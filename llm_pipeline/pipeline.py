"""LLM-only replanning pipeline driven directly by segmentation evidence."""

from __future__ import annotations

import os
import sys
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from llm_pipeline.catalog import resolve_llm_model
from llm_pipeline.client import RemoteTextLLMPlanner
from llm_pipeline.executable_symbols import build_runtime_symbol_registry
from llm_pipeline.failure_logic import SegmentationFirstFailureChecker, GeometricFailureChecker
from llm_pipeline.planner import TextLLMPlanner
from llm_pipeline.prompt_builder import TextOnlyContextBuilder
from llm_pipeline.segmentation_adapter import SegmentationEvidenceAdapter
from llm_pipeline.strict_parser import StrictActionParser
from llm_pipeline.region_aliases import scene_object_for_region
from llm_pipeline.region_geometry import resolve_object_regions
from llm_pipeline.grill_geometry import derive_grill_semantic_facts, infer_grill_lid_open
from llm_pipeline.pipeline_types import (
    DirectAction, FailureEvent, PlanResult, ICLMode, SceneState,
    BasePlanner, BaseContextBuilder
)

# NEW Modular Components
from llm_pipeline.geometric_builder import GeometricContextBuilder
try:
    from llm_pipeline.vlm_planner import VLMPlanner
except ImportError:
    VLMPlanner = None


PROMPT_MODE_SEGMENTATION_TEXT = 'segmentation_text_only'


@dataclass
class LLMPipelineConfig:
    model_alias: str = 'qwen'
    model_path: str = ''
    icl_mode: str = ICLMode.ZERO_SHOT.value
    max_replans: int = 10
    enable_replanning: bool = True
    use_4bit: bool = False
    device: str = 'cuda'
    headless: bool = False
    text_only: bool = True
    segmentation_first: bool = True
    pre_action_checks_enabled: bool = True
    post_action_checks_enabled: bool = True
    return_home_after_each_action: bool = False
    planner_max_new_tokens: int = 256
    planner_temperature: float = 0.0
    live_segmentation_view: bool = True
    visible_objects_only: bool = True
    prompt_mode: str = PROMPT_MODE_SEGMENTATION_TEXT
    direct_executable_names: bool = True
    explicit_move_token: bool = True
    live_view_update_stride: int = 5
    use_remote_planner: bool = False
    remote_planner_url: str = ''
    task_family: str = 'kitchen'
    scene_path: str = ''
    
    # NEW Multimodal & Prompting Flags
    enable_vision: bool = False
    system_prompt_path: str = ''
    user_prompt_path: str = ''
    exemplar_path: str = ''
    context_builder_type: str = 'geometric'  # IMPROVED ACCURACY: Default to 3D geometric reasoning

    def resolve_model_name(self) -> Tuple[str, str]:
        spec = resolve_llm_model(self.model_path or self.model_alias)
        return spec.alias, spec.path


@dataclass
class ExecutionCycleRecord:
    cycle_number: int
    is_replan: bool
    icl_mode: str
    planned_actions: List[str] = field(default_factory=list)
    completed_actions: List[str] = field(default_factory=list)
    remaining_actions: List[str] = field(default_factory=list)
    raw_output: str = ''
    inference_time_s: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    failure_event: Optional[Dict[str, Any]] = None
    prompt_bundle: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cycle_number': int(self.cycle_number),
            'is_replan': bool(self.is_replan),
            'icl_mode': self.icl_mode,
            'planned_actions': list(self.planned_actions),
            'completed_actions': list(self.completed_actions),
            'remaining_actions': list(self.remaining_actions),
            'raw_output': self.raw_output,
            'inference_time_s': float(self.inference_time_s),
            'success': bool(self.success),
            'error_message': self.error_message,
            'failure_event': dict(self.failure_event) if self.failure_event else None,
            'prompt_bundle': dict(self.prompt_bundle),
        }


class LLMOnlyReplanningPipeline:
    """Runs text-only planning on raw segmentation evidence with primitive execution."""

    def __init__(
        self,
        config: Optional[LLMPipelineConfig] = None,
        planner: Optional[BasePlanner] = None,
        context_builder: Optional[BaseContextBuilder] = None,
        segmentation_adapter=None,
        failure_checker=None,
        executor=None,
    ):
        self.config = config or LLMPipelineConfig()
        self.planner = planner
        self.context_builder = context_builder
        self.segmentation_adapter = segmentation_adapter
        self.failure_checker = failure_checker
        self.executor = executor
        self.env = None
        self.cycles: List[ExecutionCycleRecord] = []
        self.last_prompt_trace: Dict[str, Any] = {}
        self.symbol_registry = None
        self._sim_step_counter = 0

        # Default Context Builder based on config
        # IMPROVED ACCURACY: Always default to 3D Geometric Reasoning
        if self.context_builder is None:
            self.context_builder = GeometricContextBuilder(
                system_prompt_path=self.config.system_prompt_path,
                user_prompt_path=self.config.user_prompt_path
            )

    def initialize(self, env=None) -> bool:
        if env is None:
            os.environ['HEADLESS'] = 'True' if self.config.headless else 'False'
            scene_path = (self.config.scene_path or '').strip()
            task_family = (self.config.task_family or 'kitchen').strip().lower()

            if task_family == 'grill':
                if scene_path:
                    os.environ['GRILL_SCENE_FILE'] = scene_path
                grill_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'grill_task2')
                if grill_dir not in sys.path:
                    sys.path.insert(0, grill_dir)
                from grill_task_streams import ENV  # Lazy import for testability
            else:
                if scene_path:
                    os.environ['KITCHEN_SCENE_FILE'] = scene_path
                from rlbench_kitchen_streams import ENV  # Lazy import for testability

            env = ENV
        self.env = env

        if hasattr(self.context_builder, 'set_env'):
            self.context_builder.set_env(env)

        if self.segmentation_adapter is None:
            self.segmentation_adapter = SegmentationEvidenceAdapter(
                env=env,
                live_segmentation_view=(self.config.live_segmentation_view and not self.config.headless),
                visible_objects_only=self.config.visible_objects_only,
            )
        else:
            if hasattr(self.segmentation_adapter, 'set_env'):
                self.segmentation_adapter.set_env(env)

        # Get mask-discovered objects from the detector (intersection of class vocab and mask visibility)
        detected_objects = None
        detector = getattr(self.segmentation_adapter, 'detector', None)
        if detector is not None and hasattr(detector, 'task_objects'):
            detected_objects = detector.task_objects
        self.symbol_registry = build_runtime_symbol_registry(env=env, detected_objects=detected_objects)
        if hasattr(self.segmentation_adapter, 'set_symbol_registry'):
            self.segmentation_adapter.set_symbol_registry(self.symbol_registry)
        if hasattr(self.context_builder, 'set_symbol_registry'):
            self.context_builder.set_symbol_registry(self.symbol_registry)

        if self.failure_checker is None:
            # IMPROVED ACCURACY: Use 3D Geometric Failure Checking by default
            self.failure_checker = GeometricFailureChecker(
                adapter=self.segmentation_adapter,
                env=env
            )
        else:
            self.failure_checker.env = env
            self.failure_checker.adapter = self.segmentation_adapter

        if self.executor is None:
            from llm_pipeline.executor import DirectPrimitiveExecutor
            from vlm_pipeline.vlm_executor_v2 import ExecutorConfig

            self.executor = DirectPrimitiveExecutor(
                env=env,
                config=ExecutorConfig(return_home_after_each_action=self.config.return_home_after_each_action),
            )
        else:
            if hasattr(self.executor, 'set_env'):
                self.executor.set_env(env)

        if hasattr(self.executor, 'set_step_callback'):
            self.executor.set_step_callback(self._on_sim_step)
        if hasattr(self.executor, 'set_action_start_callback'):
            self.executor.set_action_start_callback(self._on_action_start)

        if self.planner is None:
            model_alias, model_name = self.config.resolve_model_name()
            if self.config.enable_vision:
                if VLMPlanner is None:
                    raise ImportError("VLMPlanner dependencies not met, but enable_vision=True")
                self.planner = VLMPlanner(
                    model_path=model_name or model_alias,
                    device=self.config.device,
                    use_4bit=self.config.use_4bit
                )
            elif self.config.use_remote_planner:
                self.planner = RemoteTextLLMPlanner(
                    server_url=self.config.remote_planner_url or None,
                    expected_model=resolve_llm_model(self.config.model_path or self.config.model_alias),
                )
            else:
                self.planner = TextLLMPlanner(
                    model_name=model_name,
                    model_alias=model_alias,
                    use_4bit=self.config.use_4bit,
                    device=self.config.device,
                )

        parser = StrictActionParser(
            valid_actions=self.symbol_registry.actions,
            valid_objects=self.symbol_registry.objects,
            valid_regions=self.symbol_registry.regions,
        )
        if hasattr(self.planner, 'parser'):
            self.planner.parser = parser

        if not getattr(self.planner, 'loaded', False):
            if not self.planner.load_model():
                return False

        self.reset_episode_state()
        self._settle_environment()
        if self.segmentation_adapter is not None:
            self.segmentation_adapter.refresh_visibility(event='initial')
        self._update_live_action_sequence([], None)
        return True

    def reset_episode_state(self) -> None:
        self.cycles = []
        self.last_prompt_trace = {}
        self._sim_step_counter = 0
        if self.executor is not None and hasattr(self.executor, 'reset_episode'):
            self.executor.reset_episode()
        if self.segmentation_adapter is not None and hasattr(self.segmentation_adapter, 'reset_tracking'):
            self.segmentation_adapter.reset_tracking()

    def preflight(self, goal_text: str) -> Dict[str, Any]:
        plan_result, prompt_trace = self.plan_once(goal_text=goal_text, failure_event=None)
        debug_snapshot = self.get_debug_snapshot()
        bundle = prompt_trace.get('bundle', {})
        prompt_contract_issues = []
        if any('image' in key and bundle.get(key) for key in bundle):
            prompt_contract_issues.append('image_key_in_prompt_bundle')
        if debug_snapshot and any('image' in key and debug_snapshot.get(key) for key in debug_snapshot):
            prompt_contract_issues.append('image_key_in_debug_snapshot')
        return {
            'model_alias': getattr(self.planner, 'model_alias', self.config.model_alias),
            'model_name': getattr(self.planner, 'model_name', self.config.model_path or self.config.model_alias),
            'icl_mode': self.config.icl_mode,
            'loaded': bool(getattr(self.planner, 'loaded', False)),
            'text_only': True,
            'image_present': False,
            'prompt_contract_ok': not prompt_contract_issues,
            'prompt_contract_issues': prompt_contract_issues,
            'dry_run_plan_success': bool(plan_result.success),
            'dry_run_action_count': len(plan_result.actions),
            'dry_run_error_message': plan_result.error_message,
            'dry_run_failure_event': plan_result.failure_event.to_dict() if plan_result.failure_event else None,
            'dry_run_raw_output': plan_result.raw_output,
            'prompt_trace': prompt_trace,
            'debug_snapshot': debug_snapshot,
        }

    def plan_once(
        self,
        goal_text: str,
        failure_event: Optional[FailureEvent],
    ) -> Tuple[PlanResult, Dict[str, Any]]:
        # 1. Capture and resolve scene state
        state = self._build_scene_state()
        
        # 2. Build prompt bundle via modular context builder
        bundle = self.context_builder.build_bundle(
            state=state,
            goal_text=goal_text,
            failure_event=failure_event,
            previous_actions=[
                action
                for cycle in self.cycles
                for action in cycle.completed_actions
            ],
            icl_mode=self.config.icl_mode
        )

        is_replan = failure_event is not None
        cycle_num = len(self.cycles) + 1
        print(f'\n{"=" * 60}')
        print(f'[LLM] {"REPLAN" if is_replan else "PLAN"} cycle {cycle_num}')
        if is_replan and failure_event is not None:
            print(f'[LLM] Failure context: {failure_event.message}')
        print(f'{"=" * 60}')

        # 3. Plan using modular planner
        # We try to use the new .plan() interface, fall back to .generate_plan() for legacy
        if hasattr(self.planner, 'plan'):
            result = self.planner.plan(bundle)
        else:
            result = self.planner.generate_plan(
                system_prompt=bundle.system_prompt,
                user_prompt=bundle.user_prompt,
                icl_mode=bundle.icl_mode,
                max_new_tokens=self.config.planner_max_new_tokens,
                temperature=self.config.planner_temperature,
                held_object=getattr(self.executor, 'held_object', None),
            )

        print(f'[LLM] Raw output:')
        for line in (result.raw_output or '').strip().splitlines():
            print(f'   {line}')
        if result.success and result.actions:
            print(f'[LLM] Parsed plan ({len(result.actions)} actions):')
            for i, action in enumerate(result.actions, 1):
                print(f'   {i:2}. {action}')
        elif result.failure_event:
            print(f'[LLM] Parse FAILED: {result.failure_event.message}')
        elif not result.success:
            print(f'[LLM] Plan FAILED: {result.error_message}')
        print(f'{"=" * 60}')

        bundle_trace = {
            key: value
            for key, value in bundle.__dict__.items()
            if value is not None and not (key in {'images', 'image_paths'} and not value)
        }
        self.last_prompt_trace = {
            'bundle': bundle_trace,
            'state': state.to_dict(),
            'system_prompt': bundle.system_prompt,
            'user_prompt': bundle.user_prompt,
        }
        if result.success:
            self._update_live_action_sequence(result.actions, None)
        else:
            self._update_live_action_sequence([], None)
        return result, dict(self.last_prompt_trace)

    def _build_scene_state(self) -> SceneState:
        """Helper to build a unified SceneState from current sensors."""
        snapshot = self.segmentation_adapter.capture_snapshot(event='planning')
        
        # Extract 3D poses and region bboxes if available
        pose_map = {}
        region_map = {}
        detector = getattr(self.segmentation_adapter, 'detector', None)
        if detector:
            # 1. Objects
            for obj_name in snapshot.visible_objects:
                pose = detector.get_object_pose(obj_name)
                if pose:
                    pose_map[obj_name] = pose
            
            # 2. Regions (Geometric Resolution)
            # Use the detector to get absolute world AABBs for all supported regions.
            # This ensures GeometricContextBuilder has the data needed for resolve_region().
            for region_name in snapshot.supported_regions:
                # Map semantic names to simulator names if needed
                scene_name = scene_object_for_region(region_name)
                
                bb = detector.get_bounding_box(scene_name)
                if bb:
                    # bb is (min, max)
                    region_map[region_name] = (np.array(bb[0]), np.array(bb[1]))

        object_region_map, object_region_descriptions = resolve_object_regions(
            {name: tuple(pose[:3]) for name, pose in pose_map.items()},
            region_map,
            snapshot.supported_regions,
        )
        if not object_region_map:
            object_region_map = dict(getattr(snapshot, 'object_region_map', {}) or {})
            object_region_descriptions = dict(getattr(snapshot, 'object_region_descriptions', {}) or {})

        pddl_state = []
        if (self.config.task_family or '').strip().lower() == 'grill':
            pddl_state = derive_grill_semantic_facts(
                object_region_map,
                lid_open=infer_grill_lid_open(self.env),
            )

        state = SceneState(
            frame_index=snapshot.frame_index,
            visible_objects=snapshot.visible_objects,
            valid_regions=snapshot.supported_regions,
            pddl_state=pddl_state,
            masks=snapshot.gripper_evidence.get('masks', {}),
            pose_map=pose_map,
            region_map=region_map,
            object_region_map=object_region_map,
            object_region_descriptions=object_region_descriptions,
            gripper_state={'status': 'holding' if getattr(self.executor, 'held_object', None) else 'empty',
                           'holding': getattr(self.executor, 'held_object', None)}
        )
        # Preserve original snapshot for text-only builders that 
        # need more evidence than the resolved symbolic state
        state._original_snapshot = snapshot
        return state

    def run(self, goal_text: str) -> Dict[str, Any]:
        if self.env is None:
            raise RuntimeError('Pipeline is not initialized')

        self.reset_episode_state()
        self._settle_environment()
        if self.segmentation_adapter is not None:
            self.segmentation_adapter.refresh_visibility(event='initial')
        started_at = time.time()
        pending_failure: Optional[FailureEvent] = None
        failure_reason: Optional[str] = None
        last_failure_event: Optional[FailureEvent] = None
        execution_skipped = not self.config.enable_replanning

        if execution_skipped:
            plan_result, prompt_trace = self.plan_once(goal_text=goal_text, failure_event=None)
            cycle = ExecutionCycleRecord(
                cycle_number=1,
                is_replan=False,
                icl_mode=self.config.icl_mode,
                planned_actions=[str(action) for action in plan_result.actions],
                raw_output=plan_result.raw_output,
                inference_time_s=plan_result.inference_time,
                prompt_bundle=dict(prompt_trace.get('bundle', {})),
            )
            if plan_result.success and plan_result.actions:
                cycle.success = True
                cycle.remaining_actions = list(cycle.planned_actions)
            else:
                cycle.success = False
                cycle.error_message = plan_result.error_message or 'planning_failed'
                if plan_result.failure_event is not None:
                    cycle.failure_event = plan_result.failure_event.to_dict()
                    last_failure_event = plan_result.failure_event
                failure_reason = cycle.error_message
            self.cycles.append(cycle)
        else:
            while len(self.cycles) <= self.config.max_replans:
                cycle_number = len(self.cycles) + 1
                is_replan = pending_failure is not None
                plan_result, prompt_trace = self.plan_once(goal_text=goal_text, failure_event=pending_failure)
                cycle = ExecutionCycleRecord(
                    cycle_number=cycle_number,
                    is_replan=is_replan,
                    icl_mode=self.config.icl_mode,
                    planned_actions=[str(action) for action in plan_result.actions],
                    raw_output=plan_result.raw_output,
                    inference_time_s=plan_result.inference_time,
                    prompt_bundle=dict(prompt_trace.get('bundle', {})),
                )

                if not plan_result.success or not plan_result.actions:
                    cycle.success = False
                    cycle.error_message = plan_result.error_message or 'planning_failed'
                    if plan_result.failure_event is not None:
                        cycle.failure_event = plan_result.failure_event.to_dict()
                        last_failure_event = plan_result.failure_event
                    self.cycles.append(cycle)
                    failure_reason = cycle.error_message
                    # If the planning failure is recoverable (e.g. missing move),
                    # feed it back as context and let the LLM replan
                    if plan_result.failure_event is not None and plan_result.failure_event.should_replan:
                        # Clear stale remaining_actions — no execution happened this cycle
                        if hasattr(self.executor, 'remaining_actions'):
                            self.executor.remaining_actions = []
                        pending_failure = plan_result.failure_event
                        continue
                    break

                execution = self.executor.execute_actions(
                    plan_result.actions,
                    self.failure_checker,
                    pre_action_checks_enabled=self.config.pre_action_checks_enabled,
                    post_action_checks_enabled=self.config.post_action_checks_enabled,
                )

                cycle.completed_actions = list(getattr(self.executor, 'completed_primitive_actions', []))
                cycle.remaining_actions = list(execution.remaining_actions)
                cycle.success = bool(execution.success)
                cycle.error_message = execution.error_message
                if execution.last_failure_event is not None:
                    cycle.failure_event = execution.last_failure_event.to_dict()
                    last_failure_event = execution.last_failure_event

                self.cycles.append(cycle)

                if execution.success:
                    failure_reason = None
                    break

                pending_failure = execution.last_failure_event
                failure_reason = execution.error_message or (pending_failure.message if pending_failure else 'execution_failed')
                if pending_failure is None or not pending_failure.should_replan:
                    break
                if len(self.cycles) > self.config.max_replans:
                    break

        success = bool(self.cycles) and self.cycles[-1].success
        planned_actions = list(self.cycles[0].planned_actions) if self.cycles else []
        completed_actions = list(getattr(self.executor, 'completed_primitive_actions', []))
        remaining_actions = list(getattr(self.executor, 'remaining_actions', []))
        held_object = getattr(self.executor, 'held_object', None)
        if execution_skipped:
            completed_actions = []
            remaining_actions = list(planned_actions) if success else []
            held_object = None
        return {
            'success': success,
            'goal_text': goal_text,
            'model_alias': getattr(self.planner, 'model_alias', self.config.model_alias),
            'model_path': getattr(self.planner, 'model_name', self.config.model_path or self.config.model_alias),
            'model_type': 'llm',
            'icl_mode': self.config.icl_mode,
            'prompt_mode': self.config.prompt_mode,
            'replan_mode': 'off' if execution_skipped else 'on',
            'replanning_enabled': bool(self.config.enable_replanning),
            'execution_skipped': execution_skipped,
            'text_only': True,
            'segmentation_first': True,
            'use_remote_planner': bool(self.config.use_remote_planner),
            'remote_planner_url': self.config.remote_planner_url or None,
            'pre_action_checks_enabled': bool(self.config.pre_action_checks_enabled),
            'post_action_checks_enabled': bool(self.config.post_action_checks_enabled),
            'planned_actions': planned_actions,
            'completed_actions': completed_actions,
            'remaining_actions': remaining_actions,
            'held_object': held_object,
            'last_failure_event': last_failure_event.to_dict() if last_failure_event else None,
            'failure_reason': failure_reason,
            'total_cycles': len(self.cycles),
            'total_replans': sum(1 for cycle in self.cycles if cycle.is_replan),
            'episode_time_s': time.time() - started_at,
            'cycles': [cycle.to_dict() for cycle in self.cycles],
        }

    def get_debug_snapshot(self) -> Dict[str, Any]:
        debug = {}
        if hasattr(self.planner, 'get_debug_info'):
            debug.update(dict(self.planner.get_debug_info()))
        if self.symbol_registry is not None:
            debug['symbol_registry'] = self.symbol_registry.to_dict()
        return debug

    def shutdown(self) -> None:
        if self.segmentation_adapter is not None and hasattr(self.segmentation_adapter, 'shutdown'):
            try:
                self.segmentation_adapter.shutdown()
            except Exception:
                pass
        if self.env is None or not hasattr(self.env, 'pr'):
            return
        try:
            self.env.pr.stop()
        except Exception:
            pass
        try:
            self.env.pr.shutdown()
        except Exception:
            pass


    def _settle_environment(self) -> None:
        if self.env is None or not hasattr(self.env, 'pr'):
            return
        for _ in range(50):
            self.env.pr.step()
        if hasattr(self.env, 'get_home_conf') and hasattr(self.env, 'set_robot_conf'):
            try:
                self.env.set_robot_conf(self.env.get_home_conf())
                for _ in range(10):
                    self.env.pr.step()
            except Exception:
                pass

    def _on_sim_step(self) -> None:
        if not self.config.live_segmentation_view or self.config.headless:
            return
        self._sim_step_counter += 1
        stride = max(1, int(self.config.live_view_update_stride))
        if self._sim_step_counter % stride != 0:
            return
        if self.segmentation_adapter is not None and hasattr(self.segmentation_adapter, 'update_live_segmentation_view'):
            self.segmentation_adapter.update_live_segmentation_view()

    def _on_action_start(self, actions: List[DirectAction], current_action_index: int) -> None:
        self._update_live_action_sequence(actions, current_action_index)

    def _update_live_action_sequence(self, actions, current_action_index: Optional[int] = None) -> None:
        if not self.config.live_segmentation_view or self.config.headless:
            return
        current_action_label = None
        if actions and current_action_index is not None and 0 <= current_action_index < len(actions):
            current_action_label = str(actions[current_action_index])
        if self.segmentation_adapter is not None and hasattr(self.segmentation_adapter, 'set_live_action_sequence'):
            self.segmentation_adapter.set_live_action_sequence(
                actions=actions or [],
                current_action_index=current_action_index,
                current_action_label=current_action_label,
            )

if __name__ == '__main__':
    """CLI Entry Point for the Replanning Pipeline."""
    import argparse
    parser = argparse.ArgumentParser(description="Run the LLM/VLM Replanning Pipeline")
    parser.add_argument("--goal", type=str, required=True, help="Task goal text")
    parser.add_argument("--model", type=str, default="qwen", help="Model alias")
    parser.add_argument("--vision", action="store_true", help="Enable vision-first reasoning (VLM)")
    args = parser.parse_args()

    config = LLMPipelineConfig(
        model_alias=args.model,
        enable_vision=args.vision,
        context_builder_type='geometric'
    )
    pipeline = LLMOnlyReplanningPipeline(config=config)
    
    print(f"[ENTRY] Initializing pipeline in {config.context_builder_type} mode...")
    if pipeline.initialize():
        print(f"[ENTRY] Starting loop for goal: {args.goal}")
        results = pipeline.run(args.goal)
        print(f"[ENTRY] Loop finished. Success: {results['success']}")
    else:
        print("[ENTRY] Optimization: Initialization failed.")

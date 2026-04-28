"""Primitive-by-primitive executor using direct executable symbols."""

from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDDLSTREAM_DIR = os.path.join(ROOT_DIR, 'pddlstream')
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if PDDLSTREAM_DIR not in sys.path:
    sys.path.insert(0, PDDLSTREAM_DIR)

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from pddlstream.algorithms.meta import solve
from pddlstream.language.constants import And, PDDLProblem

from llm_pipeline.pipeline_types import DirectAction, FailureEvent, FailureStage, FailureSource
from vlm_pipeline.vlm_executor_v2 import (
    VLMExecutorV2,
    _normalize_segments,
    _pick_grasp_segment_index,
    _place_release_segment_index,
    quaternion_from_euler,
)
try:
    from ground_truth_orchestrator import (
        quaternion_rotate_vector,
        compute_tip_attachment,
        update_attached_pose,
        create_primitive_transfer_executor,
        run_open_box,
    )
except Exception:
    quaternion_rotate_vector = None
    compute_tip_attachment = None
    update_attached_pose = None
    create_primitive_transfer_executor = None
    run_open_box = None


@dataclass
class PrimitiveExecutionOutcome:
    success: bool
    completed_actions: List[str] = field(default_factory=list)
    remaining_actions: List[str] = field(default_factory=list)
    held_object: Optional[str] = None
    last_failure_event: Optional[FailureEvent] = None
    error_message: Optional[str] = None


class AbstractBundlingHandler:
    """Strategy for scene-specific bundling rituals."""
    def __init__(self, executor):
        self.executor = executor
        self.env = executor.env

    def execute_transfer(self, p_action: DirectAction, pl_action: DirectAction) -> Tuple[bool, str]:
        raise NotImplementedError

    def execute_open(self, o_action: DirectAction) -> Tuple[bool, str]:
        raise NotImplementedError


class KitchenBundlingHandler(AbstractBundlingHandler):
    """Bundling rituals for the Kitchen scene."""
    def execute_transfer(self, p_action: DirectAction, pl_action: DirectAction) -> Tuple[bool, str]:
        obj_name = p_action.args[0]
        target_region = pl_action.args[1]
        print(f"[KITCHEN-BUNDLE] --- Starting GT Transfer Ritual: {obj_name} -> {target_region} ---")
        
        # 1. Pre-action Home
        self.executor.go_home()
        
        task_label = f"LLM Bundle: {obj_name} -> {target_region}"
        
        if create_primitive_transfer_executor is None:
            return False, "GT executors not available. Check ground_truth_orchestrator imports."

        gt_executor = create_primitive_transfer_executor(self.env, obj_name, target_region, task_name=task_label)
        
        # Attempt primary execution
        success = gt_executor.execute_all()
        
        # Fallback logic for box placements (mirroring GT)
        if not success and target_region == "box_boundary":
            print(f"[Fallback] {obj_name}: box_boundary failed, trying box-inside region.")
            gt_executor = create_primitive_transfer_executor(self.env, obj_name, "box-inside", task_name=f"{task_label} [fallback box-inside]")
            success = gt_executor.execute_all()
        
        # Post-action Home
        self.executor.go_home()
        
        if not success:
            return False, f"Transfer failed for {obj_name} to {target_region}"
            
        print(f"[KITCHEN-BUNDLE] ✓ Transfer Complete.")
        return True, ""

    def execute_open(self, o_action: DirectAction) -> Tuple[bool, str]:
        print(f"[KITCHEN-BUNDLE] --- Starting GT Open Ritual ---")
        self.executor.go_home()
        
        if run_open_box is None:
            return False, "run_open_box not available."
            
        success = run_open_box(self.env, task_name="LLM Bundle: Open Box")
        
        self.executor.go_home()
        
        if not success:
            return False, "Open failed"
            
        print(f"[KITCHEN-BUNDLE] ✓ Open Complete.")
        return True, ""


class GrillBundlingHandler(AbstractBundlingHandler):
    """Bundling rituals for the Grill scene."""
    def __init__(self, executor):
        super().__init__(executor)
        self.placed_counts = {}
        # Dynamically load Grill GT script
        try:
            import sys
            import os
            import importlib.util
            GRILL_DIR = os.path.join(ROOT_DIR, "grill_task2")
            if GRILL_DIR not in sys.path:
                sys.path.insert(0, GRILL_DIR)
            
            script_path = os.path.join(GRILL_DIR, "ground_truth_orchestrator_variation1 copy.py")
            spec = importlib.util.spec_from_file_location("grill_gt", script_path)
            self.grill_gt = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.grill_gt)
            
            # Initialize globals needed by run_grill_lid_motion
            print("[GRILL-BUNDLE] Initializing Grill Globals for Lid Motion...")
            try:
                self.grill_gt._discover_lid_joint_handle(self.env, self.env.pr)
            except Exception: pass
            try:
                self.grill_gt._discover_active_handle(self.env, self.env.pr)
            except Exception: pass
            try:
                self.grill_gt._capture_handle_anchor(self.env)
            except Exception: pass
                
        except Exception as e:
            print(f"[GRILL-BUNDLE] Warning: Could not initialize grill GT module: {e}")
            self.grill_gt = None

    def execute_transfer(self, p_action: DirectAction, pl_action: DirectAction) -> Tuple[bool, str]:
        obj_name = p_action.args[0]
        target_region = pl_action.args[1]
        is_plate = "plate" in obj_name.lower()
        print(f"[GRILL-BUNDLE] --- Starting GT Transfer Ritual: {obj_name} -> {target_region} ---")
        
        # 1. Pre-action Home
        self.executor.go_home()
        
        if self.grill_gt is None:
            return False, "Grill GT not available"
            
        target_obj = self.env.get_object(obj_name)
        if target_obj is None:
            return False, f"Object {obj_name} not found"
            
        # Dynamically compute target pose using _region_slot_pose
        count = self.placed_counts.get(target_region, 0)
        target_pose = None
        if not is_plate:
            target_pose = self.grill_gt._region_slot_pose(self.env, target_obj, target_region, slot_idx=count, slot_count=3)
            self.placed_counts[target_region] = count + 1
            
        success = self.grill_gt.run_pick_place(
            self.env,
            self.env.pr,
            obj_name=obj_name,
            target_region=target_region,
            task_name=f"LLM Bundle: {obj_name} -> {target_region}",
            is_plate=is_plate,
            target_pose=target_pose
        )
        
        self.executor.go_home()
        
        if not success:
            return False, f"Transfer failed for {obj_name} -> {target_region}"
            
        print(f"[GRILL-BUNDLE] ✓ Transfer Complete.")
        return True, ""

    def execute_open(self, o_action: DirectAction) -> Tuple[bool, str]:
        print(f"[GRILL-BUNDLE] --- Starting GT Open Ritual ---")
        self.executor.go_home()
        
        if self.grill_gt is None:
            return False, "Grill GT not available"
            
        success = self.grill_gt.run_grill_lid_motion(
            self.env,
            self.env.pr,
            direction="open",
            task_name="LLM Bundle: Open Grill Lid"
        )
        
        self.executor.go_home()
        
        if not success:
             return False, "Open failed"
             
        print(f"[GRILL-BUNDLE] ✓ Open Complete.")
        return True, ""


class UnifiedActionBundler:
    """Orchestrates bundling across scenes."""
    def __init__(self, executor):
        self.executor = executor
        self.env = executor.env
        env_type = str(type(self.env)).lower()
        if "grill" in env_type:
            self.handler = GrillBundlingHandler(executor)
        else:
            self.handler = KitchenBundlingHandler(executor)

    def try_execute_bundle(self, actions: List[DirectAction], index: int) -> Tuple[int, bool, str, Optional[FailureEvent]]:
        """
        Attempts to find and execute a bundle starting at 'index'.
        Returns: (num_consumed, success, error_message, failure_event)
        """
        action = actions[index]
        
        # 1. Pattern: [Move, Pick, Move, Place] -> Transfer
        if action.action_name == 'move' and (index + 3) < len(actions):
            next1 = actions[index + 1]
            next2 = actions[index + 2]
            next3 = actions[index + 3]
            if next1.action_name == 'pick' and next2.action_name == 'move' and next3.action_name == 'place':
                # Check if it's the same object
                if next1.args[0] == next3.args[0]:
                    ok, err = self.handler.execute_transfer(next1, next3)
                    failure = None
                    if not ok:
                        failure = FailureEvent(
                            failure_id="TAMP_EXECUTION_ERROR",
                            stage=FailureStage.AFTER_EXECUTION,
                            source=FailureSource.EXECUTOR,
                            action=f"{next1.action_name}({next1.args[0]}) -> {next3.action_name}({next3.args[1]})",
                            evidence={"error": err, "target": next3.args[1], "object": next1.args[0]},
                            should_replan=True,
                            message=err
                        )
                    return 4, ok, err, failure

        # 2. Pattern: [Move, Open] -> Open
        if action.action_name == 'move' and (index + 1) < len(actions):
            next1 = actions[index + 1]
            if next1.action_name == 'open':
                ok, err = self.handler.execute_open(next1)
                failure = None
                if not ok:
                    failure = FailureEvent(
                        failure_id="TAMP_OPEN_ERROR",
                        stage=FailureStage.AFTER_EXECUTION,
                        source=FailureSource.EXECUTOR,
                        action=f"open({next1.args[0]})",
                        evidence={"error": err, "target": next1.args[0]},
                        should_replan=True,
                        message=err
                    )
                return 2, ok, err, failure

        return 0, False, "", None


class DirectPrimitiveExecutor(VLMExecutorV2):
    """Executes direct LLM actions one primitive at a time."""

    def __init__(self, env=None, config=None):
        super().__init__(env=env, config=config)
        self.held_object: Optional[str] = None
        self.completed_primitive_actions: List[str] = []
        self.remaining_actions: List[str] = []
        self.last_failure_event: Optional[FailureEvent] = None
        self._manual_hold_context: Optional[dict] = None
        self._last_action_name: Optional[str] = None
        self._pending_pddl_segments: Optional[dict] = None
        self.action_start_callback: Optional[Callable[[List[DirectAction], int], None]] = None
        self.bundler: Optional[UnifiedActionBundler] = None
        if env is not None:
            self.set_env(env)

    def set_env(self, env) -> None:
        super().set_env(env)
        if env is not None:
            self.bundler = UnifiedActionBundler(self)

    def set_action_start_callback(self, callback: Optional[Callable[[List[DirectAction], int], None]]) -> None:
        self.action_start_callback = callback

    def reset_episode(self) -> None:
        super().reset()
        self.held_object = None
        self.completed_primitive_actions = []
        self.remaining_actions = []
        self.last_failure_event = None
        self._manual_hold_context = None
        self._last_action_name = None
        self._pending_pddl_segments = None

    def execute_actions(
        self,
        actions: List[DirectAction],
        failure_checker,
        pre_action_checks_enabled: bool = True,
        post_action_checks_enabled: bool = True,
    ) -> PrimitiveExecutionOutcome:
        self.last_failure_event = None
        
        # We need a skip_counter to skip actions handled by the bundler
        skip_counter = 0

        for index, action in enumerate(actions):
            if skip_counter > 0:
                skip_counter -= 1
                continue

            self.remaining_actions = [str(item) for item in actions[index:]]
            
            # --- Try Bundling ---
            if self.bundler:
                consumed, success, err, fail_event = self.bundler.try_execute_bundle(actions, index)
                if consumed > 0:
                    if not success:
                        return PrimitiveExecutionOutcome(
                            success=False, 
                            error_message=err,
                            last_failure_event=fail_event
                        )
                    
                    # Log bundled actions and update held_object state
                    for i in range(consumed):
                        action_item = actions[index + i]
                        self.completed_primitive_actions.append(str(action_item))
                        if action_item.action_name == 'pick':
                            self.held_object = action_item.args[0]
                        elif action_item.action_name == 'place':
                            self.held_object = None
                    
                    skip_counter = consumed - 1
                    continue
            # --- End Bundling ---

            next_action = actions[index + 1] if index + 1 < len(actions) else None
            if self.action_start_callback is not None:
                try:
                    self.action_start_callback(actions, index)
                except Exception:
                    pass

            print(f'[EXEC] ({index + 1}/{len(actions)}) {action}')

            pre_failure = None
            if pre_action_checks_enabled and failure_checker is not None:
                pre_snapshot = failure_checker.capture_snapshot(event=f'before-{index + 1}')
                pre_failure = failure_checker.precheck(action, self.held_object, pre_snapshot, last_action_name=self._last_action_name)
            if pre_failure is not None:
                print(f'[EXEC] PRE-CHECK FAILED: {pre_failure.message}')
                self.last_failure_event = pre_failure
                return PrimitiveExecutionOutcome(
                    success=False,
                    completed_actions=list(self.completed_primitive_actions),
                    remaining_actions=list(self.remaining_actions),
                    held_object=self.held_object,
                    last_failure_event=pre_failure,
                    error_message=pre_failure.message,
                )

            success, error_message = self._dispatch_action(action, next_action=next_action)
            if not success:
                print(f'[EXEC] FAILED: {error_message}')
                failure = None
                if failure_checker is not None:
                    failure = failure_checker.classify_runtime_error(action, error_message or 'Execution failed')
                self.last_failure_event = failure
                return PrimitiveExecutionOutcome(
                    success=False,
                    completed_actions=list(self.completed_primitive_actions),
                    remaining_actions=list(self.remaining_actions),
                    held_object=self.held_object,
                    last_failure_event=failure,
                    error_message=(failure.message if failure is not None else error_message),
                )

            if action.action_name == 'pick':
                self.held_object = action.args[0]
                print(f'[EXEC] Holding: {self.held_object}')
            elif action.action_name == 'place':
                self.held_object = None

            post_failure = None
            if post_action_checks_enabled and failure_checker is not None:
                post_snapshot = failure_checker.capture_snapshot(event=f'after-{index + 1}')
                post_failure = failure_checker.postcheck(action, self.held_object, post_snapshot)
            if post_failure is not None:
                if action.action_name == 'pick':
                    self.held_object = None
                print(f'[EXEC] POST-CHECK FAILED: {post_failure.message}')
                self.last_failure_event = post_failure
                return PrimitiveExecutionOutcome(
                    success=False,
                    completed_actions=list(self.completed_primitive_actions),
                    remaining_actions=list(self.remaining_actions),
                    held_object=self.held_object,
                    last_failure_event=post_failure,
                    error_message=post_failure.message,
                )

            self.completed_primitive_actions.append(str(action))
            self._last_action_name = action.action_name

            # After place or open, silently return to home pose so the next
            # move starts from a clean known configuration
            if action.action_name in ('place', 'open') and self.held_object is None:
                print(f'[EXEC] Auto-home after {action.action_name}')
                try:
                    self.go_home()
                except Exception:
                    pass
            elif self.config.return_home_after_each_action and self.held_object is None:
                self.go_home()

        self.remaining_actions = []
        return PrimitiveExecutionOutcome(
            success=True,
            completed_actions=list(self.completed_primitive_actions),
            remaining_actions=[],
            held_object=self.held_object,
            last_failure_event=None,
            error_message=None,
        )

    def _dispatch_action(self, action: DirectAction, next_action: Optional[DirectAction] = None) -> Tuple[bool, str]:
        if action.action_name == 'move':
            return self._execute_move_token(next_action=next_action)
        if action.action_name == 'pick':
            object_name = action.args[0]
            if object_name == 'mug3':
                return self._execute_cupboard_pick(object_name)
            return self._execute_pick_pddl(object_name)
        if action.action_name == 'place':
            object_name, target_region = action.args
            # Cupboard: scripted insert from hover → release → home
            if target_region in ('cupboard_boundary', 'cupboard_boundary_top'):
                return self._execute_cupboard_place_from_hover(object_name, target_region)
            return self._execute_place_pddl(object_name, target_region)
        if action.action_name == 'open':
            return self._execute_open_lid()
        return False, f"Unsupported action '{action.action_name}'"

    def _execute_move_token(self, next_action: Optional[DirectAction] = None) -> Tuple[bool, str]:
        """Execute a move token: position the arm for the next action via PDDL pre-solve.

        If the next action is pick/place, runs the full PDDL solve and executes
        only the move trajectory, caching the grasp/release segments for the
        subsequent pick/place call.  Falls back to go_home() if there is no
        actionable next step or if the PDDL solve fails.
        """
        self._pending_pddl_segments = None

        if next_action is not None and next_action.action_name == 'pick':
            object_name = next_action.args[0]
            # Skip PDDL pre-solve for cupboard picks — they have custom logic
            if object_name != 'mug3':
                result = self._presolve_pick(object_name)
                if result is not None:
                    return True, 'Success'

        if next_action is not None and next_action.action_name == 'place':
            object_name, target_region = next_action.args
            # Cupboard: carry object from pick hover → place hover (scripted, no PDDL)
            if target_region in ('cupboard_boundary', 'cupboard_boundary_top'):
                return self._move_to_cupboard_place_hover(object_name, target_region)
            # For non-cupboard regions: PDDL pre-solve moves to place hover and caches segments.
            result = self._presolve_place(object_name, target_region)
            if result is not None:
                return True, 'Success'

        # Fallback: go to home pose
        try:
            self.go_home()
        except Exception:
            if self.env is not None and hasattr(self.env, 'get_home_conf') and hasattr(self.env, 'set_robot_conf'):
                self.env.set_robot_conf(self.env.get_home_conf())
                for _ in range(10):
                    self._step_sim()
        return True, 'Success'

    def _presolve_pick(self, object_name: str) -> Optional[bool]:
        """Run PDDL solve for pick, execute move trajectory, cache pick segments."""
        self._load_pddl_files()
        env = self.env
        obj = env.get_object(object_name)
        if obj is None:
            return None

        q_start = tuple(env.get_robot_conf())
        try:
            obj.set_dynamic(False)
        except Exception:
            pass
        pose_tuple = tuple(obj.get_pose())

        init = [
            ('conf', q_start),
            ('at-conf', q_start),
            ('hand-empty',),
            ('movable', object_name),
            ('pose', pose_tuple),
            ('at-pose', object_name, pose_tuple),
        ]
        goal = And(('holding', object_name))
        problem = PDDLProblem(
            domain_pddl=self.domain_pddl,
            constant_map={},
            stream_pddl=self.stream_pddl,
            stream_map=self._get_stream_map(),
            init=init,
            goal=goal,
        )
        plan, _, _ = solve(problem, algorithm='adaptive', verbose=False, max_time=60)
        if not plan:
            return None

        # Execute only the move trajectories; cache the pick action data
        pick_actions = []
        for action in plan:
            if action.name == 'move':
                _, _, traj = action.args
                self._execute_trajectory(traj)
            elif action.name == 'pick':
                pick_actions.append(action)

        if pick_actions:
            self._pending_pddl_segments = {
                'type': 'pick',
                'object_name': object_name,
                'actions': pick_actions,
            }
        return True

    def _presolve_place(self, object_name: str, target_region: str) -> Optional[bool]:
        """Run PDDL solve for place, execute move trajectory, cache place segments."""
        self._load_pddl_files()
        env = self.env
        env.set_target_region(target_region)
        q_start = tuple(env.get_robot_conf())
        init = [
            ('conf', q_start),
            ('at-conf', q_start),
            ('holding', object_name),
            ('movable', object_name),
            ('region', target_region),
        ]
        goal = And(('hand-empty',), ('in-region', object_name, target_region))
        problem = PDDLProblem(
            domain_pddl=self.domain_pddl,
            constant_map={},
            stream_pddl=self.stream_pddl,
            stream_map=self._get_stream_map(),
            init=init,
            goal=goal,
        )
        plan, _, _ = solve(problem, algorithm='adaptive', verbose=False, max_time=60)
        if not plan:
            return None

        # Execute only the move trajectories; cache the place action data
        place_actions = []
        for action in plan:
            if action.name == 'move':
                _, _, traj = action.args
                self._execute_trajectory(traj)
            elif action.name == 'place':
                place_actions.append(action)

        if place_actions:
            self._pending_pddl_segments = {
                'type': 'place',
                'object_name': object_name,
                'target_region': target_region,
                'actions': place_actions,
            }
        return True

    def _execute_pick_pddl(self, object_name: str) -> Tuple[bool, str]:
        env = self.env
        obj = env.get_object(object_name)
        if obj is None:
            return False, f"Object '{object_name}' not found"

        # Use cached pick segments from a preceding move's PDDL pre-solve
        cached = self._pending_pddl_segments
        if cached is not None and cached.get('type') == 'pick' and cached.get('object_name') == object_name:
            pick_actions = cached['actions']
            self._pending_pddl_segments = None
        else:
            # Fallback: full PDDL solve (no preceding move did the pre-solve)
            self._pending_pddl_segments = None
            self._load_pddl_files()
            q_start = tuple(env.get_robot_conf())
            obj.set_dynamic(False)
            pose_tuple = tuple(obj.get_pose())

            init = [
                ('conf', q_start),
                ('at-conf', q_start),
                ('hand-empty',),
                ('movable', object_name),
                ('pose', pose_tuple),
                ('at-pose', object_name, pose_tuple),
            ]
            goal = And(('holding', object_name))
            problem = PDDLProblem(
                domain_pddl=self.domain_pddl,
                constant_map={},
                stream_pddl=self.stream_pddl,
                stream_map=self._get_stream_map(),
                init=init,
                goal=goal,
            )
            print("DEBUG [Executor]: Calling PDDL solve (adaptive)...")
            plan, _, _ = solve(problem, algorithm='adaptive', verbose=False, max_time=60)
            if not plan:
                return False, 'No PDDL plan found'
            pick_actions = []
            for action in plan:
                if action.name == 'move':
                    _, _, traj = action.args
                    self._execute_trajectory(traj)
                elif action.name == 'pick':
                    pick_actions.append(action)

        # Execute pick segments (grasp from hover)
        for pick_action in pick_actions:
            o, _, _, _, _, traj_tuple = pick_action.args
            segments = _normalize_segments(traj_tuple)
            if not segments:
                return False, 'Pick has empty trajectory'
            target_obj = env.get_object(o)
            grasp_idx = _pick_grasp_segment_index(env, target_obj, segments)
            for seg in segments[:grasp_idx + 1]:
                self._execute_trajectory(seg)
            target_obj.set_dynamic(True)
            env.gripper.actuate(0.0, 0.1)
            for _ in range(10):
                self._step_sim()
            env.gripper.grasp(target_obj)
            for seg in segments[grasp_idx + 1:]:
                self._execute_trajectory(seg)
                
            # RETRACT TO HOME (GT Ritual)
            self.go_home()
            
        return True, 'Success'

    def _execute_place_pddl(self, object_name: str, target_region: str) -> Tuple[bool, str]:
        env = self.env

        # Use cached place segments from a preceding move's PDDL pre-solve
        cached = self._pending_pddl_segments
        if cached is not None and cached.get('type') == 'place' and cached.get('object_name') == object_name:
            place_actions = cached['actions']
            self._pending_pddl_segments = None
        else:
            # Fallback: full PDDL solve (no preceding move did the pre-solve)
            self._pending_pddl_segments = None
            self._load_pddl_files()
            env.set_target_region(target_region)
            q_start = tuple(env.get_robot_conf())
            init = [
                ('conf', q_start),
                ('at-conf', q_start),
                ('holding', object_name),
                ('movable', object_name),
                ('region', target_region),
            ]
            goal = And(('hand-empty',), ('in-region', object_name, target_region))
            problem = PDDLProblem(
                domain_pddl=self.domain_pddl,
                constant_map={},
                stream_pddl=self.stream_pddl,
                stream_map=self._get_stream_map(),
                init=init,
                goal=goal,
            )
            print("DEBUG [Executor]: Calling PDDL solve for PLACE (adaptive)...")
            plan, _, _ = solve(problem, algorithm='adaptive', verbose=False, max_time=60)
            if not plan:
                return False, 'No PDDL plan found'
            place_actions = []
            for action in plan:
                if action.name == 'move':
                    _, _, traj = action.args
                    self._execute_trajectory(traj)
                elif action.name == 'place':
                    place_actions.append(action)

        # Execute place segments (release from hover)
        for place_action in place_actions:
            o, p, _, _, _, _, traj_tuple = place_action.args
            segments = _normalize_segments(traj_tuple)
            if not segments:
                return False, 'Place has empty trajectory'
            release_idx = _place_release_segment_index(env, p, segments)
            for seg in segments[:release_idx + 1]:
                self._execute_trajectory(seg)
            target_obj = env.get_object(o)
            env.gripper.release()
            target_obj.set_dynamic(True)
            hold_q = env.get_robot_conf()
            env.gripper.actuate(1.0, velocity=0.2)
            for _ in range(60):
                env.set_robot_conf(hold_q)
                self._step_sim()
            if len(segments) > release_idx + 1:
                for seg in segments[release_idx + 1:]:
                    self._execute_trajectory(seg)
            
            # RETRACT TO HOME (GT Ritual)
            self.go_home()
            
        return True, 'Success'

    def _execute_cupboard_pick(self, object_name: str) -> Tuple[bool, str]:
        env = self.env
        home_q = env.get_home_conf()
        env.set_robot_conf(home_q)
        for _ in range(10):
            self._step_sim()

        obj = env.get_object(object_name)
        if obj is None:
            return False, f"Object '{object_name}' not found"

        obj.set_dynamic(False)
        pose = obj.get_pose()
        hover_dists = [0.35, 0.30, 0.40]
        q_hover = None
        successful_grasp_quat = None
        original_conf = env.get_robot_conf()
        base_ry = np.pi / 2
        grasp_quats = [
            quaternion_from_euler(0, base_ry, 0),
            quaternion_from_euler(np.pi, base_ry, 0),
        ]

        target_z = pose[2]
        grasp_depth_offset = 0.03
        grasp_pos = None
        hover_pos = None

        for h_dist in hover_dists:
            if q_hover is not None:
                break
            hover_pos = [pose[0] - h_dist, pose[1], target_z]
            grasp_pos = [pose[0] + grasp_depth_offset, pose[1], target_z]
            for grasp_rot in grasp_quats:
                path_configs = env.robot.solve_ik_via_sampling(
                    hover_pos,
                    quaternion=grasp_rot,
                    max_configs=50,
                    max_time_ms=1000,
                    ignore_collisions=True,
                )
                if path_configs is None or len(path_configs) == 0:
                    continue
                for q in path_configs:
                    env.set_robot_conf(q)
                    if env.robot.check_collision():
                        continue
                    try:
                        path_check = env.robot.get_linear_path(
                            position=grasp_pos,
                            quaternion=grasp_rot,
                            steps=20,
                            ignore_collisions=True,
                        )
                    except Exception:
                        path_check = None
                    if path_check:
                        q_hover = q
                        successful_grasp_quat = grasp_rot
                        break
                if q_hover is not None:
                    break

        if q_hover is None:
            env.set_robot_conf(original_conf)
            return False, 'Could not find valid horizontal hover configuration'

        env.set_robot_conf(original_conf)
        traj_to_hover = env._interpolate_joint_path(home_q, q_hover, steps=100, check_collisions=False)
        if traj_to_hover:
            self._execute_trajectory(traj_to_hover, steps=10)

        env.gripper.release()
        for _ in range(30):
            self._step_sim()

        path_approach = env.robot.get_linear_path(
            position=grasp_pos,
            quaternion=successful_grasp_quat,
            steps=200,
            ignore_collisions=True,
        )
        if path_approach:
            traj_approach = path_approach._path_points.reshape(-1, 7).tolist()
            for conf in traj_approach:
                env.set_robot_conf(conf)
                self._step_sim()
        else:
            return False, 'Could not plan approach trajectory'

        env.gripper.actuate(0.0, 0.1)
        for _ in range(50):
            self._step_sim()

        gripper_tip = env.robot.get_tip()
        tip_pos = np.array(gripper_tip.get_position())
        mug_current_pos = np.array(obj.get_position())
        mug_offset = mug_current_pos - tip_pos

        retrieve_pos = list(hover_pos)
        retrieve_pos[2] += 0.02
        try:
            path_retrieve = env.robot.get_linear_path(
                position=retrieve_pos,
                quaternion=successful_grasp_quat,
                steps=200,
                ignore_collisions=True,
            )
        except Exception:
            path_retrieve = None

        if path_retrieve:
            traj_retrieve = path_retrieve._path_points.reshape(-1, 7).tolist()
            for conf in traj_retrieve:
                env.set_robot_conf(conf)
                self._step_sim()
                new_tip_pos = np.array(gripper_tip.get_position())
                obj.set_position((new_tip_pos + mug_offset).tolist())
        else:
            traj_fallback = env._interpolate_joint_path(env.get_robot_conf(), q_hover, steps=100, check_collisions=False)
            if traj_fallback:
                for conf in traj_fallback:
                    env.set_robot_conf(conf)
                    self._step_sim()
                    new_tip_pos = np.array(gripper_tip.get_position())
                    obj.set_position((new_tip_pos + mug_offset).tolist())
            else:
                return False, 'Could not retrieve to hover'

        self._manual_hold_context = {
            'object_name': object_name,
            'grasp_quat': successful_grasp_quat,
        }
        return True, 'Success'

    def _move_to_cupboard_place_hover(self, object_name: str, target_region: str) -> Tuple[bool, str]:
        """Move from pick hover → home → place hover while carrying the object.
        Routes through home to avoid IK failures from awkward pick hover configs."""
        env = self.env
        obj = env.get_object(object_name)
        if obj is None:
            return False, f"Object '{object_name}' not found"

        gripper_tip = env.robot.get_tip()

        # Compute rigid tip-frame attachment (same as GT compute_tip_attachment)
        if compute_tip_attachment is not None:
            mug_tip_offset_local, mug_tip_quat_local = compute_tip_attachment(gripper_tip, obj)
        else:
            tip_pos = np.array(gripper_tip.get_position())
            mug_tip_offset_local = np.array(obj.get_position()) - tip_pos
            mug_tip_quat_local = None

        # Helper: move object along with arm during joint trajectory
        def _track_object_during_traj(traj):
            for conf in traj:
                env.set_robot_conf(conf)
                self._step_sim()
                if update_attached_pose is not None and mug_tip_quat_local is not None:
                    update_attached_pose(gripper_tip, obj, mug_tip_offset_local, mug_tip_quat_local)
                else:
                    new_tip_pos = np.array(gripper_tip.get_position())
                    obj.set_position((new_tip_pos + mug_tip_offset_local).tolist())

        # ---- Step 1: Pick hover → Home (carrying the object) ----
        home_q = env.get_home_conf()
        current_conf = env.get_robot_conf()
        traj_to_home = env._interpolate_joint_path(current_conf, home_q, steps=100, check_collisions=False)
        if traj_to_home:
            _track_object_during_traj(traj_to_home)

        # ---- Step 2: Find placement position ----
        try:
            place_pose = env.find_best_placement(obj, target_region)
        except Exception:
            place_pose = [0.0, 0.3, 0.77, 0, 0, 0, 1]

        # Use region surface Z
        region_obj = (getattr(env, 'regions', {}) or {}).get(target_region)
        place_surface_z = None
        if region_obj is not None:
            try:
                bb = region_obj.get_bounding_box()
                place_surface_z = float(region_obj.get_position()[2]) + float(bb[5])
            except Exception:
                pass
        if place_surface_z is None:
            place_surface_z = float(place_pose[2])

        min_x, _max_x, _min_y, _max_y, min_z, _max_z = obj.get_bounding_box()
        place_object_z = place_surface_z - float(min_z) + 0.0002
        final_place_obj_pos = np.array([place_pose[0], place_pose[1], place_object_z], dtype=float)

        # ---- Step 3: Search for IK at place hover (from home config) ----
        hover_z_offset = float(os.environ.get('GT_PICK_PLACE_HOVER_Z', '0.30'))
        place_quats = [
            quaternion_from_euler(np.pi, 0, angle)
            for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False)
        ]

        q_place_hover = None
        successful_place_quat = None
        successful_place_pos = None
        pre_search_conf = env.get_robot_conf()  # Should be home now

        for place_quat in place_quats:
            # Compute where the gripper tip must be for the obj to land at final_place_obj_pos
            if quaternion_rotate_vector is not None:
                tip_offset_world = quaternion_rotate_vector(place_quat, mug_tip_offset_local)
                candidate_place_pos = (final_place_obj_pos - tip_offset_world).tolist()
            else:
                candidate_place_pos = final_place_obj_pos.tolist()

            candidate_hover_pos = [
                candidate_place_pos[0],
                candidate_place_pos[1],
                candidate_place_pos[2] + max(0.08, hover_z_offset),
            ]

            try:
                path_configs = env.robot.solve_ik_via_sampling(
                    candidate_hover_pos,
                    quaternion=place_quat,
                    max_configs=20,
                    max_time_ms=500,
                    ignore_collisions=True,
                )
            except Exception:
                continue
            if path_configs is None or len(path_configs) == 0:
                continue
            for q in path_configs:
                env.set_robot_conf(q)
                if env.robot.check_collision():
                    continue
                try:
                    place_configs = env.robot.solve_ik_via_sampling(
                        candidate_place_pos,
                        quaternion=place_quat,
                        max_configs=5,
                        max_time_ms=200,
                        ignore_collisions=True,
                    )
                except Exception:
                    continue
                if place_configs is not None and len(place_configs) > 0:
                    q_place_hover = q
                    successful_place_quat = place_quat
                    successful_place_pos = candidate_place_pos
                    break
            if q_place_hover is not None:
                break

        env.set_robot_conf(pre_search_conf)
        if q_place_hover is None:
            return False, 'Could not find place hover configuration'

        # ---- Step 4: Home → Place hover (carrying the object) ----
        current_conf = env.get_robot_conf()
        traj_to_place = env._interpolate_joint_path(current_conf, q_place_hover, steps=150, check_collisions=False)
        if traj_to_place:
            _track_object_during_traj(traj_to_place)

        # Cache for the subsequent place action
        self._pending_cupboard_place = {
            'object_name': object_name,
            'place_pos': successful_place_pos,
            'place_quat': successful_place_quat,
            'mug_tip_offset_local': mug_tip_offset_local,
            'mug_tip_quat_local': mug_tip_quat_local,
        }
        return True, 'Success'

    def _execute_cupboard_place_from_hover(self, object_name: str, target_region: str) -> Tuple[bool, str]:
        """From place hover: lower into cupboard, release, lift, return home."""
        env = self.env
        obj = env.get_object(object_name)
        if obj is None:
            return False, f"Object '{object_name}' not found"
        gripper_tip = env.robot.get_tip()

        # Use cached data from _move_to_cupboard_place_hover
        cached = getattr(self, '_pending_cupboard_place', None)
        if cached is not None and cached.get('object_name') == object_name:
            place_pos = cached['place_pos']
            successful_place_quat = cached['place_quat']
            mug_tip_offset_local = cached.get('mug_tip_offset_local')
            mug_tip_quat_local = cached.get('mug_tip_quat_local')
            self._pending_cupboard_place = None
        else:
            # Fallback: re-compute (shouldn't normally happen)
            self._pending_cupboard_place = None
            if compute_tip_attachment is not None:
                mug_tip_offset_local, mug_tip_quat_local = compute_tip_attachment(gripper_tip, obj)
            else:
                tip_pos = np.array(gripper_tip.get_position())
                mug_tip_offset_local = np.array(obj.get_position()) - tip_pos
                mug_tip_quat_local = None
            try:
                place_pose = env.find_best_placement(obj, target_region)
            except Exception:
                place_pose = [0.0, 0.3, 0.77, 0, 0, 0, 1]
            place_z = place_pose[2] + 0.015
            place_pos = [place_pose[0], place_pose[1], place_z]
            successful_place_quat = quaternion_from_euler(np.pi, 0, 0)

        # Lower from hover → place position
        try:
            path_lower = env.robot.get_linear_path(
                position=place_pos,
                quaternion=successful_place_quat,
                steps=100,
                ignore_collisions=True,
            )
        except Exception:
            path_lower = None
        if path_lower:
            traj_lower = path_lower._path_points.reshape(-1, 7).tolist()
            for conf in traj_lower:
                env.set_robot_conf(conf)
                self._step_sim()
                if update_attached_pose is not None and mug_tip_quat_local is not None:
                    update_attached_pose(gripper_tip, obj, mug_tip_offset_local, mug_tip_quat_local)
                else:
                    new_tip_pos = np.array(gripper_tip.get_position())
                    obj.set_position((new_tip_pos + mug_tip_offset_local).tolist())

        # Release
        env.gripper.actuate(1.0, 0.1)
        for _ in range(30):
            self._step_sim()
        obj.set_dynamic(True)
        for _ in range(50):
            self._step_sim()

        # Lift
        try:
            lift_pos = [place_pos[0], place_pos[1], place_pos[2] + 0.15]
            path_lift = env.robot.get_linear_path(
                position=lift_pos,
                quaternion=successful_place_quat,
                steps=50,
                ignore_collisions=True,
            )
            if path_lift:
                traj_lift = path_lift._path_points.reshape(-1, 7).tolist()
                self._execute_trajectory(traj_lift)
        except Exception:
            pass

        # Return to home (part of place action)
        self.go_home()

        self._manual_hold_context = None
        return True, 'Success'


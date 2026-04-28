#!/usr/bin/env python3
"""
VLM Pipeline with Replanning
=============================
Runs VLM planning with automatic replanning on failures.

When execution fails:
1. Captures completed actions
2. Captures error message
3. Re-prompts VLM with failure context
4. Continues with new plan

Run: python -m vlm_pipeline.vlm_with_replanning --goal "your goal here"
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm_pipeline.model_registry import (
    PROMPT_MODE_TEXT_VISIBLE,
    PROMPT_MODE_VISION_GOAL,
    format_model_listing,
    resolve_model_spec,
)
from vlm_pipeline.planner_factory import create_local_planner
from vlm_pipeline.vlm_planner import ActionSkeleton, PlanResult
from vlm_pipeline.vlm_executor_v2 import VLMExecutorV2, ExecutionResult, ExecutionStatus
from vlm_pipeline.vlm_context_aggregator import VLMContextAggregator, PromptBundle

DISCOVERY_FAILURE_CODE_DEFAULT = "NEW_OBJECT_INTRODUCED_IN_SCENE"


@dataclass
class ReplanConfig:
    """Configuration for replanning."""
    max_replans: int = 3
    use_mock_vlm: bool = False
    use_remote_vlm: bool = False  # Use remote VLM server
    remote_vlm_url: str = ""  # URL of remote VLM server
    model: str = "qwen-vl"
    model_type: str = ""
    planner_input_mode: str = ""
    use_4bit: bool = False
    use_vision: bool = True  # Auto-derived from model type unless overridden upstream
    visible_objects_only: bool = False  # Use segmentation-visible objects only in context
    live_segmentation_view: bool = False  # Show live segmentation mask window
    replan_on_discovery: bool = False  # Trigger replan when target object becomes newly visible
    discovery_targets: List[str] = None  # PDDL names that trigger replan on first visibility
    discovery_failure_code: str = DISCOVERY_FAILURE_CODE_DEFAULT
    replan_on_discovery_after_plan_complete: bool = False  # Allow discovery-triggered replan even if current plan just finished
    live_view_update_stride: int = 5  # Viewer refresh every N simulator steps
    headless: bool = False
    save_logs: bool = True
    log_dir: str = "vlm_replan_logs"

    def __post_init__(self):
        if self.discovery_targets is None:
            self.discovery_targets = []

        spec = resolve_model_spec(self.model, self.model_type)
        self.model = self.model or spec.alias
        self.model_type = spec.model_type
        if not self.planner_input_mode:
            self.planner_input_mode = spec.prompt_mode
        self.use_vision = spec.use_vision
        if self.planner_input_mode == PROMPT_MODE_TEXT_VISIBLE:
            self.visible_objects_only = True


@dataclass
class ReplanCycle:
    """Information about a single plan-execute cycle."""
    cycle_number: int
    is_replan: bool
    plan: List[str]
    completed_actions: List[str]
    failed_action: Optional[str]
    error_message: Optional[str]
    success: bool


class VLMReplanningPipeline:
    """
    VLM Pipeline with replanning capability.
    
    When execution fails, re-prompts the VLM with:
    - Original task description
    - Actions completed so far
    - Error message explaining failure
    - Request for corrected remaining actions
    """
    
    def __init__(self, config: ReplanConfig = None):
        self.config = config or ReplanConfig()
        self.model_spec = resolve_model_spec(self.config.model, self.config.model_type)
        
        # Initialize components
        self.context_aggregator = VLMContextAggregator(
            visible_objects_only=self.config.visible_objects_only,
            enable_live_segmentation_view=self.config.live_segmentation_view,
            planner_input_mode=self.config.planner_input_mode,
        )
        self.executor = VLMExecutorV2()
        
        # Initialize planner
        if self.config.use_remote_vlm:
            from vlm_pipeline.vlm_client import RemoteVLMPlanner
            url = self.config.remote_vlm_url or os.environ.get("VLM_SERVER_URL", "http://localhost:8000")
            print(f"[ReplanPipeline] Using REMOTE planner: {url}")
            self.planner = RemoteVLMPlanner(
                server_url=url,
                use_vision=self.config.use_vision,
                expected_model=self.model_spec,
            )
        else:
            if self.config.use_mock_vlm:
                print("[ReplanPipeline] Using MOCK planner")
            else:
                print(
                    f"[ReplanPipeline] Using local {self.model_spec.model_type.upper()} "
                    f"{self.model_spec.alias}: {self.model_spec.path}"
                )
            self.planner = create_local_planner(
                model=self.config.model,
                model_type=self.model_spec.model_type,
                use_4bit=self.config.use_4bit,
                use_mock=self.config.use_mock_vlm,
            )
        
        self.env = None
        
        # Tracking
        self.all_completed_actions: List[ActionSkeleton] = []
        self.cycles: List[ReplanCycle] = []
        self.original_goal = ""
        self.run_started_at: Optional[datetime] = None
        self.discovery_targets: Set[str] = set(self.config.discovery_targets or [])
        self._sim_step_counter: int = 0
        self._last_visibility_snapshot: Dict[str, List[str]] = {
            "visible_scene": [],
            "visible_pddl": [],
            "newly_visible_scene": [],
            "newly_visible_pddl": [],
        }
        
        # Logging
        if self.config.save_logs:
            os.makedirs(self.config.log_dir, exist_ok=True)
    
    def initialize(self) -> bool:
        """Initialize the pipeline."""
        print("\n" + "=" * 60)
        print("MODULAR REPLANNING PIPELINE")
        print("=" * 60)
        
        print(f"[ReplanPipeline] Model alias: {self.model_spec.alias}")
        print(f"[ReplanPipeline] Model path : {self.model_spec.path}")
        print(f"[ReplanPipeline] Model type : {self.model_spec.model_type}")
        print(f"[ReplanPipeline] Prompt mode: {self.config.planner_input_mode}")

        # Load planner
        if not self.planner.load_model():
            print("[ReplanPipeline] ERROR: Failed to load planner")
            return False
        
        # Load environment
        print("[ReplanPipeline] Loading environment...")
        try:
            os.environ["HEADLESS"] = "True" if self.config.headless else "False"
            
            from rlbench_kitchen_streams import ENV
            self.env = ENV
            
            self.context_aggregator.set_env(self.env)
            self.context_aggregator.configure_visibility(
                visible_objects_only=self.config.visible_objects_only,
                enable_live_segmentation_view=self.config.live_segmentation_view
            )
            self.context_aggregator.set_planner_input_mode(self.config.planner_input_mode)
            self.executor.set_env(self.env)

            if self.config.live_segmentation_view:
                print("[ReplanPipeline] Live segmentation viewer ENABLED")
                self.executor.set_step_callback(self._on_sim_step)
            else:
                self.executor.set_step_callback(None)

            if self.config.replan_on_discovery:
                print(f"[ReplanPipeline] Discovery-triggered replanning ENABLED for: {sorted(self.discovery_targets)}")
                if self.config.replan_on_discovery_after_plan_complete:
                    print("[ReplanPipeline] Discovery replan after plan completion ENABLED")
                self.executor.set_post_action_callback(self._post_action_discovery_check)
            else:
                self.executor.set_post_action_callback(None)
            
            # Settle physics
            print("[ReplanPipeline] Settling physics...")
            for _ in range(50):
                self.env.pr.step()
            
            # Go home
            home_q = self.env.get_home_conf()
            self.env.set_robot_conf(home_q)
            for _ in range(10):
                self.env.pr.step()

            if self.config.visible_objects_only or self.config.live_segmentation_view:
                self._last_visibility_snapshot = self.context_aggregator.update_visible_objects(event="initial")

            self._update_live_action_sequence([], None)
            
            print("[ReplanPipeline] Environment ready")
            
        except Exception as e:
            print(f"[ReplanPipeline] ERROR loading environment: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("[ReplanPipeline] Initialization complete!")
        return True

    def _on_sim_step(self):
        """Refresh live segmentation viewer during motion (throttled)."""
        if not self.config.live_segmentation_view:
            return
        self._sim_step_counter += 1
        stride = max(1, int(self.config.live_view_update_stride))
        if self._sim_step_counter % stride != 0:
            return
        self.context_aggregator.update_live_segmentation_view()

    def _update_live_action_sequence(self,
                                     actions: Optional[List[ActionSkeleton]],
                                     current_action_index: Optional[int] = None):
        """Push the latest plan/replan action list into the live viewer."""
        if not self.config.live_segmentation_view:
            return

        current_action_label = None
        if actions and current_action_index is not None and 0 <= current_action_index < len(actions):
            current_action_label = str(actions[current_action_index])

        try:
            self.context_aggregator.set_live_action_sequence(
                actions=actions or [],
                current_action_index=current_action_index,
                current_action_label=current_action_label,
            )
        except Exception:
            pass

    def _post_action_discovery_check(self,
                                     executed_actions: List[ActionSkeleton],
                                     completed_actions: int,
                                     total_actions: int) -> Optional[str]:
        """
        Post-action hook: update segmentation visibility and optionally request replan.
        """
        if not (self.config.visible_objects_only or self.config.replan_on_discovery or self.config.live_segmentation_view):
            return None

        plan_complete = completed_actions >= total_actions
        if plan_complete and not self.config.replan_on_discovery_after_plan_complete:
            return None

        event = f"after-action-{completed_actions}"
        snapshot = self.context_aggregator.update_visible_objects(event=event)
        self._last_visibility_snapshot = snapshot

        newly_visible = set(snapshot.get("newly_visible_pddl", []))
        if not newly_visible:
            return None

        # If no targets specified, any newly visible object can trigger.
        if not self.discovery_targets:
            matched = newly_visible
        else:
            matched = newly_visible.intersection(self.discovery_targets)

        if self.config.replan_on_discovery and matched:
            matched_list = sorted(matched)
            when = "after completing current plan" if plan_complete else "during current plan"
            return (
                f"{self.config.discovery_failure_code}: Scene changed {when}. "
                f"Newly visible object(s) detected via segmentation {matched_list}. "
                "Recompute actions using current visibility."
            )

        return None
    
    def build_replan_prompt(self, 
                            original_goal: str,
                            completed_actions: List[ActionSkeleton],
                            failed_action: Optional[ActionSkeleton],
                            error_message: str,
                            remaining_actions: List[ActionSkeleton],
                            state_text: str = "",
                            visible_objects_text: str = "") -> str:
        """
        Build a replanning prompt with failure context.
        """
        completed_str = "\n".join([f"  {i+1}. {a}" for i, a in enumerate(completed_actions)]) if completed_actions else "  (none)"
        remaining_str = "\n".join([f"  - {a}" for a in remaining_actions]) if remaining_actions else "  (none)"
        failed_str = f"  [failed] {failed_action}" if failed_action is not None else "  (no direct action failure; execution was interrupted due to scene change)"

        if self.config.planner_input_mode == PROMPT_MODE_VISION_GOAL:
            return f"""=== REPLANNING REQUIRED ===

ORIGINAL TASK:
{original_goal}

ACTIONS COMPLETED SUCCESSFULLY:
{completed_str}

FAILED ACTION:
{failed_str}

ERROR MESSAGE:
  {error_message}

ACTIONS THAT WERE NOT EXECUTED:
{remaining_str}

## VISUAL CONTEXT:
Use the attached image of the current scene to infer the remaining actions.
Do not rely on any explicit visible-object list text.

Output ONLY the remaining actions needed, starting from action 1:
1. action(args)
2. action(args)
..."""

        if self.config.planner_input_mode == PROMPT_MODE_TEXT_VISIBLE:
            return f"""=== REPLANNING REQUIRED ===

ORIGINAL TASK:
{original_goal}

ACTIONS COMPLETED SUCCESSFULLY:
{completed_str}

FAILED ACTION:
{failed_str}

ERROR MESSAGE:
  {error_message}

ACTIONS THAT WERE NOT EXECUTED:
{remaining_str}

CURRENT STATE SNAPSHOT:
{state_text if state_text else '(unavailable)'}

## COMPACT VISIBILITY:
{visible_objects_text if visible_objects_text else '(unavailable)'}

Plan using ONLY the updated rich visible-only state above. No images are available.
Use the failure reason to infer a corrected remaining action sequence.
Do not repeat completed actions.
Do not output commentary, analysis, or <think> blocks.

Output ONLY the remaining actions needed, starting from action 1:
1. action(args)
2. action(args)
..."""

        return f"""=== REPLANNING REQUIRED ===

ORIGINAL TASK:
{original_goal}

ACTIONS COMPLETED SUCCESSFULLY:
{completed_str}

FAILED ACTION:
{failed_str}

ERROR MESSAGE:
  {error_message}

ACTIONS THAT WERE NOT EXECUTED:
{remaining_str}

CURRENT STATE SNAPSHOT:
{state_text if state_text else '(unavailable)'}

---

Please provide a CORRECTED action sequence to complete the remaining task.
Consider:
1. The actions already completed (don't repeat them)
2. The error message explains what went wrong
3. You may need to add actions to resolve the blockage/issue
4. Prioritize currently unblocked progress first; defer risky retries until later

Output ONLY the remaining actions needed, starting from action 1:
1. action(args)
2. action(args)
..."""

    def _extract_failed_pick_object(self, failed_action: Optional[ActionSkeleton]) -> Optional[str]:
        """
        Extract object name when the failed action is pick(obj).
        """
        if failed_action is None:
            return None
        if failed_action.action_name != "pick" or len(failed_action.args) != 1:
            return None
        return failed_action.args[0]

    def _defer_failed_object_if_first(self,
                                      plan: List[ActionSkeleton],
                                      failed_action: Optional[ActionSkeleton]) -> List[ActionSkeleton]:
        """
        If replan starts by immediately retrying the same failed pick(obj) and there are
        alternative actions available, move that pick/place pair to the end to break loops.
        """
        failed_obj = self._extract_failed_pick_object(failed_action)
        if failed_obj is None or len(plan) < 4:
            return plan

        first = plan[0]
        if first.action_name != "pick" or len(first.args) != 1 or first.args[0] != failed_obj:
            return plan

        if len(plan) < 2:
            return plan
        second = plan[1]
        if second.action_name != "place" or len(second.args) < 1 or second.args[0] != failed_obj:
            return plan

        remaining = plan[2:]
        if not remaining:
            return plan

        print(f"[ReplanPipeline] Deferring immediate retry of failed object '{failed_obj}'")
        return remaining + [first, second]
    
    def plan_initial(self, goal: str) -> PlanResult:
        """Generate initial plan."""
        print(f"\n[ReplanPipeline] Initial planning for: {goal}")
        
        # Capture context
        bundle = self.context_aggregator.create_prompt_bundle(goal)
        
        # Query VLM
        if self.config.use_vision:
            result = self.planner.generate_plan(
                image=bundle.composite_image,
                system_prompt=bundle.system_prompt,
                user_prompt=bundle.user_prompt
            )
        else:
            if hasattr(self.planner, "generate_plan_text_only"):
                result = self.planner.generate_plan_text_only(
                    system_prompt=bundle.system_prompt,
                    user_prompt=bundle.user_prompt
                )
            else:
                result = self.planner.generate_plan(
                    image=bundle.composite_image,
                    system_prompt=bundle.system_prompt,
                    user_prompt=bundle.user_prompt
                )
        
        print(f"[ReplanPipeline] VLM inference time: {result.inference_time:.2f}s")
        
        if result.success:
            print(f"[ReplanPipeline] Initial plan ({len(result.skeleton)} actions):")
            for i, action in enumerate(result.skeleton, 1):
                print(f"  {i}. {action}")
            self._update_live_action_sequence(result.skeleton, 0 if result.skeleton else None)
        else:
            print(f"[ReplanPipeline] Planning failed: {result.error_message}")
            self._update_live_action_sequence([], None)
        
        return result
    
    def plan_replan(self, 
                    original_goal: str,
                    completed_actions: List[ActionSkeleton],
                    failed_action: Optional[ActionSkeleton],
                    error_message: str,
                    remaining_actions: List[ActionSkeleton]) -> PlanResult:
        """Generate a replan after failure."""
        print(f"\n[ReplanPipeline] REPLANNING...")
        print(f"[ReplanPipeline] Completed: {len(completed_actions)} actions")
        print(f"[ReplanPipeline] Failed: {failed_action if failed_action is not None else '(scene-change interruption)'}")
        print(f"[ReplanPipeline] Error: {error_message}")
        
        completed_str = "\n".join([f"  {i+1}. {a}" for i, a in enumerate(completed_actions)]) if completed_actions else "  (none)"
        remaining_str = "\n".join([f"  - {a}" for a in remaining_actions]) if remaining_actions else "  (none)"
        failed_str = (
            f"  [failed] {failed_action}"
            if failed_action is not None
            else "  (no direct action failure; execution was interrupted due to scene change)"
        )
        previous_plan = [str(action) for action in completed_actions]
        if remaining_actions:
            previous_plan.extend(str(action) for action in remaining_actions)
        elif failed_action is not None:
            previous_plan.append(str(failed_action))

        failure_context = (
            "ACTIONS COMPLETED SUCCESSFULLY:\n"
            f"{completed_str}\n\n"
            "FAILED ACTION:\n"
            f"{failed_str}\n\n"
            "ERROR MESSAGE:\n"
            f"  {error_message}\n\n"
            "ACTIONS THAT WERE NOT EXECUTED:\n"
            f"{remaining_str}"
        )
        # Capture fresh replan context (scene may have changed) using the shared prompt builder.
        bundle = self.context_aggregator.create_prompt_bundle(
            goal=original_goal,
            is_replan=True,
            failure_context=failure_context,
            previous_plan=previous_plan,
        )
        replan_user_prompt = bundle.user_prompt

        print(f"\n[ReplanPipeline] Replan prompt:\n{replan_user_prompt[:500]}...")
        
        # Query VLM with replan prompt
        if self.config.use_vision:
            result = self.planner.generate_plan(
                image=bundle.composite_image,
                system_prompt=bundle.system_prompt,
                user_prompt=replan_user_prompt
            )
        else:
            if hasattr(self.planner, "generate_plan_text_only"):
                result = self.planner.generate_plan_text_only(
                    system_prompt=bundle.system_prompt,
                    user_prompt=replan_user_prompt
                )
            else:
                result = self.planner.generate_plan(
                    image=bundle.composite_image,
                    system_prompt=bundle.system_prompt,
                    user_prompt=replan_user_prompt
                )
        
        print(f"[ReplanPipeline] VLM inference time: {result.inference_time:.2f}s")
        
        if result.success:
            print(f"[ReplanPipeline] Replan ({len(result.skeleton)} actions):")
            for i, action in enumerate(result.skeleton, 1):
                print(f"  {i}. {action}")
            self._update_live_action_sequence(result.skeleton, 0 if result.skeleton else None)
        else:
            print(f"[ReplanPipeline] Replanning failed: {result.error_message}")
        
        return result
    
    def execute_plan(self, skeleton: List[ActionSkeleton]) -> ExecutionResult:
        """Execute a plan."""
        print(f"\n[ReplanPipeline] Executing {len(skeleton)} actions...")
        return self.executor.execute_plan(skeleton)
    
    def run(self, goal: str) -> Dict[str, Any]:
        """
        Run the full pipeline with replanning.
        
        Returns summary of execution.
        """
        print("\n" + "=" * 60)
        print("STARTING EXECUTION WITH REPLANNING")
        print("=" * 60)
        print(f"Goal: {goal}")
        print(f"Max replans: {self.config.max_replans}")
        if self.config.visible_objects_only:
            print("Context mode: visible-objects-only (segmentation)")
        if self.config.live_segmentation_view:
            print("Live masks: enabled")
        if self.config.replan_on_discovery:
            print(f"Discovery replan targets: {sorted(self.discovery_targets)}")
        
        self.original_goal = goal
        self.run_started_at = datetime.now()
        self.all_completed_actions = []
        self.cycles = []
        
        # Initial planning
        plan_result = self.plan_initial(goal)
        
        if not plan_result.success:
            return self._generate_summary(success=False, reason="Initial planning failed")
        
        current_plan = plan_result.skeleton
        replan_count = 0
        
        while replan_count <= self.config.max_replans:
            cycle_num = len(self.cycles) + 1
            is_replan = replan_count > 0
            
            print(f"\n{'='*60}")
            print(f"EXECUTION CYCLE {cycle_num} {'(REPLAN)' if is_replan else '(INITIAL)'}")
            print(f"{'='*60}")
            
            # Execute current plan
            exec_result = self.execute_plan(current_plan)
            
            # Get what was executed
            executed_this_cycle = self.executor.get_executed_plan()
            remaining = self.executor.get_remaining_plan()
            
            # Track completed actions globally
            self.all_completed_actions.extend(executed_this_cycle)
            
            # Log this cycle
            cycle = ReplanCycle(
                cycle_number=cycle_num,
                is_replan=is_replan,
                plan=[str(a) for a in current_plan],
                completed_actions=[str(a) for a in executed_this_cycle],
                failed_action=str(exec_result.current_action) if exec_result.current_action else None,
                error_message=exec_result.error_message,
                success=exec_result.status == ExecutionStatus.SUCCESS
            )
            self.cycles.append(cycle)
            
            # Check if successful
            if exec_result.status == ExecutionStatus.SUCCESS:
                print("\n" + "=" * 60)
                print("✓ EXECUTION SUCCESSFUL!")
                print("=" * 60)
                return self._generate_summary(success=True)
            
            # Failed - check if we can replan
            if replan_count >= self.config.max_replans:
                print("\n" + "=" * 60)
                print(f"✗ MAX REPLANS REACHED ({self.config.max_replans})")
                print("=" * 60)
                return self._generate_summary(success=False, reason="Max replans exceeded")
            
            # Attempt replan
            print("\n" + "-" * 60)
            print("INITIATING REPLAN...")
            print("-" * 60)
            
            replan_result = self.plan_replan(
                original_goal=goal,
                completed_actions=self.all_completed_actions,
                failed_action=exec_result.current_action,
                error_message=exec_result.error_message or "Unknown error",
                remaining_actions=remaining
            )
            
            if not replan_result.success or not replan_result.skeleton:
                print("[ReplanPipeline] Replan failed - no valid plan generated")
                return self._generate_summary(success=False, reason="Replan failed")

            replan_result.skeleton = self._defer_failed_object_if_first(
                replan_result.skeleton,
                exec_result.current_action
            )
            
            # Update for next iteration
            current_plan = replan_result.skeleton
            self._update_live_action_sequence(current_plan, 0 if current_plan else None)
            replan_count += 1
            
            # Reset executor for new plan
            self.executor.reset()
        
        return self._generate_summary(success=False, reason="Unexpected exit")
    
    def _generate_summary(self, success: bool, reason: str = None) -> Dict[str, Any]:
        """Generate execution summary."""
        episode_time_s = None
        if self.run_started_at is not None:
            episode_time_s = max(0.0, (datetime.now() - self.run_started_at).total_seconds())

        summary = {
            "goal": self.original_goal,
            "success": success,
            "total_cycles": len(self.cycles),
            "total_replans": sum(1 for c in self.cycles if c.is_replan),
            "total_actions_completed": len(self.all_completed_actions),
            "completed_actions": [str(a) for a in self.all_completed_actions],
            "cycles": [asdict(c) for c in self.cycles],
            "failure_reason": reason if not success else None,
            "episode_time_s": episode_time_s,
            "model_alias": self.model_spec.alias,
            "model_path": self.model_spec.path,
            "model_type": self.model_spec.model_type,
            "prompt_mode": self.config.planner_input_mode,
            "use_vision": self.config.use_vision,
            "visible_objects_only": self.config.visible_objects_only,
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Goal: {self.original_goal}")
        print(f"Success: {'✓ YES' if success else '✗ NO'}")
        print(f"Total cycles: {len(self.cycles)}")
        print(f"Replans used: {summary['total_replans']}/{self.config.max_replans}")
        print(f"Actions completed: {len(self.all_completed_actions)}")
        
        if self.all_completed_actions:
            print("\nCompleted actions:")
            for i, a in enumerate(self.all_completed_actions, 1):
                print(f"  {i}. {a}")
        
        if not success and reason:
            print(f"\nFailure reason: {reason}")
        
        print("=" * 60)
        
        # Save log
        if self.config.save_logs:
            self._save_log(summary)
        
        return summary
    
    def _save_log(self, summary: Dict):
        """Save execution log."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(self.config.log_dir, f"replan_{timestamp}.json")
        
        with open(log_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"[ReplanPipeline] Log saved to: {log_path}")
    
    def shutdown(self):
        """Clean up."""
        print("\n[ReplanPipeline] Shutting down...")
        try:
            self.context_aggregator.shutdown()
        except Exception:
            pass
        if self.env:
            try:
                self.env.pr.stop()
                self.env.pr.shutdown()
            except:
                pass
        print("[ReplanPipeline] Done")


def main():
    parser = argparse.ArgumentParser(description="VLM Pipeline with Replanning")
    parser.add_argument("--goal", type=str, required=True,
                        help="Task goal in natural language")
    parser.add_argument("--max-replans", type=int, default=3,
                        help="Maximum number of replan attempts (default: 3)")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock planner (no GPU required)")
    parser.add_argument("--remote", action="store_true",
                        help="Use remote planner server (set VLM_SERVER_URL env var)")
    parser.add_argument("--remote-url", type=str, default="",
                        help="Remote planner server URL")
    parser.add_argument("--model", type=str, default="qwen-vl",
                        help="Registered model alias or Hugging Face model path")
    parser.add_argument("--model-type", type=str, default="", choices=["", "vlm", "llm"],
                        help="Explicit model type when --model is a custom path")
    parser.add_argument("--planner-input-mode", type=str, default="",
                        choices=["", PROMPT_MODE_VISION_GOAL, PROMPT_MODE_TEXT_VISIBLE],
                        help="Override prompt mode (default: derived from model type)")
    parser.add_argument("--4bit", action="store_true",
                        help="Enable 4-bit quantization for local Hugging Face models")
    parser.add_argument("--text-only", action="store_true",
                        help="Force text-only prompting regardless of model registry defaults")
    parser.add_argument("--visible-only", action="store_true",
                        help="Force segmentation-visible objects only in context")
    parser.add_argument("--live-masks", action="store_true",
                        help="Show live segmentation masks with visible object panel")
    parser.add_argument("--replan-on-discovery", action="store_true",
                        help="Trigger replan when discovery target becomes newly visible")
    parser.add_argument("--discovery-targets", type=str, default="",
                        help="Comma-separated PDDL object names that trigger discovery-based replan (empty = any newly visible object)")
    parser.add_argument("--discovery-failure-code", type=str, default=DISCOVERY_FAILURE_CODE_DEFAULT,
                        help=f"Failure code used when discovery triggers replan (default: {DISCOVERY_FAILURE_CODE_DEFAULT})")
    parser.add_argument("--replan-on-discovery-after-complete", action="store_true",
                        help="Also trigger discovery replan when new object appears right after completing current plan")
    parser.add_argument("--live-mask-stride", type=int, default=5,
                        help="Refresh live mask window every N simulator steps (default: 5)")
    parser.add_argument("--headless", action="store_true",
                        help="Run simulation headless")
    parser.add_argument("--list-models", action="store_true",
                        help="List registered planner models and exit")

    args = parser.parse_args()

    if args.list_models:
        print(format_model_listing())
        return

    spec = resolve_model_spec(args.model, args.model_type)
    planner_input_mode = args.planner_input_mode or spec.prompt_mode
    if args.text_only:
        planner_input_mode = PROMPT_MODE_TEXT_VISIBLE
    use_vision = planner_input_mode == PROMPT_MODE_VISION_GOAL
    visible_only = args.visible_only or planner_input_mode == PROMPT_MODE_TEXT_VISIBLE

    config = ReplanConfig(
        max_replans=args.max_replans,
        use_mock_vlm=args.mock,
        use_remote_vlm=args.remote,
        remote_vlm_url=args.remote_url,
        model=args.model,
        model_type=args.model_type,
        planner_input_mode=planner_input_mode,
        use_4bit=args.__dict__["4bit"],
        use_vision=use_vision,
        visible_objects_only=visible_only,
        live_segmentation_view=args.live_masks,
        replan_on_discovery=args.replan_on_discovery,
        discovery_targets=[s.strip() for s in args.discovery_targets.split(",") if s.strip()],
        discovery_failure_code=args.discovery_failure_code,
        replan_on_discovery_after_plan_complete=args.replan_on_discovery_after_complete,
        live_view_update_stride=max(1, args.live_mask_stride),
        headless=args.headless,
    )

    pipeline = VLMReplanningPipeline(config)

    try:
        if not pipeline.initialize():
            print("Failed to initialize pipeline!")
            return

        result = pipeline.run(args.goal)

        print("\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)
        print(json.dumps(result, indent=2))
    finally:
        pipeline.shutdown()

if __name__ == "__main__":
    main()

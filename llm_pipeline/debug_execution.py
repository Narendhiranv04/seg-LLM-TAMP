#!/usr/bin/env python3
"""
Debug script for llm_pipeline Execution and Failure Capture.
Simulates LLM output and tests the executor + failure checker.
"""

import os
import sys
import numpy as np
import time

# Set scene environment variable BEFORE importing environment dependencies
os.environ['KITCHEN_SCENE_FILE'] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'task1_variation2.ttt')
os.environ['HEADLESS'] = 'False'  # Show GUI for this test

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_pipeline.pipeline import LLMOnlyReplanningPipeline, LLMPipelineConfig
from llm_pipeline.pipeline_types import PlanResult, DirectAction, FailureStage, FailureSource

class MockPlanner:
    def __init__(self):
        self.actions = []
        self.loaded = True

    def set_actions(self, actions_list):
        """actions_list: list of (name, args_tuple)"""
        self.actions = [DirectAction(name, args) for name, args in actions_list]

    def plan(self, bundle):
        return PlanResult(
            success=True,
            actions=self.actions,
            raw_output="\n".join([str(a) for a in self.actions]),
            inference_time=0.1
        )

    def load_model(self):
        return True

def debug_execution_flow():
    print("\n" + "=" * 60)
    print("DEBUGGING LLM_PIPELINE EXECUTION & FAILURE CAPTURE")
    print("=" * 60)

    # 1. Setup Environment
    print("[Debug] Initializing Environment...")
    # Load env (scene file was already set at module load)
    from rlbench_kitchen_streams import ENV
    env = ENV
    
    # 2. Setup Pipeline with Mock Planner
    print("[Debug] Initializing Pipeline...")
    config = LLMPipelineConfig(
        headless=False,
        visible_objects_only=True,
        enable_vision=False
    )
    
    pipeline = LLMOnlyReplanningPipeline(config=config)
    pipeline.initialize(env=env)
    mock_planner = MockPlanner()
    pipeline.planner = mock_planner
    
    # 3. Define a clean execution plan
    # 3. Define a clean execution plan for K2
    print("[Debug] Setting up execution plan (K2 Sequence)...")
    mock_planner.set_actions([
        # Subtask 1: Cupboard Mug -> Placement Boundary
        ('move', ('mug3',)),
        ('pick', ('mug3',)),
        ('move', ('placement_boundary',)),
        ('place', ('mug3', 'placement_boundary')),
        
        # # Subtask 2: Table grocery -> Cupboard
        ('move', ('sugar',)),
        ('pick', ('sugar',)),
        ('move', ('cupboard_boundary',)),
        ('place', ('sugar', 'cupboard_boundary')),

        # Subtask 3: Box-top mug -> Placement Boundary
        ('move', ('mug5',)),
        ('pick', ('mug5',)),
        ('move', ('placement_boundary',)),
        ('place', ('mug5', 'placement_boundary')),

        # Subtask 4: Open box lid
        ('move', ('box_lid',)),
        ('open', ('box_lid',)),

        # Subtask 5: Box grocery -> cupboard
        ('move', ('soup',)),
        ('pick', ('soup',)),
        ('move', ('cupboard_boundary',)),
        ('place', ('soup', 'cupboard_boundary')),

        # Subtask 6: Table mug 1 -> box
        ('move', ('mug2',)),
        ('pick', ('mug2',)),
        ('move', ('box_boundary',)),
        ('place', ('mug2', 'box_boundary')),

        # Subtask 7: Table mug 2 -> box
        ('move', ('mug3',)),
        ('pick', ('mug3',)),
        ('move', ('box_boundary',)),
        ('place', ('mug3', 'box_boundary'))
    ])

    # 4. Run one cycle of the pipeline
    print("\n[Debug] Starting Pipeline Execution Cycle...")
    goal = "Move the mug on the box to the placement boundary"
    
    # We use run which handles planning + execution loop
    cycle_result = pipeline.run(goal_text=goal)
    
    print("\n" + "=" * 60)
    print("EXECUTION CYCLE RESULT")
    print("=" * 60)
    print(f"Success: {cycle_result['success']}")
    if cycle_result.get('last_failure_event'):
        failure_event = cycle_result['last_failure_event']
        print(f"Failure ID: {failure_event.get('failure_id')}")
        print(f"Failure Source: {failure_event.get('source')}")
        print(f"Failure Message: {failure_event.get('message')}")
    else:
        print("No failure event captured.")
    print("=" * 60)

    # 5. Verification
    print("\n[Debug] Verification Summary:")
    if cycle_result['success']:
        print("✅ Success: Pipeline successfully executed the full K1 sequence.")
    else:
        print("❌ Failure: Pipeline failed during execution.")
        
    print("Completed Primitive Actions:", cycle_result.get('completed_actions', []))

    if env:
        # Give user a moment to see the GUI
        time.sleep(2)
        env.pr.stop()
        env.pr.shutdown()

if __name__ == "__main__":
    debug_execution_flow()

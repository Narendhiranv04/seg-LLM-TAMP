#!/usr/bin/env python3
"""
Debug script for llm_pipeline State Recognition.
Isolates the GeometricContextBuilder and Segmentation adapter.
"""

import os
import sys
import numpy as np
import time

# Set scene environment variable BEFORE importing environment dependencies
os.environ['KITCHEN_SCENE_FILE'] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'task1_variation2.ttt')
os.environ['HEADLESS'] = 'True'

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_pipeline.pipeline import LLMOnlyReplanningPipeline, LLMPipelineConfig
from llm_pipeline.geometric_builder import GeometricContextBuilder
from llm_pipeline.segmentation_adapter import SegmentationEvidenceAdapter
from rlbench_kitchen_streams import ENV

def debug_state_recognition():
    print("\n" + "=" * 60)
    print("DEBUGGING LLM_PIPELINE STATE RECOGNITION")
    print("=" * 60)

    # 1. Setup Environment
    print("[Debug] Initializing Environment...")
    env = ENV
    
    # 2. Setup Modules
    print("[Debug] Initializing Modules...")
    adapter = SegmentationEvidenceAdapter(env=env, visible_objects_only=True)
    builder = GeometricContextBuilder()
    builder.set_env(env) # Ensure it has env for geometric resolution if needed
    
    # 3. Settle Environment
    print("[Debug] Settling Environment...")
    for _ in range(50):
        env.pr.step()
    
    # 4. Capture Scene State (Logic from LLMOnlyReplanningPipeline)
    print("[Debug] Capturing Scene State...")
    snapshot = adapter.refresh_visibility(event='debug')
    
    # Helper to build a unified SceneState from current sensors.
    def build_scene_state():
        from llm_pipeline.pipeline_types import SceneState
        cur_snapshot = adapter.capture_snapshot(event='debug-planning')
        
        # Extract 3D poses and region bboxes if available
        pose_map = {}
        region_map = {}
        detector = getattr(adapter, 'detector', None)
        if detector:
            for obj_name in cur_snapshot.visible_objects:
                pose = detector.get_object_pose(obj_name)
                if pose:
                    pose_map[obj_name] = pose
            
            for region_name in cur_snapshot.supported_regions:
                scene_name = region_name
                if region_name == 'box-top' or region_name == 'box-inside':
                    scene_name = 'box_base'
                elif region_name == 'shelf-lower':
                    scene_name = 'cupboard'
                
                bb = detector.get_bounding_box(scene_name)
                if bb:
                    region_map[region_name] = (np.array(bb[0]), np.array(bb[1]))

        return SceneState(
            frame_index=cur_snapshot.frame_index,
            visible_objects=cur_snapshot.visible_objects,
            valid_regions=cur_snapshot.supported_regions,
            masks=cur_snapshot.gripper_evidence.get('masks', {}),
            pose_map=pose_map,
            region_map=region_map,
            gripper_state={'status': 'empty', 'holding': None}
        )

    state = build_scene_state()

    print("\n--- CAPTURED SCENE STATE ---")
    print(f"Visible Objects: {state.visible_objects}")
    print(f"Valid Regions: {state.valid_regions}")
    print(f"Pose Map Keys: {list(state.pose_map.keys())}")

    # 5. Build Prompt Bundle (The "State Recognition" Output)
    print("\n[Debug] Building Prompt Bundle...")
    goal = "Move all mugs to placement_boundary"
    bundle = builder.build_bundle(state=state, goal_text=goal)

    print("\n" + "=" * 60)
    print("GENERATED OBSERVATION TEXT")
    print("=" * 60)
    # The builder puts state text in user_prompt
    print(bundle.user_prompt)
    print("=" * 60)

    # 6. Success/Failure verification
    print("\n[Debug] Verification Summary:")
    if state.visible_objects:
        print(f"✅ Success: Detected {len(state.visible_objects)} objects.")
    else:
        print("❌ Failure: No objects detected.")
    
    if "mug" in bundle.user_prompt.lower():
        print("✅ Success: Prompt contains object references.")
    else:
        print("❌ Failure: Prompt missing object details.")

    if env:
        env.pr.stop()
        env.pr.shutdown()

if __name__ == "__main__":
    debug_state_recognition()

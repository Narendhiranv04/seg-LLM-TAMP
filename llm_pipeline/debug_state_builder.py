#!/usr/bin/env python3
"""Debug the exact scene-state path used by the maintained LLM pipeline."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.canonical_variants import get_variant_spec  # noqa: E402
from llm_pipeline.pipeline import LLMPipelineConfig, LLMOnlyReplanningPipeline  # noqa: E402


class NoOpPlanner:
    """Loaded planner stub so pipeline initialization does not load a model."""

    loaded = True

    def load_model(self) -> bool:
        return True


def _load_env(task_family: str, scene_path: str, headless: bool):
    os.environ["HEADLESS"] = "True" if headless else "False"

    if task_family == "kitchen":
        os.environ["KITCHEN_SCENE_FILE"] = scene_path
        return importlib.import_module("rlbench_kitchen_streams").ENV

    if task_family == "grill":
        grill_dir = REPO_ROOT / "grill_task2"
        sys.path.insert(0, str(grill_dir))
        os.environ["GRILL_SCENE_FILE"] = scene_path
        return importlib.import_module("grill_task_streams").ENV

    raise ValueError(f"Unsupported task family: {task_family}")


def _state_summary(state) -> dict[str, Any]:
    snapshot = getattr(state, "_original_snapshot", None)
    object_evidence = {}
    if snapshot is not None:
        for name, evidence in snapshot.object_evidence.items():
            object_evidence[name] = {
                "mask_regions": list(evidence.mask_regions),
                "camera_hits": list(evidence.camera_hits),
                "pixel_count": int(evidence.pixel_count),
                "camera_pixels": dict(evidence.camera_pixels),
                "bbox_cameras": sorted(evidence.bbox.keys()),
                "centroid_cameras": sorted(evidence.centroid.keys()),
                "newly_visible": bool(evidence.newly_visible),
                "region_votes": dict(evidence.region_votes),
                "gripper_proximity": evidence.gripper_proximity,
            }

    return {
        "frame_index": int(state.frame_index),
        "visible_objects": list(state.visible_objects),
        "valid_regions": list(state.valid_regions),
        "pose_map_keys": sorted(state.pose_map.keys()),
        "region_map_keys": sorted(state.region_map.keys()),
        "gripper_state": dict(state.gripper_state),
        "snapshot": None if snapshot is None else {
            "visible_objects": list(snapshot.visible_objects),
            "newly_visible_objects": list(snapshot.newly_visible_objects),
            "visible_regions": list(snapshot.visible_regions),
            "supported_regions": list(snapshot.supported_regions),
            "object_evidence": object_evidence,
            "gripper_evidence": dict(snapshot.gripper_evidence),
        },
    }


def debug_state_recognition(args: argparse.Namespace) -> int:
    variant = get_variant_spec(args.variant)
    scene_path = str(Path(args.scene_path).resolve()) if args.scene_path else variant.scene_path
    goal_text = args.goal or variant.goal_text

    print("=" * 72)
    print("DEBUGGING LLM_PIPELINE SCENE STATE")
    print("=" * 72)
    print(f"Variant: {variant.variant_id}")
    print(f"Task family: {variant.task_family}")
    print(f"Scene: {scene_path}")
    print(f"Headless: {args.headless}")

    env = None
    pipeline = None
    try:
        print("\n[Debug] Loading environment...")
        env = _load_env(variant.task_family, scene_path, headless=args.headless)

        print("[Debug] Initializing normal pipeline setup with no model load...")
        config = LLMPipelineConfig(
            headless=args.headless,
            live_segmentation_view=False,
            visible_objects_only=True,
            enable_vision=False,
            use_remote_planner=False,
        )
        pipeline = LLMOnlyReplanningPipeline(config=config, planner=NoOpPlanner())
        if not pipeline.initialize(env=env):
            print("[Debug] ERROR: pipeline.initialize() returned False")
            return 1

        if args.settle_steps > 0:
            print(f"[Debug] Settling for {args.settle_steps} extra steps...")
            for _ in range(args.settle_steps):
                env.pr.step()

        print("[Debug] Calling LLMOnlyReplanningPipeline._build_scene_state()...")
        state = pipeline._build_scene_state()
        summary = _state_summary(state)

        print("\n--- SCENE STATE SUMMARY ---")
        print(f"Visible objects: {summary['visible_objects']}")
        print(f"Valid regions: {summary['valid_regions']}")
        print(f"Pose map keys: {summary['pose_map_keys']}")
        print(f"Region map keys: {summary['region_map_keys']}")

        snapshot = summary["snapshot"] or {}
        print(f"Visible regions: {snapshot.get('visible_regions', [])}")
        print(f"Newly visible objects: {snapshot.get('newly_visible_objects', [])}")

        print("\n--- OBJECT EVIDENCE ---")
        object_evidence = snapshot.get("object_evidence", {})
        if object_evidence:
            for name, evidence in object_evidence.items():
                print(
                    f"{name}: regions={evidence['mask_regions']} "
                    f"cameras={evidence['camera_hits']} pixels={evidence['pixel_count']}"
                )
        else:
            print("(none)")

        if not args.skip_prompt:
            print("\n[Debug] Building prompt bundle through pipeline.context_builder...")
            bundle = pipeline.context_builder.build_bundle(state=state, goal_text=goal_text)
            print("\n--- GENERATED USER PROMPT ---")
            print(bundle.user_prompt)

        if args.json:
            print("\n--- JSON SUMMARY ---")
            print(json.dumps(summary, indent=2, sort_keys=True))

        if state.visible_objects:
            print(f"\n[Debug] PASS: detected {len(state.visible_objects)} visible object(s).")
            return 0

        print("\n[Debug] FAIL: no visible objects detected.")
        return 2

    finally:
        if pipeline is not None:
            pipeline.shutdown()
        elif env is not None:
            env.pr.stop()
            env.pr.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build scene state using the exact llm_pipeline state-building path."
    )
    parser.add_argument(
        "--variant",
        default="K2",
        choices=["K1", "K2", "K3", "G1", "G2", "G3"],
        help="Canonical scene variant to load.",
    )
    parser.add_argument(
        "--scene-path",
        default="",
        help="Optional scene path override for the selected variant.",
    )
    parser.add_argument(
        "--goal",
        default="",
        help="Optional goal text override for prompt generation.",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run CoppeliaSim headless.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=0,
        help="Extra simulator steps after pipeline initialization.",
    )
    parser.add_argument(
        "--skip-prompt",
        action="store_true",
        help="Only print state evidence; do not build the prompt bundle.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON summary after the text summary.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(debug_state_recognition(parse_args()))

#!/usr/bin/env python3
"""Debug the exact scene-state path used by the maintained LLM pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from llm_pipeline.region_geometry import NON_REGION_OBJECTS


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.canonical_variants import get_variant_spec  # noqa: E402
from llm_pipeline.pipeline import LLMPipelineConfig, LLMOnlyReplanningPipeline  # noqa: E402


DEFAULT_REPORT_DIR = REPO_ROOT / "Status Doc" / "scene_state_reports"


class NoOpPlanner:
    """Loaded planner stub so pipeline initialization does not load a model."""

    loaded = True

    def load_model(self) -> bool:
        return True


class NoOpExecutor:
    """Executor stub for recognition-only debug runs."""

    held_object = None

    def set_env(self, env) -> None:
        self.env = env

    def set_step_callback(self, callback) -> None:
        self.step_callback = callback

    def set_action_start_callback(self, callback) -> None:
        self.action_start_callback = callback


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


def _lid_joint_debug_info(env) -> dict[str, Any]:
    lid_joint = getattr(env, "lid_joint", None)
    current_angle = None
    target_position = None
    if lid_joint is not None:
        try:
            current_angle = float(lid_joint.get_joint_position())
        except Exception:
            current_angle = None
        try:
            target_position = float(lid_joint.get_joint_target_position())
        except Exception:
            target_position = None

    closed_angle = getattr(env, "_closed_lid_angle", None)
    try:
        closed_angle = None if closed_angle is None else float(closed_angle)
    except Exception:
        closed_angle = None

    initial_angle = getattr(env, "_initial_lid_angle", None)
    try:
        initial_angle = None if initial_angle is None else float(initial_angle)
    except Exception:
        initial_angle = None

    delta_from_closed = None
    if current_angle is not None and closed_angle is not None:
        delta_from_closed = current_angle - closed_angle

    return {
        "current_angle": current_angle,
        "target_position": target_position,
        "closed_reference_angle": closed_angle,
        "initial_angle_after_env_load": initial_angle,
        "delta_from_closed_reference": delta_from_closed,
        "delta_target_from_closed_reference": (
            None
            if target_position is None or closed_angle is None
            else target_position - closed_angle
        ),
        "delta_current_from_target": (
            None
            if current_angle is None or target_position is None
            else current_angle - target_position
        ),
        "preserve_scene_lid_pose": bool(getattr(env, "_preserve_scene_lid_pose", False)),
    }


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
        "pddl_state": list(state.pddl_state),
        "object_region_map": dict(getattr(state, "object_region_map", {}) or {}),
        "object_region_descriptions": dict(getattr(state, "object_region_descriptions", {}) or {}),
        "gripper_state": dict(state.gripper_state),
        "snapshot": None if snapshot is None else {
            "visible_objects": list(snapshot.visible_objects),
            "newly_visible_objects": list(snapshot.newly_visible_objects),
            "visible_regions": list(snapshot.visible_regions),
            "supported_regions": list(snapshot.supported_regions),
            "object_region_map": dict(getattr(snapshot, "object_region_map", {}) or {}),
            "object_region_descriptions": dict(getattr(snapshot, "object_region_descriptions", {}) or {}),
            "object_evidence": object_evidence,
            "gripper_evidence": dict(snapshot.gripper_evidence),
        },
    }


def _detector_debug_info(pipeline) -> dict[str, Any]:
    adapter = getattr(pipeline, "segmentation_adapter", None)
    detector = getattr(adapter, "detector", None)
    if detector is None:
        return {}
    return {
        "min_pixels_per_camera": getattr(detector, "min_pixels_per_camera", None),
        "persistence_frames": getattr(detector, "persistence_frames", None),
    }


def _format_list(values: list[str]) -> str:
    return ", ".join(values) if values else "(none)"


def _format_scene_report(
    *,
    variant_id: str,
    task_family: str,
    scene_path: str,
    headless: bool,
    settle_steps: int,
    summary: dict[str, Any],
) -> str:
    snapshot = summary["snapshot"] or {}
    object_evidence = snapshot.get("object_evidence", {})
    visible_objects = summary["visible_objects"]
    visible_regions = snapshot.get("visible_regions", [])
    object_region_map = summary.get("object_region_map", {})
    object_region_descriptions = summary.get("object_region_descriptions", {})
    lid_joint = summary.get("lid_joint", {})
    detector = summary.get("detector", {})

    if not visible_objects:
        result = "FAIL"
        result_note = "No visible objects were detected."
    elif any(name not in object_region_map and name not in NON_REGION_OBJECTS for name in visible_objects):
        result = "PARTIAL"
        result_note = "Objects were detected, but one or more objects have missing geometric region assignment."
    else:
        result = "PASS"
        result_note = "Objects were detected with geometric region assignments."

    lines = [
        f"# Scene State Report: {variant_id}",
        "",
        "## Run",
        "",
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Variant: {variant_id}",
        f"- Task family: {task_family}",
        f"- Scene: `{scene_path}`",
        f"- Headless: {headless}",
        f"- Settle steps: {settle_steps}",
        f"- Result: {result}",
        f"- Summary: {result_note}",
        "",
        "## Detected State",
        "",
        f"- Visible objects: {_format_list(visible_objects)}",
        f"- Newly visible objects: {_format_list(snapshot.get('newly_visible_objects', []))}",
        f"- Visible regions: {_format_list(visible_regions)}",
        f"- Supported regions: {_format_list(snapshot.get('supported_regions', []))}",
        f"- Pose map objects: {_format_list(summary['pose_map_keys'])}",
        f"- Gripper: {summary['gripper_state'].get('status', 'unknown')}",
        f"- Detector min pixels per camera: {detector.get('min_pixels_per_camera')}",
        f"- Detector persistence frames: {detector.get('persistence_frames')}",
        f"- Lid joint current angle: {lid_joint.get('current_angle')}",
        f"- Lid joint target position: {lid_joint.get('target_position')}",
        f"- Lid joint closed reference: {lid_joint.get('closed_reference_angle')}",
        f"- Lid joint initial angle after env load: {lid_joint.get('initial_angle_after_env_load')}",
        f"- Lid joint delta from closed reference: {lid_joint.get('delta_from_closed_reference')}",
        f"- Lid joint target delta from closed reference: {lid_joint.get('delta_target_from_closed_reference')}",
        f"- Lid joint current delta from target: {lid_joint.get('delta_current_from_target')}",
        f"- Preserve scene lid pose: {lid_joint.get('preserve_scene_lid_pose')}",
        "",
        "## Semantic Facts",
        "",
        _format_list(summary.get("pddl_state", [])),
        "",
        "## Geometric Object Locations",
        "",
    ]

    if visible_objects:
        lines.extend([
            "| Object | Geometric region | Description | Visual regions |",
            "| --- | --- | --- | --- |",
        ])
        for name in sorted(visible_objects):
            evidence = object_evidence.get(name, {})
            geometric_region = object_region_map.get(name)
            description = object_region_descriptions.get(name)
            if name in NON_REGION_OBJECTS:
                geometric_region = geometric_region or "(not applicable)"
                description = description or "fixture/openable object"
            lines.append(
                "| "
                f"{name} | "
                f"{geometric_region or '(unresolved)'} | "
                f"{description or '(none)'} | "
                f"{_format_list(evidence.get('mask_regions', []))} |"
            )
    else:
        lines.append("(none)")

    lines.extend([
        "",
        "## Object Evidence",
        "",
    ])

    if object_evidence:
        lines.extend([
            "| Object | Visual regions | Cameras | Pixels | Region votes |",
            "| --- | --- | --- | ---: | --- |",
        ])
        for name in sorted(object_evidence):
            evidence = object_evidence[name]
            region_votes = ", ".join(
                f"{region}: {votes:g}"
                for region, votes in sorted(evidence.get("region_votes", {}).items())
            )
            lines.append(
                "| "
                f"{name} | "
                f"{_format_list(evidence.get('mask_regions', []))} | "
                f"{_format_list(evidence.get('camera_hits', []))} | "
                f"{evidence.get('pixel_count', 0)} | "
                f"{region_votes or '(none)'} |"
            )
    else:
        lines.append("(none)")

    lines.extend([
        "",
        "## Review Notes",
        "",
    ])
    if object_evidence:
        for name in sorted(object_evidence):
            evidence = object_evidence[name]
            regions = evidence.get("mask_regions", [])
            if len(regions) == 0:
                lines.append(f"- {name}: no visual region evidence.")
            elif len(regions) > 1:
                lines.append(f"- {name}: multiple visual region candidates ({_format_list(regions)}).")
        if lines[-1] == "":
            lines.append("- No obvious region ambiguity in this snapshot.")
    else:
        lines.append("- No object evidence available.")

    lines.append("")
    return "\n".join(lines)


def _write_reports(
    *,
    args: argparse.Namespace,
    variant_id: str,
    task_family: str,
    scene_path: str,
    summary: dict[str, Any],
) -> tuple[Path, Path | None]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{variant_id}_scene_state"
    report_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json" if args.json else None

    report = _format_scene_report(
        variant_id=variant_id,
        task_family=task_family,
        scene_path=scene_path,
        headless=args.headless,
        settle_steps=args.settle_steps,
        summary=summary,
    )
    report_path.write_text(report, encoding="utf-8")

    if json_path is not None:
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return report_path, json_path


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
            task_family=variant.task_family,
            scene_path=scene_path,
        )
        pipeline = LLMOnlyReplanningPipeline(config=config, planner=NoOpPlanner(), executor=NoOpExecutor())
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
        summary["lid_joint"] = _lid_joint_debug_info(env)
        summary["detector"] = _detector_debug_info(pipeline)

        snapshot = summary["snapshot"] or {}

        if not args.skip_prompt:
            print("\n[Debug] Building prompt bundle through pipeline.context_builder...")
            bundle = pipeline.context_builder.build_bundle(state=state, goal_text=goal_text)
            print("\n--- GENERATED USER PROMPT ---")
            print(bundle.user_prompt)

        report_path, json_path = _write_reports(
            args=args,
            variant_id=variant.variant_id,
            task_family=variant.task_family,
            scene_path=scene_path,
            summary=summary,
        )
        print(f"\n[Debug] Report written: {report_path}")
        if json_path is not None:
            print(f"[Debug] JSON written: {json_path}")

        if state.visible_objects:
            visible_regions = snapshot.get("visible_regions", [])
            print(
                f"\n[Debug] PASS: detected {len(state.visible_objects)} visible object(s), "
                f"{len(visible_regions)} visible region(s)."
            )
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
        default=False,
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
        help="Do not build the prompt bundle.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Save the machine-readable JSON summary next to the Markdown report.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="Directory for saved Markdown and optional JSON reports.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(debug_state_recognition(parse_args()))

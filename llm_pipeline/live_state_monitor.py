#!/usr/bin/env python3
"""Live scene-state monitor for manual LLM-readiness verification."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.canonical_variants import get_variant_spec
from llm_pipeline.debug_state_builder import (
    DEFAULT_REPORT_DIR,
    NoOpExecutor,
    NoOpPlanner,
    _format_scene_report,
    _load_env,
    _state_summary,
)
from llm_pipeline.pipeline import LLMPipelineConfig, LLMOnlyReplanningPipeline
from llm_pipeline.region_geometry import NON_REGION_OBJECTS


DEFAULT_LIVE_REPORT_DIR = DEFAULT_REPORT_DIR.parent / "live_scene_state_reports"


def _step_simulation(env: Any, *, headless: bool) -> None:
    env.pr.step()
    if not headless and hasattr(env.pr, "step_ui"):
        try:
            env.pr.step_ui()
        except Exception:
            pass


def _advance_simulation_until_next_poll(
    env: Any,
    *,
    seconds: float,
    headless: bool,
    step_sleep: float,
) -> None:
    if seconds <= 0:
        _step_simulation(env, headless=headless)
        return

    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        _step_simulation(env, headless=headless)
        time.sleep(min(max(step_sleep, 0.0), remaining))


def _compact_signature(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the stable state subset used to detect meaningful live changes."""
    visible_objects = [
        name
        for name in summary.get("visible_objects", [])
        if name not in NON_REGION_OBJECTS
    ]
    object_region_map = summary.get("object_region_map", {}) or {}
    return {
        "visible_objects": sorted(visible_objects),
        "object_region_map": {
            name: object_region_map.get(name)
            for name in sorted(visible_objects)
            if object_region_map.get(name)
        },
        "gripper_state": dict(summary.get("gripper_state", {}) or {}),
    }


def _signature_changes(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> list[str]:
    if previous is None:
        return ["Initial frame captured."]

    changes = []
    old_objects = set(previous.get("visible_objects", []))
    new_objects = set(current.get("visible_objects", []))
    for name in sorted(new_objects - old_objects):
        changes.append(f"{name}: appeared")
    for name in sorted(old_objects - new_objects):
        changes.append(f"{name}: disappeared")

    old_regions = previous.get("object_region_map", {}) or {}
    new_regions = current.get("object_region_map", {}) or {}
    for name in sorted(old_objects | new_objects):
        old_region = old_regions.get(name)
        new_region = new_regions.get(name)
        if old_region != new_region:
            changes.append(
                f"{name}: region changed {old_region or '(none)'} -> {new_region or '(none)'}"
            )

    old_gripper = previous.get("gripper_state", {}) or {}
    new_gripper = current.get("gripper_state", {}) or {}
    if old_gripper != new_gripper:
        changes.append(f"gripper: {old_gripper} -> {new_gripper}")

    return changes


def _format_live_report(
    *,
    frame_number: int,
    changes: list[str],
    variant_id: str,
    task_family: str,
    scene_path: str,
    headless: bool,
    settle_steps: int,
    summary: dict[str, Any],
) -> str:
    base_report = _format_scene_report(
        variant_id=variant_id,
        task_family=task_family,
        scene_path=scene_path,
        headless=headless,
        settle_steps=settle_steps,
        summary=summary,
    )
    delta_lines = [
        "",
        "## Live Monitor",
        "",
        f"- Captured frame: {frame_number}",
        "",
        "## Changes Since Previous Capture",
        "",
    ]
    delta_lines.extend(f"- {change}" for change in changes)
    delta_lines.append("")
    return base_report + "\n".join(delta_lines)


def _write_live_capture(
    *,
    output_dir: Path,
    save_json: bool,
    frame_number: int,
    changes: list[str],
    variant_id: str,
    task_family: str,
    scene_path: str,
    headless: bool,
    settle_steps: int,
    summary: dict[str, Any],
    signature: dict[str, Any],
) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{variant_id}_frame{frame_number:04d}_scene_state"
    report_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json" if save_json else None

    report = _format_live_report(
        frame_number=frame_number,
        changes=changes,
        variant_id=variant_id,
        task_family=task_family,
        scene_path=scene_path,
        headless=headless,
        settle_steps=settle_steps,
        summary=summary,
    )
    report_path.write_text(report, encoding="utf-8")

    if json_path is not None:
        json_payload = {
            "frame_number": int(frame_number),
            "changes": list(changes),
            "signature": signature,
            "summary": summary,
        }
        json_path.write_text(json.dumps(json_payload, indent=2, sort_keys=True), encoding="utf-8")

    return report_path, json_path


def monitor_live_state(args: argparse.Namespace) -> int:
    variant = get_variant_spec(args.variant)
    scene_path = str(Path(args.scene_path).resolve()) if args.scene_path else variant.scene_path
    output_dir = Path(args.output_dir).resolve()

    print("=" * 72)
    print("LIVE LLM_PIPELINE SCENE STATE MONITOR")
    print("=" * 72)
    print(f"Variant: {variant.variant_id}")
    print(f"Task family: {variant.task_family}")
    print(f"Scene: {scene_path}")
    print(f"Headless: {args.headless}")
    print(f"Poll interval: {args.poll_interval:.3f}s")
    print(f"Simulation step sleep: {args.step_sleep:.3f}s")
    print(f"Output dir: {output_dir}")

    env = None
    pipeline = None
    start_time = time.monotonic()
    captured_frames = 0
    polls = 0
    previous_signature = None

    try:
        env = _load_env(variant.task_family, scene_path, headless=args.headless)
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
            print("[Live] ERROR: pipeline.initialize() returned False")
            return 1

        if args.settle_steps > 0:
            print(f"[Live] Settling for {args.settle_steps} extra steps...")
            for _ in range(args.settle_steps):
                _step_simulation(env, headless=args.headless)

        while True:
            if args.max_seconds > 0 and (time.monotonic() - start_time) >= args.max_seconds:
                print("[Live] Reached max seconds.")
                break
            if args.max_polls > 0 and polls >= args.max_polls:
                print("[Live] Reached max polls.")
                break
            if args.max_captures > 0 and captured_frames >= args.max_captures:
                print("[Live] Reached max captures.")
                break

            polls += 1
            state = pipeline._build_scene_state()
            summary = _state_summary(state)
            signature = _compact_signature(summary)
            changes = _signature_changes(previous_signature, signature)

            if changes:
                captured_frames += 1
                report_path, json_path = _write_live_capture(
                    output_dir=output_dir,
                    save_json=args.json,
                    frame_number=captured_frames,
                    changes=changes,
                    variant_id=variant.variant_id,
                    task_family=variant.task_family,
                    scene_path=scene_path,
                    headless=args.headless,
                    settle_steps=args.settle_steps,
                    summary=summary,
                    signature=signature,
                )
                print(f"[Live] Capture {captured_frames}: {report_path}")
                if json_path is not None:
                    print(f"[Live] JSON {captured_frames}: {json_path}")
                for change in changes:
                    print(f"  - {change}")
                previous_signature = signature

            _advance_simulation_until_next_poll(
                env,
                seconds=args.poll_interval,
                headless=args.headless,
                step_sleep=args.step_sleep,
            )

        print(f"[Live] Done. Polls={polls}, captures={captured_frames}")
        return 0

    finally:
        if pipeline is not None:
            pipeline.shutdown()
        elif env is not None:
            env.pr.stop()
            env.pr.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor scene state and write a report whenever object-region state changes."
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
        "--poll-interval",
        type=float,
        default=0.5,
        help="Seconds between scene-state polls.",
    )
    parser.add_argument(
        "--step-sleep",
        type=float,
        default=0.02,
        help="Seconds to sleep between simulator steps while waiting for the next poll.",
    )
    parser.add_argument(
        "--max-captures",
        type=int,
        default=0,
        help="Stop after this many changed-state captures. 0 means unlimited.",
    )
    parser.add_argument(
        "--max-polls",
        type=int,
        default=0,
        help="Stop after this many polls. 0 means unlimited.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Stop after this many seconds. 0 means unlimited.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Save a machine-readable JSON capture next to each Markdown report.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_LIVE_REPORT_DIR),
        help="Directory for live Markdown and optional JSON reports.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(monitor_live_state(parse_args()))

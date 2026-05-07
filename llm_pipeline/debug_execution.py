#!/usr/bin/env python3
"""Executor-only debug harness for llm_pipeline.

This bypasses model calls by feeding hand-written DirectAction sequences through
LLMOnlyReplanningPipeline and DirectPrimitiveExecutor.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.canonical_variants import get_variant_spec
from llm_pipeline.executable_symbols import build_runtime_symbol_registry
from llm_pipeline.pipeline_types import (
    DirectAction,
    FailureEvent,
    FailureSource,
    FailureStage,
    PlanResult,
    PromptBundle,
    SegmentationObjectEvidence,
    SegmentationSnapshot,
)


ACTION_LINE = re.compile(r"^(move|pick|open|close)\(([^,)]+)\)$|^place\(([^,]+),\s*([^)]+)\)$|^move$")
SEQUENCE_SEPARATOR = "---"
DEFAULT_SEQUENCE_DIR = ROOT_DIR / "llm_pipeline" / "debug_sequences"
PROGRESS_KEYS = {
    "status",
    "last_run",
    "last_exit_code",
    "last_success",
    "completed_actions",
    "last_completed_action",
    "failure_id",
    "failure_source",
    "failure_message",
}


@dataclass(frozen=True)
class DebugSequence:
    name: str
    variant: str
    actions: tuple[DirectAction, ...]
    goal: str = ""
    source_path: Optional[Path] = None


class MockPlanner:
    def __init__(self, actions: Iterable[DirectAction]):
        self.actions = list(actions)
        self.loaded = True

    def plan(self, bundle):
        del bundle
        return PlanResult(
            success=True,
            actions=list(self.actions),
            raw_output="\n".join(str(action) for action in self.actions),
            inference_time=0.0,
        )

    def load_model(self):
        return True


class ExecutorOnlyContextBuilder:
    def set_env(self, env) -> None:
        self.env = env

    def set_symbol_registry(self, symbol_registry) -> None:
        self.symbol_registry = symbol_registry

    def build_bundle(self, state, goal_text, failure_event=None, previous_actions=None, icl_mode="zero_shot"):
        return PromptBundle(
            goal_text=goal_text,
            system_prompt="Executor-only debug run.",
            user_prompt="Hand-written actions are supplied by MockPlanner.",
            visible_objects=list(state.visible_objects),
            valid_regions=list(state.valid_regions),
            icl_mode=icl_mode,
            previous_actions=tuple(previous_actions or ()),
            failure_context=failure_event.message if failure_event else None,
        )


class ExecutorOnlySegmentationAdapter:
    def __init__(self, sequence: "DebugSequence"):
        self.sequence = sequence
        self.env = None
        self.symbol_registry = None
        self.frame_index = 0

    def set_env(self, env) -> None:
        self.env = env

    def set_symbol_registry(self, symbol_registry) -> None:
        self.symbol_registry = symbol_registry

    def reset_tracking(self) -> None:
        self.frame_index = 0

    def refresh_visibility(self, event: str = ""):
        snapshot = self._build_snapshot()
        return {
            "visible_objects": list(snapshot.visible_objects),
            "newly_visible_objects": [],
            "visible_regions": list(snapshot.visible_regions),
        }

    def capture_snapshot(self, event: str = ""):
        del event
        return self._build_snapshot()

    def update_live_segmentation_view(self):
        return set()

    def set_live_action_sequence(self, actions, current_action_index=None, current_action_label=None) -> None:
        del actions, current_action_index, current_action_label

    def shutdown(self) -> None:
        return None

    def _build_snapshot(self) -> SegmentationSnapshot:
        self.frame_index += 1
        registry = self.symbol_registry or build_runtime_symbol_registry(env=self.env)
        valid_objects = set(registry.objects)
        valid_regions = set(registry.regions)
        objects = []
        regions = []
        for action in self.sequence.actions:
            if action.action_name in {"pick", "open", "close"} and action.args:
                objects.append(action.args[0])
            elif action.action_name == "place":
                objects.append(action.args[0])
                regions.append(action.args[1])
            elif action.action_name == "move" and action.args:
                target = action.args[0]
                if target in valid_objects:
                    objects.append(target)
                elif target in valid_regions:
                    regions.append(target)

        visible_objects = _ordered_unique(name for name in objects if name in valid_objects)
        visible_regions = _ordered_unique(name for name in regions if name in valid_regions)
        object_evidence = {
            name: SegmentationObjectEvidence(name=name, visible=True)
            for name in visible_objects
        }
        return SegmentationSnapshot(
            frame_index=self.frame_index,
            visible_objects=visible_objects,
            newly_visible_objects=[],
            object_evidence=object_evidence,
            gripper_evidence={},
            supported_regions=list(registry.regions),
            visible_regions=visible_regions,
            object_region_map={},
            object_region_descriptions={},
        )


class ExecutorOnlyFailureChecker:
    def __init__(self, adapter: ExecutorOnlySegmentationAdapter):
        self.adapter = adapter
        self.env = None

    def capture_snapshot(self, event: str = ""):
        return self.adapter.capture_snapshot(event=event)

    def precheck(self, action, held_object, snapshot, last_action_name=None):
        del action, held_object, snapshot, last_action_name
        return None

    def postcheck(self, action, held_object, snapshot):
        del action, held_object, snapshot
        return None

    def classify_runtime_error(self, action, error_message):
        return FailureEvent(
            failure_id="TAMP_EXECUTION_ERROR",
            stage=FailureStage.AFTER_EXECUTION,
            source=FailureSource.EXECUTOR,
            action=str(action),
            evidence={"error": error_message},
            should_replan=False,
            message=error_message,
        )


def _ordered_unique(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _transfer(obj_name: str, target_region: str) -> List[DirectAction]:
    return [
        DirectAction("move", (obj_name,)),
        DirectAction("pick", (obj_name,)),
        DirectAction("move", (target_region,)),
        DirectAction("place", (obj_name, target_region)),
    ]


def _lid(action_name: str, lid_name: str) -> List[DirectAction]:
    return [
        DirectAction("move", (lid_name,)),
        DirectAction(action_name, (lid_name,)),
    ]


def _actions(*chunks: Iterable[DirectAction]) -> tuple[DirectAction, ...]:
    flattened: List[DirectAction] = []
    for chunk in chunks:
        flattened.extend(chunk)
    return tuple(flattened)


BUILTIN_SEQUENCES = {
    "K1": DebugSequence(
        name="K1_gt_as_is",
        variant="K1",
        goal="Replay K1 ground-truth concrete sequence through llm_pipeline executor.",
        actions=_actions(
            _transfer("mug2", "placement_boundary"),
            _transfer("mug3", "placement_boundary"),
            _lid("open", "box_lid"),
            _transfer("soup", "cupboard_lower"),
            _transfer("spam", "cupboard_lower"),
            _transfer("mug2", "box_storage"),
            _transfer("mug3", "box_storage"),
        ),
    ),
    "K2": DebugSequence(
        name="K2_gt_as_is",
        variant="K2",
        goal="Replay K2 ground-truth concrete sequence through llm_pipeline executor.",
        actions=_actions(
            _transfer("mug3", "placement_boundary"),
            _transfer("sugar", "cupboard_lower"),
            _transfer("mug2", "placement_boundary"),
            _lid("open", "box_lid"),
            _transfer("soup", "cupboard_lower"),
            _transfer("mug2", "box_storage"),
            _transfer("mug3", "box_storage"),
        ),
    ),
    "K3": DebugSequence(
        name="K3_gt_as_is",
        variant="K3",
        goal="Replay K3 ground-truth concrete sequence through llm_pipeline executor.",
        actions=_actions(
            _transfer("mug3", "placement_boundary"),
            _transfer("sugar", "cupboard_lower"),
            _transfer("mug2", "placement_boundary"),
            _lid("open", "box_lid"),
            _transfer("soup", "cupboard_lower"),
            _transfer("mug2", "box_storage"),
            _transfer("mug3", "box_storage"),
            _transfer("mug1", "box_storage"),
        ),
    ),
    "G1": DebugSequence(
        name="G1_gt_as_is",
        variant="G1",
        goal="Replay G1 ground-truth concrete sequence through llm_pipeline executor.",
        actions=_actions(
            _lid("open", "grill_lid"),
            _transfer("spam", "table"),
            _transfer("chicken", "inside_grill"),
            _lid("close", "grill_lid"),
            _transfer("plate", "plate_boundary"),
            _lid("open", "grill_lid"),
            _transfer("chicken", "plate-top"),
        ),
    ),
    "G2": DebugSequence(
        name="G2_gt_as_is",
        variant="G2",
        goal="Replay G2 ground-truth concrete sequence through llm_pipeline executor.",
        actions=_actions(
            _lid("open", "grill_lid"),
            _transfer("plate", "plate_boundary"),
            _transfer("steak", "plate-top"),
            _transfer("chicken", "inside_grill"),
            _transfer("steak1", "inside_grill"),
            _lid("close", "grill_lid"),
            _lid("open", "grill_lid"),
            _transfer("chicken", "plate-top"),
            _transfer("steak1", "plate-top"),
        ),
    ),
    "G3": DebugSequence(
        name="G3_gt_as_is",
        variant="G3",
        goal="Replay G3 ground-truth concrete sequence through llm_pipeline executor.",
        actions=_actions(
            _lid("open", "grill_lid"),
            _transfer("spam", "table"),
            _transfer("plate", "plate_boundary"),
            _transfer("steak", "plate-top"),
            _transfer("chicken", "inside_grill"),
            _transfer("steak1", "inside_grill"),
            _lid("close", "grill_lid"),
            _lid("open", "grill_lid"),
            _transfer("chicken", "plate-top"),
            _transfer("steak1", "plate-top"),
        ),
    ),
}


def _parse_action_line(line: str, line_number: int) -> DirectAction:
    match = ACTION_LINE.fullmatch(line)
    if not match:
        raise ValueError(
            f"Line {line_number}: expected move, move(target), pick(obj), "
            "place(obj, region), open(lid), or close(lid)"
        )
    if line == "move":
        return DirectAction("move", ())
    if line.startswith("place("):
        return DirectAction("place", (match.group(3).strip(), match.group(4).strip()))
    return DirectAction(match.group(1), (match.group(2).strip(),))


def _sequence_from_block(
    block: List[tuple[int, str]],
    fallback_variant: str,
    index: int,
    source_path: Optional[Path] = None,
) -> DebugSequence:
    name = f"sequence_{index}"
    variant = fallback_variant
    goal = ""
    actions: List[DirectAction] = []

    for line_number, raw_line in block:
        line = raw_line.strip()
        lower = line.lower()
        if lower.startswith("[") and lower.endswith("]"):
            name = line[1:-1].strip() or name
            continue
        if lower.startswith("name:"):
            name = line.split(":", 1)[1].strip() or name
            continue
        if lower.startswith("variant:"):
            variant = line.split(":", 1)[1].strip().upper() or variant
            continue
        if lower.startswith("goal:"):
            goal = line.split(":", 1)[1].strip()
            continue
        if ":" in line:
            key = line.split(":", 1)[0].strip().lower()
            if key in PROGRESS_KEYS:
                continue
        actions.append(_parse_action_line(line, line_number))

    if not actions:
        raise ValueError(f"Sequence '{name}' has no actions.")
    get_variant_spec(variant)
    return DebugSequence(
        name=name,
        variant=variant,
        goal=goal,
        actions=tuple(actions),
        source_path=source_path,
    )


def load_sequences(path: Path, default_variant: str) -> List[DebugSequence]:
    blocks: List[List[tuple[int, str]]] = [[]]
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == SEQUENCE_SEPARATOR:
            if blocks[-1]:
                blocks.append([])
            continue
        blocks[-1].append((line_number, line))

    return [
        _sequence_from_block(
            block,
            fallback_variant=default_variant,
            index=index,
            source_path=path,
        )
        for index, block in enumerate(blocks, start=1)
        if block
    ]


def _default_sequence_file(variant: str, sequence_dir: Path) -> Path:
    return sequence_dir / f"{variant.upper()}_gt_as_is.txt"


def _load_default_sequence(variant: str, sequence_dir: Path) -> DebugSequence:
    path = _default_sequence_file(variant, sequence_dir)
    if path.exists():
        sequences = load_sequences(path, default_variant=variant)
        if len(sequences) != 1:
            raise ValueError(f"Default sequence file {path} must contain exactly one sequence.")
        return sequences[0]
    return BUILTIN_SEQUENCES[variant]


def _configure_scene_env(sequence: DebugSequence, headless: bool) -> None:
    spec = get_variant_spec(sequence.variant)
    os.environ["HEADLESS"] = "True" if headless else "False"
    os.environ["COPPELIASIM_HEADLESS"] = "1" if headless else "0"
    if spec.task_family == "grill":
        os.environ["GRILL_SCENE_FILE"] = spec.scene_path
        os.environ["GRILL_ALLOW_SCENE_OVERRIDE"] = "True"
        os.environ["GRILL_SCENE_FILE_OVERRIDE"] = spec.scene_path
        os.environ.setdefault("GRILL_LID_TRAVEL_ANGLE", f"{math.radians(95.0):.6f}")
        os.environ.setdefault("GRILL_PRESERVE_SCENE_LID_POSE", "True")
        os.environ.setdefault("GRILL_LID_CLOSED_ANGLE", "0.0")
        os.environ.setdefault("GRILL_LID_OPEN_ANGLE", f"{math.radians(95.0):.6f}")
        os.environ.setdefault("GRILL_LID_AUTOCALIBRATE", "False")
        grill_dir = str(ROOT_DIR / "grill_task2")
        if grill_dir not in sys.path:
            sys.path.insert(0, grill_dir)
    else:
        os.environ["KITCHEN_SCENE_FILE"] = spec.scene_path


def _load_env_for_sequence(sequence: DebugSequence):
    spec = get_variant_spec(sequence.variant)
    if spec.task_family == "grill":
        script_path = ROOT_DIR / "grill_task2" / "ground_truth_orchestrator_variation1 copy.py"
        module_name = "grill_gt"
        if module_name in sys.modules:
            grill_gt = sys.modules[module_name]
        else:
            module_spec = importlib.util.spec_from_file_location(module_name, script_path)
            if module_spec is None or module_spec.loader is None:
                raise RuntimeError(f"Could not load grill GT module from {script_path}")
            grill_gt = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = grill_gt
            module_spec.loader.exec_module(grill_gt)
        _prepare_grill_env_like_gt(grill_gt, grill_gt.ENV)
        return grill_gt.ENV

    from rlbench_kitchen_streams import ENV

    return ENV


def _prepare_grill_env_like_gt(grill_gt, env) -> None:
    """Run the GT grill startup prep without executing any task sequence."""
    if getattr(env, "_llm_debug_gt_startup_prepared", False):
        return

    pr = env.pr
    print("[debug-startup] Preparing grill env with GT startup path...")
    grill_gt._discover_lid_joint_handle(env, pr)
    grill_gt._discover_active_handle(env, pr)
    grill_gt._capture_handle_anchor(env)
    grill_gt._restore_handle_anchor(env, pr)

    if getattr(grill_gt, "LID_AUTOCALIBRATE", False):
        grill_gt._calibrate_lid_from_waypoints(env, pr)

    pre = grill_gt._get_lid_joint_angle(env)
    d_close_pre = grill_gt._handle_waypoint_distance(env, grill_gt.CLOSE_WP_NAME)
    d_open_pre = grill_gt._handle_waypoint_distance(env, grill_gt.OPEN_WP_NAME)
    preserve_scene_lid_pose = bool(getattr(env, "_preserve_scene_lid_pose", False))
    scene_closed_angle = getattr(env, "_closed_lid_angle", None)

    if preserve_scene_lid_pose and scene_closed_angle is not None:
        grill_gt.LID_CLOSED_ANGLE = float(scene_closed_angle)
        grill_gt.LID_OPEN_ANGLE = grill_gt._open_angle_for_closed_pose(
            env,
            grill_gt.LID_CLOSED_ANGLE,
        )
        os.environ["GRILL_LID_CLOSED_ANGLE"] = f"{grill_gt.LID_CLOSED_ANGLE:.6f}"
        os.environ["GRILL_LID_OPEN_ANGLE"] = f"{grill_gt.LID_OPEN_ANGLE:.6f}"
    elif getattr(grill_gt, "USE_INITIAL_LID_AS_CLOSED", False) and pre is not None:
        closed_est, inferred = grill_gt._estimate_closed_angle_from_current(
            env,
            pr,
            pre,
            d_close_pre,
            d_open_pre,
        )
        if inferred:
            grill_gt.LID_OPEN_ANGLE = float(pre)
            grill_gt.LID_CLOSED_ANGLE = float(closed_est)
        else:
            grill_gt.LID_CLOSED_ANGLE = float(pre)
            grill_gt.LID_OPEN_ANGLE = grill_gt._open_angle_for_closed_pose(
                env,
                grill_gt.LID_CLOSED_ANGLE,
            )

    grill_gt._enforce_min_open_travel()
    print(
        f"[debug-startup] Lid targets from scene: "
        f"open={grill_gt.LID_OPEN_ANGLE:.3f}, closed={grill_gt.LID_CLOSED_ANGLE:.3f}"
    )

    try:
        env.stabilize_startup_state(steps=15)
        if preserve_scene_lid_pose:
            print("[debug-startup] Preserving scene-authored lid pose.")
        elif getattr(grill_gt, "KEEP_LID_COLLISION_OFF_UNTIL_OPEN", False):
            grill_gt._set_lid_servo_lock(env, True)
            try:
                env.set_lid_collision_enabled(False)
            except Exception:
                pass
            grill_gt._set_lid_joint_angle(env, pr, grill_gt.LID_CLOSED_ANGLE, steps=90)
            grill_gt._force_lid_closed(env, pr, steps=100)
        else:
            grill_gt._set_lid_servo_lock(env, True)
            reached, final_closed = grill_gt._close_lid_until_contact(
                env,
                pr,
                grill_gt.LID_CLOSED_ANGLE,
                steps=130,
                backoff=0.05,
            )
            if final_closed is not None:
                grill_gt.LID_CLOSED_ANGLE = float(final_closed)
                grill_gt._enforce_min_open_travel()
            print(
                f"[debug-startup] contact-safe close: reached={reached} | "
                f"closed_angle={grill_gt.LID_CLOSED_ANGLE:.3f}, "
                f"open_angle={grill_gt.LID_OPEN_ANGLE:.3f}"
            )

        if not getattr(grill_gt, "KEEP_LID_COLLISION_OFF_UNTIL_OPEN", False):
            try:
                env.set_lid_collision_enabled(True)
            except Exception:
                pass
    except Exception as exc:
        print(f"[debug-startup] Warning: startup stabilization failed: {exc}")

    grill_gt.step(pr, 3)
    grill_gt.go_home(env, pr)

    if preserve_scene_lid_pose:
        print("[debug-startup] Preserving scene-authored lid pose after home.")
    elif getattr(grill_gt, "KEEP_LID_COLLISION_OFF_UNTIL_OPEN", False):
        try:
            env.set_lid_collision_enabled(False)
        except Exception:
            pass
        grill_gt._set_lid_joint_angle(env, pr, grill_gt.LID_CLOSED_ANGLE, steps=35)
        grill_gt._force_lid_closed(env, pr, steps=50)
    else:
        reached_home, final_home_closed = grill_gt._close_lid_until_contact(
            env,
            pr,
            grill_gt.LID_CLOSED_ANGLE,
            steps=90,
            backoff=0.05,
        )
        if final_home_closed is not None:
            grill_gt.LID_CLOSED_ANGLE = float(final_home_closed)
            grill_gt.LID_OPEN_ANGLE = grill_gt._open_angle_for_closed_pose(
                env,
                grill_gt.LID_CLOSED_ANGLE,
            )
            grill_gt._enforce_min_open_travel()
        print(
            f"[debug-startup] post-home contact-safe close: reached={reached_home} | "
            f"closed_angle={grill_gt.LID_CLOSED_ANGLE:.3f}, "
            f"open_angle={grill_gt.LID_OPEN_ANGLE:.3f}"
        )

    post_home = grill_gt._get_lid_joint_angle(env)
    if post_home is not None:
        print(f"[debug-startup] Lid angle after home lock: {post_home:.3f} rad")
    grill_gt.step(pr, 3)
    env._llm_debug_gt_startup_prepared = True


def run_pipeline_sequence(
    sequence: DebugSequence,
    headless: bool,
    max_replans: int,
    live_masks: bool,
    use_scene_state: bool,
) -> dict:
    from llm_pipeline.pipeline import LLMOnlyReplanningPipeline, LLMPipelineConfig

    spec = get_variant_spec(sequence.variant)
    _configure_scene_env(sequence, headless=headless)
    env = _load_env_for_sequence(sequence)
    planner = MockPlanner(sequence.actions)
    config = LLMPipelineConfig(
        headless=headless,
        visible_objects_only=True,
        enable_vision=False,
        max_replans=max_replans,
        task_family=spec.task_family,
        scene_path=spec.scene_path,
        live_segmentation_view=bool(live_masks and not headless),
    )
    segmentation_adapter = None
    failure_checker = None
    context_builder = None
    if not use_scene_state:
        segmentation_adapter = ExecutorOnlySegmentationAdapter(sequence)
        failure_checker = ExecutorOnlyFailureChecker(segmentation_adapter)
        context_builder = ExecutorOnlyContextBuilder()

    pipeline = LLMOnlyReplanningPipeline(
        config=config,
        planner=planner,
        segmentation_adapter=segmentation_adapter,
        failure_checker=failure_checker,
        context_builder=context_builder,
    )
    try:
        if not pipeline.initialize(env=env):
            raise RuntimeError("pipeline_initialize_failed")
        return pipeline.run(goal_text=sequence.goal or spec.goal_text)
    finally:
        pipeline.shutdown()


def run_executor_direct(sequence: DebugSequence, headless: bool) -> dict:
    spec = get_variant_spec(sequence.variant)
    _configure_scene_env(sequence, headless=headless)
    env = _load_env_for_sequence(sequence)

    # Import only after the GT-style env is constructed. This keeps executor-direct
    # startup as close as possible to the working GT entrypoint.
    from llm_pipeline.executor import DirectPrimitiveExecutor

    executor = DirectPrimitiveExecutor(env=env)
    try:
        executor.reset_episode()
        execution = executor.execute_actions(
            list(sequence.actions),
            failure_checker=None,
            pre_action_checks_enabled=False,
            post_action_checks_enabled=False,
        )
        return {
            "success": bool(execution.success),
            "goal_text": sequence.goal or spec.goal_text,
            "mode": "executor_direct",
            "completed_actions": list(execution.completed_actions),
            "remaining_actions": list(execution.remaining_actions),
            "held_object": execution.held_object,
            "last_failure_event": (
                execution.last_failure_event.to_dict()
                if execution.last_failure_event is not None
                else None
            ),
            "failure_reason": execution.error_message,
        }
    finally:
        try:
            env.pr.stop()
        except Exception:
            pass
        try:
            env.pr.shutdown()
        except Exception:
            pass


def _progress_metadata(sequence: DebugSequence, result: Optional[dict], status: str) -> List[str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"status: {status}",
        f"last_run: {timestamp}",
    ]
    if result is None:
        return lines

    completed = list(result.get("completed_actions", []))
    success = bool(result.get("success", False))
    lines.extend(
        [
            f"last_exit_code: {0 if success else 1}",
            f"last_success: {success}",
            f"completed_actions: {len(completed)}/{len(sequence.actions)}",
            f"last_completed_action: {completed[-1] if completed else ''}",
        ]
    )
    failure_event = result.get("last_failure_event") or {}
    if failure_event:
        lines.extend(
            [
                f"failure_id: {failure_event.get('failure_id', '')}",
                f"failure_source: {failure_event.get('source', '')}",
                f"failure_message: {failure_event.get('message', '')}",
            ]
        )
    else:
        lines.extend(
            [
                "failure_id: ",
                "failure_source: ",
                "failure_message: ",
            ]
        )
    return lines


def _line_key(line: str) -> str:
    if ":" not in line:
        return ""
    return line.split(":", 1)[0].strip().lower()


def _update_sequence_file_progress(
    sequence: DebugSequence,
    result: Optional[dict] = None,
    status: str = "running",
) -> None:
    path = sequence.source_path
    if path is None or not path.exists():
        return

    lines = path.read_text().splitlines()
    blocks: List[tuple[int, int]] = []
    start = 0
    for index, line in enumerate(lines):
        if line.strip() == SEQUENCE_SEPARATOR:
            blocks.append((start, index))
            start = index + 1
    blocks.append((start, len(lines)))

    target_block = None
    for start, end in blocks:
        block = lines[start:end]
        block_name = ""
        block_variant = ""
        for raw_line in block:
            key = _line_key(raw_line)
            if key == "name":
                block_name = raw_line.split(":", 1)[1].strip()
            elif key == "variant":
                block_variant = raw_line.split(":", 1)[1].strip().upper()
        if block_name == sequence.name:
            target_block = (start, end)
            break
        if not block_name and block_variant == sequence.variant:
            target_block = (start, end)

    if target_block is None:
        return

    start, end = target_block
    block = [
        line
        for line in lines[start:end]
        if _line_key(line) not in PROGRESS_KEYS
    ]

    insert_at = 0
    for index, line in enumerate(block):
        key = _line_key(line)
        if key in {"name", "variant", "goal"}:
            insert_at = index + 1

    progress_lines = _progress_metadata(sequence, result=result, status=status)
    updated_block = block[:insert_at] + progress_lines + block[insert_at:]
    updated_lines = lines[:start] + updated_block + lines[end:]
    path.write_text("\n".join(updated_lines).rstrip() + "\n")


def _select_sequences(args: argparse.Namespace) -> List[DebugSequence]:
    sequence_dir = Path(args.sequence_dir)
    if args.sequence_file:
        variant = args.variant.upper()
        sequences = load_sequences(Path(args.sequence_file), default_variant=variant)
    elif args.all:
        sequences = [
            _load_default_sequence(variant, sequence_dir)
            for variant in sorted(BUILTIN_SEQUENCES)
        ]
    else:
        variant = args.variant.upper()
        if variant not in BUILTIN_SEQUENCES:
            raise ValueError(f"Unknown variant '{args.variant}'. Expected one of: {', '.join(sorted(BUILTIN_SEQUENCES))}")
        sequences = [_load_default_sequence(variant, sequence_dir)]

    if args.sequence:
        wanted = args.sequence.strip()
        sequences = [sequence for sequence in sequences if sequence.name == wanted]
        if not sequences:
            raise ValueError(f"No sequence named '{wanted}' found.")
    return sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run llm_pipeline executor with hand-written actions.")
    parser.add_argument("--variant", default="K2", help="Scene variant to load.")
    parser.add_argument("--all", action="store_true", help="Run default sequence files for all variants.")
    parser.add_argument(
        "--sequence-dir",
        default=str(DEFAULT_SEQUENCE_DIR),
        help="Directory containing per-variant debug sequence files.",
    )
    parser.add_argument("--sequence-file", default="", help="Optional text file containing one or more sequences.")
    parser.add_argument("--sequence", default="", help="Optional sequence name to run from --sequence-file.")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--gui", action="store_true", help="Run with simulator GUI.")
    display.add_argument("--headless", action="store_true", help="Run without simulator GUI.")
    parser.add_argument("--no-live-masks", action="store_true", help="Disable the separate live segmentation window.")
    parser.add_argument(
        "--with-scene-state",
        action="store_true",
        help="In --pipeline-wrapper mode, use real segmentation/failure checks.",
    )
    parser.add_argument(
        "--pipeline-wrapper",
        action="store_true",
        help="Run through LLMOnlyReplanningPipeline. Default directly tests DirectPrimitiveExecutor.",
    )
    parser.add_argument("--max-replans", type=int, default=0, help="Replan retries after executor failure.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    headless = bool(args.headless)
    if not args.headless and not args.gui:
        headless = False

    sequences = _select_sequences(args)
    if not args.sequence and len(sequences) > 1:
        names = [sequence.name for sequence in sequences]
        if len(set(names)) != len(names):
            raise ValueError("Sequence names must be unique when running a multi-sequence file.")

        overall_status = 0
        for sequence in sequences:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--variant",
                sequence.variant,
                "--sequence",
                sequence.name,
                "--max-replans",
                str(args.max_replans),
            ]
            if args.sequence_file:
                command.extend(["--sequence-file", args.sequence_file])
            else:
                command.extend(["--sequence-dir", args.sequence_dir])
            if args.no_live_masks:
                command.append("--no-live-masks")
            if args.with_scene_state:
                command.append("--with-scene-state")
            if args.pipeline_wrapper:
                command.append("--pipeline-wrapper")
            command.append("--headless" if headless else "--gui")
            print("\n" + "#" * 72)
            print(f"RUNNING SEQUENCE IN FRESH PROCESS: {sequence.name} ({sequence.variant})")
            print("#" * 72)
            result = subprocess.run(command, cwd=str(ROOT_DIR), text=True)
            if result.returncode != 0:
                overall_status = result.returncode
        return overall_status

    for sequence in sequences:
        print("\n" + "=" * 72)
        print(f"EXECUTOR DEBUG: {sequence.name} ({sequence.variant})")
        print("=" * 72)
        print(f"Mode: {'pipeline-wrapper' if args.pipeline_wrapper else 'executor-direct'}")
        print(f"Action count: {len(sequence.actions)}")
        _update_sequence_file_progress(sequence, status="running")
        try:
            if args.pipeline_wrapper:
                result = run_pipeline_sequence(
                    sequence,
                    headless=headless,
                    max_replans=args.max_replans,
                    live_masks=not args.no_live_masks,
                    use_scene_state=args.with_scene_state,
                )
            else:
                result = run_executor_direct(sequence, headless=headless)
        except Exception as exc:
            result = {
                "success": False,
                "completed_actions": [],
                "last_failure_event": {
                    "failure_id": "DEBUG_EXECUTION_ERROR",
                    "source": "debug_execution",
                    "message": str(exc),
                },
                "failure_reason": str(exc),
            }
            _update_sequence_file_progress(sequence, result=result, status="failed")
            raise
        else:
            _update_sequence_file_progress(
                sequence,
                result=result,
                status="passed" if result["success"] else "failed",
            )
        print("\n" + "-" * 72)
        print(f"Success: {result['success']}")
        if result.get("last_failure_event"):
            failure_event = result["last_failure_event"]
            print(f"Failure ID: {failure_event.get('failure_id')}")
            print(f"Failure Source: {failure_event.get('source')}")
            print(f"Failure Message: {failure_event.get('message')}")
        else:
            print("No failure event captured.")
        print("Completed Primitive Actions:")
        for action in result.get("completed_actions", []):
            print(f"  {action}")
        if not result["success"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

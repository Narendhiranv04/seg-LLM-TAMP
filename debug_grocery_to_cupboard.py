#!/usr/bin/env python3
"""
Run an isolated grocery -> cupboard_boundary transfer test.

The full kitchen GT has two grocery-to-cupboard paths:
  - grocery in box  -> cupboard_boundary, via run_box_pick_place
  - grocery on table -> cupboard_boundary, via run_standard_pick_place

This harness lets either path be tested without running the whole kitchen
sequence. For box-source groceries it can open the box first, matching the full
GT ordering before the grocery-in-box task.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np


ROOT_DIR = os.path.dirname(__file__)
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "grocery_to_cupboard")
GROCERY_CANDIDATES = ("soup", "spam", "mustard", "sugar", "crackers")
KITCHEN_VARIANTS = {
    "K1": os.path.join(ROOT_DIR, "task1_variation1.ttt"),
    "K2": os.path.join(ROOT_DIR, "task1_variation2.ttt"),
    "K3": os.path.join(ROOT_DIR, "task1_variation3.ttt"),
}


def _configure_qt(headless):
    os.environ["COPPELIASIM_HEADLESS"] = "1" if headless else "0"
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
    coppelia_root = os.environ.get("COPPELIASIM_ROOT") or os.path.expanduser("~/CoppeliaSim")
    os.environ.setdefault("QT_PLUGIN_PATH", coppelia_root)
    for candidate in [
        os.path.join(coppelia_root, "platforms"),
        os.path.join(coppelia_root, "Qt", "plugins", "platforms"),
        os.path.join(coppelia_root, "qt", "plugins", "platforms"),
    ]:
        if os.path.isdir(candidate):
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", candidate)
            break


def _parse_args():
    parser = argparse.ArgumentParser(description="Test grocery -> cupboard_boundary transfer")
    parser.add_argument(
        "--variant",
        choices=sorted(KITCHEN_VARIANTS),
        default="K1",
        help="Kitchen variant scene to load when --scene is not provided",
    )
    parser.add_argument("--scene", default="", help="Explicit .ttt scene path. Overrides --variant.")
    parser.add_argument(
        "--source",
        choices=("auto", "box", "table", "both"),
        default="auto",
        help="Which grocery source path to test",
    )
    parser.add_argument(
        "--grocery",
        default="",
        help="Explicit grocery name. Defaults to runtime source selection.",
    )
    parser.add_argument(
        "--target-region",
        default="cupboard_boundary",
        choices=("cupboard_boundary", "cupboard_boundary_top"),
        help="Cupboard region to place into",
    )
    parser.add_argument("--output", default="", help="Summary JSON path")
    parser.add_argument("--headless", action="store_true", help="Run CoppeliaSim headless")
    parser.add_argument("--keep-open", action="store_true", help="Keep simulator open after the run")
    parser.add_argument("--record-video", action="store_true", help="Record videos during the isolated task")
    parser.add_argument("--video-dir", default="", help="Video directory when --record-video is set")
    parser.add_argument("--settle-steps", type=int, default=50, help="Initial physics settle steps")
    open_group = parser.add_mutually_exclusive_group()
    open_group.add_argument(
        "--open-box-first",
        action="store_true",
        help="Open the box before running a box-source grocery transfer",
    )
    open_group.add_argument(
        "--no-open-box-first",
        action="store_true",
        help="Do not open the box before a box-source grocery transfer",
    )
    return parser.parse_args()


def _prepare_env(args):
    scene_path = os.path.abspath(args.scene) if args.scene else os.path.abspath(KITCHEN_VARIANTS[args.variant])
    _configure_qt(args.headless)
    os.environ["HEADLESS"] = "True" if args.headless else "False"
    os.environ["KITCHEN_SCENE_FILE"] = scene_path
    os.environ["GT_RECORD_VIDEO"] = "1" if args.record_video else "0"
    sys.path.append(os.path.join(ROOT_DIR, "pddlstream"))
    return scene_path


def _world_bounds(region_obj):
    min_x, max_x, min_y, max_y, min_z, max_z = region_obj.get_bounding_box()
    rx, ry, rz = region_obj.get_position()
    return (
        rx + min_x,
        rx + max_x,
        ry + min_y,
        ry + max_y,
        rz + min_z,
        rz + max_z,
    )


def _is_in_region(env, obj, region_name, tol=0.02):
    region = env.regions.get(region_name)
    if region is None or obj is None:
        return False
    x, y, z = obj.get_position()
    min_x, max_x, min_y, max_y, min_z, max_z = _world_bounds(region)
    return (
        (min_x - tol) <= x <= (max_x + tol)
        and (min_y - tol) <= y <= (max_y + tol)
        and (min_z - tol) <= z <= (max_z + tol)
    )


def _region_center(env, region_name):
    region = env.regions.get(region_name)
    if region is None:
        return None
    min_x, max_x, min_y, max_y, min_z, max_z = _world_bounds(region)
    return np.array(
        [(min_x + max_x) * 0.5, (min_y + max_y) * 0.5, (min_z + max_z) * 0.5],
        dtype=float,
    )


def _obj_name(obj, fallback):
    try:
        return obj.get_name()
    except Exception:
        return fallback


def _discover_unique_objects(env, candidate_names):
    seen = set()
    names = []
    for name in candidate_names:
        obj = env.get_object(name)
        if obj is None:
            continue
        try:
            key = obj.get_handle()
        except Exception:
            key = _obj_name(obj, name)
        if key in seen:
            continue
        seen.add(key)
        names.append(_obj_name(obj, name))
    return names


def _select_table_grocery(env):
    table_first = []
    fallback = []
    for name in _discover_unique_objects(env, GROCERY_CANDIDATES):
        obj = env.get_object(name)
        in_cupboard = _is_in_region(env, obj, "cupboard_boundary") or _is_in_region(
            env, obj, "cupboard_boundary_top"
        )
        in_box = _is_in_region(env, obj, "box_boundary") or _is_in_region(env, obj, "box-inside")
        if _is_in_region(env, obj, "table") and (not in_cupboard) and (not in_box):
            table_first.append(name)
        elif (not in_cupboard) and (not in_box):
            fallback.append(name)
    if table_first:
        return table_first[0]
    if fallback:
        return fallback[0]
    return None


def _select_box_grocery(env):
    in_box = []
    remaining = []
    for name in _discover_unique_objects(env, GROCERY_CANDIDATES):
        obj = env.get_object(name)
        in_cupboard = _is_in_region(env, obj, "cupboard_boundary") or _is_in_region(
            env, obj, "cupboard_boundary_top"
        )
        if _is_in_region(env, obj, "box_boundary") or _is_in_region(env, obj, "box-inside"):
            in_box.append(name)
        elif not in_cupboard:
            remaining.append(name)
    if in_box:
        return in_box[0]

    box_center = _region_center(env, "box_boundary")
    if box_center is not None and remaining:
        return min(
            remaining,
            key=lambda name: float(
                np.linalg.norm(np.array(env.get_object(name).get_position(), dtype=float) - box_center)
            ),
        )
    return None


def _select_grocery(env, source, explicit_grocery=""):
    if explicit_grocery:
        obj = env.get_object(explicit_grocery)
        if obj is None:
            raise RuntimeError(f"Requested grocery '{explicit_grocery}' was not found")
        if source == "auto":
            source = "box" if (
                _is_in_region(env, obj, "box_boundary")
                or _is_in_region(env, obj, "box-inside")
            ) else "table"
        return _obj_name(obj, explicit_grocery), source

    if source == "box":
        selected = _select_box_grocery(env)
        if selected is None:
            raise RuntimeError("No grocery candidate found for source=box")
        return selected, "box"
    if source == "table":
        selected = _select_table_grocery(env)
        if selected is None:
            raise RuntimeError("No grocery candidate found for source=table")
        return selected, "table"

    selected = _select_box_grocery(env)
    if selected is not None:
        return selected, "box"
    selected = _select_table_grocery(env)
    if selected is not None:
        return selected, "table"
    raise RuntimeError("No grocery candidate found")


def _pose_list(obj):
    if obj is None:
        return None
    return [float(v) for v in obj.get_pose()]


def _pos_list(obj):
    if obj is None:
        return None
    return [float(v) for v in obj.get_position()]


def _default_output_path(args, source):
    scene_key = args.variant if not args.scene else os.path.splitext(os.path.basename(args.scene))[0]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(DEFAULT_OUTPUT_DIR, f"{scene_key}_{source}_grocery_to_cupboard_{stamp}.json")


def _write_summary(path, summary):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Summary written to {path}")


def _should_open_box_first(args, source):
    if source != "box":
        return False
    if args.open_box_first:
        return True
    if args.no_open_box_first:
        return False
    if args.scene and os.path.basename(args.scene) == "kitchen_grocery.ttt":
        return False
    return True


def _run_transfer(env, gt, args, requested_source):
    grocery_name, source = _select_grocery(env, requested_source, explicit_grocery=args.grocery)
    grocery = env.get_object(grocery_name)
    start_pose = _pose_list(grocery)
    start_pos = _pos_list(grocery)
    print(f"Selected grocery: {grocery_name} (source={source})")
    print(f"Start position: {start_pos}")

    opened_box = False
    if _should_open_box_first(args, source):
        print("Opening box before box-source grocery transfer...")
        opened_box = bool(gt.run_open_box(env, task_name="Debug: Open Box Before Grocery Transfer"))
        gt.go_home(env)
        if not opened_box:
            print("WARNING: Box did not validate open; continuing with grocery transfer for diagnosis.")

    start_time = time.time()
    if source == "box":
        success = gt.run_box_pick_place(
            env,
            object_name=grocery_name,
            target_region=args.target_region,
            task_name=f"Debug: {grocery_name} in box -> {args.target_region}",
        )
    else:
        success = gt.run_standard_pick_place(
            env,
            object_name=grocery_name,
            target_region=args.target_region,
            task_name=f"Debug: {grocery_name} on table -> {args.target_region}",
        )
    elapsed = time.time() - start_time

    end_pose = _pose_list(grocery)
    end_pos = _pos_list(grocery)
    in_target = _is_in_region(env, grocery, args.target_region, tol=0.03)
    validator_passed = gt.validate_object_in_region(grocery, args.target_region)
    displacement = None
    if start_pos is not None and end_pos is not None:
        displacement = float(np.linalg.norm(np.array(end_pos) - np.array(start_pos)))

    if success:
        gt.go_home(env)

    return {
        "success": bool(success),
        "grocery": grocery_name,
        "source": source,
        "target_region": args.target_region,
        "opened_box_first": bool(opened_box),
        "start_position": start_pos,
        "end_position": end_pos,
        "start_pose": start_pose,
        "end_pose": end_pose,
        "displacement_m": displacement,
        "in_target_bbox": bool(in_target),
        "target_validator_passed": bool(validator_passed),
        "execution_time_s": float(elapsed),
    }


def _run_test(args, scene_path):
    from rlbench_kitchen_streams import ENV
    import ground_truth_orchestrator as gt

    if args.source == "both" and args.grocery:
        raise RuntimeError("--grocery can only be used with one source, not --source both")

    env = ENV
    pr = env.pr

    print("=" * 70)
    print("GROCERY -> CUPBOARD_BOUNDARY DEBUG TEST")
    print(f"Scene : {scene_path}")
    print(f"Source: {args.source}")
    print(f"Target: {args.target_region}")
    print("=" * 70)

    output_source = args.source if args.source != "auto" else "auto"
    output_path = os.path.abspath(args.output or _default_output_path(args, output_source))
    video_dir = os.path.abspath(
        args.video_dir
        or os.path.join(os.path.dirname(output_path), "videos")
    )

    if args.record_video:
        print(f"Initializing video recorder at {video_dir}")
        gt.VIDEO_RECORDER = gt.VideoRecorder(env, output_dir=video_dir, fps=30)
    else:
        gt.VIDEO_RECORDER = None

    print("Settling physics...")
    gt.step_and_record(pr, max(0, int(args.settle_steps)))

    requested_sources = ["box", "table"] if args.source == "both" else [args.source]
    transfers = []
    for requested_source in requested_sources:
        transfer = _run_transfer(env, gt, args, requested_source)
        transfers.append(transfer)
        if not (transfer["success"] and transfer["target_validator_passed"]):
            break

    if gt.VIDEO_RECORDER:
        gt.VIDEO_RECORDER.release()
        gt.VIDEO_RECORDER = None

    ok = all(t["success"] and t["target_validator_passed"] for t in transfers)
    if args.source == "both":
        summary = {
            "success": bool(ok),
            "variant": args.variant,
            "scene_path": scene_path,
            "target_region": args.target_region,
            "requested_source": args.source,
            "transfers": transfers,
            "record_video": bool(args.record_video),
            "video_dir": video_dir if args.record_video else None,
        }
    else:
        summary = dict(transfers[0])
        summary.update(
            {
                "variant": args.variant,
                "scene_path": scene_path,
                "record_video": bool(args.record_video),
                "video_dir": video_dir if args.record_video else None,
            }
        )

    _write_summary(output_path, summary)

    if ok:
        print("PASS: grocery -> cupboard completed.")
        return True
    print("FAIL: grocery -> cupboard did not validate cleanly.")
    return False


def main():
    args = _parse_args()
    scene_path = _prepare_env(args)
    ok = False
    try:
        ok = _run_test(args, scene_path)
        if args.keep_open:
            from rlbench_kitchen_streams import ENV
            print("Keeping simulator open. Press Ctrl+C to exit.")
            while True:
                ENV.pr.step()
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        if not args.keep_open:
            try:
                from rlbench_kitchen_streams import ENV
                ENV.pr.stop()
                ENV.pr.shutdown()
            except Exception:
                pass
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Run an isolated mug-on-box -> placement_boundary transfer test.

This is intentionally narrower than the full kitchen GT benchmark: it loads one
kitchen scene, selects the mug currently on top of the box, runs only
run_box_pick_place(..., target_region="placement_boundary"), and writes a small
JSON summary for quick comparison across trajectory changes.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np


ROOT_DIR = os.path.dirname(__file__)
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "mug_on_box_to_placement")
MUG_CANDIDATES = ("mug2", "mug1", "mug4", "mug3")
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
    parser = argparse.ArgumentParser(
        description="Test only the mug_on_box -> placement_boundary kitchen transfer"
    )
    parser.add_argument(
        "--variant",
        choices=sorted(KITCHEN_VARIANTS),
        default="K1",
        help="Kitchen variant scene to load when --scene is not provided",
    )
    parser.add_argument(
        "--scene",
        default="",
        help="Explicit .ttt scene path. Overrides --variant.",
    )
    parser.add_argument(
        "--mug",
        default="",
        help="Explicit mug name to pick. Defaults to live mug_on_box selection.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Summary JSON path. Defaults under outputs/mug_on_box_to_placement/.",
    )
    parser.add_argument("--headless", action="store_true", help="Run CoppeliaSim headless")
    parser.add_argument("--keep-open", action="store_true", help="Keep simulator open after the run")
    parser.add_argument("--record-video", action="store_true", help="Record videos during the isolated task")
    parser.add_argument(
        "--video-dir",
        default="",
        help="Video directory when --record-video is set",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=50,
        help="Initial physics settle steps before selecting the mug",
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


def _select_mug_on_box(env, explicit_mug=""):
    if explicit_mug:
        obj = env.get_object(explicit_mug)
        if obj is None:
            raise RuntimeError(f"Requested mug '{explicit_mug}' was not found")
        return _obj_name(obj, explicit_mug)

    discovered = _discover_unique_objects(env, MUG_CANDIDATES)
    for name in discovered:
        if _is_in_region(env, env.get_object(name), "box_boundary"):
            return name
    if discovered:
        print("WARNING: No mug was inside box_boundary; using first discovered mug.")
        return discovered[0]
    raise RuntimeError("No mug candidates found in scene")


def _pose_list(obj):
    if obj is None:
        return None
    return [float(v) for v in obj.get_pose()]


def _pos_list(obj):
    if obj is None:
        return None
    return [float(v) for v in obj.get_position()]


def _default_output_path(args):
    scene_key = args.variant if not args.scene else os.path.splitext(os.path.basename(args.scene))[0]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(DEFAULT_OUTPUT_DIR, f"{scene_key}_mug_on_box_to_placement_{stamp}.json")


def _write_summary(path, summary):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Summary written to {path}")


def _run_test(args, scene_path):
    from rlbench_kitchen_streams import ENV
    import ground_truth_orchestrator as gt

    env = ENV
    pr = env.pr
    output_path = os.path.abspath(args.output or _default_output_path(args))
    video_dir = os.path.abspath(
        args.video_dir
        or os.path.join(os.path.dirname(output_path), "videos")
    )

    print("=" * 70)
    print("MUG ON BOX -> PLACEMENT_BOUNDARY DEBUG TEST")
    print(f"Scene : {scene_path}")
    print(f"Output: {output_path}")
    print("=" * 70)

    if args.record_video:
        print(f"Initializing video recorder at {video_dir}")
        gt.VIDEO_RECORDER = gt.VideoRecorder(env, output_dir=video_dir, fps=30)
    else:
        gt.VIDEO_RECORDER = None

    print("Settling physics...")
    gt.step_and_record(pr, max(0, int(args.settle_steps)))

    mug_name = _select_mug_on_box(env, explicit_mug=args.mug)
    mug = env.get_object(mug_name)
    start_pose = _pose_list(mug)
    start_pos = _pos_list(mug)
    print(f"Selected mug_on_box: {mug_name}")
    print(f"Start position: {start_pos}")

    gt.reset_carry_height_tracks()
    start_time = time.time()
    success = gt.run_box_pick_place(
        env,
        object_name=mug_name,
        target_region="placement_boundary",
        task_name=f"Debug: {mug_name} on box -> placement_boundary",
    )
    elapsed = time.time() - start_time

    end_pose = _pose_list(mug)
    end_pos = _pos_list(mug)
    in_placement = _is_in_region(env, mug, "placement_boundary", tol=0.03)
    on_table = gt.validate_object_in_region(mug, "placement_boundary")
    displacement = None
    if start_pos is not None and end_pos is not None:
        displacement = float(np.linalg.norm(np.array(end_pos) - np.array(start_pos)))

    if success:
        gt.go_home(env)

    if gt.VIDEO_RECORDER:
        gt.VIDEO_RECORDER.release()
        gt.VIDEO_RECORDER = None

    summary = {
        "success": bool(success),
        "mug": mug_name,
        "variant": args.variant,
        "scene_path": scene_path,
        "target_region": "placement_boundary",
        "start_position": start_pos,
        "end_position": end_pos,
        "start_pose": start_pose,
        "end_pose": end_pose,
        "displacement_m": displacement,
        "in_placement_boundary_bbox": bool(in_placement),
        "placement_validator_passed": bool(on_table),
        "execution_time_s": float(elapsed),
        "carry_height_tracks": gt.get_carry_height_tracks(),
        "record_video": bool(args.record_video),
        "video_dir": video_dir if args.record_video else None,
    }
    _write_summary(output_path, summary)

    if success and in_placement:
        print("PASS: mug_on_box -> placement_boundary completed.")
        return True
    print("FAIL: mug_on_box -> placement_boundary did not validate cleanly.")
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

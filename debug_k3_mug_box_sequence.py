#!/usr/bin/env python3
"""
Run the focused K3 mug/box sequence:

1. mug on top of box -> placement_boundary
2. mug in cupboard -> placement_boundary
3. open box lid
4. put those two mugs into box_boundary
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime


ROOT_DIR = os.path.dirname(__file__)
K3_SCENE = os.path.join(ROOT_DIR, "task1_variation3.ttt")
V3_HELPERS = os.path.join(
    ROOT_DIR,
    "variation_3_hard",
    "ground_truth_orchestrator_variation3_hard.py",
)
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "k3_mug_box_sequence")


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
    parser = argparse.ArgumentParser(description="Test the K3 mug/lid/box subsequence")
    parser.add_argument("--headless", action="store_true", help="Run CoppeliaSim headless")
    parser.add_argument("--keep-open", action="store_true", help="Keep simulator open after the run")
    parser.add_argument("--record-video", action="store_true", help="Record videos during the sequence")
    parser.add_argument("--video-dir", default="", help="Video directory when --record-video is set")
    parser.add_argument("--output", default="", help="Summary JSON path")
    parser.add_argument("--settle-steps", type=int, default=50, help="Initial physics settle steps")
    return parser.parse_args()


def _prepare_env(args):
    _configure_qt(args.headless)
    os.environ["HEADLESS"] = "True" if args.headless else "False"
    os.environ["KITCHEN_SCENE_FILE"] = os.path.abspath(K3_SCENE)
    os.environ["GT_RECORD_VIDEO"] = "1" if args.record_video else "0"
    sys.path.insert(0, ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "pddlstream"))
    return os.path.abspath(K3_SCENE)


def _load_v3_helpers():
    spec = importlib.util.spec_from_file_location("k3_gt_helpers", V3_HELPERS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _default_output_path():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(DEFAULT_OUTPUT_DIR, f"K3_mug_box_sequence_{stamp}.json")


def _write_summary(path, summary):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Summary written to {path}")


def _pos_list(obj):
    if obj is None:
        return None
    return [float(v) for v in obj.get_position()]


def _object_positions(env, names):
    return {name: _pos_list(env.get_object(name)) for name in names if name}


def _result(label, success):
    print(f"[{'PASS' if success else 'FAIL'}] {label}")
    return {"task": label, "success": bool(success)}


def _run_sequence(args, scene_path):
    helpers = _load_v3_helpers()
    base = helpers.base
    from rlbench_kitchen_streams import ENV

    env = ENV
    pr = env.pr
    output_path = os.path.abspath(args.output or _default_output_path())
    video_dir = os.path.abspath(args.video_dir or os.path.join(os.path.dirname(output_path), "videos"))

    print("=" * 70)
    print("K3 MUG / BOX DEBUG SEQUENCE")
    print(f"Scene : {scene_path}")
    print(f"Output: {output_path}")
    print("=" * 70)

    if args.record_video:
        print(f"Initializing video recorder at {video_dir}")
        base.VIDEO_RECORDER = base.VideoRecorder(env, output_dir=video_dir, fps=30)
    else:
        base.VIDEO_RECORDER = None

    print("Settling physics...")
    base.step_and_record(pr, max(0, int(args.settle_steps)))
    env.set_robot_conf(env.get_home_conf())
    base.step_and_record(pr, 10)

    lid_closed_ref = None
    lid_obj = env.get_object("box_lid")
    if lid_obj is not None:
        lid_closed_ref = list(lid_obj.get_position())

    picked = helpers._classify_variation_objects(env)
    mug_on_box = picked["mug_on_box"]
    mug_in_cupboard = picked["mug_in_cupboard"]
    print("\nSelected objects:")
    print(f"  mug_on_box     : {mug_on_box}")
    print(f"  mug_in_cupboard: {mug_in_cupboard}")

    if mug_on_box is None or mug_in_cupboard is None:
        missing = []
        if mug_on_box is None:
            missing.append("mug_on_box")
        if mug_in_cupboard is None:
            missing.append("mug_in_cupboard")
        summary = {
            "success": False,
            "scene_path": scene_path,
            "missing": missing,
            "tasks": [],
        }
        _write_summary(output_path, summary)
        return False

    results = []
    start_time = time.time()
    base.reset_carry_height_tracks()

    success = base.run_box_pick_place(
        env,
        object_name=mug_on_box,
        target_region="placement_boundary",
        task_name="Debug K3.1: Mug on Box -> Placement",
    )
    results.append(_result("mug_on_box -> placement_boundary", success))
    base.go_home(env)
    if not success:
        return _finish(env, base, args, output_path, video_dir, scene_path, picked, results, start_time)

    success = base.run_cupboard_pick_place(
        env,
        object_name=mug_in_cupboard,
        target_region="placement_boundary",
        task_name="Debug K3.2: Cupboard Mug -> Placement",
    )
    results.append(_result("mug_in_cupboard -> placement_boundary", success))
    base.go_home(env)
    if not success:
        return _finish(env, base, args, output_path, video_dir, scene_path, picked, results, start_time)

    success = base.run_open_box(env, task_name="Debug K3.3: Open Box Lid")
    lid_ok = base.ensure_lid_open_for_box_tasks(
        env,
        lid_closed_ref,
        min_xy=float(os.environ.get("V3_MIN_LID_OPEN_FOR_MUGS", os.environ.get("LID_OPEN_TARGET_DISPLACEMENT", "0.45"))),
        retries=2 if not success else 1,
    )
    success = bool(success or lid_ok)
    results.append(_result("open box lid", success))
    base.go_home(env)
    if not success:
        return _finish(env, base, args, output_path, video_dir, scene_path, picked, results, start_time)

    runtime_table_mugs = helpers._select_runtime_table_mugs(
        env,
        preferred_names=[mug_on_box, mug_in_cupboard],
        max_count=2,
    )
    print(f"\nRuntime mugs for box placement: {runtime_table_mugs}")
    if len(runtime_table_mugs) < 2:
        results.append(_result("find both mugs for box placement", False))
        return _finish(env, base, args, output_path, video_dir, scene_path, picked, results, start_time)

    box_slots = helpers._compute_box_slot_poses(env, runtime_table_mugs)
    for task_idx, mug_name in enumerate(runtime_table_mugs, start=4):
        slot_pose = box_slots.get(mug_name)
        success = helpers._run_table_mug_to_box_with_fallback(
            env,
            pr,
            mug_name,
            task_idx,
            slot_pose,
        )
        results.append(_result(f"{mug_name} -> box_boundary", success))
        base.go_home(env)
        if not success:
            break

    return _finish(env, base, args, output_path, video_dir, scene_path, picked, results, start_time)


def _finish(env, base, args, output_path, video_dir, scene_path, picked, results, start_time):
    names = [picked.get("mug_on_box"), picked.get("mug_in_cupboard")]
    summary = {
        "success": all(item["success"] for item in results),
        "scene_path": scene_path,
        "selected": {
            "mug_on_box": picked.get("mug_on_box"),
            "mug_in_cupboard": picked.get("mug_in_cupboard"),
        },
        "tasks": results,
        "final_positions": _object_positions(env, names),
        "carry_height_tracks": base.get_carry_height_tracks(),
        "execution_time_s": float(max(0.0, time.time() - start_time)),
        "record_video": bool(args.record_video),
        "video_dir": video_dir if args.record_video else None,
    }
    _write_summary(output_path, summary)

    if base.VIDEO_RECORDER:
        base.VIDEO_RECORDER.release()
        base.VIDEO_RECORDER = None

    if summary["success"]:
        print("PASS: K3 mug/box sequence completed.")
    else:
        print("FAIL: K3 mug/box sequence did not validate cleanly.")
    return bool(summary["success"])


def main():
    args = _parse_args()
    scene_path = _prepare_env(args)
    ok = False
    try:
        ok = _run_sequence(args, scene_path)
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

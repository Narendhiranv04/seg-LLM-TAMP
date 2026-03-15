#!/usr/bin/env python3
"""
Run open-grill via PDDLStream (Step-1 + Step-2).

Executed segments:
1) home -> handle-facing hover
2) horizontal hover -> handle grasp
Then closes gripper on the handle.
"""

import os
import sys
import argparse
import numpy as np


def _configure_qt():
    """Keep Qt quiet and point it to CoppeliaSim plugins."""
    os.environ.setdefault("COPPELIASIM_HEADLESS", "0")
    os.environ.pop("QT_PLUGIN_PATH", None)
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")

    coppelia_root = os.environ.get("COPPELIASIM_ROOT") or os.path.expanduser("~/CoppeliaSim")
    candidate_dirs = [
        os.path.join(coppelia_root, "platforms"),
        os.path.join(coppelia_root, "Qt", "plugins", "platforms"),
        os.path.join(coppelia_root, "qt", "plugins", "platforms"),
    ]
    for candidate in candidate_dirs:
        if candidate and os.path.isdir(candidate):
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", candidate)
            break


_configure_qt()

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "pddlstream"))

from pddlstream.language.constants import PDDLProblem
from pddlstream.algorithms.meta import solve
from pddlstream.utils import read


def _is_conf(x):
    if not isinstance(x, (list, tuple, np.ndarray)):
        return False
    if len(x) != 7:
        return False
    try:
        [float(v) for v in x]
        return True
    except Exception:
        return False


def _extract_segments(traj_obj):
    """Convert traj object to list of trajectory segments."""
    if traj_obj is None:
        return []

    if isinstance(traj_obj, (list, tuple)) and len(traj_obj) > 0 and _is_conf(traj_obj[0]):
        return [list(traj_obj)]

    segments = []
    if isinstance(traj_obj, (list, tuple)):
        for seg in traj_obj:
            if isinstance(seg, (list, tuple)) and len(seg) > 0 and _is_conf(seg[0]):
                segments.append(list(seg))
    return segments


def _interpolate_segment(segment, steps_per_segment=15):
    if segment is None or len(segment) == 0:
        return []
    if len(segment) == 1:
        return [segment[0]]

    dense = []
    for i in range(len(segment) - 1):
        q1 = np.array(segment[i], dtype=float)
        q2 = np.array(segment[i + 1], dtype=float)
        for t in np.linspace(0.0, 1.0, max(2, int(steps_per_segment)), endpoint=False):
            dense.append(((1.0 - t) * q1 + t * q2).tolist())
    dense.append(list(segment[-1]))
    return dense


def _close_gripper(env, pr, target=0.0, velocity=0.14, max_steps=120):
    """Close the gripper robustly with simulation stepping."""
    for _ in range(max(1, int(max_steps))):
        try:
            done = bool(env.gripper.actuate(float(target), float(velocity)))
        except Exception:
            done = False
        pr.step()
        if done:
            break


def main():
    parser = argparse.ArgumentParser(description="PDDL open-grill Step-1+Step-2 runner")
    parser.add_argument(
        "--x-offset",
        type=float,
        default=0.12,
        help="X offset before handle for Step-1 hover (meters).",
    )
    parser.add_argument(
        "--grasp-clearance",
        type=float,
        default=0.006,
        help="Remaining X clearance from handle midpoint for Step-2 grasp (meters).",
    )
    args = parser.parse_args()

    os.environ["HEADLESS"] = "False"
    os.environ["GRILL_SCENE_FILE"] = os.path.join(THIS_DIR, "grill.variation1.ttt")
    os.environ.setdefault("GRILL_PRESERVE_SCENE_LID_POSE", "True")
    os.environ["GRILL_OPEN_STEP1_X_OFFSET"] = str(float(args.x_offset))
    os.environ["GRILL_OPEN_STEP2_X_CLEARANCE"] = str(float(args.grasp_clearance))

    from grill_task_streams import ENV, get_stream_map

    env = ENV
    pr = env.pr
    home_q = list(env.get_home_conf())
    env.set_robot_conf(home_q)

    for _ in range(15):
        pr.step()

    print("=" * 72)
    print("PDDL OPEN-GRILL TEST (STEP-1 + STEP-2)")
    print("=" * 72)
    print(f"Scene: {os.environ.get('GRILL_SCENE_FILE')}")
    print(f"Step-1 X offset: {os.environ.get('GRILL_OPEN_STEP1_X_OFFSET')} m")
    print(f"Step-2 grasp clearance: {os.environ.get('GRILL_OPEN_STEP2_X_CLEARANCE')} m")

    domain_pddl = read(os.path.join(THIS_DIR, "pddl", "grill_task_domain.pddl"))
    stream_pddl = read(os.path.join(THIS_DIR, "pddl", "grill_task_streams.pddl"))

    lid_name = "lid"
    init = [
        ("lid", lid_name),
        ("grill-closed", lid_name),
        ("hand-empty",),
        ("conf", tuple(home_q)),
        ("at-conf", tuple(home_q)),
    ]
    goal = ("grill-open", lid_name)

    problem = PDDLProblem(
        domain_pddl=domain_pddl,
        constant_map={},
        stream_pddl=stream_pddl,
        stream_map=get_stream_map(),
        init=init,
        goal=goal,
    )

    print("Solving...")
    plan, cost, _evaluations = solve(problem, algorithm="adaptive", verbose=False)

    if not plan:
        print("ERROR: No plan found.")
        print("Press Ctrl+C to close.")
        try:
            while True:
                pr.step()
        except KeyboardInterrupt:
            pass
        pr.stop()
        pr.shutdown()
        return

    print(f"Plan found (cost={cost}):")
    for act in plan:
        print(f"  - {act.name}")

    executed = False
    for act in plan:
        if act.name != "open-grill":
            continue

        _o, _g, _q1, _q2, traj = act.args
        segments = _extract_segments(traj)
        if not segments:
            print("WARNING: open-grill trajectory had no executable segments.")
            continue

        seg_count = min(2, len(segments))
        print(f"Executing {seg_count} segment(s) for Step-1+Step-2...")
        for idx in range(seg_count):
            dense = _interpolate_segment(segments[idx], steps_per_segment=12)
            print(f"  Segment {idx + 1}: {len(dense)} points")
            for q in dense:
                env.set_robot_conf(q)
                pr.step()
        executed = True
        break

    if not executed:
        print("ERROR: open-grill action not found in plan.")
        print("Press Ctrl+C to close.")
        try:
            while True:
                pr.step()
        except KeyboardInterrupt:
            pass
        pr.stop()
        pr.shutdown()
        return

    print("Closing gripper for Step-2 grasp...")
    # Never attach/re-parent the handle to gripper; keep it part of the lid.
    try:
        env.gripper.release()
    except Exception:
        pass
    _close_gripper(env, pr, target=0.0, velocity=0.14, max_steps=140)
    for _ in range(10):
        pr.step()

    print("Done. Robot should be at handle grasp pose with gripper closed.")
    print("Press Ctrl+C to close.")
    try:
        while True:
            pr.step()
    except KeyboardInterrupt:
        pass

    pr.stop()
    pr.shutdown()


if __name__ == "__main__":
    main()

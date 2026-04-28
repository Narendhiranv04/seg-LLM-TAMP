#!/usr/bin/env python3
"""
Run open-grill via PDDLStream (Step-1 only).

This uses the grill_task2 domain + streams and executes the trajectory returned by
`sample-open-grill`, but executes only segment-1:
  home -> handle-facing hover pose.
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
    """
    Convert traj object to a list of trajectory segments.
    Segment = list of 7D joint configurations.
    """
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


def main():
    parser = argparse.ArgumentParser(description="PDDL open-grill Step-1 runner")
    parser.add_argument(
        "--x-offset",
        type=float,
        default=0.12,
        help="X offset before handle for open-grill Step-1 hover (meters).",
    )
    args = parser.parse_args()

    os.environ["HEADLESS"] = "False"
    os.environ["GRILL_SCENE_FILE"] = os.path.join(THIS_DIR, "grill.variation1.ttt")
    os.environ.setdefault("GRILL_PRESERVE_SCENE_LID_POSE", "True")
    os.environ["GRILL_OPEN_STEP1_X_OFFSET"] = str(float(args.x_offset))

    from grill_task_streams import ENV, get_stream_map

    env = ENV
    pr = env.pr
    home_q = list(env.get_home_conf())
    env.set_robot_conf(home_q)

    # Settle briefly.
    for _ in range(15):
        pr.step()

    print("=" * 72)
    print("PDDL OPEN-GRILL TEST (STEP-1)")
    print("=" * 72)
    print(f"Scene: {os.environ.get('GRILL_SCENE_FILE')}")
    print(f"Step-1 X offset: {os.environ.get('GRILL_OPEN_STEP1_X_OFFSET')} m")

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

    for act in plan:
        if act.name != "open-grill":
            continue
        _o, _g, _q1, _q2, traj = act.args
        segments = _extract_segments(traj)
        if not segments:
            print("WARNING: open-grill trajectory had no executable segments.")
            continue

        print(f"Trajectory has {len(segments)} segment(s); executing Step-1 only.")
        seg = segments[0]
        dense = _interpolate_segment(seg, steps_per_segment=12)
        print(f"  Segment 1: {len(dense)} points")
        for q in dense:
            env.set_robot_conf(q)
            pr.step()

    print("Done. Robot should now be at grill grasp-hover (Step-1).")
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

#!/usr/bin/env python3
"""
Record and replay-test the kitchen box-lid opening trajectory on kitchen_box.ttt.
"""

import argparse
import os
import sys
import time
import numpy as np


ROOT_DIR = os.path.dirname(__file__)
DEFAULT_SCENE = os.path.join(ROOT_DIR, "kitchen_box.ttt")
DEFAULT_OUTPUT = os.path.join(ROOT_DIR, "precomputed_paths", "kitchen_box_lid_open.json")


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
    parser = argparse.ArgumentParser(description="Record a kitchen box-lid open replay path")
    parser.add_argument("--scene", default=DEFAULT_SCENE, help="Scene used for isolated box-lid testing")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Replay JSON to write")
    parser.add_argument("--replay", action="store_true", help="Replay --output instead of recording a new path")
    parser.add_argument("--headless", action="store_true", help="Run CoppeliaSim headless")
    parser.add_argument("--keep-open", action="store_true", help="Keep the simulator open after the run")
    parser.add_argument("--slide-dist", type=float, default=None, help="Override BOX_LID_SLIDE_DIST")
    parser.add_argument("--target-open", type=float, default=None, help="Override LID_OPEN_TARGET_DISPLACEMENT")
    parser.add_argument("--min-slide", type=float, default=None, help="Override LID_SLIDE_MIN_DIST")
    parser.add_argument(
        "--open-margin",
        type=float,
        default=0.02,
        help="Extra raw lid displacement beyond target used for deterministic opening",
    )
    motion_group = parser.add_mutually_exclusive_group()
    motion_group.add_argument(
        "--direct-home-to-grasp",
        action="store_true",
        help="Interpolate from home directly to the final lid grasp joint configuration",
    )
    motion_group.add_argument(
        "--offset-home-to-grasp",
        action="store_true",
        help="Interpolate home to a retreat-offset pre-grasp, then move horizontally into grasp",
    )
    return parser.parse_args()


def _prepare_env(args):
    _configure_qt(args.headless)
    os.environ["HEADLESS"] = "True" if args.headless else "False"
    os.environ["KITCHEN_SCENE_FILE"] = os.path.abspath(args.scene)
    if args.replay:
        os.environ["KITCHEN_BOX_LID_REPLAY_PATH"] = os.path.abspath(args.output)
    if args.slide_dist is not None:
        os.environ["BOX_LID_SLIDE_DIST"] = str(args.slide_dist)
    else:
        target_open = args.target_open if args.target_open is not None else 0.45
        os.environ.setdefault("BOX_LID_SLIDE_DIST", str(float(target_open) + float(args.open_margin)))
    if args.target_open is not None:
        os.environ["LID_OPEN_TARGET_DISPLACEMENT"] = str(args.target_open)
        os.environ.setdefault("LID_OPEN_MIN_DISPLACEMENT", str(args.target_open))
    if args.min_slide is not None:
        os.environ["LID_SLIDE_MIN_DIST"] = str(args.min_slide)
    sys.path.append(os.path.join(ROOT_DIR, "pddlstream"))


def _tip_position_at_conf(env, conf, restore_conf):
    env.set_robot_conf(conf)
    try:
        tip = env.robot.get_tip()
    except Exception:
        tip = env.robot.arm.get_tip()
    pos = np.array(tip.get_position(), dtype=float)
    env.set_robot_conf(restore_conf)
    return pos


def _plan_offset_home_to_grasp(env, gt, home_q, q_grasp, grasp_quat, traj_hover_to_center):
    restore_conf = env.get_robot_conf()
    hover_pos = _tip_position_at_conf(env, traj_hover_to_center[0], restore_conf)
    center_pos = _tip_position_at_conf(env, traj_hover_to_center[-1], restore_conf)
    grasp_pos = _tip_position_at_conf(env, q_grasp, restore_conf)

    retreat_vec = hover_pos - center_pos
    retreat_dist = float(np.linalg.norm(retreat_vec))
    horizontal_vec = np.array([retreat_vec[0], retreat_vec[1], 0.0], dtype=float)
    horizontal_norm = float(np.linalg.norm(horizontal_vec))
    if horizontal_norm < 1e-6:
        print("ERROR: Could not derive a horizontal retreat direction.")
        return None, None, None

    horizontal_offset = (horizontal_vec / horizontal_norm) * retreat_dist
    pregrasp_pos = grasp_pos + horizontal_offset
    print(
        "[OffsetGrasp] Retreat distance after opening: "
        f"{retreat_dist:.3f}m; horizontal offset={horizontal_offset.tolist()}"
    )
    print(f"[OffsetGrasp] Pre-grasp target: {pregrasp_pos.tolist()}")

    env.set_robot_conf(q_grasp)
    try:
        configs = env.robot.solve_ik_via_sampling(
            pregrasp_pos.tolist(),
            quaternion=grasp_quat,
            max_configs=20,
            max_time_ms=500,
            ignore_collisions=True,
        )
    finally:
        env.set_robot_conf(restore_conf)
    if configs is None or len(configs) == 0:
        print("ERROR: Could not solve IK for retreat-offset pre-grasp pose.")
        return None, None, None

    q_pregrasp = list(configs[0])
    motion_to_pregrasp = gt.interpolate_path(env, home_q, q_pregrasp, steps=80)
    path_to_grasp = env._get_linear_path(
        q_pregrasp,
        grasp_pos.tolist(),
        grasp_quat,
        ignore_collisions=True,
        steps=50,
    )
    if not path_to_grasp:
        print("ERROR: Could not plan horizontal pre-grasp -> grasp movement.")
        return None, None, None

    approach = path_to_grasp._path_points.reshape(-1, 7).tolist()
    return motion_to_pregrasp, approach, {
        "retreat_distance": retreat_dist,
        "retreat_vector": retreat_vec.tolist(),
        "horizontal_offset": horizontal_offset.tolist(),
        "pregrasp_position": pregrasp_pos.tolist(),
        "grasp_position": grasp_pos.tolist(),
    }


def _plan_return_to_initial_grasp(env, q_grasp, grasp_quat):
    restore_conf = env.get_robot_conf()
    grasp_pos = _tip_position_at_conf(env, q_grasp, restore_conf)
    env.set_robot_conf(restore_conf)
    path = env._get_linear_path(
        restore_conf,
        grasp_pos.tolist(),
        grasp_quat,
        ignore_collisions=True,
        steps=50,
    )
    if not path:
        print("WARNING: Could not plan linear return to initial grasp pose; using joint interpolation fallback.")
        return None
    return path._path_points.reshape(-1, 7).tolist()


def _lid_open_xy(lid, lid_pos_before):
    pos = np.array(lid.get_position()[:2], dtype=float)
    before = np.array(lid_pos_before[:2], dtype=float)
    return float(np.linalg.norm(pos - before))


def _strict_open_target(target_open_xy, open_margin):
    env_target = os.environ.get("KITCHEN_BOX_LID_STRICT_OPEN_TARGET")
    if env_target is not None:
        return float(env_target)
    return float(target_open_xy) + max(0.0, float(open_margin))


def _extend_open_to_strict_target(env, gt, lid, lid_pos_before, strict_target):
    correction_traj = []
    attempts = int(os.environ.get("KITCHEN_BOX_LID_STRICT_OPEN_ATTEMPTS", "4"))
    min_push = float(os.environ.get("KITCHEN_BOX_LID_MIN_CORRECTION_PUSH", "0.015"))
    push_margin = float(os.environ.get("KITCHEN_BOX_LID_CORRECTION_MARGIN", "0.01"))

    for attempt in range(max(1, attempts)):
        current = _lid_open_xy(lid, lid_pos_before)
        remaining = float(strict_target) - current
        print(f"[StrictOpen] raw lid displacement after attempt {attempt}: {current:.3f}m")
        if remaining <= 0:
            return correction_traj, current, True

        push = max(min_push, remaining + push_margin)
        try:
            try:
                tip = env.robot.get_tip()
            except Exception:
                tip = env.robot.arm.get_tip()
            tip_pos = np.array(tip.get_position(), dtype=float)
            tip_quat = tip.get_quaternion()
            target_pos = [float(tip_pos[0] + push), float(tip_pos[1]), float(tip_pos[2])]
            print(f"[StrictOpen] pushing +X by {push:.3f}m toward raw target {strict_target:.3f}m")
            path = env.robot.get_linear_path(
                position=target_pos,
                quaternion=tip_quat,
                steps=35,
                ignore_collisions=True,
            )
            if path is None:
                print("[StrictOpen] correction path was empty.")
                break
            traj = path._path_points.reshape(-1, 7).tolist()
            gt.execute_trajectory(env, traj, steps=4)
            correction_traj.extend(traj)
        except Exception as exc:
            print(f"[StrictOpen] correction path failed: {exc}")
            break

    current = _lid_open_xy(lid, lid_pos_before)
    return correction_traj, current, current >= float(strict_target)


def _record_open_path(args):
    from rlbench_kitchen_streams import ENV
    import ground_truth_orchestrator as gt

    env = ENV
    pr = env.pr
    target_open_xy = float(os.environ.get("LID_OPEN_TARGET_DISPLACEMENT", "0.45"))
    target_open_xy = max(target_open_xy, float(os.environ.get("LID_OPEN_MIN_DISPLACEMENT", "0.45")))
    strict_open_xy = _strict_open_target(target_open_xy, args.open_margin)

    print("=" * 70)
    print("KITCHEN BOX-LID OPEN TRAJECTORY REPLAY" if args.replay else "KITCHEN BOX-LID OPEN TRAJECTORY RECORD")
    print(f"Scene : {os.environ['KITCHEN_SCENE_FILE']}")
    print(f"Output: {os.path.abspath(args.output)}")
    print(f"Strict open target: {strict_open_xy:.3f}m")
    print("=" * 70)

    lid = env.get_object("box_lid")
    if lid is None:
        print("ERROR: box_lid not found in scene.")
        return False

    print("Settling physics...")
    gt.step_and_record(pr, 50)

    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    gt.step_and_record(pr, 10)

    if args.replay:
        return gt.run_open_box(env, task_name="Kitchen box-lid replay test")

    lid_pos_before = list(lid.get_position())
    env.gripper.actuate(1.0, 0.1)
    gt.step_and_record(pr, 10)

    print("Computing grasp trajectory...")
    grasp_quat, q_hover, q_grasp, approach_parts = env.compute_lid_grasp_trajectory(lid)
    traj_hover_to_center, traj_center_to_edge = approach_parts
    grasp_motion_metadata = {}

    if args.offset_home_to_grasp:
        print("Using retreat-offset home -> pre-grasp interpolation plus horizontal grasp movement.")
        motion_to_hover, approach, grasp_motion_metadata = _plan_offset_home_to_grasp(
            env,
            gt,
            home_q,
            q_grasp,
            grasp_quat,
            traj_hover_to_center,
        )
        if motion_to_hover is None or approach is None:
            return False
        retreat = approach[::-1]
        home_return = motion_to_hover[::-1]
    elif args.direct_home_to_grasp:
        print("Using direct home -> grasp joint interpolation.")
        motion_to_hover = gt.interpolate_path(env, home_q, q_grasp, steps=80)
        approach = [q_grasp]
        retreat = [q_grasp]
        home_return = motion_to_hover[::-1]
    else:
        approach = traj_hover_to_center + traj_center_to_edge
        retreat = approach[::-1]
        print("Planning motion to hover...")
        motion_to_hover = env.compute_motion_plan(home_q, q_hover)
        if not motion_to_hover:
            print("ERROR: Could not plan motion to hover.")
            return False
        home_return = motion_to_hover[::-1]

    if args.offset_home_to_grasp:
        move_label = "Moving to retreat-offset pre-grasp..."
    elif args.direct_home_to_grasp:
        move_label = "Moving to grasp..."
    else:
        move_label = "Moving to hover..."
    print(move_label)
    gt.execute_trajectory(env, motion_to_hover, steps=5)

    if args.offset_home_to_grasp:
        print("Moving horizontally into grasp...")
        gt.execute_trajectory(env, approach, steps=5)
    elif args.direct_home_to_grasp:
        print("Reached direct grasp configuration.")
    else:
        print("Approaching grasp...")
        gt.execute_trajectory(env, approach, steps=5)

    print("Grasping lid...")
    env.gripper.actuate(0.0, 0.1)
    gt.step_and_record(pr, 30)
    env.gripper.grasp(lid)

    print("Computing slide trajectory...")
    _q_open_start, _q_open_end, open_traj, _return_traj = env.compute_slide_lid_trajectory(
        lid, grasp_quat, initial_conf=q_grasp
    )
    if not open_traj:
        print("ERROR: Failed to compute slide trajectory.")
        return False

    print("Sliding lid open...")
    gt.execute_trajectory(env, open_traj, steps=5)

    raw_open_xy = _lid_open_xy(lid, lid_pos_before)
    print(f"[StrictOpen] raw lid displacement after planned slide: {raw_open_xy:.3f}m")
    correction_traj, raw_open_xy, strict_opened = _extend_open_to_strict_target(
        env, gt, lid, lid_pos_before, strict_open_xy
    )
    if correction_traj:
        open_traj = open_traj + correction_traj

    if not strict_opened:
        print(
            f"ERROR: Lid did not reach strict raw target. "
            f"displacement={raw_open_xy:.3f}m target={strict_open_xy:.3f}m"
        )
        return False

    print("Releasing lid...")
    env.gripper.release()
    env.gripper.actuate(1.0, 0.1)
    gt.step_and_record(pr, 50)

    print("Returning to initial grasp pose...")
    return_to_grasp = _plan_return_to_initial_grasp(env, q_grasp, grasp_quat)
    if return_to_grasp is None:
        return_to_grasp = gt.interpolate_path(env, env.get_robot_conf(), q_grasp, steps=50)
    gt.execute_trajectory(env, return_to_grasp, steps=5)

    print("Retreating from initial grasp pose...")
    gt.execute_trajectory(env, retreat, steps=5)

    print("Following the same trajectory back home...")
    gt.execute_trajectory(env, home_return, steps=5)
    gt.step_and_record(pr, 50)

    raw_final_xy = _lid_open_xy(lid, lid_pos_before)
    lid_pos_after = list(lid.get_position())
    if raw_final_xy < strict_open_xy:
        print(
            f"ERROR: Lid did not open enough. "
            f"displacement={raw_final_xy:.3f}m target={strict_open_xy:.3f}m"
        )
        return False

    segments = {
        "motion_to_hover": motion_to_hover,
        "approach": approach,
        "open": open_traj,
        "return": return_to_grasp,
        "retreat": retreat,
        "home_return": home_return,
    }
    metadata = {
        "lid_pos_before": lid_pos_before,
        "lid_pos_after": list(lid_pos_after),
        "displacement_xy": float(raw_final_xy),
        "strict_open_target_xy": float(strict_open_xy),
        "strict_open_margin": float(args.open_margin),
        "strict_correction_waypoints": len(correction_traj),
        "target_open_xy": float(target_open_xy),
        "box_lid_slide_dist": os.environ.get("BOX_LID_SLIDE_DIST"),
        "lid_slide_min_dist": os.environ.get("LID_SLIDE_MIN_DIST"),
        "grasp_motion_mode": (
            "offset_home_to_grasp"
            if args.offset_home_to_grasp
            else "direct_home_to_grasp"
            if args.direct_home_to_grasp
            else "hover_center_edge"
        ),
        "grasp_motion": grasp_motion_metadata,
    }
    gt.record_box_lid_open_replay(os.path.abspath(args.output), env, lid, segments, metadata=metadata)
    print(f"PASS: saved box-lid replay with raw displacement {raw_final_xy:.3f}m")
    return True


def main():
    args = _parse_args()
    _prepare_env(args)
    ok = False
    env = None
    try:
        ok = _record_open_path(args)
        if args.keep_open:
            from rlbench_kitchen_streams import ENV
            env = ENV
            print("Keeping simulator open. Press Ctrl+C to exit.")
            while True:
                env.pr.step()
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        if not args.keep_open:
            try:
                from rlbench_kitchen_streams import ENV
                env = ENV
                env.pr.stop()
                env.pr.shutdown()
            except Exception:
                pass
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

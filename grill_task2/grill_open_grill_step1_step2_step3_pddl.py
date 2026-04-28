#!/usr/bin/env python3
"""
Run open-grill via PDDLStream (Step-1 + Step-2 + Step-3).

Executed sequence:
1) home -> handle-facing hover
2) horizontal hover -> handle grasp + close gripper
3) smooth hinge-centered circular arc to target opening angle (default 90 deg)
"""

import os
import sys
import argparse
import time
import numpy as np
from pyrep.backend import sim


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


def _quat_to_rot(quat_xyzw):
    q = np.array(quat_xyzw, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3, dtype=float)
    q = q / n
    x, y, z, w = q.tolist()
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def _rot_to_quat(rot):
    R = np.array(rot, dtype=float).reshape(3, 3)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = float(np.sqrt(tr + 1.0) * 2.0)
        w = 0.25 * s
        x = float((R[2, 1] - R[1, 2]) / s)
        y = float((R[0, 2] - R[2, 0]) / s)
        z = float((R[1, 0] - R[0, 1]) / s)
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = float(np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0)
            w = float((R[2, 1] - R[1, 2]) / s)
            x = 0.25 * s
            y = float((R[0, 1] + R[1, 0]) / s)
            z = float((R[0, 2] + R[2, 0]) / s)
        elif R[1, 1] > R[2, 2]:
            s = float(np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0)
            w = float((R[0, 2] - R[2, 0]) / s)
            x = float((R[0, 1] + R[1, 0]) / s)
            y = 0.25 * s
            z = float((R[1, 2] + R[2, 1]) / s)
        else:
            s = float(np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0)
            w = float((R[1, 0] - R[0, 1]) / s)
            x = float((R[0, 2] + R[2, 0]) / s)
            y = float((R[1, 2] + R[2, 1]) / s)
            z = 0.25 * s
    q = np.array([x, y, z, w], dtype=float)
    n = float(np.linalg.norm(q))
    if n > 1e-12:
        q = q / n
    return q.tolist()


def _unit(vec, eps=1e-12):
    v = np.array(vec, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < eps:
        return None
    return v / n


def _face_handle_quat_candidates(facing_dir, hinge_axis=None):
    """
    Generate orientation candidates where tool keeps facing the handle center.
    This allows natural hinge-like rotation instead of fixed-horizontal lock.
    """
    f = _unit(facing_dir)
    if f is None:
        f = np.array([1.0, 0.0, 0.0], dtype=float)

    up = _unit(hinge_axis) if hinge_axis is not None else None
    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(up, f))) > 0.92:
        up = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(up, f))) > 0.92:
        up = np.array([0.0, 1.0, 0.0], dtype=float)

    # Candidate A: local +X faces handle.
    xA = f
    zA = np.cross(xA, up)
    if float(np.linalg.norm(zA)) < 1e-8:
        zA = np.array([0.0, 1.0, 0.0], dtype=float)
    zA = zA / (np.linalg.norm(zA) + 1e-12)
    yA = np.cross(zA, xA)
    yA = yA / (np.linalg.norm(yA) + 1e-12)
    RA = np.column_stack((xA, yA, zA))

    # Candidate B: local +Z faces handle.
    zB = xA
    xB = np.cross(up, zB)
    if float(np.linalg.norm(xB)) < 1e-8:
        xB = np.array([1.0, 0.0, 0.0], dtype=float)
    xB = xB / (np.linalg.norm(xB) + 1e-12)
    yB = np.cross(zB, xB)
    yB = yB / (np.linalg.norm(yB) + 1e-12)
    RB = np.column_stack((xB, yB, zB))

    out = [_rot_to_quat(RA), _rot_to_quat(RB)]
    # Small rolls around facing direction for IK robustness.
    for roll in (0.16, -0.16):
        out.append(_rot_to_quat(_axis_angle_matrix(f, roll) @ RA))
        out.append(_rot_to_quat(_axis_angle_matrix(f, roll) @ RB))
    return out


def _shape_rot_pos(shape_obj):
    """Return world rotation (3x3) and position (3,) for a shape."""
    m = np.array(shape_obj.get_matrix(), dtype=float)
    if m.size == 12:
        m = m.reshape(3, 4)
        rot = m[:, :3]
        pos = m[:, 3]
        return rot, pos
    if m.size == 16:
        m = m.reshape(4, 4)
        rot = m[:3, :3]
        pos = m[:3, 3]
        return rot, pos
    # Fallback
    pos = np.array(shape_obj.get_position(), dtype=float)
    return np.eye(3, dtype=float), pos


def _axis_angle_matrix(axis, angle):
    axis = np.array(axis, dtype=float)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return np.eye(3, dtype=float)
    axis = axis / n
    x, y, z = axis.tolist()
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


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


def _interpolate_segment(segment, steps_per_segment=14):
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


def _close_gripper(env, pr, target=0.0, velocity=0.14, max_steps=140):
    """Close the gripper robustly with simulation stepping."""
    for _ in range(max(1, int(max_steps))):
        try:
            done = bool(env.gripper.actuate(float(target), float(velocity)))
        except Exception:
            done = False
        pr.step()
        if done:
            break


def _execute_segment(env, pr, segment, label):
    dense = _interpolate_segment(segment, steps_per_segment=12)
    print(f"  {label}: {len(dense)} points")
    for q in dense:
        env.set_robot_conf(q)
        pr.step()


def _set_lid_angle_sync(env, angle):
    """Hard-sync lid joint angle for tightly coupled robot+lid motion."""
    lid_joint = getattr(env, "lid_joint", None)
    if lid_joint is None:
        return
    try:
        lid_joint.set_joint_position(float(angle), disable_dynamics=True)
    except Exception:
        try:
            sim.simSetJointPosition(int(lid_joint.get_handle()), float(angle))
        except Exception:
            pass
    try:
        lid_joint.set_joint_target_position(float(angle))
    except Exception:
        pass
    try:
        lid_joint.set_joint_target_velocity(0.0)
    except Exception:
        pass


def _move_robot_and_lid_smooth(env, pr, q_from, q_to, lid_from, lid_to, max_joint_step):
    """
    Move robot and lid together with small interpolation steps.
    This avoids large per-step jumps that break handle contact.
    """
    q_from = np.array(q_from, dtype=float)
    q_to = np.array(q_to, dtype=float)
    max_delta = float(np.max(np.abs(q_to - q_from))) if q_from.size else 0.0
    n_sub = max(1, int(np.ceil(max_delta / max(1e-4, float(max_joint_step)))))
    for k in range(1, n_sub + 1):
        t = float(k) / float(n_sub)
        q_sub = (1.0 - t) * q_from + t * q_to
        lid_sub = (1.0 - t) * float(lid_from) + t * float(lid_to)
        env.set_robot_conf(q_sub.tolist())
        _set_lid_angle_sync(env, lid_sub)
        try:
            env.gripper.actuate(0.0, 0.10)
        except Exception:
            pass
        pr.step()
    return n_sub


def _quat_align_abs(q1_xyzw, q2_xyzw):
    q1 = np.array(q1_xyzw, dtype=float)
    q2 = np.array(q2_xyzw, dtype=float)
    n1 = float(np.linalg.norm(q1))
    n2 = float(np.linalg.norm(q2))
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    q1 = q1 / n1
    q2 = q2 / n2
    return float(abs(np.dot(q1, q2)))


def _execute_hinged_arc_online(env, pr, open_deg):
    """
    Step-3 online controller:
    compute IK at each circular-arc point in real time (no precomputed segment).
    If a point is unreachable, skip to the next arc point.
    """
    try:
        env.set_lid_servo_lock(False)
    except Exception:
        pass
    try:
        env.set_lid_collision_enabled(True)
    except Exception:
        pass

    lid_joint = getattr(env, "lid_joint", None)
    if lid_joint is None:
        raise RuntimeError("lid_joint not available for online Step-3")
    tip_obj = env.robot.get_tip()
    # Prefer the physical handle object used by environment initialization.
    handle_obj = getattr(env, "handle", None)
    handle_name = "env.handle"
    if handle_obj is None:
        try:
            handle_obj = env.get_object("handle")
            handle_name = "handle"
        except Exception:
            handle_obj = None
    if handle_obj is None:
        try:
            handle_obj = env.get_object("handle_visual")
            handle_name = "handle_visual"
        except Exception:
            handle_obj = None
    if handle_obj is None:
        raise RuntimeError("handle object not found for online Step-3")
    try:
        handle_dist = float(np.linalg.norm(np.array(handle_obj.get_position(), dtype=float) - np.array(tip_obj.get_position(), dtype=float)))
    except Exception:
        handle_dist = float("nan")
    print(f"  Step-3 handle frame: {handle_name} (dist={handle_dist:.3f} m)")

    try:
        current_joint = float(lid_joint.get_joint_position())
    except Exception:
        current_joint = 0.0
    _set_lid_angle_sync(env, current_joint)

    # Increase finger force so the handle is clamped while tracing the arc.
    try:
        grip_force = float(os.environ.get("GRILL_OPEN_STEP3_GRIP_FORCE", "260.0"))
        env.gripper.set_joint_forces([grip_force, grip_force])
    except Exception:
        pass

    try:
        sign = float(env._infer_open_rotation_sign(current_joint=current_joint, handle_obj=handle_obj))
    except Exception:
        sign = 1.0
    target_joint = float(current_joint + sign * np.deg2rad(float(open_deg)))

    try:
        hinge_pos = np.array(lid_joint.get_position(), dtype=float)
    except Exception:
        hp = np.array(handle_obj.get_position(), dtype=float)
        hinge_pos = np.array([hp[0], hp[1] + 0.25, hp[2] - 0.12], dtype=float)

    try:
        m12 = sim.simGetObjectMatrix(int(lid_joint.get_handle()), sim.sim_handle_world)
        m = np.array(m12, dtype=float).reshape(3, 4)
        hinge_axis = np.array([m[0, 2], m[1, 2], m[2, 2]], dtype=float)
    except Exception:
        hinge_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    an = float(np.linalg.norm(hinge_axis))
    hinge_axis = (hinge_axis / an) if an > 1e-12 else np.array([1.0, 0.0, 0.0], dtype=float)

    hR0, hpos0 = _shape_rot_pos(handle_obj)
    tip0 = np.array(tip_obj.get_position(), dtype=float)
    tipR0 = _quat_to_rot(np.array(tip_obj.get_quaternion(), dtype=float))
    p_rel_handle = hR0.T @ (tip0 - hpos0)
    R_rel_handle = hR0.T @ tipR0
    r_tip0 = tip0 - hinge_pos
    handle_tip_dist0 = float(np.linalg.norm(tip0 - hpos0))

    n_pts = max(100, int(os.environ.get("GRILL_OPEN_STEP3_ONLINE_POINTS", "220")))
    angles = np.linspace(current_joint, target_joint, n_pts)
    max_skip_streak = max(1, int(os.environ.get("GRILL_OPEN_STEP3_ONLINE_MAX_SKIP_STREAK", "35")))
    min_reached_deg = float(os.environ.get("GRILL_OPEN_STEP3_MIN_REACHED_DEG", "80.0"))
    max_joint_step = float(os.environ.get("GRILL_OPEN_STEP3_MAX_JOINT_STEP_RAD", "0.006"))
    soft_contact_tol = float(os.environ.get("GRILL_OPEN_STEP3_CONTACT_SOFT_TOL", "0.022"))
    hard_contact_tol = float(os.environ.get("GRILL_OPEN_STEP3_CONTACT_HARD_TOL", "0.045"))
    hard_contact_streak_limit = max(1, int(os.environ.get("GRILL_OPEN_STEP3_HARD_CONTACT_STREAK", "5")))
    progress_every = max(5, int(os.environ.get("GRILL_OPEN_STEP3_PROGRESS_EVERY", "15")))
    sample_trials = max(1, int(os.environ.get("GRILL_OPEN_STEP3_SAMPLE_TRIALS", "10")))
    sample_max_configs = max(1, int(os.environ.get("GRILL_OPEN_STEP3_SAMPLE_MAX_CONFIGS", "8")))
    sample_max_time_ms = max(5, int(os.environ.get("GRILL_OPEN_STEP3_SAMPLE_MAX_TIME_MS", "30")))
    pose_pos_tol = float(os.environ.get("GRILL_OPEN_STEP3_POSE_POS_TOL", "0.010"))
    pose_ori_tol = float(os.environ.get("GRILL_OPEN_STEP3_POSE_ORI_TOL", "0.060"))
    no_joint_interp = os.environ.get("GRILL_OPEN_STEP3_NO_JOINT_INTERP", "True") == "True"
    use_sampling_fallback = os.environ.get("GRILL_OPEN_STEP3_USE_SAMPLING_FALLBACK", "False") == "True"
    max_joint_jump = float(os.environ.get("GRILL_OPEN_STEP3_MAX_JOINT_JUMP_RAD", "0.16"))
    orient_mode = os.environ.get("GRILL_OPEN_STEP3_ORIENT_MODE", "face-handle").strip().lower()
    enable_correction = os.environ.get("GRILL_OPEN_STEP3_ENABLE_CORRECTION", "False") == "True"
    pos_mode = os.environ.get("GRILL_OPEN_STEP3_POS_MODE", "auto").strip().lower()
    if pos_mode == "auto":
        pos_mode = "tip-arc" if orient_mode in {"face-handle", "face_handle", "face"} else "handle-lock"
    enforce_contact_default = "False" if orient_mode in {"face-handle", "face_handle", "face"} else "True"
    enforce_contact = os.environ.get("GRILL_OPEN_STEP3_ENFORCE_CONTACT", enforce_contact_default) == "True"
    face_tip_mode = (orient_mode in {"face-handle", "face_handle", "face"}) and (
        pos_mode in {"tip-arc", "tip_arc", "arc"}
    )
    pose_pos_tol_soft = float(
        os.environ.get(
            "GRILL_OPEN_STEP3_POSE_POS_TOL_SOFT",
            "0.030" if face_tip_mode else f"{max(0.015, pose_pos_tol):.3f}",
        )
    )
    pose_ori_tol_soft = float(
        os.environ.get(
            "GRILL_OPEN_STEP3_POSE_ORI_TOL_SOFT",
            "0.250" if face_tip_mode else f"{max(0.120, pose_ori_tol):.3f}",
        )
    )
    max_joint_jump_soft = float(
        os.environ.get(
            "GRILL_OPEN_STEP3_MAX_JOINT_JUMP_RAD_SOFT",
            f"{(1.8 if face_tip_mode else 1.3) * max_joint_jump:.3f}",
        )
    )

    solved = 0
    skipped = 0
    skip_streak = 0
    reached = float(current_joint)
    q_curr = np.array(env.get_robot_conf(), dtype=float)
    worst_contact_err = 0.0
    total_substeps = 0
    t0 = time.time()
    hard_contact_streak = 0

    print(f"  Step-3 online arc points: {n_pts} (target {open_deg:.1f} deg)")
    print(
        f"  Step-3 mode: orient={orient_mode}, pos={pos_mode}, enforce_contact={enforce_contact} | "
        f"tol=({pose_pos_tol:.3f},{pose_ori_tol:.3f}) soft=({pose_pos_tol_soft:.3f},{pose_ori_tol_soft:.3f})"
    )
    for i, a in enumerate(angles[1:], start=1):
        delta = float(a - current_joint)
        R_delta = _axis_angle_matrix(hinge_axis, delta)
        hpos_des = hinge_pos + (R_delta @ (hpos0 - hinge_pos))
        hR_des = R_delta @ hR0
        if pos_mode in {"tip-arc", "tip_arc", "arc"}:
            tip_pos_des = hinge_pos + (R_delta @ r_tip0)
        else:
            tip_pos_des = hpos_des + (hR_des @ p_rel_handle)
        tip_R_des = hR_des @ R_rel_handle
        tip_quat_des = _rot_to_quat(tip_R_des)

        tip_quat_curr = None
        try:
            tip_quat_curr = np.array(tip_obj.get_quaternion(), dtype=float).tolist()
        except Exception:
            pass

        if orient_mode in {"face-handle", "face_handle", "face"}:
            facing = hpos_des - tip_pos_des
            quat_targets = _face_handle_quat_candidates(facing, hinge_axis=hinge_axis)
            if tip_quat_curr is not None:
                quat_targets = [tip_quat_curr] + quat_targets
        else:
            quat_targets = [tip_quat_des]
            if tip_quat_curr is not None:
                quat_targets.append(tip_quat_curr)
            quat_targets.append(_rot_to_quat(_axis_angle_matrix(hinge_axis, 0.10) @ tip_R_des))
            quat_targets.append(_rot_to_quat(_axis_angle_matrix(hinge_axis, -0.10) @ tip_R_des))

        cfgs_tagged = []
        # Fast path: local Jacobian IK for each orientation target.
        for q_ref in quat_targets:
            try:
                q_jac = env.robot.solve_ik_via_jacobian(
                    tip_pos_des.tolist(),
                    quaternion=q_ref,
                )
                if q_jac is not None and len(q_jac) > 0:
                    cfgs_tagged.append((np.array(q_jac, dtype=float), q_ref))
            except Exception:
                pass
        # Optional fallback: bounded sampling IK (strict limits to avoid stalls).
        if len(cfgs_tagged) == 0 and use_sampling_fallback:
            for ignore_collisions in (False, True):
                for q_ref in quat_targets:
                    try:
                        cfgs = env.robot.solve_ik_via_sampling(
                            tip_pos_des.tolist(),
                            quaternion=q_ref,
                            trials=sample_trials,
                            max_configs=sample_max_configs,
                            max_time_ms=sample_max_time_ms,
                            ignore_collisions=ignore_collisions,
                        )
                    except Exception:
                        cfgs = None
                    if cfgs is not None and len(cfgs) > 0:
                        for c in cfgs:
                            cfgs_tagged.append((np.array(c, dtype=float), q_ref))
                if len(cfgs_tagged) > 0:
                    break

        if len(cfgs_tagged) == 0:
            skipped += 1
            skip_streak += 1
            if (skip_streak % 10) == 1:
                print(f"  [skip] arc point {i}/{n_pts - 1} unreachable, skip_streak={skip_streak}")
            if skip_streak >= max_skip_streak:
                print(f"  [warn] large skip streak ({skip_streak}), continuing with later arc points")
            try:
                env.gripper.actuate(0.0, 0.10)
            except Exception:
                pass
            pr.step()
            continue

        # Filter candidates by actual achieved TCP pose quality.
        q_backup = np.array(env.get_robot_conf(), dtype=float)
        strict_valid = []
        soft_valid = []
        for qc, q_ref in cfgs_tagged[:48]:
            env.set_robot_conf(qc.tolist())
            tip_pc = np.array(tip_obj.get_position(), dtype=float)
            tip_qc = np.array(tip_obj.get_quaternion(), dtype=float)
            pos_err = float(np.linalg.norm(tip_pc - tip_pos_des))
            align = _quat_align_abs(tip_qc, q_ref)
            ori_err = float(1.0 - align)
            move_cost = float(np.linalg.norm(qc - q_curr))
            if move_cost > max_joint_jump_soft:
                continue
            cont_err = 0.0
            if tip_quat_curr is not None:
                cont_err = float(1.0 - _quat_align_abs(q_ref, tip_quat_curr))
            if face_tip_mode:
                score = 4.0 * pos_err + 0.6 * ori_err + 0.08 * move_cost + 0.35 * cont_err
            else:
                score = 6.0 * pos_err + 2.0 * ori_err + 0.10 * move_cost + 0.35 * cont_err

            if (move_cost <= max_joint_jump) and (pos_err <= pose_pos_tol) and (ori_err <= pose_ori_tol):
                strict_valid.append((score, qc, q_ref))
            elif (pos_err <= pose_pos_tol_soft) and (ori_err <= pose_ori_tol_soft):
                # Degraded acceptance for hinge-follow mode to avoid deadlock.
                soft_valid.append((score + 0.35, qc, q_ref))
        env.set_robot_conf(q_backup.tolist())

        if len(strict_valid) > 0:
            valid = strict_valid
        elif len(soft_valid) > 0:
            valid = soft_valid
        else:
            skipped += 1
            skip_streak += 1
            if (skip_streak % 8) == 1:
                print(
                    f"  [skip] arc point {i}/{n_pts - 1} rejected by pose tolerance "
                    f"(tol_pos={pose_pos_tol:.3f}, tol_ori={pose_ori_tol:.3f}, "
                    f"soft_pos={pose_pos_tol_soft:.3f}, soft_ori={pose_ori_tol_soft:.3f})"
                )
            try:
                env.gripper.actuate(0.0, 0.10)
            except Exception:
                pass
            pr.step()
            continue

        best = min(valid, key=lambda it: float(it[0]))
        q_best = best[1]
        q_best_ref = best[2]
        if no_joint_interp:
            # Important: avoid joint-space interpolation drift between two IK-valid
            # arc points; this was causing early handle escape from finger gap.
            env.set_robot_conf(q_best.tolist())
            _set_lid_angle_sync(env, float(a))
            try:
                env.gripper.actuate(0.0, 0.10)
            except Exception:
                pass
            pr.step()
            total_substeps += 1
        else:
            total_substeps += _move_robot_and_lid_smooth(
                env=env,
                pr=pr,
                q_from=q_curr,
                q_to=q_best,
                lid_from=reached,
                lid_to=a,
                max_joint_step=max_joint_step,
            )

        # Contact consistency check in handle-local frame.
        hR_now, hpos_now = _shape_rot_pos(handle_obj)
        tip_now = np.array(tip_obj.get_position(), dtype=float)
        if pos_mode in {"tip-arc", "tip_arc", "arc"}:
            # In hinge-style mode, allow contact point rotation while preserving
            # the clamp radius around handle center.
            dist_now = float(np.linalg.norm(tip_now - hpos_now))
            contact_err = float(abs(dist_now - handle_tip_dist0))
            tip_contact_now = hpos_now
        else:
            tip_contact_now = hpos_now + (hR_now @ p_rel_handle)
            contact_err = float(np.linalg.norm(tip_now - tip_contact_now))
        if contact_err > worst_contact_err:
            worst_contact_err = contact_err
        if enable_correction and (contact_err > soft_contact_tol):
            # Corrective IK at the same arc angle using actual handle pose.
            if orient_mode in {"face-handle", "face_handle", "face"}:
                facing_now = hpos_now - tip_now
                corr_quats = [q_best_ref] + _face_handle_quat_candidates(facing_now, hinge_axis=hinge_axis)
            else:
                tip_R_now = hR_now @ R_rel_handle
                corr_quats = [_rot_to_quat(tip_R_now)]
            corr_cfgs = []
            for cq in corr_quats[:6]:
                try:
                    q_corr_jac = env.robot.solve_ik_via_jacobian(
                        tip_contact_now.tolist(),
                        quaternion=cq,
                    )
                    if q_corr_jac is not None and len(q_corr_jac) > 0:
                        corr_cfgs.append(np.array(q_corr_jac, dtype=float))
                except Exception:
                    pass
            if len(corr_cfgs) == 0:
                for ignore_collisions in (False, True):
                    try:
                        out = env.robot.solve_ik_via_sampling(
                            tip_contact_now.tolist(),
                            quaternion=q_best_ref,
                            trials=max(1, sample_trials // 2),
                            max_configs=max(2, sample_max_configs // 2),
                            max_time_ms=max(5, sample_max_time_ms // 2),
                            ignore_collisions=ignore_collisions,
                        )
                    except Exception:
                        out = None
                    if out is not None and len(out) > 0:
                        corr_cfgs.extend(out)
                    if len(corr_cfgs) > 0:
                        break
            if len(corr_cfgs) > 0:
                q_corr = min(corr_cfgs, key=lambda cc: float(np.linalg.norm(np.array(cc, dtype=float) - q_best)))
                q_corr = np.array(q_corr, dtype=float)
                if no_joint_interp:
                    env.set_robot_conf(q_corr.tolist())
                    _set_lid_angle_sync(env, float(a))
                    try:
                        env.gripper.actuate(0.0, 0.10)
                    except Exception:
                        pass
                    pr.step()
                    total_substeps += 1
                else:
                    total_substeps += _move_robot_and_lid_smooth(
                        env=env,
                        pr=pr,
                        q_from=q_best,
                        q_to=q_corr,
                        lid_from=a,
                        lid_to=a,
                        max_joint_step=max_joint_step,
                    )
                q_best = q_corr
                hR_now, hpos_now = _shape_rot_pos(handle_obj)
                tip_now = np.array(tip_obj.get_position(), dtype=float)
                tip_contact_now = hpos_now + (hR_now @ p_rel_handle)
                contact_err = float(np.linalg.norm(tip_now - tip_contact_now))
                if contact_err > worst_contact_err:
                    worst_contact_err = contact_err
        if contact_err > hard_contact_tol and (i % 12 == 0):
            print(f"  [warn] contact error high at arc point {i}/{n_pts - 1}: {contact_err:.3f} m")
        if enforce_contact and (contact_err > hard_contact_tol):
            hard_contact_streak += 1
            if hard_contact_streak >= hard_contact_streak_limit:
                raise RuntimeError(
                    f"Step-3 contact lost repeatedly (>{hard_contact_tol:.3f} m for "
                    f"{hard_contact_streak} points). Aborting to avoid cranky motion."
                )
        else:
            hard_contact_streak = 0
        if (i % progress_every) == 0:
            elapsed = time.time() - t0
            reached_deg_now = abs(float(np.rad2deg(a - current_joint)))
            print(
                f"  [progress] {i}/{n_pts - 1} | reached={reached_deg_now:.1f} deg | "
                f"solved={solved} skipped={skipped} | t={elapsed:.1f}s"
            )

        q_curr = q_best
        solved += 1
        skip_streak = 0
        reached = float(a)

    reached_deg = abs(float(np.rad2deg(reached - current_joint)))
    print(
        f"  Step-3 reached: {reached_deg:.1f} deg | solved_points={solved} | skipped_points={skipped} | "
        f"substeps={total_substeps} | worst_contact_err={worst_contact_err:.3f} m"
    )
    if solved <= 0:
        raise RuntimeError("Step-3 failed: no online IK arc points solved")
    if reached_deg < min_reached_deg:
        raise RuntimeError(
            f"Step-3 failed: reached only {reached_deg:.1f} deg (< {min_reached_deg:.1f} deg)"
        )


def main():
    parser = argparse.ArgumentParser(description="PDDL open-grill Step-1+Step-2+Step-3 runner")
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
    parser.add_argument(
        "--open-deg",
        type=float,
        default=90.0,
        help="Step-3 opening arc angle in degrees (default: 90).",
    )
    args = parser.parse_args()

    os.environ["HEADLESS"] = "False"
    os.environ["GRILL_SCENE_FILE"] = os.path.join(THIS_DIR, "grill.variation1.ttt")
    os.environ.setdefault("GRILL_PRESERVE_SCENE_LID_POSE", "True")
    os.environ["GRILL_OPEN_STEP1_X_OFFSET"] = str(float(args.x_offset))
    os.environ["GRILL_OPEN_STEP2_X_CLEARANCE"] = str(float(args.grasp_clearance))
    os.environ["GRILL_OPEN_STEP3_TARGET_DEG"] = str(float(args.open_deg))
    # Force Step-3 to be generated online during execution (IK per arc point).
    os.environ["GRILL_OPEN_USE_ONLINE_STEP3"] = "True"

    from grill_task_streams import ENV, get_stream_map

    env = ENV
    pr = env.pr
    home_q = list(env.get_home_conf())
    env.set_robot_conf(home_q)
    for _ in range(15):
        pr.step()

    print("=" * 72)
    print("PDDL OPEN-GRILL TEST (STEP-1 + STEP-2 + STEP-3)")
    print("=" * 72)
    print(f"Scene: {os.environ.get('GRILL_SCENE_FILE')}")
    print(f"Step-1 X offset: {os.environ.get('GRILL_OPEN_STEP1_X_OFFSET')} m")
    print(f"Step-2 grasp clearance: {os.environ.get('GRILL_OPEN_STEP2_X_CLEARANCE')} m")
    print(f"Step-3 open angle: {os.environ.get('GRILL_OPEN_STEP3_TARGET_DEG')} deg")

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
        if len(segments) < 2:
            print(f"ERROR: expected at least 2 segments (Step-1/2), got {len(segments)}.")
            break

        print("Executing Step-1 and Step-2...")
        _execute_segment(env, pr, segments[0], "Segment 1 (home -> hover)")
        _execute_segment(env, pr, segments[1], "Segment 2 (hover -> grasp)")

        print("Closing gripper on handle...")
        # Never attach/re-parent the handle to gripper; keep it part of the lid.
        try:
            env.gripper.release()
        except Exception:
            pass
        _close_gripper(env, pr, target=0.0, velocity=0.14, max_steps=140)
        # Let fingers settle around the handle before starting arc.
        for _ in range(35):
            try:
                env.gripper.actuate(0.0, 0.08)
            except Exception:
                pass
            pr.step()

        print("Executing Step-3 circular hinged arc...")
        try:
            _execute_hinged_arc_online(env, pr, float(args.open_deg))
        except RuntimeError as e:
            print(f"ERROR: {e}")
            executed = False
            break
        executed = True
        break

    if not executed:
        print("ERROR: open-grill execution failed.")
        print("Press Ctrl+C to close.")
        try:
            while True:
                pr.step()
        except KeyboardInterrupt:
            pass
        pr.stop()
        pr.shutdown()
        return

    print("Done. Smooth circular hinged motion complete.")
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

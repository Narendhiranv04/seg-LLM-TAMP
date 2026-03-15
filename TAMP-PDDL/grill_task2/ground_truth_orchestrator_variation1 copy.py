"""
Ground Truth Orchestrator - Grill Task Variation 1

Scene:
  grill_task2/grill.variation1.ttt

Sequence requested:
1. Open grill (circular hinge motion)
2. Pick plate (vertical grasp) -> place plate on plate_boundary (horizontal placement)
3. Pick steak inside grill -> place on plate
4. Pick chicken + steak outside -> place on grill
5. Close grill (circular hinge motion)
6. Open grill (circular hinge motion)
7. Pick chicken + steak inside grill -> place on plate
"""

import os
import sys
import math
import time
import numpy as np

from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.const import JointMode
from pyrep.backend import sim


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)

# Scene selection:
# Use the editable variation scene by default.
DEFAULT_SCENE_CLOSED = os.path.join(THIS_DIR, "GrillClosed_SteakIn_SteakChickenOut.ttt")
DEFAULT_SCENE_VARIATION = os.path.join(THIS_DIR, "grill.variation1.ttt")
ALLOW_SCENE_OVERRIDE = os.environ.get("GRILL_ALLOW_SCENE_OVERRIDE", "False") == "True"
SCENE_OVERRIDE = os.environ.get("GRILL_SCENE_FILE_OVERRIDE", "").strip()
if ALLOW_SCENE_OVERRIDE and SCENE_OVERRIDE:
    SCENE_PATH = SCENE_OVERRIDE
else:
    SCENE_PATH = DEFAULT_SCENE_VARIATION if os.path.exists(DEFAULT_SCENE_VARIATION) else DEFAULT_SCENE_CLOSED

# Force this scene for shared env.
os.environ["GRILL_SCENE_FILE"] = SCENE_PATH
os.environ["HEADLESS"] = "False"
# Set robust lid defaults before ENV is imported/constructed.
os.environ.setdefault("GRILL_LID_TRAVEL_ANGLE", f"{math.radians(95.0):.6f}")
os.environ.setdefault("GRILL_LID_CLOSED_ANGLE", "0.0")
os.environ.setdefault("GRILL_LID_OPEN_ANGLE", f"{math.radians(95.0):.6f}")
os.environ.setdefault("GRILL_LID_AUTOCALIBRATE", "False")

if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from grill_task_streams import ENV  # noqa: E402
from grill_task_env import quaternion_from_euler  # noqa: E402

LID_TRAVEL_ANGLE = float(os.environ.get("GRILL_LID_TRAVEL_ANGLE", f"{math.radians(95.0):.6f}"))
LID_CLOSED_ANGLE = float(os.environ.get("GRILL_LID_CLOSED_ANGLE", "0.0"))
LID_OPEN_ANGLE = float(os.environ.get("GRILL_LID_OPEN_ANGLE", str(LID_CLOSED_ANGLE + LID_TRAVEL_ANGLE)))
MIN_OPEN_TRAVEL_RAD = float(os.environ.get("GRILL_MIN_OPEN_TRAVEL_RAD", f"{math.radians(95.0):.6f}"))
FORCE_MIN_OPEN_TRAVEL = os.environ.get("GRILL_FORCE_MIN_OPEN_TRAVEL", "False") == "True"
LID_ANGLE_TOL = float(os.environ.get("GRILL_LID_ANGLE_TOL", "0.06"))
LID_AUTOCALIBRATE = os.environ.get("GRILL_LID_AUTOCALIBRATE", "False") == "True"
HANDLE_PROBE_AT_STARTUP = os.environ.get("GRILL_HANDLE_PROBE_AT_STARTUP", "False") == "True"
USE_LID_WAYPOINT_SNAP = os.environ.get("GRILL_USE_LID_WAYPOINT_SNAP", "False") == "True"
STARTUP_CLOSE_SCAN = os.environ.get("GRILL_STARTUP_CLOSE_SCAN", "False") == "True"
USE_INITIAL_LID_AS_CLOSED = os.environ.get("GRILL_USE_INITIAL_LID_AS_CLOSED", "True") == "True"
KEEP_LID_COLLISION_OFF_UNTIL_OPEN = os.environ.get(
    "GRILL_KEEP_LID_COLLISION_OFF_UNTIL_OPEN", "False"
) == "True"
ENABLE_HANDLE_ATTACH = os.environ.get("GRILL_ENABLE_HANDLE_ATTACH", "False") == "True"
HANDLE_REANCHOR_ALWAYS = os.environ.get("GRILL_HANDLE_REANCHOR_ALWAYS", "True") == "True"
ACTIVE_LID_JOINT_HANDLE = None
ACTIVE_HANDLE_SHAPE_HANDLE = None
HANDLE_ANCHOR_PARENT_HANDLE = None
HANDLE_ANCHOR_REL_MATRIX = None
CLOSE_WP_NAME = "waypoint23"
OPEN_WP_NAME = "waypoint29"


def _enforce_min_open_travel():
    global LID_OPEN_ANGLE, LID_CLOSED_ANGLE
    if not FORCE_MIN_OPEN_TRAVEL:
        return
    delta = float(LID_OPEN_ANGLE - LID_CLOSED_ANGLE)
    sign = 1.0 if delta >= 0.0 else -1.0
    if abs(delta) < 1e-6:
        sign = 1.0
    if abs(delta) < float(MIN_OPEN_TRAVEL_RAD):
        LID_OPEN_ANGLE = float(LID_CLOSED_ANGLE + sign * float(MIN_OPEN_TRAVEL_RAD))


def step(pr, n=1):
    for _ in range(max(1, int(n))):
        pr.step()


def _interp_traj(traj, steps_per_segment=12):
    if traj is None or len(traj) == 0:
        return []
    if len(traj) == 1:
        return [traj[0]]
    dense = []
    for i in range(len(traj) - 1):
        q1 = np.array(traj[i], dtype=float)
        q2 = np.array(traj[i + 1], dtype=float)
        for t in np.linspace(0.0, 1.0, max(1, int(steps_per_segment)), endpoint=False):
            dense.append(((1.0 - t) * q1 + t * q2).tolist())
    dense.append(list(traj[-1]))
    return dense


def execute_trajectory(env, pr, traj, steps_per_segment=12):
    dense = _interp_traj(traj, steps_per_segment=steps_per_segment)
    for q in dense:
        env.set_robot_conf(q)
        pr.step()


def _move_to_conf(env, pr, q_target, label="move"):
    q_current = env.get_robot_conf()
    motion = env.compute_motion_plan(q_current, q_target)
    if motion:
        print(f"[{label}] Planned motion with {len(motion)} waypoints.")
        execute_trajectory(env, pr, motion, steps_per_segment=4)
        return True
    print(f"[{label}] Motion planner failed, using direct interpolation.")
    q1 = np.array(q_current, dtype=float)
    q2 = np.array(q_target, dtype=float)
    fallback = [((1.0 - t) * q1 + t * q2).tolist() for t in np.linspace(0.0, 1.0, 80)]
    execute_trajectory(env, pr, fallback, steps_per_segment=1)
    return True


def go_home(env, pr):
    print("\n--- Returning to Home ---")
    _move_to_conf(env, pr, env.get_home_conf(), label="home")
    step(pr, 20)
    print("Reached Home.")


def _obj_handle(obj):
    try:
        return int(obj.get_handle())
    except Exception:
        return None


def _target_is_grasped(env, target_obj):
    if target_obj is None:
        return False
    target_h = _obj_handle(target_obj)
    if target_h is None:
        return False
    try:
        grabbed = env.gripper.get_grasped_objects()
    except Exception:
        grabbed = []
    return any(_obj_handle(o) == target_h for o in grabbed)


def _get_world_bounds(env, obj):
    return env._get_world_bounding_box(obj)


def _region_object(env, region_name):
    if region_name in env.regions:
        return env.regions[region_name]
    return env.get_object(region_name)


def _is_in_region(env, obj, region_name, tol_xy=0.02, tol_z=0.05):
    region = _region_object(env, region_name)
    if region is None or obj is None:
        return False
    x, y, z = obj.get_position()
    min_x, max_x, min_y, max_y, min_z, max_z = _get_world_bounds(env, region)
    return (
        (min_x - tol_xy) <= x <= (max_x + tol_xy)
        and (min_y - tol_xy) <= y <= (max_y + tol_xy)
        and (min_z - tol_z) <= z <= (max_z + tol_z)
    )


def _scene_shape_names():
    names = []
    try:
        handles = sim.simGetObjectsInTree(sim.sim_handle_scene, sim.sim_handle_all, 0)
        if isinstance(handles, int):
            handles = [handles]
        for h in handles:
            try:
                if sim.simGetObjectType(h) != sim.sim_object_shape_type:
                    continue
            except Exception:
                continue
            name = None
            try:
                name = sim.simGetObjectAlias(h, 5)
            except Exception:
                try:
                    name = sim.simGetObjectName(h)
                except Exception:
                    pass
            if name:
                names.append(str(name))
    except Exception:
        try:
            env.set_lid_collision_enabled(True)
        except Exception:
            pass
        pass
    return names


def _candidate_names(base):
    suffixes = ["", "#0", "#1", "#2", "_0", "_1", "_2", "0", "1", "2"]
    out = []
    for s in suffixes:
        out.append(f"{base}{s}")
    return out


def _discover_meat_objects(env):
    # 1) Explicit common aliases (physical objects first, not visual-only).
    name_candidates = []
    for base in ("steak", "steak1", "steak2", "chicken", "chicken1", "chicken2"):
        name_candidates.extend(_candidate_names(base))

    # 2) Scene names containing steak/chicken (best effort), excluding visuals.
    for n in _scene_shape_names():
        ln = n.lower()
        if ("steak" in ln) or ("chicken" in ln):
            if all(k not in ln for k in ("boundary", "grill", "lid", "handle", "joint", "visual")):
                name_candidates.append(n)

    meats = []
    seen = set()
    for name in name_candidates:
        obj = env.get_object(name)
        if obj is None:
            continue
        h = _obj_handle(obj)
        if h is None or h in seen:
            continue
        seen.add(h)
        ln = str(name).lower()
        label = "steak" if "steak" in ln else ("chicken" if "chicken" in ln else "meat")
        if "visual" in ln:
            continue
        meats.append({"name": name, "obj": obj, "label": label, "handle": h})
    meats.sort(key=lambda m: (m["label"], m["name"]))
    return meats


def _discover_plate(env):
    candidates = []
    for base in ("plate", "plate_visual"):
        candidates.extend(_candidate_names(base))
    for n in _scene_shape_names():
        ln = n.lower()
        if ("plate" in ln) and ("boundary" not in ln):
            candidates.append(n)
    seen = set()
    for name in candidates:
        obj = env.get_object(name)
        if obj is None:
            continue
        h = _obj_handle(obj)
        if h is None or h in seen:
            continue
        seen.add(h)
        return {"name": name, "obj": obj, "handle": h}
    return None


def _classify_meats(env, meats):
    in_grill = []
    on_plate = []
    outside = []
    for m in meats:
        obj = m["obj"]
        inside_grill = _is_in_region(env, obj, "grill-top", tol_xy=0.08, tol_z=0.12)
        inside_plate = _is_in_region(env, obj, "plate-top", tol_xy=0.05, tol_z=0.10) or _is_in_region(
            env, obj, "plate_boundary", tol_xy=0.05, tol_z=0.10
        )
        if inside_grill:
            in_grill.append(m)
        elif inside_plate:
            on_plate.append(m)
        else:
            outside.append(m)
    return in_grill, on_plate, outside


def _region_slot_pose(env, obj, region_name, slot_idx=0, slot_count=1):
    region = _region_object(env, region_name)
    if region is None:
        return env.sample_stable_pose(obj, region_name)

    min_x, max_x, min_y, max_y, _min_z, max_z = _get_world_bounds(env, region)
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    span_x = max_x - min_x
    span_y = max_y - min_y
    if region_name == "grill-top":
        # Wider spacing for grill meats to avoid overlap.
        if slot_count <= 1:
            off = np.array([0.0, 0.0], dtype=float)
        elif slot_count == 2:
            if span_x >= span_y:
                offs = [np.array([-0.28 * span_x, 0.0]), np.array([0.28 * span_x, 0.0])]
            else:
                offs = [np.array([0.0, -0.28 * span_y]), np.array([0.0, 0.28 * span_y])]
            off = offs[min(slot_idx, 1)]
        elif slot_count == 3:
            if span_x >= span_y:
                offs = [np.array([-0.30 * span_x, 0.0]), np.array([0.0, 0.0]), np.array([0.30 * span_x, 0.0])]
            else:
                offs = [np.array([0.0, -0.30 * span_y]), np.array([0.0, 0.0]), np.array([0.0, 0.30 * span_y])]
            off = offs[min(slot_idx, 2)]
        else:
            cols = int(math.ceil(math.sqrt(slot_count)))
            rows = int(math.ceil(slot_count / cols))
            col = slot_idx % cols
            row = slot_idx // cols
            dx = (col - (cols - 1) / 2.0) * max(0.22 * span_x, 0.03)
            dy = (row - (rows - 1) / 2.0) * max(0.22 * span_y, 0.03)
            off = np.array([dx, dy], dtype=float)
    else:
        r = max(0.01, 0.20 * min(span_x, span_y))
        if slot_count <= 1:
            off = np.array([0.0, 0.0], dtype=float)
        elif slot_count == 2:
            offs = [np.array([-r, 0.0]), np.array([r, 0.0])]
            off = offs[min(slot_idx, 1)]
        elif slot_count == 3:
            offs = [np.array([0.0, 0.0]), np.array([-r, 0.0]), np.array([r, 0.0])]
            off = offs[min(slot_idx, 2)]
        else:
            cols = int(math.ceil(math.sqrt(slot_count)))
            rows = int(math.ceil(slot_count / cols))
            col = slot_idx % cols
            row = slot_idx // cols
            dx = (col - (cols - 1) / 2.0) * max(0.6 * r, 0.02)
            dy = (row - (rows - 1) / 2.0) * max(0.6 * r, 0.02)
            off = np.array([dx, dy], dtype=float)

    pose = list(obj.get_pose())
    margin_x = 0.01 * span_x
    margin_y = 0.01 * span_y
    pose[0] = float(np.clip(cx + off[0], min_x + margin_x, max_x - margin_x))
    pose[1] = float(np.clip(cy + off[1], min_y + margin_y, max_y - margin_y))

    # Use current world bottom offset to avoid local-frame bbox artifacts.
    try:
        obj_pos = np.array(obj.get_position(), dtype=float)
        obj_world_min_z = float(_get_world_bounds(env, obj)[4])
        bottom_offset = float(obj_pos[2] - obj_world_min_z)
    except Exception:
        bottom_offset = 0.01
    z_clear = 0.005 if region_name == "grill-top" else 0.002
    pose[2] = float(max_z + bottom_offset + z_clear)
    return pose


def grasp_object(env, pr, target_obj, is_plate=False):
    target_obj.set_dynamic(False)
    env.gripper.actuate(0.0, 0.1)
    step(pr, 35)
    env.gripper.grasp(target_obj)
    step(pr, 12)
    if not is_plate:
        target_obj.set_dynamic(True)
    step(pr, 8)


def release_object(env, pr, target_obj, allow_drop=False, freeze_after_release=True, settle_steps=40):
    detached = False
    for _ in range(3):
        try:
            env.gripper.release()
        except Exception:
            pass
        if target_obj is not None:
            try:
                target_obj.set_parent(None, keep_in_place=True)
            except Exception:
                try:
                    target_obj.set_parent(None)
                except Exception:
                    pass
            try:
                target_obj.set_collidable(True)
                target_obj.set_respondable(True)
                # Keep object kinematic while opening fingers to avoid impulsive fling.
                target_obj.set_dynamic(False)
            except Exception:
                pass

        # Open slower for a stable, non-explosive release.
        _open_gripper_fully(env, pr, velocity=0.18, max_steps=120)
        detached = (target_obj is None) or (not _target_is_grasped(env, target_obj))
        if detached:
            break

    if not detached:
        print("WARNING: release may still be attached; continuing.")

    total_settle = max(1, int(settle_steps))
    if allow_drop and (target_obj is not None):
        # Optional immediate gravity window (kept off for plate; see post-retreat settle).
        drop_steps = max(6, min(14, total_settle // 2))
        try:
            target_obj.set_dynamic(True)
        except Exception:
            pass
        step(pr, drop_steps)
        total_settle = max(1, total_settle - drop_steps)

    if freeze_after_release and (target_obj is not None):
        try:
            target_obj.set_dynamic(False)
        except Exception:
            pass
    step(pr, total_settle)


def settle_placed_object(env, pr, target_obj, drop_steps=20, settle_steps=12):
    """
    After gripper retreats, briefly enable gravity so object settles onto support,
    then freeze to keep final placement stable.
    """
    if target_obj is None:
        return
    try:
        target_obj.set_collidable(True)
        target_obj.set_respondable(True)
        if int(drop_steps) > 0:
            target_obj.set_dynamic(True)
    except Exception:
        return
    if int(drop_steps) > 0:
        step(pr, max(1, int(drop_steps)))
    try:
        target_obj.set_dynamic(False)
    except Exception:
        pass
    step(pr, max(1, int(settle_steps)))


def settle_on_region_without_snap(
    env,
    pr,
    target_obj,
    target_region,
    max_drop_steps=80,
    settle_steps=12,
    support_tol=0.006,
):
    """
    Physics-only settle: enable gravity, wait until object bottom reaches/supports
    the region top (within tolerance), then freeze. No pose snapping.
    """
    if (target_obj is None) or (target_region is None):
        return
    region = _region_object(env, target_region)
    if region is None:
        settle_placed_object(env, pr, target_obj, drop_steps=max_drop_steps // 2, settle_steps=settle_steps)
        return
    try:
        _min_x, _max_x, _min_y, _max_y, _min_z, region_max_z = _get_world_bounds(env, region)
        target_obj.set_collidable(True)
        target_obj.set_respondable(True)
        target_obj.set_dynamic(True)
    except Exception:
        return

    stable_hits = 0
    last_min_z = None
    for _ in range(max(1, int(max_drop_steps))):
        step(pr, 1)
        try:
            obj_min_z = float(_get_world_bounds(env, target_obj)[4])
        except Exception:
            continue
        if obj_min_z <= float(region_max_z + support_tol):
            if (last_min_z is not None) and (abs(obj_min_z - last_min_z) < 8e-4):
                stable_hits += 1
            else:
                stable_hits = 1
            if stable_hits >= 5:
                break
        last_min_z = obj_min_z

    try:
        target_obj.set_dynamic(False)
    except Exception:
        pass
    step(pr, max(1, int(settle_steps)))


def lift_object_if_submerged(env, pr, target_obj, target_region, clearance=0.003):
    """
    Anti-submerge correction: only lift object in Z if its world bottom is
    below the support top + clearance. Keeps XY/orientation unchanged.
    """
    if (target_obj is None) or (target_region is None):
        return
    region = _region_object(env, target_region)
    if region is None:
        return
    try:
        _min_x, _max_x, _min_y, _max_y, _min_z, region_max_z = _get_world_bounds(env, region)
        obj_min_z = float(_get_world_bounds(env, target_obj)[4])
    except Exception:
        return
    target_min_z = float(region_max_z + float(clearance))
    if obj_min_z >= target_min_z:
        return
    dz = float(target_min_z - obj_min_z)
    try:
        pose = list(target_obj.get_pose())
        pose[2] = float(pose[2] + dz)
        target_obj.set_pose(pose)
        target_obj.set_dynamic(False)
    except Exception:
        return
    step(pr, 4)


def stabilize_object_on_region(
    env,
    pr,
    target_obj,
    target_region,
    target_xy=None,
    clearance=0.004,
):
    """
    Snap object just above region top (no penetration), then freeze.
    This avoids the 'submerge then shoot up' instability.
    """
    if (target_obj is None) or (target_region is None):
        return
    region = _region_object(env, target_region)
    if region is None:
        return

    try:
        min_x, max_x, min_y, max_y, _min_z, max_z = _get_world_bounds(env, region)
    except Exception:
        return

    try:
        pose = list(target_obj.get_pose())
    except Exception:
        return

    if target_xy is not None:
        tx, ty = float(target_xy[0]), float(target_xy[1])
    else:
        tx, ty, _tz = target_obj.get_position()
    margin = 0.006
    tx = float(np.clip(tx, min_x + margin, max_x - margin))
    ty = float(np.clip(ty, min_y + margin, max_y - margin))
    pose[0] = tx
    pose[1] = ty

    # Iteratively lift using current world bbox min-z.
    for _ in range(3):
        try:
            target_obj.set_pose(pose)
            o_min_z = float(_get_world_bounds(env, target_obj)[4])
        except Exception:
            break
        desired_min_z = float(max_z + clearance)
        dz = desired_min_z - o_min_z
        if abs(dz) < 5e-4:
            break
        pose[2] = float(pose[2] + dz)

    try:
        target_obj.set_pose(pose)
        target_obj.set_collidable(True)
        target_obj.set_respondable(True)
        target_obj.set_dynamic(False)
    except Exception:
        pass
    step(pr, 8)


def run_pick_place(
    env,
    pr,
    obj_name,
    target_region,
    task_name,
    is_plate=False,
    target_pose=None,
):
    print("\n" + "=" * 68)
    print(f"TASK: {task_name}")
    print(f"Pick '{obj_name}' -> Place in '{target_region}'")
    print("=" * 68)

    target_obj = env.get_object(obj_name)
    if target_obj is None:
        print(f"ERROR: object '{obj_name}' not found")
        return False

    pose_before = list(target_obj.get_pose())
    pos_before = np.array(target_obj.get_position(), dtype=float)
    target_obj.set_dynamic(False)

    # Pick
    try:
        q_hover, hover_quat = env.compute_hover_config(target_obj, list(pose_before), hover_offset=0.15)
    except Exception as e:
        print(f"ERROR: hover failed: {e}")
        return False

    _move_to_conf(env, pr, q_hover, label="move->pick_hover")
    step(pr, 10)

    try:
        _grasp, _q1, _q2, (approach_traj, retreat_traj) = env.compute_pick_trajectory(
            target_obj,
            list(pose_before),
            preferred_orientation=hover_quat,
            is_plate=is_plate,
        )
    except Exception as e:
        print(f"ERROR: pick trajectory failed: {e}")
        return False

    current_q = env.get_robot_conf()
    if approach_traj and not np.allclose(current_q, approach_traj[0], atol=0.05):
        _move_to_conf(env, pr, approach_traj[0], label="align_pick_start")

    execute_trajectory(env, pr, approach_traj, steps_per_segment=8)
    print("Grasping...")
    grasp_object(env, pr, target_obj, is_plate=is_plate)
    execute_trajectory(env, pr, retreat_traj, steps_per_segment=8)
    step(pr, 8)

    # Place
    if target_pose is None:
        place_pose = env.sample_stable_pose(target_obj, target_region)
    else:
        place_pose = list(target_pose)
        while len(place_pose) < 7:
            place_pose.append(0.0)
        if len(place_pose) == 7 and np.linalg.norm(np.array(place_pose[3:], dtype=float)) < 1e-5:
            place_pose[3:] = [0.0, 0.0, 0.0, 1.0]

    if is_plate:
        # Cleaner plate placement:
        # move planner -> pre-place HIGH side hover
        # linear down -> pre-place LOW
        # linear horizontal in -> place
        # release + settle
        # linear up -> linear horizontal out
        region = _region_object(env, target_region)
        if region is None:
            print(f"ERROR: region '{target_region}' not found for plate placement")
            return False
        min_x, max_x, min_y, max_y, _min_z, max_z = _get_world_bounds(env, region)
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        # Release slightly above support and settle after retreat (no snap).
        release_height = float(os.environ.get("GRILL_PLATE_RELEASE_HEIGHT", "0.015"))
        hover_z_offset = float(os.environ.get("GRILL_PLATE_HOVER_Z_OFFSET", "0.012"))
        # Keep pre-place hover and post-release retrieve hover at the same height.
        pre_place_z_offset = hover_z_offset
        post_release_lift = hover_z_offset
        place_z = float(max_z + release_height)
        pre_place_z = float(place_z + pre_place_z_offset)

        hover_dx = float(os.environ.get("GRILL_PLATE_HOVER_DX", "-0.16"))
        place_x_offset = float(os.environ.get("GRILL_PLATE_PLACE_X_OFFSET", "-0.03"))
        place_pos = [cx + place_x_offset, cy, place_z]
        pre_place_low_pos = [place_pos[0] + hover_dx, cy, place_z]
        pre_place_high_pos = [place_pos[0] + hover_dx, cy, pre_place_z]

        # Horizontal gripper, fingers vertical.
        quat_candidates = [
            quaternion_from_euler(np.pi / 2, 0.0, np.pi / 2),
            quaternion_from_euler(np.pi / 2, np.pi / 2, np.pi / 2),
            quaternion_from_euler(np.pi / 2, -np.pi / 2, np.pi / 2),
            quaternion_from_euler(np.pi / 2, np.pi, np.pi / 2),
        ]

        chosen = None
        q_curr = np.array(env.get_robot_conf(), dtype=float)
        for qrot in quat_candidates:
            try:
                cfg_pre = env.robot.solve_ik_via_sampling(
                    pre_place_high_pos, quaternion=qrot, max_configs=25, max_time_ms=240, ignore_collisions=True
                )
            except Exception:
                cfg_pre = None
            if cfg_pre is None or len(cfg_pre) == 0:
                continue
            q_pre = min(cfg_pre, key=lambda c: float(np.linalg.norm(np.array(c, dtype=float) - q_curr)))

            # Avoid messy direct interpolation fallback: require a real planner path
            # from post-pick to pre-place high hover.
            motion_to_pre = env.compute_motion_plan(env.get_robot_conf(), q_pre)
            if not motion_to_pre:
                continue

            # Linear down -> horizontal in -> up -> out.
            path_down = env._get_linear_path(q_pre, pre_place_low_pos, qrot, ignore_collisions=True, steps=50)
            if not path_down:
                continue
            traj_down = path_down._path_points.reshape(-1, 7).tolist()
            q_low = traj_down[-1]

            path_in = env._get_linear_path(q_low, place_pos, qrot, ignore_collisions=True, steps=70)
            if not path_in:
                continue
            traj_in = path_in._path_points.reshape(-1, 7).tolist()
            q_place = traj_in[-1]

            # IMPORTANT: retreat back horizontally first (same z), then go up.
            # Going up first can hook/lift the plate and launch it.
            path_out_low = env._get_linear_path(q_place, pre_place_low_pos, qrot, ignore_collisions=True, steps=70)
            if not path_out_low:
                continue
            traj_out_low = path_out_low._path_points.reshape(-1, 7).tolist()
            q_out_low = traj_out_low[-1]

            path_up = env._get_linear_path(q_out_low, pre_place_high_pos, qrot, ignore_collisions=True, steps=55)
            if not path_up:
                continue
            traj_up = path_up._path_points.reshape(-1, 7).tolist()

            chosen = (motion_to_pre, traj_down, traj_in, traj_out_low, traj_up)
            break

        if chosen is None:
            print("ERROR: could not compute horizontal plate placement path")
            return False

        motion_to_pre, traj_down, traj_in, traj_out_low, traj_up = chosen
        print("[move->plate_pre_place_hover] Planned motion with {} waypoints.".format(len(motion_to_pre)))
        execute_trajectory(env, pr, motion_to_pre, steps_per_segment=4)
        execute_trajectory(env, pr, traj_down, steps_per_segment=5)  # vertical down to low pre-place
        execute_trajectory(env, pr, traj_in, steps_per_segment=6)  # horizontal in

        print("Releasing...")
        release_object(
            env,
            pr,
            target_obj,
            allow_drop=False,
            freeze_after_release=True,
            settle_steps=8,
        )
        lift_object_if_submerged(
            env,
            pr,
            target_obj,
            target_region=target_region,
            clearance=float(os.environ.get("GRILL_PLATE_MIN_CLEARANCE", "0.003")),
        )
        # Hard guard: never retreat while still grasping the plate.
        if _target_is_grasped(env, target_obj):
            print("[plate] WARNING: still grasped after release, forcing detach...")
            for _ in range(3):
                try:
                    env.gripper.release()
                except Exception:
                    pass
                _open_gripper_fully(env, pr, velocity=0.2, max_steps=90)
                step(pr, 4)
                if not _target_is_grasped(env, target_obj):
                    break
        if _target_is_grasped(env, target_obj):
            print("ERROR: plate remained attached after forced release")
            return False

        execute_trajectory(env, pr, traj_out_low, steps_per_segment=6)  # horizontal retreat first
        execute_trajectory(env, pr, traj_up, steps_per_segment=5)  # then move up
        lift_object_if_submerged(
            env,
            pr,
            target_obj,
            target_region=target_region,
            clearance=float(os.environ.get("GRILL_PLATE_MIN_CLEARANCE", "0.003")),
        )
        if os.environ.get("GRILL_PLATE_ENABLE_PHYSICS_SETTLE", "False") == "True":
            settle_on_region_without_snap(
                env,
                pr,
                target_obj,
                target_region=target_region,
                max_drop_steps=int(os.environ.get("GRILL_PLATE_POST_RETREAT_DROP_STEPS", "140")),
                settle_steps=int(os.environ.get("GRILL_PLATE_POST_RETREAT_SETTLE_STEPS", "16")),
                support_tol=float(os.environ.get("GRILL_PLATE_SUPPORT_TOL", "0.010")),
            )
        else:
            # Keep plate exactly where it was placed; avoid post-release re-enabling
            # dynamics that can cause upward launch on complex meshes.
            try:
                target_obj.set_dynamic(False)
                target_obj.set_collidable(True)
                target_obj.set_respondable(True)
            except Exception:
                pass
            step(pr, 8)
        step(pr, 10)

        pos_after = np.array(target_obj.get_position(), dtype=float)
        disp = float(np.linalg.norm(pos_after - pos_before))
        in_region = _is_in_region(env, target_obj, target_region, tol_xy=0.04, tol_z=0.06)
        if (disp < 0.03) or (not in_region):
            print(f"ERROR: validation failed for '{obj_name}' (disp={disp:.3f}, in_region={in_region})")
            return False
        print(f"✓ Validation passed: moved {disp:.3f}m and inside '{target_region}'")
        return True

    try:
        _g2, q_place_start, _q_place_end, (down_traj, up_traj) = env.compute_place_trajectory(
            target_obj,
            place_pose,
            region_name=target_region,
            is_plate=is_plate,
        )
    except Exception as e:
        print(f"ERROR: place trajectory failed: {e}")
        return False

    _move_to_conf(env, pr, q_place_start, label="move->place_hover")
    execute_trajectory(env, pr, down_traj, steps_per_segment=8)
    print("Releasing...")
    release_object(env, pr, target_obj, allow_drop=False, freeze_after_release=True, settle_steps=6)
    execute_trajectory(env, pr, up_traj, steps_per_segment=8)
    if target_region in {"grill-top", "plate-top", "plate_boundary"}:
        settle_placed_object(
            env,
            pr,
            target_obj,
            drop_steps=int(os.environ.get("GRILL_OBJECT_POST_RETREAT_DROP_STEPS", "30")),
            settle_steps=int(os.environ.get("GRILL_OBJECT_POST_RETREAT_SETTLE_STEPS", "14")),
        )
    step(pr, 10)

    pos_after = np.array(target_obj.get_position(), dtype=float)
    disp = float(np.linalg.norm(pos_after - pos_before))
    in_region = _is_in_region(env, target_obj, target_region, tol_xy=0.04, tol_z=0.08)
    if (disp < 0.03) or (not in_region):
        print(f"ERROR: validation failed for '{obj_name}' (disp={disp:.3f}, in_region={in_region})")
        return False

    print(f"✓ Validation passed: moved {disp:.3f}m and inside '{target_region}'")
    return True


def _get_dummy_pose(name):
    try:
        d = Dummy(name)
        return list(d.get_position()), list(d.get_orientation())
    except Exception:
        return None, None


def _quat_to_rot(q):
    x, y, z, w = [float(v) for v in q]
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


def _rot_to_quat(R):
    """
    Convert rotation matrix to quaternion [x, y, z, w].
    """
    m = np.array(R, dtype=float)
    t = float(m[0, 0] + m[1, 1] + m[2, 2])
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=float)
    n = float(np.linalg.norm(q))
    if n < 1e-10:
        return [0.0, 0.0, 0.0, 1.0]
    return (q / n).tolist()


def _unit_xy(v):
    vv = np.array([float(v[0]), float(v[1]), 0.0], dtype=float)
    n = float(np.linalg.norm(vv))
    if n < 1e-8:
        return None
    return vv / n


def _handle_axes(handle_obj):
    try:
        pose = list(handle_obj.get_pose())
        R = _quat_to_rot(pose[3:7])
        return [R[:, 0], R[:, 1], R[:, 2]]
    except Exception:
        return [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]


def _handle_direction_and_perp(handle_obj):
    axes = _handle_axes(handle_obj)
    # Pick most horizontal axis as handle orientation direction.
    best = None
    best_n = -1.0
    for a in axes:
        h = _unit_xy(a)
        if h is None:
            continue
        n = float(np.linalg.norm(h[:2]))
        if n > best_n:
            best_n = n
            best = h
    if best is None:
        best = np.array([1.0, 0.0, 0.0], dtype=float)
    # Perpendicular in XY plane.
    perp = np.array([-best[1], best[0], 0.0], dtype=float)
    return best, perp / max(1e-8, float(np.linalg.norm(perp)))


def _compute_handle_hover_config(env, handle_obj, direction="open"):
    handle_pos = np.array(handle_obj.get_position(), dtype=float)
    handle_dir, handle_perp = _handle_direction_and_perp(handle_obj)
    if direction == "close":
        # Close: approach from a side perpendicular to handle orientation.
        side_dirs = [handle_perp, -handle_perp]
        dists = [0.11, 0.10, 0.09]
        zoffs = [0.065, 0.055]
    else:
        # Open: keep deterministic front approach (negative X first), which
        # is the stable behavior for this scene.
        side_dirs = [np.array([-1.0, 0.0, 0.0], dtype=float), np.array([1.0, 0.0, 0.0], dtype=float)]
        dists = [0.09, 0.08, 0.10]
        zoffs = [0.07, 0.06]
    hover_offsets = []
    for sdir in side_dirs:
        for d in dists:
            for z in zoffs:
                hover_offsets.append(sdir * d + np.array([0.0, 0.0, z], dtype=float))
                hover_offsets.append(sdir * d + handle_dir * 0.01 + np.array([0.0, 0.0, z], dtype=float))
                hover_offsets.append(sdir * d - handle_dir * 0.01 + np.array([0.0, 0.0, z], dtype=float))

    # Gripper horizontal toward handle, fingers vertically aligned.
    orientations = [
        quaternion_from_euler(np.pi - 0.60, 0, np.pi / 2),
        quaternion_from_euler(np.pi - 0.55, 0, np.pi / 2),
        quaternion_from_euler(np.pi - 0.65, 0, np.pi / 2),
        quaternion_from_euler(np.pi - 0.60, 0, 0.0),
        quaternion_from_euler(np.pi - 0.60, 0, np.pi),
    ]
    original = env.get_robot_conf()
    q_curr = np.array(original, dtype=float)
    best = None
    for off in hover_offsets:
        hover_pos = handle_pos + off
        approach_dir = off.copy()
        approach_dir[2] = 0.0
        an = float(np.linalg.norm(approach_dir))
        if an > 1e-8:
            approach_dir = approach_dir / an
        else:
            approach_dir = np.array([1.0, 0.0, 0.0], dtype=float)
        for q in orientations:
            try:
                cfgs = env.robot.solve_ik_via_sampling(
                    hover_pos.tolist(),
                    quaternion=q,
                    max_configs=10,
                    max_time_ms=180,
                    ignore_collisions=True,
                )
                if cfgs is not None and len(cfgs) > 0:
                    c = min(cfgs, key=lambda cc: float(np.linalg.norm(np.array(cc, dtype=float) - q_curr)))
                    score = float(np.linalg.norm(np.array(c, dtype=float) - q_curr))
                    if (best is None) or (score < best[0]):
                        best = (score, list(c), q, hover_pos.tolist(), approach_dir.tolist())
            except Exception:
                pass
    env.set_robot_conf(original)
    if best is None:
        return None, None, None, None
    return best[1], best[2], best[3], best[4]


def _compute_handle_grasp_config(env, handle_obj, orient_q, approach_dir=None, direction="open"):
    handle_pos = np.array(handle_obj.get_position(), dtype=float)
    handle_dir, handle_perp = _handle_direction_and_perp(handle_obj)
    if approach_dir is None:
        approach_dir = handle_perp
    else:
        norm_dir = _unit_xy(approach_dir)
        approach_dir = norm_dir if norm_dir is not None else handle_perp
    # Keep approach along same perpendicular side used for hover.
    if direction == "close":
        ds = [0.018, 0.016, 0.014]
    else:
        ds = [0.016, 0.014, 0.012]
    grasp_offsets = []
    for d in ds:
        for tz in [0.012, 0.010, 0.008]:
            grasp_offsets.append(approach_dir * d + np.array([0.0, 0.0, tz], dtype=float))
            grasp_offsets.append(approach_dir * d + handle_dir * 0.004 + np.array([0.0, 0.0, tz], dtype=float))
            grasp_offsets.append(approach_dir * d - handle_dir * 0.004 + np.array([0.0, 0.0, tz], dtype=float))

    orientation_candidates = [
        orient_q,
        quaternion_from_euler(np.pi - 0.60, 0, np.pi / 2),
        quaternion_from_euler(np.pi - 0.55, 0, np.pi / 2),
        quaternion_from_euler(np.pi - 0.65, 0, np.pi / 2),
        quaternion_from_euler(np.pi - 0.60, 0, 0.0),
    ]
    original = env.get_robot_conf()
    for off in grasp_offsets:
        grasp_pos = handle_pos + off
        for q in orientation_candidates:
            if q is None:
                continue
            try:
                cfgs = env.robot.solve_ik_via_sampling(
                    grasp_pos.tolist(),
                    quaternion=q,
                    max_configs=20,
                    max_time_ms=250,
                    ignore_collisions=True,
                )
                if cfgs is not None and len(cfgs) > 0:
                    env.set_robot_conf(original)
                    return cfgs[0], grasp_pos.tolist()
            except Exception:
                pass
    env.set_robot_conf(original)
    return None, None


def _orientation_candidates_for_open_face(facing_dir):
    """
    Generate horizontal gripper orientations that face the handle while keeping
    fingers vertical as the first preference.
    """
    f = _unit_xy(facing_dir)
    if f is None:
        f = np.array([1.0, 0.0, 0.0], dtype=float)
    up = np.array([0.0, 0.0, 1.0], dtype=float)

    # Candidate A: local +X points toward handle, local +Y is vertical.
    xA = f / (np.linalg.norm(f) + 1e-12)
    yA = up
    zA = np.cross(xA, yA)
    if float(np.linalg.norm(zA)) < 1e-8:
        zA = np.array([0.0, 1.0, 0.0], dtype=float)
    zA = zA / (np.linalg.norm(zA) + 1e-12)
    yA = np.cross(zA, xA)
    yA = yA / (np.linalg.norm(yA) + 1e-12)
    RA = np.column_stack((xA, yA, zA))
    qA = _rot_to_quat(RA)

    # Candidate B: local +Z points toward handle, local +Y is vertical.
    zB = xA
    yB = up
    xB = np.cross(yB, zB)
    if float(np.linalg.norm(xB)) < 1e-8:
        xB = np.array([1.0, 0.0, 0.0], dtype=float)
    xB = xB / (np.linalg.norm(xB) + 1e-12)
    yB = np.cross(zB, xB)
    yB = yB / (np.linalg.norm(yB) + 1e-12)
    RB = np.column_stack((xB, yB, zB))
    qB = _rot_to_quat(RB)

    yaw = float(np.arctan2(f[1], f[0]))
    return [
        qA,
        qB,
        quaternion_from_euler(np.pi / 2.0, 0.0, yaw),
        quaternion_from_euler(np.pi / 2.0, 0.0, yaw + np.pi),
        quaternion_from_euler(np.pi / 2.0 + 0.10, 0.0, yaw),
        quaternion_from_euler(np.pi / 2.0 - 0.10, 0.0, yaw),
        # Legacy fallbacks.
        quaternion_from_euler(np.pi - 0.60, 0.0, np.pi / 2.0),
        quaternion_from_euler(np.pi - 0.55, 0.0, np.pi / 2.0),
        quaternion_from_euler(np.pi - 0.65, 0.0, np.pi / 2.0),
    ]


def _open_face_alignment_score(q, facing_dir):
    """
    Heuristic alignment score in [0,1]: 1 means end-effector forward is aligned
    with facing_dir in the XY plane. We check both local X and local Z axes to
    avoid assumptions about tool-frame forward convention.
    """
    fd = _unit_xy(facing_dir)
    if fd is None:
        return 0.0
    try:
        R = _quat_to_rot(q)
    except Exception:
        return 0.0
    ax = _unit_xy(R[:, 0])
    az = _unit_xy(R[:, 2])
    scores = []
    if ax is not None:
        scores.append(float(np.clip(np.dot(ax, fd), -1.0, 1.0)))
    if az is not None:
        scores.append(float(np.clip(np.dot(az, fd), -1.0, 1.0)))
    if not scores:
        return 0.0
    # map [-1,1] -> [0,1]
    return 0.5 * (max(scores) + 1.0)


def _compute_open_hover_config_strict(env, handle_obj):
    """
    Step-1 open-lid hover:
    - horizontal approach, facing the handle
    - fingers vertical (preferred)
    Returns: (q_hover, hover_quat, hover_pos, approach_from_handle, hover_dist_xy, hover_z)
    """
    try:
        handle_pos = np.array(handle_obj.get_position(), dtype=float)
    except Exception:
        return None, None, None, None, None, None
    try:
        tip_pos = np.array(env.robot.get_tip().get_position(), dtype=float)
    except Exception:
        tip_pos = handle_pos - np.array([0.18, 0.0, -0.04], dtype=float)

    # Approach direction from current robot side (dynamic, no fixed side).
    approach_from_handle = _unit_xy(tip_pos - handle_pos)
    if approach_from_handle is None:
        _hdir, hperp = _handle_direction_and_perp(handle_obj)
        approach_from_handle = _unit_xy(hperp)
    if approach_from_handle is None:
        approach_from_handle = np.array([1.0, 0.0, 0.0], dtype=float)

    # Dynamic hover distance/height based on current robot-handle geometry.
    d_now = float(np.linalg.norm((tip_pos - handle_pos)[:2]))
    # Keep a larger standoff before grasp to avoid blocking near the lid.
    hover_dist = float(np.clip(0.85 * d_now, 0.14, 0.20))
    hover_z = float(np.clip(0.04 + 0.35 * abs(float(tip_pos[2] - handle_pos[2])), 0.055, 0.09))
    hover_pos = handle_pos + approach_from_handle * hover_dist + np.array([0.0, 0.0, hover_z], dtype=float)
    facing_dir = -approach_from_handle

    original = env.get_robot_conf()
    q_curr = np.array(original, dtype=float)
    best = None
    t0 = time.time()
    max_search_s = float(os.environ.get("GRILL_OPEN_HOVER_SEARCH_SEC", "2.8"))
    for q in _orientation_candidates_for_open_face(facing_dir):
        if (time.time() - t0) > max_search_s:
            break
        try:
            cfgs = env.robot.solve_ik_via_sampling(
                hover_pos.tolist(),
                quaternion=q,
                max_configs=6,
                max_time_ms=80,
                ignore_collisions=True,
            )
        except Exception:
            cfgs = None
        if cfgs is None or len(cfgs) == 0:
            continue
        c = min(cfgs, key=lambda cc: float(np.linalg.norm(np.array(cc, dtype=float) - q_curr)))
        score = float(np.linalg.norm(np.array(c, dtype=float) - q_curr))
        # Prefer IK targets with a valid motion plan from current state.
        try:
            motion = env.compute_motion_plan(original, list(c))
        except Exception:
            motion = None
        if motion is None:
            continue
        motion_cost = float(len(motion))
        total = score + 0.03 * motion_cost
        if (best is None) or (total < best[0]):
            best = (total, list(c), q, hover_pos.tolist(), approach_from_handle.tolist(), hover_dist, hover_z)

    env.set_robot_conf(original)
    if best is None:
        return None, None, None, None, None, None
    return best[1], best[2], best[3], best[4], float(best[5]), float(best[6])


def _compute_open_grasp_config_strict(env, handle_obj, hover_quat, approach_from_handle):
    """
    Step-2 open-lid grasp:
    move straight toward handle from hover side, keeping hover orientation.
    """
    try:
        handle_pos = np.array(handle_obj.get_position(), dtype=float)
    except Exception:
        return None, None
    approach = _unit_xy(approach_from_handle)
    if approach is None:
        approach = np.array([1.0, 0.0, 0.0], dtype=float)
    facing_dir = -approach

    # Dynamic shallow insertion from hover side.
    try:
        tip_pos = np.array(env.robot.get_tip().get_position(), dtype=float)
    except Exception:
        tip_pos = handle_pos + approach * 0.08
    d_now = float(np.linalg.norm((tip_pos - handle_pos)[:2]))
    grasp_dist = float(np.clip(0.20 * d_now, 0.010, 0.022))
    grasp_z = float(np.clip(0.15 * abs(float(tip_pos[2] - handle_pos[2])), 0.006, 0.014))
    grasp_pos = handle_pos + approach * grasp_dist + np.array([0.0, 0.0, grasp_z], dtype=float)

    orientation_candidates = [hover_quat] + _orientation_candidates_for_open_face(facing_dir)
    original = env.get_robot_conf()
    q_curr = np.array(original, dtype=float)
    best = None
    t0 = time.time()
    max_search_s = float(os.environ.get("GRILL_OPEN_GRASP_SEARCH_SEC", "2.0"))
    for q in orientation_candidates:
        if q is None:
            continue
        if (time.time() - t0) > max_search_s:
            break
        try:
            cfgs = env.robot.solve_ik_via_sampling(
                grasp_pos.tolist(),
                quaternion=q,
                max_configs=8,
                max_time_ms=80,
                ignore_collisions=True,
            )
        except Exception:
            cfgs = None
        if cfgs is None or len(cfgs) == 0:
            continue
        c = min(cfgs, key=lambda cc: float(np.linalg.norm(np.array(cc, dtype=float) - q_curr)))
        score = float(np.linalg.norm(np.array(c, dtype=float) - q_curr))
        # Require a clean straight forward motion from hover to grasp.
        lin_path = None
        try:
            lin_path = env._get_linear_path(original, grasp_pos.tolist(), q, ignore_collisions=True, steps=55)
        except Exception:
            lin_path = None
        if lin_path is None:
            continue
        try:
            motion = env.compute_motion_plan(original, list(c))
        except Exception:
            motion = None
        if motion is None:
            continue
        motion_cost = float(len(motion))
        lin_cost = float(len(lin_path._path_points.reshape(-1, 7).tolist()))
        total = score + 0.03 * motion_cost + 0.01 * lin_cost
        if (best is None) or (total < best[0]):
            q_from_lin = lin_path._path_points[-7:].tolist()
            best = (total, list(q_from_lin), grasp_pos.tolist())
    env.set_robot_conf(original)
    if best is None:
        return None, None
    return best[1], best[2]


def _hinge_pose_and_axis(env):
    """
    Return hinge position and joint axis in world frame.
    """
    hinge_pos = None
    hinge_axis = None
    h_joint = _resolve_lid_joint_handle(env)
    if h_joint is not None:
        try:
            hinge_pos = np.array(sim.simGetObjectPosition(int(h_joint), sim.sim_handle_world), dtype=float)
        except Exception:
            hinge_pos = None
        try:
            jm = _sim_matrix_to_4x4(sim.simGetObjectMatrix(int(h_joint), sim.sim_handle_world))
            hinge_axis = np.array([jm[0, 2], jm[1, 2], jm[2, 2]], dtype=float)
        except Exception:
            hinge_axis = None
    if hinge_pos is None:
        hp = _hinge_position()
        if hp is not None:
            hinge_pos = np.array(hp, dtype=float)
    if hinge_axis is None:
        hinge_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    n = float(np.linalg.norm(hinge_axis))
    if n < 1e-8:
        hinge_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        hinge_axis = hinge_axis / n
    if hinge_pos is None:
        return None, None
    return hinge_pos.tolist(), hinge_axis.tolist()


def _arc_waypoints_about_axis(handle_pos, hinge_pos, hinge_axis, rotation_amount, n=30):
    """
    Circular arc around explicit hinge axis (continuous and frame-invariant).
    """
    h0 = np.array(handle_pos, dtype=float)
    p = np.array(hinge_pos, dtype=float)
    axis = np.array(hinge_axis, dtype=float)
    an = float(np.linalg.norm(axis))
    if an < 1e-8:
        return _arc_waypoints(handle_pos, hinge_pos, rotation_amount=rotation_amount, n=n)
    axis = axis / an

    r0 = h0 - p
    waypoints = []
    for off in np.linspace(0.0, float(rotation_amount), max(6, int(n))):
        R = _axis_angle_matrix(axis, float(off))
        pos = p + (R @ r0)
        waypoints.append((pos.astype(float), float(off)))
    return waypoints


def _sample_handle_position(handle_obj, pr=None, samples=3):
    pts = []
    for i in range(max(1, int(samples))):
        try:
            pts.append(np.array(handle_obj.get_position(), dtype=float))
        except Exception:
            pass
        if (pr is not None) and (i < int(samples) - 1):
            step(pr, 1)
    if not pts:
        return None
    return np.mean(np.array(pts, dtype=float), axis=0)


def _open_gripper_fully(env, pr, velocity=0.35, max_steps=140):
    for _ in range(max(1, int(max_steps))):
        done = False
        try:
            done = bool(env.gripper.actuate(1.0, velocity=float(velocity)))
        except Exception:
            try:
                env.gripper.actuate(1.0, velocity=float(velocity))
            except Exception:
                pass
        step(pr, 1)
        if done:
            break
    step(pr, 8)


def _close_gripper_fully(env, pr, velocity=0.18, max_steps=120):
    for _ in range(max(1, int(max_steps))):
        done = False
        try:
            done = bool(env.gripper.actuate(0.0, velocity=float(velocity)))
        except Exception:
            try:
                env.gripper.actuate(0.0, velocity=float(velocity))
            except Exception:
                pass
        step(pr, 1)
        if done:
            break
    step(pr, 6)


def _gripper_detects(env, obj):
    try:
        return bool(env.gripper._proximity_sensor.is_detected(obj))
    except Exception:
        return False


def _obj_from_handle(h):
    try:
        t = sim.simGetObjectType(int(h))
    except Exception:
        return None
    try:
        if t == sim.sim_object_shape_type:
            return Shape(int(h))
        if t == sim.sim_object_joint_type:
            return Joint(int(h))
        if t == sim.sim_object_dummy_type:
            return Dummy(int(h))
    except Exception:
        return None
    return None


def _capture_handle_anchor(env):
    global HANDLE_ANCHOR_PARENT_HANDLE, HANDLE_ANCHOR_REL_MATRIX
    handle = _get_active_handle(env)
    if handle is None:
        return False
    try:
        parent = handle.get_parent()
        if parent is None:
            return False
        HANDLE_ANCHOR_PARENT_HANDLE = int(parent.get_handle())
        HANDLE_ANCHOR_REL_MATRIX = np.array(handle.get_matrix(relative_to=parent), dtype=float)
        return True
    except Exception:
        return False


def _restore_handle_anchor(env, pr=None):
    if not HANDLE_REANCHOR_ALWAYS:
        return False
    handle = _get_active_handle(env)
    if handle is None:
        return False
    if (HANDLE_ANCHOR_PARENT_HANDLE is None) or (HANDLE_ANCHOR_REL_MATRIX is None):
        return False
    parent = _obj_from_handle(HANDLE_ANCHOR_PARENT_HANDLE)
    if parent is None:
        return False
    try:
        cur_parent = handle.get_parent()
        cur_h = int(cur_parent.get_handle()) if cur_parent is not None else -1
    except Exception:
        cur_h = -1
    try:
        if cur_h != int(HANDLE_ANCHOR_PARENT_HANDLE):
            handle.set_parent(parent, keep_in_place=True)
        handle.set_matrix(np.array(HANDLE_ANCHOR_REL_MATRIX, dtype=float), relative_to=parent)
        try:
            handle.set_dynamic(False)
        except Exception:
            pass
        try:
            handle.set_collidable(False)
            handle.set_respondable(False)
        except Exception:
            pass
        if pr is not None:
            step(pr, 2)
        return True
    except Exception:
        return False


def _attempt_handle_attach(env, pr, handle_obj, repeats=8):
    """
    Try explicit attach without open/close oscillations.
    """
    if handle_obj is None:
        return False
    for _ in range(max(1, int(repeats))):
        try:
            env.gripper.grasp(handle_obj)
        except Exception:
            pass
        step(pr, 6)
        try:
            grabbed = env.gripper.get_grasped_objects()
        except Exception:
            grabbed = []
        if any(_obj_handle(o) == _obj_handle(handle_obj) for o in grabbed):
            return True
    return False


def _hinge_position():
    try:
        return list(Joint("lid_joint").get_position())
    except Exception:
        return None


def _get_joint_alias(handle):
    try:
        return str(sim.simGetObjectAlias(int(handle), 5))
    except Exception:
        try:
            return str(sim.simGetObjectName(int(handle)))
        except Exception:
            return f"joint@{int(handle)}"


def _get_shape_alias(handle):
    try:
        return str(sim.simGetObjectAlias(int(handle), 5))
    except Exception:
        try:
            return str(sim.simGetObjectName(int(handle)))
        except Exception:
            return f"shape@{int(handle)}"


def _axis_angle_matrix(axis, angle):
    axis = np.array(axis, dtype=float)
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        return np.eye(3, dtype=float)
    x, y, z = (axis / n).tolist()
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


def _sim_matrix_to_4x4(m12):
    m = np.eye(4, dtype=float)
    m[:3, :4] = np.array(list(m12), dtype=float).reshape(3, 4)
    return m


def _mat4_to_sim_matrix(m4):
    return np.array(m4, dtype=float)[:3, :4].reshape(-1).tolist()


def _snap_lid_to_waypoint(env, pr, waypoint_name, iters=2):
    """
    Rotate lid parent object around hinge axis so handle reaches waypoint.
    Works even if lid joint value is decoupled from mesh pose.
    """
    handle = _get_active_handle(env)
    target_pos, _ = _get_dummy_pose(waypoint_name)
    if handle is None or target_pos is None:
        return False
    h_handle = int(handle.get_handle())
    # Walk ancestors and choose the lid-like node if available.
    chain = []
    cur = h_handle
    while cur >= 0:
        chain.append(cur)
        try:
            cur = int(sim.simGetObjectParent(cur))
        except Exception:
            cur = -1
    h_lid_parent = -1
    scored = []
    for h in chain[1:]:
        moved = 0.0
        try:
            m = _sim_matrix_to_4x4(sim.simGetObjectMatrix(h, sim.sim_handle_world))
            m_test = np.array(m, copy=True)
            m_test[0, 3] += 0.005
            sim.simSetObjectMatrix(h, sim.sim_handle_world, _mat4_to_sim_matrix(m_test))
            step(pr, 1)
            p_test = np.array(handle.get_position(), dtype=float)
            sim.simSetObjectMatrix(h, sim.sim_handle_world, _mat4_to_sim_matrix(m))
            step(pr, 1)
            p_base = np.array(handle.get_position(), dtype=float)
            moved = float(np.linalg.norm(p_test - p_base))
        except Exception:
            moved = 0.0
        if moved > 1e-4:
            alias = _get_joint_alias(h).lower()
            pref = 2 if ("lid" in alias) else (1 if ("joint" in alias or "grill" in alias) else 0)
            scored.append((pref, moved, h))
    if scored:
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        h_lid_parent = int(scored[0][2])
    elif len(chain) > 1:
        h_lid_parent = chain[1]
    if h_lid_parent < 0:
        return False

    h_joint = _resolve_lid_joint_handle(env)
    if h_joint is None:
        return False

    try:
        hinge = np.array(sim.simGetObjectPosition(int(h_joint), sim.sim_handle_world), dtype=float)
    except Exception:
        try:
            hinge = np.array(_hinge_position(), dtype=float)
        except Exception:
            return False
    try:
        jm = _sim_matrix_to_4x4(sim.simGetObjectMatrix(int(h_joint), sim.sim_handle_world))
        axes = [
            np.array([jm[0, 0], jm[1, 0], jm[2, 0]], dtype=float),  # local X
            np.array([jm[0, 1], jm[1, 1], jm[2, 1]], dtype=float),  # local Y
            np.array([jm[0, 2], jm[1, 2], jm[2, 2]], dtype=float),  # local Z (joint axis in Coppelia)
        ]
    except Exception:
        axes = [
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 1.0], dtype=float),
        ]

    target = np.array(target_pos, dtype=float)
    success = False
    for _ in range(max(1, int(iters))):
        cur_h = np.array(handle.get_position(), dtype=float)
        d_before = float(np.linalg.norm(cur_h - target))
        best = None

        for axis in axes:
            axis_n = axis / (np.linalg.norm(axis) + 1e-12)
            v = cur_h - hinge
            u = target - hinge
            v = v - axis_n * float(np.dot(v, axis_n))
            u = u - axis_n * float(np.dot(u, axis_n))
            nv = float(np.linalg.norm(v))
            nu = float(np.linalg.norm(u))
            if nv < 1e-9 or nu < 1e-9:
                continue
            v /= nv
            u /= nu
            cross = np.cross(v, u)
            sin_t = float(np.dot(axis_n, cross))
            cos_t = float(np.clip(np.dot(v, u), -1.0, 1.0))
            base_angle = float(np.arctan2(sin_t, cos_t))

            for angle in [base_angle, -base_angle, 0.5 * base_angle, -0.5 * base_angle]:
                try:
                    lid_m = _sim_matrix_to_4x4(sim.simGetObjectMatrix(h_lid_parent, sim.sim_handle_world))
                except Exception:
                    continue
                R = _axis_angle_matrix(axis_n, angle)
                R_old = lid_m[:3, :3]
                t_old = lid_m[:3, 3]
                lid_m_new = np.eye(4, dtype=float)
                lid_m_new[:3, :3] = R @ R_old
                lid_m_new[:3, 3] = R @ (t_old - hinge) + hinge
                try:
                    try:
                        sim.simResetDynamicObject(int(h_lid_parent))
                    except Exception:
                        pass
                    sim.simSetObjectMatrix(h_lid_parent, sim.sim_handle_world, _mat4_to_sim_matrix(lid_m_new))
                    step(pr, 1)
                    h_new = np.array(handle.get_position(), dtype=float)
                    d_new = float(np.linalg.norm(h_new - target))
                except Exception:
                    continue
                if (best is None) or (d_new < best[0]):
                    best = (d_new, h_lid_parent, lid_m_new)
                # restore before next try
                try:
                    sim.simSetObjectMatrix(h_lid_parent, sim.sim_handle_world, _mat4_to_sim_matrix(lid_m))
                    step(pr, 1)
                except Exception:
                    pass

        if best is None:
            break

        d_best, best_parent, best_m = best
        if d_best >= d_before - 1e-5:
            break
        try:
            sim.simSetObjectMatrix(best_parent, sim.sim_handle_world, _mat4_to_sim_matrix(best_m))
            step(pr, 2)
        except Exception:
            break
        if d_best < 0.02:
            success = True
            break
        success = True
    if success:
        return True

    # Translation fallback: move lid parent so handle directly reaches target.
    try:
        cur_h = np.array(handle.get_position(), dtype=float)
        delta = target - cur_h
        if float(np.linalg.norm(delta)) > 1e-6:
            lid_m = _sim_matrix_to_4x4(sim.simGetObjectMatrix(h_lid_parent, sim.sim_handle_world))
            lid_m[:3, 3] = lid_m[:3, 3] + delta
            try:
                sim.simResetDynamicObject(int(h_lid_parent))
            except Exception:
                pass
            sim.simSetObjectMatrix(h_lid_parent, sim.sim_handle_world, _mat4_to_sim_matrix(lid_m))
            step(pr, 2)
            new_h = np.array(handle.get_position(), dtype=float)
            d_new = float(np.linalg.norm(new_h - target))
            if d_new < 0.03:
                return True
            # Last resort: nudge by object position API if matrix writes are ignored.
            try:
                p = np.array(sim.simGetObjectPosition(h_lid_parent, sim.sim_handle_world), dtype=float)
                sim.simSetObjectPosition(h_lid_parent, sim.sim_handle_world, (p + delta).tolist())
                step(pr, 2)
                new_h = np.array(handle.get_position(), dtype=float)
                d_new = float(np.linalg.norm(new_h - target))
                return d_new < 0.03
            except Exception:
                pass
    except Exception:
        pass
    return False


def _handle_waypoint_distance(env, waypoint_name):
    handle = _get_active_handle(env)
    target_pos, _ = _get_dummy_pose(waypoint_name)
    if handle is None or target_pos is None:
        return None
    return float(np.linalg.norm(np.array(handle.get_position(), dtype=float) - np.array(target_pos, dtype=float)))


def _resolve_lid_joint_handle(env):
    global ACTIVE_LID_JOINT_HANDLE
    if ACTIVE_LID_JOINT_HANDLE is not None:
        return int(ACTIVE_LID_JOINT_HANDLE)
    try:
        if getattr(env, "lid_joint", None) is not None:
            return int(env.lid_joint.get_handle())
    except Exception:
        pass
    return None


def _handle_from_shape_handle(shape_handle):
    try:
        return Shape(_get_shape_alias(int(shape_handle)))
    except Exception:
        try:
            return Shape(int(shape_handle))
        except Exception:
            return None


def _get_active_handle(env):
    global ACTIVE_HANDLE_SHAPE_HANDLE
    if ACTIVE_HANDLE_SHAPE_HANDLE is not None:
        obj = _handle_from_shape_handle(ACTIVE_HANDLE_SHAPE_HANDLE)
        if obj is not None:
            return obj
    # Fallback default
    return env.get_object("handle_visual") or env.get_object("handle")


def _discover_active_handle(env, pr):
    """
    Pick the real grill-handle shape by checking which handle candidate responds
    to lid-joint motion. This avoids grabbing a static/non-lid handle visual.
    """
    global ACTIVE_HANDLE_SHAPE_HANDLE
    if not HANDLE_PROBE_AT_STARTUP:
        # Stable default: avoid probing by moving the lid at startup.
        for nm in ("handle_visual", "handle"):
            try:
                o = env.get_object(nm)
                if o is not None:
                    h = int(o.get_handle())
                    ACTIVE_HANDLE_SHAPE_HANDLE = h
                    print(f"[startup] Selected handle (direct): {_get_shape_alias(h)} (h={h})")
                    return h
            except Exception:
                pass

    # Candidate shape handles containing 'handle'
    candidates = []
    try:
        objs = sim.simGetObjectsInTree(sim.sim_handle_scene, sim.sim_handle_all, 0)
        if isinstance(objs, int):
            objs = [objs]
        for h in objs:
            try:
                if sim.simGetObjectType(int(h)) != sim.sim_object_shape_type:
                    continue
                alias = _get_shape_alias(int(h)).lower()
                if "handle" not in alias:
                    continue
                if "panda" in alias:
                    continue
                candidates.append(int(h))
            except Exception:
                pass
    except Exception:
        pass

    # include explicit defaults
    for nm in ("handle_visual", "handle"):
        try:
            o = env.get_object(nm)
            if o is not None:
                candidates.append(int(o.get_handle()))
        except Exception:
            pass
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return None

    h_joint = _resolve_lid_joint_handle(env)
    wp23, _ = _get_dummy_pose("waypoint23")
    wp29, _ = _get_dummy_pose("waypoint29")
    p23 = np.array(wp23, dtype=float) if wp23 is not None else None
    p29 = np.array(wp29, dtype=float) if wp29 is not None else None

    # probe joint motion
    cur = None
    lo = hi = None
    if h_joint is not None:
        try:
            cur = float(sim.simGetJointPosition(int(h_joint)))
            lo, hi = cur - 0.25, cur + 0.25
            cyc, interval = sim.simGetJointInterval(int(h_joint))
            if (not cyc) and (interval is not None) and (len(interval) >= 2):
                l = float(interval[0])
                u = float(interval[0] + interval[1])
                lo, hi = max(lo, l), min(hi, u)
        except Exception:
            cur = None

    def _pos_of(sh):
        try:
            return np.array(sim.simGetObjectPosition(int(sh), sim.sim_handle_world), dtype=float)
        except Exception:
            o = _handle_from_shape_handle(sh)
            if o is None:
                return None
            return np.array(o.get_position(), dtype=float)

    best = None
    for sh in candidates:
        p0 = _pos_of(sh)
        if p0 is None:
            continue
        moved = 0.0
        if (h_joint is not None) and (cur is not None):
            for a in (lo, hi):
                try:
                    sim.simSetJointPosition(int(h_joint), float(a))
                    try:
                        sim.simSetJointTargetPosition(int(h_joint), float(a))
                    except Exception:
                        pass
                    step(pr, 1)
                    pa = _pos_of(sh)
                    if pa is not None:
                        moved = max(moved, float(np.linalg.norm(pa - p0)))
                except Exception:
                    pass
            try:
                sim.simSetJointPosition(int(h_joint), float(cur))
                try:
                    sim.simSetJointTargetPosition(int(h_joint), float(cur))
                except Exception:
                    pass
                step(pr, 1)
            except Exception:
                pass

        d_ref = float("inf")
        if p23 is not None:
            d_ref = min(d_ref, float(np.linalg.norm(p0 - p23)))
        if p29 is not None:
            d_ref = min(d_ref, float(np.linalg.norm(p0 - p29)))
        score = (10.0 * moved) - d_ref
        if (best is None) or (score > best[0]):
            best = (score, sh, moved, d_ref)

    if best is None:
        return None
    _, sh, moved, d_ref = best
    ACTIVE_HANDLE_SHAPE_HANDLE = int(sh)
    print(
        f"[startup] Selected handle: {_get_shape_alias(sh)} (h={sh}, moved={moved:.4f}, d_ref={d_ref:.4f})"
    )
    return int(sh)


def _discover_lid_joint_handle(env, pr):
    """
    Detect the joint that actually moves grill handle_visual by probing joints.
    """
    global ACTIVE_LID_JOINT_HANDLE
    # 1) Strong preference: explicit scene joint name.
    try:
        explicit = Joint("lid_joint")
        h = int(explicit.get_handle())
        ACTIVE_LID_JOINT_HANDLE = h
        print(f"[startup] Selected lid joint (explicit): {_get_joint_alias(h)} (h={h})")
        return h
    except Exception:
        pass

    # 2) Next preference: environment-provided lid joint object.
    try:
        if getattr(env, "lid_joint", None) is not None:
            h = int(env.lid_joint.get_handle())
            ACTIVE_LID_JOINT_HANDLE = h
            print(f"[startup] Selected lid joint (env): {_get_joint_alias(h)} (h={h})")
            return h
    except Exception:
        pass

    # 3) Fallback: choose by joint alias tokens only, never Panda joints.
    candidates = []
    try:
        objs = sim.simGetObjectsInTree(sim.sim_handle_scene, sim.sim_handle_all, 0)
        if isinstance(objs, int):
            objs = [objs]
        for h in objs:
            try:
                if sim.simGetObjectType(int(h)) != sim.sim_object_joint_type:
                    continue
                alias = _get_joint_alias(int(h)).lower()
                if "panda" in alias:
                    continue
                if any(t in alias for t in ("lid", "hinge", "cover", "grill")):
                    candidates.append(int(h))
            except Exception:
                pass
    except Exception:
        pass

    if candidates:
        h = int(candidates[0])
        ACTIVE_LID_JOINT_HANDLE = h
        print(f"[startup] Selected lid joint (name-fallback): {_get_joint_alias(h)} (h={h})")
        return h

    print("[startup] WARNING: could not resolve lid joint handle.")
    return _resolve_lid_joint_handle(env)


def _relax_joint_interval(h):
    prev = None
    try:
        cyc, interval = sim.simGetJointInterval(int(h))
        prev = (bool(cyc), list(interval) if interval is not None else None)
    except Exception:
        prev = None
    # Broaden limits to allow full scan / locking to true closed pose.
    try:
        sim.simSetJointInterval(int(h), False, [-3.2, 6.4])  # [min, range]
    except Exception:
        pass
    return prev


def _restore_joint_interval(h, prev):
    if prev is None:
        return
    cyc, interval = prev
    try:
        if interval is None:
            interval = [-3.2, 6.4]
        sim.simSetJointInterval(int(h), bool(cyc), interval)
    except Exception:
        pass


def _set_lid_joint_angle(env, pr, target_angle, steps=70):
    h = _resolve_lid_joint_handle(env)
    if h is None:
        return False
    _set_lid_servo_lock(env, True)
    prev_interval = _relax_joint_interval(h)
    try:
        cur = float(sim.simGetJointPosition(int(h)))
    except Exception:
        try:
            cur = _get_lid_joint_angle(env)
        except Exception:
            _restore_joint_interval(h, prev_interval)
            return False
    for a in np.linspace(cur, float(target_angle), max(2, int(steps))):
        try:
            sim.simSetJointPosition(int(h), float(a))
        except Exception:
            pass
        try:
            sim.simSetJointTargetPosition(int(h), float(a))
        except Exception:
            pass
        # Also nudge mapped joint object if this is it.
        try:
            if getattr(env, "lid_joint", None) is not None and int(env.lid_joint.get_handle()) == int(h):
                env.lid_joint.set_joint_position(float(a), disable_dynamics=True)
                env.lid_joint.set_joint_target_position(float(a))
                env.lid_joint.set_joint_target_velocity(0.0)
        except Exception:
            pass
        step(pr, 1)
    _restore_joint_interval(h, prev_interval)
    return True


def _get_lid_joint_angle(env):
    h = _resolve_lid_joint_handle(env)
    if h is None:
        return None
    try:
        return float(sim.simGetJointPosition(int(h)))
    except Exception:
        try:
            if getattr(env, "lid_joint", None) is not None:
                return float(env.lid_joint.get_joint_position())
        except Exception:
            pass
        return None


def _force_lid_closed(env, pr, steps=80):
    """Hard-hold lid at closed angle for a short period."""
    h = _resolve_lid_joint_handle(env)
    if h is None:
        return False
    _set_lid_servo_lock(env, True)
    prev_interval = _relax_joint_interval(h)
    target = float(LID_CLOSED_ANGLE)
    for _ in range(max(1, int(steps))):
        try:
            sim.simSetJointPosition(int(h), target)
        except Exception:
            pass
        try:
            sim.simSetJointTargetPosition(int(h), target)
        except Exception:
            pass
        try:
            if getattr(env, "lid_joint", None) is not None and int(env.lid_joint.get_handle()) == int(h):
                env.lid_joint.set_joint_position(target, disable_dynamics=True)
                env.lid_joint.set_joint_target_position(target)
                env.lid_joint.set_joint_target_velocity(0.0)
        except Exception:
            pass
        step(pr, 1)
    _restore_joint_interval(h, prev_interval)
    return True


def _calibrate_lid_from_waypoints(env, pr):
    """
    Calibrate closed/open lid angles by scanning only the resolved lid joint and
    minimizing handle distance to the scene waypoints.
    """
    global LID_CLOSED_ANGLE, LID_OPEN_ANGLE, CLOSE_WP_NAME, OPEN_WP_NAME

    h = _resolve_lid_joint_handle(env)
    if h is None:
        return False
    prev_interval = _relax_joint_interval(h)
    handle = _get_active_handle(env)
    if handle is None:
        _restore_joint_interval(h, prev_interval)
        return False

    wp23, _ = _get_dummy_pose("waypoint23")
    wp29, _ = _get_dummy_pose("waypoint29")
    if wp23 is None or wp29 is None:
        _restore_joint_interval(h, prev_interval)
        return False

    # Infer closed/open waypoint by height: closed handle is usually lower.
    if float(wp23[2]) <= float(wp29[2]):
        CLOSE_WP_NAME, OPEN_WP_NAME = "waypoint23", "waypoint29"
        close_target = np.array(wp23, dtype=float)
        open_target = np.array(wp29, dtype=float)
    else:
        CLOSE_WP_NAME, OPEN_WP_NAME = "waypoint29", "waypoint23"
        close_target = np.array(wp29, dtype=float)
        open_target = np.array(wp23, dtype=float)

    try:
        cur = float(sim.simGetJointPosition(int(h)))
    except Exception:
        _restore_joint_interval(h, prev_interval)
        return False

    # Scan wide regardless of original limits.
    lo, hi = -3.0, 3.0

    best_close = (cur, float("inf"))
    best_open = (cur, float("inf"))

    for a in np.linspace(lo, hi, 121):
        try:
            sim.simSetJointPosition(int(h), float(a))
        except Exception:
            pass
        try:
            sim.simSetJointTargetPosition(int(h), float(a))
        except Exception:
            pass
        step(pr, 1)
        hp = np.array(handle.get_position(), dtype=float)
        dc = float(np.linalg.norm(hp - close_target))
        do = float(np.linalg.norm(hp - open_target))
        if dc < best_close[1]:
            best_close = (float(a), dc)
        if do < best_open[1]:
            best_open = (float(a), do)

    LID_CLOSED_ANGLE = float(best_close[0])
    LID_OPEN_ANGLE = float(best_open[0])
    if abs(LID_OPEN_ANGLE - LID_CLOSED_ANGLE) < 1e-4:
        # Joint angle might be decoupled from mesh; keep a nonzero travel target.
        LID_OPEN_ANGLE = float(LID_CLOSED_ANGLE + LID_TRAVEL_ANGLE)
    _enforce_min_open_travel()
    print(
        f"[startup] Lid auto-calibration: close_wp={CLOSE_WP_NAME}, open_wp={OPEN_WP_NAME}, "
        f"closed={LID_CLOSED_ANGLE:.3f} (d={best_close[1]:.3f}), "
        f"open={LID_OPEN_ANGLE:.3f} (d={best_open[1]:.3f})"
    )
    _restore_joint_interval(h, prev_interval)
    return True


def _set_lid_servo_lock(env, lock=True):
    # Prefer environment-level API when available.
    try:
        if hasattr(env, "set_lid_servo_lock"):
            env.set_lid_servo_lock(bool(lock))
            return True
    except Exception:
        pass

    h = _resolve_lid_joint_handle(env)
    if h is None:
        return False
    try:
        j = env.lid_joint if (getattr(env, "lid_joint", None) is not None and int(env.lid_joint.get_handle()) == int(h)) else Joint(int(h))
    except Exception:
        return False
    try:
        j.set_joint_mode(JointMode.FORCE)
    except Exception:
        pass
    try:
        j.set_motor_enabled(True)
    except Exception:
        pass
    try:
        j.set_control_loop_enabled(bool(lock))
    except Exception:
        pass
    try:
        j.set_motor_locked_at_zero_velocity(bool(lock))
    except Exception:
        pass
    try:
        j.set_joint_force(float(os.environ.get("GRILL_LID_MAX_FORCE", "400.0")))
    except Exception:
        pass
    if lock:
        try:
            cur = float(j.get_joint_position())
        except Exception:
            cur = float(LID_CLOSED_ANGLE)
        try:
            j.set_joint_target_position(cur)
        except Exception:
            pass
        try:
            j.set_joint_target_velocity(0.0)
        except Exception:
            pass
    return True


def _estimate_closed_angle_from_current(env, pr, current_angle, d_close, d_open):
    """
    Estimate a closed angle using a tiny local probe around the current angle.
    Assumes current pose is open-like when d_open << d_close.
    """
    # If current already looks closed, keep it.
    if (d_close is not None) and (d_open is not None) and (d_close <= d_open):
        return float(current_angle), False

    eps = 0.20
    a0 = float(current_angle)
    _set_lid_joint_angle(env, pr, a0 + eps, steps=3)
    d_plus = _handle_waypoint_distance(env, CLOSE_WP_NAME)
    _set_lid_joint_angle(env, pr, a0 - eps, steps=6)
    d_minus = _handle_waypoint_distance(env, CLOSE_WP_NAME)
    _set_lid_joint_angle(env, pr, a0, steps=3)

    # Choose direction that gets closer to close waypoint.
    sign = 1.0
    if (d_plus is not None) and (d_minus is not None):
        sign = 1.0 if d_plus < d_minus else -1.0
    elif d_minus is not None:
        sign = -1.0

    closed_est = float(a0 + sign * abs(float(LID_TRAVEL_ANGLE)))
    return closed_est, True


def _set_lid_target_position(env, angle):
    h = _resolve_lid_joint_handle(env)
    if h is None:
        return False
    ok = False
    try:
        sim.simSetJointTargetPosition(int(h), float(angle))
        ok = True
    except Exception:
        pass
    try:
        if getattr(env, "lid_joint", None) is not None and int(env.lid_joint.get_handle()) == int(h):
            env.lid_joint.set_joint_target_position(float(angle))
            ok = True
    except Exception:
        pass
    return ok


def _iter_food_objects(env):
    names = [
        "steak", "steak1", "steak2", "steak3",
        "chicken", "chicken1", "chicken2", "chicken3",
    ]
    seen = set()
    objs = []
    for n in names:
        try:
            o = env.get_object(n)
        except Exception:
            o = None
        if o is None:
            continue
        h = _obj_handle(o)
        if h is None or h in seen:
            continue
        seen.add(h)
        objs.append(o)
    return objs


def _lid_hits_food(env):
    lid = getattr(env, "grill_lid", None)
    if lid is None:
        return False
    for o in _iter_food_objects(env):
        try:
            contacts = lid.get_contact(o, get_contact_normal=False)
            if contacts:
                return True
        except Exception:
            continue
    return False


def _close_lid_until_contact(env, pr, target_angle, steps=120, backoff=0.05):
    """
    Close lid toward target angle but stop slightly early if it contacts food.
    Returns (reached_target_without_contact, final_angle).
    """
    cur = _get_lid_joint_angle(env)
    if cur is None:
        return False, None
    target = float(target_angle)
    if abs(target - cur) < 1e-4:
        return True, float(cur)

    _set_lid_servo_lock(env, True)
    sign = 1.0 if (target - cur) >= 0.0 else -1.0
    final = float(cur)
    touched = False
    for a in np.linspace(cur, target, max(2, int(steps))):
        _set_lid_target_position(env, float(a))
        step(pr, 2)
        ja = _get_lid_joint_angle(env)
        if ja is not None:
            final = float(ja)
        else:
            final = float(a)
        if _lid_hits_food(env):
            touched = True
            break

    if touched:
        # Back off a little so the lid remains hinged and stable.
        relief = float(final - sign * abs(float(backoff)))
        _set_lid_target_position(env, relief)
        step(pr, 20)
        ja = _get_lid_joint_angle(env)
        final = float(relief if ja is None else ja)
        return False, final
    return True, final


def _arc_waypoints(handle_pos, hinge_pos, rotation_amount, n=22):
    h = np.array(handle_pos, dtype=float)
    p = np.array(hinge_pos, dtype=float)
    vec = h - p
    radius = float(np.sqrt(vec[1] ** 2 + vec[2] ** 2))
    theta0 = float(np.arctan2(vec[2], vec[1]))
    waypoints = []
    for off in np.linspace(0.0, rotation_amount, max(5, int(n))):
        theta = theta0 + off
        y = p[1] + radius * np.cos(theta)
        z = p[2] + radius * np.sin(theta)
        x = h[0]
        waypoints.append((np.array([x, y, z], dtype=float), float(off)))
    return waypoints


def _arc_waypoints_close(handle_pos, hinge_pos, n=20):
    return _arc_waypoints(handle_pos, hinge_pos, rotation_amount=(-np.pi / 2.1), n=n)


def _arc_waypoints_open(handle_pos, hinge_pos, n=20):
    return _arc_waypoints(handle_pos, hinge_pos, rotation_amount=(np.pi / 2.1), n=n)


def _arc_trajectory(env, waypoints, base_tilt):
    traj = []
    prev = np.array(env.get_robot_conf(), dtype=float)
    original = env.get_robot_conf()
    if waypoints:
        # Keep the first point exactly at current grasp configuration so we
        # don't "retreat" immediately after grasp.
        traj.append(prev.tolist())
        start_idx = 1
    else:
        start_idx = 0
    for pos, off in waypoints[start_idx:]:
        solved = None
        for var in [0.0, 0.05, -0.05, 0.1, -0.1, 0.15, -0.15]:
            q = quaternion_from_euler(base_tilt + off + var, 0, np.pi / 2)
            try:
                cfgs = env.robot.solve_ik_via_sampling(
                    pos.tolist(), quaternion=q, max_configs=30, max_time_ms=150, ignore_collisions=True
                )
            except Exception:
                cfgs = None
            if cfgs is None or len(cfgs) == 0:
                continue
            best = min(cfgs, key=lambda c: float(np.linalg.norm(np.array(c, dtype=float) - prev)))
            solved = list(best)
            break
        if solved is None:
            solved = prev.tolist()
        traj.append(solved)
        prev = np.array(solved, dtype=float)
    env.set_robot_conf(original)
    return traj


def _smooth_trajectory(trajectory, window_size=3):
    if trajectory is None or len(trajectory) < window_size:
        return trajectory
    arr = np.array(trajectory, dtype=float)
    out = []
    for i in range(len(trajectory)):
        lo = max(0, i - window_size // 2)
        hi = min(len(trajectory), i + window_size // 2 + 1)
        out.append(np.mean(arr[lo:hi], axis=0).tolist())
    return out


def _execute_smooth(env, pr, traj, steps_per_waypoint=8, pause_steps=1):
    if traj is None or len(traj) == 0:
        return
    if len(traj) == 1:
        env.set_robot_conf(traj[0])
        step(pr, 5)
        return
    for i in range(len(traj) - 1):
        q1 = np.array(traj[i], dtype=float)
        q2 = np.array(traj[i + 1], dtype=float)
        for t in np.linspace(0.0, 1.0, max(2, int(steps_per_waypoint)), endpoint=False):
            env.set_robot_conf(((1.0 - t) * q1 + t * q2).tolist())
            step(pr, pause_steps)
    env.set_robot_conf(traj[-1])
    step(pr, max(2, pause_steps * 2))


def _run_open_lid_motion_clean(env, pr, task_name):
    """
    Clean open-lid sequence (closed -> 95 deg open):
      1) horizontal hover facing handle (fingers vertical),
      2) straight approach + close gripper on handle,
      3) continuous hinge-centered circular motion to target angle,
      4) release only at the end, then perpendicular retreat to hover distance.
    """
    print("\n" + "=" * 68)
    print(f"TASK: {task_name}")
    print("Action: OPEN grill (clean hover->grasp->arc)")
    print("=" * 68)

    try:
        env.set_lid_collision_enabled(True)
    except Exception:
        pass

    handle = _get_active_handle(env)
    if handle is None:
        print("ERROR: handle object not found")
        return False

    hinge, hinge_axis = _hinge_pose_and_axis(env)
    if hinge is None:
        print("ERROR: hinge pose unavailable for open action")
        return False
    hinge_np = np.array(hinge, dtype=float)
    hinge_axis_np = np.array(hinge_axis, dtype=float)
    axis_n = float(np.linalg.norm(hinge_axis_np))
    if axis_n < 1e-8:
        hinge_axis_np = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        hinge_axis_np = hinge_axis_np / axis_n

    _open_gripper_fully(env, pr, velocity=0.32, max_steps=120)

    # -------------------------
    # 1) Hover facing handle (perpendicular to handle axis)
    # -------------------------
    handle_pos = _sample_handle_position(handle, pr=pr, samples=3)
    if handle_pos is None:
        handle_pos = np.array(handle.get_position(), dtype=float)
    handle_dir, handle_perp = _handle_direction_and_perp(handle)
    handle_perp = _unit_xy(handle_perp)
    if handle_perp is None:
        handle_perp = np.array([1.0, 0.0, 0.0], dtype=float)
    try:
        tip_pos_now = np.array(env.robot.get_tip().get_position(), dtype=float)
    except Exception:
        tip_pos_now = handle_pos + handle_perp * 0.25

    hover_dist = float(os.environ.get("GRILL_OPEN_HOVER_DIST", "0.20"))
    hover_z = float(os.environ.get("GRILL_OPEN_HOVER_Z", "0.09"))
    grasp_dist = float(os.environ.get("GRILL_OPEN_GRASP_DIST", "0.018"))
    grasp_z = float(os.environ.get("GRILL_OPEN_GRASP_Z", "0.010"))

    q_current = env.get_robot_conf()
    q_hover = None
    hover_q = None
    hover_pos = None
    approach_dir = None
    best_score = None

    # Evaluate both perpendicular sides and choose the smoother planned path.
    side_options = [handle_perp, -handle_perp]
    side_options.sort(
        key=lambda s: float(np.linalg.norm((handle_pos + s * hover_dist + np.array([0.0, 0.0, hover_z])) - tip_pos_now))
    )
    for side in side_options:
        cand_hover_pos = (handle_pos + side * hover_dist + np.array([0.0, 0.0, hover_z], dtype=float)).tolist()
        facing_dir = -np.array(side, dtype=float)
        for q_try in _orientation_candidates_for_open_face(facing_dir):
            try:
                cfgs = env.robot.solve_ik_via_sampling(
                    cand_hover_pos,
                    quaternion=q_try,
                    max_configs=8,
                    max_time_ms=120,
                    ignore_collisions=True,
                )
            except Exception:
                cfgs = None
        if cfgs is None or len(cfgs) == 0:
            continue
        c = min(
            cfgs,
            key=lambda cc: float(np.linalg.norm(np.array(cc, dtype=float) - np.array(q_current, dtype=float))),
        )
        motion = env.compute_motion_plan(q_current, list(c))
        if not motion:
            continue
        align = _open_face_alignment_score(q_try, facing_dir)
        if align < 0.75:
            continue
        score = float(len(motion)) - 8.0 * float(align)
        if (best_score is None) or (score < best_score):
            best_score = score
            q_hover = list(c)
            hover_q = q_try
            hover_pos = list(cand_hover_pos)
            approach_dir = np.array(side, dtype=float)
    if q_hover is None or hover_pos is None or approach_dir is None:
        print("ERROR: failed to find clean hover config")
        return False

    print(f"[open] Step 1/4 hover target: {np.round(np.array(hover_pos), 4).tolist()}")
    _move_to_conf(env, pr, q_hover, label="open->hover")
    step(pr, 8)

    # -------------------------
    # 2) Forward grasp
    # -------------------------
    handle_pos = _sample_handle_position(handle, pr=pr, samples=3)
    if handle_pos is None:
        handle_pos = np.array(handle.get_position(), dtype=float)
    grasp_pos = (handle_pos + approach_dir * grasp_dist + np.array([0.0, 0.0, grasp_z], dtype=float)).tolist()

    print(f"[open] Step 2/4 grasp target: {np.round(np.array(grasp_pos), 4).tolist()}")
    # Motion-plan to grasp config first, then short linear forward if available.
    q_grasp = None
    try:
        cfgs = env.robot.solve_ik_via_sampling(
            grasp_pos,
            quaternion=hover_q,
            max_configs=12,
            max_time_ms=180,
            ignore_collisions=True,
        )
    except Exception:
        cfgs = None
    if cfgs is not None and len(cfgs) > 0:
        q_grasp = min(
            cfgs,
            key=lambda cc: float(np.linalg.norm(np.array(cc, dtype=float) - np.array(env.get_robot_conf(), dtype=float))),
        )
    if q_grasp is None:
        print("ERROR: failed to solve grasp config")
        return False
    _move_to_conf(env, pr, list(q_grasp), label="open->grasp")

    path_in = None
    try:
        path_in = env._get_linear_path(
            env.get_robot_conf(),
            grasp_pos,
            hover_q,
            ignore_collisions=True,
            steps=40,
        )
    except Exception:
        path_in = None
    if path_in is not None:
        traj_in = path_in._path_points.reshape(-1, 7).tolist()
        execute_trajectory(env, pr, traj_in, steps_per_segment=2)
    step(pr, 10)

    _close_gripper_fully(env, pr, velocity=0.14, max_steps=150)
    step(pr, 12)

    # -------------------------
    # 3) Continuous arc to 95 deg
    # -------------------------
    start_handle = _sample_handle_position(handle, pr=pr, samples=2)
    if start_handle is None:
        start_handle = np.array(handle.get_position(), dtype=float)
    start_tip = np.array(env.robot.get_tip().get_position(), dtype=float)
    start_q = env.get_robot_conf()
    current_joint = _get_lid_joint_angle(env)
    if current_joint is None:
        current_joint = float(LID_CLOSED_ANGLE)
    # Enforce at least 95 deg opening travel while gripper remains closed.
    nominal_target = float(LID_OPEN_ANGLE)
    nominal_delta = float(nominal_target - current_joint)
    min_open = float(math.radians(95.0))
    if abs(nominal_delta) >= min_open:
        target_joint = nominal_target
    else:
        sign = -1.0 if nominal_delta < 0.0 else 1.0
        if abs(nominal_delta) < 1e-3:
            # Prefer opening direction away from closed angle baseline.
            sign = -1.0 if float(LID_CLOSED_ANGLE) > current_joint else 1.0
        target_joint = float(current_joint + sign * min_open)
    rotation_amount = float(target_joint - current_joint)

    # Build continuous tip waypoints around hinge using current grasp offset.
    r_tip = start_tip - hinge_np
    n_arc = max(18, int(os.environ.get("GRILL_OPEN_ARC_SEGMENTS", "28")))
    arc_confs = [list(start_q)]
    prev = np.array(start_q, dtype=float)
    for off in np.linspace(0.0, rotation_amount, n_arc)[1:]:
        R = _axis_angle_matrix(hinge_axis_np, float(off))
        tip_target = hinge_np + (R @ r_tip)
        # Keep orientation mostly fixed to reduce erratic flips while maintaining stable grasp.
        solved = None
        arc_facing = -np.array(R @ approach_dir, dtype=float)
        for q_try in [hover_q] + _orientation_candidates_for_open_face(arc_facing):
            try:
                cfgs = env.robot.solve_ik_via_sampling(
                    tip_target.tolist(),
                    quaternion=q_try,
                    max_configs=8,
                    max_time_ms=100,
                    ignore_collisions=True,
                )
            except Exception:
                cfgs = None
            if cfgs is None or len(cfgs) == 0:
                continue
            cand = min(cfgs, key=lambda cc: float(np.linalg.norm(np.array(cc, dtype=float) - prev)))
            solved = list(cand)
            break
        if solved is None:
            solved = prev.tolist()
        arc_confs.append(solved)
        prev = np.array(solved, dtype=float)

    dense = _interp_traj(arc_confs, steps_per_segment=10)
    print(
        f"[open] Step 3/4 continuous arc ({current_joint:.3f} -> {target_joint:.3f} rad, {len(dense)} points)"
    )
    _set_lid_servo_lock(env, True)
    dn = max(1, len(dense) - 1)
    for i, q in enumerate(dense):
        frac = float(i) / float(dn)
        a = float(current_joint + frac * (target_joint - current_joint))
        env.set_robot_conf(q)
        _set_lid_target_position(env, a)
        step(pr, 1)
    _set_lid_joint_angle(env, pr, target_joint, steps=40)
    step(pr, 10)

    # -------------------------
    # 4) Release and perpendicular retreat
    # -------------------------
    print("[open] Step 4/4 release + retreat")
    try:
        env.gripper.release()
    except Exception:
        pass
    _open_gripper_fully(env, pr, velocity=0.35, max_steps=140)

    final_handle = _sample_handle_position(handle, pr=pr, samples=2)
    if final_handle is None:
        final_handle = np.array(handle.get_position(), dtype=float)
    Rf = _axis_angle_matrix(hinge_axis_np, rotation_amount)
    retreat_dir = _unit_xy(Rf @ approach_dir)
    if retreat_dir is None:
        retreat_dir = _unit_xy(approach_dir)
    if retreat_dir is None:
        retreat_dir = np.array([1.0, 0.0, 0.0], dtype=float)
    retreat_pos = (
        final_handle
        + retreat_dir * hover_dist
        + np.array([0.0, 0.0, hover_z], dtype=float)
    ).tolist()
    path_retreat = None
    try:
        path_retreat = env._get_linear_path(
            env.get_robot_conf(),
            retreat_pos,
            hover_q,
            ignore_collisions=True,
            steps=75,
        )
    except Exception:
        path_retreat = None
    if path_retreat is not None:
        traj_retreat = path_retreat._path_points.reshape(-1, 7).tolist()
        execute_trajectory(env, pr, traj_retreat, steps_per_segment=4)
    else:
        # Last-resort fallback.
        q_post, _q_post_or, _q_post_pos, _q_post_dir = _compute_handle_hover_config(env, handle, direction="open")
        if q_post is not None:
            _move_to_conf(env, pr, q_post, label="open->post_hover")
    step(pr, 8)

    final_joint = _get_lid_joint_angle(env)
    if final_joint is not None:
        jerr = abs(float(final_joint) - float(target_joint))
        ok = (jerr <= float(LID_ANGLE_TOL))
        print(
            f"[open] Lid angle: {final_joint:.3f} rad | target: {target_joint:.3f} rad | err: {jerr:.3f}"
        )
        return bool(ok)
    moved = float(np.linalg.norm(np.array(handle.get_position(), dtype=float) - np.array(start_handle, dtype=float)))
    print(f"[open] Handle moved: {moved:.3f}m")
    return bool(moved > 0.04)


def run_grill_lid_motion(env, pr, direction, task_name):
    global LID_OPEN_ANGLE
    """
    direction: 'open' or 'close'
    Performs handle grasp + circular arc + detach + release.
    """
    if direction == "open":
        return _run_open_lid_motion_clean(env, pr, task_name)

    print("\n" + "=" * 68)
    print(f"TASK: {task_name}")
    print(f"Action: {direction.upper()} grill (circular arc)")
    print("=" * 68)

    _restore_handle_anchor(env, pr)

    handle = _get_active_handle(env)
    if handle is None:
        print("ERROR: handle object not found")
        return False

    target_wp = CLOSE_WP_NAME
    # Ensure gripper is open once before planning approach.
    _open_gripper_fully(env, pr, velocity=0.32, max_steps=120)
    # Use live handle object so hover/grasp stays aligned with current handle orientation.
    q_hover, hover_q, _hover_pos, approach_dir = _compute_handle_hover_config(
        env, handle, direction=direction
    )
    if q_hover is None:
        print("ERROR: could not find hover config for handle")
        return False
    _move_to_conf(env, pr, q_hover, label=f"{direction}->hover")
    step(pr, 10)

    q_grasp, grasp_pos = _compute_handle_grasp_config(
        env, handle, hover_q, approach_dir=approach_dir, direction=direction
    )
    if q_grasp is None:
        print("ERROR: could not find grasp config for handle")
        return False
    print(f"[{direction}] Grasp target: {np.round(np.array(grasp_pos), 4).tolist()}")
    # Close: prefer Cartesian approach for cleaner perpendicular pre-grasp.
    path_in = None
    try:
        path_in = env._get_linear_path(
            env.get_robot_conf(),
            grasp_pos,
            hover_q,
            ignore_collisions=True,
            steps=55,
        )
    except Exception:
        path_in = None
    if path_in is not None:
        traj_in = path_in._path_points.reshape(-1, 7).tolist()
        execute_trajectory(env, pr, traj_in, steps_per_segment=3)
    else:
        q_curr = np.array(env.get_robot_conf(), dtype=float)
        q_goal = np.array(q_grasp, dtype=float)
        direct = [((1.0 - t) * q_curr + t * q_goal).tolist() for t in np.linspace(0.0, 1.0, 55)]
        execute_trajectory(env, pr, direct, steps_per_segment=1)
    step(pr, 20)

    # Close once at grasp point and attempt explicit attachment.
    _close_gripper_fully(env, pr, velocity=0.18, max_steps=110)
    grasped_handle = False
    if ENABLE_HANDLE_ATTACH:
        grasped_handle = _attempt_handle_attach(env, pr, handle, repeats=10)
        if not grasped_handle:
            grasped_handle = _attempt_handle_attach(env, pr, env.get_object("lid_visual"), repeats=8)
        if not grasped_handle:
            grasped_handle = _attempt_handle_attach(env, pr, env.get_object("lid"), repeats=8)
    else:
        grasped_handle = _gripper_detects(env, handle)
    if not grasped_handle:
        print(f"[{direction}] WARNING: handle not explicitly attached; continuing with contact grasp.")

    hinge = _hinge_position()
    if hinge is None:
        hpos = handle.get_position()
        hinge = [hpos[0], hpos[1] + 0.25, hpos[2] - 0.12]
        print(f"[{direction}] Hinge joint unavailable, estimated hinge: {hinge}")

    start_pos = list(handle.get_position())
    current_joint = None
    target_joint = None
    if getattr(env, "lid_joint", None) is not None:
        try:
            current_joint = float(env.lid_joint.get_joint_position())
            target_joint = float(LID_OPEN_ANGLE if direction == "open" else LID_CLOSED_ANGLE)
        except Exception:
            current_joint = None
            target_joint = None

    current_handle_pos = list(handle.get_position())
    contact_limited = False
    if (current_joint is not None) and (target_joint is not None):
        # Execute true robot circular motion while synchronizing hinge joint
        # progression. This preserves the hinge while keeping visible grasp+arc.
        rotation_amount = float(target_joint - current_joint)
        waypoints = _arc_waypoints(current_handle_pos, hinge, rotation_amount=rotation_amount, n=24)
        base_tilt = (np.pi - 0.6)
        arc_traj = _arc_trajectory(env, waypoints, base_tilt=base_tilt)
        if not arc_traj:
            print("ERROR: no arc trajectory generated")
            return False
        arc_traj = _smooth_trajectory(arc_traj, window_size=5)
        dense = _interp_traj(arc_traj, steps_per_segment=10)
        print(
            f"[{direction}] Executing synchronized grasp arc + hinge "
            f"({current_joint:.3f} -> {target_joint:.3f} rad, {len(dense)} points)..."
        )
        _set_lid_servo_lock(env, True)
        n = max(1, len(dense) - 1)
        for i, q in enumerate(dense):
            f = float(i) / float(n)
            a = float(current_joint + f * (target_joint - current_joint))
            env.set_robot_conf(q)
            _set_lid_target_position(env, a)
            step(pr, 1)
            if direction == "close" and _lid_hits_food(env):
                contact_limited = True
                # Back off slightly to avoid over-forcing into food.
                relief = float(a - np.sign(target_joint - current_joint) * 0.05)
                _set_lid_target_position(env, relief)
                step(pr, 20)
                target_joint = float(_get_lid_joint_angle(env) or relief)
                print(f"[{direction}] Contact-limited close at {target_joint:.3f} rad")
                break
        step(pr, 25)
    else:
        # Fallback only when joint state is unavailable.
        _set_lid_servo_lock(env, False)
        step(pr, 3)
        rotation_amount = float((np.pi / 2.1) if direction == "open" else (-np.pi / 2.1))
        waypoints = _arc_waypoints(current_handle_pos, hinge, rotation_amount=rotation_amount, n=24)
        base_tilt = (np.pi - 0.6)
        arc_traj = _arc_trajectory(env, waypoints, base_tilt=base_tilt)
        if not arc_traj:
            print("ERROR: no arc trajectory generated")
            return False
        arc_traj = _smooth_trajectory(arc_traj, window_size=5)
        print(f"[{direction}] Executing circular arc with {len(arc_traj)} waypoints...")
        _execute_smooth(env, pr, arc_traj, steps_per_waypoint=12, pause_steps=2)
        step(pr, 25)

    # Hard-enforce the final lid angle for uniform open/close amount.
    if target_joint is not None:
        _set_lid_joint_angle(env, pr, target_joint, steps=75)
        step(pr, 10)
        _set_lid_servo_lock(env, True)
        step(pr, 3)
    # Visual fallback can destabilize this scene; keep it opt-in.
    if USE_LID_WAYPOINT_SNAP:
        _snap_lid_to_waypoint(env, pr, target_wp, iters=3)

    # Release directly after arc (no retreat/detach).
    try:
        env.gripper.release()
    except Exception:
        pass
    _open_gripper_fully(env, pr, velocity=0.35, max_steps=140)
    _restore_handle_anchor(env, pr)
    step(pr, 20)
    did_post_hover = False
    # Explicit horizontal retrieve from handle after release.
    try:
        final_handle_pos = np.array(handle.get_position(), dtype=float)
        retreat_dist = float(os.environ.get("GRILL_CLOSE_RETRIEVE_DIST", "0.11"))
        retreat_z = float(os.environ.get("GRILL_CLOSE_RETRIEVE_Z_OFFSET", "0.0"))
        a = np.array(approach_dir, dtype=float)
        a[2] = 0.0
        an = float(np.linalg.norm(a))
        if an > 1e-8:
            a = a / an
            retreat_pos = (
                final_handle_pos
                + a * retreat_dist
                + np.array([0.0, 0.0, retreat_z], dtype=float)
            ).tolist()
            path_retreat = None
            try:
                path_retreat = env._get_linear_path(
                    env.get_robot_conf(),
                    retreat_pos,
                    hover_q,
                    ignore_collisions=True,
                    steps=65,
                )
            except Exception:
                path_retreat = None
            if path_retreat is not None:
                traj_retreat = path_retreat._path_points.reshape(-1, 7).tolist()
                execute_trajectory(env, pr, traj_retreat, steps_per_segment=4)
                step(pr, 8)
                did_post_hover = True
    except Exception:
        pass

    # Move to a safe post-action hover near the final handle pose.
    if not did_post_hover:
        try:
            q_post, _q_post_or, _q_post_pos, _q_post_dir = _compute_handle_hover_config(
                env, handle, direction=direction
            )
            if q_post is not None:
                _move_to_conf(env, pr, q_post, label=f"{direction}->post_hover")
                step(pr, 8)
        except Exception:
            pass

    # Validation with joint-angle target if available.
    if target_joint is not None and getattr(env, "lid_joint", None) is not None:
        try:
            final_joint = float(env.lid_joint.get_joint_position())
            jerr = abs(final_joint - float(target_joint))
            if direction == "open" and jerr > float(LID_ANGLE_TOL):
                # Retry hard-lock once more; some scenes spring back right after arc.
                print(f"[{direction}] retrying hinge lock to target {target_joint:.3f} rad...")
                _set_lid_servo_lock(env, True)
                _set_lid_joint_angle(env, pr, target_joint, steps=100)
                step(pr, 16)
                final_joint = float(env.lid_joint.get_joint_position())
                jerr = abs(final_joint - float(target_joint))
            ok = (jerr <= float(LID_ANGLE_TOL)) or bool(contact_limited and direction == "close")
            print(
                f"[{direction}] Lid angle: {final_joint:.3f} rad | "
                f"target: {target_joint:.3f} rad | err: {jerr:.3f}"
            )
            if not ok:
                print(f"WARNING: {direction} angle mismatch.")
        except Exception:
            ok = False
    else:
        moved = float(np.linalg.norm(np.array(handle.get_position(), dtype=float) - np.array(start_pos, dtype=float)))
        ok = moved > 0.04
        print(f"[{direction}] Handle moved: {moved:.3f}m")

    return bool(ok)


def _pick_by_label(items, label):
    for it in items:
        if it["label"] == label:
            return it
    return None


def main():
    global LID_CLOSED_ANGLE, LID_OPEN_ANGLE
    env = ENV
    pr = env.pr

    print("=" * 72)
    print("GROUND TRUTH ORCHESTRATOR - GRILL VARIATION 1")
    print("=" * 72)
    print(f"Scene: {SCENE_PATH}")
    if (not ALLOW_SCENE_OVERRIDE) and SCENE_OVERRIDE:
        print(f"[startup] Ignoring GRILL_SCENE_FILE_OVERRIDE='{SCENE_OVERRIDE}' (set GRILL_ALLOW_SCENE_OVERRIDE=True to use it).")
    _discover_lid_joint_handle(env, pr)
    _discover_active_handle(env, pr)
    _capture_handle_anchor(env)
    _restore_handle_anchor(env, pr)
    if LID_AUTOCALIBRATE:
        _calibrate_lid_from_waypoints(env, pr)
    pre = _get_lid_joint_angle(env)
    d_close_pre = _handle_waypoint_distance(env, CLOSE_WP_NAME)
    d_open_pre = _handle_waypoint_distance(env, OPEN_WP_NAME)
    if USE_INITIAL_LID_AS_CLOSED and (pre is not None):
        # If initial pose looks open-like, infer closed angle from local
        # gradient around the current joint value.
        closed_est, inferred = _estimate_closed_angle_from_current(
            env, pr, pre, d_close_pre, d_open_pre
        )
        if inferred:
            LID_OPEN_ANGLE = float(pre)
            LID_CLOSED_ANGLE = float(closed_est)
        else:
            LID_CLOSED_ANGLE = float(pre)
            LID_OPEN_ANGLE = float(LID_CLOSED_ANGLE + abs(float(LID_TRAVEL_ANGLE)))
    _enforce_min_open_travel()
    print(f"[startup] Lid targets from scene: open={LID_OPEN_ANGLE:.3f}, closed={LID_CLOSED_ANGLE:.3f}")
    if pre is not None:
        print(f"Lid angle before startup lock: {pre:.3f} rad")
    if (d_close_pre is not None) and (d_open_pre is not None):
        print(f"[startup] pre-lock handle distances: d_close={d_close_pre:.3f}, d_open={d_open_pre:.3f}")
    if ACTIVE_LID_JOINT_HANDLE is not None:
        print(f"Active lid joint: {_get_joint_alias(ACTIVE_LID_JOINT_HANDLE)} (h={ACTIVE_LID_JOINT_HANDLE})")
    # Stabilize startup state (freeze task objects + hold lid closed angle).
    try:
        env.stabilize_startup_state(steps=15)
        print(f"Lid targets: closed={LID_CLOSED_ANGLE:.3f} rad, open={LID_OPEN_ANGLE:.3f} rad")
        _set_lid_servo_lock(env, True)
        if KEEP_LID_COLLISION_OFF_UNTIL_OPEN:
            try:
                env.set_lid_collision_enabled(False)
            except Exception:
                pass
            _set_lid_joint_angle(env, pr, LID_CLOSED_ANGLE, steps=90)
            _force_lid_closed(env, pr, steps=100)
        else:
            reached, final_closed = _close_lid_until_contact(
                env, pr, LID_CLOSED_ANGLE, steps=130, backoff=0.05
            )
            if final_closed is not None:
                LID_CLOSED_ANGLE = float(final_closed)
                _enforce_min_open_travel()
                print(
                    f"[startup] contact-safe close: reached={reached} | "
                    f"closed_angle={LID_CLOSED_ANGLE:.3f}, open_angle={LID_OPEN_ANGLE:.3f}"
                )
        snapped = False
        if USE_LID_WAYPOINT_SNAP:
            snapped = _snap_lid_to_waypoint(env, pr, CLOSE_WP_NAME, iters=4)
        d_close = _handle_waypoint_distance(env, CLOSE_WP_NAME)
        d_open = _handle_waypoint_distance(env, OPEN_WP_NAME)
        # Optional heavy fallback (disabled by default to avoid destabilization).
        if STARTUP_CLOSE_SCAN and (
            d_close is not None and d_open is not None and (d_close > d_open + 0.03)
        ):
            print(
                f"[startup] Lid appears open (d_close={d_close:.3f} > d_open={d_open:.3f}). "
                "Running fallback close-angle scan..."
            )
            best = None
            for a in np.linspace(-3.2, 3.2, 41):
                _set_lid_joint_angle(env, pr, float(a), steps=3)
                d = _handle_waypoint_distance(env, CLOSE_WP_NAME)
                if d is None:
                    continue
                if (best is None) or (d < best[0]):
                    best = (float(d), float(a))
            if best is not None:
                best_d, best_a = best
                LID_CLOSED_ANGLE = float(best_a)
                LID_OPEN_ANGLE = float(LID_CLOSED_ANGLE + LID_TRAVEL_ANGLE)
                _enforce_min_open_travel()
                print(
                    f"[startup] Fallback closed angle selected: {LID_CLOSED_ANGLE:.3f} "
                    f"(d_close={best_d:.3f}) | open={LID_OPEN_ANGLE:.3f}"
                )
                _set_lid_joint_angle(env, pr, LID_CLOSED_ANGLE, steps=60)
                _force_lid_closed(env, pr, steps=80)
                if USE_LID_WAYPOINT_SNAP:
                    snapped = _snap_lid_to_waypoint(env, pr, CLOSE_WP_NAME, iters=4) or snapped
                d_close = _handle_waypoint_distance(env, CLOSE_WP_NAME)
                d_open = _handle_waypoint_distance(env, OPEN_WP_NAME)
        post = _get_lid_joint_angle(env)
        if post is not None:
            print(f"Lid angle after startup lock: {post:.3f} rad")
        print(
            f"[startup] snap_to_{CLOSE_WP_NAME}: {snapped} | "
            f"d_close={d_close if d_close is not None else 'NA'} | "
            f"d_open={d_open if d_open is not None else 'NA'}"
        )
        if not KEEP_LID_COLLISION_OFF_UNTIL_OPEN:
            try:
                env.set_lid_collision_enabled(True)
            except Exception:
                pass
        print("Startup stabilization applied (objects frozen, lid set to closed target).")
    except Exception:
        pass

    # Minimal settle only.
    step(pr, 3)
    go_home(env, pr)
    # Re-enforce closed start immediately after home move.
    if KEEP_LID_COLLISION_OFF_UNTIL_OPEN:
        try:
            env.set_lid_collision_enabled(False)
        except Exception:
            pass
        _set_lid_joint_angle(env, pr, LID_CLOSED_ANGLE, steps=35)
        _force_lid_closed(env, pr, steps=50)
        if not KEEP_LID_COLLISION_OFF_UNTIL_OPEN:
            try:
                env.set_lid_collision_enabled(True)
            except Exception:
                pass
    else:
        reached_home, final_home_closed = _close_lid_until_contact(
            env, pr, LID_CLOSED_ANGLE, steps=90, backoff=0.05
        )
        if final_home_closed is not None:
            LID_CLOSED_ANGLE = float(final_home_closed)
            _enforce_min_open_travel()
        print(
            f"[startup] post-home contact-safe close: reached={reached_home} | "
            f"closed_angle={LID_CLOSED_ANGLE:.3f}, open_angle={LID_OPEN_ANGLE:.3f}"
        )
    snapped2 = False
    if USE_LID_WAYPOINT_SNAP:
        snapped2 = _snap_lid_to_waypoint(env, pr, CLOSE_WP_NAME, iters=2)
    d_close2 = _handle_waypoint_distance(env, CLOSE_WP_NAME)
    post_home = _get_lid_joint_angle(env)
    if post_home is not None:
        print(f"Lid angle after home lock: {post_home:.3f} rad")
    print(f"[startup] post-home snap_to_{CLOSE_WP_NAME}: {snapped2} | d_close={d_close2 if d_close2 is not None else 'NA'}")
    step(pr, 3)

    meats = _discover_meat_objects(env)
    plate = _discover_plate(env)
    if plate is None:
        print("ERROR: plate object not found in scene.")
        pr.stop()
        pr.shutdown()
        return

    print("\nDiscovered task objects:")
    print(f"  plate: {plate['name']}")
    for m in meats:
        print(f"  meat : {m['name']} ({m['label']})")

    in_grill, on_plate, outside = _classify_meats(env, meats)
    print(f"\nInitial classification: in_grill={len(in_grill)}, on_plate={len(on_plate)}, outside={len(outside)}")

    # Select steak initially inside grill (requested in step 3).
    inside_steak = _pick_by_label(in_grill, "steak")
    if inside_steak is None and in_grill:
        inside_steak = in_grill[0]

    outside_chicken = _pick_by_label(outside, "chicken")
    outside_steak = _pick_by_label([m for m in outside if m != outside_chicken], "steak")

    results = []

    # 1) Open grill
    ok = run_grill_lid_motion(env, pr, direction="open", task_name="Task 1: Open Grill")
    results.append(("Task 1: open grill", ok))
    go_home(env, pr)

    # 2) Plate the plate (vertical pick, horizontal place on plate_boundary)
    plate_pose = _region_slot_pose(env, plate["obj"], "plate_boundary", slot_idx=0, slot_count=1)
    ok = run_pick_place(
        env,
        pr,
        obj_name=plate["name"],
        target_region="plate_boundary",
        task_name="Task 2: Plate -> plate_boundary",
        is_plate=True,
        target_pose=plate_pose,
    )
    results.append(("Task 2: plate -> plate_boundary", ok))
    go_home(env, pr)

    # 3) Steak inside grill -> plate
    if inside_steak is None:
        print("WARNING: No inside steak found for Task 3.")
        ok = False
    else:
        pose_plate_center = _region_slot_pose(env, inside_steak["obj"], "plate-top", slot_idx=0, slot_count=3)
        ok = run_pick_place(
            env,
            pr,
            obj_name=inside_steak["name"],
            target_region="plate-top",
            task_name="Task 3: Steak in Grill -> Plate",
            is_plate=False,
            target_pose=pose_plate_center,
        )
    results.append(("Task 3: steak_in_grill -> plate", ok))
    go_home(env, pr)

    # 4) Chicken + outside steak -> grill
    to_grill = [x for x in [outside_chicken, outside_steak] if x is not None]
    if not to_grill:
        print("WARNING: No outside chicken/steak found for Task 4.")
        results.append(("Task 4a: chicken outside -> grill", False))
        results.append(("Task 4b: steak outside -> grill", False))
    else:
        for i, m in enumerate(to_grill):
            target_pose = _region_slot_pose(env, m["obj"], "grill-top", slot_idx=i, slot_count=max(2, len(to_grill)))
            ok = run_pick_place(
                env,
                pr,
                obj_name=m["name"],
                target_region="grill-top",
                task_name=f"Task 4.{i+1}: {m['label']} outside -> grill",
                is_plate=False,
                target_pose=target_pose,
            )
            results.append((f"Task 4.{i+1}: {m['label']} outside -> grill", ok))
            go_home(env, pr)

    # 5) Close grill
    ok = run_grill_lid_motion(env, pr, direction="close", task_name="Task 5: Close Grill")
    results.append(("Task 5: close grill", ok))
    go_home(env, pr)

    # 6) Open grill
    ok = run_grill_lid_motion(env, pr, direction="open", task_name="Task 6: Open Grill")
    results.append(("Task 6: open grill", ok))
    go_home(env, pr)

    # 7) Chicken + steak inside grill -> plate
    in_grill_now, _on_plate_now, _outside_now = _classify_meats(env, meats)
    to_plate = []
    chicken_in = _pick_by_label(in_grill_now, "chicken")
    steak_in = _pick_by_label([m for m in in_grill_now if m != chicken_in], "steak")
    if chicken_in is not None:
        to_plate.append(chicken_in)
    if steak_in is not None:
        to_plate.append(steak_in)
    for m in in_grill_now:
        if m not in to_plate:
            to_plate.append(m)
    to_plate = to_plate[:2]

    if not to_plate:
        print("WARNING: No meats found inside grill for Task 7.")
        results.append(("Task 7a: chicken_in_grill -> plate", False))
        results.append(("Task 7b: steak_in_grill -> plate", False))
    else:
        # Keep slot 0 occupied by the first plated steak from Task 3.
        for i, m in enumerate(to_plate):
            pose_plate = _region_slot_pose(env, m["obj"], "plate-top", slot_idx=i + 1, slot_count=3)
            ok = run_pick_place(
                env,
                pr,
                obj_name=m["name"],
                target_region="plate-top",
                task_name=f"Task 7.{i+1}: {m['label']} in grill -> plate",
                is_plate=False,
                target_pose=pose_plate,
            )
            results.append((f"Task 7.{i+1}: {m['label']} in grill -> plate", ok))
            go_home(env, pr)

    print("\n" + "=" * 72)
    print("EXECUTION SUMMARY - GRILL VARIATION 1")
    print("=" * 72)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nTotal: {passed}/{total} tasks passed")
    print("=" * 72)

    print("\nGrill variation-1 orchestration complete. Press Ctrl+C to close.")
    try:
        while True:
            pr.step()
    except KeyboardInterrupt:
        pass

    pr.stop()
    pr.shutdown()


if __name__ == "__main__":
    main()

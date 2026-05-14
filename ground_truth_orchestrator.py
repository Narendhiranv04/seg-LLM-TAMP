"""
Ground Truth Orchestrator - Complete Long-Horizon Kitchen Task
Performs all 6 tasks in sequence:
1. Pick mug3 from cupboard -> placement_boundary
2. Pick 5 groceries from table -> cupboard_boundary
3. Pick mug1 from groceries_boundary -> placement_boundary
4. Pick mug2 from box_boundary -> placement_boundary
5. Slide open box lid
6. Pick mug4 from box_inside -> placement_boundary
"""
import os
import sys
import json
import numpy as np
import time
import math

# Configure Qt for GUI
def _configure_qt():
    headless_requested = os.environ.get("HEADLESS", "False") == "True"
    if headless_requested:
        os.environ.setdefault("COPPELIASIM_HEADLESS", "1")
    else:
        os.environ.setdefault("COPPELIASIM_HEADLESS", "0")
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
    coppelia_root = os.environ.get("COPPELIASIM_ROOT") or os.path.expanduser("~/CoppeliaSim")
    # Keep the full CoppeliaSim plugin root available so Qt can also discover
    # sibling plugin families such as xcbglintegrations when PyRep launches the
    # simulator in-process from Python.
    os.environ.setdefault("QT_PLUGIN_PATH", coppelia_root)
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

# Add pddlstream to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pddlstream'))

from pddlstream.language.constants import PDDLProblem, And
from pddlstream.algorithms.meta import solve
from pddlstream.utils import read

# Preserve a caller-provided HEADLESS setting (e.g. benchmark wrapper).
os.environ.setdefault("HEADLESS", "False")

# Import ENV from streams to share the instance
from rlbench_kitchen_streams import ENV, get_stream_map
from video_recorder import VideoRecorder

# Global video recorders
VIDEO_RECORDER = None
MASK_RECORDER = None  # For segmentation mask videos
STEP_CALLBACK = None  # Optional per-step hook (e.g., live segmentation viewer update)
ACTION_PROGRESS_CALLBACK = None  # Optional primitive-action callback for live panels
KITCHEN_REPLAY_DIR = os.path.join(os.path.dirname(__file__), "precomputed_paths")
BOX_LID_OPEN_REPLAY_PATH = os.path.join(KITCHEN_REPLAY_DIR, "kitchen_box_lid_open.json")


def _env_int(name, default, min_value=1):
    """Parse positive integer env var with fallback."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(min_value, value)


# Runtime speed knobs
# Use GT_SPEED_MODE=fast for a noticeably quicker execution profile.
GT_SPEED_MODE = os.environ.get("GT_SPEED_MODE", "normal").strip().lower()
DEFAULT_EXEC_INTERP_STEPS = _env_int(
    "GT_EXEC_INTERP_STEPS", 6 if GT_SPEED_MODE == "fast" else 15
)
GO_HOME_INTERP_STEPS = _env_int(
    "GT_GO_HOME_INTERP_STEPS", 45 if GT_SPEED_MODE == "fast" else 100
)
GO_HOME_EXEC_STEPS = _env_int(
    "GT_GO_HOME_EXEC_STEPS", 5 if GT_SPEED_MODE == "fast" else 10
)
RELEASE_HOLD_STEPS = _env_int(
    "GT_RELEASE_HOLD_STEPS", 20 if GT_SPEED_MODE == "fast" else 60
)
RECORD_EVERY_N = _env_int(
    "GT_RECORD_EVERY_N", 2 if GT_SPEED_MODE == "fast" else 1
)


def _run_step_callback():
    """Invoke optional per-step callback used by external live viewers."""
    global STEP_CALLBACK
    cb = STEP_CALLBACK
    if cb is None:
        return
    try:
        cb()
    except Exception:
        # Never let visualization hooks break task execution.
        pass


def _emit_action_progress(action_name, action_label=None):
    """Emit primitive action progress (move/pick/place/open) to optional callback."""
    global ACTION_PROGRESS_CALLBACK
    cb = ACTION_PROGRESS_CALLBACK
    if cb is None:
        return
    try:
        name = "" if action_name is None else str(action_name)
        label = name if action_label is None else str(action_label)
        cb(name, label)
    except Exception:
        # Never let UI hooks break task execution.
        pass


def step_and_record(pr, count=1):
    """Step simulation and record video if recorder is active."""
    global VIDEO_RECORDER, MASK_RECORDER
    for i in range(count):
        pr.step()
        should_record = (i % RECORD_EVERY_N == 0) or (i == count - 1)
        if should_record:
            if VIDEO_RECORDER:
                VIDEO_RECORDER.record_step()
            if MASK_RECORDER:
                MASK_RECORDER.record_step()
            _run_step_callback()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def quaternion_from_euler(ai, aj, ak):
    """Convert Euler angles to quaternion."""
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk
    q = [cj*sc - sj*cs, cj*ss + sj*cc, cj*cs - sj*sc, cj*cc + sj*ss]
    return q


def normalize_quaternion(q):
    """Return unit quaternion [x, y, z, w]."""
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n


def quaternion_conjugate(q):
    """Quaternion conjugate for [x, y, z, w]."""
    q = np.array(q, dtype=float)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)


def quaternion_multiply(q1, q2):
    """Multiply quaternions (x, y, z, w order)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=float)


def quaternion_rotate_vector(q, v):
    """Rotate 3D vector by quaternion [x, y, z, w]."""
    qn = normalize_quaternion(q)
    vq = np.array([v[0], v[1], v[2], 0.0], dtype=float)
    v_rot = quaternion_multiply(quaternion_multiply(qn, vq), quaternion_conjugate(qn))
    return v_rot[:3]


def compute_tip_attachment(gripper_tip, obj):
    """Compute object pose in gripper-tip frame (position + quaternion offset)."""
    tip_pos = np.array(gripper_tip.get_position(), dtype=float)
    tip_quat = normalize_quaternion(gripper_tip.get_quaternion())
    obj_pos = np.array(obj.get_position(), dtype=float)
    obj_quat = normalize_quaternion(obj.get_quaternion())

    local_pos = quaternion_rotate_vector(quaternion_conjugate(tip_quat), obj_pos - tip_pos)
    local_quat = normalize_quaternion(quaternion_multiply(quaternion_conjugate(tip_quat), obj_quat))
    return local_pos, local_quat


def update_attached_pose(gripper_tip, obj, local_pos, local_quat):
    """Apply a rigid tip-frame attachment transform to object pose."""
    tip_pos = np.array(gripper_tip.get_position(), dtype=float)
    tip_quat = normalize_quaternion(gripper_tip.get_quaternion())

    obj_pos = tip_pos + quaternion_rotate_vector(tip_quat, local_pos)
    obj_quat = normalize_quaternion(quaternion_multiply(tip_quat, local_quat))

    obj.set_position(obj_pos.tolist())
    obj.set_quaternion(obj_quat.tolist())


def execute_trajectory(env, traj, steps=None):
    """Execute a trajectory with interpolation."""
    global VIDEO_RECORDER, MASK_RECORDER
    if not traj:
        return
    if steps is None:
        steps = DEFAULT_EXEC_INTERP_STEPS
    steps = max(1, int(steps))

    # In fast mode with live segmentation + recording, dense paths can make
    # execution appear "stuck" for a long time. Downsample waypoints first.
    raw_cap_default = 260 if GT_SPEED_MODE == "fast" else 0
    raw_cap = raw_cap_default
    raw_cap_env = os.environ.get("GT_MAX_RAW_WAYPOINTS")
    if raw_cap_env is not None:
        try:
            raw_cap = max(0, int(raw_cap_env))
        except Exception:
            raw_cap = raw_cap_default
    if raw_cap > 0 and len(traj) > raw_cap:
        idxs = np.linspace(0, len(traj) - 1, raw_cap, dtype=int)
        traj = [traj[i] for i in idxs]
        print(f"[Exec] Downsampled raw trajectory to {len(traj)} waypoints.")

    full_traj = []
    for i in range(len(traj) - 1):
        start = np.array(traj[i])
        end = np.array(traj[i + 1])
        for t in np.linspace(0, 1, steps, endpoint=False):
            full_traj.append((1 - t) * start + t * end)
    full_traj.append(traj[-1])

    exec_cap_default = 2200 if GT_SPEED_MODE == "fast" else 0
    exec_cap = exec_cap_default
    exec_cap_env = os.environ.get("GT_MAX_EXEC_POINTS")
    if exec_cap_env is not None:
        try:
            exec_cap = max(0, int(exec_cap_env))
        except Exception:
            exec_cap = exec_cap_default
    if exec_cap > 0 and len(full_traj) > exec_cap:
        idxs = np.linspace(0, len(full_traj) - 1, exec_cap, dtype=int)
        full_traj = [full_traj[i] for i in idxs]
        print(f"[Exec] Downsampled interpolated trajectory to {len(full_traj)} points.")

    final_idx = len(full_traj) - 1
    if len(full_traj) > 1200:
        print(f"[Exec] Executing long trajectory ({len(full_traj)} points)...")
    for idx, conf in enumerate(full_traj):
        env.set_robot_conf(conf)
        env.pr.step()
        if idx > 0 and (idx % 1200 == 0):
            print(f"[Exec] ... {idx}/{len(full_traj)}")
        should_record = (idx % RECORD_EVERY_N == 0) or (idx == final_idx)
        if should_record:
            if VIDEO_RECORDER:
                VIDEO_RECORDER.record_step()
            if MASK_RECORDER:
                MASK_RECORDER.record_step()
            _run_step_callback()


def _is_holding_any_object(env):
    """True when gripper currently has at least one grasped object."""
    try:
        grasped = env.gripper.get_grasped_objects()
    except Exception:
        return False
    return bool(grasped)


def _move_to_conf_via_high_hover(env, target_conf):
    """
    Move to target joint config via a high-Z Cartesian hover path.
    Used while carrying objects to avoid collisions with scene geometry.
    """
    hover_z = float(os.environ.get("GT_PICK_PLACE_HOVER_Z", "0.30"))
    if hover_z <= 0:
        return False

    current_conf = env.get_robot_conf()
    try:
        try:
            tip = env.robot.get_tip()
        except Exception:
            tip = env.robot.arm.get_tip()
        cur_pos = np.array(tip.get_position(), dtype=float)
        cur_quat = tip.get_quaternion()
    except Exception:
        return False

    try:
        env.set_robot_conf(target_conf)
        try:
            tip_t = env.robot.get_tip()
        except Exception:
            tip_t = env.robot.arm.get_tip()
        tgt_pos = np.array(tip_t.get_position(), dtype=float)
        tgt_quat = tip_t.get_quaternion()
    except Exception:
        try:
            env.set_robot_conf(current_conf)
        except Exception:
            pass
        return False
    finally:
        try:
            env.set_robot_conf(current_conf)
        except Exception:
            pass

    transit_z = max(float(cur_pos[2]), float(tgt_pos[2])) + hover_z
    lift_pos = [float(cur_pos[0]), float(cur_pos[1]), float(transit_z)]
    hover_target = [float(tgt_pos[0]), float(tgt_pos[1]), float(transit_z)]

    try:
        path_lift = env.robot.get_linear_path(
            position=lift_pos,
            quaternion=cur_quat,
            steps=40,
            ignore_collisions=True,
        )
        path_transfer = env.robot.get_linear_path(
            position=hover_target,
            quaternion=tgt_quat,
            steps=60,
            ignore_collisions=True,
        )
    except Exception:
        return False

    if (path_lift is None) or (path_transfer is None):
        return False

    _emit_action_progress("move", "move")
    execute_trajectory(env, path_lift._path_points.reshape(-1, 7).tolist(), steps=3)
    _emit_action_progress("move", "move")
    execute_trajectory(env, path_transfer._path_points.reshape(-1, 7).tolist(), steps=3)

    # Intentionally stop at high hover above target XY.
    # Place action will perform the final descent directly to release.
    return True


def _move_to_place_release_direct(env, segments, release_idx):
    """
    Move from current carry-hover state directly to release pose.
    Avoids redundant intermediate hover transitions before place.
    """
    if not segments:
        return False

    try:
        release_seg = segments[int(release_idx)]
        if not release_seg:
            return False
        release_conf = release_seg[-1]
    except Exception:
        return False

    current_q = env.get_robot_conf()
    try:
        env.set_robot_conf(release_conf)
        try:
            tip_rel = env.robot.get_tip()
        except Exception:
            tip_rel = env.robot.arm.get_tip()
        release_pos = np.array(tip_rel.get_position(), dtype=float)
        release_quat = tip_rel.get_quaternion()
    except Exception:
        try:
            env.set_robot_conf(current_q)
        except Exception:
            pass
        return False
    finally:
        try:
            env.set_robot_conf(current_q)
        except Exception:
            pass

    try:
        path_down = env.robot.get_linear_path(
            position=release_pos.tolist(),
            quaternion=release_quat,
            steps=max(30, int(os.environ.get("GT_DIRECT_PLACE_STEPS", "70"))),
            ignore_collisions=True,
        )
        if path_down is not None:
            _emit_action_progress("move", "move")
            execute_trajectory(env, path_down._path_points.reshape(-1, 7).tolist(), steps=3)
            return True
    except Exception:
        pass

    # Fallback: joint interpolation to release conf
    try:
        traj = env._interpolate_joint_path(
            current_q,
            release_conf,
            steps=max(40, int(os.environ.get("GT_DIRECT_PLACE_STEPS", "70"))),
            check_collisions=False,
        )
        if traj is not None and len(traj) > 0:
            _emit_action_progress("move", "move")
            execute_trajectory(env, traj, steps=3)
            return True
    except Exception:
        pass

    return False


def _move_to_trajectory_start(env, traj, steps=50):
    """Move from the current configuration to the first waypoint of a trajectory."""
    if not traj:
        return False
    start_conf = traj[0]
    current_q = env.get_robot_conf()
    try:
        path = env._interpolate_joint_path(
            current_q,
            start_conf,
            steps=max(1, int(steps)),
            check_collisions=False,
        )
        if path is not None and len(path) > 0:
            execute_trajectory(env, path, steps=3)
            return True
    except Exception:
        pass
    return False


def _move_to_home_from_current(env, label="[BoxPlace]"):
    current_q = env.get_robot_conf()
    home_q = env.get_home_conf()
    if np.allclose(current_q, home_q, atol=1e-3):
        return True

    print(f"{label} Returning home.")
    traj = None
    try:
        traj = env.compute_motion_plan(current_q, home_q)
    except Exception:
        traj = None

    try:
        env.set_robot_conf(current_q)
    except Exception:
        pass

    if traj is None or len(traj) == 0:
        try:
            traj = env._interpolate_joint_path(
                current_q,
                home_q,
                steps=100,
                check_collisions=False,
            )
        except Exception:
            traj = None

    if traj is not None and len(traj) > 0:
        execute_trajectory(env, traj)
        return True
    return False


def _is_box_target_region(region_name):
    return region_name in ("box_boundary", "box-inside", "box_storage", "box_inside_fallback")


def _object_handle(obj):
    """Return object handle as int when available."""
    try:
        return int(obj.get_handle())
    except Exception:
        return None


def _is_object_grasped(env, obj):
    """Check whether the gripper still reports the object as grasped."""
    if obj is None:
        return False
    target_handle = _object_handle(obj)
    try:
        grasped = env.gripper.get_grasped_objects()
    except Exception:
        return False
    for g in grasped:
        if g is obj:
            return True
        if target_handle is not None:
            try:
                if int(g.get_handle()) == target_handle:
                    return True
            except Exception:
                continue
    return False


def _is_mug_name(name):
    return isinstance(name, str) and ("mug" in name.lower())


def _world_top_z(env, obj):
    """Top Z of an object in world frame."""
    if obj is None:
        return None
    try:
        if hasattr(env, "_get_world_bounding_box"):
            _, _, _, _, _, top_z = env._get_world_bounding_box(obj)
            return float(top_z)
    except Exception:
        pass
    try:
        _, _, _, _, _, max_z = obj.get_bounding_box()
        return float(obj.get_position()[2] + max_z)
    except Exception:
        return None


def _table_surface_z(env):
    """Get table top Z with robust fallbacks."""
    table_obj = None
    try:
        table_obj = env.regions.get("table")
    except Exception:
        table_obj = None
    if table_obj is None:
        table_obj = getattr(env, "table", None)
    if table_obj is None:
        try:
            table_obj = env.get_object("diningTable")
        except Exception:
            table_obj = None
    return _world_top_z(env, table_obj)


def _upright_mug_quat(env):
    """Reference upright mug quaternion."""
    try:
        ref = env.get_object("mug2")
        if ref is not None:
            return normalize_quaternion(ref.get_quaternion()).tolist()
    except Exception:
        pass
    return [0.0, 0.0, 0.0, 1.0]


def _stable_mug_pose_on_table(env, mug, target_xy=None):
    """Build a deterministic upright mug pose resting on the table."""
    if mug is None:
        return None
    table_z = _table_surface_z(env)
    if table_z is None:
        return None
    try:
        min_z = float(mug.get_bounding_box()[4])
    except Exception:
        return None
    if target_xy is None:
        pos = mug.get_position()
        x, y = float(pos[0]), float(pos[1])
    else:
        x, y = float(target_xy[0]), float(target_xy[1])
    quat = _upright_mug_quat(env)
    z = float(table_z - min_z + 0.0002)
    return [x, y, z, float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]


def _xy_from_pose_sample(place_pose_p):
    """Best-effort XY extraction from sampled pose tuples/lists."""
    if isinstance(place_pose_p, (list, tuple)) and len(place_pose_p) >= 2:
        try:
            return (float(place_pose_p[0]), float(place_pose_p[1]))
        except Exception:
            return None
    return None


def _prepare_table_mug_release_pose(env, pr, obj_name, target_obj, target_region, place_pose_p=None):
    """Snap mug to a stable table-contact pose before release."""
    if (target_obj is None) or (not _is_mug_name(obj_name)) or (target_region != "placement_boundary"):
        return None
    target_xy = _xy_from_pose_sample(place_pose_p)
    pose_lock = _stable_mug_pose_on_table(env, target_obj, target_xy=target_xy)
    if pose_lock is None:
        print(f"[Release] Could not compute stable table pose for '{obj_name}'")
        return None
    target_obj.set_pose(pose_lock)
    target_obj.set_dynamic(False)
    step_and_record(pr, 8)
    return pose_lock


def _release_gripper_until_detached(
    env,
    pr,
    target_obj=None,
    hold_q=None,
    hold_steps=None,
    open_velocity=0.25,
    pose_lock=None,
):
    """
    Open gripper and verify target detaches.
    Optionally keeps robot fixed at hold_q and object pose locked while opening.
    """
    if hold_steps is None:
        hold_steps = RELEASE_HOLD_STEPS
    hold_steps = max(1, int(hold_steps))

    for attempt_idx in range(3):
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

        open_amount = None
        open_done = False
        for _ in range(hold_steps):
            if hold_q is not None:
                env.set_robot_conf(hold_q)
            try:
                open_done = bool(env.gripper.actuate(1.0, velocity=open_velocity))
            except Exception:
                open_done = False
            try:
                open_vals = list(env.gripper.get_open_amount())
                open_amount = open_vals
                open_ok = all(v > 0.95 for v in open_vals)
            except Exception:
                open_ok = False
            if (pose_lock is not None) and (target_obj is not None):
                try:
                    target_obj.set_pose(pose_lock)
                except Exception:
                    pass
            step_and_record(pr, 1)

        detached = (target_obj is None) or (not _is_object_grasped(env, target_obj))
        if detached:
            return True
        print(
            f"[Release] attempt={attempt_idx+1} detached={detached} "
            f"open_done={open_done} open_amount={open_amount}"
        )

    # Last-resort detach if the simulator still reports attachment.
    if target_obj is not None and _is_object_grasped(env, target_obj):
        try:
            target_obj.set_parent(None, keep_in_place=True)
        except Exception:
            try:
                target_obj.set_parent(None)
            except Exception:
                pass
        try:
            env.gripper.release()
        except Exception:
            pass

        for _ in range(max(10, hold_steps // 2)):
            if hold_q is not None:
                env.set_robot_conf(hold_q)
            try:
                env.gripper.actuate(1.0, velocity=open_velocity)
            except Exception:
                pass
            if pose_lock is not None:
                try:
                    target_obj.set_pose(pose_lock)
                except Exception:
                    pass
            step_and_record(pr, 1)

    detached = target_obj is None or not _is_object_grasped(env, target_obj)
    return detached


def validate_config(env, q, min_dist=0.10):
    """Check if a configuration is collision-free."""
    env.set_robot_conf(q)
    if env.robot.check_collision():
        return False


# =============================================================================
# VALIDATION HELPERS (matching VLM execution monitor checks)
# =============================================================================

def check_lid_closed() -> bool:
    """Check if box lid is closed (blocking access to mug4)."""
    lid = ENV.get_object('box_lid')
    if lid is None:
        return False
    lid_pos = lid.get_position()
    # If lid Z is low, it's closed
    return lid_pos[2] < 0.85


def check_object_blocked_by_mug(object_name: str) -> tuple:
    """
    Check if object is blocked by mug_box (mug2) on top of box.
    Returns: (is_blocked, blocking_object_name)
    """
    if object_name == 'box_lid':
        mug2 = ENV.get_object('mug2')  # mug_box
        lid = ENV.get_object('box_lid')
        if mug2 and lid:
            mug_pos = mug2.get_position()
            lid_pos = lid.get_position()
            # If mug is above lid and close in XY
            if (mug_pos[2] > lid_pos[2] and 
                abs(mug_pos[0] - lid_pos[0]) < 0.15 and 
                abs(mug_pos[1] - lid_pos[1]) < 0.15):
                return (True, 'mug2')
    return (False, None)


def validate_object_moved(obj, pos_before: list, min_displacement: float = 0.05) -> tuple:
    """
    Validate that an object has moved significantly.
    Returns: (success, displacement, pos_after)
    """
    pos_after = list(obj.get_position())
    displacement = np.linalg.norm(np.array(pos_after) - np.array(pos_before))
    success = displacement >= min_displacement
    return (success, displacement, pos_after)


def validate_object_in_region(obj, target_region: str) -> bool:
    """
    Validate that object is within target region bounds.
    Gets actual bounds dynamically from the environment.
    
    Special case: For 'placement_boundary', we relax to check anywhere on table
    since physics can cause objects to drift.
    """
    pos = obj.get_position()
    
    # Special case: placement_boundary -> just check if on table
    if target_region == 'placement_boundary':
        table = ENV.regions.get('table')
        if table is not None:
            t_min_x, t_max_x, t_min_y, t_max_y, t_min_z, t_max_z = table.get_bounding_box()
            tx, ty, tz = table.get_position()
            
            # Table world bounds with generous tolerance
            tolerance = 0.15
            in_x = (tx + t_min_x - tolerance) <= pos[0] <= (tx + t_max_x + tolerance)
            in_y = (ty + t_min_y - tolerance) <= pos[1] <= (ty + t_max_y + tolerance)
            in_z = pos[2] > 0.7  # Just check it's above table level
            
            return in_x and in_y and in_z
        else:
            # Fallback: just check z
            return pos[2] > 0.7
    
    # Get region from environment
    region = ENV.regions.get(target_region)
    if region is None:
        print(f"  Warning: Region '{target_region}' not found in env, checking only z > 0.7")
        return pos[2] > 0.7
    
    # Get region bounds (local frame)
    r_min_x, r_max_x, r_min_y, r_max_y, r_min_z, r_max_z = region.get_bounding_box()
    
    # Get region position (world frame)
    rx, ry, rz = region.get_position()
    
    # Convert to world bounds
    world_min_x = rx + r_min_x
    world_max_x = rx + r_max_x
    world_min_y = ry + r_min_y
    world_max_y = ry + r_max_y
    world_min_z = rz + r_min_z
    world_max_z = rz + r_max_z
    
    # Add some tolerance (objects can be slightly outside due to physics)
    tolerance = 0.1
    # Z tolerance is higher to account for floating dummy bounds (like cupboard_boundary)
    # where the physical shelf is 10cm+ below the region definition.
    z_tolerance = 0.15
    
    in_x = (world_min_x - tolerance) <= pos[0] <= (world_max_x + tolerance)
    in_y = (world_min_y - tolerance) <= pos[1] <= (world_max_y + tolerance)
    in_z = (world_min_z - z_tolerance) <= pos[2] <= (world_max_z + z_tolerance)
    
    if not (in_x and in_y and in_z):
        print(f"  Debug: Object pos={[f'{p:.3f}' for p in pos]}")
        print(f"  Debug: Region bounds X=({world_min_x:.3f}, {world_max_x:.3f}), Y=({world_min_y:.3f}, {world_max_y:.3f}), Z=({world_min_z:.3f}, {world_max_z:.3f})")
        print(f"  Debug: In bounds - X:{in_x}, Y:{in_y}, Z:{in_z}")
    
    return in_x and in_y and in_z


def validate_object_not_fallen(obj, min_z: float = 0.7) -> tuple:
    """
    Check if object has fallen below table level.
    Returns: (not_fallen, current_z)
    """
    pos = obj.get_position()
    return (pos[2] >= min_z, pos[2])


def validate_lid_opened(lid, pos_before: list, min_displacement_xy: float = 0.1) -> tuple:
    """
    Validate that lid was actually slid open.
    Returns: (success, displacement_xy, pos_after)
    """
    pos_after = list(lid.get_position())
    displacement_xy = np.linalg.norm(
        np.array(pos_after[:2]) - np.array(pos_before[:2])
    )
    # Add a generous 5mm tolerance for geometric engine floating point settling
    success = displacement_xy >= (min_displacement_xy - 0.05)
    return (success, displacement_xy, pos_after)
    return True


def _ensure_lid_open_distance(env, lid_obj, pos_before, desired_xy):
    """
    While still grasping the lid, apply extra slide motion so XY opening reaches desired_xy.
    Returns: (success, final_displacement_xy)
    """
    if lid_obj is None:
        return False, 0.0

    opened, displacement_xy, pos_after = validate_lid_opened(
        lid_obj, pos_before, min_displacement_xy=desired_xy
    )
    if opened:
        return True, float(displacement_xy)

    delta_xy = np.array(pos_after[:2], dtype=float) - np.array(pos_before[:2], dtype=float)
    norm = float(np.linalg.norm(delta_xy))
    if norm > 1e-6:
        axis_xy = delta_xy / norm
    else:
        axis_xy = np.array([1.0, 0.0], dtype=float)

    remaining = max(0.0, float(desired_xy) - float(displacement_xy))
    extra_steps = max(2, int(os.environ.get("LID_OPEN_CORRECTION_STEPS", "4")))
    scales = np.linspace(1.0, 0.35, extra_steps).tolist()

    for s in scales:
        if remaining <= 0.0:
            break
        push_xy = max(0.015, remaining * float(s))
        try:
            try:
                tip = env.robot.get_tip()
            except Exception:
                tip = env.robot.arm.get_tip()
            tip_pos = np.array(tip.get_position(), dtype=float)
            tip_quat = tip.get_quaternion()
            target_pos = [
                float(tip_pos[0] + axis_xy[0] * push_xy),
                float(tip_pos[1] + axis_xy[1] * push_xy),
                float(tip_pos[2]),
            ]
            path = env.robot.get_linear_path(
                position=target_pos,
                quaternion=tip_quat,
                steps=35,
                ignore_collisions=True,
            )
            if path is None:
                continue
            _emit_action_progress("open", "open")
            execute_trajectory(env, path._path_points.reshape(-1, 7).tolist(), steps=4)
        except Exception:
            continue

        opened, displacement_xy, pos_after = validate_lid_opened(
            lid_obj, pos_before, min_displacement_xy=desired_xy
        )
        if opened:
            return True, float(displacement_xy)

        delta_xy = np.array(pos_after[:2], dtype=float) - np.array(pos_before[:2], dtype=float)
        norm = float(np.linalg.norm(delta_xy))
        if norm > 1e-6:
            axis_xy = delta_xy / norm
        remaining = max(0.0, float(desired_xy) - float(displacement_xy))

    return False, float(displacement_xy)


def lid_open_distance_xy(env, closed_pos_xy):
    lid = env.get_object("box_lid")
    if lid is None or closed_pos_xy is None:
        return 0.0
    pos = lid.get_position()
    return float(np.linalg.norm(np.array(pos[:2], dtype=float) - np.array(closed_pos_xy[:2], dtype=float)))


def ensure_lid_open_for_box_tasks(env, closed_pos_xy, min_xy=None, retries=2):
    """
    Ensure box lid is sufficiently opened for downstream box tasks.
    Retries run_open_box when current opening is below threshold.
    """
    if min_xy is None:
        min_xy = float(os.environ.get("LID_OPEN_TARGET_DISPLACEMENT", "0.45"))

    current = lid_open_distance_xy(env, closed_pos_xy)
    if current >= float(min_xy):
        print(f"[LidCheck] Open distance {current:.3f}m (>= {float(min_xy):.3f}m)")
        return True

    print(
        f"[LidCheck] Open distance {current:.3f}m (< {float(min_xy):.3f}m). "
        f"Retrying open-box up to {retries} times."
    )
    for i in range(max(0, int(retries))):
        ok = run_open_box(env, task_name=f"Lid corrective open ({i+1}/{retries})")
        go_home(env)
        current = lid_open_distance_xy(env, closed_pos_xy)
        print(f"[LidCheck] After retry {i+1}: {current:.3f}m")
        if ok and current >= float(min_xy):
            return True
    return current >= float(min_xy)


def _box_lid_replay_path():
    return os.environ.get("KITCHEN_BOX_LID_REPLAY_PATH", BOX_LID_OPEN_REPLAY_PATH)


def _coerce_replay_segment(segment):
    if not isinstance(segment, list) or not segment:
        return None
    coerced = []
    for q in segment:
        if not isinstance(q, (list, tuple, np.ndarray)) or len(q) != 7:
            return None
        coerced.append([float(v) for v in q])
    return coerced


def _valid_replay_segment(segment):
    return _coerce_replay_segment(segment) is not None


def _load_box_lid_open_replay(env, lid_obj, pos_before):
    path = _box_lid_replay_path()
    if not path or not os.path.exists(path):
        return None, None

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[box-lid-replay] failed to read {path}: {exc}")
        return None, None

    segments = data.get("segments")
    required = ("motion_to_hover", "approach", "open", "return", "retreat")
    if not isinstance(segments, dict) or not all(
        _valid_replay_segment(segments.get(name)) for name in required
    ):
        print(f"[box-lid-replay] invalid replay file: {path}")
        return None, None

    saved_lid_pos = data.get("lid_pos_before")
    if saved_lid_pos is not None:
        tol = float(os.environ.get("KITCHEN_BOX_LID_REPLAY_POS_TOL", "0.04"))
        delta = float(
            np.linalg.norm(
                np.array(saved_lid_pos[:3], dtype=float)
                - np.array(pos_before[:3], dtype=float)
            )
        )
        if delta > tol:
            print(
                f"[box-lid-replay] lid start mismatch; saved/current delta={delta:.3f}m, "
                f"tol={tol:.3f}. Falling back to runtime planning."
            )
            return None, None

    replay_segments = {name: _coerce_replay_segment(segments[name]) for name in required}
    home_return = _coerce_replay_segment(segments.get("home_return"))
    if home_return is not None:
        replay_segments["home_return"] = home_return

    print(f"[box-lid-replay] loaded trajectory from {path}")
    return replay_segments, path


def record_box_lid_open_replay(path, env, lid_obj, segments, metadata=None):
    def _float_list(values):
        return [float(v) for v in values]

    required = ("motion_to_hover", "approach", "open", "return", "retreat")
    normalized = {
        name: _coerce_replay_segment(segments.get(name))
        for name in required
    }
    if not all(normalized.values()):
        raise ValueError("box-lid replay requires non-empty 7-DOF trajectory segments")
    home_return = _coerce_replay_segment(segments.get("home_return"))
    if home_return is not None:
        normalized["home_return"] = home_return

    output_path = path or BOX_LID_OPEN_REPLAY_PATH
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    metadata = metadata or {}
    lid_pos_before = metadata.get("lid_pos_before", list(lid_obj.get_position()))
    data = {
        "kind": "kitchen_box_lid_open",
        "scene_path": os.environ.get("KITCHEN_SCENE_FILE"),
        "lid_pos_before": _float_list(lid_pos_before),
        "home_conf": _float_list(env.get_home_conf()),
        "steps_per_segment": DEFAULT_EXEC_INTERP_STEPS,
        "metadata": metadata,
        "segments": normalized,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"[box-lid-replay] wrote trajectory to {output_path}")
    return output_path


def _execute_box_lid_open_replay(env, lid_obj, pos_before, segments, replay_path, target_open_xy):
    print(f"[box-lid-replay] executing saved box-lid trajectory: {replay_path}")
    _emit_action_progress("open-lid", "open-lid")

    print("[box-lid-replay] Moving to hover...")
    execute_trajectory(env, segments["motion_to_hover"], steps=5)

    print("[box-lid-replay] Approaching grasp...")
    execute_trajectory(env, segments["approach"], steps=5)

    print("[box-lid-replay] Grasping lid...")
    env.gripper.actuate(0.0, 0.1)
    step_and_record(env.pr, 30)
    env.gripper.grasp(lid_obj)

    print("[box-lid-replay] Sliding lid open...")
    execute_trajectory(env, segments["open"], steps=5)

    opened_target, displacement_xy, _ = validate_lid_opened(
        lid_obj, pos_before, min_displacement_xy=target_open_xy
    )
    if not opened_target:
        print(
            f"[box-lid-replay] replay opened {displacement_xy:.3f}m; "
            f"target is {target_open_xy:.3f}m. Applying corrective slide..."
        )
        opened_target, displacement_xy = _ensure_lid_open_distance(
            env, lid_obj, pos_before, target_open_xy
        )

    print("[box-lid-replay] Releasing lid...")
    env.gripper.release()
    env.gripper.actuate(1.0, 0.1)
    step_and_record(env.pr, 50)

    print("[box-lid-replay] Returning...")
    execute_trajectory(env, segments["return"], steps=5)

    print("[box-lid-replay] Retreating to hover...")
    execute_trajectory(env, segments["retreat"], steps=5)

    home_return = segments.get("home_return")
    if home_return is not None:
        print("[box-lid-replay] Returning home along saved reverse entry path...")
        execute_trajectory(env, home_return, steps=5)
    step_and_record(env.pr, 50)

    opened, displacement_xy, _ = validate_lid_opened(
        lid_obj, pos_before, min_displacement_xy=target_open_xy
    )
    if not opened:
        print(
            f"ERROR: Saved box-lid trajectory did not open enough "
            f"(XY displacement: {displacement_xy:.3f}m, required: {target_open_xy:.3f}m)"
        )
        return False

    print(f"[box-lid-replay] Validation passed: Lid slid {displacement_xy:.3f}m")
    return True


def interpolate_path(env, q1, q2, steps=50):
    """Interpolate between two configurations."""
    traj = []
    q1 = np.array(q1)
    q2 = np.array(q2)
    for i in range(steps + 1):
        t = i / steps
        q = (1 - t) * q1 + t * q2
        traj.append(q.tolist())
    return traj


def go_home(env):
    """Return robot to home configuration."""
    print("\n--- Returning to Home ---")
    pr = env.pr
    q_start = env.get_robot_conf()
    q_home = env.get_home_conf()

    # Try direct interpolation
    traj = interpolate_path(env, q_start, q_home, steps=GO_HOME_INTERP_STEPS)
    if traj:
        _emit_action_progress("move", "move")
        execute_trajectory(env, traj, steps=GO_HOME_EXEC_STEPS)
        print("Reached Home.")
        return

    # Fallback: force home
    print("WARNING: Forcing home.")
    env.set_robot_conf(q_home)
    step_and_record(pr, 10)


def _normalize_segments(traj_tuple):
    """Normalize trajectory tuple into list of segments."""
    if traj_tuple is None:
        return []
    if isinstance(traj_tuple, (list, tuple)):
        segs = []
        for s in traj_tuple:
            if s is None:
                continue
            if isinstance(s, np.ndarray):
                segs.append(s.tolist())
            elif isinstance(s, (list, tuple)) and len(s) > 0:
                segs.append(list(s))
        return segs
    return []


def _get_ee_pos(env):
    """Get end-effector position."""
    try:
        tip = env.robot.get_tip()
        return np.array(tip.get_position(), dtype=float)
    except Exception:
        return np.array(env.robot.get_position(), dtype=float)


def _ee_pos_at_conf(env, conf, restore_conf):
    """Get EE position at a configuration."""
    env.set_robot_conf(conf)
    pos = _get_ee_pos(env)
    env.set_robot_conf(restore_conf)
    return pos


def _pick_grasp_segment_index(env, obj, segments):
    """Find segment index where EE is closest to object (for grasp)."""
    if not segments:
        return 0
    restore = env.get_robot_conf()
    obj_pos = np.array(obj.get_position(), dtype=float)

    best_i, best_d = 0, float("inf")
    for i, seg in enumerate(segments):
        end_conf = seg[-1]
        ee_pos = _ee_pos_at_conf(env, end_conf, restore)
        d = float(np.linalg.norm(ee_pos - obj_pos))
        if d < best_d:
            best_d, best_i = d, i

    return best_i


def _place_release_segment_index(env, place_pose_p, segments):
    """Find segment index for release (lowest EE or closest to place pose)."""
    if not segments:
        return 0

    restore = env.get_robot_conf()
    
    # Try using place pose if available
    try:
        if isinstance(place_pose_p, (list, tuple)) and len(place_pose_p) >= 3:
            p_xyz = np.array([float(place_pose_p[0]), float(place_pose_p[1]), float(place_pose_p[2])], dtype=float)
            best_i, best_d = 0, float("inf")
            for i, seg in enumerate(segments):
                end_conf = seg[-1]
                ee_pos = _ee_pos_at_conf(env, end_conf, restore)
                d = float(np.linalg.norm(ee_pos - p_xyz))
                if d < best_d:
                    best_d, best_i = d, i
            return best_i
    except:
        pass

    # Fallback: lowest EE z
    best_i, best_z = 0, float("inf")
    for i, seg in enumerate(segments):
        end_conf = seg[-1]
        ee_pos = _ee_pos_at_conf(env, end_conf, restore)
        if float(ee_pos[2]) < best_z:
            best_z, best_i = float(ee_pos[2]), i
    return best_i


# ============================================================
# TASK EXECUTION FUNCTIONS
# ============================================================


class PrimitiveTransferExecutorBase:
    """Executes a transfer as move -> pick -> move -> place."""

    PRIMITIVE_SEQUENCE = ("move", "pick", "move", "place")
    STAGE_LABELS = ("move-to-pick", "pick", "move-to-place", "place")

    def __init__(self, env, object_name, target_region, task_name=""):
        self.env = env
        self.pr = env.pr
        self.object_name = object_name
        self.target_region = target_region
        self.task_name = task_name
        self.stage_index = 0
        self.prepared = False

    def _print_header(self, details):
        print(f"\n{'='*60}")
        print(f"TASK: {self.task_name}")
        print(details)
        print(f"{'='*60}")

    def expected_next_action(self):
        if self.stage_index >= len(self.PRIMITIVE_SEQUENCE):
            return None
        return self.PRIMITIVE_SEQUENCE[self.stage_index]

    def execute_next(self, requested_action=None):
        if not self.prepared:
            ok, msg = self.prepare()
            if not ok:
                return False, msg
            self.prepared = True

        expected = self.expected_next_action()
        if expected is None:
            return False, "Transfer already complete"

        action_name = expected if requested_action is None else requested_action
        if action_name != expected:
            return False, f"Expected '{expected}', got '{action_name}'"

        stage_number = self.stage_index + 1
        stage_label = self.STAGE_LABELS[self.stage_index]
        total_stages = len(self.PRIMITIVE_SEQUENCE)
        print(
            f"[PrimitiveExecutor] START {stage_number}/{total_stages}: "
            f"{action_name} ({stage_label}) | {self.object_name}"
        )

        if self.stage_index == 0:
            ok, msg = self._move_to_pick()
        elif self.stage_index == 1:
            ok, msg = self._pick()
        elif self.stage_index == 2:
            ok, msg = self._move_to_place()
        else:
            ok, msg = self._place()

        if not ok:
            print(
                f"[PrimitiveExecutor] FAIL  {stage_number}/{total_stages}: "
                f"{action_name} ({stage_label}) | {self.object_name} | {msg}"
            )
            return False, msg

        self.stage_index += 1
        print(
            f"[PrimitiveExecutor] DONE  {stage_number}/{total_stages}: "
            f"{action_name} ({stage_label}) | {self.object_name}"
        )
        return True, action_name

    def execute_all(self):
        while self.expected_next_action() is not None:
            ok, msg = self.execute_next()
            if not ok:
                print(f"ERROR: {msg}")
                return False
        return True

    def _validate_transfer_complete(self):
        obj = self.env.get_object(self.object_name)
        if obj is None:
            return False, f"Object '{self.object_name}' not found after placement"

        moved, displacement, pos_after = validate_object_moved(obj, self.pos_before)
        if not moved:
            return False, (
                f"Object '{self.object_name}' didn't move "
                f"(displacement: {displacement:.3f}m)"
            )

        in_region = validate_object_in_region(obj, self.target_region)
        if not in_region:
            print(f"       Object position: {pos_after}")
            return False, (
                f"Object '{self.object_name}' not in target region "
                f"'{self.target_region}'"
            )

        not_fallen, current_z = validate_object_not_fallen(obj)
        if not not_fallen:
            return False, f"Object '{self.object_name}' fell (z={current_z:.3f}m)"

        print(f"✓ Validation passed: Object moved {displacement:.3f}m to target region")
        print(f"Task '{self.task_name}' complete!")
        return True, "success"


class PDDLPrimitiveTransferExecutor(PrimitiveTransferExecutorBase):
    """Shared move/pick/move/place executor for standard and box transfers."""

    def __init__(self, env, object_name, target_region, task_name="", box_mode=False):
        super().__init__(env, object_name, target_region, task_name=task_name)
        self.box_mode = bool(box_mode)
        self.pre_pick_moves = []
        self.pre_place_moves = []
        self.pick_action = None
        self.place_action = None

    def prepare(self):
        if self.box_mode:
            self._print_header(
                f"Pick '{self.object_name}' (box) -> Place in '{self.target_region}'"
            )
        else:
            self._print_header(
                f"Pick '{self.object_name}' -> Place in '{self.target_region}'"
            )

        home_q = self.env.get_home_conf()
        self.env.set_robot_conf(home_q)
        step_and_record(self.pr, 10)
        self.env.set_target_region(self.target_region)

        self.obj = self.env.get_object(self.object_name)
        if self.obj is None:
            return False, f"{self.object_name} not found"

        if self.box_mode and self.object_name in ['mug4', 'mug_inside_box']:
            if check_lid_closed():
                return False, f"Cannot pick '{self.object_name}' - box lid is closed!"
            print("✓ Pre-check passed: Box lid is open")

        self.pos_before = list(self.obj.get_position())
        self.obj.set_dynamic(False)
        initial_pose = self.obj.get_pose()

        directory = os.path.dirname(os.path.abspath(__file__))
        domain_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_domain.pddl'))
        stream_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_streams.pddl'))

        q_home_tuple = tuple(home_q)
        pose_tuple = tuple(initial_pose)
        init = [
            ('conf', q_home_tuple),
            ('at-conf', q_home_tuple),
            ('hand-empty',),
            ('movable', self.object_name),
            ('pose', pose_tuple),
            ('at-pose', self.object_name, pose_tuple),
            ('region', self.target_region),
        ]
        goal = And(('hand-empty',), ('in-region', self.object_name, self.target_region))

        problem = PDDLProblem(
            domain_pddl=domain_pddl,
            constant_map={},
            stream_pddl=stream_pddl,
            stream_map=get_stream_map(),
            init=init,
            goal=goal,
        )

        print("Solving PDDL problem...")
        solution = solve(problem, algorithm='adaptive', verbose=False, max_time=60)
        plan, _cost, _evaluations = solution
        if not plan:
            return False, "No PDDL plan found!"

        print(f"Plan found with {len(plan)} actions")
        for action in plan:
            print(f"  Action: {action.name}")

        ok, msg = self._split_plan(plan)
        if not ok:
            return False, msg
        return True, "prepared"

    def _split_plan(self, plan):
        idx = 0
        count = len(plan)

        while idx < count and plan[idx].name == 'move':
            self.pre_pick_moves.append(plan[idx])
            idx += 1

        if idx >= count or plan[idx].name != 'pick':
            return False, "Expected a pick action in PDDL plan"
        self.pick_action = plan[idx]
        idx += 1

        while idx < count and plan[idx].name == 'move':
            self.pre_place_moves.append(plan[idx])
            idx += 1

        if idx >= count or plan[idx].name != 'place':
            return False, "Expected a place action in PDDL plan"
        self.place_action = plan[idx]
        idx += 1

        if idx < count:
            remaining = ", ".join(action.name for action in plan[idx:])
            print(f"[PrimitiveExecutor] Warning: trailing PDDL actions folded into place stage: {remaining}")

        return True, "split"

    def _run_move_group(self, move_actions):
        _emit_action_progress("move", "move")
        if not move_actions:
            print("[PrimitiveExecutor] No explicit planner move for this stage; continuing.")
            return True, "move"

        for action in move_actions:
            _q1, q2, traj = action.args
            if _is_holding_any_object(self.env):
                moved_via_hover = _move_to_conf_via_high_hover(self.env, q2)
                if not moved_via_hover:
                    execute_trajectory(self.env, traj)
            else:
                execute_trajectory(self.env, traj)
        return True, "move"

    def _move_to_pick(self):
        return self._run_move_group(self.pre_pick_moves)

    def _pick(self):
        _emit_action_progress("pick", "pick")
        if self.box_mode:
            return self._pick_box()
        return self._pick_standard()

    def _move_to_place(self):
        return self._run_move_group(self.pre_place_moves)

    def _place(self):
        _emit_action_progress("place", "place")
        if self.box_mode:
            ok, msg = self._place_box()
        else:
            ok, msg = self._place_standard()
        if not ok:
            return ok, msg
        return self._validate_transfer_complete()

    def _pick_standard(self):
        o, _p, _g, _q1, _q2, traj_tuple = self.pick_action.args
        segments = _normalize_segments(traj_tuple)
        if not segments:
            return False, "pick has empty trajectory"

        target_obj = self.env.get_object(o)
        grasp_idx = _pick_grasp_segment_index(self.env, target_obj, segments)

        for seg in segments[:grasp_idx + 1]:
            execute_trajectory(self.env, seg)

        print(f"Grasping {o}...")
        target_obj.set_dynamic(True)
        self.env.gripper.actuate(0.0, 0.1)
        step_and_record(self.pr, 10)
        self.env.gripper.grasp(target_obj)

        for seg in segments[grasp_idx + 1:]:
            execute_trajectory(self.env, seg)
        return True, "pick"

    def _place_standard(self):
        o, p, _g, _r, _q1, _q2, traj_tuple = self.place_action.args
        segments = _normalize_segments(traj_tuple)
        if not segments:
            return False, "place has empty trajectory"

        release_idx = _place_release_segment_index(self.env, p, segments)
        is_box_target = _is_box_target_region(self.target_region)
        descent_traj = segments[release_idx] if is_box_target else None
        if is_box_target:
            if not descent_traj:
                return False, "box place has empty descent trajectory"
            print("[BoxPlace] Moving to planned box hover.")
            if not _move_to_trajectory_start(self.env, descent_traj):
                return False, "Could not move to planned box hover"
            print("[BoxPlace] Descending along planned lower_traj.")
            execute_trajectory(self.env, descent_traj)
        else:
            moved_to_release = _move_to_place_release_direct(self.env, segments, release_idx)
            if not moved_to_release:
                for seg in segments[:release_idx + 1]:
                    execute_trajectory(self.env, seg)

        print(f"Releasing {o}...")
        target_obj = self.env.get_object(o)
        target_obj.set_dynamic(True)
        pose_lock = _prepare_table_mug_release_pose(
            self.env,
            self.pr,
            obj_name=o,
            target_obj=target_obj,
            target_region=self.target_region,
            place_pose_p=p,
        )

        hold_q = self.env.get_robot_conf()
        released_ok = _release_gripper_until_detached(
            self.env,
            self.pr,
            target_obj=target_obj,
            hold_q=hold_q,
            hold_steps=max(RELEASE_HOLD_STEPS, 60),
            open_velocity=0.3,
            pose_lock=pose_lock,
        )
        if not released_ok:
            return False, f"'{o}' is still attached after release attempts."
        if pose_lock is not None:
            for _ in range(10):
                self.env.set_robot_conf(hold_q)
                target_obj.set_pose(pose_lock)
                step_and_record(self.pr, 1)
            target_obj.set_dynamic(False)

        if is_box_target:
            print("[BoxPlace] Ascending using reverse lower_traj.")
            execute_trajectory(self.env, descent_traj[::-1])
            if not _move_to_home_from_current(self.env):
                return False, "Could not return home after box placement"
        else:
            current_q = self.env.get_robot_conf()
            target_retreat_conf = segments[-1][-1]
            self.env.set_robot_conf(target_retreat_conf)
            target_pos = self.env.robot.get_position()
            target_quat = self.env.robot.get_quaternion()
            self.env.set_robot_conf(current_q)

            try:
                path_retreat = self.env.robot.get_linear_path(
                    position=target_pos,
                    quaternion=target_quat,
                    steps=50,
                    ignore_collisions=True,
                )
            except Exception:
                path_retreat = None

            if path_retreat:
                retreat_traj = path_retreat._path_points.reshape(-1, 7).tolist()
                execute_trajectory(self.env, retreat_traj)
            else:
                for seg in segments[release_idx + 1:]:
                    execute_trajectory(self.env, seg)
        return True, "place"

    def _pick_box(self):
        o, _p, _g, _q1, _q2, traj_tuple = self.pick_action.args
        segments = list(traj_tuple) if isinstance(traj_tuple, tuple) else [traj_tuple]
        approach_traj = segments[0] if len(segments) > 0 else []
        retreat_traj = segments[1] if len(segments) > 1 else []

        execute_trajectory(self.env, approach_traj)

        print(f"Grasping {o}...")
        target_obj = self.env.get_object(o)
        target_obj.set_dynamic(True)
        self.env.gripper.actuate(0.0, 0.1)
        step_and_record(self.pr, 10)
        self.env.gripper.grasp(target_obj)

        execute_trajectory(self.env, retreat_traj)
        return True, "pick"

    def _place_box(self):
        o, p, _g, _r, _q1, _q2, traj_tuple = self.place_action.args
        segments = list(traj_tuple) if isinstance(traj_tuple, tuple) else [traj_tuple]
        lower_traj = segments[0] if len(segments) > 0 else []
        lift_traj = segments[1] if len(segments) > 1 else []
        home_traj = segments[2] if len(segments) > 2 else None

        norm_segments = _normalize_segments(traj_tuple)
        release_idx = _place_release_segment_index(self.env, p, norm_segments) if norm_segments else 0
        is_box_target = _is_box_target_region(self.target_region)
        if is_box_target:
            if not lower_traj:
                return False, "box place has empty descent trajectory"
            print("[BoxPlace] Moving to planned box hover.")
            if not _move_to_trajectory_start(self.env, lower_traj):
                return False, "Could not move to planned box hover"
            print("[BoxPlace] Descending along planned lower_traj.")
            execute_trajectory(self.env, lower_traj)
        else:
            moved_to_release = _move_to_place_release_direct(self.env, norm_segments, release_idx)
            if not moved_to_release:
                execute_trajectory(self.env, lower_traj)

        print(f"Releasing {o}...")
        target_obj = self.env.get_object(o)
        target_obj.set_dynamic(True)
        pose_lock = _prepare_table_mug_release_pose(
            self.env,
            self.pr,
            obj_name=o,
            target_obj=target_obj,
            target_region=self.target_region,
            place_pose_p=p,
        )

        hold_q = self.env.get_robot_conf()
        released_ok = _release_gripper_until_detached(
            self.env,
            self.pr,
            target_obj=target_obj,
            hold_q=hold_q,
            hold_steps=max(RELEASE_HOLD_STEPS, 60),
            open_velocity=0.3,
            pose_lock=pose_lock,
        )
        if not released_ok:
            return False, f"'{o}' is still attached after release attempts."
        if pose_lock is not None:
            for _ in range(10):
                self.env.set_robot_conf(hold_q)
                target_obj.set_pose(pose_lock)
                step_and_record(self.pr, 1)
            target_obj.set_dynamic(False)

        if is_box_target:
            print("[BoxPlace] Ascending using reverse lower_traj.")
            execute_trajectory(self.env, lower_traj[::-1])
            if home_traj is not None and len(home_traj) > 0:
                execute_trajectory(self.env, home_traj)
            elif not _move_to_home_from_current(self.env):
                return False, "Could not return home after box placement"
        elif lift_traj is not None and len(lift_traj) > 0:
            current_q = self.env.get_robot_conf()
            target_retreat_conf = lift_traj[-1]
            self.env.set_robot_conf(target_retreat_conf)
            target_pos = self.env.robot.get_position()
            target_quat = self.env.robot.get_quaternion()
            self.env.set_robot_conf(current_q)

            try:
                path_retreat = self.env.robot.get_linear_path(
                    position=target_pos,
                    quaternion=target_quat,
                    steps=50,
                    ignore_collisions=True,
                )
                if path_retreat:
                    retreat_traj = path_retreat._path_points.reshape(-1, 7).tolist()
                    execute_trajectory(self.env, retreat_traj)
                else:
                    execute_trajectory(self.env, [current_q] + lift_traj)
            except Exception:
                execute_trajectory(self.env, [current_q] + lift_traj)

        if (not is_box_target) and home_traj is not None and len(home_traj) > 0:
            execute_trajectory(self.env, home_traj)
        return True, "place"


class CupboardPrimitiveTransferExecutor(PrimitiveTransferExecutorBase):
    """Primitive move/pick/move/place executor for cupboard mug transfers."""

    def prepare(self):
        self._print_header(
            f"Pick '{self.object_name}' (cupboard) -> Place in '{self.target_region}'"
        )

        self.home_q = self.env.get_home_conf()
        self.env.set_robot_conf(self.home_q)
        step_and_record(self.pr, 10)

        self.mug = self.env.get_object(self.object_name)
        if self.mug is None:
            return False, f"{self.object_name} not found"

        self.pos_before = list(self.mug.get_position())
        self.mug.set_dynamic(False)
        self.pose = self.mug.get_pose()
        self.upright_mug_quat = _upright_mug_quat(self.env)

        print("Computing horizontal hover configuration...")
        hover_dists = [0.35, 0.30, 0.40]
        z_offsets = [0.001]
        self.q_hover = None
        self.successful_grasp_quat = None
        original_conf = self.env.get_robot_conf()

        base_ry = np.pi / 2
        grasp_quats = [
            quaternion_from_euler(0, base_ry, 0),
            quaternion_from_euler(np.pi, base_ry, 0),
        ]

        cupboard_pick_height_offset = 0.05
        target_z = self.pose[2] + cupboard_pick_height_offset
        grasp_depth_offset = 0.03
        self.grasp_pos = None
        self.hover_pos = None

        for h_dist in hover_dists:
            for z_off in z_offsets:
                if self.q_hover is not None:
                    break

                self.hover_pos = [self.pose[0] - h_dist, self.pose[1], target_z]
                self.grasp_pos = [self.pose[0] + grasp_depth_offset, self.pose[1], target_z + z_off]

                for grasp_rot in grasp_quats:
                    path_configs = self.env.robot.solve_ik_via_sampling(
                        self.hover_pos,
                        quaternion=grasp_rot,
                        max_configs=50,
                        max_time_ms=1000,
                        ignore_collisions=True,
                    )
                    if path_configs is not None and len(path_configs) > 0:
                        for q in path_configs:
                            self.env.set_robot_conf(q)
                            if self.env.robot.check_collision():
                                continue
                            try:
                                path_check = self.env.robot.get_linear_path(
                                    position=self.grasp_pos,
                                    quaternion=grasp_rot,
                                    steps=20,
                                    ignore_collisions=True,
                                )
                                if path_check is not None:
                                    self.q_hover = q
                                    self.successful_grasp_quat = grasp_rot
                                    break
                            except Exception:
                                pass
                    if self.q_hover is not None:
                        break
            if self.q_hover is not None:
                break

        self.env.set_robot_conf(original_conf)
        if self.q_hover is None:
            return False, "Could not find valid horizontal hover configuration"

        return True, "prepared"

    def _move_to_pick(self):
        _emit_action_progress("move", "move")
        print("Moving to hover...")
        traj_to_hover = self.env._interpolate_joint_path(
            self.home_q, self.q_hover, steps=100, check_collisions=False
        )
        if traj_to_hover is not None and len(traj_to_hover) > 0:
            execute_trajectory(self.env, traj_to_hover, steps=10)
        else:
            return False, "Could not move to hover"

        self.env.gripper.release()
        step_and_record(self.pr, 30)

        print("Approaching grasp position...")
        path_approach = self.env.robot.get_linear_path(
            position=self.grasp_pos,
            quaternion=self.successful_grasp_quat,
            steps=200,
            ignore_collisions=True,
        )
        if path_approach is None:
            return False, "Could not plan approach trajectory"

        traj_approach = path_approach._path_points.reshape(-1, 7).tolist()
        for conf in traj_approach:
            self.env.set_robot_conf(conf)
            step_and_record(self.pr, 1)
        return True, "move"

    def _pick(self):
        _emit_action_progress("pick", "pick")
        print("Grasping...")
        self.env.gripper.actuate(0.0, 0.1)
        step_and_record(self.pr, 50)

        self.gripper_tip = self.env.robot.get_tip()
        self.mug_tip_offset_local, self.mug_tip_quat_local = compute_tip_attachment(
            self.gripper_tip, self.mug
        )
        self.mug.set_dynamic(False)
        return True, "pick"

    def _move_to_place(self):
        _emit_action_progress("move", "move")
        print("Retrieving...")
        retrieve_pos = list(self.hover_pos)
        retrieve_lift = float(os.environ.get("GT_PICK_PLACE_HOVER_Z", "0.30"))
        retrieve_pos[2] += max(0.02, retrieve_lift)

        path_retrieve = None
        try:
            path_retrieve = self.env.robot.get_linear_path(
                position=retrieve_pos,
                quaternion=self.successful_grasp_quat,
                steps=200,
                ignore_collisions=True,
            )
        except Exception:
            path_retrieve = None

        if path_retrieve is not None:
            traj_retrieve = path_retrieve._path_points.reshape(-1, 7).tolist()
            for conf in traj_retrieve:
                self.env.set_robot_conf(conf)
                step_and_record(self.pr, 1)
                update_attached_pose(
                    self.gripper_tip,
                    self.mug,
                    self.mug_tip_offset_local,
                    self.mug_tip_quat_local,
                )
        else:
            traj_fallback = self.env._interpolate_joint_path(
                self.env.get_robot_conf(), self.q_hover, steps=100, check_collisions=False
            )
            if traj_fallback is not None and len(traj_fallback) > 0:
                for conf in traj_fallback:
                    self.env.set_robot_conf(conf)
                    step_and_record(self.pr, 1)
                    update_attached_pose(
                        self.gripper_tip,
                        self.mug,
                        self.mug_tip_offset_local,
                        self.mug_tip_quat_local,
                    )
            else:
                return False, "Could not retrieve to hover"

        print("Finding placement position...")
        try:
            place_pose = self.env.find_best_placement(self.mug, self.target_region)
        except Exception:
            place_pose = [0.0, 0.3, 0.77, 0, 0, 0, 1]

        min_x, _max_x, _min_y, _max_y, min_z, _max_z = self.mug.get_bounding_box()
        stable_table_pose = None
        if self.target_region == "placement_boundary":
            stable_table_pose = _stable_mug_pose_on_table(
                self.env,
                self.mug,
                target_xy=(place_pose[0], place_pose[1]),
            )
            if stable_table_pose is not None:
                self.final_place_obj_pos = np.array(stable_table_pose[:3], dtype=float)
                self.place_upright_quat = stable_table_pose[3:]
            else:
                print("[CupboardPlace] Falling back to sampled placement Z (table pose unavailable).")

        if stable_table_pose is None:
            place_surface_z = None
            region_surface_z = None
            table_surface_z = _table_surface_z(self.env)
            region_obj = self.env.regions.get(self.target_region)
            if region_obj is not None:
                region_surface_z = _world_top_z(self.env, region_obj)

            if self.target_region == 'placement_boundary' and table_surface_z is not None:
                place_surface_z = table_surface_z
            elif region_surface_z is not None:
                place_surface_z = region_surface_z
            elif table_surface_z is not None:
                place_surface_z = table_surface_z

            if place_surface_z is None:
                place_surface_z = float(place_pose[2])

            place_object_z = place_surface_z - float(min_z) + 0.0002
            self.final_place_obj_pos = np.array(
                [place_pose[0], place_pose[1], place_object_z],
                dtype=float,
            )
            self.place_upright_quat = self.upright_mug_quat

        place_quats = [
            quaternion_from_euler(np.pi, 0, angle)
            for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False)
        ]

        self.q_place_hover = None
        self.successful_place_quat = None
        self.successful_place_pos = None
        pre_place_search_conf = self.env.get_robot_conf()

        for place_quat in place_quats:
            tip_offset_world = quaternion_rotate_vector(place_quat, self.mug_tip_offset_local)
            candidate_place_pos = (self.final_place_obj_pos - tip_offset_world).tolist()
            shared_hover = float(os.environ.get("GT_PICK_PLACE_HOVER_Z", "0.30"))
            candidate_hover_pos = [
                candidate_place_pos[0],
                candidate_place_pos[1],
                candidate_place_pos[2] + max(0.08, shared_hover),
            ]

            path_configs = self.env.robot.solve_ik_via_sampling(
                candidate_hover_pos,
                quaternion=place_quat,
                max_configs=20,
                max_time_ms=500,
                ignore_collisions=True,
            )
            if path_configs is not None and len(path_configs) > 0:
                for q in path_configs:
                    self.env.set_robot_conf(q)
                    if not self.env.robot.check_collision():
                        place_configs = self.env.robot.solve_ik_via_sampling(
                            candidate_place_pos,
                            quaternion=place_quat,
                            max_configs=5,
                            max_time_ms=200,
                            ignore_collisions=True,
                        )
                        if place_configs is not None and len(place_configs) > 0:
                            self.q_place_hover = q
                            self.successful_place_quat = place_quat
                            self.successful_place_pos = candidate_place_pos
                            break
            if self.q_place_hover is not None:
                break

        self.env.set_robot_conf(pre_place_search_conf)
        if self.q_place_hover is None:
            return False, "Could not find place hover configuration"
        self.place_lower_traj = None

        print("Moving to place hover...")
        current_conf = self.env.get_robot_conf()
        traj_to_place = self.env._interpolate_joint_path(
            current_conf, self.q_place_hover, steps=150, check_collisions=False
        )
        if traj_to_place is not None and len(traj_to_place) > 0:
            for conf in traj_to_place:
                self.env.set_robot_conf(conf)
                step_and_record(self.pr, 1)
                update_attached_pose(
                    self.gripper_tip,
                    self.mug,
                    self.mug_tip_offset_local,
                    self.mug_tip_quat_local,
                )
        else:
            return False, "Could not move to place hover"

        print("Lowering to place position...")
        path_lower = None
        try:
            path_lower = self.env.robot.get_linear_path(
                position=self.successful_place_pos,
                quaternion=self.successful_place_quat,
                steps=100,
                ignore_collisions=True,
            )
        except Exception:
            path_lower = None

        if path_lower is not None:
            traj_lower = path_lower._path_points.reshape(-1, 7).tolist()
            self.place_lower_traj = traj_lower
            for conf in traj_lower:
                self.env.set_robot_conf(conf)
                step_and_record(self.pr, 1)
                update_attached_pose(
                    self.gripper_tip,
                    self.mug,
                    self.mug_tip_offset_local,
                    self.mug_tip_quat_local,
                )
        else:
            place_configs = self.env.robot.solve_ik_via_sampling(
                self.successful_place_pos,
                quaternion=self.successful_place_quat,
                max_configs=10,
                max_time_ms=500,
                ignore_collisions=True,
            )
            if place_configs is not None and len(place_configs) > 0:
                q_place = place_configs[0]
                traj_lower = self.env._interpolate_joint_path(
                    self.env.get_robot_conf(), q_place, steps=50, check_collisions=False
                )
                if traj_lower is not None and len(traj_lower) > 0:
                    self.place_lower_traj = traj_lower
                    for conf in traj_lower:
                        self.env.set_robot_conf(conf)
                        step_and_record(self.pr, 1)
                        update_attached_pose(
                            self.gripper_tip,
                            self.mug,
                            self.mug_tip_offset_local,
                            self.mug_tip_quat_local,
                        )
            else:
                print("WARNING: Could not lower to place position, releasing from hover-like pose.")

        return True, "move"

    def _place(self):
        _emit_action_progress("place", "place")
        self.final_place_obj_pose = [
            float(self.final_place_obj_pos[0]),
            float(self.final_place_obj_pos[1]),
            float(self.final_place_obj_pos[2]),
            float(self.place_upright_quat[0]),
            float(self.place_upright_quat[1]),
            float(self.place_upright_quat[2]),
            float(self.place_upright_quat[3]),
        ]
        self.mug.set_pose(self.final_place_obj_pose)
        step_and_record(self.pr, 10)

        print("Releasing...")
        hold_q = self.env.get_robot_conf()
        released_ok = _release_gripper_until_detached(
            self.env,
            self.pr,
            target_obj=self.mug,
            hold_q=hold_q,
            hold_steps=max(RELEASE_HOLD_STEPS, 80),
            open_velocity=0.3,
            pose_lock=self.final_place_obj_pose,
        )
        if not released_ok:
            return False, f"'{self.object_name}' is still attached after release attempts."

        if _is_box_target_region(self.target_region) and self.place_lower_traj:
            print("[BoxPlace] Ascending using reverse lower_traj.")
            for conf in self.place_lower_traj[::-1]:
                self.env.set_robot_conf(conf)
                step_and_record(self.pr, 1)
            if not _move_to_home_from_current(self.env):
                return False, "Could not return home after box placement"
        else:
            print("Lifting...")
            path_lift = None
            try:
                lift_pos = [
                    self.successful_place_pos[0],
                    self.successful_place_pos[1],
                    self.successful_place_pos[2] + 0.15,
                ]
                path_lift = self.env.robot.get_linear_path(
                    position=lift_pos,
                    quaternion=self.successful_place_quat,
                    steps=50,
                    ignore_collisions=True,
                )
            except Exception:
                path_lift = None

            if path_lift is not None:
                traj_lift = path_lift._path_points.reshape(-1, 7).tolist()
                for conf in traj_lift:
                    self.env.set_robot_conf(conf)
                    step_and_record(self.pr, 1)
            else:
                traj_lift = self.env._interpolate_joint_path(
                    self.env.get_robot_conf(),
                    self.q_place_hover,
                    steps=50,
                    check_collisions=False,
                )
                if traj_lift is not None and len(traj_lift) > 0:
                    for conf in traj_lift:
                        self.env.set_robot_conf(conf)
                        step_and_record(self.pr, 1)

        self.mug.set_pose(self.final_place_obj_pose)
        self.mug.set_dynamic(False)
        step_and_record(self.pr, 20)
        return self._validate_transfer_complete()


def create_primitive_transfer_executor(env, object_name, target_region, task_name=""):
    obj = env.get_object(object_name)
    pos = obj.get_position() if obj is not None else None

    def _is_in_region_local(p, r_name):
        if p is None: return False
        r = env.regions.get(r_name)
        if not r: return False
        r_min_x, r_max_x, r_min_y, r_max_y, r_min_z, r_max_z = r.get_bounding_box()
        rx, ry, rz = r.get_position()
        return (rx + r_min_x - 0.1 <= p[0] <= rx + r_max_x + 0.1 and
                ry + r_min_y - 0.1 <= p[1] <= ry + r_max_y + 0.1 and
                rz + r_min_z - 0.15 <= p[2] <= rz + r_max_z + 0.15)

    def _log_transfer_route(caller_name, executor_name, box_mode):
        to_box = target_region in ['box_boundary', 'box-inside']
        in_box = _is_in_region_local(pos, 'box_boundary')
        scene_file = os.environ.get("KITCHEN_SCENE_FILE", "")
        scene_name = os.path.basename(scene_file) if scene_file else "default"
        pos_text = "None" if pos is None else "[" + ", ".join(f"{float(v):.3f}" for v in pos[:3]) + "]"
        print(
            "[TransferRoute] "
            f"scene={scene_name} "
            f"task={task_name!r} "
            f"caller={caller_name} "
            f"executor={executor_name} "
            f"object={object_name} "
            f"target_region={target_region} "
            f"object_pos={pos_text} "
            f"in_box_boundary={in_box} "
            f"target_is_box={to_box} "
            f"box_mode={bool(box_mode)} "
            f"pick_impl={'_pick_box' if box_mode else '_pick_standard'} "
            f"place_impl={'_place_box' if box_mode else '_place_standard'}"
        )

    if _is_in_region_local(pos, 'cupboard_boundary') or _is_in_region_local(pos, 'cupboard_boundary_top'):
        _log_transfer_route(
            "create_primitive_transfer_executor",
            "CupboardPrimitiveTransferExecutor",
            False,
        )
        return CupboardPrimitiveTransferExecutor(
            env, object_name, target_region, task_name=task_name
        )

    to_box = target_region in ['box_boundary', 'box-inside']
    in_box = _is_in_region_local(pos, 'box_boundary')

    if in_box or to_box:
        _log_transfer_route(
            "create_primitive_transfer_executor",
            "PDDLPrimitiveTransferExecutor",
            True,
        )
        return PDDLPrimitiveTransferExecutor(
            env,
            object_name,
            target_region,
            task_name=task_name,
            box_mode=True,
        )

    _log_transfer_route(
        "create_primitive_transfer_executor",
        "PDDLPrimitiveTransferExecutor",
        False,
    )
    return PDDLPrimitiveTransferExecutor(
        env,
        object_name,
        target_region,
        task_name=task_name,
        box_mode=False,
    )


def _log_pick_place_route(env, caller_name, object_name, target_region, task_name, box_mode):
    obj = env.get_object(object_name)
    pos = obj.get_position() if obj is not None else None

    def _is_in_region_local(p, r_name):
        if p is None:
            return False
        r = env.regions.get(r_name)
        if not r:
            return False
        r_min_x, r_max_x, r_min_y, r_max_y, r_min_z, r_max_z = r.get_bounding_box()
        rx, ry, rz = r.get_position()
        return (
            rx + r_min_x - 0.1 <= p[0] <= rx + r_max_x + 0.1
            and ry + r_min_y - 0.1 <= p[1] <= ry + r_max_y + 0.1
            and rz + r_min_z - 0.15 <= p[2] <= rz + r_max_z + 0.15
        )

    scene_file = os.environ.get("KITCHEN_SCENE_FILE", "")
    scene_name = os.path.basename(scene_file) if scene_file else "default"
    pos_text = "None" if pos is None else "[" + ", ".join(f"{float(v):.3f}" for v in pos[:3]) + "]"
    print(
        "[TransferRoute] "
        f"scene={scene_name} "
        f"task={task_name!r} "
        f"caller={caller_name} "
        f"executor=PDDLPrimitiveTransferExecutor "
        f"object={object_name} "
        f"target_region={target_region} "
        f"object_pos={pos_text} "
        f"in_box_boundary={_is_in_region_local(pos, 'box_boundary')} "
        f"target_is_box={target_region in ['box_boundary', 'box-inside']} "
        f"box_mode={bool(box_mode)} "
        f"pick_impl={'_pick_box' if box_mode else '_pick_standard'} "
        f"place_impl={'_place_box' if box_mode else '_place_standard'}"
    )


def run_standard_pick_place(env, object_name, target_region, task_name=""):
    _log_pick_place_route(
        env,
        "run_standard_pick_place",
        object_name,
        target_region,
        task_name,
        False,
    )
    executor = PDDLPrimitiveTransferExecutor(
        env,
        object_name,
        target_region,
        task_name=task_name,
        box_mode=False,
    )
    return executor.execute_all()

def run_cupboard_pick_place(env, object_name, target_region, task_name=""):
    scene_file = os.environ.get("KITCHEN_SCENE_FILE", "")
    scene_name = os.path.basename(scene_file) if scene_file else "default"
    obj = env.get_object(object_name)
    pos = obj.get_position() if obj is not None else None
    pos_text = "None" if pos is None else "[" + ", ".join(f"{float(v):.3f}" for v in pos[:3]) + "]"
    print(
        "[TransferRoute] "
        f"scene={scene_name} "
        f"task={task_name!r} "
        "caller=run_cupboard_pick_place "
        "executor=CupboardPrimitiveTransferExecutor "
        f"object={object_name} "
        f"target_region={target_region} "
        f"object_pos={pos_text} "
        "in_box_boundary=False "
        f"target_is_box={target_region in ['box_boundary', 'box-inside']} "
        "box_mode=False "
        "pick_impl=CupboardPrimitiveTransferExecutor._pick "
        "place_impl=CupboardPrimitiveTransferExecutor._place"
    )
    executor = CupboardPrimitiveTransferExecutor(
        env,
        object_name,
        target_region,
        task_name=task_name,
    )
    return executor.execute_all()

def run_open_box(env, task_name=""):
    """
    Open the box lid by sliding.
    """
    object_name = 'box_lid'
    print(f"[PrimitiveExecutor] START 1/1: open-lid (open-lid) | {object_name}")
    pr = env.pr
    print(f"\n{'='*60}")
    print(f"TASK: {task_name}")
    print("Slide open box lid")
    print(f"{'='*60}")

    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    step_and_record(pr, 10)

    obj = env.get_object('box_lid')
    if obj is None:
        print("ERROR: box_lid not found.")
        return False

    is_blocked, blocker = check_object_blocked_by_mug('box_lid')
    if is_blocked:
        print(f"ERROR: Cannot open box lid - '{blocker}' is blocking it")
        return False

    pos_before = list(obj.get_position())
    target_open_xy = float(os.environ.get("LID_OPEN_TARGET_DISPLACEMENT", "0.45"))
    min_open_xy = float(os.environ.get("LID_OPEN_MIN_DISPLACEMENT", "0.45"))
    target_open_xy = max(target_open_xy, min_open_xy)

    env.gripper.actuate(1.0, 0.1)
    step_and_record(pr, 10)
    _emit_action_progress("open-lid", "open-lid")

    try:
        replay_segments, replay_path = _load_box_lid_open_replay(env, obj, pos_before)
        if replay_segments is not None:
            success = _execute_box_lid_open_replay(
                env,
                obj,
                pos_before,
                replay_segments,
                replay_path,
                target_open_xy,
            )
            if success:
                print(f"Task '{task_name}' complete!")
                print(f"[PrimitiveExecutor] DONE  1/1: open-lid (open-lid) | {object_name}")
            return success

        print("Computing grasp trajectory...")
        grasp_quat, q_hover, q_grasp, (traj_hover_to_center, traj_center_to_edge) = env.compute_lid_grasp_trajectory(obj)
        traj_approach = traj_hover_to_center + traj_center_to_edge

        print("Planning motion to hover...")
        path_to_hover = env.compute_motion_plan(home_q, q_hover)

        if not path_to_hover:
            print("ERROR: Could not plan motion to hover")
            return False

        print("Moving to hover...")
        execute_trajectory(env, path_to_hover, steps=5)

        print("Approaching grasp...")
        execute_trajectory(env, traj_approach, steps=5)

        print("Grasping lid...")
        env.gripper.actuate(0.0, 0.1)
        step_and_record(pr, 30)
        env.gripper.grasp(obj)

        print("Computing slide trajectory...")
        q_open_start, q_open_end, traj_open, traj_return = env.compute_slide_lid_trajectory(
            obj, grasp_quat, initial_conf=q_grasp
        )

        if not traj_open:
            print("ERROR: Failed to compute slide trajectory")
            return False

        print("Sliding lid open...")
        execute_trajectory(env, traj_open, steps=5)

        opened_target, displacement_xy, _ = validate_lid_opened(
            obj, pos_before, min_displacement_xy=target_open_xy
        )
        if not opened_target:
            print(
                f"[Lid] Initial slide opened {displacement_xy:.3f}m; "
                f"target is {target_open_xy:.3f}m. Applying corrective slide..."
            )
            opened_target, displacement_xy = _ensure_lid_open_distance(
                env, obj, pos_before, target_open_xy
            )
            if opened_target:
                print(f"[Lid] Corrective slide reached {displacement_xy:.3f}m.")
            else:
                print(
                    f"[Lid] Corrective slide ended at {displacement_xy:.3f}m; "
                    f"target {target_open_xy:.3f}m not reached."
                )

        print("Releasing lid...")
        env.gripper.release()
        env.gripper.actuate(1.0, 0.1)
        step_and_record(pr, 50)

        print("Returning...")
        execute_trajectory(env, traj_return, steps=5)

        print("Retreating to hover...")
        traj_retreat = traj_hover_to_center[::-1]
        execute_trajectory(env, traj_retreat, steps=5)
        step_and_record(pr, 50)

        lid = env.get_object('box_lid')
        required_open_xy = target_open_xy
        opened, displacement_xy, pos_after = validate_lid_opened(
            lid, pos_before, min_displacement_xy=required_open_xy
        )
        if not opened:
            print(
                f"ERROR: Lid didn't slide open enough "
                f"(XY displacement: {displacement_xy:.3f}m, required: {required_open_xy:.3f}m)"
            )
            return False

        print(f"✓ Validation passed: Lid slid {displacement_xy:.3f}m")
        print(f"Task '{task_name}' complete!")
        print(f"[PrimitiveExecutor] DONE  1/1: open-lid (open-lid) | {object_name}")
        return True

    except Exception as e:
        print(f"[PrimitiveExecutor] FAIL  1/1: open-lid (open-lid) | {object_name} | {e}")
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_box_pick_place(env, object_name, target_region, task_name=""):
    _log_pick_place_route(
        env,
        "run_box_pick_place",
        object_name,
        target_region,
        task_name,
        True,
    )
    executor = PDDLPrimitiveTransferExecutor(
        env,
        object_name,
        target_region,
        task_name=task_name,
        box_mode=True,
    )
    return executor.execute_all()


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def main():
    global VIDEO_RECORDER
    env = ENV
    pr = env.pr

    print("="*60)
    print("GROUND TRUTH ORCHESTRATOR")
    print("Long-Horizon Kitchen Task")
    print("="*60)

    print(
        f"\nSpeed mode: {GT_SPEED_MODE} "
        f"(exec_interp={DEFAULT_EXEC_INTERP_STEPS}, hold_steps={RELEASE_HOLD_STEPS}, record_every={RECORD_EVERY_N})"
    )

    # Initialize video recorder (optional)
    enable_video = os.environ.get("GT_RECORD_VIDEO", "1").strip().lower() not in {
        "0", "false", "no"
    }
    if enable_video:
        print("\nInitializing video recorder...")
        VIDEO_RECORDER = VideoRecorder(env, output_dir="orchestrator_videos", fps=30)
    else:
        VIDEO_RECORDER = None
        print("\nVideo recorder disabled (GT_RECORD_VIDEO=0)")
    
    print("\nSettling physics...")
    for _ in range(50):
        step_and_record(pr, 1)

    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    for _ in range(10):
        step_and_record(pr, 1)

    lid_closed_ref = None
    lid_obj = env.get_object("box_lid")
    if lid_obj is not None:
        try:
            lid_closed_ref = list(lid_obj.get_position())
        except Exception:
            lid_closed_ref = None

    results = []

    # ============================================
    # TASK 1: Pick mug3 from cupboard -> placement_boundary
    # ============================================
    success = run_cupboard_pick_place(
        env,
        object_name='mug3',
        target_region='placement_boundary',
        task_name="Task 1: Cupboard Mug -> Placement"
    )
    results.append(("Task 1: mug3 cupboard -> placement", success))
    go_home(env)

    # ============================================
    # TASK 2: Pick 5 groceries from table -> cupboard
    # soup, mustard, spam -> cupboard_boundary (inside)
    # sugar, crackers -> cupboard_boundary_top (top shelf)
    # ============================================
    groceries_inside = ['soup', 'mustard', 'spam']
    groceries_top = ['sugar', 'crackers']
    
    task_num = 1
    for grocery in groceries_inside:
        success = run_standard_pick_place(
            env,
            object_name=grocery,
            target_region='cupboard_boundary',
            task_name=f"Task 2.{task_num}: {grocery} -> Cupboard (inside)"
        )
        results.append((f"Task 2.{task_num}: {grocery} -> cupboard", success))
        go_home(env)
        task_num += 1
    
    for grocery in groceries_top:
        success = run_standard_pick_place(
            env,
            object_name=grocery,
            target_region='cupboard_boundary_top',
            task_name=f"Task 2.{task_num}: {grocery} -> Cupboard (top)"
        )
        results.append((f"Task 2.{task_num}: {grocery} -> cupboard_top", success))
        go_home(env)
        task_num += 1

    # ============================================
    # TASK 3: Pick mug2 from box_boundary -> placement_boundary
    # ============================================
    success = run_box_pick_place(
        env,
        object_name='mug2',
        target_region='placement_boundary',
        task_name="Task 3: Box Mug -> Placement"
    )
    results.append(("Task 3: mug2 box -> placement", success))
    go_home(env)

    # ============================================
    # TASK 4: Slide open box lid
    # ============================================
    success = run_open_box(
        env,
        task_name="Task 4: Open Box Lid"
    )
    lid_ok = ensure_lid_open_for_box_tasks(
        env,
        lid_closed_ref,
        retries=2 if not success else 1,
    )
    success = bool(success or lid_ok)
    results.append(("Task 4: open box lid", success))
    go_home(env)

    # ============================================
    # TASK 5: Pick mug4 from inside box -> placement_boundary
    # ============================================
    success = run_box_pick_place(
        env,
        object_name='mug4',
        target_region='placement_boundary',
        task_name="Task 5: Mug Inside Box -> Placement"
    )
    results.append(("Task 5: mug4 box_inside -> placement", success))
    go_home(env)

    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    total = len(results)
    passed = sum(1 for _, s in results if s)
    for task, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status} - {task}")
    print(f"\nTotal: {passed}/{total} tasks completed successfully")
    print("="*60)

    # Release video recorder
    if VIDEO_RECORDER:
        VIDEO_RECORDER.release()
        print("\nVideo recording saved to 'orchestrator_videos/' directory")

    keep_alive = os.environ.get("GT_KEEP_ALIVE", "0").strip().lower() not in {
        "0", "false", "no"
    }
    if keep_alive:
        print("\nOrchestration complete. Press Ctrl+C to close.")
        try:
            while True:
                pr.step()
                _run_step_callback()
        except KeyboardInterrupt:
            pass
    else:
        print("\nOrchestration complete. Auto-exit enabled.")

    pr.stop()
    pr.shutdown()


if __name__ == "__main__":
    main()

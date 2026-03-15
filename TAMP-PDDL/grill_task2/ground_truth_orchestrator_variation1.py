"""
Minimal startup script for Grill Task 2, Variation 1.

This file intentionally keeps only environment loading and closed-lid initialization.
"""

import math
import os
import sys
import numpy as np

from pyrep import PyRep
from pyrep.const import JointMode
from pyrep.backend import sim
from pyrep.objects.joint import Joint


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)

DEFAULT_SCENE_CLOSED = os.path.join(THIS_DIR, "GrillClosed_SteakIn_SteakChickenOut.ttt")
DEFAULT_SCENE_VARIATION = os.path.join(THIS_DIR, "grill.variation1.ttt")

ALLOW_SCENE_OVERRIDE = os.environ.get("GRILL_ALLOW_SCENE_OVERRIDE", "False") == "True"
SCENE_OVERRIDE = os.environ.get("GRILL_SCENE_FILE_OVERRIDE", "").strip()

if ALLOW_SCENE_OVERRIDE and SCENE_OVERRIDE:
    SCENE_PATH = SCENE_OVERRIDE
else:
    # Match the original variation-1 loader behavior: use variation scene by default.
    SCENE_PATH = DEFAULT_SCENE_VARIATION

if not os.path.exists(SCENE_PATH):
    raise FileNotFoundError(f"Scene file not found: {SCENE_PATH}")

os.environ["GRILL_SCENE_FILE"] = SCENE_PATH
os.environ["HEADLESS"] = "False"
os.environ.setdefault("GRILL_LID_TRAVEL_ANGLE", f"{math.radians(95.0):.6f}")
os.environ.setdefault("GRILL_LID_OPEN_ANGLE", f"{math.radians(95.0):.6f}")
os.environ.setdefault("GRILL_LID_AUTOCALIBRATE", "False")
os.environ.setdefault("GRILL_USE_SCENE_INITIAL_LID_ANGLE", "True")
os.environ.setdefault("GRILL_PRESERVE_SCENE_LID_POSE", "True")

if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def step(pr, n=1):
    for _ in range(max(1, int(n))):
        pr.step()


def _probe_scene_lid_angle(scene_path):
    """Read the initial lid joint angle directly from the .ttt scene."""
    probe = None
    try:
        probe = PyRep()
        probe.launch(scene_path, headless=True)
        j = Joint("lid_joint")
        try:
            return float(j.get_joint_position())
        except Exception:
            return float(sim.simGetJointPosition(int(j.get_handle())))
    except Exception:
        return None
    finally:
        if probe is not None:
            try:
                probe.stop()
            except Exception:
                pass
            try:
                probe.shutdown()
            except Exception:
                pass


def _resolve_lid_joint_handle(env):
    try:
        return int(Joint("lid_joint").get_handle())
    except Exception:
        pass

    lid_joint = getattr(env, "lid_joint", None)
    try:
        return int(lid_joint.get_handle())
    except Exception:
        return None


def _set_lid_servo_lock(env, lock=True):
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
        if getattr(env, "lid_joint", None) is not None and int(env.lid_joint.get_handle()) == int(h):
            j = env.lid_joint
        else:
            j = Joint(int(h))
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
            cur = float(os.environ.get("GRILL_LID_CLOSED_ANGLE", "0.0"))
        try:
            j.set_joint_target_position(cur)
        except Exception:
            pass
        try:
            j.set_joint_target_velocity(0.0)
        except Exception:
            pass
    return True


def _relax_joint_interval(h):
    prev = None
    try:
        cyc, interval = sim.simGetJointInterval(int(h))
        prev = (bool(cyc), list(interval) if interval is not None else None)
    except Exception:
        prev = None
    try:
        sim.simSetJointInterval(int(h), False, [-3.2, 6.4])
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


def _set_lid_joint_angle(env, pr, target_angle, steps=90):
    h = _resolve_lid_joint_handle(env)
    if h is None:
        return False

    _set_lid_servo_lock(env, True)
    prev_interval = _relax_joint_interval(h)

    try:
        cur = float(sim.simGetJointPosition(int(h)))
    except Exception:
        try:
            if getattr(env, "lid_joint", None) is not None:
                cur = float(env.lid_joint.get_joint_position())
            else:
                _restore_joint_interval(h, prev_interval)
                return False
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


def _force_lid_closed(env, pr, steps=100, angle=None):
    if angle is None:
        angle = float(os.environ.get("GRILL_LID_CLOSED_ANGLE", "0.0"))
    _set_lid_servo_lock(env, True)
    lid_handle = _resolve_lid_joint_handle(env)
    if lid_handle is None:
        step(pr, 20)
        return False

    prev_interval = _relax_joint_interval(lid_handle)

    for _ in range(max(1, int(steps))):
        try:
            sim.simSetJointPosition(int(lid_handle), float(angle))
        except Exception:
            pass
        try:
            sim.simSetJointTargetPosition(int(lid_handle), float(angle))
        except Exception:
            pass

        lid_joint = getattr(env, "lid_joint", None)
        if lid_joint is not None:
            try:
                lid_joint.set_joint_position(float(angle), disable_dynamics=True)
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

        step(pr, 1)

    _restore_joint_interval(lid_handle, prev_interval)
    return True


def set_closed_lid(env, pr, angle=None):
    if angle is None:
        angle = float(os.environ.get("GRILL_LID_CLOSED_ANGLE", "0.0"))

    preserve_scene_pose = os.environ.get("GRILL_PRESERVE_SCENE_LID_POSE", "False") == "True"
    if preserve_scene_pose:
        # In preserve mode, keep the exact authored scene pose and only hold it.
        _set_lid_servo_lock(env, True)
        step(pr, 8)
        return

    try:
        env.stabilize_startup_state(steps=15)
    except Exception:
        pass

    _set_lid_servo_lock(env, True)

    try:
        env.set_lid_collision_enabled(False)
    except Exception:
        pass

    _set_lid_joint_angle(env, pr, target_angle=float(angle), steps=90)
    _force_lid_closed(env, pr, steps=100, angle=float(angle))

    try:
        env.set_lid_collision_enabled(True)
    except Exception:
        pass

    step(pr, 3)


def main():
    use_scene_lid = os.environ.get("GRILL_USE_SCENE_INITIAL_LID_ANGLE", "True") == "True"
    preserve_scene_pose = os.environ.get("GRILL_PRESERVE_SCENE_LID_POSE", "False") == "True"
    scene_lid_angle = _probe_scene_lid_angle(SCENE_PATH) if use_scene_lid else None
    if scene_lid_angle is not None:
        os.environ["GRILL_LID_CLOSED_ANGLE"] = f"{scene_lid_angle:.6f}"
        os.environ["GRILL_LID_OPEN_ANGLE"] = f"{scene_lid_angle + math.radians(95.0):.6f}"
    else:
        os.environ.setdefault("GRILL_LID_CLOSED_ANGLE", "0.0")

    # Import after closed-angle env vars are finalized.
    from grill_task_streams import ENV  # noqa: WPS433,E402

    env = ENV
    pr = env.pr
    lid_closed_angle = float(os.environ.get("GRILL_LID_CLOSED_ANGLE", "0.0"))

    print("=" * 72)
    print("GRILL VARIATION 1 - ENV LOAD ONLY")
    print("=" * 72)
    print(f"Scene: {SCENE_PATH}")
    if (not ALLOW_SCENE_OVERRIDE) and SCENE_OVERRIDE:
        print(
            f"[startup] Ignoring GRILL_SCENE_FILE_OVERRIDE='{SCENE_OVERRIDE}' "
            "(set GRILL_ALLOW_SCENE_OVERRIDE=True to use it)."
        )
    if scene_lid_angle is not None:
        print(f"Closed lid target from scene: {lid_closed_angle:.3f} rad")
    else:
        print(f"Closed lid target (fallback): {lid_closed_angle:.3f} rad")
    print(f"Preserve scene lid pose: {preserve_scene_pose}")

    set_closed_lid(env, pr, angle=lid_closed_angle)
    print("Environment loaded. Grill lid held at closed position.")
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

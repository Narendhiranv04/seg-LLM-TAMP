# grill_task_env.py
# Environment for the grill task scene (grill_task.ttt)

import os
import numpy as np
import math
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import ConfigurationPathAlgorithms, JointMode
from pyrep.backend import sim

DEFAULT_SCENE_FILE = os.path.join(os.path.dirname(__file__), "grill.variation1.ttt")
SCENE_FILE = os.environ.get("GRILL_SCENE_FILE", DEFAULT_SCENE_FILE)
DEFAULT_TUNED_GRILL_HOME_CONF = [
    -0.0617,
    -0.1561,
    -0.7635,
    -1.5610,
    -0.1102,
    1.3807,
    -0.0551,
]


def quaternion_from_euler(ai, aj, ak):
    """Convert Euler angles (XYZ convention) to quaternion [qx, qy, qz, qw]."""
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk
    q = [cj * sc - sj * cs, cj * ss + sj * cc, cj * cs - sj * sc, cj * cc + sj * ss]
    return q


class GrillTaskEnv:
    def __init__(self, headless=True):
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=headless)
        # Optional mode: preserve the exact lid pose authored in the scene.
        self._preserve_scene_lid_pose = os.environ.get("GRILL_PRESERVE_SCENE_LID_POSE", "False") == "True"
        self._closed_lid_angle = float(os.environ.get("GRILL_LID_CLOSED_ANGLE", "0.0"))
        self._lid_collision_backup = None
        if self._preserve_scene_lid_pose:
            try:
                pre_lid = Joint('lid_joint')
                self._closed_lid_angle = float(pre_lid.get_joint_position())
                os.environ["GRILL_LID_CLOSED_ANGLE"] = f"{self._closed_lid_angle:.6f}"
            except Exception:
                pass
        else:
            # Force closed lid pose before starting simulation so the initial
            # visible frame begins closed.
            try:
                pre_lid = Joint('lid_joint')
                try:
                    pre_lid.set_joint_position(float(self._closed_lid_angle), disable_dynamics=True)
                except Exception:
                    pass
                try:
                    sim.simSetJointPosition(int(pre_lid.get_handle()), float(self._closed_lid_angle))
                except Exception:
                    pass
            except Exception:
                pass
        self.pr.start()

        # ---- Robot ----
        self.robot = Panda()
        self.gripper = PandaGripper()
        # Store the initial joint configuration as "home" for retreating
        self.home_conf = self.robot.get_joint_positions()

        # ---- Cameras ----
        self.cams = {}
        cam_names = ['cam_over_shoulder_left', 'cam_over_shoulder_right', 
                     'cam_overhead', 'cam_wrist', 'cam_front']
        for name in cam_names:
            try:
                self.cams[name] = VisionSensor(name)
                self.cams[name].set_explicit_handling(1)
                self.cams[name].set_resolution([640, 480])
            except Exception:
                pass

        # Directory for saving/loading named joint configurations
        import os as _os
        self.state_dir = _os.path.join(_os.path.dirname(__file__), "data_states")
        _os.makedirs(self.state_dir, exist_ok=True)

        def _safe_shape(name):
            try:
                return Shape(name)
            except Exception:
                return None

        def _scene_shape_aliases():
            aliases = []
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
                    alias = None
                    try:
                        alias = sim.simGetObjectAlias(h, 5)
                    except Exception:
                        try:
                            alias = sim.simGetObjectName(h)
                        except Exception:
                            pass
                    if alias:
                        aliases.append(str(alias))
            except Exception:
                pass
            return aliases

        def _find_shape(include_tokens, exclude_tokens=()):
            tokens = [t.lower() for t in include_tokens]
            excludes = [t.lower() for t in exclude_tokens]
            for alias in _scene_shape_aliases():
                low = alias.lower()
                if all(t in low for t in tokens) and all(e not in low for e in excludes):
                    obj = _safe_shape(alias)
                    if obj is not None:
                        return obj
            return None

        # ---- Objects based on scene hierarchy ----
        # From the scene hierarchy image:
        # - steak (with steak_visual)
        # - chicken (with chicken_visual)
        # - grill_root -> grill -> grill_visual, Shape1
        # - lid_joint -> lid -> lid_visual, handle_visual
        # - dish_rack with pillars and success positions
        # - plate with plate_visual
        
        # Core movable objects (single-name defaults + robust fallback lookup)
        self.steak = _safe_shape('steak') or _safe_shape('steak_visual') or _find_shape(
            include_tokens=['steak'],
            exclude_tokens=['boundary', 'grill', 'lid', 'handle', 'joint', 'visual'],
        )
        self.chicken = _safe_shape('chicken') or _safe_shape('chicken_visual') or _find_shape(
            include_tokens=['chicken'],
            exclude_tokens=['boundary', 'grill', 'lid', 'handle', 'joint', 'visual'],
        )
        self.spam = _safe_shape('spam') or _safe_shape('spam_visual') or _find_shape(
            include_tokens=['spam'],
            exclude_tokens=['boundary', 'grill', 'lid', 'handle', 'joint', 'visual'],
        )

        # Grill components
        self.grill = _safe_shape('grill_visual') or _safe_shape('grill')
        self.grill_lid = _safe_shape('lid_visual') or _safe_shape('lid')

        # grill_boundary is the top surface where objects are placed
        self.grill_surface = _safe_shape('grill_boundary')

        # Try to get the lid joint for hinged rotation
        try:
            self.lid_joint = Joint('lid_joint')
            self._initial_lid_angle = float(self.lid_joint.get_joint_position())
        except Exception:
            self.lid_joint = None
            self._initial_lid_angle = None
            print("Warning: 'lid_joint' not found, lid rotation may not work")

        if self._preserve_scene_lid_pose:
            if self.lid_joint is not None:
                try:
                    self._closed_lid_angle = float(self.lid_joint.get_joint_position())
                    os.environ["GRILL_LID_CLOSED_ANGLE"] = f"{self._closed_lid_angle:.6f}"
                except Exception:
                    pass
        else:
            self._closed_lid_angle = float(os.environ.get("GRILL_LID_CLOSED_ANGLE", "0.0"))

        if self.lid_joint is not None:
            # Hard-hold the lid in closed state during startup.
            self.set_lid_servo_lock(True)
            if not self._preserve_scene_lid_pose:
                # Do not step here before objects are frozen; it can destabilize
                # the plate/meat at startup in this scene.
                self._set_lid_angle_hard(float(self._closed_lid_angle), hold_steps=0)

        # Try to get the physical handle first (avoid visual-only shape).
        self.handle = _safe_shape('handle') or _safe_shape('handle_visual')
        if self.handle is None:
            self.handle = _find_shape(
                include_tokens=['handle'],
                exclude_tokens=['boundary'],
            )
            if self.handle is None:
                print("Warning: handle shape not found")

        # Plate (for placing meat)
        self.plate = _safe_shape('plate') or _safe_shape('plate_visual')
        if self.plate is None:
            self.plate = _find_shape(
                include_tokens=['plate'],
                exclude_tokens=['boundary', 'visual'],
            )
            if self.plate is None:
                print("Warning: plate shape not found")

        # Plate boundary (placement surface on plate)
        self.plate_boundary = _safe_shape('plate_boundary')
        if self.plate_boundary is None:
            print("Warning: 'plate_boundary' shape not found")

        # Dish rack
        self.dish_rack = _safe_shape('dish_rack')
        self.placement_boundary = _safe_shape('placement_boundary') or _safe_shape('box_boundary')

        # Success region for placing meat on grill
        self.grill_boundary = _safe_shape('grill_boundary')
        if self.grill_boundary is None:
            print("Warning: 'grill_boundary' shape not found")

        # Object name mapping
        self.name_to_obj = {
            'steak': self.steak,
            'chicken': self.chicken,
            'meat1': self.steak,      # Alias
            'meat2': self.chicken,    # Alias
            'grill_lid': self.grill_lid,
            'lid': self.grill_lid,
        }
        if self.spam:
            self.name_to_obj['spam'] = self.spam
        
        # Add plate if exists
        if self.plate:
            self.name_to_obj['plate'] = self.plate

        # Define regions
        # grill-top uses the grill_boundary dummy for correct placement area
        # plate-top uses the plate_boundary for correct placement area
        self.regions = {
            'grill-top': self.grill_boundary if self.grill_boundary else self.grill,
            'table': self.placement_boundary if self.placement_boundary else self.grill,
        }
        
        if self.plate:
            self.regions['plate'] = self.plate
        if self.plate_boundary:
            self.regions['plate-top'] = self.plate_boundary
            self.regions['plate_boundary'] = self.plate_boundary  # Alias underscore
            self.regions['plate-boundary'] = self.plate_boundary  # Alias hyphen
        elif self.plate:
            # Fallback to plate_visual if no boundary
            self.regions['plate-top'] = self.plate
            self.regions['plate_boundary'] = self.plate
            self.regions['plate-boundary'] = self.plate
        if self.dish_rack:
            self.regions['dish_rack'] = self.dish_rack
        if self.placement_boundary:
            self.regions['placement_boundary'] = self.placement_boundary
            self.regions['placement-boundary'] = self.placement_boundary
            self.regions['box_boundary'] = self.placement_boundary
            self.regions['box-top'] = self.placement_boundary

        legacy_adjust = os.environ.get("GRILL_ADJUST_HOME_CONF", "").strip().lower()
        home_mode = os.environ.get("GRILL_HOME_ADJUST_MODE", "").strip().lower()
        if not home_mode:
            if legacy_adjust in {"false", "0", "no"}:
                home_mode = "off"
            else:
                home_mode = "fixed"
        self.scene_home_adjust_mode = home_mode
        if self.scene_home_adjust_mode == "fixed":
            self._apply_fixed_scene_home_conf()
        elif self.scene_home_adjust_mode == "search":
            self._configure_scene_home_conf()

        # Register every scene shape name containing meat/plate aliases so
        # dynamic lookups in variants resolve reliably (e.g., #0/#1 suffixes).
        for alias in _scene_shape_aliases():
            low = alias.lower()
            if any(k in low for k in ('steak', 'chicken', 'spam', 'plate')):
                obj = _safe_shape(alias)
                if obj is not None:
                    self.name_to_obj.setdefault(alias, obj)

        # Optional startup stabilization for known movable task objects.
        # Keep ON by default: freeze in-place + hold lid at initial joint angle.
        self.enable_startup_stabilization = os.environ.get(
            "GRILL_STARTUP_STABILIZE",
            "True",
        ) == "True"
        self._task_object_handles = {}
        self._task_initial_poses = {}
        self._register_task_objects_for_stability()
        if self.enable_startup_stabilization:
            self.stabilize_startup_state(steps=10)

    def get_object(self, name):
        """Get object by name."""
        if name in self.name_to_obj:
            return self.name_to_obj[name]
        try:
            # Dynamic lookup
            obj = Shape(name)
            self.name_to_obj[name] = obj
            return obj
        except Exception:
            # Soft fallback: resolve by substring match in known aliases.
            low = str(name).lower()
            for alias, obj in self.name_to_obj.items():
                if obj is None:
                    continue
                al = str(alias).lower()
                if (al == low) or (low in al):
                    self.name_to_obj[name] = obj
                    return obj
            return None

    def _apply_fixed_scene_home_conf(self):
        tuned_q = np.array(DEFAULT_TUNED_GRILL_HOME_CONF, dtype=float)
        original_conf = list(self.robot.get_joint_positions())
        try:
            self.set_robot_conf(tuned_q.tolist())
            if self.robot.check_collision():
                self.set_robot_conf(original_conf)
                self.home_conf = list(original_conf)
                print("[startup] WARNING: Tuned grill home conf collides; using scene-authored home pose.")
                return False
            self.home_conf = tuned_q.tolist()
            tip_pos = np.array(self.robot.get_tip().get_position(), dtype=float)
            print(
                "[startup] Using tuned grill home conf: "
                f"tip={np.round(tip_pos, 4).tolist()}"
            )
            return True
        except Exception:
            self.set_robot_conf(original_conf)
            self.home_conf = list(original_conf)
            print("[startup] WARNING: Failed to apply tuned grill home conf; using scene-authored home pose.")
            return False

    def _scene_home_target_candidates(self, reference_obj):
        try:
            min_x, max_x, min_y, _max_y, _min_z, max_z = self._get_world_bounding_box(reference_obj)
            center_x = 0.5 * (min_x + max_x)
            target_y = float(min_y - float(os.environ.get("GRILL_HOME_NEG_Y_MARGIN", "0.14")))
            target_z = float(max_z + float(os.environ.get("GRILL_HOME_Z_LIFT", "0.24")))
        except Exception:
            ref_pos = np.array(reference_obj.get_position(), dtype=float)
            center_x = float(ref_pos[0])
            target_y = float(ref_pos[1] - 0.18)
            target_z = float(ref_pos[2] + 0.24)

        try:
            current_tip = np.array(self.robot.get_tip().get_position(), dtype=float)
            target_y = min(float(target_y), float(current_tip[1] - 0.10))
            target_z = max(float(target_z), float(current_tip[2] + 0.10))
        except Exception:
            pass

        x_shift = float(os.environ.get("GRILL_HOME_X_SHIFT", "0.0"))
        base = np.array([center_x + x_shift, target_y, target_z], dtype=float)
        nudges = [
            np.array([0.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 0.03, 0.0], dtype=float),
            np.array([-0.05, 0.02, -0.02], dtype=float),
            np.array([0.05, 0.02, -0.02], dtype=float),
            np.array([0.0, -0.02, 0.03], dtype=float),
        ]
        return [(base + delta).tolist() for delta in nudges]

    def _configure_scene_home_conf(self):
        reference_obj = self.grill_boundary or self.grill_surface or self.grill
        if reference_obj is None:
            return False

        original_conf = list(self.robot.get_joint_positions())
        try:
            current_tip_quat = list(self.robot.get_tip().get_quaternion())
        except Exception:
            current_tip_quat = quaternion_from_euler(np.pi, 0.0, 0.0)

        candidate_quats = [
            quaternion_from_euler(np.pi, 0.0, 0.0),
            quaternion_from_euler(np.pi, 0.0, -0.35),
            current_tip_quat,
        ]

        for target_pos in self._scene_home_target_candidates(reference_obj):
            for target_quat in candidate_quats:
                try:
                    ik_solutions = self.robot.solve_ik_via_sampling(
                        target_pos,
                        quaternion=target_quat,
                        max_configs=4,
                        max_time_ms=60,
                        ignore_collisions=True,
                    )
                except Exception:
                    ik_solutions = None
                if ik_solutions is None or len(ik_solutions) == 0:
                    continue
                for q_home in ik_solutions:
                    try:
                        self.set_robot_conf(q_home)
                        if self.robot.check_collision():
                            continue
                        self.home_conf = list(q_home)
                        tip_pos = np.array(self.robot.get_tip().get_position(), dtype=float)
                        print(
                            "[startup] Adjusted grill home conf: "
                            f"target={np.round(np.array(target_pos, dtype=float), 4).tolist()} "
                            f"tip={np.round(tip_pos, 4).tolist()}"
                        )
                        return True
                    except Exception:
                        continue

        self.set_robot_conf(original_conf)
        self.home_conf = list(original_conf)
        print("[startup] WARNING: Could not adjust grill home conf; using scene-authored home pose.")
        return False

    def _register_task_objects_for_stability(self):
        """Collect all meat/plate objects and remember their initial poses."""
        def _add(obj):
            if obj is None:
                return
            try:
                h = int(obj.get_handle())
            except Exception:
                return
            if h in self._task_object_handles:
                return
            self._task_object_handles[h] = obj
            try:
                self._task_initial_poses[h] = list(obj.get_pose())
            except Exception:
                pass

        # Primary mapped objects
        _add(self.steak)
        _add(self.chicken)
        _add(self.spam)
        _add(self.plate)

        # Additional known physical object names used in grill variations.
        extra_names = [
            "steak1",
            "steak2",
            "steak3",
            "chicken1",
            "chicken2",
            "chicken3",
            "spam1",
            "spam2",
            "spam3",
            "plate1",
            "plate2",
        ]
        for name in extra_names:
            try:
                _add(self.get_object(name))
            except Exception:
                pass

    def stabilize_task_objects(self):
        """Freeze task objects to avoid startup physics explosions."""
        for obj in self._task_object_handles.values():
            try:
                obj.set_dynamic(False)
            except Exception:
                pass

    def reset_task_objects(self, restore_pose=False, freeze=True):
        """Optionally restore task objects to startup poses."""
        for h, obj in self._task_object_handles.items():
            try:
                pose = self._task_initial_poses.get(h)
                if restore_pose and pose is not None:
                    obj.set_pose(list(pose))
                if freeze:
                    obj.set_dynamic(False)
            except Exception:
                pass

    def stabilize_startup_state(self, steps=20):
        """
        Keep startup state stable without changing scene's collidable/respondable flags.
        """
        self.stabilize_task_objects()
        hold_angle = self._closed_lid_angle
        if self.lid_joint is not None:
            self.set_lid_servo_lock(True)
            if not self._preserve_scene_lid_pose:
                # Prevent grill contents from pushing the lid off-hinge while we
                # establish the closed pose.
                self.set_lid_collision_enabled(False)
                self._set_lid_angle_hard(float(hold_angle), hold_steps=2)
        for _ in range(max(1, int(steps))):
            if self.lid_joint is not None:
                try:
                    self.lid_joint.set_joint_target_position(float(hold_angle))
                except Exception:
                    pass
                try:
                    self.lid_joint.set_joint_target_velocity(0.0)
                except Exception:
                    pass
            self.pr.step()
        if self.lid_joint is not None and (not self._preserve_scene_lid_pose):
            self.set_lid_collision_enabled(True)

    def set_lid_servo_lock(self, lock=True):
        """Enable/disable a strong servo hold on the lid joint."""
        if self.lid_joint is None:
            return False
        try:
            self.lid_joint.set_joint_mode(JointMode.FORCE)
        except Exception:
            pass
        try:
            self.lid_joint.set_motor_enabled(True)
        except Exception:
            pass
        try:
            self.lid_joint.set_joint_force(float(os.environ.get("GRILL_LID_MAX_FORCE", "400.0")))
        except Exception:
            pass
        try:
            self.lid_joint.set_motor_locked_at_zero_velocity(bool(lock))
        except Exception:
            pass
        try:
            self.lid_joint.set_control_loop_enabled(bool(lock))
        except Exception:
            pass
        if lock:
            try:
                cur = float(self.lid_joint.get_joint_position())
            except Exception:
                cur = float(self._closed_lid_angle)
            try:
                self.lid_joint.set_joint_target_position(cur)
            except Exception:
                pass
            try:
                self.lid_joint.set_joint_target_velocity(0.0)
            except Exception:
                pass
        return True

    def _set_lid_angle_hard(self, angle, hold_steps=0):
        """Set lid joint angle robustly, even with dynamics enabled."""
        if self.lid_joint is None:
            return False
        try:
            self.lid_joint.set_joint_position(float(angle), disable_dynamics=True)
        except Exception:
            try:
                sim.simSetJointPosition(int(self.lid_joint.get_handle()), float(angle))
            except Exception:
                pass
        try:
            self.lid_joint.set_joint_target_position(float(angle))
        except Exception:
            pass
        try:
            self.lid_joint.set_joint_target_velocity(0.0)
        except Exception:
            pass
        for _ in range(max(0, int(hold_steps))):
            try:
                self.lid_joint.set_joint_target_position(float(angle))
            except Exception:
                pass
            self.pr.step()
        return True

    def set_lid_collision_enabled(self, enabled=True):
        """
        Enable/disable lid collision+response temporarily.
        This is useful while force-setting a closed lid pose.
        """
        lid = getattr(self, "grill_lid", None)
        if lid is None:
            return False
        if enabled:
            backup = self._lid_collision_backup
            if backup is None:
                return True
            col, resp = backup
            try:
                lid.set_collidable(bool(col))
            except Exception:
                pass
            try:
                lid.set_respondable(bool(resp))
            except Exception:
                pass
            self._lid_collision_backup = None
            return True

        if self._lid_collision_backup is None:
            try:
                col = bool(lid.is_collidable())
            except Exception:
                col = True
            try:
                resp = bool(lid.is_respondable())
            except Exception:
                resp = True
            self._lid_collision_backup = (col, resp)
        try:
            lid.set_collidable(False)
        except Exception:
            pass
        try:
            lid.set_respondable(False)
        except Exception:
            pass
        return True

    def get_robot_conf(self):
        """Get current robot joint configuration."""
        return self.robot.get_joint_positions()

    def get_home_conf(self):
        """Return the stored home configuration captured at startup."""
        return list(self.home_conf)

    def set_robot_conf(self, q):
        """Set robot joint configuration."""
        self.robot.set_joint_positions(q)
        try:
            self.robot.set_joint_target_positions(q)
        except Exception:
            pass

    def save_conf(self, name, q=None):
        """Save a joint configuration under a given name."""
        import os as _os
        if q is None:
            q = self.get_robot_conf()
        path = _os.path.join(self.state_dir, f"{name}.npy")
        np.save(path, np.array(q, dtype=np.float32))

    def load_conf(self, name):
        """Load a previously saved joint configuration."""
        import os as _os
        path = _os.path.join(self.state_dir, f"{name}.npy")
        if not _os.path.exists(path):
            raise FileNotFoundError(f"Saved conf '{name}' not found at {path}")
        return np.load(path).tolist()

    def _get_world_bounding_box(self, obj):
        """Get the axis-aligned bounding box of an object in world coordinates."""
        min_x, max_x, min_y, max_y, min_z, max_z = obj.get_bounding_box()
        corners = np.array([
            [min_x, min_y, min_z], [min_x, min_y, max_z],
            [min_x, max_y, min_z], [min_x, max_y, max_z],
            [max_x, min_y, min_z], [max_x, min_y, max_z],
            [max_x, max_y, min_z], [max_x, max_y, max_z]
        ])

        # Transform to world
        matrix = obj.get_matrix()
        m = np.array(matrix)
        if m.size == 16:
            m = m.reshape(4, 4)
        elif m.size == 12:
            m = m.reshape(3, 4)

        world_corners = []
        for c in corners:
            c_h = np.append(c, 1.0)
            if m.shape == (4, 4):
                wc = np.dot(m, c_h)[:3]
            else:
                wc = np.dot(m, c_h)
            world_corners.append(wc)
        world_corners = np.array(world_corners)

        w_min = np.min(world_corners, axis=0)
        w_max = np.max(world_corners, axis=0)
        return w_min[0], w_max[0], w_min[1], w_max[1], w_min[2], w_max[2]

    def _get_linear_path(self, q_start, target_pos, target_quat, ignore_collisions=False, steps=50):
        """Plan a linear Cartesian path."""
        self.set_robot_conf(q_start)
        try:
            path = self.robot.get_linear_path(
                position=target_pos, 
                quaternion=target_quat, 
                steps=steps, 
                ignore_collisions=ignore_collisions
            )
            return path
        except Exception:
            return None

    def _interpolate_joint_path(self, q1, q2, steps=50, check_collisions=True):
        """Generate a simple joint-space interpolation, optionally collision-checked."""
        traj = []
        q1 = np.array(q1)
        q2 = np.array(q2)

        if steps < 50:
            steps = 50

        for i in range(steps + 1):
            t = i / steps
            q = (1 - t) * q1 + t * q2
            q_list = q.tolist()

            if check_collisions:
                self.set_robot_conf(q_list)
                if self.robot.check_collision():
                    return None

            traj.append(q_list)
        return traj

    def compute_hover_config(self, obj, pose, hover_offset=0.15, preferred_orientation=None):
        """
        Return a valid configuration q_hover strictly above the object.
        Also returns the orientation used so pick can use the same.
        
        Args:
            obj: The target object
            pose: Object pose [x,y,z,qx,qy,qz,qw]
            hover_offset: Height above object
            preferred_orientation: If provided, use this orientation (from previous pick computation)
        
        Returns:
            (q_hover, hover_orientation) - joint config and the quaternion used
        """
        original_conf = self.get_robot_conf()
        try:
            min_x, max_x, min_y, max_y, min_z, max_z = obj.get_bounding_box()
            top_z_local = max_z
            hover_z = pose[2] + top_z_local + hover_offset

            target_pos = [pose[0], pose[1], hover_z]

            # If preferred orientation is given, try that first
            if preferred_orientation is not None:
                path_configs = self.robot.solve_ik_via_sampling(
                    target_pos, quaternion=preferred_orientation, 
                    max_configs=5, max_time_ms=100, 
                    ignore_collisions=True
                )
                if path_configs is not None and len(path_configs) > 0:
                    q_hover = path_configs[0]
                    self.set_robot_conf(q_hover)
                    if not self.robot.check_collision():
                        return q_hover, preferred_orientation

            # Sample orientations pointing down
            grasp_quats = []
            for angle in np.linspace(0, 2 * np.pi, 16):
                q = quaternion_from_euler(np.pi, 0, angle)
                grasp_quats.append((angle, q))

            for angle, grasp_rot in grasp_quats:
                path_configs = self.robot.solve_ik_via_sampling(
                    target_pos, quaternion=grasp_rot, 
                    max_configs=5, max_time_ms=100, 
                    ignore_collisions=True
                )
                if path_configs is not None and len(path_configs) > 0:
                    q_hover = path_configs[0]
                    # Check collision
                    self.set_robot_conf(q_hover)
                    if not self.robot.check_collision():
                        return q_hover, grasp_rot

            raise RuntimeError("Could not find valid hover configuration")
        finally:
            self.set_robot_conf(original_conf)

    def _get_path(self, q_start, target_pos, target_quat, trials=20, max_time_ms=2000.0):
        # Helper to plan path using RRTConnect as last resort
        self.set_robot_conf(q_start)
        try:
            path = self.robot.get_path(position=target_pos, quaternion=target_quat,
                                       ignore_collisions=False,
                                       algorithm=ConfigurationPathAlgorithms.RRTConnect,
                                       max_configs=5, trials=trials, max_time_ms=max_time_ms)
            return path
        except Exception:
            return None

    def compute_retreat_to_home(self, q_start):
        """Plan a retreat from the current config back to the stored home pose."""
        q_home = list(self.home_conf)

        # 1. Try simple interpolation first (fastest)
        traj = self._interpolate_joint_path(q_start, q_home, steps=50, check_collisions=True)
        if traj:
            return q_home, traj

        # 2. If blocked, try lifting first
        self.set_robot_conf(q_start)
        curr_pos = self.robot.get_position()
        curr_quat = self.robot.get_quaternion()

        # Lift by 20cm
        lift_pos = [curr_pos[0], curr_pos[1], curr_pos[2] + 0.2]
        path_lift = self.robot.get_linear_path(
            position=lift_pos, quaternion=curr_quat, 
            steps=30, ignore_collisions=False
        )

        if path_lift:
            q_lift_end = path_lift._path_points[-7:].tolist()
            traj_home = self._interpolate_joint_path(q_lift_end, q_home, steps=50, check_collisions=True)
            if traj_home:
                traj_lift = path_lift._path_points.reshape(-1, 7).tolist()
                traj_lift[0] = list(q_start)
                return q_home, traj_lift + traj_home

        return None, None

    def compute_motion_plan(self, q1, q2):
        """Plan a path from q1 to q2 with collision checking."""
        try:
            # Special case: retreat to home
            if np.allclose(q2, self.home_conf, atol=1e-3):
                _, traj = self.compute_retreat_to_home(q1)
                return traj

            # 1. Try simple interpolation first
            traj = self._interpolate_joint_path(q1, q2, steps=50, check_collisions=True)
            if traj:
                return traj

            # 2. Try lifting first, then move
            self.set_robot_conf(q1)
            p1 = self.robot.get_position()
            quat1 = self.robot.get_quaternion()

            p_lift = [p1[0], p1[1], p1[2] + 0.25]
            path_lift = self.robot.get_linear_path(
                position=p_lift, quaternion=quat1, 
                steps=30, ignore_collisions=False
            )

            if path_lift:
                q_lift = path_lift._path_points[-7:].tolist()
                traj_lift = path_lift._path_points.reshape(-1, 7).tolist()
                traj_lift[0] = list(q1)

                traj_rest = self._interpolate_joint_path(q_lift, q2, steps=100, check_collisions=True)
                if traj_rest:
                    return traj_lift + traj_rest

            # 3. Try via home
            if not np.allclose(q1, self.home_conf, atol=1e-3):
                _, traj_to_home = self.compute_retreat_to_home(q1)
                if traj_to_home:
                    traj_from_home = self._interpolate_joint_path(
                        self.home_conf, q2, steps=100, check_collisions=True
                    )
                    if traj_from_home:
                        return traj_to_home + traj_from_home

            # 4. Last resort: RRTConnect directly from q1 to q2 (if IK-sampling gives a target)
            # Find a target IK configuration for q2 if it was't already one.
            # (In compute_motion_plan, q2 is usually a joint config, so we can use it as-is if we had an OMPL for joint space)
            # Standard pyrep get_path with None pos/quat doesn't work well for joint-to-joint RRT.
            # But we can try q1 -> q2 via RRT if we treat q2 as a configuration target.
            # For now, we rely on the above fallbacks. 
            # Optimization: If the target q2 is home, it's already handled.
            
            return None
        except Exception:
            return None

    def sample_stable_pose(self, obj, region_name):
        """Return a stable 7D pose (x,y,z,qx,qy,qz,qw) for obj in region."""
        from pyrep.objects.dummy import Dummy
        
        region = self.regions.get(region_name)
        if not region:
            print(f"Region {region_name} not found, returning current pose")
            return obj.get_pose()

        current_pose = obj.get_pose()
        
        # Check if region is a Dummy (point in space) vs Shape (has bounding box)
        if isinstance(region, Dummy):
            # For Dummy objects, use the position directly with small random offset
            dummy_pos = region.get_position()
            # Add small random offset within a radius (e.g., 5cm)
            radius = 0.05
            offset_x = np.random.uniform(-radius, radius)
            offset_y = np.random.uniform(-radius, radius)
            
            sample_x = dummy_pos[0] + offset_x
            sample_y = dummy_pos[1] + offset_y
            sample_z = dummy_pos[2] + 0.01  # Slightly above the dummy point
            
            print(f"DEBUG: Using Dummy position for grill-top: {dummy_pos}")
            print(f"DEBUG: Sampled place position: [{sample_x:.3f}, {sample_y:.3f}, {sample_z:.3f}]")
        else:
            # For Shape objects, use bounding box
            w_min_x, w_max_x, w_min_y, w_max_y, w_min_z, w_max_z = self._get_world_bounding_box(region)

            # Sample x and y within world bounds (with padding)
            padding = 0.03
            if (w_max_x - w_min_x) < 2 * padding:
                padding = 0
            if (w_max_y - w_min_y) < 2 * padding:
                padding = 0

            sample_x = np.random.uniform(w_min_x + padding, w_max_x - padding)
            sample_y = np.random.uniform(w_min_y + padding, w_max_y - padding)

            # For grill-top, place on top of grill surface
            if region_name == 'grill-top':
                sample_z = w_max_z + 0.03  # Higher above grill surface to avoid clipping
            elif region_name in ['plate-top', 'plate_boundary', 'plate']:
                sample_z = w_max_z + 0.02  # Slightly above plate surface
            else:
                sample_z = w_max_z + 0.005

        new_pose = list(current_pose)
        new_pose[0] = sample_x
        new_pose[1] = sample_y
        new_pose[2] = sample_z

        return new_pose

    def compute_pick_trajectory(self, obj, pose, preferred_orientation=None, is_plate=False):
        """
        Return grasp, q_start, q_end, and trajectory for picking obj at pose.
        Uses top-down vertical grasp strategy.
        
        Args:
            obj: The target object
            pose: Object pose [x,y,z,qx,qy,qz,qw]
            preferred_orientation: If provided (from hover), try this orientation first
            is_plate: If True, go DEEPER for better grasp on plate
        
        Returns:
            (grasp, q_start, q_end, (approach_traj, retreat_traj))
        """
        original_conf = self.get_robot_conf()
        
        try:
            try:
                obj_name = str(obj.get_name()).lower()
            except Exception:
                obj_name = ""
            # 1. Analyze Object Geometry
            min_x, max_x, min_y, max_y, min_z, max_z = obj.get_bounding_box()
            obj_height = max_z - min_z
            obj_width = max_x - min_x
            obj_length = max_y - min_y
            top_z_local = max_z
            
            print(f"DEBUG pick: Object bounding box height = {obj_height:.4f}")

            # 2. Define Grasp Strategy (Top-Down)
            if is_plate:
                # For plate: slightly deeper to get a good grip on the rim
                grasp_depths = [0.025, 0.03, 0.035, 0.04, 0.045]
                print(f"DEBUG pick: PLATE mode - grasp depths: {grasp_depths}")
                valid_depths = grasp_depths  # Use all for plate
            else:
                is_flat_pick = (
                    ("spam" in obj_name)
                    or (obj_height < 0.03 and max(obj_width, obj_length) > 0.04)
                )
                if is_flat_pick:
                    max_depth = max(0.006, float(obj_height - 0.002))
                    min_depth = min(0.008, max_depth)
                    grasp_depths = list(np.linspace(min_depth, max_depth, 5))
                    grasp_depths.extend([max_depth * 0.9, obj_height * 0.5])
                    print(f"DEBUG pick: FLAT/SPAM mode - grasp depths: {[round(d, 4) for d in grasp_depths]}")
                    valid_depths = [
                        float(d) for d in grasp_depths
                        if 0.003 <= float(d) <= max(0.003, float(obj_height - 0.001))
                    ]
                else:
                    grasp_depths = [0.02, 0.04, 0.06, 0.08]
                    valid_depths = [d for d in grasp_depths if d < (obj_height - 0.01)]
                if not valid_depths:
                    valid_depths = [max(0.004, float(obj_height / 2.0))]

            # 3. Define Grasp Orientations
            grasp_quats = []
            if is_plate:
                # For plate: try 90-degree rotated orientations first
                for angle in [np.pi/2, -np.pi/2, 0, np.pi]:
                    q = quaternion_from_euler(np.pi, 0, angle)
                    grasp_quats.append(q)
            elif preferred_orientation is not None:
                grasp_quats.append(preferred_orientation)
            
            angles = np.linspace(0, 2 * np.pi, 16)
            for angle in angles:
                q = quaternion_from_euler(np.pi, 0, angle)
                grasp_quats.append(q)

            # 4. Iterate and Solve
            for depth in valid_depths:
                target_z = pose[2] + top_z_local - depth
                target_pos = [pose[0], pose[1], target_z]

                hover_offset = 0.15
                hover_pos = [target_pos[0], target_pos[1], target_pos[2] + hover_offset]
                
                print(f"DEBUG pick: Trying depth={depth:.3f}, target_z={target_z:.3f}")

                for grasp_rot in grasp_quats:
                    try:
                        # A. Solve IK for Grasp Pose - increased sampling for plate
                        path_configs = self.robot.solve_ik_via_sampling(
                            target_pos, quaternion=grasp_rot,
                            max_configs=10, max_time_ms=200,
                            ignore_collisions=True
                        )
                        if path_configs is None or len(path_configs) == 0:
                            continue
                        q_grasp = path_configs[0]

                        # B. Solve IK for Hover Pose
                        path_configs_hover = self.robot.solve_ik_via_sampling(
                            hover_pos, quaternion=grasp_rot,
                            max_configs=10, max_time_ms=200,
                            ignore_collisions=True
                        )
                        if path_configs_hover is None or len(path_configs_hover) == 0:
                            continue
                        q_hover = path_configs_hover[0]

                        # Validate hover is collision-free (skip for plate since it's in dish rack)
                        self.set_robot_conf(q_hover)
                        if not is_plate and self.robot.check_collision():
                            continue

                        # C. Plan Hover -> Grasp (Linear Approach)
                        path2 = self._get_linear_path(q_hover, target_pos, grasp_rot, ignore_collisions=True)
                        if not path2:
                            continue

                        q_grasp_actual = path2._path_points[-7:].tolist()

                        # D. Plan Grasp -> Hover (Linear Retract)
                        path3 = self._get_linear_path(q_grasp_actual, hover_pos, grasp_rot, ignore_collisions=True)
                        if not path3:
                            path2.remove()
                            continue

                        q_hover_end = path3._path_points[-7:].tolist()

                        def get_configs(p):
                            return p._path_points.reshape(-1, 7).tolist()

                        t_approach = get_configs(path2)
                        t_retreat = get_configs(path3)

                        grasp = [0] * 7
                        print(f"DEBUG pick: SUCCESS at depth={depth:.3f}")
                        return grasp, q_hover, q_hover_end, (t_approach, t_retreat)

                    except Exception:
                        continue

            print(f"DEBUG: compute_pick_trajectory failed for {obj} at {pose}")
            raise RuntimeError("Could not find valid grasp configuration")
        finally:
            self.set_robot_conf(original_conf)

    def compute_place_trajectory(self, obj, pose, region_name=None, is_plate=False):
        """
        Return grasp, q_start, q_end, and trajectory for placing obj at pose.
        
        For regular objects: top-down vertical approach
        For plates: HORIZONTAL approach (gripper fingers vertical, approach from side)
        
        Args:
            obj: Object to place
            pose: Target pose [x,y,z,qx,qy,qz,qw]
            region_name: Target region name
            is_plate: If True, use HORIZONTAL approach for plate placement
        """
        original_conf = self.get_robot_conf()
        try:
            min_x, max_x, min_y, max_y, min_z, max_z = obj.get_bounding_box()
            top_z_local = max_z

            if is_plate:
                # HORIZONTAL APPROACH for plates
                # Plate should be placed FLAT on the surface
                # Gripper approaches from the side (negative X direction), fingers vertical
                
                place_z = pose[2] + 0.05  # Height of the target surface + small offset
                horizontal_offset = 0.20  # How far back to hover before approaching
                
                # Hover position: same height as place, but offset in X (approach from front)
                target_pos_hover = [pose[0] - horizontal_offset, pose[1], place_z]
                # Place position: at target location
                target_pos_place = [pose[0], pose[1], place_z]
                
                print(f"DEBUG plate place: HORIZONTAL approach")
                print(f"  Hover (offset): {target_pos_hover}")
                print(f"  Place target: {target_pos_place}")
                
                # Gripper orientation for horizontal approach:
                # Gripper pointing forward (+X), fingers opening vertically (up/down)
                # euler(0, π/2, 0) = gripper horizontal, fingers vertical
                grasp_quats = []
                # Try different yaw angles for horizontal approach
                for yaw in [0, np.pi/4, -np.pi/4, np.pi/2, -np.pi/2]:
                    # Gripper horizontal (pointing +X), fingers vertical
                    q = quaternion_from_euler(np.pi/2, yaw, np.pi/2)
                    grasp_quats.append(q)
                    # Also try slight tilts
                    q2 = quaternion_from_euler(np.pi/2 + 0.1, yaw, np.pi/2)
                    grasp_quats.append(q2)
                    q3 = quaternion_from_euler(np.pi/2 - 0.1, yaw, np.pi/2)
                    grasp_quats.append(q3)
            else:
                # VERTICAL APPROACH for regular objects (top-down)
                hover_z = pose[2] + top_z_local + 0.15
                place_z = pose[2] + 0.05

                target_pos_hover = [pose[0], pose[1], hover_z]
                target_pos_place = [pose[0], pose[1], place_z]
                
                print(f"DEBUG place: VERTICAL approach")
                print(f"  Hover (above): {target_pos_hover}")
                print(f"  Place target: {target_pos_place}")

                grasp_quats = []
                angles = np.linspace(0, 2 * np.pi, 32)
                for angle in angles:
                    q = quaternion_from_euler(np.pi, 0, angle)
                    grasp_quats.append(q)

            for i, grasp_rot in enumerate(grasp_quats):
                try:
                    # A. Solve IK for Hover Pose
                    path_configs_hover = self.robot.solve_ik_via_sampling(
                        target_pos_hover, quaternion=grasp_rot,
                        max_configs=10, max_time_ms=200,
                        ignore_collisions=True
                    )
                    if path_configs_hover is None or len(path_configs_hover) == 0:
                        continue
                    q_hover = path_configs_hover[0]

                    # B. Solve IK for Place Pose
                    path_configs_place = self.robot.solve_ik_via_sampling(
                        target_pos_place, quaternion=grasp_rot,
                        max_configs=10, max_time_ms=200,
                        ignore_collisions=True
                    )
                    if path_configs_place is None or len(path_configs_place) == 0:
                        continue
                    q_place = path_configs_place[0]

                    # C. Plan Hover -> Place (Linear path)
                    path_forward = self._get_linear_path(q_hover, target_pos_place, grasp_rot, ignore_collisions=True, steps=50)
                    if not path_forward:
                        continue

                    # D. Plan Place -> Hover (Linear Return - retreat)
                    path_back = self._get_linear_path(q_place, target_pos_hover, grasp_rot, ignore_collisions=True, steps=50)
                    if not path_back:
                        path_forward.remove()
                        continue

                    def get_configs(p):
                        return p._path_points.reshape(-1, 7).tolist()

                    t_forward = get_configs(path_forward)
                    t_back = get_configs(path_back)

                    grasp = [0] * 7
                    print(f"DEBUG: Found valid place config at orientation {i}")
                    return grasp, q_hover, q_hover, (t_forward, t_back)

                except Exception as e:
                    continue

            raise RuntimeError(f"Could not find valid place configuration for region {region_name}")
        finally:
            self.set_robot_conf(original_conf)

    def compute_close_grill_trajectory(self, lid):
        """
        Compute trajectory to close the grill lid (hinged rotation).
        This involves grasping the handle and rotating the lid down.
        """
        original_conf = self.get_robot_conf()
        try:
            # For now, return a placeholder - actual implementation needs
            # to understand the lid joint mechanism in the scene
            if self.lid_joint is not None:
                # Get current joint position and target
                current_angle = self.lid_joint.get_joint_position()
                # Assuming 0 is closed, and current is open
                # We need to plan a trajectory that follows the arc

            # Placeholder: return a simple approach-grasp-rotate sequence
            grasp = [0] * 7
            q_start = self.get_home_conf()
            q_end = self.get_home_conf()
            traj = ([], [])  # Placeholder empty trajectories

            raise RuntimeError("Close grill trajectory not fully implemented yet")
        finally:
            self.set_robot_conf(original_conf)

    def _get_grill_handle_midpoint(self):
        """Return (handle_obj, midpoint_xyz) for the grill handle in world frame."""
        handle = self.handle or self.get_object('handle') or self.get_object('handle_visual')
        if handle is None:
            raise RuntimeError("Grill handle object not found")

        try:
            min_x, max_x, min_y, max_y, min_z, max_z = self._get_world_bounding_box(handle)
            midpoint = np.array(
                [
                    0.5 * (float(min_x) + float(max_x)),
                    0.5 * (float(min_y) + float(max_y)),
                    0.5 * (float(min_z) + float(max_z)),
                ],
                dtype=float,
            )
        except Exception:
            midpoint = np.array(handle.get_position(), dtype=float)
        return handle, midpoint

    def _open_grill_step1_orientation_candidates(self, hover_pos, handle_midpoint):
        """
        Horizontal handle-facing orientations with vertical fingers preferred.
        """
        facing = np.array(handle_midpoint, dtype=float) - np.array(hover_pos, dtype=float)
        facing[2] = 0.0
        n = float(np.linalg.norm(facing[:2]))
        if n < 1e-9:
            facing = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            facing = facing / n
        yaw = float(np.arctan2(facing[1], facing[0]))

        return [
            # Primary: horizontal tool approach, fingers vertical.
            quaternion_from_euler(np.pi / 2.0, yaw, np.pi / 2.0),
            quaternion_from_euler(np.pi / 2.0 + 0.08, yaw, np.pi / 2.0),
            quaternion_from_euler(np.pi / 2.0 - 0.08, yaw, np.pi / 2.0),
            # Opposite-yaw fallback.
            quaternion_from_euler(np.pi / 2.0, yaw + np.pi, np.pi / 2.0),
            # Legacy fallback used elsewhere in this codebase.
            quaternion_from_euler(np.pi - 0.60, 0.0, np.pi / 2.0),
        ]

    def _compute_open_grill_step1_hover(self, q_home, handle_midpoint):
        """
        Step-1 (open-grill): home -> handle-facing hover in front of handle.

        Geometry requested:
        - same Y and Z as handle midpoint
        - X offset "before" handle from the home-side of the robot
        """
        q_home = list(q_home)
        self.set_robot_conf(q_home)

        tip_home = np.array(self.robot.get_tip().get_position(), dtype=float)
        handle_mid = np.array(handle_midpoint, dtype=float)

        x_offset = float(os.environ.get("GRILL_OPEN_STEP1_X_OFFSET", "0.12"))
        x_dir = 1.0 if (tip_home[0] - handle_mid[0]) >= 0.0 else -1.0
        hover_pos = np.array(
            [
                float(handle_mid[0] + x_dir * x_offset),
                float(handle_mid[1]),
                float(handle_mid[2]),
            ],
            dtype=float,
        )

        best = None
        q_home_np = np.array(q_home, dtype=float)
        for quat in self._open_grill_step1_orientation_candidates(hover_pos, handle_mid):
            try:
                ik_solutions = self.robot.solve_ik_via_sampling(
                    hover_pos.tolist(),
                    quaternion=quat,
                    max_configs=20,
                    max_time_ms=250,
                    ignore_collisions=True,
                )
            except Exception:
                ik_solutions = None
            if ik_solutions is None or len(ik_solutions) == 0:
                continue

            for q_hover in ik_solutions:
                q_hover = list(q_hover)
                try:
                    self.set_robot_conf(q_hover)
                    if self.robot.check_collision():
                        continue
                except Exception:
                    continue

                motion = self.compute_motion_plan(q_home, q_hover)
                if motion is None or len(motion) == 0:
                    continue

                score = float(len(motion)) + 0.01 * float(np.linalg.norm(np.array(q_hover, dtype=float) - q_home_np))
                if (best is None) or (score < best[0]):
                    best = (score, q_hover, quat, motion)

        if best is None:
            raise RuntimeError("Could not find open-grill step-1 hover config from home")

        _score, q_hover_best, quat_best, motion_best = best
        return q_hover_best, quat_best, hover_pos.tolist(), motion_best

    def _compute_open_grill_step2_grasp(self, q_hover, hover_quat, hover_pos, handle_midpoint):
        """
        Step-2 (open-grill): hover -> grasp via horizontal approach.

        Geometry requested:
        - keep same Y and Z as handle midpoint
        - move horizontally along X toward handle
        """
        q_hover = list(q_hover)
        hover_np = np.array(hover_pos, dtype=float)
        handle_mid = np.array(handle_midpoint, dtype=float)

        grasp_clearance = float(os.environ.get("GRILL_OPEN_STEP2_X_CLEARANCE", "0.015"))
        x_dir = 1.0 if (hover_np[0] - handle_mid[0]) >= 0.0 else -1.0
        grasp_pos = np.array(
            [
                float(handle_mid[0] + x_dir * grasp_clearance),
                float(handle_mid[1]),
                float(handle_mid[2]),
            ],
            dtype=float,
        )

        # Preferred: pure Cartesian horizontal move with hover orientation.
        path = self._get_linear_path(
            q_start=q_hover,
            target_pos=grasp_pos.tolist(),
            target_quat=hover_quat,
            ignore_collisions=True,
            steps=60,
        )
        if path:
            traj = path._path_points.reshape(-1, 7).tolist()
            return list(traj[-1]), grasp_pos.tolist(), traj

        # Fallback: IK + joint-space motion planning.
        best = None
        q_hover_np = np.array(q_hover, dtype=float)
        candidate_quats = [hover_quat] + self._open_grill_step1_orientation_candidates(hover_np, handle_mid)
        for quat in candidate_quats:
            try:
                ik_solutions = self.robot.solve_ik_via_sampling(
                    grasp_pos.tolist(),
                    quaternion=quat,
                    max_configs=20,
                    max_time_ms=250,
                    ignore_collisions=True,
                )
            except Exception:
                ik_solutions = None
            if ik_solutions is None or len(ik_solutions) == 0:
                continue

            for q_grasp in ik_solutions:
                q_grasp = list(q_grasp)
                motion = self.compute_motion_plan(q_hover, q_grasp)
                if motion is None or len(motion) == 0:
                    continue
                score = float(len(motion)) + 0.01 * float(
                    np.linalg.norm(np.array(q_grasp, dtype=float) - q_hover_np)
                )
                if (best is None) or (score < best[0]):
                    best = (score, q_grasp, motion)

        if best is None:
            raise RuntimeError("Could not find open-grill step-2 grasp config from hover")

        _score, q_grasp_best, motion_best = best
        return q_grasp_best, grasp_pos.tolist(), motion_best

    def compute_open_grill_trajectory(self, lid):
        """
        Open-grill PDDL motion pipeline:
        1) home -> handle-facing hover
        2) horizontal hover -> handle grasp
        3) (optional) hinge-centered circular opening arc

        Returns: grasp, q_start, q_end, traj
        """
        original_conf = self.get_robot_conf()
        try:
            q_home = list(self.get_home_conf())
            _handle, handle_mid = self._get_grill_handle_midpoint()
            q_hover, hover_quat, hover_pos, traj_home_to_hover = self._compute_open_grill_step1_hover(
                q_home=q_home,
                handle_midpoint=handle_mid,
            )
            q_grasp, _grasp_pos, traj_hover_to_grasp = self._compute_open_grill_step2_grasp(
                q_hover=q_hover,
                hover_quat=hover_quat,
                hover_pos=hover_pos,
                handle_midpoint=handle_mid,
            )

            # Placeholder grasp descriptor (matches the rest of this project).
            grasp = [0] * 7
            use_online_step3 = os.environ.get("GRILL_OPEN_USE_ONLINE_STEP3", "True") == "True"
            if use_online_step3:
                # Keep stream output to Step-1/2 only; Step-3 is computed online
                # during execution (IK per arc point).
                traj = (traj_home_to_hover, traj_hover_to_grasp)
                return grasp, q_home, q_grasp, traj

            q_open_end, _target_joint, traj_grasp_to_open = self._compute_open_grill_step3_arc(
                q_grasp=q_grasp,
                hover_quat=hover_quat,
                hover_pos=hover_pos,
                handle_midpoint=handle_mid,
            )
            traj = (traj_home_to_hover, traj_hover_to_grasp, traj_grasp_to_open)
            return grasp, q_home, q_open_end, traj
        finally:
            self.set_robot_conf(original_conf)

    @staticmethod
    def _axis_angle_matrix(axis, angle):
        """Rodrigues rotation matrix for axis-angle."""
        axis = np.array(axis, dtype=float)
        n = float(np.linalg.norm(axis))
        if n < 1e-9:
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

    @staticmethod
    def _sim_matrix_to_4x4(m12):
        """Convert Coppelia 12-float matrix to homogeneous 4x4."""
        m = np.array(m12, dtype=float).reshape(3, 4)
        out = np.eye(4, dtype=float)
        out[:3, :4] = m
        return out

    @staticmethod
    def _quat_to_rot(quat_xyzw):
        """Quaternion [x,y,z,w] -> 3x3 rotation matrix."""
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

    @staticmethod
    def _rot_to_quat(rot):
        """3x3 rotation matrix -> quaternion [x,y,z,w]."""
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

    def _infer_open_rotation_sign(self, current_joint, handle_obj=None):
        """
        Infer opening direction by a tiny local joint probe.
        Chooses the direction that raises the handle in +Z.
        """
        mode = str(os.environ.get("GRILL_OPEN_STEP3_SIGN", "auto")).strip().lower()
        if mode in {"1", "+1", "+", "pos", "positive"}:
            return 1.0
        if mode in {"-1", "-", "neg", "negative"}:
            return -1.0
        if self.lid_joint is None:
            return 1.0
        if handle_obj is None:
            handle_obj = self.handle
        if handle_obj is None:
            return 1.0

        eps = float(os.environ.get("GRILL_OPEN_STEP3_PROBE_RAD", "0.08"))
        try:
            h = int(self.lid_joint.get_handle())
            base = float(current_joint)
            z0 = float(handle_obj.get_position()[2])

            scores = {}
            for sgn in (1.0, -1.0):
                a = float(base + sgn * eps)
                try:
                    sim.simSetJointPosition(h, a)
                except Exception:
                    pass
                try:
                    sim.simSetJointTargetPosition(h, a)
                except Exception:
                    pass
                try:
                    self.lid_joint.set_joint_position(a, disable_dynamics=True)
                except Exception:
                    pass
                try:
                    self.lid_joint.set_joint_target_position(a)
                except Exception:
                    pass
                self.pr.step()
                z = float(handle_obj.get_position()[2])
                scores[sgn] = z - z0

            # Restore.
            try:
                sim.simSetJointPosition(h, float(base))
            except Exception:
                pass
            try:
                sim.simSetJointTargetPosition(h, float(base))
            except Exception:
                pass
            try:
                self.lid_joint.set_joint_position(float(base), disable_dynamics=True)
            except Exception:
                pass
            try:
                self.lid_joint.set_joint_target_position(float(base))
            except Exception:
                pass
            self.pr.step()

            return 1.0 if scores.get(1.0, -1e9) >= scores.get(-1.0, -1e9) else -1.0
        except Exception:
            return 1.0

    def _compute_open_grill_step3_arc(self, q_grasp, hover_quat, hover_pos, handle_midpoint):
        """
        Step-3 (open-grill): smooth hinge-centered circular arc to target angle.
        """
        q_grasp = list(q_grasp)
        self.set_robot_conf(q_grasp)

        if self.lid_joint is None:
            raise RuntimeError("lid_joint not available for open-grill step-3")

        handle_obj = self.handle or self.get_object('handle') or self.get_object('handle_visual')
        if handle_obj is None:
            raise RuntimeError("handle object not available for open-grill step-3")

        try:
            current_joint = float(self.lid_joint.get_joint_position())
        except Exception:
            current_joint = 0.0

        target_deg = float(os.environ.get("GRILL_OPEN_STEP3_TARGET_DEG", "90.0"))
        target_rad = float(np.deg2rad(target_deg))
        sign = self._infer_open_rotation_sign(current_joint=current_joint, handle_obj=handle_obj)
        target_joint = float(current_joint + sign * target_rad)
        rotation_amount = float(target_joint - current_joint)

        try:
            hinge_pos = np.array(self.lid_joint.get_position(), dtype=float)
        except Exception:
            hpos = np.array(handle_midpoint, dtype=float)
            hinge_pos = np.array([hpos[0], hpos[1] + 0.25, hpos[2] - 0.12], dtype=float)

        try:
            m = self._sim_matrix_to_4x4(
                sim.simGetObjectMatrix(int(self.lid_joint.get_handle()), sim.sim_handle_world)
            )
            hinge_axis = np.array([m[0, 2], m[1, 2], m[2, 2]], dtype=float)
        except Exception:
            hinge_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        an = float(np.linalg.norm(hinge_axis))
        if an < 1e-9:
            hinge_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            hinge_axis = hinge_axis / an

        tip_obj = self.robot.get_tip()
        tip_start = np.array(tip_obj.get_position(), dtype=float)
        tip_quat_start = np.array(tip_obj.get_quaternion(), dtype=float)
        tip_rot_start = self._quat_to_rot(tip_quat_start)
        handle_start = np.array(handle_obj.get_position(), dtype=float)
        # Contact relation from handle -> tip at grasp start.
        rel_tip_from_handle = tip_start - handle_start

        n_arc = max(18, int(os.environ.get("GRILL_OPEN_STEP3_ARC_POINTS", "32")))
        arc_confs = [list(q_grasp)]
        prev = np.array(q_grasp, dtype=float)
        min_ik_success_ratio = float(os.environ.get("GRILL_OPEN_STEP3_MIN_IK_RATIO", "0.95"))

        all_offsets = np.linspace(0.0, rotation_amount, n_arc)
        for idx, off in enumerate(all_offsets[1:], start=1):
            R = self._axis_angle_matrix(hinge_axis, float(off))
            handle_target = hinge_pos + (R @ (handle_start - hinge_pos))
            tip_target = handle_target + (R @ rel_tip_from_handle)

            tip_rot_target = R @ tip_rot_start
            q_base = self._rot_to_quat(tip_rot_target)
            q_var_p = self._rot_to_quat(self._axis_angle_matrix(hinge_axis, 0.08) @ tip_rot_target)
            q_var_n = self._rot_to_quat(self._axis_angle_matrix(hinge_axis, -0.08) @ tip_rot_target)

            quat_candidates = [q_base, q_var_p, q_var_n, hover_quat]

            solved = None
            for quat in quat_candidates:
                try:
                    cfgs = self.robot.solve_ik_via_sampling(
                        tip_target.tolist(),
                        quaternion=quat,
                        max_configs=8,
                        max_time_ms=120,
                        ignore_collisions=True,
                    )
                except Exception:
                    cfgs = None
                if cfgs is None or len(cfgs) == 0:
                    continue
                cand = min(cfgs, key=lambda cc: float(np.linalg.norm(np.array(cc, dtype=float) - prev)))
                # Reject candidates that don't move enough toward the next arc point.
                step_norm = float(np.linalg.norm(np.array(cand, dtype=float) - prev))
                if step_norm < float(os.environ.get("GRILL_OPEN_STEP3_MIN_STEP_NORM", "0.003")):
                    continue
                solved = list(cand)
                break

            if solved is None:
                raise RuntimeError(
                    f"open-grill step-3 IK failed at waypoint {idx}/{n_arc - 1}; "
                    "cannot guarantee true circular robot motion"
                )

            arc_confs.append(solved)
            prev = np.array(solved, dtype=float)

        # Safety check: ensure substantial robot motion (avoid fake lid-only opening).
        conf_deltas = []
        for i in range(len(arc_confs) - 1):
            d = float(np.linalg.norm(np.array(arc_confs[i + 1], dtype=float) - np.array(arc_confs[i], dtype=float)))
            conf_deltas.append(d)
        total_joint_travel = float(np.sum(conf_deltas)) if conf_deltas else 0.0
        min_joint_travel = float(os.environ.get("GRILL_OPEN_STEP3_MIN_JOINT_TRAVEL", "0.45"))
        if total_joint_travel < min_joint_travel:
            raise RuntimeError(
                f"open-grill step-3 joint travel too small ({total_joint_travel:.3f} < {min_joint_travel:.3f}); "
                "rejecting non-moving robot arc"
            )

        ik_success = len(arc_confs) - 1
        ratio = float(ik_success) / float(max(1, n_arc - 1))
        if ratio < min_ik_success_ratio:
            raise RuntimeError(
                f"open-grill step-3 IK ratio too low ({ratio:.2f} < {min_ik_success_ratio:.2f})"
            )

        self._open_grill_last_target_joint = float(target_joint)
        return list(arc_confs[-1]), float(target_joint), arc_confs

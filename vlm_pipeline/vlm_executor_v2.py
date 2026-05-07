"""
VLM Executor v2 (With PDDL Execution)
=====================================
Executes VLM-generated action skeletons using PDDL-based planning,
matching ground_truth_orchestrator.py logic.

Features:
- Uses PDDL planning for each pick-place action
- Supports cupboard horizontal grasp
- Supports box lid opening
- Robust homing with collision avoidance
"""

import os
import sys
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# PDDLStream
from pddlstream.algorithms.meta import solve
from pddlstream.language.constants import PDDLProblem, And
from pddlstream.utils import read


@dataclass
class ActionSkeleton:
    """Lightweight action skeleton used by the executor."""
    action_name: str
    args: Tuple[str, ...]

    def __str__(self):
        return f"{self.action_name}({', '.join(self.args)})"


def _ground_truth_orchestrator():
    import ground_truth_orchestrator as gt

    return gt

# ============================================================
# OBJECT NAME MAPPING (VLM names <-> Environment names)
# ============================================================

VLM_TO_ENV_OBJECT = {
    # Mugs
    'mug_box': 'mug2',           # On top of box
    'mug_inside_box': 'mug4',    # Inside the box
    'mug_table': 'mug1',         # On table
    'mug_cupboard': 'mug3',      # In cupboard
    
    # Direct names
    'mug1': 'mug1',
    'mug2': 'mug2',
    'mug3': 'mug3',
    'mug4': 'mug4',
    
    # Groceries (same names)
    'soup': 'soup',
    'mustard': 'mustard',
    'spam': 'spam',
    'sugar': 'sugar',
    'crackers': 'crackers',
    
    # Lid
    'box_lid': 'box_lid',
    'lid': 'box_lid',
}

VLM_TO_ENV_REGION = {
    'box_top': 'box_boundary',
    'box_inside': 'box_boundary',
    'box_boundary': 'box_boundary',
    'table': 'placement_boundary',
    'placement_boundary': 'placement_boundary',
    'cupboard_boundary': 'cupboard_boundary',
    'cupboard_boundary_top': 'cupboard_boundary_top',
    'cupboard': 'cupboard_boundary',
    'groceries_boundary': 'cupboard_boundary',
}

# Objects that require special handling
CUPBOARD_OBJECTS = {'mug3', 'mug_cupboard'}


class ExecutionStatus(Enum):
    """Status of action execution."""
    SUCCESS = "success"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    COLLISION = "collision"
    IK_FAILURE = "ik_failure"
    UNKNOWN_OBJECT = "unknown_object"


@dataclass
class ExecutionResult:
    """Result of executing an action or plan."""
    status: ExecutionStatus
    completed_actions: int
    total_actions: int
    current_action: Optional[ActionSkeleton] = None
    error_message: Optional[str] = None
    trajectory_executed: bool = False
    failure_reason: Optional[str] = None
    blocking_object: Optional[str] = None


@dataclass
class ExecutorConfig:
    """Configuration for the executor."""
    interpolation_steps: int = 15
    min_collision_distance: float = 0.10
    escape_lift_heights: List[float] = field(default_factory=lambda: [0.25, 0.15, 0.10])
    pause_between_actions: float = 0.1  # seconds
    enable_unknown_object_detection: bool = True

    # New performance-related options
    return_home_after_each_action: bool = False
    return_home_joint_tolerance: float = 2e-3
    home_interp_waypoints: int = 20


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
    """Find segment index for release."""
    if not segments:
        return 0
    restore = env.get_robot_conf()
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
    best_i, best_z = 0, float("inf")
    for i, seg in enumerate(segments):
        end_conf = seg[-1]
        ee_pos = _ee_pos_at_conf(env, end_conf, restore)
        if float(ee_pos[2]) < best_z:
            best_z, best_i = float(ee_pos[2]), i
    return best_i


class VLMExecutorV2:
    """
    Executes VLM action skeletons using PDDL-based planning.
    """
    
    # Retry settings for recoverable failures
    MAX_RETRIES = 2
    RECOVERABLE_FAILURES = ['PLACEMENT_FAILED', 'GRASP_FAILED', 'placement_failed', 'grasp_failed']
    
    def __init__(self, 
                 env=None,
                 config: ExecutorConfig = None,
                 unknown_object_callback: Callable = None):
        self.env = env  # Will be set externally via set_env()
        self.config = config or ExecutorConfig()
        self.unknown_object_callback = unknown_object_callback
        
        # Execution state
        self.current_plan: List[ActionSkeleton] = []
        self.executed_actions: List[ActionSkeleton] = []
        self.current_action_idx: int = 0
        self.is_paused: bool = False
        self.is_interrupted: bool = False
        self._active_transfer = None

        # Optional callbacks (all opt-in; default keeps prior behavior)
        self.post_action_callback: Optional[Callable[[List[ActionSkeleton], int, int], Optional[str]]] = None
        self.step_callback: Optional[Callable[[], None]] = None
        
        # Execution monitor for validation
        self.monitor = None
        
        # PDDL paths (will be loaded on first use)
        self.domain_pddl = None
        self.stream_pddl = None
        
    def set_env(self, env):
        """Set the environment instance."""
        self.env = env
        # Initialize monitor with env
        from vlm_pipeline.execution_monitor import ExecutionMonitor
        self.monitor = ExecutionMonitor(env)

    def set_post_action_callback(self, callback: Optional[Callable[[List[ActionSkeleton], int, int], Optional[str]]]):
        """Set callback invoked after each successfully executed action unit."""
        self.post_action_callback = callback

    def set_step_callback(self, callback: Optional[Callable[[], None]]):
        """Set callback invoked after each simulation step."""
        self.step_callback = callback

    def _step_sim(self, n: int = 1):
        """Step simulation with optional external callback (e.g., live viewers)."""
        if self.env is None:
            return
        for _ in range(max(1, n)):
            self.env.pr.step()
            if self.step_callback is not None:
                try:
                    self.step_callback()
                except Exception:
                    # Callback failures must never break execution.
                    pass
        
    def _load_pddl_files(self):
        """Load PDDL files if not already loaded."""
        if self.domain_pddl is None:
            directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.domain_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_domain.pddl'))
            self.stream_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_streams.pddl'))

    def _get_stream_map(self):
        """
        Lazy import to avoid creating RLBench ENV at module import time.
        This preserves GUI/headless selection set during pipeline initialization.
        """
        from rlbench_kitchen_streams import get_stream_map
        return get_stream_map()
        
    def reset(self):
        """Reset executor state."""
        self.current_plan = []
        self.current_action_idx = 0
        self.is_paused = False
        self.is_interrupted = False
        self.executed_actions = []
        self._active_transfer = None
    
    # =========================================================================
    # TRAJECTORY EXECUTION
    # =========================================================================
    
    def _execute_trajectory(self, traj, steps=15):
        """Execute a trajectory with interpolation."""
        if not traj or not self.env:
            return
        full_traj = []
        for i in range(len(traj) - 1):
            start = np.array(traj[i])
            end = np.array(traj[i + 1])
            for t in np.linspace(0, 1, steps, endpoint=False):
                full_traj.append((1 - t) * start + t * end)
        full_traj.append(traj[-1])

        for conf in full_traj:
            self.env.set_robot_conf(conf)
            self._step_sim()
    
    def _interpolate_path(self, q1, q2, steps=50):
        """Linear configuration space interpolation."""
        traj = []
        q1 = np.array(q1)
        q2 = np.array(q2)
        for i in range(steps + 1):
            t = i / steps
            q = (1 - t) * q1 + t * q2
            traj.append(q.tolist())
        return traj
    
    def go_home(self) -> bool:
        """Return to home configuration (skip if already near home)."""
        if not self.env:
            return False

        q_start = np.array(self.env.get_robot_conf(), dtype=float)
        q_home = np.array(self.env.get_home_conf(), dtype=float)

        max_joint_delta = float(np.max(np.abs(q_start - q_home)))
        if max_joint_delta <= self.config.return_home_joint_tolerance:
            print(f"  [ExecutorV2] Already at Home (max joint delta={max_joint_delta:.6f}), skipping.")
            return True

        print(f"  [ExecutorV2] Returning to Home... (max joint delta={max_joint_delta:.6f})")
        traj = self._interpolate_path(
            q_start,
            q_home,
            steps=max(2, int(self.config.home_interp_waypoints))
        )

        if traj:
            # Avoid double interpolation for homing
            self._execute_trajectory(traj, steps=1)
        else:
            self.env.set_robot_conf(q_home.tolist())
            self._step_sim(2)

        return True
    
    # =========================================================================
    # RETRY-ENABLED EXECUTION WRAPPER
    # =========================================================================
    
    def _execute_with_retry(self, object_name: str, target_region: str, 
                            use_cupboard_method: bool = False) -> Tuple[bool, str]:
        """
        Execute pick+place with retry logic for recoverable failures.
        
        Before each attempt:
        1. Query current object position (handles knocked-over objects)
        2. Execute with fresh motion planning
        3. Validate result
        4. Retry if PLACEMENT_FAILED or GRASP_FAILED
        
        Returns:
            (success, message)
        """
        last_error = ""
        
        for attempt in range(self.MAX_RETRIES + 1):
            # Query current object state (position/orientation)
            obj = self.env.get_object(object_name)
            if obj is None:
                return False, f"Object '{object_name}' not found"
            
            pos_before = list(obj.get_position())
            ori_before = list(obj.get_orientation())
            
            if attempt > 0:
                print(f"  [ExecutorV2] 🔄 RETRY {attempt}/{self.MAX_RETRIES} for {object_name}")
                print(f"  [ExecutorV2] Current position: {pos_before}")
                print(f"  [ExecutorV2] Current orientation: {ori_before}")
                # Return to home before retry
                self.go_home()
            
            # Execute the pick+place
            if use_cupboard_method:
                success, msg = self._execute_cupboard_pick_place(object_name, target_region)
            else:
                success, msg = self._execute_pick_place_pddl(object_name, target_region)
            
            # If execution failed, check if it's retryable
            if not success:
                last_error = msg
                # Check if it's a retryable error
                is_retryable = any(rf in msg.lower() for rf in ['no pddl plan', 'ik', 'motion'])
                if is_retryable and attempt < self.MAX_RETRIES:
                    print(f"  [ExecutorV2] ⚠️ Execution failed: {msg}")
                    continue
                else:
                    return False, msg
            
            # Post-validation
            if self.monitor:
                feedback = self.monitor.validate_pick_place(object_name, target_region, pos_before)
                self.monitor.log_feedback(feedback)
                
                if not feedback.success:
                    last_error = feedback.message
                    failure_type = feedback.failure_type.value if hasattr(feedback.failure_type, 'value') else str(feedback.failure_type)
                    
                    # Check if this is a retryable failure
                    is_retryable = failure_type.lower() in ['placement_failed', 'grasp_failed']
                    
                    if is_retryable and attempt < self.MAX_RETRIES:
                        print(f"  [ExecutorV2] ⚠️ Validation failed ({failure_type}): {feedback.message}")
                        print(f"  [ExecutorV2] Object may have been knocked/dropped - will retry with fresh planning")
                        continue
                    else:
                        return False, feedback.message
            
            # Success!
            return True, "Success"
        
        return False, f"Max retries exceeded. Last error: {last_error}"
    
    # =========================================================================
    # ACTION EXECUTORS (PDDL-based)
    # =========================================================================
    
    def _execute_pick_place_pddl(self, object_name: str, target_region: str) -> Tuple[bool, str]:
        """Execute standard pick and place using PDDL planning."""
        # Load PDDL files if needed
        self._load_pddl_files()
        
        env = self.env
        home_q = env.get_home_conf()
        env.set_robot_conf(home_q)
        for _ in range(10):
            self._step_sim()

        env.set_target_region(target_region)

        obj = env.get_object(object_name)
        if obj is None:
            return False, f"Object '{object_name}' not found"

        obj.set_dynamic(False)
        initial_pose = obj.get_pose()

        q_home_tuple = tuple(home_q)
        pose_tuple = tuple(initial_pose)

        init = [
            ('conf', q_home_tuple),
            ('at-conf', q_home_tuple),
            ('hand-empty',),
            ('movable', object_name),
            ('pose', pose_tuple),
            ('at-pose', object_name, pose_tuple),
            ('region', target_region),
        ]

        goal = And(('hand-empty',), ('in-region', object_name, target_region))

        problem = PDDLProblem(
            domain_pddl=self.domain_pddl,
            constant_map={},
            stream_pddl=self.stream_pddl,
            stream_map=self._get_stream_map(),
            init=init,
            goal=goal,
        )

        solution = solve(problem, algorithm='adaptive', verbose=False, max_time=60)
        plan, cost, evaluations = solution

        if not plan:
            return False, "No PDDL plan found"

        # Execute plan
        for action in plan:
            if action.name == 'move':
                q1, q2, traj = action.args
                self._execute_trajectory(traj)

            elif action.name == 'pick':
                o, p, g, q1, q2, traj_tuple = action.args
                segments = _normalize_segments(traj_tuple)
                if not segments:
                    return False, "Pick has empty trajectory"

                target_obj = env.get_object(o)
                grasp_idx = _pick_grasp_segment_index(env, target_obj, segments)

                for seg in segments[:grasp_idx + 1]:
                    self._execute_trajectory(seg)

                target_obj.set_dynamic(True)
                env.gripper.actuate(0.0, 0.1)
                for _ in range(10):
                    self._step_sim()
                env.gripper.grasp(target_obj)

                for seg in segments[grasp_idx + 1:]:
                    self._execute_trajectory(seg)

            elif action.name == 'place':
                o, p, g, r, q1, q2, traj_tuple = action.args
                segments = _normalize_segments(traj_tuple)
                if not segments:
                    return False, "Place has empty trajectory"

                release_idx = _place_release_segment_index(env, p, segments)

                for seg in segments[:release_idx + 1]:
                    self._execute_trajectory(seg)

                target_obj = env.get_object(o)
                env.gripper.release()
                target_obj.set_dynamic(True)

                hold_q = env.get_robot_conf()
                env.gripper.actuate(1.0, velocity=0.2)
                for _ in range(60):
                    env.set_robot_conf(hold_q)
                    self._step_sim()

                if len(segments) > release_idx + 1:
                    for seg in segments[release_idx + 1:]:
                        self._execute_trajectory(seg)

        return True, "Success"
    
    def _execute_cupboard_pick_place(self, object_name: str, target_region: str) -> Tuple[bool, str]:
        """Execute cupboard pick with horizontal grasp and place."""
        env = self.env
        home_q = env.get_home_conf()
        env.set_robot_conf(home_q)
        for _ in range(10):
            self._step_sim()

        mug = env.get_object(object_name)
        if mug is None:
            return False, f"Object '{object_name}' not found"

        mug.set_dynamic(False)
        pose = mug.get_pose()

        # Find horizontal hover configuration
        hover_dists = [0.35, 0.30, 0.40]
        q_hover = None
        successful_grasp_quat = None
        original_conf = env.get_robot_conf()

        base_ry = np.pi / 2
        grasp_quats = [
            quaternion_from_euler(0, base_ry, 0),
            quaternion_from_euler(np.pi, base_ry, 0),
        ]

        target_z = pose[2]
        grasp_depth_offset = 0.03
        grasp_pos = None
        hover_pos = None

        for h_dist in hover_dists:
            if q_hover is not None:
                break
            hover_pos = [pose[0] - h_dist, pose[1], target_z]
            grasp_pos = [pose[0] + grasp_depth_offset, pose[1], target_z]

            for grasp_rot in grasp_quats:
                path_configs = env.robot.solve_ik_via_sampling(
                    hover_pos, quaternion=grasp_rot, max_configs=50, max_time_ms=1000, ignore_collisions=True
                )
                if path_configs is not None and len(path_configs) > 0:
                    for q in path_configs:
                        env.set_robot_conf(q)
                        if env.robot.check_collision():
                            continue
                        try:
                            path_check = env.robot.get_linear_path(
                                position=grasp_pos, quaternion=grasp_rot, steps=20, ignore_collisions=True
                            )
                            if path_check:
                                q_hover = q
                                successful_grasp_quat = grasp_rot
                                break
                        except:
                            pass
                if q_hover is not None:
                    break

        if q_hover is None:
            env.set_robot_conf(original_conf)
            return False, "Could not find valid horizontal hover configuration"

        env.set_robot_conf(original_conf)

        # Move to hover
        traj_to_hover = env._interpolate_joint_path(home_q, q_hover, steps=100, check_collisions=False)
        if traj_to_hover:
            self._execute_trajectory(traj_to_hover, steps=10)

        env.gripper.release()
        for _ in range(30):
            self._step_sim()

        # Approach to grasp
        path_approach = env.robot.get_linear_path(
            position=grasp_pos, quaternion=successful_grasp_quat, steps=200, ignore_collisions=True
        )
        if path_approach:
            traj_approach = path_approach._path_points.reshape(-1, 7).tolist()
            for conf in traj_approach:
                env.set_robot_conf(conf)
                self._step_sim()

        # Grasp
        env.gripper.actuate(0.0, 0.1)
        for _ in range(50):
            self._step_sim()

        # Track mug
        gripper_tip = env.robot.get_tip()
        tip_pos = np.array(gripper_tip.get_position())
        mug_current_pos = np.array(mug.get_position())
        mug_offset = mug_current_pos - tip_pos

        # Retrieve
        retrieve_pos = list(hover_pos)
        retrieve_pos[2] += 0.02
        try:
            path_retrieve = env.robot.get_linear_path(
                position=retrieve_pos, quaternion=successful_grasp_quat, steps=200, ignore_collisions=True
            )
        except:
            path_retrieve = None

        if path_retrieve:
            traj_retrieve = path_retrieve._path_points.reshape(-1, 7).tolist()
            for conf in traj_retrieve:
                env.set_robot_conf(conf)
                self._step_sim()
                new_tip_pos = np.array(gripper_tip.get_position())
                mug.set_position((new_tip_pos + mug_offset).tolist())
        else:
            traj_fallback = env._interpolate_joint_path(env.get_robot_conf(), q_hover, steps=100, check_collisions=False)
            if traj_fallback:
                for conf in traj_fallback:
                    env.set_robot_conf(conf)
                    self._step_sim()
                    new_tip_pos = np.array(gripper_tip.get_position())
                    mug.set_position((new_tip_pos + mug_offset).tolist())

        # Place
        try:
            place_pose = env.find_best_placement(mug, target_region)
        except:
            place_pose = [0.0, 0.3, 0.77, 0, 0, 0, 1]

        min_x, max_x, min_y, max_y, min_z, max_z = mug.get_bounding_box()
        mug_top_z = max_z

        hover_z = place_pose[2] + mug_top_z + 0.12
        place_z = place_pose[2] + 0.015

        place_hover_pos = [place_pose[0], place_pose[1], hover_z]
        place_pos = [place_pose[0], place_pose[1], place_z]

        place_quats = [quaternion_from_euler(np.pi, 0, angle) for angle in np.linspace(0, 2 * np.pi, 16)]

        q_place_hover = None
        successful_place_quat = None

        for place_quat in place_quats:
            path_configs = env.robot.solve_ik_via_sampling(
                place_hover_pos, quaternion=place_quat, max_configs=20, max_time_ms=500, ignore_collisions=True
            )
            if path_configs is not None and len(path_configs) > 0:
                for q in path_configs:
                    env.set_robot_conf(q)
                    if not env.robot.check_collision():
                        place_configs = env.robot.solve_ik_via_sampling(
                            place_pos, quaternion=place_quat, max_configs=5, max_time_ms=200, ignore_collisions=True
                        )
                        if place_configs is not None and len(place_configs) > 0:
                            q_place_hover = q
                            successful_place_quat = place_quat
                            break
            if q_place_hover is not None:
                break

        if q_place_hover is None:
            return False, "Could not find place hover configuration"

        # Move to place hover
        current_conf = env.get_robot_conf()
        traj_to_place = env._interpolate_joint_path(current_conf, q_place_hover, steps=150, check_collisions=False)
        if traj_to_place:
            for conf in traj_to_place:
                env.set_robot_conf(conf)
                self._step_sim()
                new_tip_pos = np.array(gripper_tip.get_position())
                mug.set_position((new_tip_pos + mug_offset).tolist())

        # Lower to place
        try:
            path_lower = env.robot.get_linear_path(
                position=place_pos, quaternion=successful_place_quat, steps=100, ignore_collisions=True
            )
        except:
            path_lower = None

        if path_lower:
            traj_lower = path_lower._path_points.reshape(-1, 7).tolist()
            for conf in traj_lower:
                env.set_robot_conf(conf)
                self._step_sim()
                new_tip_pos = np.array(gripper_tip.get_position())
                mug.set_position((new_tip_pos + mug_offset).tolist())

        # Release
        env.gripper.actuate(1.0, 0.1)
        for _ in range(30):
            self._step_sim()

        mug.set_dynamic(True)
        for _ in range(50):
            self._step_sim()

        # Lift
        try:
            lift_pos = [place_pos[0], place_pos[1], place_pos[2] + 0.15]
            path_lift = env.robot.get_linear_path(
                position=lift_pos, quaternion=successful_place_quat, steps=50, ignore_collisions=True
            )
            if path_lift:
                traj_lift = path_lift._path_points.reshape(-1, 7).tolist()
                self._execute_trajectory(traj_lift)
        except:
            pass

        return True, "Success"
    
    def _execute_open_lid(self) -> Tuple[bool, str]:
        """Open the box lid by sliding."""
        env = self.env
        home_q = env.get_home_conf()
        env.set_robot_conf(home_q)
        for _ in range(10):
            self._step_sim()

        obj = env.get_object('box_lid')
        if obj is None:
            return False, "box_lid not found"

        env.gripper.actuate(1.0, 0.1)
        for _ in range(10):
            self._step_sim()

        try:
            grasp_quat, q_hover, q_grasp, (traj_hover_to_center, traj_center_to_edge) = env.compute_lid_grasp_trajectory(obj)
            traj_approach = traj_hover_to_center + traj_center_to_edge

            path_to_hover = env.compute_motion_plan(home_q, q_hover)

            if path_to_hover:
                self._execute_trajectory(path_to_hover, steps=5)
                self._execute_trajectory(traj_approach, steps=5)

                env.gripper.actuate(0.0, 0.1)
                for _ in range(30):
                    self._step_sim()
                env.gripper.grasp(obj)

                q_open_start, q_open_end, traj_open, traj_return = env.compute_slide_lid_trajectory(
                    obj, grasp_quat, initial_conf=q_grasp
                )

                if traj_open:
                    self._execute_trajectory(traj_open, steps=5)

                    env.gripper.release()
                    env.gripper.actuate(1.0, 0.1)
                    for _ in range(50):
                        self._step_sim()

                    self._execute_trajectory(traj_return, steps=5)
                    traj_retreat = traj_hover_to_center[::-1]
                    self._execute_trajectory(traj_retreat, steps=5)

                    for _ in range(50):
                        self._step_sim()

                    return True, "Success"
                else:
                    return False, "Failed to compute slide trajectory"
            else:
                return False, "Could not plan motion to hover"

        except Exception as e:
            return False, f"Error: {e}"
    
    def _reset_active_transfer(self):
        self._active_transfer = None

    def _future_place_region(self, skeleton: List[ActionSkeleton], action_idx: int, obj_vlm: str) -> Optional[str]:
        for future_action in skeleton[action_idx + 1:]:
            if future_action.action_name == 'place' and len(future_action.args) >= 2:
                if future_action.args[0] == obj_vlm:
                    return future_action.args[1]
        return None

    def _normalize_move_args(
        self,
        action: ActionSkeleton,
        skeleton: Optional[List[ActionSkeleton]] = None,
        action_idx: Optional[int] = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        args = list(action.args)
        if len(args) == 1:
            obj_vlm = args[0]
            region_vlm = None
            if skeleton is not None and action_idx is not None:
                region_vlm = self._future_place_region(skeleton, action_idx, obj_vlm)
            if region_vlm is None and self._active_transfer is not None:
                obj_env = VLM_TO_ENV_OBJECT.get(obj_vlm, obj_vlm)
                if self._active_transfer.object_name == obj_env:
                    region_vlm = self._active_transfer.target_region
            if region_vlm is None:
                return None, None, f"move({obj_vlm}) requires a later matching place({obj_vlm}, region)"
        elif len(args) == 2:
            a0, a1 = args
            if a0 in VLM_TO_ENV_REGION and a1 not in VLM_TO_ENV_REGION:
                obj_vlm, region_vlm = a1, a0
            else:
                obj_vlm, region_vlm = a0, a1
        else:
            return None, None, f"move expects 1 or 2 args, got {len(args)}"

        obj_env = VLM_TO_ENV_OBJECT.get(obj_vlm, obj_vlm)
        region_env = VLM_TO_ENV_REGION.get(region_vlm, region_vlm)
        return obj_env, region_env, None

    def _ensure_active_transfer(
        self,
        object_name: str,
        target_region: str,
        task_name: str = "",
    ) -> Tuple[Optional[object], Optional[str]]:
        if self.env is None:
            return None, "Environment not set"

        if self._active_transfer is None:
            gt = _ground_truth_orchestrator()
            self._active_transfer = gt.create_primitive_transfer_executor(
                self.env,
                object_name,
                target_region,
                task_name=task_name,
            )
        elif (
            self._active_transfer.object_name != object_name
            or self._active_transfer.target_region != target_region
        ):
            return None, (
                f"Active transfer is for '{self._active_transfer.object_name}' -> "
                f"'{self._active_transfer.target_region}', not '{object_name}' -> '{target_region}'"
            )
        return self._active_transfer, None

    # =========================================================================
    # SINGLE ACTION EXECUTION (for demo/testing)
    # =========================================================================
    
    def execute_single_action(self, action: ActionSkeleton) -> ExecutionResult:
        """
        Execute a single primitive action or open-lid action.
        """
        try:
            if action.action_name in ['open-lid', 'open_lid']:
                if self._active_transfer is not None:
                    msg = "Cannot open-lid while a move/pick/place transfer is in progress"
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        completed_actions=0,
                        total_actions=1,
                        current_action=action,
                        error_message=msg,
                    )
                success, msg = self._execute_open_lid()
            elif action.action_name == 'move':
                obj_env, region_env, error = self._normalize_move_args(action)
                if error:
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        completed_actions=0,
                        total_actions=1,
                        current_action=action,
                        error_message=error,
                    )
                transfer, error = self._ensure_active_transfer(
                    obj_env,
                    region_env,
                    task_name=f"{obj_env} -> {region_env}",
                )
                if error:
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        completed_actions=0,
                        total_actions=1,
                        current_action=action,
                        error_message=error,
                    )
                success, msg = transfer.execute_next('move')
            elif action.action_name == 'pick':
                obj_vlm = action.args[0]
                obj_env = VLM_TO_ENV_OBJECT.get(obj_vlm, obj_vlm)
                if self._active_transfer is None:
                    msg = 'Standalone pick requires a prior move(object, region) or execute_plan lookahead'
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        completed_actions=0,
                        total_actions=1,
                        current_action=action,
                        error_message=msg,
                    )
                if self._active_transfer.object_name != obj_env:
                    msg = f"Active transfer is for '{self._active_transfer.object_name}', not '{obj_env}'"
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        completed_actions=0,
                        total_actions=1,
                        current_action=action,
                        error_message=msg,
                    )
                success, msg = self._active_transfer.execute_next('pick')
            elif action.action_name == 'place':
                if self._active_transfer is None:
                    msg = 'Standalone place requires an active move/pick transfer'
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        completed_actions=0,
                        total_actions=1,
                        current_action=action,
                        error_message=msg,
                    )
                obj_vlm = action.args[0]
                region_vlm = action.args[1] if len(action.args) > 1 else 'placement_boundary'
                obj_env = VLM_TO_ENV_OBJECT.get(obj_vlm, obj_vlm)
                region_env = VLM_TO_ENV_REGION.get(region_vlm, region_vlm)
                if self._active_transfer.object_name != obj_env or self._active_transfer.target_region != region_env:
                    msg = (
                        f"Active transfer is for '{self._active_transfer.object_name}' -> "
                        f"'{self._active_transfer.target_region}', not '{obj_env}' -> '{region_env}'"
                    )
                    return ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        completed_actions=0,
                        total_actions=1,
                        current_action=action,
                        error_message=msg,
                    )
                success, msg = self._active_transfer.execute_next('place')
                if success and self._active_transfer.expected_next_action() is None:
                    self._reset_active_transfer()
            else:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    completed_actions=0,
                    total_actions=1,
                    current_action=action,
                    error_message=f"Unknown action: {action.action_name}",
                )

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS if success else ExecutionStatus.FAILED,
                completed_actions=1 if success else 0,
                total_actions=1,
                current_action=action,
                error_message=None if success else msg,
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                completed_actions=0,
                total_actions=1,
                current_action=action,
                error_message=str(e),
            )


    def execute_pick_place_pair(self, pick_action: ActionSkeleton, place_action: ActionSkeleton) -> ExecutionResult:
        """
        Execute a pick+place pair together with retry logic.
        """
        if pick_action.action_name != 'pick' or place_action.action_name != 'place':
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                completed_actions=0,
                total_actions=2,
                error_message="Expected pick+place pair"
            )
        
        obj_vlm = pick_action.args[0]
        region_vlm = place_action.args[1] if len(place_action.args) > 1 else 'placement_boundary'
        
        obj_env = VLM_TO_ENV_OBJECT.get(obj_vlm, obj_vlm)
        region_env = VLM_TO_ENV_REGION.get(region_vlm, region_vlm)
        
        # Determine which method to use based on WHERE we're picking FROM:
        # - mug3 (mug_cupboard): picked FROM cupboard with horizontal grasp
        # - Everything else (groceries, other mugs): standard vertical grasp via PDDL
        use_cupboard_method = (obj_env == 'mug3')
        
        # Execute with retry logic (handles knocked-over objects, failed grasps, etc.)
        success, msg = self._execute_with_retry(obj_env, region_env, use_cupboard_method)
        
        return ExecutionResult(
            status=ExecutionStatus.SUCCESS if success else ExecutionStatus.FAILED,
            completed_actions=2 if success else 0,
            total_actions=2,
            error_message=None if success else msg
        )
    
    # =========================================================================
    # PLAN EXECUTION
    # =========================================================================
    
    def execute_plan(self, skeleton: List[ActionSkeleton]) -> ExecutionResult:
        """
        Execute the action skeleton using primitive move/pick/place/open-lid actions.
        Explicit move actions run stage-by-stage; legacy pick/place pairs remain supported.
        """
        self.current_plan = skeleton
        self.current_action_idx = 0
        self.is_interrupted = False
        self.executed_actions = []
        self._reset_active_transfer()

        print(f"\n[ExecutorV2] Starting plan execution ({len(skeleton)} actions)")
        print(f"[ExecutorV2] Plan: {[str(a) for a in skeleton]}")

        self.go_home()

        while self.current_action_idx < len(skeleton):
            if self.is_interrupted:
                return ExecutionResult(
                    status=ExecutionStatus.INTERRUPTED,
                    completed_actions=self.current_action_idx,
                    total_actions=len(skeleton),
                    failure_reason="Execution interrupted",
                )

            action = skeleton[self.current_action_idx]
            print(f"\n[ExecutorV2] Processing Action {self.current_action_idx + 1}/{len(skeleton)}: {action}")

            try:
                recently_executed: List[ActionSkeleton] = []

                if action.action_name in ['open-lid', 'open_lid']:
                    if self._active_transfer is not None:
                        msg = 'Cannot open-lid while a move/pick/place transfer is in progress'
                        return ExecutionResult(
                            status=ExecutionStatus.FAILED,
                            completed_actions=self.current_action_idx,
                            total_actions=len(skeleton),
                            current_action=action,
                            error_message=msg,
                            failure_reason=msg,
                        )

                    print("  [ExecutorV2] Executing: open_lid")
                    if self.monitor:
                        precheck = self.monitor.check_open_lid_preconditions()
                        if not precheck.success:
                            self.monitor.log_feedback(precheck)
                            return ExecutionResult(
                                status=ExecutionStatus.FAILED,
                                completed_actions=self.current_action_idx,
                                total_actions=len(skeleton),
                                current_action=action,
                                error_message=precheck.message,
                                failure_reason=precheck.message,
                            )

                    lid = self.env.get_object('box_lid')
                    pos_before = list(lid.get_position()) if lid else [0, 0, 0]
                    success, msg = self._execute_open_lid()

                    if success and self.monitor:
                        feedback = self.monitor.validate_open_lid(pos_before)
                        self.monitor.log_feedback(feedback)
                        if not feedback.success:
                            success = False
                            msg = feedback.message

                    if not success:
                        print(f"  [ExecutorV2] ✗ FAILED: {msg}")
                        return ExecutionResult(
                            status=ExecutionStatus.FAILED,
                            completed_actions=self.current_action_idx,
                            total_actions=len(skeleton),
                            current_action=action,
                            error_message=msg,
                            failure_reason=msg,
                        )

                    print("  [ExecutorV2] ✓ SUCCESS")
                    self.executed_actions.append(action)
                    recently_executed = [action]
                    self.current_action_idx += 1

                elif action.action_name == 'move':
                    obj_env, region_env, error = self._normalize_move_args(
                        action,
                        skeleton=skeleton,
                        action_idx=self.current_action_idx,
                    )
                    if error:
                        return ExecutionResult(
                            status=ExecutionStatus.FAILED,
                            completed_actions=self.current_action_idx,
                            total_actions=len(skeleton),
                            current_action=action,
                            error_message=error,
                            failure_reason=error,
                        )

                    transfer, error = self._ensure_active_transfer(
                        obj_env,
                        region_env,
                        task_name=f"{obj_env} -> {region_env}",
                    )
                    if error:
                        return ExecutionResult(
                            status=ExecutionStatus.FAILED,
                            completed_actions=self.current_action_idx,
                            total_actions=len(skeleton),
                            current_action=action,
                            error_message=error,
                            failure_reason=error,
                        )

                    success, msg = transfer.execute_next('move')
                    if not success:
                        return ExecutionResult(
                            status=ExecutionStatus.FAILED,
                            completed_actions=self.current_action_idx,
                            total_actions=len(skeleton),
                            current_action=action,
                            error_message=msg,
                            failure_reason=msg,
                        )

                    print(f"  [ExecutorV2] ✓ SUCCESS: move for '{obj_env}'")
                    self.executed_actions.append(action)
                    recently_executed = [action]
                    self.current_action_idx += 1

                elif action.action_name == 'pick' and self._active_transfer is not None:
                    obj_vlm = action.args[0]
                    obj_env = VLM_TO_ENV_OBJECT.get(obj_vlm, obj_vlm)
                    if self._active_transfer.object_name != obj_env:
                        msg = (
                            f"Active transfer is for '{self._active_transfer.object_name}', "
                            f"not '{obj_env}'"
                        )
                        return ExecutionResult(
                            status=ExecutionStatus.FAILED,
                            completed_actions=self.current_action_idx,
                            total_actions=len(skeleton),
                            current_action=action,
                            error_message=msg,
                            failure_reason=msg,
                        )

                    success, msg = self._active_transfer.execute_next('pick')
                    if not success:
                        return ExecutionResult(
                            status=ExecutionStatus.FAILED,
                            completed_actions=self.current_action_idx,
                            total_actions=len(skeleton),
                            current_action=action,
                            error_message=msg,
                            failure_reason=msg,
                        )

                    print(f"  [ExecutorV2] ✓ SUCCESS: pick '{obj_env}'")
                    self.executed_actions.append(action)
                    recently_executed = [action]
                    self.current_action_idx += 1

                elif action.action_name == 'place' and self._active_transfer is not None:
                    obj_vlm = action.args[0]
                    region_vlm = action.args[1] if len(action.args) > 1 else 'placement_boundary'
                    obj_env = VLM_TO_ENV_OBJECT.get(obj_vlm, obj_vlm)
                    region_env = VLM_TO_ENV_REGION.get(region_vlm, region_vlm)
                    if (
                        self._active_transfer.object_name != obj_env
                        or self._active_transfer.target_region != region_env
                    ):
                        msg = (
                            f"Active transfer is for '{self._active_transfer.object_name}' -> "
                            f"'{self._active_transfer.target_region}', not '{obj_env}' -> '{region_env}'"
                        )
                        return ExecutionResult(
                            status=ExecutionStatus.FAILED,
                            completed_actions=self.current_action_idx,
                            total_actions=len(skeleton),
                            current_action=action,
                            error_message=msg,
                            failure_reason=msg,
                        )

                    success, msg = self._active_transfer.execute_next('place')
                    if not success:
                        return ExecutionResult(
                            status=ExecutionStatus.FAILED,
                            completed_actions=self.current_action_idx,
                            total_actions=len(skeleton),
                            current_action=action,
                            error_message=msg,
                            failure_reason=msg,
                        )

                    print(f"  [ExecutorV2] ✓ SUCCESS: place '{obj_env}' -> '{region_env}'")
                    self.executed_actions.append(action)
                    recently_executed = [action]
                    self.current_action_idx += 1
                    if self._active_transfer.expected_next_action() is None:
                        self._reset_active_transfer()

                elif action.action_name == 'pick':
                    if self.current_action_idx + 1 >= len(skeleton):
                        raise Exception("Found 'pick' without following 'place'")

                    next_action = skeleton[self.current_action_idx + 1]
                    if next_action.action_name != 'place':
                        raise Exception(f"Expected 'place' after 'pick', found {next_action.action_name}")

                    vlm_obj_name = action.args[0]
                    target_obj = next_action.args[0]
                    vlm_region = next_action.args[1]

                    if vlm_obj_name != target_obj:
                        raise Exception(f"Pick object ({vlm_obj_name}) does not match Place object ({target_obj})")

                    env_obj_name = VLM_TO_ENV_OBJECT.get(vlm_obj_name, vlm_obj_name)
                    env_region = VLM_TO_ENV_REGION.get(vlm_region, vlm_region)

                    print(f"  [ExecutorV2] Pick+Place: '{vlm_obj_name}' -> '{vlm_region}'")
                    print(f"  [ExecutorV2] Mapped to: '{env_obj_name}' -> '{env_region}'")

                    if self.monitor:
                        precheck = self.monitor.check_pick_preconditions(env_obj_name)
                        if not precheck.success:
                            self.monitor.log_feedback(precheck)
                            return ExecutionResult(
                                status=ExecutionStatus.FAILED,
                                completed_actions=self.current_action_idx,
                                total_actions=len(skeleton),
                                current_action=action,
                                error_message=precheck.message,
                                failure_reason=precheck.message,
                            )

                    use_cupboard_method = (vlm_obj_name in CUPBOARD_OBJECTS or env_obj_name == 'mug3')
                    if use_cupboard_method:
                        print("  [ExecutorV2] Using: cupboard_pick_place (horizontal grasp)")
                    else:
                        print("  [ExecutorV2] Using: PDDL pick_place")

                    success, msg = self._execute_with_retry(env_obj_name, env_region, use_cupboard_method)
                    if not success:
                        print(f"  [ExecutorV2] ✗ FAILED: {msg}")
                        return ExecutionResult(
                            status=ExecutionStatus.FAILED,
                            completed_actions=self.current_action_idx,
                            total_actions=len(skeleton),
                            current_action=action,
                            error_message=msg,
                            failure_reason=msg,
                        )

                    print("  [ExecutorV2] ✓ SUCCESS")
                    self.executed_actions.append(action)
                    self.executed_actions.append(next_action)
                    recently_executed = [action, next_action]
                    self.current_action_idx += 2

                else:
                    if action.action_name == 'place':
                        raise Exception("Orphaned 'place' action found (not preceded by 'pick' or move/pick/move')")
                    raise Exception(f"Unknown action type: {action.action_name}")

                if self.config.return_home_after_each_action:
                    self.go_home()

                import time
                time.sleep(self.config.pause_between_actions)

                if self.post_action_callback and recently_executed:
                    try:
                        interrupt_reason = self.post_action_callback(
                            recently_executed,
                            self.current_action_idx,
                            len(skeleton),
                        )
                    except Exception as cb_error:
                        print(f"  [ExecutorV2] Post-action callback error: {cb_error}")
                        interrupt_reason = None

                    if interrupt_reason:
                        print(f"  [ExecutorV2] ⚠️ Interrupt requested: {interrupt_reason}")
                        return ExecutionResult(
                            status=ExecutionStatus.INTERRUPTED,
                            completed_actions=self.current_action_idx,
                            total_actions=len(skeleton),
                            current_action=None,
                            error_message=interrupt_reason,
                            failure_reason=interrupt_reason,
                        )

            except Exception as e:
                import traceback
                traceback.print_exc()
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    completed_actions=self.current_action_idx,
                    total_actions=len(skeleton),
                    current_action=action,
                    error_message=str(e),
                    failure_reason=str(e),
                )

        if self._active_transfer is not None and self._active_transfer.expected_next_action() is not None:
            msg = (
                'Plan ended before completing transfer; '
                f"expected '{self._active_transfer.expected_next_action()}' next"
            )
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                completed_actions=self.current_action_idx,
                total_actions=len(skeleton),
                current_action=None,
                error_message=msg,
                failure_reason=msg,
            )

        print(f"\n[ExecutorV2] Plan completed successfully!")
        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            completed_actions=len(skeleton),
            total_actions=len(skeleton),
            trajectory_executed=True,
        )


    def get_remaining_plan(self) -> List[ActionSkeleton]:
        return self.current_plan[self.current_action_idx:]
    
    def get_executed_plan(self) -> List[ActionSkeleton]:
        return self.executed_actions.copy()
    
    def get_failure_summary(self) -> Dict:
        """Get failure summary for replanning context."""
        if self.monitor:
            return self.monitor.get_failure_summary()
        return {"has_failures": False}


# Backward compatibility: also export as VLMExecutor
VLMExecutor = VLMExecutorV2

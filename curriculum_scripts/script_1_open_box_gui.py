import os
import sys
import numpy as np

def _configure_qt():
    """Keep Qt quiet and point it at CoppeliaSim's plugins without forcing offscreen."""
    # GUI MODE: Set headless to 0
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

# Add pddlstream to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pddlstream'))

from pddlstream.language.constants import PDDLProblem, And
from pddlstream.algorithms.meta import solve
from pddlstream.utils import read

# Set HEADLESS env var to False BEFORE importing streams
os.environ["HEADLESS"] = "False"

# Import ENV from streams to share the instance
from rlbench_kitchen_streams import ENV, get_stream_map

import argparse

def execute_trajectory(env, traj):
    if not traj: return
    full_traj = []
    for i in range(len(traj)-1):
        start = np.array(traj[i])
        end = np.array(traj[i+1])
        steps = 30 # Interpolation steps
        for t in np.linspace(0, 1, steps, endpoint=False):
            full_traj.append((1-t)*start + t*end)
    full_traj.append(traj[-1])
    
    for conf in full_traj:
        env.set_robot_conf(conf)
        env.pr.step()

def main():
    parser = argparse.ArgumentParser(description='Open Box Task')
    args = parser.parse_args()

    # Use the shared ENV
    env = ENV
    pr = env.pr

    print("Settling physics...")
    for _ in range(50):
        pr.step()

    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    env.save_conf("home", home_q)
    for _ in range(10):
        pr.step()

    # --- CONFIGURATION ---
    target_object_name = 'box_lid'
    
    print(f"Target Object: {target_object_name}")
    
    obj = env.get_object(target_object_name)
    if obj is None:
        print(f"ERROR: {target_object_name} not found.")
        pr.stop()
        pr.shutdown()
        return

    # Force static for stability before picking
    # obj.set_dynamic(False) 
    # For lid, it might need to be dynamic but constrained? 
    # Or we set it dynamic when we grasp.
    
    # --- STEP 1: MANUAL HOVER TEST ---
    print("\n--- STEP 1: Testing Hover Motion ---")
    try:
        # 1. Compute the grasp details to find a valid hover config
        print("Computing valid grasp and hover configuration...")
        grasp, q_hover, q_grasp, traj_approach = env.compute_lid_grasp_trajectory(obj)
        
        # 2. Plan motion from Home to Hover
        print("Planning motion from Home to Hover...")
        path_to_hover = env.compute_motion_plan(home_q, q_hover)
        
        if path_to_hover:
            print("Path found! Executing motion to Hover...")
            execute_trajectory(env, path_to_hover)
            print("Reached Hover Position.")
            
            # Optional: Visualize the grasp approach (Hover -> Grasp)
            # print("Executing Approach (Hover -> Grasp)...")
            # execute_trajectory(env, traj_approach)
            
        else:
            print("ERROR: Could not plan motion from Home to Hover.")
            
    except Exception as e:
        print(f"ERROR during hover calculation: {e}")
        import traceback
        traceback.print_exc()
    
    print("Step 1 Complete. Stopping here for verification.")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            pr.step()
    except KeyboardInterrupt:
        pr.stop()
        pr.shutdown()
        return

    # --- PDDL PROBLEM ---
    print(f"Setting up PDDL problem: Open {target_object_name}...")
    
    directory = os.path.dirname(os.path.abspath(__file__))
    domain_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_domain.pddl'))
    stream_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_streams.pddl'))

    q_home_tuple = tuple(home_q)

    init = [
        ('conf', q_home_tuple),
        ('at-conf', q_home_tuple),
        ('hand-empty',),
        
        ('lid', target_object_name),
        ('closed', target_object_name),
    ]

    # Goal: Lid opened AND Robot back at home (retreat)
    goal = And(('hand-empty',), ('opened', target_object_name), ('at-conf', q_home_tuple))

    problem = PDDLProblem(
        domain_pddl=domain_pddl,
        constant_map={},
        stream_pddl=stream_pddl,
        stream_map=get_stream_map(),
        init=init,
        goal=goal,
    )

    print("Solving PDDL problem...")
    solution = solve(problem, algorithm='adaptive', verbose=True, max_time=60)
    plan, cost, evaluations = solution

    if plan:
        print(f"PDDL Plan found with cost: {cost}")
        for action in plan:
            print(f"Action: {action.name} {action.args}")
            
            if action.name == 'move':
                q1, q2, traj = action.args
                execute_trajectory(env, traj)
            
            elif action.name == 'grasp-lid':
                o, g, q1, q2, traj = action.args
                # traj is approach
                execute_trajectory(env, traj)
                
                # Grasp
                print(f"Grasping {o}...")
                target_obj = env.get_object(o)
                target_obj.set_dynamic(True) 
                
                env.gripper.actuate(0.0, 0.1)
                for _ in range(20): pr.step()
                env.gripper.grasp(target_obj)
                
            elif action.name == 'open-lid':
                o, g, q1, q2, traj = action.args
                # traj is circular open
                execute_trajectory(env, traj)
                
                # Release
                print(f"Releasing {o}...")
                target_obj = env.get_object(o)
                env.gripper.release()
                
                # Open gripper
                env.gripper.actuate(1.0, 0.2)
                for _ in range(30): pr.step()
                
                # Lift up slightly to clear?
                # The robot is at q2 (end of open).
                # The next action should be 'move' to home.
                # But we might want a small retreat from the lid first?
                # The planner will find a path from q2 to home.
                # If q2 is close to lid, sample-motion might fail or find a path.
                # Let's trust sample-motion.
                
    else:
        print("ERROR: No PDDL plan found!")

    # Keep open
    print("Sequence complete. Press Ctrl+C to close.")
    try:
        while True:
            pr.step()
    except KeyboardInterrupt:
        pass
    pr.stop()
    pr.shutdown()


if __name__ == "__main__":
    main()

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
    parser = argparse.ArgumentParser(description='Open Box Task - Grasp Test')
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

    # Ensure gripper is open initially
    env.gripper.actuate(1.0, 0.1)
    for _ in range(10): pr.step()

    # --- STEP 1: HOVER & GRASP TEST ---
    print("\n--- STEP 1: Testing Hover & Grasp Motion ---")
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
            
            # 3. Execute Approach (Hover -> Grasp)
            print("Executing Approach (Hover -> Grasp)...")
            execute_trajectory(env, traj_approach)
            print("Reached Grasp Position.")
            
            # 4. Close Gripper
            print("Closing Gripper...")
            # Close gripper (0.0 is closed, 1.0 is open)
            env.gripper.actuate(0.0, 0.1)
            
            # Wait for gripper to close
            for _ in range(30): 
                pr.step()
                
            # Attach object to gripper (Simulated Grasp)
            # Note: In RLBench/PyRep, we might need to explicitly call grasp if using the high-level gripper
            # But PandaGripper usually handles it if we use actuate?
            # Let's use the grasp method if available or just rely on physics/actuate
            # The env.gripper is a PandaGripper instance.
            # It has a grasp(obj) method which creates a dummy link.
            
            print("Attaching object...")
            env.gripper.grasp(obj)
            
            print("Grasp Complete.")
            
        else:
            print("ERROR: Could not plan motion from Home to Hover.")
            
    except Exception as e:
        print(f"ERROR during grasp calculation: {e}")
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

if __name__ == "__main__":
    main()

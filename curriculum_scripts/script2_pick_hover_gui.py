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

from pddlstream.language.constants import PDDLProblem
from pddlstream.algorithms.meta import solve
from pddlstream.utils import read

# Set HEADLESS env var to False BEFORE importing streams
os.environ["HEADLESS"] = "False"

# Import ENV from streams to share the instance
from rlbench_kitchen_streams import ENV, get_stream_map
# from video_recorder import VideoRecorder # Disable video recorder


def main():
    # Use the shared ENV
    env = ENV
    pr = env.pr

    # recorder = VideoRecorder(env) # Disable video recorder

    # SETTLE PHYSICS: Step simulation to let objects settle
    print("Settling physics...")
    for _ in range(50):
        pr.step()

    # Ensure we are at home and save it once
    home_q = env.get_home_conf()
    env.set_robot_conf(home_q)
    env.save_conf("home", home_q)
    for _ in range(10):
        pr.step()
        # recorder.record_step()

    # Get the mug object and its current pose
    mug = env.get_object("mug_box")
    if mug is None:
        print("ERROR: mug_box not found in scene.")
        # recorder.release()
        pr.stop()
        pr.shutdown()
        return

    # Force the mug to be static so it doesn't jitter/slide
    mug.set_dynamic(False)

    pose = mug.get_pose()  # [x, y, z, qx, qy, qz, qw]

    # --- PDDL PLANNING START ---
    print("Setting up PDDL problem for 'pick' action...")
    
    directory = os.path.dirname(os.path.abspath(__file__))
    domain_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_domain.pddl'))
    stream_pddl = read(os.path.join(directory, 'pddl/rlbench_kitchen_streams.pddl'))

    q_home_tuple = tuple(home_q)
    pose_tuple = tuple(pose)

    # Define PDDL problem
    # We want to pick the mug.
    # We provide home config and mug pose.
    init = [
        ('conf', q_home_tuple),
        ('at-conf', q_home_tuple),
        ('hand-empty',),
        ('movable', 'mug_box'),
        ('pose', pose_tuple),
        ('at-pose', 'mug_box', pose_tuple),
    ]

    goal = ('holding', 'mug_box')

    problem = PDDLProblem(
        domain_pddl=domain_pddl,
        constant_map={},
        stream_pddl=stream_pddl,
        stream_map=get_stream_map(),
        init=init,
        goal=goal,
    )

    print("Solving PDDL problem...")
    solution = solve(problem, algorithm='adaptive', verbose=False)
    plan, cost, evaluations = solution

    if plan:
        print(f"PDDL Plan found with cost: {cost}")
        for action in plan:
            print(f"Action: {action.name}")
            
            if action.name == 'move':
                # args: ?q1 ?q2 ?t
                q1, q2, traj = action.args
                print(f"Executing move via PDDL plan...")
                
                # Interpolate trajectory for smoother video
                full_traj = []
                # traj is a list of waypoints (tuples)
                for i in range(len(traj)-1):
                    start = np.array(traj[i])
                    end = np.array(traj[i+1])
                    steps = 60
                    for t in np.linspace(0, 1, steps, endpoint=False):
                        full_traj.append((1-t)*start + t*end)
                full_traj.append(traj[-1])

                print(f"DEBUG: Executing trajectory with {len(full_traj)} points.")
                if len(full_traj) > 0:
                    print(f"DEBUG: Start Config: {full_traj[0][:3]}...")
                    print(f"DEBUG: End Config:   {full_traj[-1][:3]}...")

                for conf in full_traj:
                    env.set_robot_conf(conf)
                    pr.step()
                    # recorder.record_step()
            
            elif action.name == 'pick':
                # args: ?o ?p ?g ?q1 ?q2 ?t
                # ?t is the full pick trajectory (approach -> grasp -> retreat)
                o, p, g, q1, q2, traj = action.args
                print(f"Executing pick via PDDL plan...")
                
                # The trajectory returned by sample-pick-kin is a list of configs
                # It includes approach, grasp, and retreat.
                # We need to execute it carefully.
                # Usually, the trajectory is just a sequence of configs.
                # But we also need to close the gripper at the right time.
                # The trajectory from compute_pick_trajectory is:
                # approach (linear) -> grasp config -> retreat (linear)
                # Wait, compute_pick_trajectory returns:
                # grasp, q_hover, q_hover_end, full_traj
                # full_traj = t2 + t3
                # t2 is hover -> grasp
                # t3 is grasp -> hover_end
                
                # So we should execute t2, close gripper, execute t3.
                # But the PDDL action just gives us one big trajectory ?t.
                # We need to split it or know when to grasp.
                # Since we know the structure (hover -> grasp -> hover), the grasp happens in the middle.
                # Actually, t2 ends at grasp config. t3 starts at grasp config.
                # So the middle point is the grasp config.
                
                # Let's find the split point.
                # The trajectory is a tuple of tuples.
                # We can assume the "deepest" point (lowest Z) is the grasp point?
                # Or simply, since we constructed it as t2 + t3, and t2 and t3 are roughly equal length?
                # No, let's look at how it was constructed in rlbench_kitchen_env.py:
                # t2 = get_configs(path2) (7 points usually)
                # t3 = get_configs(path3) (7 points usually)
                # full_traj = t2 + t3
                # So the grasp is at index len(t2)-1.
                
                # Since we don't have t2 length here easily, let's just assume the middle.
                # Or better, let's execute until we are close to the object, grasp, then continue.
                
                # Actually, let's just execute the whole thing, but pause in the middle to grasp.
                # The trajectory has 14 points (if 7+7).
                # The grasp is at index 6 (0-indexed) if t2 has 7 points.
                
                # Let's try to detect the grasp point by checking Z height of end effector?
                # Or just assume index len(traj)//2 is the grasp point.
                
                mid_idx = len(traj) // 2
                
                # Execute first half (Approach)
                approach_traj = traj[:mid_idx]
                
                full_approach = []
                for i in range(len(approach_traj)-1):
                    start = np.array(approach_traj[i])
                    end = np.array(approach_traj[i+1])
                    steps = 30
                    for t in np.linspace(0, 1, steps, endpoint=False):
                        full_approach.append((1-t)*start + t*end)
                full_approach.append(approach_traj[-1])
                
                print("Approaching...")
                for conf in full_approach:
                    env.set_robot_conf(conf)
                    pr.step()
                
                # GRASP
                print("Grasping...")
                # Make mug dynamic so we can pick it up
                mug.set_dynamic(True) 
                # Close gripper
                env.gripper.actuate(0.0, 0.1) # 0.0 is closed? 
                # Wait for gripper to close
                for _ in range(20):
                    pr.step()
                
                # Attach object if close enough (Simulated Grasp)
                # PyRep gripper has a grasp method
                env.gripper.grasp(mug)
                
                # Execute second half (Retreat)
                retreat_traj = traj[mid_idx:]
                
                full_retreat = []
                for i in range(len(retreat_traj)-1):
                    start = np.array(retreat_traj[i])
                    end = np.array(retreat_traj[i+1])
                    steps = 30
                    for t in np.linspace(0, 1, steps, endpoint=False):
                        full_retreat.append((1-t)*start + t*end)
                full_retreat.append(retreat_traj[-1])
                
                print("Retreating...")
                for conf in full_retreat:
                    env.set_robot_conf(conf)
                    pr.step()

    else:
        print("ERROR: No PDDL plan found for pick action!")
    
    # --- PDDL PLANNING END ---

    # A few extra frames to show final pose + line
    for _ in range(20):
        pr.step()
        # recorder.record_step()

    # recorder.release()

    # Leave sim running so you can inspect; close manually with Ctrl+C
    print("Pick-hover complete. Press Ctrl+C to close.")
    try:
        while True:
            pr.step()
    except KeyboardInterrupt:
        pass

    pr.stop()
    pr.shutdown()


if __name__ == "__main__":
    main()

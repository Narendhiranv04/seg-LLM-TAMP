"""
Run Ground Truth Orchestrator with Segmentation Viewer Windows

Displays all camera views and detected objects in separate OpenCV windows
while running the orchestrator tasks.

Windows shown:
1. Camera Views - Combined view of all 5 cameras
2. Detected Objects - List of visible objects (updates live)

Usage:
    python run_with_viewer.py
"""
import os
import sys

# Configure Qt for GUI - MUST be before any Qt imports
def _configure_qt():
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

import numpy as np

# Add pddlstream to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pddlstream'))

# Set HEADLESS env var to False BEFORE importing streams
os.environ["HEADLESS"] = "False"

# Import environment and streams
from rlbench_kitchen_streams import ENV, get_stream_map
from video_recorder import VideoRecorder

# Import ground truth orchestrator functions
import ground_truth_orchestrator as gt_orch
from ground_truth_orchestrator import (
    step_and_record, go_home, 
    run_standard_pick_place, run_cupboard_pick_place, 
    run_box_pick_place, run_open_box
)

# Import the segmentation viewer (cv2 is imported lazily inside)
from segmentation_viewer import SegmentationViewer


class ViewerOrchestrator:
    """Orchestrator with live camera and object viewing windows."""
    
    def __init__(self, env):
        self.env = env
        self.pr = env.pr
        self.viewer = SegmentationViewer(env, window_scale=0.5)
        self.video_recorder = None
        
    def step_with_view(self, count=1):
        """Step simulation and update viewer."""
        for _ in range(count):
            self.pr.step()
            if self.video_recorder:
                self.video_recorder.record_step()
        # Update viewer after steps
        return self.viewer.update()
    
    def run(self):
        """Run full orchestration with live viewing."""
        env = self.env
        pr = self.pr
        
        print("=" * 60)
        print("GROUND TRUTH ORCHESTRATOR WITH SEGMENTATION VIEWER")
        print("=" * 60)
        print("\nTwo windows will open:")
        print("  1. Camera Views - All 5 camera feeds in a grid")
        print("  2. Detected Objects - List of visible objects")
        print("\nPress 'q' in either window to quit early.")
        print("=" * 60)
        
        # Initialize video recorder
        self.video_recorder = VideoRecorder(env, output_dir="orchestrator_videos", fps=30)
        gt_orch.VIDEO_RECORDER = self.video_recorder
        
        # Initial view update
        self.viewer.update()
        
        # Settle physics
        print("\nSettling physics...")
        for i in range(50):
            pr.step()
            self.video_recorder.record_step()
            if i % 10 == 0:
                if not self.viewer.update():
                    print("User quit.")
                    return
        
        # Home position
        home_q = env.get_home_conf()
        env.set_robot_conf(home_q)
        if not self.step_with_view(10):
            return
        
        results = []
        user_quit = False
        
        try:
            # === TASK 1: Pick mug3 from cupboard ===
            print("\n" + "=" * 50)
            print("TASK 1: Cupboard Mug -> Placement")
            print("=" * 50)
            success = run_cupboard_pick_place(
                env,
                object_name='mug3',
                target_region='placement_boundary',
                task_name="Task 1: Cupboard Mug -> Placement"
            )
            results.append(("Task 1: mug3 -> placement", success))
            if not self.viewer.update():
                user_quit = True
            go_home(env)
            if not self.step_with_view(5):
                user_quit = True
            
            if user_quit:
                raise KeyboardInterrupt("User quit")
            
            # === TASK 2: Groceries to cupboard ===
            groceries_inside = ['soup', 'mustard', 'spam']
            groceries_top = ['sugar', 'crackers']
            
            task_num = 1
            for grocery in groceries_inside:
                print(f"\n--- Task 2.{task_num}: {grocery} -> Cupboard ---")
                success = run_standard_pick_place(
                    env,
                    object_name=grocery,
                    target_region='cupboard_boundary',
                    task_name=f"Task 2.{task_num}: {grocery}"
                )
                results.append((f"Task 2.{task_num}: {grocery}", success))
                if not self.viewer.update():
                    raise KeyboardInterrupt("User quit")
                go_home(env)
                self.step_with_view(5)
                task_num += 1
            
            for grocery in groceries_top:
                print(f"\n--- Task 2.{task_num}: {grocery} -> Cupboard (top) ---")
                success = run_standard_pick_place(
                    env,
                    object_name=grocery,
                    target_region='cupboard_boundary_top',
                    task_name=f"Task 2.{task_num}: {grocery}"
                )
                results.append((f"Task 2.{task_num}: {grocery}", success))
                if not self.viewer.update():
                    raise KeyboardInterrupt("User quit")
                go_home(env)
                self.step_with_view(5)
                task_num += 1
            
            # === TASK 3: Pick mug2 from box ===
            print("\n" + "=" * 50)
            print("TASK 3: Box Mug -> Placement")
            print("=" * 50)
            success = run_box_pick_place(
                env,
                object_name='mug2',
                target_region='placement_boundary',
                task_name="Task 3: Box Mug"
            )
            results.append(("Task 3: mug2", success))
            self.viewer.update()
            go_home(env)
            self.step_with_view(5)
            
            # === TASK 4: Open box (mug4 becomes visible!) ===
            print("\n" + "=" * 50)
            print("TASK 4: OPENING BOX")
            print(">>> Watch 'Detected Objects' window - mug4 will appear!")
            print("=" * 50)
            
            # Show before state
            self.viewer.update()
            
            success = run_open_box(
                env,
                task_name="Task 4: Open Box"
            )
            results.append(("Task 4: open box", success))
            
            # Show after state - mug4 should now be visible!
            self.viewer.update()
            
            go_home(env)
            self.step_with_view(5)
            
            # === TASK 5: Pick mug4 from inside box ===
            print("\n" + "=" * 50)
            print("TASK 5: Mug Inside Box -> Placement")
            print("=" * 50)
            success = run_box_pick_place(
                env,
                object_name='mug4',
                target_region='placement_boundary',
                task_name="Task 5: mug4"
            )
            results.append(("Task 5: mug4", success))
            self.viewer.update()
            go_home(env)
            self.step_with_view(5)
            
        except KeyboardInterrupt:
            print("\n--- Execution interrupted ---")
        except Exception as e:
            print(f"\n!!! ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        # === SUMMARY ===
        print("\n" + "=" * 60)
        print("EXECUTION SUMMARY")
        print("=" * 60)
        total = len(results)
        passed = sum(1 for _, s in results if s)
        for task, success in results:
            status = "PASS" if success else "FAIL"
            print(f"  [{status}] {task}")
        print(f"\nTotal: {passed}/{total} tasks completed")
        print("=" * 60)
        
        # Save videos
        if self.video_recorder:
            self.video_recorder.release()
            print("\nVideos saved to orchestrator_videos/")
        
        # Keep viewer open for inspection
        print("\nViewer windows open. Press 'q' to close.")
        try:
            while True:
                pr.step()
                if not self.viewer.update():
                    break
        except KeyboardInterrupt:
            pass
        
        self.viewer.close()
        pr.stop()
        pr.shutdown()


def main():
    orchestrator = ViewerOrchestrator(ENV)
    orchestrator.run()


if __name__ == "__main__":
    main()

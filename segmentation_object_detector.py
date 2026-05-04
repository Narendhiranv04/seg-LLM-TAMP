"""
Segmentation-Based Object Detection

Gets the visible object list from segmentation masks instead of hardcoded RLBench list.
Only objects that appear in camera masks are considered "known" to the planner.

Usage:
    detector = SegmentationObjectDetector(env)
    visible_objects = detector.get_visible_objects()
    
    # Check for new objects
    if detector.check_for_new_objects():
        new_objs = detector.get_newly_detected()
        print(f"NEW OBJECTS FOUND: {new_objs}")
        # Trigger replan...
"""

import numpy as np
import re
import os
from pyrep.backend import sim
from pyrep.objects.vision_sensor import VisionSensor
import colorsys


class SegmentationObjectDetector:
    """
    Detects visible objects from segmentation masks.
    Replaces hardcoded object lists with vision-based detection.
    """
    
    def __init__(self, env):
        self.env = env
        
        # Build handle -> name mapping from scene
        self.handle_to_name = {}
        self.name_to_handle = {}
        self.handle_to_task_name = {}
        self._build_handle_mapping()
        
        # Mask cameras (or regular cameras with render mode switch)
        self.cameras = {}
        self._setup_cameras()
        
        # Object tracking
        self.known_objects = set()      # All objects ever seen
        self.current_visible = set()    # Objects visible right now
        self.newly_detected = set()     # Objects detected this frame that weren't known before
        self.frame_idx = 0
        self.last_seen_frame = {}
        self.camera_hits = {}
        self.pixel_totals = {}

        # Fusion settings
        is_grill_env = self._is_grill_env()
        default_min_pixels = 50 if is_grill_env else 10
        configured_min_pixels = int(os.environ.get('LIVE_SEG_MIN_PIXELS', str(default_min_pixels)))
        self.min_pixels_per_camera = max(configured_min_pixels, 50) if is_grill_env else configured_min_pixels
        self.persistence_frames = int(os.environ.get('LIVE_SEG_PERSIST_FRAMES', '30'))
        
        default_kitchen_objects = {
            'mug1', 'mug2', 'mug3', 'mug4',
            'soup', 'mustard', 'spam', 'sugar', 'crackers',
            'box_lid'
        }
        default_grill_objects = {
            'steak', 'steak1', 'steak2', 'steak3',
            'chicken', 'chicken1', 'chicken2', 'chicken3',
            'spam', 'spam1', 'spam2', 'spam3',
            'plate', 'grill_lid'
        }
        env_task_objects = {
            str(name).strip()
            for name, obj in (getattr(self.env, 'name_to_obj', {}) or {}).items()
            if str(name).strip() and obj is not None
        }
        if self._is_grill_env():
            env_task_objects -= {'meat1', 'meat2'}
        default_task_objects = default_grill_objects if self._is_grill_env() else default_kitchen_objects
        self.task_objects = default_task_objects | env_task_objects
        self.env_aliases = {
            'mug1': ['mug1', 'mug_table', 'mug_table_shape'],
            'mug2': ['mug2', 'mug_box', 'mug_box_shape'],
            'mug3': ['mug3', 'mug_cupboard', 'mug_cupboard_shape'],
            'mug4': ['mug4', 'mug_inside_box', 'mug_inside_box_shape'],
            'soup': ['soup', 'can'],
            'mustard': ['mustard', 'bottle'],
            'spam': ['spam', 'tin'],
            'sugar': ['sugar', 'food_box'],
            'crackers': ['crackers', 'cereal'],
            'box_lid': ['box_lid'],
            'steak': ['steak', 'steak_visual'],
            'steak1': ['steak1', 'steak1_visual'],
            'steak2': ['steak2', 'steak2_visual'],
            'steak3': ['steak3', 'steak3_visual'],
            'chicken': ['chicken', 'chicken_visual'],
            'chicken1': ['chicken1', 'chicken1_visual'],
            'chicken2': ['chicken2', 'chicken2_visual'],
            'chicken3': ['chicken3', 'chicken3_visual'],
            'spam1': ['spam1', 'spam1_visual'],
            'spam2': ['spam2', 'spam2_visual'],
            'spam3': ['spam3', 'spam3_visual'],
            'plate': ['plate', 'plate_visual'],
            'grill_lid': ['grill_lid', 'lid', 'lid_visual'],
            'lid': ['lid', 'lid_visual'],
        }
        self.handle_to_region_name = {}
        self._build_task_handle_mapping()
        self._build_region_handle_mapping()
        
        print(f"[SegmentationDetector] Initialized with {len(self.handle_to_name)} scene objects")

    def _is_grill_env(self):
        module_name = self.env.__class__.__module__.lower()
        class_name = self.env.__class__.__name__.lower()
        region_names = {str(name).lower() for name in (getattr(self.env, "regions", {}) or {}).keys()}
        object_names = {str(name).lower() for name in (getattr(self.env, "name_to_obj", {}) or {}).keys()}
        return (
            "grill" in module_name
            or "grill" in class_name
            or hasattr(self.env, "grill_lid")
            or hasattr(self.env, "grill_boundary")
            or "grill-top" in region_names
            or "inside_grill" in region_names
            or "grill_lid" in object_names
        )
    
    def _build_handle_mapping(self):
        """Build mapping from CoppeliaSim handles to object names."""
        try:
            handles = sim.simGetObjectsInTree(
                sim.sim_handle_scene,
                sim.sim_object_shape_type,
                0
            )
            for h in handles:
                try:
                    name = sim.simGetObjectName(h)
                    self.handle_to_name[h] = name
                    self.name_to_handle[name] = h
                except:
                    pass
        except Exception as e:
            print(f"[SegmentationDetector] Error building handle map: {e}")
    
    def _setup_cameras(self):
        """Setup cameras for mask capture."""
        cam_names = ['left', 'right', 'overhead', 'wrist', 'front']
        
        # Try to find dedicated mask cameras first
        mask_cam_mapping = {
            'left': 'cam_over_shoulder_left_mask',
            'right': 'cam_over_shoulder_right_mask',
            'overhead': 'cam_overhead_mask',
            'wrist': 'cam_wrist_mask',
            'front': 'cam_front_mask',
        }
        env_cam_mapping = {
            'left': ['left', 'cam_over_shoulder_left'],
            'right': ['right', 'cam_over_shoulder_right'],
            'overhead': ['overhead', 'cam_overhead'],
            'wrist': ['wrist', 'cam_wrist'],
            'front': ['front', 'cam_front'],
        }
        
        for name in cam_names:
            # Try mask camera
            try:
                mask_cam = VisionSensor(mask_cam_mapping[name])
                mask_cam.set_explicit_handling(1)
                self.cameras[name] = ('mask', mask_cam)
            except:
                # Use regular camera with render mode switch
                for env_cam_name in env_cam_mapping[name]:
                    if env_cam_name in self.env.cams:
                        self.cameras[name] = ('render_mode', self.env.cams[env_cam_name])
                        break

    def _build_task_handle_mapping(self):
        """Map shape handles to canonical task objects via env object trees."""
        try:
            for task_name, aliases in self.env_aliases.items():
                root_obj = None
                for alias in aliases:
                    try:
                        root_obj = self.env.get_object(alias)
                        if root_obj is not None:
                            break
                    except Exception:
                        continue
                if root_obj is None:
                    continue

                try:
                    root_handle = root_obj.get_handle()
                    shape_handles = sim.simGetObjectsInTree(
                        int(root_handle),
                        sim.sim_object_shape_type,
                        0,
                    )
                    for handle in shape_handles:
                        self.handle_to_task_name.setdefault(int(handle), task_name)
                except Exception:
                    continue
        except Exception:
            pass

        # Fallback mapping from raw scene names.
        for handle, scene_name in self.handle_to_name.items():
            if handle in self.handle_to_task_name:
                continue
            task_name = self._canonical_task_name(scene_name)
            if task_name is not None:
                self.handle_to_task_name[handle] = task_name

        counts = {name: 0 for name in self.task_objects}
        for task_name in self.handle_to_task_name.values():
            if task_name in counts:
                counts[task_name] += 1
        print(f"[SegmentationDetector] Task-handle mapping: {counts}")

    def _build_region_handle_mapping(self):
        """Map shape handles to scene regions (table, box, cupboard)."""
        regions = getattr(self.env, 'regions', {}) or {}
        for region_name, region_obj in regions.items():
            try:
                # Handle can be the region dummy or its shape if it has one
                root_handle = int(region_obj.get_handle())
                shape_handles = sim.simGetObjectsInTree(
                    root_handle,
                    sim.sim_object_shape_type,
                    0,
                )
                # If no child shapes, use the root itself as the handle (detector sees it in masks)
                if not shape_handles:
                    shape_handles = [root_handle]
                
                for h in shape_handles:
                    self.handle_to_region_name[int(h)] = region_name
            except Exception:
                continue
        print(f"[SegmentationDetector] Region-handle mapping: {list(set(self.handle_to_region_name.values()))}")
    
    def _decode_mask(self, rgb):
        """Decode RGB-encoded handles: handle = R + G*256 + B*256*256"""
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
        
        handles = (
            rgb[:, :, 0].astype(np.int32) +
            rgb[:, :, 1].astype(np.int32) * 256 +
            rgb[:, :, 2].astype(np.int32) * 256 * 256
        )
        return handles
    
    def _capture_mask(self, cam_name):
        """Capture segmentation mask from camera."""
        from pyrep.const import RenderMode
        
        entry = self.cameras.get(cam_name)
        if entry is None:
            return None
        
        mode, cam = entry
        
        try:
            if mode == 'mask':
                cam.handle_explicitly()
                rgb = cam.capture_rgb()
                return self._decode_mask(rgb)
            
            elif mode == 'render_mode':
                original = cam.get_render_mode()
                cam.set_render_mode(RenderMode.OPENGL_COLOR_CODED)
                cam.handle_explicitly()
                rgb = cam.capture_rgb()
                cam.set_render_mode(original)
                return self._decode_mask(rgb)
        except:
            return None
        
        return None
    
    def _canonical_task_name(self, scene_name):
        """Map scene shape names to stable task object names."""
        if not scene_name:
            return None

        n = scene_name.lower()
        if 'mug_inside_box' in n or re.search(r'\bmug4\b', n):
            return 'mug4'
        if 'mug_cupboard' in n or re.search(r'\bmug3\b', n):
            return 'mug3'
        if 'mug_box' in n or re.search(r'\bmug2\b', n):
            return 'mug2'
        if 'mug_table' in n or re.search(r'\bmug1\b', n):
            return 'mug1'
        if 'box_lid' in n:
            return 'box_lid'
        if 'plate' in n and 'boundary' not in n:
            return 'plate'
        grill_meat = self._canonical_grill_meat_name(n)
        if grill_meat is not None:
            return grill_meat
        if 'lid' in n and 'box' not in n:
            return 'grill_lid'
        for name in ('soup', 'mustard', 'spam', 'sugar', 'crackers'):
            if name in n:
                return name
        return None

    def _canonical_grill_meat_name(self, lowered_scene_name):
        for meat_name in ('steak', 'chicken', 'spam'):
            match = re.search(rf'(^|[^a-z0-9])({meat_name}[0-9]*)([^a-z0-9]|$)', lowered_scene_name)
            if match:
                return match.group(2)
        return None
    
    def _get_objects_from_mask(self, handle_mask):
        """Extract canonical task object -> pixel_count from a decoded mask."""
        if handle_mask is None:
            return {}
        
        unique_handles, counts = np.unique(handle_mask, return_counts=True)
        objects = {}

        for h, pix in zip(unique_handles, counts):
            if h <= 0:
                continue
            canonical = self.handle_to_task_name.get(int(h))
            if canonical is None:
                continue
            if int(pix) < self.min_pixels_per_camera:
                continue
            objects[canonical] = objects.get(canonical, 0) + int(pix)

        return objects
    
    def update(self):
        """
        Capture all cameras and update visible object list.
        
        Returns:
            set: Currently visible objects
        """
        self.frame_idx += 1
        all_visible_pixels = {}
        camera_hits = {}

        for cam_name in self.cameras:
            handle_mask = self._capture_mask(cam_name)
            objects = self._get_objects_from_mask(handle_mask)
            for obj_name, pix in objects.items():
                all_visible_pixels[obj_name] = all_visible_pixels.get(obj_name, 0) + int(pix)
                camera_hits.setdefault(obj_name, []).append(cam_name)
                self.last_seen_frame[obj_name] = self.frame_idx

        visible_now = set(all_visible_pixels.keys())
        persisted = {
            obj for obj, last in self.last_seen_frame.items()
            if (self.frame_idx - last) <= self.persistence_frames
        }
        self.current_visible = (visible_now | persisted) & self.task_objects

        # Check for new objects
        self.newly_detected = self.current_visible - self.known_objects
        
        # Update known objects
        self.known_objects.update(self.current_visible)
        self.camera_hits = camera_hits
        self.pixel_totals = all_visible_pixels
        
        return self.current_visible
    
    def get_visible_objects(self):
        """Get current visible object list."""
        return self.current_visible.copy()
    
    def check_for_new_objects(self):
        """Check if any new objects were detected since last known state."""
        return len(self.newly_detected) > 0
    
    def get_newly_detected(self):
        """Get objects that were just detected (not previously known)."""
        return self.newly_detected.copy()
    
    def get_object_pose(self, obj_name):
        """
        Get object pose from ground truth (for PDDL init).
        We detect objects via vision but get pose from backend.
        """
        obj = self.env.get_object(obj_name)
        if obj:
            return tuple(obj.get_pose())
        return None

    def get_bounding_box(self, scene_name):
        """
        Get absolute world-coordinate axis-aligned bounding box for a scene object.
        Transforms all 8 corners of the local BB to world space to handle rotation.
        Returns: (min_x, min_y, min_z), (max_x, max_y, max_z)
        """
        try:
            handle = self.name_to_handle.get(scene_name)
            if handle is None:
                # Try region map first
                regions = getattr(self.env, 'regions', {}) or {}
                region_obj = regions.get(scene_name)
                if region_obj:
                    handle = int(region_obj.get_handle())
                else:
                    obj = self.env.get_object(scene_name)
                    if obj is None: return None
                    handle = int(obj.get_handle())
            
            # 1. Get local bounding box parameters
            # sim_objfloatparam_objbbmin_x = 15, ..., sim_objfloatparam_objbbmax_z = 20
            # Check if it's a shape first. Dummies don't have BB params.
            obj_type = sim.simGetObjectType(int(handle))
            if obj_type != sim.sim_object_shape_type:
                # If it's a dummy or other, use a small 1cm box around its position
                pos = sim.simGetObjectPosition(int(handle), -1)
                w_min = np.array(pos) - 0.005
                w_max = np.array(pos) + 0.005
                return tuple(w_min), tuple(w_max)

            l_min = [sim.simGetObjectFloatParameter(int(handle), i) for i in range(15, 18)]
            l_max = [sim.simGetObjectFloatParameter(int(handle), i) for i in range(18, 21)]
            
            # 2. Define 8 corners in local space
            corners = np.array([
                [l_min[0], l_min[1], l_min[2]], [l_min[0], l_min[1], l_max[2]],
                [l_min[0], l_max[1], l_min[2]], [l_min[0], l_max[1], l_max[2]],
                [l_max[0], l_min[1], l_min[2]], [l_max[0], l_min[1], l_max[2]],
                [l_max[0], l_max[1], l_min[2]], [l_max[0], l_max[1], l_max[2]]
            ])
            
            # 3. Get world transformation matrix (12 floats for 3x4 or 16 for 4x4)
            matrix_raw = sim.simGetObjectMatrix(int(handle), -1)
            m = np.array(matrix_raw).reshape(3, 4)
            
            # 4. Transform all corners to world space
            world_corners = []
            for c in corners:
                # Add homogeneous coordinate 1.0
                c_h = np.append(c, 1.0)
                wc = np.dot(m, c_h)
                world_corners.append(wc)
            
            world_corners = np.array(world_corners)
            
            # 5. Find min/max across all transformed corners (AABB)
            w_min = np.min(world_corners, axis=0)
            w_max = np.max(world_corners, axis=0)
            
            return tuple(w_min), tuple(w_max)
            
        except Exception as e:
            print(f"[SegmentationDetector] Error getting accurate BB for {scene_name}: {e}")
            return None

    def reset_known(self):
        """Reset the known objects (start fresh)."""
        self.known_objects = set()
        self.newly_detected = set()
    
    
    def build_pddl_init(self, target_region):
        """
        Build PDDL init state from visible objects.
        
        Returns:
            list: PDDL init predicates
        """
        self.update()  # Refresh visible objects
        
        init = [
            ('hand-empty',),
            ('region', target_region),
        ]
        
        home_q = tuple(self.env.get_home_conf())
        init.append(('conf', home_q))
        init.append(('at-conf', home_q))
        init.append(('is-home', home_q))
        
        movable_objects = {
            'mug1', 'mug2', 'mug3', 'mug4',
            'soup', 'mustard', 'spam', 'sugar', 'crackers'
        }

        # Add only VISIBLE objects
        for obj_name in self.current_visible:
            if obj_name not in movable_objects:
                continue
            pose = self.get_object_pose(obj_name)
            if pose:
                init.append(('movable', obj_name))
                init.append(('pose', pose))
                init.append(('at-pose', obj_name, pose))
        
        # Add lid-specific predicates if lid is visible
        if 'box_lid' in self.current_visible:
            init.append(('lid', 'box_lid'))
            
            # Check if lid is closed (using backend - this is physics state)
            lid = self.env.get_object('box_lid')
            if lid and lid.get_position()[2] < 0.85:
                init.append(('closed', 'box_lid'))
        
        # Add obstruction predicates based on current state
        if 'mug2' in self.current_visible and 'box_lid' in self.current_visible:
            mug2 = self.env.get_object('mug2')
            lid = self.env.get_object('box_lid')
            if mug2 and lid:
                mug_pos = mug2.get_position()
                lid_pos = lid.get_position()
                if (mug_pos[2] > lid_pos[2] and
                    abs(mug_pos[0] - lid_pos[0]) < 0.15 and
                    abs(mug_pos[1] - lid_pos[1]) < 0.15):
                    init.append(('obstructs', 'mug2', 'box_lid'))
        
        return init
    
    def print_status(self):
        """Print current detection status."""
        print(f"\n[SegmentationDetector] Status:")
        print(f"  Visible: {len(self.current_visible)} objects")
        print(f"  Known: {len(self.known_objects)} objects")
        print(f"  New: {len(self.newly_detected)} objects")
        if self.current_visible:
            print(f"  Objects: {sorted(self.current_visible)}")
        if self.newly_detected:
            print(f"  NEW: {sorted(self.newly_detected)}")


# Convenience function
def get_visible_objects_from_segmentation(env):
    """One-shot function to get visible objects."""
    detector = SegmentationObjectDetector(env)
    return detector.update()


if __name__ == "__main__":
    print("Import and use SegmentationObjectDetector")
    print("Example:")
    print("  detector = SegmentationObjectDetector(env)")
    print("  visible = detector.get_visible_objects()")

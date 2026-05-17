"""
Live Segmentation Viewer using Multiprocessing

Runs the visualization in a separate process to avoid Qt conflicts.
Uses shared memory to pass image data between processes.
"""

import multiprocessing as mp
import numpy as np
import time
import os
import re
from multiprocessing import shared_memory
import signal
import sys


def _pick_mp_context():
    """Pick a stable process context for GUI viewer subprocesses."""
    for method in ("forkserver", "spawn", "fork"):
        try:
            return mp.get_context(method)
        except Exception:
            continue
    return mp.get_context()


# Viewer process - runs completely isolated from CoppeliaSim
def viewer_process(shm_name, shape, stop_event, title="Live Segmentation"):
    """
    Viewer process that displays images from shared memory.
    Uses OpenCV in its own process (no Qt conflicts).
    """
    import cv2
    
    # Connect to shared memory
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        img_array = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    except Exception as e:
        print(f"[Viewer] Failed to connect to shared memory: {e}")
        return
    
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, shape[1], shape[0])
    
    print(f"[Viewer] Started - Press 'q' to quit")
    
    last_update = time.time()
    while not stop_event.is_set():
        try:
            # Read image from shared memory
            frame = img_array.copy()
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Display
            cv2.imshow(title, frame_bgr)
            
            # Check for quit
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            
        except Exception as e:
            print(f"[Viewer] Error: {e}")
            break
    
    cv2.destroyAllWindows()
    shm.close()
    print("[Viewer] Closed")


class LiveSegmentationViewer:
    """
    Manages a live segmentation viewer in a separate process.
    
    Usage:
        viewer = LiveSegmentationViewer(env)
        viewer.start()
        
        # In your loop:
        viewer.update()
        
        # When done:
        viewer.stop()
    """
    
    def __init__(self, env, width=1700, height=680, detector=None):
        self.env = env
        self.detector = detector
        self.width = width
        self.height = height
        self.shape = (height, width, 3)
        
        # Shared memory for image data
        self.shm = None
        self.img_array = None
        
        # Viewer process
        self.viewer_proc = None
        self.stop_event = None
        self._mp_ctx = _pick_mp_context()
        
        # Segmentation helpers
        self.handle_to_name = {}
        self.name_to_handle = {}
        self.handle_to_color = {}
        self.handle_to_task_name = {}
        self._mask_camera_cache = {}
        self._warn_once = {}

        self.camera_names = ['left', 'right', 'overhead', 'wrist', 'front']
        self.detected_objects = set()
        self.object_camera_hits = {}
        self.object_pixel_counts = {}
        self.frame_idx = 0
        self.last_seen_frame = {}
        self.latest_composite_frame = None

        # Camera-mask fusion settings
        self.min_pixels_per_camera = int(os.environ.get("LIVE_SEG_MIN_PIXELS", "10"))
        self.persistence_frames = int(os.environ.get("LIVE_SEG_PERSIST_FRAMES", "10"))
        self.action_sequence = []
        self.current_action_index = -1
        self.current_action_label = ""
        self.completed_action_count = 0
        self.total_action_count = 0
        self.scene_state_label = ""
        self.scene_state_desired = ""
        self.scene_state_regions = {}
        self.scene_state_visible = []
        self.scene_state_new = []
        self.scene_state_held = None
        self.scene_state_failure = None
        self.scene_state_info = {}

        self.default_object_order = [
            "mug1", "mug2", "mug3", "mug4",
            "soup", "mustard", "spam", "sugar", "crackers",
            "box_lid",
            "grill_lid", "plate",
            "steak", "steak1", "steak2", "steak3",
            "chicken", "chicken1", "chicken2", "chicken3",
            "spam1", "spam2", "spam3",
        ]
        self.color_palette = [
            (255, 70, 70),
            (255, 160, 40),
            (255, 235, 0),
            (230, 80, 255),
            (0, 220, 220),
            (170, 255, 0),
            (70, 150, 255),
            (255, 100, 190),
            (120, 80, 255),
            (255, 255, 255),
            (255, 180, 120),
            (120, 255, 180),
        ]
        self.task_objects = self._build_task_objects()
        self.display_names = {name: name for name in self.task_objects}
        self.task_colors = self._build_task_colors()

        self._build_handle_mapping()
        self._build_task_handle_mapping()
        self._generate_colors()

    def set_action_sequence(self, actions):
        """Set full ordered action sequence displayed in the side panel."""
        if actions is None:
            self.action_sequence = []
            return
        self.action_sequence = [str(a) for a in actions]
        if self.current_action_index >= len(self.action_sequence):
            self.current_action_index = -1

    def set_current_action(self, action_index=None, action_label=None):
        """Set currently active action index (0-based) and optional label."""
        if action_index is None:
            self.current_action_index = -1
        else:
            try:
                self.current_action_index = int(action_index)
            except Exception:
                self.current_action_index = -1
        self.current_action_label = "" if action_label is None else str(action_label)

    def set_action_progress(self, completed_action_count=None, total_action_count=None):
        if completed_action_count is not None:
            try:
                self.completed_action_count = max(0, int(completed_action_count))
            except Exception:
                pass
        if total_action_count is not None:
            try:
                self.total_action_count = max(0, int(total_action_count))
            except Exception:
                pass

    def set_scene_state(
        self,
        snapshot=None,
        label="",
        desired="",
        held_object=None,
        failure_event=None,
        scene_state_info=None,
        completed_action_count=None,
        total_action_count=None,
    ):
        self.scene_state_label = str(label or "")
        self.scene_state_desired = str(desired or "")
        self.scene_state_held = held_object
        if scene_state_info is not None:
            self.scene_state_info = dict(scene_state_info)
        if snapshot is not None:
            self.scene_state_visible = list(getattr(snapshot, "visible_objects", []) or [])
            self.scene_state_new = list(getattr(snapshot, "newly_visible_objects", []) or [])
            self.scene_state_regions = dict(getattr(snapshot, "object_region_map", {}) or {})
        self.scene_state_failure = self._format_failure(failure_event)
        self.set_action_progress(completed_action_count, total_action_count)

    def _format_failure(self, failure_event):
        if failure_event is None:
            return None
        failure_id = getattr(failure_event, "failure_id", None)
        layer = getattr(failure_event, "failure_layer", None)
        if hasattr(layer, "value"):
            layer = layer.value
        message = getattr(failure_event, "message", "") or ""
        return {
            "failure_id": str(failure_id or ""),
            "failure_layer": str(layer or ""),
            "message": str(message),
        }

    def _short_list(self, values, max_items=7):
        values = [str(v) for v in (values or []) if str(v)]
        if not values:
            return "(none)"
        shown = values[:max_items]
        suffix = "" if len(values) <= max_items else f", +{len(values) - max_items}"
        return ", ".join(shown) + suffix

    def _draw_wrapped_line(self, draw, xy, label, value, fill, font, max_chars=54, line_h=15, max_lines=2):
        x, y = xy
        text = f"{label}: {value}"
        chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)] or [text]
        for chunk in chunks[:max_lines]:
            draw.text((x, y), chunk, fill=fill, font=font)
            y += line_h
        return y
        
    def _build_handle_mapping(self):
        """Build handle -> name mapping from scene."""
        if self.detector is not None:
            self.handle_to_name = dict(getattr(self.detector, "handle_to_name", {}) or {})
            self.name_to_handle = dict(getattr(self.detector, "name_to_handle", {}) or {})
            if self.handle_to_name:
                return

        from pyrep.backend import sim
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
                    continue
        except Exception as e:
            print(f"[LiveViewer] Error scanning scene: {e}")
    
    def _scene_name(self, obj):
        if obj is None:
            return None
        try:
            name = obj.get_name()
            if name:
                return str(name).strip()
        except Exception:
            pass
        return None

    def _build_task_objects(self):
        detector_objects = getattr(self.detector, "task_objects", None)
        if detector_objects:
            detected_names = {str(name).strip() for name in detector_objects if str(name).strip()}
            ordered = [name for name in self.default_object_order if name in detected_names]
            extras = sorted(name for name in detected_names if name not in self.default_object_order)
            return ordered + extras

        support_handles = set()
        for attr in ("table", "box", "cupboard"):
            obj = getattr(self.env, attr, None)
            if obj is None:
                continue
            try:
                support_handles.add(int(obj.get_handle()))
            except Exception:
                pass

        names = []
        seen_names = set()
        seen_handles = set()
        for obj in (getattr(self.env, "name_to_obj", {}) or {}).values():
            if obj is None:
                continue
            try:
                handle = int(obj.get_handle())
            except Exception:
                continue
            if handle in seen_handles:
                continue
            seen_handles.add(handle)
            if handle in support_handles:
                continue

            scene_name = self._scene_name(obj)
            if not scene_name:
                continue
            if scene_name.endswith("_boundary"):
                continue
            if scene_name in {"box_base", "diningTable", "cupboard"}:
                continue
            if scene_name not in seen_names:
                names.append(scene_name)
                seen_names.add(scene_name)

        ordered = [name for name in self.default_object_order if name in seen_names]
        extras = sorted(name for name in names if name not in self.default_object_order)
        return ordered + extras

    def _build_task_colors(self):
        return {
            name: self.color_palette[idx % len(self.color_palette)]
            for idx, name in enumerate(self.task_objects)
        }

    def _generate_colors(self):
        """Generate colors with raw scene-object emphasis."""
        for handle in self.handle_to_name.keys():
            task_name = self.handle_to_task_name.get(handle)
            if task_name is not None:
                self.handle_to_color[handle] = self.task_colors.get(task_name, (255, 255, 255))
            else:
                self.handle_to_color[handle] = (45, 45, 45)

    def _canonical_task_name(self, scene_name):
        """Normalize low-level scene shape names to raw tracked object names."""
        if not scene_name:
            return None
        if scene_name in self.task_objects:
            return scene_name

        lowered = scene_name.lower()
        for task_name in self.task_objects:
            if lowered == task_name.lower():
                return task_name
        for task_name in self.task_objects:
            if re.search(rf"\b{re.escape(task_name.lower())}\b", lowered):
                return task_name
        return None

    def _build_task_handle_mapping(self):
        """Pre-map every shape handle to a raw scene object (if any)."""
        if self.detector is not None:
            self.handle_to_task_name.update(dict(getattr(self.detector, "handle_to_task_name", {}) or {}))

        try:
            from pyrep.backend import sim
            for obj in (getattr(self.env, "name_to_obj", {}) or {}).values():
                if obj is None:
                    continue
                scene_name = self._scene_name(obj)
                task_name = self._canonical_task_name(scene_name)
                if task_name is None:
                    continue

                try:
                    root_handle = int(obj.get_handle())
                    self.handle_to_task_name[root_handle] = task_name
                    shape_handles = sim.simGetObjectsInTree(
                        root_handle,
                        sim.sim_object_shape_type,
                        0,
                    )
                    for handle in shape_handles:
                        self.handle_to_task_name[int(handle)] = task_name
                except Exception:
                    continue
        except Exception:
            pass

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
        print(f"[LiveViewer] Task-handle mapping: {counts}")
    
    def start(self):
        """Start the viewer process."""
        # Create shared memory
        nbytes = int(np.prod(self.shape))
        self.shm = shared_memory.SharedMemory(create=True, size=nbytes)
        self.img_array = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf)
        self.img_array.fill(30)  # Dark gray background
        
        # Start viewer process
        self.stop_event = self._mp_ctx.Event()
        self.stop_event.clear()
        self.viewer_proc = self._mp_ctx.Process(
            target=viewer_process,
            args=(self.shm.name, self.shape, self.stop_event, "Live Segmentation Masks")
        )
        self.viewer_proc.start()
        
        print(f"[LiveViewer] Started (shared memory: {self.shm.name})")
        time.sleep(0.5)  # Let viewer initialize
    
    def stop(self):
        """Stop the viewer process."""
        if self.viewer_proc:
            self.stop_event.set()
            self.viewer_proc.join(timeout=2)
            if self.viewer_proc.is_alive():
                self.viewer_proc.terminate()
            self.viewer_proc = None
        
        if self.shm:
            self.shm.close()
            self.shm.unlink()
            self.shm = None
        
        print("[LiveViewer] Stopped")
    
    def _decode_mask(self, rgb_image):
        """Decode RGB-encoded handles."""
        if rgb_image.dtype != np.uint8:
            if rgb_image.max() <= 1.0:
                rgb_image = (rgb_image * 255).astype(np.uint8)
            else:
                rgb_image = rgb_image.astype(np.uint8)
        
        return (
            rgb_image[:, :, 0].astype(np.int32) +
            rgb_image[:, :, 1].astype(np.int32) * 256 +
            rgb_image[:, :, 2].astype(np.int32) * 256 * 256
        )
    
    def _get_objects_from_mask(self, handle_mask):
        """Extract canonical task objects and pixels from a handle mask."""
        detected_pixels = {}
        unique, counts = np.unique(handle_mask, return_counts=True)
        for h, pix in zip(unique, counts):
            if h <= 0:
                continue
            task_name = self.handle_to_task_name.get(int(h))
            if task_name is None:
                continue
            if int(pix) < self.min_pixels_per_camera:
                continue
            detected_pixels[task_name] = detected_pixels.get(task_name, 0) + int(pix)
        return detected_pixels
    
    def _colorize_mask(self, handle_mask):
        """Convert handle mask to colored image."""
        h, w = handle_mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        colored[:, :] = (18, 18, 18)
        
        for handle in np.unique(handle_mask):
            if handle <= 0:
                continue
            mask = handle_mask == handle
            color = self.handle_to_color.get(handle, (128, 128, 128))
            colored[mask] = color

        # White edges make small table objects visually obvious.
        edges = np.zeros((h, w), dtype=bool)
        edges[1:, :] |= handle_mask[1:, :] != handle_mask[:-1, :]
        edges[:, 1:] |= handle_mask[:, 1:] != handle_mask[:, :-1]
        colored[edges] = (255, 255, 255)
        
        return colored
    
    def _try_get_mask_camera(self, cam_name):
        """Try to get mask camera."""
        from pyrep.objects.vision_sensor import VisionSensor
        
        if cam_name in ['left', 'right']:
            mask_name = f'cam_over_shoulder_{cam_name}_mask'
        else:
            mask_name = f'cam_{cam_name}_mask'
        
        if mask_name not in self._mask_camera_cache:
            try:
                cam = VisionSensor(mask_name)
                cam.set_explicit_handling(1)
                self._mask_camera_cache[mask_name] = cam
            except:
                self._mask_camera_cache[mask_name] = None
        
        return self._mask_camera_cache.get(mask_name)

    def _get_env_camera(self, cam_name):
        """Resolve canonical viewer camera names against kitchen/grill env keys."""
        cams = getattr(self.env, "cams", {}) or {}
        aliases = {
            "left": ("left", "cam_over_shoulder_left"),
            "right": ("right", "cam_over_shoulder_right"),
            "overhead": ("overhead", "cam_overhead"),
            "wrist": ("wrist", "cam_wrist"),
            "front": ("front", "cam_front"),
        }
        for key in aliases.get(cam_name, (cam_name,)):
            cam = cams.get(key)
            if cam is not None:
                return cam
        return None
    
    def _capture_camera(self, cam_name):
        """Capture segmentation from one camera."""
        if self.detector is not None and hasattr(self.detector, "_capture_mask"):
            try:
                handle_mask = self.detector._capture_mask(cam_name)
                if handle_mask is not None:
                    detected = self._get_objects_from_mask(handle_mask)
                    colorized = self._colorize_mask(handle_mask)
                    return colorized, detected
            except Exception:
                pass

        cam = self._get_env_camera(cam_name)
        if cam is None:
            return None, {}
        
        try:
            # Try mask camera first
            mask_cam = self._try_get_mask_camera(cam_name)
            
            if mask_cam:
                mask_cam.handle_explicitly()
                mask_rgb = mask_cam.capture_rgb()
                handle_mask = self._decode_mask(mask_rgb)
                detected = self._get_objects_from_mask(handle_mask)
                colorized = self._colorize_mask(handle_mask)
                return colorized, detected
            
            # Try render mode switch
            try:
                from pyrep.const import RenderMode
                original = cam.get_render_mode()
                cam.set_render_mode(RenderMode.OPENGL_COLOR_CODED)
                cam.handle_explicitly()
                mask_rgb = cam.capture_rgb()
                cam.set_render_mode(original)
                
                if mask_rgb.max() > 0.01:
                    handle_mask = self._decode_mask(mask_rgb)
                    if len(np.unique(handle_mask)) > 1:
                        detected = self._get_objects_from_mask(handle_mask)
                        colorized = self._colorize_mask(handle_mask)
                        return colorized, detected
            except:
                pass
            
            # Fallback to RGB
            cam.handle_explicitly()
            rgb = cam.capture_rgb()
            rgb_uint8 = (rgb * 255).astype(np.uint8) if rgb.max() <= 1 else rgb.astype(np.uint8)
            return rgb_uint8, {}
            
        except Exception as e:
            return None, {}
    
    def _filter_objects(self, objects):
        """Compatibility shim; objects are already canonical task names."""
        return {o for o in objects if o in self.task_objects}
    
    def update(self):
        """Capture all cameras and update the shared image."""
        if self.img_array is None:
            return set()
        
        from PIL import Image, ImageDraw, ImageFont
        
        # Camera layout
        cam_w, cam_h = 240, 180
        panel_w = 960
        obj_panel_w = 500
        action_panel_x = cam_w * 3 + obj_panel_w
        
        # Capture all cameras and fuse object evidence.
        self.frame_idx += 1
        images = {}
        fused_pixels = {}
        fused_hits = {}

        for cam_name in self.camera_names:
            colorized, detected_pixels = self._capture_camera(cam_name)
            images[cam_name] = colorized

            for obj_name, pixels in detected_pixels.items():
                fused_pixels[obj_name] = fused_pixels.get(obj_name, 0) + int(pixels)
                fused_hits.setdefault(obj_name, []).append(cam_name)
                self.last_seen_frame[obj_name] = self.frame_idx

        # Keep objects for a short time even if briefly occluded.
        fused_visible = set()
        for obj_name in self.task_objects:
            if obj_name in fused_pixels:
                fused_visible.add(obj_name)
                continue
            last = self.last_seen_frame.get(obj_name)
            if last is not None and (self.frame_idx - last) <= self.persistence_frames:
                fused_visible.add(obj_name)

        self.detected_objects = self._filter_objects(fused_visible)
        self.object_camera_hits = fused_hits
        self.object_pixel_counts = fused_pixels
        
        # Build composite image
        grid_w = cam_w * 3
        grid_h = cam_h * 2
        total_w = grid_w + panel_w
        total_h = max(grid_h, 650)
        
        canvas = Image.new('RGB', (total_w, total_h), (30, 30, 30))
        draw = ImageDraw.Draw(canvas)
        
        # Load font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except:
            font = font_small = ImageFont.load_default()
        
        # Place camera images
        positions = {
            'left': (0, 0),
            'overhead': (cam_w, 0),
            'right': (cam_w * 2, 0),
            'front': (0, cam_h),
            'wrist': (cam_w, cam_h),
        }
        
        for cam_name, pos in positions.items():
            img = images.get(cam_name)
            if img is not None:
                pil_img = Image.fromarray(img).resize((cam_w, cam_h), Image.LANCZOS)
                canvas.paste(pil_img, pos)
                
                # Label
                draw.rectangle([pos[0], pos[1], pos[0]+70, pos[1]+18], fill=(0,0,0,180))
                draw.text((pos[0]+4, pos[1]+2), cam_name.upper(), fill=(255,255,255), font=font_small)
        
        # Panel
        px = grid_w
        draw.rectangle([px, 0, total_w, total_h], fill=(40, 40, 40))
        draw.text((px+10, 10), "VISIBLE OBJECTS", fill=(255,255,255), font=font)
        draw.text((px+10, 30), "(fused from all 5 masks)", fill=(150,150,150), font=font_small)
        draw.line([(px+10, 50), (px + obj_panel_w - 10, 50)], fill=(100,100,100))
        draw.text((px+10, 55), f"Count: {len(self.detected_objects)}", fill=(200,200,200), font=font_small)
        draw.text((px+10, 68), "Cams: L O R F W", fill=(130, 130, 130), font=font_small)
        
        # List objects
        y = 85
        for name in self.task_objects:
            if name not in self.detected_objects:
                continue
            if y > 220:
                draw.text((px+15, y), "...", fill=(150,150,150), font=font_small)
                break
            
            color = self.task_colors.get(name, (150, 150, 150))
            draw.ellipse([px+12, y+3, px+20, y+11], fill=color)
            
            hits = self.object_camera_hits.get(name, [])
            hit_flags = {
                "L": "L" if "left" in hits else ".",
                "O": "O" if "overhead" in hits else ".",
                "R": "R" if "right" in hits else ".",
                "F": "F" if "front" in hits else ".",
                "W": "W" if "wrist" in hits else ".",
            }
            cams_text = f"[{hit_flags['L']}{hit_flags['O']}{hit_flags['R']}{hit_flags['F']}{hit_flags['W']}]"
            seen_now = name in self.object_pixel_counts
            text_color = (255, 255, 255) if seen_now else (160, 160, 160)
            label = self.display_names.get(name, name)
            region = self.scene_state_regions.get(name, "")
            region_text = f" -> {region}" if region else ""
            draw.text((px+25, y), f"{label:<12} {cams_text}{region_text}", fill=text_color, font=font_small)
            y += 15

        info = self.scene_state_info or {}
        state_y = max(y + 12, 238)
        draw.line([(px + 10, state_y - 8), (px + obj_panel_w - 10, state_y - 8)], fill=(100, 100, 100))
        draw.text((px + 10, state_y), "SCENE STATE", fill=(255, 255, 255), font=font)
        state_y += 20
        state_y = self._draw_wrapped_line(draw, (px + 10, state_y), "Checkpoint", self.scene_state_label or "(none)", (210, 210, 210), font_small, max_chars=62, max_lines=1)
        if self.scene_state_desired:
            state_y = self._draw_wrapped_line(draw, (px + 10, state_y), "Desired", self.scene_state_desired, (210, 210, 210), font_small, max_chars=62, max_lines=1)
        state_y = self._draw_wrapped_line(draw, (px + 10, state_y), "Visible objects", self._short_list(info.get("visible_objects", self.scene_state_visible), 9), (220, 220, 220), font_small, max_chars=62)
        state_y = self._draw_wrapped_line(draw, (px + 10, state_y), "Newly visible", self._short_list(info.get("newly_visible_objects", self.scene_state_new), 8), (200, 220, 255), font_small, max_chars=62)
        state_y = self._draw_wrapped_line(draw, (px + 10, state_y), "Visible regions", self._short_list(info.get("visible_regions", []), 7), (220, 220, 220), font_small, max_chars=62)
        state_y = self._draw_wrapped_line(draw, (px + 10, state_y), "Supported regions", self._short_list(info.get("supported_regions", []), 8), (180, 200, 220), font_small, max_chars=62, max_lines=2)
        state_y = self._draw_wrapped_line(draw, (px + 10, state_y), "Pose map objects", self._short_list(info.get("pose_map_keys", []), 8), (220, 220, 220), font_small, max_chars=62)
        gripper = dict(info.get("gripper", {}) or {})
        gripper_text = (
            f"{gripper.get('status', 'unknown')}, "
            f"{gripper.get('open_closed', 'unknown')}, "
            f"holding={gripper.get('holding') or '(none)'}, "
            f"open_amount={gripper.get('open_amount')}"
        )
        state_y = self._draw_wrapped_line(draw, (px + 10, state_y), "Gripper", gripper_text, (220, 220, 220), font_small, max_chars=62, max_lines=2)
        lid = dict(info.get("lid", {}) or {})
        lid_bits = [lid.get("name") or "lid", lid.get("state", "unknown")]
        if lid.get("current_angle") is not None:
            lid_bits.append(f"angle={lid.get('current_angle')}")
        state_y = self._draw_wrapped_line(draw, (px + 10, state_y), "Lid", ", ".join(str(x) for x in lid_bits), (220, 220, 220), font_small, max_chars=62)
        semantic_facts = info.get("pddl_state", []) or []
        if semantic_facts:
            state_y = self._draw_wrapped_line(draw, (px + 10, state_y), "Semantic facts", self._short_list(semantic_facts, 4), (190, 230, 190), font_small, max_chars=62, max_lines=2)
        if self.scene_state_failure:
            state_y = self._draw_wrapped_line(
                draw,
                (px + 10, state_y),
                "Failure",
                f"{self.scene_state_failure['failure_layer']} {self.scene_state_failure['failure_id']}",
                (255, 180, 180),
                font_small,
                max_chars=62,
            )
            if self.scene_state_failure["message"]:
                state_y = self._draw_wrapped_line(draw, (px + 10, state_y), "Message", self.scene_state_failure["message"], (255, 190, 190), font_small, max_chars=62, max_lines=2)

        # Divider between object panel and action panel.
        draw.line([(action_panel_x, 8), (action_panel_x, total_h - 8)], fill=(95, 95, 95), width=1)

        # Action sequence panel
        ax = action_panel_x + 10
        draw.text((ax, 10), "ACTION SEQUENCE", fill=(255, 255, 255), font=font)
        if self.current_action_label:
            draw.text((ax, 30), f"Now: {self.current_action_label}", fill=(185, 230, 185), font=font_small)
        else:
            draw.text((ax, 30), "Now: (idle)", fill=(150, 150, 150), font=font_small)
        total_actions = self.total_action_count or len(self.action_sequence)
        done_actions = min(self.completed_action_count, total_actions) if total_actions else self.completed_action_count
        draw.text((ax, 45), f"Done: {done_actions}/{total_actions}", fill=(170, 205, 255), font=font_small)
        draw.line([(ax, 62), (total_w - 10, 62)], fill=(100, 100, 100))

        ay = 72
        if self.action_sequence:
            for idx, action in enumerate(self.action_sequence):
                if ay > total_h - 18:
                    draw.text((ax, ay), "...", fill=(150, 150, 150), font=font_small)
                    break
                is_current = (idx == self.current_action_index)
                is_done = idx < self.completed_action_count
                if is_current:
                    draw.rectangle(
                        [ax - 3, ay - 1, total_w - 12, ay + 13],
                        fill=(52, 95, 52),
                        outline=(95, 140, 95),
                    )
                text = f"{idx + 1}. {action}"
                if is_current:
                    tcolor = (255, 255, 255)
                elif is_done:
                    tcolor = (145, 215, 145)
                else:
                    tcolor = (190, 190, 190)
                draw.text((ax, ay), text, fill=tcolor, font=font_small)
                ay += 15
        else:
            draw.text((ax, ay), "No action sequence set.", fill=(150, 150, 150), font=font_small)
        
        # Timestamp
        ts = time.strftime("%H:%M:%S")
        draw.text((total_w-70, total_h-18), ts, fill=(100,100,100), font=font_small)
        
        # Copy to shared memory
        result = np.array(canvas.resize((self.width, self.height), Image.LANCZOS))
        self.latest_composite_frame = result.copy()
        np.copyto(self.img_array, result)
        
        return self.detected_objects
    
    def get_detected_objects(self):
        """Get current detected objects."""
        return self.detected_objects.copy()

    def get_latest_frame(self):
        """Get the most recently rendered composite panel frame (RGB)."""
        if self.latest_composite_frame is None:
            return None
        return self.latest_composite_frame.copy()


if __name__ == "__main__":
    print("This module should be imported.")
    print("Use: python run_live_segmentation.py")

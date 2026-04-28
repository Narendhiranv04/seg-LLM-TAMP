"""
Live Segmentation Viewer using Tkinter (in-process).

This avoids multiprocessing GUI crashes on some Qt/Tk setups while still
opening a separate viewer window.
"""

import numpy as np
import time
import re
import os


class TkinterSegmentationViewer:
    """Live segmentation viewer using a Tkinter window in the main process."""

    def __init__(self, env, width=1260, height=480):
        self.env = env
        self.width = width
        self.height = height

        # Compatibility with code paths that check for subprocess liveness.
        self.viewer_proc = None

        # Tk widgets/state
        self.root = None
        self.canvas_label = None
        self._photo = None
        self._closed = False

        # Segmentation
        self.handle_to_name = {}
        self.name_to_handle = {}
        self.handle_to_color = {}
        self.handle_to_task_name = {}
        self._mask_cache = {}

        self.camera_names = ["left", "right", "overhead", "wrist", "front"]
        self.detected_objects = set()
        self.object_camera_hits = {}
        self.object_pixel_counts = {}
        self.frame_idx = 0
        self.last_seen_frame = {}
        self.latest_composite_frame = None
        self.min_pixels_per_camera = int(os.environ.get("LIVE_SEG_MIN_PIXELS", "10"))
        self.persistence_frames = int(os.environ.get("LIVE_SEG_PERSIST_FRAMES", "10"))
        self.action_sequence = []
        self.current_action_index = -1
        self.current_action_label = ""

        self.default_object_order = [
            "mug1", "mug2", "mug3", "mug4",
            "soup", "mustard", "spam", "sugar", "crackers",
            "box_lid",
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

        self._build_handles()
        self._build_task_handle_mapping()
        self._make_colors()

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

    def _build_handles(self):
        from pyrep.backend import sim
        try:
            handles = sim.simGetObjectsInTree(sim.sim_handle_scene, sim.sim_object_shape_type, 0)
            for handle in handles:
                try:
                    name = sim.simGetObjectName(handle)
                    self.handle_to_name[handle] = name
                    self.name_to_handle[name] = handle
                except Exception:
                    pass
        except Exception:
            pass

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

    def _make_colors(self):
        for handle in self.handle_to_name.keys():
            task_name = self.handle_to_task_name.get(handle)
            if task_name is not None:
                self.handle_to_color[handle] = self.task_colors.get(task_name, (255, 255, 255))
            else:
                self.handle_to_color[handle] = (45, 45, 45)

    def _canonical_task_name(self, scene_name):
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
        print(f"[TkViewer] Task-handle mapping: {counts}")

    def start(self):
        import tkinter as tk

        self.root = tk.Tk()
        self.root.title("Live Segmentation Masks")
        self.root.geometry(f"{self.width}x{self.height}")
        self.root.configure(bg="#1f1f1f")

        self.canvas_label = tk.Label(self.root, bg="#1f1f1f")
        self.canvas_label.pack(fill=tk.BOTH, expand=True)

        def _on_close():
            self._closed = True
            try:
                self.root.destroy()
            except Exception:
                pass
            self.root = None

        self.root.protocol("WM_DELETE_WINDOW", _on_close)
        print("[TkViewer] Started (in-process)")

    def stop(self):
        if self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass
            self.root = None
        self._closed = True
        print("[TkViewer] Stopped")

    def _decode(self, rgb):
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1 else rgb.astype(np.uint8)
        return (
            rgb[:, :, 0].astype(np.int32)
            + rgb[:, :, 1].astype(np.int32) * 256
            + rgb[:, :, 2].astype(np.int32) * 256 * 256
        )

    def _get_mask_cam(self, name):
        from pyrep.objects.vision_sensor import VisionSensor

        mask_name = f"cam_over_shoulder_{name}_mask" if name in ["left", "right"] else f"cam_{name}_mask"
        if mask_name not in self._mask_cache:
            try:
                cam = VisionSensor(mask_name)
                cam.set_explicit_handling(1)
                self._mask_cache[mask_name] = cam
            except Exception:
                self._mask_cache[mask_name] = None
        return self._mask_cache.get(mask_name)

    def _capture(self, cam_name):
        from pyrep.const import RenderMode

        cam = self.env.cams.get(cam_name)
        if not cam:
            return np.zeros((180, 240, 3), dtype=np.uint8), set()

        try:
            mask_cam = self._get_mask_cam(cam_name)
            if mask_cam:
                mask_cam.handle_explicitly()
                rgb = mask_cam.capture_rgb()
                handles = self._decode(rgb)

                h, w = handles.shape
                colored = np.zeros((h, w, 3), dtype=np.uint8)
                colored[:, :] = (18, 18, 18)
                detected = {}
                unique, counts = np.unique(handles, return_counts=True)
                for handle, pix in zip(unique, counts):
                    if handle <= 0:
                        continue
                    task_name = self.handle_to_task_name.get(int(handle))
                    if task_name is None or int(pix) < self.min_pixels_per_camera:
                        continue
                    detected[task_name] = detected.get(task_name, 0) + int(pix)
                    colored[handles == handle] = self.task_colors.get(task_name, (128, 128, 128))
                edges = np.zeros((h, w), dtype=bool)
                edges[1:, :] |= handles[1:, :] != handles[:-1, :]
                edges[:, 1:] |= handles[:, 1:] != handles[:, :-1]
                colored[edges] = (255, 255, 255)
                return colored, detected

            # Fallback path: switch camera to color-coded rendering.
            original_mode = cam.get_render_mode()
            cam.set_render_mode(RenderMode.OPENGL_COLOR_CODED)
            cam.handle_explicitly()
            rgb = cam.capture_rgb()
            cam.set_render_mode(original_mode)
            cam.handle_explicitly()

            handles = self._decode(rgb)
            if len(np.unique(handles)) > 1:
                h, w = handles.shape
                colored = np.zeros((h, w, 3), dtype=np.uint8)
                colored[:, :] = (18, 18, 18)
                detected = {}
                unique, counts = np.unique(handles, return_counts=True)
                for handle, pix in zip(unique, counts):
                    if handle <= 0:
                        continue
                    task_name = self.handle_to_task_name.get(int(handle))
                    if task_name is None or int(pix) < self.min_pixels_per_camera:
                        continue
                    detected[task_name] = detected.get(task_name, 0) + int(pix)
                    colored[handles == handle] = self.task_colors.get(task_name, (128, 128, 128))
                edges = np.zeros((h, w), dtype=bool)
                edges[1:, :] |= handles[1:, :] != handles[:-1, :]
                edges[:, 1:] |= handles[:, 1:] != handles[:, :-1]
                colored[edges] = (255, 255, 255)
                return colored, detected

            # Last fallback: plain RGB
            cam.handle_explicitly()
            rgb = cam.capture_rgb()
            rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1 else rgb.astype(np.uint8)
            return rgb, {}
        except Exception:
            try:
                cam.set_render_mode(RenderMode.OPENGL3)
            except Exception:
                pass
            return np.zeros((180, 240, 3), dtype=np.uint8), {}

    def _filter(self, objects):
        return {obj for obj in objects if obj in self.task_objects}

    def update(self):
        if self._closed or self.root is None:
            return self.detected_objects

        from PIL import Image, ImageDraw, ImageFont, ImageTk

        cam_w, cam_h = 240, 180
        panel_w = 620
        obj_panel_w = 300
        action_panel_x = cam_w * 3 + obj_panel_w

        self.frame_idx += 1
        fused_pixels = {}
        fused_hits = {}
        images = {}

        for name in self.camera_names:
            img, detected = self._capture(name)
            images[name] = img
            for obj_name, pixels in detected.items():
                fused_pixels[obj_name] = fused_pixels.get(obj_name, 0) + int(pixels)
                fused_hits.setdefault(obj_name, []).append(name)
                self.last_seen_frame[obj_name] = self.frame_idx

        visible = set()
        for obj_name in self.task_objects:
            if obj_name in fused_pixels:
                visible.add(obj_name)
                continue
            last = self.last_seen_frame.get(obj_name)
            if last is not None and (self.frame_idx - last) <= self.persistence_frames:
                visible.add(obj_name)

        self.detected_objects = self._filter(visible)
        self.object_pixel_counts = fused_pixels
        self.object_camera_hits = fused_hits

        total_w = cam_w * 3 + panel_w
        total_h = cam_h * 2
        canvas = Image.new("RGB", (total_w, total_h), (30, 30, 30))
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except Exception:
            font = ImageFont.load_default()
            font_small = font

        positions = {
            "left": (0, 0),
            "overhead": (cam_w, 0),
            "right": (cam_w * 2, 0),
            "front": (0, cam_h),
            "wrist": (cam_w, cam_h),
        }

        for name, pos in positions.items():
            frame = images.get(name)
            if frame is None:
                continue
            pil_frame = Image.fromarray(frame).resize((cam_w, cam_h), Image.LANCZOS)
            canvas.paste(pil_frame, pos)
            draw.text((pos[0] + 4, pos[1] + 2), name.upper(), fill=(255, 255, 0), font=font)

        px = cam_w * 3
        draw.rectangle([px, 0, px + panel_w, total_h], fill=(40, 40, 40))
        draw.text((px + 10, 10), "VISIBLE OBJECTS", fill=(255, 255, 255), font=font)
        draw.text((px + 10, 30), "(fused from all 5 masks)", fill=(150, 150, 150), font=font_small)
        draw.line([(px + 10, 50), (px + obj_panel_w - 10, 50)], fill=(100, 100, 100))
        draw.text((px + 10, 55), f"Count: {len(self.detected_objects)}", fill=(200, 200, 200), font=font_small)
        draw.text((px + 10, 68), "Cams: L O R F W", fill=(130, 130, 130), font=font_small)

        y = 85
        for obj in self.task_objects:
            if obj not in self.detected_objects:
                continue
            if y > total_h - 20:
                break
            color = self.task_colors.get(obj, (150, 150, 150))
            draw.ellipse([px + 10, y + 2, px + 18, y + 10], fill=color)
            hits = self.object_camera_hits.get(obj, [])
            hit_flags = {
                "L": "L" if "left" in hits else ".",
                "O": "O" if "overhead" in hits else ".",
                "R": "R" if "right" in hits else ".",
                "F": "F" if "front" in hits else ".",
                "W": "W" if "wrist" in hits else ".",
            }
            cams_text = f"[{hit_flags['L']}{hit_flags['O']}{hit_flags['R']}{hit_flags['F']}{hit_flags['W']}]"
            seen_now = obj in self.object_pixel_counts
            txt_color = (255, 255, 255) if seen_now else (160, 160, 160)
            label = self.display_names.get(obj, obj)
            draw.text((px + 22, y), f"{label:<16} {cams_text}", fill=txt_color, font=font_small)
            y += 15

        # Divider between object panel and action panel.
        draw.line([(action_panel_x, 8), (action_panel_x, total_h - 8)], fill=(95, 95, 95), width=1)

        # Action sequence panel
        ax = action_panel_x + 10
        draw.text((ax, 10), "ACTION SEQUENCE", fill=(255, 255, 255), font=font)
        if self.current_action_label:
            draw.text((ax, 30), f"Now: {self.current_action_label}", fill=(185, 230, 185), font=font_small)
        else:
            draw.text((ax, 30), "Now: (idle)", fill=(150, 150, 150), font=font_small)
        draw.line([(ax, 50), (total_w - 10, 50)], fill=(100, 100, 100))

        ay = 60
        if self.action_sequence:
            for idx, action in enumerate(self.action_sequence):
                if ay > total_h - 18:
                    draw.text((ax, ay), "...", fill=(150, 150, 150), font=font_small)
                    break
                is_current = idx == self.current_action_index
                if is_current:
                    draw.rectangle(
                        [ax - 3, ay - 1, total_w - 12, ay + 13],
                        fill=(52, 95, 52),
                        outline=(95, 140, 95),
                    )
                txt_color = (255, 255, 255) if is_current else (210, 210, 210)
                draw.text((ax, ay), f"{idx + 1}. {action}", fill=txt_color, font=font_small)
                ay += 15
        else:
            draw.text((ax, ay), "No action sequence set.", fill=(150, 150, 150), font=font_small)

        draw.text((total_w - 70, total_h - 18), time.strftime("%H:%M:%S"), fill=(100, 100, 100), font=font_small)

        result = canvas.resize((self.width, self.height), Image.LANCZOS)
        self.latest_composite_frame = np.array(result, copy=True)
        self._photo = ImageTk.PhotoImage(result)
        self.canvas_label.configure(image=self._photo)

        try:
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            self._closed = True
            self.root = None

        return self.detected_objects

    def get_detected_objects(self):
        return self.detected_objects.copy()

    def get_latest_frame(self):
        """Get the most recently rendered composite panel frame (RGB)."""
        if self.latest_composite_frame is None:
            return None
        return self.latest_composite_frame.copy()

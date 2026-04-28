"""
VLM Context Aggregator (Module 1)
=================================
Captures 5 camera views, builds object state from environment,
and creates prompt bundles for the VLM planner.

Tracks objects that are expected but not yet visible in frames.
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm_pipeline.model_registry import (
    PROMPT_MODE_STATE_TEXT,
    PROMPT_MODE_TEXT_VISIBLE,
    PROMPT_MODE_VISION_GOAL,
)


@dataclass
class ObjectState:
    """Represents one segmentation-observed object for planning."""
    name: str
    pddl_name: str
    obj_type: str
    location: str = ""
    region: Optional[str] = None
    is_pickable: bool = True
    blocked_by: Optional[str] = None
    state: Optional[str] = None
    visible_in_frames: bool = True
    visible_cameras: List[str] = field(default_factory=list)
    mask_pixels: int = 0
    mask_regions: List[str] = field(default_factory=list)


@dataclass
class SceneState:
    """Complete segmentation-driven scene state for VLM context."""
    objects: List[ObjectState]
    regions: List[Dict[str, str]]
    robot_gripper_state: str = "empty"
    robot_holding: Optional[str] = None
    lid_state: str = "unknown"
    expected_objects: List[str] = field(default_factory=list)
    missing_objects: List[str] = field(default_factory=list)
    visibility_mode: str = "segmentation-only"
    visible_objects: List[str] = field(default_factory=list)
    newly_visible_objects: List[str] = field(default_factory=list)
    visible_regions: List[str] = field(default_factory=list)


@dataclass
class PromptBundle:
    """Bundle of all context for VLM query."""
    composite_image: np.ndarray
    individual_frames: Dict[str, np.ndarray]
    state_text: str
    visible_objects_text: str
    goal_text: str
    system_prompt: str
    user_prompt: str
    known_objects: List[str] = field(default_factory=list)
    known_regions: List[str] = field(default_factory=list)
    is_replan: bool = False
    failure_context: Optional[str] = None
    unknown_object_crop: Optional[np.ndarray] = None
    previous_plan: Optional[List[str]] = None


class VLMContextAggregator:
    """
    Aggregates visual and state context for the VLM planner.
    """
    
    def __init__(self,
                 env=None,
                 visible_objects_only: bool = False,
                 enable_live_segmentation_view: bool = False,
                 planner_input_mode: str = PROMPT_MODE_STATE_TEXT):
        """
        Initialize the context aggregator.
        
        Args:
            env: RLBenchKitchenEnv instance (optional, can be set later)
            visible_objects_only: If True, include only segmentation-visible objects in state text
            enable_live_segmentation_view: If True, show live segmentation masks + visible list panel
            planner_input_mode: Controls whether prompts use full state text, visible-object text, or vision+goal only
        """
        self.env = env
        
        # Object/region catalogs are derived from segmentation-visible raw scene names.
        self.expected_objects: Dict[str, Dict[str, str]] = {}
        self.regions: List[Dict[str, str]] = [
            {'name': 'placement_boundary', 'description': 'placement_boundary'},
            {'name': 'cupboard_boundary', 'description': 'cupboard_boundary'},
            {'name': 'cupboard_boundary_top', 'description': 'cupboard_boundary_top'},
            {'name': 'groceries_boundary', 'description': 'groceries_boundary'},
            {'name': 'box_boundary', 'description': 'box_boundary'},
            {'name': 'table', 'description': 'table'},
        ]
        
        # Camera names
        self.camera_names = ['left', 'right', 'overhead', 'wrist', 'front']

        # Visibility-mode settings (opt-in; default behavior remains unchanged)
        self.visible_objects_only = visible_objects_only
        self.enable_live_segmentation_view = enable_live_segmentation_view
        self.planner_input_mode = planner_input_mode

        # Segmentation components (lazy initialized)
        self.segmentation_detector = None
        self.live_segmentation_viewer = None
        self.live_viewer_backend = "none"
        self._visibility_init_attempted = False
        self._visibility_warning_printed = False

        # Track raw scene names observed via segmentation.
        self.visible_scene_objects: Set[str] = set()
        self.visible_pddl_objects: Set[str] = set()
        self.newly_visible_scene_objects: Set[str] = set()
        self.newly_visible_pddl_objects: Set[str] = set()
        self.all_seen_scene_objects: Set[str] = set()
        self.all_seen_pddl_objects: Set[str] = set()
        self.latest_segmentation_snapshot: Dict[str, Any] = {}

        if self.env is not None:
            self._refresh_scene_catalog()
            self._initialize_visibility_tools()
        
    def set_env(self, env):
        """Set the environment instance."""
        self.env = env
        self._refresh_scene_catalog()
        self._initialize_visibility_tools()

    def set_planner_input_mode(self, planner_input_mode: str):
        """Update prompt-building mode for the active planner backend."""
        self.planner_input_mode = planner_input_mode or PROMPT_MODE_STATE_TEXT

    def configure_visibility(self,
                             visible_objects_only: Optional[bool] = None,
                             enable_live_segmentation_view: Optional[bool] = None):
        """
        Configure segmentation-backed visibility mode.
        """
        if visible_objects_only is not None:
            self.visible_objects_only = visible_objects_only
        if enable_live_segmentation_view is not None:
            self.enable_live_segmentation_view = enable_live_segmentation_view
        self._initialize_visibility_tools()

    def _infer_object_type(self, name: str) -> str:
        lowered = (name or '').lower()
        if lowered.startswith('mug'):
            return 'mug'
        if lowered == 'soup':
            return 'can'
        if lowered == 'mustard':
            return 'bottle'
        if lowered == 'spam':
            return 'tin'
        if lowered == 'sugar':
            return 'box'
        if lowered == 'crackers':
            return 'cereal'
        if lowered == 'box_lid':
            return 'lid'
        return 'object'

    def _refresh_scene_catalog(self):
        """Build raw object and region catalogs without semantic mug-role aliases."""
        object_names: List[str] = []
        if self.segmentation_detector is not None and getattr(self.segmentation_detector, 'task_objects', None):
            object_names = list(self.segmentation_detector.task_objects)
        elif self.env is not None:
            seen = set()
            for alias, obj in getattr(self.env, 'name_to_obj', {}).items():
                if obj is None:
                    continue
                try:
                    scene_name = obj.get_name() or alias
                except Exception:
                    scene_name = alias
                if scene_name in {'box_base', 'diningTable', 'cupboard'}:
                    continue
                if scene_name.endswith('_boundary'):
                    continue
                if scene_name in seen:
                    continue
                seen.add(scene_name)
                object_names.append(scene_name)

        ordered_object_names = []
        for name in ('mug1', 'mug2', 'mug3', 'mug4', 'soup', 'mustard', 'spam', 'sugar', 'crackers', 'box_lid'):
            if name in object_names and name not in ordered_object_names:
                ordered_object_names.append(name)
        for name in sorted(object_names):
            if name not in ordered_object_names:
                ordered_object_names.append(name)

        self.expected_objects = {
            name: {'type': self._infer_object_type(name), 'scene_name': name}
            for name in ordered_object_names
        }

        region_names: List[str] = []
        if self.segmentation_detector is not None and getattr(self.segmentation_detector, 'region_names', None):
            region_names = list(self.segmentation_detector.region_names)
        elif self.env is not None and hasattr(self.env, 'regions'):
            region_names = [
                name for name in self.env.regions.keys()
                if name not in {'box-top', 'box-inside', 'shelf-lower'}
            ]
        else:
            region_names = ['placement_boundary', 'cupboard_boundary', 'cupboard_boundary_top', 'groceries_boundary', 'box_boundary', 'table']

        ordered_region_names = []
        for name in ('placement_boundary', 'cupboard_boundary_top', 'cupboard_boundary', 'groceries_boundary', 'box_boundary', 'table'):
            if name in region_names and name not in ordered_region_names:
                ordered_region_names.append(name)
        for name in sorted(region_names):
            if name not in ordered_region_names:
                ordered_region_names.append(name)

        self.regions = [{'name': name, 'description': name} for name in ordered_region_names]

    def _initialize_visibility_tools(self):
        """
        Lazy-init segmentation detector/viewer only when visibility features are enabled.
        """
        if self.env is None:
            return

        if self.segmentation_detector is None:
            try:
                from segmentation_object_detector import SegmentationObjectDetector
                self.segmentation_detector = SegmentationObjectDetector(self.env)
                self._refresh_scene_catalog()
                print("[Context Aggregator] Segmentation detector initialized")
            except Exception as e:
                if not self._visibility_warning_printed:
                    print(f"[Context Aggregator] WARNING: Segmentation detector unavailable: {e}")
                    self._visibility_warning_printed = True
                self.segmentation_detector = None

        if self.enable_live_segmentation_view and self.live_segmentation_viewer is None:
            try:
                backend_pref = os.environ.get("LIVE_SEG_VIEWER_BACKEND", "auto").strip().lower()

                def _start_backend(name: str) -> bool:
                    if name == "tkinter":
                        from tkinter_segmentation_viewer import TkinterSegmentationViewer
                        viewer = TkinterSegmentationViewer(self.env, width=1260, height=480)
                        backend = "tkinter"
                    else:
                        from live_segmentation_viewer import LiveSegmentationViewer
                        viewer = LiveSegmentationViewer(self.env, width=1260, height=480)
                        backend = "opencv"

                    viewer.start()
                    proc = getattr(viewer, "viewer_proc", None)
                    if proc is not None:
                        time.sleep(0.7)
                        if not proc.is_alive():
                            try:
                                viewer.stop()
                            except Exception:
                                pass
                            return False

                    self.live_segmentation_viewer = viewer
                    self.live_viewer_backend = backend
                    return True

                ok = False
                if backend_pref == "tkinter":
                    ok = _start_backend("tkinter")
                    if not ok:
                        print("[Context Aggregator] Tkinter viewer failed, trying OpenCV viewer")
                        ok = _start_backend("opencv")
                elif backend_pref == "opencv":
                    ok = _start_backend("opencv")
                    if not ok:
                        print("[Context Aggregator] OpenCV viewer failed, trying Tkinter viewer")
                        ok = _start_backend("tkinter")
                else:
                    ok = _start_backend("opencv")
                    if not ok:
                        print("[Context Aggregator] OpenCV viewer failed, falling back to Tkinter viewer")
                        ok = _start_backend("tkinter")

                if ok:
                    print(f"[Context Aggregator] Live segmentation viewer started (backend: {self.live_viewer_backend})")
                else:
                    self.live_segmentation_viewer = None
                    self.live_viewer_backend = "none"
                    raise RuntimeError("No live segmentation viewer backend could be started")
            except Exception as e:
                if not self._visibility_warning_printed:
                    print(f"[Context Aggregator] WARNING: Live segmentation viewer unavailable: {e}")
                    self._visibility_warning_printed = True
                self.live_segmentation_viewer = None
                self.live_viewer_backend = "none"

        self._visibility_init_attempted = True

    def _scene_name_to_pddl_name(self, scene_name: str) -> str:
        """Use raw segmentation object names directly in planner prompts."""
        return scene_name

    def update_live_segmentation_view(self) -> Set[str]:
        """
        Update live segmentation window (if enabled).
        Returns currently detected scene names from the viewer.
        """
        if self.live_segmentation_viewer is None:
            return set()
        try:
            proc = getattr(self.live_segmentation_viewer, "viewer_proc", None)
            if proc is not None and not proc.is_alive():
                print("[Context Aggregator] WARNING: Live segmentation viewer process is not alive")
                return set()
            detected = self.live_segmentation_viewer.update()
            return set(detected) if detected is not None else set()
        except Exception:
            return set()

    def set_live_action_sequence(self,
                                 actions: Optional[List[Any]],
                                 current_action_index: Optional[int] = None,
                                 current_action_label: Optional[str] = None):
        """Push the current action sequence into the live segmentation viewer."""
        if self.live_segmentation_viewer is None:
            return
        try:
            self.live_segmentation_viewer.set_action_sequence(actions)
            self.live_segmentation_viewer.set_current_action(
                action_index=current_action_index,
                action_label=current_action_label,
            )
            self.update_live_segmentation_view()
        except Exception:
            pass

    def update_visible_objects(self, event: str = "") -> Dict[str, List[str]]:
        """Refresh segmentation-visible raw object and region sets."""
        self._initialize_visibility_tools()
        self.update_live_segmentation_view()

        if self.segmentation_detector is None:
            return {
                "visible_scene": sorted(self.visible_scene_objects),
                "visible_pddl": sorted(self.visible_pddl_objects),
                "newly_visible_scene": [],
                "newly_visible_pddl": [],
                "visible_regions": sorted(self.latest_segmentation_snapshot.get("visible_regions", [])),
            }

        try:
            self.segmentation_detector.update()
            snapshot = self.segmentation_detector.get_current_snapshot()
        except Exception as e:
            if not self._visibility_warning_printed:
                print(f"[Context Aggregator] WARNING: Segmentation update failed: {e}")
                self._visibility_warning_printed = True
            snapshot = {
                "visible_objects": [],
                "newly_visible_objects": [],
                "visible_regions": [],
                "camera_hits": {},
                "pixel_totals": {},
                "region_camera_hits": {},
                "region_pixel_totals": {},
                "object_regions": {},
                "known_objects": [],
            }

        current_scene = set(snapshot.get("visible_objects", []))
        current_pddl = set(current_scene)
        newly_scene = set(snapshot.get("newly_visible_objects", []))
        newly_pddl = set(newly_scene)

        self.latest_segmentation_snapshot = snapshot
        self.visible_scene_objects = current_scene
        self.visible_pddl_objects = current_pddl
        self.newly_visible_scene_objects = newly_scene
        self.newly_visible_pddl_objects = newly_pddl
        self.all_seen_scene_objects.update(current_scene)
        self.all_seen_pddl_objects.update(current_pddl)

        label = f" ({event})" if event else ""
        print(f"[Context Aggregator] Visible objects{label}: {len(current_pddl)} -> {sorted(current_pddl)}")
        if snapshot.get("visible_regions"):
            print(f"[Context Aggregator] Visible regions{label}: {sorted(snapshot.get('visible_regions', []))}")
        if newly_pddl:
            print(f"[Context Aggregator] NEW object(s) entered scene{label}: {sorted(newly_pddl)}")

        return {
            "visible_scene": sorted(current_scene),
            "visible_pddl": sorted(current_pddl),
            "newly_visible_scene": sorted(newly_scene),
            "newly_visible_pddl": sorted(newly_pddl),
            "visible_regions": sorted(snapshot.get("visible_regions", [])),
        }

    def shutdown(self):
        """Cleanup for live viewer process."""
        if self.live_segmentation_viewer is not None:
            try:
                self.live_segmentation_viewer.stop()
            except Exception:
                pass
            self.live_segmentation_viewer = None
            self.live_viewer_backend = "none"
        
    def capture_frames(self) -> Dict[str, np.ndarray]:
        """
        Capture frames from all 5 cameras.
        
        Returns:
            Dictionary mapping camera name to RGB numpy array
        """
        if self.env is None:
            raise RuntimeError("Environment not set. Call set_env() first.")
        
        frames = {}
        for name in self.camera_names:
            if name in self.env.cams:
                cam = self.env.cams[name]
                # Trigger capture
                cam.handle_explicitly()
                # Get RGB image
                img = cam.capture_rgb()
                # Convert to uint8 if needed
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                frames[name] = img
        
        return frames
    
    def stitch_frames(self, frames: Dict[str, np.ndarray], 
                      layout: str = "grid") -> np.ndarray:
        """
        Stitch multiple camera frames into a single composite image.
        
        Args:
            frames: Dictionary of camera frames
            layout: 'grid' (2x3) or 'horizontal' (1x5)
            
        Returns:
            Composite image as numpy array
        """
        # Get frames in order
        ordered_frames = []
        for name in self.camera_names:
            if name in frames:
                ordered_frames.append(frames[name])
        
        if not ordered_frames:
            raise ValueError("No frames to stitch")
        
        # Ensure all frames have same size
        target_h, target_w = ordered_frames[0].shape[:2]
        resized = []
        for f in ordered_frames:
            if f.shape[:2] != (target_h, target_w):
                from PIL import Image

                pil_img = Image.fromarray(f)
                pil_img = pil_img.resize((target_w, target_h))
                f = np.array(pil_img)
            resized.append(f)
        
        if layout == "grid":
            # 2x3 grid (pad with black if needed)
            while len(resized) < 6:
                resized.append(np.zeros_like(resized[0]))
            
            row1 = np.concatenate(resized[:3], axis=1)
            row2 = np.concatenate(resized[3:6], axis=1)
            composite = np.concatenate([row1, row2], axis=0)
        else:
            # Horizontal strip
            composite = np.concatenate(resized, axis=1)
        
        return composite
    
    def _get_default_scene_state(self) -> SceneState:
        """Fallback state when no live environment is attached."""
        return SceneState(
            objects=[],
            regions=self.regions,
            robot_gripper_state='empty',
            robot_holding=None,
            lid_state='unknown',
            expected_objects=list(self.expected_objects.keys()),
            missing_objects=[],
            visibility_mode='offline-empty',
            visible_objects=[],
            newly_visible_objects=[],
            visible_regions=[],
        )

    def _get_world_bounding_box(self, obj) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Get an object's world-space axis-aligned bounding box."""
        if obj is None:
            return None

        if self.env is not None and hasattr(self.env, '_get_world_bounding_box'):
            try:
                return self.env._get_world_bounding_box(obj)
            except Exception:
                pass

        try:
            min_x, max_x, min_y, max_y, min_z, max_z = obj.get_bounding_box()
            ox, oy, oz = obj.get_position()
            return (
                ox + min_x,
                ox + max_x,
                oy + min_y,
                oy + max_y,
                oz + min_z,
                oz + max_z,
            )
        except Exception:
            return None

    def _get_grasped_pddl_objects(self) -> Set[str]:
        """Map currently grasped PyRep objects back to planner-facing PDDL names."""
        grasped: Set[str] = set()
        if self.env is None or not hasattr(self.env, 'gripper'):
            return grasped

        handle_to_pddl: Dict[int, str] = {}
        for pddl_name, info in self.expected_objects.items():
            try:
                obj = self.env.get_object(info['scene_name'])
                if obj is not None:
                    handle_to_pddl[int(obj.get_handle())] = pddl_name
            except Exception:
                continue

        try:
            grasped_objects = self.env.gripper.get_grasped_objects()
        except Exception:
            grasped_objects = []

        for obj in grasped_objects:
            try:
                handle = int(obj.get_handle())
                if handle in handle_to_pddl:
                    grasped.add(handle_to_pddl[handle])
                    continue
            except Exception:
                pass

            try:
                obj_name = obj.get_name()
            except Exception:
                obj_name = ''

            for pddl_name, info in self.expected_objects.items():
                if obj_name in {pddl_name, info['scene_name']}:
                    grasped.add(pddl_name)
                    break

        return grasped

    def _build_object_state(self,
                            pddl_name: str,
                            grasped_pddl: Set[str],
                            snapshot: Dict[str, Any]) -> Optional[ObjectState]:
        """Build one raw segmentation-driven object record."""
        info = self.expected_objects.get(
            pddl_name,
            {'type': self._infer_object_type(pddl_name), 'scene_name': pddl_name},
        )
        visible_cameras = list(snapshot.get('camera_hits', {}).get(pddl_name, []))
        mask_pixels = int(snapshot.get('pixel_totals', {}).get(pddl_name, 0))
        mask_regions = list(snapshot.get('object_regions', {}).get(pddl_name, []))
        is_holding = pddl_name in grasped_pddl

        return ObjectState(
            name=pddl_name,
            pddl_name=pddl_name,
            obj_type=info['type'],
            location='',
            region=mask_regions[0] if mask_regions else None,
            is_pickable=not is_holding and pddl_name != 'box_lid',
            blocked_by=None,
            state=None,
            visible_in_frames=(pddl_name in self.visible_pddl_objects) if self.visible_pddl_objects else False,
            visible_cameras=visible_cameras,
            mask_pixels=mask_pixels,
            mask_regions=mask_regions,
        )

    def get_scene_state(self) -> SceneState:
        """Extract a segmentation-driven scene state with raw scene object names."""
        if self.env is None:
            return self._get_default_scene_state()

        snapshot = self.latest_segmentation_snapshot
        if self.segmentation_detector is not None:
            update = self.update_visible_objects(event='scene-state')
            snapshot = self.latest_segmentation_snapshot
            visible_names = set(update.get('visible_pddl', []))
            visible_regions = sorted(update.get('visible_regions', []))
        else:
            visible_names = set()
            visible_regions = []

        grasped_pddl = self._get_grasped_pddl_objects()
        robot_holding = sorted(grasped_pddl)[0] if grasped_pddl else None
        robot_gripper_state = 'holding' if robot_holding else 'empty'

        object_names = sorted(visible_names | grasped_pddl)
        objects = []
        for pddl_name in object_names:
            obj_state = self._build_object_state(pddl_name, grasped_pddl, snapshot)
            if obj_state is not None:
                objects.append(obj_state)

        return SceneState(
            objects=objects,
            regions=self.regions,
            robot_gripper_state=robot_gripper_state,
            robot_holding=robot_holding,
            lid_state='unknown',
            expected_objects=list(self.expected_objects.keys()),
            missing_objects=[],
            visibility_mode='segmentation-only',
            visible_objects=sorted(visible_names),
            newly_visible_objects=sorted(list(self.newly_visible_pddl_objects)),
            visible_regions=visible_regions,
        )

    def state_to_pddl_text(self, state: SceneState) -> str:
        """Convert segmentation observations into planner-facing text."""
        lines = ["=== SEGMENTATION OBSERVATION ===", ""]
        lines.append("## Robot:")
        lines.append(f"- gripper: {state.robot_gripper_state}")
        if state.robot_holding:
            lines.append(f"- holding: {state.robot_holding}")
        lines.append("")

        lines.append("## Visible Objects:")
        if not state.objects:
            lines.append("- (none)")
        for obj in state.objects:
            facts = []
            if obj.visible_cameras:
                facts.append(f"cameras={'|'.join(obj.visible_cameras)}")
            if obj.mask_pixels:
                facts.append(f"mask_pixels={obj.mask_pixels}")
            if obj.mask_regions:
                facts.append(f"mask_regions={'|'.join(obj.mask_regions)}")
            if obj.state:
                facts.append(f"state={obj.state}")
            if obj.blocked_by:
                facts.append(f"blocked_by={obj.blocked_by}")
            if not facts:
                facts.append("detected=true")
            lines.append(f"- {obj.pddl_name}: {', '.join(facts)}")
        lines.append("")

        lines.append("## Visible Regions:")
        lines.append(f"- {', '.join(state.visible_regions) if state.visible_regions else '(none)'}")
        lines.append("")

        lines.append("## Allowed Regions:")
        for region in state.regions:
            lines.append(f"- {region['name']}")

        return '\n'.join(lines)

    def visible_objects_to_text(self, state: SceneState) -> str:
        """Compact segmentation summary for text-only planning."""
        lines = ["=== SEGMENTATION SUMMARY ===", ""]
        lines.append(f"- gripper: {state.robot_gripper_state}")
        if state.robot_holding:
            lines.append(f"- holding: {state.robot_holding}")
        lines.append(f"- visible-objects: {', '.join(state.visible_objects) if state.visible_objects else '(none)'}")
        lines.append(f"- visible-regions: {', '.join(state.visible_regions) if state.visible_regions else '(none)'}")
        if state.newly_visible_objects:
            lines.append(f"- newly-visible: {', '.join(state.newly_visible_objects)}")

        for obj in state.objects:
            facts = []
            if obj.visible_cameras:
                facts.append(f"cameras={'|'.join(obj.visible_cameras)}")
            if obj.mask_regions:
                facts.append(f"mask_regions={'|'.join(obj.mask_regions)}")
            if obj.state:
                facts.append(f"state={obj.state}")
            facts.append('pickable' if obj.is_pickable else 'not-pickable')
            lines.append(f"- {obj.pddl_name}: {', '.join(facts)}")

        return '\n'.join(lines)

    def build_system_prompt(self) -> str:
        """Build the system prompt for the VLM."""
        return """You are a robot task planner for a kitchen manipulation task. You control a Panda robot arm.

Your job is to output a sequence of executable actions to achieve the goal using the segmentation observation.
Use object and region names EXACTLY as they appear in the observation text. Do not rename objects into semantic roles.

AVAILABLE ACTIONS:
1. move(object) or move(object, region)
2. pick(object)
3. place(object, region)
4. open-lid(lid)

PLANNING RULES:
- Plan only from the current segmentation observation and goal.
- Do not invent hidden objects, hidden blockers, or semantic aliases.
- Use raw object names such as mug1, mug2, mug3, mug4 exactly as given.
- Every transfer must follow: move(object) -> pick(object) -> move(object, region) -> place(object, region).
- Output ONLY a numbered list of executable actions.
- Do not output commentary, reasoning, markdown, or prose.

Do NOT include any explanation, just the action sequence."""
    
    def build_user_prompt(self, state: SceneState, goal: str, 
                          is_replan: bool = False,
                          failure_context: Optional[str] = None,
                          previous_plan: Optional[List[str]] = None) -> str:
        """
        Build the user prompt with state and goal.
        """
        state_text = self.state_to_pddl_text(state)
        visible_text = self.visible_objects_to_text(state)

        if self.planner_input_mode == PROMPT_MODE_VISION_GOAL:
            if is_replan:
                return f"""=== REPLANNING REQUIRED ===

PREVIOUS PLAN WAS:
{chr(10).join(previous_plan) if previous_plan else 'Unknown'}

FAILURE REASON:
{failure_context}

## GOAL:
{goal}

## VISUAL CONTEXT:
The attached image shows 5 camera views of the current scene.
Use the image and the failure context to produce ONLY the remaining actions.
Do not rely on any explicit visible-object list text."""

            return f"""## GOAL:
{goal}

## VISUAL CONTEXT:
The attached image shows 5 camera views of the scene:
- Top row: left shoulder, right shoulder, overhead
- Bottom row: wrist camera, front view

Use the image and the goal to infer the action sequence.
Do not rely on any explicit visible-object list text in the prompt."""

        if self.planner_input_mode == PROMPT_MODE_TEXT_VISIBLE:
            guidance = """Plan using ONLY the rich visible-only state above. No images are available.

Use only the objects and facts currently visible through segmentation tracking.
Do not assume hidden objects, hidden blockers, or hidden placements.
If the scene later changes or a failure occurs, the system will replan with an updated visible-only state.
Output only executable actions and no commentary."""
            if is_replan:
                return f"""=== REPLANNING REQUIRED ===

PREVIOUS PLAN WAS:
{chr(10).join(previous_plan) if previous_plan else 'Unknown'}

FAILURE REASON:
{failure_context}

{state_text}

## COMPACT VISIBILITY:
{visible_text}

## GOAL:
{goal}

Use the failure reason together with the updated visible-only state to produce the corrected remaining action sequence.
{guidance}"""

            return f"""{state_text}

## COMPACT VISIBILITY:
{visible_text}

## GOAL:
{goal}

{guidance}"""

        if is_replan:
            prompt = f"""=== REPLANNING REQUIRED ===

PREVIOUS PLAN WAS:
{chr(10).join(previous_plan) if previous_plan else 'Unknown'}

FAILURE REASON:
{failure_context}

The attached image shows the current scene with the obstruction/unknown object highlighted.

{state_text}

## GOAL:
{goal}

Please provide a NEW action sequence that avoids the failure. Consider alternative paths or object orderings."""
        else:
            visibility_clause = ""
            if state.visibility_mode == "visible-only":
                visibility_clause = """
## IMPORTANT OBSERVABILITY RULE:
Only objects currently visible in segmentation masks are known/available for planning now.
If additional objects appear later, the system will replan."""

            prompt = f"""{state_text}

## GOAL:
{goal}
{visibility_clause}

## VISUAL CONTEXT:
The attached image shows 5 camera views of the scene:
- Top row: left shoulder, right shoulder, overhead
- Bottom row: wrist camera, front view

Based on the visual scene and state description, provide the action sequence to achieve the goal."""

        return prompt

    def create_prompt_bundle(self, goal: str,
                             is_replan: bool = False,
                             failure_context: Optional[str] = None,
                             previous_plan: Optional[List[str]] = None,
                             unknown_object_crop: Optional[np.ndarray] = None) -> PromptBundle:
        """
        Create complete prompt bundle for VLM query.
        
        Args:
            goal: Natural language goal
            is_replan: Whether this is a replanning query
            failure_context: What failed (for replanning)
            previous_plan: Previous plan being executed
            unknown_object_crop: Cropped image of unknown object
            
        Returns:
            PromptBundle ready for VLM
        """
        # Keep live viewer moving even when not using visible-only state mode.
        if self.enable_live_segmentation_view and not self.visible_objects_only:
            event = "replan-prompt" if is_replan else "plan-prompt"
            self.update_visible_objects(event=event)

        # Capture frames
        frames = self.capture_frames()
        
        # Stitch into composite
        composite = self.stitch_frames(frames)
        
        # Get scene state
        state = self.get_scene_state()
        
        # Build prompts
        system_prompt = self.build_system_prompt()
        state_text = self.state_to_pddl_text(state)
        visible_objects_text = self.visible_objects_to_text(state)
        user_prompt = self.build_user_prompt(
            state, goal, is_replan, failure_context, previous_plan
        )
        
        print(f"[Context Aggregator] Captured {len(frames)} frames")
        print(f"[Context Aggregator] State text preview:\n{state_text}")
        print(f"[Context Aggregator] User prompt length: {len(user_prompt)}")
        
        return PromptBundle(
            composite_image=composite,
            individual_frames=frames,
            state_text=state_text,
            visible_objects_text=visible_objects_text,
            goal_text=goal,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            is_replan=is_replan,
            failure_context=failure_context,
            unknown_object_crop=unknown_object_crop,
            known_objects=[obj.pddl_name for obj in state.objects],
            known_regions=[r['name'] for r in state.regions],
            previous_plan=previous_plan
        )

    
    def create_prompt_bundle_offline(self, goal: str, 
                                      frames: Optional[Dict[str, np.ndarray]] = None,
                                      state: Optional[SceneState] = None) -> PromptBundle:
        """
        Create prompt bundle without live environment (for testing).
        
        Args:
            goal: Natural language goal
            frames: Pre-captured frames (or None for dummy)
            state: Pre-built state (or None for default)
            
        Returns:
            PromptBundle
        """
        # Use provided frames or create dummy
        if frames is None:
            frames = {name: np.zeros((480, 640, 3), dtype=np.uint8) 
                     for name in self.camera_names}
        
        # Stitch
        composite = self.stitch_frames(frames)
        
        # Use provided state or default
        if state is None:
            state = self.get_scene_state()
        
        # Build prompts
        system_prompt = self.build_system_prompt()
        state_text = self.state_to_pddl_text(state)
        visible_objects_text = self.visible_objects_to_text(state)
        user_prompt = self.build_user_prompt(state, goal)
        
        return PromptBundle(
            composite_image=composite,
            individual_frames=frames,
            state_text=state_text,
            visible_objects_text=visible_objects_text,
            goal_text=goal,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            is_replan=False,
            failure_context=None,
            unknown_object_crop=None,
            known_objects=[obj.pddl_name for obj in state.objects],
            known_regions=[r['name'] for r in state.regions],
            previous_plan=None
        )
    
    def trigger_replan(self, failure_reason: str,
                       previous_plan: List[str],
                       goal: str,
                       unknown_object_crop: Optional[np.ndarray] = None) -> PromptBundle:
        """
        Create a replanning prompt bundle after a failure/interruption.
        
        This is the placeholder for the interruption system.
        Call this when:
        - An unknown object is detected in the scene
        - Execution fails (collision, IK failure, etc.)
        - User manually triggers replanning
        
        Args:
            failure_reason: Description of why replanning is needed
            previous_plan: The plan that was being executed
            goal: The original goal
            unknown_object_crop: Cropped image of unknown/blocking object
            
        Returns:
            PromptBundle for replanning query
        """
        return self.create_prompt_bundle(
            goal=goal,
            is_replan=True,
            failure_context=failure_reason,
            previous_plan=previous_plan,
            unknown_object_crop=unknown_object_crop
        )


# ============================================================================
# TESTING
# ============================================================================

def test_offline():
    """Test the context aggregator without environment."""
    print("Testing VLMContextAggregator (offline mode)...")
    
    aggregator = VLMContextAggregator()
    
    goal = "Move mug2 and mug4 to placement_boundary. Move soup to cupboard_boundary."
    
    bundle = aggregator.create_prompt_bundle_offline(goal)
    
    print("\n=== SYSTEM PROMPT ===")
    print(bundle.system_prompt)
    print("\n=== USER PROMPT ===")
    print(bundle.user_prompt)
    print("\n=== COMPOSITE IMAGE SHAPE ===")
    print(bundle.composite_image.shape)
    
    # Test replan prompt
    print("\n" + "="*50)
    print("Testing REPLAN prompt...")
    
    replan_bundle = aggregator.create_prompt_bundle_offline(
        goal=goal,
        state=aggregator.get_scene_state()
    )
    
    # Manually set replan context
    replan_bundle.is_replan = True
    replan_bundle.failure_context = "Unknown object detected blocking path to box_lid"
    replan_bundle.previous_plan = ["1. pick(mug2)", "2. place(mug2, placement_boundary)"]
    
    print("Replan bundle created successfully.")


if __name__ == "__main__":
    test_offline()

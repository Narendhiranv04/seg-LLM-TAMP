"""Modular context builder using 3D geometric resolution and multi-view stitching."""

from __future__ import annotations
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from llm_pipeline.pipeline_types import (
    BaseContextBuilder, SceneState, PromptBundle, FailureEvent, ICLMode
)
from llm_pipeline.geometric_utils import resolve_region
from llm_pipeline.region_aliases import PLANNER_HIDDEN_REGIONS, normalize_region_name, region_semantics


class GeometricContextBuilder(BaseContextBuilder):
    """
    Builds multimodal prompts using 3D geometric resolution.
    Ported logic from seg2/vlm_context_aggregator.py
    """

    def __init__(
        self,
        system_prompt_path: Optional[str] = None,
        user_prompt_path: Optional[str] = None,
        camera_names: List[str] = None,
        layout: str = "grid"
    ):
        self.system_prompt_template = self._load_template(system_prompt_path)
        self.user_prompt_template = self._load_template(user_prompt_path)
        self.camera_names = camera_names or ['left', 'right', 'overhead', 'wrist', 'front']
        self.layout = layout
        self.env = None

    def set_env(self, env) -> None:
        self.env = env

    def set_symbol_registry(self, symbol_registry: Any) -> None:
        pass # Geometric builder uses standalone resolution for now

    def _load_template(self, path: Optional[str]) -> Optional[str]:
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                return f.read()
        return None

    def stitch_frames(self, frames: Dict[str, np.ndarray]) -> np.ndarray:
        """Stitch camera frames into a single composite image."""
        ordered_frames = [frames[name] for name in self.camera_names if name in frames]
        if not ordered_frames:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        target_h, target_w = ordered_frames[0].shape[:2]
        resized = []
        for f in ordered_frames:
            if f.shape[:2] != (target_h, target_w):
                import cv2
                f = cv2.resize(f, (target_w, target_h))
            resized.append(f)

        if self.layout == "grid":
            # 2x3 grid
            while len(resized) < 6:
                resized.append(np.zeros_like(resized[0]))
            row1 = np.concatenate(resized[:3], axis=1)
            row2 = np.concatenate(resized[3:6], axis=1)
            return np.concatenate([row1, row2], axis=0)
        else:
            return np.concatenate(resized, axis=1)

    def build_bundle(
        self,
        state: SceneState,
        goal_text: str,
        failure_event: Optional[FailureEvent] = None,
        previous_actions: List[str] = None,
        icl_mode: str = ICLMode.ZERO_SHOT.value
    ) -> PromptBundle:
        
        # 1. Image Stitching
        composite = None
        if state.images:
            # If state already has images (e.g. from a previous perception step)
            # we assume they are the individual frames.
            # In a real environment, state.images would be a dict Cam -> Array
            pass
        
        # 2. State to PDDL-style Text
        obs_lines = []
        obs_lines.append("## Robot State:")
        obs_lines.append(f"- gripper: {state.gripper_state.get('status', 'empty')}")
        if state.gripper_state.get('holding'):
            obs_lines.append(f"- holding: {state.gripper_state['holding']}")
        
        region_map = getattr(state, 'region_map', {}) # Fallback to state's pre-resolved map if available
        object_region_map = getattr(state, 'object_region_map', {}) or {}
        object_region_descriptions = getattr(state, 'object_region_descriptions', {}) or {}
        valid_regions = [
            region for region in state.valid_regions
            if normalize_region_name(region) not in set(PLANNER_HIDDEN_REGIONS)
        ]
        if valid_regions:
            obs_lines.append("\n## Valid Target Regions:")
            obs_lines.append(", ".join(valid_regions))
            obs_lines.append("\n## Region Meanings:")
            for region in valid_regions:
                meaning = region_semantics(region)
                if meaning:
                    obs_lines.append(f"- {region}: {meaning}")
        
        obs_lines.append("\n## Object States (Geometric):")
        for obj_name in state.visible_objects:
            pos = state.pose_map.get(obj_name)
            r_id = object_region_map.get(obj_name)
            r_desc = object_region_descriptions.get(obj_name)
            if pos:
                if not r_id:
                    r_id, r_desc = resolve_region(pos, region_map, state.valid_regions)
                obs_lines.append(f"- {obj_name}: region={r_id}, description={r_desc}, pose={tuple(np.round(pos, 3))}")
            elif r_id:
                obs_lines.append(f"- {obj_name}: region={r_id}, description={r_desc or '(none)'}, pose=unresolved")
            else:
                obs_lines.append(f"- {obj_name}: visible but pose unresolved")
        
        # Use the pre-computed pddl_state if available
        if state.pddl_state:
            obs_lines.extend(state.pddl_state)
        else:
            obs_lines.append("(No symbolic state retrieved)")

        observation_text = "\n".join(obs_lines)

        # 3. Assemble Prompts
        system_prompt = self.system_prompt_template or "Perform the task."
        
        user_prompt = ""
        if failure_event:
            user_prompt += "=== REPLANNING AFTER FAILURE ===\n"
            user_prompt += f"FAILURE_ID: {failure_event.failure_id}\n"
            user_prompt += f"STAGE: {failure_event.stage.value}\n"
            user_prompt += f"ACTION_FAILED: {failure_event.action or '(none)'}\n"
            user_prompt += f"ERROR: {failure_event.message}\n"
            if previous_actions:
                user_prompt += f"COMPLETED_ACTIONS: {', '.join(previous_actions)}\n"
            user_prompt += "================================\n\n"
        
        user_prompt += f"### Current State\n{observation_text}\n\n### Goal\n{goal_text}"

        return PromptBundle(
            goal_text=goal_text,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            visible_objects=state.visible_objects,
            valid_regions=state.valid_regions,
            icl_mode=icl_mode,
            images=[state.images] if state.images is not None else None,
            previous_actions=tuple(previous_actions) if previous_actions else (),
            failure_context=failure_event.message if failure_event else None
        )

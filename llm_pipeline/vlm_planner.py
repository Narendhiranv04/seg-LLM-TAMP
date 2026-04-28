"""Modular VLM planner wrapper."""

from __future__ import annotations
import time
from typing import Optional, List
import numpy as np
from llm_pipeline.pipeline_types import BasePlanner, PromptBundle, PlanResult, DirectAction


class VLMPlanner(BasePlanner):
    """
    Modular wrapper for VLM inference.
    Delegates to the existing vlm_pipeline.vlm_planner.VLMPlanner
    but follows the BasePlanner interface.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        use_4bit: bool = False,
        trust_remote_code: bool = True
    ):
        from vlm_pipeline.vlm_planner import VLMPlanner as LegacyVLMPlanner
        self.legacy_planner = LegacyVLMPlanner(
            model_id=model_path,
            device=device,
            load_in_4bit=use_4bit,
            trust_remote_code=trust_remote_code
        )
        self.legacy_planner.load_model()

    def plan(self, bundle: PromptBundle) -> PlanResult:
        start_time = time.time()
        
        # 1. Prepare image for legacy planner
        # The legacy planner expects a single composite image or a List[Image]
        composite = None
        if bundle.images and len(bundle.images) > 0:
            composite = bundle.images[0]  # Assume first image is the composite

        # 2. Call legacy planner
        # Note: Legacy planner uses its own state resolution internally if we give it env,
        # but we want to use the text we already built in the bundle.
        # We'll use a slightly modified call or direct inference if needed.
        
        # For now, we utilize the generate_plan method which takes system/user prompts
        legacy_result = self.legacy_planner.generate_plan(
            system_prompt=bundle.system_prompt,
            user_prompt=bundle.user_prompt,
            composite_image=composite
        )
        
        # 3. Map legacy ActionSkeleton to DirectAction
        actions = []
        if legacy_result.success:
            for skeleton in legacy_result.skeleton:
                actions.append(DirectAction(
                    action_name=skeleton.action_name,
                    args=skeleton.args
                ))
        
        return PlanResult(
            success=legacy_result.success,
            actions=actions,
            raw_output=legacy_result.raw_output,
            inference_time=time.time() - start_time,
            error_message=legacy_result.error_message
        )

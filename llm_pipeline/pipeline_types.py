"""Shared datatypes for the maintained text-only pipeline."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ICLMode(str, Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT_SHARED_1 = "few_shot_shared_1"


class FailureStage(str, Enum):
    BEFORE_EXECUTION = "before_execution"
    AFTER_EXECUTION = "after_execution"


class FailureSource(str, Enum):
    SEGMENTATION = "segmentation"
    PDDL = "pddl"
    GEOMETRY = "geometry"
    EXECUTOR = "executor"
    VALIDATION = "validation"
    PARSER = "parser"


@dataclass(frozen=True)
class DirectAction:
    """A directly executable action line from the LLM."""

    action_name: str
    args: Tuple[str, ...]

    def __str__(self) -> str:
        if not self.args:
            return self.action_name
        return f"{self.action_name}({', '.join(self.args)})"


@dataclass
class PlanResult:
    success: bool
    actions: List[DirectAction]
    raw_output: str
    inference_time: float
    error_message: Optional[str] = None
    failure_event: Optional['FailureEvent'] = None


@dataclass(frozen=True)
class PromptBundle:
    """Unified prompt bundle for both LLM and VLM planners."""

    goal_text: str
    system_prompt: str
    user_prompt: str
    visible_objects: List[str]
    valid_regions: List[str]
    icl_mode: str
    images: Optional[List[np.ndarray]] = None  # Local images
    image_paths: Optional[List[str]] = None      # Remote image references
    previous_actions: Tuple[str, ...] = ()
    failure_context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextPromptBundle:
    """Intermediate bundle for text-only planning."""

    goal_text: str
    observation_text: str
    visible_objects_text: str
    icl_mode: str
    previous_actions: Tuple[str, ...] = ()
    failure_context: Optional[str] = None


@dataclass
class SegmentationObjectEvidence:
    """Per-object evidence fused only from segmentation masks."""

    name: str
    visible: bool
    camera_hits: List[str] = field(default_factory=list)
    pixel_count: int = 0
    camera_pixels: Dict[str, int] = field(default_factory=dict)
    bbox: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)
    centroid: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    mask_regions: List[str] = field(default_factory=list)
    region_votes: Dict[str, float] = field(default_factory=dict)
    newly_visible: bool = False
    gripper_proximity: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "visible": self.visible,
            "camera_hits": list(self.camera_hits),
            "pixel_count": int(self.pixel_count),
            "camera_pixels": {k: int(v) for k, v in self.camera_pixels.items()},
            "bbox": {k: tuple(v) for k, v in self.bbox.items()},
            "centroid": {k: tuple(v) for k, v in self.centroid.items()},
            "mask_regions": list(self.mask_regions),
            "region_votes": {k: float(v) for k, v in self.region_votes.items()},
            "newly_visible": bool(self.newly_visible),
            "gripper_proximity": self.gripper_proximity,
        }


@dataclass
class SegmentationSnapshot:
    """Current fused segmentation evidence."""

    frame_index: int
    visible_objects: List[str]
    newly_visible_objects: List[str]
    object_evidence: Dict[str, SegmentationObjectEvidence]
    gripper_evidence: Dict[str, Any] = field(default_factory=dict)
    supported_regions: List[str] = field(default_factory=list)
    visible_regions: List[str] = field(default_factory=list)
    object_region_map: Dict[str, str] = field(default_factory=dict)
    object_region_descriptions: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "visible_objects": list(self.visible_objects),
            "newly_visible_objects": list(self.newly_visible_objects),
            "object_evidence": {
                name: evidence.to_dict()
                for name, evidence in self.object_evidence.items()
            },
            "gripper_evidence": dict(self.gripper_evidence),
            "supported_regions": list(self.supported_regions),
            "visible_regions": list(self.visible_regions),
            "object_region_map": dict(self.object_region_map),
            "object_region_descriptions": dict(self.object_region_descriptions),
        }


@dataclass
class FailureEvent:
    """A structured failure event for replanning decisions."""

    failure_id: str
    stage: FailureStage
    source: FailureSource
    action: Optional[str]
    evidence: Dict[str, Any]
    should_replan: bool = True
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "stage": self.stage.value,
            "source": self.source.value,
            "action": self.action,
            "evidence": dict(self.evidence),
            "should_replan": bool(self.should_replan),
            "message": self.message,
        }


@dataclass
class SceneState:
    """Unified state representation of the environment at a specific time."""

    frame_index: int
    visible_objects: List[str]
    valid_regions: List[str]
    pddl_state: List[str] = field(default_factory=list)  # symbolic facts
    masks: Dict[str, np.ndarray] = field(default_factory=dict)  # camera-name -> mask
    pose_map: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)  # obj -> (x,y,z)
    images: Optional[List[np.ndarray]] = None  # Stitched or raw RGB views
    gripper_state: Dict[str, Any] = field(default_factory=dict)
    region_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # region -> (min, max)
    object_region_map: Dict[str, str] = field(default_factory=dict)  # obj -> canonical region
    object_region_descriptions: Dict[str, str] = field(default_factory=dict)  # obj -> human-readable location

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "visible_objects": list(self.visible_objects),
            "valid_regions": list(self.valid_regions),
            "pddl_state": list(self.pddl_state),
            "pose_map": {k: list(v) for k, v in self.pose_map.items()},
            "gripper_state": dict(self.gripper_state),
            "object_region_map": dict(self.object_region_map),
            "object_region_descriptions": dict(self.object_region_descriptions),
        }


class BaseContextBuilder:
    """Abstract base class for building a PromptBundle from a SceneState."""

    def build_bundle(
        self,
        state: SceneState,
        goal_text: str,
        failure_event: Optional[FailureEvent] = None,
        previous_actions: List[str] = None,
    ) -> PromptBundle:
        raise NotImplementedError


class BasePlanner:
    """Abstract base class for planners (Local/Remote, LLM/VLM)."""

    def plan(self, bundle: PromptBundle) -> PlanResult:
        raise NotImplementedError


class BaseFailureChecker:
    """Abstract base class for validating action execution."""

    def check(
        self,
        action: DirectAction,
        state: SceneState,
        held_object: Optional[str] = None,
    ) -> Optional[FailureEvent]:
        raise NotImplementedError

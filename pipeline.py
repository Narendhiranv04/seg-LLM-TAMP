"""Root-level entry point for the LLM/VLM replanning pipeline.

This file is a re-export of the modular implementation in llm_pipeline.pipeline.
Use the flags in LLMPipelineConfig to toggle between Text-Only, Multimodal, and Geometric modes.
"""

from llm_pipeline.pipeline import (
    LLMPipelineConfig,
    LLMOnlyReplanningPipeline,
    ExecutionCycleRecord,
    PROMPT_MODE_SEGMENTATION_TEXT
)

# Also re-export common types for downstream scripts
from llm_pipeline.types import (
    SceneState,
    PromptBundle,
    DirectAction,
    FailureEvent,
    PlanResult,
    ICLMode
)

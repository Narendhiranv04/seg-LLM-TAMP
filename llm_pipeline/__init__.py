"""Maintained text-only LLM planning pipeline."""

from llm_pipeline.catalog import FINAL_MODEL_TRIO, SHARED_ICL_MODES, list_candidate_llms
from llm_pipeline.strict_parser import StrictActionParser, StrictParseError
from llm_pipeline.pipeline_types import (
    DirectAction,
    FailureEvent,
    FailureSource,
    FailureStage,
    ICLMode,
    PlanResult,
    SegmentationObjectEvidence,
    SegmentationSnapshot,
    TextPromptBundle,
)

__all__ = [
    "DirectAction",
    "FailureEvent",
    "FailureSource",
    "FailureStage",
    "FINAL_MODEL_TRIO",
    "ICLMode",
    "LLMOnlyReplanningPipeline",
    "LLMPipelineConfig",
    "PlanResult",
    "SHARED_ICL_MODES",
    "SegmentationObjectEvidence",
    "SegmentationSnapshot",
    "StrictActionParser",
    "StrictParseError",
    "TextOnlyContextBuilder",
    "TextPromptBundle",
    "list_candidate_llms",
]


def __getattr__(name):
    if name in {"LLMOnlyReplanningPipeline", "LLMPipelineConfig"}:
        from llm_pipeline.pipeline import LLMOnlyReplanningPipeline, LLMPipelineConfig

        mapping = {
            "LLMOnlyReplanningPipeline": LLMOnlyReplanningPipeline,
            "LLMPipelineConfig": LLMPipelineConfig,
        }
        return mapping[name]
    if name == "TextOnlyContextBuilder":
        from llm_pipeline.prompt_builder import TextOnlyContextBuilder

        return TextOnlyContextBuilder
    raise AttributeError(f"module 'llm_pipeline' has no attribute {name!r}")

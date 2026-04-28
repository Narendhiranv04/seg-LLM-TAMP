"""LLM model catalog for the maintained text-only pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from vlm_pipeline.model_registry import ModelSpec, list_models, resolve_model_spec

from llm_pipeline.pipeline_types import ICLMode


FINAL_MODEL_TRIO = ["qwen", "deepseek-r1", "mistral-nemo"]
SHARED_ICL_MODES = [ICLMode.ZERO_SHOT.value, ICLMode.FEW_SHOT_SHARED_1.value]


@dataclass(frozen=True)
class LLMModelChoice:
    alias: str
    path: str
    description: str


def resolve_llm_model(alias_or_path: str) -> ModelSpec:
    spec = resolve_model_spec(alias_or_path, "llm")
    if spec.model_type != "llm":
        raise ValueError(f"Model '{alias_or_path}' is not an LLM entry")
    return spec


def list_candidate_llms() -> List[LLMModelChoice]:
    choices: List[LLMModelChoice] = []
    for spec in list_models():
        if spec.model_type != "llm":
            continue
        choices.append(
            LLMModelChoice(
                alias=spec.alias,
                path=spec.path,
                description=spec.description,
            )
        )
    return choices

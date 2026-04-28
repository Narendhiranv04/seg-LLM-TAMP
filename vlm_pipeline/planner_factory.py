"""
Planner factory helpers for local Hugging Face backends.
"""

from vlm_pipeline.llm_planner import LLMPlanner
from vlm_pipeline.model_registry import ModelSpec, resolve_model_spec
from vlm_pipeline.vlm_planner import MockVLMPlanner, VLMPlanner


class ModelLoadError(ValueError):
    pass



def create_local_planner(model: str = "",
                         model_type: str = "",
                         use_4bit: bool = False,
                         device: str = "cuda",
                         use_mock: bool = False):
    if use_mock:
        return MockVLMPlanner()

    spec = resolve_model_spec(model=model, model_type=model_type)
    if spec.model_type == "llm":
        return LLMPlanner(
            model_name=spec.path,
            use_4bit=use_4bit,
            device=device,
            model_alias=spec.alias,
            model_type=spec.model_type,
        )

    return VLMPlanner(
        model_name=spec.path,
        use_4bit=use_4bit,
        device=device,
        model_alias=spec.alias,
        model_type=spec.model_type,
    )



def resolve_planner_spec(model: str = "", model_type: str = "") -> ModelSpec:
    return resolve_model_spec(model=model, model_type=model_type)

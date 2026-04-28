"""
Shared model registry for local and remote planner selection.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List


PROMPT_MODE_STATE_TEXT = "state_text"
PROMPT_MODE_VISION_GOAL = "vision_goal_only"
PROMPT_MODE_TEXT_VISIBLE = "text_visible_objects"


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    path: str
    model_type: str  # "vlm" or "llm"
    description: str
    prompt_mode: str
    use_vision: bool

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)



def _make_spec(alias: str, path: str, model_type: str, description: str) -> ModelSpec:
    is_vlm = model_type == "vlm"
    return ModelSpec(
        alias=alias,
        path=path,
        model_type=model_type,
        description=description,
        prompt_mode=PROMPT_MODE_VISION_GOAL if is_vlm else PROMPT_MODE_TEXT_VISIBLE,
        use_vision=is_vlm,
    )


MODEL_SPECS: Dict[str, ModelSpec] = {
    "qwen-vl": _make_spec(
        "qwen-vl",
        "Qwen/Qwen3-VL-8B-Thinking",
        "vlm",
        "Qwen3 Vision-Language Model for scene understanding",
    ),
    "gsarch": _make_spec(
        "gsarch",
        "gsarch/ViGoRL-MCTS-SFT-7b-Spatial",
        "vlm",
        "Spatial reasoning VLM optimized for robotic tasks",
    ),
    "ms-phi4": _make_spec(
        "ms-phi4",
        "microsoft/Phi-4-reasoning-vision-15B",
        "vlm",
        "Visual Chain-of-Thought model for image-to-logic reasoning",
    ),
    "spatial-ladder": _make_spec(
        "spatial-ladder",
        "hongxingli/SpatialLadder-3B",
        "vlm",
        "Lightweight spatial VLM for fast localization and distance estimation (smoke test before evaluation)",
    ),
    "internvl-3.5": _make_spec(
        "internvl-3.5",
        "OpenGVLab/InternVL3_5-8B",
        "vlm",
        "High-resolution VLM for dense perception",
    ),
    "qwen": _make_spec(
        "qwen",
        "Qwen/Qwen3-8B",
        "llm",
        "Qwen3 language model for planning",
    ),
    "selene": _make_spec(
        "selene",
        "AItalai/Selene-1-Mini-Llama-3.1-8B",
        "llm",
        "Selene Mini for planning validation",
    ),
    "deepseek-r1": _make_spec(
        "deepseek-r1",
        "casperhansen/deepseek-r1-distill-qwen-7b-awq",
        "llm",
        "Reasoning model for structured planning text",
    ),
    "deepseek-r1-qwen-12b": _make_spec(
        "deepseek-r1-qwen-12b",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "llm",
        "Compatibility alias for the official DeepSeek Qwen-14B distilled checkpoint",
    ),
    "deepseek-r1-qwen-14b": _make_spec(
        "deepseek-r1-qwen-14b",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "llm",
        "Official DeepSeek Qwen-14B distilled reasoning model",
    ),
    "deepseek-r1-llama-8b": _make_spec(
        "deepseek-r1-llama-8b",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "llm",
        "Official DeepSeek Llama-8B distilled reasoning model",
    ),
    "mistral-nemo": _make_spec(
        "mistral-nemo",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "llm",
        "Logic-focused instruct model for plan validation",
    ),
}


DEFAULT_MODEL_ALIAS = "qwen-vl"



def list_models() -> List[ModelSpec]:
    return [MODEL_SPECS[name] for name in sorted(MODEL_SPECS.keys())]



def infer_model_type(model_name_or_path: str) -> str:
    lowered = (model_name_or_path or "").lower()
    if any(token in lowered for token in ("-vl", "vision", "internvl", "spatialladder", "vigor")):
        return "vlm"
    return "llm"



def resolve_model_spec(model: str = "", model_type: str = "") -> ModelSpec:
    key = (model or "").strip()
    explicit_type = (model_type or "").strip().lower()

    if not key:
        key = DEFAULT_MODEL_ALIAS

    if key in MODEL_SPECS:
        spec = MODEL_SPECS[key]
        if explicit_type and explicit_type != spec.model_type:
            raise ValueError(
                f"Model '{key}' is registered as type '{spec.model_type}', not '{explicit_type}'"
            )
        return spec

    inferred_type = explicit_type or infer_model_type(key)
    if inferred_type not in {"vlm", "llm"}:
        raise ValueError(f"Unsupported model type: {inferred_type}")

    alias = key.rsplit("/", 1)[-1] if "/" in key else key
    return _make_spec(
        alias=alias,
        path=key,
        model_type=inferred_type,
        description="Custom Hugging Face model path",
    )



def format_model_listing() -> str:
    lines = ["Available planner models:"]
    for spec in list_models():
        lines.append(
            f"  - {spec.alias}: type={spec.model_type}, path={spec.path} :: {spec.description}"
        )
    return "\n".join(lines)

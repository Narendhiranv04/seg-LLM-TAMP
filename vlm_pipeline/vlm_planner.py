"""
VLM Planner (Module 2)
======================
Vision-Language Model planner using Qwen2-VL-7B-Instruct.
Takes visual context + state + goal and outputs action skeletons.
"""

import os
import sys
import re
import time
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Try to import planner dependencies
try:
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None
    try:
        from transformers import AutoModelForVision2Seq
    except ImportError:
        AutoModelForVision2Seq = None
    try:
        from transformers import Phi4MultimodalForCausalLM
    except ImportError:
        Phi4MultimodalForCausalLM = None
    try:
        from transformers import Qwen2VLForConditionalGeneration
    except ImportError:
        Qwen2VLForConditionalGeneration = None
    try:
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError:
        Qwen3VLForConditionalGeneration = None
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoModel = None
    AutoModelForCausalLM = None
    AutoProcessor = None
    AutoModelForImageTextToText = None
    AutoModelForVision2Seq = None
    Phi4MultimodalForCausalLM = None
    Qwen2VLForConditionalGeneration = None
    Qwen3VLForConditionalGeneration = None
    print("Warning: planner dependencies not installed.")
    print("Install with: pip install transformers torch accelerate")

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    process_vision_info = None
    HAS_QWEN_VL_UTILS = False


@dataclass
class ActionSkeleton:
    """Represents a single action in the plan skeleton."""
    action_name: str  # 'move', 'pick', 'place', 'open-lid'
    args: Tuple[str, ...]  # ('mug2',) or ('mug2', 'placement_boundary')
    
    def __str__(self):
        return f"{self.action_name}({', '.join(self.args)})"
    
    def to_tuple(self):
        """Convert to tuple format for compatibility."""
        return (self.action_name,) + self.args


@dataclass
class PlanResult:
    """Result from the VLM planner."""
    success: bool
    skeleton: List[ActionSkeleton]
    raw_output: str
    inference_time: float
    error_message: Optional[str] = None


class VLMPlanner:
    """
    Vision-Language Model planner using Qwen2-VL-7B-Instruct.
    """
    
    # Valid action patterns
    VALID_ACTIONS = {
        'move': 2,      # move(object[, region])
        'pick': 1,      # pick(object)
        'place': 2,     # place(object, region)
        'open-lid': 1,  # open-lid(lid)
        'open_lid': 1,  # alternate format
    }
    
    # Known objects and regions for validation. These can be overridden
    # at runtime from the segmentation-derived prompt bundle.
    KNOWN_OBJECTS = {
        'mug1', 'mug2', 'mug3', 'mug4',
        'soup', 'mustard', 'spam', 'sugar', 'crackers', 'box_lid'
    }
    
    KNOWN_REGIONS = {
        'table', 'box_boundary', 'placement_boundary',
        'cupboard_boundary', 'cupboard_boundary_top', 'groceries_boundary'
    }
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
                 use_4bit: bool = False,  # 2B fits without quantization
                 device: str = "cuda",
                 model_alias: str = "",
                 model_type: str = "vlm"):
        """
        Initialize the VLM planner.
        
        Args:
            model_name: HuggingFace model name
            use_4bit: Whether to use 4-bit quantization
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.device = device
        self.model_alias = model_alias or model_name
        self.model_type = model_type
        
        self.model = None
        self.processor = None
        self.loaded = False
        self.model_family = ""
        self.model_loader_name = ""
        self.last_request_summary: Dict[str, Any] = {}
        self.known_objects = set(type(self).KNOWN_OBJECTS)
        self.known_regions = set(type(self).KNOWN_REGIONS)
        
    def set_known_entities(self,
                           objects: Optional[List[str]] = None,
                           regions: Optional[List[str]] = None):
        """Update planner validation symbols from the current observation."""
        self.known_objects = set(objects) if objects else set(type(self).KNOWN_OBJECTS)
        self.known_regions = set(regions) if regions else set(type(self).KNOWN_REGIONS)

    def _infer_model_family(self) -> str:
        lowered = self.model_name.lower()
        if ("qwen" in lowered and "-vl" in lowered) or "qwen2-vl" in lowered or "qwen3-vl" in lowered:
            return "qwen_vl"
        if "phi-4" in lowered and "vision" in lowered:
            return "phi4_multimodal"
        if "internvl" in lowered:
            return "internvl"
        if "vigorl" in lowered or "gsarch" in lowered:
            return "vigorl"
        if "spatialladder" in lowered or "spatial-ladder" in lowered:
            return "spatial_ladder"
        return "generic_vlm" if self.model_type == "vlm" else "generic_text"

    def _candidate_model_loaders(self) -> List[Tuple[str, Any]]:
        family = self._infer_model_family()
        loaders: List[Tuple[str, Any]] = []
        seen = set()

        def _append(name: str, cls):
            if cls is None or name in seen:
                return
            loaders.append((name, cls))
            seen.add(name)

        if family == "qwen_vl":
            _append("Qwen3VLForConditionalGeneration", Qwen3VLForConditionalGeneration)
            _append("Qwen2VLForConditionalGeneration", Qwen2VLForConditionalGeneration)
            _append("AutoModelForImageTextToText", AutoModelForImageTextToText)
            _append("AutoModelForVision2Seq", AutoModelForVision2Seq)
            _append("AutoModelForCausalLM", AutoModelForCausalLM)
            _append("AutoModel", AutoModel)
            return loaders

        if family == "phi4_multimodal":
            _append("Phi4MultimodalForCausalLM", Phi4MultimodalForCausalLM)
            _append("AutoModelForCausalLM", AutoModelForCausalLM)
            _append("AutoModelForImageTextToText", AutoModelForImageTextToText)
            _append("AutoModelForVision2Seq", AutoModelForVision2Seq)
            _append("AutoModel", AutoModel)
            return loaders

        if family in {"internvl", "vigorl", "spatial_ladder"}:
            _append("AutoModel", AutoModel)
            _append("AutoModelForImageTextToText", AutoModelForImageTextToText)
            _append("AutoModelForVision2Seq", AutoModelForVision2Seq)
            _append("AutoModelForCausalLM", AutoModelForCausalLM)
            return loaders

        if self.model_type == "vlm":
            _append("AutoModelForImageTextToText", AutoModelForImageTextToText)
            _append("AutoModelForVision2Seq", AutoModelForVision2Seq)

        _append("AutoModelForCausalLM", AutoModelForCausalLM)
        _append("AutoModel", AutoModel)
        return loaders

    def _load_processor(self):
        processor_kwargs = {"trust_remote_code": True}
        if self._infer_model_family() == "qwen_vl":
            processor_kwargs["use_fast"] = False

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name, **processor_kwargs)
        except TypeError as exc:
            if processor_kwargs.get("use_fast") is False:
                print(f"[VLM Planner] Processor load with use_fast=False failed: {exc}. Retrying with default processor config.")
                processor_kwargs.pop("use_fast", None)
                self.processor = AutoProcessor.from_pretrained(self.model_name, **processor_kwargs)
            else:
                raise

    def load_model(self) -> bool:
        """
        Load the active VLM model.

        Returns:
            True if successful, False otherwise
        """
        if not HAS_TRANSFORMERS:
            print("ERROR: planner dependencies are not available.")
            return False

        print(f"Loading VLM: {self.model_name}")
        print(f"Using 4-bit quantization: {self.use_4bit}")
        self.model_family = self._infer_model_family()
        print(f"Detected model family: {self.model_family}")
        model_loaders = self._candidate_model_loaders()
        if not model_loaders:
            print("ERROR: No compatible VLM loaders are available in this environment.")
            return False

        def _load_from_pretrained(model_kwargs):
            last_error = None
            for loader_name, loader_cls in model_loaders:
                try:
                    print(f"Trying loader: {loader_name}")
                    model = loader_cls.from_pretrained(self.model_name, **model_kwargs)
                    self.model_loader_name = loader_name
                    print(f"Loaded with: {loader_name}")
                    return model
                except Exception as exc:
                    last_error = exc
                    print(f"[Loader {loader_name}] failed: {exc}")
            if last_error is not None:
                raise last_error
            raise RuntimeError("No model loader available")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        try:
            self._load_processor()

            if self.use_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    print("Configuring 4-bit quantization...")

                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )

                    self.model = _load_from_pretrained({
                        "quantization_config": bnb_config,
                        "device_map": "auto",
                        "trust_remote_code": True,
                        "low_cpu_mem_usage": True,
                    })
                except Exception as exc:
                    print(f"4-bit quantization failed: {exc}")
                    print("Cleaning up memory and falling back to float16...")
                    self.model = None
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.use_4bit = False

            if not self.use_4bit:
                print("Loading with float16 precision...")
                self.model = _load_from_pretrained({
                    "device_map": "auto",
                    "torch_dtype": torch.float16,
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                })

            if hasattr(self.model, "eval"):
                self.model.eval()

            self.loaded = True
            print("VLM loaded successfully!")
            return True

        except Exception as e:
            print(f"Error loading VLM: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _is_qwen_style_model(self) -> bool:
        return self._infer_model_family() == "qwen_vl"

    def _numpy_to_pil(self, image: np.ndarray):
        from PIL import Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        return Image.fromarray(image)

    def _record_request(self,
                        system_prompt: str,
                        user_prompt: str,
                        use_vision: bool,
                        image: Optional[np.ndarray] = None):
        self.last_request_summary = {
            "timestamp": datetime.now().isoformat(),
            "model_alias": self.model_alias,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "use_vision": use_vision,
            "image_present": image is not None,
            "image_shape": list(image.shape) if image is not None else None,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "system_prompt_length": len(system_prompt),
            "user_prompt_length": len(user_prompt),
        }

    def _decode_generated_text(self, generated_ids, inputs) -> str:
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        if hasattr(self.processor, "batch_decode"):
            return self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        raise RuntimeError("Processor cannot decode model output")
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string."""
        from PIL import Image
        import base64
        import io
        
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def _build_generation_kwargs(self,
                                 max_new_tokens: int,
                                 temperature: float,
                                 pad_token_id: Optional[int] = None) -> Dict[str, Any]:
        capped_tokens = max(64, min(int(max_new_tokens), 1024))
        kwargs: Dict[str, Any] = {
            "max_new_tokens": capped_tokens,
            "do_sample": temperature > 0,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
        if pad_token_id is not None:
            kwargs["pad_token_id"] = pad_token_id
        return kwargs

    def _build_format_repair_prompt(self, user_prompt: str, bad_output: str) -> str:
        prior_output = bad_output.strip() or "(empty output)"
        return f"""{user_prompt}

FORMAT REPAIR:
Your previous answer was not parseable as executable actions:
{prior_output}

Rewrite the answer as ONLY a numbered list of executable actions using exactly these action forms:
1. pick(object)
2. place(object, region)
3. open-lid(box_lid)

Do not include reasoning, markdown, comments, or any text before/after the actions."""

    def _generate_multimodal_output(self,
                                    image: np.ndarray,
                                    system_prompt: str,
                                    user_prompt: str,
                                    max_new_tokens: int,
                                    temperature: float) -> str:
        pil_image = self._numpy_to_pil(image)

        if self._is_qwen_style_model() and HAS_QWEN_VL_UTILS and hasattr(self.processor, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"data:image/png;base64,{self._image_to_base64(image)}"
                        },
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            print(f"[VLM Planner] Input text length: {len(text)}")
            print(f"[VLM Planner] Input text preview: {text[:200]}...")

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            tokenizer = getattr(self.processor, "tokenizer", None)
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(tokenizer, "eos_token_id", None)
            generation_kwargs = self._build_generation_kwargs(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_token_id=pad_token_id,
            )
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **generation_kwargs)
            return self._decode_generated_text(generated_ids, inputs)

        prompt_text = f"{system_prompt}\n\n{user_prompt}"
        inputs = self.processor(
            text=[prompt_text],
            images=[pil_image],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        generation_kwargs = self._build_generation_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)
        return self._decode_generated_text(generated_ids, inputs)

    def _generate_text_output(self,
                              system_prompt: str,
                              user_prompt: str,
                              max_new_tokens: int,
                              temperature: float) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        if hasattr(self.processor, "apply_chat_template"):
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = f"{system_prompt}\n\n{user_prompt}\n"

        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        generation_kwargs = self._build_generation_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)
        return self._decode_generated_text(generated_ids, inputs)

    def generate_plan(self,
                      image: np.ndarray,
                      system_prompt: str,
                      user_prompt: str,
                      max_new_tokens: int = 1024,
                      temperature: float = 0.1) -> PlanResult:
        """
        Generate an action plan from visual context and prompts.
        """
        if not self.loaded:
            return PlanResult(
                success=False,
                skeleton=[],
                raw_output="",
                inference_time=0,
                error_message="Model not loaded. Call load_model() first."
            )

        start_time = time.time()
        self._record_request(system_prompt, user_prompt, use_vision=True, image=image)

        try:
            output_text = self._generate_multimodal_output(
                image=image,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            print(f"[VLM Planner] Raw output: {output_text}")

            skeleton = self.parse_plan(output_text)
            if not skeleton:
                print("[VLM Planner] No parseable actions found. Retrying once with format repair.")
                repaired_output = self._generate_multimodal_output(
                    image=image,
                    system_prompt=system_prompt,
                    user_prompt=self._build_format_repair_prompt(user_prompt, output_text),
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                )
                print(f"[VLM Planner] Repaired raw output: {repaired_output}")
                repaired_skeleton = self.parse_plan(repaired_output)
                if repaired_skeleton:
                    output_text = repaired_output
                    skeleton = repaired_skeleton

            inference_time = time.time() - start_time
            error_message = None if skeleton else "No parseable actions found in VLM output"
            return PlanResult(
                success=len(skeleton) > 0,
                skeleton=skeleton,
                raw_output=output_text,
                inference_time=inference_time,
                error_message=error_message,
            )
        except Exception as e:
            inference_time = time.time() - start_time
            return PlanResult(
                success=False,
                skeleton=[],
                raw_output="",
                inference_time=inference_time,
                error_message=str(e),
            )

    def generate_plan_text_only(self,
                                system_prompt: str,
                                user_prompt: str,
                                max_new_tokens: int = 1024,
                                temperature: float = 0.1) -> PlanResult:
        """
        Generate plan without image (text-only mode for testing).
        Falls back to text-only generation on the loaded model.
        """
        if not self.loaded:
            return PlanResult(
                success=False,
                skeleton=[],
                raw_output="",
                inference_time=0,
                error_message="Model not loaded."
            )

        start_time = time.time()
        self._record_request(system_prompt, user_prompt, use_vision=False)

        try:
            output_text = self._generate_text_output(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            skeleton = self.parse_plan(output_text)

            if not skeleton:
                print("[VLM Planner] Text-only parse yielded no actions. Retrying once with format repair.")
                repaired_output = self._generate_text_output(
                    system_prompt=system_prompt,
                    user_prompt=self._build_format_repair_prompt(user_prompt, output_text),
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                )
                repaired_skeleton = self.parse_plan(repaired_output)
                if repaired_skeleton:
                    output_text = repaired_output
                    skeleton = repaired_skeleton

            inference_time = time.time() - start_time
            error_message = None if skeleton else "No parseable actions found in VLM output"
            return PlanResult(
                success=len(skeleton) > 0,
                skeleton=skeleton,
                raw_output=output_text,
                inference_time=inference_time,
                error_message=error_message,
            )
        except Exception as e:
            return PlanResult(
                success=False,
                skeleton=[],
                raw_output="",
                inference_time=time.time() - start_time,
                error_message=str(e)
            )

    def get_debug_info(self) -> Dict[str, Any]:
        return {
            "model_alias": self.model_alias,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "model_family": self.model_family,
            "model_loader_name": self.model_loader_name,
            "loaded": self.loaded,
            "last_request": self.last_request_summary,
        }

    def parse_plan(self, text: str) -> List[ActionSkeleton]:
        """
        Parse VLM output text into action skeletons.

        Handles formats like:
        - "1. pick(mug2)"
        - "pick(mug2)"
        - "- pick(mug2)"

        Args:
            text: Raw output text from VLM

        Returns:
            List of ActionSkeleton objects
        """
        if not text:
            return []

        think_close = re.search(r'</think\s*>', text, flags=re.IGNORECASE)
        if think_close:
            text = text[think_close.end():]
        else:
            text = re.sub(r'</?think\s*>', '', text, flags=re.IGNORECASE)

        actions = []
        seen_pick_place_pairs = set()  # Track (object, region) pairs to detect duplicates
        action_pattern = re.compile(r'([A-Za-z][A-Za-z0-9_-]*)\s*\(\s*([^)]*)\s*\)')

        def _normalize_token(token: str) -> str:
            """Normalize an action/object/region token."""
            return token.strip().lower().replace(" ", "_")

        def _normalize_action_name(name: str) -> str:
            """
            Normalize common model output action variants to canonical names.
            """
            n = _normalize_token(name).replace("-", "_")
            if n in {"pick", "pickup", "pick_up", "grasp"}:
                return "pick"
            if n in {"place", "put", "put_down", "putdown"}:
                return "place"
            if n in {"open_lid", "openlid", "open_box_lid", "open_boxlid", "open_box"}:
                return "open-lid"
            if n in {"move_object_to", "move_to", "move_object", "move"}:
                return "move"
            return n.replace("_", "-")

        def _append_action(action_name: str, args: List[str]):
            """Append action with duplicate-pair protection."""
            if len(actions) >= 25:
                return

            action = ActionSkeleton(action_name=action_name, args=tuple(args))
            if action_name == 'place' and len(actions) >= 1:
                prev = actions[-1]
                if prev.action_name == 'pick':
                    pair_key = (prev.args[0], args[1])
                    if pair_key in seen_pick_place_pairs:
                        print(f"[Parser] Skipping duplicate pick-place: {pair_key}")
                        actions.pop()
                        return
                    seen_pick_place_pairs.add(pair_key)
            actions.append(action)

        def _consume_match(match) -> bool:
            action_name = _normalize_action_name(match.group(1))
            args_raw = match.group(2).strip()
            args = []
            if args_raw:
                args = [_normalize_token(a) for a in args_raw.split(',') if a.strip()]

            if action_name == "open-lid":
                if len(args) == 0:
                    args = ["box_lid"]
                elif len(args) != 1:
                    print(f"Warning: {action_name} expects 1 arg, got {len(args)}")
                    return False
                _append_action("open-lid", args)
                return len(actions) >= 25

            if action_name == "move":
                if len(args) not in {1, 2}:
                    print(f"Warning: {action_name} expects 1 or 2 args, got {len(args)}")
                    return False
                if len(args) == 2 and args[0] in self.known_regions and args[1] in self.known_objects:
                    args = [args[1], args[0]]
                _append_action("move", args)
                return len(actions) >= 25

            if action_name == "pick":
                if len(args) != 1:
                    print(f"Warning: {action_name} expects 1 arg, got {len(args)}")
                    return False
                _append_action("pick", args)
            elif action_name == "place":
                if len(args) != 2:
                    print(f"Warning: {action_name} expects 2 args, got {len(args)}")
                    return False
                if args[0] in self.known_regions and args[1] in self.known_objects:
                    args = [args[1], args[0]]
                _append_action("place", args)
            elif action_name == "open_lid":
                if len(args) == 0:
                    args = ["box_lid"]
                if len(args) != 1:
                    print(f"Warning: open_lid expects 1 arg, got {len(args)}")
                    return False
                _append_action("open-lid", args)

            return len(actions) >= 25

        # Parse line by line first, allowing multiple actions per line.
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r'^\s*(?:\d+[.)]|[-*])\s*', '', line)
            for match in action_pattern.finditer(line):
                if _consume_match(match):
                    print(f"[Parser] Capping plan at 25 actions")
                    return actions

        # Fallback: scan the whole response in case the model emitted actions inline
        # after reasoning text or without newlines.
        if not actions:
            for match in action_pattern.finditer(text):
                if _consume_match(match):
                    print(f"[Parser] Capping plan at 25 actions")
                    return actions

        return actions

    def validate_plan(self, skeleton: List[ActionSkeleton]) -> Tuple[bool, List[str]]:
        """Validate syntax and generic execution discipline for a plan."""
        errors = []
        holding = None

        for i, action in enumerate(skeleton):
            for arg in action.args:
                if arg not in self.known_objects and arg not in self.known_regions:
                    errors.append(f"Step {i+1}: Unknown object/region '{arg}'")

            if action.action_name == 'move':
                if len(action.args) == 0 or len(action.args) > 2:
                    errors.append(f"Step {i+1}: move expects 1 or 2 arguments")
                    continue

                move_obj = action.args[0]
                move_region = action.args[1] if len(action.args) > 1 else None
                if move_obj in self.known_regions and move_region in self.known_objects:
                    move_obj, move_region = move_region, move_obj

                if move_obj not in self.known_objects:
                    errors.append(f"Step {i+1}: Unknown object/region '{move_obj}'")
                if move_region is not None and move_region not in self.known_regions:
                    errors.append(f"Step {i+1}: Unknown object/region '{move_region}'")
                if holding is not None and move_obj != holding:
                    errors.append(f"Step {i+1}: Cannot move {move_obj}, currently holding {holding}")

            elif action.action_name == 'pick':
                if len(action.args) != 1:
                    errors.append(f"Step {i+1}: pick expects 1 argument")
                    continue
                obj = action.args[0]
                if holding is not None:
                    errors.append(f"Step {i+1}: Cannot pick {obj}, already holding {holding}")
                holding = obj

            elif action.action_name == 'place':
                if len(action.args) != 2:
                    errors.append(f"Step {i+1}: place expects 2 arguments")
                    continue
                obj = action.args[0]
                if holding != obj:
                    errors.append(f"Step {i+1}: Cannot place {obj}, currently holding {holding}")
                holding = None

            elif action.action_name == 'open-lid':
                if len(action.args) != 1:
                    errors.append(f"Step {i+1}: open-lid expects 1 argument")
                    continue
                if holding is not None:
                    errors.append(f"Step {i+1}: Cannot open lid while holding {holding}")

        return len(errors) == 0, errors


class MockVLMPlanner(VLMPlanner):
    """
    Mock VLM planner for testing without GPU/model.
    Returns predefined plans based on goal keywords.
    Supports replanning by detecting error context.
    """
    
    def __init__(self):
        super().__init__()
        self.loaded = True  # Pretend we're loaded
        self.replan_count = 0  # Track replans for demo
        
    def load_model(self) -> bool:
        print("MockVLMPlanner: Using mock model (no GPU required)")
        return True

    def generate_plan(self,
                      image: np.ndarray,
                      system_prompt: str,
                      user_prompt: str,
                      **kwargs) -> PlanResult:
        """Return a mock plan based on goal analysis."""
        start_time = time.time()
        self._record_request(system_prompt, user_prompt, use_vision=image is not None, image=image)
        
        # Analyze the prompt to determine appropriate plan
        prompt_lower = user_prompt.lower()
        goal_lower = prompt_lower
        
        skeleton = []
        
        # =====================================================================
        # DETECT IF THIS IS A REPLAN (contains error context)
        # =====================================================================
        is_replan = 'replanning required' in prompt_lower or 'failed action' in prompt_lower
        
        if is_replan:
            self.replan_count += 1
            print(f"[MockVLM] Detected REPLAN request (replan #{self.replan_count})")
            
            # Check what error occurred and generate corrected plan
            if 'no pddl plan found' in prompt_lower or 'lid_closed' in prompt_lower or 'mug4' in prompt_lower:
                # The mug4 failed because lid is closed
                # Generate correct sequence: move mug2 → open lid → pick mug4
                print("[MockVLM] Error was about mug4/lid - generating corrected plan")
                skeleton = [
                    ActionSkeleton('pick', ('mug2',)),
                    ActionSkeleton('place', ('mug2', 'placement_boundary')),
                    ActionSkeleton('open-lid', ('box_lid',)),
                    ActionSkeleton('pick', ('mug4',)),
                    ActionSkeleton('place', ('mug4', 'placement_boundary')),
                    ActionSkeleton('pick', ('mug3',)),
                    ActionSkeleton('place', ('mug3', 'placement_boundary')),
                    ActionSkeleton('pick', ('soup',)),
                    ActionSkeleton('place', ('soup', 'cupboard_boundary')),
                    ActionSkeleton('pick', ('mustard',)),
                    ActionSkeleton('place', ('mustard', 'cupboard_boundary')),
                    ActionSkeleton('pick', ('spam',)),
                    ActionSkeleton('place', ('spam', 'cupboard_boundary')),
                    ActionSkeleton('pick', ('sugar',)),
                    ActionSkeleton('place', ('sugar', 'cupboard_boundary')),
                    ActionSkeleton('pick', ('crackers',)),
                    ActionSkeleton('place', ('crackers', 'cupboard_boundary')),
                ]
            elif 'object_blocked' in prompt_lower or 'blocked by mug2' in prompt_lower:
                # Lid was blocked by mug2
                print("[MockVLM] Error was about blocked lid - move mug2 first")
                skeleton = [
                    ActionSkeleton('pick', ('mug2',)),
                    ActionSkeleton('place', ('mug2', 'placement_boundary')),
                    ActionSkeleton('open-lid', ('box_lid',)),
                ]
            else:
                # Generic replan - give full correct sequence
                print("[MockVLM] Generic replan - giving full correct sequence")
                skeleton = [
                    ActionSkeleton('pick', ('mug2',)),
                    ActionSkeleton('place', ('mug2', 'placement_boundary')),
                    ActionSkeleton('open-lid', ('box_lid',)),
                    ActionSkeleton('pick', ('mug4',)),
                    ActionSkeleton('place', ('mug4', 'placement_boundary')),
                    ActionSkeleton('pick', ('soup',)),
                    ActionSkeleton('place', ('soup', 'cupboard_boundary')),
                ]
        
        # =====================================================================
        # INITIAL PLAN - Deliberately make a mistake to demo replanning
        # =====================================================================
        elif 'groceries' in goal_lower and 'mugs' in goal_lower:
            # Full task - deliberately try mug4 first (will fail)
            print("[MockVLM] Initial plan: Deliberately trying mug4 first (will fail)")
            skeleton = [
                ActionSkeleton('pick', ('mug4',)),
                ActionSkeleton('place', ('mug4', 'placement_boundary')),
            ]
        
        # =====================================================================
        # SPECIFIC GOAL PATTERNS
        # =====================================================================
        
        # Test 1: Open lid first (WRONG - should fail with object_blocked)
        elif 'open' in goal_lower and 'lid' in goal_lower and 'mug' not in goal_lower:
            skeleton = [
                ActionSkeleton('open-lid', ('box_lid',)),
            ]
        
        # Move mug2 to table (correct)
        elif 'mug2' in goal_lower or ('mug' in goal_lower and 'box' in goal_lower and 'table' in goal_lower):
            skeleton = [
                ActionSkeleton('pick', ('mug2',)),
                ActionSkeleton('place', ('mug2', 'placement_boundary')),
            ]
        
        # Mugs on table only
        elif 'mug' in goal_lower and 'table' in goal_lower:
            skeleton = [
                ActionSkeleton('pick', ('mug2',)),
                ActionSkeleton('place', ('mug2', 'placement_boundary')),
                ActionSkeleton('open-lid', ('box_lid',)),
                ActionSkeleton('pick', ('mug4',)),
                ActionSkeleton('place', ('mug4', 'placement_boundary')),
                ActionSkeleton('pick', ('mug3',)),
                ActionSkeleton('place', ('mug3', 'placement_boundary')),
            ]
        
        # Soup to cupboard
        elif 'soup' in goal_lower and 'cupboard' in goal_lower:
            skeleton = [
                ActionSkeleton('pick', ('soup',)),
                ActionSkeleton('place', ('soup', 'cupboard_boundary')),
            ]
        
        # Default - give complete task
        if not skeleton:
            skeleton = [
                ActionSkeleton('pick', ('mug2',)),
                ActionSkeleton('place', ('mug2', 'placement_boundary')),
                ActionSkeleton('open-lid', ('box_lid',)),
                ActionSkeleton('pick', ('mug4',)),
                ActionSkeleton('place', ('mug4', 'placement_boundary')),
                ActionSkeleton('pick', ('soup',)),
                ActionSkeleton('place', ('soup', 'cupboard_boundary')),
            ]
        
        raw_output = "\n".join([f"{i+1}. {a}" for i, a in enumerate(skeleton)])
        
        print(f"[MockVLM] Generated plan ({len(skeleton)} actions)")
        print(f"[MockVLM] Plan: {[str(a) for a in skeleton]}")
        
        return PlanResult(
            success=True,
            skeleton=skeleton,
            raw_output=raw_output,
            inference_time=time.time() - start_time
        )
        
        return PlanResult(
            success=True,
            skeleton=skeleton,
            raw_output=raw_output,
            inference_time=time.time() - start_time
        )


# ============================================================================
# TESTING
# ============================================================================

def test_parser():
    """Test the plan parser."""
    print("Testing plan parser...")
    
    planner = MockVLMPlanner()
    
    test_texts = [
        """1. pick(mug2)
2. place(mug2, placement_boundary)
3. open-lid(box_lid)
4. pick(mug4)
5. place(mug4, placement_boundary)""",
        
        """- pick(mug2)
- place(mug2, placement_boundary)
- open_lid(box_lid)""",
        
        """The robot should:
pick(soup)
then place(soup, cupboard_boundary)"""
    ]
    
    for text in test_texts:
        print(f"\nInput:\n{text}\n")
        actions = planner.parse_plan(text)
        print("Parsed actions:")
        for a in actions:
            print(f"  {a}")


def test_validation():
    """Test plan validation."""
    print("\nTesting plan validation...")
    
    planner = MockVLMPlanner()
    
    # Valid plan
    valid_plan = [
        ActionSkeleton('pick', ('mug2',)),
        ActionSkeleton('place', ('mug2', 'placement_boundary')),
        ActionSkeleton('open-lid', ('box_lid',)),
        ActionSkeleton('pick', ('mug4',)),
        ActionSkeleton('place', ('mug4', 'placement_boundary')),
    ]
    
    is_valid, errors = planner.validate_plan(valid_plan)
    print(f"\nValid plan test: {'PASS' if is_valid else 'FAIL'}")
    if errors:
        print(f"  Errors: {errors}")
    
    # Invalid plan - pick mug4 before opening lid
    invalid_plan = [
        ActionSkeleton('pick', ('mug4',)),
        ActionSkeleton('place', ('mug4', 'placement_boundary')),
    ]
    
    is_valid, errors = planner.validate_plan(invalid_plan)
    print(f"\nInvalid plan test (should fail): {'PASS' if not is_valid else 'FAIL'}")
    if errors:
        print(f"  Errors: {errors}")
    
    # Invalid plan - open lid while mug2 on top
    invalid_plan2 = [
        ActionSkeleton('open-lid', ('box_lid',)),
    ]
    
    is_valid, errors = planner.validate_plan(invalid_plan2)
    print(f"\nInvalid plan test 2 (should fail): {'PASS' if not is_valid else 'FAIL'}")
    if errors:
        print(f"  Errors: {errors}")


def test_mock_planner():
    """Test mock planner generation."""
    print("\nTesting mock planner...")
    
    planner = MockVLMPlanner()
    
    # Dummy image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    result = planner.generate_plan(
        image=image,
        system_prompt="You are a robot planner.",
        user_prompt="Move mug2 to placement_boundary, then open lid and move mug4 to placement_boundary."
    )
    
    print(f"\nSuccess: {result.success}")
    print(f"Inference time: {result.inference_time:.3f}s")
    print(f"Raw output:\n{result.raw_output}")
    print(f"\nSkeleton:")
    for a in result.skeleton:
        print(f"  {a}")


if __name__ == "__main__":
    test_parser()
    test_validation()
    test_mock_planner()

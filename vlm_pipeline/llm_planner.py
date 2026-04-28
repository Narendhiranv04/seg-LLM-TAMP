"""
Text-only Hugging Face planner backend.
"""

import inspect
import time
from datetime import datetime
from typing import Any, Dict

import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_LLM_DEPS = True
except ImportError:
    HAS_LLM_DEPS = False
    AutoModelForCausalLM = None
    AutoTokenizer = None

from vlm_pipeline.vlm_planner import PlanResult, VLMPlanner


class LLMPlanner(VLMPlanner):
    """
    Text-only planner that shares parsing/validation with VLMPlanner.
    """

    def __init__(self,
                 model_name: str = "Qwen/Qwen3-8B",
                 use_4bit: bool = False,
                 device: str = "cuda",
                 model_alias: str = "",
                 model_type: str = "llm"):
        super().__init__(model_name=model_name, use_4bit=use_4bit, device=device)
        self.model_alias = model_alias or model_name
        self.model_type = model_type
        self.tokenizer = None
        self.last_request_summary: Dict[str, Any] = {}

    def load_model(self) -> bool:
        if not HAS_LLM_DEPS:
            print("ERROR: Text LLM dependencies not available.")
            print("Install with: pip install transformers torch accelerate")
            return False

        print(f"Loading LLM: {self.model_name}")
        print(f"Using 4-bit quantization: {self.use_4bit}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "torch_dtype": dtype,
            }
            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"

            if self.use_4bit:
                from transformers import BitsAndBytesConfig

                print("Configuring 4-bit quantization...")
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            self.loaded = True
            print("LLM loaded successfully!")
            return True
        except Exception as exc:
            print(f"Error loading LLM: {exc}")
            import traceback
            traceback.print_exc()
            return False

    def _record_request(self, system_prompt: str, user_prompt: str):
        self.last_request_summary = {
            "timestamp": datetime.now().isoformat(),
            "model_alias": self.model_alias,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "use_vision": False,
            "image_present": False,
            "image_shape": None,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "system_prompt_length": len(system_prompt),
            "user_prompt_length": len(user_prompt),
        }

    def _build_prompt_text(self, system_prompt: str, user_prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            try:
                signature = inspect.signature(self.tokenizer.apply_chat_template)
                if "enable_thinking" in signature.parameters:
                    template_kwargs["enable_thinking"] = False
            except (TypeError, ValueError):
                pass
            try:
                return self.tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                template_kwargs.pop("enable_thinking", None)
                return self.tokenizer.apply_chat_template(messages, **template_kwargs)
        return f"{system_prompt}\n\n{user_prompt}\n"

    def _decode_generation(self, prompt_text: str, max_new_tokens: int, temperature: float) -> str:
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
        )

        target_device = self.model.device if hasattr(self.model, "device") else self.device
        inputs = {name: tensor.to(target_device) for name, tensor in inputs.items()}

        do_sample = temperature >= 0.3
        generate_kwargs = {
            "max_new_tokens": min(max_new_tokens, 512),
            "temperature": temperature,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if not do_sample:
            generate_kwargs.pop("temperature", None)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                **generate_kwargs,
            )

        input_len = inputs["input_ids"].shape[1]
        trimmed = generated_ids[:, input_len:]
        return self.tokenizer.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _repair_user_prompt(self, user_prompt: str) -> str:
        return (
            f"{user_prompt}\n\n"
            "FORMAT REPAIR:\n"
            "Rewrite your answer as ONLY a numbered list of actions.\n"
            "Allowed actions: pick(object), place(object, region), open-lid(lid).\n"
            "Do not include commentary, reasoning, markdown, or <think> blocks."
        )

    def generate_plan(self,
                      image: np.ndarray,
                      system_prompt: str,
                      user_prompt: str,
                      max_new_tokens: int = 1024,
                      temperature: float = 0.0) -> PlanResult:
        return self.generate_plan_text_only(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def generate_plan_text_only(self,
                                system_prompt: str,
                                user_prompt: str,
                                max_new_tokens: int = 1024,
                                temperature: float = 0.0) -> PlanResult:
        if not self.loaded:
            return PlanResult(
                success=False,
                skeleton=[],
                raw_output="",
                inference_time=0,
                error_message="Model not loaded.",
            )

        start_time = time.time()
        self._record_request(system_prompt, user_prompt)

        try:
            prompt_text = self._build_prompt_text(system_prompt, user_prompt)
            output_text = self._decode_generation(prompt_text, max_new_tokens=max_new_tokens, temperature=temperature)

            skeleton = self.parse_plan(output_text)
            repair_attempted = False
            repair_succeeded = False
            if not skeleton:
                repair_attempted = True
                repair_prompt = self._build_prompt_text(system_prompt, self._repair_user_prompt(user_prompt))
                repaired_output = self._decode_generation(repair_prompt, max_new_tokens=256, temperature=0.0)
                repaired_skeleton = self.parse_plan(repaired_output)
                if repaired_skeleton:
                    output_text = repaired_output
                    skeleton = repaired_skeleton
                    repair_succeeded = True
                else:
                    output_text = f"{output_text}\n\n[FORMAT_REPAIR_ATTEMPT]\n{repaired_output}"

            inference_time = time.time() - start_time
            self.last_request_summary["format_repair_attempted"] = repair_attempted
            self.last_request_summary["format_repair_succeeded"] = repair_succeeded
            return PlanResult(
                success=len(skeleton) > 0,
                skeleton=skeleton,
                raw_output=output_text,
                inference_time=inference_time,
            )
        except Exception as exc:
            return PlanResult(
                success=False,
                skeleton=[],
                raw_output="",
                inference_time=time.time() - start_time,
                error_message=str(exc),
            )

    def get_debug_info(self) -> Dict[str, Any]:
        return {
            "model_alias": self.model_alias,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "loaded": self.loaded,
            "last_request": self.last_request_summary,
        }

"""Text-only Hugging Face planner with strict executable output parsing."""

from __future__ import annotations

import inspect
import time
from datetime import datetime
from typing import Any, Dict, Optional

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_LLM_DEPS = True
except ImportError:  # pragma: no cover
    HAS_LLM_DEPS = False
    AutoModelForCausalLM = None
    AutoTokenizer = None
    torch = None

from llm_pipeline.strict_parser import StrictActionParser, StrictParseError
from llm_pipeline.pipeline_types import FailureEvent, FailureSource, FailureStage, PlanResult


class TextLLMPlanner:
    """Text-only planner that emits directly executable action lines."""

    def __init__(
        self,
        model_name: str,
        model_alias: str,
        use_4bit: bool = False,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.model_alias = model_alias or model_name
        self.use_4bit = use_4bit
        self.device = device
        self.tokenizer = None
        self.model = None
        self.loaded = False
        self.parser = StrictActionParser()
        self.last_request_summary: Dict[str, Any] = {}

    def load_model(self) -> bool:
        if not HAS_LLM_DEPS:
            print("ERROR: Text LLM dependencies are not available.")
            return False

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

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            self.loaded = True
            return True
        except Exception as exc:  # pragma: no cover - depends on model install/runtime
            print(f"Error loading LLM '{self.model_name}': {exc}")
            return False

    def _record_request(self, system_prompt: str, user_prompt: str, icl_mode: str, held_object: Optional[str]) -> None:
        self.last_request_summary = {
            "timestamp": datetime.now().isoformat(),
            "model_alias": self.model_alias,
            "model_name": self.model_name,
            "model_type": "llm",
            "text_only": True,
            "icl_mode": icl_mode,
            "held_object": held_object,
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
        inputs = self.tokenizer(prompt_text, return_tensors="pt", padding=True)
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
            generated_ids = self.model.generate(**inputs, **generate_kwargs)

        input_len = inputs["input_ids"].shape[1]
        trimmed = generated_ids[:, input_len:]
        return self.tokenizer.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _build_parse_failure(self, exc: StrictParseError, raw_output: str) -> FailureEvent:
        return FailureEvent(
            failure_id=exc.failure_id,
            stage=FailureStage.BEFORE_EXECUTION,
            source=FailureSource.VALIDATION,
            action=None,
            evidence={
                "line_number": exc.line_number,
                "raw_output": raw_output,
            },
            should_replan=False,
            message=str(exc),
        )

    def generate_plan(
        self,
        system_prompt: str,
        user_prompt: str,
        icl_mode: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        held_object: Optional[str] = None,
    ) -> PlanResult:
        if not self.loaded:
            return PlanResult(
                success=False,
                actions=[],
                raw_output="",
                inference_time=0.0,
                error_message="Model not loaded.",
            )

        started_at = time.time()
        self._record_request(system_prompt, user_prompt, icl_mode, held_object)
        try:
            prompt_text = self._build_prompt_text(system_prompt, user_prompt)
            raw_output = self._decode_generation(prompt_text, max_new_tokens=max_new_tokens, temperature=temperature)
            actions = self.parser.parse(raw_output, held_object=held_object)
            return PlanResult(
                success=True,
                actions=actions,
                raw_output=raw_output,
                inference_time=time.time() - started_at,
            )
        except StrictParseError as exc:
            raw_output = raw_output if 'raw_output' in locals() else ''
            return PlanResult(
                success=False,
                actions=[],
                raw_output=raw_output,
                inference_time=time.time() - started_at,
                error_message=str(exc),
                failure_event=self._build_parse_failure(exc, raw_output),
            )
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            return PlanResult(
                success=False,
                actions=[],
                raw_output="",
                inference_time=time.time() - started_at,
                error_message=str(exc),
            )

    def get_debug_info(self) -> Dict[str, Any]:
        return {
            "model_alias": self.model_alias,
            "model_name": self.model_name,
            "model_type": "llm",
            "loaded": self.loaded,
            "last_request": self.last_request_summary,
        }


class MockTextLLMPlanner(TextLLMPlanner):
    """Scriptable test double for text-only planning."""

    def __init__(self, scripted_output: str = ""):
        super().__init__(model_name="mock-llm", model_alias="mock-llm")
        self.scripted_output = scripted_output
        self.loaded = True

    def load_model(self) -> bool:
        self.loaded = True
        return True

    def generate_plan(
        self,
        system_prompt: str,
        user_prompt: str,
        icl_mode: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        held_object: Optional[str] = None,
    ) -> PlanResult:
        self._record_request(system_prompt, user_prompt, icl_mode, held_object)
        started_at = time.time()
        try:
            actions = self.parser.parse(self.scripted_output, held_object=held_object)
            return PlanResult(
                success=True,
                actions=actions,
                raw_output=self.scripted_output,
                inference_time=time.time() - started_at,
            )
        except StrictParseError as exc:
            return PlanResult(
                success=False,
                actions=[],
                raw_output=self.scripted_output,
                inference_time=time.time() - started_at,
                error_message=str(exc),
                failure_event=self._build_parse_failure(exc, self.scripted_output),
            )

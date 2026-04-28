#!/usr/bin/env python3
"""Remote inference server for the maintained text-only LLM planner."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False
    FastAPI = None
    HTTPException = None
    CORSMiddleware = None
    BaseModel = object
    uvicorn = None

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from llm_pipeline.catalog import list_candidate_llms, resolve_llm_model
from llm_pipeline.planner import TextLLMPlanner
from llm_pipeline.pipeline_types import FailureEvent
PROMPT_MODE_LLM_SEGMENTATION = 'segmentation_text_only'


def format_model_listing() -> str:
    lines = ['Available LLM models:']
    for choice in list_candidate_llms():
        lines.append(f'- {choice.alias}: {choice.path} -- {choice.description}')
    return '\n'.join(lines)


class PlanRequest(BaseModel):
    system_prompt: str
    user_prompt: str
    icl_mode: str
    max_new_tokens: int = 512
    temperature: float = 0.0
    held_object: Optional[str] = None


class ActionResponse(BaseModel):
    action_name: str
    args: List[str]


class FailureEventResponse(BaseModel):
    failure_id: str
    stage: str
    source: str
    action: Optional[str] = None
    evidence: Dict[str, Any]
    should_replan: bool = True
    message: str = ''


class PlanResponse(BaseModel):
    success: bool
    actions: List[ActionResponse]
    raw_output: str
    inference_time: float
    error_message: Optional[str] = None
    failure_event: Optional[FailureEventResponse] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_alias: str
    model_type: str
    prompt_mode: str
    gpu_available: bool


class LLMServer:
    def __init__(self, model: str = 'qwen', use_4bit: bool = False, device: str = 'cuda'):
        self.model_spec = resolve_llm_model(model)
        self.use_4bit = use_4bit
        self.device = device
        self.planner = TextLLMPlanner(
            model_name=self.model_spec.path,
            model_alias=self.model_spec.alias,
            use_4bit=self.use_4bit,
            device=self.device,
        )
        self.loaded = False
        self.last_request_summary: Dict[str, Any] = {}

    def load_model(self) -> bool:
        self.loaded = self.planner.load_model()
        if self.loaded and hasattr(self.planner, 'get_debug_info'):
            debug = self.planner.get_debug_info()
            self.last_request_summary = debug.get('last_request', {})
        return self.loaded

    def generate_plan(self, request: PlanRequest) -> PlanResponse:
        result = self.planner.generate_plan(
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            icl_mode=request.icl_mode,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            held_object=request.held_object,
        )
        if hasattr(self.planner, 'get_debug_info'):
            debug = self.planner.get_debug_info()
            self.last_request_summary = debug.get('last_request', {})
        return PlanResponse(
            success=result.success,
            actions=[
                ActionResponse(action_name=action.action_name, args=list(action.args))
                for action in result.actions
            ],
            raw_output=result.raw_output,
            inference_time=result.inference_time,
            error_message=result.error_message,
            failure_event=self._encode_failure_event(result.failure_event),
        )

    def _encode_failure_event(self, failure_event: Optional[FailureEvent]) -> Optional[FailureEventResponse]:
        if failure_event is None:
            return None
        return FailureEventResponse(**failure_event.to_dict())


def create_app(model: str = 'qwen', use_4bit: bool = False, device: str = 'cuda') -> FastAPI:
    app = FastAPI(
        title='Maintained LLM Planner Server',
        description='Remote inference server for the llm_pipeline text-only planner',
        version='1.0.0',
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    server = LLMServer(model=model, use_4bit=use_4bit, device=device)

    @app.on_event('startup')
    async def startup_event():
        server.load_model()

    @app.get('/health', response_model=HealthResponse)
    async def health_check():
        import torch
        return HealthResponse(
            status='ok' if server.loaded else 'model_not_loaded',
            model_loaded=server.loaded,
            model_name=server.model_spec.path,
            model_alias=server.model_spec.alias,
            model_type='llm',
            prompt_mode=PROMPT_MODE_LLM_SEGMENTATION,
            gpu_available=torch.cuda.is_available(),
        )

    @app.get('/debug/last-request')
    async def debug_last_request():
        return {
            'model_alias': server.model_spec.alias,
            'model_name': server.model_spec.path,
            'model_type': 'llm',
            'prompt_mode': PROMPT_MODE_LLM_SEGMENTATION,
            'last_request': server.last_request_summary,
        }

    @app.post('/plan', response_model=PlanResponse)
    async def generate_plan(request: PlanRequest):
        if not server.loaded:
            raise HTTPException(status_code=503, detail='Model not loaded')
        return server.generate_plan(request)

    @app.get('/')
    async def root():
        return {
            'service': 'Maintained LLM Planner Server',
            'model_alias': server.model_spec.alias,
            'model_name': server.model_spec.path,
            'model_type': 'llm',
            'endpoints': {
                '/health': 'GET - Check server health',
                '/plan': 'POST - Generate direct LLM action plan',
                '/debug/last-request': 'GET - Inspect last request',
            },
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description='Maintained LLM Planner Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--model', default='qwen', help='Registered LLM alias or Hugging Face path')
    parser.add_argument('--device', default='cuda', help='Torch device hint')
    parser.add_argument('--no-4bit', action='store_true', help='Disable 4-bit quantization')
    parser.add_argument('--list-models', action='store_true', help='List registered planner models and exit')
    args = parser.parse_args()

    if args.list_models:
        print(format_model_listing())
        return

    if not HAS_FASTAPI:
        print('ERROR: FastAPI not installed. Install with: pip install fastapi uvicorn')
        sys.exit(1)

    print("=" * 60)
    print("MAINTAINED LLM PLANNER SERVER")
    print("=" * 60)
    print(f"Model: {args.model}")
    print("Model type: llm")
    print(f"4-bit quantization: {not args.no_4bit}")
    print(f"Device hint: {args.device}")
    print(f"Server: http://{args.host}:{args.port}")
    print("=" * 60)

    app = create_app(model=args.model, use_4bit=not args.no_4bit, device=args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

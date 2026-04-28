#!/usr/bin/env python3
"""
Planner inference server.
Runs on the remote GPU box and exposes VLM/LLM planning via HTTP API.
"""

import os
import sys
import base64
import argparse
import numpy as np
from io import BytesIO
from typing import Optional, List, Dict, Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm_pipeline.model_registry import format_model_listing, resolve_model_spec
from vlm_pipeline.planner_factory import create_local_planner
from vlm_pipeline.vlm_planner import ActionSkeleton, PlanResult


class PlanRequest(BaseModel):
    """Request to generate a plan."""
    image_base64: Optional[str] = None
    system_prompt: str
    user_prompt: str
    goal: str
    max_new_tokens: int = 1024
    temperature: float = 0.1
    use_vision: bool = True


class ActionResponse(BaseModel):
    """Single action in response."""
    action_name: str
    args: List[str]


class PlanResponse(BaseModel):
    """Response with generated plan."""
    success: bool
    actions: List[ActionResponse]
    raw_output: str
    inference_time: float
    error_message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: str
    model_alias: str
    model_type: str
    prompt_mode: str
    gpu_available: bool


class VLMServer:
    """Planner inference server."""

    def __init__(self, model: str = "qwen-vl", model_type: str = "", use_4bit: bool = True):
        self.model_spec = resolve_model_spec(model, model_type)
        self.use_4bit = use_4bit
        self.planner = None
        self.loaded = False
        self.last_request_summary: Dict[str, Any] = {}

    def load_model(self) -> bool:
        print(f"[VLMServer] Loading model alias: {self.model_spec.alias}")
        print(f"[VLMServer] Loading model path : {self.model_spec.path}")
        print(f"[VLMServer] Model type        : {self.model_spec.model_type}")
        self.planner = create_local_planner(
            model=self.model_spec.path,
            model_type=self.model_spec.model_type,
            use_4bit=self.use_4bit,
            use_mock=False,
        )
        if hasattr(self.planner, "model_alias"):
            self.planner.model_alias = self.model_spec.alias
        if hasattr(self.planner, "model_name"):
            self.planner.model_name = self.model_spec.path
        self.loaded = self.planner.load_model()
        if self.loaded:
            print("[VLMServer] Model loaded successfully!")
        else:
            print("[VLMServer] Failed to load model!")
        return self.loaded

    def generate_plan(self, request: PlanRequest) -> PlanResponse:
        if not self.loaded:
            return PlanResponse(
                success=False,
                actions=[],
                raw_output="",
                inference_time=0,
                error_message="Model not loaded",
            )

        try:
            if request.use_vision:
                if not request.image_base64:
                    return PlanResponse(
                        success=False,
                        actions=[],
                        raw_output="",
                        inference_time=0,
                        error_message="image_base64 is required when use_vision=True",
                    )

                image_bytes = base64.b64decode(request.image_base64)
                image_array = np.load(BytesIO(image_bytes), allow_pickle=True)
                result = self.planner.generate_plan(
                    image=image_array,
                    system_prompt=request.system_prompt,
                    user_prompt=request.user_prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                )
            else:
                result = self.planner.generate_plan_text_only(
                    system_prompt=request.system_prompt,
                    user_prompt=request.user_prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                )

            if hasattr(self.planner, "get_debug_info"):
                debug_info = self.planner.get_debug_info()
                self.last_request_summary = debug_info.get("last_request", {})

            actions = [
                ActionResponse(action_name=a.action_name, args=list(a.args))
                for a in result.skeleton
            ]
            return PlanResponse(
                success=result.success,
                actions=actions,
                raw_output=result.raw_output,
                inference_time=result.inference_time,
                error_message=result.error_message,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return PlanResponse(
                success=False,
                actions=[],
                raw_output="",
                inference_time=0,
                error_message=str(exc),
            )



def create_app(model: str = "qwen-vl", model_type: str = "", use_4bit: bool = True) -> FastAPI:
    app = FastAPI(
        title="Planner Inference Server",
        description="Remote Hugging Face planner inference for robot task planning",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    server = VLMServer(model=model, model_type=model_type, use_4bit=use_4bit)

    @app.on_event("startup")
    async def startup_event():
        server.load_model()

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        import torch
        return HealthResponse(
            status="ok" if server.loaded else "model_not_loaded",
            model_loaded=server.loaded,
            model_name=server.model_spec.path,
            model_alias=server.model_spec.alias,
            model_type=server.model_spec.model_type,
            prompt_mode=server.model_spec.prompt_mode,
            gpu_available=torch.cuda.is_available(),
        )

    @app.get("/debug/last-request")
    async def debug_last_request():
        return {
            "model_alias": server.model_spec.alias,
            "model_name": server.model_spec.path,
            "model_type": server.model_spec.model_type,
            "prompt_mode": server.model_spec.prompt_mode,
            "last_request": server.last_request_summary,
        }

    @app.post("/plan", response_model=PlanResponse)
    async def generate_plan(request: PlanRequest):
        if not server.loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return server.generate_plan(request)

    @app.get("/")
    async def root():
        return {
            "service": "Planner Inference Server",
            "model_alias": server.model_spec.alias,
            "model_name": server.model_spec.path,
            "model_type": server.model_spec.model_type,
            "endpoints": {
                "/health": "GET - Check server health",
                "/plan": "POST - Generate action plan",
                "/debug/last-request": "GET - Inspect the last planner request",
            },
        }

    return app



def main():
    parser = argparse.ArgumentParser(description="Planner Inference Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model", type=str, default="qwen-vl", help="Registered model alias or Hugging Face model path")
    parser.add_argument("--model-type", type=str, default="", choices=["", "vlm", "llm"], help="Explicit model type when --model is a custom path")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--list-models", action="store_true", help="List registered planner models and exit")
    args = parser.parse_args()

    if args.list_models:
        print(format_model_listing())
        return

    if not HAS_FASTAPI:
        print("ERROR: FastAPI not installed.")
        print("Install with: pip install fastapi uvicorn")
        sys.exit(1)

    print("=" * 60)
    print("PLANNER INFERENCE SERVER")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Model type override: {args.model_type or '(auto)'}")
    print(f"4-bit quantization: {not args.no_4bit}")
    print(f"Server: http://{args.host}:{args.port}")
    print("=" * 60)

    app = create_app(model=args.model, model_type=args.model_type, use_4bit=not args.no_4bit)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

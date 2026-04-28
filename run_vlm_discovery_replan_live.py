#!/usr/bin/env python3
"""
Run replanning pipeline with discovery-triggered failure/replan in live view mode.

Defaults are tuned for:
- live segmentation window enabled
- replan when new object appears after open-lid completes
- explicit failure code NEW_OBJECT_INTRODUCED_IN_SCENE
- model-driven prompt mode selection:
  * VLM -> image + goal
  * LLM -> visible-object text + goal
"""

import argparse
import os


def _configure_qt():
    os.environ.setdefault("COPPELIASIM_HEADLESS", "0")
    os.environ.pop("QT_PLUGIN_PATH", None)
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
    coppelia_root = os.environ.get("COPPELIASIM_ROOT") or os.path.expanduser("~/CoppeliaSim")
    for candidate in [
        os.path.join(coppelia_root, "platforms"),
        os.path.join(coppelia_root, "Qt", "plugins", "platforms"),
    ]:
        if candidate and os.path.isdir(candidate):
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", candidate)
            break


def main():
    _configure_qt()

    from vlm_pipeline.model_registry import (
        PROMPT_MODE_TEXT_VISIBLE,
        PROMPT_MODE_VISION_GOAL,
        format_model_listing,
        resolve_model_spec,
    )
    from vlm_pipeline.vlm_with_replanning import (
        DISCOVERY_FAILURE_CODE_DEFAULT,
        ReplanConfig,
        VLMReplanningPipeline,
    )

    parser = argparse.ArgumentParser(description="Live modular discovery-triggered replanning runner")
    parser.add_argument(
        "--goal",
        type=str,
        default="Move all mugs inside the box and all groceries into the cupboard",
        help="Natural language goal for the planner",
    )
    parser.add_argument("--max-replans", type=int, default=3, help="Maximum replan attempts")
    parser.add_argument("--mock", action="store_true", help="Use mock planner")
    parser.add_argument("--remote", action="store_true", help="Use remote planner server")
    parser.add_argument("--remote-url", type=str, default="", help="Remote planner server URL")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen-vl",
        help="Registered model alias or Hugging Face model path",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="",
        choices=["", "vlm", "llm"],
        help="Explicit model type when --model is a custom path",
    )
    parser.add_argument(
        "--planner-input-mode",
        type=str,
        default="",
        choices=["", PROMPT_MODE_VISION_GOAL, PROMPT_MODE_TEXT_VISIBLE],
        help="Override prompt mode (default: derived from model type)",
    )
    parser.add_argument("--4bit", action="store_true", help="Enable 4-bit quantization for local Hugging Face models")
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Force text-only prompting regardless of model registry defaults",
    )
    parser.add_argument(
        "--discovery-targets",
        type=str,
        default="",
        help="Comma-separated discovery targets (empty = any newly visible object)",
    )
    parser.add_argument("--live-mask-stride", type=int, default=5, help="Refresh live masks every N sim steps")
    parser.add_argument("--headless", action="store_true", help="Run simulation headless")
    parser.add_argument(
        "--discovery-failure-code",
        type=str,
        default=DISCOVERY_FAILURE_CODE_DEFAULT,
        help="Failure code for discovery-triggered replan",
    )
    parser.add_argument("--list-models", action="store_true", help="List registered planner models and exit")
    args = parser.parse_args()

    if args.list_models:
        print(format_model_listing())
        return

    spec = resolve_model_spec(args.model, args.model_type)
    planner_input_mode = args.planner_input_mode or spec.prompt_mode
    if args.text_only:
        planner_input_mode = PROMPT_MODE_TEXT_VISIBLE
    use_vision = planner_input_mode == PROMPT_MODE_VISION_GOAL
    visible_objects_only = planner_input_mode == PROMPT_MODE_TEXT_VISIBLE

    config = ReplanConfig(
        max_replans=args.max_replans,
        use_mock_vlm=args.mock,
        use_remote_vlm=args.remote,
        remote_vlm_url=args.remote_url,
        model=args.model,
        model_type=args.model_type,
        planner_input_mode=planner_input_mode,
        use_4bit=args.__dict__["4bit"],
        use_vision=use_vision,
        visible_objects_only=visible_objects_only,
        live_segmentation_view=True,
        replan_on_discovery=True,
        discovery_targets=[s.strip() for s in args.discovery_targets.split(",") if s.strip()],
        discovery_failure_code=args.discovery_failure_code,
        replan_on_discovery_after_plan_complete=True,
        live_view_update_stride=max(1, args.live_mask_stride),
        headless=args.headless,
    )

    pipeline = VLMReplanningPipeline(config)
    try:
        if not pipeline.initialize():
            print("Failed to initialize pipeline")
            return

        result = pipeline.run(args.goal)
        print(result)

        print("\nPress Ctrl+C to exit...")
        while True:
            pipeline.env.pr.step()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    main()

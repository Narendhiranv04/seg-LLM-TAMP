#!/usr/bin/env python3
"""Run one kitchen model trial and emit a structured JSON record."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.canonical_variants import get_variant_spec
from evaluation.metrics import collect_failure_occurrences, score_variant_completion
from vlm_pipeline.model_registry import resolve_model_spec


def _configure_qt() -> None:
    os.environ.setdefault('COPPELIASIM_HEADLESS', '0')
    os.environ.pop('QT_PLUGIN_PATH', None)
    os.environ.setdefault('QT_LOGGING_RULES', '*.debug=false;qt.qpa.*=false')
    coppelia_root = os.environ.get('COPPELIASIM_ROOT') or os.path.expanduser('~/CoppeliaSim')
    for candidate in [
        os.path.join(coppelia_root, 'platforms'),
        os.path.join(coppelia_root, 'Qt', 'plugins', 'platforms'),
    ]:
        if candidate and os.path.isdir(candidate):
            os.environ.setdefault('QT_QPA_PLATFORM_PLUGIN_PATH', candidate)
            break


def _fetch_json(url: str) -> Dict[str, Any]:
    if requests is None:
        raise RuntimeError('requests is required for remote benchmark inspection.')
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def _repo_setup(headless: bool, scene_path: str) -> None:
    os.environ['KITCHEN_SCENE_FILE'] = scene_path
    os.environ['HEADLESS'] = 'True' if headless else 'False'
    os.environ['COPPELIASIM_HEADLESS'] = '1' if headless else '0'
    pddlstream_path = str(ROOT_DIR / 'pddlstream')
    existing = os.environ.get('PYTHONPATH', '').strip()
    os.environ['PYTHONPATH'] = f"{pddlstream_path}:{existing}" if existing else pddlstream_path


def _verify_remote_server(remote_url: str, expected_model) -> Dict[str, Any]:
    issues = []
    health = _fetch_json(f'{remote_url}/health')
    if not health.get('model_loaded'):
        issues.append('remote_server_model_not_loaded')
    if health.get('model_type') and health['model_type'] != expected_model.model_type:
        issues.append(f"model_type_mismatch:{health.get('model_type')}")
    if health.get('model_name') and health['model_name'] != expected_model.path:
        issues.append(f"model_path_mismatch:{health.get('model_name')}")
    if health.get('model_alias') and health['model_alias'] != expected_model.alias:
        issues.append(f"model_alias_mismatch:{health.get('model_alias')}")
    return {'ok': not issues, 'issues': issues, 'health': health}


def _collect_debug_snapshot(pipeline, remote: bool, remote_url: str) -> Dict[str, Any]:
    if remote:
        return {
            'health': _fetch_json(f'{remote_url}/health'),
            'debug': _fetch_json(f'{remote_url}/debug/last-request'),
        }
    if hasattr(pipeline.planner, 'get_debug_info'):
        return {'health': {}, 'debug': pipeline.planner.get_debug_info()}
    return {'health': {}, 'debug': {}}


def _verify_prompt_modality(model_spec, debug_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    debug_payload = debug_snapshot.get('debug', {}) or {}
    last_request = debug_payload.get('last_request', {}) or {}
    user_prompt = (last_request.get('user_prompt') or '')
    user_prompt_lower = user_prompt.lower()
    visible_text_present = 'visible-objects:' in user_prompt_lower or 'current observation (visible objects)' in user_prompt_lower
    observed_use_vision = last_request.get('use_vision')
    observed_image_present = last_request.get('image_present')
    expected_use_vision = bool(model_spec.use_vision)
    issues = []
    if observed_use_vision is not None and bool(observed_use_vision) != expected_use_vision:
        issues.append('use_vision_mismatch')
    if observed_image_present is not None and bool(observed_image_present) != expected_use_vision:
        issues.append('image_present_mismatch')
    if expected_use_vision and visible_text_present:
        issues.append('visible_object_text_leaked_into_vlm_prompt')
    if (not expected_use_vision) and (not visible_text_present):
        issues.append('visible_object_text_missing_from_llm_prompt')
    return {
        'expected_use_vision': expected_use_vision,
        'observed_use_vision': observed_use_vision,
        'observed_image_present': observed_image_present,
        'visible_objects_text_present': visible_text_present,
        'issues': issues,
        'ok': not issues,
    }


def run_trial(variant_id: str,
              model: str,
              model_type: str = '',
              trial_index: int = 1,
              remote: bool = False,
              remote_url: str = '',
              max_replans: int = 3,
              headless: bool = False,
              preflight_only: bool = False,
              allow_unsupported_model: bool = False,
              output_path: Optional[Path] = None,
              goal_override: Optional[str] = None,
              live_segmentation_view: bool = True,
              replan_on_discovery: bool = True,
              live_mask_stride: int = 5) -> Dict[str, Any]:
    variant_spec = get_variant_spec(variant_id)
    if not variant_spec.model_eval_supported:
        raise RuntimeError(
            f'Variant {variant_spec.variant_id} is not supported for model benchmarking: '
            f'{variant_spec.model_eval_reason or "unsupported"}'
        )

    model_spec = resolve_model_spec(model, model_type)
    if model_spec.alias == 'spatial-ladder' and not allow_unsupported_model:
        raise RuntimeError('spatial-ladder is excluded from evaluation until smoke-tested with this backend.')

    _configure_qt()
    _repo_setup(headless=headless, scene_path=variant_spec.scene_path)
    goal_text = goal_override or variant_spec.goal_text

    from vlm_pipeline.vlm_with_replanning import (
        DISCOVERY_FAILURE_CODE_DEFAULT,
        ReplanConfig,
        VLMReplanningPipeline,
    )

    planner_input_mode = model_spec.prompt_mode
    use_vision = model_spec.use_vision
    visible_objects_only = not use_vision

    config = ReplanConfig(
        max_replans=max_replans,
        use_mock_vlm=False,
        use_remote_vlm=remote,
        remote_vlm_url=remote_url,
        model=model,
        model_type=model_type,
        planner_input_mode=planner_input_mode,
        use_4bit=False,
        use_vision=use_vision,
        visible_objects_only=visible_objects_only,
        live_segmentation_view=live_segmentation_view,
        replan_on_discovery=replan_on_discovery,
        discovery_targets=[],
        discovery_failure_code=DISCOVERY_FAILURE_CODE_DEFAULT,
        replan_on_discovery_after_plan_complete=True,
        live_view_update_stride=max(1, int(live_mask_stride)),
        headless=headless,
        save_logs=True,
    )

    pipeline = VLMReplanningPipeline(config)
    output_path = output_path.resolve() if output_path else None
    try:
        init_ok = pipeline.initialize()
        if not init_ok:
            raise RuntimeError('pipeline_initialize_failed')

        remote_validation = None
        if remote:
            remote_validation = _verify_remote_server(remote_url, model_spec)

        if preflight_only:
            plan_result = pipeline.plan_initial(goal_text)
            debug_snapshot = _collect_debug_snapshot(pipeline, remote=remote, remote_url=remote_url)
            prompt_verification = _verify_prompt_modality(model_spec, debug_snapshot)
            record = {
                'variant_id': variant_spec.variant_id,
                'task_family': variant_spec.task_family,
                'scene_path': variant_spec.scene_path,
                'trial_index': int(trial_index),
                'preflight_only': True,
                'model_alias': model_spec.alias,
                'model_path': model_spec.path,
                'model_type': model_spec.model_type,
                'prompt_mode': planner_input_mode,
                'use_vision': use_vision,
                'visible_objects_only': visible_objects_only,
                'goal_text': goal_text,
                'server_validation': remote_validation,
                'prompt_verification': prompt_verification,
                'dry_run_plan_success': bool(plan_result.success),
                'dry_run_action_count': len(plan_result.skeleton),
                'dry_run_error_message': plan_result.error_message,
                'dry_run_raw_output_preview': (plan_result.raw_output or '')[:500],
                'dry_run_inference_time_s': plan_result.inference_time,
                'preflight_success': bool((not remote or (remote_validation or {}).get('ok')) and prompt_verification['ok'] and plan_result.success and len(plan_result.skeleton) > 0),
                'debug_snapshot': debug_snapshot,
            }
        else:
            summary = pipeline.run(goal_text)
            debug_snapshot = _collect_debug_snapshot(pipeline, remote=remote, remote_url=remote_url)
            prompt_verification = _verify_prompt_modality(model_spec, debug_snapshot)
            completion = score_variant_completion(variant_spec.variant_id, summary.get('completed_actions', []))
            failure_occurrences = collect_failure_occurrences(summary.get('cycles', []), summary.get('failure_reason'))
            episode_success = bool(summary.get('success')) and (
                completion['completed_gt_subtasks'] >= completion['gt_total_subtasks']
            )
            record = {
                'variant_id': variant_spec.variant_id,
                'task_family': variant_spec.task_family,
                'scene_path': variant_spec.scene_path,
                'trial_index': int(trial_index),
                'preflight_only': False,
                'action_sequence_length': variant_spec.action_sequence_length,
                'gt_total_subtasks': completion['gt_total_subtasks'],
                'completed_gt_subtasks': completion['completed_gt_subtasks'],
                'subtask_completion_rate': completion['subtask_completion_rate'],
                'bucket_breakdown': completion['bucket_breakdown'],
                'observed_subtasks': completion['observed_subtasks'],
                'extra_observed_subtasks': completion['extra_observed_subtasks'],
                'model_alias': summary.get('model_alias', model_spec.alias),
                'model_path': summary.get('model_path', model_spec.path),
                'model_type': summary.get('model_type', model_spec.model_type),
                'prompt_mode': summary.get('prompt_mode', planner_input_mode),
                'use_vision': bool(summary.get('use_vision', use_vision)),
                'visible_objects_only': bool(summary.get('visible_objects_only', visible_objects_only)),
                'goal_text': goal_text,
                'episode_success': episode_success,
                'raw_episode_success': bool(summary.get('success')),
                'total_cycles': int(summary.get('total_cycles', 0)),
                'total_replans': int(summary.get('total_replans', 0)),
                'completed_actions': list(summary.get('completed_actions', [])),
                'failure_reason': summary.get('failure_reason'),
                'failure_occurrences': failure_occurrences,
                'episode_time_s': summary.get('episode_time_s'),
                'server_validation': remote_validation,
                'prompt_verification': prompt_verification,
                'debug_snapshot': debug_snapshot,
                'raw_summary': summary,
            }

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(record, indent=2))
        return record
    finally:
        try:
            pipeline.shutdown()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description='Run one model benchmark trial')
    parser.add_argument('--variant', required=True, help='Variant id (currently kitchen variants only)')
    parser.add_argument('--model', required=True, help='Registered model alias or HF path')
    parser.add_argument('--model-type', default='', choices=['', 'vlm', 'llm'], help='Explicit model type for custom HF paths')
    parser.add_argument('--trial-index', type=int, default=1, help='1-based trial index')
    parser.add_argument('--remote', action='store_true', help='Use the remote planner server')
    parser.add_argument('--remote-url', default=os.environ.get('VLM_SERVER_URL', 'http://localhost:8080'), help='Remote planner server URL')
    parser.add_argument('--max-replans', type=int, default=3, help='Maximum replans during execution')
    parser.add_argument('--goal', default='', help='Optional goal override. Default uses the canonical variant goal.')
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument('--gui', action='store_true', help='Run with simulator GUI (default)')
    display_group.add_argument('--headless', action='store_true', help='Run without simulator GUI')
    parser.add_argument('--no-live-masks', action='store_true', help='Disable the live segmentation viewer window')
    parser.add_argument('--no-discovery-replan', action='store_true', help='Disable discovery-triggered replanning')
    parser.add_argument('--live-mask-stride', type=int, default=5, help='Refresh live masks every N simulator steps')
    parser.add_argument('--preflight-only', action='store_true', help='Only validate model loading and prompt modality')
    parser.add_argument('--allow-unsupported-model', action='store_true', help='Allow benchmarking a model that is marked smoke-test-only')
    parser.add_argument('--output', default='', help='Output JSON path')
    args = parser.parse_args()

    record = run_trial(
        variant_id=args.variant,
        model=args.model,
        model_type=args.model_type,
        trial_index=args.trial_index,
        remote=args.remote,
        remote_url=args.remote_url,
        max_replans=args.max_replans,
        headless=bool(args.headless),
        preflight_only=args.preflight_only,
        allow_unsupported_model=args.allow_unsupported_model,
        output_path=Path(args.output) if args.output else None,
        goal_override=args.goal or None,
        live_segmentation_view=not args.no_live_masks,
        replan_on_discovery=not args.no_discovery_replan,
        live_mask_stride=args.live_mask_stride,
    )
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()

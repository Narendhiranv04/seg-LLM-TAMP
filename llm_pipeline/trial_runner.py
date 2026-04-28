"""Run one LLM-only kitchen trial and emit a structured JSON record."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.canonical_variants import get_variant_spec
from llm_pipeline.metrics import collect_failure_occurrences, score_variant_completion
from llm_pipeline.pipeline import LLMPipelineConfig, LLMOnlyReplanningPipeline


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


def _repo_setup(headless: bool, scene_path: str) -> None:
    os.environ['KITCHEN_SCENE_FILE'] = scene_path
    os.environ['HEADLESS'] = 'True' if headless else 'False'
    os.environ['COPPELIASIM_HEADLESS'] = '1' if headless else '0'
    pddlstream_path = str(ROOT_DIR / 'pddlstream')
    existing = os.environ.get('PYTHONPATH', '').strip()
    os.environ['PYTHONPATH'] = f"{pddlstream_path}:{existing}" if existing else pddlstream_path


def _text_only_contract_issues(preflight_record: Dict[str, Any]) -> list[str]:
    issues = list(preflight_record.get('prompt_contract_issues', []))
    prompt_trace = preflight_record.get('prompt_trace', {}) or {}
    bundle = prompt_trace.get('bundle', {}) or {}
    debug_snapshot = preflight_record.get('debug_snapshot', {}) or {}
    if any('image' in key for key in bundle):
        issues.append('image_key_in_prompt_bundle')
    if any('image' in key for key in debug_snapshot):
        issues.append('image_key_in_debug_snapshot')
    for field_name in ('system_prompt', 'user_prompt'):
        if field_name not in prompt_trace:
            issues.append(f'missing_{field_name}')
    return sorted(set(issues))


def run_trial(
    variant_id: str,
    model_alias: str,
    icl_mode: str,
    trial_index: int = 1,
    max_replans: int = 10,
    headless: bool = False,
    remote: bool = False,
    remote_url: str = '',
    replan_mode: str = 'on',
    preflight_only: bool = False,
    output_path: Optional[Path] = None,
    goal_override: Optional[str] = None,
) -> Dict[str, Any]:
    variant_spec = get_variant_spec(variant_id)
    if variant_spec.task_family != 'kitchen' or not variant_spec.model_eval_supported:
        raise RuntimeError(
            f'Variant {variant_spec.variant_id} is not supported for the LLM-only kitchen runner: '
            f"{variant_spec.model_eval_reason or 'unsupported'}"
        )

    _configure_qt()
    _repo_setup(headless=headless, scene_path=variant_spec.scene_path)
    goal_text = goal_override or variant_spec.goal_text

    replanning_enabled = replan_mode != 'off'
    config = LLMPipelineConfig(
        model_alias=model_alias,
        icl_mode=icl_mode,
        max_replans=max_replans,
        enable_replanning=replanning_enabled,
        headless=headless,
        use_remote_planner=remote,
        remote_planner_url=remote_url,
        text_only=True,
        segmentation_first=True,
        pre_action_checks_enabled=replanning_enabled,
        post_action_checks_enabled=replanning_enabled,
    )
    pipeline = LLMOnlyReplanningPipeline(config=config)

    output_path = output_path.resolve() if output_path else None
    try:
        if not pipeline.initialize():
            raise RuntimeError('pipeline_initialize_failed')

        preflight = pipeline.preflight(goal_text)
        preflight_issues = _text_only_contract_issues(preflight)
        preflight['prompt_contract_issues'] = preflight_issues
        preflight['prompt_contract_ok'] = not preflight_issues
        preflight['preflight_success'] = (
            bool(preflight.get('loaded'))
            and bool(preflight.get('dry_run_plan_success'))
            and not preflight_issues
            and preflight.get('image_present') is False
        )

        if preflight_only:
            record = {
                'variant_id': variant_spec.variant_id,
                'task_family': variant_spec.task_family,
                'scene_path': variant_spec.scene_path,
                'trial_index': int(trial_index),
                'preflight_only': True,
                'model_alias': model_alias,
                'model_type': 'llm',
                'icl_mode': icl_mode,
                'goal_text': goal_text,
                'text_only': True,
                'segmentation_first': True,
                'replan_mode': replan_mode,
                'use_remote_planner': bool(remote),
                'remote_planner_url': remote_url or None,
                'pre_action_checks_enabled': replanning_enabled,
                'post_action_checks_enabled': replanning_enabled,
                'preflight_success': bool(preflight['preflight_success']),
                'preflight': preflight,
            }
        else:
            summary = pipeline.run(goal_text)
            completion = score_variant_completion(variant_spec.variant_id, summary.get('completed_actions', []))
            failure_occurrences = collect_failure_occurrences(summary.get('cycles', []), summary.get('failure_reason'))
            execution_skipped = bool(summary.get('execution_skipped', False))
            episode_success = None if execution_skipped else bool(summary.get('success')) and (
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
                'model_alias': summary.get('model_alias', model_alias),
                'model_type': 'llm',
                'icl_mode': icl_mode,
                'prompt_mode': summary.get('prompt_mode', icl_mode),
                'goal_text': goal_text,
                'text_only': True,
                'segmentation_first': True,
                'replan_mode': summary.get('replan_mode', replan_mode),
                'planning_success': bool(summary.get('success')),
                'execution_skipped': execution_skipped,
                'use_remote_planner': bool(remote),
                'remote_planner_url': remote_url or None,
                'pre_action_checks_enabled': bool(summary.get('pre_action_checks_enabled', replanning_enabled)),
                'post_action_checks_enabled': bool(summary.get('post_action_checks_enabled', replanning_enabled)),
                'episode_success': episode_success,
                'raw_episode_success': None if execution_skipped else bool(summary.get('success')),
                'total_cycles': int(summary.get('total_cycles', 0)),
                'total_replans': int(summary.get('total_replans', 0)),
                'planned_actions': list(summary.get('planned_actions', [])),
                'completed_actions': list(summary.get('completed_actions', [])),
                'remaining_actions': list(summary.get('remaining_actions', [])),
                'failure_reason': summary.get('failure_reason'),
                'failure_occurrences': failure_occurrences,
                'episode_time_s': summary.get('episode_time_s'),
                'preflight': preflight,
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
    parser = argparse.ArgumentParser(description='Run one LLM-only kitchen benchmark trial')
    parser.add_argument('--variant', required=True, help='Variant id (K1/K2/K3)')
    parser.add_argument('--model', required=True, help='Registered LLM alias or custom HF path')
    parser.add_argument('--icl-mode', required=True, choices=['zero_shot', 'few_shot_shared_1'], help='Prompt mode to evaluate')
    parser.add_argument('--trial-index', type=int, default=1, help='1-based trial index')
    parser.add_argument('--max-replans', type=int, default=3, help='Maximum replans during execution')
    parser.add_argument('--goal', default='', help='Optional goal override')
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument('--gui', action='store_true', help='Run with simulator GUI (default)')
    display_group.add_argument('--headless', action='store_true', help='Run without simulator GUI')
    parser.add_argument('--remote', action='store_true', help='Use the maintained remote LLM planner server')
    parser.add_argument('--remote-url', default=os.environ.get('LLM_SERVER_URL', os.environ.get('VLM_SERVER_URL', 'http://localhost:8000')), help='Remote planner server URL')
    parser.add_argument('--replan-mode', choices=['on', 'off'], default='on', help='Use full execution+replanning (on) or first-plan-only mode with no failure checks (off)')
    parser.add_argument('--preflight-only', action='store_true', help='Only run text-only load/prompt validation')
    parser.add_argument('--output', default='', help='Optional JSON output path')
    args = parser.parse_args()

    record = run_trial(
        variant_id=args.variant,
        model_alias=args.model,
        icl_mode=args.icl_mode,
        trial_index=args.trial_index,
        max_replans=args.max_replans,
        headless=args.headless,
        remote=args.remote,
        remote_url=args.remote_url,
        replan_mode=args.replan_mode,
        preflight_only=args.preflight_only,
        output_path=Path(args.output) if args.output else None,
        goal_override=args.goal or None,
    )
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()

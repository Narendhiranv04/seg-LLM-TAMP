"""Run multi-model LLM-only benchmark batches."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.canonical_variants import DEFAULT_MODEL_VARIANTS, get_variant_spec
from llm_pipeline.catalog import FINAL_MODEL_TRIO, SHARED_ICL_MODES, list_candidate_llms
from llm_pipeline.metrics import aggregate_model_records


def _parse_csv(raw: str, default: List[str]) -> List[str]:
    if not raw.strip():
        return list(default)
    return [token.strip() for token in raw.split(',') if token.strip()]


def _candidate_model_aliases() -> List[str]:
    return [choice.alias for choice in list_candidate_llms()]


def _trial_command(
    variant_id: str,
    model_alias: str,
    icl_mode: str,
    output_path: Path,
    trial_index: int,
    max_replans: int,
    headless: bool,
    remote: bool,
    remote_url: str,
    replan_mode: str,
    preflight_only: bool,
    goal: str,
) -> List[str]:
    command = [
        sys.executable,
        '-m',
        'llm_pipeline.trial_runner',
        '--variant', variant_id,
        '--model', model_alias,
        '--icl-mode', icl_mode,
        '--trial-index', str(trial_index),
        '--max-replans', str(max_replans),
        '--output', str(output_path),
    ]
    if headless:
        command.append('--headless')
    if remote:
        command.extend(['--remote', '--remote-url', remote_url])
    if replan_mode != 'on':
        command.extend(['--replan-mode', replan_mode])
    if preflight_only:
        command.append('--preflight-only')
    if goal:
        command.extend(['--goal', goal])
    return command


def _run_and_load(command: List[str], output_path: Path) -> Dict:
    result = subprocess.run(command, cwd=str(ROOT_DIR), text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Trial command failed: {' '.join(command)}")
    return json.loads(output_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the LLM-only benchmark batch')
    parser.add_argument('--models', default='', help='Comma-separated LLM aliases. Default: all registered LLMs')
    parser.add_argument('--icl-modes', default=','.join(SHARED_ICL_MODES), help='Comma-separated prompt modes')
    parser.add_argument('--variants', default=','.join(DEFAULT_MODEL_VARIANTS), help='Comma-separated kitchen variants')
    parser.add_argument('--trials', type=int, default=1, help='Trials per execution variant')
    parser.add_argument('--max-replans', type=int, default=3, help='Maximum replans during execution')
    parser.add_argument('--goal', default='', help='Optional goal override for all runs')
    parser.add_argument('--headless', action='store_true', help='Run without simulator GUI')
    parser.add_argument('--remote', action='store_true', help='Use the maintained remote LLM planner server')
    parser.add_argument('--remote-url', default=os.environ.get('LLM_SERVER_URL', os.environ.get('VLM_SERVER_URL', 'http://localhost:8000')), help='Remote planner server URL')
    parser.add_argument('--replan-mode', choices=['on', 'off'], default='on', help='Use full execution+replanning (on) or first-plan-only mode with no failure checks (off)')
    parser.add_argument('--skip-preflight', action='store_true', help='Skip model/mode preflight checks')
    parser.add_argument('--execute-all-models', action='store_true', help='Run full execution for every requested model, not just the maintained trio')
    parser.add_argument('--output-root', default='llm_pipeline/results/benchmarks', help='Root directory for benchmark outputs')
    args = parser.parse_args()

    requested_models = _parse_csv(args.models, _candidate_model_aliases())
    icl_modes = _parse_csv(args.icl_modes, list(SHARED_ICL_MODES))
    variants = [token.upper() for token in _parse_csv(args.variants, list(DEFAULT_MODEL_VARIANTS))]
    for variant_id in variants:
        spec = get_variant_spec(variant_id)
        if spec.task_family != 'kitchen' or not spec.model_eval_supported:
            raise RuntimeError(f'Unsupported variant for LLM-only benchmark: {variant_id}')

    execution_models = list(requested_models) if args.execute_all_models else [
        alias for alias in requested_models if alias in FINAL_MODEL_TRIO
    ]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = (ROOT_DIR / args.output_root / timestamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    preflight_records = []
    execution_records = []

    if not args.skip_preflight:
        preflight_dir = run_dir / 'preflight'
        preflight_dir.mkdir(parents=True, exist_ok=True)
        preflight_variant = variants[0]
        for model_alias in requested_models:
            for icl_mode in icl_modes:
                output_path = preflight_dir / f'{model_alias}__{icl_mode}.json'
                command = _trial_command(
                    variant_id=preflight_variant,
                    model_alias=model_alias,
                    icl_mode=icl_mode,
                    output_path=output_path,
                    trial_index=0,
                    max_replans=args.max_replans,
                    headless=args.headless,
                    remote=args.remote,
                    remote_url=args.remote_url,
                    replan_mode=args.replan_mode,
                    preflight_only=True,
                    goal=args.goal,
                )
                print(f'[LLM Benchmark] Preflight {model_alias} :: {icl_mode}')
                preflight_records.append(_run_and_load(command, output_path))

    trials_dir = run_dir / 'trials'
    trials_dir.mkdir(parents=True, exist_ok=True)
    for model_alias in execution_models:
        for icl_mode in icl_modes:
            combo_dir = trials_dir / model_alias / icl_mode
            combo_dir.mkdir(parents=True, exist_ok=True)
            for variant_id in variants:
                variant_dir = combo_dir / variant_id
                variant_dir.mkdir(parents=True, exist_ok=True)
                for trial_index in range(1, max(1, args.trials) + 1):
                    output_path = variant_dir / f'trial_{trial_index:03d}.json'
                    command = _trial_command(
                        variant_id=variant_id,
                        model_alias=model_alias,
                        icl_mode=icl_mode,
                        output_path=output_path,
                        trial_index=trial_index,
                        max_replans=args.max_replans,
                        headless=args.headless,
                        remote=args.remote,
                        remote_url=args.remote_url,
                        replan_mode=args.replan_mode,
                        preflight_only=False,
                        goal=args.goal,
                    )
                    print(f'[LLM Benchmark] Run {model_alias} :: {icl_mode} :: {variant_id} :: trial {trial_index}/{args.trials}')
                    execution_records.append(_run_and_load(command, output_path))

    aggregate_by_combo: Dict[str, Dict] = {}
    for model_alias in execution_models:
        for icl_mode in icl_modes:
            combo_key = f'{model_alias}::{icl_mode}'
            combo_records = [
                record
                for record in execution_records
                if record.get('model_alias') == model_alias and record.get('icl_mode') == icl_mode
            ]
            if combo_records:
                aggregate_by_combo[combo_key] = aggregate_model_records(combo_records)

    summary = {
        'created_at': datetime.now().isoformat(),
        'requested_models': requested_models,
        'execution_models': execution_models,
        'final_model_trio': list(FINAL_MODEL_TRIO),
        'icl_modes': icl_modes,
        'variants': variants,
        'trials_per_variant': int(args.trials),
        'text_only': True,
        'segmentation_first': True,
        'replan_mode': args.replan_mode,
        'use_remote_planner': bool(args.remote),
        'remote_planner_url': args.remote_url or None,
        'preflight_records': preflight_records,
        'execution_record_count': len(execution_records),
        'execution_aggregates': aggregate_by_combo,
    }
    summary_path = run_dir / 'benchmark_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'[LLM Benchmark] Summary written to {summary_path}')


if __name__ == '__main__':
    main()

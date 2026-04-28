#!/usr/bin/env python3
"""Run repeated model trials for one currently loaded server model and aggregate the results."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.canonical_variants import DEFAULT_MODEL_VARIANTS, get_variant_spec
from evaluation.metrics import aggregate_model_records
from vlm_pipeline.model_registry import resolve_model_spec


def _parse_variants(raw: str) -> List[str]:
    if not raw.strip():
        return list(DEFAULT_MODEL_VARIANTS)
    return [token.strip().upper() for token in raw.split(',') if token.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description='Run repeated model trials for one selected model')
    parser.add_argument('--model', required=True, help='Registered model alias or HF path')
    parser.add_argument('--model-type', default='', choices=['', 'vlm', 'llm'], help='Explicit model type for custom HF paths')
    parser.add_argument('--variants', default=','.join(DEFAULT_MODEL_VARIANTS), help='Comma-separated kitchen variants to evaluate')
    parser.add_argument('--trials', type=int, default=10, help='Trials per variant')
    parser.add_argument('--remote', action='store_true', help='Use the remote planner server')
    parser.add_argument('--remote-url', default=os.environ.get('VLM_SERVER_URL', 'http://localhost:8080'), help='Remote planner server URL')
    parser.add_argument('--max-replans', type=int, default=3, help='Maximum replans during execution')
    parser.add_argument('--goal', default='', help='Optional goal override for all trials in this benchmark batch')
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument('--gui', action='store_true', help='Run with simulator GUI (default)')
    display_group.add_argument('--headless', action='store_true', help='Run without simulator GUI')
    parser.add_argument('--no-live-masks', action='store_true', help='Disable the live segmentation viewer window')
    parser.add_argument('--no-discovery-replan', action='store_true', help='Disable discovery-triggered replanning')
    parser.add_argument('--live-mask-stride', type=int, default=5, help='Refresh live masks every N simulator steps')
    parser.add_argument('--skip-preflight', action='store_true', help='Skip the initial dry-run validation')
    parser.add_argument('--allow-unsupported-model', action='store_true', help='Allow smoke-test-only models')
    parser.add_argument('--output-root', default='evaluation_results/models', help='Root directory for benchmark outputs')
    args = parser.parse_args()

    model_spec = resolve_model_spec(args.model, args.model_type)
    variants = []
    skipped = []
    for variant_id in _parse_variants(args.variants):
        spec = get_variant_spec(variant_id)
        if spec.pending or not spec.model_eval_supported:
            skipped.append({'variant_id': variant_id, 'reason': spec.model_eval_reason or 'unsupported'})
            continue
        variants.append(variant_id)
    if not variants:
        raise RuntimeError('No benchmarkable variants selected for model evaluation.')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = (ROOT_DIR / args.output_root / model_spec.alias / timestamp).resolve()
    trials_dir = run_dir / 'trials'
    trials_dir.mkdir(parents=True, exist_ok=True)
    records = []
    preflight_record = None

    if not args.skip_preflight:
        preflight_output = run_dir / 'preflight.json'
        preflight_cmd = [
            sys.executable,
            str(ROOT_DIR / 'run_model_trial.py'),
            '--variant', variants[0],
            '--model', args.model,
            '--trial-index', '0',
            '--preflight-only',
            '--output', str(preflight_output),
        ]
        if args.model_type:
            preflight_cmd.extend(['--model-type', args.model_type])
        if args.remote:
            preflight_cmd.extend(['--remote', '--remote-url', args.remote_url])
        if args.allow_unsupported_model:
            preflight_cmd.append('--allow-unsupported-model')
        if args.goal:
            preflight_cmd.extend(['--goal', args.goal])
        if args.headless:
            preflight_cmd.append('--headless')
        if args.no_live_masks:
            preflight_cmd.append('--no-live-masks')
        if args.no_discovery_replan:
            preflight_cmd.append('--no-discovery-replan')
        if args.live_mask_stride != 5:
            preflight_cmd.extend(['--live-mask-stride', str(args.live_mask_stride)])
        print(f'[Model Benchmark] Running preflight for {model_spec.alias}')
        preflight_result = subprocess.run(preflight_cmd, cwd=str(ROOT_DIR), text=True)
        if preflight_result.returncode != 0:
            raise RuntimeError(f'Preflight failed for model {model_spec.alias}')
        preflight_record = json.loads(preflight_output.read_text())
        if not preflight_record.get('preflight_success'):
            raise RuntimeError(f'Preflight validation did not pass for model {model_spec.alias}')

    for variant_id in variants:
        variant_dir = trials_dir / variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        for trial_index in range(1, max(1, args.trials) + 1):
            output_path = variant_dir / f'trial_{trial_index:03d}.json'
            command = [
                sys.executable,
                str(ROOT_DIR / 'run_model_trial.py'),
                '--variant', variant_id,
                '--model', args.model,
                '--trial-index', str(trial_index),
                '--max-replans', str(args.max_replans),
                '--output', str(output_path),
            ]
            if args.model_type:
                command.extend(['--model-type', args.model_type])
            if args.remote:
                command.extend(['--remote', '--remote-url', args.remote_url])
            if args.allow_unsupported_model:
                command.append('--allow-unsupported-model')
            if args.goal:
                command.extend(['--goal', args.goal])
            if args.headless:
                command.append('--headless')
            if args.no_live_masks:
                command.append('--no-live-masks')
            if args.no_discovery_replan:
                command.append('--no-discovery-replan')
            if args.live_mask_stride != 5:
                command.extend(['--live-mask-stride', str(args.live_mask_stride)])
            print(f'[Model Benchmark] Running {model_spec.alias} on {variant_id} trial {trial_index}/{args.trials}')
            result = subprocess.run(command, cwd=str(ROOT_DIR), text=True)
            if result.returncode != 0:
                raise RuntimeError(f'Model trial failed for {model_spec.alias} {variant_id} trial {trial_index}')
            records.append(json.loads(output_path.read_text()))

    summary = {
        'created_at': datetime.now().isoformat(),
        'model_alias': model_spec.alias,
        'model_path': model_spec.path,
        'model_type': model_spec.model_type,
        'variants_requested': variants,
        'variants_skipped': skipped,
        'trials_per_variant': int(args.trials),
        'preflight_record': preflight_record,
        'aggregate': aggregate_model_records(records),
    }
    summary_path = run_dir / 'benchmark_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'[Model Benchmark] Summary written to {summary_path}')


if __name__ == '__main__':
    main()

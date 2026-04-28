#!/usr/bin/env python3
"""Run repeated ground-truth trials and aggregate the results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.canonical_variants import DEFAULT_GT_VARIANTS, get_variant_spec
from evaluation.metrics import aggregate_gt_records


def _parse_variants(raw: str) -> List[str]:
    if not raw.strip():
        return list(DEFAULT_GT_VARIANTS)
    return [token.strip().upper() for token in raw.split(',') if token.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description='Run repeated ground-truth benchmark trials')
    parser.add_argument('--variants', type=str, default=','.join(DEFAULT_GT_VARIANTS), help='Comma-separated variant ids')
    parser.add_argument('--trials', type=int, default=10, help='Trials per variant')
    parser.add_argument('--output-root', type=str, default='evaluation_results/ground_truth', help='Root directory for benchmark outputs')
    parser.add_argument('--gui', action='store_true', help='Run with simulator GUI')
    parser.add_argument('--record-video', action='store_true', help='Enable GT video capture')
    parser.add_argument('--keep-alive', action='store_true', help='Keep simulator alive after each GT run')
    args = parser.parse_args()

    variants = _parse_variants(args.variants)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = (ROOT_DIR / args.output_root / timestamp).resolve()
    trials_dir = run_dir / 'trials'
    trials_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for variant_id in variants:
        spec = get_variant_spec(variant_id)
        if spec.pending:
            print(f'[GT Benchmark] Skipping pending variant: {variant_id}')
            continue
        if not spec.gt_runner_path:
            print(f'[GT Benchmark] Skipping variant without GT runner: {variant_id}')
            continue
        variant_dir = trials_dir / variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        for trial_index in range(1, max(1, args.trials) + 1):
            output_path = variant_dir / f'trial_{trial_index:03d}.json'
            command = [
                sys.executable,
                str(ROOT_DIR / 'run_ground_truth_trial.py'),
                '--variant', variant_id,
                '--trial-index', str(trial_index),
                '--output', str(output_path),
            ]
            if args.gui:
                command.append('--gui')
            if args.record_video:
                command.append('--record-video')
            if args.keep_alive:
                command.append('--keep-alive')
            print(f'[GT Benchmark] Running {variant_id} trial {trial_index}/{args.trials}')
            result = subprocess.run(command, cwd=str(ROOT_DIR), text=True)
            if result.returncode != 0:
                raise RuntimeError(f'GT trial failed for {variant_id} trial {trial_index}')
            records.append(json.loads(output_path.read_text()))

    summary = {
        'created_at': datetime.now().isoformat(),
        'variants_requested': variants,
        'trials_per_variant': int(args.trials),
        'records_path': str(trials_dir),
        'aggregate': aggregate_gt_records(records),
    }
    summary_path = run_dir / 'benchmark_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'[GT Benchmark] Summary written to {summary_path}')


if __name__ == '__main__':
    main()

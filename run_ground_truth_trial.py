#!/usr/bin/env python3
"""Run one ground-truth trial and emit a structured JSON record."""

from __future__ import annotations

import argparse
import json
import os
import selectors
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.canonical_variants import get_variant_spec


def _repo_env(headless: bool, record_video: bool, keep_alive: bool) -> Dict[str, str]:
    env = os.environ.copy()
    pddlstream_path = str(ROOT_DIR / 'pddlstream')
    existing = env.get('PYTHONPATH', '').strip()
    env['PYTHONPATH'] = f"{pddlstream_path}:{existing}" if existing else pddlstream_path
    env['GT_RECORD_VIDEO'] = '1' if record_video else '0'
    env['GT_KEEP_ALIVE'] = '1' if keep_alive else '0'
    env['HEADLESS'] = 'True' if headless else 'False'
    env['COPPELIASIM_HEADLESS'] = '1' if headless else '0'
    return env


def _run_command(command: list[str], cwd: str, env: Dict[str, str], live_output: bool) -> subprocess.CompletedProcess[str]:
    if not live_output:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
        )

    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    selector = selectors.DefaultSelector()

    if proc.stdout is not None:
        selector.register(proc.stdout, selectors.EVENT_READ, ("stdout", stdout_chunks, sys.stdout))
    if proc.stderr is not None:
        selector.register(proc.stderr, selectors.EVENT_READ, ("stderr", stderr_chunks, sys.stderr))

    while selector.get_map():
        events = selector.select(timeout=0.2)
        if not events and proc.poll() is not None:
            break
        for key, _mask in events:
            stream_name, chunks, target = key.data
            line = key.fileobj.readline()
            if line == "":
                try:
                    selector.unregister(key.fileobj)
                except Exception:
                    pass
                continue
            chunks.append(line)
            target.write(line)
            target.flush()

    for stream_name, pipe, chunks in (
        ("stdout", proc.stdout, stdout_chunks),
        ("stderr", proc.stderr, stderr_chunks),
    ):
        if pipe is None:
            continue
        try:
            rest = pipe.read()
        except Exception:
            rest = ""
        if rest:
            chunks.append(rest)
            target = sys.stdout if stream_name == "stdout" else sys.stderr
            target.write(rest)
            target.flush()

    return subprocess.CompletedProcess(
        command,
        proc.wait(),
        "".join(stdout_chunks),
        "".join(stderr_chunks),
    )


def run_trial(variant_id: str,
              trial_index: int,
              output_path: Optional[Path] = None,
              headless: bool = True,
              record_video: bool = False,
              keep_alive: bool = False) -> Dict[str, Any]:
    spec = get_variant_spec(variant_id)
    if spec.pending:
        raise RuntimeError(f'Variant {spec.variant_id} is pending and cannot be benchmarked yet.')
    if not spec.gt_runner_path:
        raise RuntimeError(f'Variant {spec.variant_id} has no executable ground-truth runner.')

    output_path = output_path.resolve() if output_path else None
    output_dir = output_path.parent if output_path else Path(tempfile.mkdtemp(prefix='gt_trial_'))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f'{spec.variant_id}_trial_{trial_index:03d}.runner_summary.json'
    stdout_path = output_dir / f'{spec.variant_id}_trial_{trial_index:03d}.stdout.log'
    stderr_path = output_dir / f'{spec.variant_id}_trial_{trial_index:03d}.stderr.log'

    env = _repo_env(headless=headless, record_video=record_video, keep_alive=keep_alive)
    env['GT_SUMMARY_JSON'] = str(summary_path)
    if spec.task_family == 'grill':
        env['GRILL_ALLOW_SCENE_OVERRIDE'] = 'True'
        env['GRILL_SCENE_FILE_OVERRIDE'] = spec.scene_path

    command = [sys.executable, spec.gt_runner_path]
    result = _run_command(
        command,
        cwd=str(ROOT_DIR),
        env=env,
        live_output=not headless,
    )
    stdout_path.write_text(result.stdout or '')
    stderr_path.write_text(result.stderr or '')

    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        failure_reason = f'runner_returncode:{result.returncode}'
        summary = {
            'variant_id': spec.variant_id,
            'task_family': spec.task_family,
            'scene_path': spec.scene_path,
            'gt_total_subtasks': spec.gt_total_subtasks,
            'gt_completed_subtasks': 0,
            'episode_success': False,
            'execution_time_s': None,
            'failure_reason': failure_reason,
            'task_results': [],
        }

    gt_total = int(summary.get('gt_total_subtasks') or spec.gt_total_subtasks or 0)
    gt_completed = int(summary.get('gt_completed_subtasks') or 0)
    completion_rate = float(gt_completed / gt_total) if gt_total else 0.0
    record = {
        'variant_id': spec.variant_id,
        'task_family': spec.task_family,
        'scene_path': spec.scene_path,
        'trial_index': int(trial_index),
        'action_sequence_length': spec.action_sequence_length,
        'gt_total_subtasks': gt_total,
        'gt_completed_subtasks': gt_completed,
        'subtask_completion_rate': completion_rate,
        'episode_success': bool(summary.get('episode_success')),
        'execution_time_s': summary.get('execution_time_s'),
        'failure_reason': summary.get('failure_reason'),
        'task_results': summary.get('task_results', []),
        'subprocess_returncode': int(result.returncode),
        'stdout_log_path': str(stdout_path),
        'stderr_log_path': str(stderr_path),
        'runner_summary_path': str(summary_path),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, indent=2))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description='Run one ground-truth benchmark trial')
    parser.add_argument('--variant', required=True, help='Variant id (K1/K2/K3/G1/G2/G3)')
    parser.add_argument('--trial-index', type=int, default=1, help='1-based trial index')
    parser.add_argument('--output', type=str, default='', help='Output JSON path')
    parser.add_argument('--gui', action='store_true', help='Run with simulator GUI instead of headless mode')
    parser.add_argument('--record-video', action='store_true', help='Enable GT video capture for this trial')
    parser.add_argument('--keep-alive', action='store_true', help='Keep simulator alive after GT run')
    args = parser.parse_args()

    record = run_trial(
        variant_id=args.variant,
        trial_index=args.trial_index,
        output_path=Path(args.output) if args.output else None,
        headless=not args.gui,
        record_video=args.record_video,
        keep_alive=args.keep_alive,
    )
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()

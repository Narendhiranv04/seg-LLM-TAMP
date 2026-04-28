"""Metrics helpers for direct-action LLM pipeline evaluation."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from evaluation.canonical_variants import get_variant_spec
from evaluation.metrics import aggregate_model_records, collect_failure_occurrences


ACTION_PATTERN = re.compile(r'^\s*([A-Za-z0-9_-]+)\((.*?)\)\s*$')
MUG_OBJECTS = {'mug1', 'mug2', 'mug3', 'mug4'}
GROCERY_OBJECTS = {'soup', 'mustard', 'spam', 'sugar', 'crackers'}
BOX_REGIONS = {'box_boundary', 'box_top', 'box_inside', 'box-top', 'box-inside'}
PLACEMENT_REGIONS = {'placement_boundary', 'table'}
CUPBOARD_REGIONS = {'cupboard_boundary', 'cupboard_boundary_top', 'cupboard', 'groceries_boundary'}


def _normalize_token(token: Optional[str]) -> str:
    return (token or '').strip().lower().replace(' ', '_')


def _normalize_action_name(name: str) -> str:
    token = _normalize_token(name).replace('-', '_')
    if token in {'pick', 'pickup', 'pick_up', 'grasp'}:
        return 'pick'
    if token in {'place', 'put', 'put_down', 'putdown'}:
        return 'place'
    if token in {'open', 'open_lid', 'openlid', 'open_box', 'open_box_lid'}:
        return 'open_lid'
    return token


def _normalize_region(region: Optional[str]) -> str:
    token = _normalize_token(region)
    if token in BOX_REGIONS:
        return 'box_boundary'
    if token in PLACEMENT_REGIONS:
        return 'placement_boundary'
    if token in CUPBOARD_REGIONS:
        return 'cupboard_boundary'
    return token


def parse_action_string(action: Any) -> Optional[Dict[str, Any]]:
    text = str(action).strip()
    if text == 'move':
        return {'action': 'move', 'args': [], 'raw': text}
    match = ACTION_PATTERN.match(text)
    if not match:
        return None
    name = _normalize_action_name(match.group(1))
    raw_args = match.group(2).strip()
    args = []
    if raw_args:
        args = [_normalize_token(part) for part in raw_args.split(',') if part.strip()]
    return {'action': name, 'args': args, 'raw': text}


def _bucket_for_transfer(object_name: str, region_name: str) -> Optional[str]:
    obj = _normalize_token(object_name)
    region = _normalize_region(region_name)
    if obj in MUG_OBJECTS and region == 'placement_boundary':
        return 'mug_to_placement'
    if obj in MUG_OBJECTS and region == 'box_boundary':
        return 'mug_to_box'
    if obj in GROCERY_OBJECTS and region == 'cupboard_boundary':
        return 'grocery_to_cupboard'
    return None


def collapse_actions_to_subtasks(actions: Sequence[Any]) -> List[str]:
    parsed = [parse_action_string(action) for action in actions]
    parsed = [item for item in parsed if item is not None]
    subtasks: List[str] = []
    idx = 0
    while idx < len(parsed):
        current = parsed[idx]
        name = current['action']
        args = current['args']
        if name == 'move':
            idx += 1
            continue
        if name == 'open_lid':
            subtasks.append('open_lid')
            idx += 1
            continue
        if name == 'pick' and idx + 1 < len(parsed):
            nxt = parsed[idx + 1]
            if nxt['action'] == 'place' and len(args) == 1 and len(nxt['args']) >= 2:
                obj = args[0]
                place_obj = nxt['args'][0]
                region = nxt['args'][1]
                if obj == place_obj:
                    bucket = _bucket_for_transfer(obj, region)
                    if bucket:
                        subtasks.append(bucket)
                        idx += 2
                        continue
        idx += 1
    return subtasks


def score_variant_completion(variant_id: str, completed_actions: Sequence[Any]) -> Dict[str, Any]:
    spec = get_variant_spec(variant_id)
    expected = dict(spec.expected_subtask_buckets)
    observed_buckets = collapse_actions_to_subtasks(completed_actions)
    observed_counts = Counter(observed_buckets)
    bucket_breakdown: Dict[str, Dict[str, int]] = {}
    matched_total = 0
    for bucket_name, expected_count in expected.items():
        observed_count = int(observed_counts.get(bucket_name, 0))
        matched_count = min(observed_count, int(expected_count))
        matched_total += matched_count
        bucket_breakdown[bucket_name] = {
            'expected': int(expected_count),
            'observed': observed_count,
            'matched': matched_count,
        }

    extras = {
        bucket_name: int(count)
        for bucket_name, count in observed_counts.items()
        if bucket_name not in expected or count > expected.get(bucket_name, 0)
    }
    gt_total = int(spec.gt_total_subtasks or 0)
    completion_rate = float(matched_total / gt_total) if gt_total else 0.0
    return {
        'variant_id': spec.variant_id,
        'gt_total_subtasks': gt_total,
        'completed_gt_subtasks': int(matched_total),
        'subtask_completion_rate': completion_rate,
        'observed_subtasks': observed_buckets,
        'observed_subtask_counts': dict(observed_counts),
        'bucket_breakdown': bucket_breakdown,
        'extra_observed_subtasks': extras,
    }


__all__ = [
    'aggregate_model_records',
    'collect_failure_occurrences',
    'collapse_actions_to_subtasks',
    'parse_action_string',
    'score_variant_completion',
]

"""Scoring and aggregation helpers for evaluation runs."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from evaluation.canonical_variants import VARIANTS, get_variant_spec


ACTION_PATTERN = re.compile(r'^\s*([A-Za-z0-9_-]+)\((.*?)\)\s*$')
MUG_OBJECTS = {
    'mug_box', 'mug_inside_box', 'mug_table', 'mug_cupboard',
    'mug1', 'mug2', 'mug3', 'mug4',
}
GROCERY_OBJECTS = {'soup', 'mustard', 'spam', 'sugar', 'crackers'}
MEAT_OBJECTS = {'steak', 'steak1', 'steak2', 'chicken', 'chicken1', 'chicken2'}
PLATE_OBJECTS = {'plate'}
BOX_REGIONS = {'box_boundary', 'box_top', 'box_inside', 'box-top', 'box-inside'}
PLACEMENT_REGIONS = {'placement_boundary', 'table'}
CUPBOARD_REGIONS = {'cupboard_boundary', 'cupboard_boundary_top', 'cupboard', 'groceries_boundary'}
PLATE_REGIONS = {'plate', 'plate_boundary', 'plate_boundary_top'}
GRILL_REGIONS = {'grill', 'grill_top', 'grill-top'}
GENERIC_TERMINAL_REASONS = {
    'max replans exceeded',
    'replan failed',
    'initial planning failed',
    'unexpected exit',
}


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_std(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mean_val = _safe_mean(values)
    variance = sum((float(v) - mean_val) ** 2 for v in values) / (len(values) - 1)
    return float(math.sqrt(max(0.0, variance)))


def _normalize_token(token: Optional[str]) -> str:
    return (token or '').strip().lower().replace(' ', '_')


def _normalize_action_name(name: str) -> str:
    token = _normalize_token(name).replace('-', '_')
    if token in {'pick', 'pickup', 'pick_up', 'grasp'}:
        return 'pick'
    if token in {'place', 'put', 'put_down', 'putdown'}:
        return 'place'
    if token in {'open_lid', 'openlid', 'open_box', 'open_box_lid'}:
        return 'open_lid'
    if token in {'close_lid', 'closelid', 'close_box', 'close_grill'}:
        return 'close_lid'
    if token in {'open_grill'}:
        return 'open_grill'
    if token in {'close_grill_lid'}:
        return 'close_grill'
    return token


def _normalize_region(region: Optional[str]) -> str:
    token = _normalize_token(region)
    if token in BOX_REGIONS:
        return 'box_boundary'
    if token in PLACEMENT_REGIONS:
        return 'placement_boundary'
    if token in CUPBOARD_REGIONS:
        return 'cupboard_boundary'
    if token in PLATE_REGIONS:
        return 'plate'
    if token in GRILL_REGIONS:
        return 'grill'
    return token


def parse_action_string(action: Any) -> Optional[Dict[str, Any]]:
    text = str(action).strip()
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
    if obj in PLATE_OBJECTS and region == 'plate':
        return 'plate_to_boundary'
    if obj in MEAT_OBJECTS and region == 'plate':
        return 'meat_to_plate'
    if obj in MEAT_OBJECTS and region == 'grill':
        return 'meat_to_grill'
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
        if name in {'open_lid', 'open_grill'}:
            subtasks.append('open_lid' if name == 'open_lid' else 'open_grill')
            idx += 1
            continue
        if name in {'close_lid', 'close_grill'}:
            subtasks.append('close_grill')
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


def classify_failure_message(message: Optional[str]) -> str:
    text = (message or '').strip().lower()
    if not text:
        return 'unknown'
    if 'new_object_introduced_in_scene' in text or 'newly visible object' in text:
        return 'new_object_introduced_in_scene'
    if 'cannot open lid' in text or 'on top' in text or 'blocked by' in text or 'object_blocked' in text:
        return 'object_blocked'
    if 'lid is closed' in text or 'lid_closed' in text or 'cannot pick' in text and 'closed' in text:
        return 'lid_closed'
    if 'not found' in text or 'object_not_found' in text:
        return 'object_not_found'
    if 'orphaned' in text and 'place' in text:
        return 'orphan_place'
    if 'pick/place different objects' in text or 'pick mismatch' in text or 'pick_place_mismatch' in text:
        return 'pick_mismatch'
    if 'grasp failed' in text or "didn't move" in text or 'not grasped' in text:
        return 'grasp_failed'
    if 'placement failed' in text or 'not in target region' in text:
        return 'placement_failed'
    if 'valid joint configuration' in text or 'inverse kinematics' in text or 'ik solution' in text:
        return 'no_ik_solution'
    if 'motion planner failed' in text or 'no motion plan' in text:
        return 'no_motion_plan'
    if 'no valid grasp' in text or 'no grasp found' in text:
        return 'no_grasp_found'
    if 'pddl' in text and 'no plan' in text:
        return 'pddl_no_plan'
    if 'collision' in text:
        return 'collision_detected'
    if 'dropped' in text or 'fell during transport' in text or 'object fell' in text:
        return 'object_dropped'
    if text in GENERIC_TERMINAL_REASONS:
        return 'unknown'
    return 'unknown'


def collect_failure_occurrences(cycles: Sequence[Dict[str, Any]], failure_reason: Optional[str]) -> Dict[str, Any]:
    messages: List[str] = []
    for cycle in cycles or []:
        error_message = (cycle or {}).get('error_message')
        if error_message:
            messages.append(str(error_message))
    normalized_failure_reason = (failure_reason or '').strip()
    if normalized_failure_reason and normalized_failure_reason.lower() not in GENERIC_TERMINAL_REASONS:
        messages.append(normalized_failure_reason)
    categories = [classify_failure_message(message) for message in messages]
    counts = Counter(categories)
    return {
        'messages': messages,
        'categories': categories,
        'counts': dict(counts),
    }


def _aggregate_failure_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_occurrences: Counter = Counter()
    trials_affected: Counter = Counter()
    for record in records:
        failure_info = record.get('failure_occurrences', {}) or {}
        categories = [str(cat) for cat in failure_info.get('categories', [])]
        total_occurrences.update(categories)
        trials_affected.update(set(categories))
    categories = sorted(set(total_occurrences) | set(trials_affected))
    return {
        'categories': categories,
        'total_occurrences': {name: int(total_occurrences.get(name, 0)) for name in categories},
        'trials_affected': {name: int(trials_affected.get(name, 0)) for name in categories},
    }


def aggregate_gt_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_variant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_variant[str(record.get('variant_id'))].append(record)

    variant_summaries: Dict[str, Dict[str, Any]] = {}
    success_values: List[float] = []
    completion_values: List[float] = []
    execution_times: List[float] = []
    for variant_id in sorted(by_variant):
        rows = by_variant[variant_id]
        success = [1.0 if row.get('episode_success') else 0.0 for row in rows]
        completion = [float(row.get('subtask_completion_rate', 0.0)) for row in rows]
        times = [float(row.get('execution_time_s', 0.0)) for row in rows if row.get('execution_time_s') is not None]
        variant_summaries[variant_id] = {
            'trials': len(rows),
            'success_rate': _safe_mean(success),
            'mean_subtask_completion_rate': _safe_mean(completion),
            'mean_execution_time_s': _safe_mean(times),
            'std_execution_time_s': _safe_std(times),
            'gt_total_subtasks': rows[0].get('gt_total_subtasks'),
            'action_sequence_length': rows[0].get('action_sequence_length'),
        }
        success_values.extend(success)
        completion_values.extend(completion)
        execution_times.extend(times)

    return {
        'record_count': len(records),
        'supported_variants': sorted(by_variant),
        'variants': variant_summaries,
        'overall': {
            'success_rate': _safe_mean(success_values),
            'mean_subtask_completion_rate': _safe_mean(completion_values),
            'mean_execution_time_s': _safe_mean(execution_times),
            'std_execution_time_s': _safe_std(execution_times),
        },
    }


def aggregate_model_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_variant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_variant[str(record.get('variant_id'))].append(record)

    variant_summaries: Dict[str, Dict[str, Any]] = {}
    success_values: List[float] = []
    raw_success_values: List[float] = []
    completion_values: List[float] = []
    replans_values: List[float] = []
    episode_times: List[float] = []

    for variant_id in sorted(by_variant):
        rows = by_variant[variant_id]
        success = [1.0 if row.get('episode_success') else 0.0 for row in rows]
        raw_success = [1.0 if row.get('raw_episode_success') else 0.0 for row in rows]
        completion = [float(row.get('subtask_completion_rate', 0.0)) for row in rows]
        replans = [float(row.get('total_replans', 0)) for row in rows]
        times = [float(row.get('episode_time_s', 0.0)) for row in rows if row.get('episode_time_s') is not None]
        variant_summaries[variant_id] = {
            'trials': len(rows),
            'episode_success_rate': _safe_mean(success),
            'raw_execution_success_rate': _safe_mean(raw_success),
            'mean_subtask_completion_rate': _safe_mean(completion),
            'mean_completed_gt_subtasks': _safe_mean([float(row.get('completed_gt_subtasks', 0)) for row in rows]),
            'mean_replans': _safe_mean(replans),
            'mean_episode_time_s': _safe_mean(times),
            'std_episode_time_s': _safe_std(times),
            'gt_total_subtasks': rows[0].get('gt_total_subtasks'),
            'action_sequence_length': rows[0].get('action_sequence_length'),
            'model_alias': rows[0].get('model_alias'),
            'model_type': rows[0].get('model_type'),
            'prompt_mode': rows[0].get('prompt_mode'),
        }
        success_values.extend(success)
        raw_success_values.extend(raw_success)
        completion_values.extend(completion)
        replans_values.extend(replans)
        episode_times.extend(times)

    return {
        'record_count': len(records),
        'supported_variants': sorted(by_variant),
        'variants': variant_summaries,
        'overall': {
            'episode_success_rate': _safe_mean(success_values),
            'raw_execution_success_rate': _safe_mean(raw_success_values),
            'mean_subtask_completion_rate': _safe_mean(completion_values),
            'mean_replans': _safe_mean(replans_values),
            'mean_episode_time_s': _safe_mean(episode_times),
            'std_episode_time_s': _safe_std(episode_times),
        },
        'failure_counts': _aggregate_failure_counts(records),
    }

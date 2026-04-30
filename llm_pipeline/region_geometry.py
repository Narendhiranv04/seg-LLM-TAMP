"""Canonical geometric object-to-region resolution."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

import numpy as np

from llm_pipeline.region_aliases import (
    BOX_INSIDE_FALLBACK_REGION,
    PLANNER_HIDDEN_REGIONS,
    normalize_region_name,
)


RegionBounds = Tuple[np.ndarray, np.ndarray]

PRIMARY_REGION_PRIORITY = (
    "cupboard_upper",
    "cupboard_lower",
    "box_lid_top",
    "box_storage",
    "groceries_boundary",
    "placement_boundary",
    "table",
)
FALLBACK_REGION_PRIORITY = tuple(PLANNER_HIDDEN_REGIONS)

REGION_DESCRIPTIONS = {
    "cupboard_upper": "on upper cupboard shelf",
    "cupboard_lower": "on lower cupboard shelf",
    "box_lid_top": "on top of the box lid",
    "box_storage": "inside the box storage target",
    "groceries_boundary": "in groceries area",
    "placement_boundary": "in placement area",
    "table": "on table",
    "box_inside_fallback": "inside broad box fallback",
    "cupboard_fallback": "inside broad cupboard fallback",
}

REGION_PADDING = {
    "cupboard_upper": 0.05,
    "cupboard_lower": 0.05,
    "box_lid_top": 0.04,
    "box_storage": 0.05,
    "groceries_boundary": 0.05,
    "placement_boundary": 0.05,
    "table": 0.04,
    "box_inside_fallback": 0.04,
    "cupboard_fallback": 0.04,
}

REGION_Z_MARGIN = {
    "cupboard_upper": (0.15, 0.20),
    "cupboard_lower": (0.15, 0.20),
    "box_lid_top": (0.05, 0.12),
    "box_storage": (0.15, 0.20),
    "groceries_boundary": (0.15, 0.20),
    "placement_boundary": (0.15, 0.20),
    "table": (0.05, 0.12),
    "box_inside_fallback": (0.15, 0.20),
    "cupboard_fallback": (0.20, 0.25),
}


def is_inside_xy(point: Tuple[float, float, float], world_min: np.ndarray, world_max: np.ndarray, padding: float = 0.04) -> bool:
    """Check whether a point lies inside a region footprint."""
    return (
        point[0] >= float(world_min[0]) - padding
        and point[0] <= float(world_max[0]) + padding
        and point[1] >= float(world_min[1]) - padding
        and point[1] <= float(world_max[1]) + padding
    )


def _normalize_region_map(region_map: Mapping[str, RegionBounds]) -> Dict[str, RegionBounds]:
    normalized = {}
    for region_name, bounds in (region_map or {}).items():
        canonical = normalize_region_name(region_name)
        if not canonical:
            continue
        w_min, w_max = bounds
        normalized[canonical] = (np.array(w_min, dtype=float), np.array(w_max, dtype=float))
    return normalized


def _ordered_regions(valid_regions: Iterable[str], region_map: Mapping[str, RegionBounds]) -> list[str]:
    available = set(_normalize_region_map(region_map).keys())
    valid = {normalize_region_name(region) for region in (valid_regions or [])}
    if valid:
        available &= valid

    primary = [region for region in PRIMARY_REGION_PRIORITY if region in available]
    fallback = [region for region in FALLBACK_REGION_PRIORITY if region in available and region not in primary]
    extras = sorted(region for region in available if region not in set(primary + fallback))
    return primary + fallback + extras


def point_matches_region(point: Tuple[float, float, float], region_name: str, bounds: RegionBounds) -> bool:
    """Return whether an object center is geometrically compatible with a region."""
    canonical = normalize_region_name(region_name)
    w_min, w_max = bounds
    if not is_inside_xy(point, w_min, w_max, padding=REGION_PADDING.get(canonical, 0.04)):
        return False

    z_min = float(w_min[2])
    z_max = float(w_max[2])
    below, above = REGION_Z_MARGIN.get(canonical, (0.10, 0.15))
    z = float(point[2])

    if canonical == BOX_INSIDE_FALLBACK_REGION:
        return z >= z_min - below and z <= z_max + above
    if canonical == "cupboard_fallback":
        return z >= z_min - below and z <= z_max + above
    if (z_max - z_min) < 0.01:
        return z >= z_min - below and z <= z_min + above
    return z >= z_min - below and z <= z_max + above


def resolve_region(
    obj_pos: Tuple[float, float, float],
    region_map: Mapping[str, RegionBounds],
    valid_regions: Iterable[str] | None = None,
) -> Tuple[str, str]:
    """Resolve one object pose to a canonical region id and description."""
    normalized_map = _normalize_region_map(region_map)
    for region_name in _ordered_regions(valid_regions or normalized_map.keys(), normalized_map):
        if point_matches_region(obj_pos, region_name, normalized_map[region_name]):
            return region_name, REGION_DESCRIPTIONS.get(region_name, region_name)
    return "table", REGION_DESCRIPTIONS["table"]


def resolve_object_regions(
    pose_map: Mapping[str, Tuple[float, float, float]],
    region_map: Mapping[str, RegionBounds],
    valid_regions: Iterable[str] | None = None,
) -> tuple[Dict[str, str], Dict[str, str]]:
    """Resolve every object pose to canonical object-region maps."""
    object_region_map = {}
    object_region_descriptions = {}
    for object_name, pose in (pose_map or {}).items():
        region_name, description = resolve_region(tuple(pose[:3]), region_map, valid_regions)
        object_region_map[object_name] = region_name
        object_region_descriptions[object_name] = description
    return object_region_map, object_region_descriptions

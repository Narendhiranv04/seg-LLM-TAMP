"""Canonical kitchen region names and compatibility aliases."""

from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping, TypeVar


KITCHEN_REGION_ALIASES = {
    "box_boundary": "box_storage",
    "box-top": "box_lid_top",
    "box_top": "box_lid_top",
    "box-inside": "box_inside_fallback",
    "box_inside": "box_inside_fallback",
    "cupboard_boundary": "cupboard_lower",
    "cupboard_boundary_top": "cupboard_upper",
    "shelf-lower": "cupboard_lower",
    "shelf_lower": "cupboard_lower",
}

GRILL_REGION_ALIASES = {
    "plate-boundary": "plate_boundary",
}

CANONICAL_KITCHEN_REGION_ORDER = (
    "table",
    "placement_boundary",
    "cupboard_lower",
    "cupboard_upper",
    "box_storage",
    "groceries_boundary",
    "box_lid_top",
    "box_inside_fallback",
)

CANONICAL_GRILL_REGION_ORDER = (
    "table",
    "grill-top",
    "plate-top",
    "plate_boundary",
    "dish_rack",
)

BOX_STORAGE_REGION = "box_storage"
BOX_INSIDE_FALLBACK_REGION = "box_inside_fallback"
BOX_LID_TOP_REGION = "box_lid_top"
CUPBOARD_TARGET_REGIONS = ("cupboard_lower", "cupboard_upper")
PLANNER_HIDDEN_REGIONS = (BOX_INSIDE_FALLBACK_REGION,)
CANONICAL_REGION_SCENE_OBJECTS = {
    "box_storage": "box_boundary",
    "box_lid_top": "box_lid",
    "box_inside_fallback": "box_base",
    "cupboard_lower": "cupboard_boundary",
    "cupboard_upper": "cupboard_boundary_top",
}

REGION_SEMANTICS = {
    "table": "broad table surface; use only when no specific table subregion applies",
    "placement_boundary": "specific destination area on the table for placing completed objects",
    "cupboard_lower": "lower shelf inside the cupboard",
    "cupboard_upper": "upper shelf inside the cupboard",
    "box_storage": "inside-box storage target for putting objects into the box",
    "groceries_boundary": "groceries/source area on the table",
    "box_lid_top": "support surface on top of the box lid for objects resting on the lid",
    "box_inside_fallback": "internal broad box evidence; not a planner target",
}


def normalize_region_name(region_name: str | None) -> str:
    token = str(region_name or "").strip()
    canonical = KITCHEN_REGION_ALIASES.get(token, token)
    return GRILL_REGION_ALIASES.get(canonical, canonical)


def region_semantics(region_name: str | None) -> str:
    canonical = normalize_region_name(region_name)
    return REGION_SEMANTICS.get(canonical, "")


def normalize_region_names(region_names: Iterable[str]) -> list[str]:
    seen = set()
    ordered = []
    for region_name in region_names:
        canonical = normalize_region_name(region_name)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        ordered.append(canonical)
    return ordered


def scene_object_for_region(region_name: str) -> str:
    canonical = normalize_region_name(region_name)
    return CANONICAL_REGION_SCENE_OBJECTS.get(canonical, canonical)


_T = TypeVar("_T")


class RegionAliasMap(dict):
    """Dict that stores canonical keys but accepts legacy kitchen region aliases."""

    def __init__(self, values: Mapping[str, _T] | None = None):
        super().__init__()
        if values:
            for key, value in values.items():
                self[key] = value

    def __setitem__(self, key: str, value: _T) -> None:
        super().__setitem__(normalize_region_name(key), value)

    def __getitem__(self, key: str) -> _T:
        return super().__getitem__(normalize_region_name(key))

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            return super().__contains__(normalize_region_name(key))
        return super().__contains__(key)

    def get(self, key: str, default: _T | None = None) -> _T | None:
        return super().get(normalize_region_name(key), default)

    def pop(self, key: str, default=None):
        return super().pop(normalize_region_name(key), default)

    def update(self, values: Mapping[str, _T] | MutableMapping[str, _T], **kwargs) -> None:
        for key, value in dict(values, **kwargs).items():
            self[key] = value

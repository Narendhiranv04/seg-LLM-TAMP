"""Runtime action, object, and region symbols for the maintained LLM pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from llm_pipeline.region_aliases import (
    CANONICAL_GRILL_REGION_ORDER,
    CANONICAL_KITCHEN_REGION_ORDER,
    normalize_region_names,
)


ACTION_SYMBOLS: Tuple[str, ...] = ("move", "pick", "place", "open")
DEFAULT_OBJECT_ORDER: Tuple[str, ...] = (
    "mug1",
    "mug2",
    "mug3",
    "mug4",
    "soup",
    "mustard",
    "spam",
    "sugar",
    "crackers",
    "box_lid",
)
GRILL_OBJECT_ORDER: Tuple[str, ...] = (
    "steak",
    "steak1",
    "steak2",
    "steak3",
    "chicken",
    "chicken1",
    "chicken2",
    "chicken3",
    "spam",
    "spam1",
    "spam2",
    "spam3",
    "plate",
    "grill_lid",
    "lid",
)
DEFAULT_REGION_ORDER: Tuple[str, ...] = CANONICAL_KITCHEN_REGION_ORDER
GRILL_REGION_ORDER: Tuple[str, ...] = CANONICAL_GRILL_REGION_ORDER
EXECUTABLE_OBJECTS: Tuple[str, ...] = DEFAULT_OBJECT_ORDER
EXECUTABLE_REGIONS: Tuple[str, ...] = DEFAULT_REGION_ORDER


@dataclass(frozen=True)
class RuntimeSymbolRegistry:
    actions: Tuple[str, ...] = ACTION_SYMBOLS
    objects: Tuple[str, ...] = EXECUTABLE_OBJECTS
    regions: Tuple[str, ...] = EXECUTABLE_REGIONS

    def to_dict(self) -> dict:
        return {
            "actions": list(self.actions),
            "objects": list(self.objects),
            "regions": list(self.regions),
        }


def _dedupe_preserve_order(items: Iterable[str]) -> Tuple[str, ...]:
    seen = set()
    ordered = []
    for item in items:
        token = str(item).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return tuple(ordered)


def _scene_name(obj) -> Optional[str]:
    if obj is None:
        return None
    try:
        name = obj.get_name()
        if name:
            return str(name).strip()
    except Exception:
        pass
    return None


def _iter_unique_scene_objects(env):
    seen_handles = set()
    for obj in (getattr(env, "name_to_obj", {}) or {}).values():
        if obj is None:
            continue
        try:
            handle = int(obj.get_handle())
        except Exception:
            continue
        if handle in seen_handles:
            continue
        seen_handles.add(handle)
        yield handle, obj


def _objects_from_env(env) -> Tuple[str, ...]:
    if env is None:
        return ()

    names = []
    for name in (getattr(env, "name_to_obj", {}) or {}).keys():
        token = str(name).strip()
        if _is_grill_env(env) and token in {"meat1", "meat2"}:
            continue
        if token:
            names.append(token)
    return _dedupe_preserve_order(names)


def _ordered_known_then_extras(items: Iterable[str], known_order: Tuple[str, ...]) -> Tuple[str, ...]:
    item_set = {str(item).strip() for item in items if str(item).strip()}
    ordered = [name for name in known_order if name in item_set]
    extras = sorted(name for name in item_set if name not in set(known_order))
    return _dedupe_preserve_order(ordered + extras)


def _objects_from_detected(detected_objects=None, env=None) -> Tuple[str, ...]:
    """Build object symbols from detector/env data with kitchen defaults as fallback."""
    candidates = []
    if detected_objects is None:
        candidates.extend(_objects_from_env(env))
    else:
        candidates.extend(detected_objects)
        candidates.extend(_objects_from_env(env))

    known_order = DEFAULT_OBJECT_ORDER + tuple(
        name for name in GRILL_OBJECT_ORDER if name not in set(DEFAULT_OBJECT_ORDER)
    )
    return _ordered_known_then_extras(candidates, known_order) or EXECUTABLE_OBJECTS


def _is_grill_env(env) -> bool:
    if env is None:
        return False
    module_name = env.__class__.__module__.lower()
    class_name = env.__class__.__name__.lower()
    if "grill" in module_name or "grill" in class_name:
        return True
    return hasattr(env, "grill_lid") or hasattr(env, "grill_boundary")


def _regions_from_env(env) -> Tuple[str, ...]:
    if env is None:
        return EXECUTABLE_REGIONS

    region_map = getattr(env, "regions", {}) or {}
    if _is_grill_env(env):
        region_set = set(normalize_region_names(region_map.keys()))
        names = tuple(region for region in GRILL_REGION_ORDER if region in region_set)
        return names or GRILL_REGION_ORDER

    names = _ordered_known_then_extras(normalize_region_names(region_map.keys()), DEFAULT_REGION_ORDER)
    return names or EXECUTABLE_REGIONS


def build_runtime_symbol_registry(env=None, detected_objects=None, context_aggregator=None) -> RuntimeSymbolRegistry:
    """Build the symbol registry.

    Args:
        env: Environment (used for regions via env.regions).
        detected_objects: List of object names discovered from segmentation masks.
            If provided, these names are retained and ordered before env extras.
            If no detector/env data is available, falls back to kitchen defaults.
    """
    del context_aggregator
    return RuntimeSymbolRegistry(
        actions=ACTION_SYMBOLS,
        objects=_objects_from_detected(detected_objects, env=env),
        regions=_regions_from_env(env),
    )

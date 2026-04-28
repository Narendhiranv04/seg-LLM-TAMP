"""Runtime action, object, and region symbols for the maintained LLM pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


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
DEFAULT_REGION_ORDER: Tuple[str, ...] = (
    "table",
    "placement_boundary",
    "cupboard_boundary",
    "cupboard_boundary_top",
    "box_boundary",
    "groceries_boundary",
)
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


def _objects_from_detected(detected_objects=None) -> Tuple[str, ...]:
    """Build object list from mask-detected objects (intersection with class vocab).

    If detected_objects is provided, returns only those that appear in
    DEFAULT_OBJECT_ORDER (intersection), preserving canonical order.
    Falls back to the full DEFAULT_OBJECT_ORDER if nothing is provided.
    """
    if detected_objects is None:
        return EXECUTABLE_OBJECTS

    detected_set = set(detected_objects)
    # Intersection: only objects in BOTH the class vocab AND the masks
    ordered = [name for name in DEFAULT_OBJECT_ORDER if name in detected_set]
    # Include any extras from masks not in the default vocab
    extras = sorted(name for name in detected_objects if name not in set(DEFAULT_OBJECT_ORDER))
    return _dedupe_preserve_order(ordered + extras) or EXECUTABLE_OBJECTS


def _regions_from_env(env) -> Tuple[str, ...]:
    if env is None:
        return EXECUTABLE_REGIONS

    region_map = getattr(env, "regions", {}) or {}
    names = [name for name in DEFAULT_REGION_ORDER if name in region_map]
    extras = sorted(
        name
        for name in region_map.keys()
        if name not in set(DEFAULT_REGION_ORDER) and name not in {"box-top", "box-inside", "shelf-lower"}
    )
    return _dedupe_preserve_order(names + extras) or EXECUTABLE_REGIONS


def build_runtime_symbol_registry(env=None, detected_objects=None, context_aggregator=None) -> RuntimeSymbolRegistry:
    """Build the symbol registry.

    Args:
        env: Environment (used for regions via env.regions).
        detected_objects: List of object names discovered from segmentation masks.
            If provided, the object list is the intersection of this with DEFAULT_OBJECT_ORDER.
            If None, falls back to DEFAULT_OBJECT_ORDER.
    """
    del context_aggregator
    return RuntimeSymbolRegistry(
        actions=ACTION_SYMBOLS,
        objects=_objects_from_detected(detected_objects),
        regions=_regions_from_env(env),
    )


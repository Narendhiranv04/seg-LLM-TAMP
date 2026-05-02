"""Grill-specific semantic facts derived from geometric scene state."""

from __future__ import annotations

from typing import Mapping


MEAT_OBJECTS = frozenset({"spam", "steak", "chicken", "meat1", "meat2"})


def derive_grill_semantic_facts(
    object_region_map: Mapping[str, str],
    *,
    lid_open: bool | None = None,
) -> list[str]:
    """Return grill facts without conflating grill-top with inside-grill."""
    facts = []
    if lid_open is True:
        facts.append("grill_lid_open")
    elif lid_open is False:
        facts.append("grill_lid_closed")

    for object_name, region_name in sorted((object_region_map or {}).items()):
        if object_name in MEAT_OBJECTS:
            if region_name == "grill-top":
                facts.append(f"on_grill_top({object_name})")
            elif region_name == "prep_area":
                facts.append(f"in_prep_area({object_name})")
            elif region_name == "plate-top":
                facts.append(f"on_plate({object_name})")
            elif region_name == "table":
                facts.append(f"on_table({object_name})")
        elif object_name == "plate":
            if region_name == "dish_rack":
                facts.append("plate_at_dish_rack")
            elif region_name == "plate_boundary":
                facts.append("plate_at_boundary")

    return facts


def infer_grill_lid_open(env) -> bool | None:
    """Best-effort lid-open check for grill scenes."""
    lid_joint = getattr(env, "lid_joint", None)
    if lid_joint is None:
        return None
    try:
        current = float(lid_joint.get_joint_position())
    except Exception:
        return None

    closed = float(getattr(env, "_closed_lid_angle", 0.0))
    return abs(current - closed) > 0.25

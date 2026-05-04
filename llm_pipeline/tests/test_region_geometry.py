import numpy as np

from llm_pipeline.region_geometry import resolve_object_regions, resolve_region


def _bounds(x0, y0, z0, x1, y1, z1):
    return np.array((x0, y0, z0)), np.array((x1, y1, z1))


def test_box_lid_top_beats_box_storage_and_table() -> None:
    region_map = {
        'table': _bounds(-1, -1, 0.0, 1, 1, 0.05),
        'box_storage': _bounds(-0.2, -0.2, 0.0, 0.2, 0.2, 0.25),
        'box_lid_top': _bounds(-0.2, -0.2, 0.30, 0.2, 0.2, 0.32),
    }

    region, _ = resolve_region((0.0, 0.0, 0.34), region_map)

    assert region == 'box_lid_top'


def test_cupboard_lower_handles_nearby_lower_shelf_objects() -> None:
    region_map = {
        'cupboard_lower': _bounds(0.45, -0.15, 0.35, 0.85, 0.15, 0.38),
    }

    region, _ = resolve_region((0.4, 0.0, 0.45), region_map)

    assert region == 'cupboard_lower'


def test_groceries_boundary_beats_table() -> None:
    region_map = {
        'table': _bounds(-1, -1, 0.0, 1, 1, 0.05),
        'groceries_boundary': _bounds(0.2, 0.2, 0.05, 0.6, 0.6, 0.06),
    }

    region, _ = resolve_region((0.4, 0.4, 0.12), region_map)

    assert region == 'groceries_boundary'


def test_inside_grill_beats_table_for_meat() -> None:
    region_map = {
        'table': _bounds(-1, -1, 0.0, 1, 1, 0.05),
        'grill-top': _bounds(-0.2, -0.2, 0.12, 0.2, 0.2, 0.14),
    }

    region, description = resolve_region((0.0, 0.0, 0.20), region_map)

    assert region == 'inside_grill'
    assert description == 'inside grill'


def test_plate_top_beats_plate_boundary_for_meat() -> None:
    region_map = {
        'plate_boundary': _bounds(-0.2, -0.2, 0.02, 0.2, 0.2, 0.04),
        'plate-top': _bounds(-0.15, -0.15, 0.05, 0.15, 0.15, 0.06),
    }

    region, description = resolve_region((0.0, 0.0, 0.10), region_map)

    assert region == 'plate-top'
    assert description == 'on plate'


def test_prep_area_beats_table_for_raw_meat() -> None:
    region_map = {
        'table': _bounds(-1, -1, 0.0, 1, 1, 0.05),
        'prep_area': _bounds(0.25, 0.25, 0.05, 0.55, 0.55, 0.06),
    }

    region, description = resolve_region((0.4, 0.4, 0.11), region_map)

    assert region == 'prep_area'
    assert description == 'in prep area'


def test_box_fallback_used_only_when_primary_absent() -> None:
    fallback_only = {'box_inside_fallback': _bounds(-0.2, -0.2, 0.0, 0.2, 0.2, 0.25)}
    with_primary = {
        **fallback_only,
        'box_lid_top': _bounds(-0.2, -0.2, 0.30, 0.2, 0.2, 0.32),
    }

    assert resolve_region((0.0, 0.0, 0.20), fallback_only)[0] == 'box_inside_fallback'
    assert resolve_region((0.0, 0.0, 0.34), with_primary)[0] == 'box_lid_top'


def test_resolve_object_regions_returns_maps() -> None:
    pose_map = {'mug3': (0.6, 0.0, 0.45)}
    region_map = {'cupboard_lower': _bounds(0.45, -0.15, 0.35, 0.85, 0.15, 0.38)}

    object_region_map, descriptions = resolve_object_regions(pose_map, region_map)

    assert object_region_map == {'mug3': 'cupboard_lower'}
    assert descriptions['mug3'] == 'on lower cupboard shelf'


def test_resolve_object_regions_skips_non_region_fixtures() -> None:
    pose_map = {
        'box_lid': (0.0, 0.0, 0.34),
        'grill_lid': (0.0, 0.0, 0.34),
        'mug2': (0.0, 0.0, 0.34),
    }
    region_map = {
        'box_lid_top': _bounds(-0.2, -0.2, 0.30, 0.2, 0.2, 0.32),
    }

    object_region_map, descriptions = resolve_object_regions(pose_map, region_map)

    assert object_region_map == {'mug2': 'box_lid_top'}
    assert descriptions == {'mug2': 'on top of the box lid'}

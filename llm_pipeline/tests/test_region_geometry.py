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


def test_cupboard_lower_beats_cupboard_fallback() -> None:
    region_map = {
        'cupboard_fallback': _bounds(0.4, -0.2, 0.0, 0.9, 0.2, 1.2),
        'cupboard_lower': _bounds(0.45, -0.15, 0.35, 0.85, 0.15, 0.38),
    }

    region, _ = resolve_region((0.6, 0.0, 0.45), region_map)

    assert region == 'cupboard_lower'


def test_groceries_boundary_beats_table() -> None:
    region_map = {
        'table': _bounds(-1, -1, 0.0, 1, 1, 0.05),
        'groceries_boundary': _bounds(0.2, 0.2, 0.05, 0.6, 0.6, 0.06),
    }

    region, _ = resolve_region((0.4, 0.4, 0.12), region_map)

    assert region == 'groceries_boundary'


def test_fallback_used_only_when_primary_absent() -> None:
    fallback_only = {'cupboard_fallback': _bounds(0.4, -0.2, 0.0, 0.9, 0.2, 1.2)}
    with_primary = {
        **fallback_only,
        'cupboard_lower': _bounds(0.45, -0.15, 0.35, 0.85, 0.15, 0.38),
    }

    assert resolve_region((0.6, 0.0, 0.45), fallback_only)[0] == 'cupboard_fallback'
    assert resolve_region((0.6, 0.0, 0.45), with_primary)[0] == 'cupboard_lower'


def test_resolve_object_regions_returns_maps() -> None:
    pose_map = {'mug3': (0.6, 0.0, 0.45)}
    region_map = {'cupboard_lower': _bounds(0.45, -0.15, 0.35, 0.85, 0.15, 0.38)}

    object_region_map, descriptions = resolve_object_regions(pose_map, region_map)

    assert object_region_map == {'mug3': 'cupboard_lower'}
    assert descriptions['mug3'] == 'on lower cupboard shelf'

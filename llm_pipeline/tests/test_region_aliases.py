from llm_pipeline.region_aliases import normalize_region_name, normalize_region_names, scene_object_for_region


def test_kitchen_region_aliases_normalize_to_public_symbols() -> None:
    assert normalize_region_name("box_boundary") == "box_storage"
    assert normalize_region_name("box-top") == "box_lid_top"
    assert normalize_region_name("box-inside") == "box_inside_fallback"
    assert normalize_region_name("cupboard_boundary") == "cupboard_lower"
    assert normalize_region_name("cupboard_boundary_top") == "cupboard_upper"
    assert normalize_region_name("shelf-lower") == "cupboard_lower"


def test_normalize_region_names_deduplicates_aliases() -> None:
    assert normalize_region_names(["box_boundary", "box_storage", "box-top"]) == [
        "box_storage",
        "box_lid_top",
    ]


def test_grill_region_aliases_normalize_to_public_symbols() -> None:
    assert normalize_region_name("grill-top") == "inside_grill"
    assert normalize_region_name("grill_top") == "inside_grill"
    assert normalize_region_name("plate-boundary") == "plate_boundary"
    assert normalize_region_name("prep-area") == "prep_area"
    assert normalize_region_name("prep_area") == "prep_area"
    assert normalize_region_names(["grill-top", "inside_grill", "plate-boundary", "plate_boundary"]) == [
        "inside_grill",
        "plate_boundary",
    ]


def test_grill_regions_map_to_scene_objects() -> None:
    assert scene_object_for_region("inside_grill") == "grill_boundary"
    assert scene_object_for_region("grill-top") == "grill_boundary"
    assert scene_object_for_region("prep_area") == "prep_area"
    assert scene_object_for_region("plate-top") == "plate_boundary"

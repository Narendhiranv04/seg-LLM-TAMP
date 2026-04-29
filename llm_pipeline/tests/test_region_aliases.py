from llm_pipeline.region_aliases import normalize_region_name, normalize_region_names


def test_kitchen_region_aliases_normalize_to_public_symbols() -> None:
    assert normalize_region_name("box_boundary") == "box_storage"
    assert normalize_region_name("box-top") == "box_lid_top"
    assert normalize_region_name("box-inside") == "box_inside_fallback"
    assert normalize_region_name("cupboard_boundary") == "cupboard_lower"
    assert normalize_region_name("cupboard_boundary_top") == "cupboard_upper"
    assert normalize_region_name("shelf-lower") == "cupboard_fallback"


def test_normalize_region_names_deduplicates_aliases() -> None:
    assert normalize_region_names(["box_boundary", "box_storage", "box-top"]) == [
        "box_storage",
        "box_lid_top",
    ]

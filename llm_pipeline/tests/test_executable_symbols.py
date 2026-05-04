from llm_pipeline.executable_symbols import build_runtime_symbol_registry


class GrillEnv:
    grill_lid = object()
    name_to_obj = {
        "steak": object(),
        "steak1": object(),
        "chicken": object(),
        "meat1": object(),
        "meat2": object(),
        "plate": object(),
        "grill_lid": object(),
    }

    regions = {
        "grill-top": object(),
        "inside_grill": object(),
        "prep_area": object(),
        "plate-top": object(),
        "plate_boundary": object(),
        "plate-boundary": object(),
        "dish_rack": object(),
        "placement_boundary": object(),
        "box_boundary": object(),
        "box-top": object(),
    }


def test_grill_symbol_registry_hides_kitchen_region_aliases() -> None:
    registry = build_runtime_symbol_registry(env=GrillEnv())

    assert registry.regions == (
        "prep_area",
        "inside_grill",
        "plate-top",
        "plate_boundary",
        "dish_rack",
    )


def test_grill_symbol_registry_preserves_numbered_meats_without_meat_aliases() -> None:
    registry = build_runtime_symbol_registry(env=GrillEnv())

    assert "steak" in registry.objects
    assert "steak1" in registry.objects
    assert "chicken" in registry.objects
    assert "meat1" not in registry.objects
    assert "meat2" not in registry.objects

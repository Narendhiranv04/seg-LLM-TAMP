from llm_pipeline.executable_symbols import build_runtime_symbol_registry


class GrillEnv:
    grill_lid = object()

    regions = {
        "grill-top": object(),
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
        "grill-top",
        "plate-top",
        "plate_boundary",
        "dish_rack",
    )

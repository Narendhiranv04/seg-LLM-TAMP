from llm_pipeline.executable_symbols import build_runtime_symbol_registry


class FakeEnv:
    def __init__(self):
        self.name_to_obj = {
            'steak': object(),
            'chicken': object(),
            'plate': object(),
        }
        self.regions = {
            'plate-top': object(),
            'grill-top': object(),
            'table': object(),
            'dish_rack': object(),
        }


def test_runtime_symbols_keep_detected_grill_objects() -> None:
    registry = build_runtime_symbol_registry(
        env=FakeEnv(),
        detected_objects={'steak', 'chicken', 'plate', 'grill_lid'},
    )

    assert 'steak' in registry.objects
    assert 'chicken' in registry.objects
    assert 'plate' in registry.objects
    assert 'grill_lid' in registry.objects


def test_runtime_symbols_include_grill_regions_from_env() -> None:
    registry = build_runtime_symbol_registry(env=FakeEnv(), detected_objects={'steak'})

    assert registry.regions == ('table', 'grill-top', 'plate-top', 'dish_rack')


def test_runtime_symbols_fall_back_to_kitchen_defaults_without_env_or_detector() -> None:
    registry = build_runtime_symbol_registry()

    assert 'mug1' in registry.objects
    assert 'box_boundary' in registry.regions

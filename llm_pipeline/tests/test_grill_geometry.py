from llm_pipeline.grill_geometry import derive_grill_semantic_facts, infer_grill_lid_open


class Joint:
    def __init__(self, angle):
        self.angle = angle

    def get_joint_position(self):
        return self.angle


class Env:
    def __init__(self, angle, closed=0.0):
        self.lid_joint = Joint(angle)
        self._closed_lid_angle = closed


def test_grill_semantic_facts_separate_grill_top_from_inside_grill() -> None:
    facts = derive_grill_semantic_facts(
        {
            "steak": "grill-top",
            "chicken": "table",
            "spam": "plate-top",
            "plate": "dish_rack",
        },
        lid_open=True,
    )

    assert "grill_lid_open" in facts
    assert "on_grill_top(steak)" in facts
    assert "on_table(chicken)" in facts
    assert "on_plate(spam)" in facts
    assert "plate_at_dish_rack" in facts
    assert not any(fact.startswith("inside_grill") for fact in facts)


def test_infer_grill_lid_open_from_joint_angle() -> None:
    assert infer_grill_lid_open(Env(0.0)) is False
    assert infer_grill_lid_open(Env(0.5)) is True

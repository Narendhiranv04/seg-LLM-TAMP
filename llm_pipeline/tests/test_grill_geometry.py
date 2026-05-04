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


def test_grill_semantic_facts_treat_grill_top_as_inside_grill() -> None:
    facts = derive_grill_semantic_facts(
        {
            "steak": "grill-top",
            "steak1": "grill-top",
            "chicken": "table",
            "spam": "plate-top",
            "chicken1": "prep_area",
            "plate": "dish_rack",
        },
        lid_open=True,
    )

    assert "grill_lid_open" in facts
    assert "inside_grill(steak)" in facts
    assert "inside_grill(steak1)" in facts
    assert "on_table(chicken)" in facts
    assert "on_plate(spam)" in facts
    assert "in_prep_area(chicken1)" in facts
    assert "plate_at_dish_rack" in facts


def test_infer_grill_lid_open_from_joint_angle() -> None:
    assert infer_grill_lid_open(Env(0.0)) is False
    assert infer_grill_lid_open(Env(0.5)) is True

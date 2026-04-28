from llm_pipeline.strict_parser import StrictActionParser, StrictParseError


parser = StrictActionParser()


def test_parses_direct_ground_truth_actions_with_move() -> None:
    actions = parser.parse(
        'move\n'
        'pick(mug2)\n'
        'place(mug2, placement_boundary)\n'
        'open(box_lid)\n'
    )
    assert [str(action) for action in actions] == [
        'move',
        'pick(mug2)',
        'place(mug2, placement_boundary)',
        'open(box_lid)',
    ]


def test_parses_replan_while_already_holding_object() -> None:
    actions = parser.parse('move\nplace(mug2, placement_boundary)', held_object='mug2')
    assert [str(action) for action in actions] == ['move', 'place(mug2, placement_boundary)']


def test_rejects_numbered_or_bulleted_output() -> None:
    try:
        parser.parse('1. pick(mug2)')
    except StrictParseError as exc:
        assert 'Numbering or bullets are not allowed' in str(exc)
        assert exc.failure_id == 'unknown_action_token'
    else:
        raise AssertionError('Expected numbered output to fail strict parsing')


def test_rejects_orphan_place() -> None:
    try:
        parser.parse('place(mug2, placement_boundary)')
    except StrictParseError as exc:
        assert "Cannot place 'mug2' without first picking it" in str(exc)
        assert exc.failure_id == 'orphan_place'
    else:
        raise AssertionError('Expected orphan place to fail strict parsing')


def test_rejects_pick_place_mismatch() -> None:
    try:
        parser.parse('pick(mug2)\nplace(mug3, placement_boundary)')
    except StrictParseError as exc:
        assert "Cannot place 'mug3' while holding 'mug2'" in str(exc)
        assert exc.failure_id == 'pick_place_mismatch'
    else:
        raise AssertionError('Expected mismatched place to fail strict parsing')


def test_rejects_alias_names_and_missing_terminal_place() -> None:
    try:
        parser.parse('pick(mug_box)')
    except StrictParseError as exc:
        assert "Unknown or unpickable object 'mug_box'" in str(exc)
        assert exc.failure_id == 'unknown_action_token'
    else:
        raise AssertionError('Expected alias object names to fail strict parsing')

    try:
        parser.parse('pick(mug2)')
    except StrictParseError as exc:
        assert "Plan ended while still holding 'mug2'" in str(exc)
        assert exc.failure_id == 'missing_post_pick_place'
    else:
        raise AssertionError('Expected missing post-pick place to fail strict parsing')

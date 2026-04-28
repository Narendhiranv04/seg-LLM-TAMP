from llm_pipeline.metrics import collapse_actions_to_subtasks, parse_action_string, score_variant_completion


def test_open_action_counts_in_completion_metrics() -> None:
    actions = [
        'move',
        'pick(mug2)',
        'place(mug2, placement_boundary)',
        'open(box_lid)',
    ]
    assert parse_action_string('move') == {'action': 'move', 'args': [], 'raw': 'move'}
    assert collapse_actions_to_subtasks(actions) == ['mug_to_placement', 'open_lid']

    completion = score_variant_completion('K1', actions)
    assert completion['completed_gt_subtasks'] == 2
    assert completion['bucket_breakdown']['open_lid']['matched'] == 1

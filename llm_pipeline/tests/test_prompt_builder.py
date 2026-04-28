from llm_pipeline.prompt_builder import TextOnlyContextBuilder
from llm_pipeline.pipeline_types import SegmentationObjectEvidence, SegmentationSnapshot


def _snapshot() -> SegmentationSnapshot:
    return SegmentationSnapshot(
        frame_index=1,
        visible_objects=['mug2', 'box_lid'],
        newly_visible_objects=['mug2'],
        object_evidence={
            'mug2': SegmentationObjectEvidence(
                name='mug2',
                visible=True,
                camera_hits=['overhead'],
                pixel_count=16,
                camera_pixels={'overhead': 16},
                bbox={'overhead': (0.4000, 0.3000, 0.6000, 0.5000)},
                centroid={'overhead': (0.5000, 0.4000)},
                mask_regions=['box_boundary'],
                newly_visible=True,
            ),
            'box_lid': SegmentationObjectEvidence(
                name='box_lid',
                visible=True,
                camera_hits=['overhead'],
                pixel_count=20,
                camera_pixels={'overhead': 20},
                bbox={'overhead': (0.3000, 0.2000, 0.7000, 0.5200)},
                centroid={'overhead': (0.5000, 0.3600)},
                mask_regions=['box_boundary'],
            ),
        },
        gripper_evidence={},
        supported_regions=['table', 'placement_boundary', 'cupboard_boundary', 'cupboard_boundary_top', 'box_boundary'],
        visible_regions=['box_boundary'],
    )


def test_prompt_bundle_stays_text_only() -> None:
    builder = TextOnlyContextBuilder()
    bundle = builder.build_bundle(
        goal_text='Move mug2 to placement_boundary.',
        snapshot=_snapshot(),
        icl_mode='few_shot_shared_1',
        failure_context='failure_id=placement_failed',
        previous_actions=['move', 'pick(mug2)'],
        held_object='mug2',
    )
    assert set(bundle.__dict__.keys()) == {
        'goal_text',
        'observation_text',
        'visible_objects_text',
        'failure_context',
        'icl_mode',
        'previous_actions',
    }
    assert not any('image' in key for key in bundle.__dict__)

    system_prompt = builder.build_system_prompt(bundle)
    user_prompt = builder.build_user_prompt(bundle)

    assert 'SHARED FEW-SHOT EXEMPLAR' in system_prompt
    assert 'pick(mug2)' not in system_prompt
    assert 'CURRENT SEGMENTATION SNAPSHOT:' in user_prompt
    assert 'VISIBLE OBJECT EVIDENCE:' in user_prompt
    assert 'COMPACT SEGMENTATION SUMMARY:' in user_prompt
    assert 'PREVIOUS ACTIONS (already executed, do not repeat):' in user_prompt
    assert 'move' in user_prompt
    assert 'pick(mug2)' in user_prompt
    assert 'FAILURE CONTEXT:' in user_prompt
    assert 'available_actions=move, pick, place, open' in user_prompt
    assert 'Executable action formats for this run:' in user_prompt
    assert 'open(box_lid)' in user_prompt
    assert 'Return executable action lines only.' in user_prompt
    assert 'state_text' not in user_prompt
    assert 'mask_regions=box_boundary' in user_prompt
    assert 'visible_regions=box_boundary' in user_prompt
    assert 'box_lid_state' not in user_prompt
    assert 'region_hint=' not in user_prompt


def test_zero_shot_system_prompt_has_no_shared_exemplar() -> None:
    builder = TextOnlyContextBuilder()
    bundle = builder.build_bundle(
        goal_text='Open the lid.',
        snapshot=_snapshot(),
        icl_mode='zero_shot',
    )
    system_prompt = builder.build_system_prompt(bundle)
    assert 'SHARED FEW-SHOT EXEMPLAR' not in system_prompt
    assert 'Valid action lines:' not in system_prompt
    assert 'Follow the executable action contract given in the user prompt.' in system_prompt
    assert 'mug_box' not in system_prompt

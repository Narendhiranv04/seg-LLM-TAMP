from llm_pipeline.failure_logic import SegmentationFirstFailureChecker
from llm_pipeline.pipeline_types import (
    DirectAction,
    FailureSource,
    FailureStage,
    SegmentationObjectEvidence,
    SegmentationSnapshot,
)


class FakeAdapter:
    def capture_snapshot(self, event=''):
        raise AssertionError('capture_snapshot should not be called in this unit test')

    def is_lid_open(self, snapshot: SegmentationSnapshot) -> bool:
        evidence = snapshot.object_evidence.get('box_lid')
        return evidence is not None and 'box_boundary' not in set(evidence.mask_regions)

    def blocking_objects_for_lid(self, snapshot: SegmentationSnapshot):
        return []


adapter = FakeAdapter()
checker = SegmentationFirstFailureChecker(adapter=adapter, env=None)


def _snapshot(object_evidence, newly_visible=None, visible_regions=None):
    return SegmentationSnapshot(
        frame_index=1,
        visible_objects=sorted([name for name, evidence in object_evidence.items() if evidence.visible]),
        newly_visible_objects=list(newly_visible or []),
        object_evidence=object_evidence,
        gripper_evidence={},
        supported_regions=['table', 'placement_boundary', 'cupboard_boundary', 'cupboard_boundary_top', 'box_boundary'],
        visible_regions=list(visible_regions or []),
    )


def test_precheck_flags_missing_pick_object() -> None:
    snapshot = _snapshot({})
    failure = checker.precheck(DirectAction('pick', ('mug4',)), held_object=None, snapshot=snapshot)
    assert failure is not None
    assert failure.failure_id == 'pick_object_missing'
    assert failure.stage == FailureStage.BEFORE_EXECUTION
    assert failure.source == FailureSource.SEGMENTATION


def test_postcheck_flags_bad_place_region() -> None:
    snapshot = _snapshot(
        {
            'mug2': SegmentationObjectEvidence(name='mug2', visible=True, mask_regions=['cupboard_boundary']),
        },
        visible_regions=['cupboard_boundary'],
    )
    failure = checker.postcheck(
        DirectAction('place', ('mug2', 'placement_boundary')),
        held_object=None,
        snapshot=snapshot,
    )
    assert failure is not None
    assert failure.failure_id == 'placement_failed'
    assert failure.stage == FailureStage.AFTER_EXECUTION


def test_postcheck_triggers_replan_for_new_visibility() -> None:
    snapshot = _snapshot(
        {
            'box_lid': SegmentationObjectEvidence(name='box_lid', visible=True, mask_regions=['placement_boundary']),
            'mug4': SegmentationObjectEvidence(name='mug4', visible=True, mask_regions=['box_boundary']),
        },
        newly_visible=['mug4'],
        visible_regions=['placement_boundary', 'box_boundary'],
    )
    failure = checker.postcheck(DirectAction('open', ('box_lid',)), held_object=None, snapshot=snapshot)
    assert failure is not None
    assert failure.failure_id == 'new_object_discovered'
    assert failure.stage == FailureStage.AFTER_EXECUTION
    assert failure.should_replan is True


def test_move_postcheck_can_trigger_new_visibility_replan() -> None:
    snapshot = _snapshot(
        {
            'mustard': SegmentationObjectEvidence(name='mustard', visible=True, mask_regions=['table']),
        },
        newly_visible=['mustard'],
        visible_regions=['table'],
    )
    failure = checker.postcheck(DirectAction('move', ()), held_object=None, snapshot=snapshot)
    assert failure is not None
    assert failure.failure_id == 'new_object_discovered'


def test_runtime_error_maps_to_pddl_failure() -> None:
    failure = checker.classify_runtime_error(DirectAction('pick', ('mug2',)), 'No PDDL plan found')
    assert failure.failure_id == 'pddl_no_plan'
    assert failure.source == FailureSource.PDDL

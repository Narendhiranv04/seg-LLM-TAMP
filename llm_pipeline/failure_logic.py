"""Segmentation-first failure checks and structured replanning events."""

from __future__ import annotations

from typing import Iterable, Optional

from llm_pipeline.segmentation_adapter import SegmentationEvidenceAdapter
from llm_pipeline.pipeline_types import (
    DirectAction,
    FailureEvent,
    FailureLayer,
    FailureSource,
    FailureStage,
    SegmentationSnapshot,
)
from llm_pipeline.region_aliases import normalize_region_name, regions_match_for_target


LAYER_1_FAILURE_IDS = frozenset({
    'missing_preceding_move',
    'invalid_executor_state',
    'pick_object_missing',
    'lid_missing',
    'geometric_discovery_fail',
    'empty_pick_trajectory',
    'empty_place_trajectory',
    'lid_hover_planning_fail',
    'lid_slide_planning_fail',
    'pddl_no_plan',
    'no_ik_solution',
    'no_motion_plan',
    'no_grasp_found',
    'executor_failure',
    'unsupported_action',
    'unknown_action_token',
    'pick_place_mismatch',
    'orphan_place',
    'missing_post_pick_place',
})

LAYER_2_FAILURE_IDS = frozenset({
    'grasp_failed',
    'object_dropped',
    'object_did_not_move',
    'object_missing_after_place',
    'placement_failed',
    'geometric_placement_failed',
    'lid_not_open_enough',
    'lid_not_closed_enough',
    'new_object_discovered',
})


def failure_layer_for_id(failure_id: str) -> FailureLayer:
    """Return the single logical layer for a known failure id."""
    if failure_id in LAYER_2_FAILURE_IDS:
        return FailureLayer.LAYER_2
    return FailureLayer.LAYER_1


class SegmentationFirstFailureChecker:
    """Produces structured failures from segmentation and low-level runtime signals."""

    def __init__(
        self,
        adapter: SegmentationEvidenceAdapter,
        env=None,
        gripper_threshold: float = 0.4,
        replan_on_new_visibility: bool = True,
    ):
        self.adapter = adapter
        self.env = env
        self.gripper_threshold = gripper_threshold
        self.replan_on_new_visibility = replan_on_new_visibility
        self.discovery_ignore_objects = {'box_lid'}

    def capture_snapshot(self, event: str = '') -> SegmentationSnapshot:
        return self.adapter.capture_snapshot(event=event)

    def precheck(
        self,
        action: DirectAction,
        held_object: Optional[str],
        snapshot: SegmentationSnapshot,
        last_action_name: Optional[str] = None,
    ) -> Optional[FailureEvent]:
        if action.action_name == 'move':
            return None

        if last_action_name != 'move':
            return FailureEvent(
                failure_id='missing_preceding_move',
                stage=FailureStage.BEFORE_EXECUTION,
                source=FailureSource.EXECUTOR,
                action=str(action),
                evidence={'last_action': last_action_name or '(none)', 'expected': 'move'},
                failure_layer=FailureLayer.LAYER_1,
                should_replan=True,
                message=f'Action {action} requires a preceding move to position the arm, but last action was {last_action_name or "(none)"}',
            )

        if action.action_name == 'pick':
            object_name = action.args[0]
            if held_object is not None:
                return FailureEvent(
                    failure_id='invalid_executor_state',
                    stage=FailureStage.BEFORE_EXECUTION,
                    source=FailureSource.EXECUTOR,
                    action=str(action),
                    evidence={'held_object': held_object, 'expected_empty_gripper': True},
                    failure_layer=FailureLayer.LAYER_1,
                    should_replan=False,
                    message=f'Cannot pick {object_name} while already holding {held_object}',
                )

            evidence = snapshot.object_evidence.get(object_name)
            if evidence is None or not evidence.visible:
                return FailureEvent(
                    failure_id='pick_object_missing',
                    stage=FailureStage.BEFORE_EXECUTION,
                    source=FailureSource.SEGMENTATION,
                    action=str(action),
                    evidence={'object_name': object_name, 'visible_objects': snapshot.visible_objects},
                    failure_layer=FailureLayer.LAYER_1,
                    message=f'Cannot pick {object_name} because it is not visible in the segmentation snapshot',
                )
            return None

        if action.action_name == 'place':
            object_name, target_region = action.args
            target_region = normalize_region_name(target_region)
            if held_object != object_name:
                return FailureEvent(
                    failure_id='invalid_executor_state',
                    stage=FailureStage.BEFORE_EXECUTION,
                    source=FailureSource.EXECUTOR,
                    action=str(action),
                    evidence={'held_object': held_object, 'place_object': object_name, 'target_region': target_region},
                    failure_layer=FailureLayer.LAYER_1,
                    should_replan=False,
                    message=f'Cannot place {object_name} while holding {held_object}',
                )
            return None

        if held_object is not None:
            return FailureEvent(
                failure_id='invalid_executor_state',
                stage=FailureStage.BEFORE_EXECUTION,
                source=FailureSource.EXECUTOR,
                action=str(action),
                evidence={'held_object': held_object, 'expected_empty_gripper': True},
                failure_layer=FailureLayer.LAYER_1,
                should_replan=False,
                message=f'Cannot {action.action_name} the lid while holding {held_object}',
            )

        lid_name = action.args[0] if action.args else 'box_lid'
        lid_evidence = snapshot.object_evidence.get(lid_name)
        if lid_evidence is None or not lid_evidence.visible:
            return FailureEvent(
                failure_id='lid_missing',
                stage=FailureStage.BEFORE_EXECUTION,
                source=FailureSource.SEGMENTATION,
                action=str(action),
                evidence={'object_name': lid_name, 'visible_objects': snapshot.visible_objects},
                failure_layer=FailureLayer.LAYER_1,
                message=f'Cannot {action.action_name} the lid because {lid_name} is not visible in the segmentation snapshot',
            )
        return None

    def postcheck(
        self,
        action: DirectAction,
        held_object: Optional[str],
        snapshot: SegmentationSnapshot,
    ) -> Optional[FailureEvent]:
        if action.action_name == 'move':
            return self._maybe_new_visibility_failure(action, snapshot)

        if action.action_name == 'pick':
            object_name = action.args[0]
            evidence = snapshot.object_evidence.get(object_name)
            if self._picked_object_confirmed(evidence):
                return self._maybe_new_visibility_failure(action, snapshot)
            if evidence is None or not evidence.visible:
                return self._maybe_new_visibility_failure(action, snapshot)
            return FailureEvent(
                failure_id='grasp_failed',
                stage=FailureStage.AFTER_EXECUTION,
                source=FailureSource.SEGMENTATION,
                action=str(action),
                evidence=evidence.to_dict(),
                failure_layer=FailureLayer.LAYER_2,
                message=f'{object_name} is not confirmed near the gripper after pick execution',
            )

        if action.action_name == 'place':
            object_name, target_region = action.args
            target_region = normalize_region_name(target_region)
            evidence = snapshot.object_evidence.get(object_name)
            if evidence is None or not evidence.visible:
                return FailureEvent(
                    failure_id='object_dropped',
                    stage=FailureStage.AFTER_EXECUTION,
                    source=FailureSource.SEGMENTATION,
                    action=str(action),
                    evidence={'object_name': object_name, 'target_region': target_region},
                    failure_layer=FailureLayer.LAYER_2,
                    message=f'{object_name} is no longer visible after place execution',
                )

            object_region_map = getattr(snapshot, 'object_region_map', {}) or {}
            observed_region = normalize_region_name(object_region_map.get(object_name))
            if observed_region and regions_match_for_target(observed_region, target_region):
                return self._maybe_new_visibility_failure(action, snapshot)
            return FailureEvent(
                failure_id='placement_failed',
                stage=FailureStage.AFTER_EXECUTION,
                source=FailureSource.GEOMETRY,
                action=str(action),
                evidence={
                    **evidence.to_dict(),
                    'target_region': target_region,
                    'geometric_region': observed_region or None,
                    'object_region_map': dict(object_region_map),
                },
                failure_layer=FailureLayer.LAYER_2,
                message=f'{object_name} is geometrically resolved in {observed_region or "(unresolved)"}, not target region {target_region}',
            )

        lid_name = action.args[0] if action.args else 'box_lid'
        lid_open = self._is_lid_open(snapshot, lid_name)
        if action.action_name == 'open' and not lid_open:
            lid_evidence = snapshot.object_evidence.get(lid_name)
            return FailureEvent(
                failure_id='lid_not_open_enough',
                stage=FailureStage.AFTER_EXECUTION,
                source=FailureSource.SEGMENTATION,
                action=str(action),
                evidence=lid_evidence.to_dict() if lid_evidence is not None else {},
                failure_layer=FailureLayer.LAYER_2,
                message=f'{lid_name} is still observed closed after the open action completed',
            )
        if action.action_name == 'close' and lid_open:
            lid_evidence = snapshot.object_evidence.get(lid_name)
            return FailureEvent(
                failure_id='lid_not_closed_enough',
                stage=FailureStage.AFTER_EXECUTION,
                source=FailureSource.SEGMENTATION,
                action=str(action),
                evidence=lid_evidence.to_dict() if lid_evidence is not None else {},
                failure_layer=FailureLayer.LAYER_2,
                message='The lid is still observed open after the close action completed',
            )
        return self._maybe_new_visibility_failure(action, snapshot)

    def classify_runtime_error(self, action: DirectAction, message: str) -> FailureEvent:
        lowered = (message or '').lower()
        stage = FailureStage.BEFORE_EXECUTION
        layer = FailureLayer.LAYER_1
        if 'not found after placement' in lowered:
            failure_id = 'object_missing_after_place'
            source = FailureSource.VALIDATION
            stage = FailureStage.AFTER_EXECUTION
            layer = FailureLayer.LAYER_2
        elif "didn't move" in lowered or 'did not move' in lowered:
            failure_id = 'object_did_not_move'
            source = FailureSource.VALIDATION
            stage = FailureStage.AFTER_EXECUTION
            layer = FailureLayer.LAYER_2
        elif 'fell' in lowered:
            failure_id = 'object_dropped'
            source = FailureSource.VALIDATION
            stage = FailureStage.AFTER_EXECUTION
            layer = FailureLayer.LAYER_2
        elif 'not in target region' in lowered or 'validation failed' in lowered:
            failure_id = 'placement_failed'
            source = FailureSource.VALIDATION
            stage = FailureStage.AFTER_EXECUTION
            layer = FailureLayer.LAYER_2
        elif "lid didn't slide open enough" in lowered or 'not open enough' in lowered:
            failure_id = 'lid_not_open_enough'
            source = FailureSource.VALIDATION
            stage = FailureStage.AFTER_EXECUTION
            layer = FailureLayer.LAYER_2
        elif 'empty trajectory' in lowered and action.action_name == 'pick':
            failure_id = 'empty_pick_trajectory'
            source = FailureSource.GEOMETRY
        elif 'empty trajectory' in lowered and action.action_name == 'place':
            failure_id = 'empty_place_trajectory'
            source = FailureSource.GEOMETRY
        elif 'motion to hover' in lowered:
            failure_id = 'lid_hover_planning_fail'
            source = FailureSource.GEOMETRY
        elif 'slide trajectory' in lowered:
            failure_id = 'lid_slide_planning_fail'
            source = FailureSource.GEOMETRY
        elif 'no pddl plan' in lowered or 'no solution' in lowered:
            failure_id = 'pddl_no_plan'
            source = FailureSource.PDDL
        elif 'ik' in lowered or 'configuration' in lowered:
            failure_id = 'no_ik_solution'
            source = FailureSource.GEOMETRY
        elif 'motion' in lowered or 'path' in lowered or 'trajectory' in lowered:
            failure_id = 'no_motion_plan'
            source = FailureSource.GEOMETRY
        elif 'grasp' in lowered:
            failure_id = 'no_grasp_found'
            source = FailureSource.GEOMETRY
        else:
            failure_id = 'executor_failure'
            source = FailureSource.EXECUTOR
        return FailureEvent(
            failure_id=failure_id,
            stage=stage,
            source=source,
            action=str(action),
            evidence={'runtime_message': message},
            failure_layer=layer,
            message=message,
        )

    def render_failure_context(
        self,
        failure_event: FailureEvent,
        completed_actions: Optional[Iterable[str]] = None,
        remaining_actions: Optional[Iterable[str]] = None,
    ) -> str:
        def _strip_move_annotation(a: str) -> str:
            """Strip display annotations like move(\u2192pick) → move."""
            if a.startswith('move(') and '\u2192' in a:
                return 'move'
            return a

        layer_value = (
            failure_event.failure_layer.value
            if isinstance(failure_event.failure_layer, FailureLayer)
            else str(failure_event.failure_layer)
        )
        lines = [
            f'failure_id={failure_event.failure_id}',
            f'failure_layer={layer_value}',
            f'stage={failure_event.stage.value}',
            f'action={failure_event.action or "(none)"}',
            f'message={failure_event.message}',
        ]
        # Include truncated raw_output if it's a parsing failure (useful for LLM)
        if failure_event.evidence and 'raw_output' in failure_event.evidence:
            raw = (failure_event.evidence['raw_output'] or '').strip()
            if raw:
                lines.append(f'your_previous_output=\n{raw}')
        if completed_actions is not None:
            joined = ', '.join(
                _strip_move_annotation(str(a)) for a in completed_actions
            ) or '(none)'
            lines.append(f'completed_actions={joined}')
        if remaining_actions is not None:
            joined = ', '.join(
                _strip_move_annotation(str(a)) for a in remaining_actions
            ) or '(none)'
            lines.append(f'remaining_actions={joined}')
        return '\n'.join(lines)

    def _maybe_new_visibility_failure(
        self,
        action: DirectAction,
        snapshot: SegmentationSnapshot,
    ) -> Optional[FailureEvent]:
        if not self.replan_on_new_visibility:
            return None

        action_object = action.args[0] if action.args else None
        discovered = [
            name
            for name in snapshot.newly_visible_objects
            if name not in self.discovery_ignore_objects and name != action_object
        ]
        if not discovered:
            return None

        return FailureEvent(
            failure_id='new_object_discovered',
            stage=FailureStage.AFTER_EXECUTION,
            source=FailureSource.SEGMENTATION,
            action=str(action),
            evidence={
                'newly_visible_objects': discovered,
                'all_newly_visible_objects': list(snapshot.newly_visible_objects),
            },
            failure_layer=FailureLayer.LAYER_2,
            message=f'Newly visible objects require replanning: {", ".join(discovered)}',
        )

    def _picked_object_confirmed(self, evidence) -> bool:
        """Heuristic to check if the object is confirmed in the gripper by mask proximity."""
        if evidence is None or evidence.gripper_proximity is None:
            return False
        return evidence.gripper_proximity <= self.gripper_threshold

    def _is_lid_open(self, snapshot: SegmentationSnapshot, lid_name: str) -> bool:
        try:
            return bool(self.adapter.is_lid_open(snapshot, lid_name=lid_name))
        except TypeError:
            return bool(self.adapter.is_lid_open(snapshot))

from llm_pipeline.geometric_utils import GeometricReasoner

class GeometricFailureChecker(SegmentationFirstFailureChecker):
    """Refined failure checker using 3D geometric reasoning for higher accuracy."""

    def __init__(self, adapter: SegmentationEvidenceAdapter, env=None, gripper_threshold: float = 0.4):
        super().__init__(adapter, env, gripper_threshold)
        self.reasoner = GeometricReasoner()

    def precheck(
        self,
        action: DirectAction,
        held_object: Optional[str],
        snapshot: SegmentationSnapshot,
        last_action_name: Optional[str] = None,
    ) -> Optional[FailureEvent]:
        # Perform base checks first (missing move, etc)
        base_failure = super().precheck(action, held_object, snapshot, last_action_name)
        if base_failure:
            return base_failure

        # Geometric Pick/Place Validation
        detector = getattr(self.adapter, 'detector', None)
        if not detector: return None

        if action.action_name == 'pick':
            obj_name = action.args[0]
            obj_pose = detector.get_object_pose(obj_name)
            if not obj_pose:
                return FailureEvent(
                    failure_id='geometric_discovery_fail',
                    stage=FailureStage.BEFORE_EXECUTION,
                    source=FailureSource.GEOMETRY,
                    action=str(action),
                    evidence={'object_name': obj_name},
                    failure_layer=FailureLayer.LAYER_1,
                    message=f'Cannot pick {obj_name}: 3D pose could not be resolved from current view.'
                )

        if action.action_name == 'place':
            obj_name, region_name = action.args
            region_name = normalize_region_name(region_name)
            region_pose = detector.get_object_pose(region_name)
            if not region_pose:
                 return FailureEvent(
                    failure_id='geometric_discovery_fail',
                    stage=FailureStage.BEFORE_EXECUTION,
                    source=FailureSource.GEOMETRY,
                    action=str(action),
                    evidence={'region_name': region_name},
                    failure_layer=FailureLayer.LAYER_1,
                    message=f'Cannot place in {region_name}: Target region pose could not be resolved.'
                )

        return None

    def postcheck(
        self,
        action: DirectAction,
        held_object: Optional[str],
        snapshot: SegmentationSnapshot,
    ) -> Optional[FailureEvent]:
        # Pick Success Verification (3D Proximity)
        if action.action_name == 'pick':
            obj_name = action.args[0]
            detector = getattr(self.adapter, 'detector', None)
            if detector:
                obj_pose = detector.get_object_pose(obj_name)
                # If object is too far from gripper, it's a grasp failure
                if obj_pose:
                    # Logic here would involve robot flange pos, simplified for now
                    pass

        # Place Success Verification (3D Containment/Accuracy)
        if action.action_name == 'place':
            obj_name, region_name = action.args
            region_name = normalize_region_name(region_name)
            object_region_map = getattr(snapshot, 'object_region_map', {}) or {}
            observed_region = normalize_region_name(object_region_map.get(obj_name))
            if observed_region and regions_match_for_target(observed_region, region_name):
                return super().postcheck(action, held_object, snapshot)

            detector = getattr(self.adapter, 'detector', None)
            if detector:
                obj_pose = detector.get_object_pose(obj_name)
                region_pose = detector.get_object_pose(region_name)
                
                if obj_pose and region_pose:
                    is_contained = self.reasoner.is_contained_3d(obj_pose, region_name, region_pose)
                    if not is_contained:
                        return FailureEvent(
                            failure_id='geometric_placement_failed',
                            stage=FailureStage.AFTER_EXECUTION,
                            source=FailureSource.GEOMETRY,
                            action=str(action),
                            evidence={'is_contained': False, 'obj_pose': obj_pose, 'region': region_name},
                            failure_layer=FailureLayer.LAYER_2,
                            message=f'Geometric Verification: {obj_name} is NOT contained within {region_name} after placement.'
                        )

        return super().postcheck(action, held_object, snapshot)

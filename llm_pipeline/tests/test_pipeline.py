from llm_pipeline.executor import PrimitiveExecutionOutcome
from llm_pipeline.pipeline import LLMPipelineConfig, LLMOnlyReplanningPipeline
from llm_pipeline.planner import MockTextLLMPlanner
from llm_pipeline.strict_parser import StrictActionParser, StrictParseError
from llm_pipeline.pipeline_types import (
    FailureEvent,
    FailureSource,
    FailureStage,
    PlanResult,
    SegmentationObjectEvidence,
    SegmentationSnapshot,
)


class QueuePlanner:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.model_alias = 'mock-llm'
        self.model_name = 'mock-llm'
        self.loaded = True
        self.parser = StrictActionParser()
        self.requests = []

    def load_model(self):
        self.loaded = True
        return True

    def generate_plan(self, system_prompt, user_prompt, icl_mode, max_new_tokens=256, temperature=0.0, held_object=None):
        self.requests.append(
            {
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'icl_mode': icl_mode,
                'held_object': held_object,
            }
        )
        raw_output = self.outputs.pop(0)
        try:
            actions = self.parser.parse(raw_output, held_object=held_object)
            return PlanResult(
                success=True,
                actions=actions,
                raw_output=raw_output,
                inference_time=0.01,
            )
        except StrictParseError as exc:
            return PlanResult(
                success=False,
                actions=[],
                raw_output=raw_output,
                inference_time=0.01,
                error_message=str(exc),
                failure_event=FailureEvent(
                    failure_id=exc.failure_id,
                    stage=FailureStage.BEFORE_EXECUTION,
                    source=FailureSource.VALIDATION,
                    action=None,
                    evidence={'line_number': exc.line_number, 'raw_output': raw_output},
                    should_replan=False,
                    message=str(exc),
                ),
            )

    def get_debug_info(self):
        return {
            'model_alias': self.model_alias,
            'model_name': self.model_name,
            'loaded': self.loaded,
            'last_request': self.requests[-1] if self.requests else {},
        }


class FakeSegmentationAdapter:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.refresh_calls = []
        self.capture_calls = []
        self.live_updates = 0
        self.action_sequence_calls = []
        self.reset_calls = 0
        self.shutdown_called = False
        self.symbol_registry = None

    def set_env(self, env):
        self.env = env

    def set_symbol_registry(self, symbol_registry):
        self.symbol_registry = symbol_registry

    def reset_tracking(self):
        self.reset_calls += 1

    def refresh_visibility(self, event=''):
        self.refresh_calls.append(event)
        return {
            'visible_objects': list(self.snapshot.visible_objects),
            'newly_visible_objects': list(self.snapshot.newly_visible_objects),
            'visible_regions': list(self.snapshot.visible_regions),
        }

    def capture_snapshot(self, event=''):
        self.capture_calls.append(event)
        return self.snapshot

    def update_live_segmentation_view(self):
        self.live_updates += 1
        return set(self.snapshot.visible_objects)

    def set_live_action_sequence(self, actions, current_action_index=None, current_action_label=None):
        self.action_sequence_calls.append(
            {
                'actions': [str(action) for action in actions],
                'current_action_index': current_action_index,
                'current_action_label': current_action_label,
            }
        )

    def shutdown(self):
        self.shutdown_called = True


class FakeFailureChecker:
    def __init__(self, adapter, snapshot):
        self.adapter = adapter
        self.snapshot = snapshot
        self.events = []
        self.render_calls = []
        self.env = None

    def capture_snapshot(self, event=''):
        self.events.append(event)
        return self.snapshot

    def render_failure_context(self, failure_event: FailureEvent, completed_actions=None, remaining_actions=None) -> str:
        self.render_calls.append(
            {
                'failure_event': failure_event,
                'completed_actions': list(completed_actions or []),
                'remaining_actions': list(remaining_actions or []),
            }
        )
        completed = ', '.join(completed_actions or []) or '(none)'
        remaining = ', '.join(remaining_actions or []) or '(none)'
        return '\n'.join(
            [
                f'failure_id={failure_event.failure_id}',
                f'action={failure_event.action}',
                f'message={failure_event.message}',
                f'completed_actions={completed}',
                f'remaining_actions={remaining}',
            ]
        )


class FakeExecutor:
    def __init__(self):
        self.completed_primitive_actions = []
        self.remaining_actions = []
        self.held_object = None
        self.last_failure_event = None
        self.calls = []
        self.step_callback = None
        self.action_start_callback = None

    def set_env(self, env):
        self.env = env

    def set_step_callback(self, callback):
        self.step_callback = callback

    def set_action_start_callback(self, callback):
        self.action_start_callback = callback

    def reset_episode(self):
        self.completed_primitive_actions = []
        self.remaining_actions = []
        self.held_object = None
        self.last_failure_event = None
        self.calls = []

    def execute_actions(self, actions, failure_checker, pre_action_checks_enabled=True, post_action_checks_enabled=True):
        del failure_checker, pre_action_checks_enabled, post_action_checks_enabled
        rendered = [str(action) for action in actions]
        self.calls.append(rendered)
        if self.action_start_callback is not None and actions:
            self.action_start_callback(actions, 0)
        if self.step_callback is not None:
            self.step_callback()

        if len(self.calls) == 1:
            pick_index = next(index for index, action in enumerate(actions) if action.action_name == 'pick')
            self.completed_primitive_actions.extend(rendered[: pick_index + 1])
            self.remaining_actions = rendered[pick_index + 1 :]
            self.held_object = 'mug2'
            failure = FailureEvent(
                failure_id='placement_failed',
                stage=FailureStage.AFTER_EXECUTION,
                source=FailureSource.SEGMENTATION,
                action=self.remaining_actions[-1],
                evidence={'target_region': 'placement_boundary'},
                message='place failed after pick',
            )
            self.last_failure_event = failure
            return PrimitiveExecutionOutcome(
                success=False,
                completed_actions=list(self.completed_primitive_actions),
                remaining_actions=list(self.remaining_actions),
                held_object=self.held_object,
                last_failure_event=failure,
                error_message=failure.message,
            )

        self.completed_primitive_actions.extend(rendered)
        self.remaining_actions = []
        self.held_object = None
        self.last_failure_event = None
        return PrimitiveExecutionOutcome(
            success=True,
            completed_actions=list(self.completed_primitive_actions),
            remaining_actions=[],
            held_object=None,
            last_failure_event=None,
            error_message=None,
        )


class FakeEnv:
    class PR:
        def step(self):
            return None

        def stop(self):
            return None

        def shutdown(self):
            return None

    def __init__(self):
        self.pr = self.PR()
        self.startup_lid_hold_calls = 0

    def get_home_conf(self):
        return [0.0] * 7

    def set_robot_conf(self, conf):
        self.conf = list(conf)

    def hold_startup_lid_pose(self):
        self.startup_lid_hold_calls += 1
        return True


def _snapshot() -> SegmentationSnapshot:
    return SegmentationSnapshot(
        frame_index=1,
        visible_objects=['mug2', 'box_lid'],
        newly_visible_objects=[],
        object_evidence={
            'mug2': SegmentationObjectEvidence(name='mug2', visible=True, mask_regions=['box_storage']),
            'box_lid': SegmentationObjectEvidence(name='box_lid', visible=True, mask_regions=['box_lid_top']),
        },
        gripper_evidence={},
        supported_regions=['table', 'placement_boundary', 'cupboard_lower', 'cupboard_upper', 'box_storage', 'box_lid_top'],
        visible_regions=['box_storage', 'box_lid_top'],
        object_region_map={'mug2': 'box_storage'},
        object_region_descriptions={'mug2': 'inside the box storage target'},
    )


def test_pipeline_replans_with_previous_direct_actions() -> None:
    planner = QueuePlanner(
        [
            'move\npick(mug2)\nmove\nplace(mug2, placement_boundary)',
            'move\nplace(mug2, placement_boundary)\nmove\nopen(box_lid)',
        ]
    )
    snapshot = _snapshot()
    segmentation_adapter = FakeSegmentationAdapter(snapshot)
    failure_checker = FakeFailureChecker(segmentation_adapter, snapshot)
    executor = FakeExecutor()
    pipeline = LLMOnlyReplanningPipeline(
        config=LLMPipelineConfig(model_alias='mock-llm', icl_mode='zero_shot', max_replans=2, live_view_update_stride=1),
        planner=planner,
        segmentation_adapter=segmentation_adapter,
        failure_checker=failure_checker,
        executor=executor,
    )

    assert pipeline.initialize(env=FakeEnv()) is True
    summary = pipeline.run('Move mug2 to placement_boundary and then open the lid.')

    assert summary['success'] is True
    assert summary['total_replans'] == 1
    assert summary['completed_actions'] == [
        'move(→pick)',
        'pick(mug2)',
        'move(→place)',
        'place(mug2, placement_boundary)',
        'move(→open)',
        'open(box_lid)',
    ]
    assert '=== REPLANNING AFTER FAILURE ===' in planner.requests[1]['user_prompt']
    assert 'COMPLETED_ACTIONS: move(→pick), pick(mug2)' in planner.requests[1]['user_prompt']
    assert 'pick(mug2)' in planner.requests[1]['user_prompt']
    assert 'ERROR: place failed after pick' in planner.requests[1]['user_prompt']
    assert '## Object States (Geometric):' in planner.requests[0]['user_prompt']
    assert 'region=box_storage' in planner.requests[0]['user_prompt']
    assert segmentation_adapter.refresh_calls[:2] == ['initial', 'initial']
    assert segmentation_adapter.action_sequence_calls[0]['actions'] == []
    assert segmentation_adapter.live_updates >= 1


def test_initialize_holds_startup_lid_pose_during_settle() -> None:
    planner = QueuePlanner(['move\nopen(box_lid)'])
    snapshot = _snapshot()
    segmentation_adapter = FakeSegmentationAdapter(snapshot)
    env = FakeEnv()
    pipeline = LLMOnlyReplanningPipeline(
        config=LLMPipelineConfig(model_alias='mock-llm', icl_mode='zero_shot'),
        planner=planner,
        segmentation_adapter=segmentation_adapter,
        failure_checker=FakeFailureChecker(segmentation_adapter, snapshot),
        executor=FakeExecutor(),
    )

    assert pipeline.initialize(env=env) is True
    assert env.startup_lid_hold_calls == 60


def test_pipeline_preflight_reports_no_image_input() -> None:
    planner = QueuePlanner(['move\nopen(box_lid)'])
    snapshot = _snapshot()
    segmentation_adapter = FakeSegmentationAdapter(snapshot)
    pipeline = LLMOnlyReplanningPipeline(
        config=LLMPipelineConfig(model_alias='mock-llm', icl_mode='few_shot_shared_1'),
        planner=planner,
        segmentation_adapter=segmentation_adapter,
        failure_checker=FakeFailureChecker(segmentation_adapter, snapshot),
        executor=FakeExecutor(),
    )

    assert pipeline.initialize(env=FakeEnv()) is True
    preflight = pipeline.preflight('Open the lid.')

    assert preflight['loaded'] is True
    assert preflight['image_present'] is False
    assert preflight['prompt_contract_ok'] is True
    assert preflight['dry_run_failure_event'] is None
    assert 'image' not in ''.join(preflight['prompt_trace']['bundle'].keys())


def test_pipeline_reports_validation_failure_before_execution() -> None:
    planner = MockTextLLMPlanner(scripted_output='1. pick(mug_box)')
    snapshot = _snapshot()
    segmentation_adapter = FakeSegmentationAdapter(snapshot)
    pipeline = LLMOnlyReplanningPipeline(
        config=LLMPipelineConfig(model_alias='mock-llm', icl_mode='zero_shot', max_replans=0),
        planner=planner,
        segmentation_adapter=segmentation_adapter,
        failure_checker=FakeFailureChecker(segmentation_adapter, snapshot),
        executor=FakeExecutor(),
    )

    assert pipeline.initialize(env=FakeEnv()) is True
    summary = pipeline.run('Move mug2 to placement_boundary.')

    assert summary['success'] is False
    assert summary['last_failure_event']['failure_id'] == 'unknown_action_token'
    assert summary['last_failure_event']['stage'] == 'before_execution'
    assert summary['last_failure_event']['source'] == 'validation'


def test_pipeline_plan_only_mode_skips_execution_and_failure_checks() -> None:
    planner = QueuePlanner(['move\npick(mug2)\nmove\nplace(mug2, placement_boundary)'])
    snapshot = _snapshot()
    segmentation_adapter = FakeSegmentationAdapter(snapshot)
    failure_checker = FakeFailureChecker(segmentation_adapter, snapshot)
    executor = FakeExecutor()
    pipeline = LLMOnlyReplanningPipeline(
        config=LLMPipelineConfig(
            model_alias='mock-llm',
            icl_mode='zero_shot',
            max_replans=2,
            enable_replanning=False,
            pre_action_checks_enabled=False,
            post_action_checks_enabled=False,
            live_view_update_stride=1,
        ),
        planner=planner,
        segmentation_adapter=segmentation_adapter,
        failure_checker=failure_checker,
        executor=executor,
    )

    assert pipeline.initialize(env=FakeEnv()) is True
    summary = pipeline.run('Move mug2 to placement_boundary.')

    assert summary['success'] is True
    assert summary['replan_mode'] == 'off'
    assert summary['replanning_enabled'] is False
    assert summary['execution_skipped'] is True
    assert summary['pre_action_checks_enabled'] is False
    assert summary['post_action_checks_enabled'] is False
    assert summary['planned_actions'] == [
        'move(→pick)',
        'pick(mug2)',
        'move(→place)',
        'place(mug2, placement_boundary)',
    ]
    assert summary['completed_actions'] == []
    assert summary['remaining_actions'] == [
        'move(→pick)',
        'pick(mug2)',
        'move(→place)',
        'place(mug2, placement_boundary)',
    ]
    assert summary['total_cycles'] == 1
    assert summary['total_replans'] == 0
    assert executor.calls == []

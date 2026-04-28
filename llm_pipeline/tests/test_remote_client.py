from llm_pipeline import client as client_module
from llm_pipeline.client import RemoteTextLLMPlanner


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=''):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _FakeRequests:
    def __init__(self, health_payload, plan_payload):
        self.health_payload = health_payload
        self.plan_payload = plan_payload
        self.get_calls = []
        self.post_calls = []

    def get(self, url, timeout=10):
        self.get_calls.append((url, timeout))
        if url.endswith('/health'):
            return _FakeResponse(payload=self.health_payload)
        if url.endswith('/debug/last-request'):
            return _FakeResponse(payload={'last_request': {'url': url}})
        return _FakeResponse(status_code=404, text='not found')

    def post(self, url, json=None, timeout=300):
        self.post_calls.append((url, json, timeout))
        if url.endswith('/plan'):
            return _FakeResponse(payload=self.plan_payload)
        return _FakeResponse(status_code=404, text='not found')


def test_remote_planner_loads_from_server_health() -> None:
    fake_requests = _FakeRequests(
        health_payload={
            'status': 'ok',
            'model_loaded': True,
            'model_alias': 'qwen',
            'model_name': 'Qwen/Qwen2.5-7B-Instruct',
            'model_type': 'llm',
            'prompt_mode': 'text_visible',
            'gpu_available': True,
        },
        plan_payload={},
    )
    old_requests = client_module.requests
    old_has_requests = client_module.HAS_REQUESTS
    client_module.requests = fake_requests
    client_module.HAS_REQUESTS = True
    try:
        planner = RemoteTextLLMPlanner(server_url='http://planner-box:8000')
        assert planner.load_model() is True
        assert planner.loaded is True
        assert planner.model_alias == 'qwen'
        assert planner.model_name == 'Qwen/Qwen2.5-7B-Instruct'
        assert fake_requests.get_calls[0][0] == 'http://planner-box:8000/health'
    finally:
        client_module.requests = old_requests
        client_module.HAS_REQUESTS = old_has_requests


def test_remote_planner_returns_server_actions() -> None:
    fake_requests = _FakeRequests(
        health_payload={'status': 'ok', 'model_loaded': True, 'model_alias': 'qwen', 'model_name': 'qwen', 'model_type': 'llm', 'prompt_mode': 'text_visible', 'gpu_available': True},
        plan_payload={
            'success': True,
            'actions': [
                {'action_name': 'move', 'args': []},
                {'action_name': 'pick', 'args': ['mug2']},
                {'action_name': 'place', 'args': ['mug2', 'placement_boundary']},
            ],
            'raw_output': 'move\npick(mug2)\nplace(mug2, placement_boundary)',
            'inference_time': 0.12,
            'error_message': None,
            'failure_event': None,
        },
    )
    old_requests = client_module.requests
    old_has_requests = client_module.HAS_REQUESTS
    client_module.requests = fake_requests
    client_module.HAS_REQUESTS = True
    try:
        planner = RemoteTextLLMPlanner(server_url='http://planner-box:8000')
        result = planner.generate_plan(
            system_prompt='system',
            user_prompt='user',
            icl_mode='zero_shot',
        )
        assert result.success is True
        assert [str(action) for action in result.actions] == [
            'move',
            'pick(mug2)',
            'place(mug2, placement_boundary)',
        ]
        assert fake_requests.post_calls[0][0] == 'http://planner-box:8000/plan'
        assert fake_requests.post_calls[0][1]['icl_mode'] == 'zero_shot'
    finally:
        client_module.requests = old_requests
        client_module.HAS_REQUESTS = old_has_requests


def test_remote_planner_falls_back_to_local_parse_when_server_returns_raw_output() -> None:
    fake_requests = _FakeRequests(
        health_payload={'status': 'ok', 'model_loaded': True, 'model_alias': 'qwen', 'model_name': 'qwen', 'model_type': 'llm', 'prompt_mode': 'text_visible', 'gpu_available': True},
        plan_payload={
            'success': False,
            'actions': [],
            'raw_output': 'move\npick(mug2)\nplace(mug2, placement_boundary)',
            'inference_time': 0.08,
            'error_message': None,
            'failure_event': None,
        },
    )
    old_requests = client_module.requests
    old_has_requests = client_module.HAS_REQUESTS
    client_module.requests = fake_requests
    client_module.HAS_REQUESTS = True
    try:
        planner = RemoteTextLLMPlanner(server_url='http://planner-box:8000')
        result = planner.generate_plan(
            system_prompt='system',
            user_prompt='user',
            icl_mode='few_shot_shared_1',
        )
        assert result.success is True
        assert [str(action) for action in result.actions] == [
            'move',
            'pick(mug2)',
            'place(mug2, placement_boundary)',
        ]
    finally:
        client_module.requests = old_requests
        client_module.HAS_REQUESTS = old_has_requests

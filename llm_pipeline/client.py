"""Remote client for the maintained text-only LLM planner server."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:  # pragma: no cover
    HAS_REQUESTS = False
    requests = None

from llm_pipeline.strict_parser import StrictActionParser, StrictParseError
from llm_pipeline.pipeline_types import FailureEvent, FailureSource, FailureStage, PlanResult


class RemoteTextLLMPlanner:
    """Drop-in remote planner for the maintained llm_pipeline."""

    def __init__(
        self,
        server_url: Optional[str] = None,
        request_timeout_s: Optional[float] = None,
        expected_model=None,
    ):
        if not HAS_REQUESTS:
            raise ImportError('requests library required for remote planner mode')

        url = (
            server_url
            or os.environ.get('LLM_SERVER_URL')
            or os.environ.get('VLM_SERVER_URL')
            or 'http://localhost:8000'
        )
        if url:
            if not url.startswith(('http://', 'https://')):
                url = f'http://{url}'
            # Add default port if it looks like an IP/hostname without a port
            # e.g. http://10.4.25.26 -> http://10.4.25.26:8000
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if not parsed.port and parsed.hostname and parsed.hostname != 'localhost':
                 url = f"{url.rstrip('/')}:8000"
        
        self.server_url = url
        timeout_env = os.environ.get('LLM_REQUEST_TIMEOUT_S', '').strip()
        if request_timeout_s is not None:
            self.request_timeout_s = float(request_timeout_s)
        elif timeout_env:
            self.request_timeout_s = float(timeout_env)
        else:
            self.request_timeout_s = 300.0

        self.expected_model = expected_model
        self.loaded = False
        self.model_alias = getattr(expected_model, 'alias', 'remote-llm')
        self.model_name = getattr(expected_model, 'path', 'remote-llm')
        self.parser = StrictActionParser()
        self.server_model_info: Dict[str, Any] = {}
        self.last_request_summary: Dict[str, Any] = {}

    def load_model(self) -> bool:
        try:
            response = requests.get(f'{self.server_url}/health', timeout=10)
            if response.status_code != 200:
                return False
            data = response.json()
            self.loaded = bool(data.get('model_loaded', False))
            self.server_model_info = data
            self.model_alias = data.get('model_alias', self.model_alias)
            self.model_name = data.get('model_name', self.model_name)
            return self.loaded
        except Exception:
            return False

    def plan(self, bundle: Any) -> PlanResult:
        """Unified interface that handles both text and multimodal bundles."""
        return self.generate_plan(
            system_prompt=bundle.system_prompt,
            user_prompt=bundle.user_prompt,
            icl_mode=bundle.icl_mode,
            held_object=bundle.state.gripper_state.get('holding') if hasattr(bundle, 'state') else None,
            bundle=bundle # Pass bundle for image extraction
        )

    def generate_plan(
        self,
        system_prompt: str,
        user_prompt: str,
        icl_mode: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        held_object: Optional[str] = None,
        bundle: Optional[Any] = None,
    ) -> PlanResult:
        started_at = time.time()
        # Support for multimodal images if present in bundle
        image_b64 = getattr(bundle, 'image_base64', None) if hasattr(bundle, 'image_base64') else None
        if not image_b64 and 'image_base64' in getattr(bundle, '__dict__', {}):
             image_b64 = bundle.__dict__['image_base64']

        request_data = {
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'goal': user_prompt, # Alias
            'icl_mode': icl_mode,
            'max_new_tokens': int(max_new_tokens),
            'temperature': float(temperature),
            'held_object': held_object,
            'use_vision': bool(image_b64 is not None),
            'image_base64': image_b64,
        }

        try:
            response = requests.post(
                f'{self.server_url}/plan',
                json=request_data,
                timeout=self.request_timeout_s,
            )
            if response.status_code != 200:
                return PlanResult(
                    success=False,
                    actions=[],
                    raw_output='',
                    inference_time=time.time() - started_at,
                    error_message=f'Server error: {response.status_code} - {response.text}',
                )

            data = response.json()
            raw_output = data.get('raw_output', '')
            failure_event = self._decode_failure_event(data.get('failure_event'))
            actions = []

            action_lines = [self._action_line(item) for item in data.get('actions', [])]
            if action_lines:
                try:
                    actions = self.parser.parse('\n'.join(action_lines), held_object=held_object)
                except StrictParseError as exc:
                    failure_event = FailureEvent(
                        failure_id=exc.failure_id,
                        stage=FailureStage.BEFORE_EXECUTION,
                        source=FailureSource.VALIDATION,
                        action=None,
                        evidence={'line_number': exc.line_number, 'raw_output': '\n'.join(action_lines)},
                        should_replan=(exc.failure_id == 'missing_preceding_move'),
                        message=str(exc),
                    )

            if not actions and raw_output.strip():
                try:
                    actions = self.parser.parse(raw_output, held_object=held_object)
                except StrictParseError as exc:
                    failure_event = FailureEvent(
                        failure_id=exc.failure_id,
                        stage=FailureStage.BEFORE_EXECUTION,
                        source=FailureSource.VALIDATION,
                        action=None,
                        evidence={'line_number': exc.line_number, 'raw_output': raw_output},
                        should_replan=(exc.failure_id == 'missing_preceding_move'),
                        message=str(exc),
                    )

            return PlanResult(
                success=bool(data.get('success', False) or actions),
                actions=actions,
                raw_output=raw_output,
                inference_time=float(data.get('inference_time', time.time() - started_at)),
                error_message=data.get('error_message'),
                failure_event=failure_event,
            )
        except Exception as exc:
            return PlanResult(
                success=False,
                actions=[],
                raw_output='',
                inference_time=time.time() - started_at,
                error_message=str(exc),
            )

    def get_debug_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            'model_alias': self.model_alias,
            'model_name': self.model_name,
            'model_type': 'llm',
            'loaded': self.loaded,
            'server_url': self.server_url,
            'health': dict(self.server_model_info),
            'last_request': dict(self.last_request_summary),
        }
        try:
            health = requests.get(f'{self.server_url}/health', timeout=10)
            if health.status_code == 200:
                info['health'] = health.json()
        except Exception as exc:
            info['health_error'] = str(exc)

        try:
            debug = requests.get(f'{self.server_url}/debug/last-request', timeout=10)
            if debug.status_code == 200:
                payload = debug.json()
                info['last_request'] = payload.get('last_request', {})
                self.last_request_summary = info['last_request']
        except Exception as exc:
            info['last_request_error'] = str(exc)
        return info

    def _action_line(self, item: Dict[str, Any]) -> str:
        action_name = item.get('action_name', '')
        args = list(item.get('args', []))
        if not args:
            return action_name
        return f"{action_name}({', '.join(args)})"

    def _decode_failure_event(self, payload: Optional[Dict[str, Any]]) -> Optional[FailureEvent]:
        if not payload:
            return None
        try:
            return FailureEvent(
                failure_id=payload['failure_id'],
                stage=FailureStage(payload['stage']),
                source=FailureSource(payload['source']),
                action=payload.get('action'),
                evidence=dict(payload.get('evidence', {})),
                should_replan=bool(payload.get('should_replan', True)),
                message=payload.get('message', ''),
            )
        except Exception:
            return None


def test_connection(server_url: Optional[str] = None) -> bool:
    """Test connection to the maintained remote LLM planner server."""
    url = server_url or os.environ.get('LLM_SERVER_URL') or os.environ.get('VLM_SERVER_URL') or 'http://localhost:8000'

    print('=' * 60)
    print('TESTING LLM PLANNER SERVER CONNECTION')
    print('=' * 60)
    print(f'Server URL: {url}')

    try:
        response = requests.get(f'{url}/health', timeout=10)
        print(f'Status Code: {response.status_code}')
        if response.status_code == 200:
            data = response.json()
            print(f"Server Status: {data.get('status')}")
            print(f"Model Loaded: {data.get('model_loaded')}")
            print(f"Model Alias: {data.get('model_alias')}")
            print(f"Model Name: {data.get('model_name')}")
            print(f"Model Type: {data.get('model_type')}")
            print(f"Prompt Mode: {data.get('prompt_mode')}")
            print(f"GPU Available: {data.get('gpu_available')}")
            print('=' * 60)
            print('Connection successful')
            return True
        print(f'Server returned error: {response.text}')
        return False
    except Exception as exc:
        print('Connection failed')
        print(f'Error: {exc}')
        print('Make sure the server is running: python -m llm_pipeline.server')
        return False


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description='Test the maintained remote LLM planner server connection')
    parser.add_argument('--url', type=str, default='', help='Server URL to test')
    args = parser.parse_args()
    test_connection(args.url or None)


if __name__ == '__main__':
    main()

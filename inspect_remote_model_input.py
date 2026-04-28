#!/usr/bin/env python3
"""
Inspect the remote planner server from the laptop.
Shows loaded model metadata plus the last request summary seen by the server.
"""

import argparse
import json
import time
from typing import Any, Dict

import requests



def fetch_json(url: str) -> Dict[str, Any]:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()



def print_snapshot(server_url: str):
    health = fetch_json(f"{server_url}/health")
    debug = fetch_json(f"{server_url}/debug/last-request")
    last_request = debug.get("last_request", {}) or {}

    print("=" * 80)
    print(f"Server URL   : {server_url}")
    print(f"Status       : {health.get('status')}")
    print(f"Model loaded : {health.get('model_loaded')}")
    print(f"Model alias  : {health.get('model_alias')}")
    print(f"Model path   : {health.get('model_name')}")
    print(f"Model type   : {health.get('model_type')}")
    print(f"Prompt mode  : {health.get('prompt_mode')}")
    print(f"GPU avail    : {health.get('gpu_available')}")
    print("-" * 80)

    if not last_request:
        print("No planner request has reached the server yet.")
        print("=" * 80)
        return

    print(f"Last request timestamp  : {last_request.get('timestamp')}")
    print(f"Request used vision     : {last_request.get('use_vision')}")
    print(f"Image present           : {last_request.get('image_present')}")
    print(f"Image shape             : {last_request.get('image_shape')}")
    print(f"System prompt length    : {last_request.get('system_prompt_length')}")
    print(f"User prompt length      : {last_request.get('user_prompt_length')}")
    print("-" * 80)
    print("SYSTEM PROMPT")
    print("-" * 80)
    print(last_request.get('system_prompt', ''))
    print("-" * 80)
    print("USER PROMPT")
    print("-" * 80)
    print(last_request.get('user_prompt', ''))
    print("=" * 80)



def main():
    parser = argparse.ArgumentParser(description="Inspect remote planner model + last input")
    parser.add_argument("--url", type=str, default="http://localhost:8080", help="Planner server URL")
    parser.add_argument("--watch", action="store_true", help="Poll continuously")
    parser.add_argument("--interval", type=float, default=3.0, help="Polling interval in seconds")
    parser.add_argument("--json", action="store_true", help="Print combined health/debug info as JSON")
    args = parser.parse_args()

    def snapshot_dict() -> Dict[str, Any]:
        return {
            "health": fetch_json(f"{args.url}/health"),
            "debug": fetch_json(f"{args.url}/debug/last-request"),
        }

    if args.watch:
        while True:
            try:
                if args.json:
                    print(json.dumps(snapshot_dict(), indent=2))
                else:
                    print_snapshot(args.url)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"Error: {exc}")
            time.sleep(max(0.5, args.interval))
    else:
        if args.json:
            print(json.dumps(snapshot_dict(), indent=2))
        else:
            print_snapshot(args.url)


if __name__ == "__main__":
    main()

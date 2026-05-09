#!/usr/bin/env python3
"""Run only the grill plate pick/place sequence for focused debugging."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
ORCHESTRATOR_PATH = THIS_DIR / "ground_truth_orchestrator_variation1 copy.py"
SCENE_BY_VARIANT = {
    "G1": THIS_DIR / "grill.variation1.ttt",
    "G2": THIS_DIR / "grill.variation2.ttt",
    "G3": THIS_DIR / "grill.variation3.ttt",
}


def _load_orchestrator(scene_path: Path, headless: bool):
    os.environ["GRILL_ALLOW_SCENE_OVERRIDE"] = "True"
    os.environ["GRILL_SCENE_FILE_OVERRIDE"] = str(scene_path)
    os.environ["HEADLESS"] = "True" if headless else "False"

    spec = importlib.util.spec_from_file_location("grill_gt_orchestrator", ORCHESTRATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load orchestrator from {ORCHESTRATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _safe_call(label, fn, default=None):
    try:
        return fn()
    except Exception as exc:
        return {"error": f"{label}: {exc}"}


def _pose_report(obj):
    if obj is None:
        return None
    return {
        "name": _safe_call("get_name", obj.get_name),
        "pose": _safe_call("get_pose", obj.get_pose),
        "position": _safe_call("get_position", obj.get_position),
        "quaternion": _safe_call("get_quaternion", obj.get_quaternion),
        "world_bounds": None,
    }


def _region_report(gt, env, region_name):
    region = gt._region_object(env, region_name)
    if region is None:
        return {"name": region_name, "found": False}
    return {
        "name": region_name,
        "found": True,
        "object_name": _safe_call("get_name", region.get_name),
        "position": _safe_call("get_position", region.get_position),
        "world_bounds": _safe_call("world_bounds", lambda: gt._get_world_bounds(env, region)),
    }


def _plate_report(gt, env, plate_obj, region_name):
    report = _pose_report(plate_obj)
    if report is None:
        return None
    report["world_bounds"] = _safe_call("world_bounds", lambda: gt._get_world_bounds(env, plate_obj))
    report["in_region"] = _safe_call(
        "in_region",
        lambda: gt._is_in_region(env, plate_obj, region_name, tol_xy=0.04, tol_z=0.06),
    )
    report["is_grasped"] = _safe_call("is_grasped", lambda: gt._target_is_grasped(env, plate_obj))
    return report


def _shutdown_env(env):
    pr = getattr(env, "pr", None)
    if pr is None:
        return
    try:
        pr.stop()
    except Exception:
        pass
    try:
        pr.shutdown()
    except Exception:
        pass


def _run_child(args):
    scene_path = SCENE_BY_VARIANT[args.variant]
    gt = _load_orchestrator(scene_path=scene_path, headless=not args.gui)
    env = gt.ENV
    pr = env.pr
    start = datetime.now()
    report = {
        "created_at": start.isoformat(timespec="seconds"),
        "variant_id": args.variant,
        "scene_path": str(scene_path),
        "trial_index": args.trial_index,
        "task": "plate -> plate_boundary",
        "success": False,
        "failure_reason": None,
        "target_region": "plate_boundary",
        "region": None,
        "target_pose": None,
        "plate_before": None,
        "plate_after": None,
    }

    try:
        plate = gt._discover_plate(env)
        if plate is None:
            report["failure_reason"] = "plate not found"
            return report

        plate_obj = plate["obj"]
        report["plate_name"] = plate.get("name")
        report["region"] = _region_report(gt, env, "plate_boundary")
        report["plate_before"] = _plate_report(gt, env, plate_obj, "plate_boundary")

        target_pose = gt._region_slot_pose(env, plate_obj, "plate_boundary", slot_idx=0, slot_count=1)
        report["target_pose"] = list(target_pose)

        ok = gt.run_pick_place_framework(
            env,
            pr,
            obj_name=plate["name"],
            target_region="plate_boundary",
            task_name="Debug: Plate -> plate_boundary",
            is_plate=True,
            target_pose=target_pose,
        )
        report["success"] = bool(ok)
        try:
            gt.go_home(env, pr)
        except Exception:
            pass
        report["plate_after"] = _plate_report(gt, env, plate_obj, "plate_boundary")
        if not ok:
            report["failure_reason"] = "plate transfer returned false"
    except Exception as exc:
        report["failure_reason"] = str(exc)
    finally:
        report["execution_time_s"] = (datetime.now() - start).total_seconds()
        _shutdown_env(env)

    return report


def _child_main(args):
    report = _run_child(args)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("success") else 1


def _run_trial(args, output_dir: Path, trial_index: int):
    variant_dir = output_dir / "trials" / args.variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    report_path = variant_dir / f"{args.variant}_plate_trial_{trial_index:03d}.json"
    stdout_path = variant_dir / f"{args.variant}_plate_trial_{trial_index:03d}.stdout.log"
    stderr_path = variant_dir / f"{args.variant}_plate_trial_{trial_index:03d}.stderr.log"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_child",
        "--variant",
        args.variant,
        "--trial-index",
        str(trial_index),
        "--report",
        str(report_path),
    ]
    if args.gui:
        cmd.append("--gui")

    env = os.environ.copy()
    env["GRILL_ALLOW_SCENE_OVERRIDE"] = "True"
    env["GRILL_SCENE_FILE_OVERRIDE"] = str(SCENE_BY_VARIANT[args.variant])
    env["HEADLESS"] = "False" if args.gui else "True"

    result = subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_path.write_text(result.stdout or "")
    stderr_path.write_text(result.stderr or "")

    if report_path.exists():
        report = json.loads(report_path.read_text())
    else:
        report = {
            "variant_id": args.variant,
            "trial_index": trial_index,
            "success": False,
            "failure_reason": f"child_returncode:{result.returncode}",
        }
        report_path.write_text(json.dumps(report, indent=2))

    report["subprocess_returncode"] = int(result.returncode)
    report["stdout_log_path"] = str(stdout_path)
    report["stderr_log_path"] = str(stderr_path)
    report_path.write_text(json.dumps(report, indent=2, default=str))
    return report


def _default_output_dir():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT_DIR / "outputs" / "plate_sequence" / stamp


def main():
    parser = argparse.ArgumentParser(description="Debug only grill plate pick/place.")
    parser.add_argument("--variant", choices=sorted(SCENE_BY_VARIANT), default="G1")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--trial-index", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--report", type=str, default="")
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._child:
        return _child_main(args)

    output_dir = args.output_dir or _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for idx in range(1, max(1, args.trials) + 1):
        print(f"[plate-debug] {args.variant} trial {idx}/{args.trials}")
        records.append(_run_trial(args, output_dir, idx))

    success_count = sum(1 for record in records if record.get("success"))
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "variant_id": args.variant,
        "trials": len(records),
        "success_count": success_count,
        "success_rate": float(success_count / len(records)) if records else 0.0,
        "records": records,
    }
    summary_path = output_dir / "plate_sequence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[plate-debug] wrote {summary_path}")
    return 0 if success_count == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

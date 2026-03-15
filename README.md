# seg-LLM-TAMP

This repository is a full snapshot of the current workspace around segmentation-aware, VLM-assisted task and motion planning on RLBench/CoppeliaSim scenes.

The active research code is mainly under `TAMP-PDDL/`. The `safe_publish_work/` directory is kept here as an additional archived workspace copy.

## Repository Layout

- `TAMP-PDDL/`: main codebase for RLBench + PDDLStream + segmentation + VLM planning.
- `TAMP-PDDL/variation_1_easy/`: easy variation ground-truth and live-segmentation runners.
- `TAMP-PDDL/variation_2/`: variation 2 ground-truth and live-segmentation runners.
- `TAMP-PDDL/variation_3_hard/`: hard variation ground-truth and live-segmentation runners.
- `TAMP-PDDL/vlm_pipeline/`: modular VLM pipeline, planner, executor, prompts, and demos.
- `TAMP-PDDL/experiments/`: experiment scripts for pure PDDL, COAST, and VLM evaluation.
- `TAMP-PDDL/RLBench/`: local RLBench checkout used by this workspace.
- `TAMP-PDDL/pddlstream/`: local PDDLStream checkout and planner binaries.
- `safe_publish_work/`: archived companion workspace snapshot.

## Clone Notes

This repo contains large videos, binaries, simulator assets, and a local virtual environment snapshot.

If you clone this repo on another machine, use Git LFS:

```bash
git lfs install
git clone <repo-url>
cd seg-LLM-TAMP
git lfs pull
```

## Environment Setup

### 1. CoppeliaSim

The RLBench stack in this workspace expects CoppeliaSim to be installed locally.

Typical environment variables:

```bash
export COPPELIASIM_ROOT=$HOME/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Several scripts auto-detect:

- `$COPPELIASIM_ROOT/platforms`
- `$COPPELIASIM_ROOT/Qt/plugins/platforms`

### 2. Python Environment

If you are on the original machine, there is already a local environment snapshot at:

```bash
TAMP-PDDL/.venv
```

To reuse it:

```bash
source TAMP-PDDL/.venv/bin/activate
```

If you want a cleaner install on a fresh machine, create a new Python environment and install the main dependencies used across the codebase:

```bash
pip install numpy pillow torch transformers accelerate bitsandbytes qwen-vl-utils
pip install pyrep
pip install -e TAMP-PDDL/RLBench
```

`TAMP-PDDL/pddlstream/` is vendored in this repo and is imported directly from the project tree by many scripts.

## How To Run

Most commands below assume:

```bash
cd TAMP-PDDL
```

### Main Ground-Truth Runs

Base orchestrator:

```bash
python orchestrator.py
```

Segmentation-aware orchestrator:

```bash
python run_segmentation_orchestrator.py
```

Run with real RLBench segmentation masks:

```bash
python run_with_segmentation.py
```

Live segmentation viewer:

```bash
python run_live_segmentation.py
```

## Variation Runs

Easy variation:

```bash
python variation_1_easy/ground_truth_orchestrator_variation1_easy.py
python variation_1_easy/run_live_segmentation_variation1_easy.py
```

Variation 2:

```bash
python variation_2/ground_truth_orchestrator_variation2.py
python variation_2/run_live_segmentation_variation2.py
```

Hard variation:

```bash
python variation_3_hard/ground_truth_orchestrator_variation3_hard.py
python variation_3_hard/run_live_segmentation_variation3_hard.py
```

If needed, switch the live viewer backend:

```bash
LIVE_SEG_VIEWER_BACKEND=tkinter python variation_1_easy/run_live_segmentation_variation1_easy.py
```

## Baselines And Experiments

Baseline shell runner:

```bash
./run_baseline.sh
```

Repeated baseline experiment runner:

```bash
./run_baseline_experiment.sh
```

Pure PDDL experiment:

```bash
python experiments/exp1_pure_pddl.py --object soup --episodes 1
```

VLM planner benchmark:

```bash
python experiments/exp4_vlm_planner.py --mock --trials 5
```

## VLM Pipeline

List built-in goals:

```bash
python -m vlm_pipeline.vlm_main --list-goals
```

Mock VLM, no RLBench environment:

```bash
python -m vlm_pipeline.vlm_main --mock --no-env --goal full_task
```

Real environment run:

```bash
python -m vlm_pipeline.vlm_main --goal full_task
```

Live discovery-triggered replanning:

```bash
python run_vlm_discovery_replan_live.py --mock
```

## Useful Files

- `TAMP-PDDL/Failure_checks_list.txt`: failure/debug checklist.
- `TAMP-PDDL/constraints_and_helpers.txt`: planning constraints and helper notes.
- `TAMP-PDDL/vlm_pipeline/README.md`: detailed VLM pipeline notes.
- `TAMP-PDDL/variation_1_easy/README.md`: easy variation notes.
- `TAMP-PDDL/variation_2/README.md`: variation 2 notes.
- `TAMP-PDDL/variation_3_hard/README.md`: hard variation notes.

## Practical Notes

- Many scripts assume GUI rendering unless `--headless` is supported explicitly.
- Large media outputs are already present under `orchestrator_videos/`, `segmentation_videos*`, and related folders.
- If Qt plugin errors appear, re-check `COPPELIASIM_ROOT` and `QT_QPA_PLATFORM_PLUGIN_PATH`.
- Some scripts append `pddlstream/` directly to `sys.path`, so running from inside `TAMP-PDDL/` is the safest default.

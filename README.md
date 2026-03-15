# seg-LLM-TAMP

This repository stores the project in two layers:

1. The repo itself contains the top-level documentation and extraction instructions.
2. The full working-tree snapshot is attached under the repository Releases as split archive assets, because GitHub blocks files over 100 MiB in normal Git history and the original workspace also exceeded Git LFS budget.

## Full Snapshot

Download the latest release assets from:

- `Releases`: `https://github.com/Narendhiranv04/seg-LLM-TAMP/releases`

The uploaded snapshot is a split `tar.zst` archive of the full workspace contents, mainly:

- `TAMP-PDDL/`
- `safe_publish_work/`

## How To Rebuild The Snapshot Locally

After downloading all parts for a directory archive:

```bash
cat TAMP-PDDL.tar.zst.part* > TAMP-PDDL.tar.zst
tar --zstd -xf TAMP-PDDL.tar.zst

cat safe_publish_work.tar.zst.part* > safe_publish_work.tar.zst
tar --zstd -xf safe_publish_work.tar.zst
```

## Main Run Commands

Most runs assume:

```bash
cd TAMP-PDDL
```

Ground-truth orchestrator:

```bash
python orchestrator.py
```

Segmentation orchestrator:

```bash
python run_segmentation_orchestrator.py
```

Real-mask segmentation run:

```bash
python run_with_segmentation.py
```

Live segmentation:

```bash
python run_live_segmentation.py
```

Variation runs:

```bash
python variation_1_easy/ground_truth_orchestrator_variation1_easy.py
python variation_1_easy/run_live_segmentation_variation1_easy.py
python variation_2/ground_truth_orchestrator_variation2.py
python variation_2/run_live_segmentation_variation2.py
python variation_3_hard/ground_truth_orchestrator_variation3_hard.py
python variation_3_hard/run_live_segmentation_variation3_hard.py
```

VLM pipeline:

```bash
python -m vlm_pipeline.vlm_main --list-goals
python -m vlm_pipeline.vlm_main --mock --no-env --goal full_task
python -m vlm_pipeline.vlm_main --goal full_task
python run_vlm_discovery_replan_live.py --mock
```

Experiments:

```bash
./run_baseline.sh
./run_baseline_experiment.sh
python experiments/exp1_pure_pddl.py --object soup --episodes 1
python experiments/exp4_vlm_planner.py --mock --trials 5
```

## Environment Notes

Typical simulator environment variables:

```bash
export COPPELIASIM_ROOT=$HOME/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

If you are on the original machine, a local environment snapshot exists at `TAMP-PDDL/.venv`.

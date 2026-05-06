# LLM Pipeline Testing Plan

This is the living test plan for bringing the LLM/VLM pipeline into alignment
with the ground-truth execution primitives. Keep this file updated as new scene
runs, failures, and fixes are discovered.

## Current Strategy

Test non-LLM components first. The LLM should only be added after the executor,
parser, state recognition, and failure checks are independently understood.

Primary question for each stage:

```text
Can this module reproduce or explain the ground-truth route without model uncertainty?
```

## Ground-Truth Baseline Status

Concrete commands:

```bash
# Single GT trial with GUI for manual inspection.
python3 run_ground_truth_trial.py --variant K1 --gui --output outputs/gt/K1_trial1.json

# Repeat all canonical GT variants headless.
python3 run_ground_truth_benchmark.py --variants K1,K2,K3,G1,G2,G3 --trials 1 --output-root outputs/gt
```

### Kitchen

Recent smoke tests:

```text
K1: 7 / 7
K2: 7 / 7
K3: 8 / 8
```

Reliable portions:

- K1 now passes the full GT sequence in the latest user-run GUI trial.
- K2 now passes the full GT sequence in the latest user-run GUI trial.
- K3 now passes the full GT sequence in the latest user-run GUI trial.
- Mug/cupboard/table/box-to-placement tasks mostly work.
- Grocery transfers mostly work.
- Box lid opening works.

Interpretation:

```text
K1, K2, and K3 are now full-scene positive GT baselines.
The next kitchen check is whether the LLM pipeline can replay the same concrete
sequences with its canonical region symbols normalized back to scene objects.
```

### Grill

Recent smoke tests:

```text
G1: 7 / 7
G2: 8 / 9
G3: 10 / 10
```

Reliable portions:

- Grill open/close primitive.
- Plate placement.
- Meat/spam from grill to table/plate.
- Outside meat to grill.

Known unreliable portion:

- G2 Task 3 fails before motion execution:
  `WARNING: No inside-grill meat found for Task 3.`
- This appears to be a selection/bookkeeping issue, not a primitive motion issue.
  The initial classification found one inside-grill meat, but the G2 Task 3
  target excludes the already selected inside target.

Interpretation:

```text
G1 and G3 are good full-scene executor baselines.
G2 should be treated as a known orchestration-selection issue.
```

## Testing Order

### Phase 1: Executor With Hand-Written GT Sequences

Start here.

Current harness:

```text
llm_pipeline/debug_execution.py
```

Concrete command:

```bash
python3 llm_pipeline/debug_execution.py
```

Current limitation:

```text
This harness is still K2-oriented and should be refactored into selectable
cases before it can cover kitchen_k1_prefix, kitchen_k1_box_only, grill_g1_full,
and grill_g3_full cleanly.
```

Goal:

- Use a mock planner or direct action list.
- Feed exact GT-style `DirectAction` sequences into the pipeline.
- Verify that `DirectPrimitiveExecutor` routes actions to the same primitives
  as GT.

Initial test cases to create/maintain:

```text
kitchen_k1_prefix
kitchen_k1_box_only
grill_g1_full
grill_g3_full
```

Success criteria:

```text
Given exact hand-written actions, llm_pipeline execution reaches the same
success/failure profile as GT.
```

### Phase 2: Executor Dispatch and Bundling Tests

Concrete command:

```text
No dedicated command yet.
```

Goal:

- Test action pairs without running entire scenes.
- Confirm that direct symbolic actions route to the intended primitive family.

Important cases:

```text
pick(mug3) + place(mug3, placement_boundary)
pick(soup) + place(soup, cupboard_boundary)
open(box_lid)
pick(mug2) + place(mug2, box_boundary)
open(grill)
close(grill)
pick(chicken) + place(chicken, inside_grill)
pick(steak) + place(steak, plate-top)
pick(plate) + place(plate, plate_boundary)
```

Success criteria:

```text
Each action pair dispatches to the expected kitchen or grill primitive path.
```

### Phase 3: Parser With Hand-Written GT Text

Concrete command:

```text
No dedicated command yet.
```

Goal:

- Verify that GT-style text output can be parsed into `DirectAction`s.
- No simulation required.

Example input:

```text
move(mug3)
pick(mug3)
move(placement_boundary)
place(mug3, placement_boundary)
```

Success criteria:

```text
Exact legal object and region names parse.
Unknown aliases fail early and clearly.
```

### Phase 2.5: Execution Concreteness With Scene State

Run before Phase 4.

Concrete command:

```text
No dedicated command yet. Next work item is to add a small non-LLM harness that
replays the six GT-as-is concrete action sequences through the LLM pipeline
executor, then derives prerequisite-preserving permutations.
```

Living status doc:

```text
Status Doc/LLM_PIPELINE_EXECUTION_CONCRETENESS.md
```

Goal:

- Verify that the scene-state object ids and region names are concrete enough
  for execution before testing failure classification.
- Use hand-written action pairs/sequences, not model output.
- Confirm that the executor accepts canonical Phase 5 symbols such as
  `steak1`, `inside_grill`, `plate_boundary`, and `dish_rack`.
- Keep `Status Doc/LLM_PIPELINE_EXECUTION_CONCRETENESS.md` as the source of
  truth for the six concrete GT sequences and permutation set.

Important cases:

```text
G2 closed grill: do not try pick(steak) while it is hidden.
G2 after opening grill: pick(steak) + place(steak, plate-top)
G2 after opening grill: pick(steak1) + place(steak1, inside_grill)
G1/G3 visible meat: pick(object) + place(object, inside_grill or plate-top)
Kitchen visible object: pick(mug2) + place(mug2, box_storage)
```

Success criteria:

```text
For objects that Phase 5 marks visible, the planned symbolic action names are
accepted by parser/executor dispatch and route to the expected primitive family.
Failures at this stage should expose symbol/dispatch/execution-contract gaps,
not failure-checker policy.
```

### Phase 4: Failure Checker Without LLM

Concrete command:

```text
No dedicated command yet.
```

Goal:

- Use fake or captured snapshots to test failure classification.

Important cases:

- Missing object before pick.
- Object not in target region after place.
- New objects visible after opening a box/grill.
- Mug placed on top of box/lid should be classified as target-region or Z
  validation failure, not high-level planning failure.
- Grill object hidden while lid is closed should be treated as not visible for
  pick preconditions.
- Grill object becoming visible after opening should be classified as a new
  object/state update, not a planner failure.

Success criteria:

```text
Failures are classified accurately enough to support future replanning.
```

Starting evidence:

```text
Use Phase 5 saved reports under Status Doc/scene_state_reports/ and
Status Doc/live_scene_state_reports/ as captured-state fixtures.
```

### Phase 5: Scene State / Segmentation Snapshot

Concrete commands:

```bash
# Kitchen pass.
python3 llm_pipeline/debug_state_builder.py --variant K1 --skip-prompt --json
python3 llm_pipeline/debug_state_builder.py --variant K2 --skip-prompt --json
python3 llm_pipeline/debug_state_builder.py --variant K3 --skip-prompt --json

# Grill pass.
python3 llm_pipeline/debug_state_builder.py --variant G1 --skip-prompt --json
python3 llm_pipeline/debug_state_builder.py --variant G2 --skip-prompt --json
python3 llm_pipeline/debug_state_builder.py --variant G3 --skip-prompt --json

# Grill live manual check.
python3 llm_pipeline/live_state_monitor.py --variant G1 --json --poll-interval 0.5
python3 llm_pipeline/live_state_monitor.py --variant G2 --json --poll-interval 0.5
python3 llm_pipeline/live_state_monitor.py --variant G3 --json --poll-interval 0.5
```

Output:

```text
Reports are saved under Status Doc/scene_state_reports/.
Live captures are saved under Status Doc/live_scene_state_reports/.
```

Goal:

- Verify scene recognition independently from planning and execution.

Important snapshots:

```text
K1 initial
K1 after opening box
G1 initial
G1 after opening grill
G3 initial
G3 after opening grill
```

For live grill checks, run from the nested repo after activating the simulator
environment:

```bash
source /home/paddy/miniconda3/etc/profile.d/conda.sh
conda activate rlbench
source ../../env_setup.sh
python3 llm_pipeline/live_state_monitor.py --variant G1 --json --poll-interval 0.5
```

The live monitor writes a new capture when visible objects, object-region
assignments, gripper state, or the grill lid joint state changes.

Success criteria:

```text
Visible objects, regions, and containment facts match what GT assumes.
```

### Phase 6: Prompt Builder With Frozen State

Concrete command:

```text
No frozen-state replay command yet.
```

Goal:

- Given a snapshot and goal, verify the generated prompt contains executable
  symbols and useful state facts.
- No real LLM call yet.

Success criteria:

```text
Prompt includes enough information for a planner to choose the GT sequence.
```

### Phase 7: Mock Planner End-to-End

Concrete command:

```bash
python3 llm_pipeline/debug_execution.py
```

Goal:

- Full pipeline loop using scripted actions:

```text
mock planner -> parser/action list -> executor -> failure checker -> metrics
```

Success criteria:

```text
Pipeline records completed actions, remaining actions, held object, failure
event, and failure reason correctly.
```

### Phase 8: Real LLM Plan-Only

Concrete commands:

```bash
# Text-only load, prompt, and dry-run plan validation.
python3 -m llm_pipeline.trial_runner --variant K1 --model <model_alias> --icl-mode zero_shot --preflight-only --output outputs/llm_preflight/K1.json

# First-plan-only mode with execution and failure checks disabled.
python3 -m llm_pipeline.trial_runner --variant K1 --model <model_alias> --icl-mode zero_shot --replan-mode off --output outputs/llm_plan_only/K1.json
```

Current limitation:

```text
The maintained llm_pipeline.trial_runner currently supports kitchen variants
K1/K2/K3 only.
```

Goal:

- Ask the real model for a plan.
- Parse it.
- Do not execute.

Success criteria:

```text
The LLM can produce legal action names, object names, and region names.
```

### Phase 9: Real LLM With Execution

Run only after non-LLM tests pass.

Concrete commands:

```bash
# Single execution trial.
python3 -m llm_pipeline.trial_runner --variant K1 --model <model_alias> --icl-mode zero_shot --gui --output outputs/llm_execution/K1_trial1.json

# Batch execution over kitchen variants.
python3 -m llm_pipeline.benchmark_runner --models <model_alias> --icl-modes zero_shot --variants K1,K2,K3 --trials 1 --output-root outputs/llm_benchmark
```

Current limitation:

```text
The maintained llm_pipeline execution runner currently supports kitchen variants
K1/K2/K3 only. Grill execution should wait for a matching LLM runner or be tested
through the mock/GT executor path first.
```

Recommended first execution scenes:

```text
G1 full
G3 full
K1 reliable prefix
Kitchen final mug-to-box as separate fragile primitive test
```

Success criteria:

```text
Real LLM plans execute through the same primitive routes already validated by
hand-written sequences.
```

## Immediate Next Step

Run the Phase 2.5 execution-concreteness gate before Phase 4 failure-checker
validation.

```text
1. Use Status Doc/LLM_PIPELINE_EXECUTION_CONCRETENESS.md as the live sequence
   list for K1-K3 and G1-G3.
2. Replay the six GT-as-is concrete sequences through the LLM pipeline executor.
3. Record parser, dispatch, and executor outcome for each scene.
4. Run the first prerequisite-preserving permutations.
5. Then start Phase 4 failure-checker tests.
```

Executor parity with hand-written GT remains the next pipeline-level target
after the execution contract and failure-checker behavior are understood.

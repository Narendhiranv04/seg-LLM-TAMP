# LLM Pipeline Execution Concreteness

Use this document to track the executor-facing gate between scene-state
validation and failure-checker validation.

This work is closer to Phase 1 and Phase 2 than Phase 4:

- Phase 1 proves hand-written sequences can run through the executor.
- Phase 2 proves symbolic action pairs dispatch to the correct primitive family.
- This gate proves Phase 5 scene-state symbols are concrete enough for those
  executor paths.

## Success Criteria

```text
For objects and regions visible in Phase 5 scene-state reports, hand-written
actions using those exact symbols parse and route to the expected executor
primitive family.
```

This phase should expose symbol, parser, dispatch, or execution-contract gaps.
It should not test failure-checker policy yet.

## Current Status

```text
Status: SEQUENCE CAPTURE STARTED
Next step: replay the GT-as-is sequences through the LLM pipeline executor
before deriving permutation tests.
```

## Inputs

Use recent Phase 5 artifacts as concrete scene-state fixtures:

```text
Status Doc/scene_state_reports/
Status Doc/live_scene_state_reports/
```

Useful starting reports:

```text
G2 closed grill static report after threshold fix
G2 live report after opening grill: 20260504_143459_G2_frame0029_scene_state.md
K2 or K3 kitchen static report with box/cupboard objects
```

Use prior GT run artifacts as the concrete sequence source:

```text
evaluation_results/K1_trial_001.runner_summary.json
evaluation_results/K2_trial_001.runner_summary.json
evaluation_results/K3_trial_001.runner_summary.json
evaluation_results/G1_trial_001.runner_summary.json
evaluation_results/G2_trial_001.runner_summary.json
evaluation_results/G3_trial_001.runner_summary.json
evaluation_results/*_trial_001.stdout.log
```

The runner summaries give task success/failure. The stdout logs give the
runtime object assignments that the kitchen GT resolves dynamically and the
exact grill object ids picked by the primitive executor.

## Contracts To Check

Object ids must be executable as written:

```text
steak
steak1
chicken
mug2
mug3
plate
```

Region ids must be executable or intentionally translated before execution:

```text
inside_grill
plate-top
plate_boundary
dish_rack
box_storage
placement_boundary
cupboard_lower
cupboard_upper
```

Important compatibility question:

```text
The LLM-facing grill region is inside_grill, while older execution/GT code may
still expect grill-top. Verify whether executor dispatch already normalizes this
or needs a compatibility layer.
```

## Test Cases

| Case | Fixture / Scene State | Hand-written action | Expected route | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| G2 closed grill hidden steak | Closed-grill G2 static report | Do not issue `pick(steak)` | Scene-state/action selection gate | Pending | Hidden object should not be selected for execution. |
| G2 opened grill inside steak | `20260504_143459_G2_frame0029_scene_state.md` | `pick(steak)` + `place(steak, plate-top)` | Grill pick/place primitive | Pending | Uses visible `steak` and `plate-top`. |
| G2 opened grill numbered steak | `20260504_143459_G2_frame0029_scene_state.md` | `pick(steak1)` + `place(steak1, inside_grill)` | Grill pick/place primitive | Pending | Checks numbered object id plus `inside_grill` compatibility. |
| G2 plate placement | G2 static/live report | `pick(plate)` + `place(plate, plate_boundary)` | Grill plate placement primitive | Pending | Plate starts at `dish_rack`. |
| Kitchen box placement | K2/K3 static report | `pick(mug2)` + `place(mug2, box_storage)` | Kitchen box placement primitive | Pending | Checks canonical `box_storage` compatibility. |
| Kitchen cupboard placement | K2/K3 static report | `pick(grocery)` + `place(grocery, cupboard_lower)` | Kitchen cupboard primitive | Pending | Use a concrete visible grocery from the selected report. |

## GT-As-Is Concrete Sequences

First replay these sequences exactly through the LLM pipeline with hand-written
actions. This is intentionally before permutation testing.

Use LLM-facing canonical names where we already changed scene state:

```text
inside_grill = GT grill-top / grill_boundary
box_storage = GT box_boundary
cupboard_lower or cupboard_upper = GT cupboard_boundary family
```

For the first GT-as-is executor test, keep the target region closest to the
current executor contract if needed, but record any normalization explicitly.

### Kitchen K1

Source: `variation_1_easy/ground_truth_orchestrator_variation1_easy.py`.
Latest confirmed GT run:

```text
Command: python3 run_ground_truth_trial.py --variant K1 --gui
Summary: /tmp/gt_trial_tzwfhr38/K1_trial_001.runner_summary.json
Stdout:  /tmp/gt_trial_tzwfhr38/K1_trial_001.stdout.log
Result:  7/7, episode_success=true
```

Runtime assignments from the recorded GT run:

```text
mug_on_box = mug2
mug_in_cupboard = mug3
grocery_in_box = soup
runtime_grocery_on_table = spam
runtime_table_mugs = [mug2, mug3]
```

Concrete replay sequence:

```text
1. pick(mug2)   -> place(mug2, placement_boundary)
2. pick(mug3)   -> place(mug3, placement_boundary)
3. open(box_lid)
4. pick(soup)   -> place(soup, cupboard_lower)
5. pick(spam)   -> place(spam, cupboard_lower)
6. pick(mug2)   -> place(mug2, box_storage)
7. pick(mug3)   -> place(mug3, box_storage)
```

Latest GT result: 7/7. This is now a full-scene positive baseline for the
LLM-pipeline GT-as-is replay.

Note: the latest stderr contains failed intermediate `cupboard_lower` placement
samples, but stdout shows the planner recovered with a successful cupboard
placement and the runner summary reports all tasks passed. Treat those sample
failures as planner search noise unless the final task result fails.

### Kitchen K2

Source: `variation_2/ground_truth_orchestrator_variation2.py`.
Latest confirmed GT run:

```text
Command: python3 run_ground_truth_trial.py --variant K2 --gui
Summary: /tmp/gt_trial_lbqo5hn9/K2_trial_001.runner_summary.json
Stdout:  /tmp/gt_trial_lbqo5hn9/K2_trial_001.stdout.log
Result:  7/7, episode_success=true
```

Runtime assignments from the recorded GT run:

```text
mug_in_cupboard = mug3
runtime_grocery_on_table = sugar
mug_on_box = mug2
runtime_grocery_in_box = soup
runtime_table_mugs = [mug2, mug3]
```

Concrete replay sequence:

```text
1. pick(mug3)   -> place(mug3, placement_boundary)
2. pick(sugar)  -> place(sugar, cupboard_lower)
3. pick(mug2)   -> place(mug2, placement_boundary)
4. open(box_lid)
5. pick(soup)   -> place(soup, cupboard_lower)
6. pick(mug2)   -> place(mug2, box_storage)
7. pick(mug3)   -> place(mug3, box_storage)
```

Latest GT result: 7/7. This is now a full-scene positive baseline for the
LLM-pipeline GT-as-is replay.

Note: the latest stderr contains failed intermediate `cupboard_lower` placement
samples, but stdout shows the planner recovered and the runner summary reports
all tasks passed. Treat those sample failures as planner search noise unless
the final task result fails.

### Kitchen K3

Source: `variation_3_hard/ground_truth_orchestrator_variation3_hard.py`.
Latest confirmed GT run:

```text
Command: python3 run_ground_truth_trial.py --variant K3 --gui
Summary: /tmp/gt_trial_gniqd5mi/K3_trial_001.runner_summary.json
Stdout:  /tmp/gt_trial_gniqd5mi/K3_trial_001.stdout.log
Result:  8/8, episode_success=true
```

Runtime assignments from the recorded GT run:

```text
mug_in_cupboard = mug3
runtime_grocery_on_table = sugar
mug_on_box = mug2
runtime_grocery_in_box = soup
runtime_table_mugs = [mug2, mug3, mug1]
```

Concrete replay sequence:

```text
1. pick(mug3)   -> place(mug3, placement_boundary)
2. pick(sugar)  -> place(sugar, cupboard_lower)
3. pick(mug2)   -> place(mug2, placement_boundary)
4. open(box_lid)
5. pick(soup)   -> place(soup, cupboard_lower)
6. pick(mug2)   -> place(mug2, box_storage)
7. pick(mug3)   -> place(mug3, box_storage)
8. pick(mug1)   -> place(mug1, box_storage)
```

Latest GT result: 8/8. This is now a full-scene positive baseline for the
LLM-pipeline GT-as-is replay.

Note: the latest stderr contains failed intermediate `placement_boundary` and
`cupboard_lower` placement samples, but stdout shows the planner recovered and
the runner summary reports all tasks passed. Treat those sample failures as
planner search noise unless the final task result fails.

### Grill G1

Source: `grill_task2/ground_truth_orchestrator_variation1 copy.py`,
`evaluation_results/G1_trial_001.runner_summary.json`,
`evaluation_results/G1_trial_001.stdout.log`.

Concrete replay sequence:

```text
1. open(grill_lid)
2. pick(spam)    -> place(spam, table)
3. pick(chicken) -> place(chicken, inside_grill)
4. close(grill_lid)
5. pick(plate)   -> place(plate, plate_boundary)
6. open(grill_lid)
7. pick(chicken) -> place(chicken, plate-top)
```

Recorded GT result: 7/7.

### Grill G2

Source: `grill_task2/ground_truth_orchestrator_variation1 copy.py`,
`evaluation_results/G2_trial_001.runner_summary.json`,
`evaluation_results/G2_trial_001.stdout.log`.

Concrete replay sequence:

```text
1. open(grill_lid)
2. pick(plate)   -> place(plate, plate_boundary)
3. pick(steak)   -> place(steak, plate-top)
4. pick(chicken) -> place(chicken, inside_grill)
5. pick(steak1)  -> place(steak1, inside_grill)
6. close(grill_lid)
7. open(grill_lid)
8. pick(chicken) -> place(chicken, plate-top)
9. pick(steak1)  -> place(steak1, plate-top)
```

Recorded GT result: 8/9. The recorded GT Task 3 failed with
`No inside-grill meat found for Task 3`, even though the later stdout shows
`steak` as the inside-grill meat moved to the plate at Task 9. Treat that as a
known GT bookkeeping/selection issue while checking whether the LLM pipeline can
execute the concrete object id after scene-state evidence is available.

### Grill G3

Source: `grill_task2/ground_truth_orchestrator_variation1 copy.py`,
`evaluation_results/G3_trial_001.runner_summary.json`,
`evaluation_results/G3_trial_001.stdout.log`.

Concrete replay sequence:

```text
1. open(grill_lid)
2. pick(spam)    -> place(spam, table)
3. pick(plate)   -> place(plate, plate_boundary)
4. pick(steak)   -> place(steak, plate-top)
5. pick(chicken) -> place(chicken, inside_grill)
6. pick(steak1)  -> place(steak1, inside_grill)
7. close(grill_lid)
8. open(grill_lid)
9. pick(chicken) -> place(chicken, plate-top)
10. pick(steak1) -> place(steak1, plate-top)
```

Recorded GT result: 10/10.

## Permutation Plan

Do not start here until the GT-as-is replay is recorded.

Allowed permutations should preserve physical and scene-state prerequisites:

```text
Kitchen:
- Swap independent grocery-to-cupboard tasks when both objects are visible and
  reachable.
- Swap final table-mug-to-box order after box_lid has been opened.
- Do not move groceries from box before open(box_lid).
- Do not place mugs into box before open(box_lid).

Grill:
- Swap outside meats sent into inside_grill when both are visible and outside.
- Swap final grilled meats sent to plate-top after close/open cooking cycle.
- Do not pick hidden inside_grill meats while the grill is closed.
- Do not close the grill before the outside meats have been placed inside it.
- Do not move cooked meats to plate-top before the required close/open cycle,
  unless the test is intentionally checking failure handling.
```

Candidate first permutation set:

| Scene | Permutation | Why |
| --- | --- | --- |
| K1 | Swap final `mug2`/`mug3` box order | Tests box insertion order without changing prerequisites. |
| K2 | Swap `sugar -> cupboard` and `mug2 -> placement_boundary` | Both happen before opening the box in GT. |
| K3 | Try final box order `mug1`, `mug2`, `mug3` | Isolates multi-mug slot sensitivity. |
| G1 | Move plate before close/open cycle only as a negative-control permutation | Should expose whether the plan violates intended grill ordering. |
| G2 | Swap chicken/steak1 outside-to-grill order | Both are outside meats before cooking. |
| G3 | Swap chicken/steak1 outside-to-grill order and final plate order | Both preserve the grill cycle. |

## Implementation Plan

1. Replay GT-as-is sequences through the LLM pipeline with hand-written action
   lists.
2. Record parser, dispatch, and executor outcome per scene in this document.
3. Check parser normalization for canonical Phase 5 region symbols.
4. Add only the smallest harness changes needed to select `K1`-`K3` and
   `G1`-`G3` cases.
5. Derive and run prerequisite-preserving permutations.
6. Move to Phase 4 only after GT-as-is and first permutations are understood.

## Work Log

| Date | Item | Status | Notes |
| --- | --- | --- | --- |
| 2026-05-04 | Capture GT-as-is concrete sequences for K1-K3 and G1-G3 | Done | Derived from GT orchestrator code plus recorded runner summaries/stdout logs. |
| 2026-05-04 | Refresh K1 GT baseline | Done | User reran `python3 run_ground_truth_trial.py --variant K1 --gui`; latest summary is 7/7. |
| 2026-05-04 | Refresh K2 GT baseline | Done | User reran `python3 run_ground_truth_trial.py --variant K2 --gui`; latest summary is 7/7. |
| 2026-05-04 | Refresh K3 GT baseline | Done | User reran `python3 run_ground_truth_trial.py --variant K3 --gui`; latest summary is 8/8. |
| 2026-05-04 | Replay GT-as-is sequences through LLM pipeline | Pending | Next action. |
| 2026-05-04 | Derive first permutation set | Drafted | Do after GT-as-is replay is recorded. |

## Open Questions

- Should `inside_grill` be normalized to `grill-top` at executor boundary, or
  should grill execution natively accept `inside_grill`?
- Should `dish_rack` ever be a placement target, or only a detected source
  region for the plate?
- Which kitchen report should be the canonical fixture for box/cupboard
  execution-concreteness checks?

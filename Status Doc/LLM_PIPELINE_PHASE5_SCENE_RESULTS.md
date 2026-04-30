# LLM Pipeline Phase 5 Scene Results

Use this document to record phase 5 scene-state / segmentation snapshot
successes and failures across all six scenes.

Success criteria from `LLM_PIPELINE_TESTING_PLAN.md`:

```text
Visible objects, regions, and containment facts match what GT assumes.
```

## Result Key

```text
PASS = visible objects are detected and each task object has a concrete geometric region assignment.
FAIL = no usable scene state was produced.
PARTIAL = objects are detected, but region/containment evidence still needs cleanup.
BLOCKED = the scene could not be launched or checked.
```

## Run Metadata

```text
Date: 2026-04-30
Tester: manual terminal runs
Branch/commit: kitchen-region-rename-latest / through d4ce4b0
Command(s): python llm_pipeline/debug_state_builder.py --variant <scene> --skip-prompt --json
Notes: Kitchen static scene-state reports now pass with geometric region assignments. Grill validation has moved forward with G1 passing; G2/G3 still need fresh static and dynamic checks.
```

## Manual Verification Plan

Open each variation in CoppeliaSim and compare the visible objects, regions, and
containment facts against the saved `debug_state_builder.py` report. Use those
manual checks to correct the scene-state builder.

Current immediate focus:

1. Kitchen static snapshots are concrete enough to move on.
2. Grill domain next.
3. Start with base static scene-state reports for G1, G2, and G3.
4. After static grill accuracy is understood, test progression/dynamic snapshots
   after actions such as opening the grill and moving meat to the plate.

## K1

| Scenario | Result | Visible objects match GT? | Regions match GT? | Containment facts match GT? | Notes / failure reason |
| --- | --- | --- | --- | --- | --- |
| Initial snapshot | PASS | Yes | Yes | Yes | Latest report `20260430_123924_K1_scene_state.md`: detected mug2, mug3, spam, box_lid, cupboard with geometric assignments. Visual region evidence still has multiple candidates, but canonical geometry resolves task objects. |
| After opening box |  |  |  |  |  |
| Other scenario |  |  |  |  |  |

## K2

| Scenario | Result | Visible objects match GT? | Regions match GT? | Containment facts match GT? | Notes / failure reason |
| --- | --- | --- | --- | --- | --- |
| Initial snapshot | PASS | Yes | Yes | Yes | Latest report `20260430_124152_K2_scene_state.md`: detected mug2, mug3, sugar, box_lid, cupboard with geometric assignments. Visual evidence remains multi-region for some objects, but canonical geometry resolves task objects. |
| After opening box |  |  |  |  |  |
| Other scenario |  |  |  |  |  |

## K3

| Scenario | Result | Visible objects match GT? | Regions match GT? | Containment facts match GT? | Notes / failure reason |
| --- | --- | --- | --- | --- | --- |
| Initial snapshot | PASS | Yes | Yes | Yes | Latest report `20260430_140905_K3_scene_state.md`: detected mug1, mug2, mug3, spam, sugar, box_lid, cupboard with geometric assignments. Live monitor also captured meaningful K3 state transitions after manual movement. |
| After opening box |  |  |  |  |  |
| Other scenario |  |  |  |  |  |

## G1

| Scenario | Result | Visible objects match GT? | Regions match GT? | Containment facts match GT? | Notes / failure reason |
| --- | --- | --- | --- | --- | --- |
| Initial snapshot | PASS | Yes | Yes | Yes | Latest report `20260430_211632_G1_scene_state.md`: detected spam, chicken, plate, grill_lid. Geometric assignments resolve spam to grill-top, plate to dish_rack, and chicken/grill_lid to table. Chicken and grill_lid still lack visual region evidence, so keep visual masks as debug-only evidence. |
| After opening grill |  |  |  |  |  |
| Other scenario |  |  |  |  |  |

## G2

| Scenario | Result | Visible objects match GT? | Regions match GT? | Containment facts match GT? | Notes / failure reason |
| --- | --- | --- | --- | --- | --- |
| Initial snapshot | PARTIAL | Yes | Partial | Partial | Detected steak, chicken, plate, grill_lid. steak maps to grill-top and plate maps to plate/dish_rack; chicken and grill_lid lack mask_regions. |
| After opening grill |  |  |  |  |  |
| Other scenario |  |  |  |  |  |

## G3

| Scenario | Result | Visible objects match GT? | Regions match GT? | Containment facts match GT? | Notes / failure reason |
| --- | --- | --- | --- | --- | --- |
| Initial snapshot | PARTIAL | Yes | Partial | Partial | Detected spam, steak, chicken, plate, grill_lid. spam/steak map to grill-top and plate maps to plate/dish_rack; chicken and grill_lid lack mask_regions. |
| After opening grill |  |  |  |  |  |
| Other scenario |  |  |  |  |  |

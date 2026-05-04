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
Branch/commit: kitchen-region-rename-latest / through 64210d9
Command(s): python llm_pipeline/debug_state_builder.py --variant <scene> --skip-prompt --json
Notes: Kitchen static scene-state reports pass with geometric region assignments. Grill static and live scene-state checks now use grill-specific regions, preserve numbered meat ids, and report `inside_grill` for `grill_boundary`.
```

## Manual Verification Plan

Open each variation in CoppeliaSim and compare the visible objects, regions, and
containment facts against the saved `debug_state_builder.py` report. Use those
manual checks to correct the scene-state builder.

Current status:

1. Kitchen static snapshots are concrete enough to move on.
2. Grill static snapshots are concrete enough to move on.
3. Grill dynamic/live snapshots are working after manual grill state changes.
4. `inside_grill` is the LLM-facing region for scene object `grill_boundary`; `grill-top` remains only as a backward-compatible executable alias.
5. Numbered grill meats stay distinct object ids, e.g. `steak` and `steak1`, instead of collapsing to broad meat labels.
6. Grill segmentation uses a stricter default threshold so tiny closed-lid mask leaks do not count as visible objects.
7. Next phase: failure checker validation without LLM calls.

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
| Initial snapshot | PASS | Yes | Yes | Yes | Latest report `20260504_141705_G1_scene_state.md`: closed grill detected chicken, plate, grill_lid with grill-specific regions and `grill_lid_closed`; hidden inside object is not falsely reported visible. |
| After opening grill | PASS | Yes | Yes | Yes | Live monitor captures grill lid changes and refreshed object-region state after manual changes. |
| Other scenario |  |  |  |  |  |

## G2

| Scenario | Result | Visible objects match GT? | Regions match GT? | Containment facts match GT? | Notes / failure reason |
| --- | --- | --- | --- | --- | --- |
| Initial snapshot | PASS | Yes | Yes | Yes | Fresh closed-grill report after threshold fix no longer treats tiny 18-pixel mask leaks as visible. Visible objects/regions match the camera-visible state. |
| After opening grill | PASS | Yes | Yes | Yes | Live report `20260504_143459_G2_frame0029_scene_state.md`: detector threshold 50, lid open, steak/steak1/chicken visible with `inside_grill` evidence and semantic facts. |
| Other scenario |  |  |  |  |  |

## G3

| Scenario | Result | Visible objects match GT? | Regions match GT? | Containment facts match GT? | Notes / failure reason |
| --- | --- | --- | --- | --- | --- |
| Initial snapshot | PASS | Yes | Yes | Yes | Latest report `20260504_141810_G3_scene_state.md`: closed grill detected steak1, chicken, plate, grill_lid with grill-specific regions and `grill_lid_closed`; hidden inside objects are not falsely reported visible. |
| After opening grill | PASS | Yes | Yes | Yes | Live monitor path has been validated on grill scenes; state changes produce new captures with updated lid/object-region facts. |
| Other scenario |  |  |  |  |  |

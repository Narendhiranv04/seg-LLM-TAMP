# LLM Pipeline Phase 5 Scene Results

Use this document to record phase 5 scene-state / segmentation snapshot
successes and failures across all six scenes.

Success criteria from `LLM_PIPELINE_TESTING_PLAN.md`:

```text
Visible objects, regions, and containment facts match what GT assumes.
```

## Result Key

```text
PASS =
FAIL =
PARTIAL =
BLOCKED =
```

## Run Metadata

```text
Date: 2026-04-29
Tester: manual terminal runs
Branch/commit: main / through 863ac95
Command(s): python llm_pipeline/debug_state_builder.py --variant <scene> --skip-prompt --json
Notes: Initial object recognition is working for tested kitchen and grill scenes. Region and containment facts still need cleanup, especially grill outside meat/table assignment and some broad kitchen multi-region assignments.
```

## K1

| Scenario | Result | Visible objects match GT? | Regions match GT? | Containment facts match GT? | Notes / failure reason |
| --- | --- | --- | --- | --- | --- |
| Initial snapshot | PARTIAL | Yes | Partial | Partial | Detected mug2, mug3, spam, box_lid, cupboard. Region evidence exists, but several objects have broad/multiple regions. |
| After opening box |  |  |  |  |  |
| Other scenario |  |  |  |  |  |

## K2

| Scenario | Result | Visible objects match GT? | Regions match GT? | Containment facts match GT? | Notes / failure reason |
| --- | --- | --- | --- | --- | --- |
| Initial snapshot |  |  |  |  | Not run/reported yet in this pass. |
| After opening box |  |  |  |  |  |
| Other scenario |  |  |  |  |  |

## K3

| Scenario | Result | Visible objects match GT? | Regions match GT? | Containment facts match GT? | Notes / failure reason |
| --- | --- | --- | --- | --- | --- |
| Initial snapshot | PARTIAL | Yes | Partial | Partial | Detected mug1, mug2, mug3, spam, sugar, box_lid, cupboard. Region evidence exists, but several objects have broad/multiple regions. |
| After opening box |  |  |  |  |  |
| Other scenario |  |  |  |  |  |

## G1

| Scenario | Result | Visible objects match GT? | Regions match GT? | Containment facts match GT? | Notes / failure reason |
| --- | --- | --- | --- | --- | --- |
| Initial snapshot | PARTIAL | Yes | Partial | Partial | Detected spam, chicken, plate, grill_lid. spam maps to grill-top and plate maps to plate/dish_rack; chicken and grill_lid lack mask_regions. |
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

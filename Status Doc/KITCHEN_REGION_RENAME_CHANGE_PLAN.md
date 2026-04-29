# Kitchen Region Rename Change Plan

This document records the planned public-symbol migration for kitchen scene
regions. The first implementation should not rename CoppeliaSim `.ttt` scene
objects. Existing scene objects such as `box_boundary`, `box_lid`,
`cupboard_boundary`, and `cupboard_boundary_top` should stay in the scenes and
be mapped to clearer public symbols in Python.

## Current Issue

The kitchen scene-state layer currently mixes task targets, broad fallback
regions, and visual evidence regions under similar names.

- `box_boundary` is the real task target for putting mugs inside the box.
- `box-top` and `box-inside` are both currently mapped through `box_base`.
- `box-top` should instead represent the top of the closed `box_lid`.
- `box-inside` is useful as a looser fallback for `box_boundary`.
- `cupboard_boundary` is a lower-shelf scene object and target region.
- `cupboard_boundary_top` is an upper-shelf scene object and target region.
- `shelf-lower` is broad cupboard geometry and should be treated as fallback
  evidence, not as the precise target.

Because reports and prompts expose these names directly, the planner can see a
mix of primary targets and fallback evidence as if they were equally meaningful
task regions.

## Target Vocabulary

| Current name | New public name | Scene backing | Role |
| --- | --- | --- | --- |
| `box_boundary` | `box_storage` | `box_boundary` | Primary target for placing objects inside the box |
| `box-top` | `box_lid_top` | `box_lid` | Evidence that an object is on the closed box lid |
| `box-inside` | `box_inside_fallback` | `box_base` | Looser fallback for inside-box detection or placement |
| `cupboard_boundary` | `cupboard_lower` | `cupboard_boundary` | Lower shelf cupboard target |
| `cupboard_boundary_top` | `cupboard_upper` | `cupboard_boundary_top` | Upper shelf cupboard target |
| `shelf-lower` | `cupboard_fallback` | `cupboard` | Broad cupboard fallback evidence |

The canonical names above should be preferred in reports, prompts, parser
contracts, and future manual-verification notes. Legacy names should remain
accepted as aliases during migration.

## Pipeline Impact

- Environment and scene mapping: `rlbench_kitchen_env.py` should expose the
  canonical region names while still resolving them to existing scene objects.
  `box_lid_top` should map to `box_lid`, not `box_base`.
- Symbol registry and parser: `llm_pipeline/executable_symbols.py` should
  advertise canonical symbols. `strict_parser.py` should normalize legacy names
  before validation so old plans do not immediately break.
- Prompt and report generation: prompt builders and `debug_state_builder.py`
  should display canonical names first. Legacy names may appear only as alias
  or debug metadata when useful.
- Executor, failure logic, and metrics: all region comparisons should use a
  central normalized name. Existing checks such as `target_region ==
  "box_boundary"` should become canonical checks against `box_storage`.
- Fallback behavior: box placement fallback should become
  `box_storage -> box_inside_fallback`.
- GT and legacy VLM paths: old scripts and ground-truth orchestrators may still
  emit legacy names. They should be treated as compatibility callers while the
  alias normalizer is active.

## Migration Strategy

1. Add a central kitchen-region alias normalizer.
2. Update runtime region registration to expose canonical names.
3. Keep legacy aliases accepted by parser, executor, failure logic, metrics, and
   GT compatibility paths.
4. Update scene-state reports and prompts to prefer canonical names only.
5. Update tests and manual-verification docs to use canonical names.
6. Remove or hide legacy names from planner-facing prompts after kitchen static
   scene verification passes.
7. Only consider removing legacy aliases after kitchen static snapshots,
   progression snapshots, and executor parity are stable.

Do not expose both old and new names as separate planner choices in prompts;
that can make the model treat aliases as distinct physical regions.

## Risks And Issues

- Full public rename can break LLM output parsing unless aliases are normalized
  before prompts are changed.
- If `env.regions` exposes both canonical and legacy keys, scene-state reports
  may duplicate evidence unless they deduplicate by canonical name.
- Tests that assert literal strings such as `box_boundary` or
  `cupboard_boundary` will need updates.
- Existing GT runners may still depend on old names, so old-name support should
  remain until non-LLM kitchen validation passes.
- `box_lid_top` evidence depends on `box_lid` visibility. When the lid is open,
  it should not be treated as the same region as inside-box storage.
- `cupboard_fallback` may overlap visually with both cupboard shelf targets and
  should not override precise `cupboard_lower` or `cupboard_upper` evidence.

## Test Plan

- Run static scene-state reports for `K1`, `K2`, and `K3`.
- Verify reports show canonical region names in object evidence.
- Confirm `box_lid_top` evidence comes from `box_lid`, not `box_base`.
- Confirm objects inside or targeted to the box normalize to `box_storage`.
- Confirm `box_inside_fallback` remains available only as fallback evidence.
- Add normalizer/parser tests for:
  - `box_boundary -> box_storage`
  - `box-top -> box_lid_top`
  - `box-inside -> box_inside_fallback`
  - `cupboard_boundary -> cupboard_lower`
  - `cupboard_boundary_top -> cupboard_upper`
  - `shelf-lower -> cupboard_fallback`
- Re-run current non-LLM commands from `LLM_PIPELINE_TESTING_PLAN.md`, starting
  with `debug_state_builder.py` for kitchen variants.

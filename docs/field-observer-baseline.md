# FIELD / OBSERVER baseline

This is the current reproducible diagnostic baseline for returning to the
last coherent FIELD SPACE / OBSERVER state. It is an offline diagnostic path;
it does not change a review, create a HUMAN anchor, or run the web UI.

## Current direction

```text
canonical field
    -> HUMAN anchors
    -> FIELD visibility
    -> OBSERVER descriptors
    -> Camera Atlas
```

The eventual next step is to use the FIELD / OBSERVER descriptors to describe
the camera-view space, find view regions that are poorly represented by the
existing HUMAN anchors, and then recommend representative frames for manual
calibration. That selector is deliberately not implemented here.

## Operational modules

The reusable geometry and baseline path are:

- `supervisor/field_pitch_model.py`: centered canonical pitch geometry. The
  default model is 105 x 68 m; saved benchmark reviews may carry a confirmed
  per-video width (the current normal benchmark uses 70.8 m).
- `supervisor/field_camera_calibration.py`, `field_solver.py`, and
  `field_calibration_strategy.py`: manual HUMAN homography calibration and
  its controlled refinement strategy.
- `supervisor/field_visibility.py`: finite pitch/viewport intersection,
  visible area, boundary and goal-line visibility.
- `supervisor/field_space_observer.py`: FIELD SPACE / OBSERVER V1. It selects
  a valid projective branch and emits visible fraction, visible polygon,
  focus, visible X bounds, goal visibility, and optical-axis diagnostics.
- `supervisor/field_camera_atlas_dense.py`: turns trusted per-keyframe camera
  states into a dense per-frame Camera Atlas. Derived states remain derived;
  they are not promoted to HUMAN.
- `supervisor/check_field_space_observer.py`: validates dense homographies,
  computes and saves the Observer JSON.
- `supervisor/check_field_space_observer_visual.py`: renders selected Observer
  records as pitch-space PNGs and a contact sheet.

The browser/API workflow is separate from this offline diagnostic. Its main
entry point is `supervisor/app.py`; `FieldWorkflowManager` in
`supervisor/field_workflow.py` orchestrates trajectory, direct HUMAN
relocalization, dense Camera Atlas generation, and (when coverage is still
missing) the current active-anchor fallback.

## Exact reproduction command

Run from the repository root with the local environment:

```bash
source .venv/bin/activate

# 1. Transfer the seven completed HUMAN match-setup anchors from the easy clip
#    to keyframes in the normal clip.
PYTHONPATH=. python -m supervisor.check_camera_atlas_transfer

# 2. Merge the explicit target HUMAN active anchor(s), currently frame 2526.
PYTHONPATH=. python -m supervisor.check_camera_active_anchor_update

# 3. Recover remaining direct-transfer gaps with local Camera H_t and build
#    the dense target Camera Atlas.
PYTHONPATH=. python -m supervisor.check_camera_atlas_gap_recovery

# 4. Calculate FIELD / OBSERVER V1 for every target frame.
PYTHONPATH=. python -m supervisor.check_field_space_observer

# 5. Render the selected diagnostic frames and contact sheet.
PYTHONPATH=. python -m supervisor.check_field_space_observer_visual
```

The first three commands are the camera-state preparation required by the
current saved baseline. Steps 4 and 5 are the FIELD / OBSERVER diagnostic
itself. To render a different set of frames:

```bash
PYTHONPATH=. python -m supervisor.check_field_space_observer_visual \
  --frames 0 30 92 899 1798 2526 2697 3595
```

The commands use these inputs:

| Role | Path | Meaning |
|---|---|---|
| source HUMAN review | `supervisor/data/reviews/benchmark_easy_2min.json` | Seven completed manual `match_setup` calibrations: 55, 87, 175, 220, 345, 498, 540. |
| source video | `input/benchmark_easy_2min.mp4` | Video containing those source HUMAN anchors. |
| target review | `supervisor/data/reviews/benchmark_normal_2min.json` | Confirmed target pitch model and the manual `active_anchor` at frame 2526. |
| target video | `input/benchmark_normal_2min.mp4` | 1920x1080, 29.97 fps, 3596 frames. |

The intermediate and final outputs are:

1. `benchmark_normal_2min_camera_trajectory_v1.json`
2. `benchmark_normal_2min_camera_anchor_transfer_v1.json`
3. `benchmark_normal_2min_camera_anchor_update_v1.json`
4. `benchmark_normal_2min_active_anchor_after_update_v1.json`
5. `benchmark_normal_2min_camera_atlas_dense_v1.json`
6. `benchmark_normal_2min_field_space_observer_v1.json`
7. `benchmark_normal_2min_field_space_observer_diag_v1/`, containing the
   selected PNGs and `observer_contact_sheet.png`.

All outputs are under `supervisor/data/reviews/`. The benchmark JSON and PNG
baseline are retained as reproducibility evidence. `asset_*` review and
experiment outputs are local/ignored artifacts and are not part of this
baseline.

## Provenance and boundary

The saved normal-clip dense Atlas is intentionally a mixed research artifact:
103 soft states came from source HUMAN cross-video transfer, 56 came from
target active-HUMAN relocalization, and one is the exact target HUMAN frame.
The Observer code consumes the dense homography sequence and the target pitch
dimensions; it does not itself consume propagation, refinement, PEOPLE, or
the anchor-selector code.

The current app still exposes compatibility endpoints for propagation,
segment refinement, online adaptation, and the older coverage-based anchor
selector. They remain available for research and UI continuity, but they are
not evidence for the Observer baseline and should not be treated as the next
development direction.

## Experimental and legacy components

These components are retained because they contain useful research history:

- `field_propagation.py`, `field_segment_refinement.py`,
  `field_camera_refinement.py`, and `field_auto_refinement.py`: propagation
  and refinement paths, including partial-HUMAN correction experiments.
- `field_active_anchoring.py`, `field_anchor_selector.py`, and
  `field_camera_active_anchor.py`: successive active-anchor experiments.
- `field_camera_state*.py`, `field_camera_*.py`, and their matching
  `check_camera_*.py` scripts: Camera State, relocalization, flow, bridge,
  recovery, transfer, and Atlas experiments.
- `scripts/run_field_geometry_baseline.py` and the older `scripts/analyze_*`
  camera-state checks: earlier calibration/state investigations.
- The timestamped `before-*`/`backup-*` reviews and the saved
  `camera_*_v1.json` artifacts: historical evidence, not live inputs.

The committed planner, sparse, hub, and HUMAN-guard experiment families remain
under `supervisor/` because their scripts use module-qualified imports and are
useful research records. They are not imported by the FSO scripts. They were
left in their original location rather than moved behind compatibility
wrappers, which would add churn without improving the operational path.

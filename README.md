# RealJaenLab

RealJaenLab is a football-computer-vision research repository. It contains
early detector/dataset experiments and a newer FIELD / CAMERA workflow for
describing football-camera views with a canonical pitch.

The current development direction is:

```text
canonical field -> HUMAN anchors -> FIELD visibility
                -> OBSERVER descriptors -> Camera Atlas
```

The reproducible FIELD SPACE / OBSERVER V1 diagnostic is documented in
[docs/field-observer-baseline.md](docs/field-observer-baseline.md). It names
the exact benchmark clips, HUMAN calibration provenance, intermediate JSON
artifacts, and commands for regenerating the pitch-space figures.

## Repository areas

- `supervisor/`: FIELD geometry, calibration, Camera Atlas, Observer code, and
  the web supervisor.
- `supervisor/data/reviews/`: saved benchmark reviews and diagnostic artifacts.
- `input/`: local benchmark videos; these are ignored by Git.
- `scripts/`: earlier dataset, detector, and camera-state experiments.
- `docs/`: repository and experiment documentation.

Propagation, refinement, previous camera-state approaches, and active-anchor
selectors are retained as experimental/legacy research. They are not the next
feature to implement. The planned next step is to use FIELD / OBSERVER
descriptors to identify camera-view regions insufficiently represented by
HUMAN anchors, then design representative-frame recommendations.

The project uses a local `.venv` when available:

```bash
source .venv/bin/activate
```

No commit or push is performed by the cleanup workflow.

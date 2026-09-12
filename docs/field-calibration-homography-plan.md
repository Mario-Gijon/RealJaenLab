# RealJaenLab — Initial Field Calibration / Homography Plan

## Status

This document defines the initial direction for the FIELD calibration work in RealJaenLab.

Current application baseline:

`VIDEO FOUNDATION → FIELD → PEOPLE → POSITION → ROLE → ANALYTICS`

At this stage, only the video foundation exists. The next objective is FIELD calibration.

The immediate goal is **not** to detect players, teams, referees, tracking, or analytics.

The goal is:

> Given a football match recorded with Veo, estimate a reliable mapping between the video image and a canonical football pitch so that the visible field boundaries and markings can be projected accurately.

The first implementation will be **human-assisted**. Automation will be added later.

---

## 1. Main objective

For a selected video frame, obtain a transformation that relates image coordinates to real pitch coordinates.

The main result should be a homography:

- `H_pitch_to_image`
- `H_image_to_pitch`

This should allow:

```text
image pixel (u, v)
        ↓
pitch position (x, y) in metres
```

and:

```text
pitch position (x, y)
        ↓
image pixel (u, v)
```

Once this works correctly, RealJaenLab can project the complete geometry of the football field over the Veo image.

The resulting overlay should include, when relevant:

- touchlines;
- goal lines;
- halfway line;
- penalty areas;
- goal areas;
- center circle;
- penalty arcs;
- penalty spots;
- field limits.

A line does not need to be visible in the frame to be reconstructed once a reliable homography exists.

---

## 2. Important design decision: human-assisted first

The first version should **not try to be fully automatic**.

The user of the RealJaen Supervisor will provide the initial geometric information required to calculate the homography.

This is intentional.

The priority is:

1. obtain a geometrically correct result;
2. make the manual interaction fast and comfortable;
3. validate that a homography can represent Veo footage accurately;
4. automate progressively afterwards.

The first version should therefore optimize the user's workflow rather than trying to remove the user immediately.

### UX requirement

Creating a calibration must not become a 20–30 minute annotation task.

A reasonable initial target is:

> A useful manual calibration should normally take around 1–3 minutes once the workflow is familiar.

This target is not strict during the first prototype, but the interface must be designed with this direction in mind.

The system should avoid asking the user to mark dozens of arbitrary points.

---

## 3. Preferred manual interaction

The user should work with **semantic football markings**, not generic correspondence points whenever possible.

Example workflow:

```text
Select "Halfway line"
        ↓
click two or more positions along that line

Select "Near touchline"
        ↓
click two or more positions

Select "Penalty area front line"
        ↓
click two or more positions
```

Potential annotations:

- touchline;
- goal line;
- halfway line;
- penalty-area boundaries;
- goal-area boundaries;
- center circle;
- penalty arc;
- known intersections when clearly visible.

The application should know the corresponding geometry in pitch coordinates.

This is preferable to a workflow such as:

```text
Point 1
Point 2
Point 3
Point 4
Point 5
...
```

because semantic lines are easier for a human to identify and provide stronger geometric constraints.

---

## 4. Minimal useful calibration

The solver should not require the same annotation pattern in every frame.

A Veo frame may show:

- midfield but no goal area;
- one penalty area but no halfway line;
- only part of a touchline;
- several lines with no visible intersections;
- center-circle fragments;
- strong perspective or zoom.

The calibration system should therefore accept different combinations of geometric constraints.

Possible input types:

### Point correspondence

```text
known pitch point ↔ image point
```

Useful for intersections and clearly identifiable landmarks.

### Line correspondence

```text
known pitch line ↔ observed image line
```

This should be a first-class input type.

### Curve correspondence

Later versions may also use:

- center circle;
- penalty arcs.

A projected pitch circle becomes a conic in the image, so these features can provide additional constraints.

---

## 5. Canonical pitch model

RealJaenLab should maintain an explicit football-pitch geometry in real units.

Example:

```text
length = 105 m
width  = 68 m
```

The model should define all markings mathematically rather than relying on a background image.

Conceptually:

```text
Pitch
├── outer boundary
├── halfway line
├── center circle
├── center spot
├── left penalty area
├── right penalty area
├── left goal area
├── right goal area
├── penalty spots
└── penalty arcs
```

All relevant coordinates should be stored in metres.

### Pitch dimensions

The architecture should allow pitch dimensions to vary by venue.

Eventually the library may associate a venue with:

```text
pitch_length
pitch_width
```

However, this does not need to be implemented before the first calibration prototype if it slows down the initial experiment.

---

## 6. First technical milestone: prove the projection model

Before implementing automatic line detection, motion propagation, or neural networks, the first experiment must answer:

> Can one projective homography accurately represent the visible pitch geometry in Veo footage?

This must be tested on several manually calibrated frames.

For each frame:

1. annotate enough reliable field geometry;
2. solve the best possible homography;
3. project the complete pitch model;
4. inspect alignment across the entire visible field;
5. measure reprojection error.

This should be done on frames from different camera positions and preferably different matches.

### Why this matters

If one homography cannot align all visible markings at the same time, the problem may contain:

- lens distortion;
- panorama/dewarping effects;
- digital crop transformations;
- Veo stabilization;
- non-projective image processing.

If this occurs systematically, the architecture may need:

```text
Veo frame
   ↓
distortion / image model
   ↓
projective geometry
   ↓
homography
```

The system must discover this experimentally rather than assume a pure homography is sufficient.

---

## 7. Manual solver V1

The first real implementation should focus on producing the best possible calibration from human annotations.

The solver should eventually combine:

```text
point constraints
+
line constraints
+
optional curve constraints
```

and optimize the projection robustly.

A conceptual objective function could be:

```text
total_error =
    point_reprojection_error
  + line_reprojection_error
  + curve_reprojection_error
```

Robust losses should be considered so that one slightly inaccurate click does not destroy the solution.

The output should include:

```text
CalibrationResult
├── H_pitch_to_image
├── H_image_to_pitch
├── reprojection metrics
├── annotations used
├── visible pitch geometry
└── confidence / quality indicators
```

---

## 8. Calibration UI

The `CALIBRACIÓN` workspace should become the main editor for this process.

A desirable long-term layout is:

```text
┌─────────────────────────────────────────────────────────┐
│ VIDEO | PREVIEW | CALIBRACIÓN                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│                    CURRENT VIDEO FRAME                  │
│                                                         │
│                projected pitch overlay                 │
│                                                         │
├─────────────────────────────────────────────────────────┤
│ Selected feature: Halfway line                         │
│                                                         │
│ [Touchline] [Halfway] [Penalty area] [...]             │
│                                                         │
│ [Undo] [Clear feature] [Solve] [Accept]                │
├─────────────────────────────────────────────────────────┤
│ Error / confidence / annotation status                 │
└─────────────────────────────────────────────────────────┘
```

The first prototype can be simpler.

The critical UX principle is:

> the user should identify football features, while the application handles the geometry.

---

## 9. PREVIEW workspace

`PREVIEW` should be used to inspect intermediate and final calibration results without mixing them with editing controls.

Useful future views include:

```text
Original frame
Detected line evidence
Projected pitch model
Reprojection residuals
Confidence heatmap
Before / after refinement
```

This separation keeps `CALIBRACIÓN` focused on editing and `PREVIEW` focused on diagnosis.

---

## 10. Validation from the beginning

A calibration is not considered successful because it looks correct in one frame.

The system must be developed against multiple situations.

Minimum internal benchmark categories should eventually include:

- central camera view;
- left-side view;
- right-side view;
- wide angle;
- strong zoom;
- penalty-area view;
- midfield view;
- sunlight;
- shadow;
- different stadiums;
- different matches.

Most importantly:

> A solution must be validated on matches that were not used to design or tune it.

The target is not “works on Jaén vs Cacereño”.

The target is:

> works reliably across Real Jaén matches recorded with Veo.

---

## 11. Metrics

Visual inspection is useful, but every calibration should also expose measurable error.

Useful metrics may include:

- median line reprojection error in pixels;
- p95 reprojection error;
- maximum relevant error;
- number of supported markings;
- spatial coverage of annotations;
- calibration confidence.

Initial rough targets for 1080p footage can be investigated around:

```text
manual calibration median error: <= 3 px
manual calibration p95:          <= 8 px
```

These are starting targets, not permanent requirements.

They must be adjusted after measuring real Veo footage.

A calibration that is uncertain should be allowed to say so.

---

## 12. Future automation roadmap

Automation should be added only after the human-assisted calibration is geometrically reliable.

The intended progression is:

### Stage A — Human Calibration Gold

```text
human annotations
      ↓
high-quality homography
      ↓
full pitch overlay
```

Goal:

- establish a trustworthy geometric baseline.

### Stage B — Veo projection-model validation

Compare:

```text
homography only
```

against models that may include distortion or other corrections.

Goal:

- determine what image model Veo footage actually requires.

### Stage C — Field-line evidence

Build a detector that answers:

> Which pixels likely belong to football field markings?

Initially this can use classical computer vision.

Possible ingredients:

- brightness;
- saturation;
- local contrast;
- edge/ridge detection;
- morphology;
- Hough/LSD line proposals.

Later, semantic segmentation can replace or complement this.

At this stage the detector does **not** need to know which football line each pixel belongs to.

### Stage D — Homography refinement from lines

Given an approximate homography:

```text
approximate H
+
line evidence
      ↓
refined H
```

Project the canonical pitch and optimize its alignment with observed line evidence.

This is expected to become one of the most important automatic components.

### Stage E — Short-term temporal propagation

Use static visual content between nearby frames to estimate image motion.

Possible first methods:

- SIFT;
- AKAZE;
- Lucas-Kanade optical flow;
- RANSAC.

Later alternatives may include learned matchers if necessary.

Players and other moving objects should simply be treated as outliers.

### Stage F — Human-anchor to human-anchor propagation

Do not propagate indefinitely from one initial frame.

Instead:

```text
HUMAN anchor A
      ↓ forward

intermediate frames

      ↑ backward
HUMAN anchor B
```

Forward and backward estimates can be compared and refined using field lines.

### Stage G — Confidence estimation

Each frame calibration should eventually report confidence based on signals such as:

- line alignment;
- visual-motion inliers;
- match spatial distribution;
- forward/backward consistency;
- temporal smoothness;
- distance from the nearest human anchor;
- pitch plausibility.

Possible states:

```text
GOOD
REVIEW
BAD
```

The system should prefer requesting human intervention over silently accepting a bad calibration.

### Stage H — Full-match assisted calibration

Final workflow target:

```text
new match
    ↓
user calibrates a small number of frames
    ↓
system propagates/refines calibration
    ↓
low-confidence moments are highlighted
    ↓
user corrects only those moments
    ↓
reliable calibration for the match
```

---

## 13. Long-term human workflow target

The ideal system should not require manual calibration of thousands of frames.

A realistic design goal is:

```text
calibrate a frame
       ↓
automatic processing
       ↓
review only uncertain regions
       ↓
small correction
       ↓
continue
```

The exact number of human interventions per match is not yet known.

A future aspirational target could be roughly a handful of corrections per half or per match, depending on Veo camera behaviour.

The important requirement is that intervention remains **fast and guided**.

---

## 14. What is explicitly out of scope

During this FIELD calibration phase, do not mix in:

- player detection;
- referee detection;
- team classification;
- tracking;
- player identity;
- ball detection;
- player-position analytics;
- heatmaps;
- distance covered;
- tactical analysis.

The only problem being solved is:

> Recover and track the geometry of the football pitch in Veo footage.

---

## 15. Proposed development sequence

The recommended initial order is:

```text
EXP-001
Manual semantic calibration + high-quality homography

EXP-002
Validate homography-only model against Veo footage

EXP-003
Fast and comfortable calibration UX

EXP-004
Field-line evidence detector

EXP-005
Automatic line-based homography refinement

EXP-006
Short-term frame-to-frame propagation

EXP-007
Human-anchor segment propagation

EXP-008
Confidence and failure detection

EXP-009
Full-half / full-match assisted calibration

EXP-010
Validation on unseen Real Jaén matches
```

The experiment numbering is only organizational guidance and can change.

---

## 16. First implementation target

The next development session should start with the smallest useful problem:

> Create a fast manual calibration workflow for one frame and solve a high-quality homography from semantic football-field annotations.

Do not attempt automatic line detection in the first implementation.

The first milestone is reached when:

1. a user opens a frame in `CALIBRACIÓN`;
2. marks a small number of known football features;
3. presses Solve;
4. RealJaenLab projects the complete canonical pitch over the frame;
5. the overlay aligns accurately with visible field markings;
6. the process is quick enough to be practical.

Only after this baseline is reliable should automatic assistance be introduced.

---

## 17. Guiding principles

Throughout development, keep these principles:

**Correctness before automation.**

A human-assisted correct solution is more useful than an automatic unreliable one.

**Fast human interaction.**

Manual calibration must be semantic, guided, and minimal.

**Generalization over one-match success.**

A method is not solved because it works on one clip.

**Geometry and evidence are separate.**

Detected pixels provide evidence; the pitch model provides geometry.

**Use temporal continuity.**

Video contains much more information than isolated images.

**Failures must be visible.**

A low-confidence calibration is acceptable. A silent wrong calibration is not.

**Git history is the archive.**

Do not reintroduce old FIELD implementations merely because they existed before. New work should be justified by the new architecture.

---

## 18. Definition of success for FIELD calibration V1

FIELD Calibration V1 can be considered successful when:

- manual calibration is quick enough for practical use;
- projected pitch geometry aligns accurately with Veo frames;
- the method works across different camera views;
- calibration quality can be measured objectively;
- failures can be identified instead of silently accepted;
- the architecture supports future automatic line detection and temporal propagation;
- the solution generalizes to multiple Real Jaén matches rather than a single benchmark clip.

At that point, RealJaenLab will have a reliable geometric foundation on which later PEOPLE, POSITION, ROLE, and ANALYTICS layers can be built.

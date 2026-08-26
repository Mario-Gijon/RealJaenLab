#!/usr/bin/env python3
"""Diagnose visual-state-search aliasing against hidden HUMAN holdouts.

No production behavior is changed.

For each interior HUMAN anchor hidden from the LOO model, this script measures:

- exact hidden true-state distance from the linear temporal prior, in the SAME
  5-D PCA coordinates and per-dimension state scale used by visual search;
- raw paint objective at:
    1) temporal prior H,
    2) hidden true H,
    3) top visual-search candidate H;
- grid error of the top visual-search candidate.

This answers two key questions:

1. Did V1 simply search too far away from the temporal prior?
2. Does the current paint objective actually prefer a geometrically wrong
   semantic alias over the hidden true camera?

Hidden HUMAN H is used ONLY for diagnosis/evaluation.
Nothing is persisted.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from supervisor.field_camera_state_projective import (
    ProjectiveCameraStateConfig,
    build_projective_camera_state_model,
    evaluate_homography_visible_grid,
    homography_from_state,
    projective_jet_from_homography,
)
from supervisor.field_camera_state_visual_search import (
    VisualStateSearchConfig,
    prepare_paint_image,
    score_homography_paint,
    search_projective_state_from_image,
)
from supervisor.field_pitch_model import build_pitch_model


REVIEW = Path("supervisor/data/reviews/benchmark_easy_2min.json")


def calibration_for(calibrations, frame):
    for calibration in calibrations:
        if not isinstance(calibration, dict):
            continue
        try:
            same = int(calibration.get("frame")) == int(frame)
        except (TypeError, ValueError):
            continue
        if (
            same
            and calibration.get("source") == "manual"
            and calibration.get("purpose") == "match_setup"
            and isinstance(calibration.get("homography"), list)
        ):
            return calibration
    return None


def state_map(model):
    return {
        int(anchor["frame"]): np.asarray(anchor["state"], dtype=np.float64)
        for anchor in model.get("anchors", [])
        if isinstance(anchor, dict) and anchor.get("state") is not None
    }


def temporal_prior(model, frame):
    states = state_map(model)
    frames = sorted(states)
    left = [value for value in frames if value < frame]
    right = [value for value in frames if value > frame]
    if not left or not right:
        return None

    a = left[-1]
    b = right[0]
    alpha = (frame - a) / float(b - a)
    state = (1.0 - alpha) * states[a] + alpha * states[b]
    return a, b, alpha, state


def state_from_hidden_h(model, homography):
    feature = projective_jet_from_homography(
        homography,
        image_width=model["image_width"],
        image_height=model["image_height"],
        pitch_length_m=model["pitch_length_m"],
        pitch_width_m=model["pitch_width_m"],
        maximum_condition=model["config"]["maximum_h_condition"],
    )
    if feature is None:
        return None

    feature = np.asarray(feature, dtype=np.float64)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    std = np.asarray(model["feature_std"], dtype=np.float64)
    pca_mean = np.asarray(model["pca_mean"], dtype=np.float64)
    components = np.asarray(model["pca_components"], dtype=np.float64)

    z = (feature - mean) / std
    return (z - pca_mean) @ components.T


def decode_h(model, state):
    result = homography_from_state(model, state)
    if result.get("status") != "ok":
        return None
    return np.asarray(result["homography"], dtype=np.float64)


def state_radius(state, prior, scale):
    return float(np.sqrt(np.mean(((state - prior) / scale) ** 2)))


def grid_error(predicted, truth, width, height, length_m, pitch_width_m):
    result = evaluate_homography_visible_grid(
        predicted,
        truth,
        image_width=width,
        image_height=height,
        pitch_length_m=length_m,
        pitch_width_m=pitch_width_m,
        config=ProjectiveCameraStateConfig(dimensions=5),
    )
    return result if result.get("status") == "ok" else None


def fmt(value, digits=3):
    return "—" if value is None else f"{float(value):.{digits}f}"


def main():
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    field = review["field"]
    calibrations = field.get("calibrations", [])
    strong = sorted({int(frame) for frame in field["match_setup"]["completed_frames"]})
    interior = strong[1:-1]

    length_m = float(field["pitch_model"]["length_m"])
    pitch_width_m = float(field["pitch_model"]["width_m"])
    pitch_model = build_pitch_model(length_m, pitch_width_m)

    video_path = Path("input") / review["video"]["filename"]
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"No se pudo abrir {video_path}")

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        or review["video"].get("width")
        or 0
    )
    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        or review["video"].get("height")
        or 0
    )

    images = {}
    for frame in interior:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        if ok and image is not None:
            images[frame] = image
    capture.release()

    cfg = VisualStateSearchConfig()

    print()
    print("VISUAL STATE SEARCH · ALIASING DIAGNOSIS")
    print("Strong anchors:", strong)
    print("Interior holdouts:", interior)
    print(f"Pitch: {length_m:g} × {pitch_width_m:g} m")
    print(
        f"Current search: max prior radius={cfg.maximum_prior_radius:.2f} · "
        f"temporal penalty={cfg.temporal_penalty:.3f}"
    )
    print()

    rows = []

    for frame in interior:
        hidden = calibration_for(calibrations, frame)
        image = images.get(frame)
        if hidden is None or image is None:
            continue

        model = build_projective_camera_state_model(
            calibrations,
            [anchor for anchor in strong if anchor != frame],
            image_width=width,
            image_height=height,
            pitch_length_m=length_m,
            pitch_width_m=pitch_width_m,
            config=ProjectiveCameraStateConfig(dimensions=5),
        )
        if model.get("status") != "ok":
            continue

        temporal = temporal_prior(model, frame)
        if temporal is None:
            continue

        left, right, alpha, prior_state = temporal
        true_h = np.asarray(hidden["homography"], dtype=np.float64)
        true_state = state_from_hidden_h(model, true_h)
        if true_state is None:
            continue

        search = search_projective_state_from_image(
            model,
            prior_state,
            image,
            pitch_model,
            query_frame=frame,
            config=cfg,
        )

        scale = np.asarray(search.get("state_scale"), dtype=np.float64)
        true_radius = state_radius(true_state, prior_state, scale)

        prior_h = decode_h(model, prior_state)
        paint = prepare_paint_image(image)

        prior_paint = score_homography_paint(
            paint,
            prior_h,
            pitch_model,
            width,
            height,
            config=cfg,
        )
        true_paint = score_homography_paint(
            paint,
            true_h,
            pitch_model,
            width,
            height,
            config=cfg,
        )

        true_objective = (
            float(true_paint["score"])
            - cfg.temporal_penalty * true_radius
        )

        top = search.get("candidates", [None])[0]
        top_radius = None
        top_paint = None
        top_objective = None
        top_grid = None
        top_state_distance_to_true = None

        if isinstance(top, dict):
            top_radius = float(top["prior_radius"])
            top_paint = float(top["paint_score"])
            top_objective = float(top["objective"])

            top_h = np.asarray(top["homography"], dtype=np.float64)
            top_grid_result = grid_error(
                top_h,
                true_h,
                width,
                height,
                length_m,
                pitch_width_m,
            )
            if top_grid_result is not None:
                top_grid = float(top_grid_result["median_px"])

            top_state = np.asarray(top["state"], dtype=np.float64)
            top_state_distance_to_true = float(
                np.sqrt(
                    np.mean(
                        ((top_state - true_state) / scale) ** 2
                    )
                )
            )

        prior_grid_result = grid_error(
            prior_h,
            true_h,
            width,
            height,
            length_m,
            pitch_width_m,
        )
        prior_grid = (
            float(prior_grid_result["median_px"])
            if prior_grid_result is not None
            else None
        )

        row = {
            "frame": frame,
            "bracket": (left, right),
            "alpha": alpha,
            "true_radius": true_radius,
            "prior_grid": prior_grid,
            "prior_paint": float(prior_paint["score"]),
            "true_paint": float(true_paint["score"]),
            "true_objective": true_objective,
            "top_radius": top_radius,
            "top_paint": top_paint,
            "top_objective": top_objective,
            "top_grid": top_grid,
            "top_to_true_radius": top_state_distance_to_true,
        }
        rows.append(row)

        prefers_false = (
            top_objective is not None
            and top_objective > true_objective
        )

        print(
            f"f {frame:>4} · true-radius={fmt(true_radius)} · "
            f"prior-grid={fmt(prior_grid,1)}px"
        )
        print(
            f"       paint prior={fmt(row['prior_paint'])} · "
            f"TRUE={fmt(row['true_paint'])} · "
            f"TRUE objective={fmt(true_objective)}"
        )
        print(
            f"       TOP radius={fmt(top_radius)} · "
            f"paint={fmt(top_paint)} · objective={fmt(top_objective)} · "
            f"grid={fmt(top_grid,1)}px · "
            f"top↔true radius={fmt(top_state_distance_to_true)}"
        )
        print(
            f"       objective prefers false top over hidden truth: "
            f"{'YES' if prefers_false else 'NO'}"
        )

    if rows:
        true_radii = np.asarray([row["true_radius"] for row in rows], dtype=np.float64)
        false_preferences = sum(
            1
            for row in rows
            if row["top_objective"] is not None
            and row["top_objective"] > row["true_objective"]
        )
        print()
        print("SUMMARY")
        print(
            f"hidden TRUE state radius: median={np.median(true_radii):.3f} · "
            f"max={np.max(true_radii):.3f}"
        )
        print(
            f"current visual objective prefers a false top candidate in "
            f"{false_preferences}/{len(rows)} holdouts"
        )

        if np.max(true_radii) < cfg.maximum_prior_radius * 0.65:
            print(
                "The current radius is materially wider than the hidden truth requires."
            )

        if false_preferences:
            print(
                "Paint score is semantically aliased: tightening radius alone may help, "
                "but the visual objective also needs stronger topology/temporal constraints."
            )

    print()
    print("Hidden HUMAN H was used only for diagnosis/evaluation.")
    print("No se ha modificado la review.")


if __name__ == "__main__":
    main()

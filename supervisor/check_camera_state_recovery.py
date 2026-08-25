#!/usr/bin/env python3
"""Smoke-check safe camera recovery policy on frame 115.

Expected result:
    mode=primary_points_lines
    ignored_arcs=1
    confidence=good
    accepted=True
    state shift around the validated points-only result (~0.305 z RMS)

Nothing is written to the review.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from supervisor.field_camera_state_projective import (
    ProjectiveCameraStateConfig,
    build_projective_camera_state_model,
    evaluate_homography_manual_points,
)
from supervisor.field_camera_state_recovery import (
    recover_camera_from_projective_state,
)
from supervisor.field_camera_state_refinement import (
    should_accept_refined_camera,
)


REVIEW = Path(
    "supervisor/data/reviews/benchmark_easy_2min.json"
)


def find_calibration(calibrations, frame):
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
        ):
            return calibration
    return None


def main():
    review = json.loads(
        REVIEW.read_text(encoding="utf-8")
    )

    field = review["field"]
    pitch = field["pitch_model"]
    setup = field["match_setup"]
    calibrations = field.get("calibrations", [])

    completed = sorted(
        int(x)
        for x in setup.get("completed_frames", [])
    )

    video = (
        Path("input")
        / review["video"]["filename"]
    )

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"No se pudo abrir {video}")

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
    capture.release()

    model = build_projective_camera_state_model(
        calibrations,
        completed,
        image_width=width,
        image_height=height,
        pitch_length_m=float(pitch["length_m"]),
        pitch_width_m=float(pitch["width_m"]),
        config=ProjectiveCameraStateConfig(
            dimensions=5
        ),
    )

    if model.get("status") != "ok":
        raise SystemExit(model)

    states = {
        int(anchor["frame"]): np.asarray(
            anchor["state"],
            dtype=float,
        )
        for anchor in model["anchors"]
    }

    alpha = (115 - 87) / float(175 - 87)
    prior = (
        (1.0 - alpha) * states[87]
        + alpha * states[175]
    )

    calibration = find_calibration(
        calibrations,
        115,
    )
    if calibration is None:
        raise SystemExit(
            "No existe calibración parcial del 115."
        )

    points = []
    lines = []
    arcs = []

    for constraint in calibration.get(
        "constraints",
        [],
    ):
        kind = constraint.get("type")
        if kind == "point":
            points.append(constraint)
        elif kind == "line":
            lines.append(constraint)
        elif kind == "arc":
            arcs.append(constraint)

    result = recover_camera_from_projective_state(
        model,
        prior,
        points,
        lines,
        arcs,
        prior_confidence="good",
    )

    print()
    print("=" * 84)
    print("SAFE PROJECTIVE CAMERA RECOVERY · FRAME 115")
    print("=" * 84)
    print(
        f"constraints={len(points)}p/"
        f"{len(lines)}l/{len(arcs)}a"
    )
    print(
        f"status={result.get('status')} · "
        f"confidence={result.get('confidence')} · "
        f"accepted={should_accept_refined_camera(result)}"
    )
    print("recovery:", result.get("recovery"))

    if result.get("status") == "ok":
        print(
            f"shift-z={result['state_shift_z_rms']:.3f} · "
            f"fitσ={result['fit_rms_sigma']:.3f}"
        )

        evaluation = evaluate_homography_manual_points(
            result["homography"],
            calibration,
        )

        if evaluation.get("status") == "ok":
            print(
                f"HUMAN points median="
                f"{evaluation['median_px']:.2f}px · "
                f"P90={evaluation['p90_px']:.2f}px · "
                f"max={evaluation['max_px']:.2f}px"
            )

        print("H:")
        print(
            json.dumps(
                result["homography"],
                indent=2,
            )
        )

    print()
    print("No se ha modificado la review.")


if __name__ == "__main__":
    main()

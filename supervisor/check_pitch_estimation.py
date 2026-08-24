"""Synthetic regression checks for multi-frame physical pitch estimation."""

from __future__ import annotations

import math

import numpy as np

from supervisor.field_pitch_estimation import PitchEstimationConfig, estimate_pitch_dimensions
from supervisor.field_pitch_model import build_pitch_model


IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
STRONG_POINT_SEMANTICS = (
    "top_left_corner", "top_right_corner", "bottom_left_corner", "bottom_right_corner",
    "left_penalty_top", "right_penalty_bottom", "left_penalty_spot", "right_penalty_spot", "centre_mark",
)
STRONG_LINE_SEMANTICS = (
    "bottom_touchline", "top_touchline", "halfway_line", "left_penalty_front", "right_goal_front",
)
CAMERAS = (
    np.asarray([[13.0, 1.5, 950.0], [0.5, 13.0, 540.0], [0.0010, -0.0012, 1.0]]),
    np.asarray([[10.0, -2.0, 1020.0], [1.4, 12.0, 460.0], [-0.0008, 0.0010, 1.0]]),
    np.asarray([[15.0, 0.8, 880.0], [-1.0, 14.0, 560.0], [0.0005, 0.0007, 1.0]]),
)


def _project(point, homography):
    projected = homography @ np.asarray([point["x"], point["y"], 1.0])
    return {"x": float(projected[0] / projected[2]), "y": float(projected[1] / projected[2])}


def _strong_observations(length_m: float, width_m: float):
    model = build_pitch_model(length_m, width_m)
    landmarks = {item["id"]: item["world"] for item in model["landmarks"]}
    lines = {item["id"]: item["world"] for item in model["calibration_lines"]}
    observations = []
    for frame, homography in enumerate(CAMERAS, start=100):
        constraints = [
            {"type": "point", "semantic": semantic, "image": _project(landmarks[semantic], homography)}
            for semantic in STRONG_POINT_SEMANTICS
        ]
        constraints.extend(
            {
                "type": "line",
                "semantic": semantic,
                "image": {
                    "start": _project(lines[semantic]["start"], homography),
                    "end": _project(lines[semantic]["end"], homography),
                },
            }
            for semantic in STRONG_LINE_SEMANTICS
        )
        observations.append({"frame": frame, "image_width": IMAGE_WIDTH, "image_height": IMAGE_HEIGHT, "constraints": constraints})
    return observations


def _weak_observation():
    model = build_pitch_model(105.0, 68.0)
    landmarks = {item["id"]: item["world"] for item in model["landmarks"]}
    semantics = ("left_penalty_spot", "right_penalty_spot", "centre_circle_halfway_top", "centre_circle_halfway_bottom")
    return [{
        "frame": 500,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "constraints": [
            {"type": "point", "semantic": semantic, "image": _project(landmarks[semantic], CAMERAS[0])}
            for semantic in semantics
        ],
    }]


def _config(length_m: float, width_m: float) -> PitchEstimationConfig:
    # The production defaults retain their broader 100–110 / 64–75 range.
    # This focused synthetic check exercises the same 1.0 m / 0.1 m two-stage
    # search without requiring hundreds of slow numerical solves per case.
    return PitchEstimationConfig(
        length_min_m=length_m - 1.0,
        length_max_m=length_m + 1.0,
        width_min_m=width_m - 1.0,
        width_max_m=width_m + 1.0,
        fine_radius_m=0.5,
    )


def _assert_recovery(length_m: float, width_m: float) -> None:
    result = estimate_pitch_dimensions(_strong_observations(length_m, width_m), config=_config(length_m, width_m))
    assert result["status"] == "estimated", result
    estimate = result["pitch_model"]
    assert math.isclose(estimate["length_m"], length_m, abs_tol=0.1), estimate
    assert math.isclose(estimate["width_m"], width_m, abs_tol=0.1), estimate
    assert result["confidence"]["length_constrained"], result["confidence"]
    assert result["confidence"]["width_constrained"], result["confidence"]
    assert "bottom_touchline" in result["holdout_summary"]["key_line_holdouts"]


def run_regression_check() -> None:
    _assert_recovery(105.0, 68.0)
    _assert_recovery(107.0, 67.5)

    weak = estimate_pitch_dimensions(_weak_observation(), config=_config(105.0, 68.0))
    assert not weak["confidence"]["length_constrained"], weak["confidence"]
    assert not weak["confidence"]["width_constrained"], weak["confidence"]
    assert weak["confidence"]["length_interval_m"][1] - weak["confidence"]["length_interval_m"][0] >= 1.0
    assert weak["confidence"]["width_interval_m"][1] - weak["confidence"]["width_interval_m"][0] >= 1.0


if __name__ == "__main__":
    run_regression_check()
    print("Pitch estimation synthetic regression OK")

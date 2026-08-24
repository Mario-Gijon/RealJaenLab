"""Synthetic checks for Phase-1 FIELD pitch setup."""

from __future__ import annotations

import math

import numpy as np

from supervisor.field_pitch_model import (
    build_pitch_model,
)
from supervisor.field_pitch_setup import (
    PitchSetupConfig,
    estimate_pitch_setup,
    next_pitch_measurement_frame,
    select_pitch_measurement_frames,
)


IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080


CAMERAS = (
    np.asarray(
        [
            [13.0, 1.5, 950.0],
            [0.5, 13.0, 540.0],
            [0.0010, -0.0012, 1.0],
        ],
        dtype=np.float64,
    ),
    np.asarray(
        [
            [10.0, -2.0, 1020.0],
            [1.4, 12.0, 460.0],
            [-0.0008, 0.0010, 1.0],
        ],
        dtype=np.float64,
    ),
    np.asarray(
        [
            [15.0, 0.8, 880.0],
            [-1.0, 14.0, 560.0],
            [0.0005, 0.0007, 1.0],
        ],
        dtype=np.float64,
    ),
)


POINT_SEMANTICS = (
    "top_left_corner",
    "top_right_corner",
    "bottom_left_corner",
    "bottom_right_corner",
    "centre_mark",
    "left_penalty_top",
    "left_penalty_bottom",
    "right_penalty_top",
    "right_penalty_bottom",
    "left_goal_top",
    "right_goal_bottom",
    "left_penalty_spot",
    "right_penalty_spot",
)


LINE_SEMANTICS = (
    "top_touchline",
    "bottom_touchline",
    "halfway_line",
    "left_goal_line",
    "right_goal_line",
    "left_penalty_front",
    "right_penalty_front",
)


def _project(
    point: dict[str, float],
    homography: np.ndarray,
) -> dict[str, float]:
    source = np.asarray(
        [
            point["x"],
            point["y"],
            1.0,
        ],
        dtype=np.float64,
    )

    projected = homography @ source

    return {
        "x": float(
            projected[0] / projected[2]
        ),
        "y": float(
            projected[1] / projected[2]
        ),
    }


def _strong_observations(
    length_m: float,
    width_m: float,
) -> list[dict]:
    model = build_pitch_model(
        length_m,
        width_m,
    )

    landmarks = {
        item["id"]: item["world"]
        for item in model["landmarks"]
    }

    lines = {
        item["id"]: item["world"]
        for item
        in model["calibration_lines"]
    }

    observations = []

    for index, homography in enumerate(
        CAMERAS
    ):
        constraints = []

        for semantic in POINT_SEMANTICS:
            constraints.append(
                {
                    "type": "point",
                    "semantic": semantic,
                    "image": _project(
                        landmarks[semantic],
                        homography,
                    ),
                    # Deliberately wrong.
                    # field_pitch_estimation must
                    # regenerate world coordinates.
                    "world": {
                        "x": 9999.0,
                        "y": -9999.0,
                    },
                }
            )

        for semantic in LINE_SEMANTICS:
            world = lines[semantic]

            constraints.append(
                {
                    "type": "line",
                    "semantic": semantic,
                    "image": {
                        "start": _project(
                            world["start"],
                            homography,
                        ),
                        "end": _project(
                            world["end"],
                            homography,
                        ),
                    },
                    # Also deliberately wrong.
                    "world": {
                        "start": {
                            "x": 5000.0,
                            "y": 5000.0,
                        },
                        "end": {
                            "x": 6000.0,
                            "y": 6000.0,
                        },
                    },
                }
            )

        observations.append(
            {
                "frame": 100 + index * 500,
                "image_width": IMAGE_WIDTH,
                "image_height": IMAGE_HEIGHT,
                "constraints": constraints,
            }
        )

    return observations


def _weak_observations() -> list[dict]:
    model = build_pitch_model(
        105.0,
        68.0,
    )

    landmarks = {
        item["id"]: item["world"]
        for item in model["landmarks"]
    }

    semantics = (
        "left_penalty_spot",
        "right_penalty_spot",
        "centre_circle_halfway_top",
        "centre_circle_halfway_bottom",
    )

    constraints = [
        {
            "type": "point",
            "semantic": semantic,
            "image": _project(
                landmarks[semantic],
                CAMERAS[0],
            ),
        }
        for semantic in semantics
    ]

    return [
        {
            "frame": 500,
            "image_width": IMAGE_WIDTH,
            "image_height": IMAGE_HEIGHT,
            "constraints": constraints,
        }
    ]


def _focused_config(
    length_m: float,
    width_m: float,
) -> PitchSetupConfig:
    # Keep regression runtime practical while
    # exercising exactly the same staged algorithm.
    return PitchSetupConfig(
        reference_length_m=105.0,
        length_min_m=length_m - 1.5,
        length_max_m=length_m + 1.5,
        width_min_m=width_m - 2.0,
        width_max_m=width_m + 2.0,
        coarse_step_m=1.0,
        fine_step_m=0.1,
        refinement_radius_m=1.0,
        minimum_usable_frames=3,
    )


def _assert_pitch_recovery(
    length_m: float,
    width_m: float,
) -> None:
    result = estimate_pitch_setup(
        _strong_observations(
            length_m,
            width_m,
        ),
        config=_focused_config(
            length_m,
            width_m,
        ),
    )

    assert (
        result["status"]
        == "ready_to_confirm"
    ), result

    estimate = result["pitch_model"]

    assert math.isclose(
        estimate["length_m"],
        length_m,
        abs_tol=0.15,
    ), estimate

    assert math.isclose(
        estimate["width_m"],
        width_m,
        abs_tol=0.15,
    ), estimate

    assert result["confidence"][
        "length_constrained"
    ], result["confidence"]

    assert result["confidence"][
        "width_constrained"
    ], result["confidence"]


def _check_weak_case() -> None:
    result = estimate_pitch_setup(
        _weak_observations(),
        config=PitchSetupConfig(
            length_min_m=104.0,
            length_max_m=106.0,
            width_min_m=67.0,
            width_max_m=69.0,
            refinement_radius_m=0.5,
        ),
    )

    assert (
        result["status"]
        == "needs_more_evidence"
    ), result

    assert (
        result["needs_more_evidence"][
            "length"
        ]
        or result[
            "needs_more_evidence"
        ]["width"]
        or result[
            "needs_more_evidence"
        ]["usable_frames"]
    ), result


def _fake_scan() -> dict:
    return {
        "status": "ok",
        "recommendations": [
            {
                "frame": 100,
                "quality_score": 0.70,
                "geometry": {
                    "line_family_count": 2,
                    "intersection_count": 5,
                    "line_count": 8,
                    "sharpness": 0.70,
                },
            },
            {
                "frame": 700,
                "quality_score": 0.92,
                "geometry": {
                    "line_family_count": 3,
                    "intersection_count": 10,
                    "line_count": 13,
                    "sharpness": 0.88,
                },
            },
            {
                "frame": 1300,
                "quality_score": 0.81,
                "geometry": {
                    "line_family_count": 3,
                    "intersection_count": 8,
                    "line_count": 10,
                    "sharpness": 0.80,
                },
            },
            {
                "frame": 1900,
                "quality_score": 0.76,
                "geometry": {
                    "line_family_count": 2,
                    "intersection_count": 7,
                    "line_count": 9,
                    "sharpness": 0.75,
                },
            },
            {
                "frame": 2500,
                "quality_score": 0.84,
                "geometry": {
                    "line_family_count": 3,
                    "intersection_count": 9,
                    "line_count": 11,
                    "sharpness": 0.82,
                },
            },
            {
                "frame": 3100,
                "quality_score": 0.73,
                "geometry": {
                    "line_family_count": 2,
                    "intersection_count": 6,
                    "line_count": 8,
                    "sharpness": 0.72,
                },
            },
        ],
    }


def _check_frame_selection() -> None:
    scan = _fake_scan()

    first = select_pitch_measurement_frames(
        scan,
        target_count=4,
        maximum_count=6,
    )

    second = select_pitch_measurement_frames(
        scan,
        target_count=4,
        maximum_count=6,
    )

    assert (
        first["selected_frames"]
        == second["selected_frames"]
    )

    assert len(
        first["selected_frames"]
    ) == 4

    assert len(
        set(first["selected_frames"])
    ) == 4

    used = set(
        first["selected_frames"]
    )

    next_frame = (
        next_pitch_measurement_frame(
            scan,
            used,
            maximum_count=6,
        )
    )

    assert next_frame is not None
    assert next_frame["frame"] not in used

    used.add(next_frame["frame"])

    next_again = (
        next_pitch_measurement_frame(
            scan,
            used,
            maximum_count=6,
        )
    )

    assert next_again is not None
    assert next_again["frame"] not in used

    used.add(next_again["frame"])

    assert (
        next_pitch_measurement_frame(
            scan,
            used,
            maximum_count=6,
        )
        is None
    )


def run_regression_check() -> None:
    _check_frame_selection()

    _assert_pitch_recovery(
        105.0,
        68.0,
    )

    _assert_pitch_recovery(
        107.0,
        67.5,
    )

    _check_weak_case()


if __name__ == "__main__":
    run_regression_check()
    print(
        "Pitch setup synthetic regression OK"
    )
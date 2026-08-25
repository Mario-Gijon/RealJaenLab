#!/usr/bin/env python3
"""Smoke check for semantic-balanced Projective Camera State refinement V2.

Runs the exact frame-115 ablation:
- points only;
- points + arc;
- arc only.

Expected behavior after V2:
- points only remains close to the validated result;
- adding the arc must no longer explode state shift merely because it contains
  many samples;
- arc-only remains guarded if it cannot support a trustworthy state.

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
    evaluate_homography_manual_points,
)
from supervisor.field_camera_state_refinement import (
    ProjectiveStateRefinementConfig,
    refine_projective_state_with_constraints,
    should_accept_refined_camera,
)


REVIEW_PATH = Path(
    "supervisor/data/reviews/benchmark_easy_2min.json"
)


def _find_calibration(
    calibrations,
    frame,
):
    for calibration in calibrations:
        if not isinstance(
            calibration,
            dict,
        ):
            continue

        try:
            same_frame = (
                int(
                    calibration.get(
                        "frame"
                    )
                )
                == int(
                    frame
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if (
            same_frame
            and calibration.get(
                "source"
            )
            == "manual"
            and calibration.get(
                "purpose"
            )
            == "match_setup"
        ):
            return calibration

    return None


def _state_map(
    model,
):
    return {
        int(
            anchor[
                "frame"
            ]
        ): np.asarray(
            anchor[
                "state"
            ],
            dtype=np.float64,
        )
        for anchor in model[
            "anchors"
        ]
    }


def main():
    review = json.loads(
        REVIEW_PATH.read_text(
            encoding="utf-8"
        )
    )

    field = review[
        "field"
    ]
    pitch = field[
        "pitch_model"
    ]
    setup = field[
        "match_setup"
    ]
    calibrations = field.get(
        "calibrations",
        [],
    )

    completed = sorted(
        int(
            value
        )
        for value in setup.get(
            "completed_frames",
            [],
        )
    )

    video_path = (
        Path(
            "input"
        )
        / review[
            "video"
        ][
            "filename"
        ]
    )

    capture = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not capture.isOpened():
        raise SystemExit(
            f"No se pudo abrir {video_path}"
        )

    try:
        width = int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
            or review[
                "video"
            ].get(
                "width"
            )
            or 0
        )
        height = int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
            or review[
                "video"
            ].get(
                "height"
            )
            or 0
        )
    finally:
        capture.release()

    model = (
        build_projective_camera_state_model(
            calibrations,
            completed,
            image_width=width,
            image_height=height,
            pitch_length_m=float(
                pitch[
                    "length_m"
                ]
            ),
            pitch_width_m=float(
                pitch[
                    "width_m"
                ]
            ),
            config=ProjectiveCameraStateConfig(
                dimensions=5
            ),
        )
    )

    if model.get(
        "status"
    ) != "ok":
        raise SystemExit(
            "No se pudo construir Projective Camera State 5D."
        )

    states = _state_map(
        model
    )

    alpha = (
        115 - 87
    ) / float(
        175 - 87
    )

    prior = (
        (
            1.0
            - alpha
        )
        * states[
            87
        ]
        + alpha
        * states[
            175
        ]
    )

    calibration = _find_calibration(
        calibrations,
        115,
    )

    if calibration is None:
        raise SystemExit(
            "No existe calibración 115."
        )

    points = [
        constraint
        for constraint in calibration.get(
            "constraints",
            [],
        )
        if isinstance(
            constraint,
            dict,
        )
        and constraint.get(
            "type"
        )
        == "point"
    ]

    lines = [
        constraint
        for constraint in calibration.get(
            "constraints",
            [],
        )
        if isinstance(
            constraint,
            dict,
        )
        and constraint.get(
            "type"
        )
        == "line"
    ]

    arcs = [
        constraint
        for constraint in calibration.get(
            "constraints",
            [],
        )
        if isinstance(
            constraint,
            dict,
        )
        and constraint.get(
            "type"
        )
        == "arc"
    ]

    tests = [
        (
            "POINTS ONLY",
            points,
            [],
            [],
        ),
        (
            "POINTS + ARC",
            points,
            [],
            arcs,
        ),
        (
            "ARC ONLY",
            [],
            [],
            arcs,
        ),
    ]

    print()
    print(
        "=" * 82
    )
    print(
        "PROJECTIVE CAMERA STATE REFINEMENT V2 · ARC SEMANTIC BALANCING"
    )
    print(
        "=" * 82
    )
    print(
        f"frame 115 · bracket 87→175 · alpha={alpha:.3f}"
    )
    print(
        f"stored constraints: {len(points)}p/{len(lines)}l/{len(arcs)}a"
    )

    for name, test_points, test_lines, test_arcs in tests:
        result = (
            refine_projective_state_with_constraints(
                model,
                prior,
                test_points,
                test_lines,
                test_arcs,
                prior_confidence="good",
                config=ProjectiveStateRefinementConfig(),
            )
        )

        print()
        print(
            "-" * 82
        )
        print(
            name
        )
        print(
            "-" * 82
        )
        print(
            "status:",
            result.get(
                "status"
            ),
        )

        if result.get(
            "status"
        ) != "ok":
            print(
                json.dumps(
                    result,
                    indent=2,
                    ensure_ascii=False,
                )
            )
            continue

        print(
            f"confidence={result['confidence']} · "
            f"accepted={should_accept_refined_camera(result)}"
        )
        print(
            f"shift-z={result['state_shift_z_rms']:.3f} · "
            f"balanced fitσ={result['fit_rms_sigma']:.3f}"
        )
        print(
            f"point error={result['diagnostics']['point_error_px']} · "
            f"line error={result['diagnostics']['line_error_px']}"
        )

        arc_info = (
            result[
                "diagnostics"
            ][
                "arc_weighting"
            ]
        )
        print(
            "arc weighting:",
            arc_info,
        )

        evaluation = (
            evaluate_homography_manual_points(
                result[
                    "homography"
                ],
                calibration,
            )
        )

        if evaluation.get(
            "status"
        ) == "ok":
            print(
                f"5 HUMAN points: "
                f"median={evaluation['median_px']:.2f}px · "
                f"P90={evaluation['p90_px']:.2f}px · "
                f"max={evaluation['max_px']:.2f}px"
            )

    print()
    print(
        "No se ha modificado la review."
    )


if __name__ == "__main__":
    main()

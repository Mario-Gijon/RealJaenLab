"""Synthetic regression for HUMAN point guard V2.

Test A reproduces the useful case:
- 10 HUMAN points are clustered away from a distant touchline;
- small click noise makes point-only extrapolation imperfect;
- a CORRECT semantic touchline should be allowed to improve that extrapolation
  while every projected point remains close to its click.

Test B supplies a contradictory line and verifies that the HUMAN click envelope
is still respected.
"""

from __future__ import annotations

import math

import numpy as np

from supervisor.field_camera_calibration import (
    CameraCalibrationConfig,
    solve_field_camera_calibration,
)


def project(world: np.ndarray, h: np.ndarray) -> np.ndarray:
    hp = np.column_stack(
        [world, np.ones(len(world), dtype=np.float64)]
    ) @ h.T
    return hp[:, :2] / hp[:, 2:3]


def image_line(start, end):
    line = np.cross(
        [float(start[0]), float(start[1]), 1.0],
        [float(end[0]), float(end[1]), 1.0],
    )
    line = line / math.hypot(float(line[0]), float(line[1]))
    return line


def line_error(h, world_start, world_end, observed_start, observed_end):
    projected = project(
        np.asarray([world_start, world_end], dtype=np.float64),
        h,
    )
    line = image_line(projected[0], projected[1])
    errors = [
        abs(
            float(
                line[0] * p[0]
                + line[1] * p[1]
                + line[2]
            )
        )
        for p in (observed_start, observed_end)
    ]
    return float(np.mean(errors))


def point_errors(h, world, clicked):
    return np.linalg.norm(
        project(world, h) - clicked,
        axis=1,
    )


def point_constraint(index, world_xy, image_xy):
    return {
        "semantic": f"P{index:02d}",
        "source": "manual",
        "world": {
            "x": float(world_xy[0]),
            "y": float(world_xy[1]),
        },
        "image": {
            "x": float(image_xy[0]),
            "y": float(image_xy[1]),
        },
    }


def line_constraint(
    world_start,
    world_end,
    image_start,
    image_end,
):
    return {
        "semantic": "bottom_touchline",
        "source": "manual",
        "world": {
            "start": {
                "x": float(world_start[0]),
                "y": float(world_start[1]),
            },
            "end": {
                "x": float(world_end[0]),
                "y": float(world_end[1]),
            },
        },
        "image": {
            "start": {
                "x": float(image_start[0]),
                "y": float(image_start[1]),
            },
            "end": {
                "x": float(image_end[0]),
                "y": float(image_end[1]),
            },
        },
    }


def solve(points, lines):
    return solve_field_camera_calibration(
        points,
        lines,
        [],
        1920,
        1080,
    )


def as_h(result):
    return np.asarray(result["homography"], dtype=np.float64)


def assert_fit_within_limits(result):
    diagnostics = result["diagnostics"]
    fit = diagnostics.get("human_point_fit")
    limits = diagnostics.get("human_point_error_limits")

    assert fit is not None, diagnostics
    assert limits is not None, diagnostics

    assert float(fit["mean_px"]) <= float(limits["mean_px"]) + 1e-6
    assert float(fit["p90_px"]) <= float(limits["p90_px"]) + 1e-6
    assert float(fit["max_px"]) <= float(limits["max_px"]) + 1e-6


def main():
    print("=" * 94)
    print("FIELD CAMERA HUMAN GUARD V2 · EXTRAPOLATION REGRESSION")
    print("=" * 94)

    h_true = np.array(
        [
            [12.8, 1.15, 960.0],
            [0.55, -8.25, 545.0],
            [0.00115, 0.00072, 1.0],
        ],
        dtype=np.float64,
    )

    # Deliberately clustered points: analogous to a frame where the visible
    # semantic landmarks do not constrain the near touchline strongly.
    world = np.array(
        [
            [-20.0, -20.0],
            [-10.0, -20.0],
            [0.0, -20.0],
            [10.0, -20.0],
            [20.0, -20.0],
            [-20.0, -10.0],
            [-10.0, -10.0],
            [0.0, -10.0],
            [10.0, -10.0],
            [20.0, -10.0],
        ],
        dtype=np.float64,
    )

    clean = project(world, h_true)

    # Deterministic small click noise chosen to make distant point-only
    # extrapolation measurably wrong while clicks themselves remain realistic.
    rng = np.random.default_rng(3867)
    clicked = clean + rng.normal(
        0.0,
        1.2,
        size=clean.shape,
    )

    points = [
        point_constraint(i, w, p)
        for i, (w, p) in enumerate(
            zip(world, clicked),
            start=1,
        )
    ]

    bottom_world = np.array(
        [
            [-52.5, 32.7],
            [52.5, 32.7],
        ],
        dtype=np.float64,
    )
    bottom_true = project(bottom_world, h_true)

    correct_line = line_constraint(
        bottom_world[0],
        bottom_world[1],
        bottom_true[0],
        bottom_true[1],
    )

    baseline = solve(points, [])
    corrected = solve(points, [correct_line])

    assert baseline["status"] in {
        "solved",
        "solved_with_warnings",
    }, baseline
    assert corrected["status"] in {
        "solved",
        "solved_with_warnings",
    }, corrected

    h_baseline = as_h(baseline)
    h_corrected = as_h(corrected)

    error_before = line_error(
        h_baseline,
        bottom_world[0],
        bottom_world[1],
        bottom_true[0],
        bottom_true[1],
    )
    error_after = line_error(
        h_corrected,
        bottom_world[0],
        bottom_world[1],
        bottom_true[0],
        bottom_true[1],
    )

    pe = point_errors(
        h_corrected,
        world,
        clicked,
    )

    print()
    print("CORRECT DISTANT LINE")
    print(
        f"line error: {error_before:.2f}px -> {error_after:.2f}px"
    )
    print(
        "point fit:"
        f" mean={np.mean(pe):.2f}"
        f" p90={np.percentile(pe, 90):.2f}"
        f" max={np.max(pe):.2f}"
    )
    print(
        "selected:",
        corrected["diagnostics"].get("selected_candidate"),
    )
    print(
        "guard intervened:",
        corrected["diagnostics"].get(
            "human_point_guard_intervened"
        ),
    )
    print(
        "guard alpha:",
        corrected["diagnostics"].get("human_guard_alpha"),
    )

    assert_fit_within_limits(corrected)

    # The central V2 acceptance criterion: unlike V1, a valid far line must
    # have some power to improve the projective extrapolation.
    assert error_after < error_before - 1.0, (
        f"La línea válida no mejoró suficientemente: "
        f"{error_before:.2f}px -> {error_after:.2f}px"
    )

    wrong_bottom = bottom_true.copy()
    wrong_bottom[:, 1] += 90.0

    wrong_line = line_constraint(
        bottom_world[0],
        bottom_world[1],
        wrong_bottom[0],
        wrong_bottom[1],
    )

    contradictory = solve(points, [wrong_line])
    assert contradictory["status"] in {
        "solved",
        "solved_with_warnings",
    }, contradictory

    assert_fit_within_limits(contradictory)

    wrong_fit = contradictory["diagnostics"]["human_point_fit"]
    limits = contradictory["diagnostics"]["human_point_error_limits"]

    print()
    print("CONTRADICTORY LINE")
    print(
        "point fit:"
        f" mean={wrong_fit['mean_px']:.2f}/{limits['mean_px']:.2f}"
        f" p90={wrong_fit['p90_px']:.2f}/{limits['p90_px']:.2f}"
        f" max={wrong_fit['max_px']:.2f}/{limits['max_px']:.2f}"
    )
    print(
        "selected:",
        contradictory["diagnostics"].get(
            "selected_candidate"
        ),
    )
    print(
        "guard intervened:",
        contradictory["diagnostics"].get(
            "human_point_guard_intervened"
        ),
    )
    print(
        "guard alpha:",
        contradictory["diagnostics"].get(
            "human_guard_alpha"
        ),
    )

    print()
    print("PASS")
    print(
        "V2 deja que una línea válida corrija extrapolación, "
        "pero mantiene los landmarks cerca de los clicks HUMAN."
    )
    print()
    print(
        "READ-ONLY CHECK: no modifica review, HUMAN, CAMERA ni Dense."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

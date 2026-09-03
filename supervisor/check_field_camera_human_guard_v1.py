# Synthetic regression for HUMAN-point preservation guard V1.
from __future__ import annotations

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


def point_constraint(semantic, world_xy, image_xy):
    return {
        "semantic": semantic,
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
    semantic,
    world_start,
    world_end,
    image_start,
    image_end,
):
    return {
        "semantic": semantic,
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


def main():
    print("=" * 90)
    print("FIELD CAMERA HUMAN GUARD V1 · SYNTHETIC REGRESSION")
    print("=" * 90)

    h_true = np.array(
        [
            [12.8, 1.15, 960.0],
            [0.55, -8.25, 545.0],
            [0.00115, 0.00072, 1.0],
        ],
        dtype=np.float64,
    )

    world_points = np.array(
        [
            [-52.5, -32.7],
            [-36.0, -20.16],
            [-16.5, -20.16],
            [0.0, -9.15],
            [16.5, -20.16],
            [36.0, -20.16],
            [52.5, -32.7],
            [-36.0, 20.16],
            [0.0, 9.15],
            [36.0, 20.16],
        ],
        dtype=np.float64,
    )

    image_points = project(world_points, h_true)
    noise = np.array(
        [
            [-0.7, 0.5],
            [0.4, -0.4],
            [0.2, 0.6],
            [-0.5, -0.3],
            [0.7, 0.2],
            [-0.3, 0.5],
            [0.6, -0.5],
            [-0.4, 0.1],
            [0.3, -0.6],
            [0.1, 0.4],
        ],
        dtype=np.float64,
    )
    clicked = image_points + noise

    points = [
        point_constraint(f"P{index:02d}", world, image)
        for index, (world, image) in enumerate(
            zip(world_points, clicked),
            start=1,
        )
    ]

    baseline = solve(points, [])
    assert baseline["status"] in {"solved", "solved_with_warnings"}, baseline
    h_baseline = as_h(baseline)

    world_bottom = np.array(
        [
            [-52.5, 32.7],
            [52.5, 32.7],
        ],
        dtype=np.float64,
    )
    image_bottom = project(world_bottom, h_true)

    consistent_line = line_constraint(
        "bottom_touchline",
        world_bottom[0],
        world_bottom[1],
        image_bottom[0],
        image_bottom[1],
    )

    wrong_image_bottom = image_bottom.copy()
    wrong_image_bottom[:, 1] += 65.0
    contradictory_line = line_constraint(
        "bottom_touchline",
        world_bottom[0],
        world_bottom[1],
        wrong_image_bottom[0],
        wrong_image_bottom[1],
    )

    consistent = solve(points, [consistent_line])
    contradictory = solve(points, [contradictory_line])

    assert consistent["status"] in {
        "solved",
        "solved_with_warnings",
    }, consistent
    assert contradictory["status"] in {
        "solved",
        "solved_with_warnings",
    }, contradictory

    baseline_proj = project(world_points, h_baseline)
    consistent_shift = np.linalg.norm(
        project(world_points, as_h(consistent)) - baseline_proj,
        axis=1,
    )
    contradictory_shift = np.linalg.norm(
        project(world_points, as_h(contradictory)) - baseline_proj,
        axis=1,
    )

    cfg = CameraCalibrationConfig()

    print()
    print("CONSISTENT LINE")
    print(
        "point shift vs baseline:"
        f" mean={np.mean(consistent_shift):.3f}"
        f" p90={np.percentile(consistent_shift, 90):.3f}"
        f" max={np.max(consistent_shift):.3f}"
    )
    print(
        "guard intervened:",
        consistent["diagnostics"].get(
            "human_point_guard_intervened"
        ),
    )
    print(
        "selected:",
        consistent["diagnostics"].get(
            "selected_candidate"
        ),
    )

    print()
    print("CONTRADICTORY LINE")
    print(
        "point shift vs baseline:"
        f" mean={np.mean(contradictory_shift):.3f}"
        f" p90={np.percentile(contradictory_shift, 90):.3f}"
        f" max={np.max(contradictory_shift):.3f}"
    )
    print(
        "guard intervened:",
        contradictory["diagnostics"].get(
            "human_point_guard_intervened"
        ),
    )
    print(
        "raw best:",
        contradictory["diagnostics"].get(
            "raw_best_candidate"
        ),
    )
    print(
        "selected:",
        contradictory["diagnostics"].get(
            "selected_candidate"
        ),
    )
    print("detail:", contradictory.get("detail"))

    assert float(np.mean(contradictory_shift)) <= (
        cfg.max_human_point_shift_mean_px + 1e-6
    )
    assert float(np.percentile(contradictory_shift, 90)) <= (
        cfg.max_human_point_shift_p90_px + 1e-6
    )
    assert float(np.max(contradictory_shift)) <= (
        cfg.max_human_point_shift_max_px + 1e-6
    )

    assert bool(
        contradictory["diagnostics"].get(
            "human_point_guard_intervened"
        )
    ), (
        "La línea contradictoria no activó el HUMAN guard; "
        "revisar el test o el solver."
    )

    print()
    print("PASS")
    print(
        "Una línea contradictoria ya no puede desplazar "
        "visiblemente los puntos HUMAN."
    )
    print()
    print(
        "READ-ONLY CHECK: no modifica review, HUMAN, CAMERA ni Dense."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

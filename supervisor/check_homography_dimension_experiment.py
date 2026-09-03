#!/usr/bin/env python3
"""
Homography dimension experiment V1 for RealJaenLab.

Purpose
-------
Read a PRUEBA HOMOGRAFÍA export and test a deliberately simple hypothesis:

    Can a different pitch length/width explain the HUMAN point observations
    better than the current fixed 105 x 68 m model?

This is an OFFLINE research diagnostic. It does not modify reviews, HUMAN
anchors, FIELD, Camera Atlas, or the web application.

Important interpretation
------------------------
A better numerical fit at a different pitch size does NOT prove that the real
pitch has those dimensions. A free pitch dimension can also absorb unmodelled
camera/lens/stitching effects. The script therefore reports both in-sample
reprojection error and leave-one-out (LOO) prediction error.

Dependencies: Python 3, numpy, opencv-python.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


FORMAT = "realjaen.homography-lab"
SUPPORTED_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare exported baseline homography with pitch-dimension optimization."
    )
    parser.add_argument("--experiment", required=True, type=Path, help="Homography Lab JSON export.")
    parser.add_argument("--video", type=Path, default=None, help="Optional local video for visual output.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for report/visuals.")
    parser.add_argument("--length-min", type=float, default=100.0)
    parser.add_argument("--length-max", type=float, default=110.0)
    parser.add_argument("--width-min", type=float, default=64.0)
    parser.add_argument("--width-max", type=float, default=75.0)
    parser.add_argument("--coarse-step", type=float, default=0.25)
    parser.add_argument("--fine-step", type=float, default=0.05)
    parser.add_argument("--fine-radius", type=float, default=0.50)
    return parser.parse_args()


def load_experiment(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT:
        raise ValueError(f"Unsupported format: {payload.get('format')!r}")
    if payload.get("version") != SUPPORTED_VERSION:
        raise ValueError(f"Unsupported version: {payload.get('version')!r}")
    if not isinstance(payload.get("constraints"), list):
        raise ValueError("Missing constraints array.")
    if not isinstance(payload.get("pitch"), dict):
        raise ValueError("Missing pitch object.")
    return payload


def fit_point_constraints(payload: dict) -> list[dict]:
    points = [
        c for c in payload["constraints"]
        if c.get("type") == "point" and not bool(c.get("labTest", False))
    ]
    if len(points) < 5:
        raise ValueError(
            f"V1 requires at least 5 FIT point constraints; found {len(points)}."
        )
    return points


def as_homography(value) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Exported baseline homography is missing or invalid.")
    return matrix


def remap_world_point(
    point: dict,
    *,
    baseline_length: float,
    baseline_width: float,
    candidate_length: float,
    candidate_width: float,
) -> tuple[float, float]:
    """
    Rebuild metric landmark coordinates after changing pitch dimensions.

    Football internal markings are metric offsets from the goal line/centre.
    Therefore:
      - x=0 landmarks remain on halfway;
      - left/right-half landmarks preserve their metric offset from the
        corresponding goal line;
      - touchline landmarks move with +/- width/2;
      - internal y values (penalty-area width, goal-area width, circle radius)
        remain fixed metric distances from the pitch centre.

    This rule exactly reproduces the exported points at the baseline dimensions
    and works for the point semantics used by the current Homography Lab export.
    """
    x0 = float(point["world"]["x"])
    y0 = float(point["world"]["y"])
    half_l0 = baseline_length / 2.0
    half_w0 = baseline_width / 2.0

    if abs(x0) < 1e-9:
        x = 0.0
    elif x0 < 0:
        offset_from_left_goal_line = x0 + half_l0
        x = -candidate_length / 2.0 + offset_from_left_goal_line
    else:
        offset_from_right_goal_line = half_l0 - x0
        x = candidate_length / 2.0 - offset_from_right_goal_line

    if abs(abs(y0) - half_w0) < 1e-6:
        y = math.copysign(candidate_width / 2.0, y0)
    else:
        y = y0

    return x, y


def arrays_for_dimensions(
    points: list[dict],
    baseline_length: float,
    baseline_width: float,
    length: float,
    width: float,
    indices: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if indices is None:
        indices = range(len(points))
    indices = list(indices)
    world = np.asarray([
        remap_world_point(
            points[i],
            baseline_length=baseline_length,
            baseline_width=baseline_width,
            candidate_length=length,
            candidate_width=width,
        )
        for i in indices
    ], dtype=np.float64)
    image = np.asarray([
        [float(points[i]["image"]["x"]), float(points[i]["image"]["y"])]
        for i in indices
    ], dtype=np.float64)
    return world, image


def fit_homography(world: np.ndarray, image: np.ndarray) -> np.ndarray:
    H, _ = cv2.findHomography(world, image, method=0)
    if H is None or H.shape != (3, 3) or not np.isfinite(H).all():
        raise ValueError("Could not fit homography.")
    return H.astype(np.float64)


def project(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    mapped = (H @ homogeneous.T).T
    denom = mapped[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        image = mapped[:, :2] / denom
    return image


def error_metrics(errors: np.ndarray) -> dict:
    errors = np.asarray(errors, dtype=np.float64)
    return {
        "mean_px": float(np.mean(errors)),
        "median_px": float(np.median(errors)),
        "p90_px": float(np.percentile(errors, 90)),
        "max_px": float(np.max(errors)),
    }


def fit_and_errors(
    points: list[dict],
    baseline_length: float,
    baseline_width: float,
    length: float,
    width: float,
) -> tuple[np.ndarray, np.ndarray]:
    world, image = arrays_for_dimensions(
        points, baseline_length, baseline_width, length, width
    )
    H = fit_homography(world, image)
    predicted = project(H, world)
    errors = np.linalg.norm(predicted - image, axis=1)
    return H, errors


def loo_errors(
    points: list[dict],
    baseline_length: float,
    baseline_width: float,
    length: float,
    width: float,
) -> np.ndarray:
    result = []
    all_indices = list(range(len(points)))
    for held_out in all_indices:
        train = [i for i in all_indices if i != held_out]
        world_train, image_train = arrays_for_dimensions(
            points, baseline_length, baseline_width, length, width, train
        )
        H = fit_homography(world_train, image_train)
        world_test, image_test = arrays_for_dimensions(
            points, baseline_length, baseline_width, length, width, [held_out]
        )
        predicted = project(H, world_test)[0]
        result.append(float(np.linalg.norm(predicted - image_test[0])))
    return np.asarray(result, dtype=np.float64)


def frange(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("Grid step must be > 0.")
    count = int(math.floor((stop - start) / step + 1e-9))
    values = [start + i * step for i in range(count + 1)]
    if values[-1] < stop - 1e-8:
        values.append(stop)
    return values


def candidate_score(
    points: list[dict],
    baseline_length: float,
    baseline_width: float,
    length: float,
    width: float,
) -> tuple[float, dict]:
    loo = loo_errors(points, baseline_length, baseline_width, length, width)
    metrics = error_metrics(loo)
    # V1 selects by unseen-point mean error. We keep p90/max in the report so
    # an apparently good mean cannot hide a catastrophic held-out point.
    return metrics["mean_px"], metrics


def search_grid(
    points: list[dict],
    baseline_length: float,
    baseline_width: float,
    lengths: list[float],
    widths: list[float],
) -> dict:
    best = None
    for length in lengths:
        for width in widths:
            score, loo = candidate_score(
                points, baseline_length, baseline_width, length, width
            )
            candidate = {
                "length_m": float(length),
                "width_m": float(width),
                "score": float(score),
                "loo": loo,
            }
            if best is None or candidate["score"] < best["score"]:
                best = candidate
    assert best is not None
    return best


def optimize_dimensions(
    points: list[dict],
    baseline_length: float,
    baseline_width: float,
    args: argparse.Namespace,
) -> dict:
    coarse = search_grid(
        points,
        baseline_length,
        baseline_width,
        frange(args.length_min, args.length_max, args.coarse_step),
        frange(args.width_min, args.width_max, args.coarse_step),
    )

    fine_l_min = max(args.length_min, coarse["length_m"] - args.fine_radius)
    fine_l_max = min(args.length_max, coarse["length_m"] + args.fine_radius)
    fine_w_min = max(args.width_min, coarse["width_m"] - args.fine_radius)
    fine_w_max = min(args.width_max, coarse["width_m"] + args.fine_radius)

    fine = search_grid(
        points,
        baseline_length,
        baseline_width,
        frange(fine_l_min, fine_l_max, args.fine_step),
        frange(fine_w_min, fine_w_max, args.fine_step),
    )
    fine["coarse_best"] = coarse
    return fine


def projected_errors_for_given_H(
    points: list[dict],
    H: np.ndarray,
    baseline_length: float,
    baseline_width: float,
) -> np.ndarray:
    world, image = arrays_for_dimensions(
        points,
        baseline_length,
        baseline_width,
        baseline_length,
        baseline_width,
    )
    predicted = project(H, world)
    return np.linalg.norm(predicted - image, axis=1)


def pitch_polylines(length: float, width: float) -> list[np.ndarray]:
    hl = length / 2.0
    hw = width / 2.0
    penalty_depth = 16.5
    penalty_half_width = 20.16
    goal_depth = 5.5
    goal_half_width = 9.16
    circle_radius = 9.15

    lines = [
        np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw], [-hl, -hw]], dtype=float),
        np.array([[0, -hw], [0, hw]], dtype=float),
        np.array([[-hl, -penalty_half_width], [-hl + penalty_depth, -penalty_half_width],
                  [-hl + penalty_depth, penalty_half_width], [-hl, penalty_half_width]], dtype=float),
        np.array([[hl, -penalty_half_width], [hl - penalty_depth, -penalty_half_width],
                  [hl - penalty_depth, penalty_half_width], [hl, penalty_half_width]], dtype=float),
        np.array([[-hl, -goal_half_width], [-hl + goal_depth, -goal_half_width],
                  [-hl + goal_depth, goal_half_width], [-hl, goal_half_width]], dtype=float),
        np.array([[hl, -goal_half_width], [hl - goal_depth, -goal_half_width],
                  [hl - goal_depth, goal_half_width], [hl, goal_half_width]], dtype=float),
    ]

    theta = np.linspace(0.0, 2.0 * math.pi, 121)
    circle = np.column_stack([circle_radius * np.cos(theta), circle_radius * np.sin(theta)])
    lines.append(circle)
    return lines


def draw_projected_pitch(
    image: np.ndarray,
    H: np.ndarray,
    length: float,
    width: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    height, img_width = image.shape[:2]
    margin = max(height, img_width) * 4
    for polyline in pitch_polylines(length, width):
        projected = project(H, polyline)
        finite = np.isfinite(projected).all(axis=1)
        if finite.sum() < 2:
            continue
        pts = projected[finite]
        # Avoid feeding OpenCV absurd projective values around the horizon.
        safe = (
            (pts[:, 0] > -margin)
            & (pts[:, 0] < img_width + margin)
            & (pts[:, 1] > -margin)
            & (pts[:, 1] < height + margin)
        )
        pts = pts[safe]
        if len(pts) >= 2:
            cv2.polylines(
                image,
                [np.rint(pts).astype(np.int32).reshape(-1, 1, 2)],
                False,
                color,
                thickness,
                cv2.LINE_AA,
            )


def draw_points(image: np.ndarray, points: list[dict], color: tuple[int, int, int]) -> None:
    for c in points:
        x = int(round(float(c["image"]["x"])))
        y = int(round(float(c["image"]["y"])))
        cv2.circle(image, (x, y), 5, color, -1, cv2.LINE_AA)


def load_video_frame(video_path: Path, frame_number: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise ValueError(f"Could not read frame {frame_number} from {video_path}")
    return frame


def image_line_for_world_touchline(
    H: np.ndarray, length: float, y: float
) -> np.ndarray | None:
    projected = project(H, np.array([[-length / 2.0, y], [length / 2.0, y]], dtype=float))
    if not np.isfinite(projected).all():
        return None
    p1 = np.array([projected[0, 0], projected[0, 1], 1.0], dtype=float)
    p2 = np.array([projected[1, 0], projected[1, 1], 1.0], dtype=float)
    line = np.cross(p1, p2)
    norm = math.hypot(line[0], line[1])
    if norm < 1e-12:
        return None
    return line / norm


def y_at_x(line: np.ndarray | None, x: float) -> float | None:
    if line is None or abs(line[1]) < 1e-12:
        return None
    y = -(line[0] * x + line[2]) / line[1]
    return float(y) if math.isfinite(y) else None


def touchline_samples(
    H: np.ndarray, length: float, width: float, image_width: int | None
) -> dict:
    line = image_line_for_world_touchline(H, length, width / 2.0)
    if image_width:
        xs = [0.0, image_width * 0.25, image_width * 0.5, image_width * 0.75, image_width - 1.0]
    else:
        xs = [0.0, 480.0, 960.0, 1440.0, 1919.0]
    return {str(int(round(x))): y_at_x(line, x) for x in xs}


def main() -> None:
    args = parse_args()
    payload = load_experiment(args.experiment)
    points = fit_point_constraints(payload)

    baseline_length = float(payload["pitch"]["length_m"])
    baseline_width = float(payload["pitch"]["width_m"])
    baseline_H = as_homography(payload.get("solve", {}).get("homography"))

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.experiment.parent / f"{args.experiment.stem}_dimension_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_exported_errors = projected_errors_for_given_H(
        points, baseline_H, baseline_length, baseline_width
    )
    refit_H, refit_errors = fit_and_errors(
        points, baseline_length, baseline_width, baseline_length, baseline_width
    )
    baseline_loo = loo_errors(
        points, baseline_length, baseline_width, baseline_length, baseline_width
    )

    best = optimize_dimensions(
        points, baseline_length, baseline_width, args
    )
    best_length = best["length_m"]
    best_width = best["width_m"]
    optimized_H, optimized_errors = fit_and_errors(
        points, baseline_length, baseline_width, best_length, best_width
    )
    optimized_loo = loo_errors(
        points, baseline_length, baseline_width, best_length, best_width
    )

    frame = int(payload["frame"])
    video_frame = None
    image_width = None
    if args.video is not None:
        video_frame = load_video_frame(args.video, frame)
        image_width = int(video_frame.shape[1])

    report = {
        "format": "realjaen.homography-dimension-experiment",
        "version": 1,
        "source_experiment": str(args.experiment),
        "frame": frame,
        "fit_point_count": len(points),
        "baseline": {
            "pitch": {"length_m": baseline_length, "width_m": baseline_width},
            "exported_homography": baseline_H.tolist(),
            "exported_fit_error": error_metrics(baseline_exported_errors),
            "refit_homography": refit_H.tolist(),
            "refit_fit_error": error_metrics(refit_errors),
            "loo_error": error_metrics(baseline_loo),
            "near_touchline_y_by_image_x": touchline_samples(
                baseline_H, baseline_length, baseline_width, image_width
            ),
        },
        "optimized_dimensions": {
            "pitch": {"length_m": best_length, "width_m": best_width},
            "selection_metric": "leave_one_out_mean_px",
            "coarse_best": best["coarse_best"],
            "homography": optimized_H.tolist(),
            "fit_error": error_metrics(optimized_errors),
            "loo_error": error_metrics(optimized_loo),
            "near_touchline_y_by_image_x": touchline_samples(
                optimized_H, best_length, best_width, image_width
            ),
        },
        "search_bounds": {
            "length_m": [args.length_min, args.length_max],
            "width_m": [args.width_min, args.width_max],
            "coarse_step_m": args.coarse_step,
            "fine_step_m": args.fine_step,
            "fine_radius_m": args.fine_radius,
        },
        "interpretation_warning": (
            "A better fit at different dimensions does not prove those are the real pitch "
            "dimensions; dimensions can compensate for unmodelled camera/lens/stitching effects."
        ),
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if video_frame is not None:
        baseline_view = video_frame.copy()
        optimized_view = video_frame.copy()
        draw_projected_pitch(
            baseline_view, baseline_H, baseline_length, baseline_width, (255, 180, 40), 2
        )
        draw_projected_pitch(
            optimized_view, optimized_H, best_length, best_width, (220, 60, 220), 2
        )
        draw_points(baseline_view, points, (0, 210, 255))
        draw_points(optimized_view, points, (0, 210, 255))

        cv2.putText(
            baseline_view,
            f"BASELINE {baseline_length:.2f} x {baseline_width:.2f} m",
            (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            optimized_view,
            f"OPTIMIZED {best_length:.2f} x {best_width:.2f} m",
            (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
        )
        comparison = np.concatenate([baseline_view, optimized_view], axis=1)
        cv2.imwrite(str(output_dir / "baseline_vs_optimized.jpg"), comparison)

        overlay = video_frame.copy()
        draw_projected_pitch(
            overlay, baseline_H, baseline_length, baseline_width, (255, 180, 40), 2
        )
        draw_projected_pitch(
            overlay, optimized_H, best_length, best_width, (220, 60, 220), 2
        )
        draw_points(overlay, points, (0, 210, 255))
        cv2.putText(
            overlay,
            "baseline + optimized overlay",
            (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.imwrite(str(output_dir / "overlay_comparison.jpg"), overlay)

    print("[homography-dimension-v1]")
    print(f"  frame: {frame}")
    print(f"  FIT points: {len(points)}")
    print(
        "  exported baseline fit: "
        f"mean={report['baseline']['exported_fit_error']['mean_px']:.3f}px "
        f"p90={report['baseline']['exported_fit_error']['p90_px']:.3f}px"
    )
    print(
        "  baseline LOO: "
        f"mean={report['baseline']['loo_error']['mean_px']:.3f}px "
        f"p90={report['baseline']['loo_error']['p90_px']:.3f}px"
    )
    print(
        f"  optimized pitch: {best_length:.2f} x {best_width:.2f} m"
    )
    print(
        "  optimized fit: "
        f"mean={report['optimized_dimensions']['fit_error']['mean_px']:.3f}px "
        f"p90={report['optimized_dimensions']['fit_error']['p90_px']:.3f}px"
    )
    print(
        "  optimized LOO: "
        f"mean={report['optimized_dimensions']['loo_error']['mean_px']:.3f}px "
        f"p90={report['optimized_dimensions']['loo_error']['p90_px']:.3f}px"
    )
    print(f"  report: {report_path}")
    if video_frame is not None:
        print(f"  visual: {output_dir / 'baseline_vs_optimized.jpg'}")
        print(f"  overlay: {output_dir / 'overlay_comparison.jpg'}")
    print(
        "  NOTE: optimized dimensions are a diagnostic degree of freedom, "
        "not yet a physical pitch-size estimate."
    )


if __name__ == "__main__":
    main()

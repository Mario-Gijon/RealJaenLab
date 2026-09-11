#!/usr/bin/env python3
"""
Homography TEST-line diagnostic V1 for RealJaenLab.

Reads a PRUEBA HOMOGRAFÍA JSON export and measures how far the baseline
homography predicts a TEST line from the manually observed TEST line.

This is an offline diagnostic only. It does not modify reviews, HUMAN anchors,
FIELD, Camera Atlas, Camera State, or the web application.

Dependencies:
    numpy
    opencv-python
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


FORMAT = "realjaen.homography-lab"
SUPPORTED_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure a Homography Lab TEST line against the baseline homography."
    )
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--semantic", default="bottom_touchline")
    parser.add_argument("--samples", type=int, default=9)
    return parser.parse_args()


def load_experiment(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT:
        raise ValueError(f"Unsupported format: {payload.get('format')!r}")
    if payload.get("version") != SUPPORTED_VERSION:
        raise ValueError(f"Unsupported version: {payload.get('version')!r}")
    return payload


def as_homography(value) -> np.ndarray:
    H = np.asarray(value, dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        raise ValueError("Missing or invalid 3x3 baseline homography.")
    return H


def project(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    ph = np.column_stack([xy, np.ones(len(xy), dtype=np.float64)])
    q = (H @ ph.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return q[:, :2] / q[:, 2:3]


def line_from_points(p1, p2) -> np.ndarray:
    a = np.asarray([p1[0], p1[1], 1.0], dtype=np.float64)
    b = np.asarray([p2[0], p2[1], 1.0], dtype=np.float64)
    line = np.cross(a, b)
    norm = math.hypot(float(line[0]), float(line[1]))
    if norm < 1e-12:
        raise ValueError("Degenerate line.")
    return line / norm


def y_at_x(line: np.ndarray, x: float) -> float | None:
    if abs(float(line[1])) < 1e-12:
        return None
    y = -(float(line[0]) * x + float(line[2])) / float(line[1])
    return float(y) if math.isfinite(y) else None


def point_line_signed_distance(line: np.ndarray, point) -> float:
    # line is normalized so this is directly pixels.
    return float(line[0] * point[0] + line[1] * point[1] + line[2])


def line_angle_deg(p1, p2) -> float:
    return math.degrees(math.atan2(float(p2[1] - p1[1]), float(p2[0] - p1[0])))


def normalize_angle_delta_deg(value: float) -> float:
    while value > 90.0:
        value -= 180.0
    while value < -90.0:
        value += 180.0
    return value


def test_constraint(payload: dict, semantic: str) -> dict:
    matches = [
        c for c in payload.get("constraints", [])
        if c.get("type") == "line"
        and c.get("semantic") == semantic
        and bool(c.get("labTest"))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one TEST line with semantic {semantic!r}; found {len(matches)}."
        )
    return matches[0]


def fit_constraints(payload: dict) -> list[dict]:
    return [c for c in payload.get("constraints", []) if not bool(c.get("labTest"))]


def predicted_line_from_world(H: np.ndarray, constraint: dict):
    ws = constraint["world"]["start"]
    we = constraint["world"]["end"]
    world = np.asarray(
        [[float(ws["x"]), float(ws["y"])], [float(we["x"]), float(we["y"])]],
        dtype=np.float64,
    )
    predicted = project(H, world)
    if not np.isfinite(predicted).all():
        raise ValueError("Projected TEST world line is non-finite.")
    return predicted[0], predicted[1], line_from_points(predicted[0], predicted[1])


def observed_line(constraint: dict):
    start = np.asarray(
        [float(constraint["image"]["start"]["x"]), float(constraint["image"]["start"]["y"])],
        dtype=np.float64,
    )
    end = np.asarray(
        [float(constraint["image"]["end"]["x"]), float(constraint["image"]["end"]["y"])],
        dtype=np.float64,
    )
    return start, end, line_from_points(start, end)


def classify_error(mean_abs: float, span_abs: float, angle_diff: float) -> str:
    if abs(angle_diff) <= 1.0 and span_abs <= max(5.0, mean_abs * 0.20):
        return "approximately_parallel_offset"
    if abs(angle_diff) > 2.0:
        return "meaningful_angle_mismatch"
    return "mixed_offset_and_angle"


def load_video_frame(path: Path, frame_number: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"Could not read frame {frame_number}.")
    return frame


def clip_line_to_image(line: np.ndarray, width: int, height: int):
    candidates = []

    for x in (0.0, float(width - 1)):
        y = y_at_x(line, x)
        if y is not None and 0 <= y <= height - 1:
            candidates.append((x, y))

    if abs(float(line[0])) > 1e-12:
        for y in (0.0, float(height - 1)):
            x = -(float(line[1]) * y + float(line[2])) / float(line[0])
            if math.isfinite(x) and 0 <= x <= width - 1:
                candidates.append((x, y))

    unique = []
    for p in candidates:
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) > 1.0 for q in unique):
            unique.append(p)

    if len(unique) < 2:
        return None
    return unique[0], unique[1]


def draw_line(image, line, color, thickness=3):
    h, w = image.shape[:2]
    clipped = clip_line_to_image(line, w, h)
    if clipped:
        p1, p2 = clipped
        cv2.line(
            image,
            (int(round(p1[0])), int(round(p1[1]))),
            (int(round(p2[0])), int(round(p2[1]))),
            color,
            thickness,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    payload = load_experiment(args.experiment)
    H = as_homography(payload.get("solve", {}).get("homography"))
    constraint = test_constraint(payload, args.semantic)

    obs_start, obs_end, obs_line = observed_line(constraint)
    pred_start, pred_end, pred_line = predicted_line_from_world(H, constraint)

    x0 = min(float(obs_start[0]), float(obs_end[0]))
    x1 = max(float(obs_start[0]), float(obs_end[0]))
    sample_count = max(2, int(args.samples))
    xs = np.linspace(x0, x1, sample_count)

    samples = []
    signed_perp = []
    vertical_delta = []

    for x in xs:
        oy = y_at_x(obs_line, float(x))
        py = y_at_x(pred_line, float(x))
        if oy is None or py is None:
            continue
        signed = point_line_signed_distance(pred_line, (float(x), oy))
        vertical = py - oy
        signed_perp.append(signed)
        vertical_delta.append(vertical)
        samples.append({
            "x_px": float(x),
            "observed_y_px": float(oy),
            "predicted_y_px": float(py),
            "predicted_minus_observed_vertical_px": float(vertical),
            "signed_perpendicular_error_px": float(signed),
            "absolute_perpendicular_error_px": float(abs(signed)),
        })

    if not samples:
        raise ValueError("Could not sample TEST-line error.")

    abs_perp = np.abs(np.asarray(signed_perp, dtype=np.float64))
    vertical_arr = np.asarray(vertical_delta, dtype=np.float64)

    obs_angle = line_angle_deg(obs_start, obs_end)
    pred_angle = line_angle_deg(pred_start, pred_end)
    angle_diff = normalize_angle_delta_deg(pred_angle - obs_angle)

    mean_abs = float(abs_perp.mean())
    span_abs = float(abs_perp.max() - abs_perp.min())
    classification = classify_error(mean_abs, span_abs, angle_diff)

    report = {
        "format": "realjaen.homography-test-line-diagnostic",
        "version": 1,
        "source_experiment": str(args.experiment),
        "frame": int(payload["frame"]),
        "fit_constraint_count": len(fit_constraints(payload)),
        "test_semantic": args.semantic,
        "baseline_reprojection_error_px": payload.get("solve", {}).get("reprojection_error"),
        "observed_test_segment": {
            "start": obs_start.tolist(),
            "end": obs_end.tolist(),
            "angle_deg": obs_angle,
        },
        "predicted_world_line": {
            "projected_world_start": pred_start.tolist(),
            "projected_world_end": pred_end.tolist(),
            "angle_deg": pred_angle,
        },
        "error": {
            "mean_absolute_perpendicular_px": mean_abs,
            "median_absolute_perpendicular_px": float(np.median(abs_perp)),
            "min_absolute_perpendicular_px": float(abs_perp.min()),
            "max_absolute_perpendicular_px": float(abs_perp.max()),
            "absolute_error_span_px": span_abs,
            "mean_vertical_predicted_minus_observed_px": float(vertical_arr.mean()),
            "angle_difference_deg": float(angle_diff),
            "classification": classification,
        },
        "samples": samples,
        "interpretation": {
            "approximately_parallel_offset": (
                "The predicted and observed lines are nearly parallel and the gap changes little "
                "along the measured segment. This does not look like a strong local curvature effect."
            ),
            "meaningful_angle_mismatch": (
                "The predicted and observed lines have a meaningful angle difference."
            ),
            "mixed_offset_and_angle": (
                "The error contains both offset and angle components."
            ),
        }[classification],
    }

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.experiment.parent / f"{args.experiment.stem}_test_line_diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.video is not None:
        frame = load_video_frame(args.video, int(payload["frame"]))
        visual = frame.copy()

        # OpenCV BGR colors. Purely diagnostic.
        draw_line(visual, pred_line, (255, 180, 40), 3)
        cv2.line(
            visual,
            tuple(np.rint(obs_start).astype(int)),
            tuple(np.rint(obs_end).astype(int)),
            (40, 80, 255),
            4,
            cv2.LINE_AA,
        )

        for sample in samples:
            x = int(round(sample["x_px"]))
            oy = int(round(sample["observed_y_px"]))
            py = int(round(sample["predicted_y_px"]))
            if 0 <= x < visual.shape[1]:
                cv2.line(visual, (x, oy), (x, py), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(visual, (x, oy), 3, (40, 80, 255), -1, cv2.LINE_AA)
                cv2.circle(visual, (x, py), 3, (255, 180, 40), -1, cv2.LINE_AA)

        text = (
            f"TEST bottom line: mean={mean_abs:.1f}px "
            f"range={abs_perp.min():.1f}-{abs_perp.max():.1f}px "
            f"angle_diff={angle_diff:+.2f}deg"
        )
        cv2.putText(
            visual, text, (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.imwrite(str(output_dir / "test_line_diagnostic.jpg"), visual)

    print("[homography-test-line-v1]")
    print(f"  frame: {payload['frame']}")
    print(f"  FIT constraints: {len(fit_constraints(payload))}")
    print(f"  TEST line: {args.semantic}")
    print(f"  baseline fit error: {payload.get('solve', {}).get('reprojection_error'):.3f}px")
    print(f"  TEST perpendicular error mean: {mean_abs:.3f}px")
    print(f"  TEST perpendicular error min/max: {abs_perp.min():.3f}px / {abs_perp.max():.3f}px")
    print(f"  TEST error span: {span_abs:.3f}px")
    print(f"  line angle difference: {angle_diff:+.3f} deg")
    print(f"  classification: {classification}")
    print(f"  report: {report_path}")
    if args.video is not None:
        print(f"  visual: {output_dir / 'test_line_diagnostic.jpg'}")


if __name__ == "__main__":
    main()

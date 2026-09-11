#!/usr/bin/env python3
"""
RealJaenLab - Homography support/generalization diagnostic V4.

Purpose
-------
Test whether the experimental vertical extrapolation correction found in the
Homography Lab is genuinely transferable between camera views, rather than
merely fitting each annotated frame independently.

The script evaluates two deliberately simple OFFLINE hypotheses:

V1 - point envelope
    Correction starts below the lowest FIT point, globally.

V2 - FIT-geometry envelope
    Correction starts below a local x-dependent envelope built only from FIT
    point observations and FIT line observations.

The TEST bottom touchline is NEVER used to solve the baseline homography.
For transfer experiments, the target frame's TEST line is also NEVER used to
learn the correction scale. It is used only for evaluation.

V4 keeps the V3 generalization tests and adds a support diagnostic:
    * single-donor pairwise transfer matrix (frame A -> frame B)
    * leave-one-frame-out pooled transfer
    * own-frame optimum only as an upper-bound diagnostic
    * scale-dispersion analysis
    * relative-improvement metrics
    * FIT-point and FIT-line guards
    * horizontal FIT-support versus TEST coverage analysis
    * V2 envelope clamping / extrapolation measurements
    * inside-support versus outside-support TEST errors
    * a screening-only support hypothesis verdict
    * optional leave-one-out visualizations

This script is research-only. It does not modify reviews, HUMAN calibration,
FIELD, OBSERVER, Camera Atlas, Camera State, or the web application.

Recommended repository path:
    supervisor/check_homography_multiframe_v4.py

Dependencies:
    numpy
    opencv-python   (only required when --video is supplied)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


LAB_FORMAT = "realjaen.homography-lab"
LAB_VERSION = 1
REPORT_FORMAT = "realjaen.homography-multiframe-generalization"
REPORT_VERSION = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Cross-frame generalization diagnostic for RealJaenLab Homography Lab "
            "experiments."
        )
    )
    p.add_argument("--experiment", action="append", type=Path, default=[])
    p.add_argument("--input-dir", type=Path, default=None)
    p.add_argument("--video", type=Path, default=None)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/homography_lab/outputs/multiframe_v3"),
    )
    p.add_argument("--test-semantic", default="bottom_touchline")
    p.add_argument("--samples", type=int, default=81)
    p.add_argument(
        "--envelope-bin-width",
        type=float,
        default=120.0,
        help="Horizontal bin width in image pixels for the FIT-geometry envelope.",
    )
    p.add_argument(
        "--max-fit-point-degradation",
        type=float,
        default=1.0,
        help="Maximum allowed mean FIT-point degradation in pixels.",
    )
    p.add_argument(
        "--max-fit-line-degradation",
        type=float,
        default=1.0,
        help="Maximum allowed mean FIT-line perpendicular degradation in pixels.",
    )
    p.add_argument(
        "--min-relative-improvement",
        type=float,
        default=0.15,
        help=(
            "Minimum fractional TEST improvement for a transfer to count as useful. "
            "0.15 means 15%%."
        ),
    )
    p.add_argument(
        "--max-scale-cv",
        type=float,
        default=0.25,
        help=(
            "Maximum coefficient of variation of own-frame correction deltas "
            "(scale - 1) for the strongest automatic verdict."
        ),
    )
    return p.parse_args()


def discover(args: argparse.Namespace) -> list[Path]:
    paths = list(args.experiment)
    if args.input_dir:
        paths += sorted(args.input_dir.glob("homography_lab*_frame_*.json"))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)

    if not unique:
        raise ValueError("No experiments supplied.")
    return unique


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if payload.get("format") != LAB_FORMAT:
        raise ValueError(f"{path}: unexpected format {payload.get('format')!r}")
    if payload.get("version") != LAB_VERSION:
        raise ValueError(f"{path}: unsupported version {payload.get('version')!r}")

    H = np.asarray(payload.get("solve", {}).get("homography"), dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        raise ValueError(f"{path}: invalid or missing 3x3 homography")

    if "frame" not in payload:
        raise ValueError(f"{path}: missing frame")

    return payload


def fit_constraints(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        c for c in payload.get("constraints", [])
        if not bool(c.get("labTest"))
    ]


def fit_points(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in fit_constraints(payload) if c.get("type") == "point"]


def fit_lines(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in fit_constraints(payload) if c.get("type") == "line"]


def test_line(
    payload: dict[str, Any],
    semantic: str,
) -> dict[str, Any] | None:
    matches = [
        c for c in payload.get("constraints", [])
        if c.get("type") == "line"
        and c.get("semantic") == semantic
        and bool(c.get("labTest"))
    ]
    return matches[0] if len(matches) == 1 else None


def project(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ph = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    q = (H @ ph.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        result = q[:, :2] / q[:, 2:3]
    return result


def line_from_points(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    p = np.asarray([a[0], a[1], 1.0], dtype=np.float64)
    q = np.asarray([b[0], b[1], 1.0], dtype=np.float64)
    line = np.cross(p, q)
    norm = math.hypot(float(line[0]), float(line[1]))
    if norm < 1e-12:
        raise ValueError("Degenerate line")
    return line / norm


def point_line_distance(line: np.ndarray, point: np.ndarray) -> float:
    return abs(float(line[0] * point[0] + line[1] * point[1] + line[2]))


def y_at_x(line: np.ndarray, x: float) -> float | None:
    if abs(float(line[1])) < 1e-12:
        return None
    y = -(float(line[0]) * x + float(line[2])) / float(line[1])
    return float(y) if math.isfinite(y) else None


def image_segment(constraint: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            [
                float(constraint["image"]["start"]["x"]),
                float(constraint["image"]["start"]["y"]),
            ],
            [
                float(constraint["image"]["end"]["x"]),
                float(constraint["image"]["end"]["y"]),
            ],
        ],
        dtype=np.float64,
    )


def world_segment(constraint: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            [
                float(constraint["world"]["start"]["x"]),
                float(constraint["world"]["start"]["y"]),
            ],
            [
                float(constraint["world"]["end"]["x"]),
                float(constraint["world"]["end"]["y"]),
            ],
        ],
        dtype=np.float64,
    )


def sample_observed_test(
    payload: dict[str, Any],
    constraint: dict[str, Any],
    count: int,
) -> dict[str, np.ndarray]:
    H = np.asarray(payload["solve"]["homography"], dtype=np.float64)
    observed = image_segment(constraint)
    predicted = project(H, world_segment(constraint))

    observed_line = line_from_points(observed[0], observed[1])
    predicted_line = line_from_points(predicted[0], predicted[1])

    x0, x1 = sorted([float(observed[0, 0]), float(observed[1, 0])])
    xs = np.linspace(x0, x1, max(3, int(count)))

    valid_x: list[float] = []
    observed_y: list[float] = []
    predicted_y: list[float] = []

    for x in xs:
        oy = y_at_x(observed_line, float(x))
        py = y_at_x(predicted_line, float(x))
        if oy is not None and py is not None:
            valid_x.append(float(x))
            observed_y.append(oy)
            predicted_y.append(py)

    if len(valid_x) < 3:
        raise ValueError("Could not sample TEST line")

    return {
        "x": np.asarray(valid_x, dtype=np.float64),
        "observed_y": np.asarray(observed_y, dtype=np.float64),
        "predicted_y": np.asarray(predicted_y, dtype=np.float64),
        "observed_segment": observed,
        "predicted_segment": predicted,
    }


class ConstantEnvelope:
    name = "point_envelope"

    def __init__(self, anchor: float):
        self.anchor = float(anchor)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        return np.full_like(values, self.anchor, dtype=np.float64)

    def serialise(self) -> dict[str, Any]:
        return {"name": self.name, "anchor_y_px": self.anchor}


class GeometryEnvelope:
    name = "fit_geometry_envelope"

    def __init__(
        self,
        xs: list[float],
        ys: list[float],
        bin_width: float,
    ):
        self.bin_width = float(bin_width)
        self.support_x = np.asarray(xs, dtype=np.float64)
        self.support_y = np.asarray(ys, dtype=np.float64)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        q = np.asarray(x, dtype=np.float64)
        return np.interp(
            q,
            self.support_x,
            self.support_y,
            left=float(self.support_y[0]),
            right=float(self.support_y[-1]),
        )

    def serialise(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bin_width_px": self.bin_width,
            "support": [
                {"x_px": float(x), "anchor_y_px": float(y)}
                for x, y in zip(self.support_x, self.support_y)
            ],
        }


def make_point_envelope(payload: dict[str, Any]) -> ConstantEnvelope:
    points = fit_points(payload)
    if not points:
        raise ValueError("No FIT points")
    return ConstantEnvelope(max(float(c["image"]["y"]) for c in points))


def make_geometry_envelope(
    payload: dict[str, Any],
    bin_width: float,
    line_samples: int = 201,
) -> GeometryEnvelope:
    support: list[tuple[float, float]] = []

    for constraint in fit_points(payload):
        support.append(
            (
                float(constraint["image"]["x"]),
                float(constraint["image"]["y"]),
            )
        )

    for constraint in fit_lines(payload):
        segment = image_segment(constraint)
        ts = np.linspace(0.0, 1.0, line_samples)
        pts = (
            segment[0][None, :] * (1.0 - ts[:, None])
            + segment[1][None, :] * ts[:, None]
        )
        support.extend((float(x), float(y)) for x, y in pts)

    if not support:
        raise ValueError("No FIT geometry")

    sx = np.asarray([p[0] for p in support], dtype=np.float64)
    sy = np.asarray([p[1] for p in support], dtype=np.float64)
    xmin, xmax = float(sx.min()), float(sx.max())

    if xmax - xmin < 1e-6:
        return GeometryEnvelope(
            [xmin, xmax + 1.0],
            [float(sy.max()), float(sy.max())],
            bin_width,
        )

    edges = np.arange(xmin, xmax + bin_width, bin_width)
    if len(edges) < 2:
        edges = np.asarray([xmin, xmax], dtype=np.float64)
    if edges[-1] < xmax:
        edges = np.append(edges, xmax)

    chosen: list[tuple[float, float]] = []
    for i in range(len(edges) - 1):
        if i == len(edges) - 2:
            mask = (sx >= edges[i]) & (sx <= edges[i + 1])
        else:
            mask = (sx >= edges[i]) & (sx < edges[i + 1])

        if not mask.any():
            continue

        idxs = np.flatnonzero(mask)
        idx = idxs[np.argmax(sy[mask])]
        chosen.append((float(sx[idx]), float(sy[idx])))

    chosen.sort()

    collapsed: list[tuple[float, float]] = []
    for x, y in chosen:
        if collapsed and abs(collapsed[-1][0] - x) < 1e-9:
            collapsed[-1] = (x, max(collapsed[-1][1], y))
        else:
            collapsed.append((x, y))

    if len(collapsed) == 1:
        x, y = collapsed[0]
        collapsed.append((x + 1.0, y))

    return GeometryEnvelope(
        [p[0] for p in collapsed],
        [p[1] for p in collapsed],
        bin_width,
    )


def warp_y(
    y: np.ndarray,
    x: np.ndarray,
    envelope: ConstantEnvelope | GeometryEnvelope,
    scale: float,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    anchor = envelope(x)
    depth = np.maximum(y - anchor, 0.0)
    return y + (float(scale) - 1.0) * depth


def baseline_test_metrics(record: dict[str, Any]) -> dict[str, float]:
    sample = record["test_samples"]
    error = sample["predicted_y"] - sample["observed_y"]
    abs_error = np.abs(error)
    return {
        "mean_abs_vertical_px": float(abs_error.mean()),
        "median_abs_vertical_px": float(np.median(abs_error)),
        "p90_abs_vertical_px": float(np.percentile(abs_error, 90)),
        "max_abs_vertical_px": float(abs_error.max()),
        "signed_mean_vertical_px": float(error.mean()),
    }


def optimal_scale(record: dict[str, Any], envelope_key: str) -> float:
    sample = record["test_samples"]
    envelope = record[envelope_key]

    depth = np.maximum(
        sample["predicted_y"] - envelope(sample["x"]),
        0.0,
    )
    target = sample["observed_y"] - sample["predicted_y"]

    denom = float(np.dot(depth, depth))
    if denom < 1e-12:
        return 1.0

    return 1.0 + float(np.dot(depth, target) / denom)


def pooled_scale(
    records: list[dict[str, Any]],
    envelope_key: str,
) -> float:
    numer = 0.0
    denom = 0.0

    for record in records:
        sample = record["test_samples"]
        envelope = record[envelope_key]

        depth = np.maximum(
            sample["predicted_y"] - envelope(sample["x"]),
            0.0,
        )
        target = sample["observed_y"] - sample["predicted_y"]

        numer += float(np.dot(depth, target))
        denom += float(np.dot(depth, depth))

    return 1.0 if denom < 1e-12 else 1.0 + numer / denom


def test_metrics(
    record: dict[str, Any],
    envelope_key: str,
    scale: float,
) -> dict[str, float]:
    sample = record["test_samples"]
    corrected = warp_y(
        sample["predicted_y"],
        sample["x"],
        record[envelope_key],
        scale,
    )

    error = corrected - sample["observed_y"]
    abs_error = np.abs(error)

    return {
        "mean_abs_vertical_px": float(abs_error.mean()),
        "median_abs_vertical_px": float(np.median(abs_error)),
        "p90_abs_vertical_px": float(np.percentile(abs_error, 90)),
        "max_abs_vertical_px": float(abs_error.max()),
        "signed_mean_vertical_px": float(error.mean()),
    }


def fit_point_metrics(
    record: dict[str, Any],
    envelope_key: str,
    scale: float,
) -> dict[str, Any]:
    payload = record["payload"]
    points = fit_points(payload)

    if not points:
        return {"count": 0}

    H = np.asarray(payload["solve"]["homography"], dtype=np.float64)

    world = np.asarray(
        [[c["world"]["x"], c["world"]["y"]] for c in points],
        dtype=np.float64,
    )
    observed = np.asarray(
        [[c["image"]["x"], c["image"]["y"]] for c in points],
        dtype=np.float64,
    )

    baseline = project(H, world)
    corrected = baseline.copy()
    corrected[:, 1] = warp_y(
        baseline[:, 1],
        baseline[:, 0],
        record[envelope_key],
        scale,
    )

    baseline_error = np.linalg.norm(baseline - observed, axis=1)
    corrected_error = np.linalg.norm(corrected - observed, axis=1)
    movement = np.linalg.norm(corrected - baseline, axis=1)

    return {
        "count": len(points),
        "baseline_mean_px": float(baseline_error.mean()),
        "corrected_mean_px": float(corrected_error.mean()),
        "mean_degradation_px": float(
            corrected_error.mean() - baseline_error.mean()
        ),
        "max_projected_movement_px": float(movement.max()),
    }


def fit_line_metrics(
    record: dict[str, Any],
    envelope_key: str,
    scale: float,
    count: int,
) -> dict[str, Any]:
    payload = record["payload"]
    H = np.asarray(payload["solve"]["homography"], dtype=np.float64)

    rows: list[dict[str, Any]] = []

    for constraint in fit_lines(payload):
        observed_segment = image_segment(constraint)
        observed_line = line_from_points(
            observed_segment[0],
            observed_segment[1],
        )
        world = world_segment(constraint)

        ts = np.linspace(0.0, 1.0, max(9, int(count)))
        world_samples = (
            world[0][None, :] * (1.0 - ts[:, None])
            + world[1][None, :] * ts[:, None]
        )

        baseline = project(H, world_samples)
        corrected = baseline.copy()
        corrected[:, 1] = warp_y(
            baseline[:, 1],
            baseline[:, 0],
            record[envelope_key],
            scale,
        )

        baseline_error = np.asarray(
            [point_line_distance(observed_line, p) for p in baseline],
            dtype=np.float64,
        )
        corrected_error = np.asarray(
            [point_line_distance(observed_line, p) for p in corrected],
            dtype=np.float64,
        )
        movement = np.linalg.norm(corrected - baseline, axis=1)

        rows.append(
            {
                "semantic": constraint.get("semantic"),
                "baseline_mean_perpendicular_px": float(
                    baseline_error.mean()
                ),
                "corrected_mean_perpendicular_px": float(
                    corrected_error.mean()
                ),
                "mean_perpendicular_degradation_px": float(
                    corrected_error.mean() - baseline_error.mean()
                ),
                "max_projected_movement_px": float(movement.max()),
            }
        )

    if not rows:
        return {
            "count": 0,
            "mean_perpendicular_degradation_px": None,
            "max_projected_movement_px": 0.0,
            "per_line": [],
        }

    return {
        "count": len(rows),
        "mean_perpendicular_degradation_px": float(
            np.mean(
                [
                    row["mean_perpendicular_degradation_px"]
                    for row in rows
                ]
            )
        ),
        "max_projected_movement_px": float(
            max(row["max_projected_movement_px"] for row in rows)
        ),
        "per_line": rows,
    }


def evaluate(
    record: dict[str, Any],
    envelope_key: str,
    scale: float,
    samples: int,
) -> dict[str, Any]:
    return {
        "scale": float(scale),
        "test": test_metrics(record, envelope_key, scale),
        "fit_points": fit_point_metrics(record, envelope_key, scale),
        "fit_lines": fit_line_metrics(record, envelope_key, scale, samples),
    }


def fit_guard(
    evaluation: dict[str, Any],
    max_point_degradation: float,
    max_line_degradation: float,
) -> tuple[bool, float, float | None]:
    point_deg = float(
        evaluation["fit_points"].get("mean_degradation_px") or 0.0
    )
    line_deg_raw = evaluation["fit_lines"].get(
        "mean_perpendicular_degradation_px"
    )
    line_deg = None if line_deg_raw is None else float(line_deg_raw)
    line_for_guard = 0.0 if line_deg is None else line_deg

    passes = (
        point_deg <= max_point_degradation
        and line_for_guard <= max_line_degradation
    )
    return passes, point_deg, line_deg


def transfer_metrics(
    target: dict[str, Any],
    envelope_key: str,
    scale: float,
    samples: int,
    max_point_degradation: float,
    max_line_degradation: float,
    min_relative_improvement: float,
) -> dict[str, Any]:
    evaluation = evaluate(target, envelope_key, scale, samples)
    passes_guard, point_deg, line_deg = fit_guard(
        evaluation,
        max_point_degradation,
        max_line_degradation,
    )

    baseline = float(target["baseline_test"]["mean_abs_vertical_px"])
    corrected = float(evaluation["test"]["mean_abs_vertical_px"])
    improvement_px = baseline - corrected
    relative = improvement_px / baseline if baseline > 1e-12 else 0.0

    return {
        "scale": float(scale),
        "baseline_test_mean_px": baseline,
        "corrected_test_mean_px": corrected,
        "improvement_px": float(improvement_px),
        "relative_improvement": float(relative),
        "fit_point_degradation_px": point_deg,
        "fit_line_perpendicular_degradation_px": line_deg,
        "fit_line_max_projected_movement_px": float(
            evaluation["fit_lines"].get(
                "max_projected_movement_px",
                0.0,
            )
        ),
        "passes_fit_guard": bool(passes_guard),
        "useful_transfer": bool(
            passes_guard and relative >= min_relative_improvement
        ),
        "worsened_test": bool(corrected > baseline),
    }


def pairwise_transfer(
    records: list[dict[str, Any]],
    envelope_key: str,
    samples: int,
    max_point_degradation: float,
    max_line_degradation: float,
    min_relative_improvement: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for donor in records:
        donor_scale = optimal_scale(donor, envelope_key)

        for target in records:
            if donor is target:
                continue

            metrics = transfer_metrics(
                target,
                envelope_key,
                donor_scale,
                samples,
                max_point_degradation,
                max_line_degradation,
                min_relative_improvement,
            )
            rows.append(
                {
                    "donor_frame": donor["frame"],
                    "target_frame": target["frame"],
                    **metrics,
                }
            )

    return rows


def leave_one_out_transfer(
    records: list[dict[str, Any]],
    envelope_key: str,
    samples: int,
    max_point_degradation: float,
    max_line_degradation: float,
    min_relative_improvement: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for held_out in records:
        donors = [record for record in records if record is not held_out]
        scale = pooled_scale(donors, envelope_key)

        metrics = transfer_metrics(
            held_out,
            envelope_key,
            scale,
            samples,
            max_point_degradation,
            max_line_degradation,
            min_relative_improvement,
        )

        rows.append(
            {
                "held_out_frame": held_out["frame"],
                "donor_frames": [record["frame"] for record in donors],
                "learned_scale": float(scale),
                **metrics,
            }
        )

    return rows


def build_record(
    path: Path,
    payload: dict[str, Any],
    semantic: str,
    samples: int,
    bin_width: float,
) -> dict[str, Any] | None:
    test = test_line(payload, semantic)
    if test is None:
        return None

    record: dict[str, Any] = {
        "path": path,
        "payload": payload,
        "frame": int(payload["frame"]),
        "point_envelope": make_point_envelope(payload),
        "geometry_envelope": make_geometry_envelope(payload, bin_width),
        "test_samples": sample_observed_test(
            payload,
            test,
            samples,
        ),
    }
    record["baseline_test"] = baseline_test_metrics(record)
    return record


def safe_cv(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return None

    mean = float(arr.mean())
    if abs(mean) < 1e-12:
        return None

    return float(arr.std(ddof=1) / abs(mean))


def scale_dispersion(
    records: list[dict[str, Any]],
    envelope_key: str,
) -> dict[str, Any]:
    scales = [float(optimal_scale(record, envelope_key)) for record in records]
    deltas = [scale - 1.0 for scale in scales]
    delta_cv = safe_cv(deltas)

    return {
        "own_scales": [
            {
                "frame": record["frame"],
                "scale": scale,
                "delta_from_identity": scale - 1.0,
            }
            for record, scale in zip(records, scales)
        ],
        "scale_mean": float(np.mean(scales)),
        "scale_std": float(np.std(scales, ddof=1)) if len(scales) > 1 else 0.0,
        "scale_min": float(np.min(scales)),
        "scale_max": float(np.max(scales)),
        "delta_cv": delta_cv,
        "same_delta_direction": bool(
            all(delta >= 0.0 for delta in deltas)
            or all(delta <= 0.0 for delta in deltas)
        ),
    }


def summarise_transfers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "useful_count": 0,
            "guard_pass_count": 0,
            "worsened_count": 0,
            "useful_ratio": 0.0,
            "guard_pass_ratio": 0.0,
            "relative_improvement_mean": None,
            "relative_improvement_median": None,
            "relative_improvement_min": None,
            "relative_improvement_max": None,
        }

    rel = np.asarray(
        [row["relative_improvement"] for row in rows],
        dtype=np.float64,
    )

    useful_count = sum(bool(row["useful_transfer"]) for row in rows)
    guard_count = sum(bool(row["passes_fit_guard"]) for row in rows)
    worsened_count = sum(bool(row["worsened_test"]) for row in rows)

    return {
        "count": len(rows),
        "useful_count": useful_count,
        "guard_pass_count": guard_count,
        "worsened_count": worsened_count,
        "useful_ratio": float(useful_count / len(rows)),
        "guard_pass_ratio": float(guard_count / len(rows)),
        "relative_improvement_mean": float(rel.mean()),
        "relative_improvement_median": float(np.median(rel)),
        "relative_improvement_min": float(rel.min()),
        "relative_improvement_max": float(rel.max()),
    }


def model_verdict(
    loo_rows: list[dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
    dispersion: dict[str, Any],
    max_scale_cv: float,
) -> dict[str, Any]:
    loo_summary = summarise_transfers(loo_rows)
    pairwise_summary = summarise_transfers(pairwise_rows)

    all_loo_useful = bool(loo_rows) and all(
        row["useful_transfer"] for row in loo_rows
    )
    all_loo_guards = bool(loo_rows) and all(
        row["passes_fit_guard"] for row in loo_rows
    )

    delta_cv = dispersion.get("delta_cv")
    stable_scale = delta_cv is not None and delta_cv <= max_scale_cv
    pairwise_majority = pairwise_summary["useful_ratio"] >= (2.0 / 3.0)
    pairwise_half = pairwise_summary["useful_ratio"] >= 0.5
    no_loo_worsening = loo_summary["worsened_count"] == 0

    if (
        all_loo_useful
        and all_loo_guards
        and pairwise_majority
        and stable_scale
        and dispersion["same_delta_direction"]
    ):
        label = "promising"
        reason = (
            "The correction transfers to every held-out frame, most single-donor "
            "transfers are useful, FIT guards pass, and the required correction "
            "magnitude is reasonably stable across views."
        )
    elif (
        no_loo_worsening
        and loo_summary["useful_ratio"] >= (2.0 / 3.0)
        and pairwise_half
        and all_loo_guards
    ):
        label = "mixed"
        reason = (
            "There is evidence of cross-frame transfer, but scale stability or "
            "single-donor consistency is not strong enough to call the model "
            "view-independent."
        )
    else:
        label = "not_generalizing"
        reason = (
            "The correction does not transfer consistently enough across held-out "
            "views, worsens some TEST cases, or violates FIT guards."
        )

    return {
        "label": label,
        "reason": reason,
        "screening_only": True,
        "sample_size_frames": len(loo_rows),
        "criteria": {
            "all_leave_one_out_useful": all_loo_useful,
            "all_leave_one_out_fit_guards_pass": all_loo_guards,
            "no_leave_one_out_worsening": no_loo_worsening,
            "pairwise_useful_ratio": pairwise_summary["useful_ratio"],
            "pairwise_majority_threshold": 2.0 / 3.0,
            "delta_cv": delta_cv,
            "max_scale_cv": max_scale_cv,
            "stable_scale": stable_scale,
            "same_delta_direction": dispersion["same_delta_direction"],
        },
    }


def compact_frame_record(
    record: dict[str, Any],
    samples: int,
) -> dict[str, Any]:
    own_v1_scale = optimal_scale(record, "point_envelope")
    own_v2_scale = optimal_scale(record, "geometry_envelope")

    own_v1 = evaluate(record, "point_envelope", own_v1_scale, samples)
    own_v2 = evaluate(record, "geometry_envelope", own_v2_scale, samples)

    return {
        "frame": record["frame"],
        "source": str(record["path"]),
        "fit_points": len(fit_points(record["payload"])),
        "fit_lines": len(fit_lines(record["payload"])),
        "baseline_reprojection_error_px": record["payload"]["solve"].get(
            "reprojection_error"
        ),
        "baseline_test": record["baseline_test"],
        "v1_own_upper_bound": own_v1,
        "v2_own_upper_bound": own_v2,
        "point_envelope": record["point_envelope"].serialise(),
        "geometry_envelope": record["geometry_envelope"].serialise(),
    }



def raw_fit_support_x(record: dict[str, Any]) -> tuple[float, float]:
    """Horizontal image support directly observed by FIT points/line endpoints."""
    xs: list[float] = []

    for constraint in fit_constraints(record["payload"]):
        ctype = constraint.get("type")
        image = constraint.get("image") or {}

        if ctype == "point" and "x" in image:
            xs.append(float(image["x"]))
        elif ctype == "line":
            start = image.get("start") or {}
            end = image.get("end") or {}
            if "x" in start:
                xs.append(float(start["x"]))
            if "x" in end:
                xs.append(float(end["x"]))

    if not xs:
        raise ValueError(
            f"Frame {record['frame']}: no horizontal FIT support could be measured."
        )

    return float(min(xs)), float(max(xs))


def error_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "mean_abs_px": None,
            "median_abs_px": None,
            "p90_abs_px": None,
            "max_abs_px": None,
        }

    absolute = np.abs(values)
    return {
        "count": int(values.size),
        "mean_abs_px": float(absolute.mean()),
        "median_abs_px": float(np.median(absolute)),
        "p90_abs_px": float(np.percentile(absolute, 90)),
        "max_abs_px": float(absolute.max()),
    }


def split_test_error_by_horizontal_support(
    record: dict[str, Any],
    scale: float,
) -> dict[str, Any]:
    sample = record["test_samples"]
    xs = sample["x"]
    fit_x_min, fit_x_max = raw_fit_support_x(record)

    inside = (xs >= fit_x_min) & (xs <= fit_x_max)
    corrected_y = warp_y(
        sample["predicted_y"],
        xs,
        record["geometry_envelope"],
        scale,
    )
    error = corrected_y - sample["observed_y"]

    return {
        "inside_fit_x_support": error_stats(error[inside]),
        "outside_fit_x_support": error_stats(error[~inside]),
    }


def support_diagnostic(
    record: dict[str, Any],
    loo_v2_row: dict[str, Any],
) -> dict[str, Any]:
    sample = record["test_samples"]
    xs = sample["x"]

    fit_x_min, fit_x_max = raw_fit_support_x(record)
    test_x_min = float(np.min(xs))
    test_x_max = float(np.max(xs))

    fit_span = max(0.0, fit_x_max - fit_x_min)
    test_span = max(0.0, test_x_max - test_x_min)

    overlap_start = max(fit_x_min, test_x_min)
    overlap_end = min(fit_x_max, test_x_max)
    overlap = max(0.0, overlap_end - overlap_start)
    coverage_ratio = overlap / test_span if test_span > 1e-12 else 0.0

    inside_mask = (xs >= fit_x_min) & (xs <= fit_x_max)
    inside_fraction = float(np.mean(inside_mask)) if len(xs) else 0.0

    left_extrapolation = max(0.0, fit_x_min - test_x_min)
    right_extrapolation = max(0.0, test_x_max - fit_x_max)

    envelope = record["geometry_envelope"]
    env_x_min = float(envelope.support_x[0])
    env_x_max = float(envelope.support_x[-1])
    clamped_mask = (xs < env_x_min) | (xs > env_x_max)
    clamped_fraction = float(np.mean(clamped_mask)) if len(xs) else 0.0

    predicted_y = sample["predicted_y"]
    envelope_y = envelope(xs)
    positive_depth = np.maximum(predicted_y - envelope_y, 0.0)

    own_scale = float(optimal_scale(record, "geometry_envelope"))

    baseline_split = split_test_error_by_horizontal_support(record, 1.0)
    loo_split = split_test_error_by_horizontal_support(
        record,
        float(loo_v2_row["learned_scale"]),
    )

    return {
        "frame": int(record["frame"]),
        "fit_x_min_px": fit_x_min,
        "fit_x_max_px": fit_x_max,
        "fit_x_span_px": fit_span,
        "test_x_min_px": test_x_min,
        "test_x_max_px": test_x_max,
        "test_x_span_px": test_span,
        "test_fit_x_overlap_px": overlap,
        "test_fit_x_coverage_ratio": float(coverage_ratio),
        "test_sample_inside_fit_x_fraction": inside_fraction,
        "test_sample_outside_fit_x_fraction": float(1.0 - inside_fraction),
        "left_horizontal_extrapolation_px": float(left_extrapolation),
        "right_horizontal_extrapolation_px": float(right_extrapolation),
        "max_horizontal_extrapolation_px": float(
            max(left_extrapolation, right_extrapolation)
        ),
        "v2_envelope_x_min_px": env_x_min,
        "v2_envelope_x_max_px": env_x_max,
        "v2_test_clamped_fraction": clamped_fraction,
        "v2_predicted_depth_below_envelope_mean_px": float(
            positive_depth.mean()
        ),
        "v2_predicted_depth_below_envelope_max_px": float(
            positive_depth.max()
        ),
        "baseline_test_mean_px": float(
            record["baseline_test"]["mean_abs_vertical_px"]
        ),
        "v2_own_scale": own_scale,
        "v2_own_delta_from_identity": float(own_scale - 1.0),
        "v2_loo_scale": float(loo_v2_row["learned_scale"]),
        "v2_loo_corrected_test_mean_px": float(
            loo_v2_row["corrected_test_mean_px"]
        ),
        "v2_loo_relative_improvement": float(
            loo_v2_row["relative_improvement"]
        ),
        "baseline_error_split": baseline_split,
        "v2_loo_error_split": loo_split,
    }


def average_ranks(values: list[float]) -> np.ndarray:
    """Small scipy-free rankdata implementation with average ranks for ties."""
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)

    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        average_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = average_rank
        i = j

    return ranks


def spearman_small_sample(
    x: list[float],
    y: list[float],
) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None

    rx = average_ranks(x)
    ry = average_ranks(y)

    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(
        np.sqrt(np.dot(rx, rx) * np.dot(ry, ry))
    )
    if denom < 1e-12:
        return None

    return float(np.dot(rx, ry) / denom)


def summarise_support_hypothesis(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    outside = [
        float(row["test_sample_outside_fit_x_fraction"])
        for row in rows
    ]
    deltas = [
        float(row["v2_own_delta_from_identity"])
        for row in rows
    ]
    max_extrapolation = [
        float(row["max_horizontal_extrapolation_px"])
        for row in rows
    ]

    rho_outside = spearman_small_sample(outside, deltas)
    rho_distance = spearman_small_sample(max_extrapolation, deltas)

    most_outside = max(
        rows,
        key=lambda row: (
            row["test_sample_outside_fit_x_fraction"],
            row["max_horizontal_extrapolation_px"],
        ),
    )
    largest_delta = max(
        rows,
        key=lambda row: row["v2_own_delta_from_identity"],
    )

    supported_rows = [
        row for row in rows
        if row["test_sample_outside_fit_x_fraction"] <= 0.20
    ]
    supported_deltas = [
        float(row["v2_own_delta_from_identity"])
        for row in supported_rows
    ]
    supported_median = (
        float(np.median(supported_deltas))
        if supported_deltas
        else None
    )

    outlier_ratio = None
    if supported_median is not None and abs(supported_median) > 1e-12:
        outlier_ratio = float(
            most_outside["v2_own_delta_from_identity"]
            / supported_median
        )

    same_frame = most_outside["frame"] == largest_delta["frame"]
    strongly_outside = (
        most_outside["test_sample_outside_fit_x_fraction"] >= 0.50
    )
    clearly_larger_delta = (
        outlier_ratio is not None and outlier_ratio >= 1.50
    )

    if strongly_outside and same_frame and clearly_larger_delta:
        label = "consistent_with_support_limited_extrapolation"
        reason = (
            "The frame with the weakest horizontal FIT support is also the frame "
            "requiring the largest V2 correction, and its correction delta is "
            "substantially larger than the median of well-supported frames."
        )
    elif (
        same_frame
        and (
            (rho_outside is not None and rho_outside >= 0.5)
            or (rho_distance is not None and rho_distance >= 0.5)
        )
    ):
        label = "suggestive"
        reason = (
            "Horizontal extrapolation and required V2 correction move in the "
            "expected direction, but the evidence is not strong enough for a "
            "clear screening result."
        )
    else:
        label = "inconclusive"
        reason = (
            "The current frames do not show a clear relationship between "
            "horizontal FIT support and the V2 correction magnitude."
        )

    return {
        "label": label,
        "reason": reason,
        "screening_only": True,
        "frame_count": len(rows),
        "most_horizontally_extrapolated_frame": int(
            most_outside["frame"]
        ),
        "largest_v2_delta_frame": int(largest_delta["frame"]),
        "same_frame": bool(same_frame),
        "well_supported_frame_count": len(supported_rows),
        "well_supported_v2_delta_median": supported_median,
        "most_outside_delta_ratio_to_supported_median": outlier_ratio,
        "spearman_outside_fraction_vs_v2_delta": rho_outside,
        "spearman_max_horizontal_extrapolation_vs_v2_delta": rho_distance,
        "warning": (
            "With only a few manually annotated frames this is a mechanism "
            "diagnostic, not statistical proof or a production rule."
        ),
    }


def write_support_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fields = [
        "frame",
        "fit_x_min_px",
        "fit_x_max_px",
        "fit_x_span_px",
        "test_x_min_px",
        "test_x_max_px",
        "test_x_span_px",
        "test_fit_x_coverage_ratio",
        "test_sample_inside_fit_x_fraction",
        "test_sample_outside_fit_x_fraction",
        "left_horizontal_extrapolation_px",
        "right_horizontal_extrapolation_px",
        "max_horizontal_extrapolation_px",
        "v2_test_clamped_fraction",
        "baseline_test_mean_px",
        "v2_own_scale",
        "v2_own_delta_from_identity",
        "v2_loo_scale",
        "v2_loo_corrected_test_mean_px",
        "v2_loo_relative_improvement",
        "baseline_inside_mean_px",
        "baseline_outside_mean_px",
        "v2_loo_inside_mean_px",
        "v2_loo_outside_mean_px",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "frame": row["frame"],
                    "fit_x_min_px": row["fit_x_min_px"],
                    "fit_x_max_px": row["fit_x_max_px"],
                    "fit_x_span_px": row["fit_x_span_px"],
                    "test_x_min_px": row["test_x_min_px"],
                    "test_x_max_px": row["test_x_max_px"],
                    "test_x_span_px": row["test_x_span_px"],
                    "test_fit_x_coverage_ratio": row[
                        "test_fit_x_coverage_ratio"
                    ],
                    "test_sample_inside_fit_x_fraction": row[
                        "test_sample_inside_fit_x_fraction"
                    ],
                    "test_sample_outside_fit_x_fraction": row[
                        "test_sample_outside_fit_x_fraction"
                    ],
                    "left_horizontal_extrapolation_px": row[
                        "left_horizontal_extrapolation_px"
                    ],
                    "right_horizontal_extrapolation_px": row[
                        "right_horizontal_extrapolation_px"
                    ],
                    "max_horizontal_extrapolation_px": row[
                        "max_horizontal_extrapolation_px"
                    ],
                    "v2_test_clamped_fraction": row[
                        "v2_test_clamped_fraction"
                    ],
                    "baseline_test_mean_px": row[
                        "baseline_test_mean_px"
                    ],
                    "v2_own_scale": row["v2_own_scale"],
                    "v2_own_delta_from_identity": row[
                        "v2_own_delta_from_identity"
                    ],
                    "v2_loo_scale": row["v2_loo_scale"],
                    "v2_loo_corrected_test_mean_px": row[
                        "v2_loo_corrected_test_mean_px"
                    ],
                    "v2_loo_relative_improvement": row[
                        "v2_loo_relative_improvement"
                    ],
                    "baseline_inside_mean_px": row[
                        "baseline_error_split"
                    ]["inside_fit_x_support"]["mean_abs_px"],
                    "baseline_outside_mean_px": row[
                        "baseline_error_split"
                    ]["outside_fit_x_support"]["mean_abs_px"],
                    "v2_loo_inside_mean_px": row[
                        "v2_loo_error_split"
                    ]["inside_fit_x_support"]["mean_abs_px"],
                    "v2_loo_outside_mean_px": row[
                        "v2_loo_error_split"
                    ]["outside_fit_x_support"]["mean_abs_px"],
                }
            )

def write_pairwise_csv(
    path: Path,
    v1_rows: list[dict[str, Any]],
    v2_rows: list[dict[str, Any]],
) -> None:
    fields = [
        "model",
        "donor_frame",
        "target_frame",
        "scale",
        "baseline_test_mean_px",
        "corrected_test_mean_px",
        "improvement_px",
        "relative_improvement",
        "fit_point_degradation_px",
        "fit_line_perpendicular_degradation_px",
        "passes_fit_guard",
        "useful_transfer",
        "worsened_test",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for model, rows in (("v1", v1_rows), ("v2", v2_rows)):
            for row in rows:
                writer.writerow(
                    {
                        "model": model,
                        **{field: row.get(field) for field in fields if field != "model"},
                    }
                )


def write_loo_csv(
    path: Path,
    v1_rows: list[dict[str, Any]],
    v2_rows: list[dict[str, Any]],
) -> None:
    fields = [
        "model",
        "held_out_frame",
        "donor_frames",
        "learned_scale",
        "baseline_test_mean_px",
        "corrected_test_mean_px",
        "improvement_px",
        "relative_improvement",
        "fit_point_degradation_px",
        "fit_line_perpendicular_degradation_px",
        "passes_fit_guard",
        "useful_transfer",
        "worsened_test",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for model, rows in (("v1", v1_rows), ("v2", v2_rows)):
            for row in rows:
                csv_row = {
                    "model": model,
                    **{field: row.get(field) for field in fields if field != "model"},
                }
                csv_row["donor_frames"] = ",".join(
                    str(frame) for frame in row["donor_frames"]
                )
                writer.writerow(csv_row)


def render_loo_v2(
    record: dict[str, Any],
    loo_row: dict[str, Any],
    video: Path,
    output_dir: Path,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "--video requires opencv-python (import cv2 failed)."
        ) from exc

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, record["frame"])
    ok, image = cap.read()
    cap.release()

    if not ok or image is None:
        raise ValueError(f"Could not read frame {record['frame']} from {video}")

    sample = record["test_samples"]
    xs = sample["x"]
    observed_y = sample["observed_y"]
    baseline_y = sample["predicted_y"]
    corrected_y = warp_y(
        baseline_y,
        xs,
        record["geometry_envelope"],
        float(loo_row["learned_scale"]),
    )
    envelope_y = record["geometry_envelope"](xs)

    def draw_polyline(ys: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
        pts = np.column_stack([xs, ys])
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) >= 2:
            cv2.polylines(
                image,
                [np.rint(pts).astype(np.int32).reshape(-1, 1, 2)],
                False,
                color,
                thickness,
                cv2.LINE_AA,
            )

    # BGR diagnostic colours.
    draw_polyline(baseline_y, (255, 180, 40), 3)
    draw_polyline(corrected_y, (220, 60, 220), 3)
    draw_polyline(observed_y, (40, 80, 255), 4)
    draw_polyline(envelope_y, (0, 220, 220), 2)

    text = (
        f"frame {record['frame']} | LOO V2 "
        f"{loo_row['baseline_test_mean_px']:.1f}px -> "
        f"{loo_row['corrected_test_mean_px']:.1f}px | "
        f"s={loo_row['learned_scale']:.5f}"
    )
    cv2.putText(
        image,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(
        str(output_dir / f"frame_{record['frame']}_loo_v2.jpg"),
        image,
    )


def main() -> None:
    args = parse_args()

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for path in discover(args):
        payload = load_payload(path)
        record = build_record(
            path,
            payload,
            args.test_semantic,
            args.samples,
            args.envelope_bin_width,
        )

        if record is None:
            skipped.append(
                {
                    "path": str(path),
                    "frame": payload.get("frame"),
                    "reason": f"no unique TEST {args.test_semantic}",
                }
            )
        else:
            records.append(record)

    if len(records) < 3:
        raise ValueError(
            "V4 support/generalization diagnostic needs at least three experiments "
            "with the requested TEST line."
        )

    records.sort(key=lambda record: record["frame"])

    models = {
        "v1": "point_envelope",
        "v2": "geometry_envelope",
    }

    pairwise: dict[str, list[dict[str, Any]]] = {}
    loo: dict[str, list[dict[str, Any]]] = {}
    dispersion: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    verdicts: dict[str, dict[str, Any]] = {}

    for model_name, envelope_key in models.items():
        pairwise_rows = pairwise_transfer(
            records,
            envelope_key,
            args.samples,
            args.max_fit_point_degradation,
            args.max_fit_line_degradation,
            args.min_relative_improvement,
        )
        loo_rows = leave_one_out_transfer(
            records,
            envelope_key,
            args.samples,
            args.max_fit_point_degradation,
            args.max_fit_line_degradation,
            args.min_relative_improvement,
        )
        model_dispersion = scale_dispersion(records, envelope_key)

        pairwise[model_name] = pairwise_rows
        loo[model_name] = loo_rows
        dispersion[model_name] = model_dispersion
        summaries[model_name] = {
            "pairwise": summarise_transfers(pairwise_rows),
            "leave_one_out": summarise_transfers(loo_rows),
        }
        verdicts[model_name] = model_verdict(
            loo_rows,
            pairwise_rows,
            model_dispersion,
            args.max_scale_cv,
        )

    frame_rows = [
        compact_frame_record(record, args.samples)
        for record in records
    ]

    loo_v2_by_frame = {
        int(row["held_out_frame"]): row
        for row in loo["v2"]
    }
    support_rows = [
        support_diagnostic(
            record,
            loo_v2_by_frame[int(record["frame"])],
        )
        for record in records
    ]
    support_hypothesis = summarise_support_hypothesis(
        support_rows
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "experiment_version": "multiframe_v4",
        "evaluated_frames": [record["frame"] for record in records],
        "test_semantic": args.test_semantic,
        "models": {
            "v1": {
                "name": "point_envelope_vertical_scale",
                "description": "Global anchor at the deepest FIT point.",
            },
            "v2": {
                "name": "fit_geometry_envelope_vertical_scale",
                "description": (
                    "Local x-dependent anchor built only from FIT points and "
                    "observed FIT line segments."
                ),
                "envelope_bin_width_px": args.envelope_bin_width,
            },
        },
        "thresholds": {
            "max_fit_point_mean_degradation_px": args.max_fit_point_degradation,
            "max_fit_line_mean_perpendicular_degradation_px": (
                args.max_fit_line_degradation
            ),
            "min_relative_test_improvement": args.min_relative_improvement,
            "max_correction_delta_cv_for_promising": args.max_scale_cv,
        },
        "frames": frame_rows,
        "scale_dispersion": dispersion,
        "pairwise_transfer": pairwise,
        "leave_one_frame_out": loo,
        "summaries": summaries,
        "verdicts": verdicts,
        "horizontal_support_diagnostic": {
            "frames": support_rows,
            "hypothesis": support_hypothesis,
        },
        "skipped": skipped,
        "interpretation_notes": [
            (
                "Own-frame optimum uses that same frame's TEST line and is therefore "
                "only an upper bound, not evidence of generalization."
            ),
            (
                "Pairwise transfer learns a correction scale from one donor frame "
                "and evaluates it on a different target frame."
            ),
            (
                "Leave-one-frame-out learns from all frames except the target and "
                "is the main cross-view screening test."
            ),
            (
                "A promising verdict is still only screening evidence because the "
                "current dataset contains very few manually annotated views."
            ),
            (
                "V1 and V2 are non-projective diagnostic warps. Even a positive "
                "result does not make either one a production camera model."
            ),
        ],
    }

    report_path = output_dir / "report_v4.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pairwise_csv = output_dir / "pairwise_transfer_v4.csv"
    loo_csv = output_dir / "leave_one_out_v4.csv"

    write_pairwise_csv(
        pairwise_csv,
        pairwise["v1"],
        pairwise["v2"],
    )
    write_loo_csv(
        loo_csv,
        loo["v1"],
        loo["v2"],
    )

    support_csv = output_dir / "horizontal_support_v4.csv"
    write_support_csv(
        support_csv,
        support_rows,
    )

    if args.video:
        loo_by_frame = {
            int(row["held_out_frame"]): row
            for row in loo["v2"]
        }
        for record in records:
            render_loo_v2(
                record,
                loo_by_frame[record["frame"]],
                args.video,
                output_dir,
            )

    print("[homography-multiframe-v4]")
    print(f"  evaluated frames: {[record['frame'] for record in records]}")
    if skipped:
        print(f"  skipped inputs: {len(skipped)}")

    print()
    print("  own-frame correction scales (diagnostic upper bound)")
    for model_name in ("v1", "v2"):
        d = dispersion[model_name]
        scale_text = ", ".join(
            f"{row['frame']}={row['scale']:.5f}"
            for row in d["own_scales"]
        )
        cv_text = (
            "n/a"
            if d["delta_cv"] is None
            else f"{d['delta_cv']:.3f}"
        )
        print(
            f"    {model_name.upper()}: {scale_text} | "
            f"delta CV={cv_text} | "
            f"same direction={d['same_delta_direction']}"
        )

    print()
    print("  leave-one-frame-out transfer")
    for model_name in ("v1", "v2"):
        print(f"    {model_name.upper()}:")
        for row in loo[model_name]:
            line_deg = row["fit_line_perpendicular_degradation_px"]
            line_text = "n/a" if line_deg is None else f"{line_deg:+.2f}px"
            guard = "PASS" if row["passes_fit_guard"] else "FAIL"
            useful = "USEFUL" if row["useful_transfer"] else "NO"
            print(
                f"      hold {row['held_out_frame']}: "
                f"{row['baseline_test_mean_px']:.2f}px -> "
                f"{row['corrected_test_mean_px']:.2f}px "
                f"({row['relative_improvement'] * 100:+.1f}%, "
                f"s={row['learned_scale']:.5f}, "
                f"FIT point={row['fit_point_degradation_px']:+.2f}px, "
                f"FIT line={line_text}, guard={guard}, transfer={useful})"
            )

    print()
    print("  pairwise transfer summary")
    for model_name in ("v1", "v2"):
        summary = summaries[model_name]["pairwise"]
        print(
            f"    {model_name.upper()}: "
            f"{summary['useful_count']}/{summary['count']} useful "
            f"({summary['useful_ratio'] * 100:.1f}%), "
            f"{summary['worsened_count']} worsened"
        )

    print()
    print("  horizontal FIT-support diagnostic (V2)")
    for row in support_rows:
        print(
            f"    frame {row['frame']}: "
            f"FIT-x=[{row['fit_x_min_px']:.1f},{row['fit_x_max_px']:.1f}] "
            f"TEST-x=[{row['test_x_min_px']:.1f},{row['test_x_max_px']:.1f}] | "
            f"coverage={100.0 * row['test_fit_x_coverage_ratio']:.1f}% | "
            f"outside samples={100.0 * row['test_sample_outside_fit_x_fraction']:.1f}% | "
            f"max X extrap={row['max_horizontal_extrapolation_px']:.1f}px | "
            f"clamped={100.0 * row['v2_test_clamped_fraction']:.1f}% | "
            f"own V2 scale={row['v2_own_scale']:.5f}"
        )

    print()
    print("  support hypothesis")
    print(f"    {support_hypothesis['label']}")
    print(f"      {support_hypothesis['reason']}")
    rho = support_hypothesis[
        "spearman_outside_fraction_vs_v2_delta"
    ]
    rho_text = "n/a" if rho is None else f"{rho:+.3f}"
    ratio = support_hypothesis[
        "most_outside_delta_ratio_to_supported_median"
    ]
    ratio_text = "n/a" if ratio is None else f"{ratio:.2f}x"
    print(
        "      most outside frame="
        f"{support_hypothesis['most_horizontally_extrapolated_frame']} | "
        "largest V2 delta frame="
        f"{support_hypothesis['largest_v2_delta_frame']} | "
        f"rho(outside, delta)={rho_text} | "
        f"delta ratio={ratio_text}"
    )

    print()
    print("  screening verdict")
    for model_name in ("v1", "v2"):
        verdict = verdicts[model_name]
        print(f"    {model_name.upper()}: {verdict['label']}")
        print(f"      {verdict['reason']}")

    print()
    print(f"  report: {report_path}")
    print(f"  pairwise csv: {pairwise_csv}")
    print(f"  leave-one-out csv: {loo_csv}")
    print(f"  horizontal support csv: {support_csv}")
    if args.video:
        print(f"  visuals: {output_dir}/frame_*_loo_v2.jpg")

    print()
    print(
        "  Main question: is the large correction required by outlier frames "
        "associated with TEST geometry lying horizontally outside FIT support? "
        "Do not integrate any correction into FIELD from this diagnostic alone."
    )


if __name__ == "__main__":
    main()

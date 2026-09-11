#!/usr/bin/env python3
"""
Multi-frame homography extrapolation diagnostic V2 for RealJaenLab.

Compares two deliberately simple OFFLINE correction hypotheses:

V1 point envelope
    Correction starts below the lowest FIT point, globally.

V2 FIT-geometry envelope
    Correction starts below a local x-dependent envelope built ONLY from FIT
    point observations and FIT line observations.

The TEST bottom touchline is never used by the baseline homography. It is used
only to estimate/validate diagnostic correction scales.

This script is research-only. It does not modify reviews, HUMAN calibration,
FIELD, Camera Atlas, Camera State, or the web application.

Stable repository path:
    supervisor/check_homography_multiframe.py

Dependencies:
    numpy
    opencv-python
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

LAB_FORMAT = "realjaen.homography-lab"
LAB_VERSION = 1
REPORT_FORMAT = "realjaen.homography-multiframe-diagnostic"
REPORT_VERSION = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", action="append", type=Path, default=[])
    p.add_argument("--input-dir", type=Path, default=None)
    p.add_argument("--video", type=Path, default=None)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/homography_lab/outputs/multiframe_v2"),
    )
    p.add_argument("--test-semantic", default="bottom_touchline")
    p.add_argument("--samples", type=int, default=81)
    p.add_argument(
        "--envelope-bin-width",
        type=float,
        default=120.0,
        help="Horizontal bin width, in image pixels, for the FIT-geometry envelope.",
    )
    p.add_argument("--max-fit-point-degradation", type=float, default=1.0)
    p.add_argument("--max-fit-line-degradation", type=float, default=1.0)
    return p.parse_args()


def discover(args):
    paths = list(args.experiment)
    if args.input_dir:
        paths += sorted(args.input_dir.glob("homography_lab*_frame_*.json"))

    unique = []
    seen = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if not unique:
        raise ValueError("No experiments supplied.")
    return unique


def load_payload(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != LAB_FORMAT:
        raise ValueError(f"{path}: unexpected format")
    if payload.get("version") != LAB_VERSION:
        raise ValueError(f"{path}: unsupported version")
    H = np.asarray(payload.get("solve", {}).get("homography"), dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        raise ValueError(f"{path}: invalid/missing homography")
    return payload


def fit_constraints(payload):
    return [c for c in payload.get("constraints", []) if not bool(c.get("labTest"))]


def fit_points(payload):
    return [c for c in fit_constraints(payload) if c.get("type") == "point"]


def fit_lines(payload):
    return [c for c in fit_constraints(payload) if c.get("type") == "line"]


def test_line(payload, semantic):
    matches = [
        c for c in payload.get("constraints", [])
        if c.get("type") == "line"
        and c.get("semantic") == semantic
        and bool(c.get("labTest"))
    ]
    return matches[0] if len(matches) == 1 else None


def project(H, points):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ph = np.column_stack([pts, np.ones(len(pts))])
    q = (H @ ph.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return q[:, :2] / q[:, 2:3]


def line_from_points(a, b):
    p = np.asarray([a[0], a[1], 1.0], dtype=np.float64)
    q = np.asarray([b[0], b[1], 1.0], dtype=np.float64)
    line = np.cross(p, q)
    norm = math.hypot(float(line[0]), float(line[1]))
    if norm < 1e-12:
        raise ValueError("Degenerate line")
    return line / norm


def point_line_distance(line, point):
    return abs(float(line[0] * point[0] + line[1] * point[1] + line[2]))


def y_at_x(line, x):
    if abs(float(line[1])) < 1e-12:
        return None
    y = -(float(line[0]) * x + float(line[2])) / float(line[1])
    return float(y) if math.isfinite(y) else None


def image_segment(c):
    return np.asarray([
        [float(c["image"]["start"]["x"]), float(c["image"]["start"]["y"])],
        [float(c["image"]["end"]["x"]), float(c["image"]["end"]["y"])],
    ])


def world_segment(c):
    return np.asarray([
        [float(c["world"]["start"]["x"]), float(c["world"]["start"]["y"])],
        [float(c["world"]["end"]["x"]), float(c["world"]["end"]["y"])],
    ])


def sample_observed_test(payload, constraint, count):
    H = np.asarray(payload["solve"]["homography"], dtype=np.float64)
    obs = image_segment(constraint)
    pred = project(H, world_segment(constraint))
    obs_line = line_from_points(obs[0], obs[1])
    pred_line = line_from_points(pred[0], pred[1])

    x0, x1 = sorted([float(obs[0, 0]), float(obs[1, 0])])
    xs = np.linspace(x0, x1, max(3, count))
    oy, py, valid_x = [], [], []
    for x in xs:
        a = y_at_x(obs_line, float(x))
        b = y_at_x(pred_line, float(x))
        if a is not None and b is not None:
            valid_x.append(float(x))
            oy.append(a)
            py.append(b)
    if len(valid_x) < 3:
        raise ValueError("Could not sample TEST line")
    return {
        "x": np.asarray(valid_x),
        "observed_y": np.asarray(oy),
        "predicted_y": np.asarray(py),
        "observed_segment": obs,
        "predicted_segment": pred,
    }


class ConstantEnvelope:
    name = "point_envelope"

    def __init__(self, anchor):
        self.anchor = float(anchor)

    def __call__(self, x):
        values = np.asarray(x, dtype=np.float64)
        return np.full_like(values, self.anchor, dtype=np.float64)

    def serialise(self):
        return {"name": self.name, "anchor_y_px": self.anchor}


class GeometryEnvelope:
    name = "fit_geometry_envelope"

    def __init__(self, xs, ys, bin_width):
        self.bin_width = float(bin_width)
        self.support_x = np.asarray(xs, dtype=np.float64)
        self.support_y = np.asarray(ys, dtype=np.float64)

    def __call__(self, x):
        q = np.asarray(x, dtype=np.float64)
        return np.interp(
            q,
            self.support_x,
            self.support_y,
            left=float(self.support_y[0]),
            right=float(self.support_y[-1]),
        )

    def serialise(self):
        return {
            "name": self.name,
            "bin_width_px": self.bin_width,
            "support": [
                {"x_px": float(x), "anchor_y_px": float(y)}
                for x, y in zip(self.support_x, self.support_y)
            ],
        }


def make_point_envelope(payload):
    points = fit_points(payload)
    if not points:
        raise ValueError("No FIT points")
    return ConstantEnvelope(max(float(c["image"]["y"]) for c in points))


def make_geometry_envelope(payload, bin_width, line_samples=201):
    support = []

    for c in fit_points(payload):
        support.append((float(c["image"]["x"]), float(c["image"]["y"])))

    for c in fit_lines(payload):
        seg = image_segment(c)
        ts = np.linspace(0.0, 1.0, line_samples)
        pts = seg[0][None, :] * (1.0 - ts[:, None]) + seg[1][None, :] * ts[:, None]
        support.extend((float(x), float(y)) for x, y in pts)

    if not support:
        raise ValueError("No FIT geometry")

    sx = np.asarray([p[0] for p in support])
    sy = np.asarray([p[1] for p in support])
    xmin, xmax = float(sx.min()), float(sx.max())

    if xmax - xmin < 1e-6:
        return GeometryEnvelope([xmin, xmax + 1.0], [float(sy.max())] * 2, bin_width)

    edges = np.arange(xmin, xmax + bin_width, bin_width)
    if len(edges) < 2:
        edges = np.asarray([xmin, xmax])
    if edges[-1] < xmax:
        edges = np.append(edges, xmax)

    chosen = []
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
    collapsed = []
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


def warp_y(y, x, envelope, scale):
    y = np.asarray(y, dtype=np.float64)
    anchor = envelope(x)
    depth = np.maximum(y - anchor, 0.0)
    return y + (float(scale) - 1.0) * depth


def optimal_scale(record, envelope_key):
    sample = record["test_samples"]
    env = record[envelope_key]
    depth = np.maximum(sample["predicted_y"] - env(sample["x"]), 0.0)
    target = sample["observed_y"] - sample["predicted_y"]
    denom = float(np.dot(depth, depth))
    if denom < 1e-12:
        return 1.0
    return 1.0 + float(np.dot(depth, target) / denom)


def test_metrics(record, envelope_key, scale):
    sample = record["test_samples"]
    corrected = warp_y(
        sample["predicted_y"], sample["x"], record[envelope_key], scale
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


def fit_point_metrics(record, envelope_key, scale):
    payload = record["payload"]
    points = fit_points(payload)
    if not points:
        return {"count": 0}

    H = np.asarray(payload["solve"]["homography"], dtype=np.float64)
    world = np.asarray([[c["world"]["x"], c["world"]["y"]] for c in points], dtype=float)
    observed = np.asarray([[c["image"]["x"], c["image"]["y"]] for c in points], dtype=float)
    baseline = project(H, world)
    corrected = baseline.copy()
    corrected[:, 1] = warp_y(
        baseline[:, 1], baseline[:, 0], record[envelope_key], scale
    )

    base = np.linalg.norm(baseline - observed, axis=1)
    corr = np.linalg.norm(corrected - observed, axis=1)
    movement = np.linalg.norm(corrected - baseline, axis=1)
    return {
        "count": len(points),
        "baseline_mean_px": float(base.mean()),
        "corrected_mean_px": float(corr.mean()),
        "mean_degradation_px": float(corr.mean() - base.mean()),
        "max_projected_movement_px": float(movement.max()),
    }


def fit_line_metrics(record, envelope_key, scale, count):
    payload = record["payload"]
    H = np.asarray(payload["solve"]["homography"], dtype=np.float64)
    rows = []

    for c in fit_lines(payload):
        obs_seg = image_segment(c)
        obs_line = line_from_points(obs_seg[0], obs_seg[1])
        world_seg = world_segment(c)
        ts = np.linspace(0.0, 1.0, max(9, count))
        world = world_seg[0][None, :] * (1-ts[:, None]) + world_seg[1][None, :] * ts[:, None]
        baseline = project(H, world)
        corrected = baseline.copy()
        corrected[:, 1] = warp_y(
            baseline[:, 1], baseline[:, 0], record[envelope_key], scale
        )

        base = np.asarray([point_line_distance(obs_line, p) for p in baseline])
        corr = np.asarray([point_line_distance(obs_line, p) for p in corrected])
        movement = np.linalg.norm(corrected - baseline, axis=1)
        rows.append({
            "semantic": c.get("semantic"),
            "baseline_mean_perpendicular_px": float(base.mean()),
            "corrected_mean_perpendicular_px": float(corr.mean()),
            "mean_perpendicular_degradation_px": float(corr.mean() - base.mean()),
            "max_projected_movement_px": float(movement.max()),
        })

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
            np.mean([r["mean_perpendicular_degradation_px"] for r in rows])
        ),
        "max_projected_movement_px": float(
            max(r["max_projected_movement_px"] for r in rows)
        ),
        "per_line": rows,
    }


def evaluate(record, envelope_key, scale, samples):
    return {
        "scale": float(scale),
        "test": test_metrics(record, envelope_key, scale),
        "fit_points": fit_point_metrics(record, envelope_key, scale),
        "fit_lines": fit_line_metrics(record, envelope_key, scale, samples),
    }


def pooled_scale(records, envelope_key):
    numer = denom = 0.0
    for record in records:
        sample = record["test_samples"]
        env = record[envelope_key]
        depth = np.maximum(sample["predicted_y"] - env(sample["x"]), 0.0)
        target = sample["observed_y"] - sample["predicted_y"]
        numer += float(np.dot(depth, target))
        denom += float(np.dot(depth, depth))
    return 1.0 if denom < 1e-12 else 1.0 + numer / denom


def leave_one_out(records, envelope_key, samples, point_limit, line_limit):
    rows = []
    for held in records:
        donors = [r for r in records if r is not held]
        scale = pooled_scale(donors, envelope_key)
        ev = evaluate(held, envelope_key, scale, samples)
        point_deg = ev["fit_points"].get("mean_degradation_px") or 0.0
        line_deg = ev["fit_lines"].get("mean_perpendicular_degradation_px")
        line_deg_value = 0.0 if line_deg is None else line_deg
        rows.append({
            "held_out_frame": held["frame"],
            "donor_frames": [r["frame"] for r in donors],
            "learned_scale": float(scale),
            "baseline_test_mean_px": held["baseline_test"]["mean_abs_vertical_px"],
            "corrected_test_mean_px": ev["test"]["mean_abs_vertical_px"],
            "fit_point_degradation_px": point_deg,
            "fit_line_perpendicular_degradation_px": line_deg,
            "fit_line_max_projected_movement_px": ev["fit_lines"].get(
                "max_projected_movement_px", 0.0
            ),
            "passes_fit_guard": (
                point_deg <= point_limit and line_deg_value <= line_limit
            ),
        })
    return rows


def build_record(path, payload, semantic, samples, bin_width):
    test = test_line(payload, semantic)
    if test is None:
        return None

    record = {
        "path": path,
        "payload": payload,
        "frame": int(payload["frame"]),
        "point_envelope": make_point_envelope(payload),
        "geometry_envelope": make_geometry_envelope(payload, bin_width),
        "test_samples": sample_observed_test(payload, test, samples),
    }
    record["baseline_test"] = test_metrics(record, "point_envelope", 1.0)
    return record


def compact_model_record(record, envelope_key, samples):
    own_scale = optimal_scale(record, envelope_key)
    own = evaluate(record, envelope_key, own_scale, samples)
    return {
        "optimal_test_only_scale": float(own_scale),
        "optimal_test": own["test"],
        "optimal_fit_points": own["fit_points"],
        "optimal_fit_lines": own["fit_lines"],
        "envelope": record[envelope_key].serialise(),
    }


def load_video_frame(video, frame_number):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"Could not read frame {frame_number}")
    return frame


def draw_polyline(image, xs, ys, color, thickness):
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


def render(record, global_v2_scale, video, output_dir):
    image = load_video_frame(video, record["frame"])
    sample = record["test_samples"]

    baseline_y = sample["predicted_y"]
    corrected_y = warp_y(
        baseline_y,
        sample["x"],
        record["geometry_envelope"],
        global_v2_scale,
    )
    envelope_y = record["geometry_envelope"](sample["x"])

    # BGR: baseline cyan-ish, V2 magenta, TEST red, envelope yellow.
    draw_polyline(image, sample["x"], baseline_y, (255, 180, 40), 3)
    draw_polyline(image, sample["x"], corrected_y, (220, 60, 220), 3)
    draw_polyline(image, sample["x"], sample["observed_y"], (40, 80, 255), 4)
    draw_polyline(image, sample["x"], envelope_y, (0, 220, 220), 2)

    ev = evaluate(record, "geometry_envelope", global_v2_scale, 41)
    text = (
        f"frame {record['frame']} | TEST "
        f"{record['baseline_test']['mean_abs_vertical_px']:.1f}px -> "
        f"{ev['test']['mean_abs_vertical_px']:.1f}px | "
        f"V2 scale={global_v2_scale:.4f}"
    )
    cv2.putText(
        image, text, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255,255,255), 2, cv2.LINE_AA
    )
    cv2.imwrite(str(output_dir / f"frame_{record['frame']}_v2.jpg"), image)


def main():
    args = parse_args()
    records = []
    skipped = []

    for path in discover(args):
        payload = load_payload(path)
        record = build_record(
            path, payload, args.test_semantic, args.samples, args.envelope_bin_width
        )
        if record is None:
            skipped.append({
                "path": str(path),
                "frame": payload.get("frame"),
                "reason": f"no TEST {args.test_semantic}",
            })
        else:
            records.append(record)

    if len(records) < 2:
        raise ValueError("Need at least two experiments with a TEST bottom touchline.")
    records.sort(key=lambda r: r["frame"])

    global_v1 = pooled_scale(records, "point_envelope")
    global_v2 = pooled_scale(records, "geometry_envelope")
    loo_v1 = leave_one_out(
        records, "point_envelope", args.samples,
        args.max_fit_point_degradation, args.max_fit_line_degradation
    )
    loo_v2 = leave_one_out(
        records, "geometry_envelope", args.samples,
        args.max_fit_point_degradation, args.max_fit_line_degradation
    )

    frame_rows = []
    for record in records:
        diagnostics = record["payload"].get("solve", {}).get("diagnostics", {})
        coverage = diagnostics.get("calibration_strategy", {}).get(
            "evidence", {}
        ).get("coverage", {})
        frame_rows.append({
            "frame": record["frame"],
            "source": str(record["path"]),
            "fit_points": len(fit_points(record["payload"])),
            "fit_lines": len(fit_lines(record["payload"])),
            "baseline_reprojection_error_px": record["payload"]["solve"].get(
                "reprojection_error"
            ),
            "fit_image_span_y_ratio": coverage.get("image_span_y_ratio"),
            "baseline_test": record["baseline_test"],
            "v1": compact_model_record(record, "point_envelope", args.samples),
            "v2": compact_model_record(record, "geometry_envelope", args.samples),
        })

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "models": {
            "v1": {
                "name": "point_envelope_vertical_scale",
                "description": "Global anchor at deepest FIT point.",
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
        "guard_thresholds": {
            "max_fit_point_mean_degradation_px": args.max_fit_point_degradation,
            "max_fit_line_mean_perpendicular_degradation_px": args.max_fit_line_degradation,
        },
        "frames": frame_rows,
        "pooled_scales": {"v1": global_v1, "v2": global_v2},
        "leave_one_frame_out": {"v1": loo_v1, "v2": loo_v2},
        "skipped": skipped,
        "warning": (
            "V2 remains a diagnostic non-projective warp. Passing these tests "
            "does not make it a production camera model."
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "frame", "fit_points", "fit_lines", "baseline_test_mean_px",
            "v1_own_scale", "v1_own_test_mean_px",
            "v2_own_scale", "v2_own_test_mean_px",
            "v2_own_fit_point_degradation_px",
            "v2_own_fit_line_perpendicular_degradation_px",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in frame_rows:
            line_deg = row["v2"]["optimal_fit_lines"].get(
                "mean_perpendicular_degradation_px"
            )
            w.writerow({
                "frame": row["frame"],
                "fit_points": row["fit_points"],
                "fit_lines": row["fit_lines"],
                "baseline_test_mean_px": row["baseline_test"]["mean_abs_vertical_px"],
                "v1_own_scale": row["v1"]["optimal_test_only_scale"],
                "v1_own_test_mean_px": row["v1"]["optimal_test"]["mean_abs_vertical_px"],
                "v2_own_scale": row["v2"]["optimal_test_only_scale"],
                "v2_own_test_mean_px": row["v2"]["optimal_test"]["mean_abs_vertical_px"],
                "v2_own_fit_point_degradation_px": row["v2"]["optimal_fit_points"].get(
                    "mean_degradation_px"
                ),
                "v2_own_fit_line_perpendicular_degradation_px": line_deg,
            })

    if args.video:
        for record in records:
            render(record, global_v2, args.video, output_dir)

    print("[homography-multiframe-v2]")
    print(f"  evaluated frames: {[r['frame'] for r in records]}")
    if skipped:
        print(f"  skipped inputs: {len(skipped)}")
    print(f"  pooled scale V1 point-envelope: {global_v1:.6f}")
    print(f"  pooled scale V2 FIT-geometry-envelope: {global_v2:.6f}")
    print()
    print("  per-frame own TEST-only optimum (V1 -> V2)")
    for row in frame_rows:
        v1 = row["v1"]
        v2 = row["v2"]
        line_deg = v2["optimal_fit_lines"].get(
            "mean_perpendicular_degradation_px"
        )
        line_text = "n/a" if line_deg is None else f"{line_deg:+.2f}px"
        print(
            f"    frame {row['frame']}: baseline "
            f"{row['baseline_test']['mean_abs_vertical_px']:.2f}px | "
            f"V1 {v1['optimal_test']['mean_abs_vertical_px']:.2f}px "
            f"(s={v1['optimal_test_only_scale']:.5f}) | "
            f"V2 {v2['optimal_test']['mean_abs_vertical_px']:.2f}px "
            f"(s={v2['optimal_test_only_scale']:.5f}, "
            f"FIT-line degradation={line_text})"
        )

    print()
    print("  leave-one-frame-out V2")
    for row in loo_v2:
        line_deg = row["fit_line_perpendicular_degradation_px"]
        line_text = "n/a" if line_deg is None else f"{line_deg:+.2f}px"
        guard = "PASS" if row["passes_fit_guard"] else "FAIL"
        print(
            f"    hold frame {row['held_out_frame']}: "
            f"{row['baseline_test_mean_px']:.2f}px -> "
            f"{row['corrected_test_mean_px']:.2f}px "
            f"(scale={row['learned_scale']:.5f}, "
            f"FIT-point deg={row['fit_point_degradation_px']:+.2f}px, "
            f"FIT-line deg={line_text}, guard={guard})"
        )

    print()
    print(f"  report: {output_dir / 'report.json'}")
    print(f"  csv: {csv_path}")
    if args.video:
        print(f"  visuals: {output_dir}/frame_*_v2.jpg")
    print()
    print(
        "  V2 is promising only if TEST error transfers across frames AND all "
        "held-out FIT guards pass."
    )


if __name__ == "__main__":
    main()

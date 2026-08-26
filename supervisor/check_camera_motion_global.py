#!/usr/bin/env python3
"""Benchmark Global Camera Motion V1 against hidden HUMAN homographies.

The target HUMAN H is NEVER used to estimate camera motion.

For every consecutive pair of strong HUMAN anchors:

    HUMAN H_left
        + image-only global camera motion
        -> predicted H_right

The prediction is then compared with the hidden HUMAN H_right. The same test is
also run backwards.

This directly answers the Phase-2 question:

    Can we observe Veo camera motion without pitch lines / pitch H and transport
    a known calibration through time with useful accuracy?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from supervisor.field_camera_motion_global import (
    GlobalCameraMotionConfig,
    propagate_homography_with_global_motion,
)
from supervisor.field_camera_state_projective import (
    ProjectiveCameraStateConfig,
    evaluate_homography_manual_points,
    evaluate_homography_visible_grid,
)


DEFAULT_REVIEW = Path("supervisor/data/reviews/benchmark_easy_2min.json")


def calibration_for(
    calibrations: list[dict[str, Any]],
    frame: int,
    *,
    require_h: bool = False,
) -> dict[str, Any] | None:
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
            and (
                not require_h
                or isinstance(calibration.get("homography"), list)
            )
        ):
            return calibration
    return None


def visible_grid(
    predicted: Any,
    truth: Any,
    width: int,
    height: int,
    pitch_length: float,
    pitch_width: float,
) -> dict[str, Any] | None:
    result = evaluate_homography_visible_grid(
        predicted,
        truth,
        image_width=width,
        image_height=height,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=ProjectiveCameraStateConfig(dimensions=5),
    )
    return result if result.get("status") == "ok" else None


def fmt(value: Any, digits: int = 1) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def path_summary(path: dict[str, Any]) -> str:
    return (
        f"{path.get('status')} · "
        f"steps={path.get('step_count', '—')} · "
        f"inlier={fmt(path.get('median_inlier_ratio'), 2)} · "
        f"reproj={fmt(path.get('median_reprojection_px'), 2)}px · "
        f"sym={fmt(path.get('median_symmetric_error_px'), 2)}px"
    )


def evaluate_direction(
    *,
    video_path: Path,
    source_frame: int,
    target_frame: int,
    source_calibration: dict[str, Any],
    target_calibration: dict[str, Any],
    width: int,
    height: int,
    pitch_length: float,
    pitch_width: float,
    config: GlobalCameraMotionConfig,
) -> dict[str, Any]:
    path = propagate_homography_with_global_motion(
        video_path,
        source_frame,
        target_frame,
        source_calibration["homography"],
        config=config,
    )

    if path.get("status") != "ok":
        return {
            "source_frame": source_frame,
            "target_frame": target_frame,
            "path": path,
            "visible_grid": None,
            "manual_points": None,
        }

    predicted = path["homography"]
    truth = target_calibration["homography"]

    grid = visible_grid(
        predicted,
        truth,
        width,
        height,
        pitch_length,
        pitch_width,
    )
    manual = evaluate_homography_manual_points(
        predicted,
        target_calibration,
    )
    if manual.get("status") != "ok":
        manual = None

    return {
        "source_frame": source_frame,
        "target_frame": target_frame,
        "path": path,
        "visible_grid": grid,
        "manual_points": manual,
    }


def metric_values(
    records: list[dict[str, Any]],
    direction: str,
    metric: str,
) -> np.ndarray:
    values = []
    for record in records:
        result = record.get(direction)
        if not isinstance(result, dict):
            continue
        grid = result.get("visible_grid")
        if not isinstance(grid, dict):
            continue
        value = grid.get(metric)
        if value is not None:
            values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def print_summary(records: list[dict[str, Any]], direction: str) -> None:
    medians = metric_values(records, direction, "median_px")
    p90s = metric_values(records, direction, "p90_px")

    successful_paths = sum(
        1
        for record in records
        if isinstance(record.get(direction), dict)
        and record[direction].get("path", {}).get("status") == "ok"
    )

    print(
        f"{direction.upper():>8} paths={successful_paths}/{len(records)} · "
        f"evaluated={len(medians)}/{len(records)} · "
        f"median(grid median)={fmt(np.median(medians) if len(medians) else None)}px · "
        f"P90(grid median)={fmt(np.percentile(medians, 90) if len(medians) else None)}px · "
        f"max(grid median)={fmt(np.max(medians) if len(medians) else None)}px · "
        f"median(grid P90)={fmt(np.median(p90s) if len(p90s) else None)}px"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review",
        type=Path,
        default=DEFAULT_REVIEW,
        help="Review JSON containing strong HUMAN match_setup anchors.",
    )
    parser.add_argument(
        "--analysis-width",
        type=int,
        default=960,
    )
    parser.add_argument(
        "--step",
        type=int,
        default=2,
        help="Frame stride used by image-only camera-motion propagation.",
    )
    parser.add_argument(
        "--max-intervals",
        type=int,
        default=None,
        help="Optional quick-test limit.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional diagnostics JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    review = json.loads(args.review.read_text(encoding="utf-8"))
    field = review["field"]
    calibrations = field.get("calibrations", [])
    strong = sorted(
        {
            int(frame)
            for frame in field["match_setup"].get("completed_frames", [])
            if calibration_for(calibrations, int(frame), require_h=True) is not None
        }
    )

    if len(strong) < 2:
        raise SystemExit("Se necesitan al menos dos HUMAN fuertes.")

    pitch_length = float(field["pitch_model"]["length_m"])
    pitch_width = float(field["pitch_model"]["width_m"])
    video_path = Path("input") / review["video"]["filename"]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"No se pudo abrir {video_path}")
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
    fps = float(capture.get(cv2.CAP_PROP_FPS) or review["video"].get("fps") or 0)
    capture.release()

    cfg = GlobalCameraMotionConfig(
        analysis_width=max(320, int(args.analysis_width)),
        sample_step_frames=max(1, int(args.step)),
    )

    intervals = list(zip(strong[:-1], strong[1:]))
    if args.max_intervals is not None:
        intervals = intervals[: max(0, int(args.max_intervals))]

    print()
    print("GLOBAL CAMERA MOTION V1 · HUMAN-TO-HUMAN BENCHMARK")
    print("Motion estimator uses NO pitch H, NO field mask and NO pitch lines.")
    print("Target HUMAN H is used ONLY after propagation for evaluation.")
    print()
    print("Video:", video_path)
    print(f"Size: {width} × {height} · fps={fps:g}")
    print("Strong HUMAN anchors:", strong)
    print(f"Intervals tested: {len(intervals)}")
    print(
        f"analysis_width={cfg.analysis_width} · "
        f"step={cfg.sample_step_frames} frames · "
        f"min inliers={cfg.minimum_inliers} · "
        f"min ratio={cfg.minimum_inlier_ratio:.2f}"
    )
    print()

    records: list[dict[str, Any]] = []

    for left_frame, right_frame in intervals:
        left_cal = calibration_for(calibrations, left_frame, require_h=True)
        right_cal = calibration_for(calibrations, right_frame, require_h=True)
        if left_cal is None or right_cal is None:
            continue

        forward = evaluate_direction(
            video_path=video_path,
            source_frame=left_frame,
            target_frame=right_frame,
            source_calibration=left_cal,
            target_calibration=right_cal,
            width=width,
            height=height,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            config=cfg,
        )
        backward = evaluate_direction(
            video_path=video_path,
            source_frame=right_frame,
            target_frame=left_frame,
            source_calibration=right_cal,
            target_calibration=left_cal,
            width=width,
            height=height,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            config=cfg,
        )

        record = {
            "left_frame": left_frame,
            "right_frame": right_frame,
            "frame_span": right_frame - left_frame,
            "seconds": (
                (right_frame - left_frame) / fps
                if fps > 0
                else None
            ),
            "forward": forward,
            "backward": backward,
        }
        records.append(record)

        f_grid = forward.get("visible_grid")
        b_grid = backward.get("visible_grid")

        print(
            f"{left_frame:>6} ↔ {right_frame:<6} "
            f"span={right_frame-left_frame:>4}f "
            f"({fmt(record['seconds'], 2)}s)"
        )
        print(
            f"  FORWARD  {path_summary(forward['path'])} · "
            f"grid median={fmt(f_grid.get('median_px') if f_grid else None)}px · "
            f"P90={fmt(f_grid.get('p90_px') if f_grid else None)}px"
        )
        print(
            f"  BACKWARD {path_summary(backward['path'])} · "
            f"grid median={fmt(b_grid.get('median_px') if b_grid else None)}px · "
            f"P90={fmt(b_grid.get('p90_px') if b_grid else None)}px"
        )

    print()
    print("SUMMARY")
    print_summary(records, "forward")
    print_summary(records, "backward")

    both = []
    for record in records:
        for direction in ("forward", "backward"):
            grid = record.get(direction, {}).get("visible_grid")
            if isinstance(grid, dict) and grid.get("median_px") is not None:
                both.append(float(grid["median_px"]))
    both_values = np.asarray(both, dtype=np.float64)

    print(
        "COMBINED "
        f"evaluated={len(both_values)}/{2*len(records)} · "
        f"median={fmt(np.median(both_values) if len(both_values) else None)}px · "
        f"P90={fmt(np.percentile(both_values, 90) if len(both_values) else None)}px · "
        f"max={fmt(np.max(both_values) if len(both_values) else None)}px"
    )

    print()
    print("INTERPRETATION")
    print(
        "If propagation remains stable and endpoint errors are small, "
        "Global Camera Motion can become the H-independent backbone for "
        "trajectory/keyframes/loop-closure. Large drift means this layer "
        "must be improved before building the Camera Atlas."
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)

        def serializable(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {str(k): serializable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [serializable(v) for v in value]
            return value

        payload = {
            "version": 1,
            "algorithm": "global_camera_motion_v1",
            "review": str(args.review),
            "video": str(video_path),
            "width": width,
            "height": height,
            "fps": fps,
            "strong_anchors": strong,
            "config": serializable(cfg.__dict__),
            "records": serializable(records),
        }
        args.json_out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("JSON:", args.json_out)


if __name__ == "__main__":
    main()

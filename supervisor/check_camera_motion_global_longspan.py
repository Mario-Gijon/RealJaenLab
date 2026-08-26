#!/usr/bin/env python3
"""Long-span drift benchmark for Global Camera Motion V1.

Why this exists
---------------
The first CMG benchmark validates consecutive HUMAN-to-HUMAN intervals, mostly
1-5 seconds long. This benchmark deliberately removes intermediate HUMAN
resets.

It tests long spans from:
    - the FIRST strong HUMAN anchor to every later strong HUMAN anchor
    - the LAST strong HUMAN anchor to every earlier strong HUMAN anchor

For each pair, camera motion is estimated from the video ONLY. The destination
HUMAN H is hidden until final evaluation.

It also measures image-space round-trip closure:

    G_b_to_a @ G_a_to_b ~= I

This separates "camera-motion consistency" from HUMAN calibration differences.
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
    evaluate_homography_visible_grid,
)


DEFAULT_REVIEW = Path("supervisor/data/reviews/benchmark_easy_2min.json")


def calibration_for(
    calibrations: list[dict[str, Any]],
    frame: int,
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
            and isinstance(calibration.get("homography"), list)
        ):
            return calibration
    return None


def fmt(value: Any, digits: int = 1) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def visible_grid(
    predicted: Any,
    truth: Any,
    *,
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


def image_grid(width: int, height: int) -> np.ndarray:
    xs = np.linspace(0.08 * width, 0.92 * width, 9)
    ys = np.linspace(0.10 * height, 0.90 * height, 6)
    return np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)


def project(points: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.column_stack(
        (points, np.ones(len(points), dtype=np.float64))
    ) @ h.T
    den = q[:, 2]
    valid = (
        np.isfinite(q).all(axis=1)
        & np.isfinite(den)
        & (np.abs(den) > 1e-9)
    )
    out = np.full((len(points), 2), np.nan, dtype=np.float64)
    out[valid] = q[valid, :2] / den[valid, None]
    valid &= np.isfinite(out).all(axis=1)
    return out, valid


def closure_error(
    forward_path: dict[str, Any],
    backward_path: dict[str, Any],
    *,
    width: int,
    height: int,
) -> dict[str, Any] | None:
    if (
        forward_path.get("status") != "ok"
        or backward_path.get("status") != "ok"
    ):
        return None

    try:
        g_ab = np.asarray(
            forward_path["cumulative_transform"],
            dtype=np.float64,
        )
        g_ba = np.asarray(
            backward_path["cumulative_transform"],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None

    if g_ab.shape != (3, 3) or g_ba.shape != (3, 3):
        return None

    closure = g_ba @ g_ab
    if abs(float(closure[2, 2])) < 1e-9:
        return None
    closure /= float(closure[2, 2])

    points = image_grid(width, height)
    projected, valid = project(points, closure)
    if not valid.any():
        return None

    errors = np.linalg.norm(projected[valid] - points[valid], axis=1)
    return {
        "median_px": float(np.median(errors)),
        "p90_px": float(np.percentile(errors, 90)),
        "max_px": float(np.max(errors)),
        "matrix": closure.tolist(),
    }


def evaluate_pair(
    *,
    video_path: Path,
    source_frame: int,
    target_frame: int,
    source_h: Any,
    target_h: Any,
    width: int,
    height: int,
    pitch_length: float,
    pitch_width: float,
    config: GlobalCameraMotionConfig,
) -> dict[str, Any]:
    forward = propagate_homography_with_global_motion(
        video_path,
        source_frame,
        target_frame,
        source_h,
        config=config,
    )
    backward = propagate_homography_with_global_motion(
        video_path,
        target_frame,
        source_frame,
        target_h,
        config=config,
    )

    forward_grid = (
        visible_grid(
            forward["homography"],
            target_h,
            width=width,
            height=height,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )
        if forward.get("status") == "ok"
        else None
    )
    backward_grid = (
        visible_grid(
            backward["homography"],
            source_h,
            width=width,
            height=height,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )
        if backward.get("status") == "ok"
        else None
    )

    closure = closure_error(
        forward,
        backward,
        width=width,
        height=height,
    )

    return {
        "source_frame": int(source_frame),
        "target_frame": int(target_frame),
        "frame_span": abs(int(target_frame) - int(source_frame)),
        "forward": forward,
        "backward": backward,
        "forward_grid": forward_grid,
        "backward_grid": backward_grid,
        "roundtrip_closure": closure,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--analysis-width", type=int, default=960)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def compact_path(path: dict[str, Any]) -> str:
    return (
        f"{path.get('status')} · "
        f"steps={path.get('step_count','—')} · "
        f"inlier={fmt(path.get('median_inlier_ratio'),2)} · "
        f"reproj={fmt(path.get('median_reprojection_px'),2)}px"
    )


def main() -> None:
    args = parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))

    field = review["field"]
    calibrations = field.get("calibrations", [])
    strong = sorted(
        int(frame)
        for frame in field["match_setup"].get("completed_frames", [])
        if calibration_for(calibrations, int(frame)) is not None
    )

    if len(strong) < 3:
        raise SystemExit("Se necesitan al menos tres HUMAN fuertes.")

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
    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
        or review["video"].get("fps")
        or 0
    )
    capture.release()

    pitch_length = float(field["pitch_model"]["length_m"])
    pitch_width = float(field["pitch_model"]["width_m"])

    cfg = GlobalCameraMotionConfig(
        analysis_width=max(320, int(args.analysis_width)),
        sample_step_frames=max(1, int(args.step)),
    )

    first = strong[0]
    last = strong[-1]

    pairs: list[tuple[int, int, str]] = []
    for target in strong[1:]:
        pairs.append((first, target, "from_first"))
    for target in reversed(strong[:-1]):
        pairs.append((last, target, "from_last"))

    print()
    print("GLOBAL CAMERA MOTION V1 · LONG-SPAN DRIFT BENCHMARK")
    print("No intermediate HUMAN reset is allowed inside each tested span.")
    print("Destination HUMAN H is used ONLY for final evaluation.")
    print()
    print("Video:", video_path)
    print(f"Size: {width} × {height} · fps={fps:g}")
    print("Strong HUMAN anchors:", strong)
    print(
        f"analysis_width={cfg.analysis_width} · "
        f"step={cfg.sample_step_frames} frames"
    )
    print()

    records: list[dict[str, Any]] = []

    for source_frame, target_frame, family in pairs:
        source_cal = calibration_for(calibrations, source_frame)
        target_cal = calibration_for(calibrations, target_frame)
        if source_cal is None or target_cal is None:
            continue

        result = evaluate_pair(
            video_path=video_path,
            source_frame=source_frame,
            target_frame=target_frame,
            source_h=source_cal["homography"],
            target_h=target_cal["homography"],
            width=width,
            height=height,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            config=cfg,
        )
        result["family"] = family
        result["seconds"] = (
            result["frame_span"] / fps
            if fps > 0
            else None
        )
        records.append(result)

        fg = result["forward_grid"]
        bg = result["backward_grid"]
        cl = result["roundtrip_closure"]

        print(
            f"{source_frame:>6} → {target_frame:<6} "
            f"span={result['frame_span']:>4}f "
            f"({fmt(result['seconds'],2)}s) · {family}"
        )
        print(
            "  transport:",
            compact_path(result["forward"]),
            "· endpoint median=",
            f"{fmt(fg.get('median_px') if fg else None)}px",
            "· P90=",
            f"{fmt(fg.get('p90_px') if fg else None)}px",
        )
        print(
            "  reverse  :",
            compact_path(result["backward"]),
            "· endpoint median=",
            f"{fmt(bg.get('median_px') if bg else None)}px",
        )
        print(
            "  closure  :",
            (
                f"median={cl['median_px']:.2f}px · "
                f"P90={cl['p90_px']:.2f}px · "
                f"max={cl['max_px']:.2f}px"
                if cl is not None
                else "—"
            ),
        )

    print()
    print("SUMMARY BY SPAN")

    valid = [
        record
        for record in records
        if isinstance(record.get("forward_grid"), dict)
    ]
    valid.sort(key=lambda item: item["frame_span"])

    for record in valid:
        grid = record["forward_grid"]
        closure = record["roundtrip_closure"]
        print(
            f"{record['seconds']:>6.2f}s · "
            f"{record['source_frame']:>4}→{record['target_frame']:<4} · "
            f"endpoint={grid['median_px']:>6.1f}px · "
            f"P90={grid['p90_px']:>6.1f}px · "
            f"closure={fmt(closure.get('median_px') if closure else None,2):>6}px"
        )

    endpoint = np.asarray(
        [
            record["forward_grid"]["median_px"]
            for record in valid
        ],
        dtype=np.float64,
    )
    closure_values = np.asarray(
        [
            record["roundtrip_closure"]["median_px"]
            for record in valid
            if isinstance(record.get("roundtrip_closure"), dict)
        ],
        dtype=np.float64,
    )

    print()
    print(
        "ENDPOINT "
        f"n={len(endpoint)} · "
        f"median={fmt(np.median(endpoint) if len(endpoint) else None)}px · "
        f"P90={fmt(np.percentile(endpoint,90) if len(endpoint) else None)}px · "
        f"max={fmt(np.max(endpoint) if len(endpoint) else None)}px"
    )
    print(
        "ROUNDTRIP "
        f"n={len(closure_values)} · "
        f"median={fmt(np.median(closure_values) if len(closure_values) else None,2)}px · "
        f"P90={fmt(np.percentile(closure_values,90) if len(closure_values) else None,2)}px · "
        f"max={fmt(np.max(closure_values) if len(closure_values) else None,2)}px"
    )

    print()
    print("READING THE RESULT")
    print(
        "Low round-trip closure with growing HUMAN endpoint error means the "
        "image-motion estimator is internally consistent and part of the "
        "endpoint difference may come from accumulated model bias or HUMAN H "
        "differences. Rapidly growing closure means the raw trajectory itself "
        "is drifting and will require frequent relocalisation/keyframes."
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)

        def clean(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {str(k): clean(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(v) for v in value]
            return value

        args.json_out.write_text(
            json.dumps(
                {
                    "version": 1,
                    "benchmark": "global_camera_motion_longspan_v1",
                    "review": str(args.review),
                    "video": str(video_path),
                    "strong_anchors": strong,
                    "records": clean(records),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print("JSON:", args.json_out)


if __name__ == "__main__":
    main()

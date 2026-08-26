#!/usr/bin/env python3
"""Build and inspect Dense Camera Atlas V1 over the complete review video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from supervisor.field_camera_atlas_dense import (
    DenseCameraAtlasConfig,
    build_dense_camera_atlas,
)
from supervisor.field_camera_state_projective import (
    ProjectiveCameraStateConfig,
    evaluate_homography_visible_grid,
)


DEFAULT_REVIEW = Path("supervisor/data/reviews/benchmark_easy_2min.json")
DEFAULT_RELOCALIZATION = Path(
    "supervisor/data/reviews/camera_anchor_relocalization_v1.json"
)
DEFAULT_OUT = Path("supervisor/data/reviews/camera_atlas_dense_v1.json")


def calibration_for(calibrations, frame):
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


def fmt(value: Any, digits: int = 1) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument(
        "--relocalization",
        type=Path,
        default=DEFAULT_RELOCALIZATION,
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    review = json.loads(args.review.read_text(encoding="utf-8"))
    relocalization = json.loads(
        args.relocalization.read_text(encoding="utf-8")
    )

    field = review["field"]
    video_path = Path("input") / review["video"]["filename"]

    coverage = relocalization.get("coverage")
    if not isinstance(coverage, dict):
        raise SystemExit(
            "El JSON de relocalización no contiene `coverage`."
        )

    soft_anchors = [
        {
            "frame": int(result["frame"]),
            "homography": result["homography"],
            "confidence_tier": result.get(
                "confidence_tier", "verified"
            ),
            "anchor_frame": result.get("anchor_frame"),
            "score": result.get("score"),
            "match_count": result.get("match_count"),
            "inlier_count": result.get("inlier_count"),
            "inlier_ratio": result.get("inlier_ratio"),
            "median_reprojection_px": result.get(
                "median_reprojection_px"
            ),
            "direct_state_distance": result.get(
                "direct_state_distance"
            ),
        }
        for result in coverage.get("results", [])
        if (
            isinstance(result, dict)
            and result.get("status") == "ok"
            and isinstance(result.get("homography"), list)
        )
    ]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"No se pudo abrir {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()

    pitch_length = float(field["pitch_model"]["length_m"])
    pitch_width = float(field["pitch_model"]["width_m"])

    last_bucket = -1

    def progress(
        phase: str,
        done: int,
        total: int,
        left: int,
        right: int,
        covered: int,
    ) -> None:
        nonlocal last_bucket
        if phase != "intervals" or total <= 0:
            return
        percent = int(100 * done / total)
        bucket = percent // 10
        if bucket <= last_bucket:
            return
        last_bucket = bucket
        print(
            f"progress={min(percent,100):>3}% · "
            f"interval={left}→{right} · "
            f"covered={covered}"
        )

    print()
    print("DENSE CAMERA ATLAS V1")
    print(
        "HUMAN → direct keyframe relocalization → local Camera H_t."
    )
    print("No relocalized keyframe is promoted to HUMAN.")
    print()
    print("Video:", video_path)
    print(
        f"size={width}×{height} · fps={fps:g} · "
        f"frames={total_frames}"
    )
    print(
        f"soft keyframe anchors={len(soft_anchors)} · "
        f"high={sum(1 for item in soft_anchors if item.get('confidence_tier') == 'high')} · "
        f"verified={sum(1 for item in soft_anchors if item.get('confidence_tier') == 'verified')}"
    )
    print()

    result = build_dense_camera_atlas(
        video_path,
        soft_anchors=soft_anchors,
        image_width=width,
        image_height=height,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        start_frame=0,
        end_frame=max(0, total_frames - 1),
        config=DenseCameraAtlasConfig(),
        progress_callback=progress,
    )

    if result.get("status") != "ok":
        print(json.dumps(clean(result), indent=2, ensure_ascii=False))
        raise SystemExit(1)

    print()
    print("DENSE COVERAGE SUMMARY")
    print(
        f"frames={result['frame_range']['expected_count']} · "
        f"derived={result['derived_frame_count']} · "
        f"coverage={100.0*result['coverage_fraction']:.2f}% · "
        f"missing={result['missing_frame_count']}"
    )
    print("sources:", result["source_counts"])
    print(f"failures={len(result['failures'])}")

    # Compare dense predictions with exact HUMAN frames only as a diagnostic.
    # These values are NOT a strict holdout because the complete HUMAN set was
    # allowed to construct the atlas keyframes. The independent holdout proof is
    # ARL01's HUMAN LOO benchmark.
    records_by_frame = {
        int(record["frame"]): record
        for record in result["records"]
    }
    human_frames = sorted(
        {
            int(frame)
            for frame in field["match_setup"].get(
                "completed_frames", []
            )
        }
    )

    print()
    print("HUMAN FRAME CONSISTENCY (diagnostic, not holdout)")
    values = []
    for frame in human_frames:
        calibration = calibration_for(
            field.get("calibrations", []),
            frame,
        )
        record = records_by_frame.get(frame)
        if calibration is None or record is None:
            print(f"f={frame:>4} · unavailable")
            continue

        evaluation = evaluate_homography_visible_grid(
            record["homography"],
            calibration["homography"],
            image_width=width,
            image_height=height,
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            config=ProjectiveCameraStateConfig(dimensions=5),
        )

        median = (
            evaluation.get("median_px")
            if evaluation.get("status") == "ok"
            else None
        )
        if median is not None:
            values.append(float(median))

        print(
            f"f={frame:>4} · "
            f"source={record.get('source')} · "
            f"median={fmt(median)}px · "
            f"P90={fmt(evaluation.get('p90_px') if evaluation.get('status') == 'ok' else None)}px"
        )

    if values:
        array = np.asarray(values, dtype=np.float64)
        print(
            "consistency median="
            f"{np.median(array):.1f}px · "
            f"P90={np.percentile(array,90):.1f}px · "
            f"max={np.max(array):.1f}px"
        )

    print()
    print("FIRST MISSING FRAMES")
    if not result["missing_frames"]:
        print("None.")
    else:
        print(result["missing_frames"][:50])

    print()
    print("INTERPRETATION")
    print(
        "Near-100% dense coverage means the first HUMAN set is no longer "
        "temporally local: it has produced reusable Camera-Atlas keyframes, "
        "and those soft anchors are sufficient for the existing H_t engine "
        "to generate field homographies over the complete video."
    )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(clean(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print()
    print("JSON:", args.json_out)


if __name__ == "__main__":
    main()

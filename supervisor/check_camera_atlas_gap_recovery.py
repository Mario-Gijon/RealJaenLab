#!/usr/bin/env python3
"""Recover direct-transfer gaps with local Camera H_t and build target Dense Atlas.

This is the missing automatic-recovery stage before requesting another HUMAN.

Inputs
------
- benchmark_normal_2min_camera_anchor_update_v1.json
  Contains the merged target keyframe coverage after:
      7 source-video HUMAN anchors
      + explicit target active HUMAN anchors (currently frame 2526)

The merged keyframe set may still contain states that failed DIRECT
relocalization (currently 0, 30, 92).

Those direct failures are NOT automatically equivalent to "needs HUMAN".
If they lie between nearby relocalized keyframes, the existing Camera H_t engine
should recover them locally.

Pipeline
--------
directly known keyframe H
        ↓
local Camera H_t bridge / tail
        ↓
dense H for every video frame

No new HUMAN is introduced by this script.
"""

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


DEFAULT_VIDEO = Path("input/benchmark_normal_2min.mp4")
DEFAULT_REVIEW = Path(
    "supervisor/data/reviews/benchmark_normal_2min.json"
)
DEFAULT_UPDATE = Path(
    "supervisor/data/reviews/"
    "benchmark_normal_2min_camera_anchor_update_v1.json"
)
DEFAULT_OUT = Path(
    "supervisor/data/reviews/"
    "benchmark_normal_2min_camera_atlas_dense_v1.json"
)


def clean(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--update", type=Path, default=DEFAULT_UPDATE)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def soft_anchor_from_updated(item: dict[str, Any]) -> dict[str, Any] | None:
    if item.get("status") != "ok":
        return None
    if not isinstance(item.get("homography"), list):
        return None

    source = str(item.get("coverage_source") or "")
    confidence = str(item.get("confidence_tier") or "verified")

    # Preserve exact HUMAN provenance explicitly.
    if source == "target_active_human_exact":
        confidence = "human"

    return {
        "frame": int(item["frame"]),
        "homography": item["homography"],
        "confidence_tier": confidence,
        "anchor_frame": item.get("anchor_frame"),
        "score": item.get("score"),
        "match_count": item.get("match_count"),
        "inlier_count": item.get("inlier_count"),
        "inlier_ratio": item.get("inlier_ratio"),
        "median_reprojection_px": item.get("median_reprojection_px"),
        "direct_state_distance": item.get("direct_state_distance"),
        "coverage_source": source,
    }


def nearest_bracket(frames: list[int], target: int) -> tuple[int | None, int | None]:
    left = None
    right = None
    for frame in frames:
        if frame < target:
            left = frame
        elif frame > target:
            right = frame
            break
        else:
            return frame, frame
    return left, right


def main() -> None:
    args = parse_args()

    review = json.loads(args.review.read_text(encoding="utf-8"))
    update = json.loads(args.update.read_text(encoding="utf-8"))

    updated = update.get("updated")
    if not isinstance(updated, dict):
        raise SystemExit("El JSON AAU01 no contiene `updated`.")

    merged_results = updated.get("results")
    if not isinstance(merged_results, list):
        raise SystemExit("El JSON AAU01 no contiene `updated.results`.")

    soft_anchors = []
    for item in merged_results:
        if not isinstance(item, dict):
            continue
        anchor = soft_anchor_from_updated(item)
        if anchor is not None:
            soft_anchors.append(anchor)

    soft_anchors.sort(key=lambda item: int(item["frame"]))
    anchor_frames = [int(item["frame"]) for item in soft_anchors]

    if not soft_anchors:
        raise SystemExit("No hay keyframes con H conocida para densificar.")

    direct_unresolved = sorted(
        int(item["frame"])
        for item in merged_results
        if isinstance(item, dict) and item.get("status") != "ok"
    )

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"No se pudo abrir {args.video}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()

    field = review.get("field", {})
    pitch_model = field.get("pitch_model", {})
    if pitch_model.get("status") != "confirmed":
        raise SystemExit("El pitch_model target no está confirmado.")

    pitch_length = float(pitch_model["length_m"])
    pitch_width = float(pitch_model["width_m"])

    print()
    print("CAMERA ATLAS · LOCAL GAP RECOVERY")
    print()
    print("Target:", args.video)
    print(
        f"video={width}×{height} · fps={fps:g} · frames={total_frames}"
    )
    print(
        f"directly known keyframe H={len(soft_anchors)} · "
        f"direct unresolved keyframes={len(direct_unresolved)}"
    )
    print("Direct unresolved:", direct_unresolved)
    print()
    print("LOCAL BRACKETS FOR DIRECT FAILURES")
    for frame in direct_unresolved:
        left, right = nearest_bracket(anchor_frames, frame)
        print(
            f"f={frame:>5} · "
            f"left={left if left is not None else '—'} · "
            f"right={right if right is not None else '—'} · "
            f"left gap={frame-left if left is not None else '—'}f · "
            f"right gap={right-frame if right is not None else '—'}f"
        )

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
            f"dense={min(percent,100):>3}% · "
            f"interval={left}→{right} · "
            f"covered={covered}"
        )

    dense = build_dense_camera_atlas(
        args.video,
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

    if dense.get("status") != "ok":
        print(json.dumps(clean(dense), indent=2, ensure_ascii=False))
        raise SystemExit(1)

    records_by_frame = {
        int(record["frame"]): record
        for record in dense.get("records", [])
        if isinstance(record, dict)
    }

    print()
    print("DENSE RECOVERY SUMMARY")
    print(
        f"frames={dense['frame_range']['expected_count']} · "
        f"derived={dense['derived_frame_count']} · "
        f"coverage={100.0*dense['coverage_fraction']:.2f}% · "
        f"missing={dense['missing_frame_count']} · "
        f"failures={len(dense['failures'])}"
    )
    print("sources:", dense["source_counts"])

    print()
    print("FORMER DIRECT FAILURES AFTER LOCAL H_t")
    recovered = 0
    for frame in direct_unresolved:
        record = records_by_frame.get(frame)
        if record is None:
            print(f"f={frame:>5} · STILL MISSING")
            continue
        recovered += 1
        print(
            f"f={frame:>5} · RECOVERED · "
            f"source={record.get('source')} · "
            f"method={record.get('method')} · "
            f"left={record.get('atlas_left_keyframe', record.get('atlas_anchor_frame','—'))} · "
            f"right={record.get('atlas_right_keyframe','—')}"
        )

    print()
    print(
        f"direct failures automatically recovered="
        f"{recovered}/{len(direct_unresolved)}"
    )

    if recovered == len(direct_unresolved):
        print()
        print("ACTIVE ANCHOR DECISION")
        print(
            "No additional HUMAN is justified for these direct-transfer gaps. "
            "They are recoverable by local Camera H_t from already-known "
            "Camera Atlas states."
        )
    else:
        still_missing = [
            frame
            for frame in direct_unresolved
            if frame not in records_by_frame
        ]
        print()
        print("ACTIVE ANCHOR DECISION")
        print(
            "Only frames still missing after local H_t are legitimate "
            f"HUMAN candidates: {still_missing}"
        )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(clean(dense), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("JSON:", args.json_out)
    print(
        "PREVIEW already prefers this per-video dense filename. "
        "Reload benchmark_normal_2min.mp4 after generation."
    )


if __name__ == "__main__":
    main()

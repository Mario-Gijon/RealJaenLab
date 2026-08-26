#!/usr/bin/env python3
"""End-to-end Camera Atlas transfer benchmark across two clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from supervisor.field_camera_anchor_transfer import (
    CrossVideoAnchorTransferConfig,
    analyse_cross_video_anchor_transfer,
)
from supervisor.field_camera_atlas_dense import (
    DenseCameraAtlasConfig,
    build_dense_camera_atlas,
)
from supervisor.field_camera_motion_global import GlobalCameraMotionConfig
from supervisor.field_camera_trajectory import (
    CameraTrajectoryConfig,
    analyse_camera_trajectory,
)


DEFAULT_SOURCE_REVIEW = Path(
    "supervisor/data/reviews/benchmark_easy_2min.json"
)
DEFAULT_TARGET_VIDEO = Path("input/benchmark_normal_2min.mp4")


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


def timestamp(frame: int, fps: float) -> str:
    seconds = float(frame) / fps if fps > 0 else 0.0
    minute = int(seconds // 60)
    rest = seconds - minute * 60
    return f"{minute:02d}:{rest:05.2f}"


def fmt(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-review",
        type=Path,
        default=DEFAULT_SOURCE_REVIEW,
    )
    parser.add_argument(
        "--target-video",
        type=Path,
        default=DEFAULT_TARGET_VIDEO,
    )
    parser.add_argument("--analysis-width", type=int, default=960)
    parser.add_argument("--step", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_review = json.loads(
        args.source_review.read_text(encoding="utf-8")
    )
    source_video = Path("input") / source_review["video"]["filename"]
    target_video = args.target_video

    if not source_video.exists():
        raise SystemExit(f"No existe el vídeo fuente: {source_video}")
    if not target_video.exists():
        raise SystemExit(f"No existe el vídeo target: {target_video}")

    field = source_review["field"]
    calibrations = field.get("calibrations", [])
    strong_frames = sorted(
        {
            int(frame)
            for frame in field["match_setup"].get("completed_frames", [])
            if calibration_for(calibrations, int(frame)) is not None
        }
    )

    anchors = [
        {
            "frame": frame,
            "homography": calibration_for(calibrations, frame)["homography"],
        }
        for frame in strong_frames
    ]

    target_capture = cv2.VideoCapture(str(target_video))
    if not target_capture.isOpened():
        raise SystemExit(f"No se pudo abrir {target_video}")
    target_width = int(
        target_capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
    )
    target_height = int(
        target_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
    )
    target_fps = float(target_capture.get(cv2.CAP_PROP_FPS) or 0.0)
    target_frames = int(
        target_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    )
    target_capture.release()

    if target_frames <= 0:
        raise SystemExit("El vídeo target no tiene un frame_count válido.")

    safe_stem = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in target_video.stem
    )
    review_dir = Path("supervisor/data/reviews")
    trajectory_out = (
        review_dir / f"{safe_stem}_camera_trajectory_v1.json"
    )
    transfer_out = (
        review_dir / f"{safe_stem}_camera_anchor_transfer_v1.json"
    )
    dense_out = (
        review_dir / f"{safe_stem}_camera_atlas_dense_v1.json"
    )

    print()
    print("CAMERA ATLAS · CROSS-VIDEO TRANSFER")
    print()
    print("SOURCE VIDEO:", source_video)
    print("SOURCE HUMAN anchors:", strong_frames)
    print("TARGET VIDEO:", target_video)
    print(
        f"TARGET size={target_width}×{target_height} · "
        f"fps={target_fps:g} · frames={target_frames}"
    )
    print()
    print(
        "RULE: ZERO target HUMAN anchors. "
        "All target H must come from source HUMAN reuse."
    )

    motion_cfg = GlobalCameraMotionConfig(
        analysis_width=max(320, int(args.analysis_width)),
        sample_step_frames=max(1, int(args.step)),
    )
    trajectory_cfg = CameraTrajectoryConfig(
        sample_step_frames=max(1, int(args.step)),
    )

    last_trajectory_bucket = -1

    def trajectory_progress(
        done: int,
        total: int,
        frame: int,
        keyframes: int,
    ) -> None:
        nonlocal last_trajectory_bucket
        if total <= 0:
            return
        percent = int(100 * done / total)
        bucket = percent // 10
        if bucket <= last_trajectory_bucket:
            return
        last_trajectory_bucket = bucket
        print(
            f"trajectory={min(percent,100):>3}% · "
            f"frame={frame} · keyframes={keyframes}"
        )

    print()
    print("STEP 1 · TARGET CAMERA TRAJECTORY")

    trajectory = analyse_camera_trajectory(
        target_video,
        start_frame=0,
        end_frame=target_frames - 1,
        motion_config=motion_cfg,
        trajectory_config=trajectory_cfg,
        progress_callback=trajectory_progress,
    )

    if trajectory.get("status") != "ok":
        print(json.dumps(clean(trajectory), indent=2, ensure_ascii=False))
        raise SystemExit(1)

    trajectory_out.parent.mkdir(parents=True, exist_ok=True)
    trajectory_out.write_text(
        json.dumps(clean(trajectory), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print(
        f"trajectory samples={trajectory['sample_count']} · "
        f"keyframes={trajectory['keyframe_count']} · "
        f"coarse recurrences={trajectory['recurrence_edge_count']}"
    )

    targets = [
        {
            "frame": int(item["frame"]),
            "keyframe_index": index,
        }
        for index, item in enumerate(trajectory["keyframes"])
    ]

    transfer_cfg = CrossVideoAnchorTransferConfig(
        analysis_width=max(480, int(args.analysis_width))
    )

    last_target_bucket = -1

    def transfer_progress(
        phase: str,
        done: int,
        total: int,
        frame: int,
        covered: int,
    ) -> None:
        nonlocal last_target_bucket
        if phase != "targets" or total <= 0:
            return
        percent = int(100 * done / total)
        bucket = percent // 10
        if bucket <= last_target_bucket:
            return
        last_target_bucket = bucket
        print(
            f"transfer={min(percent,100):>3}% · "
            f"frame={frame} · covered={covered}/{done}"
        )

    print()
    print("STEP 2 · CROSS-VIDEO HUMAN RELOCALIZATION")

    transfer = analyse_cross_video_anchor_transfer(
        source_video,
        target_video,
        anchors=anchors,
        targets=targets,
        config=transfer_cfg,
        progress_callback=transfer_progress,
    )

    if transfer.get("status") != "ok":
        print(json.dumps(clean(transfer), indent=2, ensure_ascii=False))
        raise SystemExit(1)

    transfer_out.write_text(
        json.dumps(clean(transfer), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    coverage = 100.0 * float(transfer["coverage_fraction"])
    high = 100.0 * float(transfer["high_confidence_fraction"])

    print()
    print("TRANSFER SUMMARY")
    print(
        f"SOURCE HUMAN={transfer['anchor_count']} · "
        f"TARGET keyframes={transfer['target_count']} · "
        f"relocalized={transfer['covered_target_count']} "
        f"({coverage:.1f}%) · "
        f"HIGH={transfer['high_confidence_target_count']} "
        f"({high:.1f}%)"
    )
    print(
        f"verified source-anchor→target pairs="
        f"{transfer['verified_pair_count']} · "
        f"high-confidence pairs="
        f"{transfer['high_confidence_pair_count']}"
    )

    unresolved = [
        item
        for item in transfer["results"]
        if item.get("status") != "ok"
    ]
    non_high = [
        item
        for item in transfer["results"]
        if item.get("status") == "ok"
        and item.get("confidence_tier") != "high"
    ]

    print()
    print("UNRESOLVED TARGET KEYFRAMES")
    if not unresolved:
        print("None.")
    else:
        for item in unresolved[:40]:
            print(
                f"f={item['frame']:>5} "
                f"t={timestamp(item['frame'],target_fps)} · "
                f"verified={item.get('verified_candidate_count',0)}"
            )

    print()
    print("VERIFIED BUT NOT HIGH")
    if not non_high:
        print("None.")
    else:
        for item in non_high[:30]:
            print(
                f"f={item['frame']:>5} "
                f"t={timestamp(item['frame'],target_fps)} · "
                f"source HUMAN={item.get('anchor_frame','—')} · "
                f"inliers={item.get('inlier_count','—')} · "
                f"ratio={fmt(item.get('inlier_ratio'))} · "
                f"reproj={fmt(item.get('median_reprojection_px'))}px"
            )

    if transfer["covered_target_count"] != transfer["target_count"]:
        print()
        print("DENSE ATLAS NOT BUILT")
        print(
            "At least one target keyframe is unresolved. "
            "Do not hide that with one-sided flow: these are candidates "
            "for a genuinely new HUMAN anchor."
        )
        print()
        print("Trajectory JSON:", trajectory_out)
        print("Transfer JSON:", transfer_out)
        return

    print()
    print("STEP 3 · DENSE TARGET CAMERA ATLAS")

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
        for result in transfer["results"]
        if (
            result.get("status") == "ok"
            and isinstance(result.get("homography"), list)
        )
    ]

    pitch_length = float(field["pitch_model"]["length_m"])
    pitch_width = float(field["pitch_model"]["width_m"])

    last_dense_bucket = -1

    def dense_progress(
        phase: str,
        done: int,
        total: int,
        left: int,
        right: int,
        covered: int,
    ) -> None:
        nonlocal last_dense_bucket
        if phase != "intervals" or total <= 0:
            return
        percent = int(100 * done / total)
        bucket = percent // 10
        if bucket <= last_dense_bucket:
            return
        last_dense_bucket = bucket
        print(
            f"dense={min(percent,100):>3}% · "
            f"interval={left}→{right} · "
            f"covered={covered}"
        )

    dense = build_dense_camera_atlas(
        target_video,
        soft_anchors=soft_anchors,
        image_width=target_width,
        image_height=target_height,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        start_frame=0,
        end_frame=target_frames - 1,
        config=DenseCameraAtlasConfig(),
        progress_callback=dense_progress,
    )

    if dense.get("status") != "ok":
        print(json.dumps(clean(dense), indent=2, ensure_ascii=False))
        raise SystemExit(1)

    dense_out.write_text(
        json.dumps(clean(dense), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("DENSE TARGET SUMMARY")
    print(
        f"frames={dense['frame_range']['expected_count']} · "
        f"derived={dense['derived_frame_count']} · "
        f"coverage={100.0*dense['coverage_fraction']:.2f}% · "
        f"missing={dense['missing_frame_count']} · "
        f"failures={len(dense['failures'])}"
    )
    print("sources:", dense["source_counts"])

    print()
    print("OUTPUT")
    print("Trajectory:", trajectory_out)
    print("Transfer:", transfer_out)
    print("Dense Atlas:", dense_out)
    print()
    print(
        "The current PREVIEW endpoint already prefers this per-video "
        "dense filename automatically. Reload benchmark_normal_2min.mp4 "
        "and inspect PREVIEW."
    )


if __name__ == "__main__":
    main()

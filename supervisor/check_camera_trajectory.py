#!/usr/bin/env python3
"""CLI probe for Camera Trajectory V1.

Runs on the complete review video by default. No HUMAN geometry is used by the
trajectory builder. HUMAN frame numbers are printed only as temporal landmarks
after the analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from supervisor.field_camera_motion_global import GlobalCameraMotionConfig
from supervisor.field_camera_trajectory import (
    CameraTrajectoryConfig,
    analyse_camera_trajectory,
)


DEFAULT_REVIEW = Path("supervisor/data/reviews/benchmark_easy_2min.json")


def seconds(frame: int, fps: float) -> float:
    return float(frame) / fps if fps > 0 else 0.0


def timestamp(frame: int, fps: float) -> str:
    value = seconds(frame, fps)
    minutes = int(value // 60)
    secs = value - 60 * minutes
    return f"{minutes:02d}:{secs:05.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--analysis-width", type=int, default=960)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Optional prefix duration. Default: analyse full video.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional full trajectory JSON.",
    )
    return parser.parse_args()


def clean(value: Any) -> Any:
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def main() -> None:
    args = parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    video_path = Path("input") / review["video"]["filename"]

    fps_hint = float(review.get("video", {}).get("fps") or 0.0)
    end_frame = None
    if args.seconds is not None and fps_hint > 0:
        end_frame = max(0, int(round(float(args.seconds) * fps_hint)))

    mcfg = GlobalCameraMotionConfig(
        analysis_width=max(320, int(args.analysis_width)),
        sample_step_frames=max(1, int(args.step)),
    )
    tcfg = CameraTrajectoryConfig(
        sample_step_frames=max(1, int(args.step)),
    )

    last_progress_bucket = -1

    def progress(done: int, total: int, frame: int, keyframes: int) -> None:
        nonlocal last_progress_bucket
        if total <= 0:
            return
        percent = int(100 * done / total)
        bucket = percent // 10
        if bucket > last_progress_bucket:
            last_progress_bucket = bucket
            print(
                f"progress={min(percent,100):>3}% · "
                f"frame={frame} · keyframes={keyframes}"
            )

    print()
    print("CAMERA TRAJECTORY V1")
    print("Uses image motion only: NO H, NO pitch mask, NO field lines.")
    print("Keyframes are selected by geometric camera-state change, not time.")
    print()
    print("Video:", video_path)
    if args.seconds is not None:
        print(f"Requested prefix: {args.seconds:g}s")
    print(
        f"analysis_width={mcfg.analysis_width} · "
        f"step={tcfg.sample_step_frames} frames"
    )
    print(
        "keyframe thresholds: "
        f"median≥{tcfg.keyframe_median_displacement_px:g}px "
        f"or P90≥{tcfg.keyframe_p90_displacement_px:g}px"
    )
    print(
        "recurrence thresholds: "
        f"median≤{tcfg.recurrence_median_displacement_px:g}px · "
        f"P90≤{tcfg.recurrence_p90_displacement_px:g}px · "
        f"gap≥{tcfg.recurrence_min_gap_frames}f"
    )
    print()

    result = analyse_camera_trajectory(
        video_path,
        end_frame=end_frame,
        motion_config=mcfg,
        trajectory_config=tcfg,
        progress_callback=progress,
    )

    if result.get("status") != "ok":
        print()
        print("FAILED")
        print(json.dumps(clean(result), indent=2, ensure_ascii=False))
        raise SystemExit(1)

    video = result["video"]
    fps = float(video["fps"])
    start = int(video["start_frame"])
    end = int(video["end_frame"])

    print()
    print("TRAJECTORY SUMMARY")
    print(
        f"analysed={start}→{end} · "
        f"duration={seconds(end-start,fps):.2f}s · "
        f"samples={result['sample_count']}"
    )
    print(
        f"keyframes={result['keyframe_count']} · "
        f"coarse recurrence edges={result['recurrence_edge_count']} · "
        f"coarse regions={result['region_count']}"
    )

    keyframes = result["keyframes"]
    print()
    print("KEYFRAMES")
    for index, item in enumerate(keyframes[:40]):
        selection = item.get("selection", {})
        recurrence = item.get("best_recurrence")
        recurrence_text = ""
        if isinstance(recurrence, dict):
            recurrence_text = (
                f" · repeat≈{timestamp(recurrence['left_frame'],fps)}"
                f" ({recurrence['median_displacement_px']:.1f}px)"
            )
        print(
            f"#{index:02d} "
            f"f={item['frame']:>5} "
            f"t={timestamp(item['frame'],fps)} · "
            f"novelty={float(selection.get('median_displacement_px') or 0):>5.1f}px"
            f"{recurrence_text}"
        )
    if len(keyframes) > 40:
        print(f"... {len(keyframes)-40} more keyframes")

    regions = sorted(
        result["regions"],
        key=lambda item: (-int(item["visit_count"]), int(item["first_frame"])),
    )
    repeated_regions = [item for item in regions if int(item["visit_count"]) > 1]

    print()
    print("COARSE REPEATED REGIONS")
    if not repeated_regions:
        print("No repeated region passed the coarse accumulated-trajectory threshold.")
    else:
        for item in repeated_regions[:20]:
            times = [
                timestamp(frame, fps)
                for frame in item["frames"][:12]
            ]
            suffix = " ..." if len(item["frames"]) > 12 else ""
            print(
                f"region={item['region']:>3} · "
                f"visits={item['visit_count']:>2} · "
                f"times={', '.join(times)}{suffix}"
            )

    field = review.get("field", {})
    human_frames = sorted(
        {
            int(value)
            for value in field.get("match_setup", {}).get("completed_frames", [])
        }
    )
    print()
    print("HUMAN TEMPORAL LANDMARKS (not used by trajectory)")
    print(
        ", ".join(
            f"{frame}@{timestamp(frame,fps)}"
            for frame in human_frames
        )
        if human_frames
        else "—"
    )

    print()
    print("INTERPRETATION")
    print(
        "CTR01 is intentionally coarse. A high recurrence count is direct evidence "
        "that camera states revisit previously observed regions. A low recurrence "
        "count does NOT yet disprove repetition because long-term composition drift "
        "can separate identical distant views; the next layer will use direct "
        "keyframe-to-keyframe image matching for loop-closure validation."
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(clean(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print()
        print("JSON:", args.json_out)


if __name__ == "__main__":
    main()

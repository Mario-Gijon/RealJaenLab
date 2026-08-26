#!/usr/bin/env python3
"""CLI benchmark for direct visual Loop Closure V1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from supervisor.field_camera_loop_closure import (
    LoopClosureConfig,
    analyse_loop_closures,
)


DEFAULT_REVIEW = Path("supervisor/data/reviews/benchmark_easy_2min.json")
DEFAULT_TRAJECTORY = Path("supervisor/data/reviews/camera_trajectory_v1.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--analysis-width", type=int, default=960)
    parser.add_argument("--candidates-per-keyframe", type=int, default=10)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("supervisor/data/reviews/camera_loop_closure_v1.json"),
    )
    return parser.parse_args()


def timestamp(frame: int, fps: float) -> str:
    seconds = float(frame) / fps if fps > 0 else 0.0
    minutes = int(seconds // 60)
    rest = seconds - 60 * minutes
    return f"{minutes:02d}:{rest:05.2f}"


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
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))

    video_path = Path("input") / review["video"]["filename"]
    config = LoopClosureConfig(
        analysis_width=max(480, int(args.analysis_width)),
        candidates_per_keyframe=max(1, int(args.candidates_per_keyframe)),
    )

    last_feature_bucket = -1
    last_verify_bucket = -1

    def progress(
        phase: str,
        done: int,
        total: int,
        frame: int,
        verified: int,
        recurrent: int,
    ) -> None:
        nonlocal last_feature_bucket, last_verify_bucket
        if total <= 0:
            return
        percent = int(100 * done / total)
        bucket = percent // 10

        if phase == "features":
            if bucket <= last_feature_bucket:
                return
            last_feature_bucket = bucket
            print(
                f"features={min(percent,100):>3}% · "
                f"frame={frame}"
            )
        else:
            if bucket <= last_verify_bucket:
                return
            last_verify_bucket = bucket
            print(
                f"verify={min(percent,100):>3}% · "
                f"frame={frame} · "
                f"verified={verified} · "
                f"recurrences={recurrent}"
            )

    print()
    print("CAMERA LOOP CLOSURE V1")
    print("Direct distant-keyframe verification from image features.")
    print("NO HUMAN H, NO pitch lines, NO field mask.")
    print()
    print("Video:", video_path)
    print("Trajectory:", args.trajectory)
    print(
        f"keyframes={trajectory.get('keyframe_count','—')} · "
        f"coarse regions={trajectory.get('region_count','—')} · "
        f"coarse recurrence edges={trajectory.get('recurrence_edge_count','—')}"
    )
    print(
        f"analysis_width={config.analysis_width} · "
        f"candidates/keyframe={config.candidates_per_keyframe} · "
        f"gap≥{config.minimum_temporal_gap_frames}f"
    )
    print(
        "strict recurrence: "
        f"median≤{config.recurrence_median_displacement_px:g}px · "
        f"P90≤{config.recurrence_p90_displacement_px:g}px"
    )
    print()

    result = analyse_loop_closures(
        video_path,
        trajectory,
        config=config,
        progress_callback=progress,
    )

    if result.get("status") != "ok":
        print()
        print("FAILED")
        print(json.dumps(clean(result), indent=2, ensure_ascii=False))
        raise SystemExit(1)

    fps = float(result.get("video", {}).get("fps") or review["video"].get("fps") or 0.0)

    print()
    print("LOOP CLOSURE SUMMARY")
    print(
        f"keyframes={result['keyframe_count']} · "
        f"candidates={result['candidate_count']} · "
        f"verified overlaps={result['verified_edge_count']} · "
        f"strict recurrences={result['recurrence_edge_count']}"
    )
    print(
        f"CTR01 coarse regions={trajectory.get('region_count','—')} "
        f"→ direct recurrence regions={result['region_count']}"
    )

    recurrences = sorted(
        result["recurrence_edges"],
        key=lambda edge: (
            float(edge["direct_state_distance"]["median_px"]),
            -int(edge["frame_gap"]),
        ),
    )

    print()
    print("BEST DIRECT RECURRENCES")
    if not recurrences:
        print("No strict direct recurrence passed verification.")
    else:
        for edge in recurrences[:30]:
            direct = edge["direct_state_distance"]
            drift = edge.get("trajectory_disagreement")
            print(
                f"{timestamp(edge['left_frame'],fps)} "
                f"↔ {timestamp(edge['right_frame'],fps)} · "
                f"direct={direct['median_px']:.1f}px/"
                f"P90 {direct['p90_px']:.1f}px · "
                f"matches={edge['match_count']} · "
                f"inliers={edge['inlier_count']} "
                f"({edge['inlier_ratio']:.2f}) · "
                f"reproj={edge['median_reprojection_px']:.2f}px · "
                f"trajectory drift="
                f"{drift['median_px']:.1f}px"
                if isinstance(drift, dict)
                else
                f"{timestamp(edge['left_frame'],fps)} "
                f"↔ {timestamp(edge['right_frame'],fps)} · "
                f"direct={direct['median_px']:.1f}px · "
                f"inliers={edge['inlier_count']}"
            )

    regions = sorted(
        result["regions"],
        key=lambda item: (-int(item["visit_count"]), int(item["first_frame"])),
    )
    repeated = [item for item in regions if int(item["visit_count"]) > 1]

    print()
    print("DIRECT RECURRENCE REGIONS")
    if not repeated:
        print("No multi-visit region.")
    else:
        for item in repeated[:25]:
            times = [
                timestamp(frame, fps)
                for frame in item["frames"][:16]
            ]
            suffix = " ..." if len(item["frames"]) > 16 else ""
            print(
                f"region={item['region']:>3} · "
                f"visits={item['visit_count']:>2} · "
                f"times={', '.join(times)}{suffix}"
            )

    disagreements = [
        float(edge["trajectory_disagreement"]["median_px"])
        for edge in result["verified_edges"]
        if isinstance(edge.get("trajectory_disagreement"), dict)
    ]
    if disagreements:
        import numpy as np
        values = np.asarray(disagreements, dtype=np.float64)
        print()
        print(
            "TRAJECTORY DRIFT ON DIRECTLY VERIFIED EDGES "
            f"median={np.median(values):.1f}px · "
            f"P90={np.percentile(values,90):.1f}px · "
            f"max={np.max(values):.1f}px"
        )

    print()
    print("INTERPRETATION")
    print(
        "Verified overlaps prove that distant frames share enough static scene "
        "geometry to be related directly. Strict recurrences prove that the camera "
        "returns to nearly the same image-space state. Direct-vs-trajectory "
        "disagreement measures accumulated drift and will later be used as a "
        "loop-closure correction constraint in the Camera Atlas."
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

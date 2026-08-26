#!/usr/bin/env python3
"""Benchmark and coverage probe for Anchor Relocalization V1.

Two tests are performed:

1. HUMAN leave-one-out:
   hide each strong HUMAN target H, relocalize it from the OTHER HUMAN anchors,
   then compare the predicted H against the hidden target H.

2. Two-minute keyframe coverage:
   use the complete strong HUMAN anchor set from the first ~18 seconds and ask
   how many CTR01 keyframes across the full video can be directly relocalized.

Target HUMAN H is used only for evaluation in test 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from supervisor.field_camera_anchor_relocalization import (
    AnchorRelocalizationConfig,
    analyse_anchor_relocalization,
)
from supervisor.field_camera_state_projective import (
    ProjectiveCameraStateConfig,
    evaluate_homography_manual_points,
    evaluate_homography_visible_grid,
)


DEFAULT_REVIEW = Path("supervisor/data/reviews/benchmark_easy_2min.json")
DEFAULT_TRAJECTORY = Path("supervisor/data/reviews/camera_trajectory_v1.json")
DEFAULT_OUT = Path(
    "supervisor/data/reviews/camera_anchor_relocalization_v1.json"
)


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


def timestamp(frame: int, fps: float) -> str:
    value = float(frame) / fps if fps > 0 else 0.0
    minute = int(value // 60)
    second = value - 60 * minute
    return f"{minute:02d}:{second:05.2f}"


def fmt(value: Any, digits: int = 1) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--analysis-width", type=int, default=960)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def evaluate_prediction(
    predicted_h: Any,
    target_calibration: dict[str, Any],
    *,
    width: int,
    height: int,
    pitch_length: float,
    pitch_width: float,
) -> dict[str, Any]:
    grid = evaluate_homography_visible_grid(
        predicted_h,
        target_calibration["homography"],
        image_width=width,
        image_height=height,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=ProjectiveCameraStateConfig(dimensions=5),
    )
    manual = evaluate_homography_manual_points(
        predicted_h,
        target_calibration,
    )
    return {
        "visible_grid": grid,
        "manual_points": manual,
    }


def main() -> None:
    args = parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))

    field = review["field"]
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

    video_path = Path("input") / review["video"]["filename"]
    config = AnchorRelocalizationConfig(
        analysis_width=max(480, int(args.analysis_width)),
    )

    print()
    print("ANCHOR RELOCALIZATION V1")
    print("Reuse HUMAN camera states by direct distant-frame image geometry.")
    print("NO target pitch lines, NO target points, NO target HUMAN H.")
    print()
    print("Video:", video_path)
    print("Strong HUMAN anchors:", strong_frames)
    print(
        f"analysis_width={config.analysis_width} · "
        f"HIGH: inliers≥{config.high_confidence_min_inliers}, "
        f"ratio≥{config.high_confidence_min_inlier_ratio:.2f}, "
        f"reproj≤{config.high_confidence_max_reprojection_px:.2f}px"
    )

    # Test 1: LOO HUMAN relocalization.
    print()
    print("TEST 1 · HUMAN LEAVE-ONE-OUT")

    loo = analyse_anchor_relocalization(
        video_path,
        anchors=anchors,
        targets=[{"frame": frame} for frame in strong_frames],
        config=config,
        exclude_same_frame=True,
    )
    if loo.get("status") != "ok":
        print(json.dumps(clean(loo), indent=2, ensure_ascii=False))
        raise SystemExit(1)

    width = int(loo["video"]["width"])
    height = int(loo["video"]["height"])
    fps = float(loo["video"]["fps"])
    pitch_length = float(field["pitch_model"]["length_m"])
    pitch_width = float(field["pitch_model"]["width_m"])

    loo_records = []
    for result in loo["results"]:
        frame = int(result["frame"])
        target_cal = calibration_for(calibrations, frame)
        evaluation = None
        if result.get("status") == "ok" and target_cal is not None:
            evaluation = evaluate_prediction(
                result["homography"],
                target_cal,
                width=width,
                height=height,
                pitch_length=pitch_length,
                pitch_width=pitch_width,
            )

        record = {
            "frame": frame,
            "result": result,
            "evaluation": evaluation,
        }
        loo_records.append(record)

        grid = (
            evaluation.get("visible_grid")
            if isinstance(evaluation, dict)
            else None
        )
        manual = (
            evaluation.get("manual_points")
            if isinstance(evaluation, dict)
            else None
        )

        print(
            f"f={frame:>4} t={timestamp(frame,fps)} · "
            f"from={result.get('anchor_frame','—')} · "
            f"tier={result.get('confidence_tier','—')} · "
            f"inliers={result.get('inlier_count','—')} · "
            f"ratio={fmt(result.get('inlier_ratio'),2)} · "
            f"reproj={fmt(result.get('median_reprojection_px'),2)}px · "
            f"grid={fmt(grid.get('median_px') if isinstance(grid,dict) and grid.get('status')=='ok' else None)}px · "
            f"manual={fmt(manual.get('median_px') if isinstance(manual,dict) and manual.get('status')=='ok' else None)}px"
        )

    grid_values = np.asarray(
        [
            record["evaluation"]["visible_grid"]["median_px"]
            for record in loo_records
            if isinstance(record.get("evaluation"), dict)
            and isinstance(record["evaluation"].get("visible_grid"), dict)
            and record["evaluation"]["visible_grid"].get("status") == "ok"
        ],
        dtype=np.float64,
    )

    print()
    print(
        "LOO SUMMARY "
        f"covered={loo['covered_target_count']}/{loo['target_count']} · "
        f"high={loo['high_confidence_target_count']}/{loo['target_count']} · "
        f"grid median={fmt(np.median(grid_values) if len(grid_values) else None)}px · "
        f"P90={fmt(np.percentile(grid_values,90) if len(grid_values) else None)}px · "
        f"max={fmt(np.max(grid_values) if len(grid_values) else None)}px"
    )

    # Test 2: Full keyframe coverage.
    print()
    print("TEST 2 · FULL VIDEO KEYFRAME COVERAGE")
    keyframes = trajectory.get("keyframes", [])
    targets = [
        {
            "frame": int(item["frame"]),
            "keyframe_index": index,
        }
        for index, item in enumerate(keyframes)
    ]

    last_bucket = -1

    def progress(
        phase: str,
        done: int,
        total: int,
        frame: int,
        covered: int,
    ) -> None:
        nonlocal last_bucket
        if phase != "targets" or total <= 0:
            return
        percent = int(100 * done / total)
        bucket = percent // 10
        if bucket <= last_bucket:
            return
        last_bucket = bucket
        print(
            f"coverage={min(percent,100):>3}% · "
            f"frame={frame} · "
            f"covered={covered}/{done}"
        )

    coverage = analyse_anchor_relocalization(
        video_path,
        anchors=anchors,
        targets=targets,
        config=config,
        exclude_same_frame=False,
        progress_callback=progress,
    )
    if coverage.get("status") != "ok":
        print(json.dumps(clean(coverage), indent=2, ensure_ascii=False))
        raise SystemExit(1)

    print()
    print("COVERAGE SUMMARY")
    print(
        f"HUMAN anchors={coverage['anchor_count']} · "
        f"keyframes={coverage['target_count']} · "
        f"directly relocalized={coverage['covered_target_count']} "
        f"({100.0*coverage['coverage_fraction']:.1f}%) · "
        f"HIGH={coverage['high_confidence_target_count']} "
        f"({100.0*coverage['high_confidence_fraction']:.1f}%)"
    )
    print(
        f"verified anchor→keyframe pairs={coverage['verified_pair_count']} · "
        f"high-confidence pairs={coverage['high_confidence_pair_count']}"
    )

    unresolved = [
        item
        for item in coverage["results"]
        if item.get("status") != "ok"
    ]
    non_high = [
        item
        for item in coverage["results"]
        if item.get("status") == "ok"
        and item.get("confidence_tier") != "high"
    ]

    print()
    print("UNRESOLVED KEYFRAMES")
    if not unresolved:
        print("None.")
    else:
        for item in unresolved[:30]:
            print(
                f"f={item['frame']:>5} "
                f"t={timestamp(item['frame'],fps)} · "
                f"verified candidates={item.get('verified_candidate_count',0)}"
            )

    print()
    print("VERIFIED BUT NOT HIGH CONFIDENCE")
    if not non_high:
        print("None.")
    else:
        for item in non_high[:30]:
            print(
                f"f={item['frame']:>5} "
                f"t={timestamp(item['frame'],fps)} · "
                f"anchor={item.get('anchor_frame','—')} · "
                f"inliers={item.get('inlier_count','—')} · "
                f"ratio={fmt(item.get('inlier_ratio'),2)} · "
                f"reproj={fmt(item.get('median_reprojection_px'),2)}px"
            )

    print()
    print("INTERPRETATION")
    print(
        "High direct coverage means early HUMAN anchors already act as reusable "
        "camera-state references across the full two-minute video. Remaining "
        "unresolved/low-confidence keyframes are the places where graph propagation "
        "or a genuinely new HUMAN anchor may be useful."
    )

    payload = {
        "version": 1,
        "algorithm": "anchor_relocalization_benchmark_v1",
        "review": str(args.review),
        "trajectory": str(args.trajectory),
        "loo": {
            **{
                key: value
                for key, value in loo.items()
                if key != "results"
            },
            "records": loo_records,
        },
        "coverage": coverage,
    }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(clean(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print()
        print("JSON:", args.json_out)


if __name__ == "__main__":
    main()

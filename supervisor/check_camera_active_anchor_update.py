#!/usr/bin/env python3
"""Evaluate the marginal gain of newly added target ACTIVE ANCHOR HUMANs.

Experiment
----------
We already have a cross-video transfer:

    7 HUMAN from benchmark_easy_2min
        -> benchmark_normal_2min keyframes
        -> 155 / 163 covered

The Active Anchor selector recommended one new HUMAN at frame 2526 in the
target clip.

This script DOES NOT rerun or alter the original cross-video experiment.
Instead it:

1. Loads the original cross-video transfer result.
2. Loads exact manual `purpose="active_anchor"` HUMAN calibrations from the
   target review.
3. Relocalizes every target keyframe directly from those target HUMAN anchors.
4. Merges the two independent coverage sources:
       source-video HUMAN reuse
       target active-anchor HUMAN reuse
5. Reports the exact marginal gain caused by the new HUMAN(s).
6. Re-runs Active Anchor selection only on whatever remains unresolved.

No target HUMAN other than explicit `active_anchor` calibrations is used.
No pitch lines or target field points are used outside the saved HUMAN H itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from supervisor.field_camera_active_anchor import (
    ActiveAnchorConfig,
    select_next_active_anchor,
)
from supervisor.field_camera_anchor_relocalization import (
    AnchorRelocalizationConfig,
    analyse_anchor_relocalization,
)


DEFAULT_VIDEO = Path("input/benchmark_normal_2min.mp4")
DEFAULT_REVIEW = Path(
    "supervisor/data/reviews/benchmark_normal_2min.json"
)
DEFAULT_TRAJECTORY = Path(
    "supervisor/data/reviews/"
    "benchmark_normal_2min_camera_trajectory_v1.json"
)
DEFAULT_SOURCE_TRANSFER = Path(
    "supervisor/data/reviews/"
    "benchmark_normal_2min_camera_anchor_transfer_v1.json"
)
DEFAULT_OUT = Path(
    "supervisor/data/reviews/"
    "benchmark_normal_2min_camera_anchor_update_v1.json"
)
DEFAULT_SELECTION_OUT = Path(
    "supervisor/data/reviews/"
    "benchmark_normal_2min_active_anchor_after_update_v1.json"
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


def timestamp(frame: int, fps: float) -> str:
    seconds = float(frame) / fps if fps > 0 else 0.0
    minute = int(seconds // 60)
    rest = seconds - minute * 60
    return f"{minute:02d}:{rest:05.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=DEFAULT_TRAJECTORY,
    )
    parser.add_argument(
        "--source-transfer",
        type=Path,
        default=DEFAULT_SOURCE_TRANSFER,
    )
    parser.add_argument("--analysis-width", type=int, default=960)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--selection-json-out",
        type=Path,
        default=DEFAULT_SELECTION_OUT,
    )
    return parser.parse_args()


def active_human_anchors(review: dict[str, Any]) -> list[dict[str, Any]]:
    calibrations = review.get("field", {}).get("calibrations", [])
    anchors = []

    for calibration in calibrations:
        if not isinstance(calibration, dict):
            continue
        if calibration.get("source") != "manual":
            continue
        if calibration.get("purpose") != "active_anchor":
            continue
        if not isinstance(calibration.get("homography"), list):
            continue

        try:
            frame = int(calibration["frame"])
        except (KeyError, TypeError, ValueError):
            continue

        anchors.append(
            {
                "frame": frame,
                "homography": calibration["homography"],
                "calibration_id": calibration.get("id"),
                "source": "manual",
                "purpose": "active_anchor",
                "confidence_tier": "human",
            }
        )

    deduplicated: dict[int, dict[str, Any]] = {}
    for anchor in anchors:
        deduplicated[int(anchor["frame"])] = anchor
    return [
        deduplicated[frame]
        for frame in sorted(deduplicated)
    ]


def result_by_frame(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    output = {}
    for item in result.get("results", []):
        if not isinstance(item, dict):
            continue
        try:
            frame = int(item["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        output[frame] = item
    return output


def best_result(
    source_result: dict[str, Any] | None,
    active_result: dict[str, Any] | None,
    active_frames: set[int],
    frame: int,
) -> tuple[str | None, dict[str, Any] | None]:
    # Exact newly-created HUMAN always wins on its own frame.
    if frame in active_frames:
        active = active_result or {}
        exact = dict(active)
        exact["status"] = "ok"
        exact["frame"] = frame
        exact["coverage_source"] = "target_active_human_exact"
        exact["confidence_tier"] = "human"
        return "target_active_human_exact", exact

    candidates: list[tuple[str, dict[str, Any]]] = []
    if isinstance(source_result, dict) and source_result.get("status") == "ok":
        candidates.append(("source_human_cross_video", source_result))
    if isinstance(active_result, dict) and active_result.get("status") == "ok":
        candidates.append(("target_active_human_relocalization", active_result))

    if not candidates:
        return None, None

    def quality(candidate: tuple[str, dict[str, Any]]) -> tuple[float, int, float]:
        _name, item = candidate
        score = float(item.get("score") or 0.0)
        inliers = int(item.get("inlier_count") or 0)
        reproj = float(item.get("median_reprojection_px") or 999.0)
        return (score, inliers, -reproj)

    candidates.sort(key=quality, reverse=True)
    return candidates[0]


def main() -> None:
    args = parse_args()

    review = json.loads(args.review.read_text(encoding="utf-8"))
    trajectory = json.loads(
        args.trajectory.read_text(encoding="utf-8")
    )
    source_transfer = json.loads(
        args.source_transfer.read_text(encoding="utf-8")
    )

    anchors = active_human_anchors(review)
    if not anchors:
        raise SystemExit(
            "No hay ninguna calibración manual purpose='active_anchor' "
            "con H resuelta en el review target."
        )

    keyframes = trajectory.get("keyframes", [])
    targets = [
        {
            "frame": int(item["frame"]),
            "keyframe_index": index,
        }
        for index, item in enumerate(keyframes)
    ]

    source_by_frame = result_by_frame(source_transfer)
    original_unresolved = sorted(
        frame
        for frame, item in source_by_frame.items()
        if item.get("status") != "ok"
    )

    print()
    print("ACTIVE ANCHOR · MARGINAL COVERAGE UPDATE")
    print()
    print("Target video:", args.video)
    print(
        "Original cross-video coverage: "
        f"{source_transfer.get('covered_target_count','—')}/"
        f"{source_transfer.get('target_count','—')} "
        f"({100.0*float(source_transfer.get('coverage_fraction') or 0):.1f}%)"
    )
    print("Original unresolved:", original_unresolved)
    print(
        "Target ACTIVE HUMAN anchors:",
        [int(anchor["frame"]) for anchor in anchors],
    )
    print()

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
            f"active-transfer={min(percent,100):>3}% · "
            f"frame={frame} · covered={covered}/{done}"
        )

    active_transfer = analyse_anchor_relocalization(
        args.video,
        anchors=anchors,
        targets=targets,
        config=AnchorRelocalizationConfig(
            analysis_width=max(480, int(args.analysis_width)),
        ),
        exclude_same_frame=False,
        progress_callback=progress,
    )

    if active_transfer.get("status") != "ok":
        print(
            json.dumps(
                clean(active_transfer),
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(1)

    active_by_frame = result_by_frame(active_transfer)
    active_frames = {int(anchor["frame"]) for anchor in anchors}

    merged_results = []
    for target in targets:
        frame = int(target["frame"])
        source_result = source_by_frame.get(frame)
        active_result = active_by_frame.get(frame)

        coverage_source, chosen = best_result(
            source_result,
            active_result,
            active_frames,
            frame,
        )

        merged = {
            "frame": frame,
            "target": target,
            "status": "ok" if chosen is not None else "unresolved",
            "coverage_source": coverage_source,
            "source_transfer_status": (
                source_result.get("status")
                if isinstance(source_result, dict)
                else None
            ),
            "active_transfer_status": (
                active_result.get("status")
                if isinstance(active_result, dict)
                else None
            ),
        }

        if chosen is not None:
            for key in (
                "homography",
                "anchor_frame",
                "confidence_tier",
                "score",
                "match_count",
                "inlier_count",
                "inlier_ratio",
                "median_reprojection_px",
                "direct_state_distance",
            ):
                if key in chosen:
                    merged[key] = chosen[key]

            if frame in active_frames:
                exact_anchor = next(
                    anchor
                    for anchor in anchors
                    if int(anchor["frame"]) == frame
                )
                merged["homography"] = exact_anchor["homography"]
                merged["anchor_frame"] = frame
                merged["confidence_tier"] = "human"

        merged_results.append(merged)

    merged_covered = [
        item
        for item in merged_results
        if item["status"] == "ok"
    ]
    merged_unresolved = sorted(
        int(item["frame"])
        for item in merged_results
        if item["status"] != "ok"
    )

    original_unresolved_set = set(original_unresolved)
    remaining_set = set(merged_unresolved)
    newly_resolved = sorted(original_unresolved_set - remaining_set)

    source_only = sum(
        1
        for item in merged_results
        if item.get("coverage_source") == "source_human_cross_video"
    )
    active_relocalized = sum(
        1
        for item in merged_results
        if item.get("coverage_source")
        == "target_active_human_relocalization"
    )
    exact_human = sum(
        1
        for item in merged_results
        if item.get("coverage_source") == "target_active_human_exact"
    )

    total = len(targets)
    before = int(source_transfer.get("covered_target_count") or 0)
    after = len(merged_covered)

    print()
    print("MARGINAL GAIN")
    print(
        f"before={before}/{total} "
        f"({100.0*before/max(total,1):.1f}%)"
    )
    print(
        f"after ={after}/{total} "
        f"({100.0*after/max(total,1):.1f}%)"
    )
    print(
        f"gain  =+{after-before} keyframes "
        f"({100.0*(after-before)/max(total,1):.1f} percentage points)"
    )
    print()
    print("NEWLY RESOLVED ORIGINAL FAILURES")
    print(newly_resolved if newly_resolved else "None.")
    print()
    print("STILL UNRESOLVED")
    print(merged_unresolved if merged_unresolved else "None.")
    print()
    print(
        "Coverage provenance: "
        f"source-cross-video={source_only} · "
        f"active-relocalized={active_relocalized} · "
        f"active-exact-HUMAN={exact_human}"
    )

    print()
    print("ACTIVE HUMAN EFFECT ON ORIGINAL 8")
    for frame in original_unresolved:
        item = next(
            result
            for result in merged_results
            if int(result["frame"]) == frame
        )
        print(
            f"f={frame:>5} · "
            f"{'RESOLVED' if item['status']=='ok' else 'UNRESOLVED':<10} · "
            f"via={item.get('coverage_source') or '—'} · "
            f"anchor={item.get('anchor_frame','—')} · "
            f"inliers={item.get('inlier_count','—')} · "
            f"ratio="
            f"{float(item.get('inlier_ratio')):.2f}"
            if item.get("inlier_ratio") is not None
            else
            f"f={frame:>5} · "
            f"{'RESOLVED' if item['status']=='ok' else 'UNRESOLVED':<10} · "
            f"via={item.get('coverage_source') or '—'}"
        )

    next_selection = None
    if merged_unresolved:
        print()
        print("RECOMPUTING NEXT ACTIVE ANCHOR")
        next_selection = select_next_active_anchor(
            args.video,
            merged_unresolved,
            config=ActiveAnchorConfig(
                analysis_width=max(480, int(args.analysis_width)),
            ),
        )

        if next_selection.get("status") == "ok":
            best = next_selection["recommendations"][0]
            fps = float(next_selection["video"]["fps"])
            print(
                f"NEXT HUMAN: frame {best['frame']} "
                f"({timestamp(best['frame'],fps)}) · "
                f"expected gain={best['expected_gain_count']}/"
                f"{next_selection['unresolved_count']} · "
                f"covers={best['expected_unresolved_covered']}"
            )
            print(
                "This recommendation is AFTER incorporating the new HUMAN. "
                "Do not use the previous greedy plan."
            )
        else:
            print(
                json.dumps(
                    clean(next_selection),
                    indent=2,
                    ensure_ascii=False,
                )
            )
    else:
        print()
        print("ALL TARGET KEYFRAMES ARE NOW COVERED.")
        print("No additional HUMAN is currently justified.")

    payload = {
        "version": 1,
        "algorithm": "active_anchor_marginal_update_v1",
        "video": str(args.video),
        "review": str(args.review),
        "trajectory": str(args.trajectory),
        "source_transfer": str(args.source_transfer),
        "original": {
            "covered": before,
            "total": total,
            "unresolved": original_unresolved,
        },
        "active_human_anchors": anchors,
        "active_transfer": active_transfer,
        "updated": {
            "covered": after,
            "total": total,
            "coverage_fraction": after / max(float(total), 1.0),
            "newly_resolved_original_failures": newly_resolved,
            "unresolved": merged_unresolved,
            "results": merged_results,
        },
        "next_selection": next_selection,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(clean(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print()
    print("JSON:", args.json_out)

    if isinstance(next_selection, dict) and next_selection.get("status") == "ok":
        args.selection_json_out.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.selection_json_out.write_text(
            json.dumps(
                clean(next_selection),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print("Next selection JSON:", args.selection_json_out)


if __name__ == "__main__":
    main()

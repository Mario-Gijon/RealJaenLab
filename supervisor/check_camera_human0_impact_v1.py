"""Diagnose the real mechanism by which HUMAN frame 0 changed Córdoba CAMERA.

Read-only diagnostic.

This script does NOT ask "where should the next HUMAN be?" yet. It answers the
more fundamental question:

    "What exactly changed in sparse CAMERA when HUMAN 0 was introduced?"

Using the current Anchor Relocalization JSON, it compares:
- ACTUAL state: HUMAN 0 is available;
- COUNTERFACTUAL state: candidate source 0 is removed in memory.

For every trajectory keyframe it measures:
- whether HUMAN 0 directly verifies to the keyframe;
- whether HUMAN 0 becomes score-best;
- which previous HUMAN it displaces;
- how different the resulting field homography is from the counterfactual
  score-best homography;
- how broadly those changes are distributed over time and Planner V2 regions.

This is designed to reveal the signal an active HUMAN selector should optimize.
No review/CAMERA/Dense artifact is modified.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from supervisor.field_workflow import extract_human_anchors
from supervisor.field_camera_sparse_validator_v1 import (
    SparseCameraValidatorConfig,
    compare_target_homographies,
)


ROOT = Path(__file__).resolve().parents[1]
ASSET = "asset_7fbc47b7b83647bc8fd52e6a676c9425"

REVIEWS = ROOT / "supervisor" / "data" / "reviews"

VIDEO = (
    ROOT
    / "supervisor"
    / "data"
    / "library"
    / "videos"
    / "managed_upload_559cfa321d00455d9ef8b6465dc23594.mp4"
)

REVIEW = REVIEWS / f"{ASSET}.json"
RELOC = REVIEWS / f"{ASSET}_camera_anchor_relocalization_v1.json"
PLANNER_V2 = REVIEWS / f"{ASSET}_camera_anchor_planner_v2.json"

OUTPUT = REVIEWS / f"{ASSET}_camera_human0_impact_v1.json"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _valid_candidates(result):
    rows = []

    for candidate in result.get("candidates", []):
        if not isinstance(candidate, dict):
            continue

        verification = candidate.get("verification")
        if (
            not isinstance(verification, dict)
            or verification.get("status") != "ok"
        ):
            continue

        if not isinstance(candidate.get("homography"), list):
            continue

        try:
            anchor = int(candidate["anchor_frame"])
            score = float(candidate["score"])
        except (KeyError, TypeError, ValueError):
            continue

        if not math.isfinite(score):
            continue

        rows.append(
            {
                "anchor_frame": anchor,
                "score": score,
                "homography": candidate["homography"],
                "verification": verification,
            }
        )

    rows.sort(
        key=lambda item: (
            -item["score"],
            -int(item["verification"].get("inlier_count") or 0),
            float(
                item["verification"].get(
                    "median_reprojection_px"
                )
                or float("inf")
            ),
            item["anchor_frame"],
        )
    )

    return rows


def _distribution(values):
    clean = np.asarray(
        [
            float(value)
            for value in values
            if value is not None
            and math.isfinite(float(value))
        ],
        dtype=np.float64,
    )

    if not len(clean):
        return {
            "count": 0,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
        }

    return {
        "count": int(len(clean)),
        "median": float(np.median(clean)),
        "p75": float(np.percentile(clean, 75)),
        "p90": float(np.percentile(clean, 90)),
        "p95": float(np.percentile(clean, 95)),
        "max": float(np.max(clean)),
    }


def _region_map(planner):
    frame_to_region = {}
    info = {}

    for region in planner.get("recommended_regions", []):
        if not isinstance(region, dict):
            continue

        try:
            region_id = int(
                region.get(
                    "temporal_rank",
                    region.get("region"),
                )
            )
        except (TypeError, ValueError):
            continue

        info[region_id] = region

        for frame in region.get("frames", []):
            try:
                frame_to_region[int(frame)] = region_id
            except (TypeError, ValueError):
                pass

    return frame_to_region, info


def _fmt(value, digits=1):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def main():
    print("=" * 104)
    print("HUMAN 0 IMPACT DIAGNOSTIC V1 · CÓRDOBA")
    print("=" * 104)

    for label, path in (
        ("video", VIDEO),
        ("review", REVIEW),
        ("relocalization", RELOC),
        ("planner V2", PLANNER_V2),
    ):
        if not path.exists():
            print(f"ERROR: falta {label}: {path}")
            return 2

    review = _load(REVIEW)
    relocalization = _load(RELOC)
    planner_v2 = _load(PLANNER_V2)

    humans = sorted(
        int(item["frame"])
        for item in extract_human_anchors(review)
    )

    if 0 not in humans:
        print("ERROR: HUMAN 0 no existe en la review actual.")
        return 2

    field = review.get("field", {})
    pitch = field.get("pitch_model", {})

    try:
        pitch_length = float(pitch["length_m"])
        pitch_width = float(pitch["width_m"])
    except (KeyError, TypeError, ValueError):
        print("ERROR: pitch_model inválido.")
        return 2

    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        print("ERROR: no se pudo abrir vídeo.")
        return 2

    image_width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
    )
    image_height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
    )
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    )
    capture.release()

    frame_to_region, region_info = _region_map(planner_v2)

    compare_cfg = SparseCameraValidatorConfig(
        grid_x=21,
        grid_y=15,
        image_margin_px=8.0,
        minimum_common_grid_points=8,
    )

    started = time.perf_counter()

    rows = []
    source0_verified_count = 0
    source0_best_count = 0
    changed_winner_count = 0
    counterfactual_unresolved_count = 0

    displaced_counts = {}
    source0_metrics = {
        "median_state_px": [],
        "p90_state_px": [],
        "inliers": [],
        "inlier_ratio": [],
        "reprojection_px": [],
    }
    winner_change_field_delta = []

    for result in relocalization.get("results", []):
        if not isinstance(result, dict):
            continue

        try:
            frame = int(result["frame"])
        except (KeyError, TypeError, ValueError):
            continue

        candidates = _valid_candidates(result)
        candidate0 = next(
            (
                item
                for item in candidates
                if int(item["anchor_frame"]) == 0
            ),
            None,
        )

        actual_best = candidates[0] if candidates else None
        before_candidates = [
            item
            for item in candidates
            if int(item["anchor_frame"]) != 0
        ]
        before_best = (
            before_candidates[0]
            if before_candidates
            else None
        )

        if candidate0 is not None:
            source0_verified_count += 1

            verification = candidate0["verification"]
            distance = verification.get("direct_state_distance") or {}

            source0_metrics["median_state_px"].append(
                _finite(distance.get("median_px"))
            )
            source0_metrics["p90_state_px"].append(
                _finite(distance.get("p90_px"))
            )
            source0_metrics["inliers"].append(
                _finite(verification.get("inlier_count"))
            )
            source0_metrics["inlier_ratio"].append(
                _finite(verification.get("inlier_ratio"))
            )
            source0_metrics["reprojection_px"].append(
                _finite(
                    verification.get(
                        "median_reprojection_px"
                    )
                )
            )

        actual_anchor = (
            int(actual_best["anchor_frame"])
            if actual_best is not None
            else None
        )
        before_anchor = (
            int(before_best["anchor_frame"])
            if before_best is not None
            else None
        )

        source0_is_best = actual_anchor == 0

        if source0_is_best:
            source0_best_count += 1

        changed = (
            actual_anchor != before_anchor
            and actual_anchor is not None
        )

        if changed:
            changed_winner_count += 1

            displaced_counts[before_anchor] = (
                displaced_counts.get(before_anchor, 0) + 1
            )

        if before_best is None:
            counterfactual_unresolved_count += 1

        comparison = None

        if (
            source0_is_best
            and before_best is not None
        ):
            comparison = compare_target_homographies(
                actual_best["homography"],
                before_best["homography"],
                image_width=image_width,
                image_height=image_height,
                pitch_length_m=pitch_length,
                pitch_width_m=pitch_width,
                config=compare_cfg,
            )

            if comparison.get("status") == "ok":
                winner_change_field_delta.append(
                    float(comparison["median_px"])
                )

        rows.append(
            {
                "frame": frame,
                "time_s": (
                    frame / fps
                    if fps > 0
                    else None
                ),
                "planner_region": frame_to_region.get(frame),
                "source0_verified": candidate0 is not None,
                "source0_score": (
                    float(candidate0["score"])
                    if candidate0 is not None
                    else None
                ),
                "source0_is_best": source0_is_best,
                "actual_best_anchor": actual_anchor,
                "counterfactual_best_anchor": before_anchor,
                "winner_changed_by_human0": changed,
                "counterfactual_unresolved": (
                    before_best is None
                ),
                "actual_candidate_count": len(candidates),
                "counterfactual_candidate_count": len(
                    before_candidates
                ),
                "actual_best_score": (
                    float(actual_best["score"])
                    if actual_best is not None
                    else None
                ),
                "counterfactual_best_score": (
                    float(before_best["score"])
                    if before_best is not None
                    else None
                ),
                "winner_field_delta": comparison,
            }
        )

    target_count = len(rows)

    # Region impact.
    regions = []

    for region_id in sorted(region_info):
        members = [
            row
            for row in rows
            if row.get("planner_region") == region_id
        ]

        if not members:
            continue

        verified = sum(
            1
            for row in members
            if row["source0_verified"]
        )
        best = sum(
            1
            for row in members
            if row["source0_is_best"]
        )
        changed = sum(
            1
            for row in members
            if row["winner_changed_by_human0"]
        )
        deltas = [
            row["winner_field_delta"]["median_px"]
            for row in members
            if isinstance(row.get("winner_field_delta"), dict)
            and row["winner_field_delta"].get("status") == "ok"
        ]

        representative = (
            region_info[region_id].get("representative") or {}
        )

        regions.append(
            {
                "region": region_id,
                "representative_frame": representative.get("frame"),
                "target_count": len(members),
                "source0_verified_count": verified,
                "source0_verified_fraction": (
                    verified / len(members)
                ),
                "source0_best_count": best,
                "source0_best_fraction": (
                    best / len(members)
                ),
                "winner_changed_count": changed,
                "winner_changed_fraction": (
                    changed / len(members)
                ),
                "field_delta_distribution_px": _distribution(
                    deltas
                ),
            }
        )

    regions.sort(
        key=lambda item: (
            -item["winner_changed_fraction"],
            -item["source0_best_fraction"],
            item["region"],
        )
    )

    # Temporal impact in equal 30-second bins if fps is valid.
    temporal_bins = []

    if fps > 0 and frame_count > 0:
        bin_seconds = 30.0
        duration = frame_count / fps
        start = 0.0

        while start < duration:
            end = min(start + bin_seconds, duration)

            members = [
                row
                for row in rows
                if row["time_s"] is not None
                and start <= row["time_s"] < end
            ]

            if members:
                verified = sum(
                    row["source0_verified"]
                    for row in members
                )
                best = sum(
                    row["source0_is_best"]
                    for row in members
                )
                changed = sum(
                    row["winner_changed_by_human0"]
                    for row in members
                )

                temporal_bins.append(
                    {
                        "start_s": start,
                        "end_s": end,
                        "target_count": len(members),
                        "source0_verified_count": verified,
                        "source0_best_count": best,
                        "winner_changed_count": changed,
                        "winner_changed_fraction": (
                            changed / len(members)
                        ),
                    }
                )

            start = end

    changed_rows = [
        row
        for row in rows
        if row["winner_changed_by_human0"]
    ]

    # Show the changed keyframes with largest geometric consequence.
    changed_with_delta = [
        row
        for row in changed_rows
        if isinstance(row.get("winner_field_delta"), dict)
        and row["winner_field_delta"].get("status") == "ok"
    ]

    changed_with_delta.sort(
        key=lambda row: (
            -float(row["winner_field_delta"]["median_px"]),
            row["frame"],
        )
    )

    elapsed = time.perf_counter() - started

    result = {
        "status": "ok",
        "version": 1,
        "algorithm": "human0_impact_diagnostic_v1",
        "human_frames": humans,
        "target_count": target_count,
        "source0_verified_count": source0_verified_count,
        "source0_verified_fraction": (
            source0_verified_count / target_count
            if target_count
            else 0.0
        ),
        "source0_best_count": source0_best_count,
        "source0_best_fraction": (
            source0_best_count / target_count
            if target_count
            else 0.0
        ),
        "winner_changed_count": changed_winner_count,
        "winner_changed_fraction": (
            changed_winner_count / target_count
            if target_count
            else 0.0
        ),
        "counterfactual_unresolved_count": (
            counterfactual_unresolved_count
        ),
        "displaced_anchor_counts": {
            str(key): value
            for key, value in sorted(
                displaced_counts.items(),
                key=lambda item: (
                    -item[1],
                    str(item[0]),
                ),
            )
        },
        "source0_direct_metrics": {
            key: _distribution(values)
            for key, values in source0_metrics.items()
        },
        "winner_change_field_delta_px": (
            _distribution(winner_change_field_delta)
        ),
        "regions": regions,
        "temporal_bins_30s": temporal_bins,
        "largest_winner_changes": changed_with_delta[:40],
        "targets": rows,
        "elapsed_s": elapsed,
    }

    print()
    print("HUMAN actuales:", humans)
    print("targets:", target_count)
    print()
    print("DIRECT REACH OF HUMAN 0")
    print("-----------------------")
    print(
        "verified to:",
        f"{source0_verified_count}/{target_count}",
        f"({100 * result['source0_verified_fraction']:.1f}%)",
    )
    print(
        "score-best on:",
        f"{source0_best_count}/{target_count}",
        f"({100 * result['source0_best_fraction']:.1f}%)",
    )
    print(
        "winner changed vs counterfactual:",
        f"{changed_winner_count}/{target_count}",
        f"({100 * result['winner_changed_fraction']:.1f}%)",
    )
    print(
        "counterfactual unresolved:",
        counterfactual_unresolved_count,
    )

    print()
    print("WHICH HUMAN 0 DISPLACED")
    print("-----------------------")
    for anchor, count in result["displaced_anchor_counts"].items():
        print(f"anchor {anchor:>5}: {count}")

    print()
    print("HOW DIFFERENT WERE THE NEW WINNERS?")
    print("-----------------------------------")
    delta = result["winner_change_field_delta_px"]
    print(
        "field-H disagreement current(0) vs previous winner:",
        f"median={_fmt(delta['median'])}",
        f"p75={_fmt(delta['p75'])}",
        f"p90={_fmt(delta['p90'])}",
        f"p95={_fmt(delta['p95'])}",
        f"max={_fmt(delta['max'])}",
        f"n={delta['count']}",
    )

    print()
    print("HUMAN 0 DIRECT VISUAL METRICS")
    print("-----------------------------")
    for key, distribution in result["source0_direct_metrics"].items():
        print(
            f"{key:<22}",
            f"median={_fmt(distribution['median'])}",
            f"p90={_fmt(distribution['p90'])}",
            f"max={_fmt(distribution['max'])}",
            f"n={distribution['count']}",
        )

    print()
    print("IMPACT BY CAMERA REGION")
    print("-----------------------")
    print(
        f"{'REG':>4} "
        f"{'REP':>6} "
        f"{'N':>4} "
        f"{'VERIFY':>8} "
        f"{'BEST0':>8} "
        f"{'CHANGED':>8} "
        f"{'ΔH MED':>8}"
    )
    print("-" * 58)

    for region in regions:
        print(
            f"{int(region['region']):>4} "
            f"{str(region.get('representative_frame')):>6} "
            f"{int(region['target_count']):>4} "
            f"{100 * float(region['source0_verified_fraction']):>7.1f}% "
            f"{100 * float(region['source0_best_fraction']):>7.1f}% "
            f"{100 * float(region['winner_changed_fraction']):>7.1f}% "
            f"{_fmt(region['field_delta_distribution_px']['median']):>8}"
        )

    print()
    print("IMPACT OVER TIME · 30s bins")
    print("---------------------------")
    print(
        f"{'TIME':>11} "
        f"{'N':>4} "
        f"{'VERIF':>6} "
        f"{'BEST0':>6} "
        f"{'CHANGE':>7}"
    )
    print("-" * 42)

    for item in temporal_bins:
        print(
            f"{int(item['start_s']):>4}-{int(item['end_s']):<4}s "
            f"{int(item['target_count']):>4} "
            f"{int(item['source0_verified_count']):>6} "
            f"{int(item['source0_best_count']):>6} "
            f"{int(item['winner_changed_count']):>7}"
        )

    print()
    print("LARGEST KEYFRAME WINNER CHANGES")
    print("-------------------------------")
    print(
        f"{'FRAME':>6} "
        f"{'REG':>4} "
        f"{'OLD':>6} "
        f"{'NEW':>6} "
        f"{'ΔH MED':>8} "
        f"{'ΔH P90':>8}"
    )
    print("-" * 48)

    for row in changed_with_delta[:30]:
        comparison = row["winner_field_delta"]
        print(
            f"{int(row['frame']):>6} "
            f"{str(row.get('planner_region') or '-'):>4} "
            f"{str(row.get('counterfactual_best_anchor')):>6} "
            f"{str(row.get('actual_best_anchor')):>6} "
            f"{_fmt(comparison.get('median_px')):>8} "
            f"{_fmt(comparison.get('p90_px')):>8}"
        )

    OUTPUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("tiempo:", f"{elapsed:.3f}s")
    print("Guardado:")
    print(OUTPUT)
    print()
    print(
        "READ-ONLY. Este diagnóstico no selecciona todavía el próximo HUMAN."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

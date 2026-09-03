"""Córdoba causal comparison for Sparse Active Anchor Selector V3.

Runs the SAME V3 criterion on:
A) counterfactual state BEFORE HUMAN 0;
B) actual state AFTER HUMAN 0.

No disk artifact is modified except the two diagnostic JSON outputs.

Usage:
    python -m supervisor.check_camera_sparse_active_anchor_v3
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

import cv2

from supervisor.field_workflow import extract_human_anchors
from supervisor.field_camera_sparse_validator_v1 import (
    SparseCameraValidatorConfig,
    analyse_sparse_camera_consensus,
)
from supervisor.field_camera_sparse_active_anchor_v3 import (
    SparseActiveAnchorV3Config,
    analyse_sparse_active_anchor_v3,
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
V1 = REVIEWS / f"{ASSET}_camera_anchor_planner_v1.json"
V2 = REVIEWS / f"{ASSET}_camera_anchor_planner_v2.json"

OUT_BEFORE = (
    REVIEWS
    / f"{ASSET}_camera_sparse_active_v3_before_0.json"
)
OUT_AFTER = (
    REVIEWS
    / f"{ASSET}_camera_sparse_active_v3_after_0.json"
)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _filter_anchor(relocalization, excluded):
    value = deepcopy(relocalization)

    for result in value.get("results", []):
        kept = []

        for candidate in result.get("candidates", []):
            try:
                anchor = int(candidate["anchor_frame"])
            except (KeyError, TypeError, ValueError):
                continue

            if anchor == int(excluded):
                continue

            kept.append(candidate)

        result["candidates"] = kept
        result["candidate_count"] = len(kept)

        valid = [
            item
            for item in kept
            if isinstance(item.get("homography"), list)
            and item.get("score") is not None
        ]

        valid.sort(
            key=lambda item: (
                -float(item.get("score") or float("-inf")),
                -int(
                    (item.get("verification") or {}).get(
                        "inlier_count"
                    )
                    or 0
                ),
                float(
                    (item.get("verification") or {}).get(
                        "median_reprojection_px"
                    )
                    or float("inf")
                ),
            )
        )

        if not valid:
            for key in (
                "homography",
                "source",
                "anchor_frame",
                "confidence_tier",
                "score",
                "direct_state_distance",
                "match_count",
                "inlier_count",
                "inlier_ratio",
                "median_reprojection_px",
            ):
                result.pop(key, None)
            result["status"] = "unrelocalized"
            result["verified_candidate_count"] = 0
            continue

        best = valid[0]
        verification = best.get("verification") or {}

        result["status"] = "ok"
        result["verified_candidate_count"] = len(valid)
        result["homography"] = best["homography"]
        result["anchor_frame"] = int(best["anchor_frame"])
        result["score"] = float(best["score"])
        result["source"] = "counterfactual_candidate_replay"
        result["direct_state_distance"] = verification.get(
            "direct_state_distance"
        )
        result["match_count"] = verification.get("match_count")
        result["inlier_count"] = verification.get("inlier_count")
        result["inlier_ratio"] = verification.get("inlier_ratio")
        result["median_reprojection_px"] = verification.get(
            "median_reprojection_px"
        )

    return value


def _rank_map(result):
    return {
        int(item["frame"]): int(item["rank"])
        for item in result.get("recommendations", [])
    }


def _best_in_window(result, start, end):
    rows = [
        item
        for item in result.get("recommendations", [])
        if start <= int(item["frame"]) <= end
    ]

    if not rows:
        return None

    return min(
        rows,
        key=lambda item: int(item["rank"]),
    )


def _print_top(title, result, n=25):
    print()
    print(title)
    print("-" * len(title))
    print(
        f"{'RANK':>4} "
        f"{'FRAME':>6} "
        f"{'SCORE':>8} "
        f"{'GAIN-R':>6} "
        f"{'NOV-R':>6} "
        f"{'GAIN':>8} "
        f"{'N':>4} "
        f"{'NEAR H':>7}"
    )
    print("-" * 75)

    for item in result.get("recommendations", [])[:n]:
        nearest = item.get("nearest_human") or {}

        print(
            f"{int(item['rank']):>4} "
            f"{int(item['frame']):>6} "
            f"{float(item['score']):>8.3f} "
            f"{float(item['gain_rank']):>6.3f} "
            f"{float(item['candidate_novelty_rank']):>6.3f} "
            f"{float(item['raw_information_gain']):>8.3f} "
            f"{int(item['covered_conflicted_state_count']):>4} "
            f"{str(nearest.get('anchor_frame', '-')):>7}"
        )


def _print_pressure_targets(title, result, n=20):
    print()
    print(title)
    print("-" * len(title))
    print(
        f"{'FRAME':>6} "
        f"{'PRESS':>7} "
        f"{'SUP':>4} "
        f"{'PAIRS':>5} "
        f"{'E-RANK':>6} "
        f"{'M-RANK':>6} "
        f"{'N-RANK':>6} "
        f"SUPPORT"
    )
    print("-" * 82)

    for item in result.get("top_target_pressures", [])[:n]:
        support = "/".join(
            str(value)
            for value in item.get(
                "support_anchor_frames",
                [],
            )
        )

        print(
            f"{int(item['frame']):>6} "
            f"{float(item['information_pressure']):>7.3f} "
            f"{int(item['support_count']):>4} "
            f"{int(item['comparable_pair_count']):>5} "
            f"{float(item['evidence_rank']):>6.3f} "
            f"{float(item['magnitude_rank']):>6.3f} "
            f"{float(item['target_novelty_rank']):>6.3f} "
            f"[{support}]"
        )


def main():
    print("=" * 104)
    print("SPARSE ACTIVE ANCHOR SELECTOR V3 · CAUSAL BEFORE/AFTER HUMAN 0")
    print("=" * 104)

    for label, path in (
        ("video", VIDEO),
        ("review", REVIEW),
        ("relocalization", RELOC),
        ("planner V1", V1),
        ("planner V2", V2),
    ):
        if not path.exists():
            print(f"ERROR: falta {label}: {path}")
            return 2

    review = _load(REVIEW)
    relocalization_after = _load(RELOC)
    planner_v1 = _load(V1)
    planner_v2 = _load(V2)

    actual_humans = sorted(
        int(item["frame"])
        for item in extract_human_anchors(review)
    )

    if 0 not in actual_humans:
        print("ERROR: HUMAN 0 no existe actualmente.")
        return 2

    humans_before = [
        frame
        for frame in actual_humans
        if frame != 0
    ]

    relocalization_before = _filter_anchor(
        relocalization_after,
        0,
    )

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
    capture.release()

    sparse_cfg = SparseCameraValidatorConfig(
        grid_x=21,
        grid_y=15,
        image_margin_px=8.0,
        minimum_common_grid_points=8,
    )

    selector_cfg = SparseActiveAnchorV3Config(
        quality_floor=0.16,
        minimum_support_for_conflict=2,
        percentile_floor=0.04,
        quality_soft_weight=0.12,
        gain_exponent=0.62,
        novelty_exponent=0.38,
    )

    started = time.perf_counter()

    sparse_before = analyse_sparse_camera_consensus(
        relocalization_before,
        image_width=image_width,
        image_height=image_height,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        human_frames=humans_before,
        planner_v2=planner_v2,
        config=sparse_cfg,
    )

    sparse_after = analyse_sparse_camera_consensus(
        relocalization_after,
        image_width=image_width,
        image_height=image_height,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        human_frames=actual_humans,
        planner_v2=planner_v2,
        config=sparse_cfg,
    )

    if (
        sparse_before.get("status") != "ok"
        or sparse_after.get("status") != "ok"
    ):
        print(
            "ERROR sparse:",
            sparse_before.get("status"),
            sparse_after.get("status"),
        )
        return 1

    before = analyse_sparse_active_anchor_v3(
        planner_v1,
        sparse_before,
        relocalization_before,
        human_frames=humans_before,
        config=selector_cfg,
    )

    after = analyse_sparse_active_anchor_v3(
        planner_v1,
        sparse_after,
        relocalization_after,
        human_frames=actual_humans,
        config=selector_cfg,
    )

    elapsed = time.perf_counter() - started

    print()
    print("HUMAN BEFORE:", humans_before)
    print("HUMAN AFTER :", actual_humans)
    print("status:", before.get("status"), "/", after.get("status"))
    print("tiempo:", f"{elapsed:.3f}s")

    if (
        before.get("status") != "ok"
        or after.get("status") != "ok"
    ):
        print("BEFORE:", json.dumps(before, indent=2))
        print("AFTER:", json.dumps(after, indent=2))
        return 1

    _print_top(
        "A · BEFORE HUMAN 0 · TOP RECOMMENDATIONS",
        before,
    )

    _print_top(
        "B · AFTER HUMAN 0 · TOP RECOMMENDATIONS",
        after,
    )

    _print_pressure_targets(
        "TARGET INFORMATION PRESSURE · BEFORE HUMAN 0",
        before,
    )

    _print_pressure_targets(
        "TARGET INFORMATION PRESSURE · AFTER HUMAN 0",
        after,
    )

    print()
    print("=" * 104)
    print("CAUSAL RANK MOVEMENT")
    print("=" * 104)

    windows = [
        ("early 0–100", 0, 100),
        ("580-zone 500–1800", 500, 1800),
        ("3400-zone 3300–4200", 3300, 4200),
        ("5880-zone 5700–6100", 5700, 6100),
        ("6800-zone 6600–7000", 6600, 7000),
        ("late 7600–8200", 7600, 8200),
    ]

    print(
        f"{'WINDOW':<23} "
        f"{'BEFORE':>14} "
        f"{'AFTER':>14} "
        f"{'ΔRANK':>8}"
    )
    print("-" * 64)

    for label, start, end in windows:
        b = _best_in_window(before, start, end)
        a = _best_in_window(after, start, end)

        if b is None or a is None:
            print(
                f"{label:<23} "
                f"{'-':>14} "
                f"{'-':>14} "
                f"{'-':>8}"
            )
            continue

        delta = int(a["rank"]) - int(b["rank"])

        print(
            f"{label:<23} "
            f"{int(b['frame']):>6}#{int(b['rank']):<6} "
            f"{int(a['frame']):>6}#{int(a['rank']):<6} "
            f"{delta:>+8}"
        )

    before_rank = _rank_map(before)
    after_rank = _rank_map(after)

    print()
    print("FRAME PROBES")
    print("------------")

    for frame in (
        0, 24, 32, 40, 48, 56, 64,
        902, 922, 986, 1368, 1428,
        4094, 4102,
        5842, 5858, 5880, 5896,
        6808, 6824, 7798, 7814,
    ):
        b = before_rank.get(frame)
        a = after_rank.get(frame)

        if b is None and a is None:
            continue

        movement = (
            None
            if b is None or a is None
            else a - b
        )

        print(
            f"frame {frame:>5}"
            f" · before #{str(b) if b is not None else '-'}"
            f" · after #{str(a) if a is not None else '-'}"
            f" · Δ={movement if movement is not None else '-'}"
        )

    OUT_BEFORE.write_text(
        json.dumps(before, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    OUT_AFTER.write_text(
        json.dumps(after, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("Guardado:")
    print(OUT_BEFORE)
    print(OUT_AFTER)
    print()
    print(
        "READ-ONLY: no calibres nada todavía. "
        "Primero validamos si V3 explica el efecto causal de HUMAN 0."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Counterfactual replay for Córdoba active HUMAN selection.

Question
--------
Using only artifacts that already exist now, what would Sparse Validator +
Active Anchor Selector V2 have recommended if HUMAN frame 0 had NOT existed?

This is a diagnostic time-travel test:
- the current review is NOT modified;
- no SIFT is rerun;
- no CAMERA/Dense is rerun;
- candidate predictions from anchor 0 are filtered out in memory;
- HUMAN 0 is removed from the HUMAN set in memory;
- Planner/selector then run exactly on that counterfactual state.

If the selector recommends the 0/early-frame region, that retrospectively
matches the manually discovered correction that later improved CAMERA.
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
from supervisor.field_camera_sparse_active_anchor_v2 import (
    SparseActiveAnchorV2Config,
    analyse_sparse_active_anchor_v2,
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
PLANNER_V1 = REVIEWS / f"{ASSET}_camera_anchor_planner_v1.json"
PLANNER_V2 = REVIEWS / f"{ASSET}_camera_anchor_planner_v2.json"

OUTPUT_SPARSE = (
    REVIEWS
    / f"{ASSET}_camera_sparse_replay_without_0.json"
)
OUTPUT_SELECTOR = (
    REVIEWS
    / f"{ASSET}_camera_sparse_active_replay_without_0.json"
)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _remove_anchor_from_relocalization(
    relocalization,
    excluded_anchor_frame,
):
    """Filter one HUMAN source from candidates without changing disk artifacts."""
    value = deepcopy(relocalization)

    for result in value.get("results", []):
        candidates = []

        for candidate in result.get("candidates", []):
            try:
                anchor = int(candidate["anchor_frame"])
            except (KeyError, TypeError, ValueError):
                continue

            if anchor == int(excluded_anchor_frame):
                continue

            candidates.append(candidate)

        result["candidates"] = candidates
        result["candidate_count"] = len(candidates)

        valid = [
            item
            for item in candidates
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

        result["verified_candidate_count"] = len(valid)

        if not valid:
            # Remove stale top-level winner from the original state.
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
            continue

        best = valid[0]
        verification = best.get("verification") or {}

        result.update(
            {
                "status": "ok",
                "homography": best["homography"],
                "source": "counterfactual_candidate_replay",
                "anchor_frame": int(best["anchor_frame"]),
                "confidence_tier": (
                    "high"
                    if bool(best.get("high_confidence"))
                    else "verified"
                ),
                "score": float(best["score"]),
                "direct_state_distance": verification.get(
                    "direct_state_distance"
                ),
                "match_count": verification.get("match_count"),
                "inlier_count": verification.get("inlier_count"),
                "inlier_ratio": verification.get("inlier_ratio"),
                "median_reprojection_px": verification.get(
                    "median_reprojection_px"
                ),
            }
        )

    return value


def _print_top(title, rows, score_key):
    print()
    print(title)
    print("-" * len(title))
    print(
        f"{'RANK':>4} "
        f"{'FRAME':>6} "
        f"{'SCORE':>8} "
        f"{'C-RANK':>7} "
        f"{'N-RANK':>7} "
        f"{'NEAR H':>7} "
        f"{'D-MED':>8} "
        f"{'D-P90':>8}"
    )
    print("-" * 76)

    for index, item in enumerate(rows[:25], start=1):
        near = item.get("nearest_human") or {}
        dmed = near.get("median_px")
        dp90 = near.get("p90_px")

        print(
            f"{index:>4} "
            f"{int(item['frame']):>6} "
            f"{float(item[score_key]):>8.3f} "
            f"{float(item['conflict_rank']):>7.3f} "
            f"{float(item['novelty_rank']):>7.3f} "
            f"{str(near.get('anchor_frame', '-')):>7} "
            f"{('-' if dmed is None else f'{float(dmed):.1f}'):>8} "
            f"{('-' if dp90 is None else f'{float(dp90):.1f}'):>8}"
        )


def main():
    print("=" * 100)
    print("COUNTERFACTUAL REPLAY · BEFORE HUMAN 0 · CÓRDOBA")
    print("=" * 100)

    for label, path in (
        ("video", VIDEO),
        ("review", REVIEW),
        ("relocalization", RELOC),
        ("planner V1", PLANNER_V1),
        ("planner V2", PLANNER_V2),
    ):
        if not path.exists():
            print(f"ERROR: falta {label}: {path}")
            return 2

    review = _load(REVIEW)
    anchors = extract_human_anchors(review)

    actual_humans = sorted(
        int(anchor["frame"])
        for anchor in anchors
    )

    counterfactual_humans = [
        frame
        for frame in actual_humans
        if frame != 0
    ]

    if 0 not in actual_humans:
        print(
            "ERROR: HUMAN 0 no existe en la review actual; "
            "este replay ya no representa el experimento esperado."
        )
        return 2

    field = review.get("field", {})
    pitch = field.get("pitch_model", {})

    try:
        pitch_length = float(pitch["length_m"])
        pitch_width = float(pitch["width_m"])
    except (KeyError, TypeError, ValueError):
        print("ERROR: pitch_model no válido.")
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

    relocalization_actual = _load(RELOC)
    relocalization_before_0 = (
        _remove_anchor_from_relocalization(
            relocalization_actual,
            0,
        )
    )

    planner_v1 = _load(PLANNER_V1)
    planner_v2 = _load(PLANNER_V2)

    print()
    print("HUMAN actuales:", actual_humans)
    print("HUMAN counterfactual:", counterfactual_humans)
    print(
        "Anchor 0 y todas sus predicciones se eliminan SOLO en memoria."
    )

    sparse_cfg = SparseCameraValidatorConfig(
        grid_x=21,
        grid_y=15,
        image_margin_px=8.0,
        minimum_common_grid_points=8,
    )

    started = time.perf_counter()

    sparse = analyse_sparse_camera_consensus(
        relocalization_before_0,
        image_width=image_width,
        image_height=image_height,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        human_frames=counterfactual_humans,
        planner_v2=planner_v2,
        config=sparse_cfg,
    )

    if sparse.get("status") != "ok":
        print("Sparse replay failed:")
        print(json.dumps(sparse, indent=2))
        return 1

    selector_cfg = SparseActiveAnchorV2Config(
        quality_floor=0.16,
        minimum_support_for_conflict=2,
        rank_floor=0.05,
        quality_soft_weight=0.15,
    )

    selector = analyse_sparse_active_anchor_v2(
        planner_v1,
        sparse,
        relocalization_before_0,
        human_frames=counterfactual_humans,
        config=selector_cfg,
    )

    elapsed = time.perf_counter() - started

    print()
    print("status:", selector.get("status"))
    print("tiempo:", f"{elapsed:.3f}s")

    if selector.get("status") != "ok":
        print(json.dumps(selector, indent=2))
        return 1

    print()
    print("SPARSE BEFORE HUMAN 0")
    print("---------------------")
    print(
        "support count distribution:",
        sparse["support_count_distribution"],
    )

    pair = sparse["pair_p90_distribution_px"]
    print(
        "pair-p90 disagreement:",
        f"median={pair.get('median'):.1f}",
        f"p75={pair.get('p75'):.1f}",
        f"p90={pair.get('p90'):.1f}",
        f"max={pair.get('max'):.1f}",
    )

    _print_top(
        "BALANCED RECOMMENDATION · BEFORE HUMAN 0",
        selector["balanced"],
        "balanced_score",
    )

    _print_top(
        "CONFLICT-ONLY · BEFORE HUMAN 0",
        selector["conflict_only"],
        "conflict_gain",
    )

    early = [
        item
        for item in selector["balanced"]
        if int(item["frame"]) <= 100
    ]

    print()
    print("EARLY-FRAME PROBES · <=100")
    print("--------------------------")

    for item in early:
        near = item.get("nearest_human") or {}
        print(
            f"frame {int(item['frame']):>4}"
            f" · balanced rank #{int(item['balanced_rank'])}"
            f" · score={float(item['balanced_score']):.3f}"
            f" · conflict rank={float(item['conflict_rank']):.3f}"
            f" · novelty rank={float(item['novelty_rank']):.3f}"
            f" · nearest HUMAN={near.get('anchor_frame', '-')}"
        )

    OUTPUT_SPARSE.write_text(
        json.dumps(sparse, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    OUTPUT_SELECTOR.write_text(
        json.dumps(selector, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("Guardado:")
    print(OUTPUT_SPARSE)
    print(OUTPUT_SELECTOR)
    print()
    print(
        "READ-ONLY COUNTERFACTUAL: no se modificó review, HUMAN, "
        "Relocalization, CAMERA ni Dense."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

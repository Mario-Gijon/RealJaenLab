"""Córdoba checker for Sparse Active Anchor Selector V2.

This is a ranking-sensitivity diagnostic. It specifically tests whether the V1
recommendation around frame 32 remains valuable after accounting for the HUMAN
already added at frame 0.

Usage:
    python -m supervisor.check_camera_sparse_active_anchor_v2
"""

from __future__ import annotations

import json
from pathlib import Path
import time

from supervisor.field_workflow import extract_human_anchors
from supervisor.field_camera_sparse_active_anchor_v2 import (
    SparseActiveAnchorV2Config,
    analyse_sparse_active_anchor_v2,
)


ROOT = Path(__file__).resolve().parents[1]
ASSET = "asset_7fbc47b7b83647bc8fd52e6a676c9425"
REVIEWS = ROOT / "supervisor" / "data" / "reviews"

REVIEW = REVIEWS / f"{ASSET}.json"
V1 = REVIEWS / f"{ASSET}_camera_anchor_planner_v1.json"
SPARSE = REVIEWS / f"{ASSET}_camera_sparse_validator_v1.json"
RELOC = REVIEWS / f"{ASSET}_camera_anchor_relocalization_v1.json"

OUTPUT = (
    REVIEWS
    / f"{ASSET}_camera_sparse_active_anchor_v2.json"
)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value, digits=1):
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _row(item, rank_key, score_key):
    nearest = item.get("nearest_human") or {}

    return (
        f"{int(item[rank_key]):>4} "
        f"{int(item['frame']):>6} "
        f"{float(item[score_key]):>8.3f} "
        f"{float(item['conflict_gain']):>8.3f} "
        f"{float(item['conflict_rank']):>6.3f} "
        f"{float(item['novelty_rank']):>6.3f} "
        f"{str(nearest.get('anchor_frame', '-')):>7} "
        f"{_fmt(nearest.get('median_px')):>8} "
        f"{_fmt(nearest.get('p90_px')):>8} "
        f"{float(item['quality_score']):>7.3f}"
    )


def _print_ranking(title, rows, rank_key, score_key):
    print()
    print(title)
    print("-" * len(title))
    print(
        f"{'RANK':>4} "
        f"{'FRAME':>6} "
        f"{'SCORE':>8} "
        f"{'GAIN':>8} "
        f"{'C-RANK':>6} "
        f"{'N-RANK':>6} "
        f"{'NEAR H':>7} "
        f"{'D-MED':>8} "
        f"{'D-P90':>8} "
        f"{'QUALITY':>7}"
    )
    print("-" * 92)

    for item in rows[:25]:
        print(_row(item, rank_key, score_key))


def main():
    print("=" * 100)
    print("SPARSE ACTIVE ANCHOR SELECTOR V2 · MARGINAL HUMAN VALUE · CÓRDOBA")
    print("=" * 100)

    for label, path in (
        ("review", REVIEW),
        ("planner V1", V1),
        ("sparse validator", SPARSE),
        ("relocalization", RELOC),
    ):
        if not path.exists():
            print(f"ERROR: falta {label}: {path}")
            return 2

    review = _load(REVIEW)
    humans = [
        int(item["frame"])
        for item in extract_human_anchors(review)
    ]

    print()
    print("HUMAN actuales:", humans)
    print(
        "La distancia a HUMAN se usa SOLO para redundancia/novelty, "
        "no para decidir si una transferencia es correcta."
    )

    cfg = SparseActiveAnchorV2Config(
        quality_floor=0.16,
        minimum_support_for_conflict=2,
        rank_floor=0.05,
        quality_soft_weight=0.15,
    )

    started = time.perf_counter()

    result = analyse_sparse_active_anchor_v2(
        V1,
        SPARSE,
        RELOC,
        human_frames=humans,
        config=cfg,
    )

    elapsed = time.perf_counter() - started

    print()
    print("status:", result.get("status"))
    print("tiempo:", f"{elapsed:.3f}s")

    if result.get("status") != "ok":
        print(json.dumps(result, indent=2))
        return 1

    print(
        "candidates:",
        result["candidate_count"],
        "· novelty measured:",
        result["measured_novelty_candidate_count"],
    )

    dist = result["nearest_human_distance_distribution_px"]
    print(
        "nearest-HUMAN combined-distance distribution:",
        f"median={_fmt(dist.get('median'))}",
        f"p75={_fmt(dist.get('p75'))}",
        f"p90={_fmt(dist.get('p90'))}",
        f"max={_fmt(dist.get('max'))}",
    )

    _print_ranking(
        "A · CONFLICT ONLY · comportamiento V1",
        result["conflict_only"],
        "conflict_only_rank",
        "conflict_gain",
    )

    _print_ranking(
        "B · BALANCED · conflicto × novedad frente a HUMAN",
        result["balanced"],
        "balanced_rank",
        "balanced_score",
    )

    _print_ranking(
        "C · NOVELTY-FIRST · sensitivity diagnostic",
        result["novelty_first"],
        "novelty_first_rank",
        "novelty_first_score",
    )

    print()
    print("PARETO FRONTIER · conflict / novelty / quality")
    print("----------------------------------------------")

    for item in result["pareto_frontier"][:25]:
        nearest = item.get("nearest_human") or {}
        print(
            f"frame {int(item['frame']):>5}"
            f" · conflict_rank={float(item['conflict_rank']):.3f}"
            f" · novelty_rank={float(item['novelty_rank']):.3f}"
            f" · quality={float(item['quality_score']):.3f}"
            f" · near HUMAN={nearest.get('anchor_frame', '-')}"
            f" · dmed={_fmt(nearest.get('median_px'))}"
            f" · dp90={_fmt(nearest.get('p90_px'))}"
        )

    # Explicitly surface the frames that matter to the current diagnosis.
    lookup = {
        int(item["frame"]): item
        for item in result["conflict_only"]
    }

    print()
    print("DIAGNOSTIC PROBES")
    print("-----------------")

    for frame in (24, 32, 40, 48, 64, 1428, 4102, 5880, 6808, 6824):
        item = lookup.get(frame)
        if item is None:
            continue
        nearest = item.get("nearest_human") or {}

        print(
            f"frame {frame:>5}"
            f" · conflict rank #{item['conflict_only_rank']}"
            f" · balanced rank #{item['balanced_rank']}"
            f" · novelty rank={item['novelty_rank']:.3f}"
            f" · nearest HUMAN={nearest.get('anchor_frame', '-')}"
            f" · dmed={_fmt(nearest.get('median_px'))}"
            f" · dp90={_fmt(nearest.get('p90_px'))}"
        )

    OUTPUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("Guardado:")
    print(OUTPUT)
    print()
    print(
        "NO calibres ningún frame todavía: esta V2 compara criterios "
        "de selección antes de elegir el siguiente HUMAN."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

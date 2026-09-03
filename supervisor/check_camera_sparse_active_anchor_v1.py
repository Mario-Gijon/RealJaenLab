"""Córdoba checker for Sparse Active Anchor Selector V1.

Reads existing diagnostic artifacts only. Does not run SIFT, CAMERA or Dense.

Usage:
    python -m supervisor.check_camera_sparse_active_anchor_v1
"""

from __future__ import annotations

import json
from pathlib import Path
import time

from supervisor.field_workflow import (
    extract_human_anchors,
)
from supervisor.field_camera_sparse_active_anchor_v1 import (
    SparseActiveAnchorConfig,
    analyse_sparse_active_anchor,
)


ROOT = Path(__file__).resolve().parents[1]
ASSET = "asset_7fbc47b7b83647bc8fd52e6a676c9425"

REVIEWS = (
    ROOT
    / "supervisor"
    / "data"
    / "reviews"
)

REVIEW = REVIEWS / f"{ASSET}.json"

V1 = (
    REVIEWS
    / f"{ASSET}_camera_anchor_planner_v1.json"
)

SPARSE = (
    REVIEWS
    / f"{ASSET}_camera_sparse_validator_v1.json"
)

OUTPUT = (
    REVIEWS
    / f"{ASSET}_camera_sparse_active_anchor_v1.json"
)


def _load(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def main():
    print("=" * 98)
    print("SPARSE ACTIVE ANCHOR SELECTOR V1 · CÓRDOBA")
    print("=" * 98)
    print()

    for label, path in (
        ("review", REVIEW),
        ("planner V1", V1),
        ("sparse validator", SPARSE),
    ):
        print(f"{label}: {path}")
        if not path.exists():
            print(f"ERROR: falta {label}.")
            return 2

    review = _load(REVIEW)
    anchors = extract_human_anchors(review)
    human_frames = [
        int(anchor["frame"])
        for anchor in anchors
    ]

    print()
    print("HUMAN actuales:", human_frames)
    print()
    print(
        "POLÍTICA: soporte único NO se considera error por sí solo. "
        "Priorizamos conflicto medido entre ≥2 HUMAN."
    )

    cfg = SparseActiveAnchorConfig(
        quality_floor=0.16,
        minimum_support_for_conflict=2,
        conflict_gain_weight=1.0,
        quality_weight=0.12,
        greedy_preview_count=5,
    )

    started = time.perf_counter()

    result = analyse_sparse_active_anchor(
        V1,
        SPARSE,
        human_frames=human_frames,
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
        "measured conflicted states:",
        result["measured_conflict_state_count"],
    )

    distribution = result[
        "raw_pair_p90_distribution_px"
    ]

    print(
        "pair-p90 conflict distribution:",
        f"median={distribution['median']:.1f}",
        f"p75={distribution['p75']:.1f}",
        f"p90={distribution['p90']:.1f}",
        f"p95={distribution['p95']:.1f}",
        f"max={distribution['max']:.1f}",
    )

    print()
    print("TOP NEXT-HUMAN CANDIDATES")
    print("-------------------------")
    print(
        f"{'RANK':>4} "
        f"{'FRAME':>6} "
        f"{'SCORE':>8} "
        f"{'GAIN':>8} "
        f"{'N':>4} "
        f"{'QUALITY':>7} "
        f"DOMINANT SUPPORT PATTERNS"
    )
    print("-" * 108)

    for item in result["recommendations"][:30]:
        patterns = []

        for pattern in item.get(
            "covered_support_patterns",
            [],
        )[:3]:
            anchors_text = "/".join(
                str(value)
                for value in pattern[
                    "support_anchor_frames"
                ]
            )
            patterns.append(
                f"[{anchors_text}]×{pattern['count']}"
            )

        print(
            f"{int(item['rank']):>4} "
            f"{int(item['frame']):>6} "
            f"{float(item['score']):>8.3f} "
            f"{float(item['conflict_gain']):>8.3f} "
            f"{int(item['covered_conflict_count']):>4} "
            f"{float(item['quality_score']):>7.3f} "
            + " ".join(patterns)
        )

    print()
    print("GREEDY BATCH PREVIEW · DIAGNOSTIC ONLY")
    print("--------------------------------------")
    print(
        "NO calibres este lote completo. "
        "Después de un HUMAN real hay que recomputar."
    )

    for index, item in enumerate(
        result["greedy_preview_only"],
        start=1,
    ):
        print(
            f"#{index} frame {int(item['frame'])}"
            f" · conflict gain={float(item['conflict_gain']):.3f}"
            f" · directly touches "
            f"{int(item['covered_conflict_count'])} conflicted states"
            f" · quality={float(item['quality_score']):.3f}"
        )

    if result["recommendations"]:
        first = result["recommendations"][0]

        print()
        print("=" * 98)
        print("NEXT EXPERIMENTAL HUMAN")
        print("=" * 98)
        print("frame:", first["frame"])
        print(
            "directly touches conflicted states:",
            first["covered_conflict_count"],
        )
        print(
            "conflict gain:",
            f"{first['conflict_gain']:.3f}",
        )
        print(
            "quality:",
            f"{first['quality_score']:.3f}",
        )
        print()
        print(
            "IMPORTANTE: es una recomendación experimental. "
            "Calibrar UNO y volver a ejecutar Sparse Validator; "
            "no el batch completo."
        )

    OUTPUT.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Guardado:")
    print(OUTPUT)
    print()
    print(
        "READ-ONLY: no modifica review, HUMAN, CAMERA, "
        "Relocalization ni Dense."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

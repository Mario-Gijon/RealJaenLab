"""Córdoba checker for Candidate Global Hub Probe V1.

This deliberately compares:
- early-frame family associated with the later successful HUMAN 0 correction;
- candidates that Sparse Active V2/V3 ranked highly before HUMAN 0.

HUMAN 0 is included as a diagnostic oracle using IMAGE CONTENT ONLY.
Its homography is never provided to the hub probe.

Usage:
    python -m supervisor.check_camera_candidate_hub_probe_v1
"""

from __future__ import annotations

import json
from pathlib import Path
import time

from supervisor.field_camera_candidate_hub_probe_v1 import (
    CandidateHubProbeConfig,
    analyse_candidate_hub_probe,
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

PLANNER_V1 = (
    REVIEWS
    / f"{ASSET}_camera_anchor_planner_v1.json"
)

SPARSE_BEFORE = (
    REVIEWS
    / f"{ASSET}_camera_sparse_replay_without_0.json"
)

PLANNER_V2 = (
    REVIEWS
    / f"{ASSET}_camera_anchor_planner_v2.json"
)

OUTPUT = (
    REVIEWS
    / f"{ASSET}_camera_candidate_hub_probe_v1.json"
)


# Shortlist only: O(MN), not all-pairs.
CANDIDATES = [
    # Successful early family / causal oracle.
    0,
    32,
    40,
    48,

    # V3 / V2 high-ranked alternatives before HUMAN 0.
    5880,
    922,
    902,
    986,
    1368,
    3642,
    4102,
    7814,
]


def _fmt(value, digits=1):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _print_ranking(title, rows):
    print()
    print(title)
    print("-" * len(title))
    print(
        f"{'RANK':>4} "
        f"{'FRAME':>6} "
        f"{'GLOBAL':>7} "
        f"{'G%':>6} "
        f"{'CONFL':>6} "
        f"{'C%':>6} "
        f"{'MASS':>8} "
        f"{'M%':>6} "
        f"{'REG':>4}"
    )
    print("-" * 72)

    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank:>4} "
            f"{int(row['frame']):>6} "
            f"{int(row['verified_count']):>7} "
            f"{100 * float(row['verified_fraction']):>5.1f}% "
            f"{int(row['conflicted_verified_count']):>6} "
            f"{100 * float(row['conflicted_verified_fraction']):>5.1f}% "
            f"{float(row['conflict_mass_reached']):>8.2f} "
            f"{100 * float(row['conflict_mass_fraction']):>5.1f}% "
            f"{int(row['camera_region_count']):>4}"
        )


def main():
    print("=" * 104)
    print("CANDIDATE GLOBAL HUB PROBE V1 · CÓRDOBA · BEFORE HUMAN 0")
    print("=" * 104)
    print()
    print("Candidates:", CANDIDATES)
    print(
        "Frame 0 is an oracle ONLY in the sense that we inspect its pixels; "
        "its HUMAN homography is never used."
    )
    print()

    for label, path in (
        ("video", VIDEO),
        ("planner V1", PLANNER_V1),
        ("sparse-before-0", SPARSE_BEFORE),
        ("planner V2", PLANNER_V2),
    ):
        if not path.exists():
            print(f"ERROR: falta {label}: {path}")
            return 2

    cfg = CandidateHubProbeConfig(
        analysis_width=960,
        minimum_support_for_conflict=2,
        progress_every_edges=250,
    )

    started = time.perf_counter()

    result = analyse_candidate_hub_probe(
        VIDEO,
        PLANNER_V1,
        SPARSE_BEFORE,
        PLANNER_V2,
        candidate_frames=CANDIDATES,
        config=cfg,
    )

    elapsed = time.perf_counter() - started

    print()
    print("status:", result.get("status"))
    print("tiempo total:", f"{elapsed:.1f}s")

    if result.get("status") != "ok":
        print(json.dumps(result, indent=2))
        return 1

    print(
        "targets:",
        result["target_count"],
        "· conflicted:",
        result["conflicted_target_count"],
    )

    _print_ranking(
        "A · GLOBAL DIRECT REACH",
        result["by_global_reach"],
    )

    _print_ranking(
        "B · CONFLICTED-STATE DIRECT REACH",
        result["by_conflict_reach"],
    )

    _print_ranking(
        "C · CONFLICT EVIDENCE MASS REACHED",
        result["by_conflict_mass"],
    )

    print()
    print("PER-CANDIDATE DIAGNOSTICS")
    print("-------------------------")

    for row in sorted(
        (
            item
            for item in result["candidate_results"]
            if item.get("status") == "ok"
        ),
        key=lambda item: int(item["frame"]),
    ):
        dmed = row[
            "direct_state_median_distribution_px"
        ]
        dp90 = row[
            "direct_state_p90_distribution_px"
        ]

        print(
            f"frame {int(row['frame']):>5}"
            f" · global #{int(row['global_reach_rank'])}"
            f" · conflict #{int(row['conflict_reach_rank'])}"
            f" · mass #{int(row['conflict_mass_rank'])}"
            f" · reach={int(row['verified_count'])}"
            f" · conflict={int(row['conflicted_verified_count'])}"
            f" · regions={int(row['camera_region_count'])}"
            f" · state med={_fmt(dmed.get('median'))}"
            f" · state p90={_fmt(dp90.get('median'))}"
        )

        patterns = row.get(
            "dominant_support_patterns",
            [],
        )[:4]

        if patterns:
            text = " ".join(
                (
                    "["
                    + "/".join(
                        str(value)
                        for value in pattern[
                            "support_anchor_frames"
                        ]
                    )
                    + f"]×{pattern['count']}"
                )
                for pattern in patterns
            )
            print("      conflict patterns:", text)

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
        "READ-ONLY. No calibres todavía ningún candidato: "
        "esta prueba decide si global hub reach explica el efecto de HUMAN 0."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

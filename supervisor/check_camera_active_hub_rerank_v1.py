"""Active HUMAN V4 diagnostic: sparse shortlist -> global hub rerank.

This checker tests the CAMERA V4 selection architecture on the CURRENT Córdoba
state (after HUMAN 0):

    Sparse Validator / Active V3
            ↓ cheap
    top-K candidate shortlist
            ↓ expensive only for K candidates
    GLOBAL direct hub probe against 421 keyframes
            ↓
    rank by conflict evidence mass reached

No novelty penalty is applied in the final rerank. Previous experiments showed
that novelty proxies could demote the empirically useful HUMAN-0 family.

This is still diagnostic: do not automatically calibrate the winner until the
result is inspected.
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

PLANNER_V2 = (
    REVIEWS
    / f"{ASSET}_camera_anchor_planner_v2.json"
)

SPARSE_AFTER = (
    REVIEWS
    / f"{ASSET}_camera_sparse_validator_v1.json"
)

ACTIVE_V3_AFTER = (
    REVIEWS
    / f"{ASSET}_camera_sparse_active_v3_after_0.json"
)

OUTPUT = (
    REVIEWS
    / f"{ASSET}_camera_active_hub_rerank_v1.json"
)


SHORTLIST_SIZE = 16


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _build_shortlist(active_v3):
    rows = [
        item
        for item in active_v3.get("recommendations", [])
        if isinstance(item, dict)
    ]

    rows.sort(
        key=lambda item: int(item.get("rank") or 10**9)
    )

    selected = []
    seen = set()

    for item in rows:
        try:
            frame = int(item["frame"])
        except (KeyError, TypeError, ValueError):
            continue

        if frame in seen:
            continue

        selected.append(frame)
        seen.add(frame)

        if len(selected) >= SHORTLIST_SIZE:
            break

    return selected


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
    print("-" * 73)

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
    print("ACTIVE HUMAN V4 DIAGNOSTIC · SPARSE SHORTLIST → GLOBAL HUB RERANK")
    print("=" * 104)

    for label, path in (
        ("video", VIDEO),
        ("planner V1", PLANNER_V1),
        ("planner V2", PLANNER_V2),
        ("sparse after HUMAN 0", SPARSE_AFTER),
        ("active V3 after HUMAN 0", ACTIVE_V3_AFTER),
    ):
        if not path.exists():
            print(f"ERROR: falta {label}: {path}")
            return 2

    active_v3 = _load(ACTIVE_V3_AFTER)
    shortlist = _build_shortlist(active_v3)

    if not shortlist:
        print("ERROR: V3 no produjo shortlist.")
        return 2

    print()
    print("V3 shortlist:", shortlist)
    print(
        f"K={len(shortlist)}. Solo estos candidatos se verifican "
        "globalmente contra los 421 keyframes."
    )
    print()

    cfg = CandidateHubProbeConfig(
        analysis_width=960,
        minimum_support_for_conflict=2,
        progress_every_edges=250,
    )

    started = time.perf_counter()

    result = analyse_candidate_hub_probe(
        VIDEO,
        PLANNER_V1,
        SPARSE_AFTER,
        PLANNER_V2,
        candidate_frames=shortlist,
        config=cfg,
    )

    elapsed = time.perf_counter() - started

    print()
    print("status:", result.get("status"))
    print("tiempo total:", f"{elapsed:.1f}s")

    if result.get("status") != "ok":
        print(json.dumps(result, indent=2))
        return 1

    _print_ranking(
        "A · FINAL V4 SIGNAL · CONFLICT EVIDENCE MASS REACHED",
        result["by_conflict_mass"],
    )

    _print_ranking(
        "B · CONFLICTED-STATE DIRECT REACH",
        result["by_conflict_reach"],
    )

    _print_ranking(
        "C · GLOBAL DIRECT REACH",
        result["by_global_reach"],
    )

    print()
    print("V3 → HUB RERANK MOVEMENT")
    print("------------------------")
    print(
        f"{'FRAME':>6} "
        f"{'V3':>5} "
        f"{'MASS':>6} "
        f"{'CONFL':>6} "
        f"{'GLOBAL':>7} "
        f"{'MASS%':>7}"
    )
    print("-" * 45)

    v3_rank = {
        int(item["frame"]): int(item["rank"])
        for item in active_v3.get("recommendations", [])
        if isinstance(item, dict)
        and item.get("frame") is not None
        and item.get("rank") is not None
    }

    mass_rank = {
        int(item["frame"]): index
        for index, item in enumerate(
            result["by_conflict_mass"],
            start=1,
        )
    }

    conflict_rank = {
        int(item["frame"]): index
        for index, item in enumerate(
            result["by_conflict_reach"],
            start=1,
        )
    }

    global_rank = {
        int(item["frame"]): index
        for index, item in enumerate(
            result["by_global_reach"],
            start=1,
        )
    }

    row_by_frame = {
        int(item["frame"]): item
        for item in result["candidate_results"]
        if item.get("status") == "ok"
    }

    for frame in shortlist:
        row = row_by_frame.get(frame)
        if row is None:
            continue

        print(
            f"{frame:>6} "
            f"{v3_rank.get(frame, '-'):>5} "
            f"{mass_rank.get(frame, '-'):>6} "
            f"{conflict_rank.get(frame, '-'):>6} "
            f"{global_rank.get(frame, '-'):>7} "
            f"{100 * float(row['conflict_mass_fraction']):>6.1f}%"
        )

    if result["by_conflict_mass"]:
        winner = result["by_conflict_mass"][0]

        print()
        print("=" * 104)
        print("CURRENT V4 WINNER · DIAGNOSTIC")
        print("=" * 104)
        print("frame:", int(winner["frame"]))
        print(
            "global reach:",
            f"{int(winner['verified_count'])}/"
            f"{result['target_count']}",
            f"({100 * float(winner['verified_fraction']):.1f}%)",
        )
        print(
            "conflicted reach:",
            f"{int(winner['conflicted_verified_count'])}/"
            f"{result['conflicted_target_count']}",
            f"({100 * float(winner['conflicted_verified_fraction']):.1f}%)",
        )
        print(
            "conflict evidence mass reached:",
            f"{100 * float(winner['conflict_mass_fraction']):.1f}%",
        )
        print(
            "camera regions reached:",
            int(winner["camera_region_count"]),
        )

        print()
        print(
            "No lo calibres todavía hasta interpretar esta tabla: "
            "queremos comprobar que el rerank no vuelve a seleccionar "
            "simplemente otra copia de la familia ya corregida."
        )

    OUTPUT.write_text(
        json.dumps(
            {
                "status": "ok",
                "version": 1,
                "algorithm": "active_human_v4_sparse_to_global_hub",
                "shortlist_size": len(shortlist),
                "shortlist": shortlist,
                "hub_probe": result,
            },
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
        "READ-ONLY. No modifica HUMAN, review, CAMERA, "
        "Relocalization ni Dense."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

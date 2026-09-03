"""Hub Reach-Signature Novelty V1.

Cheap causal diagnostic using already-generated artifacts.

Hypothesis
----------
A useful HUMAN candidate should:
1. directly reach a large amount of current conflict evidence; and
2. have a GLOBAL REACH SIGNATURE that is not already duplicated by one current
   HUMAN anchor.

Reach signature = set of representative trajectory keyframes to which the
candidate directly verifies.

This is more appropriate than raw direct_state_distance for redundancy:
two visually distant-looking frames may still provide almost the same
relocalization domain, while two similarly sized views may reach different
camera-state families.

The test compares exactly the same candidate signatures:
- BEFORE HUMAN 0: existing HUMAN = 580/3450/3805/5101/7200
- AFTER HUMAN 0:  existing HUMAN additionally includes 0

No SIFT/RANSAC is rerun. No file is modified except diagnostic JSON output.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSET = "asset_7fbc47b7b83647bc8fd52e6a676c9425"

REVIEWS = ROOT / "supervisor" / "data" / "reviews"

RELOC = REVIEWS / f"{ASSET}_camera_anchor_relocalization_v1.json"
HUB_PROBE = REVIEWS / f"{ASSET}_camera_candidate_hub_probe_v1.json"

SPARSE_BEFORE = (
    REVIEWS
    / f"{ASSET}_camera_sparse_replay_without_0.json"
)
SPARSE_AFTER = (
    REVIEWS
    / f"{ASSET}_camera_sparse_validator_v1.json"
)

OUTPUT = (
    REVIEWS
    / f"{ASSET}_camera_hub_signature_novelty_v1.json"
)


HUMANS_BEFORE = [580, 3450, 3805, 5101, 7200]
HUMANS_AFTER = [0, 580, 3450, 3805, 5101, 7200]


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _average_percentile(values):
    if not values:
        return {}

    keys = sorted(values)
    arr = np.asarray(
        [float(values[key]) for key in keys],
        dtype=np.float64,
    )
    order = np.argsort(arr, kind="mergesort")
    ranks = np.zeros(len(arr), dtype=np.float64)

    pos = 0
    while pos < len(arr):
        end = pos + 1
        value = arr[order[pos]]

        while (
            end < len(arr)
            and np.isclose(
                arr[order[end]],
                value,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            end += 1

        avg = 0.5 * (pos + end - 1)
        rank = (
            1.0
            if len(arr) == 1
            else avg / (len(arr) - 1)
        )

        for index in order[pos:end]:
            ranks[int(index)] = rank

        pos = end

    return {
        key: float(ranks[index])
        for index, key in enumerate(keys)
    }


def _conflict_mass(sparse):
    pair_rows = []

    for target in sparse.get("targets", []):
        if not isinstance(target, dict):
            continue

        try:
            frame = int(target["frame"])
            support = int(target.get("support_count") or 0)
        except (KeyError, TypeError, ValueError):
            continue

        if target.get("is_exact_human") or support < 2:
            continue

        for pair in target.get("pair_details", []):
            if (
                not isinstance(pair, dict)
                or pair.get("status") != "ok"
            ):
                continue

            value = _finite(pair.get("median_px"))
            if value is not None:
                pair_rows.append((frame, max(0.0, value)))

    if not pair_rows:
        return {}

    values = {
        index: value
        for index, (_, value) in enumerate(pair_rows)
    }
    ranks = _average_percentile(values)

    mass = {}

    for index, (frame, _) in enumerate(pair_rows):
        contribution = 0.04 + 0.96 * ranks[index]
        mass[frame] = mass.get(frame, 0.0) + contribution

    return mass


def _human_reach_sets(relocalization, human_frames):
    humans = set(int(value) for value in human_frames)
    reach = {human: set() for human in humans}

    for target in relocalization.get("results", []):
        if not isinstance(target, dict):
            continue

        try:
            frame = int(target["frame"])
        except (KeyError, TypeError, ValueError):
            continue

        for candidate in target.get("candidates", []):
            if not isinstance(candidate, dict):
                continue

            try:
                anchor = int(candidate["anchor_frame"])
            except (KeyError, TypeError, ValueError):
                continue

            if anchor not in humans:
                continue

            verification = candidate.get("verification")
            if (
                isinstance(verification, dict)
                and verification.get("status") == "ok"
            ):
                reach[anchor].add(frame)

    return reach


def _candidate_reach_sets(hub_probe):
    result = {}

    for row in hub_probe.get("candidate_results", []):
        if not isinstance(row, dict) or row.get("status") != "ok":
            continue

        try:
            frame = int(row["frame"])
        except (KeyError, TypeError, ValueError):
            continue

        result[frame] = set(
            int(value)
            for value in row.get("verified_targets", [])
        )

    return result


def _jaccard(first, second):
    union = first | second
    if not union:
        return 0.0
    return len(first & second) / len(union)


def _weighted_jaccard(first, second, mass):
    union = first | second
    if not union:
        return 0.0

    intersection = first & second

    numerator = sum(
        float(mass.get(frame, 0.0))
        for frame in intersection
    )
    denominator = sum(
        float(mass.get(frame, 0.0))
        for frame in union
    )

    if denominator <= 1e-12:
        return _jaccard(first, second)

    return numerator / denominator


def _analyse_state(
    *,
    label,
    candidates,
    human_reach,
    conflict_mass,
    exclude_candidates,
):
    rows = []

    total_mass = sum(conflict_mass.values())

    for candidate, reach in sorted(candidates.items()):
        if candidate in set(exclude_candidates):
            continue

        mass_reached = sum(
            float(conflict_mass.get(frame, 0.0))
            for frame in reach
        )

        similarities = []

        for human, human_set in sorted(human_reach.items()):
            similarities.append(
                {
                    "human_frame": int(human),
                    "jaccard": float(
                        _jaccard(reach, human_set)
                    ),
                    "conflict_weighted_jaccard": float(
                        _weighted_jaccard(
                            reach,
                            human_set,
                            conflict_mass,
                        )
                    ),
                    "overlap_count": len(
                        reach & human_set
                    ),
                }
            )

        similarities.sort(
            key=lambda item: (
                -item["conflict_weighted_jaccard"],
                -item["jaccard"],
                item["human_frame"],
            )
        )

        closest = (
            similarities[0]
            if similarities
            else None
        )

        redundancy = (
            float(
                closest[
                    "conflict_weighted_jaccard"
                ]
            )
            if closest is not None
            else 0.0
        )

        novelty = 1.0 - redundancy

        # First diagnostic form: actual conflict mass not duplicated by the
        # closest current HUMAN reach signature.
        marginal_mass = mass_reached * novelty

        rows.append(
            {
                "frame": candidate,
                "reach_count": len(reach),
                "conflict_mass_reached": float(
                    mass_reached
                ),
                "conflict_mass_fraction": (
                    mass_reached / total_mass
                    if total_mass > 1e-12
                    else 0.0
                ),
                "closest_human": (
                    closest["human_frame"]
                    if closest
                    else None
                ),
                "max_weighted_jaccard": redundancy,
                "signature_novelty": novelty,
                "marginal_conflict_mass": float(
                    marginal_mass
                ),
                "all_human_similarities": similarities,
            }
        )

    mass_rank = _average_percentile(
        {
            int(row["frame"]): float(
                row["conflict_mass_reached"]
            )
            for row in rows
        }
    )

    novelty_rank = _average_percentile(
        {
            int(row["frame"]): float(
                row["signature_novelty"]
            )
            for row in rows
        }
    )

    marginal_rank = _average_percentile(
        {
            int(row["frame"]): float(
                row["marginal_conflict_mass"]
            )
            for row in rows
        }
    )

    for row in rows:
        frame = int(row["frame"])
        row["mass_rank"] = mass_rank[frame]
        row["novelty_rank"] = novelty_rank[frame]
        row["marginal_rank"] = marginal_rank[frame]

    by_mass = sorted(
        rows,
        key=lambda row: (
            -row["conflict_mass_reached"],
            -row["signature_novelty"],
            row["frame"],
        ),
    )

    by_marginal = sorted(
        rows,
        key=lambda row: (
            -row["marginal_conflict_mass"],
            -row["conflict_mass_reached"],
            -row["signature_novelty"],
            row["frame"],
        ),
    )

    for rank, row in enumerate(by_mass, start=1):
        row["pure_mass_rank"] = rank

    for rank, row in enumerate(by_marginal, start=1):
        row["marginal_signature_rank"] = rank

    return {
        "label": label,
        "human_frames": sorted(human_reach),
        "candidate_count": len(rows),
        "by_conflict_mass": by_mass,
        "by_marginal_signature": by_marginal,
    }


def _fmt(value, digits=3):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _print_state(title, state):
    print()
    print(title)
    print("-" * len(title))
    print(
        f"{'RANK':>4} "
        f"{'FRAME':>6} "
        f"{'REACH':>6} "
        f"{'MASS':>8} "
        f"{'M%':>6} "
        f"{'NEAR H':>7} "
        f"{'W-JAC':>7} "
        f"{'NOV':>7} "
        f"{'MARG':>8}"
    )
    print("-" * 75)

    for row in state["by_marginal_signature"]:
        print(
            f"{int(row['marginal_signature_rank']):>4} "
            f"{int(row['frame']):>6} "
            f"{int(row['reach_count']):>6} "
            f"{float(row['conflict_mass_reached']):>8.2f} "
            f"{100 * float(row['conflict_mass_fraction']):>5.1f}% "
            f"{str(row.get('closest_human')):>7} "
            f"{float(row['max_weighted_jaccard']):>7.3f} "
            f"{float(row['signature_novelty']):>7.3f} "
            f"{float(row['marginal_conflict_mass']):>8.2f}"
        )


def main():
    print("=" * 104)
    print("HUB REACH-SIGNATURE NOVELTY V1 · CÓRDOBA")
    print("=" * 104)

    for label, path in (
        ("relocalization", RELOC),
        ("hub probe", HUB_PROBE),
        ("sparse before", SPARSE_BEFORE),
        ("sparse after", SPARSE_AFTER),
    ):
        if not path.exists():
            print(f"ERROR: falta {label}: {path}")
            return 2

    relocalization = _load(RELOC)
    hub_probe = _load(HUB_PROBE)
    sparse_before = _load(SPARSE_BEFORE)
    sparse_after = _load(SPARSE_AFTER)

    candidates = _candidate_reach_sets(hub_probe)

    before_human_reach = _human_reach_sets(
        relocalization,
        HUMANS_BEFORE,
    )
    after_human_reach = _human_reach_sets(
        relocalization,
        HUMANS_AFTER,
    )

    before_mass = _conflict_mass(sparse_before)
    after_mass = _conflict_mass(sparse_after)

    before = _analyse_state(
        label="before_human_0",
        candidates=candidates,
        human_reach=before_human_reach,
        conflict_mass=before_mass,
        exclude_candidates=HUMANS_BEFORE,
    )

    after = _analyse_state(
        label="after_human_0",
        candidates=candidates,
        human_reach=after_human_reach,
        conflict_mass=after_mass,
        exclude_candidates=HUMANS_AFTER,
    )

    print()
    print("HUMAN BEFORE:", HUMANS_BEFORE)
    print("HUMAN AFTER :", HUMANS_AFTER)

    _print_state(
        "A · BEFORE HUMAN 0 · MARGINAL HUB SIGNATURE",
        before,
    )

    _print_state(
        "B · AFTER HUMAN 0 · MARGINAL HUB SIGNATURE",
        after,
    )

    before_map = {
        int(row["frame"]): row
        for row in before["by_marginal_signature"]
    }
    after_map = {
        int(row["frame"]): row
        for row in after["by_marginal_signature"]
    }

    print()
    print("CAUSAL MOVEMENT")
    print("---------------")
    print(
        f"{'FRAME':>6} "
        f"{'BEFORE':>7} "
        f"{'AFTER':>7} "
        f"{'ΔRANK':>7} "
        f"{'NOV B':>7} "
        f"{'NOV A':>7} "
        f"{'NEAR A':>7}"
    )
    print("-" * 57)

    for frame in sorted(set(before_map) | set(after_map)):
        b = before_map.get(frame)
        a = after_map.get(frame)

        before_rank = (
            int(b["marginal_signature_rank"])
            if b else None
        )
        after_rank = (
            int(a["marginal_signature_rank"])
            if a else None
        )

        movement = (
            after_rank - before_rank
            if before_rank is not None
            and after_rank is not None
            else None
        )

        print(
            f"{frame:>6} "
            f"{str(before_rank or '-'):>7} "
            f"{str(after_rank or '-'):>7} "
            f"{str(movement if movement is not None else '-'):>7} "
            f"{_fmt(b['signature_novelty'] if b else None):>7} "
            f"{_fmt(a['signature_novelty'] if a else None):>7} "
            f"{str(a.get('closest_human') if a else '-'):>7}"
        )

    result = {
        "status": "ok",
        "version": 1,
        "algorithm": "hub_reach_signature_novelty_v1",
        "policy": {
            "reruns_visual_verification": False,
            "candidate_homographies_used": False,
            "signature_is_direct_verified_target_set": True,
            "redundancy_uses_conflict_weighted_jaccard": True,
            "production_threshold_defined": False,
        },
        "before": before,
        "after": after,
    }

    OUTPUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("Guardado:")
    print(OUTPUT)
    print()
    print(
        "READ-ONLY. Si la firma funciona causalmente, la incorporaremos "
        "al selector automático V4."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

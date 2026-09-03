"""Sparse Active Anchor Selector V3: conflict evidence mass.

Experimental/read-only active HUMAN recommendation.

V2 combined:
    conflict magnitude x novelty vs current HUMAN

but treated a conflict supported by only one HUMAN pair too similarly to a
conflict reproduced across many independent HUMAN pairs.

V3 models three distinct ideas:

1. CONFLICT MAGNITUDE
   How large is the disagreement at this target?

2. CONFLICT EVIDENCE MASS
   How many independent HUMAN-pair comparisons reproduce disagreement, and how
   strong are those pair disagreements relative to the rest of THIS clip?

3. HUMAN NOVELTY
   How different is the target/candidate from the HUMAN states already known?

No fixed pixel threshold is used to decide "bad". Pair disagreements and
HUMAN distances are converted to empirical percentiles inside the clip.

The selector still does NOT certify CAMERA correctness. It only ranks where an
additional exact HUMAN is expected to provide the most marginal information.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import math

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class SparseActiveAnchorV3Config:
    quality_floor: float = 0.16
    minimum_support_for_conflict: int = 2

    # Prevent exact zero percentiles from annihilating a signal.
    percentile_floor: float = 0.04

    # Human-calibratability is only a soft tiebreaker.
    quality_soft_weight: float = 0.12

    # Final candidate score is a geometric mean of:
    # candidate conflict-evidence gain percentile and candidate HUMAN novelty.
    gain_exponent: float = 0.62
    novelty_exponent: float = 0.38

    pareto_limit: int = 50


def _load(value: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8"))


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percentiles(values: dict[int, float]) -> dict[int, float]:
    """Average empirical percentile rank in [0,1], preserving ties."""
    if not values:
        return {}

    keys = sorted(values)
    array = np.asarray(
        [float(values[key]) for key in keys],
        dtype=np.float64,
    )

    order = np.argsort(array, kind="mergesort")
    ranks = np.zeros(len(array), dtype=np.float64)

    pos = 0
    while pos < len(array):
        end = pos + 1
        value = array[order[pos]]

        while (
            end < len(array)
            and np.isclose(
                array[order[end]],
                value,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            end += 1

        average = 0.5 * (pos + end - 1)
        rank = (
            1.0
            if len(array) == 1
            else average / (len(array) - 1)
        )

        for index in order[pos:end]:
            ranks[int(index)] = rank

        pos = end

    return {
        key: float(ranks[index])
        for index, key in enumerate(keys)
    }


def _soft_percentile(
    rank: float,
    floor: float,
) -> float:
    floor = min(max(float(floor), 0.0), 0.49)
    return floor + (1.0 - floor) * min(
        max(float(rank), 0.0),
        1.0,
    )


def _quality_by_frame(v1: dict[str, Any]) -> dict[int, float]:
    result: dict[int, float] = {}

    for state in v1.get("states", []):
        if not isinstance(state, dict):
            continue
        try:
            frame = int(state["frame"])
            quality = float(state.get("quality_score") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        result[frame] = quality

    return result


def _direct_graph(v1: dict[str, Any]) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {}

    for state in v1.get("states", []):
        if not isinstance(state, dict):
            continue
        try:
            frame = int(state["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        adjacency.setdefault(frame, set())

    for edge in v1.get("edges", []):
        if not isinstance(edge, dict) or not edge.get("usable"):
            continue
        try:
            left = int(edge["left_frame"])
            right = int(edge["right_frame"])
        except (KeyError, TypeError, ValueError):
            continue

        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)

    return adjacency


def _nearest_human_state(
    relocalization: dict[str, Any],
    human_frames: set[int],
) -> dict[int, dict[str, Any]]:
    """Nearest directly verified current HUMAN, for novelty only."""
    result: dict[int, dict[str, Any]] = {}

    for target in relocalization.get("results", []):
        if not isinstance(target, dict):
            continue

        try:
            frame = int(target["frame"])
        except (KeyError, TypeError, ValueError):
            continue

        if frame in human_frames:
            continue

        rows: list[dict[str, Any]] = []

        for candidate in target.get("candidates", []):
            if not isinstance(candidate, dict):
                continue

            try:
                anchor = int(candidate["anchor_frame"])
            except (KeyError, TypeError, ValueError):
                continue

            if anchor not in human_frames:
                continue

            verification = candidate.get("verification")
            if (
                not isinstance(verification, dict)
                or verification.get("status") != "ok"
            ):
                continue

            distance = verification.get("direct_state_distance")
            if not isinstance(distance, dict):
                continue

            median = _finite(distance.get("median_px"))
            p90 = _finite(distance.get("p90_px"))
            if median is None or p90 is None:
                continue

            combined = math.sqrt(
                max(median, 0.0) * max(p90, 0.0)
            )

            rows.append(
                {
                    "anchor_frame": anchor,
                    "median_px": float(median),
                    "p90_px": float(p90),
                    "combined_px": float(combined),
                }
            )

        if rows:
            rows.sort(
                key=lambda item: (
                    item["combined_px"],
                    item["median_px"],
                    item["p90_px"],
                    item["anchor_frame"],
                )
            )
            result[frame] = rows[0]

    return result


def _collect_pair_disagreements(
    sparse: dict[str, Any],
    cfg: SparseActiveAnchorV3Config,
) -> tuple[
    dict[tuple[int, int, int], float],
    dict[int, dict[str, Any]],
]:
    """Collect every comparable HUMAN-pair disagreement across targets."""
    pair_values: dict[tuple[int, int, int], float] = {}
    target_map: dict[int, dict[str, Any]] = {}

    for target in sparse.get("targets", []):
        if not isinstance(target, dict):
            continue

        try:
            frame = int(target["frame"])
            support = int(target.get("support_count") or 0)
        except (KeyError, TypeError, ValueError):
            continue

        target_map[frame] = target

        if target.get("is_exact_human"):
            continue
        if support < int(cfg.minimum_support_for_conflict):
            continue

        for pair in target.get("pair_details", []):
            if not isinstance(pair, dict):
                continue
            if pair.get("status") != "ok":
                continue

            value = _finite(pair.get("median_px"))
            if value is None:
                continue

            try:
                left = int(pair["left_anchor_frame"])
                right = int(pair["right_anchor_frame"])
            except (KeyError, TypeError, ValueError):
                continue

            pair_values[
                (
                    frame,
                    min(left, right),
                    max(left, right),
                )
            ] = max(0.0, value)

    return pair_values, target_map


def _target_information_pressure(
    sparse: dict[str, Any],
    relocalization: dict[str, Any],
    human_frames: set[int],
    cfg: SparseActiveAnchorV3Config,
) -> tuple[
    dict[int, float],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    """Build per-target conflict evidence without fixed px thresholds."""
    pair_values, target_map = _collect_pair_disagreements(
        sparse,
        cfg,
    )

    if not pair_values:
        return {}, {}, target_map

    # Rank EVERY independent HUMAN-pair disagreement clip-wide.
    pair_rank = _percentiles(
        {
            index: value
            for index, value in enumerate(pair_values.values())
        }
    )

    keys = list(pair_values)
    pair_soft_rank: dict[
        tuple[int, int, int],
        float
    ] = {}

    for index, key in enumerate(keys):
        pair_soft_rank[key] = _soft_percentile(
            pair_rank[index],
            cfg.percentile_floor,
        )

    # Evidence mass naturally increases if disagreement is reproduced across
    # several independent HUMAN pairs.
    raw_evidence_mass: dict[int, float] = {}
    pair_count: dict[int, int] = {}

    for key, soft_rank in pair_soft_rank.items():
        frame = int(key[0])
        raw_evidence_mass[frame] = (
            raw_evidence_mass.get(frame, 0.0)
            + float(soft_rank)
        )
        pair_count[frame] = pair_count.get(frame, 0) + 1

    evidence_rank = _percentiles(raw_evidence_mass)

    raw_magnitude: dict[int, float] = {}

    for frame in raw_evidence_mass:
        target = target_map.get(frame, {})
        value = _finite(target.get("pair_p90_px"))
        if value is not None:
            raw_magnitude[frame] = value

    magnitude_rank = _percentiles(raw_magnitude)

    nearest = _nearest_human_state(
        relocalization,
        human_frames,
    )

    measured_novelty = {
        frame: float(nearest[frame]["combined_px"])
        for frame in raw_evidence_mass
        if frame in nearest
    }
    novelty_rank = _percentiles(measured_novelty)

    pressure: dict[int, float] = {}
    diagnostics: dict[int, dict[str, Any]] = {}

    for frame, evidence_mass in raw_evidence_mass.items():
        e_rank = float(evidence_rank.get(frame, 0.0))
        m_rank = float(magnitude_rank.get(frame, 0.0))

        if frame in novelty_rank:
            n_rank = float(novelty_rank[frame])
            novelty_measured = True
        else:
            # No current HUMAN directly verifies to this state => maximum
            # novelty for selection purposes, not a correctness judgement.
            n_rank = 1.0
            novelty_measured = False

        e_soft = _soft_percentile(
            e_rank,
            cfg.percentile_floor,
        )
        m_soft = _soft_percentile(
            m_rank,
            cfg.percentile_floor,
        )
        n_soft = _soft_percentile(
            n_rank,
            cfg.percentile_floor,
        )

        # Evidence is the dominant term: many independent conflicting HUMAN
        # pairs should outweigh a single dramatic pair.
        conflict_evidence = math.sqrt(
            e_soft * m_soft
        )

        # If a target is already close to a known HUMAN, its unresolved
        # information pressure decreases.
        information_pressure = (
            conflict_evidence
            * math.sqrt(n_soft)
        )

        pressure[frame] = float(information_pressure)

        target = target_map.get(frame, {})

        diagnostics[frame] = {
            "frame": frame,
            "support_count": int(
                target.get("support_count") or 0
            ),
            "comparable_pair_count": int(
                pair_count.get(frame, 0)
            ),
            "raw_evidence_mass": float(evidence_mass),
            "evidence_rank": e_rank,
            "magnitude_rank": m_rank,
            "target_novelty_rank": n_rank,
            "target_novelty_measured": novelty_measured,
            "nearest_human": nearest.get(frame),
            "information_pressure": float(
                information_pressure
            ),
            "support_anchor_frames": target.get(
                "support_anchor_frames",
                [],
            ),
            "pair_median_px": target.get(
                "pair_median_px"
            ),
            "pair_p90_px": target.get(
                "pair_p90_px"
            ),
        }

    return pressure, diagnostics, target_map


def _candidate_gain(
    candidate: int,
    *,
    adjacency: dict[int, set[int]],
    pressure: dict[int, float],
) -> tuple[float, list[int]]:
    touched = set(adjacency.get(candidate, set()))
    touched.add(candidate)

    frames = sorted(touched & set(pressure))

    gain = float(
        sum(pressure[frame] for frame in frames)
    )

    return gain, frames


def _pareto_frontier(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []

    for candidate in candidates:
        dominated = False

        for other in candidates:
            if other is candidate:
                continue

            at_least = (
                other["gain_rank"] >= candidate["gain_rank"]
                and other["candidate_novelty_rank"]
                >= candidate["candidate_novelty_rank"]
                and other["quality_score"]
                >= candidate["quality_score"]
            )

            strict = (
                other["gain_rank"] > candidate["gain_rank"]
                or other["candidate_novelty_rank"]
                > candidate["candidate_novelty_rank"]
                or other["quality_score"]
                > candidate["quality_score"]
            )

            if at_least and strict:
                dominated = True
                break

        if not dominated:
            frontier.append(candidate)

    frontier.sort(
        key=lambda item: (
            -item["score"],
            -item["gain_rank"],
            -item["candidate_novelty_rank"],
            -item["quality_score"],
            item["frame"],
        )
    )

    return frontier


def analyse_sparse_active_anchor_v3(
    v1_result: dict[str, Any] | str | Path,
    sparse_result: dict[str, Any] | str | Path,
    relocalization_result: dict[str, Any] | str | Path,
    *,
    human_frames: list[int] | set[int],
    config: SparseActiveAnchorV3Config | None = None,
) -> dict[str, Any]:
    cfg = config or SparseActiveAnchorV3Config()

    try:
        v1 = _load(v1_result)
        sparse = _load(sparse_result)
        relocalization = _load(relocalization_result)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "status": "input_read_failed",
            "error": str(error),
        }

    for label, value in (
        ("v1", v1),
        ("sparse", sparse),
        ("relocalization", relocalization),
    ):
        if value.get("status") != "ok":
            return {
                "status": f"invalid_{label}",
                "input_status": value.get("status"),
            }

    humans = {int(frame) for frame in human_frames}
    quality = _quality_by_frame(v1)
    adjacency = _direct_graph(v1)

    pressure, target_diagnostics, _ = (
        _target_information_pressure(
            sparse,
            relocalization,
            humans,
            cfg,
        )
    )

    if not pressure:
        return {"status": "no_measured_conflict_evidence"}

    nearest_candidate_human = _nearest_human_state(
        relocalization,
        humans,
    )

    candidate_frames = [
        frame
        for frame in sorted(adjacency)
        if frame not in humans
        and float(quality.get(frame, 0.0))
        >= float(cfg.quality_floor)
    ]

    raw_candidates: list[dict[str, Any]] = []

    for frame in candidate_frames:
        gain, touched = _candidate_gain(
            frame,
            adjacency=adjacency,
            pressure=pressure,
        )

        if gain <= 0.0:
            continue

        raw_candidates.append(
            {
                "frame": int(frame),
                "raw_information_gain": float(gain),
                "covered_conflicted_state_count": len(touched),
                "covered_conflicted_frames": touched,
                "quality_score": float(
                    quality.get(frame, 0.0)
                ),
                "nearest_human": nearest_candidate_human.get(
                    frame
                ),
            }
        )

    if not raw_candidates:
        return {"status": "no_candidates"}

    gain_rank = _percentiles(
        {
            int(item["frame"]): float(
                item["raw_information_gain"]
            )
            for item in raw_candidates
        }
    )

    measured_candidate_novelty = {
        int(item["frame"]): float(
            item["nearest_human"]["combined_px"]
        )
        for item in raw_candidates
        if item.get("nearest_human")
    }

    candidate_novelty_rank_measured = _percentiles(
        measured_candidate_novelty
    )

    gain_exp = max(float(cfg.gain_exponent), EPS)
    novelty_exp = max(float(cfg.novelty_exponent), EPS)
    exponent_total = gain_exp + novelty_exp

    quality_weight = min(
        max(float(cfg.quality_soft_weight), 0.0),
        0.5,
    )

    for item in raw_candidates:
        frame = int(item["frame"])
        g_rank = float(gain_rank.get(frame, 0.0))

        if frame in candidate_novelty_rank_measured:
            n_rank = float(
                candidate_novelty_rank_measured[frame]
            )
            novelty_measured = True
        else:
            n_rank = 1.0
            novelty_measured = False

        g_soft = _soft_percentile(
            g_rank,
            cfg.percentile_floor,
        )
        n_soft = _soft_percentile(
            n_rank,
            cfg.percentile_floor,
        )

        core = (
            (g_soft ** gain_exp)
            * (n_soft ** novelty_exp)
        ) ** (1.0 / exponent_total)

        q = float(item["quality_score"])

        score = core * (
            (1.0 - quality_weight)
            + quality_weight * q
        )

        item.update(
            {
                "gain_rank": g_rank,
                "candidate_novelty_rank": n_rank,
                "candidate_novelty_measured": novelty_measured,
                "score": float(score),
            }
        )

    ranking = sorted(
        raw_candidates,
        key=lambda item: (
            -item["score"],
            -item["gain_rank"],
            -item["candidate_novelty_rank"],
            -item["quality_score"],
            item["frame"],
        ),
    )

    for rank, item in enumerate(ranking, start=1):
        item["rank"] = rank

    frontier = _pareto_frontier(raw_candidates)

    pressure_values = np.asarray(
        list(pressure.values()),
        dtype=np.float64,
    )

    target_top = sorted(
        target_diagnostics.values(),
        key=lambda item: (
            -item["information_pressure"],
            -item["comparable_pair_count"],
            item["frame"],
        ),
    )

    return {
        "status": "ok",
        "version": 3,
        "algorithm": "sparse_active_anchor_selector_v3",
        "policy": {
            "fixed_pixel_conflict_threshold": False,
            "pairwise_conflict_ranked_clip_empirically": True,
            "independent_pair_evidence_accumulates": True,
            "human_distance_used_only_for_novelty": True,
            "no_automatic_human_action": True,
        },
        "config": asdict(cfg),
        "human_frames": sorted(humans),
        "conflicted_target_count": len(pressure),
        "information_pressure_distribution": {
            "median": float(np.median(pressure_values)),
            "p75": float(np.percentile(pressure_values, 75)),
            "p90": float(np.percentile(pressure_values, 90)),
            "p95": float(np.percentile(pressure_values, 95)),
            "max": float(np.max(pressure_values)),
        },
        "top_target_pressures": target_top[:80],
        "candidate_count": len(ranking),
        "recommendations": ranking,
        "pareto_frontier": frontier[
            : max(1, int(cfg.pareto_limit))
        ],
    }

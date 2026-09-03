"""Sparse Active Anchor Selector V2: marginal information vs HUMAN redundancy.

V1 ranked candidates only by how much measured multi-HUMAN conflict they touch.
That can recommend a frame almost identical to an existing HUMAN, which is not
the marginal-information objective we want.

V2 adds a strictly different use of direct_state_distance:

    NOT as a transfer correctness/safety threshold.
    ONLY as a soft redundancy/novelty signal relative to current HUMAN anchors.

For every candidate keyframe we inspect the already-computed HUMAN->target
direct visual verifications from Anchor Relocalization and measure distance to
its nearest current HUMAN. Distances are converted to an empirical percentile
within the SAME clip, avoiding a fixed pixel threshold.

Outputs three rankings:
- conflict_only: V1 behaviour;
- balanced: conflict rank x HUMAN-novelty rank;
- novelty_first: diagnostic sensitivity.

No production threshold or automatic HUMAN action is defined here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import math

import numpy as np


@dataclass(frozen=True)
class SparseActiveAnchorV2Config:
    quality_floor: float = 0.16
    minimum_support_for_conflict: int = 2

    # Small floors prevent one zero percentile from annihilating all evidence.
    rank_floor: float = 0.05

    # Quality is deliberately weak; it should break ties, not dominate.
    quality_soft_weight: float = 0.15

    pareto_limit: int = 40


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


def _average_percentile(values: dict[int, float]) -> dict[int, float]:
    """Average empirical percentile in [0,1], preserving ties."""
    if not values:
        return {}

    frames = sorted(values)
    array = np.asarray(
        [values[frame] for frame in frames],
        dtype=np.float64,
    )
    order = np.argsort(array, kind="mergesort")
    ranks = np.zeros(len(array), dtype=np.float64)

    position = 0
    while position < len(array):
        end = position + 1
        value = array[order[position]]

        while end < len(array) and np.isclose(
            array[order[end]],
            value,
            rtol=0.0,
            atol=1e-12,
        ):
            end += 1

        average = 0.5 * (position + end - 1)
        rank = (
            1.0
            if len(array) == 1
            else average / (len(array) - 1)
        )

        for index in order[position:end]:
            ranks[int(index)] = rank

        position = end

    return {
        frame: float(ranks[index])
        for index, frame in enumerate(frames)
    }


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


def _target_map(value: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}

    for item in value.get("targets", value.get("results", [])):
        if not isinstance(item, dict):
            continue
        try:
            frame = int(item["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        result[frame] = item

    return result


def _conflict_raw(
    sparse: dict[str, Any],
    cfg: SparseActiveAnchorV2Config,
) -> dict[int, float]:
    raw: dict[int, float] = {}

    for target in sparse.get("targets", []):
        if not isinstance(target, dict) or target.get("is_exact_human"):
            continue

        try:
            frame = int(target["frame"])
            support = int(target.get("support_count") or 0)
        except (KeyError, TypeError, ValueError):
            continue

        if support < int(cfg.minimum_support_for_conflict):
            continue

        disagreement = _finite(target.get("pair_p90_px"))
        if disagreement is None:
            continue

        comparable = _finite(
            target.get("comparable_pair_fraction")
        )
        if comparable is None:
            comparable = 0.0

        raw[frame] = disagreement * (
            0.60 + 0.40 * min(max(comparable, 0.0), 1.0)
        )

    return raw


def _nearest_human_state(
    relocalization: dict[str, Any],
    human_frames: set[int],
) -> dict[int, dict[str, Any]]:
    """Nearest current HUMAN in direct image-state displacement.

    This reads only candidates that already passed the robust direct verifier.
    Missing support is treated as maximally novel later, not as unsafe.
    """
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
            if not isinstance(verification, dict):
                continue

            if verification.get("status") != "ok":
                continue

            distance = verification.get("direct_state_distance")
            if not isinstance(distance, dict):
                continue

            median = _finite(distance.get("median_px"))
            p90 = _finite(distance.get("p90_px"))

            if median is None or p90 is None:
                continue

            # Median dominates; p90 makes severe projective changes count too.
            combined = math.sqrt(
                max(median, 0.0) * max(p90, 0.0)
            )

            rows.append(
                {
                    "anchor_frame": anchor,
                    "median_px": float(median),
                    "p90_px": float(p90),
                    "combined_px": float(combined),
                    "inlier_count": int(
                        verification.get("inlier_count") or 0
                    ),
                    "inlier_ratio": float(
                        verification.get("inlier_ratio") or 0.0
                    ),
                    "median_reprojection_px": _finite(
                        verification.get(
                            "median_reprojection_px"
                        )
                    ),
                }
            )

        if rows:
            rows.sort(
                key=lambda item: (
                    item["combined_px"],
                    item["median_px"],
                    item["p90_px"],
                    -item["inlier_count"],
                    item["anchor_frame"],
                )
            )
            result[frame] = rows[0]

    return result


def _candidate_conflict_gain(
    candidate: int,
    *,
    adjacency: dict[int, set[int]],
    conflict_weight: dict[int, float],
) -> tuple[float, list[int]]:
    touched = set(adjacency.get(candidate, set()))
    touched.add(candidate)

    frames = sorted(touched & set(conflict_weight))
    gain = float(sum(conflict_weight[frame] for frame in frames))
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
                other["conflict_rank"] >= candidate["conflict_rank"]
                and other["novelty_rank"] >= candidate["novelty_rank"]
                and other["quality_score"] >= candidate["quality_score"]
            )
            strict = (
                other["conflict_rank"] > candidate["conflict_rank"]
                or other["novelty_rank"] > candidate["novelty_rank"]
                or other["quality_score"] > candidate["quality_score"]
            )

            if at_least and strict:
                dominated = True
                break

        if not dominated:
            frontier.append(candidate)

    frontier.sort(
        key=lambda item: (
            -item["balanced_score"],
            -item["conflict_rank"],
            -item["novelty_rank"],
            -item["quality_score"],
            item["frame"],
        )
    )
    return frontier


def analyse_sparse_active_anchor_v2(
    v1_result: dict[str, Any] | str | Path,
    sparse_result: dict[str, Any] | str | Path,
    relocalization_result: dict[str, Any] | str | Path,
    *,
    human_frames: list[int] | set[int],
    config: SparseActiveAnchorV2Config | None = None,
) -> dict[str, Any]:
    cfg = config or SparseActiveAnchorV2Config()

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

    conflict_raw = _conflict_raw(sparse, cfg)
    if not conflict_raw:
        return {"status": "no_measured_conflicts"}

    conflict_state_rank = _average_percentile(conflict_raw)
    conflict_weight = {
        frame: 0.10 + 0.90 * rank
        for frame, rank in conflict_state_rank.items()
    }

    nearest = _nearest_human_state(
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

    # Empirical novelty is based on distance to nearest current HUMAN.
    measured_novelty = {
        frame: float(nearest[frame]["combined_px"])
        for frame in candidate_frames
        if frame in nearest
    }
    novelty_rank_measured = _average_percentile(
        measured_novelty
    )

    raw_candidates: list[dict[str, Any]] = []

    for frame in candidate_frames:
        gain, touched = _candidate_conflict_gain(
            frame,
            adjacency=adjacency,
            conflict_weight=conflict_weight,
        )

        if gain <= 0:
            continue

        raw_candidates.append(
            {
                "frame": int(frame),
                "conflict_gain": gain,
                "covered_conflict_count": len(touched),
                "covered_conflict_frames": touched,
                "quality_score": float(
                    quality.get(frame, 0.0)
                ),
                "nearest_human": nearest.get(frame),
            }
        )

    if not raw_candidates:
        return {"status": "no_candidates"}

    gain_rank = _average_percentile(
        {
            int(item["frame"]): float(item["conflict_gain"])
            for item in raw_candidates
        }
    )

    floor = min(max(float(cfg.rank_floor), 0.0), 0.49)
    quality_soft = min(
        max(float(cfg.quality_soft_weight), 0.0),
        0.5,
    )

    for item in raw_candidates:
        frame = int(item["frame"])
        c_rank = float(gain_rank.get(frame, 0.0))

        if frame in novelty_rank_measured:
            n_rank = float(novelty_rank_measured[frame])
            novelty_measured = True
        else:
            # If NO current HUMAN directly verifies to this candidate, it is
            # maximally novel for selection purposes. This is not a claim that
            # transfer is unsafe.
            n_rank = 1.0
            novelty_measured = False

        c_soft = floor + (1.0 - floor) * c_rank
        n_soft = floor + (1.0 - floor) * n_rank

        # Geometric mean rewards candidates that are simultaneously useful and
        # non-redundant instead of allowing one dimension to compensate fully.
        balanced_core = math.sqrt(c_soft * n_soft)
        q = float(item["quality_score"])
        balanced = balanced_core * (
            (1.0 - quality_soft)
            + quality_soft * q
        )

        # Sensitivity view that cares more about novelty.
        novelty_first = (
            (c_soft ** 0.35)
            * (n_soft ** 0.65)
            * (
                (1.0 - quality_soft)
                + quality_soft * q
            )
        )

        item.update(
            {
                "conflict_rank": c_rank,
                "novelty_rank": n_rank,
                "novelty_measured": novelty_measured,
                "balanced_score": float(balanced),
                "novelty_first_score": float(novelty_first),
            }
        )

    conflict_only = sorted(
        raw_candidates,
        key=lambda item: (
            -item["conflict_gain"],
            -item["covered_conflict_count"],
            -item["quality_score"],
            item["frame"],
        ),
    )

    balanced = sorted(
        raw_candidates,
        key=lambda item: (
            -item["balanced_score"],
            -item["conflict_rank"],
            -item["novelty_rank"],
            -item["quality_score"],
            item["frame"],
        ),
    )

    novelty_first = sorted(
        raw_candidates,
        key=lambda item: (
            -item["novelty_first_score"],
            -item["novelty_rank"],
            -item["conflict_rank"],
            -item["quality_score"],
            item["frame"],
        ),
    )

    for ranking in (conflict_only, balanced, novelty_first):
        for rank, item in enumerate(ranking, start=1):
            # Keep separate labels so the same dict can appear in all rankings.
            if ranking is conflict_only:
                item["conflict_only_rank"] = rank
            elif ranking is balanced:
                item["balanced_rank"] = rank
            else:
                item["novelty_first_rank"] = rank

    frontier = _pareto_frontier(raw_candidates)

    novelty_values = np.asarray(
        list(measured_novelty.values()),
        dtype=np.float64,
    )

    return {
        "status": "ok",
        "version": 2,
        "algorithm": "sparse_active_anchor_selector_v2",
        "policy": {
            "direct_state_distance_used_only_for_redundancy": True,
            "direct_state_distance_not_used_as_transfer_safety_gate": True,
            "fixed_pixel_novelty_threshold": False,
            "novelty_is_clip_empirical_percentile": True,
            "no_automatic_human_action": True,
        },
        "config": asdict(cfg),
        "human_frames": sorted(humans),
        "candidate_count": len(raw_candidates),
        "measured_novelty_candidate_count": len(measured_novelty),
        "nearest_human_distance_distribution_px": {
            "median": (
                float(np.median(novelty_values))
                if len(novelty_values)
                else None
            ),
            "p75": (
                float(np.percentile(novelty_values, 75))
                if len(novelty_values)
                else None
            ),
            "p90": (
                float(np.percentile(novelty_values, 90))
                if len(novelty_values)
                else None
            ),
            "max": (
                float(np.max(novelty_values))
                if len(novelty_values)
                else None
            ),
        },
        "conflict_only": conflict_only,
        "balanced": balanced,
        "novelty_first": novelty_first,
        "pareto_frontier": frontier[
            : max(1, int(cfg.pareto_limit))
        ],
    }

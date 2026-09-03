"""Sparse Active Anchor Selector V1.

Experimental/read-only active HUMAN recommendation after Sparse CAMERA Validator.

Unlike field_camera_active_anchor.py V1, which only considers genuinely
UNRESOLVED relocalization states, this selector considers CONFLICTED states:
states for which multiple existing HUMAN anchors produce materially different
field homographies.

Why
---
Córdoba showed that:
- a single-support target can still be accurate;
- a multi-support target can be wrong even when every direct visual match passed;
- adding one HUMAN in a conflicted region can improve a large fraction of video.

Therefore this selector does NOT treat support_count < 2 as automatically bad.
It ranks direct field-H disagreement and asks:

    "If this candidate became an exact HUMAN, how much current sparse conflict
     could it directly supervise?"

No production acceptance threshold is set here. Conflict magnitude is converted
to a robust percentile rank so one extreme outlier cannot dominate the plan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import math

import numpy as np


@dataclass(frozen=True)
class SparseActiveAnchorConfig:
    quality_floor: float = 0.16

    # Conflict must have at least two independent HUMAN predictions.
    minimum_support_for_conflict: int = 2

    # Direct visual graph only. No transitive coverage.
    include_candidate_self: bool = True

    # Soft score weights. Conflict gain dominates; calibration quality only
    # breaks near-ties.
    conflict_gain_weight: float = 1.0
    quality_weight: float = 0.12

    # Diagnostic greedy preview only. Real workflow must recompute after each
    # newly calibrated HUMAN.
    greedy_preview_count: int = 5


def _load(value: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return json.loads(
        Path(value).read_text(encoding="utf-8")
    )


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percentile_ranks(values: dict[int, float]) -> dict[int, float]:
    """Empirical [0,1] rank. Equal values receive the same average rank."""
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
        current = array[order[position]]

        while (
            end < len(array)
            and array[order[end]] == current
        ):
            end += 1

        average_position = 0.5 * (
            position + (end - 1)
        )

        if len(array) == 1:
            rank = 1.0
        else:
            rank = average_position / (len(array) - 1)

        for index in order[position:end]:
            ranks[int(index)] = rank

        position = end

    return {
        frame: float(ranks[index])
        for index, frame in enumerate(frames)
    }


def _direct_graph(
    v1: dict[str, Any],
) -> dict[int, set[int]]:
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


def _quality_by_frame(
    v1: dict[str, Any],
) -> dict[int, float]:
    quality: dict[int, float] = {}

    for state in v1.get("states", []):
        if not isinstance(state, dict):
            continue

        try:
            frame = int(state["frame"])
            value = float(state.get("quality_score") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue

        quality[frame] = value

    return quality


def _conflict_weights(
    sparse: dict[str, Any],
    cfg: SparseActiveAnchorConfig,
) -> tuple[
    dict[int, float],
    dict[int, dict[str, Any]],
]:
    raw: dict[int, float] = {}
    target_by_frame: dict[int, dict[str, Any]] = {}

    for target in sparse.get("targets", []):
        if not isinstance(target, dict):
            continue

        try:
            frame = int(target["frame"])
        except (KeyError, TypeError, ValueError):
            continue

        target_by_frame[frame] = target

        if target.get("is_exact_human"):
            continue

        support = int(target.get("support_count") or 0)
        if support < int(cfg.minimum_support_for_conflict):
            continue

        # pair_p90 emphasizes geometrically meaningful disagreement over the
        # visible pitch extent while staying robust to one local point.
        disagreement = _finite(
            target.get("pair_p90_px")
        )
        if disagreement is None:
            continue

        comparable_fraction = _finite(
            target.get("comparable_pair_fraction")
        )
        if comparable_fraction is None:
            comparable_fraction = 0.0

        # Preserve the measured disagreement for ranking, mildly downweighting
        # targets whose HUMAN predictions could barely be compared.
        raw[frame] = disagreement * (
            0.60 + 0.40 * max(
                0.0,
                min(1.0, comparable_fraction),
            )
        )

    ranks = _percentile_ranks(raw)

    # Keep a small positive floor: every measured conflict is information, while
    # high-percentile conflicts still dominate.
    weights = {
        frame: 0.10 + 0.90 * rank
        for frame, rank in ranks.items()
    }

    return weights, target_by_frame


def _candidate_record(
    candidate: int,
    *,
    conflict_weights: dict[int, float],
    target_by_frame: dict[int, dict[str, Any]],
    adjacency: dict[int, set[int]],
    quality: dict[int, float],
    human_frames: set[int],
    cfg: SparseActiveAnchorConfig,
    remaining_conflicts: set[int] | None = None,
) -> dict[str, Any]:
    direct = set(adjacency.get(candidate, set()))

    if cfg.include_candidate_self:
        direct.add(candidate)

    conflicts = set(conflict_weights)

    if remaining_conflicts is not None:
        conflicts &= remaining_conflicts

    covered = sorted(direct & conflicts)

    gain = float(
        sum(
            conflict_weights[frame]
            for frame in covered
        )
    )

    q = float(quality.get(candidate, 0.0))
    score = (
        cfg.conflict_gain_weight * gain
        + cfg.quality_weight * q
    )

    support_sets: dict[tuple[int, ...], int] = {}

    for frame in covered:
        target = target_by_frame.get(frame, {})
        anchors = tuple(
            sorted(
                int(value)
                for value in target.get(
                    "support_anchor_frames",
                    [],
                )
            )
        )
        support_sets[anchors] = (
            support_sets.get(anchors, 0) + 1
        )

    return {
        "frame": int(candidate),
        "score": float(score),
        "conflict_gain": float(gain),
        "quality_score": q,
        "covered_conflict_count": len(covered),
        "covered_conflict_frames": covered,
        "covered_support_patterns": [
            {
                "support_anchor_frames": list(pattern),
                "count": int(count),
            }
            for pattern, count in sorted(
                support_sets.items(),
                key=lambda item: (
                    -item[1],
                    item[0],
                ),
            )
        ],
        "is_existing_human": candidate in human_frames,
    }


def analyse_sparse_active_anchor(
    v1_result: dict[str, Any] | str | Path,
    sparse_result: dict[str, Any] | str | Path,
    *,
    human_frames: list[int] | set[int],
    config: SparseActiveAnchorConfig | None = None,
) -> dict[str, Any]:
    cfg = config or SparseActiveAnchorConfig()

    try:
        v1 = _load(v1_result)
        sparse = _load(sparse_result)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "status": "input_read_failed",
            "error": str(error),
        }

    if v1.get("status") != "ok":
        return {
            "status": "invalid_v1",
            "input_status": v1.get("status"),
        }

    if sparse.get("status") != "ok":
        return {
            "status": "invalid_sparse",
            "input_status": sparse.get("status"),
        }

    humans = {
        int(frame)
        for frame in human_frames
    }

    adjacency = _direct_graph(v1)
    quality = _quality_by_frame(v1)

    conflict_weights, target_by_frame = (
        _conflict_weights(sparse, cfg)
    )

    if not conflict_weights:
        return {
            "status": "no_measured_conflicts",
            "human_frames": sorted(humans),
        }

    candidates: list[dict[str, Any]] = []

    for candidate in sorted(adjacency):
        if candidate in humans:
            continue

        if (
            float(quality.get(candidate, 0.0))
            < float(cfg.quality_floor)
        ):
            continue

        record = _candidate_record(
            candidate,
            conflict_weights=conflict_weights,
            target_by_frame=target_by_frame,
            adjacency=adjacency,
            quality=quality,
            human_frames=humans,
            cfg=cfg,
        )

        if record["covered_conflict_count"] > 0:
            candidates.append(record)

    candidates.sort(
        key=lambda item: (
            -item["score"],
            -item["conflict_gain"],
            -item["covered_conflict_count"],
            -item["quality_score"],
            item["frame"],
        )
    )

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        candidate["rank"] = rank

    # Diagnostic only: if we naively removed conflicts directly covered by each
    # recommendation, what would a short batch look like? Production must NEVER
    # follow this without recomputing Sparse Validator after every HUMAN.
    remaining = set(conflict_weights)
    greedy_preview: list[dict[str, Any]] = []

    for _ in range(
        max(0, int(cfg.greedy_preview_count))
    ):
        best = None

        for candidate in sorted(adjacency):
            if candidate in humans:
                continue
            if float(
                quality.get(candidate, 0.0)
            ) < float(cfg.quality_floor):
                continue

            record = _candidate_record(
                candidate,
                conflict_weights=conflict_weights,
                target_by_frame=target_by_frame,
                adjacency=adjacency,
                quality=quality,
                human_frames=humans,
                cfg=cfg,
                remaining_conflicts=remaining,
            )

            if not record["covered_conflict_count"]:
                continue

            if (
                best is None
                or (
                    record["score"],
                    record["conflict_gain"],
                    record["covered_conflict_count"],
                    record["quality_score"],
                    -record["frame"],
                )
                > (
                    best["score"],
                    best["conflict_gain"],
                    best["covered_conflict_count"],
                    best["quality_score"],
                    -best["frame"],
                )
            ):
                best = record

        if best is None:
            break

        greedy_preview.append(best)
        remaining -= set(
            best["covered_conflict_frames"]
        )

    raw_disagreement = {
        frame: float(
            target_by_frame[frame]["pair_p90_px"]
        )
        for frame in conflict_weights
        if target_by_frame[frame].get(
            "pair_p90_px"
        ) is not None
    }

    raw_values = np.asarray(
        list(raw_disagreement.values()),
        dtype=np.float64,
    )

    return {
        "status": "ok",
        "version": 1,
        "algorithm": "sparse_active_anchor_selector_v1",
        "policy": {
            "uses_dense": False,
            "uses_transitive_coverage": False,
            "treats_single_support_as_failure": False,
            "conflict_requires_multi_human_support": True,
            "recompute_after_every_real_human": True,
        },
        "config": asdict(cfg),
        "human_frames": sorted(humans),
        "measured_conflict_state_count": len(
            conflict_weights
        ),
        "raw_pair_p90_distribution_px": {
            "median": float(
                np.median(raw_values)
            ),
            "p75": float(
                np.percentile(raw_values, 75)
            ),
            "p90": float(
                np.percentile(raw_values, 90)
            ),
            "p95": float(
                np.percentile(raw_values, 95)
            ),
            "max": float(
                np.max(raw_values)
            ),
        },
        "recommendations": candidates,
        "greedy_preview_only": greedy_preview,
        "remaining_conflict_count_after_preview": len(
            remaining
        ),
    }

"""Sparse CAMERA Validator V1.

Read-only diagnostic over HUMAN->keyframe relocalization candidates.

Purpose
-------
CAMERA Anchor Planner proposes where HUMAN calibration is useful. This module
answers the next question BEFORE Dense:

    "Given the HUMAN anchors we already have, do independent HUMAN-derived
     homographies agree on the representative camera keyframes?"

It does not decide final production thresholds yet. It measures:
- direct support count from distinct HUMAN anchors;
- pairwise field-homography disagreement in target image space;
- a robust medoid HUMAN prediction;
- disagreement between the current score-best relocalization and the medoid;
- weak CAMERA regions from Planner V2;
- exact HUMAN leave-one-out error when ground truth is available.

A HUMAN at the target frame is treated as known truth, not as a state that
needs automatic validation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json
import math

import numpy as np

from supervisor.field_camera_state_projective import (
    evaluate_homography_visible_grid,
)


EPS = 1e-10


@dataclass(frozen=True)
class SparseCameraValidatorConfig:
    grid_x: int = 21
    grid_y: int = 15
    image_margin_px: float = 8.0
    minimum_common_grid_points: int = 8


def _as_h(value: Any) -> np.ndarray | None:
    try:
        h = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None

    if h.shape != (3, 3) or not np.isfinite(h).all():
        return None
    if abs(float(h[2, 2])) < EPS:
        return None

    h = h / float(h[2, 2])

    try:
        condition = float(np.linalg.cond(h))
        determinant = float(np.linalg.det(h))
    except np.linalg.LinAlgError:
        return None

    if (
        not math.isfinite(condition)
        or condition > 1e14
        or not math.isfinite(determinant)
        or abs(determinant) < 1e-12
    ):
        return None

    return h


def _project(
    world_points: np.ndarray,
    h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        world_points,
        dtype=np.float64,
    ).reshape(-1, 2)

    homogeneous = np.column_stack(
        (
            points,
            np.ones(len(points), dtype=np.float64),
        )
    )

    projected_h = homogeneous @ h.T
    denominator = projected_h[:, 2]

    valid = (
        np.isfinite(projected_h).all(axis=1)
        & np.isfinite(denominator)
        & (np.abs(denominator) > EPS)
    )

    projected = np.full(
        (len(points), 2),
        np.nan,
        dtype=np.float64,
    )

    projected[valid] = (
        projected_h[valid, :2]
        / denominator[valid, None]
    )

    valid &= np.isfinite(projected).all(axis=1)

    return projected, valid


def _world_grid(
    pitch_length_m: float,
    pitch_width_m: float,
    cfg: SparseCameraValidatorConfig,
) -> np.ndarray:
    xs = np.linspace(
        -float(pitch_length_m) / 2.0,
        +float(pitch_length_m) / 2.0,
        max(3, int(cfg.grid_x)),
    )
    ys = np.linspace(
        -float(pitch_width_m) / 2.0,
        +float(pitch_width_m) / 2.0,
        max(3, int(cfg.grid_y)),
    )

    return np.asarray(
        [(x, y) for y in ys for x in xs],
        dtype=np.float64,
    )


def compare_target_homographies(
    first_h: Any,
    second_h: Any,
    *,
    image_width: int,
    image_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
    config: SparseCameraValidatorConfig | None = None,
) -> dict[str, Any]:
    """Compare two field->same-target-image homographies symmetrically.

    Only canonical field points that BOTH predictions place inside the target
    image are compared. Therefore neither H is treated as ground truth.
    """
    cfg = config or SparseCameraValidatorConfig()

    first = _as_h(first_h)
    second = _as_h(second_h)

    if first is None or second is None:
        return {"status": "invalid_homography"}

    world = _world_grid(
        pitch_length_m,
        pitch_width_m,
        cfg,
    )

    first_image, first_valid = _project(world, first)
    second_image, second_valid = _project(world, second)

    margin = float(cfg.image_margin_px)

    first_inside = (
        first_valid
        & (first_image[:, 0] >= margin)
        & (first_image[:, 0] <= image_width - margin)
        & (first_image[:, 1] >= margin)
        & (first_image[:, 1] <= image_height - margin)
    )

    second_inside = (
        second_valid
        & (second_image[:, 0] >= margin)
        & (second_image[:, 0] <= image_width - margin)
        & (second_image[:, 1] >= margin)
        & (second_image[:, 1] <= image_height - margin)
    )

    common = first_inside & second_inside
    count = int(np.count_nonzero(common))

    if count < int(cfg.minimum_common_grid_points):
        return {
            "status": "insufficient_common_visibility",
            "common_points": count,
        }

    errors = np.linalg.norm(
        first_image[common] - second_image[common],
        axis=1,
    )

    return {
        "status": "ok",
        "common_points": count,
        "median_px": float(np.median(errors)),
        "p90_px": float(np.percentile(errors, 90)),
        "max_px": float(np.max(errors)),
    }


def _candidate_predictions(
    result: dict[str, Any],
    *,
    exclude_anchor_frame: int | None = None,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []

    for candidate in result.get("candidates", []):
        if not isinstance(candidate, dict):
            continue

        try:
            anchor_frame = int(candidate["anchor_frame"])
        except (KeyError, TypeError, ValueError):
            continue

        if (
            exclude_anchor_frame is not None
            and anchor_frame == int(exclude_anchor_frame)
        ):
            continue

        verification = candidate.get("verification")
        if not isinstance(verification, dict):
            continue

        if verification.get("status") != "ok":
            continue

        h = _as_h(candidate.get("homography"))
        if h is None:
            continue

        score = candidate.get("score")
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = float("-inf")

        predictions.append(
            {
                "anchor_frame": anchor_frame,
                "homography": h,
                "score": score,
                "high_confidence": bool(
                    candidate.get("high_confidence")
                ),
                "verification": verification,
            }
        )

    # One prediction per HUMAN anchor frame.
    by_anchor: dict[int, dict[str, Any]] = {}

    for prediction in predictions:
        frame = int(prediction["anchor_frame"])
        previous = by_anchor.get(frame)

        if (
            previous is None
            or float(prediction["score"])
            > float(previous["score"])
        ):
            by_anchor[frame] = prediction

    return [
        by_anchor[frame]
        for frame in sorted(by_anchor)
    ]


def summarize_target_consensus(
    result: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
    human_frames: set[int] | None = None,
    exclude_anchor_frame: int | None = None,
    config: SparseCameraValidatorConfig | None = None,
) -> dict[str, Any]:
    cfg = config or SparseCameraValidatorConfig()

    try:
        target_frame = int(result["frame"])
    except (KeyError, TypeError, ValueError):
        return {"status": "invalid_target"}

    human_frames = human_frames or set()

    predictions = _candidate_predictions(
        result,
        exclude_anchor_frame=exclude_anchor_frame,
    )

    support_count = len(predictions)

    base = {
        "frame": target_frame,
        "is_exact_human": target_frame in human_frames,
        "support_count": support_count,
        "support_anchor_frames": [
            int(item["anchor_frame"])
            for item in predictions
        ],
    }

    if not predictions:
        return {
            **base,
            "status": "no_support",
        }

    if len(predictions) == 1:
        only = predictions[0]
        return {
            **base,
            "status": "single_support",
            "consensus_anchor_frame": int(
                only["anchor_frame"]
            ),
            "consensus_homography": (
                only["homography"].tolist()
            ),
            "consensus_anchor_score": float(
                only["score"]
            ),
            "comparable_pair_fraction": 0.0,
            "pair_median_px": None,
            "pair_p90_px": None,
            "pair_max_px": None,
            "current_anchor_frame": result.get(
                "anchor_frame"
            ),
            "current_to_consensus_median_px": None,
        }

    n = len(predictions)
    pair_matrix = np.full(
        (n, n),
        np.nan,
        dtype=np.float64,
    )

    pair_values: list[float] = []
    pair_details: list[dict[str, Any]] = []

    for left in range(n):
        pair_matrix[left, left] = 0.0

        for right in range(left + 1, n):
            comparison = compare_target_homographies(
                predictions[left]["homography"],
                predictions[right]["homography"],
                image_width=image_width,
                image_height=image_height,
                pitch_length_m=pitch_length_m,
                pitch_width_m=pitch_width_m,
                config=cfg,
            )

            detail = {
                "left_anchor_frame": int(
                    predictions[left]["anchor_frame"]
                ),
                "right_anchor_frame": int(
                    predictions[right]["anchor_frame"]
                ),
                **comparison,
            }
            pair_details.append(detail)

            if comparison.get("status") != "ok":
                continue

            value = float(comparison["median_px"])
            pair_matrix[left, right] = value
            pair_matrix[right, left] = value
            pair_values.append(value)

    expected_pairs = n * (n - 1) // 2
    comparable_pairs = len(pair_values)

    medoid_rows: list[dict[str, Any]] = []

    for index, prediction in enumerate(predictions):
        row = pair_matrix[index]
        mask = (
            np.isfinite(row)
            & (np.arange(n) != index)
        )
        values = row[mask]

        medoid_rows.append(
            {
                "index": index,
                "anchor_frame": int(
                    prediction["anchor_frame"]
                ),
                "comparable_count": int(len(values)),
                "median_disagreement_px": (
                    float(np.median(values))
                    if len(values)
                    else float("inf")
                ),
                "p90_disagreement_px": (
                    float(np.percentile(values, 90))
                    if len(values)
                    else float("inf")
                ),
                "score": float(prediction["score"]),
            }
        )

    # Prefer the candidate that agrees with the largest number of other HUMAN
    # predictions; among equally comparable candidates choose the medoid.
    medoid_rows.sort(
        key=lambda item: (
            -item["comparable_count"],
            item["median_disagreement_px"],
            item["p90_disagreement_px"],
            -item["score"],
            item["anchor_frame"],
        )
    )

    chosen_row = medoid_rows[0]
    chosen = predictions[int(chosen_row["index"])]

    current_h = _as_h(result.get("homography"))
    current_to_consensus = None

    if current_h is not None:
        comparison = compare_target_homographies(
            current_h,
            chosen["homography"],
            image_width=image_width,
            image_height=image_height,
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
            config=cfg,
        )

        if comparison.get("status") == "ok":
            current_to_consensus = float(
                comparison["median_px"]
            )

    if pair_values:
        pair_median = float(np.median(pair_values))
        pair_p90 = float(
            np.percentile(pair_values, 90)
        )
        pair_max = float(np.max(pair_values))
        status = "multi_support"
    else:
        pair_median = None
        pair_p90 = None
        pair_max = None
        status = "incoherent_visibility"

    return {
        **base,
        "status": status,
        "consensus_anchor_frame": int(
            chosen["anchor_frame"]
        ),
        "consensus_homography": (
            chosen["homography"].tolist()
        ),
        "consensus_anchor_score": float(
            chosen["score"]
        ),
        "consensus_comparable_count": int(
            chosen_row["comparable_count"]
        ),
        "expected_pair_count": expected_pairs,
        "comparable_pair_count": comparable_pairs,
        "comparable_pair_fraction": (
            comparable_pairs / expected_pairs
            if expected_pairs
            else 0.0
        ),
        "pair_median_px": pair_median,
        "pair_p90_px": pair_p90,
        "pair_max_px": pair_max,
        "current_anchor_frame": result.get(
            "anchor_frame"
        ),
        "current_to_consensus_median_px": (
            current_to_consensus
        ),
        "medoid_candidates": medoid_rows,
        "pair_details": pair_details,
    }


def _percentiles(
    values: list[float],
) -> dict[str, float | None]:
    clean = np.asarray(
        [
            float(value)
            for value in values
            if value is not None
            and math.isfinite(float(value))
        ],
        dtype=np.float64,
    )

    if not len(clean):
        return {
            "count": 0,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
        }

    return {
        "count": int(len(clean)),
        "median": float(np.median(clean)),
        "p75": float(np.percentile(clean, 75)),
        "p90": float(np.percentile(clean, 90)),
        "p95": float(np.percentile(clean, 95)),
        "max": float(np.max(clean)),
    }


def _region_map(
    planner_v2: dict[str, Any] | None,
) -> tuple[dict[int, int], dict[int, dict[str, Any]]]:
    frame_to_region: dict[int, int] = {}
    region_info: dict[int, dict[str, Any]] = {}

    if not isinstance(planner_v2, dict):
        return frame_to_region, region_info

    for region in planner_v2.get(
        "recommended_regions",
        [],
    ):
        if not isinstance(region, dict):
            continue

        try:
            region_id = int(
                region.get(
                    "temporal_rank",
                    region.get("region"),
                )
            )
        except (TypeError, ValueError):
            continue

        region_info[region_id] = region

        for frame in region.get("frames", []):
            try:
                frame_to_region[int(frame)] = region_id
            except (TypeError, ValueError):
                continue

    return frame_to_region, region_info


def analyse_sparse_camera_consensus(
    relocalization: dict[str, Any] | str | Path,
    *,
    image_width: int,
    image_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
    human_frames: list[int] | set[int],
    planner_v2: dict[str, Any] | str | Path | None = None,
    config: SparseCameraValidatorConfig | None = None,
) -> dict[str, Any]:
    cfg = config or SparseCameraValidatorConfig()

    if isinstance(relocalization, (str, Path)):
        try:
            relocalization = json.loads(
                Path(relocalization).read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError) as error:
            return {
                "status": "relocalization_read_failed",
                "error": str(error),
            }

    if relocalization.get("status") != "ok":
        return {
            "status": "invalid_relocalization",
            "relocalization_status": (
                relocalization.get("status")
            ),
        }

    if isinstance(planner_v2, (str, Path)):
        try:
            planner_v2 = json.loads(
                Path(planner_v2).read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError):
            planner_v2 = None

    human_set = {
        int(frame)
        for frame in human_frames
    }

    summaries: list[dict[str, Any]] = []

    for result in relocalization.get("results", []):
        if not isinstance(result, dict):
            continue

        summary = summarize_target_consensus(
            result,
            image_width=image_width,
            image_height=image_height,
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
            human_frames=human_set,
            config=cfg,
        )

        if "frame" in summary:
            summaries.append(summary)

    if not summaries:
        return {"status": "no_targets"}

    frame_to_region, region_info = _region_map(
        planner_v2
        if isinstance(planner_v2, dict)
        else None
    )

    for summary in summaries:
        region_id = frame_to_region.get(
            int(summary["frame"])
        )
        summary["planner_region"] = region_id

    automatic = [
        item
        for item in summaries
        if not item.get("is_exact_human")
    ]

    support_counts = [
        int(item.get("support_count") or 0)
        for item in automatic
    ]

    pair_medians = [
        item.get("pair_median_px")
        for item in automatic
        if item.get("pair_median_px") is not None
    ]

    pair_p90s = [
        item.get("pair_p90_px")
        for item in automatic
        if item.get("pair_p90_px") is not None
    ]

    current_diffs = [
        item.get("current_to_consensus_median_px")
        for item in automatic
        if item.get(
            "current_to_consensus_median_px"
        ) is not None
    ]

    region_summaries: list[dict[str, Any]] = []

    for region_id in sorted(region_info):
        members = [
            item
            for item in automatic
            if item.get("planner_region") == region_id
        ]

        if not members:
            continue

        region = region_info[region_id]
        representative = region.get(
            "representative",
            {},
        )

        disagreement = [
            float(item["pair_p90_px"])
            for item in members
            if item.get("pair_p90_px") is not None
        ]

        support = [
            int(item.get("support_count") or 0)
            for item in members
        ]

        single_or_none = sum(
            1
            for item in members
            if int(item.get("support_count") or 0) < 2
        )

        current_vs_consensus = [
            float(
                item[
                    "current_to_consensus_median_px"
                ]
            )
            for item in members
            if item.get(
                "current_to_consensus_median_px"
            ) is not None
        ]

        region_summaries.append(
            {
                "region": region_id,
                "state_count": len(members),
                "representative_frame": (
                    representative.get("frame")
                ),
                "support_median": (
                    float(np.median(support))
                    if support
                    else None
                ),
                "support_min": (
                    int(min(support))
                    if support
                    else None
                ),
                "single_or_no_support_fraction": (
                    single_or_none / len(members)
                ),
                "pair_p90_distribution": _percentiles(
                    disagreement
                ),
                "current_vs_consensus_distribution": (
                    _percentiles(
                        current_vs_consensus
                    )
                ),
            }
        )

    # Diagnostic ranking, not a production threshold:
    # poor support first, then large multi-HUMAN disagreement.
    weak_targets = sorted(
        automatic,
        key=lambda item: (
            int(item.get("support_count") or 0) >= 2,
            -float(
                item.get("pair_p90_px")
                if item.get("pair_p90_px") is not None
                else (
                    1e9
                    if int(
                        item.get("support_count") or 0
                    ) == 0
                    else 0.0
                )
            ),
            int(item["frame"]),
        ),
    )

    weak_regions = sorted(
        region_summaries,
        key=lambda item: (
            -float(
                item[
                    "single_or_no_support_fraction"
                ]
            ),
            -float(
                item["pair_p90_distribution"].get(
                    "p90"
                )
                or 0.0
            ),
            int(item["region"]),
        ),
    )

    return {
        "status": "ok",
        "version": 1,
        "algorithm": "sparse_camera_validator_v1",
        "config": asdict(cfg),
        "human_frames": sorted(human_set),
        "target_count": len(summaries),
        "automatic_target_count": len(automatic),
        "support_count_distribution": {
            str(count): int(
                sum(
                    1
                    for value in support_counts
                    if value == count
                )
            )
            for count in sorted(set(support_counts))
        },
        "pair_median_distribution_px": _percentiles(
            pair_medians
        ),
        "pair_p90_distribution_px": _percentiles(
            pair_p90s
        ),
        "current_vs_consensus_distribution_px": (
            _percentiles(current_diffs)
        ),
        "targets": summaries,
        "region_summaries": region_summaries,
        "weak_targets": weak_targets[:50],
        "weak_regions": weak_regions,
    }


def evaluate_human_leave_one_out(
    relocalization_result: dict[str, Any],
    true_h_by_frame: dict[int, Any],
    *,
    image_width: int,
    image_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
    config: SparseCameraValidatorConfig | None = None,
) -> dict[str, Any]:
    """Evaluate exact HUMAN targets while hiding each target's own HUMAN."""
    cfg = config or SparseCameraValidatorConfig()
    human_frames = set(int(frame) for frame in true_h_by_frame)
    rows: list[dict[str, Any]] = []

    for result in relocalization_result.get(
        "results",
        [],
    ):
        if not isinstance(result, dict):
            continue

        try:
            frame = int(result["frame"])
        except (KeyError, TypeError, ValueError):
            continue

        if frame not in true_h_by_frame:
            continue

        consensus = summarize_target_consensus(
            result,
            image_width=image_width,
            image_height=image_height,
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
            human_frames=human_frames,
            exclude_anchor_frame=frame,
            config=cfg,
        )

        true_h = true_h_by_frame[frame]
        consensus_h = consensus.get(
            "consensus_homography"
        )

        if consensus_h is not None:
            consensus_error = (
                evaluate_homography_visible_grid(
                    consensus_h,
                    true_h,
                    image_width=image_width,
                    image_height=image_height,
                    pitch_length_m=pitch_length_m,
                    pitch_width_m=pitch_width_m,
                )
            )
        else:
            consensus_error = {
                "status": "no_prediction"
            }

        # analyse_anchor_relocalization(exclude_same_frame=True) already picks
        # its normal score-best candidate at top level.
        score_best_h = result.get("homography")

        if score_best_h is not None:
            score_best_error = (
                evaluate_homography_visible_grid(
                    score_best_h,
                    true_h,
                    image_width=image_width,
                    image_height=image_height,
                    pitch_length_m=pitch_length_m,
                    pitch_width_m=pitch_width_m,
                )
            )
        else:
            score_best_error = {
                "status": "no_prediction"
            }

        rows.append(
            {
                "frame": frame,
                "support_count": consensus.get(
                    "support_count"
                ),
                "support_anchor_frames": consensus.get(
                    "support_anchor_frames"
                ),
                "pair_median_px": consensus.get(
                    "pair_median_px"
                ),
                "pair_p90_px": consensus.get(
                    "pair_p90_px"
                ),
                "consensus_anchor_frame": consensus.get(
                    "consensus_anchor_frame"
                ),
                "score_best_anchor_frame": result.get(
                    "anchor_frame"
                ),
                "consensus_visible_grid_error": (
                    consensus_error
                ),
                "score_best_visible_grid_error": (
                    score_best_error
                ),
            }
        )

    valid_consensus = [
        row
        for row in rows
        if row[
            "consensus_visible_grid_error"
        ].get("status") == "ok"
    ]

    valid_score = [
        row
        for row in rows
        if row[
            "score_best_visible_grid_error"
        ].get("status") == "ok"
    ]

    return {
        "status": (
            "ok"
            if rows
            else "no_human_targets"
        ),
        "holdout_count": len(rows),
        "consensus_valid_count": len(valid_consensus),
        "score_best_valid_count": len(valid_score),
        "consensus_median_error_distribution_px": (
            _percentiles(
                [
                    row[
                        "consensus_visible_grid_error"
                    ]["median_px"]
                    for row in valid_consensus
                ]
            )
        ),
        "score_best_median_error_distribution_px": (
            _percentiles(
                [
                    row[
                        "score_best_visible_grid_error"
                    ]["median_px"]
                    for row in valid_score
                ]
            )
        ),
        "per_human": rows,
    }

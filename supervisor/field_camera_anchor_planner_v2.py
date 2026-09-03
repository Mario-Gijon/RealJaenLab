"""CAMERA Anchor Planner V2: camera-region discovery from the V1 sparse graph.

Experimental/read-only analysis.

V1 answered:
    "Which HUMAN nodes directly cover every keyframe?"

That is intentionally conservative and over-selects HUMAN anchors.

V2 asks a different preflight question:
    "Which coherent camera-state regions exist in the sparse direct-overlap graph,
     and which calibratable frame is the best representative of each region?"

Important:
- HUMAN homographies are NOT used.
- Existing HUMAN frame numbers are NOT used.
- Dense / Relocalization / review calibrations are NOT used.
- Direct-state displacement is used only as a CAMERA-REGION affinity signal,
  NOT as a safety/acceptance threshold for homography transfer.
- Region membership may exploit graph structure, but later Sparse Validator must
  independently verify whether HUMAN-derived field homographies agree.
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
class CameraAnchorPlannerV2Config:
    minimum_regions: int = 2
    maximum_regions: int = 16

    # Local graph scale is learned from each node's closest verified edges.
    local_neighbors_for_scale: int = 3
    local_scale_percentile: float = 75.0

    # Edge affinity combines direct camera-state closeness and already-verified
    # projective quality. These are soft weights, not binary acceptance gates.
    reprojection_soft_scale_px: float = 1.5

    # Choose the smallest region count close to the best weighted modularity,
    # avoiding needless fragmentation when several resolutions are equivalent.
    modularity_plateau_tolerance: float = 0.012

    # Region representative = camera-state centrality + HUMAN calibratability
    # + preference for nodes whose graph support stays inside the region.
    representative_centrality_weight: float = 0.60
    representative_quality_weight: float = 0.25
    representative_internal_fraction_weight: float = 0.15

    alternates_per_region: int = 5


def _load_v1(value: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    path = Path(value)
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _build_index(
    v1: dict[str, Any],
) -> tuple[list[int], dict[int, dict[str, Any]], dict[int, int]]:
    states_by_frame: dict[int, dict[str, Any]] = {}

    for state in v1.get("states", []):
        if not isinstance(state, dict):
            continue
        try:
            frame = int(state["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        states_by_frame[frame] = state

    frames = sorted(states_by_frame)
    index_by_frame = {
        frame: index
        for index, frame in enumerate(frames)
    }
    return frames, states_by_frame, index_by_frame


def _usable_edge_rows(
    v1: dict[str, Any],
    index_by_frame: dict[int, int],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []

    for edge in v1.get("edges", []):
        if not isinstance(edge, dict) or not edge.get("usable"):
            continue

        try:
            left = int(edge["left_frame"])
            right = int(edge["right_frame"])
        except (KeyError, TypeError, ValueError):
            continue

        if left not in index_by_frame or right not in index_by_frame:
            continue

        verification = edge.get("verification")
        if not isinstance(verification, dict):
            continue

        distance = verification.get("direct_state_distance")
        if not isinstance(distance, dict):
            continue

        median = _valid_number(distance.get("median_px"))
        p90 = _valid_number(distance.get("p90_px"))
        ratio = _valid_number(verification.get("inlier_ratio"))
        reprojection = _valid_number(
            verification.get("median_reprojection_px")
        )

        if (
            median is None
            or p90 is None
            or ratio is None
            or reprojection is None
        ):
            continue

        rows.append(
            {
                "left": left,
                "right": right,
                "median_px": max(0.0, median),
                "p90_px": max(0.0, p90),
                "inlier_ratio": min(max(ratio, 0.0), 1.0),
                "reprojection_px": max(0.0, reprojection),
            }
        )

    return rows


def _learn_graph_scales(
    frames: list[int],
    edge_rows: list[dict[str, float | int]],
    cfg: CameraAnchorPlannerV2Config,
) -> dict[str, float]:
    incident: dict[int, list[tuple[float, float]]] = {
        frame: [] for frame in frames
    }

    for edge in edge_rows:
        left = int(edge["left"])
        right = int(edge["right"])
        pair = (
            float(edge["median_px"]),
            float(edge["p90_px"]),
        )
        incident[left].append(pair)
        incident[right].append(pair)

    local_medians: list[float] = []
    local_p90s: list[float] = []

    k = max(1, int(cfg.local_neighbors_for_scale))

    for frame in frames:
        values = sorted(
            incident.get(frame, []),
            key=lambda item: (item[0], item[1]),
        )
        for median, p90 in values[:k]:
            local_medians.append(median)
            local_p90s.append(p90)

    if not local_medians:
        return {
            "median_px": 100.0,
            "p90_px": 180.0,
        }

    percentile = min(
        max(float(cfg.local_scale_percentile), 1.0),
        99.0,
    )

    median_scale = float(
        np.percentile(local_medians, percentile)
    )
    p90_scale = float(
        np.percentile(local_p90s, percentile)
    )

    # Numerical floors only. They are not CAMERA thresholds.
    return {
        "median_px": max(median_scale, 10.0),
        "p90_px": max(p90_scale, 20.0),
    }


def _edge_weight(
    edge: dict[str, float | int],
    scales: dict[str, float],
    cfg: CameraAnchorPlannerV2Config,
) -> float:
    median = float(edge["median_px"])
    p90 = float(edge["p90_px"])
    ratio = float(edge["inlier_ratio"])
    reproj = float(edge["reprojection_px"])

    median_scale = max(scales["median_px"], EPS)
    p90_scale = max(scales["p90_px"], EPS)

    # Soft state affinity. No edge is declared unsafe because of displacement.
    state_affinity = math.exp(
        -0.5 * (median / median_scale) ** 2
        -0.25 * (p90 / p90_scale) ** 2
    )

    # All rows already passed the existing robust verifier. Here we only make
    # stronger verified edges matter somewhat more inside region discovery.
    projective_quality = (
        (0.35 + 0.65 * ratio)
        * math.exp(
            -reproj
            / max(cfg.reprojection_soft_scale_px, EPS)
        )
    )

    return max(
        float(state_affinity * projective_quality),
        1e-9,
    )


def _weighted_adjacency(
    frames: list[int],
    index_by_frame: dict[int, int],
    edge_rows: list[dict[str, float | int]],
    scales: dict[str, float],
    cfg: CameraAnchorPlannerV2Config,
) -> np.ndarray:
    n = len(frames)
    weights = np.zeros((n, n), dtype=np.float64)

    for edge in edge_rows:
        left = index_by_frame[int(edge["left"])]
        right = index_by_frame[int(edge["right"])]
        weight = _edge_weight(edge, scales, cfg)

        # If the sparse proposal happened to produce the pair more than once in
        # future versions, preserve the strongest direct verification.
        if weight > weights[left, right]:
            weights[left, right] = weight
            weights[right, left] = weight

    return weights


def _spectral_embedding(
    weights: np.ndarray,
    maximum_regions: int,
) -> tuple[np.ndarray, np.ndarray]:
    degree = weights.sum(axis=1)
    inv_sqrt = np.zeros_like(degree)
    valid = degree > EPS
    inv_sqrt[valid] = 1.0 / np.sqrt(degree[valid])

    normalized = (
        inv_sqrt[:, None]
        * weights
        * inv_sqrt[None, :]
    )

    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    dimensions = min(
        max(1, maximum_regions),
        eigenvectors.shape[1],
    )
    embedding = eigenvectors[:, :dimensions]

    return eigenvalues, embedding


def _deterministic_kmeans(
    points: np.ndarray,
    k: int,
    *,
    seed_scores: np.ndarray | None = None,
    maximum_iterations: int = 100,
) -> np.ndarray:
    n = len(points)
    if k <= 1 or n <= 1:
        return np.zeros(n, dtype=np.int32)
    if k >= n:
        return np.arange(n, dtype=np.int32)

    work = np.asarray(points, dtype=np.float64)

    # Row-normalization is standard for normalized spectral clustering.
    norms = np.linalg.norm(work, axis=1, keepdims=True)
    work = work / np.maximum(norms, EPS)

    if seed_scores is None:
        first = 0
    else:
        first = int(np.argmax(seed_scores))

    seeds = [first]
    minimum_squared = np.sum(
        (work - work[first]) ** 2,
        axis=1,
    )

    while len(seeds) < k:
        for seed in seeds:
            minimum_squared[seed] = -1.0
        next_seed = int(np.argmax(minimum_squared))
        seeds.append(next_seed)
        candidate_squared = np.sum(
            (work - work[next_seed]) ** 2,
            axis=1,
        )
        minimum_squared = np.minimum(
            minimum_squared,
            candidate_squared,
        )

    centers = work[np.asarray(seeds)].copy()
    labels = np.zeros(n, dtype=np.int32)

    for _ in range(maximum_iterations):
        squared = np.sum(
            (work[:, None, :] - centers[None, :, :]) ** 2,
            axis=2,
        )
        new_labels = np.argmin(squared, axis=1).astype(np.int32)

        # Deterministically repair empty clusters with the point farthest from
        # its current assigned center.
        counts = np.bincount(new_labels, minlength=k)
        if np.any(counts == 0):
            assigned_distance = squared[
                np.arange(n),
                new_labels,
            ]
            order = np.argsort(assigned_distance)[::-1]
            used: set[int] = set()

            for empty in np.flatnonzero(counts == 0):
                for index in order:
                    index = int(index)
                    if index in used:
                        continue
                    donor = int(new_labels[index])
                    if counts[donor] <= 1:
                        continue
                    counts[donor] -= 1
                    new_labels[index] = int(empty)
                    counts[int(empty)] += 1
                    used.add(index)
                    break

        if np.array_equal(new_labels, labels):
            labels = new_labels
            break

        labels = new_labels

        for cluster in range(k):
            members = work[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)

    return labels


def _weighted_modularity(
    weights: np.ndarray,
    labels: np.ndarray,
) -> float:
    total = float(weights.sum())
    if total <= EPS:
        return 0.0

    degree = weights.sum(axis=1)
    value = 0.0

    for cluster in np.unique(labels):
        mask = labels == cluster
        internal = float(
            weights[np.ix_(mask, mask)].sum()
        )
        degree_sum = float(degree[mask].sum())

        value += (
            internal / total
            - (degree_sum / total) ** 2
        )

    return float(value)


def _normalized_cut_score(
    weights: np.ndarray,
    labels: np.ndarray,
) -> float:
    degree = weights.sum(axis=1)
    total = 0.0
    count = 0

    for cluster in np.unique(labels):
        mask = labels == cluster
        volume = float(degree[mask].sum())
        if volume <= EPS:
            continue

        cut = float(
            weights[np.ix_(mask, ~mask)].sum()
        )
        total += cut / volume
        count += 1

    return float(total / max(count, 1))


def _region_representatives(
    frames: list[int],
    states_by_frame: dict[int, dict[str, Any]],
    weights: np.ndarray,
    labels: np.ndarray,
    cfg: CameraAnchorPlannerV2Config,
) -> list[dict[str, Any]]:
    degree = weights.sum(axis=1)
    regions: list[dict[str, Any]] = []

    for cluster in sorted(int(value) for value in np.unique(labels)):
        members = np.flatnonzero(labels == cluster)
        member_frames = [frames[int(i)] for i in members]

        internal_strength = np.asarray(
            [
                float(weights[int(i), members].sum())
                for i in members
            ],
            dtype=np.float64,
        )

        maximum_internal = max(
            float(internal_strength.max(initial=0.0)),
            EPS,
        )

        ranked: list[dict[str, Any]] = []

        for local_index, node_index_raw in enumerate(members):
            node_index = int(node_index_raw)
            frame = frames[node_index]

            centrality = (
                float(internal_strength[local_index])
                / maximum_internal
            )

            total_strength = float(degree[node_index])
            internal_fraction = (
                float(internal_strength[local_index])
                / max(total_strength, EPS)
            )

            quality = float(
                states_by_frame[frame].get(
                    "quality_score",
                    0.0,
                )
            )

            score = (
                cfg.representative_centrality_weight * centrality
                + cfg.representative_quality_weight * quality
                + cfg.representative_internal_fraction_weight
                * internal_fraction
            )

            ranked.append(
                {
                    "frame": int(frame),
                    "score": float(score),
                    "centrality": float(centrality),
                    "quality_score": float(quality),
                    "internal_fraction": float(internal_fraction),
                    "graph_strength": float(total_strength),
                }
            )

        ranked.sort(
            key=lambda item: (
                -item["score"],
                -item["centrality"],
                -item["quality_score"],
                item["frame"],
            )
        )

        region_mask = labels == cluster
        internal_weight = float(
            weights[np.ix_(region_mask, region_mask)].sum()
        )
        external_weight = float(
            weights[np.ix_(region_mask, ~region_mask)].sum()
        )

        regions.append(
            {
                "region": int(cluster),
                "state_count": int(len(members)),
                "frame_min": int(min(member_frames)),
                "frame_max": int(max(member_frames)),
                "frame_median": float(
                    np.median(member_frames)
                ),
                "internal_weight": internal_weight,
                "external_weight": external_weight,
                "cohesion_fraction": (
                    internal_weight
                    / max(
                        internal_weight + external_weight,
                        EPS,
                    )
                ),
                "representative": ranked[0],
                "alternates": ranked[
                    : max(
                        1,
                        int(cfg.alternates_per_region),
                    )
                ],
                "frames": sorted(
                    int(frame)
                    for frame in member_frames
                ),
            }
        )

    # Stable human-facing order by first appearance in the video, while region
    # IDs remain available for diagnostics.
    regions.sort(
        key=lambda item: (
            item["frame_median"],
            item["frame_min"],
        )
    )

    for rank, region in enumerate(regions, start=1):
        region["temporal_rank"] = rank

    return regions


def _partition_summary(
    frames: list[int],
    states_by_frame: dict[int, dict[str, Any]],
    weights: np.ndarray,
    embedding: np.ndarray,
    k: int,
    cfg: CameraAnchorPlannerV2Config,
) -> dict[str, Any]:
    degree = weights.sum(axis=1)
    labels = _deterministic_kmeans(
        embedding[:, :k],
        k,
        seed_scores=degree,
    )

    regions = _region_representatives(
        frames,
        states_by_frame,
        weights,
        labels,
        cfg,
    )

    return {
        "region_count": int(k),
        "modularity": _weighted_modularity(
            weights,
            labels,
        ),
        "normalized_cut": _normalized_cut_score(
            weights,
            labels,
        ),
        "representative_frames": [
            int(region["representative"]["frame"])
            for region in regions
        ],
        "region_sizes": [
            int(region["state_count"])
            for region in regions
        ],
        "labels": labels.tolist(),
        "regions": regions,
    }


def analyse_camera_anchor_planner_v2(
    v1_result: dict[str, Any] | str | Path,
    *,
    config: CameraAnchorPlannerV2Config | None = None,
) -> dict[str, Any]:
    cfg = config or CameraAnchorPlannerV2Config()

    try:
        v1 = _load_v1(v1_result)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "status": "v1_read_failed",
            "error": str(error),
        }

    if v1.get("status") != "ok":
        return {
            "status": "invalid_v1",
            "v1_status": v1.get("status"),
        }

    frames, states_by_frame, index_by_frame = _build_index(v1)

    if len(frames) < 2:
        return {
            "status": "insufficient_states",
            "state_count": len(frames),
        }

    edge_rows = _usable_edge_rows(
        v1,
        index_by_frame,
    )

    if not edge_rows:
        return {
            "status": "no_verified_edges",
        }

    scales = _learn_graph_scales(
        frames,
        edge_rows,
        cfg,
    )

    weights = _weighted_adjacency(
        frames,
        index_by_frame,
        edge_rows,
        scales,
        cfg,
    )

    degree = weights.sum(axis=1)
    isolated = [
        int(frames[index])
        for index in np.flatnonzero(degree <= EPS)
    ]

    maximum_regions = min(
        max(
            int(cfg.minimum_regions),
            int(cfg.maximum_regions),
        ),
        len(frames) - 1,
    )
    minimum_regions = min(
        max(2, int(cfg.minimum_regions)),
        maximum_regions,
    )

    eigenvalues, embedding = _spectral_embedding(
        weights,
        maximum_regions,
    )

    partitions: list[dict[str, Any]] = []

    for k in range(
        minimum_regions,
        maximum_regions + 1,
    ):
        partitions.append(
            _partition_summary(
                frames,
                states_by_frame,
                weights,
                embedding,
                k,
                cfg,
            )
        )

    best_modularity = max(
        partition["modularity"]
        for partition in partitions
    )

    tolerance = max(
        0.0,
        float(cfg.modularity_plateau_tolerance),
    )

    plateau = [
        partition
        for partition in partitions
        if partition["modularity"]
        >= best_modularity - tolerance
    ]

    recommended = min(
        plateau,
        key=lambda partition: (
            partition["region_count"],
            partition["normalized_cut"],
        ),
    )

    # Eigengaps are diagnostics only. We do not force k from a single eigengap.
    eigengaps = []
    maximum_gap_index = min(
        maximum_regions,
        len(eigenvalues) - 1,
    )

    for k in range(1, maximum_gap_index + 1):
        gap = float(
            eigenvalues[k - 1] - eigenvalues[k]
        )
        eigengaps.append(
            {
                "after_components": int(k),
                "gap": gap,
                "eigenvalue_before": float(
                    eigenvalues[k - 1]
                ),
                "eigenvalue_after": float(
                    eigenvalues[k]
                ),
            }
        )

    eigengaps.sort(
        key=lambda item: (
            -item["gap"],
            item["after_components"],
        )
    )

    compact_partitions = [
        {
            key: value
            for key, value in partition.items()
            if key not in {"labels", "regions"}
        }
        for partition in partitions
    ]

    return {
        "status": "ok",
        "version": 2,
        "algorithm": "camera_anchor_planner_v2_regions",
        "policy": {
            "uses_human_homographies": False,
            "uses_human_frame_numbers": False,
            "uses_dense": False,
            "uses_relocalization_outputs": False,
            "uses_review_calibrations": False,
            "reuses_v1_verified_sparse_graph": True,
            "direct_state_distance_is_affinity_not_safety_gate": True,
            "region_plan_requires_sparse_validation_after_human": True,
        },
        "config": asdict(cfg),
        "state_count": len(frames),
        "verified_edge_count": len(edge_rows),
        "isolated_frames": isolated,
        "learned_graph_scales": scales,
        "top_eigengaps": eigengaps[:10],
        "partition_sweep": compact_partitions,
        "best_modularity": float(best_modularity),
        "recommended_region_count": int(
            recommended["region_count"]
        ),
        "recommended_modularity": float(
            recommended["modularity"]
        ),
        "recommended_normalized_cut": float(
            recommended["normalized_cut"]
        ),
        "recommended_representative_frames": [
            int(frame)
            for frame in recommended[
                "representative_frames"
            ]
        ],
        "recommended_regions": recommended["regions"],
    }

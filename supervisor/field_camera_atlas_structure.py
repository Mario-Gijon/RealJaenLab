"""Structural Camera Atlas V2 analysis for RealJaen FIELD.

This module answers a narrower question than a tracker:

    Do the strong HUMAN camera anchors form a coherent, reusable camera-state
    space independent of time?

It consumes an already-built Camera Atlas V1 and never reads propagated or
refined data.  V2 measures:

- pairwise camera distances from canonical pitch projections;
- nearest camera-state neighbours, including temporally distant recurrence;
- correlation between temporal distance and camera-state distance;
- k-NN graph connectivity;
- a 2-D Isomap-like embedding for debug visualisation;
- leave-one-anchor-out LOCAL RECONSTRUCTABILITY from nearest camera states.

The last metric is intentionally not called prediction: the held-out target is
used to identify its nearest states in camera space.  It measures atlas density
and local smoothness, not what a production tracker could know before observing
the target frame.

No video decoding, optical flow, line detection, neural learning or review
mutation lives here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from supervisor.field_visibility import compute_visible_pitch_footprint


_SIGNATURE_WORLD = np.asarray(
    [
        (-52.5, -34.0), (0.0, -34.0), (52.5, -34.0),
        (-52.5, -20.16), (-36.0, -20.16), (0.0, -20.16),
        (36.0, -20.16), (52.5, -20.16),
        (-52.5, 0.0), (-41.5, 0.0), (-36.0, 0.0),
        (0.0, 0.0),
        (36.0, 0.0), (41.5, 0.0), (52.5, 0.0),
        (-52.5, 20.16), (-36.0, 20.16), (0.0, 20.16),
        (36.0, 20.16), (52.5, 20.16),
        (-52.5, 34.0), (0.0, 34.0), (52.5, 34.0),
        (-47.0, -9.16), (-47.0, 9.16),
        (47.0, -9.16), (47.0, 9.16),
        (0.0, -9.15), (0.0, 9.15),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class CameraAtlasStructureConfig:
    nearest_neighbors: int = 4
    reconstruction_neighbors: int = 4
    far_time_gap_frames: int = 90
    graph_initial_k: int = 2
    graph_max_k: int = 6
    reconstruction_ridge: float = 1e-4
    feature_clip_normalised: float = 5.0


def _as_h(matrix: Any) -> np.ndarray | None:
    try:
        h = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if h.shape != (3, 3) or not np.isfinite(h).all():
        return None
    scale = float(h[2, 2])
    if abs(scale) < 1e-12:
        return None
    h = h / scale
    try:
        if abs(float(np.linalg.det(h))) < 1e-12:
            return None
    except np.linalg.LinAlgError:
        return None
    return h


def _project_points(points_xy: np.ndarray, h: np.ndarray):
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    q = homogeneous @ h.T
    denominator = q[:, 2]
    valid = (
        np.isfinite(q).all(axis=1)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-10)
    )
    projected = np.full((len(points), 2), np.nan, dtype=np.float64)
    projected[valid] = q[valid, :2] / denominator[valid, None]
    valid &= np.isfinite(projected).all(axis=1)
    return projected, valid


def _signature_feature(
    h: np.ndarray,
    image_width: int,
    image_height: int,
    clip: float,
):
    projected, valid = _project_points(_SIGNATURE_WORLD, h)
    if not np.all(valid):
        return None, None

    normalised = projected.copy()
    normalised[:, 0] /= max(float(image_width), 1.0)
    normalised[:, 1] /= max(float(image_height), 1.0)
    normalised = np.clip(normalised, -clip, clip)
    return normalised.reshape(-1), projected


def _pair_distance_px(first: np.ndarray, second: np.ndarray):
    distances = np.linalg.norm(first - second, axis=1)
    return {
        "median_px": float(np.median(distances)),
        "p90_px": float(np.percentile(distances, 90)),
        "max_px": float(np.max(distances)),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        rank = 0.5 * (index + end - 1) + 1.0
        ranks[order[index:end]] = rank
        index = end
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 3 or len(second) != len(first):
        return None
    a = _rankdata(first)
    b = _rankdata(second)
    a -= np.mean(a)
    b -= np.mean(b)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-12:
        return None
    return float(np.dot(a, b) / denominator)


def _normalise_points_for_dlt(points: np.ndarray):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 4 or not np.isfinite(points).all():
        return None
    centroid = np.mean(points, axis=0)
    shifted = points - centroid
    mean_distance = float(np.mean(np.linalg.norm(shifted, axis=1)))
    if mean_distance < 1e-12:
        return None
    scale = math.sqrt(2.0) / mean_distance
    transform = np.asarray(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    homogeneous = np.column_stack((points, np.ones(len(points))))
    normalised_h = homogeneous @ transform.T
    return normalised_h[:, :2], transform


def _solve_homography_dlt(world_points: np.ndarray, image_points: np.ndarray):
    world_result = _normalise_points_for_dlt(world_points)
    image_result = _normalise_points_for_dlt(image_points)
    if world_result is None or image_result is None:
        return None

    world_norm, world_transform = world_result
    image_norm, image_transform = image_result
    rows = []

    for (x, y), (u, v) in zip(world_norm, image_norm):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u*x, u*y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v*x, v*y, v])

    try:
        _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
        normalised_h = vt[-1].reshape(3, 3)
        h = np.linalg.inv(image_transform) @ normalised_h @ world_transform
    except np.linalg.LinAlgError:
        return None

    return _as_h(h)


def _affine_reconstruction_weights(
    neighbor_features: np.ndarray,
    target_feature: np.ndarray,
    ridge: float,
) -> np.ndarray | None:
    """Ridge local-linear reconstruction with weights constrained to sum to 1."""
    neighbors = np.asarray(neighbor_features, dtype=np.float64)
    target = np.asarray(target_feature, dtype=np.float64).reshape(-1)
    if neighbors.ndim != 2 or len(neighbors) < 2:
        return None

    gram = neighbors @ neighbors.T
    rhs = neighbors @ target
    regularised = gram + float(ridge) * np.eye(len(neighbors))

    ones = np.ones((len(neighbors), 1), dtype=np.float64)
    kkt = np.block(
        [
            [regularised, ones],
            [ones.T, np.zeros((1, 1), dtype=np.float64)],
        ]
    )
    target_rhs = np.concatenate((rhs, np.asarray([1.0])))

    try:
        solution = np.linalg.solve(kkt, target_rhs)
    except np.linalg.LinAlgError:
        try:
            solution = np.linalg.lstsq(kkt, target_rhs, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None

    weights = solution[:-1]
    return weights if np.isfinite(weights).all() else None


def _connected_components(adjacency: np.ndarray) -> list[list[int]]:
    n = len(adjacency)
    seen = np.zeros(n, dtype=bool)
    components = []

    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = np.flatnonzero(np.isfinite(adjacency[node]))
            for neighbor in neighbors:
                neighbor = int(neighbor)
                if neighbor == node or seen[neighbor]:
                    continue
                seen[neighbor] = True
                stack.append(neighbor)
        components.append(sorted(component))

    return components


def _knn_graph(distance_matrix: np.ndarray, initial_k: int, max_k: int):
    n = len(distance_matrix)
    if n < 2:
        return None, [], 0

    max_allowed = min(max(1, int(max_k)), n - 1)
    start = min(max(1, int(initial_k)), max_allowed)

    for k in range(start, max_allowed + 1):
        adjacency = np.full((n, n), np.inf, dtype=np.float64)
        np.fill_diagonal(adjacency, 0.0)

        for index in range(n):
            neighbors = np.argsort(distance_matrix[index])
            neighbors = [int(j) for j in neighbors if int(j) != index][:k]
            for neighbor in neighbors:
                distance = float(distance_matrix[index, neighbor])
                adjacency[index, neighbor] = min(adjacency[index, neighbor], distance)
                adjacency[neighbor, index] = min(adjacency[neighbor, index], distance)

        components = _connected_components(adjacency)
        if len(components) == 1 or k == max_allowed:
            return adjacency, components, k

    return None, [], 0


def _all_pairs_shortest_paths(adjacency: np.ndarray) -> np.ndarray:
    distances = np.asarray(adjacency, dtype=np.float64).copy()
    n = len(distances)
    for k in range(n):
        distances = np.minimum(
            distances,
            distances[:, [k]] + distances[[k], :],
        )
    return distances


def _classical_mds(distance_matrix: np.ndarray, dimensions: int = 2):
    distances = np.asarray(distance_matrix, dtype=np.float64)
    n = len(distances)
    if n < 2 or not np.isfinite(distances).all():
        return np.zeros((n, dimensions)), []

    squared = distances ** 2
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ squared @ centering

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    positive = eigenvalues > 1e-12
    positive_values = eigenvalues[positive]
    positive_vectors = eigenvectors[:, positive]

    count = min(dimensions, len(positive_values))
    coordinates = np.zeros((n, dimensions), dtype=np.float64)
    if count:
        coordinates[:, :count] = (
            positive_vectors[:, :count]
            * np.sqrt(positive_values[:count])[None, :]
        )

    return coordinates, positive_values.tolist()


def _descriptor(
    anchor: dict[str, Any],
    h: np.ndarray,
    image_width: int,
    image_height: int,
):
    centre, valid = _project_points(np.asarray([[0.0, 0.0]]), h)
    centre_norm = None
    if valid[0]:
        centre_norm = [
            float(centre[0, 0] / max(float(image_width), 1.0)),
            float(centre[0, 1] / max(float(image_height), 1.0)),
        ]

    support, support_valid = _project_points(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        h,
    )
    zoom_proxy = None
    if np.all(support_valid):
        sx = float(np.linalg.norm(support[1] - support[0]))
        sy = float(np.linalg.norm(support[2] - support[0]))
        if sx > 0.0 and sy > 0.0:
            zoom_proxy = math.sqrt(sx * sy)

    try:
        horizon = np.linalg.inv(h).T @ np.asarray([0.0, 0.0, 1.0])
        norm = math.hypot(float(horizon[0]), float(horizon[1]))
        if norm > 1e-12:
            horizon = horizon / norm
        else:
            horizon = None
    except np.linalg.LinAlgError:
        horizon = None

    horizon_y_center_norm = None
    horizon_slope = None
    if horizon is not None and abs(float(horizon[1])) > 1e-10:
        image_x = 0.5 * image_width
        horizon_y = -(
            float(horizon[0]) * image_x + float(horizon[2])
        ) / float(horizon[1])
        horizon_y_center_norm = float(horizon_y / image_height)
        horizon_slope = float(-horizon[0] / horizon[1])

    visibility = anchor.get("visibility")
    if not isinstance(visibility, dict) or visibility.get("status") != "ok":
        visibility = compute_visible_pitch_footprint(
            h,
            image_width,
            image_height,
        )

    world_bbox = visibility.get("world_bbox", {}) if isinstance(visibility, dict) else {}
    visible_centroid = visibility.get("world_centroid") if isinstance(visibility, dict) else None

    return {
        "frame": int(anchor["frame"]),
        "field_center_image_norm": centre_norm,
        "log_zoom_proxy": (
            float(math.log(zoom_proxy))
            if zoom_proxy is not None and zoom_proxy > 0.0
            else None
        ),
        "horizon_y_center_norm": horizon_y_center_norm,
        "horizon_slope": horizon_slope,
        "visible_fraction": (
            visibility.get("visible_fraction")
            if isinstance(visibility, dict)
            else None
        ),
        "visible_world_centroid": visible_centroid,
        "visible_world_bbox": world_bbox,
    }


def analyze_camera_atlas_structure(
    atlas: dict[str, Any],
    image_width: int,
    image_height: int,
    *,
    config: CameraAtlasStructureConfig | None = None,
) -> dict[str, Any]:
    """Analyze state-space structure of strong anchors already selected by V1."""
    cfg = config or CameraAtlasStructureConfig()

    try:
        width = int(image_width)
        height = int(image_height)
    except (TypeError, ValueError):
        width = 0
        height = 0

    if width <= 0 or height <= 0:
        return {
            "status": "invalid_image_size",
            "config": asdict(cfg),
        }

    anchors_input = atlas.get("anchors", []) if isinstance(atlas, dict) else []
    if not isinstance(anchors_input, list):
        anchors_input = []

    anchors = []
    for anchor in anchors_input:
        if not isinstance(anchor, dict):
            continue
        h = _as_h(anchor.get("homography"))
        try:
            frame = int(anchor.get("frame"))
        except (TypeError, ValueError):
            continue
        if h is None:
            continue

        feature, projected = _signature_feature(
            h,
            width,
            height,
            cfg.feature_clip_normalised,
        )
        if feature is None:
            continue

        anchors.append({
            "frame": frame,
            "homography": h,
            "feature": feature,
            "projected": projected,
            "source": anchor,
            "descriptor": _descriptor(anchor, h, width, height),
        })

    anchors.sort(key=lambda item: item["frame"])
    n = len(anchors)

    if n < 2:
        return {
            "status": "not_enough_anchors",
            "anchor_count": n,
            "descriptors": [item["descriptor"] for item in anchors],
            "config": asdict(cfg),
        }

    distance_matrix = np.zeros((n, n), dtype=np.float64)
    pairwise_records = []

    for i in range(n):
        for j in range(i + 1, n):
            distance = _pair_distance_px(
                anchors[i]["projected"],
                anchors[j]["projected"],
            )
            median = distance["median_px"]
            distance_matrix[i, j] = median
            distance_matrix[j, i] = median
            pairwise_records.append({
                "frame_a": anchors[i]["frame"],
                "frame_b": anchors[j]["frame"],
                "frame_gap": abs(
                    anchors[j]["frame"] - anchors[i]["frame"]
                ),
                **distance,
            })

    k_neighbors = min(
        max(1, int(cfg.nearest_neighbors)),
        n - 1,
    )

    nearest_neighbors = []
    far_neighbor_distances = []

    for i, anchor in enumerate(anchors):
        order = [
            int(j)
            for j in np.argsort(distance_matrix[i])
            if int(j) != i
        ]
        neighbors = []
        for j in order[:k_neighbors]:
            neighbors.append({
                "frame": anchors[j]["frame"],
                "frame_gap": abs(
                    anchors[j]["frame"] - anchor["frame"]
                ),
                "median_px": float(distance_matrix[i, j]),
            })

        far_candidates = [
            j for j in order
            if abs(anchors[j]["frame"] - anchor["frame"])
            >= cfg.far_time_gap_frames
        ]
        far_neighbor = None
        if far_candidates:
            j = far_candidates[0]
            far_neighbor = {
                "frame": anchors[j]["frame"],
                "frame_gap": abs(
                    anchors[j]["frame"] - anchor["frame"]
                ),
                "median_px": float(distance_matrix[i, j]),
            }
            far_neighbor_distances.append(float(distance_matrix[i, j]))

        nearest_neighbors.append({
            "frame": anchor["frame"],
            "neighbors": neighbors,
            "nearest_far_time_neighbor": far_neighbor,
        })

    time_gaps = np.asarray(
        [record["frame_gap"] for record in pairwise_records],
        dtype=np.float64,
    )
    camera_distances = np.asarray(
        [record["median_px"] for record in pairwise_records],
        dtype=np.float64,
    )
    time_camera_spearman = _spearman(time_gaps, camera_distances)

    adjacency, components, graph_k = _knn_graph(
        distance_matrix,
        cfg.graph_initial_k,
        cfg.graph_max_k,
    )

    embedding = np.zeros((n, 2), dtype=np.float64)
    embedding_eigenvalues = []
    geodesic_status = "unavailable"

    if adjacency is not None:
        geodesic = _all_pairs_shortest_paths(adjacency)
        if np.isfinite(geodesic).all():
            embedding, embedding_eigenvalues = _classical_mds(
                geodesic,
                dimensions=2,
            )
            geodesic_status = "ok"
        else:
            geodesic_status = "graph_disconnected"

    embedding_records = [
        {
            "frame": anchors[index]["frame"],
            "x": float(embedding[index, 0]),
            "y": float(embedding[index, 1]),
        }
        for index in range(n)
    ]

    reconstruction_k = min(
        max(2, int(cfg.reconstruction_neighbors)),
        n - 1,
    )
    reconstruction_records = []

    for target_index, target in enumerate(anchors):
        order = [
            int(j)
            for j in np.argsort(distance_matrix[target_index])
            if int(j) != target_index
        ]
        neighbor_indices = order[:reconstruction_k]
        neighbor_features = np.vstack(
            [anchors[j]["feature"] for j in neighbor_indices]
        )

        weights = _affine_reconstruction_weights(
            neighbor_features,
            target["feature"],
            cfg.reconstruction_ridge,
        )

        if weights is None:
            reconstruction_records.append({
                "frame": target["frame"],
                "status": "weight_solve_failed",
            })
            continue

        reconstructed_feature = weights @ neighbor_features
        reconstructed_normalised = reconstructed_feature.reshape(-1, 2)
        reconstructed_image = reconstructed_normalised.copy()
        reconstructed_image[:, 0] *= width
        reconstructed_image[:, 1] *= height

        reconstructed_h = _solve_homography_dlt(
            _SIGNATURE_WORLD,
            reconstructed_image,
        )

        if reconstructed_h is None:
            reconstruction_records.append({
                "frame": target["frame"],
                "status": "homography_solve_failed",
                "neighbor_frames": [
                    anchors[j]["frame"] for j in neighbor_indices
                ],
                "weights": weights.tolist(),
            })
            continue

        reconstructed_projection, valid = _project_points(
            _SIGNATURE_WORLD,
            reconstructed_h,
        )

        if not np.all(valid):
            reconstruction_records.append({
                "frame": target["frame"],
                "status": "projection_failed",
            })
            continue

        error = _pair_distance_px(
            reconstructed_projection,
            target["projected"],
        )

        reconstruction_records.append({
            "frame": target["frame"],
            "status": "ok",
            "neighbor_frames": [
                anchors[j]["frame"] for j in neighbor_indices
            ],
            "neighbor_frame_gaps": [
                abs(anchors[j]["frame"] - target["frame"])
                for j in neighbor_indices
            ],
            "weights": weights.tolist(),
            "error": error,
            "reconstructed_homography": reconstructed_h.tolist(),
        })

    successful_reconstruction = [
        record
        for record in reconstruction_records
        if record.get("status") == "ok"
    ]

    reconstruction_summary = None
    if successful_reconstruction:
        median_errors = np.asarray(
            [
                record["error"]["median_px"]
                for record in successful_reconstruction
            ],
            dtype=np.float64,
        )
        p90_errors = np.asarray(
            [
                record["error"]["p90_px"]
                for record in successful_reconstruction
            ],
            dtype=np.float64,
        )

        reconstruction_summary = {
            "tested_anchors": len(successful_reconstruction),
            "median_of_median_error_px": float(
                np.median(median_errors)
            ),
            "p90_of_median_error_px": float(
                np.percentile(median_errors, 90)
            ),
            "max_median_error_px": float(
                np.max(median_errors)
            ),
            "median_p90_error_px": float(
                np.median(p90_errors)
            ),
        }

    pairwise_summary = {
        "pair_count": len(pairwise_records),
        "min_median_px": float(np.min(camera_distances)),
        "median_median_px": float(np.median(camera_distances)),
        "p90_median_px": float(np.percentile(camera_distances, 90)),
        "max_median_px": float(np.max(camera_distances)),
        "time_camera_spearman": time_camera_spearman,
        "far_time_gap_frames": int(cfg.far_time_gap_frames),
        "median_nearest_far_time_px": (
            float(np.median(far_neighbor_distances))
            if far_neighbor_distances else None
        ),
        "min_nearest_far_time_px": (
            float(np.min(far_neighbor_distances))
            if far_neighbor_distances else None
        ),
    }

    component_frames = [
        [anchors[index]["frame"] for index in component]
        for component in components
    ]

    return {
        "status": "ok",
        "anchor_count": n,
        "descriptors": [
            anchor["descriptor"] for anchor in anchors
        ],
        "pairwise_summary": pairwise_summary,
        "nearest_neighbors": nearest_neighbors,
        "graph": {
            "status": geodesic_status,
            "k": graph_k,
            "component_count": len(components),
            "components": component_frames,
        },
        "embedding_2d": {
            "method": "knn_geodesic_classical_mds",
            "coordinates": embedding_records,
            "positive_eigenvalues": embedding_eigenvalues,
        },
        "local_reconstructability": {
            "note": (
                "Coverage/smoothness diagnostic, not production prediction. "
                "The target camera is used only to select its nearest atlas "
                "states during this offline benchmark."
            ),
            "neighbors": reconstruction_k,
            "summary": reconstruction_summary,
            "per_anchor": reconstruction_records,
        },
        "config": asdict(cfg),
    }


def summarize_camera_atlas_structure(
    analysis: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(analysis, dict):
        return {"status": "invalid_analysis"}

    pairwise = analysis.get("pairwise_summary", {})
    graph = analysis.get("graph", {})
    reconstructability = analysis.get(
        "local_reconstructability",
        {},
    )

    return {
        "status": analysis.get("status"),
        "anchor_count": analysis.get("anchor_count", 0),
        "time_camera_spearman": pairwise.get(
            "time_camera_spearman"
        ),
        "nearest_far_time_median_px": pairwise.get(
            "median_nearest_far_time_px"
        ),
        "nearest_far_time_min_px": pairwise.get(
            "min_nearest_far_time_px"
        ),
        "graph_k": graph.get("k"),
        "graph_components": graph.get("component_count"),
        "local_reconstruction": reconstructability.get("summary"),
    }

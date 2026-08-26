"""Direct visual Loop Closure V1 for fixed-position football cameras.

This module validates repeated/similar camera states by comparing distant
keyframes DIRECTLY in image space.

It does not use:
    - HUMAN homographies
    - pitch lines
    - pitch geometry
    - field masks

The trajectory is used only to propose likely candidate pairs. Each candidate
is then independently verified from the two source images with local features
+ robust projective geometry.

For a camera whose optical centre is approximately fixed, pan/tilt/roll/zoom
between two views is well modelled by one global image homography. Players and
other moving objects are treated as RANSAC outliers.

Outputs:
    - directly verified visual-overlap edges
    - strict recurrence edges for almost-the-same camera states
    - direct-vs-trajectory disagreement (a drift diagnostic)
    - connected recurrence regions
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from supervisor.field_camera_trajectory import (
    CameraTrajectoryConfig,
    camera_state_distance_px,
    relative_camera_transform,
)


EPS = 1e-9


@dataclass(frozen=True)
class LoopClosureConfig:
    analysis_width: int = 960

    # Candidate generation from the coarse accumulated trajectory.
    minimum_temporal_gap_frames: int = 120
    candidates_per_keyframe: int = 10
    maximum_coarse_candidate_median_px: float = 420.0

    # SIFT feature extraction / matching.
    sift_nfeatures: int = 2200
    sift_contrast_threshold: float = 0.018
    sift_edge_threshold: float = 12.0
    ratio_test: float = 0.75

    # Robust global projective verification.
    ransac_reprojection_threshold_px: float = 3.0
    ransac_confidence: float = 0.999
    ransac_max_iters: int = 6000

    minimum_ratio_matches: int = 28
    minimum_inliers: int = 22
    minimum_inlier_ratio: float = 0.36
    maximum_median_reprojection_px: float = 2.6

    minimum_inlier_span_x_fraction: float = 0.24
    minimum_inlier_span_y_fraction: float = 0.16
    minimum_inlier_hull_area_fraction: float = 0.022

    # A strict "same camera region" threshold. Verified pairs outside this
    # threshold are still valuable overlap edges but do not merge regions.
    recurrence_median_displacement_px: float = 42.0
    recurrence_p90_displacement_px: float = 82.0

    # Grid used to measure state distance and drift.
    grid_x: int = 9
    grid_y: int = 6
    grid_margin_x_fraction: float = 0.08
    grid_margin_y_fraction: float = 0.10


def _normalise_h(value: Any) -> np.ndarray | None:
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
        not np.isfinite(condition)
        or condition > 1e14
        or not np.isfinite(determinant)
        or abs(determinant) < 1e-12
    ):
        return None
    return h


def _resize(frame: np.ndarray, analysis_width: int) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    if width <= analysis_width:
        return frame, 1.0
    scale = float(analysis_width) / float(width)
    resized = cv2.resize(
        frame,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8),
    ).apply(gray)


def _grid(
    width: int,
    height: int,
    cfg: LoopClosureConfig,
) -> np.ndarray:
    xs = np.linspace(
        cfg.grid_margin_x_fraction * width,
        (1.0 - cfg.grid_margin_x_fraction) * width,
        cfg.grid_x,
    )
    ys = np.linspace(
        cfg.grid_margin_y_fraction * height,
        (1.0 - cfg.grid_margin_y_fraction) * height,
        cfg.grid_y,
    )
    return np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)


def _project(points: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    q = np.column_stack(
        (pts, np.ones(len(pts), dtype=np.float64))
    ) @ h.T
    den = q[:, 2]
    valid = (
        np.isfinite(q).all(axis=1)
        & np.isfinite(den)
        & (np.abs(den) > EPS)
    )
    out = np.full((len(pts), 2), np.nan, dtype=np.float64)
    out[valid] = q[valid, :2] / den[valid, None]
    valid &= np.isfinite(out).all(axis=1)
    return out, valid


def _matrix_distance_from_identity(
    h: Any,
    *,
    width: int,
    height: int,
    cfg: LoopClosureConfig,
) -> dict[str, Any] | None:
    matrix = _normalise_h(h)
    if matrix is None:
        return None
    points = _grid(width, height, cfg)
    projected, valid = _project(points, matrix)
    if not valid.any():
        return None
    errors = np.linalg.norm(projected[valid] - points[valid], axis=1)
    return {
        "median_px": float(np.median(errors)),
        "p90_px": float(np.percentile(errors, 90)),
        "max_px": float(np.max(errors)),
        "valid_points": int(np.count_nonzero(valid)),
    }


def _transform_disagreement(
    first: Any,
    second: Any,
    *,
    width: int,
    height: int,
    cfg: LoopClosureConfig,
) -> dict[str, Any] | None:
    a = _normalise_h(first)
    b = _normalise_h(second)
    if a is None or b is None:
        return None

    points = _grid(width, height, cfg)
    pa, va = _project(points, a)
    pb, vb = _project(points, b)
    valid = va & vb
    if not valid.any():
        return None

    errors = np.linalg.norm(pa[valid] - pb[valid], axis=1)
    return {
        "median_px": float(np.median(errors)),
        "p90_px": float(np.percentile(errors, 90)),
        "max_px": float(np.max(errors)),
        "valid_points": int(np.count_nonzero(valid)),
    }


def _distribution(
    points: np.ndarray,
    width: int,
    height: int,
    cfg: LoopClosureConfig,
) -> tuple[bool, dict[str, float]]:
    if len(points) < 3:
        return False, {}

    span_x = float(np.ptp(points[:, 0])) / max(float(width), 1.0)
    span_y = float(np.ptp(points[:, 1])) / max(float(height), 1.0)

    hull = cv2.convexHull(points.astype(np.float32))
    hull_area = float(cv2.contourArea(hull))
    hull_fraction = hull_area / max(float(width * height), 1.0)

    ok = (
        span_x >= cfg.minimum_inlier_span_x_fraction
        and span_y >= cfg.minimum_inlier_span_y_fraction
        and hull_fraction >= cfg.minimum_inlier_hull_area_fraction
    )
    return ok, {
        "span_x_fraction": span_x,
        "span_y_fraction": span_y,
        "hull_area_fraction": hull_fraction,
    }


def _symmetric_ratio_matches(
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
    ratio: float,
) -> list[cv2.DMatch]:
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    forward_knn = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    backward_knn = matcher.knnMatch(descriptors_b, descriptors_a, k=2)

    forward: dict[tuple[int, int], cv2.DMatch] = {}
    for pair in forward_knn:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            forward[(int(first.queryIdx), int(first.trainIdx))] = first

    backward_pairs: set[tuple[int, int]] = set()
    for pair in backward_knn:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            # Convert B->A indices back to A->B order.
            backward_pairs.add((int(first.trainIdx), int(first.queryIdx)))

    mutual = [
        match
        for key, match in forward.items()
        if key in backward_pairs
    ]
    mutual.sort(key=lambda match: float(match.distance))
    return mutual


def extract_keyframe_features(
    frame: np.ndarray,
    *,
    config: LoopClosureConfig | None = None,
) -> dict[str, Any]:
    cfg = config or LoopClosureConfig()

    work, scale = _resize(frame, cfg.analysis_width)
    gray = _gray(work)

    sift = cv2.SIFT_create(
        nfeatures=cfg.sift_nfeatures,
        contrastThreshold=cfg.sift_contrast_threshold,
        edgeThreshold=cfg.sift_edge_threshold,
    )
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or not keypoints:
        return {
            "status": "no_features",
            "scale": float(scale),
            "width": int(gray.shape[1]),
            "height": int(gray.shape[0]),
        }

    points = np.asarray(
        [keypoint.pt for keypoint in keypoints],
        dtype=np.float32,
    )

    return {
        "status": "ok",
        "scale": float(scale),
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "points": points,
        "descriptors": np.asarray(descriptors, dtype=np.float32),
        "feature_count": int(len(points)),
    }


def verify_loop_candidate(
    features_a: dict[str, Any],
    features_b: dict[str, Any],
    *,
    native_width: int,
    native_height: int,
    config: LoopClosureConfig | None = None,
) -> dict[str, Any]:
    """Directly verify one distant keyframe pair."""

    cfg = config or LoopClosureConfig()

    if features_a.get("status") != "ok" or features_b.get("status") != "ok":
        return {"status": "features_unavailable"}

    scale_a = float(features_a["scale"])
    scale_b = float(features_b["scale"])
    if abs(scale_a - scale_b) > 1e-9:
        return {"status": "inconsistent_scale"}
    scale = scale_a

    matches = _symmetric_ratio_matches(
        features_a["descriptors"],
        features_b["descriptors"],
        cfg.ratio_test,
    )
    if len(matches) < cfg.minimum_ratio_matches:
        return {
            "status": "insufficient_ratio_matches",
            "match_count": int(len(matches)),
        }

    points_a = features_a["points"]
    points_b = features_b["points"]

    source = np.asarray(
        [points_a[match.queryIdx] for match in matches],
        dtype=np.float32,
    )
    target = np.asarray(
        [points_b[match.trainIdx] for match in matches],
        dtype=np.float32,
    )

    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    matrix, inlier_mask = cv2.findHomography(
        source,
        target,
        method=method,
        ransacReprojThreshold=cfg.ransac_reprojection_threshold_px,
        maxIters=cfg.ransac_max_iters,
        confidence=cfg.ransac_confidence,
    )
    if matrix is None or inlier_mask is None:
        return {
            "status": "homography_failed",
            "match_count": int(len(matches)),
        }

    matrix = _normalise_h(matrix)
    if matrix is None:
        return {"status": "invalid_homography"}

    inliers = inlier_mask.ravel() > 0
    inlier_source = source[inliers]
    inlier_target = target[inliers]
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = inlier_count / max(float(len(matches)), 1.0)

    if inlier_count < cfg.minimum_inliers:
        return {
            "status": "insufficient_inliers",
            "match_count": int(len(matches)),
            "inlier_count": inlier_count,
            "inlier_ratio": float(inlier_ratio),
        }
    if inlier_ratio < cfg.minimum_inlier_ratio:
        return {
            "status": "low_inlier_ratio",
            "match_count": int(len(matches)),
            "inlier_count": inlier_count,
            "inlier_ratio": float(inlier_ratio),
        }

    distribution_ok, distribution = _distribution(
        inlier_source,
        int(features_a["width"]),
        int(features_a["height"]),
        cfg,
    )
    if not distribution_ok:
        return {
            "status": "poor_inlier_distribution",
            "match_count": int(len(matches)),
            "inlier_count": inlier_count,
            "inlier_ratio": float(inlier_ratio),
            **distribution,
        }

    projected, valid = _project(inlier_source, matrix)
    errors = np.linalg.norm(
        projected[valid] - inlier_target[valid],
        axis=1,
    )
    median_reprojection = (
        float(np.median(errors))
        if len(errors)
        else float("inf")
    )
    if (
        not np.isfinite(median_reprojection)
        or median_reprojection > cfg.maximum_median_reprojection_px
    ):
        return {
            "status": "high_reprojection_error",
            "match_count": int(len(matches)),
            "inlier_count": inlier_count,
            "inlier_ratio": float(inlier_ratio),
            "median_reprojection_px": median_reprojection,
            **distribution,
        }

    # Convert work-resolution homography to native image resolution:
    # H_native = S^-1 H_work S
    s = np.asarray(
        [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    s_inv = np.asarray(
        [[1.0 / scale, 0.0, 0.0], [0.0, 1.0 / scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    native_matrix = _normalise_h(s_inv @ matrix @ s)
    if native_matrix is None:
        return {"status": "native_conversion_failed"}

    direct_distance = _matrix_distance_from_identity(
        native_matrix,
        width=native_width,
        height=native_height,
        cfg=cfg,
    )
    if direct_distance is None:
        return {"status": "direct_distance_failed"}

    recurrence = (
        float(direct_distance["median_px"])
        <= cfg.recurrence_median_displacement_px
        and float(direct_distance["p90_px"])
        <= cfg.recurrence_p90_displacement_px
    )

    return {
        "status": "ok",
        "homography": native_matrix.tolist(),
        "match_count": int(len(matches)),
        "inlier_count": inlier_count,
        "inlier_ratio": float(inlier_ratio),
        "median_reprojection_px": median_reprojection,
        "direct_state_distance": direct_distance,
        "is_recurrence": bool(recurrence),
        **distribution,
    }


def _trajectory_candidates(
    trajectory: dict[str, Any],
    cfg: LoopClosureConfig,
) -> list[dict[str, Any]]:
    keyframes = trajectory.get("keyframes", [])
    if not isinstance(keyframes, list):
        return []

    video = trajectory.get("video", {})
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if width <= 0 or height <= 0:
        return []

    tcfg = CameraTrajectoryConfig()

    pair_map: dict[tuple[int, int], dict[str, Any]] = {}

    # Always preserve CTR01's already-discovered recurrence pairs.
    for edge in trajectory.get("recurrence_edges", []):
        try:
            left_index = int(edge["left_keyframe_index"])
            right_index = int(edge["right_keyframe_index"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (left_index, right_index)
        pair_map[key] = {
            "left_keyframe_index": left_index,
            "right_keyframe_index": right_index,
            "left_frame": int(keyframes[left_index]["frame"]),
            "right_frame": int(keyframes[right_index]["frame"]),
            "coarse_median_px": float(edge.get("median_displacement_px") or 0.0),
            "coarse_p90_px": float(edge.get("p90_displacement_px") or 0.0),
            "candidate_source": ["ctr_recurrence"],
        }

    # For every keyframe, also take N nearest sufficiently old states in the
    # accumulated trajectory, even if CTR01's strict recurrence threshold failed.
    for right_index, right in enumerate(keyframes):
        right_frame = int(right["frame"])
        nearest: list[tuple[float, float, int, dict[str, Any]]] = []

        for left_index in range(right_index):
            left = keyframes[left_index]
            left_frame = int(left["frame"])
            if right_frame - left_frame < cfg.minimum_temporal_gap_frames:
                continue

            distance = camera_state_distance_px(
                left["cumulative_transform"],
                right["cumulative_transform"],
                image_width=width,
                image_height=height,
                config=tcfg,
            )
            if distance is None:
                continue

            median = float(distance["median_px"])
            p90 = float(distance["p90_px"])
            if median > cfg.maximum_coarse_candidate_median_px:
                continue
            nearest.append((median, p90, left_index, distance))

        nearest.sort(key=lambda item: (item[0], item[1]))
        for median, p90, left_index, _distance in nearest[: cfg.candidates_per_keyframe]:
            key = (left_index, right_index)
            if key in pair_map:
                sources = pair_map[key]["candidate_source"]
                if "nearest_trajectory" not in sources:
                    sources.append("nearest_trajectory")
                continue

            pair_map[key] = {
                "left_keyframe_index": int(left_index),
                "right_keyframe_index": int(right_index),
                "left_frame": int(keyframes[left_index]["frame"]),
                "right_frame": right_frame,
                "coarse_median_px": median,
                "coarse_p90_px": p90,
                "candidate_source": ["nearest_trajectory"],
            }

    candidates = list(pair_map.values())
    candidates.sort(
        key=lambda item: (
            int(item["right_frame"]),
            float(item["coarse_median_px"]),
            int(item["left_frame"]),
        )
    )
    return candidates


class _UnionFind:
    def __init__(self, count: int):
        self.parent = list(range(count))
        self.rank = [0] * count

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a = self.find(left)
        b = self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _regions(
    keyframes: list[dict[str, Any]],
    recurrence_edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not keyframes:
        return []

    union = _UnionFind(len(keyframes))
    for edge in recurrence_edges:
        union.union(
            int(edge["left_keyframe_index"]),
            int(edge["right_keyframe_index"]),
        )

    groups: dict[int, list[int]] = {}
    for index in range(len(keyframes)):
        groups.setdefault(union.find(index), []).append(index)

    output = []
    for indices in groups.values():
        frames = [int(keyframes[index]["frame"]) for index in indices]
        output.append(
            {
                "keyframe_indices": [int(index) for index in indices],
                "frames": frames,
                "visit_count": len(indices),
                "first_frame": min(frames),
                "last_frame": max(frames),
            }
        )

    output.sort(key=lambda item: (item["first_frame"], -item["visit_count"]))
    for region_index, item in enumerate(output):
        item["region"] = int(region_index)
    return output


def analyse_loop_closures(
    video_path: str | Path,
    trajectory: dict[str, Any],
    *,
    config: LoopClosureConfig | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    cfg = config or LoopClosureConfig()

    if trajectory.get("status") != "ok":
        return {"status": "invalid_trajectory"}

    keyframes = trajectory.get("keyframes")
    if not isinstance(keyframes, list) or not keyframes:
        return {"status": "no_keyframes"}

    video_meta = trajectory.get("video", {})
    native_width = int(video_meta.get("width") or 0)
    native_height = int(video_meta.get("height") or 0)
    if native_width <= 0 or native_height <= 0:
        return {"status": "invalid_video_dimensions"}

    candidates = _trajectory_candidates(trajectory, cfg)
    if not candidates:
        return {
            "status": "ok",
            "version": 1,
            "algorithm": "camera_loop_closure_v1",
            "config": asdict(cfg),
            "candidate_count": 0,
            "verified_edge_count": 0,
            "recurrence_edge_count": 0,
            "region_count": len(keyframes),
            "candidates": [],
            "verified_edges": [],
            "recurrence_edges": [],
            "regions": _regions(keyframes, []),
        }

    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"status": "video_open_failed", "video": str(path)}

    # Only decode and describe keyframes that occur in at least one candidate.
    needed_indices = sorted(
        {
            int(candidate["left_keyframe_index"])
            for candidate in candidates
        }
        | {
            int(candidate["right_keyframe_index"])
            for candidate in candidates
        }
    )

    feature_cache: dict[int, dict[str, Any]] = {}
    try:
        for order, keyframe_index in enumerate(needed_indices):
            frame_index = int(keyframes[keyframe_index]["frame"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                feature_cache[keyframe_index] = {
                    "status": "frame_read_failed",
                    "frame": frame_index,
                }
            else:
                feature_cache[keyframe_index] = extract_keyframe_features(
                    frame,
                    config=cfg,
                )

            if progress_callback is not None:
                progress_callback(
                    "features",
                    order + 1,
                    len(needed_indices),
                    frame_index,
                    0,
                    0,
                )
    finally:
        capture.release()

    verified_edges: list[dict[str, Any]] = []
    recurrence_edges: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []

    for order, candidate in enumerate(candidates):
        left_index = int(candidate["left_keyframe_index"])
        right_index = int(candidate["right_keyframe_index"])

        verification = verify_loop_candidate(
            feature_cache.get(left_index, {"status": "missing"}),
            feature_cache.get(right_index, {"status": "missing"}),
            native_width=native_width,
            native_height=native_height,
            config=cfg,
        )

        record = {
            **candidate,
            "verification": {
                key: value
                for key, value in verification.items()
                if key != "homography"
            },
        }

        if verification.get("status") == "ok":
            direct_h = verification["homography"]
            predicted_h = relative_camera_transform(
                keyframes[left_index]["cumulative_transform"],
                keyframes[right_index]["cumulative_transform"],
            )
            disagreement = (
                _transform_disagreement(
                    direct_h,
                    predicted_h,
                    width=native_width,
                    height=native_height,
                    cfg=cfg,
                )
                if predicted_h is not None
                else None
            )

            edge = {
                "left_keyframe_index": left_index,
                "right_keyframe_index": right_index,
                "left_frame": int(candidate["left_frame"]),
                "right_frame": int(candidate["right_frame"]),
                "frame_gap": int(candidate["right_frame"]) - int(candidate["left_frame"]),
                "candidate_source": candidate["candidate_source"],
                "coarse_median_px": float(candidate["coarse_median_px"]),
                "coarse_p90_px": float(candidate["coarse_p90_px"]),
                "direct_homography": direct_h,
                "direct_state_distance": verification["direct_state_distance"],
                "trajectory_disagreement": disagreement,
                "match_count": int(verification["match_count"]),
                "inlier_count": int(verification["inlier_count"]),
                "inlier_ratio": float(verification["inlier_ratio"]),
                "median_reprojection_px": float(
                    verification["median_reprojection_px"]
                ),
                "is_recurrence": bool(verification["is_recurrence"]),
            }
            verified_edges.append(edge)
            if edge["is_recurrence"]:
                recurrence_edges.append(edge)

            record["trajectory_disagreement"] = disagreement

        candidate_records.append(record)

        if progress_callback is not None:
            progress_callback(
                "verify",
                order + 1,
                len(candidates),
                int(candidate["right_frame"]),
                len(verified_edges),
                len(recurrence_edges),
            )

    regions = _regions(keyframes, recurrence_edges)

    return {
        "status": "ok",
        "version": 1,
        "algorithm": "camera_loop_closure_v1",
        "video": {
            "path": str(path),
            "width": native_width,
            "height": native_height,
            "fps": float(video_meta.get("fps") or 0.0),
        },
        "config": asdict(cfg),
        "keyframe_count": len(keyframes),
        "candidate_count": len(candidates),
        "verified_edge_count": len(verified_edges),
        "recurrence_edge_count": len(recurrence_edges),
        "region_count": len(regions),
        "candidates": candidate_records,
        "verified_edges": verified_edges,
        "recurrence_edges": recurrence_edges,
        "regions": regions,
    }

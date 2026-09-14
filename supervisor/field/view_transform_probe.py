"""Image-to-image view-transform probe for RealJaenLab FIELD.

This diagnostic module estimates how a Veo view changed between two video
frames without requiring the target frame to have an independently solvable
football-pitch homography.

The intended use is:

    calibrated HUMAN source frame
        -> estimate image-space source->target transform from visual features
        -> transport the source pitch projection into the target frame
        -> optionally validate/refine later with sparse target annotations

The probe deliberately compares increasingly expressive motion models:

- translation
- similarity (uniform scale + rotation + translation)
- affine
- homography

Feature matching uses OpenCV (SIFT when available, ORB as fallback) and a robust
homography consensus to reject moving players and other inconsistent matches.
The four candidate transforms are then fitted and compared on the SAME robust
consensus correspondences, so their residuals are directly comparable.

This is an experiment, not the final CameraState implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - explicit runtime message
    raise ImportError(
        "view_transform_probe requires OpenCV. Install opencv-python-headless."
    ) from exc


EPS = 1e-10
MODEL_ORDER = ("translation", "similarity", "affine", "homography")
DEFAULT_MAX_FEATURES = 5000
DEFAULT_RATIO_SIFT = 0.78
DEFAULT_RATIO_ORB = 0.82
DEFAULT_MIN_MUTUAL_MATCHES = 16
DEFAULT_MIN_CONSENSUS_INLIERS = 12
DEFAULT_ADEQUACY_P95_DIAGONAL_FRACTION = 0.0030
DEFAULT_NEAR_BEST_DIAGONAL_FRACTION = 0.0008
DEFAULT_MIN_SPATIAL_COVERAGE = 0.05


class ViewTransformProbeError(ValueError):
    """Invalid input or insufficient visual evidence for view-transform probing."""


@dataclass(frozen=True, slots=True)
class FeatureSet:
    method: str
    keypoints: tuple[Any, ...]
    descriptors: np.ndarray


# ---------------------------------------------------------------------------
# Image / feature handling
# ---------------------------------------------------------------------------


def _ensure_image(image: np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ViewTransformProbeError(f"{label} must be a grayscale or BGR image array.")
    if arr.ndim == 3 and arr.shape[2] not in (3, 4):
        raise ViewTransformProbeError(f"{label} must have 3 or 4 channels when color.")
    if arr.shape[0] < 32 or arr.shape[1] < 32:
        raise ViewTransformProbeError(f"{label} is too small for feature matching.")
    if arr.dtype != np.uint8:
        finite = np.asarray(arr, dtype=float)
        if not np.all(np.isfinite(finite)):
            raise ViewTransformProbeError(f"{label} contains non-finite values.")
        finite = np.clip(finite, 0.0, 255.0)
        arr = finite.astype(np.uint8)
    return np.ascontiguousarray(arr)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _create_detector(max_features: int) -> tuple[str, Any, int, float]:
    if hasattr(cv2, "SIFT_create"):
        return (
            "SIFT",
            cv2.SIFT_create(nfeatures=int(max_features), contrastThreshold=0.02),
            cv2.NORM_L2,
            DEFAULT_RATIO_SIFT,
        )
    return (
        "ORB",
        cv2.ORB_create(
            nfeatures=int(max_features),
            scaleFactor=1.2,
            nlevels=10,
            edgeThreshold=21,
            fastThreshold=10,
        ),
        cv2.NORM_HAMMING,
        DEFAULT_RATIO_ORB,
    )


def _detect_features(gray: np.ndarray, max_features: int) -> tuple[FeatureSet, int, float]:
    method, detector, norm, ratio = _create_detector(max_features)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None or keypoints is None or len(keypoints) < 8:
        raise ViewTransformProbeError(
            f"{method} found too few usable features ({0 if keypoints is None else len(keypoints)})."
        )
    return FeatureSet(method, tuple(keypoints), descriptors), norm, ratio


def _ratio_matches(
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
    norm: int,
    ratio: float,
) -> dict[int, Any]:
    matcher = cv2.BFMatcher(normType=norm, crossCheck=False)
    pairs = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good: dict[int, Any] = {}
    for pair in pairs:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < float(ratio) * second.distance:
            good[int(first.queryIdx)] = first
    return good


def _mutual_matches(
    source: FeatureSet,
    target: FeatureSet,
    norm: int,
    ratio: float,
) -> list[Any]:
    forward = _ratio_matches(source.descriptors, target.descriptors, norm, ratio)
    reverse = _ratio_matches(target.descriptors, source.descriptors, norm, ratio)
    result = []
    for source_index, match in forward.items():
        reverse_match = reverse.get(int(match.trainIdx))
        if reverse_match is not None and int(reverse_match.trainIdx) == source_index:
            result.append(match)
    result.sort(key=lambda item: float(item.distance))
    return result


def _grid_balance_matches(
    matches: Sequence[Any],
    keypoints: Sequence[Any],
    width: int,
    height: int,
    *,
    cols: int = 8,
    rows: int = 5,
    max_per_cell: int = 35,
) -> list[Any]:
    cells: dict[tuple[int, int], list[Any]] = {}
    for match in matches:
        x, y = keypoints[int(match.queryIdx)].pt
        cx = min(cols - 1, max(0, int(float(x) / max(width, 1) * cols)))
        cy = min(rows - 1, max(0, int(float(y) / max(height, 1) * rows)))
        cells.setdefault((cx, cy), []).append(match)
    balanced: list[Any] = []
    for items in cells.values():
        items.sort(key=lambda item: float(item.distance))
        balanced.extend(items[:max_per_cell])
    balanced.sort(key=lambda item: float(item.distance))
    return balanced


def _match_points(
    source_image: np.ndarray,
    target_image: np.ndarray,
    *,
    max_features: int,
) -> tuple[str, np.ndarray, np.ndarray, dict[str, Any]]:
    source_gray = _to_gray(source_image)
    target_gray = _to_gray(target_image)
    source_features, source_norm, ratio = _detect_features(source_gray, max_features)
    target_features, target_norm, target_ratio = _detect_features(target_gray, max_features)

    # In practice both detections use the same detector choice. Keep this check
    # explicit so future detector fallbacks cannot silently mix descriptor types.
    if source_features.method != target_features.method or source_norm != target_norm:
        raise ViewTransformProbeError("Source and target feature descriptor types differ.")
    ratio = min(ratio, target_ratio)

    matches = _mutual_matches(source_features, target_features, source_norm, ratio)
    raw_mutual_count = len(matches)
    matches = _grid_balance_matches(
        matches,
        source_features.keypoints,
        source_image.shape[1],
        source_image.shape[0],
    )
    if len(matches) < DEFAULT_MIN_MUTUAL_MATCHES:
        raise ViewTransformProbeError(
            f"Too few mutual feature matches ({len(matches)}; need at least {DEFAULT_MIN_MUTUAL_MATCHES})."
        )

    source_points = np.asarray(
        [source_features.keypoints[int(m.queryIdx)].pt for m in matches], dtype=float
    )
    target_points = np.asarray(
        [target_features.keypoints[int(m.trainIdx)].pt for m in matches], dtype=float
    )
    diagnostics = {
        "feature_method": source_features.method,
        "source_keypoint_count": len(source_features.keypoints),
        "target_keypoint_count": len(target_features.keypoints),
        "mutual_match_count_before_balance": raw_mutual_count,
        "mutual_match_count": len(matches),
        "ratio_threshold": float(ratio),
    }
    return source_features.method, source_points, target_points, diagnostics


# ---------------------------------------------------------------------------
# Robust consensus and model fitting
# ---------------------------------------------------------------------------


def _robust_consensus(
    source: np.ndarray,
    target: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    diagonal = math.hypot(width, height)
    threshold = max(3.0, diagonal * 0.0025)
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    matrix, mask = cv2.findHomography(
        source.astype(np.float32),
        target.astype(np.float32),
        method,
        ransacReprojThreshold=float(threshold),
        maxIters=10000,
        confidence=0.999,
    )
    if matrix is None or mask is None:
        raise ViewTransformProbeError("Robust visual consensus could not be estimated.")
    inliers = mask.reshape(-1).astype(bool)
    count = int(np.count_nonzero(inliers))
    if count < DEFAULT_MIN_CONSENSUS_INLIERS:
        raise ViewTransformProbeError(
            f"Too few robust visual inliers ({count}; need at least {DEFAULT_MIN_CONSENSUS_INLIERS})."
        )
    return source[inliers], target[inliers], {
        "consensus_inlier_count": count,
        "consensus_inlier_ratio": float(count / max(len(source), 1)),
        "consensus_ransac_threshold_px": float(threshold),
        "consensus_method": "USAC_MAGSAC" if method == getattr(cv2, "USAC_MAGSAC", -999) else "RANSAC",
    }


def _fit_translation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = np.median(target - source, axis=0)
    return np.array(
        [[1.0, 0.0, delta[0]], [0.0, 1.0, delta[1]], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def _fit_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) < 2:
        raise ViewTransformProbeError("Similarity fit requires at least two correspondences.")
    src_mean = source.mean(axis=0)
    dst_mean = target.mean(axis=0)
    src_centered = source - src_mean
    dst_centered = target - dst_mean
    variance = float(np.sum(src_centered * src_centered) / len(source))
    if variance < EPS:
        raise ViewTransformProbeError("Similarity source correspondences have no spatial extent.")
    covariance = (dst_centered.T @ src_centered) / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(2)
    if np.linalg.det(u @ vt) < 0:
        correction[1, 1] = -1.0
    rotation = u @ correction @ vt
    scale = float(np.trace(np.diag(singular_values) @ correction) / variance)
    translation = dst_mean - scale * (rotation @ src_mean)
    matrix = np.eye(3, dtype=float)
    matrix[:2, :2] = scale * rotation
    matrix[:2, 2] = translation
    return matrix


def _fit_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) < 3:
        raise ViewTransformProbeError("Affine fit requires at least three correspondences.")
    design = np.column_stack([source, np.ones(len(source), dtype=float)])
    params, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3:
        raise ViewTransformProbeError("Affine source correspondences are degenerate.")
    return np.array(
        [
            [params[0, 0], params[1, 0], params[2, 0]],
            [params[0, 1], params[1, 1], params[2, 1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _hartley_transform(points: np.ndarray) -> np.ndarray:
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_distance = float(np.mean(np.linalg.norm(shifted, axis=1)))
    if mean_distance < EPS:
        raise ViewTransformProbeError("Homography correspondences have no spatial extent.")
    scale = math.sqrt(2.0) / mean_distance
    return np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _apply_matrix(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=float)])
    projected = (matrix @ homogeneous.T).T
    w = projected[:, 2]
    if np.any(np.abs(w) < EPS):
        raise ViewTransformProbeError("A fitted transform maps a correspondence to infinity.")
    return projected[:, :2] / w[:, None]


def _fit_homography(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) < 4:
        raise ViewTransformProbeError("Homography fit requires at least four correspondences.")
    ts = _hartley_transform(source)
    tt = _hartley_transform(target)
    src_n = _apply_matrix(ts, source)
    dst_n = _apply_matrix(tt, target)
    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    a = np.asarray(rows, dtype=float)
    _, _, vt = np.linalg.svd(a)
    h = vt[-1].reshape(3, 3)
    matrix = np.linalg.inv(tt) @ h @ ts
    if abs(float(matrix[2, 2])) > EPS:
        matrix /= float(matrix[2, 2])
    return matrix


def _fit_model(name: str, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if name == "translation":
        return _fit_translation(source, target)
    if name == "similarity":
        return _fit_similarity(source, target)
    if name == "affine":
        return _fit_affine(source, target)
    if name == "homography":
        return _fit_homography(source, target)
    raise ViewTransformProbeError(f"Unknown model {name!r}.")


def _residual_summary(matrix: np.ndarray, source: np.ndarray, target: np.ndarray, diagonal: float) -> dict[str, float]:
    residuals = np.linalg.norm(_apply_matrix(matrix, source) - target, axis=1)
    median = float(np.median(residuals))
    p95 = float(np.percentile(residuals, 95))
    maximum = float(np.max(residuals))
    rmse = float(np.sqrt(np.mean(residuals * residuals)))
    diagonal = max(float(diagonal), EPS)
    return {
        "median_px": median,
        "p95_px": p95,
        "max_px": maximum,
        "rmse_px": rmse,
        "median_diagonal_fraction": median / diagonal,
        "p95_diagonal_fraction": p95 / diagonal,
    }


def _jacobian(matrix: np.ndarray, x: float, y: float) -> np.ndarray:
    h = matrix
    den = float(h[2, 0] * x + h[2, 1] * y + h[2, 2])
    if abs(den) < EPS:
        raise ViewTransformProbeError("Cannot evaluate transform at image center.")
    num_u = float(h[0, 0] * x + h[0, 1] * y + h[0, 2])
    num_v = float(h[1, 0] * x + h[1, 1] * y + h[1, 2])
    den2 = den * den
    return np.array(
        [
            [
                (h[0, 0] * den - num_u * h[2, 0]) / den2,
                (h[0, 1] * den - num_u * h[2, 1]) / den2,
            ],
            [
                (h[1, 0] * den - num_v * h[2, 0]) / den2,
                (h[1, 1] * den - num_v * h[2, 1]) / den2,
            ],
        ],
        dtype=float,
    )


def _interpret_matrix(matrix: np.ndarray, width: int, height: int) -> dict[str, Any]:
    cx, cy = 0.5 * width, 0.5 * height
    transformed_center = _apply_matrix(matrix, np.asarray([[cx, cy]], dtype=float))[0]
    shift = transformed_center - np.asarray([cx, cy], dtype=float)
    jac = _jacobian(matrix, cx, cy)
    u, singular_values, vt = np.linalg.svd(jac)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    major = float(max(singular_values[0], EPS))
    minor = float(max(singular_values[1], EPS))
    scale = math.sqrt(major * minor)
    diagonal = math.hypot(width, height)
    normalized = matrix / matrix[2, 2] if abs(float(matrix[2, 2])) > EPS else matrix
    return {
        "center_shift_px": [float(shift[0]), float(shift[1])],
        "center_shift_normalized": [float(shift[0] / width), float(shift[1] / height)],
        "scale_geometric": float(scale),
        "log_scale": float(math.log(max(scale, EPS))),
        "scale_major": major,
        "scale_minor": minor,
        "anisotropy_ratio": float(major / minor),
        "rotation_deg": float(math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))),
        "projective_terms_per_diagonal": [
            float(normalized[2, 0] * diagonal),
            float(normalized[2, 1] * diagonal),
        ],
    }


def _coverage(points: np.ndarray, width: int, height: int) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    area = float(abs(cv2.contourArea(hull)))
    return area / max(float(width * height), 1.0)


def _choose_model(models: dict[str, dict[str, Any]], diagonal: float) -> str:
    best_name = min(MODEL_ORDER, key=lambda name: models[name]["residual_summary"]["p95_px"])
    best_p95 = float(models[best_name]["residual_summary"]["p95_px"])
    adequate_limit = diagonal * DEFAULT_ADEQUACY_P95_DIAGONAL_FRACTION
    near_best = diagonal * DEFAULT_NEAR_BEST_DIAGONAL_FRACTION
    for name in MODEL_ORDER:
        p95 = float(models[name]["residual_summary"]["p95_px"])
        if p95 <= adequate_limit and p95 <= best_p95 + near_best:
            return name
    return best_name


# ---------------------------------------------------------------------------
# Public probe API
# ---------------------------------------------------------------------------


def probe_view_transform(
    source_image: np.ndarray,
    target_image: np.ndarray,
    *,
    max_features: int = DEFAULT_MAX_FEATURES,
) -> dict[str, Any]:
    """Estimate and compare view transforms between two already-rendered frames."""

    source_image = _ensure_image(source_image, "source_image")
    target_image = _ensure_image(target_image, "target_image")
    if source_image.shape[:2] != target_image.shape[:2]:
        raise ViewTransformProbeError("Source and target frame dimensions must match.")

    height, width = source_image.shape[:2]
    diagonal = math.hypot(width, height)
    feature_method, source_points, target_points, feature_diag = _match_points(
        source_image, target_image, max_features=max_features
    )
    consensus_source, consensus_target, consensus_diag = _robust_consensus(
        source_points, target_points, width, height
    )

    source_coverage = _coverage(consensus_source, width, height)
    target_coverage = _coverage(consensus_target, width, height)
    models: dict[str, dict[str, Any]] = {}
    for name in MODEL_ORDER:
        matrix = _fit_model(name, consensus_source, consensus_target)
        models[name] = {
            "name": name,
            "matrix": [[float(v) for v in row] for row in matrix],
            "residual_summary": _residual_summary(
                matrix, consensus_source, consensus_target, diagonal
            ),
            "transform": _interpret_matrix(matrix, width, height),
        }

    recommended = _choose_model(models, diagonal)
    warnings: list[str] = []
    if min(source_coverage, target_coverage) < DEFAULT_MIN_SPATIAL_COVERAGE:
        warnings.append(
            "Visual correspondences cover only a small part of the frame; extrapolated field projection may be unreliable."
        )
    if consensus_diag["consensus_inlier_ratio"] < 0.35:
        warnings.append(
            "The robust visual consensus uses a relatively small fraction of feature matches."
        )
    if models[recommended]["residual_summary"]["p95_diagonal_fraction"] > 0.006:
        warnings.append(
            "Even the recommended transform has a relatively high residual; consider chaining through temporally closer frames."
        )

    return {
        "image_width": int(width),
        "image_height": int(height),
        "feature_method": feature_method,
        **feature_diag,
        **consensus_diag,
        "source_spatial_coverage": float(source_coverage),
        "target_spatial_coverage": float(target_coverage),
        "recommended_model": recommended,
        "models": models,
        "warnings": warnings,
        "note": (
            "Diagnostic image-space transform. It measures how the rendered view changed; "
            "it is not yet a physical camera pose or focal length."
        ),
    }


def transform_projected_geometry(
    projected_geometry: dict[str, Any],
    matrix: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Transport already projected pitch geometry through an image-space transform."""

    h = np.asarray(matrix, dtype=float)
    if h.shape != (3, 3) or not np.all(np.isfinite(h)):
        raise ViewTransformProbeError("matrix must be a finite 3x3 transform.")

    def transform_points(values: Sequence[Sequence[float]]) -> list[list[float]]:
        if not values:
            return []
        pts = np.asarray(values, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ViewTransformProbeError("Projected geometry contains invalid points.")
        transformed = _apply_matrix(h, pts)
        return [[float(x), float(y)] for x, y in transformed]

    lines = []
    for line in projected_geometry.get("lines", []):
        item = dict(line)
        item["points"] = transform_points(line.get("points", []))
        lines.append(item)

    curves = []
    for curve in projected_geometry.get("curves", []):
        item = dict(curve)
        item["points"] = transform_points(curve.get("points", []))
        curves.append(item)

    points = []
    for point in projected_geometry.get("points", []):
        item = dict(point)
        original = point.get("point")
        item["point"] = transform_points([original])[0] if original is not None else None
        points.append(item)

    return {"lines": lines, "curves": curves, "points": points}


def read_video_frame(video_path: str | Path, frame_index: int) -> np.ndarray:
    """Read one frame by index for local diagnostic CLI use."""

    path = str(video_path)
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ViewTransformProbeError(f"Could not open video: {path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ViewTransformProbeError(f"Could not read frame {frame_index} from {path}")
    return frame


def probe_video_frames(
    video_path: str | Path,
    source_frame_index: int,
    target_frame_index: int,
) -> dict[str, Any]:
    source = read_video_frame(video_path, source_frame_index)
    target = read_video_frame(video_path, target_frame_index)
    result = probe_view_transform(source, target)
    result.update(
        {
            "video_path": str(video_path),
            "source_frame_index": int(source_frame_index),
            "target_frame_index": int(target_frame_index),
        }
    )
    return result


# ---------------------------------------------------------------------------
# Synthetic self-check
# ---------------------------------------------------------------------------


def _synthetic_scene(width: int = 960, height: int = 540) -> np.ndarray:
    rng = np.random.default_rng(20260914)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    noise = rng.normal(95.0, 18.0, size=(height, width)).clip(0, 255).astype(np.uint8)
    image[:] = noise[..., None]

    for _ in range(160):
        x = int(rng.integers(25, width - 25))
        y = int(rng.integers(25, height - 25))
        radius = int(rng.integers(3, 12))
        shade = int(rng.integers(40, 230))
        cv2.circle(image, (x, y), radius, (shade, shade, shade), -1, cv2.LINE_AA)
    for _ in range(70):
        x1 = int(rng.integers(0, width))
        y1 = int(rng.integers(0, height))
        x2 = int(rng.integers(0, width))
        y2 = int(rng.integers(0, height))
        shade = int(rng.integers(50, 230))
        cv2.line(image, (x1, y1), (x2, y2), (shade, shade, shade), 1, cv2.LINE_AA)
    for i, text in enumerate(("REAL", "JAEN", "FIELD", "CAMERA", "STATE", "VEO")):
        cv2.putText(
            image,
            text,
            (45 + (i % 3) * 285, 90 + (i // 3) * 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.25,
            (235, 235, 235),
            3,
            cv2.LINE_AA,
        )
    return cv2.GaussianBlur(image, (3, 3), 0)


def _self_check() -> None:
    source = _synthetic_scene()
    height, width = source.shape[:2]
    truth_scale = 1.16
    truth_rotation_deg = 2.8
    truth_center = (0.5 * width, 0.5 * height)
    affine = cv2.getRotationMatrix2D(truth_center, truth_rotation_deg, truth_scale)
    affine[0, 2] += 34.0
    affine[1, 2] -= 17.0
    target = cv2.warpAffine(
        source,
        affine,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    result = probe_view_transform(source, target, max_features=3500)
    similarity = result["models"]["similarity"]
    recovered_scale = float(similarity["transform"]["scale_geometric"])
    # OpenCV's image rotation convention yields a sign flip in the interpreted
    # mathematical rotation angle relative to getRotationMatrix2D's argument.
    recovered_rotation = float(similarity["transform"]["rotation_deg"])
    p95 = float(similarity["residual_summary"]["p95_px"])

    if abs(recovered_scale - truth_scale) > 0.02:
        raise AssertionError((recovered_scale, truth_scale))
    if abs(abs(recovered_rotation) - abs(truth_rotation_deg)) > 0.8:
        raise AssertionError((recovered_rotation, truth_rotation_deg))
    if p95 > 4.0:
        raise AssertionError(p95)

    geometry = {
        "lines": [{"name": "demo", "points": [[100.0, 100.0], [300.0, 100.0]]}],
        "curves": [],
        "points": [{"name": "p", "point": [200.0, 200.0]}],
    }
    transported = transform_projected_geometry(geometry, similarity["matrix"])
    if len(transported["lines"][0]["points"]) != 2:
        raise AssertionError("Projected geometry transport failed.")

    print("View transform probe self-check: OK")
    print(f"Feature method:      {result['feature_method']}")
    print(f"Consensus inliers:   {result['consensus_inlier_count']}")
    print(f"Synthetic scale:     {truth_scale:.4f}")
    print(f"Recovered scale:     {recovered_scale:.4f}")
    print(f"Synthetic rotation:  {truth_rotation_deg:+.3f} deg")
    print(f"Recovered rotation:  {recovered_rotation:+.3f} deg")
    print(f"Similarity P95:      {p95:.3f} px")
    print(f"Recommended model:   {result['recommended_model']}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Probe image-space view changes between two Veo frames.")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--source-frame", type=int)
    parser.add_argument("--target-frame", type=int)
    parser.add_argument("--json", action="store_true", help="Print full result as JSON.")
    args = parser.parse_args()

    if args.video is None:
        _self_check()
        return
    if args.source_frame is None or args.target_frame is None:
        parser.error("--video requires --source-frame and --target-frame")

    result = probe_video_frames(args.video, args.source_frame, args.target_frame)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        recommended = result["recommended_model"]
        print(f"Feature method:      {result['feature_method']}")
        print(f"Consensus inliers:   {result['consensus_inlier_count']}")
        print(f"Recommended model:   {recommended}")
        for name in MODEL_ORDER:
            summary = result["models"][name]["residual_summary"]
            transform = result["models"][name]["transform"]
            print(
                f"{name:11s} P95={summary['p95_px']:.2f}px "
                f"scale={transform['scale_geometric']:.4f} "
                f"rot={transform['rotation_deg']:+.3f}deg"
            )
        for warning in result["warnings"]:
            print(f"WARNING: {warning}")


if __name__ == "__main__":
    _main()

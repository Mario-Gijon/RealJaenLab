#!/usr/bin/env python3
"""
RealJaenLab - Uncalibrated Camera View Diversity · DOWNLOAD REVISION 3

OFFLINE research diagnostic for selecting a small number of stable, diverse
camera views from an uncalibrated football video.

DOWNLOAD REVISION 3.

This file intentionally keeps the repository filename
`check_video_view_diversity_v1.py` while the experiment is refined in place.
Do not create one Python file per iteration.

Current approach
----------------
1. Sample the video sequentially for speed.
2. Extract brightness-robust, football-field-oriented geometry descriptors.
3. Reject unstable transition/outlier frames.
4. Build an ordered pool of diverse candidates.
5. Compare only that small pool with ORB feature matching to estimate:
   - relative horizontal camera pan,
   - relative zoom/scale,
   - whether two candidates are effectively the same view.
6. Select a compact HUMAN-oriented recommendation set that tries to cover:
   - left / center / right pan regions when they exist,
   - more than one zoom/opening regime when it exists,
   - while suppressing near-duplicate views.

No homography, FIELD calibration, PEOPLE tracking, paid API, or online AI is
used. ORB and all other operations are local OpenCV algorithms.

Example
-------
PYTHONPATH=. python -m supervisor.check_video_view_diversity_v1 \
  --video supervisor/data/library/videos/managed_upload_559cfa321d00455d9ef8b6465dc23594.mp4 \
  --name real_jaen_cordoba

Default outputs
---------------
supervisor/data/reviews/<name>_view_diversity/
    report.json
    recommended_views_contact_sheet.jpg
    all_diverse_candidates_contact_sheet.jpg
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VERSION = "video_view_diversity_experimental_r3"


@dataclass(frozen=True)
class Config:
    sample_fps: float = 1.25
    max_samples: int = 500
    analysis_width: int = 640
    crop_top_fraction: float = 0.18
    crop_bottom_fraction: float = 0.985

    pitch_grid_rows: int = 4
    pitch_grid_cols: int = 10
    pitch_envelope_cols: int = 20

    line_grid_rows: int = 4
    line_grid_cols: int = 8
    line_orientation_bins: int = 6
    min_line_length_fraction: float = 0.035

    edge_grid_rows: int = 4
    edge_grid_cols: int = 10

    temporal_smoothing_radius: int = 1
    transition_percentile: float = 82.0
    density_neighbors: int = 6
    density_outlier_percentile: float = 96.0

    diverse_pool_size: int = 12
    minimum_recommendations: int = 4
    maximum_recommendations: int = 6

    orb_features: int = 1800
    orb_ratio_test: float = 0.78
    orb_min_good_matches: int = 14
    orb_min_inliers: int = 9
    orb_ransac_px: float = 3.5
    duplicate_overlap_threshold: float = 0.16

    pan_span_threshold: float = 0.10
    zoom_span_threshold: float = 0.08

    panel_width: int = 640
    panel_height: int = 410
    contact_columns: int = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a small, stable, pan/zoom-diverse set of views from an "
            "uncalibrated football video."
        )
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-fps", type=float, default=Config.sample_fps)
    parser.add_argument("--max-samples", type=int, default=Config.max_samples)
    parser.add_argument(
        "--diverse-pool-size",
        type=int,
        default=Config.diverse_pool_size,
        help="Number of diverse candidates inspected before final coverage selection.",
    )
    parser.add_argument("--crop-top", type=float, default=Config.crop_top_fraction)
    parser.add_argument("--crop-bottom", type=float, default=Config.crop_bottom_fraction)
    return parser.parse_args()


def format_time(frame: int, fps: float) -> str:
    seconds = frame / fps if fps > 0 else 0.0
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:05.2f}"


def video_metadata(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Invalid FPS: {fps}")
    if frame_count <= 0:
        raise RuntimeError(f"Invalid frame count: {frame_count}")

    return {
        "path": str(path),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": frame_count / fps,
    }


def choose_sample_frames(frame_count: int, fps: float, cfg: Config) -> np.ndarray:
    nominal = max(1, int(round(fps / max(cfg.sample_fps, 1e-6))))
    frames = np.arange(0, frame_count, nominal, dtype=np.int64)

    if len(frames) > cfg.max_samples:
        positions = np.linspace(0, len(frames) - 1, cfg.max_samples)
        frames = frames[np.round(positions).astype(np.int64)]

    frames = np.unique(frames)
    if frames[-1] != frame_count - 1:
        frames = np.unique(np.append(frames, frame_count - 1))
    return frames.astype(np.int64)


def sequential_sample(
    video_path: Path,
    target_frames: np.ndarray,
    cfg: Config,
) -> tuple[list[np.ndarray], np.ndarray, list[int]]:
    """
    Decode sequentially and retrieve only sampled frames.

    Repeated random seeking was the main performance problem in the first
    experiment; this keeps analysis latency practical.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    targets = set(int(v) for v in target_frames)
    last_target = int(target_frames[-1])
    frames: list[np.ndarray] = []
    valid_indices: list[int] = []
    failed: list[int] = []

    frame_index = 0
    while frame_index <= last_target:
        if not cap.grab():
            for missing in target_frames[target_frames >= frame_index]:
                failed.append(int(missing))
            break

        if frame_index in targets:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                target_w = min(cfg.analysis_width, w)
                scale = target_w / float(w)
                target_h = max(2, int(round(h * scale)))
                if target_w != w:
                    frame = cv2.resize(
                        frame,
                        (target_w, target_h),
                        interpolation=cv2.INTER_AREA,
                    )
                frames.append(frame)
                valid_indices.append(frame_index)
            else:
                failed.append(frame_index)

        frame_index += 1

    cap.release()
    return frames, np.asarray(valid_indices, dtype=np.int64), failed


def crop_analysis_region(frame: np.ndarray, cfg: Config) -> np.ndarray:
    h = frame.shape[0]
    y0 = int(round(h * cfg.crop_top_fraction))
    y1 = int(round(h * cfg.crop_bottom_fraction))
    y0 = max(0, min(y0, h - 2))
    y1 = max(y0 + 2, min(y1, h))
    return frame[y0:y1]


def adaptive_pitch_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Broad, exposure-robust approximation of visible grass.

    This is not semantic segmentation. It only provides a geometric cue whose
    relative shape tends to remain useful under the large sun/shadow changes in
    the Veo clip.
    """
    bgr_f = bgr.astype(np.float32)
    b, g, r = cv2.split(bgr_f)

    green_excess = g - 0.5 * (r + b)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    chromatic = (
        (hue >= 25)
        & (hue <= 95)
        & (sat >= 28)
        & (val >= 20)
    )

    positive = green_excess[chromatic]
    adaptive = (
        max(4.0, float(np.percentile(positive, 20)))
        if positive.size
        else 6.0
    )

    mask = chromatic & (green_excess >= adaptive)
    mask_u8 = mask.astype(np.uint8) * 255

    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), dtype=np.uint8),
    )
    return mask_u8


def grid_mean(image: np.ndarray, rows: int, cols: int) -> np.ndarray:
    resized = cv2.resize(
        image.astype(np.float32),
        (cols, rows),
        interpolation=cv2.INTER_AREA,
    )
    return resized.reshape(-1)


def pitch_envelope(mask: np.ndarray, columns: int) -> np.ndarray:
    """
    Coarse upper grass boundary + occupancy per horizontal slice.

    This is useful for differentiating camera compositions while remaining much
    less sensitive to exposure than raw RGB appearance.
    """
    h, w = mask.shape[:2]
    envelope: list[float] = []
    occupancy: list[float] = []

    for idx in range(columns):
        x0 = int(round(idx * w / columns))
        x1 = int(round((idx + 1) * w / columns))
        x1 = max(x0 + 1, min(x1, w))

        block = mask[:, x0:x1] > 0
        row_counts = block.mean(axis=1)
        rows = np.where(row_counts >= 0.12)[0]

        top = float(rows[0] / max(1, h - 1)) if len(rows) else 1.0
        envelope.append(top)
        occupancy.append(float(block.mean()))

    return np.asarray(envelope + occupancy, dtype=np.float32)


def line_geometry_descriptor(
    gray: np.ndarray,
    pitch_mask: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Length-weighted spatial/orientation histogram of long line segments.
    """
    h, w = gray.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = lsd.detect(enhanced)[0]

    bins = np.zeros(
        (
            cfg.line_grid_rows,
            cfg.line_grid_cols,
            cfg.line_orientation_bins,
        ),
        dtype=np.float32,
    )
    orientation_global = np.zeros(
        cfg.line_orientation_bins,
        dtype=np.float32,
    )

    expanded_pitch = cv2.dilate(
        pitch_mask,
        np.ones((11, 11), dtype=np.uint8),
        iterations=1,
    )

    minimum_length = cfg.min_line_length_fraction * w
    total_weight = 0.0
    weighted_x = 0.0
    x_values: list[float] = []
    accepted = 0

    if detected is not None:
        for raw in detected.reshape(-1, 4):
            x1, y1, x2, y2 = map(float, raw)
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length < minimum_length:
                continue

            mx = 0.5 * (x1 + x2)
            my = 0.5 * (y1 + y2)
            xi = int(np.clip(round(mx), 0, w - 1))
            yi = int(np.clip(round(my), 0, h - 1))
            if expanded_pitch[yi, xi] == 0:
                continue

            angle = math.atan2(dy, dx) % math.pi
            orientation = min(
                cfg.line_orientation_bins - 1,
                int(angle / math.pi * cfg.line_orientation_bins),
            )
            row = min(
                cfg.line_grid_rows - 1,
                int(my / max(h, 1) * cfg.line_grid_rows),
            )
            col = min(
                cfg.line_grid_cols - 1,
                int(mx / max(w, 1) * cfg.line_grid_cols),
            )

            weight = min(length / w, 0.5)
            bins[row, col, orientation] += weight
            orientation_global[orientation] += weight
            total_weight += weight
            weighted_x += weight * (mx / max(w, 1))
            x_values.append(mx / max(w, 1))
            accepted += 1

    if total_weight > 1e-6:
        bins /= total_weight
        orientation_global /= total_weight

    line_x = weighted_x / total_weight if total_weight > 1e-6 else 0.5
    line_spread = float(np.std(x_values)) if x_values else 0.0

    return (
        np.concatenate([bins.reshape(-1), orientation_global]),
        {
            "accepted_long_lines": int(accepted),
            "line_x_centroid": float(line_x),
            "line_x_spread": line_spread,
            "line_weight": float(total_weight),
        },
    )


def edge_geometry_descriptor(
    gray: np.ndarray,
    pitch_mask: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, dict[str, float]]:
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    median = float(np.median(enhanced))
    low = int(max(0, 0.66 * median))
    high = int(min(255, 1.33 * median + 20))
    edges = cv2.Canny(enhanced, low, max(low + 1, high))

    near_pitch = cv2.dilate(
        pitch_mask,
        np.ones((15, 15), dtype=np.uint8),
        iterations=1,
    )
    edges = cv2.bitwise_and(edges, near_pitch)

    ys, xs = np.where(edges > 0)
    if len(xs):
        x_centroid = float(np.mean(xs) / max(1, edges.shape[1] - 1))
        x_spread = float(np.std(xs) / max(1, edges.shape[1] - 1))
    else:
        x_centroid = 0.5
        x_spread = 0.0

    return (
        grid_mean(
            edges / 255.0,
            cfg.edge_grid_rows,
            cfg.edge_grid_cols,
        ),
        {
            "edge_x_centroid": x_centroid,
            "edge_x_spread": x_spread,
            "edge_fraction": float(np.mean(edges > 0)),
        },
    )


def l2_unit(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else vector


def descriptor_for_frame(
    frame: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, dict[str, float]]:
    crop = crop_analysis_region(frame, cfg)
    pitch = adaptive_pitch_mask(crop)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    pitch_grid = grid_mean(
        pitch / 255.0,
        cfg.pitch_grid_rows,
        cfg.pitch_grid_cols,
    )
    envelope = pitch_envelope(pitch, cfg.pitch_envelope_cols)
    line_geometry, line_diag = line_geometry_descriptor(gray, pitch, cfg)
    edge_geometry, edge_diag = edge_geometry_descriptor(gray, pitch, cfg)

    ys, xs = np.where(pitch > 0)
    if len(xs):
        pitch_x = float(np.mean(xs) / max(1, pitch.shape[1] - 1))
        pitch_y = float(np.mean(ys) / max(1, pitch.shape[0] - 1))
    else:
        pitch_x = 0.5
        pitch_y = 0.5

    descriptor = np.concatenate(
        [
            2.4 * l2_unit(pitch_grid),
            2.8 * l2_unit(envelope),
            3.2 * l2_unit(line_geometry),
            0.8 * l2_unit(edge_geometry),
        ]
    ).astype(np.float32)

    diagnostics: dict[str, float] = {
        "pitch_fraction": float(np.mean(pitch > 0)),
        "pitch_x_centroid": pitch_x,
        "pitch_y_centroid": pitch_y,
        **line_diag,
        **edge_diag,
    }

    # Fallback pan proxy used only to align disconnected ORB components.
    # Positive ~= camera more to image-right: scene geometry shifts left.
    diagnostics["fallback_pan_proxy"] = float(
        -(
            0.55 * (diagnostics["line_x_centroid"] - 0.5)
            + 0.30 * (diagnostics["edge_x_centroid"] - 0.5)
            + 0.15 * (diagnostics["pitch_x_centroid"] - 0.5)
        )
    )

    # Very weak fallback for disconnected zoom components. ORB scale is the
    # preferred signal; this only prevents arbitrary component offsets.
    diagnostics["fallback_zoom_proxy"] = float(
        0.55 * diagnostics["pitch_fraction"]
        - 0.30 * diagnostics["line_x_spread"]
        - 0.15 * diagnostics["edge_x_spread"]
    )

    if not np.isfinite(descriptor).all():
        raise ValueError("Descriptor contains NaN/Inf")
    return descriptor, diagnostics


def extract_descriptors(
    sampled_images: list[np.ndarray],
    valid_frames: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, list[dict[str, float]], np.ndarray]:
    descriptors: list[np.ndarray] = []
    diagnostics: list[dict[str, float]] = []
    kept_frames: list[int] = []

    for frame_index, image in zip(valid_frames, sampled_images):
        try:
            descriptor, diag = descriptor_for_frame(image, cfg)
        except Exception:
            continue

        descriptors.append(descriptor)
        diagnostics.append(diag)
        kept_frames.append(int(frame_index))

    if len(descriptors) < max(12, cfg.minimum_recommendations * 3):
        raise RuntimeError(
            f"Only {len(descriptors)} valid descriptors; insufficient analysis."
        )

    return (
        np.vstack(descriptors).astype(np.float64),
        diagnostics,
        np.asarray(kept_frames, dtype=np.int64),
    )


def robust_standardize(x: np.ndarray) -> np.ndarray:
    median = np.median(x, axis=0)
    mad = np.median(np.abs(x - median), axis=0)
    scale = 1.4826 * mad

    std = np.std(x, axis=0)
    scale = np.where(scale > 1e-6, scale, std)
    scale = np.where(scale > 1e-6, scale, 1.0)

    z = (x - median) / scale
    return np.clip(z, -5.0, 5.0)


def pca_reduce(
    x: np.ndarray,
    max_dims: int = 18,
    variance_target: float = 0.96,
) -> tuple[np.ndarray, dict[str, Any]]:
    centered = x - x.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)

    variance = singular**2
    total = float(variance.sum())
    if total <= 1e-12:
        return centered[:, :1], {
            "dimensions": 1,
            "explained_variance_fraction": 1.0,
        }

    cumulative = np.cumsum(variance) / total
    required = int(np.searchsorted(cumulative, variance_target) + 1)
    dims = min(
        max(6, required),
        max_dims,
        vt.shape[0],
        max(1, len(x) - 1),
    )

    reduced = centered @ vt[:dims].T

    # Gentle whitening: useful axes are comparable without allowing tiny PCA
    # components to explode.
    axis_std = np.std(reduced, axis=0)
    valid = axis_std[axis_std > 1e-8]
    reference = float(np.median(valid)) if valid.size else 1.0
    divisor = np.maximum(axis_std, 0.35 * reference)
    reduced = reduced / divisor

    return reduced, {
        "dimensions": int(dims),
        "requested_variance_fraction": variance_target,
        "explained_variance_fraction": float(cumulative[dims - 1]),
    }


def temporal_smooth(x: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return x.copy()

    result = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - radius)
        hi = min(len(x), i + radius + 1)
        result[i] = np.median(x[lo:hi], axis=0)
    return result


def pairwise_distances(x: np.ndarray) -> np.ndarray:
    squared = np.sum(x * x, axis=1, keepdims=True)
    d2 = squared + squared.T - 2.0 * (x @ x.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2, dtype=np.float64)


def transition_scores(x: np.ndarray) -> np.ndarray:
    n = len(x)
    score = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return score

    adjacent = np.linalg.norm(np.diff(x, axis=0), axis=1)
    score[0] = adjacent[0]
    score[-1] = adjacent[-1]
    if n > 2:
        score[1:-1] = 0.5 * (adjacent[:-1] + adjacent[1:])
    return score


def density_scores(distances: np.ndarray, neighbors: int) -> np.ndarray:
    n = len(distances)
    k = min(max(2, neighbors), max(2, n - 1))
    ordered = np.partition(distances, kth=k, axis=1)[:, 1 : k + 1]
    return np.median(ordered, axis=1)


def stable_candidate_mask(
    reduced: np.ndarray,
    distances: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, dict[str, float], np.ndarray, np.ndarray]:
    transitions = transition_scores(reduced)
    density = density_scores(distances, cfg.density_neighbors)

    transition_limit = float(
        np.percentile(transitions, cfg.transition_percentile)
    )
    density_limit = float(
        np.percentile(density, cfg.density_outlier_percentile)
    )

    stable = (transitions <= transition_limit) & (density <= density_limit)

    if int(stable.sum()) < max(cfg.diverse_pool_size * 2, 20):
        stable[:] = True

    return (
        stable,
        {
            "transition_limit": transition_limit,
            "density_distance_limit": density_limit,
            "stable_count": int(stable.sum()),
        },
        transitions,
        density,
    )


def medoid_of_candidates(
    distances: np.ndarray,
    candidates: np.ndarray,
) -> int:
    sub = distances[np.ix_(candidates, candidates)]
    costs = np.median(sub, axis=1)
    return int(candidates[int(np.argmin(costs))])


def diverse_candidate_pool(
    distances: np.ndarray,
    stable_mask: np.ndarray,
    density: np.ndarray,
    cfg: Config,
) -> list[int]:
    """
    Create a deliberately larger pool than the final HUMAN recommendation set.

    Final coverage selection is handled later with explicit pan/zoom information.
    """
    candidates = np.where(stable_mask)[0]
    first = medoid_of_candidates(distances, candidates)
    selected = [first]

    density_ref = float(np.median(density[candidates])) + 1e-8
    density_weight = np.clip(
        density_ref / (density + 1e-8),
        0.60,
        1.30,
    )

    while len(selected) < cfg.diverse_pool_size:
        nearest = np.min(distances[:, selected], axis=1)
        score = nearest * density_weight
        score[~stable_mask] = -1.0
        score[selected] = -1.0

        next_index = int(np.argmax(score))
        if score[next_index] <= 0:
            break
        selected.append(next_index)

    return selected


def image_by_frame(
    sampled_images: list[np.ndarray],
    sampled_frames: np.ndarray,
) -> dict[int, np.ndarray]:
    return {
        int(frame): image
        for frame, image in zip(sampled_frames, sampled_images)
    }


def orb_features(
    image: np.ndarray,
    cfg: Config,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """
    Local, free scene matching.

    We use a large part of the frame, including static stands and field lines,
    while excluding a little of the very bottom where nearby moving people tend
    to dominate.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    h, w = gray.shape[:2]
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[: int(round(h * 0.90)), :] = 255

    orb = cv2.ORB_create(
        nfeatures=cfg.orb_features,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        patchSize=31,
        fastThreshold=12,
    )
    keypoints, descriptors = orb.detectAndCompute(gray, mask)
    return keypoints, descriptors


def match_orb_pair(
    keypoints_a: list[cv2.KeyPoint],
    descriptors_a: np.ndarray | None,
    keypoints_b: list[cv2.KeyPoint],
    descriptors_b: np.ndarray | None,
    width: int,
    height: int,
    cfg: Config,
) -> dict[str, Any] | None:
    if descriptors_a is None or descriptors_b is None:
        return None
    if len(keypoints_a) < cfg.orb_min_good_matches:
        return None
    if len(keypoints_b) < cfg.orb_min_good_matches:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = matcher.knnMatch(descriptors_a, descriptors_b, k=2)

    good: list[cv2.DMatch] = []
    for pair in raw:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < cfg.orb_ratio_test * second.distance:
            good.append(first)

    if len(good) < cfg.orb_min_good_matches:
        return None

    src = np.float32(
        [keypoints_a[m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)
    dst = np.float32(
        [keypoints_b[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    affine, inlier_mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=cfg.orb_ransac_px,
        maxIters=2500,
        confidence=0.995,
        refineIters=10,
    )
    if affine is None or inlier_mask is None:
        return None

    inliers = int(inlier_mask.ravel().sum())
    if inliers < cfg.orb_min_inliers:
        return None

    inlier_ratio = float(inliers / max(1, len(good)))
    a = float(affine[0, 0])
    b = float(affine[0, 1])
    scale = math.hypot(a, b)
    if not np.isfinite(scale) or scale <= 1e-6:
        return None

    center = np.asarray(
        [0.5 * width, 0.5 * height, 1.0],
        dtype=np.float64,
    )
    mapped = affine @ center
    center_shift_x = float(
        (mapped[0] - 0.5 * width) / max(1, width)
    )

    # If camera B pans right, static scene tends to shift left from A->B.
    pan_delta = -center_shift_x
    zoom_delta = math.log(scale)

    # Approximate same-view overlap. It is used only as a duplicate penalty,
    # never as a FIELD/geometric truth claim.
    match_strength = min(1.0, inliers / 75.0)
    overlap_score = float(match_strength * inlier_ratio)

    weight = float(
        max(1.0, math.sqrt(inliers) * max(0.10, inlier_ratio))
    )

    return {
        "good_matches": int(len(good)),
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "pan_delta": pan_delta,
        "zoom_delta": zoom_delta,
        "overlap_score": overlap_score,
        "weight": weight,
    }


def connected_components(
    n: int,
    edges: list[dict[str, Any]],
) -> list[list[int]]:
    adjacency = [[] for _ in range(n)]
    for edge in edges:
        a, b = int(edge["a"]), int(edge["b"])
        adjacency[a].append(b)
        adjacency[b].append(a)

    visited = [False] * n
    components: list[list[int]] = []

    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component: list[int] = []

        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

        components.append(sorted(component))

    return components


def solve_component_coordinates(
    component: list[int],
    edges: list[dict[str, Any]],
    field: str,
) -> dict[int, float]:
    """
    Weighted least squares for x_j - x_i = observed_delta.

    First node is fixed at zero to remove the arbitrary global offset.
    """
    if len(component) == 1:
        return {component[0]: 0.0}

    local_index = {node: idx for idx, node in enumerate(component)}
    anchor = component[0]

    rows: list[np.ndarray] = []
    rhs: list[float] = []

    for edge in edges:
        a, b = int(edge["a"]), int(edge["b"])
        if a not in local_index or b not in local_index:
            continue

        weight = math.sqrt(float(edge["weight"]))
        row = np.zeros(len(component) - 1, dtype=np.float64)

        if a != anchor:
            row[local_index[a] - 1] -= weight
        if b != anchor:
            row[local_index[b] - 1] += weight

        rows.append(row)
        rhs.append(weight * float(edge[field]))

    if not rows:
        return {node: 0.0 for node in component}

    design = np.vstack(rows)
    target = np.asarray(rhs, dtype=np.float64)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)

    result = {anchor: 0.0}
    for node in component[1:]:
        result[node] = float(solution[local_index[node] - 1])
    return result


def candidate_scene_space(
    candidate_images: list[np.ndarray],
    candidate_diags: list[dict[str, float]],
    cfg: Config,
) -> dict[str, Any]:
    n = len(candidate_images)
    features = [orb_features(image, cfg) for image in candidate_images]

    edges: list[dict[str, Any]] = []
    overlap = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        h, w = candidate_images[i].shape[:2]
        for j in range(i + 1, n):
            match = match_orb_pair(
                features[i][0],
                features[i][1],
                features[j][0],
                features[j][1],
                w,
                h,
                cfg,
            )
            if match is None:
                continue

            edge = {
                "a": i,
                "b": j,
                **match,
            }
            edges.append(edge)
            overlap[i, j] = overlap[j, i] = match["overlap_score"]

    components = connected_components(n, edges)
    pan = np.zeros(n, dtype=np.float64)
    zoom = np.zeros(n, dtype=np.float64)
    component_ids = np.zeros(n, dtype=np.int32)

    fallback_pan = np.asarray(
        [diag["fallback_pan_proxy"] for diag in candidate_diags],
        dtype=np.float64,
    )
    fallback_zoom = np.asarray(
        [diag["fallback_zoom_proxy"] for diag in candidate_diags],
        dtype=np.float64,
    )

    for component_id, component in enumerate(components):
        pan_local = solve_component_coordinates(component, edges, "pan_delta")
        zoom_local = solve_component_coordinates(component, edges, "zoom_delta")

        local_pan_values = np.asarray(
            [pan_local[node] for node in component],
            dtype=np.float64,
        )
        local_zoom_values = np.asarray(
            [zoom_local[node] for node in component],
            dtype=np.float64,
        )

        # Align disconnected components using weak geometry-only proxies.
        pan_offset = float(
            np.median(fallback_pan[component] - local_pan_values)
        )
        zoom_offset = float(
            np.median(fallback_zoom[component] - local_zoom_values)
        )

        for node in component:
            pan[node] = pan_local[node] + pan_offset
            zoom[node] = zoom_local[node] + zoom_offset
            component_ids[node] = component_id

    return {
        "pan": pan,
        "zoom": zoom,
        "overlap": overlap,
        "edges": edges,
        "components": components,
        "component_ids": component_ids,
    }


def normalized_coordinate(values: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=np.float64)
    low = float(np.percentile(values, 10))
    high = float(np.percentile(values, 90))
    span = high - low

    if span <= 1e-8:
        return np.zeros_like(values), span

    center = 0.5 * (low + high)
    half = 0.5 * span
    return np.clip((values - center) / max(half, 1e-8), -1.5, 1.5), span


def categorical_bins(
    normalized: np.ndarray,
    meaningful_span: bool,
    labels: tuple[str, str, str],
) -> list[str]:
    if not meaningful_span:
        return [labels[1]] * len(normalized)

    result: list[str] = []
    for value in normalized:
        if value < -0.33:
            result.append(labels[0])
        elif value > 0.33:
            result.append(labels[2])
        else:
            result.append(labels[1])
    return result


def candidate_quality(
    candidate_sample_indices: list[int],
    transitions: np.ndarray,
    density: np.ndarray,
) -> np.ndarray:
    t = transitions[candidate_sample_indices]
    d = density[candidate_sample_indices]

    def inverse_rank(values: np.ndarray) -> np.ndarray:
        if len(values) <= 1:
            return np.ones(len(values), dtype=np.float64)
        order = np.argsort(np.argsort(values))
        return 1.0 - order / float(len(values) - 1)

    return 0.55 * inverse_rank(t) + 0.45 * inverse_rank(d)


def is_duplicate(
    candidate: int,
    selected: list[int],
    overlap: np.ndarray,
    pan_norm: np.ndarray,
    zoom_norm: np.ndarray,
    cfg: Config,
) -> bool:
    for existing in selected:
        same_scene = (
            overlap[candidate, existing] >= cfg.duplicate_overlap_threshold
        )
        same_camera_region = (
            abs(pan_norm[candidate] - pan_norm[existing]) < 0.32
            and abs(zoom_norm[candidate] - zoom_norm[existing]) < 0.38
        )
        if same_scene and same_camera_region:
            return True
    return False


def choose_best_candidate(
    candidate_ids: list[int],
    selected: list[int],
    original_distances: np.ndarray,
    pool_sample_indices: list[int],
    overlap: np.ndarray,
    pan_norm: np.ndarray,
    zoom_norm: np.ndarray,
    quality: np.ndarray,
    cfg: Config,
) -> int | None:
    available = [
        idx
        for idx in candidate_ids
        if idx not in selected
        and not is_duplicate(
            idx,
            selected,
            overlap,
            pan_norm,
            zoom_norm,
            cfg,
        )
    ]
    if not available:
        available = [
            idx for idx in candidate_ids if idx not in selected
        ]
    if not available:
        return None

    if not selected:
        # Prefer stable/dense view closest to neutral pan/zoom.
        return max(
            available,
            key=lambda idx: (
                quality[idx]
                - 0.25 * abs(pan_norm[idx])
                - 0.12 * abs(zoom_norm[idx])
            ),
        )

    selected_samples = [pool_sample_indices[idx] for idx in selected]
    all_pair_values = original_distances[
        np.ix_(pool_sample_indices, pool_sample_indices)
    ]
    distance_scale = float(
        np.percentile(all_pair_values, 85)
    ) + 1e-8

    def score(idx: int) -> float:
        sample_idx = pool_sample_indices[idx]
        novelty = float(
            np.min(original_distances[sample_idx, selected_samples])
        ) / distance_scale
        overlap_penalty = (
            max(float(overlap[idx, existing]) for existing in selected)
            if selected
            else 0.0
        )
        return (
            1.15 * novelty
            + 0.65 * quality[idx]
            - 0.95 * overlap_penalty
        )

    return max(available, key=score)


def coverage_recommendations(
    pool_sample_indices: list[int],
    original_distances: np.ndarray,
    transitions: np.ndarray,
    density: np.ndarray,
    scene_space: dict[str, Any],
    cfg: Config,
) -> dict[str, Any]:
    pan_raw = np.asarray(scene_space["pan"], dtype=np.float64)
    zoom_raw = np.asarray(scene_space["zoom"], dtype=np.float64)
    overlap = np.asarray(scene_space["overlap"], dtype=np.float64)

    pan_norm, pan_span = normalized_coordinate(pan_raw)
    zoom_norm, zoom_span = normalized_coordinate(zoom_raw)

    pan_meaningful = pan_span >= cfg.pan_span_threshold
    zoom_meaningful = zoom_span >= cfg.zoom_span_threshold

    pan_bins = categorical_bins(
        pan_norm,
        pan_meaningful,
        ("izquierda", "centro", "derecha"),
    )
    zoom_bins = categorical_bins(
        zoom_norm,
        zoom_meaningful,
        ("abierta", "media", "cerrada"),
    )

    quality = candidate_quality(
        pool_sample_indices,
        transitions,
        density,
    )

    selected: list[int] = []

    # 1. Start from the most useful neutral/central view.
    center_ids = [
        idx for idx, label in enumerate(pan_bins)
        if label == "centro"
    ]
    first = choose_best_candidate(
        center_ids or list(range(len(pool_sample_indices))),
        selected,
        original_distances,
        pool_sample_indices,
        overlap,
        pan_norm,
        zoom_norm,
        quality,
        cfg,
    )
    if first is not None:
        selected.append(first)

    # 2. Explicitly cover horizontal camera regions.
    available_pan_bins = [
        label
        for label in ("izquierda", "centro", "derecha")
        if label in pan_bins
    ]
    for label in ("izquierda", "derecha", "centro"):
        if label not in available_pan_bins:
            continue
        if any(pan_bins[idx] == label for idx in selected):
            continue

        ids = [
            idx for idx, candidate_label in enumerate(pan_bins)
            if candidate_label == label
        ]
        chosen = choose_best_candidate(
            ids,
            selected,
            original_distances,
            pool_sample_indices,
            overlap,
            pan_norm,
            zoom_norm,
            quality,
            cfg,
        )
        if chosen is not None:
            selected.append(chosen)

    # 3. If zoom range is meaningful, ensure at least two opening regimes.
    available_zoom_bins = [
        label
        for label in ("abierta", "media", "cerrada")
        if label in zoom_bins
    ]
    covered_zoom = {zoom_bins[idx] for idx in selected}

    if zoom_meaningful and len(available_zoom_bins) >= 2:
        for label in ("abierta", "cerrada", "media"):
            if label not in available_zoom_bins or label in covered_zoom:
                continue

            ids = [
                idx for idx, candidate_label in enumerate(zoom_bins)
                if candidate_label == label
            ]
            chosen = choose_best_candidate(
                ids,
                selected,
                original_distances,
                pool_sample_indices,
                overlap,
                pan_norm,
                zoom_norm,
                quality,
                cfg,
            )
            if chosen is not None:
                selected.append(chosen)
                covered_zoom.add(label)
            if len(covered_zoom) >= 2:
                break

    target_count = max(
        cfg.minimum_recommendations,
        len(available_pan_bins)
        + (1 if zoom_meaningful and len(available_zoom_bins) >= 2 else 0),
    )
    target_count = min(cfg.maximum_recommendations, target_count)

    # 4. Fill any remaining slot with the strongest non-duplicate diverse view.
    while len(selected) < target_count:
        chosen = choose_best_candidate(
            list(range(len(pool_sample_indices))),
            selected,
            original_distances,
            pool_sample_indices,
            overlap,
            pan_norm,
            zoom_norm,
            quality,
            cfg,
        )
        if chosen is None or chosen in selected:
            break
        selected.append(chosen)

    return {
        "recommended_pool_indices": selected[: cfg.maximum_recommendations],
        "pan_raw": pan_raw,
        "pan_norm": pan_norm,
        "pan_span": pan_span,
        "pan_meaningful": pan_meaningful,
        "pan_bins": pan_bins,
        "zoom_raw": zoom_raw,
        "zoom_norm": zoom_norm,
        "zoom_span": zoom_span,
        "zoom_meaningful": zoom_meaningful,
        "zoom_bins": zoom_bins,
        "quality": quality,
        "available_pan_bins": available_pan_bins,
        "available_zoom_bins": available_zoom_bins,
    }


def put_text(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    scale: float = 0.55,
) -> None:
    cv2.putText(
        image,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )


def make_panel(
    image: np.ndarray,
    lines: list[str],
    cfg: Config,
) -> np.ndarray:
    panel = np.zeros(
        (cfg.panel_height, cfg.panel_width, 3),
        dtype=np.uint8,
    )
    banner = 102
    available = cfg.panel_height - banner

    h, w = image.shape[:2]
    scale = min(
        cfg.panel_width / w,
        available / h,
    )
    resized = cv2.resize(
        image,
        (
            max(1, int(round(w * scale))),
            max(1, int(round(h * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )

    rh, rw = resized.shape[:2]
    x = (cfg.panel_width - rw) // 2
    y = banner + (available - rh) // 2
    panel[y : y + rh, x : x + rw] = resized

    for i, line in enumerate(lines[:4]):
        put_text(panel, line, (14, 23 + i * 22))
    return panel


def save_sheet(
    panels: list[np.ndarray],
    path: Path,
    columns: int,
) -> None:
    if not panels:
        return

    rows = int(math.ceil(len(panels) / columns))
    ph, pw = panels[0].shape[:2]
    sheet = np.zeros(
        (rows * ph, columns * pw, 3),
        dtype=np.uint8,
    )

    for index, panel in enumerate(panels):
        row, col = divmod(index, columns)
        sheet[
            row * ph : (row + 1) * ph,
            col * pw : (col + 1) * pw,
        ] = panel

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"Could not write image: {path}")


def build_visual_outputs(
    candidate_images: list[np.ndarray],
    candidate_frames: list[int],
    coverage: dict[str, Any],
    recommendations: list[int],
    transitions: np.ndarray,
    density: np.ndarray,
    pool_sample_indices: list[int],
    fps: float,
    output_dir: Path,
    cfg: Config,
) -> tuple[Path, Path]:
    recommended_panels: list[np.ndarray] = []
    all_panels: list[np.ndarray] = []

    recommended_rank = {
        pool_index: rank + 1
        for rank, pool_index in enumerate(recommendations)
    }

    for pool_index, (image, frame) in enumerate(
        zip(candidate_images, candidate_frames)
    ):
        recommended = pool_index in recommended_rank
        label = (
            f"RECOMENDADA {recommended_rank[pool_index]}"
            if recommended
            else "extra"
        )
        sample_index = pool_sample_indices[pool_index]

        panel = make_panel(
            image,
            [
                f"Vista candidata {pool_index + 1} · {label}",
                f"Frame {frame} · {format_time(frame, fps)}",
                (
                    f"pan={coverage['pan_bins'][pool_index]} "
                    f"({coverage['pan_norm'][pool_index]:+.2f}) · "
                    f"zoom={coverage['zoom_bins'][pool_index]} "
                    f"({coverage['zoom_norm'][pool_index]:+.2f})"
                ),
                (
                    f"transicion={transitions[sample_index]:.2f} · "
                    f"densidad={density[sample_index]:.2f}"
                ),
            ],
            cfg,
        )
        all_panels.append(panel)
        if recommended:
            recommended_panels.append(
                (
                    recommended_rank[pool_index],
                    panel,
                )
            )

    recommended_panels.sort(key=lambda item: item[0])

    recommended_path = (
        output_dir / "recommended_views_contact_sheet.jpg"
    )
    all_path = (
        output_dir / "all_diverse_candidates_contact_sheet.jpg"
    )

    save_sheet(
        [panel for _, panel in recommended_panels],
        recommended_path,
        cfg.contact_columns,
    )
    save_sheet(
        all_panels,
        all_path,
        cfg.contact_columns,
    )
    return recommended_path, all_path


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cfg = Config(
        sample_fps=args.sample_fps,
        max_samples=args.max_samples,
        diverse_pool_size=args.diverse_pool_size,
        crop_top_fraction=args.crop_top,
        crop_bottom_fraction=args.crop_bottom,
    )

    if cfg.sample_fps <= 0:
        raise ValueError("--sample-fps must be > 0")
    if not 0 <= cfg.crop_top_fraction < cfg.crop_bottom_fraction <= 1:
        raise ValueError("Invalid crop fractions")
    if cfg.diverse_pool_size < cfg.minimum_recommendations:
        raise ValueError("--diverse-pool-size is too small")

    name = args.name or video_path.stem
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (
            Path("supervisor/data/reviews")
            / f"{name}_view_diversity"
        ).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    start = time.perf_counter()
    metadata = video_metadata(video_path)
    requested = choose_sample_frames(
        metadata["frame_count"],
        metadata["fps"],
        cfg,
    )
    timings["metadata_sampling_s"] = time.perf_counter() - start

    print(
        f"[view-diversity] {metadata['frame_count']} frames · "
        f"{metadata['duration_seconds']:.1f}s · "
        f"requesting {len(requested)} samples"
    )

    start = time.perf_counter()
    images, valid_frames, decode_failures = sequential_sample(
        video_path,
        requested,
        cfg,
    )
    timings["sequential_decode_s"] = time.perf_counter() - start
    print(
        f"[view-diversity] sequential decode: "
        f"{len(valid_frames)} valid · "
        f"{len(decode_failures)} failed · "
        f"{timings['sequential_decode_s']:.2f}s"
    )

    start = time.perf_counter()
    raw, frame_diagnostics, descriptor_frames = extract_descriptors(
        images,
        valid_frames,
        cfg,
    )
    timings["descriptor_s"] = time.perf_counter() - start
    print(
        f"[view-diversity] geometry descriptors: "
        f"{len(descriptor_frames)} · "
        f"{timings['descriptor_s']:.2f}s"
    )

    frame_to_image = image_by_frame(images, valid_frames)
    frame_to_diag = {
        int(frame): diag
        for frame, diag in zip(descriptor_frames, frame_diagnostics)
    }

    start = time.perf_counter()
    standardized = robust_standardize(raw)
    reduced, pca_info = pca_reduce(standardized)
    reduced = temporal_smooth(
        reduced,
        cfg.temporal_smoothing_radius,
    )
    if not np.isfinite(reduced).all():
        raise RuntimeError("NaN/Inf in reduced descriptor space")

    distances = pairwise_distances(reduced)
    stable_mask, stability_info, transitions, density = (
        stable_candidate_mask(
            reduced,
            distances,
            cfg,
        )
    )
    pool_sample_indices = diverse_candidate_pool(
        distances,
        stable_mask,
        density,
        cfg,
    )
    timings["diverse_pool_s"] = time.perf_counter() - start

    candidate_frames = [
        int(descriptor_frames[index])
        for index in pool_sample_indices
    ]
    candidate_images = [
        frame_to_image[frame]
        for frame in candidate_frames
    ]
    candidate_diags = [
        frame_to_diag[frame]
        for frame in candidate_frames
    ]

    start = time.perf_counter()
    scene_space = candidate_scene_space(
        candidate_images,
        candidate_diags,
        cfg,
    )
    coverage = coverage_recommendations(
        pool_sample_indices,
        distances,
        transitions,
        density,
        scene_space,
        cfg,
    )
    recommendations = coverage["recommended_pool_indices"]
    timings["pan_zoom_coverage_s"] = time.perf_counter() - start

    print(
        f"[view-diversity] stable candidates: "
        f"{int(stable_mask.sum())}/{len(stable_mask)}"
    )
    print(
        f"[view-diversity] ORB graph: "
        f"{len(scene_space['edges'])} pair links · "
        f"{len(scene_space['components'])} component(s)"
    )
    print(
        f"[view-diversity] pan span={coverage['pan_span']:.3f} "
        f"({'meaningful' if coverage['pan_meaningful'] else 'weak'}) · "
        f"zoom span={coverage['zoom_span']:.3f} "
        f"({'meaningful' if coverage['zoom_meaningful'] else 'weak'})"
    )

    start = time.perf_counter()
    recommended_path, all_path = build_visual_outputs(
        candidate_images,
        candidate_frames,
        coverage,
        recommendations,
        transitions,
        density,
        pool_sample_indices,
        metadata["fps"],
        output_dir,
        cfg,
    )
    timings["visual_outputs_s"] = time.perf_counter() - start
    timings["total_s"] = time.perf_counter() - total_start

    candidate_report: list[dict[str, Any]] = []
    for pool_index, frame in enumerate(candidate_frames):
        sample_index = pool_sample_indices[pool_index]
        candidate_report.append(
            {
                "candidate": pool_index + 1,
                "frame": frame,
                "time": format_time(frame, metadata["fps"]),
                "recommended": pool_index in recommendations,
                "recommendation_rank": (
                    recommendations.index(pool_index) + 1
                    if pool_index in recommendations
                    else None
                ),
                "pan": {
                    "raw": float(coverage["pan_raw"][pool_index]),
                    "normalized": float(
                        coverage["pan_norm"][pool_index]
                    ),
                    "label": coverage["pan_bins"][pool_index],
                },
                "zoom": {
                    "raw": float(coverage["zoom_raw"][pool_index]),
                    "normalized": float(
                        coverage["zoom_norm"][pool_index]
                    ),
                    "label": coverage["zoom_bins"][pool_index],
                },
                "transition_score": float(
                    transitions[sample_index]
                ),
                "density_distance": float(
                    density[sample_index]
                ),
                "quality": float(
                    coverage["quality"][pool_index]
                ),
                "diagnostics": candidate_diags[pool_index],
            }
        )

    recommended_report = sorted(
        [
            item
            for item in candidate_report
            if item["recommended"]
        ],
        key=lambda item: int(item["recommendation_rank"]),
    )

    report = {
        "version": VERSION,
        "video": metadata,
        "config": asdict(cfg),
        "sampling": {
            "requested": int(len(requested)),
            "decoded": int(len(valid_frames)),
            "descriptor_count": int(len(descriptor_frames)),
            "decode_failures": decode_failures,
        },
        "descriptor": {
            "principle": (
                "Adaptive pitch geometry + long field/stadium line structure "
                "+ coarse pitch edges."
            ),
            "raw_dimensions": int(raw.shape[1]),
            "pca": pca_info,
            "paid_ai": False,
            "homography_required": False,
        },
        "candidate_filter": stability_info,
        "scene_matching": {
            "method": "ORB + RANSAC partial affine graph",
            "pair_links": int(len(scene_space["edges"])),
            "components": scene_space["components"],
            "pan_span": float(coverage["pan_span"]),
            "pan_meaningful": bool(coverage["pan_meaningful"]),
            "zoom_span": float(coverage["zoom_span"]),
            "zoom_meaningful": bool(
                coverage["zoom_meaningful"]
            ),
        },
        "selection": {
            "method": (
                "stable diverse pool -> ORB duplicate suppression -> "
                "pan/zoom coverage"
            ),
            "pool_count": int(len(pool_sample_indices)),
            "recommended_count": int(len(recommendations)),
            "available_pan_bins": coverage["available_pan_bins"],
            "available_zoom_bins": coverage["available_zoom_bins"],
            "recommended": recommended_report,
            "all_candidates": candidate_report,
        },
        "outputs": {
            "recommended_contact_sheet": str(recommended_path),
            "all_candidates_contact_sheet": str(all_path),
        },
        "timings_seconds": {
            key: round(value, 4)
            for key, value in timings.items()
        },
        "interpretation": [
            "Pan/zoom labels are relative camera-view proxies, not FIELD coordinates.",
            "The goal is compact HUMAN coverage, not discrete classification of the video.",
            "Near-duplicate candidates are penalized using local ORB scene overlap.",
            "Human visual validation is still required before web integration.",
        ],
    }

    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n[view-diversity] recommended frames")
    for item in recommended_report:
        print(
            f"  {item['recommendation_rank']:02d}. "
            f"frame {item['frame']} ({item['time']}) · "
            f"pan={item['pan']['label']} "
            f"({item['pan']['normalized']:+.2f}) · "
            f"zoom={item['zoom']['label']} "
            f"({item['zoom']['normalized']:+.2f})"
        )

    print("\n[view-diversity] timings")
    for key, value in timings.items():
        print(f"  {key}: {value:.3f}s")

    print(f"\nReport: {report_path}")
    print(f"Recommended sheet: {recommended_path}")
    print(f"All candidates sheet: {all_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

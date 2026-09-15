"""HUMAN view descriptor and Human Atlas coverage diagnostics for RealJaenLab FIELD.

V2 changes the descriptor approximation from one global homography fitted to all
semantic landmarks to a viewport-local homography fitted only to landmarks that
are visible or close to the current image.

Why
---
The final HUMAN geometry may include anisotropic correction and/or DERIVED
refinement.  In tight lateral views, landmarks from the opposite side of the
pitch can be projected extremely far outside the image.  A single global
homography can then be a terrible summary of the *visible* camera state even
when the local calibration is good.

V2 therefore keeps two diagnostics:

- ``approximation``: LOCAL viewport fit.  This is the model used for field
  coverage, visible centroid and effective px/m scale.
- ``global_approximation``: all semantic landmarks, diagnostic only.  It tells
  us how non-projective the complete extrapolated geometry is.

The local approximation is still only a descriptor aid.  It never replaces the
actual calibrated ``projected_geometry`` and it is not a physical optical-zoom
estimate.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

try:
    from .pitch import PitchModel
except ImportError:  # Allows direct execution from supervisor/field.
    from pitch import PitchModel


EPS = 1e-10

DEFAULT_GRID_COLS = 41
DEFAULT_GRID_ROWS = 27
DEFAULT_VIEWPORT_MARGIN_PX = 0.0

# 0.5% of image diagonal.  This is only a descriptor-quality warning threshold.
DEFAULT_POOR_P95_DIAGONAL_FRACTION = 0.005

# Prefer at least this many local semantic correspondences when possible.
DEFAULT_MIN_LOCAL_CORRESPONDENCES = 6

# Progressive expanded-view margins used to find enough local landmarks.
DEFAULT_LOCAL_MARGIN_DIAGONAL_FRACTIONS = (
    0.00,
    0.04,
    0.08,
    0.15,
    0.25,
    0.40,
)

# If the progressive margins still cannot supply a usable fit, use the nearest
# semantic points to the viewport, but never fewer than four.
DEFAULT_LOCAL_NEAREST_FALLBACK = 8


class HumanViewDescriptorError(ValueError):
    """Invalid or insufficient input for a HUMAN view descriptor."""


# ---------------------------------------------------------------------------
# Generic numeric helpers
# ---------------------------------------------------------------------------


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise HumanViewDescriptorError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise HumanViewDescriptorError(f"{label} must be finite.")
    return number


def _positive(value: Any, label: str) -> float:
    number = _finite(value, label)
    if number <= 0:
        raise HumanViewDescriptorError(f"{label} must be positive.")
    return number


def _positive_int(value: Any, label: str, minimum: int = 1) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise HumanViewDescriptorError(f"{label} must be an integer.") from exc
    if number < minimum:
        raise HumanViewDescriptorError(f"{label} must be >= {minimum}.")
    return number


def _point(value: Sequence[float], label: str) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise HumanViewDescriptorError(f"{label} must contain two coordinates.")
    return _finite(value[0], f"{label}[0]"), _finite(value[1], f"{label}[1]")


def _normalize_h(matrix: np.ndarray) -> np.ndarray:
    h = np.asarray(matrix, dtype=float)
    if h.shape != (3, 3) or not np.all(np.isfinite(h)):
        raise HumanViewDescriptorError("Homography must be a finite 3x3 matrix.")
    if abs(float(h[2, 2])) > EPS:
        h = h / float(h[2, 2])
    else:
        norm = float(np.linalg.norm(h))
        if norm < EPS:
            raise HumanViewDescriptorError("Degenerate homography.")
        h = h / norm
    return h


def _safe_apply_h(
    matrix: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project points while tolerating singular samples.

    Returns:
      xy: Nx2 array. Invalid rows contain NaN.
      valid: boolean N mask.
    """

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise HumanViewDescriptorError("Points must be an Nx2 array.")

    h = _normalize_h(matrix)
    hom = np.column_stack([pts, np.ones(len(pts), dtype=float)])
    out = (h @ hom.T).T
    w = out[:, 2]

    valid = np.isfinite(out).all(axis=1) & (np.abs(w) >= EPS)
    xy = np.full((len(pts), 2), np.nan, dtype=float)
    if np.any(valid):
        xy[valid] = out[valid, :2] / w[valid, None]
        finite_xy = np.isfinite(xy).all(axis=1)
        valid &= finite_xy
        xy[~valid] = np.nan
    return xy, valid


def _apply_h(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    xy, valid = _safe_apply_h(matrix, points)
    if not np.all(valid):
        raise HumanViewDescriptorError("Homography maps one or more required samples to infinity.")
    return xy


def _hartley(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_distance = float(np.mean(np.linalg.norm(shifted, axis=1)))
    if mean_distance < EPS:
        raise HumanViewDescriptorError("Homography samples have no spatial extent.")
    scale = math.sqrt(2.0) / mean_distance
    return np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _fit_homography(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise HumanViewDescriptorError(
            "Homography source/target must be matching Nx2 arrays."
        )
    if len(source) < 4:
        raise HumanViewDescriptorError(
            "At least four semantic correspondences are required."
        )
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise HumanViewDescriptorError("Homography correspondences must be finite.")

    ts = _hartley(source)
    tt = _hartley(target)
    src_n = _apply_h(ts, source)
    dst_n = _apply_h(tt, target)

    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])

    a = np.asarray(rows, dtype=float)
    _, singular_values, vt = np.linalg.svd(a)

    if singular_values.size < 2 or singular_values[-2] < EPS:
        raise HumanViewDescriptorError(
            "Semantic correspondences are degenerate for a homography."
        )

    h = np.linalg.inv(tt) @ vt[-1].reshape(3, 3) @ ts
    return _normalize_h(h)


def _summary(errors: np.ndarray, diagonal: float) -> dict[str, float]:
    arr = np.asarray(errors, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise HumanViewDescriptorError("Cannot summarize an empty residual set.")

    median = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    maximum = float(np.max(arr))
    rmse = float(np.sqrt(np.mean(arr * arr)))
    diagonal = max(float(diagonal), EPS)

    return {
        "median_px": median,
        "p95_px": p95,
        "max_px": maximum,
        "rmse_px": rmse,
        "median_diagonal_fraction": median / diagonal,
        "p95_diagonal_fraction": p95 / diagonal,
        "max_diagonal_fraction": maximum / diagonal,
    }


def _residual_summary(
    h: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    diagonal: float,
) -> dict[str, float]:
    predicted, valid = _safe_apply_h(h, source)
    if int(np.count_nonzero(valid)) < 1:
        raise HumanViewDescriptorError("Approximation cannot project its correspondences.")
    residuals = np.linalg.norm(predicted[valid] - target[valid], axis=1)
    return _summary(residuals, diagonal)


def _jacobian(h: np.ndarray, x: float, y: float) -> np.ndarray:
    h = _normalize_h(h)
    d = float(h[2, 0] * x + h[2, 1] * y + h[2, 2])
    if abs(d) < EPS:
        raise HumanViewDescriptorError(
            "Cannot evaluate local scale near a mapping singularity."
        )

    nu = float(h[0, 0] * x + h[0, 1] * y + h[0, 2])
    nv = float(h[1, 0] * x + h[1, 1] * y + h[1, 2])
    d2 = d * d

    return np.array(
        [
            [
                (h[0, 0] * d - nu * h[2, 0]) / d2,
                (h[0, 1] * d - nu * h[2, 1]) / d2,
            ],
            [
                (h[1, 0] * d - nv * h[2, 0]) / d2,
                (h[1, 1] * d - nv * h[2, 1]) / d2,
            ],
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Semantic points and viewport locality
# ---------------------------------------------------------------------------


def _canonical_points(pitch: PitchModel) -> dict[str, tuple[float, float]]:
    return {
        point.name: (float(point.point.x), float(point.point.y))
        for point in pitch.points
    }


def _projected_points(
    geometry: dict[str, Any],
) -> dict[str, tuple[float, float]]:
    if not isinstance(geometry, dict):
        raise HumanViewDescriptorError("projected_geometry must be an object.")

    result: dict[str, tuple[float, float]] = {}
    for item in geometry.get("points", []) or []:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        try:
            result[item["name"]] = _point(
                item.get("point"),
                f"projected point {item['name']}",
            )
        except HumanViewDescriptorError:
            continue
    return result


def _distance_to_viewport(
    image_points: np.ndarray,
    image_width: float,
    image_height: float,
) -> np.ndarray:
    """Euclidean distance to the image rectangle; zero means inside."""

    x = image_points[:, 0]
    y = image_points[:, 1]

    dx = np.where(
        x < 0.0,
        -x,
        np.where(x > image_width, x - image_width, 0.0),
    )
    dy = np.where(
        y < 0.0,
        -y,
        np.where(y > image_height, y - image_height, 0.0),
    )
    return np.hypot(dx, dy)


def _fit_is_locally_usable(
    h: np.ndarray,
    field_points: np.ndarray,
    image_points: np.ndarray,
    image_width: float,
    image_height: float,
) -> bool:
    """Conservative sanity check for a candidate local descriptor homography."""

    try:
        predicted = _apply_h(h, field_points)
    except HumanViewDescriptorError:
        return False

    if not np.all(np.isfinite(predicted)):
        return False

    # Reject a fit that turns a compact local target set into an absurd solution.
    diagonal = math.hypot(image_width, image_height)
    residuals = np.linalg.norm(predicted - image_points, axis=1)
    if not np.all(np.isfinite(residuals)):
        return False

    # This is deliberately loose: the actual geometry may be non-projective.
    if float(np.median(residuals)) > 0.15 * diagonal:
        return False

    return True


def _select_local_fit(
    names: list[str],
    field_points: np.ndarray,
    image_points: np.ndarray,
    image_width: float,
    image_height: float,
    diagonal: float,
    *,
    min_correspondences: int = DEFAULT_MIN_LOCAL_CORRESPONDENCES,
) -> dict[str, Any]:
    """Choose visible/near-viewport semantic points for the descriptor fit."""

    if len(names) < 4:
        raise HumanViewDescriptorError(
            "At least four semantic points are required for a local descriptor."
        )

    distances = _distance_to_viewport(image_points, image_width, image_height)
    visible_count = int(np.count_nonzero(distances <= EPS))

    # Prefer the first progressive viewport expansion with enough points.
    for fraction in DEFAULT_LOCAL_MARGIN_DIAGONAL_FRACTIONS:
        margin = float(fraction * diagonal)
        indices = np.flatnonzero(distances <= margin)

        if len(indices) < min_correspondences:
            continue

        try:
            h = _fit_homography(field_points[indices], image_points[indices])
        except HumanViewDescriptorError:
            continue

        if _fit_is_locally_usable(
            h,
            field_points[indices],
            image_points[indices],
            image_width,
            image_height,
        ):
            return {
                "matrix": h,
                "indices": indices,
                "selection_mode": "expanded_viewport",
                "fit_margin_px": margin,
                "fit_margin_diagonal_fraction": fraction,
                "visible_semantic_count": visible_count,
                "distances_px": distances,
            }

    # If we have 4-5 visible/very-near points, try them before bringing in
    # potentially wild far-field extrapolations.
    near_order = np.argsort(distances)
    for count in range(4, min(len(names), min_correspondences - 1) + 1):
        indices = near_order[:count]
        try:
            h = _fit_homography(field_points[indices], image_points[indices])
        except HumanViewDescriptorError:
            continue
        if _fit_is_locally_usable(
            h,
            field_points[indices],
            image_points[indices],
            image_width,
            image_height,
        ):
            return {
                "matrix": h,
                "indices": indices,
                "selection_mode": "nearest_sparse",
                "fit_margin_px": float(np.max(distances[indices])),
                "fit_margin_diagonal_fraction": float(
                    np.max(distances[indices]) / max(diagonal, EPS)
                ),
                "visible_semantic_count": visible_count,
                "distances_px": distances,
            }

    # Final fallback: nearest N semantic points, increasing until a stable fit
    # exists.  This is still local because ordering is by distance to viewport.
    start = min(
        len(names),
        max(4, DEFAULT_LOCAL_NEAREST_FALLBACK),
    )
    for count in range(start, len(names) + 1):
        indices = near_order[:count]
        try:
            h = _fit_homography(field_points[indices], image_points[indices])
        except HumanViewDescriptorError:
            continue
        if _fit_is_locally_usable(
            h,
            field_points[indices],
            image_points[indices],
            image_width,
            image_height,
        ):
            return {
                "matrix": h,
                "indices": indices,
                "selection_mode": "nearest_fallback",
                "fit_margin_px": float(np.max(distances[indices])),
                "fit_margin_diagonal_fraction": float(
                    np.max(distances[indices]) / max(diagonal, EPS)
                ),
                "visible_semantic_count": visible_count,
                "distances_px": distances,
            }

    raise HumanViewDescriptorError(
        "Could not obtain a stable local homography from visible/nearby semantic landmarks."
    )


# ---------------------------------------------------------------------------
# Coverage grid
# ---------------------------------------------------------------------------


def _grid(
    length: float,
    width: float,
    cols: int,
    rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Cell centres make visible-cell fraction a simple area approximation.
    xs = (np.arange(cols, dtype=float) + 0.5) * length / cols
    ys = (np.arange(rows, dtype=float) + 0.5) * width / rows
    xx, yy = np.meshgrid(xs, ys)

    field = np.column_stack([xx.ravel(), yy.ravel()])
    normalized = np.column_stack(
        [field[:, 0] / length, field[:, 1] / width]
    )
    return field, normalized


def _inside(
    points: np.ndarray,
    valid: np.ndarray,
    image_width: float,
    image_height: float,
    margin: float,
) -> np.ndarray:
    result = np.zeros(len(points), dtype=bool)
    if not np.any(valid):
        return result

    p = points[valid]
    inside_valid = (
        (p[:, 0] >= -margin)
        & (p[:, 0] <= image_width + margin)
        & (p[:, 1] >= -margin)
        & (p[:, 1] <= image_height + margin)
    )
    result[np.flatnonzero(valid)] = inside_valid
    return result


def _visible_bounds(
    mask: np.ndarray,
    cols: int,
    rows: int,
    length: float,
    width: float,
) -> dict[str, float]:
    m = mask.reshape(rows, cols)
    yy, xx = np.nonzero(m)

    if len(xx) == 0:
        raise HumanViewDescriptorError(
            "No canonical field area is visible in the viewport."
        )

    min_col, max_col = int(xx.min()), int(xx.max())
    min_row, max_row = int(yy.min()), int(yy.max())

    x0 = min_col * length / cols
    x1 = (max_col + 1) * length / cols
    y0 = min_row * width / rows
    y1 = (max_row + 1) * width / rows

    return {
        "x_min_m": float(x0),
        "x_max_m": float(x1),
        "y_min_m": float(y0),
        "y_max_m": float(y1),
        "x_span_m": float(x1 - x0),
        "y_span_m": float(y1 - y0),
    }


def _occupancy_3x3(
    normalized: np.ndarray,
    visible: np.ndarray,
) -> dict[str, Any]:
    x_names = ("left", "center", "right")
    y_names = ("touchline_a", "center", "touchline_b")

    matrix: list[list[float]] = []
    cells: list[dict[str, Any]] = []

    x = normalized[:, 0]
    y = normalized[:, 1]

    for row in range(3):
        matrix_row: list[float] = []
        for col in range(3):
            x_sel = (
                (x >= col / 3.0)
                & (
                    (x < (col + 1) / 3.0)
                    if col < 2
                    else (x <= 1.0)
                )
            )
            y_sel = (
                (y >= row / 3.0)
                & (
                    (y < (row + 1) / 3.0)
                    if row < 2
                    else (y <= 1.0)
                )
            )

            cell = x_sel & y_sel
            count = int(cell.sum())
            seen = int((visible & cell).sum())
            fraction = float(seen / count) if count else 0.0

            matrix_row.append(fraction)
            cells.append(
                {
                    "x_region": x_names[col],
                    "y_region": y_names[row],
                    "visible_fraction": fraction,
                    "visible_percent": 100.0 * fraction,
                    "visible_sample_count": seen,
                    "sample_count": count,
                }
            )
        matrix.append(matrix_row)

    return {"matrix": matrix, "cells": cells}


# ---------------------------------------------------------------------------
# One-HUMAN descriptor
# ---------------------------------------------------------------------------


def describe_human_view(
    projected_geometry: dict[str, Any],
    *,
    pitch_length: float,
    pitch_width: float,
    video_width: float,
    video_height: float,
    grid_cols: int = DEFAULT_GRID_COLS,
    grid_rows: int = DEFAULT_GRID_ROWS,
    viewport_margin_px: float = DEFAULT_VIEWPORT_MARGIN_PX,
    human_id: str | None = None,
    poor_p95_diagonal_fraction: float = DEFAULT_POOR_P95_DIAGONAL_FRACTION,
) -> dict[str, Any]:
    """Describe one DIRECT or ASSISTED HUMAN from final projected geometry."""

    length = _positive(pitch_length, "pitch_length")
    width = _positive(pitch_width, "pitch_width")
    image_width = _positive(video_width, "video_width")
    image_height = _positive(video_height, "video_height")
    cols = _positive_int(grid_cols, "grid_cols", 3)
    rows = _positive_int(grid_rows, "grid_rows", 3)

    margin = _finite(viewport_margin_px, "viewport_margin_px")
    if margin < 0:
        raise HumanViewDescriptorError(
            "viewport_margin_px must be non-negative."
        )

    threshold = _positive(
        poor_p95_diagonal_fraction,
        "poor_p95_diagonal_fraction",
    )
    diagonal = math.hypot(image_width, image_height)
    pitch = PitchModel(length=length, width=width)

    canonical = _canonical_points(pitch)
    projected = _projected_points(projected_geometry)

    names = [name for name in canonical if name in projected]
    if len(names) < 4:
        raise HumanViewDescriptorError(
            "Final projected geometry has fewer than four usable semantic points."
        )

    field_points = np.asarray([canonical[name] for name in names], dtype=float)
    image_points = np.asarray([projected[name] for name in names], dtype=float)

    # Global fit is diagnostic only in V2.
    global_approximation: dict[str, Any]
    try:
        global_h = _fit_homography(field_points, image_points)
        global_summary = _residual_summary(
            global_h,
            field_points,
            image_points,
            diagonal,
        )
        global_approximation = {
            "available": True,
            "model": "all_semantic_points_homography",
            "matrix": global_h.tolist(),
            "semantic_correspondence_count": len(names),
            "semantic_correspondence_names": names,
            "residual_summary": global_summary,
        }
    except HumanViewDescriptorError as error:
        global_approximation = {
            "available": False,
            "model": "all_semantic_points_homography",
            "semantic_correspondence_count": len(names),
            "semantic_correspondence_names": names,
            "error": str(error),
        }

    # Local viewport fit is the descriptor model used by coverage/scale.
    local_fit = _select_local_fit(
        names,
        field_points,
        image_points,
        image_width,
        image_height,
        diagonal,
    )
    local_h = local_fit["matrix"]
    local_indices = np.asarray(local_fit["indices"], dtype=int)
    local_names = [names[index] for index in local_indices]

    local_summary = _residual_summary(
        local_h,
        field_points[local_indices],
        image_points[local_indices],
        diagonal,
    )

    # Also report how the local H extrapolates to all semantic landmarks.  This
    # is diagnostic only and may be intentionally very poor in tight side views.
    local_all_summary = _residual_summary(
        local_h,
        field_points,
        image_points,
        diagonal,
    )

    approximation = {
        "model": "viewport_local_semantic_homography",
        "scope": "local_viewport",
        "matrix": local_h.tolist(),
        "selection_mode": local_fit["selection_mode"],
        "fit_margin_px": float(local_fit["fit_margin_px"]),
        "fit_margin_diagonal_fraction": float(
            local_fit["fit_margin_diagonal_fraction"]
        ),
        "visible_semantic_count": int(
            local_fit["visible_semantic_count"]
        ),
        "semantic_correspondence_count": len(local_names),
        "semantic_correspondence_names": local_names,
        "residual_summary": local_summary,
        "all_semantic_extrapolation_residual_summary": local_all_summary,
        "poor_p95_diagonal_fraction_threshold": threshold,
    }

    warnings: list[str] = []

    if local_summary["p95_diagonal_fraction"] > threshold:
        warnings.append(
            "Local viewport geometry is not well represented by one homography; "
            "FIELD coverage and effective scale should be used cautiously."
        )

    global_summary_for_warning = (
        global_approximation.get("residual_summary")
        if global_approximation.get("available")
        else None
    )
    if (
        global_summary_for_warning
        and global_summary_for_warning["p95_diagonal_fraction"] > threshold
    ):
        if local_summary["p95_diagonal_fraction"] <= threshold:
            warnings.append(
                "Global pitch extrapolation is strongly non-projective, but the "
                "visible/near-viewport geometry has a usable local fit. Descriptor "
                "coverage/scale use the local fit."
            )
        else:
            warnings.append(
                "Both local and global homography approximations are weak for this HUMAN."
            )

    # FIELD coverage uses the LOCAL model.
    field_grid, normalized_grid = _grid(length, width, cols, rows)
    projected_grid, valid_grid = _safe_apply_h(local_h, field_grid)

    visible = _inside(
        projected_grid,
        valid_grid,
        image_width,
        image_height,
        margin,
    )

    visible_count = int(visible.sum())
    total_count = len(visible)

    if visible_count == 0:
        raise HumanViewDescriptorError(
            "No meaningful canonical field area falls inside the viewport "
            "under the local descriptor approximation."
        )

    fraction = visible_count / total_count
    visible_field = field_grid[visible]
    centroid = visible_field.mean(axis=0)
    centroid_norm = [
        float(centroid[0] / length),
        float(centroid[1] / width),
    ]

    # Effective view scale is local px/m at the visible-field centroid.
    jacobian = _jacobian(
        local_h,
        float(centroid[0]),
        float(centroid[1]),
    )
    axis_x = jacobian[:, 0]
    axis_y = jacobian[:, 1]

    sx = float(np.linalg.norm(axis_x))
    sy = float(np.linalg.norm(axis_y))
    if min(sx, sy) < EPS:
        raise HumanViewDescriptorError(
            "Effective view scale is degenerate."
        )

    sg = math.sqrt(sx * sy)

    # Semantic visibility always uses the ACTUAL final projected geometry,
    # not either homography approximation.
    canonical_names = [point.name for point in pitch.points]
    available: list[str] = []
    visible_landmarks: list[str] = []

    for name in canonical_names:
        p = projected.get(name)
        if p is None:
            continue
        available.append(name)
        if (
            -margin <= p[0] <= image_width + margin
            and -margin <= p[1] <= image_height + margin
        ):
            visible_landmarks.append(name)

    mask = visible.reshape(rows, cols)

    return {
        "descriptor_version": 2,
        "purpose": (
            "Diagnostic HUMAN viewport descriptor. FIELD coverage and effective "
            "scale use a viewport-local homography; effective scale is not "
            "physical optical zoom."
        ),
        "human_id": human_id,
        "field_profile": {
            "length": length,
            "width": width,
        },
        "image": {
            "width": image_width,
            "height": image_height,
            "diagonal": diagonal,
        },
        # Keep this key backward-compatible with the current TEST UI.
        "approximation": approximation,
        "global_approximation": global_approximation,
        "coverage": {
            "visible_field_fraction": float(fraction),
            "visible_field_percent": float(100.0 * fraction),
            "visible_sample_count": visible_count,
            "total_sample_count": total_count,
            "grid": {
                "cols": cols,
                "rows": rows,
                "cell_center_sampling": True,
                "mask": mask.tolist(),
            },
            "visible_bounds_m": _visible_bounds(
                visible,
                cols,
                rows,
                length,
                width,
            ),
            "visible_centroid_m": [
                float(centroid[0]),
                float(centroid[1]),
            ],
            "visible_centroid_normalized": centroid_norm,
            "view_center_normalized": centroid_norm,
        },
        "effective_view_scale": {
            "evaluation_field_point_m": [
                float(centroid[0]),
                float(centroid[1]),
            ],
            "scale_x_px_per_m": sx,
            "scale_y_px_per_m": sy,
            "scale_geometric_px_per_m": sg,
            "scale_x_normalized_by_image_diagonal": sx / diagonal,
            "scale_y_normalized_by_image_diagonal": sy / diagonal,
            "scale_geometric_normalized_by_image_diagonal": sg / diagonal,
            "anisotropy_ratio": float(max(sx, sy) / min(sx, sy)),
            "axis_x_angle_deg": float(
                math.degrees(math.atan2(axis_x[1], axis_x[0]))
            ),
            "axis_y_angle_deg": float(
                math.degrees(math.atan2(axis_y[1], axis_y[0]))
            ),
            "local_jacobian_px_per_m": jacobian.tolist(),
        },
        "semantic_visibility": {
            "visible_landmark_names": visible_landmarks,
            "visible_landmark_count": len(visible_landmarks),
            "available_projected_landmark_count": len(available),
            "total_landmark_count": len(canonical_names),
            "viewport_margin_px": margin,
        },
        "occupancy_3x3": _occupancy_3x3(
            normalized_grid,
            visible,
        ),
        "warnings": warnings,
    }


def describe_human_view_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Stateless payload wrapper used by the FastAPI endpoint."""

    if not isinstance(payload, dict):
        raise HumanViewDescriptorError("Payload must be an object.")

    field_profile = payload.get("field_profile") or {}

    return describe_human_view(
        payload.get("projected_geometry"),
        pitch_length=payload.get(
            "pitch_length",
            field_profile.get("length"),
        ),
        pitch_width=payload.get(
            "pitch_width",
            field_profile.get("width"),
        ),
        video_width=payload.get("video_width"),
        video_height=payload.get("video_height"),
        grid_cols=payload.get(
            "grid_cols",
            DEFAULT_GRID_COLS,
        ),
        grid_rows=payload.get(
            "grid_rows",
            DEFAULT_GRID_ROWS,
        ),
        viewport_margin_px=payload.get(
            "viewport_margin_px",
            DEFAULT_VIEWPORT_MARGIN_PX,
        ),
        human_id=payload.get("human_id") or payload.get("state_id"),
        poor_p95_diagonal_fraction=payload.get(
            "poor_p95_diagonal_fraction",
            DEFAULT_POOR_P95_DIAGONAL_FRACTION,
        ),
    )


# ---------------------------------------------------------------------------
# Human Atlas FIELD coverage
# ---------------------------------------------------------------------------


def _descriptor_mask(
    descriptor: dict[str, Any],
) -> np.ndarray:
    try:
        grid = descriptor["coverage"]["grid"]
        mask = np.asarray(grid["mask"], dtype=bool)
        rows = int(grid["rows"])
        cols = int(grid["cols"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HumanViewDescriptorError(
            "Descriptor is missing a valid coverage grid."
        ) from exc

    if mask.shape != (rows, cols):
        raise HumanViewDescriptorError(
            f"Coverage mask shape {mask.shape} does not match declared {(rows, cols)}."
        )
    return mask


def _descriptor_field_profile(
    descriptor: dict[str, Any],
) -> tuple[float, float]:
    try:
        profile = descriptor["field_profile"]
        return (
            _positive(
                profile["length"],
                "descriptor field length",
            ),
            _positive(
                profile["width"],
                "descriptor field width",
            ),
        )
    except (KeyError, TypeError) as exc:
        raise HumanViewDescriptorError(
            "Descriptor is missing field_profile."
        ) from exc


def summarize_human_atlas(
    descriptors: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Union/marginal FIELD coverage for compatible HUMAN descriptors."""

    if (
        not isinstance(descriptors, Sequence)
        or isinstance(descriptors, (str, bytes))
        or len(descriptors) == 0
    ):
        raise HumanViewDescriptorError(
            "At least one HUMAN descriptor is required."
        )

    first = descriptors[0]
    length, width = _descriptor_field_profile(first)
    first_mask = _descriptor_mask(first)
    rows, cols = first_mask.shape

    union = np.zeros_like(first_mask, dtype=bool)
    cover_count = np.zeros_like(first_mask, dtype=np.int32)

    marginal: list[dict[str, Any]] = []
    individual: list[dict[str, Any]] = []
    total_cells = int(rows * cols)

    for index, descriptor in enumerate(descriptors):
        d_length, d_width = _descriptor_field_profile(descriptor)

        if (
            abs(d_length - length) > 1e-6
            or abs(d_width - width) > 1e-6
        ):
            raise HumanViewDescriptorError(
                "Human Atlas descriptors must share one fixed FieldProfile."
            )

        mask = _descriptor_mask(descriptor)
        if mask.shape != (rows, cols):
            raise HumanViewDescriptorError(
                "Human Atlas descriptors must use the same coverage-grid shape."
            )

        human_id = (
            descriptor.get("human_id")
            or f"human_{index + 1}"
        )

        visible_count = int(np.count_nonzero(mask))
        before_count = int(np.count_nonzero(union))

        newly_covered = mask & ~union
        new_count = int(np.count_nonzero(newly_covered))

        union |= mask
        cover_count += mask.astype(np.int32)

        after_count = int(np.count_nonzero(union))

        individual_fraction = visible_count / total_cells
        marginal_fraction = new_count / total_cells
        cumulative_fraction = after_count / total_cells

        individual.append(
            {
                "human_id": human_id,
                "visible_field_fraction": float(
                    individual_fraction
                ),
                "visible_field_percent": float(
                    100.0 * individual_fraction
                ),
            }
        )

        marginal.append(
            {
                "human_id": human_id,
                "input_index": index,
                "newly_covered_sample_count": new_count,
                "marginal_field_coverage_fraction": float(
                    marginal_fraction
                ),
                "marginal_field_coverage_percent": float(
                    100.0 * marginal_fraction
                ),
                "cumulative_field_coverage_fraction": float(
                    cumulative_fraction
                ),
                "cumulative_field_coverage_percent": float(
                    100.0 * cumulative_fraction
                ),
                "previous_union_sample_count": before_count,
                "cumulative_union_sample_count": after_count,
            }
        )

    union_count = int(np.count_nonzero(union))
    union_fraction = union_count / total_cells

    return {
        "atlas_version": 1,
        "purpose": (
            "Geometric FIELD-area coverage of the HUMAN atlas. "
            "This is not Camera-State or video-state coverage."
        ),
        "human_count": len(descriptors),
        "field_profile": {
            "length": length,
            "width": width,
        },
        "grid": {
            "cols": cols,
            "rows": rows,
            "union_mask": union.tolist(),
            "coverage_count": cover_count.tolist(),
            "total_sample_count": total_cells,
            "covered_sample_count": union_count,
        },
        "accumulated_field_coverage_fraction": float(
            union_fraction
        ),
        "accumulated_field_coverage_percent": float(
            100.0 * union_fraction
        ),
        "individual_humans": individual,
        "marginal_coverage": marginal,
    }


def summarize_human_atlas_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Stateless payload wrapper used by the FastAPI endpoint."""

    if not isinstance(payload, dict):
        raise HumanViewDescriptorError(
            "Payload must be an object."
        )
    return summarize_human_atlas(
        payload.get("descriptors"),
    )


# ---------------------------------------------------------------------------
# Synthetic self-check
# ---------------------------------------------------------------------------


def _synthetic_matrix(
    *,
    scale_px_per_m: float,
    field_center_x: float,
    field_center_y: float,
    image_width: float,
    image_height: float,
) -> np.ndarray:
    scale = float(scale_px_per_m)
    return _normalize_h(
        np.array(
            [
                [
                    scale,
                    0.0,
                    image_width / 2.0
                    - scale * field_center_x,
                ],
                [
                    0.0,
                    scale,
                    image_height / 2.0
                    - scale * field_center_y,
                ],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    )


def _synthetic_projected_geometry(
    pitch: PitchModel,
    matrix: np.ndarray,
    *,
    image_width: float,
    image_height: float,
    non_projective_strength: float = 0.0,
    corrupt_far_left: bool = False,
) -> dict[str, Any]:
    canonical = np.asarray(
        [
            [float(p.point.x), float(p.point.y)]
            for p in pitch.points
        ],
        dtype=float,
    )
    projected = _apply_h(matrix, canonical)

    if non_projective_strength:
        cx = image_width / 2.0
        cy = image_height / 2.0
        diagonal = math.hypot(image_width, image_height)

        shifted = projected - np.array([cx, cy], dtype=float)
        nx = shifted[:, 0] / diagonal
        ny = shifted[:, 1] / diagonal
        r2 = nx * nx + ny * ny

        projected[:, 0] += (
            non_projective_strength
            * diagonal
            * nx
            * r2
        )
        projected[:, 1] += (
            -0.65
            * non_projective_strength
            * diagonal
            * ny
            * r2
        )

    if corrupt_far_left:
        # Mimic the real failure mode: the visible right-side geometry remains
        # good while opposite-side extrapolated landmarks become absurdly far.
        far_left = canonical[:, 0] < 0.45 * pitch.length
        projected[far_left, 0] -= (
            12000.0
            + 180.0 * (0.45 * pitch.length - canonical[far_left, 0])
        )
        projected[far_left, 1] += (
            5000.0
            + 80.0 * canonical[far_left, 1]
        )

    return {
        "points": [
            {
                "name": p.name,
                "label": p.label,
                "category": p.category,
                "point": [
                    float(image_point[0]),
                    float(image_point[1]),
                ],
            }
            for p, image_point in zip(
                pitch.points,
                projected,
            )
        ],
        "lines": [],
        "curves": [],
    }


def _self_check() -> None:
    image_width = 1920.0
    image_height = 1080.0
    length = 105.0
    width = 68.0

    pitch = PitchModel(
        length=length,
        width=width,
    )

    wide_h = _synthetic_matrix(
        scale_px_per_m=18.0,
        field_center_x=length / 2.0,
        field_center_y=width / 2.0,
        image_width=image_width,
        image_height=image_height,
    )
    tight_h = _synthetic_matrix(
        scale_px_per_m=28.0,
        field_center_x=length / 2.0,
        field_center_y=width / 2.0,
        image_width=image_width,
        image_height=image_height,
    )
    left_h = _synthetic_matrix(
        scale_px_per_m=24.0,
        field_center_x=0.24 * length,
        field_center_y=width / 2.0,
        image_width=image_width,
        image_height=image_height,
    )
    right_h = _synthetic_matrix(
        scale_px_per_m=24.0,
        field_center_x=0.76 * length,
        field_center_y=width / 2.0,
        image_width=image_width,
        image_height=image_height,
    )

    def descriptor(
        human_id: str,
        matrix: np.ndarray,
        **geometry_options: Any,
    ) -> dict[str, Any]:
        return describe_human_view(
            _synthetic_projected_geometry(
                pitch,
                matrix,
                image_width=image_width,
                image_height=image_height,
                **geometry_options,
            ),
            pitch_length=length,
            pitch_width=width,
            video_width=image_width,
            video_height=image_height,
            human_id=human_id,
        )

    wide = descriptor("wide", wide_h)
    tight = descriptor("tight", tight_h)
    left = descriptor("left", left_h)
    right = descriptor("right", right_h)

    wide_center = wide["coverage"][
        "visible_centroid_normalized"
    ]
    tight_center = tight["coverage"][
        "visible_centroid_normalized"
    ]

    if (
        abs(wide_center[0] - 0.5) > 0.03
        or abs(wide_center[1] - 0.5) > 0.03
    ):
        raise AssertionError(
            ("wide center", wide_center)
        )

    if (
        abs(tight_center[0] - 0.5) > 0.03
        or abs(tight_center[1] - 0.5) > 0.03
    ):
        raise AssertionError(
            ("tight center", tight_center)
        )

    if not (
        tight["coverage"]["visible_field_fraction"]
        < wide["coverage"]["visible_field_fraction"]
    ):
        raise AssertionError(
            "Tighter view did not reduce visible field coverage."
        )

    if not (
        tight["effective_view_scale"][
            "scale_geometric_px_per_m"
        ]
        > wide["effective_view_scale"][
            "scale_geometric_px_per_m"
        ]
    ):
        raise AssertionError(
            "Tighter view did not increase effective px/m scale."
        )

    if not (
        left["coverage"][
            "visible_centroid_normalized"
        ][0]
        < wide["coverage"][
            "visible_centroid_normalized"
        ][0]
        < right["coverage"][
            "visible_centroid_normalized"
        ][0]
    ):
        raise AssertionError(
            "Panned views did not move centroid in expected x direction."
        )

    atlas_lr = summarize_human_atlas(
        [left, right]
    )
    if (
        atlas_lr[
            "accumulated_field_coverage_fraction"
        ]
        + 1e-12
        < max(
            left["coverage"][
                "visible_field_fraction"
            ],
            right["coverage"][
                "visible_field_fraction"
            ],
        )
    ):
        raise AssertionError(
            "Atlas union coverage is below an individual HUMAN."
        )

    duplicate_left = dict(left)
    duplicate_left["human_id"] = "left_duplicate"
    redundant = summarize_human_atlas(
        [left, duplicate_left]
    )
    redundant_gain = redundant[
        "marginal_coverage"
    ][1]["marginal_field_coverage_fraction"]

    if redundant_gain > 1e-12:
        raise AssertionError(
            (
                "Redundant HUMAN gained unexpected coverage",
                redundant_gain,
            )
        )

    mixed = dict(right)
    mixed["field_profile"] = {
        "length": 110.0,
        "width": width,
    }

    mixed_rejected = False
    try:
        summarize_human_atlas(
            [left, mixed]
        )
    except HumanViewDescriptorError:
        mixed_rejected = True

    if not mixed_rejected:
        raise AssertionError(
            "Mixed FieldProfiles were not rejected."
        )

    # Mild smooth non-projective distortion: local P95 should increase but
    # descriptor should remain valid.
    distorted = descriptor(
        "wide_distorted",
        wide_h,
        non_projective_strength=0.055,
    )
    pure_local_p95 = wide[
        "approximation"
    ]["residual_summary"]["p95_px"]
    distorted_local_p95 = distorted[
        "approximation"
    ]["residual_summary"]["p95_px"]

    if not (
        distorted_local_p95
        > pure_local_p95 + 0.05
    ):
        raise AssertionError(
            (
                "Non-projective distortion did not raise local residual",
                pure_local_p95,
                distorted_local_p95,
            )
        )

    # Critical V2 case: tight/right view with absurd opposite-side extrapolation.
    right_corrupt = descriptor(
        "right_corrupt",
        right_h,
        corrupt_far_left=True,
    )

    local_p95 = right_corrupt[
        "approximation"
    ]["residual_summary"]["p95_px"]
    global_info = right_corrupt[
        "global_approximation"
    ]

    if not global_info.get("available"):
        raise AssertionError(
            "Synthetic corrupt case unexpectedly lost global diagnostic."
        )

    global_p95 = global_info[
        "residual_summary"
    ]["p95_px"]

    if local_p95 > 5.0:
        raise AssertionError(
            (
                "Viewport-local fit was contaminated by far extrapolation",
                local_p95,
            )
        )

    if global_p95 < 100.0:
        raise AssertionError(
            (
                "Synthetic corrupt case did not produce poor global fit",
                global_p95,
            )
        )

    corrupt_center_x = right_corrupt[
        "coverage"
    ]["visible_centroid_normalized"][0]
    if corrupt_center_x < 0.62:
        raise AssertionError(
            (
                "Corrupt right view lost expected rightward center",
                corrupt_center_x,
            )
        )

    corrupt_scale = right_corrupt[
        "effective_view_scale"
    ]["scale_geometric_px_per_m"]
    if corrupt_scale < 20.0:
        raise AssertionError(
            (
                "Corrupt right view lost expected tight effective scale",
                corrupt_scale,
            )
        )

    print("HUMAN view descriptor V2 self-check: OK")
    print(
        "Wide:   "
        f"coverage {wide['coverage']['visible_field_percent']:.1f}% · "
        f"center {wide_center[0]:.3f}, {wide_center[1]:.3f} · "
        f"scale {wide['effective_view_scale']['scale_geometric_px_per_m']:.2f} px/m · "
        f"local P95 {pure_local_p95:.4f} px"
    )
    print(
        "Tight:  "
        f"coverage {tight['coverage']['visible_field_percent']:.1f}% · "
        f"center {tight_center[0]:.3f}, {tight_center[1]:.3f} · "
        f"scale {tight['effective_view_scale']['scale_geometric_px_per_m']:.2f} px/m"
    )
    print(
        "Pan X:  "
        f"left {left['coverage']['visible_centroid_normalized'][0]:.3f} · "
        f"right {right['coverage']['visible_centroid_normalized'][0]:.3f}"
    )
    print(
        "Atlas:  "
        f"left+right {atlas_lr['accumulated_field_coverage_percent']:.1f}% · "
        f"duplicate marginal +{100.0 * redundant_gain:.3f}%"
    )
    print(
        "Warp:   "
        f"pure local P95 {pure_local_p95:.4f} px -> "
        f"distorted local P95 {distorted_local_p95:.4f} px"
    )
    print(
        "Far extrapolation: "
        f"local P95 {local_p95:.4f} px · "
        f"global P95 {global_p95:.1f} px · "
        f"center_x {corrupt_center_x:.3f} · "
        f"scale {corrupt_scale:.2f} px/m"
    )
    print("Mixed-pitch rejection: OK")


if __name__ == "__main__":
    _self_check()

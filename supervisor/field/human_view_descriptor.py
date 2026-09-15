"""HUMAN view descriptor and Human Atlas coverage diagnostics for RealJaenLab FIELD.

The descriptor is derived from the final calibrated ``projected_geometry`` of a
HUMAN, not from the sparse clicks that created it. It measures geometric field
coverage, visible field centroid, effective image scale, semantic landmark
visibility and coarse 3x3 occupancy. It also combines several HUMAN descriptors
into a cumulative FIELD-coverage atlas.

Important: the fitted homography below is only a diagnostic approximation. The
actual HUMAN projection may include anisotropic correction and/or DERIVED
refinement and therefore need not be one perfect homography.
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
DEFAULT_POOR_P95_DIAGONAL_FRACTION = 0.005


class HumanViewDescriptorError(ValueError):
    """Invalid or insufficient input for a HUMAN view descriptor."""


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
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
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


def _apply_h(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise HumanViewDescriptorError("Points must be an Nx2 array.")
    hom = np.column_stack([pts, np.ones(len(pts), dtype=float)])
    out = (_normalize_h(matrix) @ hom.T).T
    w = out[:, 2]
    if np.any(np.abs(w) < EPS):
        raise HumanViewDescriptorError("Homography maps a sample to infinity.")
    xy = out[:, :2] / w[:, None]
    if not np.all(np.isfinite(xy)):
        raise HumanViewDescriptorError("Homography produced non-finite coordinates.")
    return xy


def _hartley(points: np.ndarray) -> np.ndarray:
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_distance = float(np.mean(np.linalg.norm(shifted, axis=1)))
    if mean_distance < EPS:
        raise HumanViewDescriptorError("Homography samples have no spatial extent.")
    scale = math.sqrt(2.0) / mean_distance
    return np.array(
        [[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def _fit_homography(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise HumanViewDescriptorError("Homography source/target must be matching Nx2 arrays.")
    if len(source) < 4:
        raise HumanViewDescriptorError("At least four semantic correspondences are required.")
    ts, tt = _hartley(source), _hartley(target)
    src_n, dst_n = _apply_h(ts, source), _apply_h(tt, target)
    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    a = np.asarray(rows, dtype=float)
    _, singular_values, vt = np.linalg.svd(a)
    if singular_values.size < 2 or singular_values[-2] < EPS:
        raise HumanViewDescriptorError("Semantic correspondences are degenerate.")
    h = np.linalg.inv(tt) @ vt[-1].reshape(3, 3) @ ts
    return _normalize_h(h)


def _summary(errors: np.ndarray, diagonal: float) -> dict[str, float]:
    arr = np.asarray(errors, dtype=float)
    median = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    maximum = float(np.max(arr))
    rmse = float(np.sqrt(np.mean(arr * arr)))
    diagonal = max(diagonal, EPS)
    return {
        "median_px": median,
        "p95_px": p95,
        "max_px": maximum,
        "rmse_px": rmse,
        "median_diagonal_fraction": median / diagonal,
        "p95_diagonal_fraction": p95 / diagonal,
        "max_diagonal_fraction": maximum / diagonal,
    }


def _jacobian(h: np.ndarray, x: float, y: float) -> np.ndarray:
    h = _normalize_h(h)
    d = float(h[2, 0] * x + h[2, 1] * y + h[2, 2])
    if abs(d) < EPS:
        raise HumanViewDescriptorError("Cannot evaluate local scale near a mapping singularity.")
    nu = float(h[0, 0] * x + h[0, 1] * y + h[0, 2])
    nv = float(h[1, 0] * x + h[1, 1] * y + h[1, 2])
    d2 = d * d
    return np.array([
        [(h[0, 0] * d - nu * h[2, 0]) / d2, (h[0, 1] * d - nu * h[2, 1]) / d2],
        [(h[1, 0] * d - nv * h[2, 0]) / d2, (h[1, 1] * d - nv * h[2, 1]) / d2],
    ], dtype=float)


def _canonical_points(pitch: PitchModel) -> dict[str, tuple[float, float]]:
    return {p.name: (float(p.point.x), float(p.point.y)) for p in pitch.points}


def _projected_points(geometry: dict[str, Any]) -> dict[str, tuple[float, float]]:
    if not isinstance(geometry, dict):
        raise HumanViewDescriptorError("projected_geometry must be an object.")
    result: dict[str, tuple[float, float]] = {}
    for item in geometry.get("points", []) or []:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        try:
            result[item["name"]] = _point(item.get("point"), f"projected point {item['name']}")
        except HumanViewDescriptorError:
            continue
    return result


def _grid(length: float, width: float, cols: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    # Cell centres make visible-cell fraction a simple area approximation.
    xs = (np.arange(cols, dtype=float) + 0.5) * length / cols
    ys = (np.arange(rows, dtype=float) + 0.5) * width / rows
    xx, yy = np.meshgrid(xs, ys)
    field = np.column_stack([xx.ravel(), yy.ravel()])
    normalized = np.column_stack([field[:, 0] / length, field[:, 1] / width])
    return field, normalized


def _inside(points: np.ndarray, image_width: float, image_height: float, margin: float) -> np.ndarray:
    return (
        (points[:, 0] >= -margin) & (points[:, 0] <= image_width + margin)
        & (points[:, 1] >= -margin) & (points[:, 1] <= image_height + margin)
    )


def _visible_bounds(mask: np.ndarray, cols: int, rows: int, length: float, width: float) -> dict[str, float]:
    m = mask.reshape(rows, cols)
    yy, xx = np.nonzero(m)
    if len(xx) == 0:
        raise HumanViewDescriptorError("No canonical field area is visible in the viewport.")
    min_col, max_col = int(xx.min()), int(xx.max())
    min_row, max_row = int(yy.min()), int(yy.max())
    x0, x1 = min_col * length / cols, (max_col + 1) * length / cols
    y0, y1 = min_row * width / rows, (max_row + 1) * width / rows
    return {
        "x_min_m": float(x0), "x_max_m": float(x1),
        "y_min_m": float(y0), "y_max_m": float(y1),
        "x_span_m": float(x1 - x0), "y_span_m": float(y1 - y0),
    }


def _occupancy_3x3(normalized: np.ndarray, visible: np.ndarray) -> dict[str, Any]:
    x_names = ("left", "center", "right")
    y_names = ("touchline_a", "center", "touchline_b")
    matrix: list[list[float]] = []
    cells: list[dict[str, Any]] = []
    x, y = normalized[:, 0], normalized[:, 1]
    for row in range(3):
        matrix_row: list[float] = []
        for col in range(3):
            x_sel = (x >= col / 3) & ((x < (col + 1) / 3) if col < 2 else (x <= 1))
            y_sel = (y >= row / 3) & ((y < (row + 1) / 3) if row < 2 else (y <= 1))
            cell = x_sel & y_sel
            count = int(cell.sum())
            seen = int((visible & cell).sum())
            fraction = float(seen / count) if count else 0.0
            matrix_row.append(fraction)
            cells.append({
                "x_region": x_names[col], "y_region": y_names[row],
                "visible_fraction": fraction, "visible_percent": 100 * fraction,
                "visible_sample_count": seen, "sample_count": count,
            })
        matrix.append(matrix_row)
    return {"matrix": matrix, "cells": cells}


def describe_human_view(
    projected_geometry: dict[str, Any], *, pitch_length: float, pitch_width: float,
    video_width: float, video_height: float, grid_cols: int = DEFAULT_GRID_COLS,
    grid_rows: int = DEFAULT_GRID_ROWS, viewport_margin_px: float = DEFAULT_VIEWPORT_MARGIN_PX,
    human_id: str | None = None,
    poor_p95_diagonal_fraction: float = DEFAULT_POOR_P95_DIAGONAL_FRACTION,
) -> dict[str, Any]:
    """Describe one DIRECT or ASSISTED HUMAN from its final projected geometry."""
    length, width = _positive(pitch_length, "pitch_length"), _positive(pitch_width, "pitch_width")
    image_width, image_height = _positive(video_width, "video_width"), _positive(video_height, "video_height")
    cols, rows = _positive_int(grid_cols, "grid_cols", 3), _positive_int(grid_rows, "grid_rows", 3)
    margin = _finite(viewport_margin_px, "viewport_margin_px")
    if margin < 0:
        raise HumanViewDescriptorError("viewport_margin_px must be non-negative.")
    diagonal = math.hypot(image_width, image_height)
    pitch = PitchModel(length=length, width=width)

    canonical, projected = _canonical_points(pitch), _projected_points(projected_geometry)
    names = [name for name in canonical if name in projected]
    if len(names) < 4:
        raise HumanViewDescriptorError("Final projected geometry has fewer than four usable semantic points.")
    field_points = np.asarray([canonical[n] for n in names], dtype=float)
    image_points = np.asarray([projected[n] for n in names], dtype=float)
    h = _fit_homography(field_points, image_points)
    residuals = np.linalg.norm(_apply_h(h, field_points) - image_points, axis=1)
    approximation = _summary(residuals, diagonal)

    threshold = _positive(poor_p95_diagonal_fraction, "poor_p95_diagonal_fraction")
    warnings: list[str] = []
    if approximation["p95_diagonal_fraction"] > threshold:
        warnings.append(
            "Final projected geometry is not well represented by one homography; coverage/scale are approximate."
        )

    field_grid, normalized_grid = _grid(length, width, cols, rows)
    projected_grid = _apply_h(h, field_grid)
    visible = _inside(projected_grid, image_width, image_height, margin)
    visible_count, total_count = int(visible.sum()), len(visible)
    if visible_count == 0:
        raise HumanViewDescriptorError("No meaningful canonical field area falls inside the viewport.")
    fraction = visible_count / total_count
    visible_field = field_grid[visible]
    centroid = visible_field.mean(axis=0)
    centroid_norm = [float(centroid[0] / length), float(centroid[1] / width)]

    j = _jacobian(h, float(centroid[0]), float(centroid[1]))
    axis_x, axis_y = j[:, 0], j[:, 1]
    sx, sy = float(np.linalg.norm(axis_x)), float(np.linalg.norm(axis_y))
    if min(sx, sy) < EPS:
        raise HumanViewDescriptorError("Effective view scale is degenerate.")
    sg = math.sqrt(sx * sy)

    canonical_names = [p.name for p in pitch.points]
    available, visible_landmarks = [], []
    for name in canonical_names:
        p = projected.get(name)
        if p is None:
            continue
        available.append(name)
        if -margin <= p[0] <= image_width + margin and -margin <= p[1] <= image_height + margin:
            visible_landmarks.append(name)

    mask = visible.reshape(rows, cols)
    return {
        "descriptor_version": 1,
        "purpose": "Diagnostic HUMAN viewport descriptor; effective scale is not physical optical zoom.",
        "human_id": human_id,
        "field_profile": {"length": length, "width": width},
        "image": {"width": image_width, "height": image_height, "diagonal": diagonal},
        "approximation": {
            "model": "semantic_points_homography", "matrix": h.tolist(),
            "semantic_correspondence_count": len(names), "semantic_correspondence_names": names,
            "residual_summary": approximation, "poor_p95_diagonal_fraction_threshold": threshold,
        },
        "coverage": {
            "visible_field_fraction": float(fraction), "visible_field_percent": float(100 * fraction),
            "visible_sample_count": visible_count, "total_sample_count": int(total_count),
            "grid": {"cols": cols, "rows": rows, "cell_center_sampling": True, "mask": mask.tolist()},
            "visible_bounds_m": _visible_bounds(visible, cols, rows, length, width),
            "visible_centroid_m": [float(centroid[0]), float(centroid[1])],
            "visible_centroid_normalized": centroid_norm,
            "view_center_normalized": centroid_norm,
        },
        "effective_view_scale": {
            "evaluation_field_point_m": [float(centroid[0]), float(centroid[1])],
            "scale_x_px_per_m": sx, "scale_y_px_per_m": sy, "scale_geometric_px_per_m": sg,
            "scale_x_normalized_by_image_diagonal": sx / diagonal,
            "scale_y_normalized_by_image_diagonal": sy / diagonal,
            "scale_geometric_normalized_by_image_diagonal": sg / diagonal,
            "anisotropy_ratio": max(sx, sy) / min(sx, sy),
            "axis_x_angle_deg": float(math.degrees(math.atan2(axis_x[1], axis_x[0]))),
            "axis_y_angle_deg": float(math.degrees(math.atan2(axis_y[1], axis_y[0]))),
            "local_jacobian_px_per_m": j.tolist(),
        },
        "semantic_visibility": {
            "visible_landmark_names": visible_landmarks,
            "visible_landmark_count": len(visible_landmarks),
            "available_projected_landmark_count": len(available),
            "total_landmark_count": len(canonical_names),
            "viewport_margin_px": margin,
        },
        "occupancy_3x3": _occupancy_3x3(normalized_grid, visible),
        "warnings": warnings,
    }


def describe_human_view_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HumanViewDescriptorError("Payload must be an object.")
    profile = payload.get("field_profile") or {}
    return describe_human_view(
        payload.get("projected_geometry"),
        pitch_length=payload.get("pitch_length", profile.get("length")),
        pitch_width=payload.get("pitch_width", profile.get("width")),
        video_width=payload.get("video_width"), video_height=payload.get("video_height"),
        grid_cols=payload.get("grid_cols", DEFAULT_GRID_COLS), grid_rows=payload.get("grid_rows", DEFAULT_GRID_ROWS),
        viewport_margin_px=payload.get("viewport_margin_px", DEFAULT_VIEWPORT_MARGIN_PX),
        human_id=payload.get("human_id") or payload.get("state_id"),
        poor_p95_diagonal_fraction=payload.get("poor_p95_diagonal_fraction", DEFAULT_POOR_P95_DIAGONAL_FRACTION),
    )


def _mask(descriptor: dict[str, Any]) -> np.ndarray:
    try:
        grid = descriptor["coverage"]["grid"]
        mask = np.asarray(grid["mask"], dtype=bool)
        shape = (int(grid["rows"]), int(grid["cols"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise HumanViewDescriptorError("Descriptor is missing a valid coverage grid.") from exc
    if mask.shape != shape:
        raise HumanViewDescriptorError(f"Coverage mask shape {mask.shape} does not match {shape}.")
    return mask


def _profile(descriptor: dict[str, Any]) -> tuple[float, float]:
    try:
        p = descriptor["field_profile"]
        return _positive(p["length"], "field length"), _positive(p["width"], "field width")
    except (KeyError, TypeError) as exc:
        raise HumanViewDescriptorError("Descriptor is missing field_profile.") from exc


def summarize_human_atlas(descriptors: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Combine compatible HUMAN descriptors into FIELD-area union coverage."""
    if not isinstance(descriptors, Sequence) or isinstance(descriptors, (str, bytes)) or not descriptors:
        raise HumanViewDescriptorError("At least one HUMAN descriptor is required.")
    length, width = _profile(descriptors[0])
    first_mask = _mask(descriptors[0])
    rows, cols = first_mask.shape
    union = np.zeros_like(first_mask, dtype=bool)
    cover_count = np.zeros_like(first_mask, dtype=np.int32)
    total = rows * cols
    marginal, individual = [], []

    for index, descriptor in enumerate(descriptors):
        dl, dw = _profile(descriptor)
        if abs(dl - length) > 1e-6 or abs(dw - width) > 1e-6:
            raise HumanViewDescriptorError("Human Atlas descriptors must share one fixed FieldProfile.")
        mask = _mask(descriptor)
        if mask.shape != (rows, cols):
            raise HumanViewDescriptorError("Human Atlas descriptors must share one coverage-grid shape.")
        human_id = descriptor.get("human_id") or f"human_{index + 1}"
        visible_count = int(mask.sum())
        new = mask & ~union
        new_count = int(new.sum())
        union |= mask
        cover_count += mask.astype(np.int32)
        union_count = int(union.sum())
        individual.append({
            "human_id": human_id,
            "visible_field_fraction": visible_count / total,
            "visible_field_percent": 100 * visible_count / total,
        })
        marginal.append({
            "human_id": human_id, "input_index": index,
            "newly_covered_sample_count": new_count,
            "marginal_field_coverage_fraction": new_count / total,
            "marginal_field_coverage_percent": 100 * new_count / total,
            "cumulative_field_coverage_fraction": union_count / total,
            "cumulative_field_coverage_percent": 100 * union_count / total,
        })

    union_count = int(union.sum())
    return {
        "atlas_version": 1,
        "purpose": "Geometric FIELD-area coverage only; this is not Camera-State/video-state coverage.",
        "human_count": len(descriptors),
        "field_profile": {"length": length, "width": width},
        "grid": {
            "cols": cols, "rows": rows, "union_mask": union.tolist(),
            "coverage_count": cover_count.tolist(), "total_sample_count": total,
            "covered_sample_count": union_count,
        },
        "accumulated_field_coverage_fraction": union_count / total,
        "accumulated_field_coverage_percent": 100 * union_count / total,
        "individual_humans": individual,
        "marginal_coverage": marginal,
    }


def summarize_human_atlas_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HumanViewDescriptorError("Payload must be an object.")
    return summarize_human_atlas(payload.get("descriptors"))


# ---------------------------------------------------------------------------
# Synthetic self-check
# ---------------------------------------------------------------------------


def _synthetic_h(scale: float, center_x: float, center_y: float, image_width: float, image_height: float) -> np.ndarray:
    return np.array([
        [scale, 0.0, image_width / 2 - scale * center_x],
        [0.0, scale, image_height / 2 - scale * center_y],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def _synthetic_geometry(pitch: PitchModel, h: np.ndarray, image_width: float, image_height: float, warp: float = 0.0) -> dict[str, Any]:
    names = [p.name for p in pitch.points]
    canonical = np.asarray([[p.point.x, p.point.y] for p in pitch.points], dtype=float)
    projected = _apply_h(h, canonical)
    if warp:
        cx, cy, diagonal = image_width / 2, image_height / 2, math.hypot(image_width, image_height)
        nx, ny = (projected[:, 0] - cx) / diagonal, (projected[:, 1] - cy) / diagonal
        r2 = nx * nx + ny * ny
        projected[:, 0] += warp * diagonal * nx * r2
        projected[:, 1] -= 0.65 * warp * diagonal * ny * r2
    return {"points": [
        {"name": n, "label": n, "category": "synthetic", "point": [float(p[0]), float(p[1])]}
        for n, p in zip(names, projected)
    ], "lines": [], "curves": []}


def _self_check() -> None:
    iw, ih, length, width = 1920.0, 1080.0, 105.0, 68.0
    pitch = PitchModel(length=length, width=width)

    def desc(name: str, scale: float, cx: float, cy: float, warp: float = 0.0) -> dict[str, Any]:
        h = _synthetic_h(scale, cx, cy, iw, ih)
        return describe_human_view(
            _synthetic_geometry(pitch, h, iw, ih, warp), pitch_length=length, pitch_width=width,
            video_width=iw, video_height=ih, human_id=name,
        )

    wide = desc("wide", 18.0, length / 2, width / 2)
    tight = desc("tight", 28.0, length / 2, width / 2)
    left = desc("left", 24.0, 0.24 * length, width / 2)
    right = desc("right", 24.0, 0.76 * length, width / 2)

    wc = wide["coverage"]["visible_centroid_normalized"]
    tc = tight["coverage"]["visible_centroid_normalized"]
    if abs(wc[0] - 0.5) > 0.03 or abs(wc[1] - 0.5) > 0.03:
        raise AssertionError(("wide center", wc))
    if abs(tc[0] - 0.5) > 0.03 or abs(tc[1] - 0.5) > 0.03:
        raise AssertionError(("tight center", tc))
    if tight["coverage"]["visible_field_fraction"] >= wide["coverage"]["visible_field_fraction"]:
        raise AssertionError("Tighter view did not reduce coverage.")
    if tight["effective_view_scale"]["scale_geometric_px_per_m"] <= wide["effective_view_scale"]["scale_geometric_px_per_m"]:
        raise AssertionError("Tighter view did not increase px/m scale.")
    lx = left["coverage"]["visible_centroid_normalized"][0]
    rx = right["coverage"]["visible_centroid_normalized"][0]
    if not lx < wc[0] < rx:
        raise AssertionError(("pan centers", lx, wc[0], rx))
    if len(wide["semantic_visibility"]["visible_landmark_names"]) <= len(tight["semantic_visibility"]["visible_landmark_names"]):
        raise AssertionError("Wide view should expose more semantic landmarks.")
    if "center_spot" not in tight["semantic_visibility"]["visible_landmark_names"]:
        raise AssertionError("Centered tight view should contain center_spot.")

    atlas = summarize_human_atlas([left, right])
    if atlas["accumulated_field_coverage_fraction"] < max(left["coverage"]["visible_field_fraction"], right["coverage"]["visible_field_fraction"]):
        raise AssertionError("Atlas union below individual coverage.")
    duplicate = dict(left); duplicate["human_id"] = "left_duplicate"
    redundant = summarize_human_atlas([left, duplicate])
    if redundant["marginal_coverage"][1]["marginal_field_coverage_fraction"] > 1e-12:
        raise AssertionError("Redundant HUMAN gained coverage.")

    mixed = dict(right); mixed["field_profile"] = {"length": 110.0, "width": width}
    try:
        summarize_human_atlas([left, mixed])
    except HumanViewDescriptorError:
        pass
    else:
        raise AssertionError("Mixed FieldProfiles were not rejected.")

    distorted = desc("wide_distorted", 18.0, length / 2, width / 2, warp=0.055)
    pure_p95 = wide["approximation"]["residual_summary"]["p95_px"]
    distorted_p95 = distorted["approximation"]["residual_summary"]["p95_px"]
    if distorted_p95 <= pure_p95 + 0.05:
        raise AssertionError(("distortion P95", pure_p95, distorted_p95))

    print("HUMAN view descriptor self-check: OK")
    print(f"Wide:   coverage {wide['coverage']['visible_field_percent']:.1f}% · center {wc[0]:.3f}, {wc[1]:.3f} · scale {wide['effective_view_scale']['scale_geometric_px_per_m']:.2f} px/m · P95 {pure_p95:.4f} px")
    print(f"Tight:  coverage {tight['coverage']['visible_field_percent']:.1f}% · center {tc[0]:.3f}, {tc[1]:.3f} · scale {tight['effective_view_scale']['scale_geometric_px_per_m']:.2f} px/m")
    print(f"Pan X:  left {lx:.3f} · right {rx:.3f}")
    print(f"Atlas:  left+right {atlas['accumulated_field_coverage_percent']:.1f}% · duplicate marginal +{redundant['marginal_coverage'][1]['marginal_field_coverage_percent']:.3f}%")
    print(f"Warp:   pure P95 {pure_p95:.4f} px -> distorted P95 {distorted_p95:.4f} px")
    print("Mixed-pitch rejection: OK")


if __name__ == "__main__":
    _self_check()

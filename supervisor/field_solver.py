"""Robust mixed FIELD calibration solver for RealJaen Supervisor.

Key design:
- Point-only calibration remains the stable baseline when >=4 good points exist.
- Manual semantic lines refine that baseline but are NOT allowed to destroy it.
- A manual line represents an infinite image line: its drawn endpoints only define
  orientation/position. They are NOT interpreted as world-line endpoints.
- Mixed DLT is available when points alone are insufficient.
- Final candidate is accepted only if it is numerically sane and does not
  substantially worsen the manual point fit.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


# Conservative defaults. They are intentionally biased toward preserving a good
# point calibration and using semantic lines to correct drift rather than
# completely replacing the solution.
LINE_WEIGHT = 10.0
ARC_WEIGHT = 1.5
REGULARIZATION_WEIGHT = 0.015
MAX_POINT_DEGRADATION_PX = 12.0
MAX_POINT_ERROR_ABSOLUTE_PX = 18.0
LINE_SCORE_WEIGHT = 2.5
MAX_CONDITION = 5e8
LINE_SAMPLES = 7
ARC_SAMPLES = 20
LM_ITERS = 35


def _finite_h(h: np.ndarray | None) -> np.ndarray | None:
    if h is None:
        return None
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all():
        return None
    if abs(float(h[2, 2])) < 1e-12:
        return None
    h = h / float(h[2, 2])
    try:
        cond = float(np.linalg.cond(h))
        det = float(np.linalg.det(h))
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(cond) or cond > MAX_CONDITION or abs(det) < 1e-12:
        return None
    return h


def _non_collinear(points: np.ndarray) -> bool:
    if len(points) < 4:
        return False
    centered = points - points.mean(axis=0)
    s = np.linalg.svd(centered, compute_uv=False)
    return len(s) == 2 and s[1] > max(1e-9, s[0] * 1e-4)


def _project(points: np.ndarray, h: np.ndarray) -> np.ndarray | None:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    hp = np.column_stack([pts, np.ones(len(pts))]) @ h.T
    if not np.isfinite(hp).all() or np.any(np.abs(hp[:, 2]) < 1e-10):
        return None
    out = hp[:, :2] / hp[:, 2:3]
    return out if np.isfinite(out).all() else None


def _image_line(start: dict, end: dict) -> np.ndarray | None:
    try:
        p1 = np.array([float(start["x"]), float(start["y"]), 1.0])
        p2 = np.array([float(end["x"]), float(end["y"]), 1.0])
    except (KeyError, TypeError, ValueError):
        return None
    line = np.cross(p1, p2)
    norm = math.hypot(float(line[0]), float(line[1]))
    if norm < 1e-9:
        return None
    return line / norm


def _sample_world_line(constraint: dict, count: int = LINE_SAMPLES) -> np.ndarray | None:
    try:
        a = constraint["world"]["start"]
        b = constraint["world"]["end"]
        p1 = np.array([float(a["x"]), float(a["y"])])
        p2 = np.array([float(b["x"]), float(b["y"])])
    except (KeyError, TypeError, ValueError):
        return None
    t = np.linspace(0.0, 1.0, count)
    return p1[None, :] * (1.0 - t[:, None]) + p2[None, :] * t[:, None]


def _point_data(point_constraints: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    world, image = [], []
    for item in point_constraints:
        try:
            world.append([float(item["world"]["x"]), float(item["world"]["y"])])
            image.append([float(item["image"]["x"]), float(item["image"]["y"])])
        except (KeyError, TypeError, ValueError):
            continue
    return np.asarray(world, dtype=np.float64), np.asarray(image, dtype=np.float64)


def _line_data(line_constraints: list[dict]) -> list[tuple[dict, np.ndarray, np.ndarray]]:
    result = []
    for item in line_constraints:
        line = _image_line(item.get("image", {}).get("start", {}), item.get("image", {}).get("end", {}))
        samples = _sample_world_line(item)
        if line is not None and samples is not None:
            result.append((item, line, samples))
    return result


def _baseline_from_points(world: np.ndarray, image: np.ndarray) -> np.ndarray | None:
    if len(world) < 4 or not _non_collinear(world) or not _non_collinear(image):
        return None
    h, _ = cv2.findHomography(world, image, method=0)
    return _finite_h(h)


def _mixed_dlt(point_constraints: list[dict], line_constraints: list[dict]) -> np.ndarray | None:
    """Linear initialization using point equations + line incidence equations."""
    rows: list[list[float]] = []

    for item in point_constraints:
        try:
            x, y = float(item["world"]["x"]), float(item["world"]["y"])
            u, v = float(item["image"]["x"]), float(item["image"]["y"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        rows.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])

    for item, line, samples in _line_data(line_constraints):
        # Two independent world points on the semantic world line are enough
        # for linear incidence constraints l_img^T H X = 0.
        for p in (samples[0], samples[-1]):
            x, y = float(p[0]), float(p[1])
            a, b, c = map(float, line)
            rows.append([
                a*x, a*y, a,
                b*x, b*y, b,
                c*x, c*y, c,
            ])

    if len(rows) < 8:
        return None

    a = np.asarray(rows, dtype=np.float64)
    if np.linalg.matrix_rank(a) < 8:
        return None
    try:
        _, _, vt = np.linalg.svd(a)
    except np.linalg.LinAlgError:
        return None
    h = vt[-1].reshape(3, 3)
    return _finite_h(h)


def _h_to_params(h: np.ndarray) -> np.ndarray:
    h = h / h[2, 2]
    return np.array([h[0,0], h[0,1], h[0,2],
                     h[1,0], h[1,1], h[1,2],
                     h[2,0], h[2,1]], dtype=np.float64)


def _params_to_h(p: np.ndarray) -> np.ndarray:
    return np.array([[p[0], p[1], p[2]],
                     [p[3], p[4], p[5]],
                     [p[6], p[7], 1.0]], dtype=np.float64)


def _canonical_regularization_points() -> np.ndarray:
    return np.array([
        [-52.5, -34.0], [0.0, -34.0], [52.5, -34.0],
        [-52.5,   0.0], [0.0,   0.0], [52.5,   0.0],
        [-52.5,  34.0], [0.0,  34.0], [52.5,  34.0],
        [-36.0, -20.16], [36.0, -20.16],
        [-36.0,  20.16], [36.0,  20.16],
    ], dtype=np.float64)


def _ellipse_samples(geometry: dict, count: int = ARC_SAMPLES) -> np.ndarray | None:
    try:
        c = geometry["center"]
        cx, cy = float(c["x"]), float(c["y"])
        rx = abs(float(geometry["radiusX"]))
        ry = abs(float(geometry["radiusY"]))
        rotation = float(geometry.get("rotation", 0.0))
        arc = geometry.get("arc", {})
        a0 = float(arc.get("startAngle", 0.0))
        a1 = float(arc.get("endAngle", 2.0 * math.pi))
        ccw = bool(arc.get("counterclockwise", False))
    except (KeyError, TypeError, ValueError):
        return None
    if rx < 1.0 or ry < 1.0:
        return None

    # Keep the user-traced angular span. For a wrapped arc choose the short
    # intended path consistent with the stored direction.
    if ccw:
        while a1 > a0:
            a1 -= 2.0 * math.pi
    else:
        while a1 < a0:
            a1 += 2.0 * math.pi

    angles = np.linspace(a0, a1, count)
    ca, sa = math.cos(rotation), math.sin(rotation)
    x = rx * np.cos(angles)
    y = ry * np.sin(angles)
    return np.column_stack([cx + ca*x - sa*y, cy + sa*x + ca*y])


def _arc_data(arc_constraints: list[dict]) -> list[tuple[dict, np.ndarray, np.ndarray, float]]:
    data = []
    for item in arc_constraints:
        try:
            wc = item["world"]["center"]
            center = np.array([float(wc["x"]), float(wc["y"])])
            radius = float(item["world"]["radius"])
        except (KeyError, TypeError, ValueError):
            continue
        image = item.get("image", {})
        fitted = image.get("fitted") or image.get("geometry") or image
        samples = _ellipse_samples(fitted) if isinstance(fitted, dict) else None
        if samples is None:
            raw = image.get("samples") or item.get("fit", {}).get("samples")
            if raw:
                try:
                    samples = np.asarray([[float(p["x"]), float(p["y"])] for p in raw], dtype=np.float64)
                except (KeyError, TypeError, ValueError):
                    samples = None
        if samples is not None and len(samples) >= 4 and radius > 0:
            data.append((item, samples, center, radius))
    return data


def _residual_vector(
    p: np.ndarray,
    point_constraints: list[dict],
    line_constraints: list[dict],
    arc_constraints: list[dict],
    baseline_h: np.ndarray | None,
) -> np.ndarray:
    h = _finite_h(_params_to_h(p))
    if h is None:
        return np.full(32, 1e4, dtype=np.float64)

    residuals: list[float] = []

    world, image = _point_data(point_constraints)
    if len(world):
        projected = _project(world, h)
        if projected is None:
            return np.full(32, 1e4, dtype=np.float64)
        # Human points: strong, ordinary pixel reprojection residual.
        residuals.extend((projected - image).reshape(-1).tolist())

    for item, line, samples in _line_data(line_constraints):
        projected = _project(samples, h)
        if projected is None:
            return np.full(32, 1e4, dtype=np.float64)
        # Since image line is normalized, this is signed distance in pixels.
        d = projected[:, 0] * line[0] + projected[:, 1] * line[1] + line[2]
        # Normalize by sqrt(N) so "more samples" does not create more total
        # weight. The semantic line itself gets a strong manual weight.
        scale = LINE_WEIGHT / math.sqrt(max(1, len(d)))
        residuals.extend((d * scale).tolist())

    for item, image_samples, world_center, world_radius in _arc_data(arc_constraints):
        try:
            inv = np.linalg.inv(h)
        except np.linalg.LinAlgError:
            continue
        back = _project(image_samples, inv)
        if back is None:
            continue
        radial_m = np.linalg.norm(back - world_center[None, :], axis=1) - world_radius

        # Convert metre error roughly to pixels using projected local scale
        c_img = _project(np.array([world_center, world_center + [world_radius, 0.0]]), h)
        if c_img is None:
            continue
        px_per_m = np.linalg.norm(c_img[1] - c_img[0]) / max(world_radius, 1e-6)
        d = radial_m * px_per_m
        scale = ARC_WEIGHT / math.sqrt(max(1, len(d)))
        residuals.extend((d * scale).tolist())

    # Softly keep refinement near a known-good point-only calibration.
    if baseline_h is not None:
        reg_world = _canonical_regularization_points()
        base_img = _project(reg_world, baseline_h)
        cur_img = _project(reg_world, h)
        if base_img is not None and cur_img is not None:
            residuals.extend(((cur_img - base_img) * REGULARIZATION_WEIGHT).reshape(-1).tolist())

    if not residuals:
        return np.zeros(1, dtype=np.float64)
    return np.asarray(residuals, dtype=np.float64)


def _robust_cost(r: np.ndarray, delta: float = 8.0) -> float:
    """Huber-style cost; enough for stable finite-difference LM."""
    a = np.abs(r)
    return float(np.sum(np.where(a <= delta, 0.5 * r*r, delta * (a - 0.5*delta))))


def _lm_optimize(
    initial_h: np.ndarray,
    point_constraints: list[dict],
    line_constraints: list[dict],
    arc_constraints: list[dict],
    baseline_h: np.ndarray | None,
) -> np.ndarray | None:
    p = _h_to_params(initial_h)
    damping = 1e-3
    best_r = _residual_vector(p, point_constraints, line_constraints, arc_constraints, baseline_h)
    best_cost = _robust_cost(best_r)

    for _ in range(LM_ITERS):
        r = _residual_vector(p, point_constraints, line_constraints, arc_constraints, baseline_h)
        if len(r) != len(best_r):
            return _finite_h(_params_to_h(p))

        # Finite-difference Jacobian for only 8 parameters.
        j = np.zeros((len(r), 8), dtype=np.float64)
        for k in range(8):
            step = 1e-6 * max(1.0, abs(float(p[k])))
            pp = p.copy()
            pp[k] += step
            rr = _residual_vector(pp, point_constraints, line_constraints, arc_constraints, baseline_h)
            if len(rr) != len(r):
                return _finite_h(_params_to_h(p))
            j[:, k] = (rr - r) / step

        # IRLS Huber weights.
        delta = 8.0
        abs_r = np.abs(r)
        w = np.ones_like(r)
        mask = abs_r > delta
        w[mask] = delta / np.maximum(abs_r[mask], 1e-12)
        sw = np.sqrt(w)
        jw = j * sw[:, None]
        rw = r * sw

        lhs = jw.T @ jw + damping * np.eye(8)
        rhs = -(jw.T @ rw)
        try:
            dp = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            damping *= 10.0
            continue

        if not np.isfinite(dp).all():
            return None

        candidate = p + dp
        candidate_h = _finite_h(_params_to_h(candidate))
        if candidate_h is None:
            damping *= 10.0
            continue

        cr = _residual_vector(candidate, point_constraints, line_constraints, arc_constraints, baseline_h)
        ccost = _robust_cost(cr)

        if ccost < best_cost:
            p = candidate
            best_r = cr
            best_cost = ccost
            damping = max(1e-8, damping * 0.4)
            if np.linalg.norm(dp) < 1e-7:
                break
        else:
            damping = min(1e8, damping * 4.0)

    return _finite_h(_params_to_h(p))


def _point_error(h: np.ndarray, points: list[dict]) -> float | None:
    world, image = _point_data(points)
    if not len(world):
        return None
    projected = _project(world, h)
    if projected is None:
        return None
    return float(np.mean(np.linalg.norm(projected - image, axis=1)))


def _line_errors(h: np.ndarray, lines: list[dict]) -> tuple[float | None, list[dict]]:
    details = []
    values = []
    for item, line, samples in _line_data(lines):
        projected = _project(samples, h)
        if projected is None:
            continue
        d = np.abs(projected[:, 0] * line[0] + projected[:, 1] * line[1] + line[2])
        err = float(np.mean(d))
        values.append(err)
        details.append({
            "semantic": item.get("semantic"),
            "error_px": err,
        })
    return (float(np.mean(values)) if values else None), details


def solve_field_calibration(
    point_constraints: list[dict],
    line_constraints: list[dict],
    arc_constraints: list[dict],
    image_width: int | None,
    image_height: int | None,
) -> dict:
    """Solve one FIELD frame from any useful point/line/arc mixture."""

    baseline = None
    world, image = _point_data(point_constraints)
    if len(world) >= 4:
        baseline = _baseline_from_points(world, image)

    # Stable initialization preference:
    # 1) point-only H when possible (known to work well in this project)
    # 2) mixed DLT when point-only is impossible
    initial = baseline if baseline is not None else _mixed_dlt(point_constraints, line_constraints)
    if initial is None:
        return {
            "status": "not_enough_constraints",
            "detail": "Las referencias actuales no determinan una homografía estable.",
            "diagnostics": {
                "points": len(point_constraints),
                "lines": len(line_constraints),
                "arcs": len(arc_constraints),
            },
        }

    # Point-only keeps OpenCV baseline exactly, preserving previous behaviour.
    if not line_constraints and not arc_constraints:
        pe = _point_error(initial, point_constraints)
        return {
            "status": "solved",
            "homography": initial.tolist(),
            "reprojection_error": pe,
            "diagnostics": {
                "points": len(point_constraints),
                "lines": 0,
                "arcs": 0,
                "point_error_px": pe,
                "mode": "point_baseline",
            },
        }

    # Multi-start refinement. A clustered set of points can give an excellent
    # local homography around the penalty area while extrapolating the near
    # touchline very badly. A semantic manual line is precisely the evidence
    # that should correct that projective extrapolation, so do not allow the
    # point baseline to dominate merely because it was the initializer.
    starts: list[tuple[str, np.ndarray]] = [("initial", initial)]
    mixed_start = _mixed_dlt(point_constraints, line_constraints)
    if mixed_start is not None:
        if all(np.linalg.norm(mixed_start - h) > 1e-6 for _, h in starts):
            starts.append(("mixed_dlt", mixed_start))

    baseline_point_error = _point_error(baseline, point_constraints) if baseline is not None else None
    baseline_line_error, _ = _line_errors(baseline, line_constraints) if baseline is not None else (None, [])

    candidates: list[dict] = []
    for start_name, start_h in starts:
        refined = _lm_optimize(
            start_h,
            point_constraints,
            line_constraints,
            arc_constraints,
            baseline,
        )
        if refined is None:
            continue
        pe = _point_error(refined, point_constraints)
        le, details = _line_errors(refined, line_constraints)
        if pe is None and point_constraints:
            continue
        score = (pe or 0.0) + LINE_SCORE_WEIGHT * (le or 0.0)
        candidates.append({
            "name": start_name,
            "homography": refined,
            "point_error": pe,
            "line_error": le,
            "line_details": details,
            "score": float(score),
        })

    # Keep the pure point solution available as a last-resort candidate, but
    # heavily penalize it when it contradicts a manual semantic line.
    if baseline is not None:
        baseline_details = _line_errors(baseline, line_constraints)[1]
        baseline_score = (baseline_point_error or 0.0) + LINE_SCORE_WEIGHT * (baseline_line_error or 0.0)
        candidates.append({
            "name": "point_baseline",
            "homography": baseline,
            "point_error": baseline_point_error,
            "line_error": baseline_line_error,
            "line_details": baseline_details,
            "score": float(baseline_score),
        })

    if not candidates:
        return {
            "status": "optimization_failed",
            "detail": "No se pudo refinar la calibración mixta de forma estable.",
        }

    # Manual points remain important, but a semantic line is not a weak hint.
    # Permit the homography to move enough to satisfy a far-away touchline,
    # while rejecting solutions that make the clicked landmarks obviously bad.
    if baseline_point_error is not None:
        point_guard = max(
            MAX_POINT_ERROR_ABSOLUTE_PX,
            baseline_point_error + MAX_POINT_DEGRADATION_PX,
        )
    else:
        point_guard = float("inf")

    feasible = [
        item for item in candidates
        if item["point_error"] is None or item["point_error"] <= point_guard
    ]
    chosen_pool = feasible or candidates
    chosen = min(chosen_pool, key=lambda item: item["score"])

    candidate = chosen["homography"]
    candidate_point_error = chosen["point_error"]
    candidate_line_error = chosen["line_error"]
    line_details = chosen["line_details"]
    fallback_used = chosen["name"] == "point_baseline"
    selected_candidate = chosen["name"]

    # Final sanity: representative pitch points must project finitely.
    representative = _canonical_regularization_points()
    projected = _project(representative, candidate)
    if projected is None:
        return {
            "status": "degenerate",
            "detail": "La homografía resultante no proyecta el campo de forma estable.",
        }

    # Do not reject merely because much of the pitch is outside the image.
    if image_width and image_height:
        limit = max(float(image_width), float(image_height)) * 50.0
        if float(np.max(np.abs(projected))) > limit:
            return {
                "status": "degenerate",
                "detail": "La proyección del campo es numéricamente poco plausible.",
            }

    return {
        "status": "solved",
        "homography": candidate.tolist(),
        "reprojection_error": candidate_point_error,
        "diagnostics": {
            "points": len(point_constraints),
            "lines": len(line_constraints),
            "arcs": len(arc_constraints),
            "point_error_px": candidate_point_error,
            "line_error_px": candidate_line_error,
            "line_errors": line_details,
            "baseline_point_error_px": baseline_point_error,
            "baseline_line_error_px": baseline_line_error,
            "fallback_to_point_baseline": fallback_used,
            "selected_candidate": selected_candidate,
            "point_guard_px": point_guard if np.isfinite(point_guard) else None,
            "line_improvement_px": (
                (baseline_line_error - candidate_line_error)
                if baseline_line_error is not None and candidate_line_error is not None
                else None
            ),
            "mode": "mixed_semantic_line_refinement",
        },
    }
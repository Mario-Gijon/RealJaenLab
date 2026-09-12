"""Diagnostic camera-model probe for RealJaenLab FIELD.

Compares the current pure homography against two tiny, generic post-homography
image warps:

- radial_1p: one symmetric radial coefficient k
- anisotropic_2p: separate kx / ky coefficients

This is a diagnostic experiment, not a production camera model.  H is held
fixed on purpose: the probe asks whether a small residual image warp explains
systematic error left after the current FIELD calibration.

No SciPy/OpenCV dependency is required.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np

from supervisor.field.pitch import PitchModel, Point2D


EPS = 1e-10
BOUND = 0.12
HUBER_DELTA_PX = 6.0
REGULARIZATION_WEIGHT = 24.0
LINE_FIT_SAMPLES = 81
LINE_OUTPUT_SAMPLES = 41


class CameraModelProbeError(ValueError):
    pass


def _homography(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise CameraModelProbeError("H_pitch_to_image must be a finite 3x3 matrix.")
    if abs(float(np.linalg.det(matrix))) < EPS:
        raise CameraModelProbeError("H_pitch_to_image is singular.")
    return matrix


def _image_point(value: Sequence[float], label: str) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise CameraModelProbeError(f"{label} must contain exactly two coordinates.")
    u, v = float(value[0]), float(value[1])
    if not (math.isfinite(u) and math.isfinite(v)):
        raise CameraModelProbeError(f"{label} coordinates must be finite.")
    return u, v


def _project_h(h: np.ndarray, point: Point2D) -> tuple[float, float]:
    q = h @ np.array([point.x, point.y, 1.0], dtype=float)
    if abs(float(q[2])) < EPS:
        raise CameraModelProbeError("A pitch point projects to infinity.")
    return float(q[0] / q[2]), float(q[1] / q[2])


def _norm(u: float, v: float, width: float, height: float) -> tuple[float, float, float]:
    scale = 0.5 * max(width, height)
    return (
        (u - 0.5 * width) / scale,
        (v - 0.5 * height) / scale,
        scale,
    )


def _identity(
    u: float,
    v: float,
    params: tuple[float, ...],
    width: float,
    height: float,
) -> tuple[float, float]:
    return u, v


def _radial(
    u: float,
    v: float,
    params: tuple[float, ...],
    width: float,
    height: float,
) -> tuple[float, float]:
    k = float(params[0])
    x, y, scale = _norm(u, v, width, height)
    r2 = x * x + y * y
    factor = 1.0 + k * r2
    return (
        0.5 * width + scale * x * factor,
        0.5 * height + scale * y * factor,
    )


def _anisotropic(
    u: float,
    v: float,
    params: tuple[float, ...],
    width: float,
    height: float,
) -> tuple[float, float]:
    kx, ky = float(params[0]), float(params[1])
    x, y, scale = _norm(u, v, width, height)
    r2 = x * x + y * y
    return (
        0.5 * width + scale * x * (1.0 + kx * r2),
        0.5 * height + scale * y * (1.0 + ky * r2),
    )


def _model_point(
    h: np.ndarray,
    point: Point2D,
    warp: Callable[..., tuple[float, float]],
    params: tuple[float, ...],
    width: float,
    height: float,
) -> tuple[float, float]:
    u, v = _project_h(h, point)
    return warp(u, v, params, width, height)


def _segment_points(start: Point2D, end: Point2D, count: int) -> list[Point2D]:
    return [
        Point2D(
            start.x + (end.x - start.x) * float(t),
            start.y + (end.y - start.y) * float(t),
        )
        for t in np.linspace(0.0, 1.0, max(2, count))
    ]


def _point_segment_distance(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom < EPS:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = min(1.0, max(0.0, t))
    qx, qy = ax + t * dx, ay + t * dy
    return math.hypot(px - qx, py - qy)


def _point_polyline_distance(
    point: tuple[float, float],
    polyline: Sequence[tuple[float, float]],
) -> float:
    return min(
        _point_segment_distance(point, polyline[i], polyline[i + 1])
        for i in range(len(polyline) - 1)
    )


def _huber(value: float) -> float:
    a = abs(float(value))
    if a <= HUBER_DELTA_PX:
        return 0.5 * a * a
    return HUBER_DELTA_PX * (a - 0.5 * HUBER_DELTA_PX)


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"median_px": float("nan"), "p95_px": float("nan"), "max_px": float("nan")}
    arr = np.asarray(values, dtype=float)
    return {
        "median_px": float(np.median(arr)),
        "p95_px": float(np.percentile(arr, 95)),
        "max_px": float(np.max(arr)),
    }


def _errors(
    *,
    pitch: PitchModel,
    h: np.ndarray,
    point_annotations: Sequence[dict[str, Any]],
    line_annotations: Sequence[dict[str, Any]],
    warp: Callable[..., tuple[float, float]],
    params: tuple[float, ...],
    width: float,
    height: float,
) -> tuple[list[float], list[float]]:
    point_errors: list[float] = []
    line_errors: list[float] = []

    for annotation in point_annotations:
        name = annotation.get("point_name")
        if not isinstance(name, str):
            continue
        observed = _image_point(annotation.get("point"), name)
        predicted = _model_point(
            h, pitch.point(name).point, warp, params, width, height
        )
        point_errors.append(
            math.hypot(predicted[0] - observed[0], predicted[1] - observed[1])
        )

    for annotation in line_annotations:
        name = annotation.get("line_name")
        clicks = annotation.get("points")
        if (
            not isinstance(name, str)
            or not isinstance(clicks, Sequence)
            or isinstance(clicks, (str, bytes))
            or len(clicks) != 2
        ):
            continue

        observed_clicks = [
            _image_point(clicks[0], f"{name} click 1"),
            _image_point(clicks[1], f"{name} click 2"),
        ]
        line = pitch.line(name)
        sampled = _segment_points(line.start, line.end, LINE_FIT_SAMPLES)
        polyline = [
            _model_point(h, point, warp, params, width, height)
            for point in sampled
        ]
        line_errors.extend(
            _point_polyline_distance(click, polyline)
            for click in observed_clicks
        )

    return point_errors, line_errors


def _objective(
    *,
    pitch: PitchModel,
    h: np.ndarray,
    point_annotations: Sequence[dict[str, Any]],
    line_annotations: Sequence[dict[str, Any]],
    warp: Callable[..., tuple[float, float]],
    params: tuple[float, ...],
    width: float,
    height: float,
) -> tuple[float, float, list[float], list[float]]:
    point_errors, line_errors = _errors(
        pitch=pitch,
        h=h,
        point_annotations=point_annotations,
        line_annotations=line_annotations,
        warp=warp,
        params=params,
        width=width,
        height=height,
    )
    all_errors = point_errors + line_errors
    if not all_errors:
        raise CameraModelProbeError("No usable annotations were supplied.")

    raw = float(sum(_huber(v) for v in all_errors) / len(all_errors))
    regularized = raw + REGULARIZATION_WEIGHT * sum(v * v for v in params)
    return regularized, raw, point_errors, line_errors


def _fit_1d(evaluate) -> tuple[float, ...]:
    grid = np.linspace(-BOUND, BOUND, 49)
    best_k = min(
        ((evaluate((float(k),))[0], float(k)) for k in grid),
        key=lambda item: item[0],
    )[1]
    step = float(grid[1] - grid[0])

    for _ in range(16):
        candidates = [
            max(-BOUND, min(BOUND, best_k - step)),
            best_k,
            max(-BOUND, min(BOUND, best_k + step)),
        ]
        best_k = min(
            ((evaluate((k,))[0], k) for k in candidates),
            key=lambda item: item[0],
        )[1]
        step *= 0.5
    return (float(best_k),)


def _fit_2d(evaluate) -> tuple[float, ...]:
    grid = np.linspace(-BOUND, BOUND, 21)
    best_score = float("inf")
    best = (0.0, 0.0)

    for kx in grid:
        for ky in grid:
            score = evaluate((float(kx), float(ky)))[0]
            if score < best_score:
                best_score = score
                best = (float(kx), float(ky))

    step = float(grid[1] - grid[0])
    for _ in range(16):
        candidates = []
        for dx in (-step, 0.0, step):
            for dy in (-step, 0.0, step):
                candidates.append(
                    (
                        max(-BOUND, min(BOUND, best[0] + dx)),
                        max(-BOUND, min(BOUND, best[1] + dy)),
                    )
                )
        best = min(
            ((evaluate(candidate)[0], candidate) for candidate in candidates),
            key=lambda item: item[0],
        )[1]
        step *= 0.5
    return float(best[0]), float(best[1])


def _warp_amount(
    warp: Callable[..., tuple[float, float]],
    params: tuple[float, ...],
    width: float,
    height: float,
) -> dict[str, Any]:
    samples = []
    corners = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    for u in np.linspace(0.0, width, 9):
        for v in np.linspace(0.0, height, 7):
            wu, wv = warp(float(u), float(v), params, width, height)
            samples.append(math.hypot(wu - u, wv - v))
    corner_values = []
    for u, v in corners:
        wu, wv = warp(u, v, params, width, height)
        corner_values.append(math.hypot(wu - u, wv - v))
    return {
        "max_warp_px": float(max(samples)),
        "corner_warp_px": [float(v) for v in corner_values],
    }


def _project_geometry(
    pitch: PitchModel,
    h: np.ndarray,
    warp: Callable[..., tuple[float, float]],
    params: tuple[float, ...],
    width: float,
    height: float,
) -> dict[str, Any]:
    lines = []
    for line in pitch.straight_lines:
        samples = _segment_points(line.start, line.end, LINE_OUTPUT_SAMPLES)
        lines.append(
            {
                "name": line.name,
                "label": line.label,
                "family": line.family,
                "points": [
                    list(_model_point(h, point, warp, params, width, height))
                    for point in samples
                ],
            }
        )

    curves = []
    for curve in pitch.curves:
        curves.append(
            {
                "name": curve.name,
                "label": curve.label,
                "closed": curve.closed,
                "points": [
                    list(_model_point(h, point, warp, params, width, height))
                    for point in curve.points
                ],
            }
        )

    points = []
    for semantic_point in pitch.points:
        points.append(
            {
                "name": semantic_point.name,
                "label": semantic_point.label,
                "category": semantic_point.category,
                "point": list(
                    _model_point(
                        h, semantic_point.point, warp, params, width, height
                    )
                ),
            }
        )
    return {"lines": lines, "curves": curves, "points": points}


def _model_result(
    *,
    name: str,
    pitch: PitchModel,
    h: np.ndarray,
    point_annotations: Sequence[dict[str, Any]],
    line_annotations: Sequence[dict[str, Any]],
    warp: Callable[..., tuple[float, float]],
    params: tuple[float, ...],
    width: float,
    height: float,
) -> dict[str, Any]:
    objective, raw, point_errors, line_errors = _objective(
        pitch=pitch,
        h=h,
        point_annotations=point_annotations,
        line_annotations=line_annotations,
        warp=warp,
        params=params,
        width=width,
        height=height,
    )
    result = {
        "name": name,
        "parameters": [float(v) for v in params],
        "objective": float(objective),
        "unregularized_objective": float(raw),
        "residual_summary": _summary(point_errors + line_errors),
        "point_summary": _summary(point_errors),
        "line_summary": _summary(line_errors),
        "projected_geometry": _project_geometry(
            pitch, h, warp, params, width, height
        ),
    }
    result.update(_warp_amount(warp, params, width, height))
    return result


def probe_camera_models_payload(
    solved_calibration: dict[str, Any],
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    """Compare pure H, radial-1p, and anisotropic-2p on the same annotations."""

    width = float(request_payload["video_width"])
    height = float(request_payload["video_height"])
    if width <= 0.0 or height <= 0.0:
        raise CameraModelProbeError("Positive video dimensions are required.")

    h = _homography(solved_calibration["H_pitch_to_image"])
    pitch = PitchModel(
        float(solved_calibration.get("pitch_length", 105.0)),
        float(solved_calibration.get("pitch_width", 68.0)),
    )
    point_annotations = request_payload.get("point_annotations", [])
    line_annotations = request_payload.get("annotations", [])

    def radial_eval(params):
        return _objective(
            pitch=pitch, h=h,
            point_annotations=point_annotations,
            line_annotations=line_annotations,
            warp=_radial, params=params,
            width=width, height=height,
        )

    def anisotropic_eval(params):
        return _objective(
            pitch=pitch, h=h,
            point_annotations=point_annotations,
            line_annotations=line_annotations,
            warp=_anisotropic, params=params,
            width=width, height=height,
        )

    radial_params = _fit_1d(radial_eval)
    anisotropic_params = _fit_2d(anisotropic_eval)

    models = {
        "pure_homography": _model_result(
            name="pure_homography",
            pitch=pitch, h=h,
            point_annotations=point_annotations,
            line_annotations=line_annotations,
            warp=_identity, params=(),
            width=width, height=height,
        ),
        "radial_1p": _model_result(
            name="radial_1p",
            pitch=pitch, h=h,
            point_annotations=point_annotations,
            line_annotations=line_annotations,
            warp=_radial, params=radial_params,
            width=width, height=height,
        ),
        "anisotropic_2p": _model_result(
            name="anisotropic_2p",
            pitch=pitch, h=h,
            point_annotations=point_annotations,
            line_annotations=line_annotations,
            warp=_anisotropic, params=anisotropic_params,
            width=width, height=height,
        ),
    }

    baseline_raw = models["pure_homography"]["unregularized_objective"]
    for model in models.values():
        if baseline_raw <= EPS:
            improvement = 0.0
        else:
            improvement = 100.0 * (
                baseline_raw - model["unregularized_objective"]
            ) / baseline_raw
        model["annotation_objective_improvement_percent"] = float(improvement)

    best_name = min(models, key=lambda key: models[key]["objective"])

    return {
        "probe_version": 1,
        "best_regularized_model": best_name,
        "models": models,
        "note": (
            "Diagnostic only: lower error on the same annotations does not by "
            "itself prove generalization. Compare the projected overlays across "
            "multiple frames and matches."
        ),
    }


# Synthetic self-check -------------------------------------------------------

def _synthetic_h() -> np.ndarray:
    return np.array(
        [
            [11.4, -1.8, 290.0],
            [1.5, 7.3, 170.0],
            [0.0035, -0.0013, 1.0],
        ],
        dtype=float,
    )


def _self_check() -> None:
    pitch = PitchModel()
    h = _synthetic_h()
    width, height = 1920.0, 1080.0
    truth = (0.035, -0.020)

    point_names = (
        "corner_left_a",
        "left_penalty_front_a",
        "left_penalty_front_b",
        "center_spot",
        "right_penalty_front_a",
        "right_penalty_front_b",
        "corner_right_b",
        "right_penalty_spot",
    )
    points = []
    for name in point_names:
        observed = _model_point(
            h, pitch.point(name).point, _anisotropic, truth, width, height
        )
        points.append({"point_name": name, "point": list(observed)})

    lines = []
    for name in ("touchline_b", "halfway_line", "right_penalty_front"):
        line = pitch.line(name)
        samples = _segment_points(line.start, line.end, 9)
        clicks = [
            _model_point(h, samples[2], _anisotropic, truth, width, height),
            _model_point(h, samples[6], _anisotropic, truth, width, height),
        ]
        lines.append({"line_name": name, "points": [list(v) for v in clicks]})

    result = probe_camera_models_payload(
        {
            "H_pitch_to_image": h.tolist(),
            "pitch_length": pitch.length,
            "pitch_width": pitch.width,
        },
        {
            "video_width": width,
            "video_height": height,
            "annotations": lines,
            "point_annotations": points,
        },
    )

    recovered = result["models"]["anisotropic_2p"]["parameters"]
    baseline = result["models"]["pure_homography"]["residual_summary"]["median_px"]
    corrected = result["models"]["anisotropic_2p"]["residual_summary"]["median_px"]

    assert abs(recovered[0] - truth[0]) < 0.004, recovered
    assert abs(recovered[1] - truth[1]) < 0.004, recovered
    assert corrected < baseline
    assert result["best_regularized_model"] == "anisotropic_2p"

    print("Camera model probe self-check: OK")
    print(f"Synthetic truth: kx={truth[0]:+.4f}, ky={truth[1]:+.4f}")
    print(f"Recovered:       kx={recovered[0]:+.4f}, ky={recovered[1]:+.4f}")
    print(f"Pure H median:   {baseline:.3f} px")
    print(f"Corrected median:{corrected:.3f} px")
    print(f"Best model:      {result['best_regularized_model']}")


if __name__ == "__main__":
    _self_check()

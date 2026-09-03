"""Upgrade HUMAN point guard V1 -> V2.

V1 froze the point-only projection too aggressively. V2 protects distance to
the actual HUMAN clicks and permits the maximum useful line/arc refinement
inside that click-fit envelope.
"""
from __future__ import annotations

from pathlib import Path
import shutil

TARGET = Path(__file__).with_name("field_camera_calibration.py")
BACKUP = Path(__file__).with_name(
    "field_camera_calibration.py.before_human_guard_v2"
)

V1_MARKER = "max_human_point_shift_mean_px"
V2_MARKER = "human_point_error_mean_floor_px"

OLD_CONFIG = '    # HUMAN clicks are primary geometric evidence. Lines/arcs may refine the\n    # homography, but they must not visibly drag already placed HUMAN points.\n    manual_point_full_weight: bool = True\n    max_human_point_shift_mean_px: float = 1.5\n    max_human_point_shift_p90_px: float = 2.5\n    max_human_point_shift_max_px: float = 4.0\n'
NEW_CONFIG = '    # HUMAN clicks are primary geometric evidence. Lines/arcs may move the\n    # homography to improve extrapolation, but projected semantic landmarks must\n    # remain close to the ACTUAL clicks (not frozen to the point-only H).\n    manual_point_full_weight: bool = True\n\n    # Absolute click-fit envelope, plus a small allowance over the point-only\n    # baseline when the manual clicks are mutually imperfect.\n    human_point_error_mean_floor_px: float = 3.5\n    human_point_error_p90_floor_px: float = 5.0\n    human_point_error_max_floor_px: float = 7.0\n    human_point_error_mean_degradation_px: float = 1.0\n    human_point_error_p90_degradation_px: float = 1.5\n    human_point_error_max_degradation_px: float = 2.0\n\n    # If the unconstrained mixed optimum violates the HUMAN envelope, walk from\n    # the point baseline toward it and keep the furthest feasible solution.\n    human_guard_bisection_steps: int = 28\n'
OLD_HELPERS = 'def _point_projection_shift_diagnostics(\n    h: np.ndarray,\n    reference_h: np.ndarray | None,\n    points: list[dict[str, Any]],\n) -> dict[str, Any] | None:\n    """How much a candidate moves HUMAN point projections vs point-only H.\n\n    This is deliberately different from reprojection error to the click:\n    it detects the exact failure mode seen in the UI, where adding a semantic\n    line makes already stable projected HUMAN landmarks jump elsewhere.\n    """\n\n    if reference_h is None or not points:\n        return None\n\n    world = np.asarray(\n        [item["world"] for item in points],\n        dtype=np.float64,\n    )\n\n    reference, reference_valid = _project_points(\n        world,\n        reference_h,\n    )\n    candidate, candidate_valid = _project_points(\n        world,\n        h,\n    )\n\n    valid = reference_valid & candidate_valid\n    if not np.any(valid):\n        return None\n\n    distances = np.linalg.norm(\n        candidate[valid] - reference[valid],\n        axis=1,\n    )\n\n    if not len(distances) or not np.isfinite(distances).all():\n        return None\n\n    details: list[dict[str, Any]] = []\n    valid_indices = np.flatnonzero(valid)\n\n    for local_index, point_index in enumerate(valid_indices):\n        details.append(\n            {\n                "semantic": points[int(point_index)].get("semantic"),\n                "shift_px": float(distances[local_index]),\n            }\n        )\n\n    return {\n        "count": int(len(distances)),\n        "mean_px": float(np.mean(distances)),\n        "p90_px": float(np.percentile(distances, 90)),\n        "max_px": float(np.max(distances)),\n        "per_point": details,\n    }\n\n\ndef _human_point_guard_ok(\n    shift: dict[str, Any] | None,\n    config: CameraCalibrationConfig,\n) -> bool:\n    """Allow line/arc refinement only when HUMAN point motion is imperceptible."""\n\n    if shift is None:\n        return True\n\n    return bool(\n        float(shift["mean_px"])\n        <= config.max_human_point_shift_mean_px\n        and float(shift["p90_px"])\n        <= config.max_human_point_shift_p90_px\n        and float(shift["max_px"])\n        <= config.max_human_point_shift_max_px\n    )\n\n\n'
NEW_HELPERS = 'def _manual_point_fit_diagnostics(\n    h: np.ndarray,\n    points: list[dict[str, Any]],\n) -> dict[str, Any] | None:\n    """Pixel error from projected semantic landmarks to the HUMAN clicks."""\n\n    if not points:\n        return None\n\n    world = np.asarray(\n        [item["world"] for item in points],\n        dtype=np.float64,\n    )\n    observed = np.asarray(\n        [item["image"] for item in points],\n        dtype=np.float64,\n    )\n\n    projected, valid = _project_points(world, h)\n    if not np.all(valid):\n        return None\n\n    distances = np.linalg.norm(\n        projected - observed,\n        axis=1,\n    )\n    if not len(distances) or not np.isfinite(distances).all():\n        return None\n\n    return {\n        "count": int(len(distances)),\n        "mean_px": float(np.mean(distances)),\n        "p90_px": float(np.percentile(distances, 90)),\n        "max_px": float(np.max(distances)),\n        "per_point": [\n            {\n                "semantic": item.get("semantic"),\n                "error_px": float(distances[index]),\n            }\n            for index, item in enumerate(points)\n        ],\n    }\n\n\ndef _human_point_error_limits(\n    point_baseline_h: np.ndarray | None,\n    points: list[dict[str, Any]],\n    config: CameraCalibrationConfig,\n) -> dict[str, float] | None:\n    """Feasible click-fit envelope derived from annotation noise + baseline."""\n\n    if point_baseline_h is None or not points:\n        return None\n\n    baseline = _manual_point_fit_diagnostics(\n        point_baseline_h,\n        points,\n    )\n    if baseline is None:\n        return None\n\n    return {\n        "mean_px": max(\n            config.human_point_error_mean_floor_px,\n            float(baseline["mean_px"])\n            + config.human_point_error_mean_degradation_px,\n        ),\n        "p90_px": max(\n            config.human_point_error_p90_floor_px,\n            float(baseline["p90_px"])\n            + config.human_point_error_p90_degradation_px,\n        ),\n        "max_px": max(\n            config.human_point_error_max_floor_px,\n            float(baseline["max_px"])\n            + config.human_point_error_max_degradation_px,\n        ),\n        "baseline_mean_px": float(baseline["mean_px"]),\n        "baseline_p90_px": float(baseline["p90_px"]),\n        "baseline_max_px": float(baseline["max_px"]),\n    }\n\n\ndef _human_point_fit_ok(\n    fit: dict[str, Any] | None,\n    limits: dict[str, float] | None,\n) -> bool:\n    if fit is None or limits is None:\n        return True\n\n    return bool(\n        float(fit["mean_px"]) <= float(limits["mean_px"])\n        and float(fit["p90_px"]) <= float(limits["p90_px"])\n        and float(fit["max_px"]) <= float(limits["max_px"])\n    )\n\n\ndef _blend_homographies(\n    reference_h: np.ndarray,\n    target_h: np.ndarray,\n    alpha: float,\n    max_condition: float,\n) -> np.ndarray | None:\n    """Interpolate in the eight normalised projective parameters."""\n\n    alpha = min(max(float(alpha), 0.0), 1.0)\n    p0 = _h_to_params(reference_h)\n    p1 = _h_to_params(target_h)\n    params = p0 + alpha * (p1 - p0)\n    return _finite_h(\n        _params_to_h(params),\n        max_condition,\n    )\n\n\ndef _human_guard_boundary_candidate(\n    reference_h: np.ndarray,\n    target_h: np.ndarray,\n    points: list[dict[str, Any]],\n    limits: dict[str, float] | None,\n    config: CameraCalibrationConfig,\n) -> tuple[np.ndarray, float] | None:\n    """Furthest feasible point on baseline -> mixed-optimum trajectory.\n\n    V1 either accepted the complete mixed solution or snapped all the way back\n    to point-only. V2 keeps the maximum useful line/arc correction that still\n    respects the HUMAN click-fit envelope.\n    """\n\n    if limits is None:\n        return target_h, 1.0\n\n    target_fit = _manual_point_fit_diagnostics(\n        target_h,\n        points,\n    )\n    if _human_point_fit_ok(target_fit, limits):\n        return target_h, 1.0\n\n    reference_fit = _manual_point_fit_diagnostics(\n        reference_h,\n        points,\n    )\n    if not _human_point_fit_ok(reference_fit, limits):\n        return None\n\n    low = 0.0\n    high = 1.0\n    best_h = reference_h\n\n    for _ in range(max(8, int(config.human_guard_bisection_steps))):\n        mid = 0.5 * (low + high)\n        candidate = _blend_homographies(\n            reference_h,\n            target_h,\n            mid,\n            config.max_condition,\n        )\n\n        if candidate is None:\n            high = mid\n            continue\n\n        fit = _manual_point_fit_diagnostics(\n            candidate,\n            points,\n        )\n\n        if _human_point_fit_ok(fit, limits):\n            low = mid\n            best_h = candidate\n        else:\n            high = mid\n\n    return best_h, float(low)\n\n\n'
OLD_LIMITS_ANCHOR = '    candidates: list[dict[str, Any]] = []\n'
NEW_LIMITS_ANCHOR = '    human_point_error_limits = _human_point_error_limits(\n        point_only,\n        points,\n        cfg,\n    )\n\n    candidates: list[dict[str, Any]] = []\n'
OLD_CANDIDATE = '            human_point_shift = (\n                _point_projection_shift_diagnostics(\n                    candidate,\n                    point_only,\n                    points,\n                )\n                if point_only is not None\n                else None\n            )\n\n            candidates.append(\n                {\n                    "name": f"{name}:{suffix}",\n                    "homography": candidate,\n                    "score": float(score),\n                    "human_point_shift": human_point_shift,\n                    "human_point_guard_ok": _human_point_guard_ok(\n                        human_point_shift,\n                        cfg,\n                    ),\n                }\n            )\n'
NEW_CANDIDATE = '            human_point_fit = _manual_point_fit_diagnostics(\n                candidate,\n                points,\n            )\n\n            candidates.append(\n                {\n                    "name": f"{name}:{suffix}",\n                    "homography": candidate,\n                    "score": float(score),\n                    "human_point_fit": human_point_fit,\n                    "human_point_guard_ok": _human_point_fit_ok(\n                        human_point_fit,\n                        human_point_error_limits,\n                    ),\n                    "human_guard_alpha": 1.0,\n                }\n            )\n'
OLD_CHOOSE = '    # Raw score still describes point/line/arc fit, but final selection is\n    # lexicographic when a point-only HUMAN baseline exists:\n    #   1. preserve HUMAN point geometry;\n    #   2. among solutions that preserve it, improve lines/arcs.\n    raw_best = min(\n        candidates,\n        key=lambda item: item["score"],\n    )\n\n    guarded_candidates = [\n        item\n        for item in candidates\n        if item.get("human_point_guard_ok", True)\n    ]\n    chosen_pool = guarded_candidates or candidates\n\n    chosen = min(\n        chosen_pool,\n        key=lambda item: item["score"],\n    )\n    h = chosen["homography"]\n\n    human_point_guard_intervened = bool(\n        point_only is not None\n        and not raw_best.get("human_point_guard_ok", True)\n        and raw_best["name"] != chosen["name"]\n    )\n'
NEW_CHOOSE = '    # Keep the raw optimum for diagnostics, then build a small constrained\n    # Pareto boundary instead of the V1 all-or-nothing fallback.\n    raw_candidates = list(candidates)\n    raw_best = min(\n        raw_candidates,\n        key=lambda item: item["score"],\n    )\n\n    boundary_candidates: list[dict[str, Any]] = []\n\n    if (\n        point_only is not None\n        and human_point_error_limits is not None\n    ):\n        for item in raw_candidates:\n            if item.get("human_point_guard_ok", True):\n                continue\n\n            boundary = _human_guard_boundary_candidate(\n                point_only,\n                item["homography"],\n                points,\n                human_point_error_limits,\n                cfg,\n            )\n            if boundary is None:\n                continue\n\n            boundary_h, alpha = boundary\n\n            # alpha ~= 0 is just the point baseline and is already present.\n            if alpha <= 1e-6:\n                continue\n\n            boundary_score = _candidate_score(\n                boundary_h,\n                points,\n                lines,\n                arcs,\n                cfg,\n            )\n            if not np.isfinite(boundary_score):\n                continue\n\n            boundary_fit = _manual_point_fit_diagnostics(\n                boundary_h,\n                points,\n            )\n            if not _human_point_fit_ok(\n                boundary_fit,\n                human_point_error_limits,\n            ):\n                continue\n\n            boundary_candidates.append(\n                {\n                    "name": (\n                        item["name"]\n                        + ":human_guard_boundary"\n                    ),\n                    "homography": boundary_h,\n                    "score": float(boundary_score),\n                    "human_point_fit": boundary_fit,\n                    "human_point_guard_ok": True,\n                    "human_guard_alpha": float(alpha),\n                    "guarded_from": item["name"],\n                }\n            )\n\n    candidates.extend(boundary_candidates)\n\n    guarded_candidates = [\n        item\n        for item in candidates\n        if item.get("human_point_guard_ok", True)\n    ]\n    chosen_pool = guarded_candidates or candidates\n\n    chosen = min(\n        chosen_pool,\n        key=lambda item: item["score"],\n    )\n    h = chosen["homography"]\n\n    human_point_guard_intervened = bool(\n        point_only is not None\n        and raw_best["name"] != chosen["name"]\n    )\n'
OLD_WARNING = '        warning_parts.append(\n            "una refinación de línea/arco fue descartada porque movía "\n            "demasiado los puntos HUMAN; se preservó la geometría marcada"\n        )\n'
NEW_WARNING = '        warning_parts.append(\n            "la refinación de línea/arco fue limitada para mantener "\n            "las proyecciones cerca de los puntos HUMAN marcados"\n        )\n'
OLD_DIAG = '            "raw_best_candidate": raw_best["name"],\n            "raw_best_score": raw_best["score"],\n            "human_point_guard_intervened": human_point_guard_intervened,\n            "human_point_shift": chosen.get("human_point_shift"),\n            "raw_best_human_point_shift": raw_best.get("human_point_shift"),\n            "human_point_guard_rejected_candidates": [\n                {\n                    "name": item["name"],\n                    "score": item["score"],\n                    "human_point_shift": item.get("human_point_shift"),\n                }\n                for item in candidates\n                if not item.get("human_point_guard_ok", True)\n            ],\n            "homography_condition": condition,\n'
NEW_DIAG = '            "raw_best_candidate": raw_best["name"],\n            "raw_best_score": raw_best["score"],\n            "human_point_guard_intervened": human_point_guard_intervened,\n            "human_point_error_limits": human_point_error_limits,\n            "human_point_fit": chosen.get("human_point_fit"),\n            "raw_best_human_point_fit": raw_best.get("human_point_fit"),\n            "human_guard_alpha": chosen.get("human_guard_alpha"),\n            "guarded_from": chosen.get("guarded_from"),\n            "human_point_guard_rejected_candidates": [\n                {\n                    "name": item["name"],\n                    "score": item["score"],\n                    "human_point_fit": item.get("human_point_fit"),\n                }\n                for item in raw_candidates\n                if not item.get("human_point_guard_ok", True)\n            ],\n            "human_guard_boundary_candidates": [\n                {\n                    "name": item["name"],\n                    "score": item["score"],\n                    "human_guard_alpha": item.get("human_guard_alpha"),\n                    "human_point_fit": item.get("human_point_fit"),\n                }\n                for item in boundary_candidates\n            ],\n            "homography_condition": condition,\n'


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: esperaba 1 coincidencia y encontré {count}."
        )
    return text.replace(old, new, 1)


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: no existe {TARGET}")
        return 2

    original = TARGET.read_text(encoding="utf-8")

    if V2_MARKER in original:
        print("HUMAN guard V2 ya está aplicado; no se modifica nada.")
        return 0

    if V1_MARKER not in original:
        print(
            "ERROR: no detecto HUMAN guard V1. "
            "No aplico V2 sobre una versión desconocida."
        )
        return 2

    updated = original
    updated = replace_once(updated, OLD_CONFIG, NEW_CONFIG, "config V2")
    updated = replace_once(updated, OLD_HELPERS, NEW_HELPERS, "helpers V2")
    updated = replace_once(
        updated,
        OLD_LIMITS_ANCHOR,
        NEW_LIMITS_ANCHOR,
        "point limits insertion",
    )
    updated = replace_once(
        updated,
        OLD_CANDIDATE,
        NEW_CANDIDATE,
        "candidate metadata V2",
    )
    updated = replace_once(
        updated,
        OLD_CHOOSE,
        NEW_CHOOSE,
        "constrained boundary selection",
    )
    updated = replace_once(
        updated,
        OLD_WARNING,
        NEW_WARNING,
        "warning wording V2",
    )
    updated = replace_once(
        updated,
        OLD_DIAG,
        NEW_DIAG,
        "diagnostics V2",
    )

    compile(updated, str(TARGET), "exec")

    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)

    TARGET.write_text(updated, encoding="utf-8")

    print("HUMAN point guard V2 aplicado.")
    print("Target :", TARGET)
    print("Backup :", BACKUP)
    print()
    print("Cambio conceptual:")
    print("  V1: no mover la proyección point-only")
    print("  V2: mantener la proyección cerca del CLICK HUMAN real")
    print()
    print("Envelope inicial:")
    print("  mean <= max(3.5 px, baseline + 1.0)")
    print("  p90  <= max(5.0 px, baseline + 1.5)")
    print("  max  <= max(7.0 px, baseline + 2.0)")
    print()
    print(
        "Si el óptimo mixto se pasa, V2 busca por bisección "
        "la máxima corrección compatible con esos límites."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

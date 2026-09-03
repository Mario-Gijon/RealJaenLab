# Patch field_camera_calibration.py with HUMAN-point preservation guard V1.
from __future__ import annotations

from pathlib import Path
import shutil

TARGET = Path(__file__).with_name("field_camera_calibration.py")
BACKUP = Path(__file__).with_name(
    "field_camera_calibration.py.before_human_guard_v1"
)
MARKER = "max_human_point_shift_mean_px"

OLD_CONFIG = '    point_sigma_px: float = 3.5\n    line_sigma_px: float = 1.5\n    arc_sigma_px: float = 4.0\n\n    point_huber_delta_sigma: float = 3.0\n'
NEW_CONFIG = '    point_sigma_px: float = 3.5\n    line_sigma_px: float = 1.5\n    arc_sigma_px: float = 4.0\n\n    # HUMAN clicks are primary geometric evidence. Lines/arcs may refine the\n    # homography, but they must not visibly drag already placed HUMAN points.\n    manual_point_full_weight: bool = True\n    max_human_point_shift_mean_px: float = 1.5\n    max_human_point_shift_p90_px: float = 2.5\n    max_human_point_shift_max_px: float = 4.0\n\n    point_huber_delta_sigma: float = 3.0\n'
OLD_KINDS = '        residuals.extend(point_residual.reshape(-1).tolist())\n        kinds.extend(["point"] * point_residual.size)\n'
NEW_KINDS = '        residuals.extend(point_residual.reshape(-1).tolist())\n\n        # Keep two residual kinds per point (x/y). Explicit HUMAN points are\n        # not robustly down-weighted: if the user clicked a semantic landmark,\n        # the solver should honour it instead of treating it as expendable\n        # evidence when a line disagrees.\n        for item in points:\n            kind = (\n                "manual_point"\n                if (\n                    config.manual_point_full_weight\n                    and str(item.get("source", "manual")).lower() == "manual"\n                )\n                else "point"\n            )\n            kinds.extend([kind, kind])\n'
OLD_WEIGHTS = '        if kind == "line":\n            weights[index] = 1.0\n            continue\n\n        delta = (\n            config.point_huber_delta_sigma\n            if kind == "point"\n            else config.arc_huber_delta_sigma\n        )\n'
NEW_WEIGHTS = '        if kind in {"line", "manual_point"}:\n            weights[index] = 1.0\n            continue\n\n        delta = (\n            config.point_huber_delta_sigma\n            if kind == "point"\n            else config.arc_huber_delta_sigma\n        )\n'
HELPER_ANCHOR = 'def _line_diagnostics(\n    h: np.ndarray,\n    lines: list[dict[str, Any]],\n) -> tuple[float | None, list[dict[str, Any]]]:\n'
HELPER_CODE = 'def _point_projection_shift_diagnostics(\n    h: np.ndarray,\n    reference_h: np.ndarray | None,\n    points: list[dict[str, Any]],\n) -> dict[str, Any] | None:\n    """How much a candidate moves HUMAN point projections vs point-only H.\n\n    This is deliberately different from reprojection error to the click:\n    it detects the exact failure mode seen in the UI, where adding a semantic\n    line makes already stable projected HUMAN landmarks jump elsewhere.\n    """\n\n    if reference_h is None or not points:\n        return None\n\n    world = np.asarray(\n        [item["world"] for item in points],\n        dtype=np.float64,\n    )\n\n    reference, reference_valid = _project_points(\n        world,\n        reference_h,\n    )\n    candidate, candidate_valid = _project_points(\n        world,\n        h,\n    )\n\n    valid = reference_valid & candidate_valid\n    if not np.any(valid):\n        return None\n\n    distances = np.linalg.norm(\n        candidate[valid] - reference[valid],\n        axis=1,\n    )\n\n    if not len(distances) or not np.isfinite(distances).all():\n        return None\n\n    details: list[dict[str, Any]] = []\n    valid_indices = np.flatnonzero(valid)\n\n    for local_index, point_index in enumerate(valid_indices):\n        details.append(\n            {\n                "semantic": points[int(point_index)].get("semantic"),\n                "shift_px": float(distances[local_index]),\n            }\n        )\n\n    return {\n        "count": int(len(distances)),\n        "mean_px": float(np.mean(distances)),\n        "p90_px": float(np.percentile(distances, 90)),\n        "max_px": float(np.max(distances)),\n        "per_point": details,\n    }\n\n\ndef _human_point_guard_ok(\n    shift: dict[str, Any] | None,\n    config: CameraCalibrationConfig,\n) -> bool:\n    """Allow line/arc refinement only when HUMAN point motion is imperceptible."""\n\n    if shift is None:\n        return True\n\n    return bool(\n        float(shift["mean_px"])\n        <= config.max_human_point_shift_mean_px\n        and float(shift["p90_px"])\n        <= config.max_human_point_shift_p90_px\n        and float(shift["max_px"])\n        <= config.max_human_point_shift_max_px\n    )\n\n\n'
OLD_CANDIDATE = '            candidates.append(\n                {\n                    "name": f"{name}:{suffix}",\n                    "homography": candidate,\n                    "score": float(score),\n                }\n            )\n'
NEW_CANDIDATE = '            human_point_shift = (\n                _point_projection_shift_diagnostics(\n                    candidate,\n                    point_only,\n                    points,\n                )\n                if point_only is not None\n                else None\n            )\n\n            candidates.append(\n                {\n                    "name": f"{name}:{suffix}",\n                    "homography": candidate,\n                    "score": float(score),\n                    "human_point_shift": human_point_shift,\n                    "human_point_guard_ok": _human_point_guard_ok(\n                        human_point_shift,\n                        cfg,\n                    ),\n                }\n            )\n'
OLD_CHOOSE = '    chosen = min(\n        candidates,\n        key=lambda item: item["score"],\n    )\n    h = chosen["homography"]\n'
NEW_CHOOSE = '    # Raw score still describes point/line/arc fit, but final selection is\n    # lexicographic when a point-only HUMAN baseline exists:\n    #   1. preserve HUMAN point geometry;\n    #   2. among solutions that preserve it, improve lines/arcs.\n    raw_best = min(\n        candidates,\n        key=lambda item: item["score"],\n    )\n\n    guarded_candidates = [\n        item\n        for item in candidates\n        if item.get("human_point_guard_ok", True)\n    ]\n    chosen_pool = guarded_candidates or candidates\n\n    chosen = min(\n        chosen_pool,\n        key=lambda item: item["score"],\n    )\n    h = chosen["homography"]\n\n    human_point_guard_intervened = bool(\n        point_only is not None\n        and not raw_best.get("human_point_guard_ok", True)\n        and raw_best["name"] != chosen["name"]\n    )\n'
OLD_WARNING = '    if bad_lines:\n        status = "solved_with_warnings"\n        names = [\n            str(item.get("semantic") or "línea")\n            for item in bad_lines\n        ]\n        detail = (\n            "La calibración se resolvió, pero algunas líneas "\n            "manuales siguen en conflicto: "\n            + ", ".join(names)\n            + "."\n        )\n'
NEW_WARNING = '    warning_parts: list[str] = []\n\n    if bad_lines:\n        names = [\n            str(item.get("semantic") or "línea")\n            for item in bad_lines\n        ]\n        warning_parts.append(\n            "algunas líneas manuales siguen en conflicto: "\n            + ", ".join(names)\n        )\n\n    if human_point_guard_intervened:\n        warning_parts.append(\n            "una refinación de línea/arco fue descartada porque movía "\n            "demasiado los puntos HUMAN; se preservó la geometría marcada"\n        )\n\n    if warning_parts:\n        status = "solved_with_warnings"\n        detail = "La calibración se resolvió, pero " + "; ".join(\n            warning_parts\n        ) + "."\n'
OLD_DIAG = '            "selected_candidate": chosen["name"],\n            "score": chosen["score"],\n            "homography_condition": condition,\n'
NEW_DIAG = '            "selected_candidate": chosen["name"],\n            "score": chosen["score"],\n            "raw_best_candidate": raw_best["name"],\n            "raw_best_score": raw_best["score"],\n            "human_point_guard_intervened": human_point_guard_intervened,\n            "human_point_shift": chosen.get("human_point_shift"),\n            "raw_best_human_point_shift": raw_best.get("human_point_shift"),\n            "human_point_guard_rejected_candidates": [\n                {\n                    "name": item["name"],\n                    "score": item["score"],\n                    "human_point_shift": item.get("human_point_shift"),\n                }\n                for item in candidates\n                if not item.get("human_point_guard_ok", True)\n            ],\n            "homography_condition": condition,\n'


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: esperaba exactamente 1 coincidencia y encontré {count}."
        )
    return text.replace(old, new, 1)


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: no existe {TARGET}")
        return 2

    original = TARGET.read_text(encoding="utf-8")

    if MARKER in original:
        print("HUMAN point guard V1 ya está aplicado; no se modifica nada.")
        return 0

    updated = original
    updated = replace_once(updated, OLD_CONFIG, NEW_CONFIG, "config HUMAN guard")
    updated = replace_once(updated, OLD_KINDS, NEW_KINDS, "manual point kinds")
    updated = replace_once(updated, OLD_WEIGHTS, NEW_WEIGHTS, "manual point weights")

    if HELPER_ANCHOR not in updated:
        raise RuntimeError("No encuentro el punto de inserción de HUMAN guard helpers.")
    updated = updated.replace(
        HELPER_ANCHOR,
        HELPER_CODE + HELPER_ANCHOR,
        1,
    )

    updated = replace_once(
        updated,
        OLD_CANDIDATE,
        NEW_CANDIDATE,
        "candidate HUMAN guard metadata",
    )
    updated = replace_once(
        updated,
        OLD_CHOOSE,
        NEW_CHOOSE,
        "guarded candidate selection",
    )
    updated = replace_once(
        updated,
        OLD_WARNING,
        NEW_WARNING,
        "guard intervention warning",
    )
    updated = replace_once(
        updated,
        OLD_DIAG,
        NEW_DIAG,
        "guard diagnostics",
    )

    # Do not touch disk unless every expected edit succeeded.
    compile(updated, str(TARGET), "exec")

    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)

    TARGET.write_text(updated, encoding="utf-8")

    print("HUMAN point guard V1 aplicado.")
    print("Target :", TARGET)
    print("Backup :", BACKUP)
    print()
    print("Política nueva:")
    print("  - puntos HUMAN manuales: peso completo")
    print("  - líneas/arcos: refinan")
    print("  - refinamiento rechazado si desplaza puntos HUMAN más de:")
    print("      mean 1.5 px · p90 2.5 px · max 4.0 px")
    print()
    print("Ejecuta ahora:")
    print("  python -m supervisor.check_field_camera_human_guard_v1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

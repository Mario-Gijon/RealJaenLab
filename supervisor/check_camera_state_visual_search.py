#!/usr/bin/env python3
"""LOO benchmark for visual Projective Camera State search.

No target-frame HUMAN geometry is used for search or recovery.
Hidden HUMAN H is used only after the automatic result for evaluation.
Frame 115 HUMAN points are used only after recovery for evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from supervisor.field_camera_state_projective import (
    ProjectiveCameraStateConfig,
    build_projective_camera_state_model,
    evaluate_homography_manual_points,
    evaluate_homography_visible_grid,
    homography_from_state,
)
from supervisor.field_camera_state_visual_search import (
    VisualStateSearchConfig,
    recover_from_visual_state_search,
)
from supervisor.field_pitch_model import build_pitch_model


REVIEW = Path("supervisor/data/reviews/benchmark_easy_2min.json")


def calibration_for(calibrations, frame, require_h=False):
    for c in calibrations:
        if not isinstance(c, dict):
            continue
        try:
            same = int(c.get("frame")) == int(frame)
        except (TypeError, ValueError):
            continue
        if (
            same
            and c.get("source") == "manual"
            and c.get("purpose") == "match_setup"
            and (not require_h or isinstance(c.get("homography"), list))
        ):
            return c
    return None


def state_map(model):
    return {
        int(a["frame"]): np.asarray(a["state"], dtype=np.float64)
        for a in model.get("anchors", [])
        if isinstance(a, dict) and a.get("state") is not None
    }


def temporal_prior(model, frame):
    states = state_map(model)
    frames = sorted(states)
    left = [f for f in frames if f < frame]
    right = [f for f in frames if f > frame]
    if not left or not right:
        return None
    a, b = left[-1], right[0]
    alpha = (frame - a) / float(b - a)
    state = (1.0 - alpha) * states[a] + alpha * states[b]
    return a, b, alpha, state


def decode_h(model, state):
    result = homography_from_state(model, state)
    if result.get("status") != "ok":
        return None
    return np.asarray(result["homography"], dtype=np.float64)


def grid(pred, truth, width, height, length_m, pitch_width_m):
    if pred is None:
        return None
    result = evaluate_homography_visible_grid(
        pred,
        truth,
        image_width=width,
        image_height=height,
        pitch_length_m=length_m,
        pitch_width_m=pitch_width_m,
        config=ProjectiveCameraStateConfig(dimensions=5),
    )
    return result if result.get("status") == "ok" else None


def fmt(value, digits=1):
    return "—" if value is None else f"{float(value):.{digits}f}"


def main():
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    field = review["field"]
    calibrations = field.get("calibrations", [])
    strong = sorted({int(x) for x in field["match_setup"]["completed_frames"]})
    interior = strong[1:-1]

    length_m = float(field["pitch_model"]["length_m"])
    pitch_width_m = float(field["pitch_model"]["width_m"])
    pitch_model = build_pitch_model(length_m, pitch_width_m)

    video_path = Path("input") / review["video"]["filename"]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or review["video"].get("width") or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or review["video"].get("height") or 0)

    needed = sorted(set(interior + [115]))
    images = {}
    for frame in needed:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        if ok and image is not None:
            images[frame] = image
    cap.release()

    cfg = VisualStateSearchConfig()

    print()
    print("PROJECTIVE CAMERA STATE 5D · VISUAL STATE SEARCH · LOO")
    print("Strong anchors:", strong)
    print("Interior holdouts:", interior)
    print(f"Pitch: {length_m:g} × {pitch_width_m:g} m")
    print(
        f"Search rounds={cfg.round_scales} · population={cfg.population_per_round} · "
        f"beam={cfg.beam_size} · final candidates={cfg.final_candidate_count}"
    )
    print(
        f"Final local observer: ±{cfg.observer_search_half_width_px:g}px · "
        f"confidence≥{cfg.observer_minimum_confidence:.2f}"
    )
    print()

    records = []

    for frame in interior:
        hidden = calibration_for(calibrations, frame, require_h=True)
        image = images.get(frame)

        if hidden is None or image is None:
            records.append({"frame": frame, "status": "missing_input"})
            continue

        model = build_projective_camera_state_model(
            calibrations,
            [f for f in strong if f != frame],
            image_width=width,
            image_height=height,
            pitch_length_m=length_m,
            pitch_width_m=pitch_width_m,
            config=ProjectiveCameraStateConfig(dimensions=5),
        )

        if model.get("status") != "ok":
            records.append({"frame": frame, "status": model.get("status")})
            continue

        temporal = temporal_prior(model, frame)
        if temporal is None:
            records.append({"frame": frame, "status": "not_bracketed"})
            continue

        left, right, alpha, prior_state = temporal
        prior_h = decode_h(model, prior_state)
        truth_h = np.asarray(hidden["homography"], dtype=np.float64)
        prior_grid = grid(prior_h, truth_h, width, height, length_m, pitch_width_m)

        result = recover_from_visual_state_search(
            model,
            prior_state,
            image,
            pitch_model,
            query_frame=frame,
            config=cfg,
        )

        recovered_grid = (
            grid(
                np.asarray(result["homography"], dtype=np.float64),
                truth_h,
                width,
                height,
                length_m,
                pitch_width_m,
            )
            if result.get("status") == "ok"
            and isinstance(result.get("homography"), list)
            else None
        )

        searched_h = None
        candidate = result.get("search_candidate")
        if isinstance(candidate, dict) and isinstance(candidate.get("homography"), list):
            searched_h = np.asarray(candidate["homography"], dtype=np.float64)

        searched_grid = grid(
            searched_h,
            truth_h,
            width,
            height,
            length_m,
            pitch_width_m,
        )

        record = {
            "frame": frame,
            "status": result.get("status"),
            "left": left,
            "right": right,
            "alpha": alpha,
            "prior_grid": prior_grid,
            "searched_grid": searched_grid,
            "recovered_grid": recovered_grid,
            "automatic_line_count": result.get("automatic_line_count", 0),
            "automatic_semantics": [
                (
                    item.get("semantic"),
                    float(item.get("confidence") or 0.0),
                    item.get("orientation_family"),
                )
                for item in result.get("automatic_semantics", [])
            ],
            "family_counts": result.get("family_counts", {}),
            "state_shift_z_rms": result.get("state_shift_z_rms"),
            "fit_rms_sigma": result.get("fit_rms_sigma"),
            "search_paint_score": (
                candidate.get("paint_score")
                if isinstance(candidate, dict)
                else None
            ),
            "search_prior_radius": (
                candidate.get("prior_radius")
                if isinstance(candidate, dict)
                else None
            ),
            "search_candidate_count": result.get("search", {}).get("candidate_count"),
        }
        records.append(record)

        print(
            f"f {frame:>4} · status={record['status']} · "
            f"prior={fmt(prior_grid.get('median_px') if prior_grid else None)}px · "
            f"searched={fmt(searched_grid.get('median_px') if searched_grid else None)}px · "
            f"recovered={fmt(recovered_grid.get('median_px') if recovered_grid else None)}px · "
            f"lines={record['automatic_line_count']} · "
            f"families=x{record['family_counts'].get('x_constant',0)}/"
            f"y{record['family_counts'].get('y_constant',0)} · "
            f"radius={fmt(record['search_prior_radius'],2)} · "
            f"shift-z={fmt(record['state_shift_z_rms'],2)} · "
            f"fitσ={fmt(record['fit_rms_sigma'],2)}"
        )
        print(
            "       lines:",
            ", ".join(
                f"{semantic}:{confidence:.2f}/{family}"
                for semantic, confidence, family in record["automatic_semantics"]
            ) or "—",
        )

    prior_values = np.asarray(
        [
            r["prior_grid"]["median_px"]
            for r in records
            if isinstance(r.get("prior_grid"), dict)
        ],
        dtype=np.float64,
    )
    searched_values = np.asarray(
        [
            r["searched_grid"]["median_px"]
            for r in records
            if isinstance(r.get("searched_grid"), dict)
        ],
        dtype=np.float64,
    )
    recovered_values = np.asarray(
        [
            r["recovered_grid"]["median_px"]
            for r in records
            if isinstance(r.get("recovered_grid"), dict)
        ],
        dtype=np.float64,
    )

    print()
    print("LOO SUMMARY")
    print(f"automatic recovered={len(recovered_values)}/{len(interior)}")
    print(f"prior median={fmt(float(np.median(prior_values)) if len(prior_values) else None)}px")
    print(f"searched-state median={fmt(float(np.median(searched_values)) if len(searched_values) else None)}px")
    print(f"final recovered median={fmt(float(np.median(recovered_values)) if len(recovered_values) else None)}px")
    print(
        f"final P90 medians="
        f"{fmt(float(np.percentile(recovered_values,90)) if len(recovered_values) else None)}px · "
        f"max median={fmt(float(np.max(recovered_values)) if len(recovered_values) else None)}px"
    )

    print()
    print("FRAME 115 · FULLY AUTOMATIC")

    model = build_projective_camera_state_model(
        calibrations,
        strong,
        image_width=width,
        image_height=height,
        pitch_length_m=length_m,
        pitch_width_m=pitch_width_m,
        config=ProjectiveCameraStateConfig(dimensions=5),
    )
    temporal = temporal_prior(model, 115)
    calibration_115 = calibration_for(calibrations, 115, require_h=False)
    image_115 = images.get(115)

    if temporal is None or calibration_115 is None or image_115 is None:
        print("No se pudo preparar frame 115.")
        print("No se ha modificado la review.")
        return

    left, right, alpha, prior_state = temporal
    prior_h = decode_h(model, prior_state)
    prior_human = evaluate_homography_manual_points(prior_h, calibration_115)

    result = recover_from_visual_state_search(
        model,
        prior_state,
        image_115,
        pitch_model,
        query_frame=115,
        config=cfg,
    )

    print(f"bracket={left}→115→{right} · alpha={alpha:.3f}")
    if prior_human.get("status") == "ok":
        print(
            f"prior HUMAN points median={prior_human['median_px']:.1f}px · "
            f"P90={prior_human['p90_px']:.1f}px · max={prior_human['max_px']:.1f}px"
        )

    print(
        f"automatic status={result.get('status')} · "
        f"confidence={result.get('confidence')} · "
        f"lines={result.get('automatic_line_count',0)} · "
        f"families={result.get('family_counts',{})}"
    )

    print(
        "automatic lines:",
        [
            (
                item.get("semantic"),
                round(float(item.get("confidence") or 0.0), 3),
                item.get("orientation_family"),
            )
            for item in result.get("automatic_semantics", [])
        ],
    )

    if result.get("status") != "ok" or not isinstance(result.get("homography"), list):
        print("No CAMERA: evidencia automática insuficiente o recuperación rechazada.")
        print("No se usa geometría HUMAN como fallback.")
        print("No se ha modificado la review.")
        return

    evaluation = evaluate_homography_manual_points(
        result["homography"],
        calibration_115,
    )

    if evaluation.get("status") == "ok":
        print(
            f"recovered HUMAN points median={evaluation['median_px']:.1f}px · "
            f"P90={evaluation['p90_px']:.1f}px · max={evaluation['max_px']:.1f}px"
        )

    print(
        f"search prior radius={float(result['search_candidate']['prior_radius']):.3f} · "
        f"state shift z RMS={float(result['state_shift_z_rms']):.3f} · "
        f"fit RMS sigma={float(result['fit_rms_sigma']):.3f}"
    )
    print("H:")
    print(json.dumps(result["homography"], indent=2))
    print()
    print("Los HUMAN del 115 se usan SOLO para evaluación.")
    print("No se ha modificado la review.")


if __name__ == "__main__":
    main()

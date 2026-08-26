#!/usr/bin/env python3
"""LOO benchmark for semantic-topology Visual State Search V2."""

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
from supervisor.field_camera_state_visual_search_v2 import (
    VisualTopologySearchConfig,
    recover_from_topology_search,
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
    return a, b, alpha, (1.0 - alpha) * states[a] + alpha * states[b]


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

    images = {}
    for frame in sorted(set(interior + [115])):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        if ok and image is not None:
            images[frame] = image
    cap.release()

    cfg = VisualTopologySearchConfig()

    print()
    print("PROJECTIVE CAMERA STATE 5D · VISUAL TOPOLOGY V2 · LOO")
    print("Strong anchors:", strong)
    print("Interior holdouts:", interior)
    print(f"Pitch: {length_m:g} × {pitch_width_m:g} m")
    print(
        f"radius={cfg.maximum_prior_radius:.1f} · penalty={cfg.temporal_penalty:.3f} · "
        f"rounds={cfg.round_scales}"
    )
    print("Circles/arcs used in search: NO")
    print()

    records = []

    for frame in interior:
        hidden = calibration_for(calibrations, frame, require_h=True)
        image = images.get(frame)
        if hidden is None or image is None:
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
        temporal = temporal_prior(model, frame)
        if model.get("status") != "ok" or temporal is None:
            continue

        left, right, alpha, prior_state = temporal
        truth_h = np.asarray(hidden["homography"], dtype=np.float64)
        prior_h = decode_h(model, prior_state)
        prior_grid = grid(prior_h, truth_h, width, height, length_m, pitch_width_m)

        result = recover_from_topology_search(
            model,
            prior_state,
            image,
            pitch_model,
            query_frame=frame,
            config=cfg,
        )

        candidate = result.get("search_candidate")
        searched_grid = None
        if isinstance(candidate, dict) and isinstance(candidate.get("homography"), list):
            searched_grid = grid(
                np.asarray(candidate["homography"], dtype=np.float64),
                truth_h,
                width,
                height,
                length_m,
                pitch_width_m,
            )

        recovered_grid = None
        if result.get("status") == "ok" and isinstance(result.get("homography"), list):
            recovered_grid = grid(
                np.asarray(result["homography"], dtype=np.float64),
                truth_h,
                width,
                height,
                length_m,
                pitch_width_m,
            )

        record = {
            "frame": frame,
            "status": result.get("status"),
            "prior": prior_grid,
            "searched": searched_grid,
            "recovered": recovered_grid,
            "candidate": candidate,
            "lines": result.get("automatic_semantics", []),
            "family": result.get("family_counts", {}),
            "shift": result.get("state_shift_z_rms"),
            "fit": result.get("fit_rms_sigma"),
        }
        records.append(record)

        print(
            f"f {frame:>4} · status={record['status']} · "
            f"prior={fmt(prior_grid.get('median_px') if prior_grid else None)}px · "
            f"searched={fmt(searched_grid.get('median_px') if searched_grid else None)}px · "
            f"recovered={fmt(recovered_grid.get('median_px') if recovered_grid else None)}px · "
            f"radius={fmt(candidate.get('prior_radius') if isinstance(candidate,dict) else None,2)} · "
            f"topology={fmt(candidate.get('topology_score') if isinstance(candidate,dict) else None,3)} · "
            f"coverage={fmt(candidate.get('line_coverage') if isinstance(candidate,dict) else None,2)} · "
            f"x/y={record['family'].get('x_constant',0)}/{record['family'].get('y_constant',0)} · "
            f"shift={fmt(record['shift'],2)}"
        )
        print(
            "       lines:",
            ", ".join(
                f"{item.get('semantic')}:{float(item.get('confidence') or 0):.2f}"
                for item in record["lines"]
            ) or "—",
        )

    prior_values = np.asarray(
        [r["prior"]["median_px"] for r in records if isinstance(r.get("prior"), dict)],
        dtype=np.float64,
    )
    searched_values = np.asarray(
        [r["searched"]["median_px"] for r in records if isinstance(r.get("searched"), dict)],
        dtype=np.float64,
    )
    recovered_values = np.asarray(
        [r["recovered"]["median_px"] for r in records if isinstance(r.get("recovered"), dict)],
        dtype=np.float64,
    )

    improved_search = sum(
        1
        for r in records
        if isinstance(r.get("prior"), dict)
        and isinstance(r.get("searched"), dict)
        and r["searched"]["median_px"] < r["prior"]["median_px"]
    )
    catastrophic = sum(
        1
        for r in records
        if isinstance(r.get("prior"), dict)
        and isinstance(r.get("searched"), dict)
        and r["searched"]["median_px"] > max(300.0, 2.0 * r["prior"]["median_px"])
    )

    print()
    print("LOO SUMMARY")
    print(f"searched improved={improved_search}/{len(interior)} · catastrophic={catastrophic}")
    print(f"prior median={fmt(np.median(prior_values) if len(prior_values) else None)}px")
    print(f"searched median={fmt(np.median(searched_values) if len(searched_values) else None)}px")
    print(f"automatic recovered={len(recovered_values)}/{len(interior)}")
    print(
        f"recovered median={fmt(np.median(recovered_values) if len(recovered_values) else None)}px · "
        f"P90={fmt(np.percentile(recovered_values,90) if len(recovered_values) else None)}px · "
        f"max={fmt(np.max(recovered_values) if len(recovered_values) else None)}px"
    )

    print()
    print("FRAME 115 · VISUAL TOPOLOGY V2 · FULLY AUTOMATIC")

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
    calibration_115 = calibration_for(calibrations, 115)
    image_115 = images.get(115)

    if temporal is None or calibration_115 is None or image_115 is None:
        print("No se pudo preparar 115.")
        print("No se ha modificado la review.")
        return

    left, right, alpha, prior_state = temporal
    prior_h = decode_h(model, prior_state)
    prior_eval = evaluate_homography_manual_points(prior_h, calibration_115)

    result = recover_from_topology_search(
        model,
        prior_state,
        image_115,
        pitch_model,
        query_frame=115,
        config=cfg,
    )

    print(f"bracket={left}→115→{right} · alpha={alpha:.3f}")
    if prior_eval.get("status") == "ok":
        print(
            f"prior HUMAN median={prior_eval['median_px']:.1f}px · "
            f"P90={prior_eval['p90_px']:.1f}px · max={prior_eval['max_px']:.1f}px"
        )

    candidate = result.get("search_candidate")
    print(
        f"status={result.get('status')} · confidence={result.get('confidence')} · "
        f"radius={fmt(candidate.get('prior_radius') if isinstance(candidate,dict) else None,2)} · "
        f"topology={fmt(candidate.get('topology_score') if isinstance(candidate,dict) else None,3)} · "
        f"coverage={fmt(candidate.get('line_coverage') if isinstance(candidate,dict) else None,2)}"
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
        print("No CAMERA: automatic evidence rejected.")
        print("No HUMAN fallback.")
        print("No se ha modificado la review.")
        return

    recovered_eval = evaluate_homography_manual_points(
        result["homography"],
        calibration_115,
    )
    if recovered_eval.get("status") == "ok":
        print(
            f"recovered HUMAN median={recovered_eval['median_px']:.1f}px · "
            f"P90={recovered_eval['p90_px']:.1f}px · max={recovered_eval['max_px']:.1f}px"
        )
    print(
        f"state shift z RMS={float(result['state_shift_z_rms']):.3f} · "
        f"fit RMS sigma={float(result['fit_rms_sigma']):.3f}"
    )
    print("H:")
    print(json.dumps(result["homography"], indent=2))
    print()
    print("HUMAN 115 used ONLY for evaluation.")
    print("No se ha modificado la review.")


if __name__ == "__main__":
    main()

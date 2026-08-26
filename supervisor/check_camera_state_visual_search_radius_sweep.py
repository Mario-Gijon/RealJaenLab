#!/usr/bin/env python3
"""LOO sweep for Visual Projective Camera State search radius/temporal penalty.

Purpose
-------
V1 visual search found strong semantic aliases far from the temporal prior.
The hidden HUMAN states showed a useful separation:

- hidden TRUE states: roughly radius 0.65 .. 2.52
- false V1 winners: roughly radius 3.24 .. 4.55

This benchmark changes NO core mathematics. It sweeps only:

    maximum_prior_radius
    temporal_penalty

For each interior HUMAN holdout:
- hide its calibration/H from the Projective State training model;
- build the normal 5-D temporal prior;
- search from raw image pixels only;
- recover from automatic visual lines only;
- use hidden HUMAN H only AFTERWARDS for evaluation.

Frame 115 is evaluated only after the best LOO configuration is chosen.
Nothing is persisted.
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

RADII = (2.4, 2.6, 2.8, 3.0, 3.2)
TEMPORAL_PENALTIES = (0.0, 0.01, 0.025)


def calibration_for(calibrations, frame, *, require_h=False):
    for calibration in calibrations:
        if not isinstance(calibration, dict):
            continue
        try:
            same = int(calibration.get("frame")) == int(frame)
        except (TypeError, ValueError):
            continue
        if (
            same
            and calibration.get("source") == "manual"
            and calibration.get("purpose") == "match_setup"
            and (
                not require_h
                or isinstance(calibration.get("homography"), list)
            )
        ):
            return calibration
    return None


def state_map(model):
    return {
        int(anchor["frame"]): np.asarray(
            anchor["state"],
            dtype=np.float64,
        )
        for anchor in model.get("anchors", [])
        if isinstance(anchor, dict)
        and anchor.get("state") is not None
    }


def temporal_prior(model, frame):
    states = state_map(model)
    frames = sorted(states)

    left = [value for value in frames if value < frame]
    right = [value for value in frames if value > frame]

    if not left or not right:
        return None

    a = left[-1]
    b = right[0]
    alpha = (frame - a) / float(b - a)

    state = (
        (1.0 - alpha) * states[a]
        + alpha * states[b]
    )

    return a, b, float(alpha), state


def decode_h(model, state):
    decoded = homography_from_state(model, state)
    if decoded.get("status") != "ok":
        return None
    try:
        homography = np.asarray(
            decoded["homography"],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None
    return (
        homography
        if homography.shape == (3, 3)
        and np.isfinite(homography).all()
        else None
    )


def visible_grid(
    predicted,
    truth,
    width,
    height,
    pitch_length,
    pitch_width,
):
    if predicted is None:
        return None

    result = evaluate_homography_visible_grid(
        predicted,
        truth,
        image_width=width,
        image_height=height,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=ProjectiveCameraStateConfig(
            dimensions=5
        ),
    )

    return (
        result
        if result.get("status") == "ok"
        else None
    )


def read_frames(capture, frames):
    output = {}

    for frame in sorted(set(frames)):
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame),
        )
        ok, image = capture.read()

        if ok and image is not None:
            output[int(frame)] = image

    return output


def make_config(radius, temporal_penalty):
    base = VisualStateSearchConfig()

    return VisualStateSearchConfig(
        random_seed=base.random_seed,
        round_scales=base.round_scales,
        population_per_round=base.population_per_round,
        beam_size=base.beam_size,
        final_candidate_count=base.final_candidate_count,
        minimum_state_scale=base.minimum_state_scale,
        maximum_prior_radius=float(radius),

        samples_per_line_min=base.samples_per_line_min,
        samples_per_line_max=base.samples_per_line_max,
        target_sample_spacing_px=base.target_sample_spacing_px,
        normal_offsets_px=base.normal_offsets_px,
        side_sample_distance_px=base.side_sample_distance_px,
        minimum_visible_line_px=base.minimum_visible_line_px,
        temporal_penalty=float(temporal_penalty),

        observer_search_half_width_px=base.observer_search_half_width_px,
        observer_minimum_confidence=base.observer_minimum_confidence,
        observer_max_angle_delta_deg=base.observer_max_angle_delta_deg,
    )


def evaluate_holdout(
    *,
    frame,
    strong_frames,
    calibrations,
    image,
    pitch_model,
    width,
    height,
    pitch_length,
    pitch_width,
    config,
):
    hidden = calibration_for(
        calibrations,
        frame,
        require_h=True,
    )

    if hidden is None:
        return {
            "frame": frame,
            "status": "missing_hidden_h",
        }

    training = [
        anchor
        for anchor in strong_frames
        if anchor != frame
    ]

    model = build_projective_camera_state_model(
        calibrations,
        training,
        image_width=width,
        image_height=height,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=ProjectiveCameraStateConfig(
            dimensions=5
        ),
    )

    if model.get("status") != "ok":
        return {
            "frame": frame,
            "status": model.get(
                "status",
                "projective_state_failed",
            ),
        }

    temporal = temporal_prior(
        model,
        frame,
    )

    if temporal is None:
        return {
            "frame": frame,
            "status": "not_bracketed",
        }

    left, right, alpha, prior_state = temporal

    prior_h = decode_h(
        model,
        prior_state,
    )

    if prior_h is None:
        return {
            "frame": frame,
            "status": "prior_decode_failed",
        }

    truth_h = np.asarray(
        hidden["homography"],
        dtype=np.float64,
    )

    prior_grid = visible_grid(
        prior_h,
        truth_h,
        width,
        height,
        pitch_length,
        pitch_width,
    )

    result = recover_from_visual_state_search(
        model,
        prior_state,
        image,
        pitch_model,
        query_frame=frame,
        config=config,
    )

    search = result.get(
        "search",
        {},
    )

    top_candidate = (
        search.get("candidates", [None])[0]
        if isinstance(search, dict)
        and search.get("candidates")
        else None
    )

    searched_grid = None
    searched_radius = None
    searched_paint = None

    if isinstance(top_candidate, dict):
        try:
            searched_h = np.asarray(
                top_candidate["homography"],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            searched_h = None

        if (
            searched_h is not None
            and searched_h.shape == (3, 3)
        ):
            searched_grid = visible_grid(
                searched_h,
                truth_h,
                width,
                height,
                pitch_length,
                pitch_width,
            )

        searched_radius = top_candidate.get(
            "prior_radius"
        )
        searched_paint = top_candidate.get(
            "paint_score"
        )

    recovered_grid = None

    if (
        result.get("status") == "ok"
        and isinstance(
            result.get("homography"),
            list,
        )
    ):
        recovered_h = np.asarray(
            result["homography"],
            dtype=np.float64,
        )

        recovered_grid = visible_grid(
            recovered_h,
            truth_h,
            width,
            height,
            pitch_length,
            pitch_width,
        )

    family_counts = result.get(
        "family_counts",
        {},
    )

    return {
        "frame": frame,
        "status": result.get(
            "status",
            "unknown",
        ),
        "left": left,
        "right": right,
        "alpha": alpha,
        "prior_grid": prior_grid,
        "searched_grid": searched_grid,
        "recovered_grid": recovered_grid,
        "searched_radius": searched_radius,
        "searched_paint_score": searched_paint,
        "automatic_line_count": int(
            result.get(
                "automatic_line_count",
                0,
            )
            or 0
        ),
        "family_counts": family_counts,
        "state_shift_z_rms": result.get(
            "state_shift_z_rms"
        ),
        "fit_rms_sigma": result.get(
            "fit_rms_sigma"
        ),
        "automatic_semantics": [
            {
                "semantic": item.get(
                    "semantic"
                ),
                "confidence": float(
                    item.get(
                        "confidence",
                        0.0,
                    )
                    or 0.0
                ),
                "orientation_family": item.get(
                    "orientation_family"
                ),
            }
            for item in result.get(
                "automatic_semantics",
                [],
            )
        ],
    }


def summarize(records):
    prior = np.asarray(
        [
            record["prior_grid"]["median_px"]
            for record in records
            if isinstance(
                record.get("prior_grid"),
                dict,
            )
        ],
        dtype=np.float64,
    )

    searched_records = [
        record
        for record in records
        if isinstance(
            record.get("searched_grid"),
            dict,
        )
    ]

    searched = np.asarray(
        [
            record["searched_grid"]["median_px"]
            for record in searched_records
        ],
        dtype=np.float64,
    )

    recovered_records = [
        record
        for record in records
        if isinstance(
            record.get("recovered_grid"),
            dict,
        )
    ]

    recovered = np.asarray(
        [
            record["recovered_grid"]["median_px"]
            for record in recovered_records
        ],
        dtype=np.float64,
    )

    improved_search = sum(
        1
        for record in searched_records
        if (
            isinstance(
                record.get("prior_grid"),
                dict,
            )
            and record["searched_grid"]["median_px"]
            < record["prior_grid"]["median_px"]
        )
    )

    catastrophic_search = sum(
        1
        for record in searched_records
        if (
            isinstance(
                record.get("prior_grid"),
                dict,
            )
            and record["searched_grid"]["median_px"]
            > max(
                300.0,
                2.0
                * record["prior_grid"]["median_px"],
            )
        )
    )

    improved_recovery = sum(
        1
        for record in recovered_records
        if (
            isinstance(
                record.get("prior_grid"),
                dict,
            )
            and record["recovered_grid"]["median_px"]
            < record["prior_grid"]["median_px"]
        )
    )

    return {
        "records": records,
        "holdout_count": len(records),
        "searched_count": len(searched_records),
        "recovered_count": len(recovered_records),
        "improved_search_count": improved_search,
        "catastrophic_search_count": catastrophic_search,
        "improved_recovery_count": improved_recovery,

        "prior_median_px": (
            float(np.median(prior))
            if len(prior)
            else None
        ),
        "searched_median_px": (
            float(np.median(searched))
            if len(searched)
            else None
        ),
        "searched_p90_px": (
            float(
                np.percentile(
                    searched,
                    90,
                )
            )
            if len(searched)
            else None
        ),
        "searched_max_px": (
            float(np.max(searched))
            if len(searched)
            else None
        ),
        "recovered_median_px": (
            float(np.median(recovered))
            if len(recovered)
            else None
        ),
        "recovered_p90_px": (
            float(
                np.percentile(
                    recovered,
                    90,
                )
            )
            if len(recovered)
            else None
        ),
        "recovered_max_px": (
            float(np.max(recovered))
            if len(recovered)
            else None
        ),
    }


def rank_key(configuration):
    summary = configuration["summary"]

    # We are fixing aliasing first:
    # 1. avoid catastrophic false camera states;
    # 2. make image search improve as many LOO holdouts as possible;
    # 3. keep/use recoverable automatic evidence;
    # 4. minimize searched and recovered errors.
    return (
        int(
            summary[
                "catastrophic_search_count"
            ]
        ),
        -int(
            summary[
                "improved_search_count"
            ]
        ),
        -int(
            summary[
                "recovered_count"
            ]
        ),
        (
            float(
                summary[
                    "searched_median_px"
                ]
            )
            if summary[
                "searched_median_px"
            ] is not None
            else float("inf")
        ),
        (
            float(
                summary[
                    "searched_max_px"
                ]
            )
            if summary[
                "searched_max_px"
            ] is not None
            else float("inf")
        ),
        (
            float(
                summary[
                    "recovered_median_px"
                ]
            )
            if summary[
                "recovered_median_px"
            ] is not None
            else float("inf")
        ),
    )


def fmt(value, digits=1):
    return (
        "—"
        if value is None
        else f"{float(value):.{digits}f}"
    )


def main():
    review = json.loads(
        REVIEW.read_text(
            encoding="utf-8"
        )
    )

    field = review["field"]
    calibrations = field.get(
        "calibrations",
        [],
    )

    strong_frames = sorted(
        {
            int(frame)
            for frame in field[
                "match_setup"
            ].get(
                "completed_frames",
                [],
            )
        }
    )

    interior = strong_frames[1:-1]

    pitch_length = float(
        field[
            "pitch_model"
        ][
            "length_m"
        ]
    )
    pitch_width = float(
        field[
            "pitch_model"
        ][
            "width_m"
        ]
    )

    pitch_model = build_pitch_model(
        pitch_length,
        pitch_width,
    )

    video_path = (
        Path("input")
        / review["video"]["filename"]
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise SystemExit(
            f"No se pudo abrir {video_path}"
        )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
        or review["video"].get(
            "width"
        )
        or 0
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
        or review["video"].get(
            "height"
        )
        or 0
    )

    images = read_frames(
        capture,
        interior + [115],
    )

    capture.release()

    print()
    print(
        "VISUAL PROJECTIVE STATE SEARCH · RADIUS/PENALTY LOO SWEEP"
    )
    print(
        "Strong anchors:",
        strong_frames,
    )
    print(
        "Interior holdouts:",
        interior,
    )
    print(
        f"Pitch: {pitch_length:g} × {pitch_width:g} m"
    )
    print(
        "Radii:",
        RADII,
    )
    print(
        "Temporal penalties:",
        TEMPORAL_PENALTIES,
    )
    print()

    configurations = []

    for radius in RADII:
        for penalty in TEMPORAL_PENALTIES:
            cfg = make_config(
                radius,
                penalty,
            )

            records = []

            for frame in interior:
                image = images.get(
                    frame
                )

                if image is None:
                    records.append(
                        {
                            "frame": frame,
                            "status": "frame_read_failed",
                        }
                    )
                    continue

                records.append(
                    evaluate_holdout(
                        frame=frame,
                        strong_frames=strong_frames,
                        calibrations=calibrations,
                        image=image,
                        pitch_model=pitch_model,
                        width=width,
                        height=height,
                        pitch_length=pitch_length,
                        pitch_width=pitch_width,
                        config=cfg,
                    )
                )

            summary = summarize(
                records
            )

            configuration = {
                "radius": radius,
                "penalty": penalty,
                "summary": summary,
            }

            configurations.append(
                configuration
            )

            print(
                f"radius={radius:.1f} · "
                f"penalty={penalty:.3f} · "
                f"searched={summary['searched_count']}/{summary['holdout_count']} · "
                f"improved={summary['improved_search_count']} · "
                f"catastrophic={summary['catastrophic_search_count']} · "
                f"search med={fmt(summary['searched_median_px'])}px · "
                f"search max={fmt(summary['searched_max_px'])}px · "
                f"recovered={summary['recovered_count']}/{summary['holdout_count']} · "
                f"rec med={fmt(summary['recovered_median_px'])}px · "
                f"rec max={fmt(summary['recovered_max_px'])}px"
            )

    best = min(
        configurations,
        key=rank_key,
    )

    summary = best["summary"]

    print()
    print(
        "BEST SAFE LOO CONFIG"
    )
    print(
        f"maximum_prior_radius={best['radius']:.1f}"
    )
    print(
        f"temporal_penalty={best['penalty']:.3f}"
    )
    print(
        f"search improved={summary['improved_search_count']}/{summary['holdout_count']} · "
        f"catastrophic={summary['catastrophic_search_count']}"
    )
    print(
        f"search median={fmt(summary['searched_median_px'])}px · "
        f"P90={fmt(summary['searched_p90_px'])}px · "
        f"max={fmt(summary['searched_max_px'])}px"
    )
    print(
        f"recovered={summary['recovered_count']}/{summary['holdout_count']} · "
        f"recovered median={fmt(summary['recovered_median_px'])}px · "
        f"P90={fmt(summary['recovered_p90_px'])}px · "
        f"max={fmt(summary['recovered_max_px'])}px"
    )
    print()

    for record in summary["records"]:
        prior_error = (
            record["prior_grid"][
                "median_px"
            ]
            if isinstance(
                record.get(
                    "prior_grid"
                ),
                dict,
            )
            else None
        )

        searched_error = (
            record["searched_grid"][
                "median_px"
            ]
            if isinstance(
                record.get(
                    "searched_grid"
                ),
                dict,
            )
            else None
        )

        recovered_error = (
            record["recovered_grid"][
                "median_px"
            ]
            if isinstance(
                record.get(
                    "recovered_grid"
                ),
                dict,
            )
            else None
        )

        family = record.get(
            "family_counts",
            {},
        )

        print(
            f"f {record['frame']:>4} · "
            f"status={record.get('status')} · "
            f"prior={fmt(prior_error)}px · "
            f"searched={fmt(searched_error)}px · "
            f"recovered={fmt(recovered_error)}px · "
            f"radius={fmt(record.get('searched_radius'),2)} · "
            f"paint={fmt(record.get('searched_paint_score'),3)} · "
            f"lines={record.get('automatic_line_count',0)} · "
            f"families=x{family.get('x_constant',0)}/"
            f"y{family.get('y_constant',0)}"
        )

        print(
            "       lines:",
            ", ".join(
                f"{item['semantic']}:{item['confidence']:.2f}/"
                f"{item['orientation_family']}"
                for item in record.get(
                    "automatic_semantics",
                    [],
                )
            )
            or "—",
        )

    print()
    print(
        "FRAME 115 · BEST LOO CONFIG · FULLY AUTOMATIC"
    )

    model = build_projective_camera_state_model(
        calibrations,
        strong_frames,
        image_width=width,
        image_height=height,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=ProjectiveCameraStateConfig(
            dimensions=5
        ),
    )

    temporal = temporal_prior(
        model,
        115,
    )

    calibration_115 = calibration_for(
        calibrations,
        115,
        require_h=False,
    )

    image_115 = images.get(
        115
    )

    if (
        model.get("status") != "ok"
        or temporal is None
        or calibration_115 is None
        or image_115 is None
    ):
        print(
            "No se pudo preparar frame 115."
        )
        print(
            "No se ha modificado la review."
        )
        return

    left, right, alpha, prior_state = temporal

    prior_h = decode_h(
        model,
        prior_state,
    )

    prior_human = evaluate_homography_manual_points(
        prior_h,
        calibration_115,
    )

    cfg = make_config(
        best["radius"],
        best["penalty"],
    )

    result = recover_from_visual_state_search(
        model,
        prior_state,
        image_115,
        pitch_model,
        query_frame=115,
        config=cfg,
    )

    print(
        f"bracket={left}→115→{right} · "
        f"alpha={alpha:.3f}"
    )

    if prior_human.get(
        "status"
    ) == "ok":
        print(
            f"prior HUMAN points median="
            f"{prior_human['median_px']:.1f}px · "
            f"P90={prior_human['p90_px']:.1f}px · "
            f"max={prior_human['max_px']:.1f}px"
        )

    search = result.get(
        "search",
        {},
    )

    top = (
        search.get(
            "candidates",
            [None],
        )[0]
        if isinstance(
            search,
            dict,
        )
        and search.get(
            "candidates"
        )
        else None
    )

    print(
        f"automatic status={result.get('status')} · "
        f"confidence={result.get('confidence')} · "
        f"top search radius="
        f"{fmt(top.get('prior_radius') if isinstance(top,dict) else None,2)} · "
        f"top paint="
        f"{fmt(top.get('paint_score') if isinstance(top,dict) else None,3)}"
    )

    print(
        "automatic lines:",
        [
            (
                item.get(
                    "semantic"
                ),
                round(
                    float(
                        item.get(
                            "confidence",
                            0.0,
                        )
                        or 0.0
                    ),
                    3,
                ),
                item.get(
                    "orientation_family"
                ),
            )
            for item in result.get(
                "automatic_semantics",
                [],
            )
        ],
    )

    if (
        result.get("status") != "ok"
        or not isinstance(
            result.get(
                "homography"
            ),
            list,
        )
    ):
        print(
            "No CAMERA: evidencia automática insuficiente o recuperación rechazada."
        )
        print(
            "No se usa geometría HUMAN como fallback."
        )
        print(
            "No se ha modificado la review."
        )
        return

    recovered_human = evaluate_homography_manual_points(
        result["homography"],
        calibration_115,
    )

    if recovered_human.get(
        "status"
    ) == "ok":
        print(
            f"recovered HUMAN points median="
            f"{recovered_human['median_px']:.1f}px · "
            f"P90={recovered_human['p90_px']:.1f}px · "
            f"max={recovered_human['max_px']:.1f}px"
        )

    print(
        f"state shift z RMS="
        f"{float(result['state_shift_z_rms']):.3f} · "
        f"fit RMS sigma="
        f"{float(result['fit_rms_sigma']):.3f}"
    )

    print(
        "H:"
    )
    print(
        json.dumps(
            result["homography"],
            indent=2,
        )
    )

    print()
    print(
        "Los HUMAN del 115 se usan SOLO para evaluación."
    )
    print(
        "No se ha modificado la review."
    )


if __name__ == "__main__":
    main()

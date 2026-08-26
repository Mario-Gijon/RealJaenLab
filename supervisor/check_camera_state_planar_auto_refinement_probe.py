#!/usr/bin/env python3
"""Probe: refine selected planar-flow H with automatic straight-line evidence.

This is analysis-only.

Pipeline per LOO holdout:
    strong HUMAN anchors
    -> left/right ground-plane projective flow
    -> reliability selection of ONE direct propagated H
    -> use that direct H ONLY to localize automatic straight-line searches
    -> refine validated 5-D Projective Camera State with those line constraints
    -> evaluate against hidden HUMAN H

Two 5-D refinement priors are compared:
    temporal       = normal linear temporal Projective State prior
    planar_state   = selected direct planar-flow H projected into the 5-D model

The direct propagated H is NEVER averaged and never overwritten during search.
Circles/arcs are not used.

Hidden target HUMAN geometry is used only after automatic processing for LOO
evaluation and semantic diagnostics. Frame 115 is held out from tuning.
Nothing is persisted.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from supervisor.field_camera_state_auto_lines import (
    AutoLineEvidenceConfig,
    observe_automatic_pitch_lines,
)
from supervisor.field_camera_state_planar_flow import (
    PlanarFlowConfig,
    homography_disagreement_px,
    propagate_pitch_homography,
)
from supervisor.field_camera_state_planar_selection import (
    PlanarSelectionConfig,
    select_planar_flow_camera,
)
from supervisor.field_camera_state_projective import (
    ProjectiveCameraStateConfig,
    build_projective_camera_state_model,
    evaluate_homography_manual_points,
    evaluate_homography_visible_grid,
    homography_from_state,
    projective_jet_from_homography,
)
from supervisor.field_camera_state_recovery import (
    recover_camera_from_projective_state,
)
from supervisor.field_pitch_model import build_pitch_model


REVIEW = Path("supervisor/data/reviews/benchmark_easy_2min.json")

SEARCH_WIDTHS = (12.0, 24.0, 36.0, 48.0, 60.0)
MIN_CONFIDENCES = (0.50, 0.55, 0.60)
PRIOR_MODES = ("temporal", "planar_state")

# Frozen from the preceding LOO direct-selection sweep.
SELECTION_CONFIG = PlanarSelectionConfig(
    step_decay_tau=20.0,
    inlier_ratio_power=6.0,
    reprojection_decay_px=1.0,
)


def calibration_for(calibrations, frame, require_h=False):
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

    return {
        "left": a,
        "right": b,
        "alpha": float(alpha),
        "state": (
            (1.0 - alpha) * states[a]
            + alpha * states[b]
        ),
    }


def state_from_homography(model, homography):
    feature = projective_jet_from_homography(
        homography,
        image_width=model["image_width"],
        image_height=model["image_height"],
        pitch_length_m=model["pitch_length_m"],
        pitch_width_m=model["pitch_width_m"],
        maximum_condition=model["config"]["maximum_h_condition"],
    )

    if feature is None:
        return None

    feature = np.asarray(feature, dtype=np.float64)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    std = np.asarray(model["feature_std"], dtype=np.float64)
    pca_mean = np.asarray(model["pca_mean"], dtype=np.float64)
    components = np.asarray(model["pca_components"], dtype=np.float64)

    state = (
        ((feature - mean) / std - pca_mean)
        @ components.T
    )

    return (
        state
        if np.isfinite(state).all()
        else None
    )


def decode_h(model, state):
    result = homography_from_state(
        model,
        state,
    )

    if result.get("status") != "ok":
        return None

    try:
        homography = np.asarray(
            result["homography"],
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


def normalise_line(value):
    try:
        line = np.asarray(
            value,
            dtype=np.float64,
        ).reshape(3)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(line).all():
        return None

    norm = math.hypot(
        float(line[0]),
        float(line[1]),
    )

    return (
        line / norm
        if norm > 1e-9
        else None
    )


def project_world(point, homography):
    q = np.asarray(
        [
            float(point["x"]),
            float(point["y"]),
            1.0,
        ],
        dtype=np.float64,
    ) @ np.asarray(
        homography,
        dtype=np.float64,
    ).T

    if (
        not np.isfinite(q).all()
        or abs(float(q[2])) < 1e-9
    ):
        return None

    projected = q[:2] / q[2]

    return (
        projected
        if np.isfinite(projected).all()
        else None
    )


def semantic_line_errors(
    automatic,
    truth_h,
    pitch_model,
):
    definitions = {
        item["id"]: item
        for item in pitch_model.get(
            "calibration_lines",
            [],
        )
        if isinstance(item, dict)
        and item.get("id")
    }

    output = []

    for observation in automatic.get(
        "observations",
        [],
    ):
        semantic = observation.get(
            "semantic"
        )
        definition = definitions.get(
            semantic
        )
        observed = normalise_line(
            observation.get(
                "observed_line"
            )
        )

        if definition is None or observed is None:
            continue

        world = definition["world"]
        first = project_world(
            world["start"],
            truth_h,
        )
        second = project_world(
            world["end"],
            truth_h,
        )

        if first is None or second is None:
            continue

        truth_line = normalise_line(
            np.cross(
                np.asarray(
                    [
                        first[0],
                        first[1],
                        1.0,
                    ],
                    dtype=np.float64,
                ),
                np.asarray(
                    [
                        second[0],
                        second[1],
                        1.0,
                    ],
                    dtype=np.float64,
                ),
            )
        )

        if truth_line is None:
            continue

        midpoint = 0.5 * (
            first + second
        )

        normal_error = abs(
            float(
                observed[0] * midpoint[0]
                + observed[1] * midpoint[1]
                + observed[2]
            )
        )

        output.append(
            {
                "semantic": semantic,
                "normal_error_px": normal_error,
                "confidence": float(
                    observation.get(
                        "confidence",
                        0.0,
                    )
                    or 0.0
                ),
                "orientation_family": observation.get(
                    "orientation_family"
                ),
            }
        )

    return output


def read_frames(
    capture,
    frames,
):
    output = {}

    for frame in sorted(
        set(
            int(value)
            for value in frames
        )
    ):
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame,
        )

        ok, image = capture.read()

        if ok and image is not None:
            output[frame] = image

    return output


def prepare_holdout(
    *,
    frame,
    strong,
    calibrations,
    video_path,
    flow_config,
    width,
    height,
    pitch_length,
    pitch_width,
):
    hidden = calibration_for(
        calibrations,
        frame,
        require_h=True,
    )

    if hidden is None:
        return None

    left_frame = max(
        anchor
        for anchor in strong
        if anchor < frame
    )
    right_frame = min(
        anchor
        for anchor in strong
        if anchor > frame
    )

    left_calibration = calibration_for(
        calibrations,
        left_frame,
        require_h=True,
    )
    right_calibration = calibration_for(
        calibrations,
        right_frame,
        require_h=True,
    )

    left_path = propagate_pitch_homography(
        video_path,
        left_frame,
        frame,
        left_calibration["homography"],
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=flow_config,
    )

    right_path = propagate_pitch_homography(
        video_path,
        right_frame,
        frame,
        right_calibration["homography"],
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=flow_config,
    )

    disagreement = (
        homography_disagreement_px(
            left_path["homography"],
            right_path["homography"],
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            image_width=width,
            image_height=height,
        )
        if left_path.get("status") == "ok"
        and right_path.get("status") == "ok"
        else None
    )

    selected = select_planar_flow_camera(
        left_path,
        right_path,
        disagreement_px=(
            disagreement["median_px"]
            if isinstance(
                disagreement,
                dict,
            )
            and disagreement.get(
                "status"
            ) == "ok"
            else None
        ),
        config=SELECTION_CONFIG,
    )

    if (
        selected.get("status") != "ok"
        or not isinstance(
            selected.get(
                "homography"
            ),
            list,
        )
    ):
        return None

    model = build_projective_camera_state_model(
        calibrations,
        [
            anchor
            for anchor in strong
            if anchor != frame
        ],
        image_width=width,
        image_height=height,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=ProjectiveCameraStateConfig(
            dimensions=5
        ),
    )

    if model.get("status") != "ok":
        return None

    temporal = temporal_prior(
        model,
        frame,
    )

    if temporal is None:
        return None

    truth_h = np.asarray(
        hidden["homography"],
        dtype=np.float64,
    )
    selected_h = np.asarray(
        selected["homography"],
        dtype=np.float64,
    )

    baseline_grid = visible_grid(
        selected_h,
        truth_h,
        width,
        height,
        pitch_length,
        pitch_width,
    )

    planar_state = state_from_homography(
        model,
        selected_h,
    )

    projected_h = (
        decode_h(
            model,
            planar_state,
        )
        if planar_state is not None
        else None
    )

    projected_grid = (
        visible_grid(
            projected_h,
            truth_h,
            width,
            height,
            pitch_length,
            pitch_width,
        )
        if projected_h is not None
        else None
    )

    return {
        "frame": frame,
        "truth_h": truth_h,
        "hidden": hidden,
        "left_frame": left_frame,
        "right_frame": right_frame,
        "selected": selected,
        "selected_h": selected_h,
        "baseline_grid": baseline_grid,
        "model": model,
        "temporal": temporal,
        "planar_state": planar_state,
        "projected_grid": projected_grid,
    }


def evaluate_configuration(
    holdouts,
    images,
    pitch_model,
    width,
    height,
    pitch_length,
    pitch_width,
    *,
    search_width,
    minimum_confidence,
    prior_mode,
):
    records = []

    for item in holdouts:
        frame = item["frame"]
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

        automatic = observe_automatic_pitch_lines(
            image,
            item["selected_h"],
            pitch_model,
            width,
            height,
            config=AutoLineEvidenceConfig(
                search_half_width_px=float(
                    search_width
                ),
                minimum_observer_confidence=float(
                    minimum_confidence
                ),
                observer_max_angle_delta_deg=12.0,
            ),
        )

        semantic = semantic_line_errors(
            automatic,
            item["truth_h"],
            pitch_model,
        )

        baseline = item["baseline_grid"]

        record = {
            "frame": frame,
            "status": "insufficient_automatic_geometry",
            "baseline_grid": baseline,
            "projected_grid": item[
                "projected_grid"
            ],
            "automatic_line_count": len(
                automatic.get(
                    "constraints",
                    [],
                )
            ),
            "family_counts": automatic.get(
                "family_counts",
                {},
            ),
            "automatic_semantics": automatic.get(
                "observations",
                [],
            ),
            "semantic_diagnostics": semantic,
            "recovered_grid": None,
            "state_shift_z_rms": None,
            "fit_rms_sigma": None,
        }

        if not automatic.get(
            "eligible"
        ):
            records.append(
                record
            )
            continue

        if prior_mode == "temporal":
            refinement_prior = item[
                "temporal"
            ][
                "state"
            ]
        elif prior_mode == "planar_state":
            refinement_prior = item[
                "planar_state"
            ]
        else:
            raise ValueError(
                prior_mode
            )

        if refinement_prior is None:
            record[
                "status"
            ] = "refinement_prior_unavailable"
            records.append(
                record
            )
            continue

        recovered = recover_camera_from_projective_state(
            item["model"],
            refinement_prior,
            [],
            automatic["constraints"],
            [],
            prior_confidence="good",
        )

        record[
            "state_shift_z_rms"
        ] = recovered.get(
            "state_shift_z_rms"
        )
        record[
            "fit_rms_sigma"
        ] = recovered.get(
            "fit_rms_sigma"
        )

        if (
            recovered.get(
                "status"
            ) != "ok"
            or not isinstance(
                recovered.get(
                    "homography"
                ),
                list,
            )
        ):
            record[
                "status"
            ] = "recovery_rejected"
            records.append(
                record
            )
            continue

        recovered_grid = visible_grid(
            recovered["homography"],
            item["truth_h"],
            width,
            height,
            pitch_length,
            pitch_width,
        )

        if recovered_grid is None:
            record[
                "status"
            ] = "evaluation_failed"
            records.append(
                record
            )
            continue

        record[
            "status"
        ] = "recovered"
        record[
            "recovered_grid"
        ] = recovered_grid

        records.append(
            record
        )

    return records


def summarize(records):
    baselines = np.asarray(
        [
            record[
                "baseline_grid"
            ][
                "median_px"
            ]
            for record in records
            if isinstance(
                record.get(
                    "baseline_grid"
                ),
                dict,
            )
        ],
        dtype=np.float64,
    )

    recovered_records = [
        record
        for record in records
        if isinstance(
            record.get(
                "recovered_grid"
            ),
            dict,
        )
    ]

    recovered = np.asarray(
        [
            record[
                "recovered_grid"
            ][
                "median_px"
            ]
            for record in recovered_records
        ],
        dtype=np.float64,
    )

    improved = sum(
        1
        for record in recovered_records
        if (
            isinstance(
                record.get(
                    "baseline_grid"
                ),
                dict,
            )
            and record[
                "recovered_grid"
            ][
                "median_px"
            ]
            < record[
                "baseline_grid"
            ][
                "median_px"
            ]
        )
    )

    catastrophic = sum(
        1
        for record in recovered_records
        if (
            isinstance(
                record.get(
                    "baseline_grid"
                ),
                dict,
            )
            and record[
                "recovered_grid"
            ][
                "median_px"
            ]
            > max(
                150.0,
                2.0
                * record[
                    "baseline_grid"
                ][
                    "median_px"
                ],
            )
        )
    )

    semantic_errors = np.asarray(
        [
            diagnostic[
                "normal_error_px"
            ]
            for record in records
            for diagnostic in record.get(
                "semantic_diagnostics",
                [],
            )
        ],
        dtype=np.float64,
    )

    return {
        "records": records,
        "holdout_count": len(records),
        "recovered_count": len(
            recovered_records
        ),
        "improved_count": improved,
        "catastrophic_count": catastrophic,
        "baseline_median_px": (
            float(
                np.median(
                    baselines
                )
            )
            if len(
                baselines
            )
            else None
        ),
        "recovered_median_px": (
            float(
                np.median(
                    recovered
                )
            )
            if len(
                recovered
            )
            else None
        ),
        "recovered_p90_px": (
            float(
                np.percentile(
                    recovered,
                    90,
                )
            )
            if len(
                recovered
            )
            else None
        ),
        "recovered_max_px": (
            float(
                np.max(
                    recovered
                )
            )
            if len(
                recovered
            )
            else None
        ),
        "semantic_median_px": (
            float(
                np.median(
                    semantic_errors
                )
            )
            if len(
                semantic_errors
            )
            else None
        ),
        "total_auto_lines": sum(
            int(
                record.get(
                    "automatic_line_count",
                    0,
                )
            )
            for record in records
        ),
    }


def rank_key(configuration):
    summary = configuration[
        "summary"
    ]

    return (
        int(
            summary[
                "catastrophic_count"
            ]
        ),
        -int(
            summary[
                "improved_count"
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
                    "recovered_median_px"
                ]
            )
            if summary[
                "recovered_median_px"
            ] is not None
            else float(
                "inf"
            )
        ),
        (
            float(
                summary[
                    "recovered_max_px"
                ]
            )
            if summary[
                "recovered_max_px"
            ] is not None
            else float(
                "inf"
            )
        ),
    )


def fmt(
    value,
    digits=1,
):
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

    strong = sorted(
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

    interior = strong[1:-1]

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
        / review[
            "video"
        ][
            "filename"
        ]
    )

    capture = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not capture.isOpened():
        raise SystemExit(
            f"No se pudo abrir {video_path}"
        )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
        or review[
            "video"
        ].get(
            "width"
        )
        or 0
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
        or review[
            "video"
        ].get(
            "height"
        )
        or 0
    )

    images = read_frames(
        capture,
        interior + [115],
    )

    capture.release()

    flow_config = PlanarFlowConfig()

    print()
    print(
        "PLANAR FLOW → AUTO LINES → PROJECTIVE STATE REFINEMENT · LOO PROBE"
    )
    print(
        "Strong anchors:",
        strong,
    )
    print(
        "Interior holdouts:",
        interior,
    )
    print(
        f"Pitch: {pitch_length:g} × {pitch_width:g} m"
    )
    print(
        "Circles/arcs used: NO"
    )
    print()

    print(
        "Preparing selected planar-flow cameras once..."
    )

    holdouts = []

    for frame in interior:
        prepared = prepare_holdout(
            frame=frame,
            strong=strong,
            calibrations=calibrations,
            video_path=video_path,
            flow_config=flow_config,
            width=width,
            height=height,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )

        if prepared is not None:
            holdouts.append(
                prepared
            )

            print(
                f"f {frame:>4} · "
                f"side={prepared['selected'].get('selected_side')} · "
                f"direct={fmt(prepared['baseline_grid'].get('median_px') if prepared['baseline_grid'] else None)}px · "
                f"projected-to-5D={fmt(prepared['projected_grid'].get('median_px') if prepared['projected_grid'] else None)}px · "
                f"disagreement={fmt(prepared['selected'].get('disagreement_px'))}px"
            )

    print()
    print(
        "REFINEMENT SWEEP"
    )

    configurations = []

    for search_width in SEARCH_WIDTHS:
        for confidence in MIN_CONFIDENCES:
            for prior_mode in PRIOR_MODES:
                records = evaluate_configuration(
                    holdouts,
                    images,
                    pitch_model,
                    width,
                    height,
                    pitch_length,
                    pitch_width,
                    search_width=search_width,
                    minimum_confidence=confidence,
                    prior_mode=prior_mode,
                )

                summary = summarize(
                    records
                )

                configuration = {
                    "search_width": search_width,
                    "confidence": confidence,
                    "prior_mode": prior_mode,
                    "summary": summary,
                }

                configurations.append(
                    configuration
                )

                print(
                    f"width={search_width:>4.0f}px · "
                    f"conf={confidence:.2f} · "
                    f"prior={prior_mode:<12} · "
                    f"recovered={summary['recovered_count']}/{summary['holdout_count']} · "
                    f"improved={summary['improved_count']} · "
                    f"catastrophic={summary['catastrophic_count']} · "
                    f"base med={fmt(summary['baseline_median_px'])}px · "
                    f"ref med={fmt(summary['recovered_median_px'])}px · "
                    f"P90={fmt(summary['recovered_p90_px'])}px · "
                    f"max={fmt(summary['recovered_max_px'])}px · "
                    f"semantic={fmt(summary['semantic_median_px'])}px · "
                    f"lines={summary['total_auto_lines']}"
                )

    configurations.sort(
        key=rank_key
    )

    print()
    print(
        "TOP ROBUST REFINEMENT CONFIGS"
    )

    for configuration in configurations[:8]:
        summary = configuration[
            "summary"
        ]

        print(
            f"width={configuration['search_width']:.0f}px · "
            f"conf={configuration['confidence']:.2f} · "
            f"prior={configuration['prior_mode']} · "
            f"recovered={summary['recovered_count']}/{summary['holdout_count']} · "
            f"improved={summary['improved_count']} · "
            f"cat={summary['catastrophic_count']} · "
            f"median={fmt(summary['recovered_median_px'])}px · "
            f"P90={fmt(summary['recovered_p90_px'])}px · "
            f"max={fmt(summary['recovered_max_px'])}px"
        )

    best = configurations[0]
    summary = best[
        "summary"
    ]

    print()
    print(
        "BEST REFINEMENT CONFIG"
    )
    print(
        f"search_half_width_px={best['search_width']:.0f}"
    )
    print(
        f"minimum_confidence={best['confidence']:.2f}"
    )
    print(
        f"refinement_prior={best['prior_mode']}"
    )
    print(
        f"baseline median={fmt(summary['baseline_median_px'])}px"
    )
    print(
        f"recovered={summary['recovered_count']}/{summary['holdout_count']} · "
        f"improved={summary['improved_count']} · "
        f"catastrophic={summary['catastrophic_count']}"
    )
    print(
        f"refined median={fmt(summary['recovered_median_px'])}px · "
        f"P90={fmt(summary['recovered_p90_px'])}px · "
        f"max={fmt(summary['recovered_max_px'])}px · "
        f"semantic={fmt(summary['semantic_median_px'])}px"
    )
    print()

    for record in summary[
        "records"
    ]:
        baseline_error = (
            record[
                "baseline_grid"
            ][
                "median_px"
            ]
            if isinstance(
                record.get(
                    "baseline_grid"
                ),
                dict,
            )
            else None
        )

        projected_error = (
            record[
                "projected_grid"
            ][
                "median_px"
            ]
            if isinstance(
                record.get(
                    "projected_grid"
                ),
                dict,
            )
            else None
        )

        recovered_error = (
            record[
                "recovered_grid"
            ][
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

        semantics = ", ".join(
            f"{item.get('semantic')}:"
            f"{float(item.get('confidence') or 0.0):.2f}"
            for item in record.get(
                "automatic_semantics",
                [],
            )
        ) or "—"

        print(
            f"f {record['frame']:>4} · "
            f"status={record.get('status')} · "
            f"direct={fmt(baseline_error)}px · "
            f"5Dproj={fmt(projected_error)}px · "
            f"refined={fmt(recovered_error)}px · "
            f"lines={record.get('automatic_line_count',0)} · "
            f"families=x{family.get('x_constant',0)}/"
            f"y{family.get('y_constant',0)} · "
            f"shift={fmt(record.get('state_shift_z_rms'),2)} · "
            f"fitσ={fmt(record.get('fit_rms_sigma'),2)}"
        )

        print(
            "       lines:",
            semantics,
        )

    print()
    print(
        "FRAME 115 · BEST LOO CONFIG"
    )

    prepared_115 = prepare_holdout(
        frame=115,
        strong=strong,
        calibrations=calibrations,
        video_path=video_path,
        flow_config=flow_config,
        width=width,
        height=height,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )

    # Frame 115 is rejected, not a strong anchor, so the helper's LOO training
    # list is effectively the full strong-anchor model. prepare_holdout expects
    # a hidden calibration with H; frame 115 has no strong H. Build it manually.
    if prepared_115 is None:
        left_frame = max(
            anchor
            for anchor in strong
            if anchor < 115
        )
        right_frame = min(
            anchor
            for anchor in strong
            if anchor > 115
        )

        left_calibration = calibration_for(
            calibrations,
            left_frame,
            require_h=True,
        )
        right_calibration = calibration_for(
            calibrations,
            right_frame,
            require_h=True,
        )
        target_calibration = calibration_for(
            calibrations,
            115,
            require_h=False,
        )

        left_path = propagate_pitch_homography(
            video_path,
            left_frame,
            115,
            left_calibration["homography"],
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            config=flow_config,
        )
        right_path = propagate_pitch_homography(
            video_path,
            right_frame,
            115,
            right_calibration["homography"],
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            config=flow_config,
        )
        disagreement = (
            homography_disagreement_px(
                left_path["homography"],
                right_path["homography"],
                pitch_length_m=pitch_length,
                pitch_width_m=pitch_width,
                image_width=width,
                image_height=height,
            )
            if left_path.get("status") == "ok"
            and right_path.get("status") == "ok"
            else None
        )
        selected = select_planar_flow_camera(
            left_path,
            right_path,
            disagreement_px=(
                disagreement["median_px"]
                if isinstance(disagreement, dict)
                and disagreement.get("status") == "ok"
                else None
            ),
            config=SELECTION_CONFIG,
        )

        model = build_projective_camera_state_model(
            calibrations,
            strong,
            image_width=width,
            image_height=height,
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            config=ProjectiveCameraStateConfig(dimensions=5),
        )
        temporal = temporal_prior(
            model,
            115,
        )
        selected_h = np.asarray(
            selected["homography"],
            dtype=np.float64,
        )
        planar_state = state_from_homography(
            model,
            selected_h,
        )

        prepared_115 = {
            "frame": 115,
            "hidden": target_calibration,
            "selected": selected,
            "selected_h": selected_h,
            "model": model,
            "temporal": temporal,
            "planar_state": planar_state,
        }

    target_calibration = prepared_115[
        "hidden"
    ]
    image_115 = images.get(
        115
    )

    direct_eval = evaluate_homography_manual_points(
        prepared_115[
            "selected_h"
        ],
        target_calibration,
    )

    automatic = observe_automatic_pitch_lines(
        image_115,
        prepared_115[
            "selected_h"
        ],
        pitch_model,
        width,
        height,
        config=AutoLineEvidenceConfig(
            search_half_width_px=float(
                best[
                    "search_width"
                ]
            ),
            minimum_observer_confidence=float(
                best[
                    "confidence"
                ]
            ),
            observer_max_angle_delta_deg=12.0,
        ),
    )

    print(
        f"selected planar side="
        f"{prepared_115['selected'].get('selected_side')} · "
        f"confidence="
        f"{prepared_115['selected'].get('confidence')} · "
        f"disagreement="
        f"{fmt(prepared_115['selected'].get('disagreement_px'))}px"
    )

    if direct_eval.get(
        "status"
    ) == "ok":
        print(
            f"direct planar HUMAN median="
            f"{direct_eval['median_px']:.1f}px · "
            f"P90={direct_eval['p90_px']:.1f}px · "
            f"max={direct_eval['max_px']:.1f}px"
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
            for item in automatic.get(
                "observations",
                [],
            )
        ],
    )

    print(
        "families:",
        automatic.get(
            "family_counts",
            {},
        ),
    )

    if not automatic.get(
        "eligible"
    ):
        print(
            "No refinement: automatic geometry insufficient."
        )
        print(
            "Keep direct planar-flow H."
        )
        print(
            "No HUMAN fallback."
        )
        print(
            "No se ha modificado la review."
        )
        return

    if best[
        "prior_mode"
    ] == "temporal":
        refinement_prior = prepared_115[
            "temporal"
        ][
            "state"
        ]
    else:
        refinement_prior = prepared_115[
            "planar_state"
        ]

    recovered = recover_camera_from_projective_state(
        prepared_115[
            "model"
        ],
        refinement_prior,
        [],
        automatic[
            "constraints"
        ],
        [],
        prior_confidence="good",
    )

    print(
        f"refinement status="
        f"{recovered.get('status')} · "
        f"confidence="
        f"{recovered.get('confidence')}"
    )

    if (
        recovered.get(
            "status"
        ) != "ok"
        or not isinstance(
            recovered.get(
                "homography"
            ),
            list,
        )
    ):
        print(
            "Refinement rejected. Keep direct planar-flow H."
        )
        print(
            "No HUMAN fallback."
        )
        print(
            "No se ha modificado la review."
        )
        return

    refined_eval = evaluate_homography_manual_points(
        recovered[
            "homography"
        ],
        target_calibration,
    )

    if refined_eval.get(
        "status"
    ) == "ok":
        print(
            f"refined HUMAN median="
            f"{refined_eval['median_px']:.1f}px · "
            f"P90={refined_eval['p90_px']:.1f}px · "
            f"max={refined_eval['max_px']:.1f}px"
        )

    print(
        f"state shift z RMS="
        f"{float(recovered.get('state_shift_z_rms')):.3f} · "
        f"fit RMS sigma="
        f"{float(recovered.get('fit_rms_sigma')):.3f}"
    )

    print(
        "Refined H:"
    )
    print(
        json.dumps(
            recovered[
                "homography"
            ],
            indent=2,
        )
    )

    print()
    print(
        "115 HUMAN used ONLY for evaluation."
    )
    print(
        "No se ha modificado la review."
    )


if __name__ == "__main__":
    main()

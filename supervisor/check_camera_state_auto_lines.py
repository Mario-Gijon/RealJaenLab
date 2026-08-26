#!/usr/bin/env python3
"""LOO benchmark for raw-image automatic line evidence.

Question
--------
Can real image lines correct a temporal 5-D Projective Camera State prior
WITHOUT any HUMAN geometry in the target frame?

Strong HUMAN anchors are hidden one at a time. Their H is used only after
recovery for evaluation.

Frame 115 is evaluated last with the best LOO-selected configuration. Its saved
HUMAN points are never used for fitting.
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
from supervisor.field_camera_state_projective import (
    ProjectiveCameraStateConfig,
    build_projective_camera_state_model,
    evaluate_homography_manual_points,
    evaluate_homography_visible_grid,
    homography_from_state,
)
from supervisor.field_camera_state_recovery import (
    recover_camera_from_projective_state,
)
from supervisor.field_pitch_model import build_pitch_model


REVIEW_PATH = Path(
    "supervisor/data/reviews/benchmark_easy_2min.json"
)

SEARCH_WIDTHS = (30.0, 60.0, 90.0, 120.0)
MIN_CONFIDENCES = (0.55, 0.65, 0.75)


def _find_manual_match_calibration(
    calibrations,
    frame,
):
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
        ):
            return calibration
    return None


def _state_map(model):
    return {
        int(anchor["frame"]): np.asarray(
            anchor["state"],
            dtype=np.float64,
        )
        for anchor in model.get("anchors", [])
        if isinstance(anchor, dict)
        and anchor.get("state") is not None
    }


def _temporal_prior(
    model,
    query_frame,
):
    states = _state_map(model)
    frames = sorted(states)

    left = [
        frame
        for frame in frames
        if frame < query_frame
    ]
    right = [
        frame
        for frame in frames
        if frame > query_frame
    ]

    if not left or not right:
        return None

    left_frame = left[-1]
    right_frame = right[0]

    alpha = (
        query_frame - left_frame
    ) / float(
        right_frame - left_frame
    )

    state = (
        (1.0 - alpha) * states[left_frame]
        + alpha * states[right_frame]
    )

    return {
        "left_frame": left_frame,
        "right_frame": right_frame,
        "alpha": float(alpha),
        "state": state,
    }


def _decode_h(
    model,
    state,
):
    result = homography_from_state(
        model,
        state,
    )
    if result.get("status") != "ok":
        return None
    try:
        h = np.asarray(
            result["homography"],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None
    return h if h.shape == (3, 3) else None


def _read_frame(
    capture,
    frame_number,
):
    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_number),
    )
    ok, frame = capture.read()
    if not ok or frame is None:
        return None
    return frame


def _normalise_line(line):
    try:
        value = np.asarray(
            line,
            dtype=np.float64,
        ).reshape(3)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value).all():
        return None

    norm = math.hypot(
        float(value[0]),
        float(value[1]),
    )

    return (
        value / norm
        if norm > 1e-9
        else None
    )


def _project_world_point(
    point,
    h,
):
    q = np.asarray(
        [
            float(point["x"]),
            float(point["y"]),
            1.0,
        ],
        dtype=np.float64,
    ) @ np.asarray(
        h,
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


def _semantic_line_diagnostics(
    auto_result,
    hidden_calibration,
    pitch_model,
):
    true_h = np.asarray(
        hidden_calibration["homography"],
        dtype=np.float64,
    )

    definitions = {
        item["id"]: item
        for item in pitch_model.get(
            "calibration_lines",
            [],
        )
        if isinstance(item, dict)
        and item.get("id")
    }

    diagnostics = []

    for observation in auto_result.get(
        "observations",
        [],
    ):
        semantic = observation.get("semantic")
        definition = definitions.get(semantic)

        if not isinstance(definition, dict):
            continue

        world = definition["world"]
        true_start = _project_world_point(
            world["start"],
            true_h,
        )
        true_end = _project_world_point(
            world["end"],
            true_h,
        )

        if true_start is None or true_end is None:
            continue

        true_line = _normalise_line(
            np.cross(
                np.asarray(
                    [
                        true_start[0],
                        true_start[1],
                        1.0,
                    ]
                ),
                np.asarray(
                    [
                        true_end[0],
                        true_end[1],
                        1.0,
                    ]
                ),
            )
        )

        observed_line = _normalise_line(
            observation.get(
                "observed_line"
            )
        )

        if true_line is None or observed_line is None:
            continue

        midpoint = (
            0.5
            * (
                true_start
                + true_end
            )
        )

        normal_error = abs(
            float(
                observed_line[0] * midpoint[0]
                + observed_line[1] * midpoint[1]
                + observed_line[2]
            )
        )

        true_direction = np.asarray(
            [
                -true_line[1],
                true_line[0],
            ],
            dtype=np.float64,
        )
        observed_direction = np.asarray(
            [
                -observed_line[1],
                observed_line[0],
            ],
            dtype=np.float64,
        )

        cosine = float(
            np.clip(
                abs(
                    np.dot(
                        true_direction,
                        observed_direction,
                    )
                ),
                0.0,
                1.0,
            )
        )

        angle_error = float(
            math.degrees(
                math.acos(
                    cosine
                )
            )
        )

        diagnostics.append(
            {
                "semantic": semantic,
                "confidence": observation.get(
                    "confidence"
                ),
                "normal_error_px": normal_error,
                "angle_error_deg": angle_error,
                "orientation_family": observation.get(
                    "orientation_family"
                ),
            }
        )

    return diagnostics


def _visible_grid(
    predicted_h,
    true_h,
    *,
    width,
    height,
    pitch_length,
    pitch_width,
):
    if predicted_h is None:
        return None

    result = evaluate_homography_visible_grid(
        predicted_h,
        true_h,
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


def _evaluate_holdout(
    *,
    frame,
    strong_frames,
    calibrations,
    frame_bgr,
    pitch_model,
    width,
    height,
    pitch_length,
    pitch_width,
    search_width,
    minimum_confidence,
):
    hidden = _find_manual_match_calibration(
        calibrations,
        frame,
    )

    if hidden is None or not isinstance(
        hidden.get("homography"),
        list,
    ):
        return {
            "frame": frame,
            "status": "missing_hidden_human_h",
        }

    training_frames = [
        anchor
        for anchor in strong_frames
        if anchor != frame
    ]

    model = build_projective_camera_state_model(
        calibrations,
        training_frames,
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

    temporal = _temporal_prior(
        model,
        frame,
    )

    if temporal is None:
        return {
            "frame": frame,
            "status": "not_temporally_bracketed",
        }

    prior_h = _decode_h(
        model,
        temporal["state"],
    )

    if prior_h is None:
        return {
            "frame": frame,
            "status": "prior_h_failed",
        }

    true_h = np.asarray(
        hidden["homography"],
        dtype=np.float64,
    )

    prior_grid = _visible_grid(
        prior_h,
        true_h,
        width=width,
        height=height,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )

    auto = observe_automatic_pitch_lines(
        frame_bgr,
        prior_h,
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

    semantic = _semantic_line_diagnostics(
        auto,
        hidden,
        pitch_model,
    )

    result = {
        "frame": frame,
        "status": "automatic_evidence_observed",
        "left_frame": temporal["left_frame"],
        "right_frame": temporal["right_frame"],
        "alpha": temporal["alpha"],
        "prior_grid": prior_grid,
        "automatic_line_count": len(
            auto.get(
                "constraints",
                [],
            )
        ),
        "automatic_semantics": [
            {
                "semantic": item.get("semantic"),
                "confidence": item.get("confidence"),
                "orientation_family": item.get(
                    "orientation_family"
                ),
            }
            for item in auto.get(
                "observations",
                [],
            )
        ],
        "family_counts": auto.get(
            "family_counts",
            {},
        ),
        "eligible": bool(
            auto.get("eligible")
        ),
        "semantic_line_diagnostics": semantic,
        "recovery": None,
        "recovered_grid": None,
    }

    if not auto.get("eligible"):
        result["status"] = "insufficient_automatic_geometry"
        return result

    recovered = recover_camera_from_projective_state(
        model,
        temporal["state"],
        [],
        auto["constraints"],
        [],
        prior_confidence="good",
    )

    result["recovery"] = {
        "status": recovered.get(
            "status"
        ),
        "confidence": recovered.get(
            "confidence"
        ),
        "state_shift_z_rms": recovered.get(
            "state_shift_z_rms"
        ),
        "fit_rms_sigma": recovered.get(
            "fit_rms_sigma"
        ),
        "mode": recovered.get(
            "recovery",
            {},
        ).get(
            "mode"
        ),
    }

    if (
        recovered.get("status") != "ok"
        or not isinstance(
            recovered.get("homography"),
            list,
        )
    ):
        result["status"] = "recovery_rejected"
        return result

    recovered_grid = _visible_grid(
        recovered["homography"],
        true_h,
        width=width,
        height=height,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )

    if recovered_grid is None:
        result["status"] = "recovered_h_evaluation_failed"
        return result

    result["status"] = "recovered"
    result["recovered_grid"] = recovered_grid
    return result


def _summarize_configuration(
    records,
):
    total = len(records)

    prior_values = np.asarray(
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

    recovered = [
        record
        for record in records
        if record.get("status") == "recovered"
        and isinstance(
            record.get("recovered_grid"),
            dict,
        )
    ]

    recovered_values = np.asarray(
        [
            record["recovered_grid"]["median_px"]
            for record in recovered
        ],
        dtype=np.float64,
    )

    semantic_errors = np.asarray(
        [
            diagnostic["normal_error_px"]
            for record in records
            for diagnostic in record.get(
                "semantic_line_diagnostics",
                [],
            )
            if math.isfinite(
                float(
                    diagnostic.get(
                        "normal_error_px",
                        float("nan"),
                    )
                )
            )
        ],
        dtype=np.float64,
    )

    total_lines = sum(
        int(
            record.get(
                "automatic_line_count",
                0,
            )
        )
        for record in records
    )

    lacking_families = sum(
        1
        for record in records
        if not record.get("eligible")
    )

    return {
        "holdout_count": total,
        "recovered_count": len(recovered),
        "coverage": (
            len(recovered) / total
            if total
            else 0.0
        ),
        "prior_median_grid_px": (
            float(np.median(prior_values))
            if len(prior_values)
            else None
        ),
        "recovered_median_grid_px": (
            float(np.median(recovered_values))
            if len(recovered_values)
            else None
        ),
        "recovered_p90_of_medians_px": (
            float(
                np.percentile(
                    recovered_values,
                    90,
                )
            )
            if len(recovered_values)
            else None
        ),
        "recovered_max_median_px": (
            float(np.max(recovered_values))
            if len(recovered_values)
            else None
        ),
        "median_automatic_line_semantic_error_px": (
            float(np.median(semantic_errors))
            if len(semantic_errors)
            else None
        ),
        "automatic_lines_used": total_lines,
        "holdouts_lacking_two_families": lacking_families,
        "records": records,
    }


def _rank_key(
    summary,
):
    recovered = int(
        summary.get(
            "recovered_count",
            0,
        )
    )

    median = summary.get(
        "recovered_median_grid_px"
    )

    worst = summary.get(
        "recovered_max_median_px"
    )

    return (
        -recovered,
        (
            float(median)
            if median is not None
            else float("inf")
        ),
        (
            float(worst)
            if worst is not None
            else float("inf")
        ),
    )


def _fmt(
    value,
    digits=1,
):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def main():
    review = json.loads(
        REVIEW_PATH.read_text(
            encoding="utf-8"
        )
    )

    field = review["field"]
    pitch = field["pitch_model"]
    setup = field["match_setup"]
    calibrations = field.get(
        "calibrations",
        [],
    )

    strong_frames = sorted(
        {
            int(value)
            for value in setup.get(
                "completed_frames",
                [],
            )
        }
    )

    if len(strong_frames) < 7:
        raise SystemExit(
            "Se esperaban al menos 7 anchors HUMAN fuertes."
        )

    interior = strong_frames[1:-1]

    pitch_length = float(
        pitch["length_m"]
    )
    pitch_width = float(
        pitch["width_m"]
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

    cached_frames = {}

    for frame in sorted(
        set(interior + [115])
    ):
        image = _read_frame(
            capture,
            frame,
        )
        if image is not None:
            cached_frames[frame] = image

    capture.release()

    print()
    print(
        "AUTO-LINES · PROJECTIVE CAMERA STATE 5D · LOO"
    )
    print(
        "Strong anchors:",
        strong_frames,
    )
    print(
        f"Pitch: {pitch_length:g} × {pitch_width:g} m"
    )
    print(
        "Interior holdouts:",
        interior,
    )

    configurations = []

    for search_width in SEARCH_WIDTHS:
        for minimum_confidence in MIN_CONFIDENCES:
            records = []

            for frame in interior:
                image = cached_frames.get(
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
                    _evaluate_holdout(
                        frame=frame,
                        strong_frames=strong_frames,
                        calibrations=calibrations,
                        frame_bgr=image,
                        pitch_model=pitch_model,
                        width=width,
                        height=height,
                        pitch_length=pitch_length,
                        pitch_width=pitch_width,
                        search_width=search_width,
                        minimum_confidence=minimum_confidence,
                    )
                )

            summary = _summarize_configuration(
                records
            )

            configurations.append(
                {
                    "search_width": search_width,
                    "minimum_confidence": minimum_confidence,
                    "summary": summary,
                }
            )

            print(
                f"width={search_width:>5.0f}px · "
                f"conf={minimum_confidence:.2f} · "
                f"recovered={summary['recovered_count']}/{summary['holdout_count']} · "
                f"prior med={_fmt(summary['prior_median_grid_px'])}px · "
                f"recovered med={_fmt(summary['recovered_median_grid_px'])}px · "
                f"P90 med={_fmt(summary['recovered_p90_of_medians_px'])}px · "
                f"max med={_fmt(summary['recovered_max_median_px'])}px · "
                f"semantic={_fmt(summary['median_automatic_line_semantic_error_px'])}px · "
                f"lines={summary['automatic_lines_used']} · "
                f"no-2families={summary['holdouts_lacking_two_families']}"
            )

    best = min(
        configurations,
        key=lambda item: _rank_key(
            item["summary"]
        ),
    )

    best_summary = best["summary"]

    print()
    print(
        "BEST ROBUST CONFIG"
    )
    print(
        f"search_half_width_px={best['search_width']:.0f}"
    )
    print(
        f"minimum_observer_confidence={best['minimum_confidence']:.2f}"
    )
    print(
        f"coverage={best_summary['recovered_count']}/{best_summary['holdout_count']}"
    )
    print(
        f"recovered median={_fmt(best_summary['recovered_median_grid_px'])}px · "
        f"P90 medians={_fmt(best_summary['recovered_p90_of_medians_px'])}px · "
        f"max median={_fmt(best_summary['recovered_max_median_px'])}px"
    )
    print()

    for record in best_summary["records"]:
        semantics = ", ".join(
            f"{item['semantic']}:{float(item['confidence']):.2f}"
            for item in record.get(
                "automatic_semantics",
                [],
            )
        ) or "—"

        prior_error = (
            record.get(
                "prior_grid",
                {},
            ).get(
                "median_px"
            )
            if isinstance(
                record.get(
                    "prior_grid"
                ),
                dict,
            )
            else None
        )

        recovered_error = (
            record.get(
                "recovered_grid",
                {},
            ).get(
                "median_px"
            )
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

        recovery = record.get(
            "recovery"
        ) or {}

        semantic_error_values = [
            item["normal_error_px"]
            for item in record.get(
                "semantic_line_diagnostics",
                [],
            )
        ]

        semantic_median = (
            float(
                np.median(
                    semantic_error_values
                )
            )
            if semantic_error_values
            else None
        )

        print(
            f"f {record['frame']:>4} · "
            f"status={record.get('status')} · "
            f"prior={_fmt(prior_error)}px · "
            f"recovered={_fmt(recovered_error)}px · "
            f"families=x{family.get('x_constant', 0)}/y{family.get('y_constant', 0)} · "
            f"semantic={_fmt(semantic_median)}px · "
            f"shift-z={_fmt(recovery.get('state_shift_z_rms'), 2)} · "
            f"fitσ={_fmt(recovery.get('fit_rms_sigma'), 2)}"
        )
        print(
            "       lines:",
            semantics,
        )

    print()
    print(
        "FRAME 115 · AUTOMATIC IMAGE EVIDENCE"
    )

    full_model = build_projective_camera_state_model(
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

    temporal_115 = _temporal_prior(
        full_model,
        115,
    )

    calibration_115 = _find_manual_match_calibration(
        calibrations,
        115,
    )

    image_115 = cached_frames.get(115)

    if (
        full_model.get("status") != "ok"
        or temporal_115 is None
        or calibration_115 is None
        or image_115 is None
    ):
        print(
            "No se pudo preparar el frame 115."
        )
        print(
            "No se ha modificado la review."
        )
        return

    prior_h_115 = _decode_h(
        full_model,
        temporal_115["state"],
    )

    if prior_h_115 is None:
        print(
            "No se pudo decodificar el prior de 115."
        )
        print(
            "No se ha modificado la review."
        )
        return

    prior_human = evaluate_homography_manual_points(
        prior_h_115,
        calibration_115,
    )

    auto_115 = observe_automatic_pitch_lines(
        image_115,
        prior_h_115,
        pitch_model,
        width,
        height,
        config=AutoLineEvidenceConfig(
            search_half_width_px=float(
                best["search_width"]
            ),
            minimum_observer_confidence=float(
                best["minimum_confidence"]
            ),
            observer_max_angle_delta_deg=12.0,
        ),
    )

    print(
        f"bracket={temporal_115['left_frame']}→115→{temporal_115['right_frame']} · "
        f"alpha={temporal_115['alpha']:.3f}"
    )

    print(
        "automatic lines:",
        [
            (
                item.get("semantic"),
                round(
                    float(
                        item.get(
                            "confidence",
                            0.0,
                        )
                    ),
                    3,
                ),
                item.get(
                    "orientation_family"
                ),
            )
            for item in auto_115.get(
                "observations",
                [],
            )
        ],
    )

    print(
        "family counts:",
        auto_115.get(
            "family_counts",
            {},
        ),
    )

    if prior_human.get("status") == "ok":
        print(
            f"prior HUMAN points: "
            f"median={prior_human['median_px']:.1f}px · "
            f"P90={prior_human['p90_px']:.1f}px · "
            f"max={prior_human['max_px']:.1f}px"
        )

    if not auto_115.get("eligible"):
        print(
            "automatic recovery: INSUFFICIENT EVIDENCE"
        )
        print(
            "No se usa geometría HUMAN como fallback."
        )
        print(
            "No se ha modificado la review."
        )
        return

    recovered_115 = recover_camera_from_projective_state(
        full_model,
        temporal_115["state"],
        [],
        auto_115["constraints"],
        [],
        prior_confidence="good",
    )

    print(
        f"recovery status={recovered_115.get('status')} · "
        f"confidence={recovered_115.get('confidence')}"
    )

    if (
        recovered_115.get("status") != "ok"
        or not isinstance(
            recovered_115.get("homography"),
            list,
        )
    ):
        print(
            "automatic recovery rejected."
        )
        print(
            "No se usa geometría HUMAN como fallback."
        )
        print(
            "No se ha modificado la review."
        )
        return

    recovered_human = evaluate_homography_manual_points(
        recovered_115["homography"],
        calibration_115,
    )

    if recovered_human.get("status") == "ok":
        print(
            f"recovered HUMAN points: "
            f"median={recovered_human['median_px']:.1f}px · "
            f"P90={recovered_human['p90_px']:.1f}px · "
            f"max={recovered_human['max_px']:.1f}px"
        )

    print(
        f"state shift z RMS="
        f"{float(recovered_115.get('state_shift_z_rms')):.3f}"
    )
    print(
        f"fit RMS sigma="
        f"{float(recovered_115.get('fit_rms_sigma')):.3f}"
    )
    print(
        "H:"
    )
    print(
        json.dumps(
            recovered_115["homography"],
            indent=2,
        )
    )

    print()
    print(
        "Los puntos HUMAN del 115 se han usado SOLO para evaluación."
    )
    print(
        "No se ha modificado la review."
    )


if __name__ == "__main__":
    main()

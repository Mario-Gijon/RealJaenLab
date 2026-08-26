#!/usr/bin/env python3
"""LOO benchmark for endpoint-closed planar-flow bridge correction.

The target HUMAN H is hidden from every correction. Only the surrounding strong
HUMAN anchor H values are used to close accumulated planar-flow drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from supervisor.field_camera_state_planar_bridge import (
    PlanarBridgeConfig,
    bridge_correct_homography,
    corrected_path_result,
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
    evaluate_homography_manual_points,
    evaluate_homography_visible_grid,
)


REVIEW = Path("supervisor/data/reviews/benchmark_easy_2min.json")
EXPONENTS = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)

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


def grid(
    predicted,
    truth,
    width,
    height,
    pitch_length,
    pitch_width,
):
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


def fmt(value, digits=1):
    return (
        "—"
        if value is None
        else f"{float(value):.{digits}f}"
    )


def prepare_interval(
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

    left_cal = calibration_for(
        calibrations,
        left_frame,
        require_h=True,
    )
    right_cal = calibration_for(
        calibrations,
        right_frame,
        require_h=True,
    )

    left_target = propagate_pitch_homography(
        video_path,
        left_frame,
        frame,
        left_cal["homography"],
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=flow_config,
    )

    right_target = propagate_pitch_homography(
        video_path,
        right_frame,
        frame,
        right_cal["homography"],
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=flow_config,
    )

    left_closure = propagate_pitch_homography(
        video_path,
        left_frame,
        right_frame,
        left_cal["homography"],
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=flow_config,
    )

    right_closure = propagate_pitch_homography(
        video_path,
        right_frame,
        left_frame,
        right_cal["homography"],
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=flow_config,
    )

    direct_disagreement = (
        homography_disagreement_px(
            left_target["homography"],
            right_target["homography"],
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            image_width=width,
            image_height=height,
        )
        if left_target.get("status") == "ok"
        and right_target.get("status") == "ok"
        else None
    )

    direct_selected = select_planar_flow_camera(
        left_target,
        right_target,
        disagreement_px=(
            direct_disagreement["median_px"]
            if isinstance(
                direct_disagreement,
                dict,
            )
            and direct_disagreement.get(
                "status"
            ) == "ok"
            else None
        ),
        config=SELECTION_CONFIG,
    )

    direct_grid = (
        grid(
            direct_selected["homography"],
            hidden["homography"],
            width,
            height,
            pitch_length,
            pitch_width,
        )
        if direct_selected.get("status") == "ok"
        and isinstance(
            direct_selected.get(
                "homography"
            ),
            list,
        )
        else None
    )

    alpha = (
        (frame - left_frame)
        / float(
            right_frame
            - left_frame
        )
    )

    return {
        "frame": frame,
        "hidden": hidden,
        "left_frame": left_frame,
        "right_frame": right_frame,
        "alpha": float(alpha),
        "left_cal": left_cal,
        "right_cal": right_cal,
        "left_target": left_target,
        "right_target": right_target,
        "left_closure": left_closure,
        "right_closure": right_closure,
        "direct_disagreement": direct_disagreement,
        "direct_selected": direct_selected,
        "direct_grid": direct_grid,
    }


def evaluate_exponent(
    intervals,
    *,
    exponent,
    width,
    height,
    pitch_length,
    pitch_width,
):
    records = []

    for item in intervals:
        cfg = PlanarBridgeConfig(
            correction_exponent=float(
                exponent
            )
        )

        left_correction = bridge_correct_homography(
            item["left_target"],
            item["left_closure"],
            item["right_cal"]["homography"],
            progress_to_endpoint=item[
                "alpha"
            ],
            image_width=width,
            image_height=height,
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            config=cfg,
        )

        right_correction = bridge_correct_homography(
            item["right_target"],
            item["right_closure"],
            item["left_cal"]["homography"],
            progress_to_endpoint=(
                1.0
                - item["alpha"]
            ),
            image_width=width,
            image_height=height,
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            config=cfg,
        )

        left_corrected = (
            corrected_path_result(
                item["left_target"],
                left_correction,
            )
            if left_correction.get(
                "status"
            ) == "ok"
            else left_correction
        )

        right_corrected = (
            corrected_path_result(
                item["right_target"],
                right_correction,
            )
            if right_correction.get(
                "status"
            ) == "ok"
            else right_correction
        )

        corrected_disagreement = (
            homography_disagreement_px(
                left_corrected["homography"],
                right_corrected["homography"],
                pitch_length_m=pitch_length,
                pitch_width_m=pitch_width,
                image_width=width,
                image_height=height,
            )
            if left_corrected.get(
                "status"
            ) == "ok"
            and right_corrected.get(
                "status"
            ) == "ok"
            else None
        )

        selected = select_planar_flow_camera(
            left_corrected,
            right_corrected,
            disagreement_px=(
                corrected_disagreement[
                    "median_px"
                ]
                if isinstance(
                    corrected_disagreement,
                    dict,
                )
                and corrected_disagreement.get(
                    "status"
                ) == "ok"
                else None
            ),
            config=SELECTION_CONFIG,
        )

        selected_grid = (
            grid(
                selected["homography"],
                item["hidden"]["homography"],
                width,
                height,
                pitch_length,
                pitch_width,
            )
            if selected.get(
                "status"
            ) == "ok"
            and isinstance(
                selected.get(
                    "homography"
                ),
                list,
            )
            else None
        )

        left_grid = (
            grid(
                left_corrected["homography"],
                item["hidden"]["homography"],
                width,
                height,
                pitch_length,
                pitch_width,
            )
            if left_corrected.get(
                "status"
            ) == "ok"
            else None
        )

        right_grid = (
            grid(
                right_corrected["homography"],
                item["hidden"]["homography"],
                width,
                height,
                pitch_length,
                pitch_width,
            )
            if right_corrected.get(
                "status"
            ) == "ok"
            else None
        )

        records.append(
            {
                "frame": item["frame"],
                "direct_grid": item[
                    "direct_grid"
                ],
                "direct_side": item[
                    "direct_selected"
                ].get(
                    "selected_side"
                ),
                "selected": selected,
                "selected_grid": selected_grid,
                "left_grid": left_grid,
                "right_grid": right_grid,
                "direct_disagreement": item[
                    "direct_disagreement"
                ],
                "corrected_disagreement": corrected_disagreement,
                "left_correction": left_correction,
                "right_correction": right_correction,
            }
        )

    values = np.asarray(
        [
            record[
                "selected_grid"
            ][
                "median_px"
            ]
            for record in records
            if isinstance(
                record.get(
                    "selected_grid"
                ),
                dict,
            )
        ],
        dtype=np.float64,
    )

    direct_values = np.asarray(
        [
            record[
                "direct_grid"
            ][
                "median_px"
            ]
            for record in records
            if isinstance(
                record.get(
                    "direct_grid"
                ),
                dict,
            )
        ],
        dtype=np.float64,
    )

    improved = sum(
        1
        for record in records
        if isinstance(
            record.get(
                "direct_grid"
            ),
            dict,
        )
        and isinstance(
            record.get(
                "selected_grid"
            ),
            dict,
        )
        and record[
            "selected_grid"
        ][
            "median_px"
        ]
        < record[
            "direct_grid"
        ][
            "median_px"
        ]
    )

    disagreements = np.asarray(
        [
            record[
                "corrected_disagreement"
            ][
                "median_px"
            ]
            for record in records
            if isinstance(
                record.get(
                    "corrected_disagreement"
                ),
                dict,
            )
            and record[
                "corrected_disagreement"
            ].get(
                "status"
            ) == "ok"
        ],
        dtype=np.float64,
    )

    return {
        "records": records,
        "valid_count": int(
            len(
                values
            )
        ),
        "improved_count": int(
            improved
        ),
        "direct_median_px": (
            float(
                np.median(
                    direct_values
                )
            )
            if len(
                direct_values
            )
            else None
        ),
        "median_px": (
            float(
                np.median(
                    values
                )
            )
            if len(
                values
            )
            else None
        ),
        "p90_px": (
            float(
                np.percentile(
                    values,
                    90,
                )
            )
            if len(
                values
            )
            else None
        ),
        "max_px": (
            float(
                np.max(
                    values
                )
            )
            if len(
                values
            )
            else None
        ),
        "under_50": int(
            np.count_nonzero(
                values
                < 50.0
            )
        ),
        "under_100": int(
            np.count_nonzero(
                values
                < 100.0
            )
        ),
        "corrected_disagreement_median_px": (
            float(
                np.median(
                    disagreements
                )
            )
            if len(
                disagreements
            )
            else None
        ),
    }


def rank_key(configuration):
    summary = configuration[
        "summary"
    ]

    return (
        -int(
            summary[
                "valid_count"
            ]
        ),
        (
            float(
                summary[
                    "max_px"
                ]
            )
            if summary[
                "max_px"
            ] is not None
            else float(
                "inf"
            )
        ),
        (
            float(
                summary[
                    "p90_px"
                ]
            )
            if summary[
                "p90_px"
            ] is not None
            else float(
                "inf"
            )
        ),
        (
            float(
                summary[
                    "median_px"
                ]
            )
            if summary[
                "median_px"
            ] is not None
            else float(
                "inf"
            )
        ),
        -int(
            summary[
                "improved_count"
            ]
        ),
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

    capture.release()

    flow_config = PlanarFlowConfig()

    print()
    print(
        "GROUND-PLANE FLOW · ENDPOINT-CLOSED 8D PROJECTIVE BRIDGE · LOO"
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
        "Correction exponents:",
        EXPONENTS,
    )
    print(
        "5-D PCA used in bridge correction: NO"
    )
    print()

    print(
        "Preparing target and endpoint-closure flow paths..."
    )

    intervals = []

    for frame in interior:
        prepared = prepare_interval(
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
            intervals.append(
                prepared
            )

            print(
                f"f {frame:>4} · "
                f"bracket={prepared['left_frame']}→{frame}→{prepared['right_frame']} · "
                f"alpha={prepared['alpha']:.3f} · "
                f"direct side={prepared['direct_selected'].get('selected_side')} · "
                f"direct={fmt(prepared['direct_grid'].get('median_px') if prepared['direct_grid'] else None)}px"
            )

    configurations = []

    print()
    print(
        "BRIDGE SWEEP"
    )

    for exponent in EXPONENTS:
        summary = evaluate_exponent(
            intervals,
            exponent=exponent,
            width=width,
            height=height,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )

        configurations.append(
            {
                "exponent": exponent,
                "summary": summary,
            }
        )

        print(
            f"exponent={exponent:.2f} · "
            f"valid={summary['valid_count']}/{len(interior)} · "
            f"improved={summary['improved_count']} · "
            f"direct med={fmt(summary['direct_median_px'])}px · "
            f"bridge med={fmt(summary['median_px'])}px · "
            f"P90={fmt(summary['p90_px'])}px · "
            f"max={fmt(summary['max_px'])}px · "
            f"<50={summary['under_50']} · "
            f"<100={summary['under_100']} · "
            f"L↔R bridge disagreement med="
            f"{fmt(summary['corrected_disagreement_median_px'])}px"
        )

    configurations.sort(
        key=rank_key
    )

    best = configurations[0]
    summary = best[
        "summary"
    ]

    print()
    print(
        "BEST SAFE BRIDGE CONFIG"
    )
    print(
        f"correction_exponent="
        f"{best['exponent']:.2f}"
    )
    print(
        f"direct median="
        f"{fmt(summary['direct_median_px'])}px"
    )
    print(
        f"bridge median="
        f"{fmt(summary['median_px'])}px · "
        f"P90={fmt(summary['p90_px'])}px · "
        f"max={fmt(summary['max_px'])}px · "
        f"improved={summary['improved_count']}/{len(interior)} · "
        f"<50={summary['under_50']}/{len(interior)} · "
        f"<100={summary['under_100']}/{len(interior)}"
    )
    print()

    for record in summary[
        "records"
    ]:
        direct_error = (
            record[
                "direct_grid"
            ][
                "median_px"
            ]
            if isinstance(
                record.get(
                    "direct_grid"
                ),
                dict,
            )
            else None
        )

        selected_error = (
            record[
                "selected_grid"
            ][
                "median_px"
            ]
            if isinstance(
                record.get(
                    "selected_grid"
                ),
                dict,
            )
            else None
        )

        left_error = (
            record[
                "left_grid"
            ][
                "median_px"
            ]
            if isinstance(
                record.get(
                    "left_grid"
                ),
                dict,
            )
            else None
        )

        right_error = (
            record[
                "right_grid"
            ][
                "median_px"
            ]
            if isinstance(
                record.get(
                    "right_grid"
                ),
                dict,
            )
            else None
        )

        direct_disagreement = (
            record[
                "direct_disagreement"
            ][
                "median_px"
            ]
            if isinstance(
                record.get(
                    "direct_disagreement"
                ),
                dict,
            )
            and record[
                "direct_disagreement"
            ].get(
                "status"
            ) == "ok"
            else None
        )

        corrected_disagreement = (
            record[
                "corrected_disagreement"
            ][
                "median_px"
            ]
            if isinstance(
                record.get(
                    "corrected_disagreement"
                ),
                dict,
            )
            and record[
                "corrected_disagreement"
            ].get(
                "status"
            ) == "ok"
            else None
        )

        print(
            f"f {record['frame']:>4} · "
            f"direct={fmt(direct_error)}px/"
            f"{record.get('direct_side')} · "
            f"bridge={fmt(selected_error)}px/"
            f"{record['selected'].get('selected_side')} · "
            f"left={fmt(left_error)}px · "
            f"right={fmt(right_error)}px · "
            f"disagreement {fmt(direct_disagreement)}→"
            f"{fmt(corrected_disagreement)}px"
        )

    print()
    print(
        "FRAME 115 · BEST LOO BRIDGE CONFIG"
    )

    frame = 115
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

    left_cal = calibration_for(
        calibrations,
        left_frame,
        require_h=True,
    )
    right_cal = calibration_for(
        calibrations,
        right_frame,
        require_h=True,
    )
    target_cal = calibration_for(
        calibrations,
        frame,
        require_h=False,
    )

    left_target = propagate_pitch_homography(
        video_path,
        left_frame,
        frame,
        left_cal["homography"],
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=flow_config,
    )

    right_target = propagate_pitch_homography(
        video_path,
        right_frame,
        frame,
        right_cal["homography"],
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=flow_config,
    )

    left_closure = propagate_pitch_homography(
        video_path,
        left_frame,
        right_frame,
        left_cal["homography"],
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=flow_config,
    )

    right_closure = propagate_pitch_homography(
        video_path,
        right_frame,
        left_frame,
        right_cal["homography"],
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=flow_config,
    )

    alpha = (
        (frame - left_frame)
        / float(
            right_frame
            - left_frame
        )
    )

    bridge_config = PlanarBridgeConfig(
        correction_exponent=best[
            "exponent"
        ]
    )

    left_correction = bridge_correct_homography(
        left_target,
        left_closure,
        right_cal["homography"],
        progress_to_endpoint=alpha,
        image_width=width,
        image_height=height,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=bridge_config,
    )

    right_correction = bridge_correct_homography(
        right_target,
        right_closure,
        left_cal["homography"],
        progress_to_endpoint=(
            1.0
            - alpha
        ),
        image_width=width,
        image_height=height,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        config=bridge_config,
    )

    left_corrected = corrected_path_result(
        left_target,
        left_correction,
    )
    right_corrected = corrected_path_result(
        right_target,
        right_correction,
    )

    disagreement = homography_disagreement_px(
        left_corrected["homography"],
        right_corrected["homography"],
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        image_width=width,
        image_height=height,
    )

    selected = select_planar_flow_camera(
        left_corrected,
        right_corrected,
        disagreement_px=(
            disagreement["median_px"]
            if disagreement.get(
                "status"
            ) == "ok"
            else None
        ),
        config=SELECTION_CONFIG,
    )

    direct_disagreement = homography_disagreement_px(
        left_target["homography"],
        right_target["homography"],
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        image_width=width,
        image_height=height,
    )

    direct_selected = select_planar_flow_camera(
        left_target,
        right_target,
        disagreement_px=(
            direct_disagreement[
                "median_px"
            ]
            if direct_disagreement.get(
                "status"
            ) == "ok"
            else None
        ),
        config=SELECTION_CONFIG,
    )

    direct_eval = evaluate_homography_manual_points(
        direct_selected[
            "homography"
        ],
        target_cal,
    )

    bridge_eval = evaluate_homography_manual_points(
        selected[
            "homography"
        ],
        target_cal,
    )

    print(
        f"bracket={left_frame}→115→{right_frame} · "
        f"alpha={alpha:.3f}"
    )
    print(
        f"direct side="
        f"{direct_selected.get('selected_side')} · "
        f"disagreement="
        f"{fmt(direct_selected.get('disagreement_px'))}px"
    )

    if direct_eval.get(
        "status"
    ) == "ok":
        print(
            f"direct HUMAN median="
            f"{direct_eval['median_px']:.1f}px · "
            f"P90={direct_eval['p90_px']:.1f}px · "
            f"max={direct_eval['max_px']:.1f}px"
        )

    print(
        f"bridge side="
        f"{selected.get('selected_side')} · "
        f"disagreement="
        f"{fmt(selected.get('disagreement_px'))}px"
    )

    if bridge_eval.get(
        "status"
    ) == "ok":
        print(
            f"bridge HUMAN median="
            f"{bridge_eval['median_px']:.1f}px · "
            f"P90={bridge_eval['p90_px']:.1f}px · "
            f"max={bridge_eval['max_px']:.1f}px"
        )

    print(
        "Bridge H:"
    )
    print(
        json.dumps(
            selected[
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

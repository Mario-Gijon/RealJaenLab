#!/usr/bin/env python3
"""Real-data check for Camera State -> Homography Prior V1.

Tests two separate questions:

1. Decoder-only LOO:
   if the 3-D HUMAN state were known, can neighbouring HUMAN states reconstruct
   a useful H prior without using the held-out H?

2. Frame 115:
   use the current temporal Camera State prior between 87 and 175, decode an H,
   and compare that H against the HUMAN point constraints already saved on the
   rejected frame 115. Those constraints are evaluation only; they do NOT fit
   the H prior.

No review mutation and no solver refinement occurs here.

Run:
    .venv/bin/python -m supervisor.check_camera_state_h_prior
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from supervisor.field_camera_state import (
    build_camera_state_model,
)
from supervisor.field_camera_state_h_prior import (
    estimate_homography_prior_from_state,
    evaluate_prior_against_manual_points,
    homography_prior_leave_one_out,
    summarize_homography_prior_benchmark,
)
from supervisor.field_camera_state_temporal_v2 import (
    estimate_temporal_camera_state_v2,
)


REVIEW_PATH = Path(
    "supervisor/data/reviews/benchmark_easy_2min.json"
)


def _match_setup_calibration(
    calibrations,
    frame,
):
    for calibration in calibrations:
        if (
            isinstance(
                calibration,
                dict,
            )
            and int(
                calibration.get(
                    "frame",
                    -1,
                )
            )
            == int(
                frame
            )
            and calibration.get(
                "source"
            )
            == "manual"
            and calibration.get(
                "purpose"
            )
            == "match_setup"
        ):
            return calibration

    return None


def main() -> int:
    review = json.loads(
        REVIEW_PATH.read_text(
            encoding="utf-8"
        )
    )

    field = review[
        "field"
    ]
    pitch = field[
        "pitch_model"
    ]
    match_setup = field[
        "match_setup"
    ]
    calibrations = field.get(
        "calibrations",
        [],
    )

    video_path = (
        Path(
            "input"
        )
        / review[
            "video"
        ][
            "filename"
        ]
    )

    if not video_path.exists():
        raise SystemExit(
            f"No existe vídeo: {video_path}"
        )

    capture = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not capture.isOpened():
        raise SystemExit(
            f"No se pudo abrir vídeo: {video_path}"
        )

    try:
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
    finally:
        capture.release()

    completed = [
        int(
            value
        )
        for value in match_setup.get(
            "completed_frames",
            [],
        )
    ]

    model = (
        build_camera_state_model(
            calibrations,
            completed,
            image_width=width,
            image_height=height,
            pitch_length_m=float(
                pitch[
                    "length_m"
                ]
            ),
            pitch_width_m=float(
                pitch[
                    "width_m"
                ]
            ),
        )
    )

    if model.get(
        "status"
    ) != "ok":
        print(
            json.dumps(
                model,
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(
            "No se pudo construir Camera State."
        )

    print()
    print(
        "=" * 86
    )
    print(
        "CAMERA STATE → H PRIOR V1 · DECODER-ONLY LOO"
    )
    print(
        "=" * 86
    )

    benchmark = (
        homography_prior_leave_one_out(
            model,
            calibrations,
        )
    )

    print(
        json.dumps(
            summarize_homography_prior_benchmark(
                benchmark
            ),
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        "HOLDOUTS"
    )

    for item in benchmark.get(
        "per_anchor",
        [],
    ):
        if item.get(
            "status"
        ) != "ok":
            print(
                f"  f {item.get('frame')} · "
                f"ERROR={item.get('status')}"
            )
            continue

        grid = item.get(
            "visible_grid",
            {},
        )
        points = item.get(
            "manual_points",
            {},
        )

        neighbours = ", ".join(
            f"{neighbor['frame']}"
            f"(d={neighbor['state_distance']:.2f},"
            f"w={neighbor['weight']:.2f})"
            for neighbor in item.get(
                "neighbors",
                [],
            )
        )

        grid_text = (
            f"grid med={grid['median_px']:.1f}px "
            f"p90={grid['p90_px']:.1f}px"
            if grid.get(
                "status"
            )
            == "ok"
            else f"grid={grid.get('status')}"
        )

        point_text = (
            f"HUMAN med={points['median_px']:.1f}px "
            f"p90={points['p90_px']:.1f}px"
            if points.get(
                "status"
            )
            == "ok"
            else f"HUMAN={points.get('status')}"
        )

        print(
            f"  f {item['frame']:>5} · "
            f"{grid_text} · "
            f"{point_text} · "
            f"conf={item['confidence']} · "
            f"support={item['support_ratio']:.2f}"
        )
        print(
            f"           vecinos: {neighbours}"
        )

    print()
    print(
        "=" * 86
    )
    print(
        "FRAME 115 · TEMPORAL STATE → H PRIOR · SIN USAR SUS PUNTOS"
    )
    print(
        "=" * 86
    )

    temporal = (
        estimate_temporal_camera_state_v2(
            model,
            video_path,
            query_frame=115,
        )
    )

    if temporal.get(
        "status"
    ) != "ok":
        print(
            json.dumps(
                temporal,
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    # Results showed linear time was the most robust temporal state baseline.
    # Flow remains a confidence gate.
    query_state = temporal[
        "linear_time_state"
    ]

    prior = (
        estimate_homography_prior_from_state(
            model,
            calibrations,
            query_state,
        )
    )

    if prior.get(
        "status"
    ) != "ok":
        print(
            json.dumps(
                prior,
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    frame_115_calibration = (
        _match_setup_calibration(
            calibrations,
            115,
        )
    )

    print(
        "state usado:",
        [
            round(
                float(
                    value
                ),
                3,
            )
            for value in query_state
        ],
    )
    print(
        f"temporal confidence="
        f"{temporal['confidence']}"
    )
    print(
        f"H-prior confidence="
        f"{prior['confidence']}"
    )
    print(
        f"H-prior support_ratio="
        f"{prior['support_ratio']:.3f}"
    )
    print(
        f"probe-fit median="
        f"{prior['probe_fit_median_m']:.3f} m · "
        f"max={prior['probe_fit_max_m']:.3f} m"
    )
    print(
        "vecinos del decoder:"
    )
    for neighbor in prior[
        "neighbors"
    ]:
        print(
            f"  f {neighbor['frame']} · "
            f"d_state={neighbor['state_distance']:.3f} · "
            f"w={neighbor['weight']:.3f}"
        )

    if frame_115_calibration is None:
        print(
            "No existe calibración match_setup preservada para frame 115."
        )
    else:
        evaluation = (
            evaluate_prior_against_manual_points(
                prior[
                    "homography"
                ],
                frame_115_calibration,
            )
        )

        print()
        print(
            "EVALUACIÓN CONTRA PUNTOS HUMAN DE 115 "
            "(NO usados para construir el prior)"
        )

        if evaluation.get(
            "status"
        ) != "ok":
            print(
                json.dumps(
                    evaluation,
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            print(
                f"puntos={evaluation['point_count']} · "
                f"mediana={evaluation['median_px']:.1f}px · "
                f"P90={evaluation['p90_px']:.1f}px · "
                f"máximo={evaluation['max_px']:.1f}px"
            )

            for point in evaluation[
                "per_point"
            ]:
                print(
                    f"  {point['semantic']}: "
                    f"{point['error_px']:.1f}px"
                )

    print()
    print(
        "H PRIOR:"
    )
    print(
        json.dumps(
            prior[
                "homography"
            ],
            indent=2,
        )
    )

    print()
    print(
        "=" * 86
    )
    print(
        "No se ha refinado H ni modificado la review."
    )
    print(
        "=" * 86
    )
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
23
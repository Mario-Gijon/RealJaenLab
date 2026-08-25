#!/usr/bin/env python3
"""Real-data benchmark for Temporal Camera State Observer V1.

Uses the current saved PARTIDO review and the already validated HUMAN 3-D
Camera State model.

Reports:
- LOO raw-temporal-motion -> Camera State accuracy;
- time-only interpolation baseline;
- state estimate for rejected frame 115 using anchors 87 and 175.

Nothing is persisted and no H is reconstructed.

Run:
    .venv/bin/python -m supervisor.check_camera_state_temporal
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from supervisor.field_camera_state import (
    build_camera_state_model,
)
from supervisor.field_camera_state_temporal import (
    estimate_bracketed_camera_state,
    summarize_temporal_benchmark,
    temporal_camera_state_leave_one_out,
)


REVIEW_PATH = Path(
    "supervisor/data/reviews/benchmark_easy_2min.json"
)


def _state_map(
    model,
):
    return {
        int(
            anchor[
                "frame"
            ]
        ): np.asarray(
            anchor[
                "state"
            ],
            dtype=np.float64,
        )
        for anchor in model[
            "anchors"
        ]
    }


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

    video_name = review[
        "video"
    ][
        "filename"
    ]

    video_path = (
        Path(
            "input"
        )
        / video_name
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
            f"No se pudo abrir: {video_path}"
        )

    try:
        image_width = int(
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

        image_height = int(
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
            field.get(
                "calibrations",
                [],
            ),
            completed,
            image_width=image_width,
            image_height=image_height,
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
        "=" * 82
    )
    print(
        "TEMPORAL CAMERA STATE OBSERVER V1 · LOO"
    )
    print(
        "=" * 82
    )
    print(
        "Anchors:",
        model[
            "anchor_frames"
        ],
    )
    print(
        "State dimensions:",
        model[
            "dimensions"
        ],
    )

    benchmark = (
        temporal_camera_state_leave_one_out(
            model,
            video_path,
        )
    )

    print()
    print(
        json.dumps(
            summarize_temporal_benchmark(
                benchmark
            ),
            indent=2,
            ensure_ascii=False,
        )
    )

    if benchmark.get(
        "status"
    ) == "ok":
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
                    f"  f {item['frame']:>5} · "
                    f"{item.get('benchmark_group')} · "
                    f"ERROR: {item.get('status')}"
                )
                continue

            print(
                f"  f {item['frame']:>5} · "
                f"{item['benchmark_group']:<9} · "
                f"motion={item['motion_state_rmse']:.3f} · "
                f"time={item['time_only_state_rmse']:.3f} · "
                f"Δ={item['improvement_vs_time']:+.3f} · "
                f"conf={item['confidence']} · "
                f"valid={item['valid_step_fraction']:.2f} · "
                f"residual="
                f"{item['motion_feature_residual'] if item['motion_feature_residual'] is not None else '—'}"
            )

            if item.get(
                "alpha_used"
            ) is not None:
                print(
                    f"           alpha_motion="
                    f"{item['alpha_motion_raw']:.3f} · "
                    f"alpha_time={item['alpha_time']:.3f} · "
                    f"alpha_used={item['alpha_used']:.3f} · "
                    f"w_motion={item['motion_weight']:.2f}"
                )

    print()
    print(
        "=" * 82
    )
    print(
        "PRUEBA DEL FRAME 115 · SIN H HUMAN"
    )
    print(
        "=" * 82
    )

    states = _state_map(
        model
    )

    required = {
        87,
        175,
    }

    if not required.issubset(
        states
    ):
        print(
            "No están disponibles los anchors 87 y 175."
        )
        return 0

    frame_115 = (
        estimate_bracketed_camera_state(
            model,
            video_path,
            left_frame=87,
            query_frame=115,
            right_frame=175,
        )
    )

    if frame_115.get(
        "status"
    ) != "ok":
        print(
            json.dumps(
                frame_115,
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    print(
        "state:",
        [
            round(
                float(
                    value
                ),
                3,
            )
            for value in frame_115[
                "state"
            ]
        ],
    )
    print(
        "time-only state:",
        [
            round(
                float(
                    value
                ),
                3,
            )
            for value in frame_115[
                "time_only_state"
            ]
        ],
    )
    print(
        f"confidence="
        f"{frame_115['confidence']}"
    )
    print(
        f"alpha_motion_raw="
        f"{frame_115['alpha_motion_raw']:.3f}"
    )
    print(
        f"alpha_time="
        f"{frame_115['alpha_time']:.3f}"
    )
    print(
        f"alpha_used="
        f"{frame_115['alpha_used']:.3f}"
    )
    print(
        f"motion_weight="
        f"{frame_115['motion_weight']:.2f}"
    )
    print(
        f"motion residual="
        f"{frame_115['motion_feature_residual']:.3f}"
    )
    print(
        f"valid step fraction="
        f"{frame_115['valid_step_fraction']:.3f}"
    )
    print(
        "boundary states:"
    )
    print(
        "  f87 :",
        [
            round(
                float(
                    value
                ),
                3,
            )
            for value in states[
                87
            ]
        ],
    )
    print(
        "  f175:",
        [
            round(
                float(
                    value
                ),
                3,
            )
            for value in states[
                175
            ]
        ],
    )

    print()
    print(
        "=" * 82
    )
    print(
        "No se ha reconstruido ninguna H ni modificado la review."
    )
    print(
        "=" * 82
    )
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )

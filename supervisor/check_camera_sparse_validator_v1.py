"""Córdoba checker for Sparse CAMERA Validator V1.

This diagnostic:
1. reads the latest saved HUMAN anchors from the review;
2. reads the latest sparse HUMAN->trajectory relocalization produced by CAMERA;
3. measures multi-HUMAN agreement on the 421 trajectory keyframes;
4. maps uncertainty onto Planner V2 camera regions;
5. runs a tiny exact HUMAN leave-one-out relocalization (no Dense) so agreement
   can be compared against known HUMAN ground truth.

It does NOT edit review, CAMERA, HUMAN, Dense or workflow state.

Usage:
    python -m supervisor.check_camera_sparse_validator_v1
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import cv2

from supervisor.field_workflow import (
    extract_human_anchors,
)
from supervisor.field_camera_anchor_relocalization import (
    AnchorRelocalizationConfig,
    analyse_anchor_relocalization,
)
from supervisor.field_camera_sparse_validator_v1 import (
    SparseCameraValidatorConfig,
    analyse_sparse_camera_consensus,
    evaluate_human_leave_one_out,
)


ROOT = Path(__file__).resolve().parents[1]

ASSET = "asset_7fbc47b7b83647bc8fd52e6a676c9425"

VIDEO = (
    ROOT
    / "supervisor"
    / "data"
    / "library"
    / "videos"
    / "managed_upload_559cfa321d00455d9ef8b6465dc23594.mp4"
)

REVIEWS = (
    ROOT
    / "supervisor"
    / "data"
    / "reviews"
)

REVIEW = REVIEWS / f"{ASSET}.json"
RELOCALIZATION = (
    REVIEWS
    / f"{ASSET}_camera_anchor_relocalization_v1.json"
)
PLANNER_V2 = (
    REVIEWS
    / f"{ASSET}_camera_anchor_planner_v2.json"
)

OUTPUT = (
    REVIEWS
    / f"{ASSET}_camera_sparse_validator_v1.json"
)

LOO_OUTPUT = (
    REVIEWS
    / f"{ASSET}_camera_sparse_validator_v1_loo.json"
)


def _fmt(value, digits=1):
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _distribution_line(name, distribution):
    print(
        f"{name}: "
        f"median={_fmt(distribution.get('median'))} "
        f"p75={_fmt(distribution.get('p75'))} "
        f"p90={_fmt(distribution.get('p90'))} "
        f"p95={_fmt(distribution.get('p95'))} "
        f"max={_fmt(distribution.get('max'))} "
        f"(n={distribution.get('count')})"
    )


def _load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def main():
    print("=" * 96)
    print("SPARSE CAMERA VALIDATOR V1 · CÓRDOBA")
    print("=" * 96)
    print()

    required = [
        ("video", VIDEO),
        ("review", REVIEW),
        ("relocalization", RELOCALIZATION),
        ("planner V2", PLANNER_V2),
    ]

    for label, path in required:
        print(f"{label}: {path}")
        if not path.exists():
            print(f"ERROR: falta {label}.")
            return 2

    review = _load_json(REVIEW)
    relocalization = _load_json(RELOCALIZATION)
    planner_v2 = _load_json(PLANNER_V2)

    anchors = extract_human_anchors(review)

    if not anchors:
        print("ERROR: no hay HUMAN válidos.")
        return 2

    human_frames = [
        int(anchor["frame"])
        for anchor in anchors
    ]

    true_h_by_frame = {
        int(anchor["frame"]): anchor["homography"]
        for anchor in anchors
    }

    field = review.get("field", {})
    pitch = field.get("pitch_model", {})

    try:
        pitch_length = float(pitch["length_m"])
        pitch_width = float(pitch["width_m"])
    except (KeyError, TypeError, ValueError):
        print("ERROR: pitch_model no válido.")
        return 2

    capture = cv2.VideoCapture(str(VIDEO))

    if not capture.isOpened():
        print("ERROR: no se pudo abrir el vídeo.")
        return 2

    image_width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
    )
    image_height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
    )
    capture.release()

    if image_width <= 0 or image_height <= 0:
        print("ERROR: dimensiones de vídeo no válidas.")
        return 2

    print()
    print("HUMAN ACTUALES")
    print("--------------")
    print(human_frames)
    print(
        f"pitch: {pitch_length:g} × {pitch_width:g} m"
    )
    print(
        f"image: {image_width} × {image_height}"
    )

    # Check which HUMAN sources are actually present in the current
    # relocalization JSON.
    relocalization_anchor_frames = set()

    for result in relocalization.get("results", []):
        for candidate in result.get("candidates", []):
            try:
                relocalization_anchor_frames.add(
                    int(candidate["anchor_frame"])
                )
            except (KeyError, TypeError, ValueError):
                pass

    print(
        "anchors presentes en Relocalization JSON:",
        sorted(relocalization_anchor_frames),
    )

    missing_from_relocalization = sorted(
        set(human_frames)
        - relocalization_anchor_frames
    )

    if missing_from_relocalization:
        print(
            "WARNING: faltan HUMAN en el JSON:",
            missing_from_relocalization,
        )
        print(
            "El diagnóstico sparse seguirá, pero probablemente "
            "el CAMERA guardado no sea el último recalculado."
        )

    cfg = SparseCameraValidatorConfig(
        grid_x=21,
        grid_y=15,
        image_margin_px=8.0,
        minimum_common_grid_points=8,
    )

    started = time.perf_counter()

    sparse = analyse_sparse_camera_consensus(
        relocalization,
        image_width=image_width,
        image_height=image_height,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        human_frames=human_frames,
        planner_v2=planner_v2,
        config=cfg,
    )

    print()
    print("=" * 96)
    print("SPARSE CONSENSUS · 421 KEYFRAMES")
    print("=" * 96)
    print("status:", sparse.get("status"))

    if sparse.get("status") != "ok":
        print(json.dumps(sparse, indent=2))
        return 1

    print(
        "targets:",
        sparse["target_count"],
        "· automatic:",
        sparse["automatic_target_count"],
    )
    print(
        "support count distribution:",
        sparse["support_count_distribution"],
    )

    _distribution_line(
        "pair median disagreement px",
        sparse["pair_median_distribution_px"],
    )
    _distribution_line(
        "pair p90 disagreement px",
        sparse["pair_p90_distribution_px"],
    )
    _distribution_line(
        "current best vs consensus px",
        sparse[
            "current_vs_consensus_distribution_px"
        ],
    )

    print()
    print("WEAKEST CAMERA REGIONS")
    print("----------------------")
    print(
        f"{'REGION':>7} "
        f"{'N':>4} "
        f"{'REP':>6} "
        f"{'SUP MED':>8} "
        f"{'<2 SUP':>8} "
        f"{'DIS MED':>8} "
        f"{'DIS P90':>8} "
        f"{'CUR P90':>8}"
    )
    print("-" * 74)

    for region in sparse["weak_regions"]:
        disagreement = region[
            "pair_p90_distribution"
        ]
        current = region[
            "current_vs_consensus_distribution"
        ]

        print(
            f"{int(region['region']):>7} "
            f"{int(region['state_count']):>4} "
            f"{str(region.get('representative_frame')):>6} "
            f"{_fmt(region.get('support_median')):>8} "
            f"{100 * float(region['single_or_no_support_fraction']):>7.1f}% "
            f"{_fmt(disagreement.get('median')):>8} "
            f"{_fmt(disagreement.get('p90')):>8} "
            f"{_fmt(current.get('p90')):>8}"
        )

    print()
    print("TOP UNCERTAIN NON-HUMAN KEYFRAMES")
    print("---------------------------------")
    print(
        f"{'FRAME':>6} "
        f"{'REG':>4} "
        f"{'SUP':>4} "
        f"{'PAIR MED':>9} "
        f"{'PAIR P90':>9} "
        f"{'CURRENT→CONS':>12} "
        f"{'CONS ANCHOR':>11}"
    )
    print("-" * 82)

    for item in sparse["weak_targets"][:30]:
        print(
            f"{int(item['frame']):>6} "
            f"{str(item.get('planner_region') or '-'):>4} "
            f"{int(item.get('support_count') or 0):>4} "
            f"{_fmt(item.get('pair_median_px')):>9} "
            f"{_fmt(item.get('pair_p90_px')):>9} "
            f"{_fmt(item.get('current_to_consensus_median_px')):>12} "
            f"{str(item.get('consensus_anchor_frame') or '-'):>11}"
        )

    OUTPUT.write_text(
        json.dumps(
            sparse,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    # Exact HUMAN leave-one-out. This is intentionally small: N HUMAN targets,
    # not the full video and not Dense.
    print()
    print("=" * 96)
    print("EXACT HUMAN LEAVE-ONE-OUT")
    print("=" * 96)
    print(
        "Ocultamos cada HUMAN en su propio frame y "
        "lo predecimos desde los demás."
    )

    loo_targets = [
        {"frame": int(frame)}
        for frame in human_frames
    ]

    loo_relocalization = analyse_anchor_relocalization(
        VIDEO,
        anchors=anchors,
        targets=loo_targets,
        config=AnchorRelocalizationConfig(
            analysis_width=960
        ),
        exclude_same_frame=True,
    )

    if loo_relocalization.get("status") != "ok":
        print(
            "LOO relocalization status:",
            loo_relocalization.get("status"),
        )
        loo = {
            "status": "relocalization_failed",
            "relocalization": loo_relocalization,
        }
    else:
        loo = evaluate_human_leave_one_out(
            loo_relocalization,
            true_h_by_frame,
            image_width=image_width,
            image_height=image_height,
            pitch_length_m=pitch_length,
            pitch_width_m=pitch_width,
            config=cfg,
        )

        print(
            "holdouts:",
            loo.get("holdout_count"),
            "· consensus valid:",
            loo.get("consensus_valid_count"),
            "· score-best valid:",
            loo.get("score_best_valid_count"),
        )

        _distribution_line(
            "CONSENSUS true median error px",
            loo[
                "consensus_median_error_distribution_px"
            ],
        )
        _distribution_line(
            "SCORE-BEST true median error px",
            loo[
                "score_best_median_error_distribution_px"
            ],
        )

        print()
        print(
            f"{'HUMAN':>6} "
            f"{'SUP':>4} "
            f"{'PAIR P90':>9} "
            f"{'CONS A':>7} "
            f"{'CONS ERR':>9} "
            f"{'BEST A':>7} "
            f"{'BEST ERR':>9}"
        )
        print("-" * 66)

        for row in loo.get("per_human", []):
            consensus_error = row[
                "consensus_visible_grid_error"
            ]
            best_error = row[
                "score_best_visible_grid_error"
            ]

            print(
                f"{int(row['frame']):>6} "
                f"{int(row.get('support_count') or 0):>4} "
                f"{_fmt(row.get('pair_p90_px')):>9} "
                f"{str(row.get('consensus_anchor_frame') or '-'):>7} "
                f"{_fmt(consensus_error.get('median_px')):>9} "
                f"{str(row.get('score_best_anchor_frame') or '-'):>7} "
                f"{_fmt(best_error.get('median_px')):>9}"
            )

    LOO_OUTPUT.write_text(
        json.dumps(
            loo,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - started

    print()
    print("Tiempo total:", f"{elapsed:.1f}s")
    print()
    print("Guardado:")
    print(OUTPUT)
    print(LOO_OUTPUT)
    print()
    print(
        "DIAGNÓSTICO READ-ONLY: no modifica review, HUMAN, "
        "CAMERA, Dense ni workflow."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

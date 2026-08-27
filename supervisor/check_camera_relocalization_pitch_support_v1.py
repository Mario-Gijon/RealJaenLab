#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from supervisor.field_camera_loop_closure import (
    LoopClosureConfig,
    extract_keyframe_features,
    _symmetric_ratio_matches,
)
from supervisor.field_camera_anchor_relocalization import (
    derive_target_homography,
)
from supervisor.field_camera_state_projective import (
    evaluate_homography_visible_grid,
)
from supervisor.field_workflow import extract_human_anchors
from supervisor.media_library import MediaLibrary
from supervisor.video_sources import resolve_video_source


ROOT = Path(__file__).resolve().parents[1]
REVIEWS = ROOT / "supervisor" / "data" / "reviews"
INPUT_DIR = ROOT / "input"
CATALOG = ROOT / "supervisor" / "data" / "library" / "catalog_v1.json"

ASSET = "asset_7fbc47b7b83647bc8fd52e6a676c9425"

KNOWN_BAD = [0, 312, 320, 336, 344, 4134, 4142]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise RuntimeError(path)
    return value


def normalise_h(value: Any) -> np.ndarray | None:
    try:
        h = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None

    if (
        h.shape != (3, 3)
        or not np.isfinite(h).all()
        or abs(float(h[2, 2])) < 1e-9
    ):
        return None

    return h / float(h[2, 2])


def project(points: np.ndarray, h: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    q = np.column_stack(
        (points, np.ones(len(points), dtype=np.float64))
    ) @ h.T

    result = q[:, :2] / q[:, 2, None]
    return result


def pitch_polygon_native(
    h: Any,
    length_m: float,
    width_m: float,
) -> np.ndarray:
    h = normalise_h(h)
    if h is None:
        raise RuntimeError("H inválida")

    world = np.asarray(
        [
            [-length_m / 2.0, -width_m / 2.0],
            [ length_m / 2.0, -width_m / 2.0],
            [ length_m / 2.0,  width_m / 2.0],
            [-length_m / 2.0,  width_m / 2.0],
        ],
        dtype=np.float64,
    )

    return project(world, h).astype(np.float32)


def inside_polygon(
    points: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    contour = polygon.reshape(-1, 1, 2).astype(np.float32)

    return np.asarray(
        [
            cv2.pointPolygonTest(
                contour,
                (float(point[0]), float(point[1])),
                False,
            ) >= 0
            for point in points
        ],
        dtype=bool,
    )


def native_h_from_work(
    h_work: Any,
    scale: float,
) -> np.ndarray | None:
    h = normalise_h(h_work)
    if h is None or scale <= 0:
        return None

    s = np.asarray(
        [
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    s_inv = np.asarray(
        [
            [1.0 / scale, 0.0, 0.0],
            [0.0, 1.0 / scale, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return normalise_h(s_inv @ h @ s)


def fit_homography(
    source: np.ndarray,
    target: np.ndarray,
    cfg: LoopClosureConfig,
) -> dict[str, Any]:
    if len(source) < 4:
        return {
            "status": "not_enough_matches",
            "match_count": len(source),
        }

    method = getattr(
        cv2,
        "USAC_MAGSAC",
        cv2.RANSAC,
    )

    h, mask = cv2.findHomography(
        source.astype(np.float32),
        target.astype(np.float32),
        method=method,
        ransacReprojThreshold=cfg.ransac_reprojection_threshold_px,
        maxIters=cfg.ransac_max_iters,
        confidence=cfg.ransac_confidence,
    )

    if h is None or mask is None:
        return {
            "status": "homography_failed",
            "match_count": len(source),
        }

    h = normalise_h(h)
    if h is None:
        return {
            "status": "invalid_h",
            "match_count": len(source),
        }

    inliers = mask.ravel() > 0
    count = int(np.count_nonzero(inliers))

    if count:
        q = project(source[inliers], h)
        errors = np.linalg.norm(
            q - target[inliers],
            axis=1,
        )
        reproj = float(np.median(errors))
    else:
        reproj = None

    return {
        "status": "ok",
        "homography": h,
        "mask": inliers,
        "match_count": int(len(source)),
        "inlier_count": count,
        "inlier_ratio": count / max(float(len(source)), 1.0),
        "median_reprojection_px": reproj,
    }


def fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def main() -> None:
    review = load_json(
        REVIEWS / f"{ASSET}.json"
    )

    existing = load_json(
        REVIEWS
        / f"{ASSET}_camera_anchor_relocalization_v1.json"
    )

    anchors = extract_human_anchors(review)
    truth = {
        int(anchor["frame"]): anchor["homography"]
        for anchor in anchors
    }

    human_frames = sorted(truth)

    pitch = review["field"]["pitch_model"]
    length_m = float(pitch["length_m"])
    width_m = float(pitch["width_m"])

    library = MediaLibrary(
        CATALOG,
        ROOT,
        INPUT_DIR,
    )

    source = resolve_video_source(
        ASSET,
        library=library,
        input_dir=INPUT_DIR,
    )

    cap = cv2.VideoCapture(str(source.path))
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir el vídeo")

    native_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
    )
    native_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
    )

    cfg = LoopClosureConfig(
        analysis_width=960
    )

    needed = sorted(
        set(human_frames)
        | set(KNOWN_BAD)
    )

    features = {}

    for frame in needed:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()

        if not ok or image is None:
            features[frame] = {
                "status": "frame_read_failed"
            }
            continue

        features[frame] = extract_keyframe_features(
            image,
            config=cfg,
        )

    cap.release()

    selected_by_target = {
        int(result["frame"]): int(result["anchor_frame"])
        for result in existing.get("results", [])
        if (
            isinstance(result, dict)
            and result.get("status") == "ok"
            and result.get("anchor_frame") is not None
        )
    }

    print("VIDEO:", source.path)
    print("HUMAN:", human_frames)
    print("BAD TARGETS:", KNOWN_BAD)
    print()

    rows = []

    targets = human_frames + KNOWN_BAD

    for target_frame in targets:
        target_features = features.get(target_frame, {})

        if target_features.get("status") != "ok":
            continue

        for anchor_frame in human_frames:
            if anchor_frame == target_frame:
                continue

            anchor_features = features.get(anchor_frame, {})

            if anchor_features.get("status") != "ok":
                continue

            if abs(
                float(anchor_features["scale"])
                - float(target_features["scale"])
            ) > 1e-9:
                continue

            scale = float(anchor_features["scale"])

            matches = _symmetric_ratio_matches(
                anchor_features["descriptors"],
                target_features["descriptors"],
                cfg.ratio_test,
            )

            if len(matches) < 4:
                continue

            source_points = np.asarray(
                [
                    anchor_features["points"][m.queryIdx]
                    for m in matches
                ],
                dtype=np.float32,
            )

            target_points = np.asarray(
                [
                    target_features["points"][m.trainIdx]
                    for m in matches
                ],
                dtype=np.float32,
            )

            polygon_native = pitch_polygon_native(
                truth[anchor_frame],
                length_m,
                width_m,
            )

            polygon_work = polygon_native * scale

            source_on_pitch = inside_polygon(
                source_points,
                polygon_work,
            )

            pitch_match_count = int(
                np.count_nonzero(source_on_pitch)
            )

            global_fit = fit_homography(
                source_points,
                target_points,
                cfg,
            )

            if global_fit.get("status") != "ok":
                continue

            global_mask = global_fit["mask"]

            global_pitch_inliers = int(
                np.count_nonzero(
                    global_mask & source_on_pitch
                )
            )

            global_bg_inliers = (
                int(global_fit["inlier_count"])
                - global_pitch_inliers
            )

            global_pitch_fraction = (
                global_pitch_inliers
                / max(
                    float(global_fit["inlier_count"]),
                    1.0,
                )
            )

            field_fit = fit_homography(
                source_points[source_on_pitch],
                target_points[source_on_pitch],
                cfg,
            )

            global_native = native_h_from_work(
                global_fit["homography"],
                scale,
            )

            field_native = (
                native_h_from_work(
                    field_fit["homography"],
                    scale,
                )
                if field_fit.get("status") == "ok"
                else None
            )

            global_target_h = (
                derive_target_homography(
                    truth[anchor_frame],
                    global_native,
                )
                if global_native is not None
                else None
            )

            field_target_h = (
                derive_target_homography(
                    truth[anchor_frame],
                    field_native,
                )
                if field_native is not None
                else None
            )

            global_truth_error = None
            field_truth_error = None

            if target_frame in truth:
                if global_target_h is not None:
                    result = evaluate_homography_visible_grid(
                        global_target_h,
                        truth[target_frame],
                        image_width=native_width,
                        image_height=native_height,
                        pitch_length_m=length_m,
                        pitch_width_m=width_m,
                    )
                    if result.get("status") == "ok":
                        global_truth_error = result

                if field_target_h is not None:
                    result = evaluate_homography_visible_grid(
                        field_target_h,
                        truth[target_frame],
                        image_width=native_width,
                        image_height=native_height,
                        pitch_length_m=length_m,
                        pitch_width_m=width_m,
                    )
                    if result.get("status") == "ok":
                        field_truth_error = result

            disagreement = None

            if (
                global_target_h is not None
                and field_target_h is not None
            ):
                result = evaluate_homography_visible_grid(
                    field_target_h,
                    global_target_h,
                    image_width=native_width,
                    image_height=native_height,
                    pitch_length_m=length_m,
                    pitch_width_m=width_m,
                )
                if result.get("status") == "ok":
                    disagreement = result

            rows.append({
                "target": target_frame,
                "anchor": anchor_frame,
                "selected": (
                    selected_by_target.get(target_frame)
                    == anchor_frame
                ),
                "matches": len(matches),
                "pitch_matches": pitch_match_count,
                "global_inliers": int(
                    global_fit["inlier_count"]
                ),
                "global_pitch_inliers": global_pitch_inliers,
                "global_bg_inliers": global_bg_inliers,
                "global_pitch_fraction": global_pitch_fraction,
                "global_reproj": global_fit[
                    "median_reprojection_px"
                ],
                "field_matches": (
                    field_fit.get("match_count")
                ),
                "field_inliers": (
                    field_fit.get("inlier_count")
                ),
                "field_ratio": (
                    field_fit.get("inlier_ratio")
                ),
                "field_reproj": (
                    field_fit.get(
                        "median_reprojection_px"
                    )
                ),
                "global_true": (
                    global_truth_error.get("median_px")
                    if global_truth_error else None
                ),
                "field_true": (
                    field_truth_error.get("median_px")
                    if field_truth_error else None
                ),
                "global_field_disagreement": (
                    disagreement.get("median_px")
                    if disagreement else None
                ),
            })

    print(
        "TGT".rjust(5),
        "ANC".rjust(5),
        "SEL".rjust(4),
        "MATCH".rjust(6),
        "PITCHM".rjust(7),
        "GINL".rjust(6),
        "PINL".rjust(6),
        "BGINL".rjust(6),
        "PFRAC".rjust(7),
        "GREPR".rjust(7),
        "FINL".rjust(6),
        "FREPR".rjust(7),
        "GTRUE".rjust(7),
        "FTRUE".rjust(7),
        "GvsF".rjust(7),
    )
    print("-" * 110)

    for row in rows:
        print(
            str(row["target"]).rjust(5),
            str(row["anchor"]).rjust(5),
            ("*" if row["selected"] else "").rjust(4),
            str(row["matches"]).rjust(6),
            str(row["pitch_matches"]).rjust(7),
            str(row["global_inliers"]).rjust(6),
            str(row["global_pitch_inliers"]).rjust(6),
            str(row["global_bg_inliers"]).rjust(6),
            fmt(
                row["global_pitch_fraction"],
                3,
            ).rjust(7),
            fmt(
                row["global_reproj"],
                3,
            ).rjust(7),
            str(
                row["field_inliers"]
                if row["field_inliers"] is not None
                else "—"
            ).rjust(6),
            fmt(
                row["field_reproj"],
                3,
            ).rjust(7),
            fmt(
                row["global_true"],
                1,
            ).rjust(7),
            fmt(
                row["field_true"],
                1,
            ).rjust(7),
            fmt(
                row["global_field_disagreement"],
                1,
            ).rjust(7),
        )

    print()
    print("=" * 90)
    print("HUMAN LOO · GLOBAL vs FIELD-SUPPORTED")
    print("=" * 90)

    for target in human_frames:
        group = [
            row
            for row in rows
            if (
                row["target"] == target
                and row["global_true"] is not None
            )
        ]

        if not group:
            print(target, "sin candidatos")
            continue

        best_global = min(
            group,
            key=lambda row: row["global_true"],
        )

        field_group = [
            row
            for row in group
            if row["field_true"] is not None
        ]

        best_field = (
            min(
                field_group,
                key=lambda row: row["field_true"],
            )
            if field_group else None
        )

        print(
            f"target {target}:",
            f"best global {best_global['anchor']} "
            f"→ {fmt(best_global['global_true'])} px",
            "|",
            (
                f"best field {best_field['anchor']} "
                f"→ {fmt(best_field['field_true'])} px"
                if best_field
                else "sin field-H"
            ),
        )

    print()
    print("=" * 90)
    print("CANDIDATOS ELEGIDOS EN FRAMES MALOS")
    print("=" * 90)

    for target in KNOWN_BAD:
        selected = [
            row
            for row in rows
            if (
                row["target"] == target
                and row["selected"]
            )
        ]

        if not selected:
            print(target, "sin candidato seleccionado medido")
            continue

        row = selected[0]

        print(
            f"frame {target} ← HUMAN {row['anchor']}:",
            f"global inliers={row['global_inliers']}",
            f"pitch inliers={row['global_pitch_inliers']}",
            f"background={row['global_bg_inliers']}",
            f"pitch_fraction={fmt(row['global_pitch_fraction'],3)}",
            f"field-H inliers={row['field_inliers']}",
            f"global↔field={fmt(row['global_field_disagreement'])} px",
        )

    print()
    print(
        "DIAGNÓSTICO PURO: no modifica Relocalization,"
        " Dense, HUMAN ni artifacts."
    )


if __name__ == "__main__":
    main()

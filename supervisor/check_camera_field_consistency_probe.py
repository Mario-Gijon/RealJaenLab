#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from supervisor.field_camera_state_auto_lines import (
    AutoLineEvidenceConfig,
    observe_automatic_pitch_lines,
)
from supervisor.field_pitch_model import build_pitch_model
from supervisor.field_workflow import extract_human_anchors
from supervisor.media_library import MediaLibrary
from supervisor.video_sources import resolve_video_source


ROOT = Path(__file__).resolve().parents[1]
REVIEWS = ROOT / "supervisor" / "data" / "reviews"
INPUT_DIR = ROOT / "input"
CATALOG = ROOT / "supervisor" / "data" / "library" / "catalog_v1.json"

ASSET = "asset_7fbc47b7b83647bc8fd52e6a676c9425"

# HUMAN comprobados visualmente como correctos.
HUMAN_FRAMES = [580, 3450, 3805, 5101, 7200]

# H que hemos comprobado visualmente como incorrectas.
KNOWN_BAD_DENSE = [319, 339, 4139]

# Frame 0 estaba etiquetado como relocalización directa y era incorrecto.
KNOWN_BAD_RELOCALIZED = [0]

# Endpoints que alimentaron los tres Dense incorrectos.
# Queremos saber si Dense recibió ya H incorrectas.
RELOCALIZED_ENDPOINTS = [
    312, 320,
    336, 344,
    4134, 4142,
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} no contiene un objeto JSON")
    return value


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def fmt(value: Any, decimals: int = 2) -> str:
    number = finite_number(value)
    if number is None:
        return "—"
    return f"{number:.{decimals}f}"


def read_frame(
    capture: cv2.VideoCapture,
    frame_number: int,
):
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    ok, frame = capture.read()
    if not ok or frame is None:
        return None
    return frame


def evaluate_candidate(
    *,
    frame_number: int,
    homography: Any,
    frame_bgr,
    pitch_model: dict[str, Any],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    result = observe_automatic_pitch_lines(
        frame_bgr,
        homography,
        pitch_model,
        image_width,
        image_height,
        config=AutoLineEvidenceConfig(),
    )

    observations = [
        item
        for item in result.get("observations", [])
        if isinstance(item, dict)
    ]

    offsets = [
        abs(float(item["normal_offset_px"]))
        for item in observations
        if finite_number(item.get("normal_offset_px")) is not None
    ]

    angles = [
        abs(float(item["angle_offset_deg"]))
        for item in observations
        if finite_number(item.get("angle_offset_deg")) is not None
    ]

    confidences = [
        float(item["confidence"])
        for item in observations
        if finite_number(item.get("confidence")) is not None
    ]

    supports = [
        float(item["support_length_px"])
        for item in observations
        if finite_number(item.get("support_length_px")) is not None
    ]

    paints = [
        float(item["paint_score"])
        for item in observations
        if finite_number(item.get("paint_score")) is not None
    ]

    observer_summary = (
        result.get("observer", {}).get("summary", {})
        if isinstance(result.get("observer"), dict)
        else {}
    )

    families = (
        result.get("family_counts", {})
        if isinstance(result.get("family_counts"), dict)
        else {}
    )

    family_count = sum(
        1
        for family in ("x_constant", "y_constant")
        if int(families.get(family, 0) or 0) > 0
    )

    return {
        "frame": int(frame_number),
        "status": result.get("status"),
        "eligible": bool(result.get("eligible")),
        "searched_lines": int(observer_summary.get("searched_lines", 0) or 0),
        "found_lines": int(observer_summary.get("found_lines", 0) or 0),
        "constraint_count": len(result.get("constraints", [])),
        "family_count": family_count,
        "family_x": int(families.get("x_constant", 0) or 0),
        "family_y": int(families.get("y_constant", 0) or 0),
        "median_abs_normal_offset_px": median(offsets),
        "p90_abs_normal_offset_px": percentile(offsets, 90),
        "median_abs_angle_deg": median(angles),
        "p90_abs_angle_deg": percentile(angles, 90),
        "median_confidence": median(confidences),
        "median_support_px": median(supports),
        "median_paint_score": median(paints),
        "observations": observations,
        "reason": result.get("reason"),
    }


def summarize_group(
    name: str,
    records: list[dict[str, Any]],
) -> None:
    print()
    print(f"=== {name} ===")
    print(f"frames: {len(records)}")

    if not records:
        return

    eligible = sum(bool(item["eligible"]) for item in records)

    med_offsets = [
        float(item["median_abs_normal_offset_px"])
        for item in records
        if item["median_abs_normal_offset_px"] is not None
    ]

    p90_offsets = [
        float(item["p90_abs_normal_offset_px"])
        for item in records
        if item["p90_abs_normal_offset_px"] is not None
    ]

    med_angles = [
        float(item["median_abs_angle_deg"])
        for item in records
        if item["median_abs_angle_deg"] is not None
    ]

    constraints = [
        float(item["constraint_count"])
        for item in records
    ]

    print(
        "eligible:",
        f"{eligible}/{len(records)}",
        f"({100.0 * eligible / len(records):.1f}%)",
    )
    print(
        "constraints/frame median:",
        fmt(median(constraints)),
    )
    print(
        "median of frame median offsets:",
        fmt(median(med_offsets)),
        "px",
    )
    print(
        "P90 of frame median offsets:",
        fmt(percentile(med_offsets, 90)),
        "px",
    )
    print(
        "median of frame P90 offsets:",
        fmt(median(p90_offsets)),
        "px",
    )
    print(
        "median angle:",
        fmt(median(med_angles)),
        "deg",
    )


def main() -> None:
    review_path = REVIEWS / f"{ASSET}.json"
    reloc_path = REVIEWS / f"{ASSET}_camera_anchor_relocalization_v1.json"
    dense_path = REVIEWS / f"{ASSET}_camera_atlas_dense_v1.json"

    for path in (review_path, reloc_path, dense_path):
        if not path.exists():
            raise SystemExit(f"Falta: {path}")

    review = load_json(review_path)
    reloc = load_json(reloc_path)
    dense = load_json(dense_path)

    pitch = review.get("field", {}).get("pitch_model", {})
    length_m = float(pitch["length_m"])
    width_m = float(pitch["width_m"])

    physical_pitch = build_pitch_model(
        length_m=length_m,
        width_m=width_m,
    )

    human_map = {
        int(item["frame"]): item
        for item in extract_human_anchors(review)
    }

    reloc_map = {
        int(item["frame"]): item
        for item in reloc.get("results", [])
        if (
            isinstance(item, dict)
            and "frame" in item
            and item.get("status") == "ok"
            and isinstance(item.get("homography"), list)
        )
    }

    dense_map = {
        int(item["frame"]): item
        for item in dense.get("records", [])
        if (
            isinstance(item, dict)
            and "frame" in item
            and isinstance(item.get("homography"), list)
        )
    }

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

    capture = cv2.VideoCapture(str(source.path))
    if not capture.isOpened():
        raise SystemExit(f"No se puede abrir {source.path}")

    image_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    image_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    print("VIDEO:", source.path)
    print("DIMENSIONES:", image_width, "x", image_height)
    print("PITCH:", length_m, "x", width_m)
    print()

    cases: list[dict[str, Any]] = []

    for frame in HUMAN_FRAMES:
        item = human_map.get(frame)
        if item is None:
            print("WARN: HUMAN ausente:", frame)
            continue
        cases.append({
            "group": "HUMAN",
            "frame": frame,
            "homography": item["homography"],
            "origin": "HUMAN exacto",
            "image_metrics": None,
        })

    for frame in KNOWN_BAD_RELOCALIZED:
        item = reloc_map.get(frame)
        if item is None:
            print("WARN: relocalización ausente:", frame)
            continue
        cases.append({
            "group": "RELOCALIZADA_MALA",
            "frame": frame,
            "homography": item["homography"],
            "origin": f"HUMAN {item.get('anchor_frame')}",
            "image_metrics": item,
        })

    for frame in RELOCALIZED_ENDPOINTS:
        item = reloc_map.get(frame)
        if item is None:
            print("WARN: endpoint relocalizado ausente:", frame)
            continue
        cases.append({
            "group": "ENDPOINT_RELOCALIZADO",
            "frame": frame,
            "homography": item["homography"],
            "origin": f"HUMAN {item.get('anchor_frame')}",
            "image_metrics": item,
        })

    for frame in KNOWN_BAD_DENSE:
        item = dense_map.get(frame)
        if item is None:
            print("WARN: Dense ausente:", frame)
            continue
        cases.append({
            "group": "DENSE_MALA",
            "frame": frame,
            "homography": item["homography"],
            "origin": (
                f"{item.get('atlas_left_keyframe')}"
                f"→{item.get('atlas_right_keyframe')}"
            ),
            "image_metrics": None,
        })

    evaluated: list[dict[str, Any]] = []

    for case in cases:
        frame_number = int(case["frame"])
        frame_bgr = read_frame(capture, frame_number)

        if frame_bgr is None:
            print("WARN: no se pudo leer frame", frame_number)
            continue

        diagnostics = evaluate_candidate(
            frame_number=frame_number,
            homography=case["homography"],
            frame_bgr=frame_bgr,
            pitch_model=physical_pitch,
            image_width=image_width,
            image_height=image_height,
        )

        diagnostics.update({
            "group": case["group"],
            "origin": case["origin"],
            "image_metrics": case["image_metrics"],
        })

        evaluated.append(diagnostics)

    capture.release()

    print()
    print(
        "GROUP".ljust(23),
        "FRAME".rjust(6),
        "ELIG".rjust(5),
        "LINES".rjust(7),
        "FAM".rjust(4),
        "MED_OFF".rjust(9),
        "P90_OFF".rjust(9),
        "MED_ANG".rjust(9),
        "CONF".rjust(7),
        "ORIGEN",
    )
    print("-" * 115)

    for item in evaluated:
        print(
            str(item["group"]).ljust(23),
            str(item["frame"]).rjust(6),
            ("yes" if item["eligible"] else "no").rjust(5),
            (
                f"{item['constraint_count']}/"
                f"{item['searched_lines']}"
            ).rjust(7),
            str(item["family_count"]).rjust(4),
            fmt(item["median_abs_normal_offset_px"]).rjust(9),
            fmt(item["p90_abs_normal_offset_px"]).rjust(9),
            fmt(item["median_abs_angle_deg"]).rjust(9),
            fmt(item["median_confidence"]).rjust(7),
            item["origin"],
        )

    groups = [
        "HUMAN",
        "RELOCALIZADA_MALA",
        "ENDPOINT_RELOCALIZADO",
        "DENSE_MALA",
    ]

    for group in groups:
        summarize_group(
            group,
            [
                item
                for item in evaluated
                if item["group"] == group
            ],
        )

    print()
    print("=== DETALLE DE RELOCALIZACIONES ===")

    for item in evaluated:
        if item["group"] not in {
            "RELOCALIZADA_MALA",
            "ENDPOINT_RELOCALIZADO",
        }:
            continue

        metrics = item.get("image_metrics") or {}

        print()
        print(
            f"frame {item['frame']} "
            f"← HUMAN {metrics.get('anchor_frame')}"
        )
        print(
            "  SIFT:",
            f"inliers={metrics.get('inlier_count')}",
            f"ratio={fmt(metrics.get('inlier_ratio'), 3)}",
            f"reproj={fmt(metrics.get('median_reprojection_px'), 3)} px",
            f"tier={metrics.get('confidence_tier')}",
        )
        print(
            "  FIELD:",
            f"eligible={item['eligible']}",
            f"constraints={item['constraint_count']}",
            f"families={item['family_count']}",
            f"median_offset={fmt(item['median_abs_normal_offset_px'])} px",
            f"p90_offset={fmt(item['p90_abs_normal_offset_px'])} px",
        )

        for observation in item["observations"]:
            print(
                "   ",
                observation.get("semantic"),
                f"conf={fmt(observation.get('confidence'), 3)}",
                f"offset={fmt(abs(float(observation.get('normal_offset_px', 0.0))))} px",
                f"angle={fmt(abs(float(observation.get('angle_offset_deg', 0.0))))}°",
                f"support={fmt(observation.get('support_length_px'))} px",
                f"paint={fmt(observation.get('paint_score'), 3)}",
            )

    print()
    print(
        "NOTA: este checker NO cambia ninguna H, "
        "no escribe artifacts y no decide thresholds."
    )


if __name__ == "__main__":
    main()

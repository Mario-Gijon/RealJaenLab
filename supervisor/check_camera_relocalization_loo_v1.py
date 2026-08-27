#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from supervisor.field_camera_anchor_relocalization import (
    AnchorRelocalizationConfig,
    analyse_anchor_relocalization,
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

KNOWN_BAD_RELOCALIZED = [
    0,
    312,
    320,
    336,
    344,
    4134,
    4142,
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON inválido: {path}")
    return value


def normalise_h(value: Any) -> np.ndarray | None:
    try:
        h = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None

    if h.shape != (3, 3) or not np.isfinite(h).all():
        return None

    if abs(float(h[2, 2])) < 1e-9:
        return None

    return h / float(h[2, 2])


def project(
    world: np.ndarray,
    h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.column_stack(
        (world, np.ones(len(world), dtype=np.float64))
    ) @ h.T

    denominator = q[:, 2]

    valid = (
        np.isfinite(q).all(axis=1)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-9)
    )

    image = np.full(
        (len(world), 2),
        np.nan,
        dtype=np.float64,
    )

    image[valid] = (
        q[valid, :2]
        / denominator[valid, None]
    )

    valid &= np.isfinite(image).all(axis=1)

    return image, valid


def candidate_disagreement(
    first_h: Any,
    second_h: Any,
    *,
    image_width: int,
    image_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
) -> dict[str, Any] | None:
    first = normalise_h(first_h)
    second = normalise_h(second_h)

    if first is None or second is None:
        return None

    xs = np.linspace(
        -pitch_length_m / 2.0,
        pitch_length_m / 2.0,
        17,
    )
    ys = np.linspace(
        -pitch_width_m / 2.0,
        pitch_width_m / 2.0,
        13,
    )

    world = np.asarray(
        [(x, y) for y in ys for x in xs],
        dtype=np.float64,
    )

    a, va = project(world, first)
    b, vb = project(world, second)

    margin = 12.0

    visible_a = (
        va
        & (a[:, 0] >= margin)
        & (a[:, 0] <= image_width - margin)
        & (a[:, 1] >= margin)
        & (a[:, 1] <= image_height - margin)
    )

    visible_b = (
        vb
        & (b[:, 0] >= margin)
        & (b[:, 0] <= image_width - margin)
        & (b[:, 1] >= margin)
        & (b[:, 1] <= image_height - margin)
    )

    common = visible_a & visible_b

    if int(np.count_nonzero(common)) < 4:
        return None

    errors = np.linalg.norm(
        a[common] - b[common],
        axis=1,
    )

    return {
        "count": int(len(errors)),
        "median_px": float(np.median(errors)),
        "p90_px": float(np.percentile(errors, 90)),
        "max_px": float(np.max(errors)),
    }


def fmt(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"

    if not math.isfinite(number):
        return "—"

    return f"{number:.{digits}f}"


def valid_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
    output = []

    for candidate in result.get("candidates", []):
        if (
            isinstance(candidate, dict)
            and isinstance(candidate.get("homography"), list)
            and candidate.get("verification", {}).get("status") == "ok"
        ):
            output.append(candidate)

    return output


def add_consensus(
    candidates: list[dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
) -> None:
    for candidate in candidates:
        disagreements = []

        for other in candidates:
            if other is candidate:
                continue

            result = candidate_disagreement(
                candidate["homography"],
                other["homography"],
                image_width=image_width,
                image_height=image_height,
                pitch_length_m=pitch_length_m,
                pitch_width_m=pitch_width_m,
            )

            if result is not None:
                disagreements.append(
                    float(result["median_px"])
                )

        candidate["_consensus_count"] = len(disagreements)

        candidate["_consensus_median_px"] = (
            float(np.median(disagreements))
            if disagreements
            else None
        )

        candidate["_consensus_p90_px"] = (
            float(np.percentile(disagreements, 90))
            if disagreements
            else None
        )


def verification_metrics(candidate: dict[str, Any]) -> dict[str, Any]:
    verification = candidate.get("verification", {})

    return {
        "matches": verification.get("match_count"),
        "inliers": verification.get("inlier_count"),
        "ratio": verification.get("inlier_ratio"),
        "reprojection": verification.get(
            "median_reprojection_px"
        ),
        "span_x": verification.get(
            "span_x_fraction"
        ),
        "span_y": verification.get(
            "span_y_fraction"
        ),
        "hull": verification.get(
            "hull_area_fraction"
        ),
        "state_median": (
            verification
            .get("direct_state_distance", {})
            .get("median_px")
        ),
        "state_p90": (
            verification
            .get("direct_state_distance", {})
            .get("p90_px")
        ),
    }


def main() -> None:
    review = load_json(
        REVIEWS / f"{ASSET}.json"
    )

    existing_reloc = load_json(
        REVIEWS
        / f"{ASSET}_camera_anchor_relocalization_v1.json"
    )

    pitch = review["field"]["pitch_model"]

    length_m = float(pitch["length_m"])
    width_m = float(pitch["width_m"])

    anchors = extract_human_anchors(review)

    if len(anchors) != 5:
        print(
            f"AVISO: esperaba 5 HUMAN y hay {len(anchors)}"
        )

    truth = {
        int(item["frame"]): item["homography"]
        for item in anchors
    }

    human_frames = sorted(truth)

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
        raise SystemExit(
            f"No se pudo abrir {source.path}"
        )

    image_width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
    )
    image_height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
    )

    capture.release()

    print("VIDEO:", source.path)
    print(
        "HUMAN ground truth:",
        human_frames,
    )
    print(
        "PITCH:",
        length_m,
        "x",
        width_m,
    )

    print()
    print(
        "Ejecutando leave-one-out real:"
        " cada HUMAN target se excluye a sí mismo."
    )

    loo = analyse_anchor_relocalization(
        source.path,
        anchors=anchors,
        targets=[
            {"frame": frame}
            for frame in human_frames
        ],
        config=AnchorRelocalizationConfig(
            analysis_width=960
        ),
        exclude_same_frame=True,
    )

    if loo.get("status") != "ok":
        raise SystemExit(
            f"LOO falló: {loo.get('status')}"
        )

    chosen_errors = []
    oracle_errors = []
    consensus_errors = []

    print()
    print("=" * 100)
    print("LOO CONTRA HUMAN GROUND TRUTH")
    print("=" * 100)

    for result in loo.get("results", []):
        frame = int(result["frame"])
        true_h = truth[frame]

        candidates = valid_candidates(result)

        add_consensus(
            candidates,
            image_width=image_width,
            image_height=image_height,
            pitch_length_m=length_m,
            pitch_width_m=width_m,
        )

        for candidate in candidates:
            evaluation = evaluate_homography_visible_grid(
                candidate["homography"],
                true_h,
                image_width=image_width,
                image_height=image_height,
                pitch_length_m=length_m,
                pitch_width_m=width_m,
            )

            candidate["_truth_grid"] = (
                evaluation
                if evaluation.get("status") == "ok"
                else None
            )

        valid_truth = [
            candidate
            for candidate in candidates
            if isinstance(
                candidate.get("_truth_grid"),
                dict,
            )
        ]

        print()
        print(
            f"TARGET HUMAN {frame}"
            f" | candidatos verificados={len(candidates)}"
            f" | elegido={result.get('anchor_frame')}"
        )

        print(
            "ANCHOR".rjust(7),
            "SEL".rjust(4),
            "SCORE".rjust(8),
            "INL".rjust(5),
            "RATIO".rjust(7),
            "REPR".rjust(7),
            "SPANX".rjust(7),
            "SPANY".rjust(7),
            "HULL".rjust(7),
            "STATE".rjust(8),
            "CONS".rjust(8),
            "TRUE_MED".rjust(10),
            "TRUE_P90".rjust(10),
        )

        print("-" * 105)

        for candidate in sorted(
            candidates,
            key=lambda item: -float(
                item.get("score", -1e9)
            ),
        ):
            metrics = verification_metrics(candidate)
            grid = candidate.get("_truth_grid")

            selected = (
                int(candidate["anchor_frame"])
                == int(result.get("anchor_frame"))
            )

            print(
                str(candidate["anchor_frame"]).rjust(7),
                ("*" if selected else "").rjust(4),
                fmt(candidate.get("score"), 3).rjust(8),
                str(metrics["inliers"]).rjust(5),
                fmt(metrics["ratio"], 3).rjust(7),
                fmt(metrics["reprojection"], 3).rjust(7),
                fmt(metrics["span_x"], 3).rjust(7),
                fmt(metrics["span_y"], 3).rjust(7),
                fmt(metrics["hull"], 3).rjust(7),
                fmt(metrics["state_median"], 1).rjust(8),
                fmt(
                    candidate.get(
                        "_consensus_median_px"
                    ),
                    1,
                ).rjust(8),
                fmt(
                    grid.get("median_px")
                    if grid else None,
                    1,
                ).rjust(10),
                fmt(
                    grid.get("p90_px")
                    if grid else None,
                    1,
                ).rjust(10),
            )

        selected_candidates = [
            candidate
            for candidate in valid_truth
            if int(candidate["anchor_frame"])
            == int(result.get("anchor_frame"))
        ]

        if selected_candidates:
            selected_error = float(
                selected_candidates[0]
                ["_truth_grid"]["median_px"]
            )
            chosen_errors.append(selected_error)
        else:
            selected_error = None

        if valid_truth:
            oracle = min(
                valid_truth,
                key=lambda item: float(
                    item["_truth_grid"]["median_px"]
                ),
            )

            oracle_error = float(
                oracle["_truth_grid"]["median_px"]
            )

            oracle_errors.append(oracle_error)

            consensus_available = [
                candidate
                for candidate in valid_truth
                if candidate.get(
                    "_consensus_median_px"
                ) is not None
            ]

            consensus_best = (
                min(
                    consensus_available,
                    key=lambda item: float(
                        item["_consensus_median_px"]
                    ),
                )
                if consensus_available
                else None
            )

            if consensus_best is not None:
                consensus_error = float(
                    consensus_best
                    ["_truth_grid"]["median_px"]
                )
                consensus_errors.append(
                    consensus_error
                )
            else:
                consensus_error = None

            print()
            print(
                "  seleccionado actual:",
                result.get("anchor_frame"),
                "→ error real mediano",
                fmt(selected_error),
                "px",
            )

            print(
                "  oracle:",
                oracle["anchor_frame"],
                "→ error real mediano",
                fmt(oracle_error),
                "px",
            )

            if consensus_best is not None:
                print(
                    "  mejor consenso:",
                    consensus_best["anchor_frame"],
                    "→ consenso",
                    fmt(
                        consensus_best[
                            "_consensus_median_px"
                        ]
                    ),
                    "px"
                    " → error real",
                    fmt(consensus_error),
                    "px",
                )

    print()
    print("=" * 100)
    print("RESUMEN LOO")
    print("=" * 100)

    def summary(
        name: str,
        values: list[float],
    ) -> None:
        if not values:
            print(name, "sin valores")
            return

        arr = np.asarray(
            values,
            dtype=np.float64,
        )

        print(
            name,
            "median=",
            fmt(np.median(arr)),
            "px",
            "p90=",
            fmt(np.percentile(arr, 90)),
            "px",
            "max=",
            fmt(np.max(arr)),
            "px",
        )

    summary(
        "selector actual:",
        chosen_errors,
    )

    summary(
        "oracle candidato:",
        oracle_errors,
    )

    summary(
        "candidato de mayor consenso:",
        consensus_errors,
    )

    print()
    print("=" * 100)
    print("CONSENSO EN FRAMES VISUALMENTE MALOS")
    print("=" * 100)

    existing_results = {
        int(item["frame"]): item
        for item in existing_reloc.get(
            "results",
            [],
        )
        if (
            isinstance(item, dict)
            and "frame" in item
        )
    }

    for frame in KNOWN_BAD_RELOCALIZED:
        result = existing_results.get(frame)

        if result is None:
            print()
            print(frame, "no aparece en relocalización")
            continue

        candidates = valid_candidates(result)

        add_consensus(
            candidates,
            image_width=image_width,
            image_height=image_height,
            pitch_length_m=length_m,
            pitch_width_m=width_m,
        )

        print()
        print(
            f"FRAME MALO {frame}"
            f" | elegido={result.get('anchor_frame')}"
            f" | candidatos={len(candidates)}"
        )

        print(
            "ANCHOR".rjust(7),
            "SEL".rjust(4),
            "SCORE".rjust(8),
            "INL".rjust(5),
            "RATIO".rjust(7),
            "REPR".rjust(7),
            "STATE".rjust(8),
            "CONS_MED".rjust(10),
            "CONS_P90".rjust(10),
        )

        print("-" * 80)

        for candidate in sorted(
            candidates,
            key=lambda item: -float(
                item.get("score", -1e9)
            ),
        ):
            metrics = verification_metrics(candidate)

            selected = (
                int(candidate["anchor_frame"])
                == int(result.get("anchor_frame"))
            )

            print(
                str(candidate["anchor_frame"]).rjust(7),
                ("*" if selected else "").rjust(4),
                fmt(candidate.get("score"), 3).rjust(8),
                str(metrics["inliers"]).rjust(5),
                fmt(metrics["ratio"], 3).rjust(7),
                fmt(metrics["reprojection"], 3).rjust(7),
                fmt(metrics["state_median"], 1).rjust(8),
                fmt(
                    candidate.get(
                        "_consensus_median_px"
                    ),
                    1,
                ).rjust(10),
                fmt(
                    candidate.get(
                        "_consensus_p90_px"
                    ),
                    1,
                ).rjust(10),
            )

    print()
    print(
        "NOTA: diagnóstico puro."
        " No cambia Relocalization,"
        " Dense ni ningún artifact."
    )


if __name__ == "__main__":
    main()

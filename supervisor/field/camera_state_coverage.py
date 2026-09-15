"""Camera-State coverage diagnostics for the RealJaenLab HUMAN Atlas.

V1 answers a different question from ``human_view_descriptor``:

    FIELD coverage
        "Has this canonical piece of grass appeared in at least one HUMAN?"

    Camera-State coverage
        "Does this target view have at least one sufficiently close/reliable
         HUMAN anchor, or a small set of anchors that consistently explain it?"

The target frame does NOT need manual pitch annotations.  Evidence comes from
the existing image-to-image ``view_transform_probe`` between each HUMAN anchor
and the target frame.

For each HUMAN -> target candidate this module combines:

- visual evidence quality:
  robust inliers, inlier ratio, spatial match coverage and transform residual;
- relative view severity:
  image-centre shift, scale change, rotation, anisotropy and projective terms;
- optional predicted target ViewDescriptor:
  transport the HUMAN projected geometry through the image transform and
  describe the transported view numerically;
- optional multi-anchor agreement:
  if two independent HUMAN anchors predict essentially the same target state,
  a target can be considered covered even when neither anchor is extremely
  close by itself.

This is deliberately diagnostic V1.  Thresholds are exposed and the result
keeps its component scores.  It is not yet the production CameraState model,
interpolator, temporal tracker or full-match calibrator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


EPS = 1e-10


class CameraStateCoverageError(ValueError):
    """Invalid input for Camera-State coverage diagnostics."""


@dataclass(frozen=True, slots=True)
class CoverageConfig:
    """Tunable V1 thresholds.

    Characteristic scales are not hard category boundaries.  They normalize
    heterogeneous Camera-State changes into one dimensionless diagnostic
    distance.
    """

    # Evidence-quality ramps.
    inlier_count_bad: float = 12.0
    inlier_count_good: float = 80.0
    inlier_ratio_bad: float = 0.25
    inlier_ratio_good: float = 0.85
    spatial_coverage_bad: float = 0.02
    spatial_coverage_good: float = 0.20
    p95_fraction_good: float = 0.0015
    p95_fraction_bad: float = 0.0080
    mutual_match_bad: float = 16.0
    mutual_match_good: float = 150.0

    # Relative image-transform Camera-State scales.
    shift_norm_scale: float = 0.30
    log_scale_scale: float = 0.35
    rotation_deg_scale: float = 8.0
    log_anisotropy_scale: float = 0.22
    projective_norm_scale: float = 0.18

    # Descriptor delta scales.
    descriptor_center_scale: float = 0.28
    descriptor_log_scale_scale: float = 0.40
    descriptor_log_anisotropy_scale: float = 0.25
    descriptor_axis_angle_deg_scale: float = 10.0

    # Agreement between two predicted target descriptors.
    agreement_center_scale: float = 0.12
    agreement_log_scale_scale: float = 0.18
    agreement_log_anisotropy_scale: float = 0.18
    agreement_axis_angle_deg_scale: float = 7.0
    max_consensus_disagreement: float = 1.00

    # Final classification.
    observable_evidence_min: float = 0.30
    strong_evidence_min: float = 0.50
    consensus_evidence_min: float = 0.48
    single_anchor_max_distance: float = 1.45
    consensus_anchor_max_distance: float = 2.20
    uncertain_max_distance: float = 1.95
    single_anchor_score_min: float = 0.32
    uncertain_score_min: float = 0.16


DEFAULT_CONFIG = CoverageConfig()


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CameraStateCoverageError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise CameraStateCoverageError(f"{label} must be finite.")
    return number


def _optional_finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _ascending_ramp(value: float, bad: float, good: float) -> float:
    if good <= bad:
        raise CameraStateCoverageError("Ascending ramp requires good > bad.")
    return _clip01((value - bad) / (good - bad))


def _descending_ramp(value: float, good: float, bad: float) -> float:
    if bad <= good:
        raise CameraStateCoverageError("Descending ramp requires bad > good.")
    return _clip01((bad - value) / (bad - good))


def _weighted_rms(values: Sequence[float], weights: Sequence[float]) -> float:
    if len(values) != len(weights) or not values:
        raise CameraStateCoverageError("weighted RMS requires matching non-empty inputs.")
    arr = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(arr)) or not np.all(np.isfinite(w)) or np.sum(w) <= 0:
        raise CameraStateCoverageError("weighted RMS received invalid values.")
    return float(math.sqrt(float(np.sum(w * arr * arr) / np.sum(w))))


def _angle_delta_deg(a: float, b: float) -> float:
    """Smallest unsigned difference for 180-degree image-axis directions."""

    raw = abs(float(a) - float(b)) % 180.0
    return float(min(raw, 180.0 - raw))


def _model_rank(name: str | None) -> int:
    return {
        "translation": 0,
        "similarity": 1,
        "affine": 2,
        "homography": 3,
    }.get(str(name), 4)


# ---------------------------------------------------------------------------
# ViewDescriptor summaries / distances
# ---------------------------------------------------------------------------


def _descriptor_state(descriptor: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(descriptor, dict):
        return None

    coverage = descriptor.get("coverage") or {}
    center = (
        coverage.get("view_center_normalized")
        or coverage.get("visible_centroid_normalized")
    )
    scale = descriptor.get("effective_view_scale") or {}

    if (
        not isinstance(center, Sequence)
        or isinstance(center, (str, bytes))
        or len(center) != 2
    ):
        return None

    cx = _optional_finite(center[0])
    cy = _optional_finite(center[1])
    geometric_scale = _optional_finite(scale.get("scale_geometric_px_per_m"))
    anisotropy = _optional_finite(scale.get("anisotropy_ratio"))
    axis_angle = _optional_finite(scale.get("axis_x_angle_deg"))

    if (
        cx is None
        or cy is None
        or geometric_scale is None
        or geometric_scale <= EPS
        or anisotropy is None
        or anisotropy <= EPS
    ):
        return None

    return {
        "center_x": cx,
        "center_y": cy,
        "scale_geometric_px_per_m": geometric_scale,
        "log_scale": float(math.log(geometric_scale)),
        "anisotropy_ratio": anisotropy,
        "log_anisotropy": float(math.log(anisotropy)),
        "axis_x_angle_deg": axis_angle if axis_angle is not None else 0.0,
    }


def _descriptor_delta_distance(
    anchor_descriptor: dict[str, Any] | None,
    target_descriptor: dict[str, Any] | None,
    config: CoverageConfig,
) -> tuple[float | None, dict[str, float] | None]:
    source = _descriptor_state(anchor_descriptor)
    target = _descriptor_state(target_descriptor)
    if source is None or target is None:
        return None, None

    center = math.hypot(
        target["center_x"] - source["center_x"],
        target["center_y"] - source["center_y"],
    )
    log_scale = abs(target["log_scale"] - source["log_scale"])
    log_anisotropy = abs(target["log_anisotropy"] - source["log_anisotropy"])
    angle = _angle_delta_deg(
        target["axis_x_angle_deg"],
        source["axis_x_angle_deg"],
    )

    normalized = {
        "center": center / config.descriptor_center_scale,
        "log_scale": log_scale / config.descriptor_log_scale_scale,
        "log_anisotropy": log_anisotropy / config.descriptor_log_anisotropy_scale,
        "axis_angle": angle / config.descriptor_axis_angle_deg_scale,
    }

    distance = _weighted_rms(
        list(normalized.values()),
        [0.40, 0.35, 0.10, 0.15],
    )
    return distance, normalized


def _prediction_disagreement(
    first: dict[str, Any] | None,
    second: dict[str, Any] | None,
    config: CoverageConfig,
) -> tuple[float | None, dict[str, float] | None]:
    a = _descriptor_state(first)
    b = _descriptor_state(second)
    if a is None or b is None:
        return None, None

    center = math.hypot(
        a["center_x"] - b["center_x"],
        a["center_y"] - b["center_y"],
    )
    log_scale = abs(a["log_scale"] - b["log_scale"])
    log_anisotropy = abs(a["log_anisotropy"] - b["log_anisotropy"])
    angle = _angle_delta_deg(a["axis_x_angle_deg"], b["axis_x_angle_deg"])

    normalized = {
        "center": center / config.agreement_center_scale,
        "log_scale": log_scale / config.agreement_log_scale_scale,
        "log_anisotropy": log_anisotropy / config.agreement_log_anisotropy_scale,
        "axis_angle": angle / config.agreement_axis_angle_deg_scale,
    }
    distance = _weighted_rms(
        list(normalized.values()),
        [0.45, 0.35, 0.08, 0.12],
    )
    return distance, normalized


# ---------------------------------------------------------------------------
# Probe quality / relative Camera-State distance
# ---------------------------------------------------------------------------


def _chosen_probe_model(probe: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if not isinstance(probe, dict):
        raise CameraStateCoverageError("candidate probe must be an object.")

    model_name = str(probe.get("recommended_model") or "")
    models = probe.get("models")
    if not isinstance(models, dict) or model_name not in models:
        raise CameraStateCoverageError(
            "probe is missing its recommended transform model."
        )

    model = models[model_name]
    if not isinstance(model, dict):
        raise CameraStateCoverageError("probe recommended model is invalid.")
    return model_name, model


def _evidence_quality(
    probe: dict[str, Any],
    config: CoverageConfig,
) -> tuple[float, dict[str, float]]:
    _, model = _chosen_probe_model(probe)
    residual = model.get("residual_summary") or {}

    inlier_count = _finite(
        probe.get("consensus_inlier_count", 0),
        "consensus_inlier_count",
    )
    inlier_ratio = _finite(
        probe.get("consensus_inlier_ratio", 0),
        "consensus_inlier_ratio",
    )
    source_coverage = _finite(
        probe.get("source_spatial_coverage", 0),
        "source_spatial_coverage",
    )
    target_coverage = _finite(
        probe.get("target_spatial_coverage", 0),
        "target_spatial_coverage",
    )
    mutual_matches = _finite(
        probe.get("mutual_match_count", 0),
        "mutual_match_count",
    )

    p95_fraction = _optional_finite(residual.get("p95_diagonal_fraction"))
    if p95_fraction is None:
        p95_px = _optional_finite(residual.get("p95_px"))
        width = _optional_finite(probe.get("image_width"))
        height = _optional_finite(probe.get("image_height"))
        if p95_px is not None and width and height:
            p95_fraction = p95_px / math.hypot(width, height)
        else:
            p95_fraction = config.p95_fraction_bad

    component = {
        "inlier_count": _ascending_ramp(
            inlier_count,
            config.inlier_count_bad,
            config.inlier_count_good,
        ),
        "inlier_ratio": _ascending_ramp(
            inlier_ratio,
            config.inlier_ratio_bad,
            config.inlier_ratio_good,
        ),
        "spatial_coverage": _ascending_ramp(
            min(source_coverage, target_coverage),
            config.spatial_coverage_bad,
            config.spatial_coverage_good,
        ),
        "residual": _descending_ramp(
            p95_fraction,
            config.p95_fraction_good,
            config.p95_fraction_bad,
        ),
        "mutual_matches": _ascending_ramp(
            mutual_matches,
            config.mutual_match_bad,
            config.mutual_match_good,
        ),
    }

    weighted = (
        0.20 * component["inlier_count"]
        + 0.22 * component["inlier_ratio"]
        + 0.28 * component["spatial_coverage"]
        + 0.22 * component["residual"]
        + 0.08 * component["mutual_matches"]
    )

    # A relation with one catastrophic evidence dimension should not obtain an
    # unrealistically high score just because the others are excellent.
    minimum = min(component.values())
    quality = weighted * (0.55 + 0.45 * minimum)

    return float(_clip01(quality)), component


def _transform_state_distance(
    probe: dict[str, Any],
    config: CoverageConfig,
) -> tuple[float, dict[str, float]]:
    _, model = _chosen_probe_model(probe)
    transform = model.get("transform") or {}

    shift = transform.get("center_shift_normalized") or [0.0, 0.0]
    if (
        not isinstance(shift, Sequence)
        or isinstance(shift, (str, bytes))
        or len(shift) != 2
    ):
        shift = [0.0, 0.0]

    sx = _optional_finite(shift[0]) or 0.0
    sy = _optional_finite(shift[1]) or 0.0
    shift_norm = math.hypot(sx, sy)

    log_scale = _optional_finite(transform.get("log_scale"))
    if log_scale is None:
        scale = _optional_finite(transform.get("scale_geometric"))
        log_scale = math.log(max(scale or 1.0, EPS))

    rotation = abs(_optional_finite(transform.get("rotation_deg")) or 0.0)

    anisotropy = _optional_finite(transform.get("anisotropy_ratio"))
    if anisotropy is None or anisotropy <= EPS:
        anisotropy = 1.0
    log_anisotropy = abs(math.log(anisotropy))

    projective = transform.get("projective_terms_per_diagonal") or [0.0, 0.0]
    if (
        not isinstance(projective, Sequence)
        or isinstance(projective, (str, bytes))
        or len(projective) != 2
    ):
        projective = [0.0, 0.0]

    p0 = _optional_finite(projective[0]) or 0.0
    p1 = _optional_finite(projective[1]) or 0.0
    projective_norm = math.hypot(p0, p1)

    normalized = {
        "center_shift": shift_norm / config.shift_norm_scale,
        "log_scale": abs(log_scale) / config.log_scale_scale,
        "rotation": rotation / config.rotation_deg_scale,
        "log_anisotropy": log_anisotropy / config.log_anisotropy_scale,
        "projective": projective_norm / config.projective_norm_scale,
    }

    distance = _weighted_rms(
        list(normalized.values()),
        [0.30, 0.30, 0.10, 0.12, 0.18],
    )
    return distance, normalized


# ---------------------------------------------------------------------------
# Optional transported descriptor derivation
# ---------------------------------------------------------------------------


def derive_predicted_target_descriptor(
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    """Derive a target ViewDescriptor from HUMAN geometry + probe transform.

    This imports the already-existing FIELD modules lazily so the pure scoring
    self-check below does not require OpenCV.
    """

    if not isinstance(candidate, dict):
        raise CameraStateCoverageError("candidate must be an object.")

    geometry = candidate.get("projected_geometry")
    field_profile = candidate.get("field_profile") or {}
    probe = candidate.get("probe")

    if not isinstance(geometry, dict) or not isinstance(field_profile, dict):
        return None
    if not isinstance(probe, dict):
        return None

    model_name, model = _chosen_probe_model(probe)
    matrix = model.get("matrix")
    if matrix is None:
        return None

    width = _optional_finite(probe.get("image_width"))
    height = _optional_finite(probe.get("image_height"))
    length = _optional_finite(field_profile.get("length"))
    pitch_width = _optional_finite(field_profile.get("width"))

    if (
        width is None
        or height is None
        or length is None
        or pitch_width is None
        or min(width, height, length, pitch_width) <= 0
    ):
        return None

    try:
        from .view_transform_probe import transform_projected_geometry
        from .human_view_descriptor import describe_human_view
    except ImportError:
        try:
            from view_transform_probe import transform_projected_geometry
            from human_view_descriptor import describe_human_view
        except ImportError as exc:
            raise CameraStateCoverageError(
                "Could not import FIELD transform/descriptor helpers."
            ) from exc

    transported = transform_projected_geometry(geometry, matrix)
    return describe_human_view(
        transported,
        pitch_length=length,
        pitch_width=pitch_width,
        video_width=width,
        video_height=height,
        human_id=f"predicted_from_{candidate.get('anchor_id', '?')}",
    )


# ---------------------------------------------------------------------------
# Candidate / target evaluation
# ---------------------------------------------------------------------------


def evaluate_camera_state_candidate(
    candidate: dict[str, Any],
    *,
    config: CoverageConfig = DEFAULT_CONFIG,
    derive_prediction: bool = True,
) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        raise CameraStateCoverageError("candidate must be an object.")

    anchor_id = str(candidate.get("anchor_id") or candidate.get("state_id") or "")
    if not anchor_id:
        raise CameraStateCoverageError("candidate requires anchor_id.")

    probe = candidate.get("probe")
    if not isinstance(probe, dict):
        raise CameraStateCoverageError(f"{anchor_id}: candidate requires probe.")

    model_name, model = _chosen_probe_model(probe)
    evidence, evidence_components = _evidence_quality(probe, config)
    transform_distance, transform_components = _transform_state_distance(
        probe,
        config,
    )

    anchor_descriptor = candidate.get("anchor_descriptor")
    predicted_descriptor = candidate.get("predicted_descriptor")

    prediction_error: str | None = None
    if predicted_descriptor is None and derive_prediction:
        try:
            predicted_descriptor = derive_predicted_target_descriptor(candidate)
        except Exception as error:  # diagnostic path must not destroy probe score
            prediction_error = str(error)
            predicted_descriptor = None

    descriptor_distance, descriptor_components = _descriptor_delta_distance(
        anchor_descriptor,
        predicted_descriptor,
        config,
    )

    if descriptor_distance is None:
        state_distance = transform_distance
        distance_source = "image_transform"
    else:
        state_distance = (
            0.55 * transform_distance
            + 0.45 * descriptor_distance
        )
        distance_source = "image_transform+predicted_descriptor"

    # Proximity likelihood.  A strong visual probe is necessary but not
    # sufficient: a visually matchable target can still be too far from the
    # HUMAN anchor to trust it as a direct atlas representative.
    proximity = math.exp(-0.5 * (state_distance / 1.20) ** 2)
    score = float(evidence * proximity)

    return {
        "anchor_id": anchor_id,
        "anchor_frame_index": candidate.get("anchor_frame_index"),
        "anchor_type": candidate.get("anchor_type"),
        "recommended_model": model_name,
        "model_complexity_rank": _model_rank(model_name),
        "evidence_score": evidence,
        "evidence_components": evidence_components,
        "transform_state_distance": transform_distance,
        "transform_distance_components": transform_components,
        "descriptor_state_distance": descriptor_distance,
        "descriptor_distance_components": descriptor_components,
        "state_distance": float(state_distance),
        "state_distance_source": distance_source,
        "candidate_score": score,
        "predicted_descriptor": predicted_descriptor,
        "prediction_error": prediction_error,
        "probe_warnings": list(probe.get("warnings") or []),
        "probe_summary": {
            "consensus_inlier_count": probe.get("consensus_inlier_count"),
            "consensus_inlier_ratio": probe.get("consensus_inlier_ratio"),
            "source_spatial_coverage": probe.get("source_spatial_coverage"),
            "target_spatial_coverage": probe.get("target_spatial_coverage"),
            "mutual_match_count": probe.get("mutual_match_count"),
            "p95_px": (model.get("residual_summary") or {}).get("p95_px"),
            "p95_diagonal_fraction": (
                model.get("residual_summary") or {}
            ).get("p95_diagonal_fraction"),
            "transform": model.get("transform"),
        },
    }


def _best_consensus_pair(
    evaluations: Sequence[dict[str, Any]],
    config: CoverageConfig,
) -> dict[str, Any] | None:
    eligible = [
        item
        for item in evaluations
        if item["evidence_score"] >= config.consensus_evidence_min
        and item["state_distance"] <= config.consensus_anchor_max_distance
        and isinstance(item.get("predicted_descriptor"), dict)
    ]

    best: dict[str, Any] | None = None

    for index, first in enumerate(eligible):
        for second in eligible[index + 1 :]:
            disagreement, components = _prediction_disagreement(
                first.get("predicted_descriptor"),
                second.get("predicted_descriptor"),
                config,
            )
            if disagreement is None:
                continue

            pair_score = (
                0.5 * (first["candidate_score"] + second["candidate_score"])
                * math.exp(-0.5 * disagreement * disagreement)
            )

            item = {
                "anchor_ids": [first["anchor_id"], second["anchor_id"]],
                "prediction_disagreement": disagreement,
                "disagreement_components": components,
                "pair_score": float(pair_score),
            }
            if best is None or item["pair_score"] > best["pair_score"]:
                best = item

    return best


def evaluate_target_camera_state(
    candidates: Sequence[dict[str, Any]],
    *,
    target_frame_index: int | None = None,
    target_timestamp_seconds: float | None = None,
    config: CoverageConfig = DEFAULT_CONFIG,
    derive_predictions: bool = True,
) -> dict[str, Any]:
    """Evaluate whether one target frame is represented by the HUMAN Atlas."""

    if (
        not isinstance(candidates, Sequence)
        or isinstance(candidates, (str, bytes))
        or len(candidates) == 0
    ):
        raise CameraStateCoverageError(
            "At least one HUMAN->target candidate is required."
        )

    evaluations: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for raw in candidates:
        try:
            evaluations.append(
                evaluate_camera_state_candidate(
                    raw,
                    config=config,
                    derive_prediction=derive_predictions,
                )
            )
        except CameraStateCoverageError as error:
            anchor = (
                raw.get("anchor_id")
                if isinstance(raw, dict)
                else "?"
            )
            failures.append(
                {
                    "anchor_id": str(anchor),
                    "error": str(error),
                }
            )

    if not evaluations:
        return {
            "target_frame_index": target_frame_index,
            "target_timestamp_seconds": target_timestamp_seconds,
            "status": "unobservable",
            "covered": False,
            "coverage_mode": None,
            "best_anchor_id": None,
            "best_candidate": None,
            "candidates": [],
            "candidate_failures": failures,
            "consensus": None,
            "warnings": [
                "No HUMAN anchor produced a usable visual relation to this target."
            ],
        }

    evaluations.sort(
        key=lambda item: (
            -item["candidate_score"],
            item["state_distance"],
            -item["evidence_score"],
            item["model_complexity_rank"],
        )
    )
    best = evaluations[0]

    observable = any(
        item["evidence_score"] >= config.observable_evidence_min
        for item in evaluations
    )

    strong_single = (
        best["evidence_score"] >= config.strong_evidence_min
        and best["state_distance"] <= config.single_anchor_max_distance
        and best["candidate_score"] >= config.single_anchor_score_min
    )

    consensus = _best_consensus_pair(evaluations, config)
    strong_consensus = (
        consensus is not None
        and consensus["prediction_disagreement"]
        <= config.max_consensus_disagreement
    )

    if strong_single:
        status = "covered"
        coverage_mode = "single_anchor"
    elif strong_consensus:
        status = "covered"
        coverage_mode = "multi_anchor_consensus"
    elif not observable:
        status = "unobservable"
        coverage_mode = None
    elif (
        best["state_distance"] <= config.uncertain_max_distance
        and best["candidate_score"] >= config.uncertain_score_min
    ):
        status = "uncertain"
        coverage_mode = "marginal_anchor"
    else:
        status = "uncovered"
        coverage_mode = None

    warnings: list[str] = []
    if status == "uncovered":
        warnings.append(
            "Visual evidence exists, but every HUMAN anchor is too distant or weak "
            "for conservative Camera-State coverage."
        )
    elif status == "uncertain":
        warnings.append(
            "A HUMAN anchor is plausible but not strong enough for strict coverage."
        )
    elif status == "unobservable":
        warnings.append(
            "The target cannot be judged reliably from current visual matching; "
            "this is not automatically a missing CameraState."
        )

    return {
        "target_frame_index": target_frame_index,
        "target_timestamp_seconds": target_timestamp_seconds,
        "status": status,
        "covered": status == "covered",
        "coverage_mode": coverage_mode,
        "best_anchor_id": best["anchor_id"],
        "best_candidate": best,
        "candidates": evaluations,
        "candidate_failures": failures,
        "consensus": consensus,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Multi-frame coverage summary
# ---------------------------------------------------------------------------


def _problem_clusters(
    evaluations: Sequence[dict[str, Any]],
    *,
    max_gap_frames: int,
) -> list[dict[str, Any]]:
    problem = [
        item
        for item in evaluations
        if item.get("status") in {"uncovered", "uncertain"}
        and isinstance(item.get("target_frame_index"), int)
    ]
    problem.sort(key=lambda item: item["target_frame_index"])

    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for item in problem:
        if not current:
            current = [item]
            continue

        if (
            item["target_frame_index"]
            - current[-1]["target_frame_index"]
            <= max_gap_frames
        ):
            current.append(item)
        else:
            clusters.append(current)
            current = [item]

    if current:
        clusters.append(current)

    result: list[dict[str, Any]] = []
    for cluster in clusters:
        frames = [item["target_frame_index"] for item in cluster]
        statuses: dict[str, int] = {}
        anchors: dict[str, int] = {}

        for item in cluster:
            status = str(item.get("status"))
            statuses[status] = statuses.get(status, 0) + 1
            anchor = item.get("best_anchor_id")
            if anchor:
                anchors[str(anchor)] = anchors.get(str(anchor), 0) + 1

        result.append(
            {
                "start_frame": min(frames),
                "end_frame": max(frames),
                "sample_count": len(cluster),
                "status_counts": statuses,
                "most_common_best_anchor": (
                    max(anchors, key=anchors.get)
                    if anchors
                    else None
                ),
            }
        )
    return result


def summarize_camera_state_coverage(
    target_evaluations: Sequence[dict[str, Any]],
    *,
    max_problem_gap_frames: int = 90,
) -> dict[str, Any]:
    """Summarize sampled target evaluations into Camera-State coverage metrics."""

    if (
        not isinstance(target_evaluations, Sequence)
        or isinstance(target_evaluations, (str, bytes))
        or len(target_evaluations) == 0
    ):
        raise CameraStateCoverageError(
            "At least one target evaluation is required."
        )

    valid_status = {"covered", "uncertain", "uncovered", "unobservable"}
    counts = {name: 0 for name in valid_status}

    anchor_usage: dict[str, int] = {}
    coverage_modes: dict[str, int] = {}

    for item in target_evaluations:
        if not isinstance(item, dict):
            raise CameraStateCoverageError(
                "Each target evaluation must be an object."
            )
        status = str(item.get("status"))
        if status not in valid_status:
            raise CameraStateCoverageError(
                f"Unknown target status {status!r}."
            )
        counts[status] += 1

        if status == "covered":
            anchor = item.get("best_anchor_id")
            if anchor:
                anchor_usage[str(anchor)] = anchor_usage.get(str(anchor), 0) + 1
            mode = item.get("coverage_mode")
            if mode:
                coverage_modes[str(mode)] = coverage_modes.get(str(mode), 0) + 1

    total = len(target_evaluations)
    observable = total - counts["unobservable"]
    strict_covered = counts["covered"]
    covered_or_uncertain = counts["covered"] + counts["uncertain"]

    observable_strict_fraction = (
        strict_covered / observable
        if observable
        else 0.0
    )
    observable_relaxed_fraction = (
        covered_or_uncertain / observable
        if observable
        else 0.0
    )
    all_sample_strict_fraction = strict_covered / total

    return {
        "purpose": (
            "Diagnostic sampled Camera-State coverage of the HUMAN Atlas. "
            "This is separate from geometric FIELD coverage."
        ),
        "sample_count": total,
        "status_counts": counts,
        "observable_sample_count": observable,
        "unobservable_sample_count": counts["unobservable"],
        "strict_camera_state_coverage_fraction": float(
            observable_strict_fraction
        ),
        "strict_camera_state_coverage_percent": float(
            100.0 * observable_strict_fraction
        ),
        "covered_or_uncertain_fraction": float(
            observable_relaxed_fraction
        ),
        "covered_or_uncertain_percent": float(
            100.0 * observable_relaxed_fraction
        ),
        "all_sample_strict_coverage_fraction": float(
            all_sample_strict_fraction
        ),
        "all_sample_strict_coverage_percent": float(
            100.0 * all_sample_strict_fraction
        ),
        "coverage_mode_counts": coverage_modes,
        "best_anchor_usage_counts": anchor_usage,
        "problem_clusters": _problem_clusters(
            target_evaluations,
            max_gap_frames=int(max_problem_gap_frames),
        ),
        "interpretation": {
            "covered": (
                "At least one strong HUMAN anchor, or a consistent pair of HUMANs, "
                "represents the target CameraState conservatively."
            ),
            "uncertain": (
                "The target is near the atlas boundary; it may calibrate but should "
                "not count as strict coverage yet."
            ),
            "uncovered": (
                "Visual evidence exists, but current HUMAN anchors do not safely "
                "represent this target state. This is a candidate for a new HUMAN."
            ),
            "unobservable": (
                "Visual matching itself is insufficient. Do not interpret this "
                "automatically as a missing CameraState."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Payload wrappers
# ---------------------------------------------------------------------------


def evaluate_target_camera_state_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CameraStateCoverageError("Payload must be an object.")

    return evaluate_target_camera_state(
        payload.get("candidates"),
        target_frame_index=payload.get("target_frame_index"),
        target_timestamp_seconds=payload.get("target_timestamp_seconds"),
        derive_predictions=bool(payload.get("derive_predictions", True)),
    )


def summarize_camera_state_coverage_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CameraStateCoverageError("Payload must be an object.")

    return summarize_camera_state_coverage(
        payload.get("target_evaluations"),
        max_problem_gap_frames=int(
            payload.get("max_problem_gap_frames", 90)
        ),
    )


# ---------------------------------------------------------------------------
# Synthetic self-check
# ---------------------------------------------------------------------------


def _synthetic_descriptor(
    center_x: float,
    center_y: float,
    scale: float,
    *,
    anisotropy: float = 1.05,
    angle: float = 0.0,
) -> dict[str, Any]:
    return {
        "coverage": {
            "view_center_normalized": [center_x, center_y],
        },
        "effective_view_scale": {
            "scale_geometric_px_per_m": scale,
            "anisotropy_ratio": anisotropy,
            "axis_x_angle_deg": angle,
        },
    }


def _synthetic_probe(
    *,
    shift_x: float,
    shift_y: float,
    relative_scale: float,
    rotation_deg: float = 0.0,
    anisotropy: float = 1.03,
    projective_x: float = 0.02,
    projective_y: float = 0.01,
    inliers: int = 100,
    inlier_ratio: float = 0.85,
    spatial_coverage: float = 0.20,
    p95_fraction: float = 0.0015,
    mutual_matches: int = 180,
    model: str = "homography",
) -> dict[str, Any]:
    transform = {
        "center_shift_normalized": [shift_x, shift_y],
        "scale_geometric": relative_scale,
        "log_scale": math.log(max(relative_scale, EPS)),
        "rotation_deg": rotation_deg,
        "anisotropy_ratio": anisotropy,
        "projective_terms_per_diagonal": [
            projective_x,
            projective_y,
        ],
    }
    p95_px = p95_fraction * math.hypot(1920, 1080)
    return {
        "image_width": 1920,
        "image_height": 1080,
        "consensus_inlier_count": inliers,
        "consensus_inlier_ratio": inlier_ratio,
        "source_spatial_coverage": spatial_coverage,
        "target_spatial_coverage": spatial_coverage,
        "mutual_match_count": mutual_matches,
        "recommended_model": model,
        "models": {
            model: {
                "matrix": [
                    [relative_scale, 0.0, shift_x * 1920.0],
                    [0.0, relative_scale, shift_y * 1080.0],
                    [0.0, 0.0, 1.0],
                ],
                "residual_summary": {
                    "p95_px": p95_px,
                    "p95_diagonal_fraction": p95_fraction,
                },
                "transform": transform,
            }
        },
        "warnings": [],
    }


def _self_check() -> None:
    anchor_a = _synthetic_descriptor(0.50, 0.42, 9.0)
    anchor_b = _synthetic_descriptor(0.78, 0.50, 25.0)

    # 1) Near one HUMAN: strong single-anchor coverage.
    near_target = _synthetic_descriptor(0.55, 0.43, 10.2)
    near = evaluate_target_camera_state(
        [
            {
                "anchor_id": "A",
                "anchor_descriptor": anchor_a,
                "predicted_descriptor": near_target,
                "probe": _synthetic_probe(
                    shift_x=0.05,
                    shift_y=0.01,
                    relative_scale=1.12,
                    model="similarity",
                ),
            },
            {
                "anchor_id": "B",
                "anchor_descriptor": anchor_b,
                "predicted_descriptor": near_target,
                "probe": _synthetic_probe(
                    shift_x=-0.22,
                    shift_y=-0.05,
                    relative_scale=0.41,
                    rotation_deg=3.0,
                    projective_x=0.12,
                    projective_y=0.08,
                    model="homography",
                ),
            },
        ],
        target_frame_index=100,
        derive_predictions=False,
    )
    if near["status"] != "covered" or near["best_anchor_id"] != "A":
        raise AssertionError(("near", near["status"], near["best_anchor_id"]))

    # 2) Far from every anchor but still visually matchable -> uncovered.
    far_target = _synthetic_descriptor(0.05, 0.85, 52.0, anisotropy=1.45, angle=15.0)
    far = evaluate_target_camera_state(
        [
            {
                "anchor_id": "A",
                "anchor_descriptor": anchor_a,
                "predicted_descriptor": far_target,
                "probe": _synthetic_probe(
                    shift_x=-0.44,
                    shift_y=0.38,
                    relative_scale=4.8,
                    rotation_deg=15.0,
                    anisotropy=1.45,
                    projective_x=0.42,
                    projective_y=0.25,
                    model="homography",
                ),
            },
            {
                "anchor_id": "B",
                "anchor_descriptor": anchor_b,
                "predicted_descriptor": far_target,
                "probe": _synthetic_probe(
                    shift_x=-0.56,
                    shift_y=0.31,
                    relative_scale=2.1,
                    rotation_deg=12.0,
                    anisotropy=1.35,
                    projective_x=0.36,
                    projective_y=0.22,
                    model="homography",
                ),
            },
        ],
        target_frame_index=200,
        derive_predictions=False,
    )
    if far["status"] != "uncovered":
        raise AssertionError(("far", far["status"]))

    # 3) Poor visual evidence is unobservable, not automatically uncovered.
    weak = evaluate_target_camera_state(
        [
            {
                "anchor_id": "A",
                "anchor_descriptor": anchor_a,
                "predicted_descriptor": near_target,
                "probe": _synthetic_probe(
                    shift_x=0.03,
                    shift_y=0.01,
                    relative_scale=1.05,
                    inliers=13,
                    inlier_ratio=0.28,
                    spatial_coverage=0.025,
                    p95_fraction=0.0078,
                    mutual_matches=18,
                    model="homography",
                ),
            }
        ],
        target_frame_index=300,
        derive_predictions=False,
    )
    if weak["status"] != "unobservable":
        raise AssertionError(("weak", weak["status"]))

    # 4) A target between two anchors: each relation is moderately distant,
    # but both predict the same target state -> multi-anchor coverage.
    between_prediction_a = _synthetic_descriptor(0.65, 0.46, 16.2, anisotropy=1.08, angle=1.5)
    between_prediction_b = _synthetic_descriptor(0.66, 0.45, 16.0, anisotropy=1.07, angle=1.8)

    between = evaluate_target_camera_state(
        [
            {
                "anchor_id": "A",
                "anchor_descriptor": anchor_a,
                "predicted_descriptor": between_prediction_a,
                "probe": _synthetic_probe(
                    shift_x=0.45,
                    shift_y=0.10,
                    relative_scale=3.60,
                    rotation_deg=4.0,
                    anisotropy=1.15,
                    projective_x=0.36,
                    projective_y=0.10,
                    inliers=85,
                    inlier_ratio=0.78,
                    spatial_coverage=0.15,
                    p95_fraction=0.0022,
                    mutual_matches=140,
                    model="homography",
                ),
            },
            {
                "anchor_id": "B",
                "anchor_descriptor": anchor_b,
                "predicted_descriptor": between_prediction_b,
                "probe": _synthetic_probe(
                    shift_x=-0.44,
                    shift_y=-0.09,
                    relative_scale=0.30,
                    rotation_deg=3.5,
                    anisotropy=1.12,
                    projective_x=0.36,
                    projective_y=0.09,
                    inliers=90,
                    inlier_ratio=0.80,
                    spatial_coverage=0.16,
                    p95_fraction=0.0020,
                    mutual_matches=150,
                    model="homography",
                ),
            },
        ],
        target_frame_index=400,
        derive_predictions=False,
    )
    if between["status"] != "covered":
        raise AssertionError(("between", between["status"]))
    if between["coverage_mode"] != "multi_anchor_consensus":
        raise AssertionError(("between mode", between["coverage_mode"]))

    summary = summarize_camera_state_coverage(
        [near, far, weak, between],
        max_problem_gap_frames=120,
    )

    if summary["status_counts"]["covered"] != 2:
        raise AssertionError(summary["status_counts"])
    if summary["status_counts"]["uncovered"] != 1:
        raise AssertionError(summary["status_counts"])
    if summary["status_counts"]["unobservable"] != 1:
        raise AssertionError(summary["status_counts"])

    # Observable samples are near/far/between = 3, two are strictly covered.
    expected = 100.0 * 2.0 / 3.0
    if abs(summary["strict_camera_state_coverage_percent"] - expected) > 1e-6:
        raise AssertionError(summary["strict_camera_state_coverage_percent"])

    print("Camera-State coverage V1 self-check: OK")
    print(
        "Near:      "
        f"{near['status']} via {near['coverage_mode']} · "
        f"best {near['best_anchor_id']} · "
        f"score {near['best_candidate']['candidate_score']:.3f} · "
        f"distance {near['best_candidate']['state_distance']:.3f}"
    )
    print(
        "Far:       "
        f"{far['status']} · "
        f"best {far['best_anchor_id']} · "
        f"distance {far['best_candidate']['state_distance']:.3f}"
    )
    print(
        "Weak:      "
        f"{weak['status']} · "
        f"evidence {weak['best_candidate']['evidence_score']:.3f}"
    )
    print(
        "Between:   "
        f"{between['status']} via {between['coverage_mode']} · "
        f"best {between['best_anchor_id']} · "
        f"consensus disagreement "
        f"{(between['consensus'] or {}).get('prediction_disagreement', float('nan')):.3f}"
    )
    print(
        "Summary:   "
        f"strict observable coverage "
        f"{summary['strict_camera_state_coverage_percent']:.1f}% · "
        f"all-sample strict "
        f"{summary['all_sample_strict_coverage_percent']:.1f}%"
    )
    print(
        "Statuses:  "
        f"{summary['status_counts']}"
    )


if __name__ == "__main__":
    _self_check()

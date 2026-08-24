"""Phase-1 FIELD physical pitch setup.

This module orchestrates the algorithmic part of the first HUMAN phase:

raw FIELD view scan
    -> select geometry-rich measurement frames
    -> HUMAN semantic observations
    -> staged physical pitch estimation
    -> decide whether more evidence is required

It deliberately does NOT implement Camera State, Camera Atlas, propagation,
refinement, PEOPLE, semantic recognition, or a Veo warp model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

from supervisor.field_pitch_estimation import (
    PitchEstimationConfig,
    estimate_pitch_dimensions,
    summarize_pitch_estimation,
)


@dataclass(frozen=True)
class PitchSetupConfig:
    """Deterministic Phase-1 search and acceptance policy."""

    reference_length_m: float = 105.0

    length_min_m: float = 100.0
    length_max_m: float = 110.0
    width_min_m: float = 64.0
    width_max_m: float = 75.0

    coarse_step_m: float = 1.0
    fine_step_m: float = 0.1
    refinement_radius_m: float = 1.0

    initial_measurement_frames: int = 4
    maximum_measurement_frames: int = 6
    minimum_usable_frames: int = 3


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    return result if math.isfinite(result) else default


def _measurement_utility(
    recommendation: dict[str, Any],
) -> float:
    """Score one scanner-selected camera state for HUMAN measurement.

    The FIELD view scanner already owns camera/view diversity. We therefore
    rank those already-distinct states only by how useful their visible pitch
    geometry appears for manual calibration.
    """

    geometry = recommendation.get("geometry")
    if not isinstance(geometry, dict):
        geometry = {}

    quality = min(
        1.0,
        max(0.0, _finite(recommendation.get("quality_score"))),
    )

    line_families = min(
        1.0,
        max(
            0.0,
            _finite(geometry.get("line_family_count")) / 3.0,
        ),
    )

    intersections = min(
        1.0,
        max(
            0.0,
            _finite(geometry.get("intersection_count")) / 8.0,
        ),
    )

    line_count = min(
        1.0,
        max(
            0.0,
            _finite(geometry.get("line_count")) / 10.0,
        ),
    )

    sharpness = min(
        1.0,
        max(0.0, _finite(geometry.get("sharpness"))),
    )

    return float(
        0.50 * quality
        + 0.20 * line_families
        + 0.15 * intersections
        + 0.10 * line_count
        + 0.05 * sharpness
    )


def _rank_measurement_recommendations(
    scan_result: dict[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(scan_result, dict):
        return []

    recommendations = scan_result.get("recommendations", [])
    if not isinstance(recommendations, list):
        return []

    ranked: list[dict[str, Any]] = []
    seen_frames: set[int] = set()

    for raw in recommendations:
        if not isinstance(raw, dict):
            continue

        try:
            frame = int(raw["frame"])
        except (KeyError, TypeError, ValueError):
            continue

        if frame < 0 or frame in seen_frames:
            continue

        seen_frames.add(frame)

        candidate = dict(raw)
        candidate["frame"] = frame
        candidate["measurement_utility"] = _measurement_utility(raw)

        ranked.append(candidate)

    ranked.sort(
        key=lambda item: (
            -float(item["measurement_utility"]),
            -_finite(item.get("quality_score")),
            int(item["frame"]),
        )
    )

    return ranked


def select_pitch_measurement_frames(
    scan_result: dict[str, Any],
    *,
    target_count: int = 4,
    maximum_count: int = 6,
) -> dict[str, Any]:
    """Create the initial Phase-1 HUMAN measurement queue."""

    maximum = max(0, int(maximum_count))
    target = min(max(0, int(target_count)), maximum)

    eligible = _rank_measurement_recommendations(scan_result)[:maximum]

    selected = eligible[:target]
    remaining = eligible[target:]

    # HUMAN works through the selected queue in temporal order even though
    # selection itself is quality-driven.
    selected = sorted(
        selected,
        key=lambda item: int(item["frame"]),
    )

    return {
        "status": "ok" if selected else "no_measurement_frames",
        "target_count": target,
        "maximum_count": maximum,
        "selected_frames": [item["frame"] for item in selected],
        "selected": selected,
        "remaining_frames": [item["frame"] for item in remaining],
        "remaining": remaining,
        "policy": {
            "source": "field_view_scanner_recommendations",
            "scanner_owns_view_diversity": True,
            "measurement_utility": {
                "quality_score": 0.50,
                "line_family_count": 0.20,
                "intersection_count": 0.15,
                "line_count": 0.10,
                "sharpness": 0.05,
            },
        },
    }


def next_pitch_measurement_frame(
    scan_result: dict[str, Any],
    already_used_frames: Iterable[int],
    *,
    maximum_count: int = 6,
) -> dict[str, Any] | None:
    """Return the strongest still-unused Phase-1 measurement state."""

    maximum = max(0, int(maximum_count))

    used: set[int] = set()

    for value in already_used_frames:
        try:
            used.add(int(value))
        except (TypeError, ValueError):
            continue

    if len(used) >= maximum:
        return None

    candidates = _rank_measurement_recommendations(scan_result)[:maximum]

    for candidate in candidates:
        if candidate["frame"] not in used:
            return candidate

    return None


def _estimation_config(
    *,
    length_min_m: float,
    length_max_m: float,
    width_min_m: float,
    width_max_m: float,
    setup: PitchSetupConfig,
    coarse_step_m: float | None = None,
    fine_radius_m: float | None = None,
) -> PitchEstimationConfig:
    return PitchEstimationConfig(
        length_min_m=float(length_min_m),
        length_max_m=float(length_max_m),
        width_min_m=float(width_min_m),
        width_max_m=float(width_max_m),
        coarse_step_m=float(
            setup.coarse_step_m
            if coarse_step_m is None
            else coarse_step_m
        ),
        fine_radius_m=float(
            setup.refinement_radius_m
            if fine_radius_m is None
            else fine_radius_m
        ),
        fine_step_m=float(setup.fine_step_m),
    )


def _result_pitch(
    result: dict[str, Any],
) -> tuple[float, float] | None:
    if not isinstance(result, dict):
        return None

    model = result.get("pitch_model")
    if not isinstance(model, dict):
        return None

    length = _finite(
        model.get("length_m"),
        float("nan"),
    )
    width = _finite(
        model.get("width_m"),
        float("nan"),
    )

    if not (
        math.isfinite(length)
        and math.isfinite(width)
    ):
        return None

    return float(length), float(width)


def _forced_model_diagnostics(
    observations: list[dict[str, Any]],
    *,
    length_m: float,
    width_m: float,
    setup: PitchSetupConfig,
) -> dict[str, Any]:
    """Evaluate exactly one physical pitch model.

    Confidence flags from this call must NOT be used because a one-candidate
    profile is boundary-limited by construction.
    """

    config = _estimation_config(
        length_min_m=length_m,
        length_max_m=length_m,
        width_min_m=width_m,
        width_max_m=width_m,
        setup=setup,
        coarse_step_m=max(
            setup.fine_step_m,
            1e-6,
        ),
        fine_radius_m=0.0,
    )

    return estimate_pitch_dimensions(
        observations,
        config=config,
    )


def _evidence_gaps(
    final_result: dict[str, Any],
) -> list[str]:
    observability = final_result.get(
        "observability",
        {},
    )

    if not isinstance(observability, dict):
        return [
            "touchline",
            "goal_line_structure",
            "centre_or_halfway",
        ]

    gaps: list[str] = []

    if not observability.get("reaches_touchline"):
        gaps.append("touchline")

    if not observability.get("reaches_goal_line_structure"):
        gaps.append("goal_line_structure")

    if not observability.get("includes_centre_or_halfway"):
        gaps.append("centre_or_halfway")

    return gaps


def estimate_pitch_setup(
    observations: list[dict[str, Any]],
    *,
    config: PitchSetupConfig | None = None,
) -> dict[str, Any]:
    """Run staged physical-pitch estimation for Phase 1.

    Coordinate-descent V1:

        W @ reference L
        -> L @ best W
        -> W @ best L

    The two width profiles cover the full configured width range. The final
    width confidence is taken from the third stage, after length has been
    updated, rather than from the initial width profile at reference length.

    No Cartesian L x W fine grid is evaluated.
    """

    setup = config or PitchSetupConfig()

    observations = [
        item
        for item in observations
        if isinstance(item, dict)
    ]

    if not observations:
        return {
            "status": "failed",
            "detail": (
                "No se recibieron observaciones HUMAN "
                "para calibrar el terreno."
            ),
            "pitch_model": None,
            "confidence": {},
            "profiles": {},
            "diagnostics": {},
            "needs_more_evidence": {
                "length": True,
                "width": True,
                "usable_frames": True,
            },
            "config": asdict(setup),
        }

    # ------------------------------------------------------------
    # Stage 1: initial width profile at the reference length
    # ------------------------------------------------------------

    width_profile = estimate_pitch_dimensions(
        observations,
        config=_estimation_config(
            length_min_m=setup.reference_length_m,
            length_max_m=setup.reference_length_m,
            width_min_m=setup.width_min_m,
            width_max_m=setup.width_max_m,
            setup=setup,
        ),
    )

    width_pitch = _result_pitch(width_profile)

    if width_pitch is None:
        return {
            "status": "failed",
            "detail": (
                "No se pudo resolver el perfil "
                "inicial de anchura."
            ),
            "pitch_model": None,
            "confidence": {},
            "profiles": {
                "initial_width": summarize_pitch_estimation(
                    width_profile
                ),
            },
            "diagnostics": {},
            "needs_more_evidence": {
                "length": True,
                "width": True,
                "usable_frames": True,
            },
            "config": asdict(setup),
        }

    _, initial_width = width_pitch

    # ------------------------------------------------------------
    # Stage 2: full length profile at the Stage-1 width
    # ------------------------------------------------------------

    length_profile = estimate_pitch_dimensions(
        observations,
        config=_estimation_config(
            length_min_m=setup.length_min_m,
            length_max_m=setup.length_max_m,
            width_min_m=initial_width,
            width_max_m=initial_width,
            setup=setup,
        ),
    )

    length_pitch = _result_pitch(length_profile)

    if length_pitch is None:
        return {
            "status": "failed",
            "detail": (
                "No se pudo resolver el perfil "
                "de longitud."
            ),
            "pitch_model": None,
            "confidence": {},
            "profiles": {
                "initial_width": summarize_pitch_estimation(
                    width_profile
                ),
                "length": summarize_pitch_estimation(
                    length_profile
                ),
            },
            "diagnostics": {},
            "needs_more_evidence": {
                "length": True,
                "width": True,
                "usable_frames": True,
            },
            "config": asdict(setup),
        }

    estimated_length, _ = length_pitch

    # ------------------------------------------------------------
    # Stage 3: FINAL full width profile at the updated length
    # ------------------------------------------------------------

    refined_width_profile = estimate_pitch_dimensions(
        observations,
        config=_estimation_config(
            length_min_m=estimated_length,
            length_max_m=estimated_length,
            width_min_m=setup.width_min_m,
            width_max_m=setup.width_max_m,
            setup=setup,
        ),
    )

    refined_pitch = _result_pitch(refined_width_profile)

    if refined_pitch is None:
        return {
            "status": "failed",
            "detail": (
                "No se pudo resolver el perfil "
                "final de anchura."
            ),
            "pitch_model": None,
            "confidence": {},
            "profiles": {
                "initial_width": summarize_pitch_estimation(
                    width_profile
                ),
                "length": summarize_pitch_estimation(
                    length_profile
                ),
                "refined_width": summarize_pitch_estimation(
                    refined_width_profile
                ),
            },
            "diagnostics": {},
            "needs_more_evidence": {
                "length": True,
                "width": True,
                "usable_frames": True,
            },
            "config": asdict(setup),
        }

    _, estimated_width = refined_pitch

    # ------------------------------------------------------------
    # Exact final-model diagnostics
    # ------------------------------------------------------------

    final_result = _forced_model_diagnostics(
        observations,
        length_m=estimated_length,
        width_m=estimated_width,
        setup=setup,
    )

    # Length confidence comes from Stage 2, where length was free over the
    # complete configured interval.
    length_confidence = length_profile.get(
        "confidence",
        {},
    )

    # Width confidence comes from Stage 3, where width was re-estimated over
    # the complete configured interval after updating the length estimate.
    # Stage 1 is only an initializer and must not decide final width confidence.
    width_confidence = refined_width_profile.get(
        "confidence",
        {},
    )

    final_observability = final_result.get(
        "observability",
        {},
    )

    usable_frames = int(
        final_observability.get(
            "frames_usable",
            0,
        )
        or 0
    )

    width_constrained = bool(
        width_confidence.get(
            "width_constrained"
        )
    )

    length_constrained = bool(
        length_confidence.get(
            "length_constrained"
        )
    )

    enough_frames = (
        usable_frames
        >= setup.minimum_usable_frames
    )

    needs_more_evidence = {
        "length": not length_constrained,
        "width": not width_constrained,
        "usable_frames": not enough_frames,
    }

    status = (
        "ready_to_confirm"
        if not any(
            needs_more_evidence.values()
        )
        else "needs_more_evidence"
    )

    return {
        "status": status,
        "pitch_model": {
            "length_m": float(
                estimated_length
            ),
            "width_m": float(
                estimated_width
            ),
            "status": "estimated",
        },
        "confidence": {
            "length_interval_m": length_confidence.get(
                "length_interval_m"
            ),
            "width_interval_m": width_confidence.get(
                "width_interval_m"
            ),
            "length_constrained": (
                length_constrained
            ),
            "width_constrained": (
                width_constrained
            ),
            "length_boundary_limited": (
                length_confidence.get(
                    "boundary_limited",
                    {},
                ).get("length")
            ),
            "width_boundary_limited": (
                width_confidence.get(
                    "boundary_limited",
                    {},
                ).get("width")
            ),
        },
        "profiles": {
            "initial_width": width_profile,
            "length": length_profile,
            "refined_width": refined_width_profile,
        },
        "diagnostics": {
            "final_model": final_result,
            "final_summary": summarize_pitch_estimation(
                final_result
            ),
            "evidence_gaps": _evidence_gaps(
                final_result
            ),
        },
        "needs_more_evidence": (
            needs_more_evidence
        ),
        "config": asdict(setup),
    }


def summarize_pitch_setup(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Compact Phase-1 result intended for the future UI."""

    model = result.get("pitch_model")
    confidence = result.get(
        "confidence",
        {},
    )

    diagnostics = result.get(
        "diagnostics",
        {},
    )

    return {
        "status": result.get("status"),
        "pitch_model": model,
        "length_interval_m": confidence.get(
            "length_interval_m"
        ),
        "width_interval_m": confidence.get(
            "width_interval_m"
        ),
        "length_constrained": confidence.get(
            "length_constrained"
        ),
        "width_constrained": confidence.get(
            "width_constrained"
        ),
        "needs_more_evidence": result.get(
            "needs_more_evidence",
            {},
        ),
        "evidence_gaps": diagnostics.get(
            "evidence_gaps",
            [],
        ),
        "final_summary": diagnostics.get(
            "final_summary",
            {},
        ),
    }

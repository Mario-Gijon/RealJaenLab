"""Thin orchestration for recovering rejected PARTIDO camera calibrations.

This module deliberately owns workflow policy only.  The validated projective
state and refinement mathematics remain in their dedicated core modules.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from supervisor.field_camera_state_projective import (
    ProjectiveCameraStateConfig,
    build_projective_camera_state_model,
)
from supervisor.field_camera_state_recovery import (
    recover_camera_from_projective_state,
)


def _frame_values(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    frames: set[int] = set()
    for item in value:
        try:
            frame = int(item)
        except (TypeError, ValueError):
            continue
        if frame >= 0:
            frames.add(frame)
    return sorted(frames)


def _manual_match_calibration(
    calibrations: list[dict[str, Any]], frame: int
) -> dict[str, Any] | None:
    for calibration in calibrations:
        try:
            same_frame = int(calibration.get("frame")) == frame
        except (AttributeError, TypeError, ValueError):
            continue
        if (
            same_frame
            and calibration.get("source") == "manual"
            and calibration.get("purpose") == "match_setup"
        ):
            return calibration
    return None


def _constraints_by_type(
    calibration: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        "point": [], "line": [], "arc": []
    }
    for constraint in calibration.get("constraints", []):
        if isinstance(constraint, dict) and constraint.get("type") in grouped:
            grouped[constraint["type"]].append(constraint)
    return grouped["point"], grouped["line"], grouped["arc"]


def _unrecovered(frame: int, reason: str, **details: Any) -> dict[str, Any]:
    return {"frame": frame, "reason": reason, **details}


def recover_match_setup_cameras(
    calibrations: list[dict[str, Any]],
    match_setup: dict[str, Any],
    pitch_model: dict[str, Any],
    image_width: int,
    image_height: int,
    video_name: str | None = None,
) -> dict[str, Any]:
    """Recover bracketed rejected PARTIDO frames from 5-D projective state.

    Inputs are read-only.  Returned derived cameras are intentionally separate
    from manual calibrations and never become training anchors.
    """

    if not isinstance(calibrations, list) or not isinstance(match_setup, dict):
        return {"version": 1, "status": "invalid_input", "dimensions": 5,
                "anchor_frames": [], "derived_frames": [], "unrecovered_frames": []}
    try:
        width = int(image_width)
        height = int(image_height)
        length_m = float(pitch_model["length_m"])
        pitch_width_m = float(pitch_model["width_m"])
    except (KeyError, TypeError, ValueError):
        return {"version": 1, "status": "invalid_input", "dimensions": 5,
                "anchor_frames": [], "derived_frames": [], "unrecovered_frames": []}
    if width <= 0 or height <= 0 or length_m <= 0 or pitch_width_m <= 0:
        return {"version": 1, "status": "invalid_input", "dimensions": 5,
                "anchor_frames": [], "derived_frames": [], "unrecovered_frames": []}

    manual_calibrations = [item for item in calibrations if isinstance(item, dict)]
    completed = _frame_values(match_setup.get("completed_frames"))
    rejected = _frame_values(match_setup.get("rejected_frames"))
    model = build_projective_camera_state_model(
        manual_calibrations,
        completed,
        image_width=width,
        image_height=height,
        pitch_length_m=length_m,
        pitch_width_m=pitch_width_m,
        config=ProjectiveCameraStateConfig(dimensions=5),
    )

    if model.get("status") != "ok":
        return {
            "version": 1,
            "status": model.get("status", "model_failed"),
            "dimensions": 5,
            "anchor_frames": model.get("anchor_frames", []),
            "derived_frames": [],
            "unrecovered_frames": [
                _unrecovered(frame, "projective_state_unavailable")
                for frame in rejected
            ],
            "video": video_name,
        }

    anchors = {
        int(anchor["frame"]): np.asarray(anchor["state"], dtype=np.float64)
        for anchor in model.get("anchors", [])
        if isinstance(anchor, dict)
    }
    anchor_frames = sorted(anchors)
    derived: list[dict[str, Any]] = []
    unrecovered: list[dict[str, Any]] = []

    for frame in rejected:
        calibration = _manual_match_calibration(manual_calibrations, frame)
        if calibration is None:
            unrecovered.append(_unrecovered(frame, "no_saved_manual_geometry"))
            continue
        points, lines, arcs = _constraints_by_type(calibration)
        if not (points or lines or arcs):
            unrecovered.append(_unrecovered(frame, "no_saved_manual_geometry"))
            continue
        left = max((anchor for anchor in anchor_frames if anchor < frame), default=None)
        right = min((anchor for anchor in anchor_frames if anchor > frame), default=None)
        if left is None or right is None:
            unrecovered.append(_unrecovered(frame, "not_temporally_bracketed",
                                           left_anchor_frame=left, right_anchor_frame=right))
            continue
        alpha = (frame - left) / float(right - left)
        prior = (1.0 - alpha) * anchors[left] + alpha * anchors[right]
        recovery = recover_camera_from_projective_state(
            model, prior, points, lines, arcs, prior_confidence="good"
        )
        if recovery.get("status") != "ok" or not isinstance(recovery.get("homography"), list):
            unrecovered.append(_unrecovered(
                frame, "recovery_rejected", left_anchor_frame=left,
                right_anchor_frame=right, recovery=recovery.get("recovery"),
                confidence=recovery.get("confidence"),
            ))
            continue
        derived.append({
            "frame": frame,
            "homography": recovery["homography"],
            "source": "camera_state_refined",
            "purpose": "derived_camera",
            "confidence": recovery.get("confidence"),
            "left_anchor_frame": left,
            "right_anchor_frame": right,
            "alpha": float(alpha),
            "state_shift_z_rms": recovery.get("state_shift_z_rms"),
            "fit_rms_sigma": recovery.get("fit_rms_sigma"),
            "recovery": recovery.get("recovery", {}),
        })

    return {
        "version": 1,
        "status": "ok",
        "dimensions": 5,
        "anchor_frames": anchor_frames,
        "derived_frames": derived,
        "unrecovered_frames": unrecovered,
        "video": video_name,
    }

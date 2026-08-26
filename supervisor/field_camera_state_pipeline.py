"""Workflow orchestration for derived PARTIDO camera states.

This module deliberately contains no camera mathematics. Strong HUMAN
``match_setup`` anchors are joined with the validated planar-flow/bridge/fusion
modules, while the existing 5-D recovery stays available only for rejected
frames containing useful partial HUMAN geometry.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from supervisor.field_camera_state_planar_bridge import (
    PlanarBridgeConfig,
    bridge_correct_homography,
    corrected_path_result,
)
from supervisor.field_camera_state_planar_bridge_fusion import (
    PlanarBridgeFusionConfig,
    fuse_corrected_planar_paths,
)
from supervisor.field_camera_state_planar_flow import (
    PlanarFlowConfig,
    _normalise_h,
    estimate_ground_plane_step,
    homography_disagreement_px,
)
from supervisor.field_camera_state_planar_selection import PlanarSelectionConfig
from supervisor.field_camera_state_projective import (
    ProjectiveCameraStateConfig,
    build_projective_camera_state_model,
)
from supervisor.field_camera_state_recovery import recover_camera_from_projective_state


CAMERA_STATE_VERSION = 2
CAMERA_STATE_ALGORITHM = "camera_ht_v1"


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


def _strong_match_anchors(
    calibrations: list[dict[str, Any]], completed_frames: list[int]
) -> list[tuple[int, dict[str, Any]]]:
    anchors: list[tuple[int, dict[str, Any]]] = []
    for frame in completed_frames:
        calibration = _manual_match_calibration(calibrations, frame)
        if (
            calibration is not None
            and isinstance(calibration.get("homography"), list)
            and len(calibration["homography"]) == 3
        ):
            anchors.append((frame, calibration))
    return anchors


def _constraints_by_type(
    calibration: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {"point": [], "line": [], "arc": []}
    for constraint in calibration.get("constraints", []):
        if isinstance(constraint, dict) and constraint.get("type") in grouped:
            grouped[constraint["type"]].append(constraint)
    return grouped["point"], grouped["line"], grouped["arc"]


def _unrecovered(frame: int, reason: str, **details: Any) -> dict[str, Any]:
    return {"frame": frame, "reason": reason, **details}


def _flow_quality(path: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(path, dict):
        return None
    return {
        "step_count": path.get("step_count"),
        "median_inlier_ratio": path.get("median_inlier_ratio"),
        "median_reprojection_px": path.get("median_reprojection_px"),
    }


class _IntervalFrameCache:
    """Read each decoded source frame at most once for one anchor interval."""

    def __init__(self, video_path: str | Path):
        self._path = Path(video_path)
        self._capture = cv2.VideoCapture(str(self._path))
        self._frames: dict[int, np.ndarray | None] = {}

    @property
    def opened(self) -> bool:
        return self._capture.isOpened()

    def frame(self, frame_index: int) -> np.ndarray | None:
        if frame_index not in self._frames:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = self._capture.read()
            self._frames[frame_index] = frame if ok and frame is not None else None
        return self._frames[frame_index]

    def close(self) -> None:
        self._capture.release()


def _flow_success_result(
    start_frame: int,
    end_frame: int,
    homography: np.ndarray,
    steps: list[dict[str, Any]],
    config: PlanarFlowConfig,
) -> dict[str, Any]:
    """Mirror ``propagate_pitch_homography`` aggregate diagnostics exactly."""

    inlier_ratios = [
        float(step["inlier_ratio"])
        for step in steps
        if step.get("status") == "ok" and step.get("inlier_ratio") is not None
    ]
    reprojections = [
        float(step["median_reprojection_px"])
        for step in steps
        if step.get("status") == "ok" and step.get("median_reprojection_px") is not None
    ]
    tracks = [
        int(step["inlier_count"])
        for step in steps
        if step.get("status") == "ok" and step.get("inlier_count") is not None
    ]
    return {
        "status": "ok",
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "frame_span": abs(int(end_frame) - int(start_frame)),
        "step_count": len(steps),
        "homography": homography.tolist(),
        "median_inlier_ratio": float(np.median(inlier_ratios)) if inlier_ratios else None,
        "minimum_inlier_ratio": float(np.min(inlier_ratios)) if inlier_ratios else None,
        "median_reprojection_px": float(np.median(reprojections)) if reprojections else None,
        "median_inlier_count": float(np.median(tracks)) if tracks else None,
        "steps": steps,
        "config": asdict(config),
    }


class _CachedDirectionalFlow:
    """Cache the exact 2-frame prefixes used by ``propagate_pitch_homography``."""

    def __init__(
        self,
        frames: _IntervalFrameCache,
        start_frame: int,
        end_frame: int,
        start_homography: Any,
        *,
        pitch_length_m: float,
        pitch_width_m: float,
        config: PlanarFlowConfig,
    ):
        self._frames = frames
        self._start = int(start_frame)
        self._end = int(end_frame)
        self._direction = 1 if self._end >= self._start else -1
        self._pitch_length_m = pitch_length_m
        self._pitch_width_m = pitch_width_m
        self._config = config
        self._prefix: dict[int, dict[str, Any]] = {}
        self._failure: dict[str, Any] | None = None
        self._initial_failure: dict[str, Any] | None = None

        h = _normalise_h(start_homography)
        if h is None:
            self._initial_failure = {"status": "invalid_start_homography"}
            return
        if not frames.opened:
            self._initial_failure = {"status": "video_open_failed", "video": str(frames._path)}
            return
        if frames.frame(self._start) is None:
            self._initial_failure = {"status": "frame_read_failed", "frame": self._start}
            return
        self._prefix[self._start] = {"homography": h, "steps": []}
        self._build_even_prefixes()

    def _build_even_prefixes(self) -> None:
        distance = abs(self._end - self._start)
        last_even = self._start + self._direction * (distance // 2 * 2)
        current = self._start
        while current != last_even and self._failure is None:
            self._extend(current, current + self._direction * 2, cache_prefix=True)
            current += self._direction * 2

    def _extend(
        self,
        previous_index: int,
        current_index: int,
        *,
        cache_prefix: bool,
    ) -> dict[str, Any]:
        previous = self._prefix.get(previous_index)
        if previous is None:
            return self._failure or {"status": "path_failed"}
        current_frame = self._frames.frame(current_index)
        if current_frame is None:
            result = {
                "status": "frame_read_failed",
                "frame": current_index,
                "steps": previous["steps"],
            }
            if cache_prefix:
                self._failure = result
            return result
        step = estimate_ground_plane_step(
            self._frames.frame(previous_index), current_frame, previous["homography"],
            pitch_length_m=self._pitch_length_m,
            pitch_width_m=self._pitch_width_m,
            config=self._config,
        )
        step_record = {
            "from_frame": int(previous_index),
            "to_frame": int(current_index),
            **{key: value for key, value in step.items() if key != "matrix"},
        }
        steps = [*previous["steps"], step_record]
        if step.get("status") != "ok":
            result = {
                "status": "path_failed",
                "failed_from_frame": step_record["from_frame"],
                "failed_to_frame": step_record["to_frame"],
                "failure": step,
                "steps": steps,
                "homography": previous["homography"].tolist(),
                "config": asdict(self._config),
            }
            if cache_prefix:
                self._failure = result
            return result
        # Reuse the core module's exact normalisation instead of reimplementing
        # projective composition in orchestration.
        homography = _normalise_h(np.asarray(step["matrix"], dtype=np.float64) @ previous["homography"])
        if homography is None:
            result = {"status": "composition_failed", "steps": steps, "config": asdict(self._config)}
            if cache_prefix:
                self._failure = result
            return result
        result = {"homography": homography, "steps": steps}
        if cache_prefix:
            self._prefix[current_index] = result
        return result

    def path_to(self, target_frame: int) -> dict[str, Any]:
        if self._initial_failure is not None:
            return self._initial_failure
        target = int(target_frame)
        distance = (target - self._start) * self._direction
        total = (self._end - self._start) * self._direction
        if distance < 0 or distance > total:
            return {"status": "invalid_target_frame"}
        if distance % 2 == 0:
            prefix = self._prefix.get(target)
            if prefix is None:
                return self._failure or {"status": "path_failed"}
            return _flow_success_result(self._start, target, prefix["homography"], prefix["steps"], self._config)
        previous = target - self._direction
        prefix = self._prefix.get(previous)
        if prefix is None:
            return self._failure or {"status": "path_failed"}
        result = self._extend(previous, target, cache_prefix=False)
        if result.get("status") is not None or "homography" not in result:
            return result
        return _flow_success_result(self._start, target, result["homography"], result["steps"], self._config)


def _camera_ht_between_anchors(
    video_path: str | Path,
    left_frame: int,
    left_homography: list[list[float]],
    right_frame: int,
    right_homography: list[list[float]],
    *,
    image_width: int,
    image_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Derive all interior cameras between one pair of exact HUMAN anchors."""

    flow_config = PlanarFlowConfig(sample_step_frames=2)
    bridge_config = PlanarBridgeConfig(correction_exponent=2.0)
    selection_config = PlanarSelectionConfig(
        step_decay_tau=20.0,
        inlier_ratio_power=6.0,
        reprojection_decay_px=1.0,
    )
    fusion_config = PlanarBridgeFusionConfig(
        maximum_fusion_disagreement_px=90.0,
        reliability_power=1.0,
    )
    derived: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    span = float(right_frame - left_frame)
    frame_cache = _IntervalFrameCache(video_path)
    try:
        left_paths = _CachedDirectionalFlow(
            frame_cache, left_frame, right_frame, left_homography,
            pitch_length_m=pitch_length_m, pitch_width_m=pitch_width_m,
            config=flow_config,
        )
        right_paths = _CachedDirectionalFlow(
            frame_cache, right_frame, left_frame, right_homography,
            pitch_length_m=pitch_length_m, pitch_width_m=pitch_width_m,
            config=flow_config,
        )
        # Endpoint closures are now the cached full directional paths.
        left_closure = left_paths.path_to(right_frame)
        right_closure = right_paths.path_to(left_frame)
        for frame in range(left_frame + 1, right_frame):
            left_raw = left_paths.path_to(frame)
            right_raw = right_paths.path_to(frame)
            left_corrected = corrected_path_result(
                left_raw,
                bridge_correct_homography(
                    left_raw, left_closure, right_homography,
                    progress_to_endpoint=(frame - left_frame) / span,
                    image_width=image_width, image_height=image_height,
                    pitch_length_m=pitch_length_m, pitch_width_m=pitch_width_m,
                    config=bridge_config,
                ),
            )
            right_corrected = corrected_path_result(
                right_raw,
                bridge_correct_homography(
                    right_raw, right_closure, left_homography,
                    progress_to_endpoint=(right_frame - frame) / span,
                    image_width=image_width, image_height=image_height,
                    pitch_length_m=pitch_length_m, pitch_width_m=pitch_width_m,
                    config=bridge_config,
                ),
            )
            disagreement = homography_disagreement_px(
                left_corrected.get("homography"), right_corrected.get("homography"),
                pitch_length_m=pitch_length_m, pitch_width_m=pitch_width_m,
                image_width=image_width, image_height=image_height,
            )
            disagreement_px = disagreement.get("median_px") if disagreement.get("status") == "ok" else None
            fusion = fuse_corrected_planar_paths(
                left_corrected, right_corrected,
                disagreement_px=disagreement_px,
                image_width=image_width, image_height=image_height,
                pitch_length_m=pitch_length_m, pitch_width_m=pitch_width_m,
                selection_config=selection_config, fusion_config=fusion_config,
            )
            if fusion.get("status") != "ok" or not isinstance(fusion.get("homography"), list):
                failures.append(_unrecovered(
                    frame, "camera_ht_unavailable", left_anchor_frame=left_frame,
                    right_anchor_frame=right_frame, fusion_status=fusion.get("status"),
                ))
                continue
            selection = fusion.get("selection", {})
            derived.append({
                "frame": frame,
                "homography": fusion["homography"],
                "source": CAMERA_STATE_ALGORITHM,
                "purpose": "derived_camera",
                "method": fusion.get("mode", "selected_only"),
                "confidence": selection.get("confidence"),
                "left_anchor_frame": left_frame,
                "right_anchor_frame": right_frame,
                "corrected_disagreement_px": disagreement_px,
                "selected_side": selection.get("selected_side"),
                "fusion_weights": fusion.get("weights", {}),
                "flow_quality": {
                    "left": _flow_quality(left_corrected),
                    "right": _flow_quality(right_corrected),
                },
            })
    finally:
        frame_cache.close()
    return derived, failures


def _recover_rejected_manual_geometry(
    calibrations: list[dict[str, Any]],
    completed: list[int],
    rejected: list[int],
    *,
    image_width: int,
    image_height: int,
    pitch_length_m: float,
    pitch_width_m: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Preserve V1 partial-HUMAN recovery without making it a training path."""

    model = build_projective_camera_state_model(
        calibrations, completed,
        image_width=image_width, image_height=image_height,
        pitch_length_m=pitch_length_m, pitch_width_m=pitch_width_m,
        config=ProjectiveCameraStateConfig(dimensions=5),
    )
    if model.get("status") != "ok":
        return [], [_unrecovered(frame, "projective_state_unavailable") for frame in rejected]

    anchors = {
        int(anchor["frame"]): np.asarray(anchor["state"], dtype=np.float64)
        for anchor in model.get("anchors", []) if isinstance(anchor, dict)
    }
    anchor_frames = sorted(anchors)
    recovered: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for frame in rejected:
        calibration = _manual_match_calibration(calibrations, frame)
        if calibration is None:
            failures.append(_unrecovered(frame, "no_saved_manual_geometry"))
            continue
        points, lines, arcs = _constraints_by_type(calibration)
        if not (points or lines or arcs):
            failures.append(_unrecovered(frame, "no_saved_manual_geometry"))
            continue
        left = max((anchor for anchor in anchor_frames if anchor < frame), default=None)
        right = min((anchor for anchor in anchor_frames if anchor > frame), default=None)
        if left is None or right is None:
            failures.append(_unrecovered(frame, "not_temporally_bracketed", left_anchor_frame=left, right_anchor_frame=right))
            continue
        alpha = (frame - left) / float(right - left)
        prior = (1.0 - alpha) * anchors[left] + alpha * anchors[right]
        recovery = recover_camera_from_projective_state(model, prior, points, lines, arcs, prior_confidence="good")
        if recovery.get("status") != "ok" or not isinstance(recovery.get("homography"), list):
            failures.append(_unrecovered(
                frame, "recovery_rejected", left_anchor_frame=left,
                right_anchor_frame=right, recovery=recovery.get("recovery"),
                confidence=recovery.get("confidence"),
            ))
            continue
        recovered.append({
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
    return recovered, failures


def recover_match_setup_cameras(
    calibrations: list[dict[str, Any]],
    match_setup: dict[str, Any],
    pitch_model: dict[str, Any],
    image_width: int,
    image_height: int,
    *,
    video_path: str | Path,
    video_name: str | None = None,
) -> dict[str, Any]:
    """Build dense derived H_t cameras between strong PARTIDO anchors."""

    base = {
        "version": CAMERA_STATE_VERSION,
        "algorithm": CAMERA_STATE_ALGORITHM,
        "dimensions": 5,
        "anchor_frames": [],
        "derived_frames": [],
        "unrecovered_frames": [],
        "video": video_name,
    }
    if not isinstance(calibrations, list) or not isinstance(match_setup, dict):
        return {**base, "status": "invalid_input"}
    try:
        width = int(image_width)
        height = int(image_height)
        length_m = float(pitch_model["length_m"])
        pitch_width_m = float(pitch_model["width_m"])
    except (KeyError, TypeError, ValueError):
        return {**base, "status": "invalid_input"}
    if width <= 0 or height <= 0 or length_m <= 0 or pitch_width_m <= 0:
        return {**base, "status": "invalid_input"}

    manual_calibrations = [item for item in calibrations if isinstance(item, dict)]
    completed = _frame_values(match_setup.get("completed_frames"))
    rejected = _frame_values(match_setup.get("rejected_frames"))
    anchors = _strong_match_anchors(manual_calibrations, completed)
    anchor_frames = [frame for frame, _ in anchors]
    if len(anchors) < 2:
        return {
            **base,
            "status": "insufficient_anchors",
            "anchor_frames": anchor_frames,
            "unrecovered_frames": [_unrecovered(frame, "not_temporally_bracketed") for frame in rejected],
        }

    derived_by_frame: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for (left_frame, left), (right_frame, right) in zip(anchors, anchors[1:]):
        segment, segment_failures = _camera_ht_between_anchors(
            video_path, left_frame, left["homography"], right_frame, right["homography"],
            image_width=width, image_height=height,
            pitch_length_m=length_m, pitch_width_m=pitch_width_m,
        )
        derived_by_frame.update({entry["frame"]: entry for entry in segment})
        failures.extend(segment_failures)

    # Exact partial HUMAN recovery wins at its rejected frame, yet stays derived.
    recovered, recovery_failures = _recover_rejected_manual_geometry(
        manual_calibrations, completed, rejected,
        image_width=width, image_height=height,
        pitch_length_m=length_m, pitch_width_m=pitch_width_m,
    )
    derived_by_frame.update({entry["frame"]: entry for entry in recovered})
    failures.extend(recovery_failures)
    failures = [item for item in failures if item.get("frame") not in derived_by_frame]
    unrecovered_by_frame = {item["frame"]: item for item in failures}
    return {
        **base,
        "status": "ok",
        "anchor_frames": anchor_frames,
        "derived_frames": [derived_by_frame[frame] for frame in sorted(derived_by_frame)],
        "unrecovered_frames": [unrecovered_by_frame[frame] for frame in sorted(unrecovered_by_frame)],
    }

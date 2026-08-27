"""Camera Trajectory V1 built only from image-space camera motion.

This layer is deliberately independent of football-pitch calibration.

Input:
    continuous video

Uses:
    Global Camera Motion V1 (KLT + RANSAC projective image motion)

Produces:
    - a cumulative relative camera trajectory
    - camera keyframes selected by geometric novelty, not time
    - coarse repeated-state / loop candidates
    - coarse camera-state regions

No HUMAN H, pitch lines, field mask or Camera Atlas H is used here.

Important
---------
This is a trajectory/coverage experiment, not the final relocaliser. Long-term
composition drift can make two truly identical distant views look less similar.
Later loop-closure will validate/refine candidates with direct image matching.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from supervisor.field_camera_motion_global import (
    GlobalCameraMotionConfig,
    camera_motion_descriptor,
    estimate_global_camera_step,
)


EPS = 1e-9


@dataclass(frozen=True)
class CameraTrajectoryConfig:
    sample_step_frames: int = 2

    # A new keyframe is created only when the camera view has moved far enough
    # from the most recent keyframe. These are image-space geometric thresholds.
    keyframe_min_gap_frames: int = 8
    keyframe_median_displacement_px: float = 48.0
    keyframe_p90_displacement_px: float = 85.0

    # Coarse recurrence / region thresholds. These intentionally operate on the
    # accumulated trajectory only; direct image validation comes later.
    recurrence_min_gap_frames: int = 120
    recurrence_median_displacement_px: float = 32.0
    recurrence_p90_displacement_px: float = 58.0

    state_grid_x: int = 9
    state_grid_y: int = 6
    state_grid_margin_x_fraction: float = 0.08
    state_grid_margin_y_fraction: float = 0.10


def _normalise_h(value: Any) -> np.ndarray | None:
    try:
        h = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if h.shape != (3, 3) or not np.isfinite(h).all():
        return None
    if abs(float(h[2, 2])) < EPS:
        return None
    h = h / float(h[2, 2])
    try:
        condition = float(np.linalg.cond(h))
        determinant = float(np.linalg.det(h))
    except np.linalg.LinAlgError:
        return None
    if (
        not np.isfinite(condition)
        or condition > 1e14
        or not np.isfinite(determinant)
        or abs(determinant) < 1e-12
    ):
        return None
    return h


def _state_grid(
    width: int,
    height: int,
    cfg: CameraTrajectoryConfig,
) -> np.ndarray:
    xs = np.linspace(
        cfg.state_grid_margin_x_fraction * width,
        (1.0 - cfg.state_grid_margin_x_fraction) * width,
        cfg.state_grid_x,
    )
    ys = np.linspace(
        cfg.state_grid_margin_y_fraction * height,
        (1.0 - cfg.state_grid_margin_y_fraction) * height,
        cfg.state_grid_y,
    )
    return np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)


def _project(points: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.column_stack(
        (points, np.ones(len(points), dtype=np.float64))
    ) @ h.T
    den = q[:, 2]
    valid = (
        np.isfinite(q).all(axis=1)
        & np.isfinite(den)
        & (np.abs(den) > EPS)
    )
    out = np.full((len(points), 2), np.nan, dtype=np.float64)
    out[valid] = q[valid, :2] / den[valid, None]
    valid &= np.isfinite(out).all(axis=1)
    return out, valid


def relative_camera_transform(
    source_cumulative: Any,
    target_cumulative: Any,
) -> np.ndarray | None:
    """Return image transform source-state -> target-state."""

    source = _normalise_h(source_cumulative)
    target = _normalise_h(target_cumulative)
    if source is None or target is None:
        return None
    try:
        relative = target @ np.linalg.inv(source)
    except np.linalg.LinAlgError:
        return None
    return _normalise_h(relative)


def camera_state_distance_px(
    source_cumulative: Any,
    target_cumulative: Any,
    *,
    image_width: int,
    image_height: int,
    config: CameraTrajectoryConfig | None = None,
) -> dict[str, Any] | None:
    """Geometric distance between two camera states in image pixels.

    If the camera view is identical, source->target is approximately identity
    and this distance approaches zero.
    """

    cfg = config or CameraTrajectoryConfig()
    relative = relative_camera_transform(source_cumulative, target_cumulative)
    if relative is None:
        return None

    points = _state_grid(image_width, image_height, cfg)
    projected, valid = _project(points, relative)
    if not valid.any():
        return None

    errors = np.linalg.norm(projected[valid] - points[valid], axis=1)
    if len(errors) == 0:
        return None

    return {
        "median_px": float(np.median(errors)),
        "p90_px": float(np.percentile(errors, 90)),
        "max_px": float(np.max(errors)),
        "valid_points": int(np.count_nonzero(valid)),
        "relative_transform": relative.tolist(),
    }


class _UnionFind:
    def __init__(self, count: int):
        self.parent = list(range(count))
        self.rank = [0] * count

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a = self.find(left)
        b = self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _camera_regions(
    keyframes: list[dict[str, Any]],
    recurrence_edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not keyframes:
        return []

    union = _UnionFind(len(keyframes))
    for edge in recurrence_edges:
        union.union(int(edge["left_keyframe_index"]), int(edge["right_keyframe_index"]))

    groups: dict[int, list[int]] = {}
    for index in range(len(keyframes)):
        root = union.find(index)
        groups.setdefault(root, []).append(index)

    regions = []
    ordered_groups = sorted(
        groups.values(),
        key=lambda indices: min(keyframes[index]["frame"] for index in indices),
    )
    for region_index, indices in enumerate(ordered_groups):
        frames = [int(keyframes[index]["frame"]) for index in indices]
        regions.append(
            {
                "region": int(region_index),
                "keyframe_indices": [int(index) for index in indices],
                "frames": frames,
                "visit_count": len(indices),
                "first_frame": min(frames),
                "last_frame": max(frames),
            }
        )
    return regions


def analyse_camera_trajectory(
    video_path: str | Path,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    motion_config: GlobalCameraMotionConfig | None = None,
    trajectory_config: CameraTrajectoryConfig | None = None,
    progress_callback=None,
    motion_estimator=None,
) -> dict[str, Any]:
    """Build a raw H-independent camera trajectory and coarse state atlas."""

    tcfg = trajectory_config or CameraTrajectoryConfig()
    motion_estimator = motion_estimator or estimate_global_camera_step
    mcfg = motion_config or GlobalCameraMotionConfig(
        sample_step_frames=tcfg.sample_step_frames
    )

    # Keep one source of truth for the stride.
    if int(mcfg.sample_step_frames) != int(tcfg.sample_step_frames):
        mcfg = GlobalCameraMotionConfig(
            **{
                **asdict(mcfg),
                "sample_step_frames": int(tcfg.sample_step_frames),
            }
        )

    started_at = perf_counter()
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"status": "video_open_failed", "video": str(path)}

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if width <= 0 or height <= 0:
        capture.release()
        return {"status": "invalid_video_dimensions"}

    start = max(0, int(start_frame))
    if end_frame is None:
        end = max(start, total_frames - 1) if total_frames > 0 else start
    else:
        end = max(start, int(end_frame))
        if total_frames > 0:
            end = min(end, total_frames - 1)

    step = max(1, int(tcfg.sample_step_frames))
    sampled_frames = list(range(start, end + 1, step))
    if sampled_frames[-1] != end:
        sampled_frames.append(end)

    if start > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, previous_frame = capture.read()
    if not ok or previous_frame is None:
        capture.release()
        return {"status": "frame_read_failed", "frame": start}

    cumulative = np.eye(3, dtype=np.float64)

    first_descriptor = camera_motion_descriptor(
        cumulative,
        image_width=width,
        image_height=height,
    )
    trajectory = [
        {
            "frame": int(start),
            "cumulative_transform": cumulative.tolist(),
            "descriptor": first_descriptor,
            "step_quality": None,
        }
    ]
    keyframes = [
        {
            "frame": int(start),
            "trajectory_index": 0,
            "cumulative_transform": cumulative.tolist(),
            "descriptor": first_descriptor,
            "selection": {
                "reason": "start",
                "median_displacement_px": 0.0,
                "p90_displacement_px": 0.0,
            },
        }
    ]

    current_capture_index = start
    last_decoded_frame = start
    decoded_frame_count = 1
    motion_estimation_count = 0
    trailing_unreadable_frame_count = 0
    effective_end_frame = end
    try:
        # Decode forward once.  ``previous_frame`` always remains the previous
        # *sampled* image; decoded intermediates merely advance the codec.
        next_sample_index = 1
        for current_index in range(start + 1, end + 1):
            ok, current_frame = capture.read()
            if not ok or current_frame is None:
                trailing_unreadable_frame_count = max(0, end - last_decoded_frame)
                tolerance_frames = int(math.ceil(max(fps, 1.0) * 3.0))
                if end - last_decoded_frame <= tolerance_frames:
                    effective_end_frame = last_decoded_frame
                    break
                return {
                    "status": "frame_read_failed",
                    "frame": int(current_index),
                    "trajectory": trajectory,
                    "keyframes": keyframes,
                }

            # Physical decoding and sampled camera motion are deliberately
            # separate: intermediate frames only advance the codec.
            last_decoded_frame = current_index
            decoded_frame_count += 1
            if next_sample_index >= len(sampled_frames) or current_index != sampled_frames[next_sample_index]:
                continue

            motion_estimation_count += 1
            motion = motion_estimator(
                previous_frame,
                current_frame,
                config=mcfg,
            )
            if motion.get("status") != "ok":
                return {
                    "status": "trajectory_step_failed",
                    "failed_from_frame": int(current_capture_index),
                    "failed_to_frame": int(current_index),
                    "failure": {
                        key: value
                        for key, value in motion.items()
                        if key != "matrix"
                    },
                    "trajectory": trajectory,
                    "keyframes": keyframes,
                    "video": {
                        "path": str(path),
                        "width": width,
                        "height": height,
                        "fps": fps,
                        "total_frames": total_frames,
                        "reported_frame_count": total_frames,
                        "reported_end_frame": end,
                        "effective_end_frame": last_decoded_frame,
                        "trailing_unreadable_frame_count": 0,
                    },
                }

            sample_index = next_sample_index
            next_sample_index += 1

            step_matrix = _normalise_h(motion["matrix"])
            cumulative = _normalise_h(step_matrix @ cumulative)
            if cumulative is None:
                return {
                    "status": "trajectory_composition_failed",
                    "frame": int(current_index),
                    "trajectory": trajectory,
                    "keyframes": keyframes,
                }

            descriptor = camera_motion_descriptor(
                cumulative,
                image_width=width,
                image_height=height,
            )

            trajectory_entry = {
                "frame": int(current_index),
                "cumulative_transform": cumulative.tolist(),
                "descriptor": descriptor,
                "step_quality": {
                    key: value
                    for key, value in motion.items()
                    if key not in {"matrix", "descriptor"}
                },
            }
            trajectory.append(trajectory_entry)

            last_keyframe = keyframes[-1]
            distance = camera_state_distance_px(
                last_keyframe["cumulative_transform"],
                cumulative,
                image_width=width,
                image_height=height,
                config=tcfg,
            )

            frame_gap = int(current_index) - int(last_keyframe["frame"])
            create_keyframe = False
            reason = None
            if (
                distance is not None
                and frame_gap >= tcfg.keyframe_min_gap_frames
                and (
                    float(distance["median_px"])
                    >= tcfg.keyframe_median_displacement_px
                    or float(distance["p90_px"])
                    >= tcfg.keyframe_p90_displacement_px
                )
            ):
                create_keyframe = True
                reason = "geometric_novelty"

            if create_keyframe:
                keyframes.append(
                    {
                        "frame": int(current_index),
                        "trajectory_index": len(trajectory) - 1,
                        "cumulative_transform": cumulative.tolist(),
                        "descriptor": descriptor,
                        "selection": {
                            "reason": reason,
                            "median_displacement_px": float(distance["median_px"]),
                            "p90_displacement_px": float(distance["p90_px"]),
                        },
                    }
                )

            previous_frame = current_frame
            current_capture_index = current_index

            if progress_callback is not None:
                progress_callback(
                    sample_index,
                    len(sampled_frames) - 1,
                    int(current_index),
                    len(keyframes),
                )

    finally:
        capture.release()

    recurrence_edges: list[dict[str, Any]] = []
    for right_index in range(len(keyframes)):
        right = keyframes[right_index]
        best_prior = None

        for left_index in range(right_index):
            left = keyframes[left_index]
            frame_gap = int(right["frame"]) - int(left["frame"])
            if frame_gap < tcfg.recurrence_min_gap_frames:
                continue

            distance = camera_state_distance_px(
                left["cumulative_transform"],
                right["cumulative_transform"],
                image_width=width,
                image_height=height,
                config=tcfg,
            )
            if distance is None:
                continue

            if (
                float(distance["median_px"])
                <= tcfg.recurrence_median_displacement_px
                and float(distance["p90_px"])
                <= tcfg.recurrence_p90_displacement_px
            ):
                edge = {
                    "left_keyframe_index": int(left_index),
                    "right_keyframe_index": int(right_index),
                    "left_frame": int(left["frame"]),
                    "right_frame": int(right["frame"]),
                    "frame_gap": frame_gap,
                    "median_displacement_px": float(distance["median_px"]),
                    "p90_displacement_px": float(distance["p90_px"]),
                }
                recurrence_edges.append(edge)
                if (
                    best_prior is None
                    or edge["median_displacement_px"]
                    < best_prior["median_displacement_px"]
                ):
                    best_prior = edge

        if best_prior is not None:
            right["best_recurrence"] = best_prior

    regions = _camera_regions(keyframes, recurrence_edges)

    effective_sampled_end_frame = int(trajectory[-1]["frame"])
    return {
        "status": "ok",
        "version": 1,
        "algorithm": "camera_trajectory_v1",
        "video": {
            "path": str(path),
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "reported_frame_count": total_frames,
            "effective_frame_count": effective_end_frame + 1,
            "start_frame": start,
            "end_frame": end,
            "reported_end_frame": end,
            "effective_end_frame": effective_end_frame,
            "effective_sampled_end_frame": effective_sampled_end_frame,
            "last_decoded_frame": last_decoded_frame,
            "trailing_unreadable_frame_count": trailing_unreadable_frame_count,
        },
        "motion_config": asdict(mcfg),
        "trajectory_config": asdict(tcfg),
        "sample_count": len(trajectory),
        "keyframe_count": len(keyframes),
        "recurrence_edge_count": len(recurrence_edges),
        "region_count": len(regions),
        "elapsed_seconds": perf_counter() - started_at,
        "decoded_frame_count": decoded_frame_count,
        "sampled_frame_count": len(trajectory),
        "motion_estimation_count": motion_estimation_count,
        "trajectory": trajectory,
        "keyframes": keyframes,
        "recurrence_edges": recurrence_edges,
        "regions": regions,
    }

"""Focused acquisition checks for sequential Camera Trajectory decoding."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from supervisor.field_camera_trajectory import CameraTrajectoryConfig, analyse_camera_trajectory


def _tiny_video(path: Path, frames: int = 9) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (96, 54))
    assert writer.isOpened()
    for index in range(frames):
        image = np.full((54, 96, 3), index * 20, dtype=np.uint8)
        writer.write(image)
    writer.release()


def main() -> None:
    with TemporaryDirectory() as temporary:
        def run(frames: int, expected_pairs: list[tuple[int, int]]) -> None:
            path = Path(temporary) / f"tiny-{frames}.mp4"; _tiny_video(path, frames)
            pairs: list[tuple[int, int]] = []
            def motion(previous, current, **_kwargs):
                pairs.append((round(float(previous.mean()) / 20), round(float(current.mean()) / 20)))
                return {"status": "ok", "matrix": np.eye(3).tolist()}
            result = analyse_camera_trajectory(path, trajectory_config=CameraTrajectoryConfig(sample_step_frames=2), motion_estimator=motion)
            assert result.get("status") == "ok", result
            assert pairs == expected_pairs, pairs
            assert result["motion_estimation_count"] == len(expected_pairs)
            assert result["decoded_frame_count"] == frames
            assert result["sampled_frame_count"] == len(expected_pairs) + 1
        run(9, [(0, 2), (2, 4), (4, 6), (6, 8)])
        run(10, [(0, 2), (2, 4), (4, 6), (6, 8), (8, 9)])
    print("check_camera_trajectory_sequential: ok")


if __name__ == "__main__": main()

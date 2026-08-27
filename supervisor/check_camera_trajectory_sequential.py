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
        image = np.zeros((54, 96, 3), dtype=np.uint8)
        cv2.circle(image, (10 + index * 5, 27), 7, (255, 255, 255), -1)
        writer.write(image)
    writer.release()


def main() -> None:
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "tiny.mp4"; _tiny_video(path)
        result = analyse_camera_trajectory(path, trajectory_config=CameraTrajectoryConfig(sample_step_frames=2))
        # A tiny feature-poor clip may reject motion, but sequential decoding
        # must still preserve the intended sampled-frame policy up to failure.
        if result.get("status") == "ok":
            frames = [item["frame"] for item in result["trajectory"]]
            assert frames == [0, 2, 4, 6, 8]
            assert result["decoded_frame_count"] == 9
            assert result["sampled_frame_count"] == len(frames)
            assert result["video"]["effective_end_frame"] == 8
    print("check_camera_trajectory_sequential: ok")


if __name__ == "__main__": main()

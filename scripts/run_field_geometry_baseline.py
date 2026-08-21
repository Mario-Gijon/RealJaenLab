#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


DEFAULT_INPUT = Path("input/benchmark_normal_30s.mp4")
DEFAULT_OUTPUT = Path(
    "outputs/field_geometry_baseline/field_geometry_normal.mp4"
)
DEFAULT_SAMPLES = Path(
    "outputs/field_geometry_baseline/samples"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dynamic football-pitch geometry baseline for Veo footage. "
            "Estimates grass region and white pitch-line candidates per frame."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=DEFAULT_SAMPLES,
    )
    parser.add_argument(
        "--process-width",
        type=int,
        default=1280,
        help=(
            "Maximum width used for geometry processing. "
            "The overlay is projected back to original resolution."
        ),
    )
    parser.add_argument(
        "--line-min-length",
        type=int,
        default=70,
    )
    parser.add_argument(
        "--save-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if num_labels <= 1:
        return np.zeros_like(mask)

    # Ignore label 0 (background).
    largest = 1 + np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    )

    result = np.zeros_like(mask)
    result[labels == largest] = 255
    return result


def estimate_grass_mask(frame: np.ndarray) -> np.ndarray:
    """
    Estimate the dominant grass/playing-surface region.

    This is deliberately a baseline rather than a trained model:
    - HSV green prior
    - excess-green prior
    - morphology
    - largest connected component
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # Broad football-grass hue range. Deliberately permissive because
    # shadows / sun / camera WB can move the hue considerably.
    hsv_green = (
        (h >= 25)
        & (h <= 105)
        & (s >= 35)
        & (v >= 30)
    )

    b, g, r = cv2.split(
        frame.astype(np.int16)
    )
    excess_green = 2 * g - r - b

    exg_green = excess_green >= 20

    # Requiring at least some saturation avoids many grey/white objects.
    candidate = (
        (hsv_green | exg_green)
        & (s >= 25)
        & (v >= 25)
    ).astype(np.uint8) * 255

    height, width = candidate.shape

    # Avoid tiny isolated green regions in stands/graphics.
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (21, 21),
    )

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        kernel_open,
        iterations=1,
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        kernel_close,
        iterations=2,
    )

    component = largest_component(candidate)

    # Fill small holes from players / markings.
    component = cv2.morphologyEx(
        component,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (31, 31),
        ),
        iterations=1,
    )

    # If the largest region is implausibly tiny, return no field.
    coverage = (
        cv2.countNonZero(component)
        / float(width * height)
    )

    if coverage < 0.12:
        return np.zeros_like(component)

    return component


def field_hull(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    if cv2.contourArea(contour) < 1000:
        return None

    hull = cv2.convexHull(contour)

    epsilon = 0.01 * cv2.arcLength(
        hull,
        True,
    )

    polygon = cv2.approxPolyDP(
        hull,
        epsilon,
        True,
    )

    return polygon


def sample_segment_fraction(
    mask: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    thickness: int = 5,
) -> float:
    sample = np.zeros_like(mask)
    cv2.line(
        sample,
        (x1, y1),
        (x2, y2),
        255,
        thickness,
    )

    denom = cv2.countNonZero(sample)

    if denom == 0:
        return 0.0

    overlap = cv2.bitwise_and(
        sample,
        mask,
    )

    return cv2.countNonZero(overlap) / denom


def detect_pitch_lines(
    frame: np.ndarray,
    grass_mask: np.ndarray,
    min_line_length: int,
) -> list[tuple[int, int, int, int]]:
    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV,
    )

    _, s, v = cv2.split(hsv)

    # Candidate white paint: bright and relatively unsaturated.
    white = (
        (s <= 85)
        & (v >= 145)
    ).astype(np.uint8) * 255

    # Allow line pixels slightly outside the green mask because the touchline
    # itself can form the visible grass boundary.
    field_context = cv2.dilate(
        grass_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (17, 17),
        ),
        iterations=1,
    )

    white = cv2.bitwise_and(
        white,
        field_context,
    )

    white = cv2.morphologyEx(
        white,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 3),
        ),
    )

    edges = cv2.Canny(
        white,
        50,
        150,
        apertureSize=3,
    )

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=45,
        minLineLength=min_line_length,
        maxLineGap=25,
    )

    if lines is None:
        return []

    accepted = []

    # OpenCV may return HoughLinesP as either (N, 1, 4) or (N, 4)
    # depending on build/version. Normalize both cases.
    lines = np.asarray(lines).reshape(-1, 4)

    for raw in lines:
        x1, y1, x2, y2 = map(int, raw)

        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))

        if length < min_line_length:
            continue

        white_fraction = sample_segment_fraction(
            white,
            x1,
            y1,
            x2,
            y2,
            thickness=7,
        )

        field_fraction = sample_segment_fraction(
            field_context,
            x1,
            y1,
            x2,
            y2,
            thickness=7,
        )

        if white_fraction < 0.20:
            continue

        if field_fraction < 0.75:
            continue

        accepted.append(
            (x1, y1, x2, y2)
        )

    return accepted


def overlay_geometry(
    frame: np.ndarray,
    grass_mask: np.ndarray,
    polygon: np.ndarray | None,
    lines: list[tuple[int, int, int, int]],
) -> np.ndarray:
    output = frame.copy()

    # Translucent green estimated playing surface.
    green_layer = np.zeros_like(output)
    green_layer[:] = (0, 180, 0)

    mask_bool = grass_mask > 0

    blended = cv2.addWeighted(
        output,
        0.72,
        green_layer,
        0.28,
        0,
    )

    output[mask_bool] = blended[mask_bool]

    # Dynamic outer contour in red.
    if polygon is not None:
        cv2.polylines(
            output,
            [polygon],
            True,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    # White-line candidates in cyan.
    for x1, y1, x2, y2 in lines:
        cv2.line(
            output,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            3,
            cv2.LINE_AA,
        )

    coverage = (
        cv2.countNonZero(grass_mask)
        / grass_mask.size
    )

    cv2.putText(
        output,
        f"field coverage: {coverage * 100:.1f}%",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        f"line candidates: {len(lines)}",
        (18, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


def process_frame(
    frame: np.ndarray,
    process_width: int,
    min_line_length: int,
):
    original_h, original_w = frame.shape[:2]

    if original_w > process_width:
        scale = process_width / original_w
        proc_w = process_width
        proc_h = int(round(original_h * scale))

        proc = cv2.resize(
            frame,
            (proc_w, proc_h),
            interpolation=cv2.INTER_AREA,
        )
    else:
        proc = frame
        scale = 1.0

    grass_mask = estimate_grass_mask(proc)
    polygon = field_hull(grass_mask)

    lines = detect_pitch_lines(
        proc,
        grass_mask,
        max(
            30,
            int(round(min_line_length * scale)),
        ),
    )

    debug = overlay_geometry(
        proc,
        grass_mask,
        polygon,
        lines,
    )

    if scale != 1.0:
        debug = cv2.resize(
            debug,
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )

    return debug, grass_mask, polygon, lines


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    cap = cv2.VideoCapture(
        str(args.input)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open {args.input}"
        )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )
    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:
        fps = 30.0

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create {args.output}"
        )

    if args.save_samples:
        args.samples_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    sample_seconds = {
        0,
        5,
        10,
        15,
        20,
        25,
    }

    frame_idx = 0
    line_counts = []
    coverages = []
    start = time.perf_counter()

    print("=== FIELD GEOMETRY BASELINE ===")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(
        f"Video: {width}x{height} @ {fps:.2f} FPS "
        f"| {total_frames} frames"
    )
    print()

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            debug, grass_mask, _, lines = process_frame(
                frame,
                args.process_width,
                args.line_min_length,
            )

            writer.write(debug)

            line_counts.append(len(lines))
            coverages.append(
                cv2.countNonZero(grass_mask)
                / grass_mask.size
            )

            second = int(
                frame_idx / fps
            )

            if (
                args.save_samples
                and second in sample_seconds
                and abs(
                    frame_idx - round(second * fps)
                ) <= 0
            ):
                sample_path = (
                    args.samples_dir
                    / f"field_{second:02d}s.jpg"
                )
                cv2.imwrite(
                    str(sample_path),
                    debug,
                )

            frame_idx += 1

            if (
                frame_idx == 1
                or frame_idx % 100 == 0
                or frame_idx == total_frames
            ):
                elapsed = (
                    time.perf_counter() - start
                )
                speed = (
                    frame_idx / elapsed
                    if elapsed > 0
                    else 0.0
                )
                print(
                    f"\r{frame_idx}/{total_frames} "
                    f"| {speed:.2f} FPS "
                    f"| lines={len(lines):2d}",
                    end="",
                    flush=True,
                )

    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - start

    mean_lines = (
        float(np.mean(line_counts))
        if line_counts
        else 0.0
    )
    mean_coverage = (
        float(np.mean(coverages))
        if coverages
        else 0.0
    )

    print()
    print()
    print("Done.")
    print(
        f"Processed {frame_idx} frames "
        f"in {elapsed:.2f}s "
        f"({frame_idx / elapsed:.2f} FPS)"
    )
    print(
        f"Mean field coverage: "
        f"{mean_coverage * 100:.1f}%"
    )
    print(
        f"Mean line candidates/frame: "
        f"{mean_lines:.1f}"
    )

    if args.save_samples:
        print(
            f"Samples: {args.samples_dir}"
        )


if __name__ == "__main__":
    main()
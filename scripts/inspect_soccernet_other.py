#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np


SOCCERNET_ROOT = Path("datasets/raw/soccernet_gamestate")
OUTPUT_ROOT = Path("datasets/audits")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a SoccerNet GameState contact sheet containing "
            "only annotations with role/category 'other'."
        )
    )
    parser.add_argument("--samples", type=int, default=36)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def draw_box(image, bbox, label):
    x, y, w, h = map(float, bbox)

    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))

    ih, iw = image.shape[:2]

    x1 = max(0, min(x1, iw - 1))
    y1 = max(0, min(y1, ih - 1))
    x2 = max(0, min(x2, iw - 1))
    y2 = max(0, min(y2, ih - 1))

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.putText(
        image,
        label,
        (x1, max(22, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def resize_tile(image, target_w=640, target_h=360):
    h, w = image.shape[:2]

    scale = min(target_w / w, target_h / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    resized = cv2.resize(
        image,
        (nw, nh),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x = (target_w - nw) // 2
    y = (target_h - nh) // 2

    canvas[y:y + nh, x:x + nw] = resized

    return canvas


def build_image_cache(clip_dir: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in clip_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def collect_other_annotations():
    records = []
    other_annotations_seen = 0
    missing_images = 0

    for split in ("train", "valid", "test"):
        split_root = SOCCERNET_ROOT / split

        for label_path in sorted(
            split_root.glob("SNGS-*/Labels-GameState.json")
        ):
            clip_dir = label_path.parent
            data = load_json(label_path)

            categories = {
                int(category["id"]): str(category["name"]).lower()
                for category in data.get("categories", [])
            }

            images = {
                str(image["image_id"]): image
                for image in data.get("images", [])
                if image.get("image_id") is not None
            }

            other_annotations = []

            for ann in data.get("annotations", []):
                category_name = categories.get(
                    int(ann.get("category_id", -1)),
                    "",
                )

                attrs = ann.get("attributes") or {}
                role = str(attrs.get("role", "")).lower()

                if category_name == "other" or role == "other":
                    other_annotations.append(
                        (ann, category_name, role)
                    )

            if not other_annotations:
                continue

            other_annotations_seen += len(other_annotations)

            # Images are nested inside SoccerNet clip directories,
            # so build a filename -> full path cache once per relevant clip.
            image_cache = build_image_cache(clip_dir)

            for ann, category_name, role in other_annotations:
                bbox_image = ann.get("bbox_image")

                if not isinstance(bbox_image, dict):
                    continue

                required = ("x", "y", "w", "h")

                if not all(key in bbox_image for key in required):
                    continue

                image_id = str(ann.get("image_id"))
                image_meta = images.get(image_id)

                if not image_meta:
                    continue

                file_name = image_meta.get("file_name")

                if not file_name:
                    continue

                image_path = image_cache.get(Path(str(file_name)).name)

                if image_path is None:
                    missing_images += 1
                    continue

                bbox = [
                    bbox_image["x"],
                    bbox_image["y"],
                    bbox_image["w"],
                    bbox_image["h"],
                ]

                records.append(
                    {
                        "split": split,
                        "clip": clip_dir.name,
                        "image_path": image_path,
                        "bbox": bbox,
                        "role": role or "none",
                        "category": category_name or "unknown",
                        "track_id": ann.get("track_id"),
                    }
                )

    return records, other_annotations_seen, missing_images


def build_contact_sheet(records, output_path):
    cols = 3
    tile_w = 640
    image_h = 360
    footer_h = 40
    tile_h = image_h + footer_h
    header_h = 45

    rows = math.ceil(len(records) / cols)

    sheet = np.zeros(
        (header_h + rows * tile_h, cols * tile_w, 3),
        dtype=np.uint8,
    )

    cv2.putText(
        sheet,
        "SoccerNet GameState - examples annotated as OTHER",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for index, record in enumerate(records):
        image = cv2.imread(str(record["image_path"]))

        if image is None:
            continue

        draw_box(
            image,
            record["bbox"],
            "OTHER",
        )

        tile_image = resize_tile(
            image,
            target_w=tile_w,
            target_h=image_h,
        )

        footer = np.zeros(
            (footer_h, tile_w, 3),
            dtype=np.uint8,
        )

        caption = (
            f'{record["split"]}/{record["clip"]}/'
            f'{record["image_path"].name} '
            f'track={record["track_id"]}'
        )

        cv2.putText(
            footer,
            caption[:100],
            (8, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        tile = np.vstack([tile_image, footer])

        row = index // cols
        col = index % cols

        y0 = header_h + row * tile_h
        x0 = col * tile_w

        sheet[
            y0:y0 + tile_h,
            x0:x0 + tile_w,
        ] = tile

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        sheet,
    )


def main():
    args = parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be greater than 0.")

    print("Searching SoccerNet GameState for 'other' annotations...")

    records, seen, missing_images = collect_other_annotations()

    print(f"'other' annotations present in JSON: {seen}")
    print(f"'other' annotations with resolved images: {len(records)}")
    print(f"Missing image resolutions: {missing_images}")

    if not records:
        raise RuntimeError(
            "The annotations exist, but no corresponding images could be resolved."
        )

    rng = random.Random(args.seed)
    rng.shuffle(records)

    selected = records[: min(args.samples, len(records))]

    output_path = (
        OUTPUT_ROOT
        / "soccernet_other_contact_sheet.jpg"
    )

    build_contact_sheet(
        selected,
        output_path,
    )

    print(f"Samples rendered: {len(selected)}")
    print(f"Contact sheet: {output_path}")
    print("No dataset files were modified.")


if __name__ == "__main__":
    main()
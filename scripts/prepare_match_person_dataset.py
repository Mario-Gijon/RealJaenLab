#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VEO_ROOT = Path("datasets/raw/football_images_veo")
SOCCERNET_ROOT = Path("datasets/raw/soccernet_gamestate")
OUTPUT_ROOT = Path("datasets/processed/match_person_v1")

VEO_POSITIVE_CLASSES = {"player", "referee"}
SOCCERNET_POSITIVE_CLASSES = {"player", "goalkeeper", "referee"}

CATEGORY = {
    "id": 1,
    "name": "match_person",
    "supercategory": "person",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build match_person_v1 by combining the Veo football dataset "
            "and SoccerNet GameState into one COCO-format, one-class dataset."
        )
    )
    parser.add_argument(
        "--train-stride",
        type=int,
        default=5,
        help=(
            "Keep one of every N SoccerNet train frames per clip. "
            "Default 5 ~= 5 FPS from the 25 FPS clips."
        ),
    )
    parser.add_argument(
        "--valid-stride",
        type=int,
        default=25,
        help=(
            "Keep one of every N SoccerNet validation frames per clip. "
            "Default 25 ~= 1 FPS."
        ),
    )
    parser.add_argument(
        "--test-stride",
        type=int,
        default=25,
        help=(
            "Keep one of every N SoccerNet test frames per clip. "
            "Default 25 ~= 1 FPS."
        ),
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help=(
            "How to materialize images in the processed dataset. "
            "Hardlink uses essentially no extra image storage when source "
            "and destination are on the same filesystem."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate and print what would be created without writing it.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild datasets/processed/match_person_v1 if it exists.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def valid_bbox(bbox: Any, width: int, height: int) -> bool:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False

    try:
        x, y, w, h = map(float, bbox)
    except (TypeError, ValueError):
        return False

    if w <= 0 or h <= 0 or x < 0 or y < 0:
        return False

    if x + w > width + 1 or y + h > height + 1:
        return False

    return True


def materialize(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() or destination.is_symlink():
        destination.unlink()

    if mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            # Cross-device hardlinks are impossible. Fall back to symlink
            # rather than unexpectedly copying tens of GB.
            destination.symlink_to(source.resolve())
            return "symlink-fallback"

    if mode == "symlink":
        destination.symlink_to(source.resolve())
        return "symlink"

    shutil.copy2(source, destination)
    return "copy"


def bbox_from_soccernet(ann: dict[str, Any]) -> list[float] | None:
    box = ann.get("bbox_image")

    if not isinstance(box, dict):
        return None

    required = ("x", "y", "w", "h")

    if not all(key in box for key in required):
        return None

    return [
        float(box["x"]),
        float(box["y"]),
        float(box["w"]),
        float(box["h"]),
    ]


def all_veo_records_for_train():
    """Use every Veo image only in TRAIN.

    The Roboflow train/valid/test split leaks the same source matches across
    splits, so none of its original validation/test partitions are used for
    evaluation in match_person_v1.
    """
    for split in ("train", "valid", "test"):
        yield from veo_split_records(split)


def veo_split_records(split: str):
    split_dir = VEO_ROOT / split
    annotation_path = split_dir / "_annotations.coco.json"
    data = load_json(annotation_path)

    categories = {
        int(cat["id"]): str(cat["name"]).strip().lower()
        for cat in data.get("categories", [])
    }

    images = {
        int(img["id"]): img
        for img in data.get("images", [])
    }

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for ann in data.get("annotations", []):
        image_id = int(ann["image_id"])
        category_name = categories.get(
            int(ann["category_id"]),
            "",
        )

        if category_name not in VEO_POSITIVE_CLASSES:
            continue

        annotations_by_image[image_id].append(ann)

    for image_id, image_meta in images.items():
        file_name = str(image_meta["file_name"])
        source_path = split_dir / file_name

        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        width = int(image_meta["width"])
        height = int(image_meta["height"])

        positive_boxes = []

        for ann in annotations_by_image.get(image_id, []):
            bbox = ann.get("bbox")

            if valid_bbox(bbox, width, height):
                positive_boxes.append([float(v) for v in bbox])

        yield {
            "source": "veo",
            "source_split": split,
            "source_clip": None,
            "source_path": source_path,
            "source_file_name": file_name,
            "width": width,
            "height": height,
            "boxes": positive_boxes,
        }


def soccernet_image_cache(clip_dir: Path) -> dict[str, Path]:
    return {
        p.name: p
        for p in clip_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".webp"
        }
    }


def soccernet_split_records(split: str, stride: int):
    split_dir = SOCCERNET_ROOT / split

    for label_path in sorted(
        split_dir.glob("SNGS-*/Labels-GameState.json")
    ):
        clip_dir = label_path.parent
        clip_name = clip_dir.name
        data = load_json(label_path)

        categories = {
            int(cat["id"]): str(cat["name"]).strip().lower()
            for cat in data.get("categories", [])
        }

        images = sorted(
            data.get("images", []),
            key=lambda img: str(img.get("file_name", "")),
        )

        annotations_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for ann in data.get("annotations", []):
            image_id = str(ann.get("image_id"))

            attrs = ann.get("attributes") or {}
            role = str(attrs.get("role", "")).strip().lower()
            category = categories.get(
                int(ann.get("category_id", -1)),
                "",
            )

            semantic_class = role or category

            if semantic_class not in SOCCERNET_POSITIVE_CLASSES:
                continue

            annotations_by_image[image_id].append(ann)

        image_cache = soccernet_image_cache(clip_dir)

        for index, image_meta in enumerate(images):
            if index % stride != 0:
                continue

            image_id = str(image_meta["image_id"])
            file_name = str(image_meta["file_name"])
            source_path = image_cache.get(Path(file_name).name)

            if source_path is None:
                raise FileNotFoundError(
                    f"Could not resolve {clip_name}/{file_name}"
                )

            width = int(image_meta["width"])
            height = int(image_meta["height"])

            positive_boxes = []

            for ann in annotations_by_image.get(image_id, []):
                bbox = bbox_from_soccernet(ann)

                if bbox is not None and valid_bbox(
                    bbox,
                    width,
                    height,
                ):
                    positive_boxes.append(bbox)

            yield {
                "source": "soccernet",
                "source_split": split,
                "source_clip": clip_name,
                "source_path": source_path,
                "source_file_name": file_name,
                "width": width,
                "height": height,
                "boxes": positive_boxes,
            }


def output_filename(record: dict[str, Any], sequence: int) -> str:
    suffix = record["source_path"].suffix.lower()

    if record["source"] == "soccernet":
        clip = record["source_clip"]
        stem = record["source_path"].stem
        return f"sn_{clip}_{stem}_{sequence:06d}{suffix}"

    # Roboflow names are already globally distinctive, but prefix anyway
    # to make dataset provenance obvious.
    return f"veo_{sequence:06d}_{record['source_path'].name}"


def build_split(
    split: str,
    soccernet_stride: int,
    link_mode: str,
    dry_run: bool,
):
    # The original Roboflow Veo split has source-video leakage between
    # train/valid/test. To avoid optimistic metrics, all Veo images are used
    # only for training. Validation and test are SoccerNet-only.
    if split == "train":
        records = list(all_veo_records_for_train())
    else:
        records = []

    veo_count = len(records)

    soccer_records = list(
        soccernet_split_records(
            split,
            stride=soccernet_stride,
        )
    )
    records.extend(soccer_records)

    output_dir = OUTPUT_ROOT / split

    images_json = []
    annotations_json = []
    manifest = []

    image_id = 1
    annotation_id = 1
    positive_boxes = 0
    negative_images = 0
    link_methods = Counter()
    source_counts = Counter()

    for sequence, record in enumerate(records, start=1):
        file_name = output_filename(record, sequence)
        destination = output_dir / file_name

        source_counts[record["source"]] += 1

        if not record["boxes"]:
            negative_images += 1

        if not dry_run:
            method = materialize(
                record["source_path"],
                destination,
                link_mode,
            )
            link_methods[method] += 1

        images_json.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": record["width"],
                "height": record["height"],
            }
        )

        for bbox in record["boxes"]:
            x, y, w, h = bbox

            annotations_json.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )

            annotation_id += 1
            positive_boxes += 1

        manifest.append(
            {
                "output_file": file_name,
                "source": record["source"],
                "source_split": record["source_split"],
                "source_clip": record["source_clip"],
                "source_path": str(record["source_path"]),
                "positive_boxes": len(record["boxes"]),
            }
        )

        image_id += 1

    coco = {
        "info": {
            "description": (
                "RealJaenLab match_person_v1: Veo Football_images + "
                "subsampled SoccerNet GameState."
            ),
            "version": "1.0",
        },
        "licenses": [],
        "categories": [CATEGORY],
        "images": images_json,
        "annotations": annotations_json,
    }

    summary = {
        "split": split,
        "veo_images": veo_count,
        "soccernet_images": len(soccer_records),
        "total_images": len(records),
        "positive_boxes": positive_boxes,
        "negative_images": negative_images,
        "soccernet_stride": soccernet_stride,
        "link_methods": dict(link_methods),
        "source_counts": dict(source_counts),
    }

    if not dry_run:
        write_json(
            output_dir / "_annotations.coco.json",
            coco,
        )
        write_json(
            output_dir / "_manifest.json",
            manifest,
        )

    return summary


def main() -> None:
    args = parse_args()

    for value, name in (
        (args.train_stride, "--train-stride"),
        (args.valid_stride, "--valid-stride"),
        (args.test_stride, "--test-stride"),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0.")

    if OUTPUT_ROOT.exists() and not args.dry_run:
        if not args.overwrite:
            raise FileExistsError(
                f"{OUTPUT_ROOT} already exists. "
                "Use --overwrite to rebuild it."
            )

        shutil.rmtree(OUTPUT_ROOT)

    if not args.dry_run:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Building match_person_v1")
    print()
    print("Split policy:")
    print("  Veo: ALL original train/valid/test images -> TRAIN only")
    print("  SoccerNet: preserve official train/valid/test splits")
    print("  RealJaen benchmark clips: never included")
    print()
    print("Positive mapping:")
    print("  Veo: Player + Referee -> match_person")
    print(
        "  SoccerNet: player + goalkeeper + referee "
        "-> match_person"
    )
    print("Ignored as positives:")
    print("  Football/ball, pitch, other, staff/background")
    print()

    summaries = []

    for split, stride in (
        ("train", args.train_stride),
        ("valid", args.valid_stride),
        ("test", args.test_stride),
    ):
        print(f"Processing {split}...")
        summary = build_split(
            split=split,
            soccernet_stride=stride,
            link_mode=args.link_mode,
            dry_run=args.dry_run,
        )
        summaries.append(summary)

        print(
            f"  images={summary['total_images']} "
            f"(veo={summary['veo_images']}, "
            f"soccernet={summary['soccernet_images']})"
        )
        print(
            f"  match_person boxes="
            f"{summary['positive_boxes']}"
        )
        print(
            f"  images with no positive match_person="
            f"{summary['negative_images']}"
        )

    dataset_summary = {
        "dataset": "match_person_v1",
        "positive_class": "match_person",
        "split_policy": {
            "veo": "all source splits merged into train only due to source-video leakage",
            "soccernet": "official train/valid/test preserved",
            "realjaen_benchmarks": "excluded",
        },
        "positive_mapping": {
            "veo": sorted(VEO_POSITIVE_CLASSES),
            "soccernet": sorted(
                SOCCERNET_POSITIVE_CLASSES
            ),
        },
        "ignored_positive_classes": [
            "football",
            "ball",
            "pitch",
            "other",
        ],
        "splits": summaries,
    }

    print()
    print("Summary:")
    for summary in summaries:
        print(
            f"  {summary['split']}: "
            f"{summary['total_images']} images, "
            f"{summary['positive_boxes']} boxes"
        )

    if args.dry_run:
        print()
        print("DRY RUN: no files were created.")
        return

    write_json(
        OUTPUT_ROOT / "dataset_summary.json",
        dataset_summary,
    )

    print()
    print(f"Dataset created at: {OUTPUT_ROOT}")
    print(
        "Images were materialized using "
        f"link mode: {args.link_mode}"
    )
    print(
        "Raw datasets were not modified."
    )


if __name__ == "__main__":
    main()
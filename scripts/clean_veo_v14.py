#!/usr/bin/env python3

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


KEEP_CLASSES = [
    "ball",
    "linesman",
    "referee",
    "team_a",
    "team_b",
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input",
        type=Path,
    )

    parser.add_argument(
        "output",
        type=Path,
    )

    args = parser.parse_args()

    input_root = args.input.resolve()
    output_root = args.output.resolve()

    images_dir = output_root / "images"

    if output_root.exists():
        raise RuntimeError(
            f"Output already exists: {output_root}\n"
            "Remove it manually if you really want to recreate it."
        )

    images_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    #
    # Category map canonical 0..4
    #

    categories = [
        {
            "id": i,
            "name": name,
            "supercategory": "football",
        }
        for i, name in enumerate(KEEP_CLASSES)
    ]

    name_to_new_id = {
        c["name"]: c["id"]
        for c in categories
    }

    new_images = []
    new_annotations = []

    image_id = 1
    ann_id = 1

    removed_classes = Counter()
    kept_classes = Counter()

    clipped_boxes = 0
    invalid_boxes = 0
    copied_images = 0

    source_split_counts = Counter()

    #
    # Merge original train/valid/test into
    # one canonical pool.
    #

    for split in ["train", "valid", "test"]:

        split_dir = input_root / split

        annotation_file = (
            split_dir /
            "_annotations.coco.json"
        )

        if not annotation_file.exists():
            continue

        data = json.loads(
            annotation_file.read_text()
        )

        old_categories = {
            c["id"]: c["name"]
            for c in data["categories"]
        }

        anns_by_image = {}

        for ann in data["annotations"]:
            anns_by_image.setdefault(
                ann["image_id"],
                [],
            ).append(ann)

        for old_image in data["images"]:

            old_file = (
                split_dir /
                old_image["file_name"]
            )

            if not old_file.exists():
                raise FileNotFoundError(
                    old_file
                )

            width = int(old_image["width"])
            height = int(old_image["height"])

            #
            # Unique filename even if two splits
            # happened to contain the same basename.
            #
            extension = old_file.suffix.lower()

            new_filename = (
                f"{image_id:06d}_"
                f"{split}"
                f"{extension}"
            )

            destination = (
                images_dir /
                new_filename
            )

            shutil.copy2(
                old_file,
                destination,
            )

            new_images.append(
                {
                    "id": image_id,
                    "file_name": new_filename,
                    "width": width,
                    "height": height,

                    # Custom metadata useful for
                    # our later audit/split.
                    "source_split": split,
                    "source_file_name":
                        old_image["file_name"],
                }
            )

            copied_images += 1
            source_split_counts[split] += 1

            #
            # Clean annotations.
            #

            for old_ann in anns_by_image.get(
                old_image["id"],
                [],
            ):

                old_name = old_categories[
                    old_ann["category_id"]
                ]

                if old_name not in KEEP_CLASSES:
                    removed_classes[
                        old_name
                    ] += 1
                    continue

                x, y, w, h = map(
                    float,
                    old_ann["bbox"],
                )

                old_x1 = x
                old_y1 = y
                old_x2 = x + w
                old_y2 = y + h

                #
                # Clip to image.
                #

                x1 = max(
                    0.0,
                    min(float(width), old_x1),
                )

                y1 = max(
                    0.0,
                    min(float(height), old_y1),
                )

                x2 = max(
                    0.0,
                    min(float(width), old_x2),
                )

                y2 = max(
                    0.0,
                    min(float(height), old_y2),
                )

                if (
                    x1 != old_x1
                    or y1 != old_y1
                    or x2 != old_x2
                    or y2 != old_y2
                ):
                    clipped_boxes += 1

                new_w = x2 - x1
                new_h = y2 - y1

                if new_w <= 0 or new_h <= 0:
                    invalid_boxes += 1
                    continue

                new_annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id":
                            name_to_new_id[
                                old_name
                            ],
                        "bbox": [
                            x1,
                            y1,
                            new_w,
                            new_h,
                        ],
                        "area":
                            new_w * new_h,
                        "iscrowd":
                            int(
                                old_ann.get(
                                    "iscrowd",
                                    0,
                                )
                            ),
                    }
                )

                ann_id += 1

                kept_classes[
                    old_name
                ] += 1

            image_id += 1

    coco = {
        "info": {
            "description":
                "RealJaenLab cleaned version of "
                "Veo Football Detection v14",
        },
        "licenses": [],
        "categories": categories,
        "images": new_images,
        "annotations": new_annotations,
    }

    annotation_path = (
        output_root /
        "annotations.coco.json"
    )

    annotation_path.write_text(
        json.dumps(
            coco,
            indent=2,
        )
    )

    #
    # Report
    #

    report = []

    report.append(
        "=== Veo Football Detection v14 clean ==="
    )

    report.append(
        f"Images: {len(new_images)}"
    )

    report.append(
        f"Annotations: {len(new_annotations)}"
    )

    report.append("")
    report.append("Kept classes:")

    for name in KEEP_CLASSES:
        report.append(
            f"  {name:12s}: "
            f"{kept_classes[name]}"
        )

    report.append("")
    report.append("Removed annotations:")

    for name, count in sorted(
        removed_classes.items()
    ):
        report.append(
            f"  {name:25s}: {count}"
        )

    report.append("")
    report.append(
        f"Clipped bounding boxes: "
        f"{clipped_boxes}"
    )

    report.append(
        f"Invalid boxes removed: "
        f"{invalid_boxes}"
    )

    report.append("")
    report.append("Original source splits:")

    for split in ["train", "valid", "test"]:
        report.append(
            f"  {split:5s}: "
            f"{source_split_counts[split]}"
        )

    report_text = "\n".join(
        report
    )

    print(report_text)

    (
        output_root /
        "clean_report.txt"
    ).write_text(
        report_text + "\n"
    )


if __name__ == "__main__":
    main()

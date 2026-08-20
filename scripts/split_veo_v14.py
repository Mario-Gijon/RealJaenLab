#!/usr/bin/env python3

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------
# pHash
# ---------------------------------------------------------

def phash(path: Path):
    img = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if img is None:
        raise RuntimeError(
            f"Cannot read {path}"
        )

    img = cv2.resize(
        img,
        (32, 32),
        interpolation=cv2.INTER_AREA,
    )

    dct = cv2.dct(
        np.float32(img)
    )

    low = dct[:8, :8]

    flat = low.flatten()

    median = np.median(
        flat[1:]
    )

    return (
        low > median
    ).flatten()


class UnionFind:

    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(
                self.parent[x]
            )

        return self.parent[x]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def hardlink_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_split(
    split_name,
    image_indexes,
    images,
    annotations_by_image,
    categories,
    source_root,
    output_root,
):
    split_dir = (
        output_root /
        split_name
    )

    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_images = []
    new_annotations = []

    new_image_id = 1
    new_ann_id = 1

    class_counts = Counter()

    category_names = {
        c["id"]: c["name"]
        for c in categories
    }

    for index in sorted(image_indexes):

        old_image = images[index]

        src = (
            source_root /
            "images" /
            old_image["file_name"]
        )

        dst = (
            split_dir /
            old_image["file_name"]
        )

        hardlink_or_copy(
            src,
            dst,
        )

        new_image = dict(old_image)
        new_image["id"] = new_image_id

        new_images.append(
            new_image
        )

        for ann in annotations_by_image.get(
            old_image["id"],
            [],
        ):
            new_ann = dict(ann)

            new_ann["id"] = new_ann_id
            new_ann["image_id"] = (
                new_image_id
            )

            new_annotations.append(
                new_ann
            )

            class_counts[
                category_names[
                    ann["category_id"]
                ]
            ] += 1

            new_ann_id += 1

        new_image_id += 1

    coco = {
        "info": {
            "description":
                "RealJaenLab V14 clean "
                f"{split_name} split"
        },
        "licenses": [],
        "categories": categories,
        "images": new_images,
        "annotations": new_annotations,
    }

    (
        split_dir /
        "_annotations.coco.json"
    ).write_text(
        json.dumps(
            coco,
            indent=2,
        )
    )

    return class_counts


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

    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    random.seed(args.seed)

    input_root = args.input.resolve()
    output_root = args.output.resolve()

    if output_root.exists():
        raise RuntimeError(
            f"Output already exists: "
            f"{output_root}"
        )

    data = json.loads(
        (
            input_root /
            "annotations.coco.json"
        ).read_text()
    )

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    n = len(images)

    annotations_by_image = defaultdict(
        list
    )

    for ann in annotations:
        annotations_by_image[
            ann["image_id"]
        ].append(ann)

    print(
        f"Images: {n}"
    )

    #
    # Calculate pHash
    #

    print(
        "Computing pHash..."
    )

    hashes = []

    for i, image in enumerate(images):

        path = (
            input_root /
            "images" /
            image["file_name"]
        )

        hashes.append(
            phash(path)
        )

        if (
            (i + 1) % 100 == 0
            or i + 1 == n
        ):
            print(
                f"\r{i + 1}/{n}",
                end="",
                flush=True,
            )

    print()

    hashes = np.stack(hashes)

    #
    # Build strict near-duplicate groups
    #

    uf = UnionFind(n)

    print(
        "Grouping near-duplicates..."
    )

    for i in range(n):

        diff = (
            hashes[i + 1:]
            != hashes[i]
        )

        if len(diff) == 0:
            continue

        distances = np.count_nonzero(
            diff,
            axis=1,
        )

        matches = np.where(
            distances
            <= args.phash_threshold
        )[0]

        for relative_j in matches:

            j = (
                i + 1
                + int(relative_j)
            )

            uf.union(
                i,
                j,
            )

    groups_dict = defaultdict(list)

    for i in range(n):
        groups_dict[
            uf.find(i)
        ].append(i)

    groups = list(
        groups_dict.values()
    )

    #
    # Shuffle groups, but then greedily choose
    # validation groups close to requested size.
    #

    random.shuffle(groups)

    target_valid = round(
        n * args.valid_ratio
    )

    #
    # Sort with random tie-breaking but prefer
    # smaller groups when filling validation,
    # so one giant sequence doesn't dominate.
    #

    groups.sort(
        key=lambda g: (
            len(g),
            random.random(),
        )
    )

    valid_groups = []
    valid_count = 0

    for group in groups:

        if valid_count >= target_valid:
            break

        remaining = (
            target_valid
            - valid_count
        )

        #
        # Add it if it gets us reasonably
        # closer to the target.
        #

        before_error = abs(
            target_valid
            - valid_count
        )

        after_error = abs(
            target_valid
            - (
                valid_count
                + len(group)
            )
        )

        if (
            after_error <= before_error
            or valid_count
            < target_valid * 0.75
        ):
            valid_groups.append(
                group
            )

            valid_count += len(
                group
            )

    valid_indexes = set()

    for group in valid_groups:
        valid_indexes.update(group)

    train_indexes = set(
        range(n)
    ) - valid_indexes

    #
    # Extra safety:
    # verify no pHash <= threshold pair
    # crosses split.
    #

    print(
        "Checking split leakage..."
    )

    leakage = []

    train_list = sorted(
        train_indexes
    )

    valid_list = sorted(
        valid_indexes
    )

    for i in train_list:

        diff = (
            hashes[valid_list]
            != hashes[i]
        )

        distances = np.count_nonzero(
            diff,
            axis=1,
        )

        bad = np.where(
            distances
            <= args.phash_threshold
        )[0]

        for pos in bad:
            leakage.append(
                (
                    i,
                    valid_list[
                        int(pos)
                    ],
                    int(
                        distances[pos]
                    ),
                )
            )

    if leakage:
        raise RuntimeError(
            "Cross-split pHash leakage "
            f"detected: {leakage[:10]}"
        )

    #
    # Write
    #

    train_counts = write_split(
        "train",
        train_indexes,
        images,
        annotations_by_image,
        categories,
        input_root,
        output_root,
    )

    valid_counts = write_split(
        "valid",
        valid_indexes,
        images,
        annotations_by_image,
        categories,
        input_root,
        output_root,
    )

    #
    # Report
    #

    report = []

    report.append(
        "=== VEO V14 group-aware split ==="
    )

    report.append(
        f"pHash threshold: "
        f"{args.phash_threshold}"
    )

    report.append(
        f"seed: {args.seed}"
    )

    report.append("")

    report.append(
        f"Total images: {n}"
    )

    report.append(
        f"Groups: {len(groups)}"
    )

    report.append(
        f"Train images: "
        f"{len(train_indexes)}"
    )

    report.append(
        f"Valid images: "
        f"{len(valid_indexes)}"
    )

    report.append("")

    report.append(
        "Train annotations:"
    )

    for name, count in sorted(
        train_counts.items()
    ):
        report.append(
            f"  {name:12s}: {count}"
        )

    report.append("")

    report.append(
        "Valid annotations:"
    )

    for name, count in sorted(
        valid_counts.items()
    ):
        report.append(
            f"  {name:12s}: {count}"
        )

    report.append("")

    report.append(
        "Cross-split near-duplicate "
        "leakage: 0"
    )

    text = "\n".join(
        report
    )

    print()
    print(text)

    (
        output_root /
        "split_report.txt"
    ).write_text(
        text + "\n"
    )


if __name__ == "__main__":
    main()

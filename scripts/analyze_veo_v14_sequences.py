#!/usr/bin/env python3

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------
# Perceptual hash
# ---------------------------------------------------------

def phash(image_path: Path) -> np.ndarray:
    """
    64-bit perceptual hash based on DCT.
    Returns bool array of length 64.
    """

    img = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if img is None:
        raise RuntimeError(
            f"Cannot read {image_path}"
        )

    img = cv2.resize(
        img,
        (32, 32),
        interpolation=cv2.INTER_AREA,
    )

    img = np.float32(img)

    dct = cv2.dct(img)

    low = dct[:8, :8]

    # Ignore DC coefficient when estimating median.
    values = low.flatten()
    median = np.median(values[1:])

    return (
        low > median
    ).flatten()


def hamming(a, b):
    return int(
        np.count_nonzero(a != b)
    )


def sha256(path: Path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# ---------------------------------------------------------
# Union-find
# ---------------------------------------------------------

class UnionFind:

    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = (
                self.parent[
                    self.parent[x]
                ]
            )
            x = self.parent[x]

        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


# ---------------------------------------------------------
# Contact sheets
# ---------------------------------------------------------

def make_sheet(
    records,
    output,
    cols=4,
    thumb_width=480,
):
    images = []

    for record in records:

        img = cv2.imread(
            str(record["path"])
        )

        if img is None:
            continue

        h, w = img.shape[:2]

        scale = thumb_width / w

        thumb = cv2.resize(
            img,
            (
                thumb_width,
                int(h * scale),
            ),
            interpolation=cv2.INTER_AREA,
        )

        label = (
            f"ID {record['id']} | "
            f"{record['source_split']}"
        )

        cv2.rectangle(
            thumb,
            (0, 0),
            (thumb_width, 32),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            thumb,
            label,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        images.append(thumb)

    if not images:
        return

    cell_h = max(
        img.shape[0]
        for img in images
    )

    rows = (
        len(images) + cols - 1
    ) // cols

    sheet = np.zeros(
        (
            rows * cell_h,
            cols * thumb_width,
            3,
        ),
        dtype=np.uint8,
    )

    sheet[:] = 230

    for i, img in enumerate(images):

        row = i // cols
        col = i % cols

        y = row * cell_h
        x = col * thumb_width

        ih, iw = img.shape[:2]

        sheet[
            y:y + ih,
            x:x + iw,
        ] = img

    cv2.imwrite(
        str(output),
        sheet,
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset",
        type=Path,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/dataset_inspection/"
            "veo_v14_sequences"
        ),
    )

    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[6, 8, 10, 12],
    )

    parser.add_argument(
        "--max-groups",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    root = args.dataset.resolve()

    annotations_path = (
        root /
        "annotations.coco.json"
    )

    data = json.loads(
        annotations_path.read_text()
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    print(
        "=== Loading images ==="
    )

    for image in data["images"]:

        path = (
            root /
            "images" /
            image["file_name"]
        )

        records.append(
            {
                "id": image["id"],
                "file_name":
                    image["file_name"],
                "source_file_name":
                    image.get(
                        "source_file_name",
                        "",
                    ),
                "source_split":
                    image.get(
                        "source_split",
                        "?",
                    ),
                "path": path,
            }
        )

    n = len(records)

    print(
        f"Images: {n}"
    )

    # -----------------------------------------------------
    # SHA256 exact duplicates
    # -----------------------------------------------------

    print()
    print(
        "=== Exact duplicate scan ==="
    )

    sha_groups = defaultdict(list)

    for i, record in enumerate(records):

        digest = sha256(
            record["path"]
        )

        record["sha256"] = digest

        sha_groups[
            digest
        ].append(i)

        if (
            (i + 1) % 100 == 0
            or i + 1 == n
        ):
            print(
                f"\rSHA256: {i + 1}/{n}",
                end="",
                flush=True,
            )

    print()

    exact_duplicates = [
        indexes
        for indexes
        in sha_groups.values()
        if len(indexes) > 1
    ]

    print(
        "Exact duplicate groups:",
        len(exact_duplicates),
    )

    print(
        "Images involved:",
        sum(
            len(x)
            for x in exact_duplicates
        ),
    )

    # -----------------------------------------------------
    # pHash
    # -----------------------------------------------------

    print()
    print(
        "=== Computing pHash ==="
    )

    hashes = []

    for i, record in enumerate(records):

        hashes.append(
            phash(
                record["path"]
            )
        )

        if (
            (i + 1) % 100 == 0
            or i + 1 == n
        ):
            print(
                f"\rpHash: {i + 1}/{n}",
                end="",
                flush=True,
            )

    print()

    hashes = np.stack(hashes)

    # -----------------------------------------------------
    # Pairwise distances
    # -----------------------------------------------------

    print()
    print(
        "=== Pairwise distances ==="
    )

    distances = np.full(
        (n, n),
        64,
        dtype=np.uint8,
    )

    np.fill_diagonal(
        distances,
        0,
    )

    for i in range(n):

        diff = (
            hashes[i + 1:]
            != hashes[i]
        )

        if len(diff) == 0:
            continue

        d = np.count_nonzero(
            diff,
            axis=1,
        )

        distances[
            i,
            i + 1:
        ] = d

        distances[
            i + 1:,
            i
        ] = d

    # -----------------------------------------------------
    # Nearest-neighbour statistics
    # -----------------------------------------------------

    nearest = []

    for i in range(n):

        row = distances[i].copy()

        row[i] = 255

        nearest.append(
            int(row.min())
        )

    nearest = np.array(nearest)

    print()
    print(
        "Nearest-neighbour pHash distance:"
    )

    for percentile in [
        0,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
        100,
    ]:

        value = np.percentile(
            nearest,
            percentile,
        )

        print(
            f"  p{percentile:3d}: "
            f"{value:.1f}"
        )

    # -----------------------------------------------------
    # Candidate groups at several thresholds
    # -----------------------------------------------------

    report_lines = []

    report_lines.append(
        "Veo v14 sequence similarity audit"
    )

    report_lines.append(
        f"Images: {n}"
    )

    report_lines.append(
        f"Exact duplicate groups: "
        f"{len(exact_duplicates)}"
    )

    report_lines.append("")

    for threshold in args.thresholds:

        print()
        print(
            f"=== threshold <= "
            f"{threshold} ==="
        )

        uf = UnionFind(n)

        edge_count = 0

        for i in range(n):

            js = np.where(
                distances[i, i + 1:]
                <= threshold
            )[0]

            for relative_j in js:

                j = (
                    i + 1
                    + int(relative_j)
                )

                uf.union(i, j)
                edge_count += 1

        components = defaultdict(list)

        for i in range(n):

            components[
                uf.find(i)
            ].append(i)

        groups = [
            indexes
            for indexes
            in components.values()
            if len(indexes) > 1
        ]

        groups.sort(
            key=len,
            reverse=True,
        )

        isolated = sum(
            1
            for indexes
            in components.values()
            if len(indexes) == 1
        )

        print(
            f"Edges: {edge_count}"
        )

        print(
            f"Groups (>1): "
            f"{len(groups)}"
        )

        print(
            f"Isolated images: "
            f"{isolated}"
        )

        print(
            "Largest groups:",
            [
                len(g)
                for g in groups[:15]
            ],
        )

        report_lines.append(
            f"Threshold <= {threshold}"
        )

        report_lines.append(
            f"  edges: {edge_count}"
        )

        report_lines.append(
            f"  groups: {len(groups)}"
        )

        report_lines.append(
            f"  isolated: {isolated}"
        )

        report_lines.append(
            "  largest: "
            + str(
                [
                    len(g)
                    for g
                    in groups[:20]
                ]
            )
        )

        report_lines.append("")

        threshold_dir = (
            args.output /
            f"threshold_{threshold:02d}"
        )

        threshold_dir.mkdir(
            exist_ok=True,
        )

        # Create contact sheets for
        # largest candidate groups.

        for group_number, indexes in enumerate(
            groups[:args.max_groups],
            start=1,
        ):

            selected_records = [
                records[i]
                for i in indexes
            ]

            make_sheet(
                selected_records,
                threshold_dir /
                (
                    f"group_"
                    f"{group_number:03d}_"
                    f"n{len(indexes):03d}.jpg"
                ),
            )

    # -----------------------------------------------------
    # Closest pairs
    # -----------------------------------------------------

    pairs = []

    for i in range(n):
        for j in range(i + 1, n):

            pairs.append(
                (
                    int(
                        distances[i, j]
                    ),
                    i,
                    j,
                )
            )

    pairs.sort(
        key=lambda x: x[0]
    )

    closest_dir = (
        args.output /
        "closest_pairs"
    )

    closest_dir.mkdir(
        exist_ok=True,
    )

    for rank, (
        distance,
        i,
        j,
    ) in enumerate(
        pairs[:50],
        start=1,
    ):

        make_sheet(
            [
                records[i],
                records[j],
            ],
            closest_dir /
            (
                f"{rank:03d}_"
                f"d{distance:02d}.jpg"
            ),
            cols=2,
            thumb_width=700,
        )

    # -----------------------------------------------------
    # Original filenames
    # -----------------------------------------------------

    filename_report = (
        args.output /
        "source_filenames.txt"
    )

    with filename_report.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in records:

            f.write(
                f"{record['id']:04d}\t"
                f"{record['source_split']}\t"
                f"{record['source_file_name']}\n"
            )

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    report_lines.append(
        "Nearest-neighbour percentiles"
    )

    for percentile in [
        0,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
        100,
    ]:

        report_lines.append(
            f"  p{percentile:3d}: "
            f"{np.percentile(nearest, percentile):.1f}"
        )

    report_path = (
        args.output /
        "report.txt"
    )

    report_path.write_text(
        "\n".join(report_lines)
        + "\n"
    )

    print()
    print(
        "=== DONE ==="
    )

    print(
        args.output.resolve()
    )

    print()
    print(
        "Important files:"
    )

    print(
        "  report.txt"
    )

    print(
        "  source_filenames.txt"
    )

    print(
        "  closest_pairs/"
    )

    for threshold in args.thresholds:
        print(
            f"  threshold_{threshold:02d}/"
        )


if __name__ == "__main__":
    main()

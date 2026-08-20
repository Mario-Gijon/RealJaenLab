#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from PIL import Image

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from torchvision import transforms


def make_contact_sheet(
    records,
    output_path,
    cols=4,
    thumb_width=420,
):
    images = []

    for record in records:
        img = cv2.imread(str(record["path"]))

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
            (thumb_width, 30),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            thumb,
            label,
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
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

    sheet = np.full(
        (
            rows * cell_h,
            cols * thumb_width,
            3,
        ),
        235,
        dtype=np.uint8,
    )

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
        str(output_path),
        sheet,
    )


def load_dataset(root):
    data = json.loads(
        (
            root /
            "annotations.coco.json"
        ).read_text()
    )

    records = []

    for image in data["images"]:
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
                "path":
                    root /
                    "images" /
                    image["file_name"],
            }
        )

    return records


def extract_embeddings(
    records,
    model,
    transform,
    device,
    batch_size,
):
    embeddings = []

    for start in range(
        0,
        len(records),
        batch_size,
    ):
        batch_records = records[
            start:start + batch_size
        ]

        batch = []

        for record in batch_records:
            image = Image.open(
                record["path"]
            ).convert("RGB")

            batch.append(
                transform(image)
            )

        x = torch.stack(batch).to(device)

        with torch.inference_mode():

            if device.type == "cuda":
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    features = model(x)
            else:
                features = model(x)

        features = features.float()

        features = torch.nn.functional.normalize(
            features,
            p=2,
            dim=1,
        )

        embeddings.append(
            features.cpu().numpy()
        )

        done = min(
            start + batch_size,
            len(records),
        )

        print(
            f"\rEmbeddings: "
            f"{done}/{len(records)}",
            end="",
            flush=True,
        )

    print()

    return np.concatenate(
        embeddings,
        axis=0,
    )


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
            "veo_v14_dino"
        ),
    )

    parser.add_argument(
        "--min-clusters",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-clusters",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    root = args.dataset.resolve()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = load_dataset(root)

    print(
        f"Images: {len(records)}"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    #
    # DINOv2
    #

    print()
    print(
        "Loading DINOv2 ViT-S/14..."
    )

    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
    )

    model = model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(
                256,
                interpolation=
                    transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    embeddings_path = (
        args.output /
        "embeddings.npy"
    )

    if embeddings_path.exists():

        print(
            "Loading cached embeddings..."
        )

        embeddings = np.load(
            embeddings_path
        )

    else:

        print(
            "Extracting embeddings..."
        )

        embeddings = extract_embeddings(
            records,
            model,
            transform,
            device,
            args.batch_size,
        )

        np.save(
            embeddings_path,
            embeddings,
        )

    print(
        "Embedding shape:",
        embeddings.shape,
    )

    #
    # Try several cluster counts.
    #

    report = []

    report.append(
        "Veo v14 - DINOv2 clustering"
    )

    report.append(
        f"Images: {len(records)}"
    )

    report.append(
        f"Embedding dimension: "
        f"{embeddings.shape[1]}"
    )

    report.append("")

    results = []

    for k in range(
        args.min_clusters,
        args.max_clusters + 1,
    ):

        clustering = (
            AgglomerativeClustering(
                n_clusters=k,
                metric="cosine",
                linkage="average",
            )
        )

        labels = clustering.fit_predict(
            embeddings
        )

        score = silhouette_score(
            embeddings,
            labels,
            metric="cosine",
        )

        counts = Counter(labels)

        sizes = sorted(
            counts.values(),
            reverse=True,
        )

        results.append(
            (
                score,
                k,
                labels,
                sizes,
            )
        )

        print(
            f"k={k:2d} | "
            f"silhouette={score:.4f} | "
            f"sizes={sizes}"
        )

        report.append(
            f"k={k:2d} "
            f"silhouette={score:.4f} "
            f"sizes={sizes}"
        )

        #
        # Contact sheets
        #

        k_dir = (
            args.output /
            f"k_{k:02d}"
        )

        k_dir.mkdir(
            exist_ok=True,
        )

        for cluster_id in sorted(
            set(labels)
        ):

            indexes = np.where(
                labels == cluster_id
            )[0]

            cluster_records = [
                records[i]
                for i in indexes
            ]

            cluster_records.sort(
                key=lambda x: x["id"]
            )

            source_counts = Counter(
                r["source_split"]
                for r
                in cluster_records
            )

            make_contact_sheet(
                cluster_records,
                k_dir /
                (
                    f"cluster_"
                    f"{cluster_id:02d}_"
                    f"n{len(cluster_records):03d}.jpg"
                ),
            )

            metadata = {
                "cluster": int(
                    cluster_id
                ),
                "size": len(
                    cluster_records
                ),
                "source_splits":
                    dict(source_counts),
                "image_ids": [
                    int(r["id"])
                    for r
                    in cluster_records
                ],
            }

            (
                k_dir /
                (
                    f"cluster_"
                    f"{cluster_id:02d}.json"
                )
            ).write_text(
                json.dumps(
                    metadata,
                    indent=2,
                )
            )

    #
    # Best mathematical silhouette
    #

    results.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    best_score, best_k, _, _ = (
        results[0]
    )

    report.append("")
    report.append(
        f"Best silhouette: "
        f"k={best_k} "
        f"score={best_score:.4f}"
    )

    (
        args.output /
        "report.txt"
    ).write_text(
        "\n".join(report)
        + "\n"
    )

    print()
    print(
        "Best mathematical k:",
        best_k,
    )

    print(
        "Silhouette:",
        f"{best_score:.4f}",
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Best silhouette is NOT "
        "automatically the correct "
        "number of matches."
    )

    print()
    print(
        "Output:",
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()

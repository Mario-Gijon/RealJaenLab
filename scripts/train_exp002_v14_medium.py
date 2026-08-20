#!/usr/bin/env python3
from pathlib import Path

import torch
from rfdetr import RFDETRMedium


DATASET = Path("datasets/veo_football_detection_v14_clean")
PRETRAINED = Path(
    "weights/pretrained/rf-detr-medium-1.8.3.pth"
)
OUTPUT = Path(
    "experiments/exp002_veo_v14_medium"
)


def main():
    print("=== RealJaenLab EXP_002 ===")
    print("Experiment: RF-DETR Medium baseline")
    print(f"Dataset:    {DATASET.resolve()}")
    print(f"Pretrained: {PRETRAINED.resolve()}")
    print(f"Output:     {OUTPUT.resolve()}")
    print()

    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    if not PRETRAINED.exists():
        raise FileNotFoundError(PRETRAINED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = RFDETRMedium(
        num_classes=5,
        pretrain_weights=PRETRAINED,
        gradient_checkpointing=True,
    )

    model.train(
        # ---------------------------
        # Dataset
        # ---------------------------
        dataset_dir=DATASET,
        dataset_file="roboflow",

        class_names=[
            "ball",
            "linesman",
            "referee",
            "team_a",
            "team_b",
        ],

        output_dir=OUTPUT,

        # ---------------------------
        # Training length
        # ---------------------------
        epochs=200,

        # ---------------------------
        # Batch
        # effective batch = 2 * 8 = 16
        # ---------------------------
        batch_size=2,
        grad_accum_steps=8,

        # ---------------------------
        # Optimisation
        # RF-DETR 1.8.3 defaults,
        # made explicit
        # ---------------------------
        lr=1e-4,
        lr_encoder=1.5e-4,
        weight_decay=1e-4,

        lr_scheduler="step",
        lr_drop=100,

        # ---------------------------
        # Resolution/training scale
        # ---------------------------
        resolution=576,
        multi_scale=True,
        expanded_scales=True,
        do_random_resize_via_padding=False,

        # ---------------------------
        # Augmentation
        # Keep stock 1.8.3 behaviour
        # for EXP_002
        # ---------------------------
        aug_config=None,
        augmentation_backend="cpu",

        # ---------------------------
        # EMA
        # ---------------------------
        use_ema=True,
        ema_decay=0.993,
        ema_tau=100,

        # ---------------------------
        # GPU / precision
        # ---------------------------
        accelerator="gpu",
        devices=1,
        amp_dtype="auto",

        # ---------------------------
        # Reproducibility
        # ---------------------------
        seed=42,

        # ---------------------------
        # Validation/checkpoints
        # ---------------------------
        eval_interval=1,
        checkpoint_interval=10,

        # Intentionally disabled.
        # We want the complete learning curve.
        early_stopping=False,

        # Do NOT touch test while tuning.
        run_test=False,

        # ---------------------------
        # Logging
        # ---------------------------
        tensorboard=True,
        wandb=False,
        progress_bar="tqdm",

        log_per_class_metrics=True,
        save_dataset_grids=True,

        notes={
            "experiment": "EXP_002",
            "purpose": "RF-DETR Medium stock baseline",
            "dataset": "veo_football_detection_v14_clean",
            "dataset_images": 847,
            "classes": [
                "ball",
                "linesman",
                "referee",
                "team_a",
                "team_b",
            ],
            "seed": 42,
        },
    )


if __name__ == "__main__":
    main()

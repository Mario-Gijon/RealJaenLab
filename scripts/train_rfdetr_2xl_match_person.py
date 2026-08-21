#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


DEFAULT_DATASET = Path("datasets/processed/match_person_v1")
DEFAULT_OUTPUT = Path("experiments/match_person_v1_rfdetr_2xl_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune RF-DETR 2XL on RealJaenLab match_person_v1."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Dataset root (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Training output directory (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum epochs (default: 30).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=880,
        help="Square training resolution (default: native 2XL 880).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Physical batch size (default: 1 for 12 GB VRAM).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=16,
        help="Gradient accumulation steps (default: 16).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Main learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--lr-encoder",
        type=float,
        default=1.5e-4,
        help="Encoder learning rate (default: 1.5e-4).",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save periodic full checkpoint every N epochs (default: 5).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early stopping patience (default: 7 epochs).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from an RF-DETR .ckpt checkpoint.",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Disable EMA to reduce VRAM if necessary.",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing.",
    )
    return parser.parse_args()


def validate_dataset(root: Path) -> None:
    required = [
        root / "train" / "_annotations.coco.json",
        root / "valid" / "_annotations.coco.json",
        root / "test" / "_annotations.coco.json",
    ]

    missing = [path for path in required if not path.is_file()]

    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Dataset is incomplete. Missing:\n" + lines
        )


def print_environment() -> None:
    print("=== ENVIRONMENT ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"VRAM: {vram_gb:.2f} GiB")

    print()


def main() -> None:
    args = parse_args()

    validate_dataset(args.dataset)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. RF-DETR 2XL training should run on the NVIDIA GPU."
        )

    print_environment()

    try:
        from rfdetr import RFDETR2XLarge
    except ImportError as exc:
        raise RuntimeError(
            "RFDETR2XLarge could not be imported. "
            "Install/activate the environment containing rfdetr[plus]."
        ) from exc

    use_ema = not args.no_ema
    gradient_checkpointing = not args.no_gradient_checkpointing

    effective_batch = args.batch_size * args.grad_accum

    print("=== TRAINING PLAN ===")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print("Model: RF-DETR 2XL")
    print("Classes: 1 (match_person)")
    print(f"Resolution: {args.resolution}x{args.resolution} (static)")
    print("Multi-scale training: False")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.grad_accum}")
    print(f"Effective batch size: {effective_batch}")
    print(f"Gradient checkpointing: {gradient_checkpointing}")
    print(f"EMA: {use_ema}")
    print(f"Max epochs: {args.epochs}")
    print(f"Early-stopping patience: {args.patience}")
    print()

    args.output.mkdir(parents=True, exist_ok=True)

    # gradient_checkpointing is a model-constructor parameter in current RF-DETR.
    model = RFDETR2XLarge(
        num_classes=1,
        resolution=args.resolution,
        gradient_checkpointing=gradient_checkpointing,
    )

    train_kwargs = dict(
        dataset_dir=str(args.dataset),
        output_dir=str(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        resolution=args.resolution,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        device="cuda",
        num_workers=args.num_workers,
        amp_dtype="auto",
        use_ema=use_ema,
        checkpoint_interval=args.checkpoint_interval,
        early_stopping=True,
        early_stopping_patience=args.patience,
        early_stopping_min_delta=0.001,
        early_stopping_use_ema=use_ema,
        # Fine-tuning can temporarily dip below the pretrained epoch-0
        # metric while adapting the detection head. Ignore the first two
        # epochs for best-checkpoint selection / early stopping.
        skip_best_epochs=2,
        tensorboard=False,
        wandb=False,
        seed=42,
        progress_bar="tqdm",
    )

    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(args.resume)
        train_kwargs["resume"] = str(args.resume)

    print("Starting training...")
    print()

    try:
        model.train(**train_kwargs)
    except torch.cuda.OutOfMemoryError as exc:
        print()
        print("CUDA OUT OF MEMORY")
        print("------------------")
        print(
            "The 2XL configuration did not fit in VRAM. "
            "Try the same command with:"
        )
        print()
        print("  --resolution 704 --no-ema")
        print()
        print(
            "Keep --batch-size 1. Do not reduce gradient accumulation "
            "for VRAM; accumulation does not increase per-step activation memory."
        )
        raise exc

    print()
    print("Training finished.")
    print(f"Checkpoints/results: {args.output}")


if __name__ == "__main__":
    main()
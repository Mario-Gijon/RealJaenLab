#!/usr/bin/env python3

from pathlib import Path

import torch
from rfdetr import RFDETRMedium


DATASET = Path("datasets/football_players_v20_clean")
OUTPUT = Path("experiments/smoke_rfdetr_medium")


def main():
    print("=== RealJaenLab RF-DETR smoke test ===")
    print(f"dataset: {DATASET.resolve()}")
    print(f"output:  {OUTPUT.resolve()}")
    print(f"cuda:    {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    print(f"gpu:     {torch.cuda.get_device_name(0)}")

    model = RFDETRMedium(
        gradient_checkpointing=True,
    )

    model.train(
        dataset_dir=str(DATASET),
        output_dir=str(OUTPUT),

        # Sólo queremos comprobar el pipeline.
        epochs=1,

        # Configuración adecuada para ~12 GB VRAM.
        batch_size=2,
        grad_accum_steps=8,

        # Mismo tamaño del dataset de Roboflow.
        resolution=576,

        # Forzar nuestra GPU.
        device="cuda",

        # Reproducibilidad.
        seed=42,

        # Queremos validación al terminar este epoch.
        eval_interval=1,

        # Guardado/logs.
        checkpoint_interval=1,
        tensorboard=True,
        wandb=False,

        # Dejamos EMA activa.
        use_ema=True,

        # Precisión automática.
        amp_dtype="auto",
    )


if __name__ == "__main__":
    main()

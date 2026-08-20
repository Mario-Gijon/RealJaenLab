# RealJaenLab

RealJaenLab es un repositorio de experimentación para análisis de fútbol mediante visión por computador.

El objetivo de esta primera fase ha sido construir y comparar detectores capaces de localizar elementos relevantes del juego en grabaciones de fútbol, especialmente en vídeos obtenidos con cámaras de tipo Veo.

El trabajo realizado incluye preparación de datasets, entrenamiento de modelos RF-DETR, comparación de distintas configuraciones e inferencia sobre partidos reales del Real Jaén.

## Estructura del repositorio

```text
RealJaenLab/
├── configs/        # Configuración y dependencias de experimentos
├── datasets/       # Datasets locales utilizados durante entrenamiento
├── docs/           # Documentación técnica
├── experiments/    # Resultados, métricas y checkpoints de entrenamiento
├── input/          # Vídeos utilizados para pruebas
├── outputs/        # Vídeos procesados y detecciones exportadas
├── scripts/        # Scripts de preparación, entrenamiento e inferencia
└── weights/        # Pesos preentrenados y modelos congelados
```

Los directorios que contienen datasets, vídeos, resultados pesados y checkpoints no están pensados para almacenarse directamente en el repositorio Git.

## Entorno

El desarrollo principal se ha realizado sobre Linux/WSL2 con GPU NVIDIA.

Configuración utilizada durante los experimentos:

```text
Python 3.12
RF-DETR 1.8.3
PyTorch 2.13
CUDA
NVIDIA GeForce RTX 4070 SUPER
```

El entorno virtual utilizado durante el desarrollo se encuentra en `.venv/`.

Para activarlo:

```bash
source .venv/bin/activate
```

## Datasets

Durante esta fase se han utilizado principalmente dos datasets.

### Football Players Detection v20

Dataset con cuatro clases:

```text
ball
goalkeeper
player
referee
```

Se utiliza una versión revisada y normalizada en:

```text
datasets/football_players_v20_clean/
```

### Veo Football Detection v14

Dataset con imágenes 1920x1080 y una perspectiva próxima a grabaciones panorámicas de fútbol.

Las clases utilizadas son:

```text
ball
linesman
referee
team_a
team_b
```

La versión preparada para entrenamiento se encuentra en:

```text
datasets/veo_football_detection_v14_clean/
```

El split de entrenamiento y validación se generó evitando que imágenes casi idénticas detectadas mediante perceptual hash quedaran en conjuntos diferentes.

Más información sobre la preparación de los datasets:

[docs/datasets.md](docs/datasets.md)

## Modelos entrenados

Se han comparado RF-DETR Medium y RF-DETR Large sobre ambos datasets.

| Experimento | Dataset | Arquitectura | Mejor mAP50:95 |
|---|---|---|---:|
| EXP_001 | Football Players v20 | RF-DETR Medium | 0.4696 |
| EXP_002 | Veo v14 | RF-DETR Medium | ~0.6326 |
| EXP_003 | Football Players v20 | RF-DETR Large | ~0.5000 |
| EXP_004 | Veo v14 | RF-DETR Large | 0.6585 |

Las métricas entre datasets distintos no son directamente comparables porque utilizan diferentes conjuntos de validación y taxonomías de clases.

Dentro de cada dataset, RF-DETR Large mejoró los resultados obtenidos con RF-DETR Medium.

El detector principal utilizado actualmente es:

```text
weights/models/football_detector_exp004_v14_large.pth
```

SHA256:

```text
fc0062fcd0edc7547200d857d5e5209c48051e331e16a09d6af1cd8d7a5d950b
```

Este modelo corresponde al mejor checkpoint EMA de EXP_004.

La configuración completa y los resultados de cada experimento están documentados en:

[docs/experiments.md](docs/experiments.md)

## Inferencia sobre vídeo

El script principal de inferencia para EXP_004 es:

```text
scripts/test_exp004_v14_large_video.py
```

Ejemplo:

```bash
python scripts/test_exp004_v14_large_video.py \
  --video input/better_look_match.mp4 \
  --infer-fps 20 \
  --predict-threshold 0.15 \
  --display-threshold 0.60 \
  --output outputs/better_look_exp004_v14_large.mp4 \
  --csv outputs/better_look_exp004_v14_large.csv
```

La inferencia genera:

- un vídeo con las detecciones dibujadas;
- un CSV con las detecciones producidas por el modelo.

Las detecciones incluyen clase, confianza y bounding box para cada frame procesado.

Los thresholds utilizados normalmente durante las pruebas son:

```text
predict_threshold = 0.15
display_threshold = 0.60
```

El primero determina qué candidatos se conservan. El segundo controla únicamente qué detecciones se muestran visualmente en el vídeo generado.

Más información:

[docs/inference.md](docs/inference.md)

## Scripts principales

Entre los scripts utilizados durante esta fase se encuentran:

```text
scripts/audit_dataset.py
scripts/clean_veo_v14.py
scripts/analyze_veo_v14_sequences.py
scripts/split_veo_v14.py

scripts/train_exp001.py
scripts/train_exp002_v14_medium.py
scripts/train_exp003_v20_large.py
scripts/train_exp004_v14_large.py

scripts/test_exp001_video.py
scripts/test_exp002_v14_medium_video.py
scripts/test_exp003_v20_large_video.py
scripts/test_exp004_v14_large_video.py
```

Los scripts de preparación permiten reconstruir las versiones utilizadas de los datasets a partir de las descargas originales.

## Documentación

La documentación técnica de esta fase se encuentra en:

```text
docs/
├── README.md
├── datasets.md
├── experiments.md
└── inference.md
```

Los documentos describen la preparación de los datos, los experimentos realizados y el proceso utilizado para probar los modelos sobre vídeo.
# RealJaenLab

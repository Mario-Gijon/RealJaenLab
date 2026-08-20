# Experimentos de detección

Los experimentos de esta fase se han realizado con RF-DETR 1.8.3 y PyTorch con CUDA.

Se han entrenado dos arquitecturas:

- RF-DETR Medium
- RF-DETR Large

Los modelos Medium utilizan una resolución de entrenamiento de 576 píxeles y los modelos Large una resolución de 704 píxeles.

Como configuración general se ha utilizado:

```text
batch_size = 2
grad_accum_steps = 8
effective_batch_size = 16

lr = 1e-4
lr_encoder = 1.5e-4
weight_decay = 1e-4

lr_scheduler = step
lr_drop = 100

multi_scale = True
expanded_scales = True

EMA enabled
seed = 42
```

Los entrenamientos completos se realizaron durante 200 épocas y se conservó el checkpoint con mejor resultado de validación.

## EXP_001 - Football Players v20 + RF-DETR Medium

Dataset:

```text
datasets/football_players_v20_clean/
```

Arquitectura:

```text
RF-DETR Medium
resolution = 576
num_classes = 4
```

Clases:

```text
ball
goalkeeper
player
referee
```

El mejor checkpoint regular apareció aproximadamente en la época 77.

Resultados principales:

```text
mAP50:95 = 0.4696
mAP50    = 0.8252
mAP75    = 0.4740
mAR      = 0.5309
```

AP50:95 por clase en el mejor checkpoint regular:

| Clase | AP50:95 |
|---|---:|
| ball | 0.1719 |
| goalkeeper | 0.5952 |
| player | 0.6152 |
| referee | 0.4960 |

Modelo congelado:

```text
weights/models/football_detector_exp001.pth
```

## EXP_002 - Veo Football Detection v14 + RF-DETR Medium

Dataset:

```text
datasets/veo_football_detection_v14_clean/
```

Arquitectura:

```text
RF-DETR Medium
resolution = 576
num_classes = 5
```

Clases:

```text
ball
linesman
referee
team_a
team_b
```

El mejor resultado total se obtuvo con EMA.

```text
best mAP50:95 ≈ 0.6326
```

El entrenamiento confirmó una mejora clara en la separación inicial entre los dos equipos respecto a los modelos que únicamente utilizan una clase general de jugador.

Modelo congelado:

```text
weights/models/football_detector_exp002_v14_medium.pth
```

SHA256:

```text
52f62f0a8081155811661793855ab4106db3bfe4a2004e0172311731076c2038
```

## EXP_003 - Football Players v20 + RF-DETR Large

Dataset:

```text
datasets/football_players_v20_clean/
```

Arquitectura:

```text
RF-DETR Large
resolution = 704
num_classes = 4
```

Clases:

```text
ball
goalkeeper
player
referee
```

El mejor checkpoint regular apareció alrededor de la época 42:

```text
mAP50:95 = 0.4934
mAP50    = 0.8639
mAP75    = 0.4892
mAR      = 0.5651
```

El mejor EMA alcanzó aproximadamente:

```text
mAP50:95 = 0.5000
```

La arquitectura Large mejoró los resultados obtenidos anteriormente con Medium sobre el mismo dataset.

En el mejor checkpoint regular:

| Clase | AP50:95 |
|---|---:|
| ball | 0.2011 |
| goalkeeper | 0.6073 |
| player | 0.6434 |
| referee | 0.5217 |

Modelo congelado:

```text
weights/models/football_detector_exp003_v20_large.pth
```

SHA256:

```text
b3e21cb76359ad0b79ba7ba46a430ecb8f536d0c206ed0f2abcedc70a29dc890
```

## EXP_004 - Veo Football Detection v14 + RF-DETR Large

Dataset:

```text
datasets/veo_football_detection_v14_clean/
```

Arquitectura:

```text
RF-DETR Large
resolution = 704
num_classes = 5
```

Clases:

```text
ball
linesman
referee
team_a
team_b
```

El mejor checkpoint regular se obtuvo en la época 25:

```text
mAP50:95 = 0.6501
mAP50    = 0.8964
mAP75    = 0.7414
mAR      = 0.7180
```

AP50:95 del modelo regular:

| Clase | AP50:95 |
|---|---:|
| ball | 0.3720 |
| linesman | 0.5450 |
| referee | 0.7795 |
| team_a | 0.7924 |
| team_b | 0.7617 |

El mejor resultado del experimento se obtuvo mediante EMA alrededor de la época 34:

```text
EMA mAP50:95 = 0.6585
EMA mAP50    = 0.9045
EMA mAR      = 0.7246
```

Al finalizar las 200 épocas:

```text
regular mAP50:95 = 0.6381
EMA mAP50:95     = 0.6393
```

Por tanto, para inferencia se utiliza el checkpoint correspondiente al mejor resultado total y no el estado final del entrenamiento.

Modelo congelado:

```text
weights/models/football_detector_exp004_v14_large.pth
```

SHA256:

```text
fc0062fcd0edc7547200d857d5e5209c48051e331e16a09d6af1cd8d7a5d950b
```

## Comparación

Resumen de los cuatro experimentos:

| Experimento | Dataset | Arquitectura | Mejor mAP50:95 |
|---|---|---|---:|
| EXP_001 | Football Players v20 | Medium | 0.4696 |
| EXP_002 | Veo v14 | Medium | 0.6326 |
| EXP_003 | Football Players v20 | Large | ~0.5000 |
| EXP_004 | Veo v14 | Large | 0.6585 |

Las métricas de V14 y Football Players v20 no deben compararse directamente porque utilizan conjuntos de validación y taxonomías diferentes.

Sí permiten comparar el efecto de la arquitectura dentro de un mismo dataset. En ambos casos RF-DETR Large produjo mejores resultados que RF-DETR Medium.

Para las pruebas actuales sobre grabaciones reales se utiliza EXP_004 como detector principal.

EXP_003 se conserva porque utiliza una taxonomía diferente y proporciona una clase específica para `goalkeeper`.

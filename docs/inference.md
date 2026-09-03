# Inferencia sobre vídeo

> **Historical note:** this page documents the original detector-inference
> phase and refers to local scripts/checkpoints that may not exist in the
> current checkout. For the current FIELD / CAMERA work, use the reproducible
> [FIELD / OBSERVER baseline](field-observer-baseline.md).

Los modelos se han probado sobre grabaciones reales del Real Jaén para comprobar su comportamiento fuera de los conjuntos utilizados durante el entrenamiento.

Las pruebas se realizan sobre vídeos almacenados en:

```text
input/
```

Los resultados se generan en:

```text
outputs/
```

Para cada ejecución se genera un vídeo con las detecciones dibujadas y un fichero CSV con las detecciones originales.

## EXP_004

Script principal:

```text
scripts/test_exp004_v14_large_video.py
```

Checkpoint:

```text
weights/models/football_detector_exp004_v14_large.pth
```

Clases:

```text
0 ball
1 linesman
2 referee
3 team_a
4 team_b
```

En las pruebas realizadas se ha utilizado normalmente:

```text
predict_threshold = 0.15
display_threshold = 0.60
```

El primer threshold determina qué detecciones se conservan como candidatos del modelo. El segundo únicamente determina qué detecciones se dibujan en el vídeo generado.

Esto permite conservar detecciones de menor confianza en el CSV sin mostrar todas ellas visualmente.

Ejemplo:

```bash
python scripts/test_exp004_v14_large_video.py \
  --video input/better_look_match.mp4 \
  --start 0 \
  --duration 52 \
  --infer-fps 20 \
  --predict-threshold 0.15 \
  --display-threshold 0.60 \
  --output outputs/better_look_exp004_v14_large_20fps_thr060.mp4 \
  --csv outputs/better_look_exp004_v14_large_20fps_thr060.csv
```

También se han utilizado fragmentos más largos para observar el comportamiento del modelo durante varios minutos consecutivos.

Las pruebas visuales muestran una detección consistente de los jugadores incluso cuando aparecen pequeños en la imagen.

La separación `team_a` y `team_b` funciona en gran parte de las secuencias, aunque una misma persona puede cambiar temporalmente de clase entre frames.

También aparecen detecciones sobre personas situadas fuera del terreno de juego.

El balón continúa siendo el objeto más difícil. El modelo puede localizar correctamente el balón durante varios frames consecutivos, pero también aparecen candidatos asociados a marcas del terreno de juego, botas u otros objetos pequeños.

Las clases producidas por el detector se consideran observaciones por frame. El resultado almacenado en los CSV conserva estas detecciones sin aplicar todavía seguimiento ni correcciones temporales.

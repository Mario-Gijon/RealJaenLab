# Datasets

Durante los primeros experimentos se han utilizado dos datasets principales.

## Football Players Detection v20

Dataset procedente de Roboflow Universe utilizado inicialmente como referencia para detección de objetos de fútbol.

La versión original contiene 372 imágenes distribuidas en:

| Split | Imágenes |
|---|---:|
| Train | 298 |
| Validation | 49 |
| Test | 25 |

Las imágenes tienen una resolución de 576x576.

Las clases utilizadas son:

| ID | Clase |
|---:|---|
| 0 | ball |
| 1 | goalkeeper |
| 2 | player |
| 3 | referee |

Se generó una versión limpia en:

```text
datasets/football_players_v20_clean/
```

El conjunto completo contiene 8.905 anotaciones:

| Clase | Train | Validation | Test |
|---|---:|---:|---:|
| ball | 258 | 45 | 24 |
| goalkeeper | 230 | 39 | 19 |
| player | 5955 | 973 | 499 |
| referee | 690 | 117 | 56 |

Este dataset resultó especialmente útil por separar explícitamente las clases `player`, `goalkeeper` y `referee`.

El balón es la clase con menor representación y también el objeto de menor tamaño en las imágenes.

## Veo Football Detection v14

El segundo dataset utilizado está formado por imágenes de fútbol con una perspectiva más próxima al tipo de grabación que se quiere analizar.

La descarga original contiene 847 imágenes de 1920x1080.

Las clases originales relevantes son:

```text
ball
linesman
person
referee
team_a
team_b
```

La clase `person` solamente contenía una anotación y no tenía suficiente representación para utilizarse durante el entrenamiento, por lo que se eliminó de la versión utilizada.

La versión limpia inicial se encuentra en:

```text
datasets/veo_football_detection_v14_clean_pool/
```

y contiene:

| Clase | Anotaciones |
|---|---:|
| ball | 619 |
| linesman | 383 |
| referee | 730 |
| team_a | 6498 |
| team_b | 6079 |

Total: 14.309 anotaciones.

Durante la revisión se detectó una fuerte similitud entre numerosas imágenes pertenecientes a las mismas secuencias de vídeo.

Para reducir la presencia de imágenes casi idénticas entre entrenamiento y validación se realizó un agrupamiento mediante perceptual hash. Las imágenes con una distancia pHash menor o igual a 6 se mantuvieron dentro del mismo grupo.

El split utilizado finalmente se encuentra en:

```text
datasets/veo_football_detection_v14_clean/
```

y fue:

| Split | Imágenes |
|---|---:|
| Train | 762 |
| Validation | 85 |

Distribución de anotaciones:

| Clase | Train | Validation |
|---|---:|---:|
| ball | 546 | 73 |
| linesman | 346 | 37 |
| referee | 651 | 79 |
| team_a | 5811 | 687 |
| team_b | 5432 | 647 |

No se utiliza un test interno para este dataset. La comparación final de los modelos se realiza también sobre vídeos externos del Real Jaén.

Una diferencia importante respecto al dataset v20 es que V14 separa directamente los dos equipos como `team_a` y `team_b`, pero no dispone de una clase específica para porteros.

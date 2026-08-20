# RealJaenLab

RealJaenLab se utiliza como entorno de experimentación para la detección de elementos de juego en grabaciones de fútbol.

Durante esta primera fase se ha trabajado principalmente en la detección de jugadores, equipos, árbitros, porteros y balón a partir de grabaciones con una perspectiva similar a la utilizada por cámaras Veo.

El trabajo realizado se ha centrado en tres partes:

- preparación y revisión de datasets;
- entrenamiento y comparación de distintas configuraciones de RF-DETR;
- validación de los modelos sobre grabaciones reales del Real Jaén.

Los experimentos se han realizado de forma local utilizando una NVIDIA GeForce RTX 4070 SUPER.

La documentación de esta fase se divide en:

- [Datasets](datasets.md)
- [Experimentos](experiments.md)
- [Inferencia sobre vídeo](inference.md)

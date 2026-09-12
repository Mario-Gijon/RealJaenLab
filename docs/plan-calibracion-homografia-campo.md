# RealJaenLab — Plan inicial de calibración del campo y homografía

## Estado

Este documento define el planteamiento inicial para la fase FIELD de RealJaenLab.

Estado actual:

`VIDEO FOUNDATION → FIELD → PEOPLE → POSITION → ROLE → ANALYTICS`

Por ahora solo existe la base de vídeo. El siguiente objetivo es FIELD.

En esta fase no queremos detectar jugadores, árbitros, equipos, hacer tracking ni analítica. El único objetivo es recuperar correctamente la geometría del terreno de juego en vídeos grabados con Veo.

> Dado un partido grabado con Veo, queremos estimar una relación fiable entre la imagen y un campo de fútbol canónico, de forma que podamos proyectar correctamente sus límites y líneas.

La primera implementación será **asistida por el usuario**. La automatización se añadirá progresivamente.

## 1. Objetivo principal

Para un fotograma seleccionado queremos obtener una transformación entre coordenadas de imagen y coordenadas reales del campo.

El resultado principal será una homografía:

- `H_pitch_to_image`
- `H_image_to_pitch`

Esto permitirá transformar:

```text
píxel de imagen (u, v)
        ↓
posición en el campo (x, y) en metros
```

y también:

```text
posición del campo (x, y)
        ↓
píxel de imagen (u, v)
```

Con una homografía fiable, RealJaenLab podrá proyectar sobre la imagen:

- líneas de banda;
- líneas de fondo;
- línea de medio campo;
- áreas de penalti;
- áreas pequeñas;
- círculo central;
- arcos de penalti;
- puntos de penalti;
- límites completos del terreno.

Una marca del campo no tiene que estar visible en el frame para poder inferirse si la transformación es correcta.

## 2. Primera versión: asistida por usuario

La primera versión **no debe intentar ser completamente automática**.

El usuario del RealJaen Supervisor proporcionará parte de la información geométrica necesaria para calcular la homografía.

La prioridad inicial será:

1. conseguir una solución geométricamente correcta;
2. hacer que la interacción sea rápida y cómoda;
3. validar que una homografía representa correctamente el vídeo Veo;
4. automatizar después las partes que realmente merezca la pena automatizar.

### Requisito de UX

Calibrar un frame no puede convertirse en una tarea de 20 o 30 minutos.

Objetivo inicial:

> Una calibración manual útil debería poder hacerse normalmente en unos 1–3 minutos cuando el usuario conozca el flujo.

No es un requisito rígido del primer prototipo, pero toda la interfaz debe diseñarse con esa dirección.

No queremos pedir al usuario decenas de puntos arbitrarios.

## 3. Interacción manual preferida

El usuario debería trabajar con **elementos semánticos del campo**.

Ejemplo:

```text
Seleccionar "Línea de medio campo"
        ↓
marcar dos o más puntos sobre ella

Seleccionar "Línea de banda cercana"
        ↓
marcar dos o más puntos

Seleccionar "Frontal del área"
        ↓
marcar dos o más puntos
```

Posibles elementos:

- línea de banda;
- línea de fondo;
- línea de medio campo;
- límites del área de penalti;
- límites del área pequeña;
- círculo central;
- arco de penalti;
- intersecciones conocidas cuando sean claramente visibles.

La aplicación debe conocer de antemano la geometría real correspondiente.

La idea es evitar un flujo tipo:

```text
Punto 1
Punto 2
Punto 3
Punto 4
...
```

Las líneas semánticas son más fáciles de identificar y aportan restricciones más fuertes.

## 4. Calibración mínima útil

El solver no debe exigir el mismo conjunto de marcas en todos los frames.

En una vista Veo puede aparecer:

- medio campo pero ningún área;
- un área pero no la línea central;
- solo parte de una banda;
- varias líneas sin intersecciones;
- fragmentos del círculo central;
- zoom fuerte;
- gran angular.

Por tanto, el sistema debe admitir diferentes combinaciones.

### Correspondencias de punto

```text
punto conocido del campo ↔ punto de imagen
```

Útiles para intersecciones y landmarks claros.

### Correspondencias de línea

```text
línea conocida del campo ↔ línea observada en la imagen
```

Deben ser una entrada de primera clase.

### Correspondencias de curva

En versiones posteriores podrán utilizarse:

- círculo central;
- arcos de penalti.

Estas curvas también aportan restricciones geométricas.

## 5. Modelo canónico del campo

RealJaenLab debe disponer de un modelo matemático del terreno en unidades reales.

Ejemplo:

```text
longitud = 105 m
anchura  = 68 m
```

El modelo debe definir matemáticamente:

```text
Campo
├── límites exteriores
├── línea de medio campo
├── círculo central
├── punto central
├── áreas de penalti
├── áreas pequeñas
├── puntos de penalti
└── arcos de penalti
```

Todas las coordenadas relevantes estarán expresadas en metros.

### Dimensiones por estadio

La arquitectura debe permitir diferentes dimensiones según el estadio.

En el futuro la biblioteca podría asociar a cada estadio:

```text
pitch_length
pitch_width
```

No es obligatorio implementarlo antes del primer prototipo si ralentiza el trabajo inicial.

## 6. Primer hito técnico: comprobar que una homografía es suficiente

Antes de automatizar líneas o movimiento de cámara, debemos responder:

> ¿Puede una sola homografía representar con precisión toda la geometría visible de un frame Veo?

Para varios frames:

1. realizar una calibración humana cuidadosa;
2. resolver la mejor homografía posible;
3. proyectar el campo completo;
4. comprobar el ajuste en toda la zona visible;
5. medir el error de reproyección.

Debe probarse con diferentes encuadres y, si es posible, distintos partidos.

Si una homografía no consigue alinear todas las líneas simultáneamente, podría haber:

- distorsión de lente;
- efectos de panorama/dewarping;
- estabilización de Veo;
- recortes digitales;
- otras transformaciones no puramente proyectivas.

En ese caso habría que estudiar algo como:

```text
frame Veo
   ↓
corrección/modelo de distorsión
   ↓
geometría proyectiva
   ↓
homografía
```

No debemos asumir que una homografía pura basta sin comprobarlo.

## 7. Solver manual V1

La primera implementación debe centrarse en conseguir la mejor calibración posible a partir de anotaciones humanas.

El solver debería combinar progresivamente:

```text
restricciones de puntos
+
restricciones de líneas
+
restricciones de curvas opcionales
```

Objetivo conceptual:

```text
error_total =
    error_reproyeccion_puntos
  + error_reproyeccion_lineas
  + error_reproyeccion_curvas
```

Conviene utilizar pérdidas robustas para que un clic ligeramente impreciso no arruine la solución.

La salida podría ser:

```text
CalibrationResult
├── H_pitch_to_image
├── H_image_to_pitch
├── métricas de reproyección
├── anotaciones utilizadas
├── geometría proyectada
└── indicadores de calidad/confianza
```

## 8. Workspace CALIBRACIÓN

`CALIBRACIÓN` debe convertirse en el editor principal.

Diseño orientativo:

```text
┌─────────────────────────────────────────────────────────┐
│ VIDEO | PREVIEW | CALIBRACIÓN                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│                FOTOGRAMA ACTUAL DEL VÍDEO              │
│                                                         │
│                 overlay del campo                      │
│                                                         │
├─────────────────────────────────────────────────────────┤
│ Elemento: Línea de medio campo                         │
│                                                         │
│ [Banda] [Medio campo] [Área] [...]                     │
│                                                         │
│ [Deshacer] [Borrar] [Resolver] [Aceptar]               │
├─────────────────────────────────────────────────────────┤
│ Error / confianza / estado                             │
└─────────────────────────────────────────────────────────┘
```

El primer prototipo puede ser mucho más simple.

Principio principal:

> El usuario identifica elementos del fútbol; la aplicación se encarga de la geometría.

## 9. Workspace PREVIEW

`PREVIEW` servirá para inspeccionar resultados sin mezclar todo con los controles de edición.

Podrá mostrar en el futuro:

```text
Frame original
Evidencia de líneas
Campo proyectado
Errores de reproyección
Mapa de confianza
Antes / después del refinamiento
```

Así `CALIBRACIÓN` queda para editar y `PREVIEW` para diagnosticar.

## 10. Validación desde el principio

Una calibración no se considera buena porque funcione en un único frame.

El benchmark interno deberá incluir progresivamente:

- vista central;
- vista izquierda;
- vista derecha;
- gran angular;
- zoom;
- área;
- medio campo;
- sol;
- sombra;
- distintos estadios;
- distintos partidos.

Y sobre todo:

> Una solución debe validarse en partidos que no se hayan utilizado para diseñarla.

El objetivo no es:

> funciona en Jaén vs Cacereño.

El objetivo es:

> funciona de forma fiable en partidos del Real Jaén grabados con Veo.

## 11. Métricas

No debemos depender solo de “se ve bien”.

Métricas útiles:

- error mediano de reproyección de líneas;
- p95 del error;
- error máximo relevante;
- número de marcas utilizadas;
- cobertura espacial;
- confianza de la calibración.

Objetivos iniciales orientativos para vídeo 1080p:

```text
error mediano calibración manual: <= 3 px
p95 calibración manual:           <= 8 px
```

Estos valores deben ajustarse después de medir vídeos Veo reales.

Una calibración incierta debe poder indicar que es incierta.

## 12. Hoja de ruta hacia automatización

La automatización se añadirá después de tener una buena base manual.

### Etapa A — Calibración humana Gold

```text
anotaciones humanas
      ↓
homografía de alta calidad
      ↓
overlay completo del campo
```

Objetivo: obtener un baseline geométrico fiable.

### Etapa B — Validación del modelo Veo

Comparar:

```text
solo homografía
```

frente a modelos con corrección de distorsión si fuera necesario.

### Etapa C — Evidencia de líneas

Construir un detector que responda:

> ¿Qué píxeles parecen pertenecer a marcas del campo?

Primero se puede probar visión clásica:

- brillo;
- saturación;
- contraste;
- bordes/ridges;
- morfología;
- Hough/LSD.

Más adelante puede utilizarse segmentación semántica.

No es necesario saber todavía qué línea concreta representa cada píxel.

### Etapa D — Refinamiento automático

```text
H aproximada
+
evidencia de líneas
      ↓
H refinada
```

Se proyecta el campo canónico y se ajusta contra las líneas observadas.

### Etapa E — Propagación temporal

Estimar movimiento entre frames cercanos usando contenido visual estático.

Primeros candidatos:

- SIFT;
- AKAZE;
- Lucas-Kanade;
- RANSAC.

Los jugadores se tratarán simplemente como outliers.

Más adelante podrán probarse matchers aprendidos si fuera necesario.

### Etapa F — Propagación entre anchors humanos

No propagar indefinidamente desde un solo frame.

Mejor:

```text
anchor HUMAN A
      ↓ hacia delante

frames intermedios

      ↑ hacia atrás
anchor HUMAN B
```

Las dos estimaciones pueden compararse y refinarse.

### Etapa G — Confianza

Cada frame debería terminar indicando su nivel de confianza mediante señales como:

- alineamiento de líneas;
- inliers del movimiento;
- distribución espacial de correspondencias;
- consistencia forward/backward;
- suavidad temporal;
- distancia al último anchor humano;
- plausibilidad geométrica.

Estados posibles:

```text
GOOD
REVIEW
BAD
```

Preferimos pedir intervención humana a aceptar silenciosamente una mala calibración.

### Etapa H — Partido completo asistido

Objetivo:

```text
partido nuevo
    ↓
usuario calibra unos pocos frames
    ↓
sistema propaga y refina
    ↓
marca momentos de baja confianza
    ↓
usuario corrige solo esos momentos
    ↓
calibración fiable del partido
```

## 13. Objetivo de intervención humana

No queremos calibrar miles de frames.

Flujo deseado:

```text
calibrar un frame
       ↓
procesamiento automático
       ↓
revisar solo zonas dudosas
       ↓
pequeña corrección
       ↓
continuar
```

Todavía no conocemos cuántas intervenciones serán necesarias.

Como objetivo aspiracional, podrían ser unas pocas por parte o partido.

Lo importante es que cada intervención sea **rápida, guiada y útil**.

## 14. Fuera de alcance

Durante FIELD no mezclaremos:

- detección de jugadores;
- detección de árbitros;
- clasificación de equipos;
- tracking;
- identificación;
- balón;
- heatmaps;
- distancias;
- análisis táctico.

El único problema es:

> Recuperar y seguir la geometría del terreno de juego en vídeo Veo.

## 15. Secuencia propuesta

```text
EXP-001
Calibración manual semántica + homografía de alta calidad

EXP-002
Validar el modelo de homografía en vídeo Veo

EXP-003
Optimizar la UX de calibración

EXP-004
Detector de evidencia de líneas

EXP-005
Refinamiento automático con líneas

EXP-006
Propagación temporal a corto plazo

EXP-007
Propagación entre anchors humanos

EXP-008
Confianza y detección de fallos

EXP-009
Calibración asistida de una parte / partido completo

EXP-010
Validación en partidos del Real Jaén no usados durante el desarrollo
```

La numeración es orientativa.

## 16. Primer objetivo de implementación

La próxima sesión debería comenzar por:

> Crear un flujo rápido de calibración manual de un único frame y resolver una homografía de alta calidad a partir de anotaciones semánticas del campo.

Todavía no intentaremos detectar líneas automáticamente.

El primer hito se cumple cuando:

1. el usuario abre un frame en `CALIBRACIÓN`;
2. marca pocos elementos conocidos;
3. pulsa Resolver;
4. RealJaenLab proyecta el campo completo;
5. el overlay se alinea correctamente;
6. el proceso es suficientemente rápido para usarlo de verdad.

Solo entonces empezaremos a automatizar.

## 17. Principios guía

### Corrección antes que automatización

Una solución asistida y correcta es mejor que una automática poco fiable.

### Interacción humana rápida

La calibración debe ser semántica, guiada y mínima.

### Generalización antes que éxito en un clip

Un método no está resuelto porque funcione en un solo partido.

### Geometría y evidencia son distintas

Los píxeles observados aportan evidencia; el modelo canónico aporta geometría.

### Aprovechar el vídeo

La continuidad temporal es una fuente de información fundamental.

### Los fallos deben ser visibles

Una calibración de baja confianza es aceptable. Una calibración mala aceptada silenciosamente no.

### Git es el archivo histórico

No reintroducir implementaciones antiguas simplemente porque existieran.

## 18. Definición de éxito para FIELD Calibration V1

FIELD Calibration V1 podrá considerarse conseguida cuando:

- la calibración manual sea rápida;
- el campo proyectado se alinee con precisión con frames Veo;
- funcione con distintos encuadres;
- la calidad pueda medirse;
- los fallos puedan detectarse;
- la arquitectura permita automatización posterior;
- generalice a varios partidos del Real Jaén y no a un único clip.

En ese momento RealJaenLab tendrá una base geométrica fiable para construir posteriormente PEOPLE, POSITION, ROLE y ANALYTICS.

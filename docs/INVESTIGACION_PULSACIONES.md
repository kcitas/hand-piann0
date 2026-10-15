# Investigación: detección de pulsaciones por dedo

11 de octubre de 2026. Todo lo que aquí aparece como "medido" se calculó en esta investigación con `scripts/benchmark_signals.py` y `scripts/benchmark_pianovam.py`; lo demás se indica como no verificado.

## 1. Punto de partida (auditoría del proyecto)

- **Dedos.** El detector sigue **8 dedos** (índice–meñique de cada mano). Los pulgares se detectan pero no son dedos de juego.
- **Datos disponibles.**

  | Archivo | Contenido | ¿Sirve para medir pulsaciones? |
  | --- | --- | --- |
  | `precision.csv`, `precision2.csv` | Precisión dedo → tecla (otro esquema) | No |
  | `traza.csv` (v1) | 62 s, sin columna `cue` | No: sin referencia temporal |
  | `traza_v2.csv` | 162 s, 21.4 fps, 76 avisos | Sí |
  | `traza_v3.csv` | 162 s, 24 fps, 76 avisos | Sí |

  La prueba `focus` (avisos cada 3 s) todavía no se ha grabado.
- **Qué se guarda.** Las trazas guardan solo **tres señales derivadas** por dedo (`tip_y`, `flex`, `tip_z`), además de la posición en el piano, la tecla, `hand_speed` y el estado del detector. **No guardan los 21 landmarks.** Por eso:
  - **sí se puede calcular** sin perder información: velocidad y aceleración de esas señales, combinaciones entre ellas, modo común, umbrales y señales por dedo;
  - **no se puede calcular** con las trazas existentes: ángulos PIP/DIP, distancias a la palma, coordenadas respecto a la orientación de la mano.
- **Etiquetas.** Un aviso (`cue`) indica **cuándo se pidió** tocar con un dedo, no cuándo se tocó. Si la persona no responde, el aviso cuenta igual.
- **Emparejamiento.** Cada pulsación se asigna al aviso más cercano de su dedo dentro de ± medio intervalo, uno a uno. Con avisos cada 1.5 s, entre el 25 % y el 46 % de las pulsaciones caen en la zona ambigua entre dos avisos. Por eso la tasa de aciertos depende de la ventana: en v3 va de 5/76 a 24/76.

## 2. Recursos externos

| Recurso | Qué contiene realmente | ¿Temporal? | ¿Etiquetas de pulsación? | Licencia | Tamaño | Uso para HandPiano |
| --- | --- | --- | --- | --- | --- | --- |
| [kunal0230/Virtual_Piano_With_Computer_Vision](https://github.com/kunal0230/Virtual_Piano_With_Computer_Vision) | Código: MediaPipe Hands (API antigua), 5 dedos por mano | Código en vivo | No (sin dataset ni evaluación) | **Sin licencia** (todos los derechos reservados): solo se pueden estudiar las ideas | ~22 MB (incluye un vídeo de demo) | Una idea (ver más abajo) |
| [anjalibibra/Virtual-Piano](https://github.com/anjalibibra/Virtual-Piano) | Código: MediaPipe Tasks, puntas de 4 dedos | Código en vivo | No | **Sin licencia** | ~8.6 MB | Ninguna nueva |
| [Zenodo 20032888](https://zenodo.org/records/20032888) (Universidad del Cauca, 2026) | 30 027 muestras de landmarks **solo x, y** (z descartada): 5 posturas estáticas y 5 gestos dinámicos de dirección | Los dinámicos sí; nº de frames no indicado | **No** (posturas y direcciones) | CC BY 4.0 | 3.2 MB | No sirve para pulsaciones |
| [Zenodo 18108472](https://zenodo.org/records/18108472) (CERTH, 2025) | 63 coordenadas de MediaPipe por muestra (21×3), normalizadas y rotadas, con aumento de datos ×6 | **No** (un frame por muestra) | **No** (clases de gesto no publicadas en la ficha) | CC BY 4.0 | 1.0 MB | Solo como referencia de normalización (origen en la muñeca, escala por la palma, rotación canónica) |
| [HaGRID v2](https://github.com/hukenovs/hagrid) | 1 086 158 imágenes FullHD, 33 gestos + `no_gesture`, cajas y landmarks de MediaPipe | **No** (imágenes estáticas) | **No** | Variante de CC BY-SA 4.0 | **1.5 TB** (versión de 512 px: ~119 GB) | No justifica la descarga: no tiene tiempo ni pulsaciones |
| [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) | Documentación oficial | — | — | — | — | Confirma el diseño actual (ver más abajo) |
| **[PianoVAM](https://huggingface.co/datasets/PianoVAM/PianoVAM_v1)** (ISMIR 2025) | 107 interpretaciones en un Disklavier: vídeo cenital, audio, **MIDI**, **landmarks de 21 puntos por frame** y **digitación** (106 semiautomáticas, **11 manuales**) | **Sí** (vídeo a 60 fps) | **Sí:** onset MIDI de cada nota + mano + dedo | **CC BY-NC-SA 4.0** | 45.3 GB en total; landmarks 12.1 GB; MIDI y TSV < 50 MB; descarga por archivo | **El único recurso evaluado con verdad de pulsación por dedo.** Usado en el benchmark |
| [PianoMotion10M](https://github.com/agnJason/PianoMotion10M) | 116 h de vídeo cenital de piano, poses de mano estimadas y MIDI ([Zenodo 13297386](https://zenodo.org/records/13297386): anotaciones 4.4 GB, MIDI 10 MB) | Sí | Indirectas (MIDI) | Datos **CC BY-NC-ND 4.0** (sin obras derivadas); código Apache-2.0 | ~7 GB sin vídeo | Posible, pero la licencia ND limita distribuir datos derivados. No descargado |
| SKY-Piano ([arXiv 2607.27296](https://arxiv.org/abs/2607.27296)) | Mocap óptica + MIDI + vídeo multivista | Sí | Sí (contacto validado frente a MIDI) | **No verificada** | No verificado | Candidato futuro; disponibilidad no comprobada |
| Typing on Any Surface / AKDD ([arXiv 2309.00174](https://arxiv.org/abs/2309.00174)) | Landmarks + C‑RNN para detectar pulsaciones sobre una superficie; ~91 % a 40 ppm | Sí | Sí (dataset propio) | Paper CC BY-NC-SA; **publicación del dataset no confirmada** | — | Referencia de método: clasificador temporal sobre landmarks |

### Qué hacen de verdad los dos pianos virtuales (código leído)

- **kunal0230:**
  - nota **fija por dedo**, sin geometría de teclas;
  - pulsa si la **`y` absoluta** de la punta, suavizada con una media exponencial (α = 0.3), baja más de 5 px por frame;
  - **suelta si sube más de 3 px por frame**, es decir, la liberación depende solo de la velocidad hacia arriba, no de volver a una posición de reposo;
  - no normaliza respecto a la mano: mover la mano entera pulsa;
  - detecta el borde de la mesa con una transformada de Hough, **pero no lo usa** para decidir la pulsación;
  - no tiene evaluación.
- **anjalibibra:**
  - la "pulsación" es **la entrada de la punta del dedo en el rectángulo de la tecla**: el mismo "hover = nota" que originó nuestro problema de teclas vecinas;
  - bloqueo hasta que el dedo sale de la tecla, más 150 ms de debounce;
  - `LIFT_THRESHOLD` está definido e importado, pero **no se usa**;
  - la ranura de dedo (0–3) **es la misma para ambas manos**;
  - usa `time.time()` como timestamp para MediaPipe, cuando un reloj monotónico sería lo correcto.
- **Qué aprovechamos:** solo **la idea de soltar por velocidad hacia arriba** (kunal) como alternativa a soltar por posición respecto a una referencia. Ataca justo el tipo de atasco que tuvimos, pero introduce el riesgo contrario: si el dedo se levanta despacio, no se suelta nunca. No es código reutilizable, porque no tiene licencia.

### MediaPipe Hand Landmarker: lo que confirma la documentación

- Ofrece 21 landmarks en coordenadas de imagen y de mundo, la lateralidad y los modos `IMAGE` / `VIDEO` / `LIVE_STREAM`.
- En `VIDEO` reutiliza la caja de la mano del frame anterior y solo vuelve a ejecutar el detector de palma si pierde la mano. Es lo que usamos.
- Valores por defecto de `num_hands` 1 y de los tres umbrales de confianza 0.5. Modelo "full" en float16, con entradas de 192 y 224 px.
- La página **no** define la semántica de `z`, **no** explica cómo determina la lateralidad y **no publica métricas de precisión**. Lo que sabemos de la lateralidad (invertida con imagen en espejo) lo medimos nosotros, y PianoVAM tiene el mismo comportamiento (abajo).

## 3. Benchmark sin umbrales

**Método** (`handpiano/interaction/benchmark.py`):
- Por cada evento conocido (dedo que pulsó y momento), la **excursión** de un dedo es el máximo de la señal en la ventana del evento menos su mediana justo antes.
- A partir de la excursión se calculan tres medidas:
  - **AUC frente a vecinos:** ¿el dedo que pulsa se mueve más que los otros dedos de su mano en ese mismo instante? 0.5 = azar. Es el problema de las teclas vecinas.
  - **AUC frente a fondo:** ¿se mueve más que ese mismo dedo en momentos sin pulsación?
  - **Rango 1:** fracción de eventos en los que el dedo que pulsa es el que más se mueve de su mano (azar = 0.25).
- No depende de ningún umbral ni de la ventana de emparejamiento.
- Se comparan 9 variantes: `tip_y`, `flex` y `tip_z` en crudo, sin el modo común de la mano (`:cm`) y como pico de velocidad (`:vel`).

### 3.1 Trazas de HandPiano (cámara frontal; eventos = avisos)

| Variante | v2: AUC vecinos | v2: rango 1 | v3: AUC vecinos | v3: rango 1 |
| --- | --- | --- | --- | --- |
| `tip_y` (producción) | 0.65 | 0.50 | 0.57 | 0.45 |
| `tip_y:cm` | **0.70** | 0.50 | 0.57 | 0.45 |
| `tip_y:vel` | 0.67 | 0.57 | 0.68 | **0.63** |
| `flex` | 0.60 | 0.51 | 0.61 | 0.46 |
| `flex:cm` | 0.64 | 0.51 | 0.64 | 0.46 |
| `flex:vel` | 0.62 | 0.55 | **0.70** | 0.59 |
| `tip_z` | 0.52 | 0.31 | 0.61 | 0.37 |
| `tip_z:cm` | 0.55 | 0.31 | 0.66 | 0.37 |
| `tip_z:vel` | 0.60 | 0.45 | 0.59 | 0.40 |

**Lectura:**
- La separación frente a los vecinos es **débil** (AUC 0.52–0.70). Aun así, el rango 1 (0.31–0.63) queda por encima del azar.
- La mejor variante **cambia entre grabaciones**. Las más estables son las de **velocidad**: `tip_y:vel` da 0.67 / 0.68.
- **Por dedo, los resultados no se reproducen** entre v2 y v3 (por ejemplo, L-MIDDLE con `tip_y`: 0.55 frente a 0.30). Con 6–14 eventos por dedo, **no hay base para elegir una señal por dedo**.

### 3.2 PianoVAM (cámara cenital; eventos = onsets MIDI con dedo anotado a mano)

Tres grabaciones con digitación manual: los landmarks JSON (27–32 MB) y el TSV. No se descargó vídeo ni audio (87 MB en total frente a 45 GB). Dos hechos **medidos**, no supuestos:
- **Las etiquetas de mano del JSON están invertidas respecto a la mano física.** Con ellas invertidas, la punta del dedo anotado cae a 7–16 px de su tecla (una tecla blanca mide 27–33 px). Sin invertir, el error es de ~300 px.
- **Vídeo y MIDI concuerdan con un margen de ±0.1 s.**

Resultados con todos los onsets sin pulgar (AUC frente a vecinos; ventana de −0.15 a +0.10 s):

| Grabación (estilo) | Onsets | Mejor variante | Producción (`tip_y`) | `tip_z` |
| --- | --- | --- | --- | --- |
| Kiss The Rain (principiante) | 99 | `flex:cm` 0.61 | 0.57 | 0.49 |
| Sonata nº 2 (avanzado, rápida) | 111 | `tip_y:cm` 0.66 | 0.61 | 0.51 |
| Gymnopédie (principiante, lenta y suave) | 117 | `tip_z:cm` 0.56 | ≤ 0.50 | 0.53 |

**Lectura:**
- **Incluso con pulsaciones físicas reales medidas por MIDI y vídeo a 60 fps, la separación entre el dedo que pulsa y sus vecinos con landmarks 2D por frame es modesta** (AUC ≤ 0.66). En un toque lento y suave desaparece.
- **Restar el modo común ayuda en 2 de 3 grabaciones.**
- **`tip_z` está cerca del azar.**
- **Aviso de transferibilidad:** la cámara de PianoVAM mira **desde arriba**, la nuestra **desde delante**. Lo que se transfiere es la conclusión ("las señales geométricas por frame separan mal a los vecinos"), no los valores ni la dirección de cada señal.

### 3.3 PianoVAM con características de los 21 landmarks (añadido tras implementar v4)

PianoVAM incluye los 21 landmarks, así que las características nuevas (`landmark_features.py`) ya se pueden evaluar con verdad medida: flexión en MCP/PIP/DIP, distancia a la palma y la combinación temporal `:vel:cm`. Se usaron coordenadas de imagen; PianoVAM no tiene landmarks de mundo. Comando: `python scripts/benchmark_pianovam.py --all-features RUTA`. Son 28 variantes, con todos los onsets sin pulgar:

| Grabación | Mejor variante | `dip_bend:vel:cm` | `palm_dist:vel:cm` | `tip_y` (producción) |
| --- | --- | --- | --- | --- |
| Kiss The Rain | `dip_bend:vel:cm` 0.70 | **0.70** | 0.63 | 0.57 |
| Sonata nº 2 | `palm_dist:vel:cm` 0.71 | **0.68** | 0.71 | 0.61 |
| Gymnopédie | `pip_bend:vel:cm` / `dip_bend:vel:cm` 0.62 | **0.62** | 0.45 | 0.46 |

**Lectura:**
- **`dip_bend:vel:cm`**, la velocidad de flexión de la última falange menos el movimiento común de la mano, es la **única variante entre las dos mejores en las tres grabaciones**. Supera a `tip_y` por 0.07–0.16 de AUC, incluida la pieza lenta, donde todas las señales de nivel fallaban.
- **Siguen siendo separaciones modestas** (≤ 0.71). El rango 1 queda en 0.37–0.51, con un azar de 0.25.
- **Es exploratorio:** 28 variantes probadas en 3 grabaciones, elegidas a la vista de los resultados, y con vista cenital. Es una **hipótesis que confirmar en trazas v4 propias** con etiquetas de teclado, no una razón para cambiar producción.

## 4. Alternativas de señal y algoritmo

| # | Alternativa | Estado | Evidencia o motivo |
| --- | --- | --- | --- |
| 1 | `tip_y` | En producción | Separación débil (0.57–0.65); mejor con modo común en v2 |
| 2 | Flexión (`flex`; ángulos) | `flex` medido; ángulos medidos en PianoVAM | `flex` ≈ `tip_y`. En PianoVAM, `dip_bend:vel:cm` es la más constante (0.62–0.70; §3.3). En trazas propias hacen falta grabaciones v4 |
| 3 | Velocidad / aceleración | Velocidad medida | Variante más estable en nuestras trazas (`tip_y:vel` 0.67 / 0.68); peor en PianoVAM con vista cenital. Aceleración: no medida (con 21–24 fps está dominada por ruido) |
| 4 | Distancias a la palma | Medida en PianoVAM | `palm_dist:vel:cm` va de 0.45 a 0.71, según la pieza: inestable |
| 5 | Normalización por tamaño y orientación | Implementada para el estudio | Los ángulos son invariantes a posición, tamaño y rotación; las distancias se dividen por la palma. Las versiones `_w` usan landmarks de mundo (metros), disponibles desde v4 |
| 6 | Señal o umbral por dedo | Medido | **Inestable entre grabaciones** con 6–14 eventos por dedo: no está justificado |
| 7 | Filtrado, histéresis, debounce | En producción | Ya existen One Euro, histéresis, confirmación por frames y periodo refractario. La liberación por velocidad hacia arriba (kunal) queda como alternativa a evaluar |
| 8 | Modo común sin cancelar pulsaciones reales | Medido | `:cm` ayuda en v2 y en 2/3 de PianoVAM. `hand_low` frente a `idle_min`: ninguno gana en v2 y v3 a nivel de detector (ver README) |
| 9 | Pérdida de seguimiento y recuperación | En producción | Reaprendizaje tras la pérdida, validado con trazas (los atascos bajaron de 25 a 2 en v2) |
| 10 | Clasificador temporal ligero | **No justificado todavía** | Nuestras etiquetas son avisos, no pulsaciones reales, y son pocas (~150 eventos). PianoVAM tiene etiquetas reales, pero otra vista, y su licencia es no comercial. La literatura (C‑RNN en AKDD, TCN en FingerType) usa datos etiquetados propios |

## 5. Conclusiones

1. **No hay una señal geométrica por frame que resuelva el problema de los vecinos.** Lo muestran nuestras dos trazas y las tres grabaciones de PianoVAM con verdad medida. Ajustar más umbrales sobre `tip_y`, `flex` o `tip_z` tiene un techo bajo.
2. **Mejoras con respaldo, aunque modesto:** restar el modo común de la mano (ya en producción) y usar la **velocidad** de la señal en lugar del nivel (aún no en producción; es una hipótesis que validar).
3. **Lo que falta no es un dataset público, es verdad propia:**
   - ningún recurso público tiene la cámara frontal y el gesto de HandPiano;
   - nuestras etiquetas son avisos, no toques.
   Sin una referencia independiente de cuándo hubo toque, ningún benchmark distingue "no tocó" de "la cámara no lo ve".

## 6. Siguientes pasos recomendados

1. ~~**Guardar los 21 landmarks por mano en las trazas** (formato v4).~~ **Hecho**: ver [TRAZAS_V4_Y_ETIQUETAS.md](TRAZAS_V4_Y_ETIQUETAS.md). Coste medido: 1,8 KB por mano y frame.
2. **Grabar la prueba `focus`** (avisos cada 3 s) para evaluar sin la ambigüedad de la ventana.
3. **Una referencia de toque independiente de la visión.** **Hecho, sin dependencias:** el protocolo `keys` registra las pulsaciones de teclas físicas como etiquetas `contact`/`release`. El micrófono sigue pendiente de aprobación, para tocar sobre la mesa.
4. **Evaluar en el benchmark** la liberación por velocidad hacia arriba y el detector basado en `:vel` antes de tocar producción.
5. **Solo con verdad propia suficiente** (cientos de toques por dedo), plantear un clasificador temporal ligero sobre landmarks. PianoVAM puede servir para pre-entrenar o explorar características, respetando su licencia CC BY-NC-SA.

## 7. Pruebas guiadas `dedos` con el piano en pantalla (13–15 de octubre)

Protocolo `dedos`: 5 toques guiados por dedo, uno cada 2 s; 40 avisos por sesión. Las pulsaciones se reproducen con el detector de producción (`replay_frames`) y se emparejan uno a uno con los avisos (`match_cues`).

**Colocación.** Subir el piano dibujado (`screen_area` = 0,30–0,62 del alto) y la barra de colocación bastaron para resolver el seguimiento: con buena colocación, cada mano se ve el 97–100 % del tiempo (`tocar_3`, `tocar_4`). Con las muñecas en el borde inferior (`dedos_4`) se veía el 46 % (izquierda) y el 86 % (derecha). `--confianza 0.3` sube el seguimiento del 97 % al 100 % en una pareja de grabaciones: es una mejora pequeña y no cambia el valor por defecto.

**Detección con las manos bien vistas.** Rejilla en las 6 sesiones (240 avisos):

| Ajuste | Aciertos | Pulsaciones de más |
| --- | --- | --- |
| Producción (`press_on` 0,12, `min_velocity` 0,8, 2 frames, `hand_low`) | 18 (7,5 %) | 56 |
| Mejor de la rejilla (0,08, 0, 1 frame, `max_press_s` 1,5, modo común `off`) | 53 (22 %) | 145 |

Ninguna combinación de `press_on` ∈ {0,08; 0,10; 0,12}, `min_velocity` ∈ {0; 0,4; 0,8}, `min_press_frames` ∈ {1; 2}, `max_press_s` ∈ {sin límite; 1,5} y modo común ∈ {`hand_low`; `off`} mejora los aciertos sin multiplicar las pulsaciones de más. Por eso no se cambió nada en producción.

**Por qué.** En `dedos_5`, con el dedo indicado tocando la mesa, su punta se mueve 0,03–0,10 tamaños de mano en la imagen, y los otros dedos de la misma mano se mueven lo mismo o más en ese instante. Tocando "en el aire" con más amplitud (`dedos_6`), el índice llega a 0,36–0,9, pero las puntas salían del piano dibujado y se perdió la mano derecha, así que los aciertos no subieron (4/40).

**Defecto observado:** dedos que se quedan en PRESSED decenas de segundos cuando la postura cambia durante una pulsación (la línea base está congelada en PRESSED). Limitarlo con `max_press_s` = 1,5 s sube 14 → 18 aciertos de 200, con 5 pulsaciones de más. Es una ganancia pequeña y no está activado.

**Modo "dedo = nota" (15 de octubre).** Manos levantadas, palma hacia la cámara; una nota = doblar el dedo. Señal `flex` (longitud aparente del dedo). Misma prueba `dedos`:

| Grabación | Postura | Umbral 0,50 | Umbral 0,35 |
| --- | --- | --- | --- |
| `notas_1` | correcta 33–42 % del tiempo | 0/40 (+1) | 3/40 (+4) |
| `notas_2` | manos no vistas 91–92 % | 0/40 | 0/40 |
| `notas_3` | manos vistas, barra de postura | 18/40 (+8) | **32/40 (+18)** |

En `notas_3`, `press_on` de 0,30 a 0,40 da 32–33/40: un tramo estable. Se adopta 0,35 (suelta en 0,175). Por dedo: índices 5/5, mano derecha 5/5 en todos; medio izquierdo 2/5, anular izquierdo 1/5. Doblar un dedo es un gesto que la webcam frontal sí ve, al contrario que el toque en la mesa.

**Siguientes pasos con más potencial que ajustar umbrales:**
1. una cámara a la altura de la mesa, mirando los dedos de lado (un móvil como webcam);
2. un gesto más visible que el toque, como doblar el dedo (ángulo de la articulación PIP);
3. medir el modo `--solo-indices` con el protocolo `indices`.

## Reproducir

```bash
python scripts/benchmark_signals.py traza_v2.csv traza_v3.csv
# PianoVAM: ver las instrucciones de descarga (~87 MB) en el docstring del script
python scripts/benchmark_pianovam.py RUTA_A_PIANOVAM
python scripts/benchmark_pianovam.py --all-features RUTA_A_PIANOVAM
# Trazas v4 con etiquetas independientes
python scripts/benchmark_landmarks.py teclado_1.csv
```

# Trazas v4 y etiquetas independientes

Complemento de [INVESTIGACION_PULSACIONES.md](INVESTIGACION_PULSACIONES.md). Explica qué guarda una grabación nueva, cómo se etiqueta la verdad de forma independiente al detector y cómo se evalúa.

## 1. Qué se perdía antes

| Etapa | Qué recibe | Qué se descartaba |
| --- | --- | --- |
| MediaPipe Hand Landmarker (modo `VIDEO`) | 21 landmarks de imagen, 21 de mundo y lateralidad por mano | — |
| `observations_from_result` | — | **Landmarks de mundo** y campos opcionales `visibility`/`presence` |
| `FingerTracker` | Landmarks crudos → filtro One Euro → suavizados | Los **crudos** solo quedaban en memoria, y la observación rechazada por salto ni eso |
| `PressDetector` | Landmarks suavizados → `tip_y`, `flex`, `tip_z` | Todo lo demás |
| `TraceRecorder` (v1–v3) | 3 señales por dedo, estado y eventos | **Los 21 landmarks**, el estado `HELD`/`LOST` (solo `tracked` sí/no), el modo común aplicado y la línea base |

**Ahora:**
- `HandObservation` conserva `world_landmarks`, `visibility` y `presence`. Los dos últimos solo si la API los rellena en **todos** los puntos; si no, quedan en `None` y nunca se rellenan con valores por defecto.
- `HandState` conserva la observación del frame (`observation`) y si se rechazó por salto (`outlier_rejected`).
- El detector de producción no cambia.

## 2. Archivos de una grabación v4

Una grabación con `--record-trace NOMBRE.csv` escribe estos archivos. **Ninguno sobrescribe nada:** si ya existe cualquiera de ellos, la app se niega antes de abrir la cámara, y la escritura usa modo exclusivo (`x`).

| Archivo | Contenido |
| --- | --- |
| `NOMBRE.csv` | Traza por dedo. Mismas 18 columnas que v2/v3, más `seq`, `hand_status`, `common_mode` y `baseline` al final |
| `NOMBRE.landmarks.csv` | Una fila por mano y frame procesado (ambas manos siempre; una mano perdida aparece como fila con coordenadas vacías) |
| `NOMBRE.meta.json` | Versión del formato, reloj, tamaño de imagen, espejo, protocolo, pasos y `PressConfig` completa |
| `NOMBRE.labels.csv` | Etiquetas independientes. Solo se escribe si hubo alguna tecla física |

**Tamaño medido:** ~1,8 KB por mano y frame, es decir ~17 MB para el protocolo `keys` (202 s a 24 fps). Copiar los datos cuesta ~4 µs por frame en el hilo de seguimiento. **No se guarda vídeo ni imagen.**

**Compatibilidad:**
- Las trazas v1–v3 siguen funcionando con `analyze_trace.py`, `compare_detectors.py` y `benchmark_signals.py`.
- Los lectores nuevos aceptan que falten las columnas nuevas. Hay un test que reproduce una traza sin ellas y obtiene el mismo resultado.

### Convenciones de `NOMBRE.landmarks.csv`

| Columnas | Unidades y convención |
| --- | --- |
| `t` | Segundos de `time.perf_counter()` en la captura del frame: monotónico, origen arbitrario. Es el mismo `t` que la traza por dedo y que las etiquetas |
| `seq` | Contador de frames procesados |
| `hand` | Mano **física** tras la identidad temporal (`LEFT`/`RIGHT`) |
| `status` | `TRACKED`: datos frescos. `HELD`: sin datos frescos; se mantiene el último estado durante un periodo de gracia. `LOST`: sin datos |
| `detected` | `True` si MediaPipe dio una mano asignada a esta en este frame |
| `detector_label` | Mano física que implica la etiqueta de MediaPipe en este frame. Si difiere de `hand`, la identidad temporal la corrigió |
| `outlier_rejected` | La detección de este frame se rechazó por salto |
| `handedness_score` | Puntuación de MediaPipe para izquierda/derecha. **No mide la calidad de los landmarks.** Es la única puntuación por mano que expone la API Tasks |
| `rejected_outliers` | Saltos rechazados acumulados para esta mano |
| `raw_{x,y,z}{i}` | Landmarks de imagen de la detección de este frame, sin suavizar. x, y: fracción del ancho/alto de la imagen (pueden salirse un poco de [0, 1]). z: profundidad relativa con origen en la muñeca, en ~la escala de x; menor = más cerca de la cámara. Imagen en espejo si `meta.mirrored` |
| `sm_{x,y,z}{i}` | Los landmarks suavizados que usó el detector. Vacíos en `LOST`; en `HELD` son los últimos (antiguos) |
| `w_{x,y,z}{i}` | `hand_world_landmarks`: metros, con origen cerca del centro de la mano |
| `vis{i}`, `pres{i}` | Visibilidad y presencia por punto **si el modelo las da**; si no, vacías. Hay que comprobar en la primera grabación real si el modelo de manos las rellena: la API las define como opcionales y la documentación no lo aclara |

`i` es el índice de MediaPipe: 0 = muñeca … 20 = punta del meñique (ver `Landmark`).

**Validación al leer** (`load_landmarks`):

| Problema | Qué hace el lector |
| --- | --- |
| Timestamp no numérico o no finito | Error que nombra la línea |
| Timestamp que no avanza dentro de una mano | Error que nombra la línea |
| Mano desconocida o columnas que faltan | Error |
| Coordenadas no finitas | Se descartan y se informa |
| Coordenadas de imagen fuera de [−1, 2] | Se descartan y se informa |
| Fila `TRACKED` sin landmarks suavizados válidos | Se trata como `LOST` |

### Ejemplo (columnas recortadas)

```csv
t,seq,hand,status,detected,detector_label,outlier_rejected,handedness_score,rejected_outliers,raw_x0,raw_y0,raw_z0,…,sm_x0,…,w_x0,…,vis0,…,pres20
1000.0000,1,LEFT,TRACKED,True,LEFT,False,0.9000,0,0.350000,0.750000,0.000000,…,0.350000,…,0.000000,…,,…,
1000.0000,1,RIGHT,TRACKED,True,RIGHT,False,0.9000,0,0.650000,0.750000,0.000000,…,0.650000,…,0.000000,…,,…,
1002.0667,63,LEFT,HELD,False,,False,0.9000,0,,,,…,0.350000,…,,…,,…,
1002.3667,72,LEFT,LOST,False,,False,,0,,,,…,,…,,…,,…,
```

## 3. Etiquetas independientes (`NOMBRE.labels.csv`)

**Regla:** una etiqueta nunca sale de la salida del detector ni del calendario de avisos. Un aviso dice cuándo se **pidió** tocar; el detector dice lo que **creyó** ver. Ninguno de los dos prueba un contacto. El código **rechaza** las fuentes `detector` y `cue`.

```csv
t_start,t_end,target,label,source,confidence,note
1001.4167,,RIGHT_INDEX,contact,keyboard,confirmed,key=J
1001.5833,,RIGHT_INDEX,release,keyboard,confirmed,key=J
1004.0000,1004.6000,ALL,ambiguous,keyboard,uncertain,key=Q (key outside the protocol's map)
1010.0000,1018.0000,ALL,rest,manual,confirmed,anotado en directo
1012.3000,1012.9000,RIGHT_RING,neighbour_motion,video_review,uncertain,se mueve con el medio
```

| Campo | Valores |
| --- | --- |
| `label` | Puntuales (`t_end` vacío): `motion_onset`, `contact`, `release`. Intervalos: `rest`, `neighbour_motion`, `insufficient_tracking`, `ambiguous` |
| `target` | Un dedo (`RIGHT_INDEX` …), una mano (`LEFT`/`RIGHT`) o `ALL` |
| `source` | `keyboard`: tecla física. `video_review`: revisión de vídeo, solo con consentimiento explícito; HandPiano no guarda vídeo. `manual`: otra anotación humana. `tracking`: solo para `insufficient_tracking` |
| `confidence` | `confirmed` o `uncertain`. Si la cámara no permite confirmar algo, se etiqueta `uncertain` o `ambiguous`; no se adivina |

### Referencia independiente: el teclado del ordenador

La cámara no puede confirmar el contacto físico; por eso el benchmark anterior se quedaba en "cuándo se pidió". La forma más simple de tener verdad **sin micrófono ni dependencias nuevas** es que los dedos pulsen teclas reales:

- **Qué se mide:** cada tecla que baja es un `contact` y cada tecla que sube un `release`, con el mismo reloj que los frames.
- **Mapa tecla → dedo:** fila central, A S D F para la mano izquierda (meñique → índice) y J K L Ñ para la derecha (índice → meñique). La tecla `;` del teclado US equivale a Ñ.

**Limitaciones conocidas:**
1. **El dedo se supone, no se mide.** Si alguien pulsa la F con el dedo medio, la etiqueta dirá índice. Por eso cada tecla solo admite un dedo y el protocolo lo pide.
2. **Latencia del momento medido:** el escaneo del teclado más la entrega del evento en Qt suman probablemente ≤ 20–30 ms. Es menos que un frame (42 ms a 24 fps), pero no está medido.
3. **Contacto ≠ pulsación de tecla.** Rozar la tecla sin hundirla no cuenta como contacto. Las teclas de portátil tienen ~1–1,5 mm de recorrido, y por eso un toque muy suave puede quedar sin etiqueta.
4. **Superficie distinta.** Las teclas no son la mesa ni un piano. Sirve para estudiar señales con la misma cámara, la misma vista y la misma persona, que es mucho más cercano que PianoVAM, pero no es idéntico al uso real.
5. **Foco:** solo se registran teclas mientras la ventana de HandPiano está activa. El banner muestra el contador de teclas registradas.

**Detalles de implementación:**
- La auto-repetición se ignora, y un mismo evento visto por varios widgets se registra una sola vez.
- Las teclas del mapa se consumen para que no escriban en los controles de la barra.
- Una tecla fuera del mapa se registra como `ambiguous`, con confianza `uncertain` y destino `ALL`.

## 4. Benchmark

La extracción de características, las etiquetas y el detector están en módulos separados:

| Pieza | Módulo |
| --- | --- |
| Características | `landmark_features.py`: `tip_y`, `flex`, `tip_z`, flexión en MCP/PIP/DIP, distancia a la palma, y sus versiones con landmarks de mundo `_w` |
| Etiquetas | `labels.py` |
| Comparación | `benchmark.py`: variantes `:cm`, `:vel` y `:vel:cm`; `events_from_labels`; `match_detections`; `session_folds` |
| Detector | Sin cambios |

**Todas las características crecen al pulsar:**
- **Ángulos:** se calculan con `atan2`, en grados (0 = dedo recto). Son invariantes a la posición, al tamaño y a la rotación de la mano.
- **Distancias:** se dividen por la longitud de la palma (muñeca → MCP del medio), así que no dependen del tamaño de la mano.

`scripts/benchmark_landmarks.py NOMBRE.csv` hace dos cosas:
1. **Puntúa todas las características** alrededor de cada contacto confirmado (ventana alineada al contacto).
2. **Compara las pulsaciones del detector** con los contactos, emparejándolas uno a uno: aciertos, fallos, falsas pulsaciones y desfase mediano.

Sin `labels.csv` no evalúa nada. Con `--cues` evalúa contra los avisos y lo advierte en la salida.

**Sin fugas:** si más adelante se entrena un modelo, la separación entre entrenamiento y evaluación se hace **por sesión** (`session_folds`, dejando una sesión fuera cada vez), nunca por frame. Los frames de una misma sesión están muy correlacionados.

## 5. Procedimiento exacto para una grabación etiquetable

0. Comprobación previa, sin cámara (unos 3 s):
   ```bash
   python scripts/check_trace_keys.py
   ```
   El script abre la ventana real, envía A S D F J K L Ñ por la cola de eventos de macOS de la propia app y valida el `.labels.csv`, que escribe en una carpeta temporal. Debe terminar con `PASS`.

   Después, la comprobación con **tus** pulsaciones físicas, sin cámara y en unos 20 s:
   ```bash
   python scripts/check_trace_keys.py --manual
   ```
   - Imprime una línea por cada tecla que macOS entrega a la ventana y avisa cuando la app gana o pierde el teclado.
   - Si aquí no llega ninguna tecla, el problema está en la entrega del sistema, no en HandPiano.
   - Durante una grabación real, `--debug-keys` imprime las mismas líneas.
1. Coloca un teclado físico (mejor externo) sobre la mesa, delante de la cámara, de modo que se vean ambas manos enteras sobre él.
2. Ejecuta:
   ```bash
   python -m handpiano --record-trace teclado_1.csv --trace-protocol keys
   ```
   Se abre la ventana con las instrucciones en pantalla. **Haz clic en la ventana de HandPiano antes de empezar a pulsar.** En macOS, al lanzarla desde la terminal o desde VS Code, el foco del teclado se queda en la terminal y las teclas no llegan.
   - Comprueba que el contador "Teclas registradas" de la barra de estado (abajo a la derecha) sube con cada pulsación.
   - Si la ventana pierde el foco, el banner lo dice desde el primer momento: "⚠ LA VENTANA NO TIENE EL FOCO DEL TECLADO".
   - Si en un paso de pulsaciones no llega ninguna tecla, el banner se pone en rojo: "⚠ NO LLEGA NINGUNA TECLA".
   - `teclado_1.csv` y `teclado_2.csv` (11 de octubre de 2026) se grabaron completas, con 116 y 124 pulsaciones detectadas, pero con **0 teclas registradas**. No tienen etiquetas y hay que repetirlas.
3. Sigue los pasos (≈ 3 min 20 s):
   - **Preparación** (8 s): dedos en A S D F / J K L Ñ, sin pulsar.
   - **Reposo** (10 s): sin pulsar.
   - **Levantar sin pulsar** (10 s): movimientos sin contacto, que sirven de negativos difíciles.
   - **8 bloques de 5 pulsaciones** guiadas, cada 3 s: una por dedo y tecla.
   - **30 s de juego libre.**
4. Al terminar quedan `teclado_1.csv`, `.landmarks.csv`, `.meta.json` y `.labels.csv`. Analiza con:
   ```bash
   python scripts/benchmark_landmarks.py teclado_1.csv
   python scripts/analyze_trace.py teclado_1.csv
   ```
5. Repite en **otra sesión** (otro día o con otra luz) con otro nombre (`teclado_2.csv`). Solo así se puede comprobar si un resultado se mantiene entre sesiones.

**Si hace falta etiquetar contacto sobre la mesa o el papel**, sin teclado, hay dos fuentes posibles que no están implementadas:
- **Revisión de vídeo**, solo con consentimiento explícito, fuera de la app y borrándolo después.
- **Micrófono**, pendiente de tu aprobación porque añade una dependencia.

En ambos casos el formato de etiquetas es el mismo.

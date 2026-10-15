# Audio: tres instrumentos reales, motor y medidas

## 1. Instrumentos y procedencia

Cada instrumento es una grabación de un instrumento **distinto y real**. Ninguna muestra se reutiliza ni se procesa para fingir otro instrumento.

| Instrumento | Grabación | Banco y procedencia | Licencia | Atribución |
| --- | --- | --- | --- | --- |
| **Grand Piano** | Steinway Model B de cola, micrófonos cercanos, sin pedal, 44,1 kHz / 24 bit estéreo | [Versilian Community Sample Library (VCSL)](https://github.com/sgossner/VCSL), carpeta `Grand Piano, Steinway B`, commit `c1ea7bcc3c` | **CC0 1.0** (dominio público) | No obligatoria. Se cita por cortesía |
| **Upright Piano** | Piano vertical Knight, muestreado por Simon Dalzell (Ivy Audio) para VSCO 2, 44,1 kHz / 24 bit estéreo | VCSL, carpeta `Upright Piano, Knight`, commit `c1ea7bcc3c` | **CC0 1.0** | No obligatoria |
| **Electric Piano** | Rhodes Mark I Stage 73 de 1977, grabado directamente del conector del arpa, 44,1 kHz / 16 bit mono | [jRhodes3d de Jeff Learman](https://github.com/jlearman/jRhodes3d-wav), commit `a886e6cebf` | **CC BY-NC 4.0** (muestras) | **Obligatoria:** "Electric piano: jRhodes3d by Jeff Learman (CC BY-NC 4.0), trimmed to 5 s and converted." |

**Licencia del piano eléctrico:**
- **Uso comercial:** la licencia es **no comercial**. Para vender HandPiano o distribuirlo comercialmente con estas muestras hace falta permiso del autor (jjlearman@gmail.com) o cambiar de banco.
- **Discrepancia en las fuentes:** el `.sfz` dice "CC-SA-BY", pero el archivo LICENSE dice CC BY-NC. Se aplica la más restrictiva.

**Bancos evaluados y descartados:**

| Banco | Motivo del descarte |
| --- | --- |
| Salamander Grand (CC BY 3.0) | 296 MiB comprimido y 1,27 GB descomprimido: demasiado para un Mac de 8 GB |
| FreePats YDP Grand (CC BY 3.0, 36 MiB SF2) | Válido, pero VCSL permite bajar solo las notas necesarias en WAV, sin descomprimir 7z ni leer SF2 |
| FreePats Upright KW (CC0) | Solo se distribuye en 7z; aquí no hay extractor sin añadir dependencias |
| Freesound "Fender Rhodes Mark II" de tim.kahn (CC BY 4.0) | **Falta el registro medio**: 41 notas, sin nada entre F3 y C#5 según su nomenclatura. Además, descargar los originales exige cuenta |
| SF2 de "Rhodes" en Musical Artifacts | Procedencia de las muestras no verificable |
| FreePats "Synthesized Piano" | Es un sintetizador, no un Rhodes |

**El repositorio no incluye ninguna muestra.** `scripts/download_sounds.py` las descarga una vez a `sounds/`, que está en `.gitignore`. Es el mismo criterio que con el modelo de MediaPipe.

## 2. Descarga, tamaño y verificación

```bash
python scripts/download_sounds.py                 # los tres (≈170 MB de descarga, ≈110 MB en disco)
python scripts/download_sounds.py --only electric  # uno solo
```

| Instrumento | Muestras | Notas grabadas | Capas de velocidad | Descarga | Caché | RAM |
| --- | --- | --- | --- | --- | --- | --- |
| Grand | 58 | 29, cada tono entero, MIDI 44–100 | 2 (`vl2`, `vl4`) | ~81 MB | 48 MB | 48 MB |
| Upright | 56 | 28, cada tono entero, MIDI 45–99 | 2 (`vl1`, `vl2`) | ~78 MB | 48 MB | 48 MB |
| Electric | 36 | 12, de A2 a C7 | 3 (capas 2, 4 y 5) | ~18 MB | 14,5 MB | 14 MB |

**Qué baja el script y cómo lo procesa:**
- **Descarga mínima:** solo los **primeros ~5 s** de cada archivo, con peticiones HTTP `Range` sobre URLs fijadas a un commit. El contenido no puede cambiar.
- **Recorte:** desde 3 ms antes del ataque, con 5 s de duración y fundido coseno en el último segundo. Se guarda en 16 bit.
- **Trazabilidad:** `bank.json` guarda la URL, el SHA-256 de los bytes descargados, la licencia y el procesado.

**Verificación de afinación:** cada muestra se compara con la nota a la que está asignada.
1. **Clase de nota:** debe coincidir a ±50 cents. Así se detecta cualquier error de semitono.
2. **Octava:** al menos el 90 % de las muestras de cada instrumento deben estar en la octava exacta. Un error de nomenclatura de octava desplazaría todas las muestras a la vez; algún fallo aislado en el registro agudo es una limitación del estimador.

Resultado de la instalación del 11 de octubre de 2026:

| Instrumento | Peor desvío de clase de nota | Octava correcta |
| --- | --- | --- |
| Grand | +17 cents | 100 % de las muestras |
| Upright | +27 cents | 98 % |
| Electric | +6 cents | 94 % |

Los desvíos de unos cents en el agudo son normales: los pianos reales se afinan con las octavas estiradas.

**La nomenclatura se midió, no se supuso:**
- **Steinway:** "C4" suena a 261,8 Hz.
- **Knight:** "A3" suena a 441 Hz, es decir A4. Sus nombres van una octava por debajo de la notación científica.
- **jRhodes:** "D4" suena a 294,4 Hz. La capa 1 es la más suave; uno de los `.sfz` del autor tiene el orden de capas invertido.

## 3. Arquitectura

```
ratón / teclado / pulsaciones de manos / juego / MIDI / tests
        │  note_on(midi, velocidad) · note_off(midi) · set_instrument(id) · set_volume(v)
        ▼
AudioEngine  (handpiano/audio/engine.py) — no depende de MediaPipe, Qt ni la cámara
   ├─ cargador (hilo propio): SoundBank ← sounds/<id>/ (validado; error claro si falta o está dañado)
   ├─ Mixer (numpy): voces polifónicas, release, robo de voces, volumen con rampa, limitador suave
   └─ salida: SoundDeviceOutput (PortAudio vía sounddevice) o NullOutput (tests / --no-audio)
```

**Hilos:**
- **Quien llama** (hilo de interfaz o de seguimiento): solo encola la orden; la llamada tarda unos 4 µs.
- **Hilo de audio:** las aplica al empezar el siguiente bloque de 256 muestras (5,8 ms).
- **Bancos:** se cargan en otro hilo. Al arrancar se precargan los tres (unos 110 MB de RAM), así que cambiar de instrumento es inmediato.

**Detalles del mezclador:**

| Aspecto | Cómo funciona |
| --- | --- |
| Elección de muestra | La raíz grabada más cercana, en la capa de velocidad que corresponda. Se transpone como mucho 1 semitono en los acústicos y 3 en el Rhodes |
| Ganancia | `(velocidad/127)^curva` |
| Release (tras `note_off`) | Exponencial hasta −60 dB en 0,30 s (cola), 0,25 s (vertical) o 0,15 s (Rhodes) |
| Nota mantenida | Termina con su muestra, a los 5 s. Ninguna nota suena indefinidamente |
| Volver a tocar una nota que suena | La anterior se apaga en 30 ms |
| Polifonía | 48 voces. Si se supera, se roba la voz más antigua, empezando por las que ya se están apagando |
| Limitador | Suave (`tanh`), para que los acordes no saturen |

**Dependencias:** `sounddevice` (PortAudio) ya venía instalado como dependencia de MediaPipe. Ahora se declara también en `pyproject.toml`, pero no añade ningún paquete. QtMultimedia no está en PySide6-Essentials y se descartó, porque PySide6-Addons es mucho más pesado.

## 4. Integración con los modos

**Piano Libre** (`handpiano/app/press_sound.py`):
- **Hilo:** se ejecuta en el hilo de seguimiento, justo después del detector, para que la nota no espere al refresco de la interfaz.
- **Una nota por pulsación:** cada **evento** `press` inicia la nota una sola vez. Mantener el dedo abajo durante muchos frames no repite nada.
- **Liberación:** el evento `release`, que lleva la tecla bloqueada al pulsar, la detiene.
- **Misma tecla con dos dedos:** **cada pulsación vuelve a golpear la nota**, como en un piano, y la nota se apaga cuando se levanta el último dedo. Antes, si otro dedo ya "tenía" la tecla, la segunda pulsación no sonaba.
  - **Medido en `teclado_1` a `teclado_4`**, reproduciendo la salida grabada del detector: 35 de 555 pulsaciones eran silenciosas, porque un dedo vecino estaba atascado en `PRESSED` sobre la misma tecla. Con la corrección, 555 de 555 generan `note_on`.
  - **Errores de audio:** se registran y nunca se propagan. Antes, una excepción desactivaba el oyente del pipeline para toda la sesión: las teclas seguían iluminándose y no sonaba nada.
- **Diagnóstico con identificador de pulsación:**
  - **Tracking Lab, grupo "Pulsación → sonido":** pulsaciones → `note_on` encoladas → voces iniciadas por el motor → liberaciones → `note_off`.
  - **`--debug-notes`:** una línea por pulsación en la consola, por ejemplo `[nota #12] RIGHT_INDEX → MIDI 72 → note_on encolada → motor: voz iniciada (grand, +9 ms) → liberada a los 0.21 s → note_off encolada → voces liberadas: 1`.
  - **Límite:** "voz iniciada" significa que el mezclador empezó a reproducir la muestra. No prueba que el sonido saliera por el altavoz.
- **Red de seguridad:** cada frame se comparan las notas que suenan con las teclas que el detector da por pulsadas. Así nunca queda una nota colgada aunque falte el `release`; por ejemplo, si la calibración desaparece en mitad de una pulsación.
- **Velocidad (provisional):** la velocidad de bajada del dedo se convierte de forma lineal a MIDI 40–120. No está calibrada.
- **El detector no cambia:** no se han tocado sus umbrales ni su lógica.

**Juego Rítmico 4K** (`handpiano/game/`, `handpiano/ui/rhythm_view.py`):
- **Carriles:** 4, con notas que caen y línea de impacto. Las ventanas de acierto son PERFECT ±50 ms, GREAT ±100 ms y GOOD ±160 ms; fuera de ellas es MISS.
- **Marcador:** puntuación, combo y precisión viven en `RhythmGame`, sin audio ni Qt.
- **Audio de un acierto:** el audio se suscribe a los aciertos y toca la nota **de la partitura** con el instrumento seleccionado.
- **Entrada:**
  - pulsaciones de manos sobre los cuatro cuartos del teclado calibrado, usando el tiempo de captura del frame, de modo que la latencia de la cámara no penaliza;
  - teclas D F J K;
  - clic en un carril.
- **Sin repeticiones:** cada pulsación se juzga una sola vez y acierta como máximo una nota. Un dedo mantenido no acierta las siguientes.
- **Canciones:** se incluyen cuatro melodías de dominio público en `songs/` (ver [songs/README.md](../songs/README.md)). Se pueden sustituir por `.mid` o partituras `.json` (formato en `handpiano/game/chart.py`). Sin canciones, el juego ofrece un **ejercicio de escala etiquetado como ejercicio**.

**Teclado y ratón como alternativa:**

| Acción | Cómo |
| --- | --- |
| Tocar notas (Piano Libre) | A W S E D F T G Y H U J K O L P Ñ, desde la primera tecla del piano. Z/X cambian la octava |
| Cambiar de instrumento | Teclas 1, 2 y 3 |
| Volumen | Teclas − y + |
| Tocar con el ratón | Clic y mantener sobre una tecla del mapa del piano |

**Grabación de trazas:** durante una grabación de trazas las teclas son **solo etiquetas**, se registran pero no suenan. Hay un test que lo comprueba.

## 5. Probar los tres instrumentos

```bash
python scripts/download_sounds.py      # una vez
python -m handpiano --audio-test       # sin cámara: teclado y ratón, acordes, 1/2/3, −/+
python scripts/audio_check.py          # diagnóstico: arpegio con cada instrumento y medidas
python -m handpiano                    # app completa: Piano Libre / Juego Rítmico 4K
python -m handpiano --instrument electric --no-audio   # opciones
```

## 6. Latencia: qué se mide y cómo medir la real

Medido en este MacBook (altavoces internos, 11 de octubre de 2026) con `scripts/audio_check.py`:

| Tramo | Valor | Cómo se obtiene |
| --- | --- | --- |
| Llamada a `note_on` | 3–5 µs | Cronómetro alrededor de la llamada |
| Orden → hilo de audio | 3–4 ms de mediana, ≤ 5,7 ms | Marca de tiempo al encolar y al aplicar |
| Buffer → DAC | 8,3 ms | `outputBufferDacTime − currentTime`, estimación de PortAudio |
| Latencia de salida | 17,3 ms | `stream.latency`, estimación de PortAudio |
| Cortes y callback | 0 cortes en 926 bloques; callback ≤ 0,9 ms de 5,8 ms | Contadores del propio stream |

**La latencia acústica no está medida.** Las estimaciones de PortAudio no incluyen el altavoz, ni Bluetooth, ni la latencia de la cámara y el detector, que ya está documentada en el README. Para medirla en tu equipo:

1. **Solo audio (tecla → sonido):**
   1. Abre `python -m handpiano --audio-test`.
   2. Graba con el móvil (audio, mejor a 48 kHz) cerca del teclado del ordenador y pulsa una tecla de nota con decisión, unas 20 veces.
   3. En un editor de audio (por ejemplo Audacity), mide en cada pulsación la distancia entre el clic mecánico de la tecla y el inicio del sonido del piano.
   4. La mediana de esas distancias es la latencia teclado → sonido. Incluye unos 5–15 ms del propio teclado.
2. **De punta a punta (dedo → sonido):**
   1. Graba en cámara lenta (240 fps) la mano tocando sobre la mesa en Piano Libre, con el sonido.
   2. Para cada pulsación, mide el tiempo entre el frame del contacto y el inicio del sonido en la pista de audio.
   3. Calibra antes la sincronía entre audio y vídeo del móvil con una palmada visible y audible, porque muchos teléfonos desfasan ambas pistas.
3. **Comprueba los cortes:** repite `scripts/audio_check.py` con la cámara funcionando. El contador de *underflows* debe seguir en 0.

## 7. Limitaciones pendientes

- **Gestos de la mano:** la app de escritorio no tenía navegación por gestos. Ahora se navega con ratón y teclado (1/2/3, −/+, botones). La selección por gestos, por ejemplo mantener el dedo sobre una tarjeta, está pendiente.
- **Sin pedal de sustain** y sin el ruido de liberación del martillo, aunque VCSL incluye esas muestras. Las notas mantenidas duran como máximo 5 s.
- **Velocidad de las pulsaciones de manos:** el mapeo es provisional y no está calibrado frente a una medida.
- **Teclas iluminadas en silencio:** quedan por un problema de **detección**, no de audio. Un dedo que el detector mantiene en `PRESSED` durante 8–22 s sigue iluminando su tecla después de que la muestra (5 s) se apague: 15 s en `teclado_1` y 20 s en `teclado_2`. Solo se puede medir con etiquetas independientes.
- **Prueba de 15 s con cámara y audio (11 de octubre de 2026):**
  - 24 fps y 0 frames descartados;
  - 0 cortes de audio en 2407 bloques;
  - orden → hilo de audio ≤ 5,6 ms;
  - el callback más lento tardó **6,5 ms**, por encima del presupuesto de 5,8 ms. No causó cortes porque PortAudio tiene unos 17 ms de margen, pero es un riesgo.
- **CPU:** no se ha medido el efecto del audio sobre el FPS con la cámara activa. El hilo de audio usa menos de 1 ms por bloque.
- **Canciones:** solo cuatro melodías sencillas de dominio público; cualquier otra hay que añadirla en `songs/` con una licencia que lo permita.
- **El piano eléctrico es CC BY-NC:** para uso comercial habría que cambiarlo o pedir permiso.

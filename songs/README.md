# Canciones del Juego Rítmico 4K

El juego llena sus cuatro huecos con los primeros cuatro archivos, en orden alfabético, que haya en esta carpeta. Vienen incluidas cuatro melodías de dominio público, transcritas a mano para HandPiano y lentas para poder tocarlas con las manos (0,55–0,75 s entre notas):

| Archivo | Canción |
| --- | --- |
| `1_himno_alegria.json` | Himno de la Alegría (Beethoven) |
| `2_estrellita.json` | Estrellita, ¿dónde estás? (tradicional) |
| `3_cumpleanos_feliz.json` | Cumpleaños Feliz |
| `4_para_elisa.json` | Para Elisa, inicio (Beethoven) |

Formatos admitidos:

- **`.mid` / `.midi`**: archivo MIDI estándar. Las notas de todos los canales salvo el 10 (percusión) se reparten en 4 carriles por altura: la más grave va al carril 1 y la más aguda al carril 4.
- **`.json`**: partitura `handpiano-chart` v1. Permite elegir el carril y la nota de cada golpe:

```json
{"format": "handpiano-chart", "version": 1,
 "title": "Mi canción", "artist": "…", "source": "de dónde sale", "license": "…",
 "notes": [{"t": 3.0, "lane": 0, "midi": 60, "duration": 0.6},
           {"t": 3.7, "lane": 2, "midi": 67, "duration": 0.6}]}
```

`t` son los segundos desde el inicio, `lane` el carril (0–3), `midi` la nota que suena al acertar y `duration` cuánto suena, en segundos (opcional; por defecto 0,4). El formato completo está en `handpiano/game/chart.py`.

Usa solo archivos cuya licencia lo permita, e indica la fuente y la licencia en `source` y `license`. Si la carpeta queda vacía, el juego ofrece un **ejercicio de escala**, etiquetado como tal.

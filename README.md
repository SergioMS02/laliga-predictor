# LaLiga Predictor

Predicciones probabilísticas (1X2, goles esperados, marcador, +2.5, ambos marcan) para cada partido de LaLiga, actualizadas con toda la información disponible hasta el pitido inicial: resultados, tiros y xG de las últimas temporadas, cuotas de mercado y su movimiento, árbitro, estadio y desplazamiento, descanso real (incluidos partidos de Champions, Europa League, Conference y Copa), lo que se juega cada equipo en la clasificación, historial directo, sanciones probables, horario, parones internacionales, derbis, cambios de entrenador, clima previsto, bajas detectadas en noticias y, cuando se publica, la alineación oficial (separando el impacto en ataque y en defensa).

Coste: 0 €. Todo funciona sin claves; las claves gratuitas (Telegram, Gemini) solo activan extras.

## Resultados (backtest honesto)

Walk-forward: cada partido se predice solo con información anterior a él. Hiperparámetros y selección de variables se eligieron con las temporadas 2016/17-2020/21 (validación) y se evalúan en 2021/22-hoy (test), que nunca se usaron para decidir nada.

| Modelo | Log-loss ↓ | Brier ↓ | RPS ↓ | Acierto |
|---|---|---|---|---|
| Meta-modelo (final) | 0.9774 | 0.5812 | 0.1960 | 53.4% |
| Dixon-Coles | 0.9833 | 0.5853 | 0.1977 | 53.0% |
| Solo Elo | 0.9864 | 0.5865 | 0.1982 | 53.2% |
| Frecuencias históricas 1X2 | 1.0660 | 0.6439 | 0.2254 | 45.7% |
| Uniforme (1/3 cada uno) | 1.0986 | 0.6667 | 0.2343 | 45.7% |

1.969 partidos de test (2021/22 a 2026/27). ejecutado con el espejo sin cuotas y sin xG (ver *Limitaciones*).

Qué aportaron las variables nuevas en test: con solo las variables originales, el meta-modelo seleccionado da 0,9795; con todas disponibles y la selección actual, 0,9774. La selección final solo conserva `form_shots` y `timing` (vuelta de parón, primeras jornadas, entre semana). El resto de grupos con datos no superó el filtro: mejoras de ±0,001, indistinguibles del ruido.

Calibración del modelo final (probabilidad predicha → frecuencia real):

| Rango | Predicho | Observado | n |
|---|---|---|---|
| 0.0-0.1 | 0.068 | 0.087 | 231 |
| 0.1-0.2 | 0.157 | 0.154 | 830 |
| 0.2-0.3 | 0.257 | 0.255 | 2058 |
| 0.3-0.4 | 0.338 | 0.329 | 1200 |
| 0.4-0.5 | 0.448 | 0.472 | 625 |
| 0.5-0.6 | 0.549 | 0.508 | 447 |
| 0.6-0.7 | 0.643 | 0.700 | 293 |
| 0.7-0.8 | 0.746 | 0.745 | 141 |
| 0.8-0.9 | 0.839 | 0.835 | 79 |

Referencias: log-loss de 1,0986 = tirar una moneda de tres caras. Las casas de apuestas en LaLiga suelen quedar en torno a 0,96-0,97 con cuotas de cierre. Cuando se ejecuta con conexión a football-data.co.uk, el backtest añade automáticamente las filas `mercado`, `mercado_cierre` y `stacker_mercado` para que veas si el modelo aporta algo frente a las cuotas. **Si no las mejora, no apuestes con él.**

## Puesta en marcha (local)

```bash
git clone <tu-repo> && cd laliga-predictor
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python -m laliga backfill-espn   # una vez (~40 min): alineaciones, árbitros y calendario europeo/Copa desde 2016
python -m laliga tune            # una vez (~10 min): elige hiperparámetros y variables con 2016-2020
python -m laliga all       # descarga datos + backtest + entrena + predice
python -m laliga predict   # a partir de aquí, cuando quieras
```

Salida en `predictions/latest.md` (tabla legible) y `predictions/latest.json` (todo el detalle: xG, bajas, árbitro, clima, cuotas, modelo usado).

Con el historial de alineaciones cargado, estima cuánto pesan realmente las bajas con tus datos:

```bash
python -m laliga calibrate-availability
```

Ejecuta `tune` después de `backfill-espn`: así la selección de variables puede valorar árbitro, alineaciones y congestión europea con datos completos. Sin ese paso, los grupos sin datos en validación se descartan solos.

## Historial y cambios importantes

Cada vez que cambia la predicción de un partido se guarda una línea en `predictions/history.jsonl` con las probabilidades y una "foto" de todo lo que las produjo (goles esperados, bajas, forma, calendario, cuotas, árbitro, clima).

Si alguna probabilidad cambia **4 puntos o más** (configurable en `movements.threshold_pts`), el sistema explica el porqué. Para ello recalcula la predicción anterior cambiando un grupo de datos cada vez y mide cuánto mueve cada uno:

```
La Coruna – Vallecano: gana La Coruna 56% → 47%
  -9.8 pts · Bajas y alineaciones: alineaciones oficiales publicadas · La Coruna: no salen de inicio: …; déficit ataque 0%→40%, defensa 0%→54%
  +1.2 pts · Árbitro designado: designado José Luis Munuera Montero
```

Causas posibles: bajas y alineaciones (con nombres), resultados y forma (con los marcadores que hubo entre medias), calendario (Europa/Copa, descanso, horario), cuotas, árbitro, clima y reentrenamiento del modelo. Lo que no se puede separar aparece como "efecto combinado".

- `predictions/latest.md`: columna *Evolución 1* (mini-gráfico de la victoria local) y sección *Cambios importantes*.
- `predictions/evolucion.md`: por partido, todas las actualizaciones con hora, si ya había alineación y la explicación de cada cambio importante.
- `predictions/movements.jsonl`: los cambios importantes en formato máquina.
- Con Telegram configurado, cada cambio importante llega como aviso con sus causas.

## Automatización gratis (GitHub Actions)

1. Crea un repositorio en GitHub y sube el proyecto.
2. *Settings → Actions → General → Workflow permissions →* **Read and write**.
3. (Opcional) *Settings → Secrets and variables → Actions*: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `GEMINI_API_KEY`.
4. Listo. Los flujos hacen:

| Flujo | Cuándo | Qué |
|---|---|---|
| `predict.yml` | Diario 06:17 UTC | Actualiza datos y predice los próximos 8 días. Resumen por Telegram de las próximas 48 h. |
| `predict.yml` | Vie-lun, cada 20 min (10-22 UTC) | Vigila partidos que empiezan en <100 min. Si ESPN ya publica la alineación, re-predice y avisa con el cambio. |
| `weekly.yml` | Martes | Guarda alineaciones/árbitros de la jornada, rehace backtest, reentrena, recalibra bajas y mide el acierto real de lo predicho. |
| `tests.yml` | Cada push | Ejecuta los tests. |

Minutos: repo público = ilimitado. Repo privado = 2.000 min/mes gratis; este uso ronda 700-1.000.

## Fuentes de datos (todas gratuitas)

| Dato | Fuente | Notas |
|---|---|---|
| Resultados, tiros, córners, tarjetas, cuotas (2005-hoy) | football-data.co.uk | Cuotas previas (Pinnacle > media > Bet365) y de cierre. Respaldo sin cuotas en GitHub. |
| Cuotas de los próximos partidos | football-data.co.uk/fixtures.csv | |
| Calendario, horarios, resultados recientes | ESPN (API pública) + openfootball | ESPN es la fuente principal en la temporada en curso. |
| Alineaciones oficiales y árbitro | ESPN `summary` | API no documentada: el parser es defensivo y si falla el modelo sigue sin ese dato. |
| Clima a la hora del partido | Open-Meteo (previsión y archivo ERA5) | Sin clave. |
| Noticias de lesiones/sanciones/dudas | Google News RSS por equipo (+ RSS que añadas) | |
| Lectura de noticias | Gemini (tier gratis), Groq/OpenRouter u Ollama local; si no, reglas | El LLM solo puede devolver un JSON validado y los nombres deben existir en la plantilla. |
| xG por partido y por jugador (2014-hoy) | Understat (API interna, mismo método que la librería soccerdata) | Modelo de fuerza sobre xG, forma de xG y peso ofensivo de cada jugador. |
| Partidos de Champions, Europa League, Conference y Copa | openfootball (histórico) + ESPN (temporada actual y `backfill-espn`) | Solo fechas: descanso real, rotaciones previsibles. |
| Cambios de entrenador | Wikidata (P286 con fecha de inicio) + `data/coaches_manual.csv` | Cobertura irregular: el CSV manual tiene prioridad (`team,date,coach`). |
| Árbitro manual | `data/referee_overrides.csv` | LaLiga publica las designaciones ~48 h antes; ver `.example.csv`. |
| Otros partidos manuales | `data/other_matches_manual.csv` (`date,team,comp`) | Por si falta algún partido de Copa o Europa. |

## Cómo funciona

```
histórico ─┬─ Dixon-Coles (goles, ponderado en el tiempo) ─────┐
           ├─ mismo modelo sobre tiros a puerta y sobre xG ────┤
           ├─ Elo con multiplicador de goles ──────────────────┤
           └─ variables de contexto (ver tabla) ───────────────├─► meta-modelo (logística ─► 1 / X / 2
cuotas y su movimiento (si las hay) ───────────────────────────┘   multinomial)
bajas: alineación oficial > noticias ─► ajusta los goles esperados (ataque y defensa por separado)
```

Variables del meta-modelo, agrupadas. `tune` prueba los grupos uno a uno (selección hacia delante sobre validación) y solo deja los que mejoran; el resto se ignora sin coste.

| Grupo | Qué mide |
|---|---|
| `sot_model`, `xg_model` | Fuerza de ataque/defensa estimada sobre tiros a puerta y sobre xG |
| `form_shots`, `form_goals`, `form_xg` | Forma reciente (media exponencial) de tiros, córners, goles, puntos y xG |
| `momentum` | Forma de las últimas 2-3 jornadas frente a la de fondo |
| `venue`, `venue_form` | Rendimiento del local en casa y del visitante fuera |
| `rest`, `travel` | Días de descanso en liga; distancia del viaje y desplazamientos a las islas |
| `congestion` | Partido europeo/Copa en los 4 días anteriores o posteriores (rotaciones), descanso real, carga de 14 días |
| `stakes` | Lo que se juega cada equipo (título, Europa, descenso) según clasificación y jornadas restantes; equipos sin nada en juego |
| `h2h` | Diferencia de goles y empates en los últimos enfrentamientos directos |
| `discipline` | Roja en el partido anterior (sanción probable) y tendencia a amarillas |
| `timing` | Entre semana, hora de inicio, vuelta de parón internacional, primeras jornadas |
| `derby` | Derbi de ciudad o regional |
| `promoted` | Recién ascendidos |
| `coach` | Entrenador nuevo (≤ 60 días) |
| `referee` | Sesgo local y tarjetas del árbitro (con contracción a la media) |
| `weather` | Lluvia, viento y temperatura extrema |
| `no_fans` | Partidos a puerta cerrada (2020-21), para que no distorsionen la ventaja de local |
| `market_move` | Movimiento de las cuotas entre la primera lectura y el cierre |

- **Dixon-Coles**: ataque/defensa por equipo, nivel medio de la liga, ventaja de local, corrección de marcadores bajos, decaimiento exponencial, regularización y prior para ascendidos. Gradiente analítico (se ajusta en milisegundos).
- **Bajas**: importancia de cada jugador = % de titularidades en los últimos 10 partidos. En ataque se pondera por su xG + xA por 90 minutos (Understat); en defensa, por su posición (portero > defensa > medio > delantero). Los goles esperados propios se multiplican por `exp(-a·déficit_ataque)` y los del rival por `exp(b·déficit_defensa)`; `calibrate-availability` estima `a` y `b` con tus datos.
- **Sin fugas**: todas las variables se calculan en orden cronológico; hay tests que alteran el futuro y comprueban que el pasado no cambia. Excepción deliberada: el calendario europeo futuro (se conoce de antemano).

## LLM gratis para las noticias

En `config.yaml`, sección `news`:

- `llm_backend: gemini` + `GEMINI_API_KEY` (Google AI Studio, gratis con límites diarios).
- `llm_backend: openai_compatible` + `openai_base_url`:
  - Ollama local (`http://localhost:11434/v1`, sin clave, p. ej. `qwen2.5:7b-instruct`).
  - Groq (`https://api.groq.com/openai/v1`) u OpenRouter con `LLM_API_KEY`.
- `llm_backend: none` (por defecto): extracción por reglas, más conservadora.

Las noticias solo cuentan para jugadores que ya aparecen en el historial de alineaciones del equipo, así que ejecuta `backfill-espn` al menos una vez.

## Comandos

| Comando | Qué hace |
|---|---|
| `laliga update` | Descarga/actualiza datos |
| `laliga tune` | Ajusta hiperparámetros y elige variables (validación 2016-2020) → `data/state/tuned.json` |
| `laliga backtest [--weather]` | Evaluación en temporadas de test → `data/processed/backtest.json` |
| `laliga train` | Entrena el meta-modelo final |
| `laliga predict [--days N] [--notify]` | Predice los próximos partidos |
| `laliga watch-lineups [--notify]` | Re-predice partidos inminentes con alineación oficial |
| `laliga backfill-espn [--seasons 2016-2026]` | Historial de alineaciones, árbitros y calendario europeo/Copa |
| `laliga calibrate-availability` | Estima el peso real de las bajas |
| `laliga evaluate-live` | Rendimiento de las predicciones ya emitidas |
| `laliga all` | update + backtest + train + predict |

Todos aceptan `--offline` (solo caché) y `-v` (registro detallado).

## Limitaciones

- Los números de arriba se obtuvieron sin cuotas, sin xG, sin alineaciones ni árbitros históricos y con calendario europeo solo desde 2020. Repite `backfill-espn` → `tune` → `backtest` en tu máquina para medirlo con todo.
- `tune` es conservador a propósito: un grupo entra solo si mejora la log-loss media en ≥0,0005, mejora en al menos el 60 % de las temporadas de validación y tiene datos en la mitad de las filas. Así se evita que variables con efecto mínimo metan ruido.
- **El mercado es muy difícil de batir.** El valor realista del proyecto es tener probabilidades bien calibradas y entender qué mueve cada partido, no ganar dinero.
- La API de ESPN no es oficial y puede cambiar. Si deja de responder, el sistema sigue con calendario de openfootball y sin alineaciones.
- Para entrenar, el clima histórico es el observado (ERA5), no la previsión que había antes del partido: sesgo optimista pequeño en esa variable.
- El efecto de árbitro y clima es pequeño. `tune` los descarta si no mejoran la validación; con `backtest --weather` puedes comprobarlo tú.
- Las variables modernas (xG desde 2014, horarios desde 2019, alineaciones y árbitros según lo que devuelva ESPN) no existen para temporadas antiguas: por eso validación y test empiezan en 2016 y 2021.
- Wikidata no tiene todos los cambios de entrenador; complétalo con `data/coaches_manual.csv` si quieres que esa variable cuente.
- El movimiento de cuotas en predicción se mide entre lecturas propias (cada ejecución guarda una); en entrenamiento, entre la cuota previa y la de cierre de football-data. Son parecidos, no idénticos.
- Google News puede limitar peticiones si se abusa; el flujo lo consulta solo para los partidos del horizonte.

## Estructura

```
config.yaml                 parámetros (los de tuned.json tienen prioridad)
src/laliga/
  data/      results · openfootball · espn · understat · other_comps · coaches · weather · news · http
  features/  sequential (Elo, forma, descanso, viaje, árbitro) · context (clasificación, h2h, Europa,
             horario, derbi, entrenador...) · availability (bajas ataque/defensa)
  models/    dixon_coles · stacker · market
  pipeline.py  backtest walk-forward y tabla de entrenamiento
  predict.py   predicción con toda la información del momento
  movements.py historial de probabilidades y explicación de cambios
  tune.py · history.py · notify.py · cli.py
tests/       102 tests, sin red (liga sintética + ejemplos reales de cada fuente)
.github/workflows/  predict · weekly · tests
```

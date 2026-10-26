"use strict";
/* ======================================================================
   Versión en inglés. La web se escribe en español; con el idioma en «EN»
   se traducen los textos al vuelo (nodos de texto y atributos), con frases
   exactas y patrones para los textos con cifras. Las fechas y los números
   pasan a formato inglés. Los nombres de equipos y jugadores no cambian.
   ====================================================================== */
const I18N = (() => {
  let lang = "es";
  try { lang = localStorage.getItem("lang") || ((navigator.language || "").startsWith("es") ? "es" : (navigator.language || "").startsWith("en") ? "en" : "es"); } catch (e) { /* sin almacenamiento */ }

  const EXACT = {
    "Quiniela del modelo": "The model's pools coupon", "gana el local": "home win", "empate": "draw", "gana el visitante": "away win",
    "Partidos": "Matches", "En directo": "Live", "Directo": "Live", "Quiniela": "Pools", "Apuestas": "Bets", "Clasificación": "Table",
    "Liga": "League", "Equipos": "Teams", "Jugadores": "Players", "Otras ligas": "Other leagues", "Aciertos": "Track record", "Más": "More",
    "Apuestas simuladas": "Simulated bets", "Jugadores y árbitros": "Players and referees", "Otras ligas, copas y Europa": "Other leagues, cups and Europe",
    "Aciertos y fuentes": "Track record and sources", "Más secciones": "More sections", "Secciones": "Sections", "Secciones principales": "Main sections",
    "Próximo partido": "Next match", "Favorito más claro": "Clearest favourite", "Partido más abierto": "Most open match",
    "Resultados al momento": "Live scores", "Próximos partidos": "Upcoming matches",
    "Cada casilla se rellena según la probabilidad. El círculo marca el resultado más probable. Pulsa un partido para ver cómo ha cambiado y por qué.":
      "Each box fills up with the probability. The circle marks the most likely result. Tap a match to see how it has changed and why.",
    "Evolución local": "Home win trend", "Novedades": "News", "Partido": "Match", "Nº": "No.",
    "Probabilidades": "Probabilities", "Más mercados": "More markets", "Alineaciones": "Line-ups", "Estadísticas": "Stats",
    "Cómo han cambiado las probabilidades": "How the probabilities have changed", "Por qué han cambiado": "Why they changed",
    "El partido en números": "The match in numbers", "Efecto de las bajas": "Effect of absences", "Marcadores más probables (%)": "Most likely scores (%)",
    "Tarjetas y córners": "Cards and corners", "Árbitro aún sin designar.": "Referee not yet appointed.", "Bajas, sanciones y dudas": "Injuries, suspensions and doubts",
    "Contexto del partido": "Match context", "Clima:": "Weather:", "Mercado:": "Market:", "Cuotas Bet365:": "Bet365 odds:", "Árbitro:": "Referee:",
    "Mejor cuota del mercado:": "Best market odds:", "Contexto:": "Context:", "Polymarket:": "Polymarket:", "Más de 2,5 goles": "Over 2.5 goals",
    "Marcan los dos": "Both teams score", "Alineaciones oficiales": "Official line-ups", "Once tipo según titularidades": "Usual XI by starts",
    "Plantilla y disponibilidad": "Squad and availability", "Puntos por partido": "Points per match", "Goles a favor por partido": "Goals scored per match",
    "Goles en contra por partido": "Goals conceded per match", "xG a favor por partido": "xG for per match", "xG en contra por partido": "xG against per match",
    "Tiros por partido": "Shots per match", "Tiros a puerta por partido": "Shots on target per match", "Porterías a cero": "Clean sheets",
    "Fuerza (Elo)": "Strength (Elo)", "Solo modelo (sin cuotas todavía)": "Model only (no odds yet)",
    "Sin cuotas de las casas para este partido: el efecto se aplica entero a la predicción.": "No bookmaker odds for this match: the full effect goes into the prediction.",
    "Probabilidad de marcar al menos un gol si es titular, según su xG por 90 minutos y los goles esperados de su equipo.":
      "Probability of scoring at least once if he starts, from his xG per 90 minutes and his team's expected goals.",
    "Últimos cambios importantes": "Latest major changes", "Bajas y alineaciones": "Absences and line-ups", "Calendario y descanso": "Schedule and rest",
    "Resultados recientes": "Recent results", "Cuotas de mercado": "Market odds", "Precio de Polymarket": "Polymarket price", "Clima": "Weather", "Árbitro": "Referee",
    "Modelo": "Model", "Efecto combinado de varios factores": "Combined effect of several factors",
    "Clasificación y proyección": "Table and projection", "El resto de la temporada simulado miles de veces con el modelo.": "The rest of the season simulated thousands of times with the model.",
    "Puntos que harán falta": "Points needed", "Puntos del último que entra en cada zona, en las simulaciones de lo que queda de temporada.": "Points of the last team inside each zone, across the simulations of the rest of the season.",
    "Simulador: ¿y si…?": "Simulator: what if…?", "Fija el resultado de los partidos que quieras y mira cómo cambia la liga.": "Set the result of any match you like and see how the league changes.",
    "Tabla de la suerte": "Luck table", "Puntos reales frente a los que merecía cada equipo por sus ocasiones.": "Actual points against the points each team deserved from its chances.",
    "Pichichi": "Top scorer", "Quién acabará como máximo goleador, simulando los goles que le quedan a cada jugador.": "Who will finish as top scorer, simulating the goals each player has left.",
    "Otros premios": "Other awards", "Máximo asistente, Zamora y equipos más y menos goleados, con las mismas simulaciones.": "Top assister, best goalkeeper and the highest- and lowest-scoring teams, from the same simulations.",
    "El modelo frente a Polymarket": "The model against Polymarket",
    "Lo que dice la simulación comparado con el precio de los mercados de predicción de Polymarket (el precio de un contrato es la probabilidad que le da el dinero). Solo se leen los precios.":
      "The simulation compared with Polymarket's prediction-market prices (a contract's price is the probability the money gives it). Prices are only read.",
    "Dónde puede acabar cada equipo": "Where each team can finish", "Probabilidad de terminar en cada puesto. Cuanto más intenso el color, más probable.": "Probability of finishing in each position. The stronger the colour, the more likely.",
    "Ordenados por su fuerza actual (Elo). Elige un equipo para ver sus números.": "Sorted by current strength (Elo). Pick a team to see its numbers.",
    "Goles, asistencias, xG, xA, minutos, tarjetas, penaltis y mapa de tiros de cada jugador.": "Goals, assists, xG, xA, minutes, cards, penalties and shot map for each player.",
    "Árbitros": "Referees", "Tarjetas, faltas, goles y victorias locales con cada árbitro.": "Cards, fouls, goals and home wins with each referee.",
    "Segunda, Liga F, Premier League, Serie A, Bundesliga y Ligue 1: clasificación, proyección y próximos partidos.": "Segunda, Liga F, Premier League, Serie A, Bundesliga and Ligue 1: table, projection and upcoming matches.",
    "Copas y Europa": "Cups and Europe", "Copa del Rey, Liga F y equipos españoles en Champions, Europa League y Conference.": "Copa del Rey, Liga F and Spanish teams in the Champions League, Europa League and Conference League.",
    "Marcadores, goles, tarjetas, cambios y estadísticas al momento (se actualiza solo cada 30 segundos). Pulsa un partido para ver el detalle y su probabilidad en directo.":
      "Scores, goals, cards, substitutions and stats as they happen (refreshes every 30 seconds). Tap a match for the details and its live probability.",
    "Quiniela oficial": "Official pools coupon",
    "El boleto real de Loterías rellenado por el modelo: la apuesta sencilla (lápiz) y una reducida con dobles y triples (marcas suaves). Se cierra cuando empieza el primer partido.":
      "The real Loterías coupon filled in by the model: the single bet (pencil) and a reduced system with doubles and trebles (soft marks). It closes when the first match starts.",
    "Historial de jornadas": "Rounds so far", "Aciertos y premios reales de cada boleto, con la tabla oficial de premios de cada jornada.": "Hits and real prizes of each coupon, with each round's official prize table.",
    "Quinigol": "Quinigol", "Goles de cada equipo (0, 1, 2 o M) en los próximos partidos de LaLiga.": "Each team's goals (0, 1, 2 or M) in the upcoming matches.",
    "Dinero ficticio: el modelo apuesta solo, con las cuotas de Bet365, y aquí queda todo lo que habría ganado o perdido.": "Play money: the model bets on its own at Bet365 odds, and everything it would have won or lost is kept here.",
    "Si lo hubieras hecho en temporadas pasadas": "If you had done it in past seasons",
    "Las mismas reglas aplicadas a cada partido de las temporadas de prueba, con la cuota de Bet365 que había antes del partido.": "The same rules applied to every match in the test seasons, at the Bet365 odds before kick-off.",
    "Aciertos de esta temporada": "This season's record", "Lo que dijo el modelo antes de cada partido frente a lo que pasó.": "What the model said before each match against what happened.",
    "Goles, goleadores, tarjetas y córners": "Goals, scorers, cards and corners", "Lo que dijo el modelo antes de cada partido en los mercados secundarios frente a lo que pasó.": "What the model said before each match in the secondary markets against what happened.",
    "Polymarket frente a las casas y al modelo": "Polymarket against the bookmakers and the model",
    "Precio de Polymarket justo antes de cada partido, comparado con la cuota de cierre de las casas y con el modelo en los mismos partidos.": "Polymarket's price just before each match, compared with the bookmakers' closing odds and the model in the same matches.",
    "Lo que se ha medido": "What has been measured", "Cada mejora se comprueba con partidos pasados antes de darla por buena; aquí, lo que salió.": "Every improvement is tested on past matches before it is accepted; here is what came out.",
    "¿Cuánto acierta a largo plazo?": "How accurate is it in the long run?", "Prueba con temporadas pasadas: cada partido se predijo solo con lo que se sabía antes de jugarse.": "Tested on past seasons: each match was predicted only with what was known before it was played.",
    "Error (log-loss)": "Error (log-loss)", "Acierta el resultado": "Gets the result right", "Casas de apuestas (previas)": "Bookmakers (pre-match)", "Este modelo + cuotas": "This model + odds",
    "Solo goles (Dixon-Coles)": "Goals only (Dixon-Coles)", "Al azar": "Random", "Probabilidad que dio el modelo": "Probability given by the model", "Lo que ocurrió": "What happened",
    "Si el modelo dice 40 %, ese resultado debería ocurrir unas 40 de cada 100 veces. Cuanto más cerca de la diagonal, más fiable.": "If the model says 40%, that result should happen about 40 times in 100. The closer to the diagonal, the more reliable.",
    "Fuentes de la última actualización": "Sources of the last update",
    "Son probabilidades, no certezas. Las apuestas y la quiniela son simuladas: no es un consejo de apuesta. Juega con responsabilidad (+18).": "These are probabilities, not certainties. The bets and the pools are simulated: this is not betting advice. Gamble responsibly (18+).",
    "Modo claro u oscuro": "Light or dark mode", "Cambiar entre modo claro y oscuro": "Switch between light and dark mode",
    "Detalle del partido": "Match details", "Pleno al 15": "Pleno al 15", "Premio (10 o más)": "Prize (10 or more)", "Pleno de 14": "All 14 right", "Apuestas": "Bets",
    "Aciertos esperados": "Expected hits", "Premio esperado": "Expected prize", "Apuesta sencilla": "Single bet", "Reducida": "Reduced system", "Contra la mayoría": "Against the crowd",
    "Fuentes de cada partido:": "Source of each match:", "Jornada": "Round", "Boleto de la quiniela": "Pools coupon", "Red.": "Red.", "Val.": "Val.",
    "Favorito del modelo": "Model favourite", "Al resultado más probable de cada partido.": "On the most likely result of each match.", "Apuesta de valor": "Value bet",
    "Solo cuando la cuota paga más de lo que dice el modelo.": "Only when the odds pay more than the model says.", "Más/menos 2,5 goles": "Over/under 2.5 goals",
    "Más o menos de 2,5 goles cuando esa cuota tiene valor.": "Over or under 2.5 goals when those odds have value.", "Kelly (fondo de 100 €)": "Kelly (€100 bankroll)",
    "Las apuestas de valor, con importe según la ventaja (Kelly) sobre un fondo de 100 €.": "The value bets, staked by edge (Kelly) on a €100 bankroll.",
    "Frente al cierre": "Against the close", "Prob. del modelo": "Model prob.", "¿Qué es «frente al cierre»?": "What is 'against the close'?",
    "Beneficio acumulado apuesta a apuesta": "Cumulative profit bet by bet", "¿Y apostando en otras casas?": "And betting with other bookmakers?",
    "Mejor cuota del mercado": "Best market odds", "Filtrar por estrategia": "Filter by strategy", "Pts al final": "Final pts", "Champions": "Champions League",
    "Europa": "Europe", "Descenso": "Relegation", "Equipo": "Team", "Goles": "Goals", "DG": "GD", "Pts": "Pts", "Forma": "Form", "PJ": "P", "G": "W", "E": "D", "P": "L",
    "Al final": "At the end", "Ahora": "Now", "Prob.": "Prob.", "Jugador": "Player", "Portero": "Goalkeeper", "Goles/partido": "Goals/match",
    "Máximo asistente": "Top assister", "Zamora (aprox.)": "Best goalkeeper (approx.)", "Equipo más goleador": "Highest-scoring team", "Equipo menos goleado": "Fewest goals conceded",
    "Goles al final": "Final goals", "Encajados al final": "Final conceded", "Campeón": "Champion", "Acabar entre los 2 primeros": "Top 2 finish", "Jugar la Champions": "Champions League qualification",
    "Descender": "Relegated", "Dif.": "Diff.", "Puntos": "Points", "Merecidos": "Deserved", "Diferencia": "Difference", "Goles / xG": "Goals / xG", "Encajados / xG": "Conceded / xG",
    "Ganar la liga": "Win the league", "Entrar en Champions (4.º)": "Champions League (4th)", "Entrar en Europa (7.º)": "Europe (7th)", "Salvarse (17.º)": "Stay up (17th)",
    "Sim.": "Sim.", "Fuerza (Elo) en las dos últimas temporadas": "Strength (Elo) over the last two seasons", "Puntos finales proyectados, día a día": "Projected final points, day by day",
    "Últimos partidos": "Last matches", "Posición": "Position", "Balance": "Record", "Goles por partido": "Goals per match", "xG por partido": "xG per match",
    "En verano todos los equipos se acercan un poco a la media: por eso hay un salto entre temporadas.": "In the summer every team is pulled a little towards the average: that is why there is a jump between seasons.",
    "Asistencias": "Assists", "Minutos": "Minutes", "Tarjetas": "Cards", "Min.": "Min.", "Buscar jugador": "Search player", "Buscar jugador…": "Search player…",
    "Partidos": "Matches", "Amarillas": "Yellows", "Rojas": "Reds", "Faltas": "Fouls", "Gana el local": "Home wins", "Árbitro": "Referee",
    "Segunda División": "Segunda División", "Ascenso": "Promotion", "Playoff": "Play-off", "Competición": "Competition",
    "No hay partidos de copa ni de Europa en los próximos días.": "No cup or European matches in the coming days.",
    "Aciertos por semana": "Hits per week", "¿Acertó?": "Right?", "Resultado": "Result", "Sí": "Yes", "No": "No", "Predicho (media)": "Predicted (mean)", "Real": "Actual", "Error": "Error", "Mercado": "Market",
    "Polymarket (antes del inicio)": "Polymarket (before kick-off)", "Cierre de las casas (Pinnacle o media)": "Bookmakers' close (Pinnacle or average)", "Nuestro modelo (con cuotas de apertura)": "Our model (with opening odds)",
    "Mezcla elegida": "Chosen blend", "Fuente": "Source", "Log-loss": "Log-loss", "Proyección de la temporada": "Season projection", "Probabilidad en directo": "Live probability", "Polymarket en goles": "Polymarket on goals",
    "Doble oportunidad y empate no válido": "Double chance and draw no bet", "Hándicap asiático": "Asian handicap", "Descanso": "Half time", "Descanso / final": "Half time / full time",
    "Marcador exacto": "Correct score", "Quinigol: goles de cada equipo": "Quinigol: each team's goals", "Apuesta": "Bet", "Cuota justa": "Fair odds", "Línea": "Line", "Gana": "Wins",
    "12 (no hay empate)": "12 (no draw)", "Marcan los dos": "Both teams score", "Empate al descanso": "Draw at half time", "Gol en la primera parte": "Goal in the first half",
    "Local": "Home", "Visitante": "Away", "Boleto": "Coupon", "Cargando detalles…": "Loading details…", "Lo que ha pasado": "What has happened", "Sin estadísticas todavía.": "No stats yet.",
    "Aún no hay incidencias.": "Nothing has happened yet.", "Descanso": "Half time", "Final": "Full time", "Aplazado": "Postponed", "Hoy": "Today", "Ayer": "Yesterday", "Mañana": "Tomorrow",
    "Posesión (%)": "Possession (%)", "Tiros": "Shots", "Tiros a puerta": "Shots on target", "Córners": "Corners", "Fueras de juego": "Offsides", "Paradas": "Saves",
    "Pases buenos": "Completed passes", "Acierto en el pase": "Pass accuracy", "Gol": "Goal", "Amarilla": "Yellow card", "Roja": "Red card", "Cambio": "Substitution",
    "Empieza el partido": "Kick-off", "Empieza la segunda parte": "Second half starts", "Penalti": "Penalty", "Penalti fallado": "Missed penalty", "Gol en propia": "Own goal",
    "Presión": "Pressing", "Le dejan el balón": "Given the ball", "Llegadas al área": "Box entries", "Llegadas que concede": "Box entries conceded", "Ocasiones (xG)": "Chances (xG)", "Ocasiones que concede": "Chances conceded",
    "Presión alta y dominio": "High press and control", "Posesión y control": "Possession and control", "Bloque bajo que sufre": "Deep block under pressure", "Bloque bajo y contra": "Deep block and counter",
    "Pocas ocasiones": "Few chances", "Equilibrado": "Balanced", "Disponible": "Available", "Titular": "Starter", "Suplente": "Sub", "Duda": "Doubt", "Baja": "Out", "Sancionado": "Suspended",
    "Porteros": "Goalkeepers", "Defensas": "Defenders", "Centrocampistas": "Midfielders", "Delanteros": "Forwards", "hoy": "today", "Aparecerán tras la próxima actualización.": "Will appear after the next update.",
    "Aparecerá tras la próxima actualización.": "Will appear after the next update.", "Aparecerá con los próximos partidos.": "Will appear with the next matches.",
    "Los mercados aparecerán en la próxima actualización.": "The markets will appear after the next update.", "No quedan partidos por simular.": "No matches left to simulate.",
    "Sin novedades": "No news", "Disponible": "Available", "Sin datos": "No data", "derbi": "derby", "vuelta de parón internacional": "back from the international break",
    "Todavía no ha terminado ninguna jornada con boleto del modelo.": "No round with a model coupon has finished yet.",
    "No hay partidos en los próximos días. La próxima jornada aparecerá en cuanto se publique el calendario.": "No matches in the coming days. The next round will appear as soon as the fixtures are published.",
    "No hay partidos de estas competiciones hoy.": "No matches in these competitions today.", "Ver más": "See more", "Quitar": "Remove",
    "Idioma": "Language", "Balance total": "Total balance", "Apuestas simuladas": "Simulated bets",
  };

  // [patrón, reemplazo] para textos con cifras o nombres
  const PATTERNS = [
    [/^Actualizado el (.+?)\. (\d+) partidos en los próximos días\.$/, "Updated $1. $2 matches in the coming days."],
    [/^Actualizado el (.+)$/, "Updated $1"], [/^en (\d+) h (\d+) min$/, "in $1 h $2 min"], [/^en (\d+) min$/, "in $1 min"], [/^en (\d+) días?$/, "in $1 days"],
    [/^(\d+) % ante (.+)$/, "$1% against $2"], [/^(\d+) partidos más hoy$/, "$1 more matches today"], [/^(\d+) abiertas?$/, "$1 open"], [/^\+(\d+) abiertas?$/, "+$1 open"],
    [/^Premio (\d+) % · reducida (\d+) %$/, "Prize $1% · reduced $2%"], [/^Sin cambios \((\d+) %\)$/, "No change ($1%)"], [/^Local (\d+) → (\d+) %$/, "Home $1 → $2%"],
    [/^(\d+) bajas? o dudas?$/, "$1 absences or doubts"], [/^(\d+) cambios? importantes?$/, "$1 major change(s)"],
    [/^Ningún cambio ha superado aún los (\d+) puntos\. Cuando ocurra, aquí verás qué lo provocó\.$/, "No change has passed $1 points yet. When one does, you will see what caused it here."],
    [/^Cambios de (\d+) puntos o más y lo que los provocó\.$/, "Changes of $1 points or more and what caused them."],
    [/^Goles esperados: (.+)$/, "Expected goals: $1"], [/^Bajas: (.+)$/, "Absences: $1"], [/^Bajas y dudas: (.+)$/, "Absences and doubts: $1"],
    [/^En el once probable: (.+)$/, "In the likely XI: $1"], [/^Modelo (\d+) %, casas (\d+) %$/, "Model $1%, bookmakers $2%"], [/^Modelo (\d+) %, Polymarket (\d+) %$/, "Model $1%, Polymarket $2%"],
    [/^Goles de (.+) en filas y de (.+) en columnas$/, "$1 goals in rows, $2 goals in columns"], [/^Más de ([\d,]+): (\d+) %$/, (m, a, b) => `Over ${a.replace(",", ".")}: ${b}%`],
    [/^Más de ([\d,]+)$/, (m, a) => `Over ${a.replace(",", ".")}`], [/^(.+): jugadores clave$/, "$1: key players"], [/^(.+) gana al descanso$/, "$1 ahead at half time"],
    [/^(.+) marca$/, "$1 scores"], [/^(.+) deja la portería a cero$/, "$1 keeps a clean sheet"], [/^(.+) \(empate devuelve\)$/, "$1 (draw refunded)"],
    [/^1X \((.+) o empate\)$/, "1X ($1 or draw)"], [/^X2 \(empate o (.+)\)$/, "X2 (draw or $1)"],
    [/^(\d+) jugadores · (.+)$/, (m, n, r) => `${n} players · ${r.replace(/bajas?/, "out").replace(/en duda/, "doubtful")}`],
    [/^Puntos por partido \((.+) en casa, (.+) fuera\)$/, "Points per match ($1 at home, $2 away)"], [/^(.+) contra (.+): local (\d+) %, empate (\d+) %, visitante (\d+) %$/, "$1 v $2: home $3%, draw $4%, away $5%"],
    [/^Detalle de (.+) contra (.+)$/, "Details of $1 v $2"], [/^(.+) · titular en el (\d+) % de partidos · xG\+xA\/90: (.+)$/, "$1 · starter in $2% of matches · xG+xA/90: $3"],
    [/^gana (.+)$/, "$1 win"], [/^Predicho (\d+) %, ocurrió (\d+) % \((\d+) casos\)$/, "Predicted $1%, happened $2% ($3 cases)"],
    [/^(\d+)º · (\d+) pts$/, "$1 · $2 pts"], [/^(\d+) puntos$/, "$1 points"], [/^vs (.+): gana (\d+) %, empate (\d+) %$/, "v $1: win $2%, draw $3%"],
    [/^(.+) partidos por jugar, simulados con el modelo \(incertidumbre incluida\)\.$/, "$1 matches left, simulated with the model (uncertainty included)."],
    [/^Comprobado el (.+)$/, "Checked $1"], [/^Semana del (.+): (\d+) de (\d+)$/, "Week of $1: $2 of $3"], [/^(.+) M\$ negociados$/, "$1 M$ traded"], [/^(.+) mil \$ negociados$/, "$1k $ traded"],
    [/^(.+) \$ negociados$/, "$1 $ traded"], [/^\((.+) negociados\)$/, "($1 traded)"], [/^(.+) negociados en este contrato$/, "$1 traded in this contract"],
    [/^no se usa: poco volumen \((.+)\)$/, "not used: low volume ($1)"], [/^no se usa: poca liquidez \((.+)\)$/, "not used: low liquidity ($1)"], [/^no se usa: en juego$/, "not used: in play"],
    [/^(.+), puesto (\d+): (.+)$/, "$1, position $2: $3"], [/^Estilo de juego: (.+)$/, "Playing style: $1"], [/^Fuerza \(Elo\) por temporada desde (\d+)$/, "Strength (Elo) by season since $1"],
    [/^Jornada (\d+)$/, "Round $1"], [/^Boleto abierto: se actualiza hasta el primer partido \(última vez (.+)\)$/, "Coupon open: updated until the first match (last time $1)"],
    [/^Más probable: (.+) \((\d+) %\)$/, "Most likely: $1 ($2%)"], [/^1 entre (.+)$/, "1 in $1"], [/^de 100 € iniciales \((.+)\)$/, "from an initial €100 ($1)"],
    [/^(\d+) ap\.$/, "$1 bets"], [/^\((\d+) ap\.\)$/, "($1 bets)"], [/^Próximos (\d+) de (\d+)\.?$/, "Next $1 of $2."], [/^Quitar mis (\d+) resultados$/, "Remove my $1 results"],
    [/^(\d+) temporadas simuladas en tu navegador\..*$/, "$1 seasons simulated in your browser. In green or red, the change against the projection without your results."],
    [/^9 de cada 10 veces basta con (\d+)$/, "Enough 9 times out of 10: $1"], [/^en (.+): (\d+-\d+) \((.+)\)$/, "at $1: $2 ($3)"], [/^vs (.+): (\d+-\d+) \((.+)\)$/, "v $1: $2 ($3)"],
    [/^Ver las (\d+) apuestas$/, "See all $1 bets"], [/^(\d+) en juego$/, "$1 in play"], [/^(\d+) partidos más hoy$/, "$1 more matches today"],
    [/^actualizado a las (.+)$/, "updated at $1"],
    [/^\. En el modelo de goles: (.+) puntos\.$/, ". In the goals model: $1 points."], [/^Quiniela (\d+)ª$/, "Pools round $1"],
    [/^Medido en ([\d.,]+) partidos de (\d+) a (\d+): las cuotas de las casas ya descuentan las bajas y los onces, así que sumar este efecto encima empeoraba la predicción \(error (.+) frente a (.+)\)\..*$/,
      "Measured on $1 matches from $2 to $3: bookmaker odds already price in absences and line-ups, so adding this effect on top made the prediction worse (error $4 against $5). That is why the final probability does not add it again; it does change the expected goals, the scorers and the prediction when there are no odds."],
    [/^Más papeletas de ver amarilla — (.+)$/, "Most likely to be booked — $1"], [/^(\d+) °C, lluvia (.+) mm, viento (.+) km\/h$/, "$1 °C, rain $2 mm, wind $3 km/h"],
    [/^(.+) juega en Europa o Copa justo después(.*)$/, "$1 plays in Europe or the Cup right after$2"], [/^(.+) viene de jugar en Europa o Copa(.*)$/, "$1 is coming off a European or Cup match$2"],
    [/^Lo que queda de temporada \((\d+) partidos\) simulado ([\d.,]+) veces con el modelo\..*$/, "The rest of the season ($1 matches) simulated $2 times with the model. Ties broken approximately by goal difference."],
    [/^Champions: 4 primeros\..*$/, "Champions League: top 4. Europe: top 7 (Champions League included). In brackets, the points range in 8 of every 10 simulations."], [/^(.+) \(duda\)$/, "$1 (doubt)"], [/^(.+) \(sanción\)$/, "$1 (suspended)"],
  ];

  function tr(s) {
    const t = s.replace(/\s+/g, " ").trim();
    if (!t) return null;
    if (EXACT[t]) return s.replace(t, EXACT[t]);
    for (const [re, rep] of PATTERNS) {
      if (re.test(t)) return s.replace(t, t.replace(re, rep));
    }
    return null;
  }
  function translate(root) {
    if (lang !== "en" || !root) return;
    const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    let n;
    while ((n = w.nextNode())) nodes.push(n);
    for (const x of nodes) {
      if (x.parentElement && x.parentElement.closest("script,style")) continue;
      const v = tr(x.nodeValue);
      if (v != null && v !== x.nodeValue) x.nodeValue = v;
    }
    const els = root.querySelectorAll ? root.querySelectorAll("[title],[aria-label],[placeholder]") : [];
    for (const e of els) for (const a of ["title", "aria-label", "placeholder"]) {
      const v0 = e.getAttribute(a);
      if (v0) { const v = tr(v0); if (v != null) e.setAttribute(a, v); }
    }
  }
  // fechas y números en formato inglés
  function patchLocale() {
    const swap = (loc) => (lang === "en" && (loc === "es-ES" || loc === "es" || loc === undefined) ? "en-GB" : loc);
    for (const proto of [Date.prototype]) {
      for (const f of ["toLocaleString", "toLocaleDateString", "toLocaleTimeString"]) {
        const orig = proto[f];
        proto[f] = function (loc, opt) { return orig.call(this, swap(loc), opt); };
      }
    }
    const nOrig = Number.prototype.toLocaleString;
    Number.prototype.toLocaleString = function (loc, opt) { return nOrig.call(this, swap(loc), opt); };
  }
  function setup() {
    document.documentElement.lang = lang;
    if (lang === "en") {
      patchLocale();
      let pending = false;
      new MutationObserver(() => {
        if (pending) return;
        pending = true;
        requestAnimationFrame(() => { pending = false; translate(document.body); });
      }).observe(document.body, { childList: true, subtree: true, characterData: false });
      translate(document.body);
    }
    const b = document.getElementById("lang");
    if (b) {
      b.textContent = lang === "en" ? "ES" : "EN";
      b.setAttribute("aria-label", lang === "en" ? "Ver en español" : "View in English");
      b.addEventListener("click", () => {
        try { localStorage.setItem("lang", lang === "en" ? "es" : "en"); } catch (e) { /* sin almacenamiento */ }
        location.reload();
      });
    }
  }
  return { setup, translate, get lang() { return lang; } };
})();
I18N.setup();

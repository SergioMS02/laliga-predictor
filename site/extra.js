"use strict";
/* ======================================================================
   Secciones ampliadas: más mercados de cada partido, previa, simulador
   «¿y si…?», puntos necesarios, tabla de la suerte, premios, estilos,
   Elo histórico, jugadores (con mapa de tiros), árbitros, otras ligas,
   copas y Europa, y Quinigol.
   Usa las utilidades de app.js (name, esc, pct, dec, fmtTs, DATA…).
   ====================================================================== */

const pc1 = (v) => (v == null ? "–" : v >= 0.995 ? ">99" : v > 0 && v < 0.005 ? "<1" : String(Math.round(100 * v)));
const fair = (p) => (p > 0 ? dec(1 / p, 2) : "–");

/* ---------- Previa y más mercados de un partido ---------- */
function previewHtml(m) {
  const t = ((DATA.extra || {}).previews || {})[`${m.home}|${m.away}`];
  return t ? `<p class="preview">${esc(t)}</p>` : "";
}

function marketsHtml(m) {
  const k = m.markets;
  if (!k) return '<p class="quiet">Los mercados aparecerán en la próxima actualización.</p>';
  const H = esc(name(m.home)), A = esc(name(m.away));
  const pmore = (m.polymarket || {}).more || {};
  const hasPm = Object.keys(pmore).length > 0;
  const pmCell = (v) => (hasPm ? `<td class="num">${v == null ? '<span class="quiet">–</span>' : `${pc1(v)} %`}</td>` : "");
  const row = (label, p, pmv) => `<tr><th scope="row" class="l">${label}</th><td class="num">${pc1(p)} %</td><td class="num quiet">${fair(p)}</td>${pmCell(pmv)}</tr>`;
  const tbl = (title, rows) => `<div class="mk"><h4>${title}</h4><table class="track mkt"><thead><tr><th scope="col" class="l">Apuesta</th><th scope="col" class="num">Prob.</th><th scope="col" class="num">Cuota justa</th>${hasPm ? '<th scope="col" class="num">Polymarket</th>' : ""}</tr></thead><tbody>${rows}</tbody></table></div>`;
  const pt = (x) => ((pmore.totals || {})[x] || {}).p;
  const pht = (pmore.half_time || {}).p || [];
  const dc = k.double_chance, tt = k.totals, ht = k.half_time;
  const ah = Object.entries(k.asian || {}).filter(([l]) => ["-1.5", "-1.0", "-0.5", "0.0", "0.5", "1.0", "1.5"].includes(l));
  const ahRows = ah.map(([l, a]) => {
    const lv = Number(l);
    const lab = lv === 0 ? "0" : `${lv > 0 ? "+" : "−"}${dec(Math.abs(lv), Math.abs(lv) % 1 ? 1 : 0)}`;
    return `<tr><th scope="row" class="l">${H} ${lab}</th><td class="num">${pc1(a.win)} %${a.push > 0.001 ? ` <span class="quiet small">(dev. ${pc1(a.push)} %)</span>` : ""}</td><td class="num quiet">${a.fair_home ? dec(a.fair_home, 2) : "–"}</td><td class="num quiet">${a.fair_away ? dec(a.fair_away, 2) : "–"}</td></tr>`;
  }).join("");
  const am = k.ah_market;
  const amTxt = am ? `<p class="quiet small">Línea del mercado: ${H} ${am.line > 0 ? "+" : ""}${dec(am.line, 2)} a ${dec(am.odds[0], 2)} / ${A} a ${dec(am.odds[1], 2)}. Cuota justa del modelo: ${am.fair[0] ? dec(am.fair[0], 2) : "–"} / ${am.fair[1] ? dec(am.fair[1], 2) : "–"}${(am.ev[0] > 0.03 || am.ev[1] > 0.03) ? ` · <b>el modelo ve valor en ${am.ev[0] > am.ev[1] ? H : A}</b>` : ""}.</p>` : "";
  const htft = k.ht_ft || {};
  const hf = Object.entries(htft).sort((a, b) => b[1] - a[1]);
  const exact = (k.exact || []).map((e) => `<li><b>${esc(e.score)}</b><span>${pc1(e.p)} %</span></li>`).join("");
  const q = k.quinigol;
  const qg = q ? `<div class="mk"><h4>Quinigol: goles de cada equipo</h4><table class="track mkt"><thead><tr><th scope="col" class="l">Equipo</th>${["0", "1", "2", "M"].map((c) => `<th scope="col" class="num">${c}</th>`).join("")}</tr></thead><tbody>
      <tr><th scope="row" class="l">${H}</th>${q.home.map((v) => `<td class="num">${pc1(v)} %</td>`).join("")}</tr>
      <tr><th scope="row" class="l">${A}</th>${q.away.map((v) => `<td class="num">${pc1(v)} %</td>`).join("")}</tr></tbody></table></div>` : "";
  return `<div class="mk-grid">
    ${tbl("Doble oportunidad y empate no válido", row(`1X (${H} o empate)`, dc["1X"]) + row(`X2 (empate o ${A})`, dc.X2) + row("12 (no hay empate)", dc["12"]) + row(`${H} (empate devuelve)`, k.draw_no_bet["1"]) + row(`${A} (empate devuelve)`, k.draw_no_bet["2"]))}
    ${tbl("Goles", ["0.5", "1.5", "2.5", "3.5", "4.5"].map((x) => row(`Más de ${x.replace(".", ",")}`, tt[x], pt(x))).join("") + row("Marcan los dos", k.btts, (pmore.btts || {}).p) + row(`${H} marca`, k.team_totals.home["0.5"]) + row(`${A} marca`, k.team_totals.away["0.5"]) + row(`${H} deja la portería a cero`, k.clean_sheet.home))}
    <div class="mk"><h4>Hándicap asiático</h4><table class="track mkt"><thead><tr><th scope="col" class="l">Línea</th><th scope="col" class="num">Gana</th><th scope="col" class="num">${H}</th><th scope="col" class="num">${A}</th></tr></thead><tbody>${ahRows}</tbody></table>${amTxt}</div>
    ${tbl("Descanso", row(`${H} gana al descanso`, ht["1"], pht[0]) + row("Empate al descanso", ht.X, pht[1]) + row(`${A} gana al descanso`, ht["2"], pht[2]) + row("Gol en la primera parte", ht["over_0.5"], ((pmore.ht_totals || {})["0.5"] || {}).p))}
    <div class="mk"><h4>Descanso / final</h4><ul class="htft">${hf.map(([lab, p]) => `<li><b>${lab}</b><span class="pbar"><i style="width:${(100 * p / (hf[0][1] || 1)).toFixed(0)}%"></i></span><span class="num">${pc1(p)} %</span></li>`).join("")}</ul></div>
    <div class="mk"><h4>Marcador exacto</h4><ul class="exact">${exact}</ul></div>
    ${qg}
  </div>
  ${hasPm ? `<p class="quiet small">Polymarket: solo contratos con al menos 2.000 $ negociados; «–» si no hay mercado o es demasiado fino.${(pmore.corners || {})["9.5"] ? ` Córners: más de 9,5 a ${pc1(pmore.corners["9.5"].p)} % en Polymarket.` : ""}</p>` : ""}
  <p class="quiet small">Calculado con la matriz de marcadores del modelo, ajustada a la probabilidad final del partido. Cuota justa = 1 / probabilidad (sin margen de casa): si la casa paga más, el modelo ve valor. Descanso: el ${pc1(0.445)} % de los goles de LaLiga llega antes del descanso.</p>`;
}

/* ---------- Puntos necesarios ---------- */
function thresholdsHtml() {
  const th = ((DATA.extra || {}).projection || {}).thresholds;
  if (!th) return "";
  const rows = [["Ganar la liga", th.title], ["Entrar en Champions (4.º)", th.ucl], ["Entrar en Europa (7.º)", th.europe], ["Salvarse (17.º)", th.safe]];
  return `<dl class="facts facts--4 thr">${rows.map(([l, v]) => `<div><dt>${l}</dt><dd>${v.median} <small>pts</small></dd><dd class="quiet small">9 de cada 10 veces basta con ${v.p90}</dd></div>`).join("")}</dl>`;
}

/* ---------- Simulador «¿y si…?» ---------- */
const SIM = { picks: {}, n: 4000, res: null };
function poissonDraw(l) { let k = 0, p = Math.exp(-l), s = p; const u = Math.random(); while (u > s && k < 15) { k++; p *= l / k; s += p; } return k; }
function runSim() {
  const ex = DATA.extra || {};
  const pr = ex.projection || {};
  const fx = pr.fixtures || [];
  const tab = ex.table || [];
  const teams = tab.map((t) => t.team);
  const idx = Object.fromEntries(teams.map((t, i) => [t, i]));
  const k = teams.length, N = SIM.n;
  const sg = pr.sigma || 0;
  const cnt = { title: new Float64Array(k), ucl: new Float64Array(k), europe: new Float64Array(k), releg: new Float64Array(k), pts: new Float64Array(k) };
  const gauss = () => { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  for (let s = 0; s < N; s++) {
    const pts = tab.map((t) => t.pts), gd = tab.map((t) => t.dg), gf = tab.map((t) => t.gf);
    const att = teams.map(() => sg * gauss()), dfn = teams.map(() => sg * gauss());
    for (const f of fx) {
      const h = idx[f.home], a = idx[f.away];
      if (h == null || a == null) continue;
      const key = `${f.home}|${f.away}`;
      let hg, ag;
      const pick = SIM.picks[key];
      if (pick) { [hg, ag] = pick === "1" ? [1, 0] : pick === "X" ? [0, 0] : [0, 1]; }
      else { hg = poissonDraw(f.lam * Math.exp(att[h] + dfn[a])); ag = poissonDraw(f.mu * Math.exp(att[a] + dfn[h])); }
      if (hg > ag) pts[h] += 3; else if (hg < ag) pts[a] += 3; else { pts[h]++; pts[a]++; }
      gd[h] += hg - ag; gd[a] += ag - hg; gf[h] += hg; gf[a] += ag;
    }
    const order = teams.map((_, i) => i).sort((x, y) => (pts[y] - pts[x]) || (gd[y] - gd[x]) || (gf[y] - gf[x]) || (Math.random() - 0.5));
    order.forEach((ti, pos) => {
      if (pos === 0) cnt.title[ti]++;
      if (pos < 4) cnt.ucl[ti]++;
      if (pos < 7) cnt.europe[ti]++;
      if (pos >= k - 3) cnt.releg[ti]++;
      cnt.pts[ti] += pts[ti];
    });
  }
  SIM.res = teams.map((t, i) => ({ team: t, title: cnt.title[i] / N, ucl: cnt.ucl[i] / N, europe: cnt.europe[i] / N, releg: cnt.releg[i] / N, pts: cnt.pts[i] / N }))
    .sort((a, b) => b.pts - a.pts);
}
function simulatorHtml() {
  const pr = (DATA.extra || {}).projection || {};
  const fx = (pr.fixtures || []).slice().sort((a, b) => a.date.localeCompare(b.date));
  if (!fx.length) return '<p class="quiet">No quedan partidos por simular.</p>';
  const base = Object.fromEntries((pr.teams || []).map((t) => [t.team, t]));
  if (!SIM.res) runSim();
  const nPicks = Object.keys(SIM.picks).length;
  const next = fx.slice(0, 30);
  const sel = (f) => { const key = `${f.home}|${f.away}`; const v = SIM.picks[key] || ""; return `<div class="simrow" data-key="${esc(key)}"><span class="simrow__m">${esc(name(f.home))} – ${esc(name(f.away))} <span class="quiet small">${new Date(f.date + "T12:00:00").toLocaleDateString("es-ES", { day: "numeric", month: "short" })}</span></span><span class="seg" role="radiogroup" aria-label="Resultado de ${esc(name(f.home))} – ${esc(name(f.away))}">${["", "1", "X", "2"].map((o) => `<button type="button" role="radio" aria-checked="${v === o}" data-o="${o}">${o || "Sim."}</button>`).join("")}</span></div>`; };
  const d = (now, was) => { const x = now - (was || 0); return Math.abs(x) < 0.005 ? "" : `<small class="${x > 0 ? "up" : "down"}">${x > 0 ? "+" : "−"}${Math.round(Math.abs(100 * x))}</small>`; };
  const res = SIM.res.map((r) => { const b = base[r.team] || {}; return `<tr><th scope="row" class="l">${esc(name(r.team))}</th><td class="num">${Math.round(r.pts)}</td><td class="num">${pc1(r.title)} % ${d(r.title, b.p_title)}</td><td class="num">${pc1(r.ucl)} % ${d(r.ucl, b.p_ucl)}</td><td class="num">${pc1(r.releg)} % ${d(r.releg, b.p_releg)}</td></tr>`; }).join("");
  return `<div class="sim">
    <div class="sim__picks"><p class="quiet small">Elige el resultado de los partidos que quieras (los demás se simulan con el modelo). Próximos ${next.length} de ${fx.length}.</p>${next.map(sel).join("")}
      ${nPicks ? `<button type="button" class="more" id="sim-reset">Quitar mis ${nPicks} resultados</button>` : ""}</div>
    <div class="sim__res"><div class="tbl-wrap"><table class="track"><thead><tr><th scope="col" class="l">Equipo</th><th scope="col" class="num">Pts</th><th scope="col" class="num">Liga</th><th scope="col" class="num">Champions</th><th scope="col" class="num">Descenso</th></tr></thead><tbody>${res}</tbody></table></div>
    <p class="quiet small">${SIM.n.toLocaleString("es-ES")} temporadas simuladas en tu navegador. En verde o rojo, cuánto cambia respecto a la proyección sin tus resultados.</p></div>
  </div>`;
}
function wireSimulator(root) {
  root.querySelectorAll(".simrow .seg button").forEach((b) => b.addEventListener("click", () => {
    const key = b.closest(".simrow").dataset.key;
    if (b.dataset.o) SIM.picks[key] = b.dataset.o; else delete SIM.picks[key];
    runSim();
    root.innerHTML = simulatorHtml();
    wireSimulator(root);
  }));
  const r = root.querySelector("#sim-reset");
  if (r) r.addEventListener("click", () => { SIM.picks = {}; runSim(); root.innerHTML = simulatorHtml(); wireSimulator(root); });
}

/* ---------- Suerte ---------- */
function luckHtml() {
  const L = (DATA.extra || {}).luck || [];
  if (!L.length) return '<p class="quiet">Aparecerá tras la próxima actualización.</p>';
  const max = Math.max(...L.map((r) => Math.abs(r.diff)), 1);
  return `<div class="tbl-wrap"><table class="track luck"><thead><tr><th scope="col" class="l">Equipo</th><th scope="col" class="num">Puntos</th><th scope="col" class="num">Merecidos</th><th scope="col" class="l">Diferencia</th><th scope="col" class="num">Goles / xG</th><th scope="col" class="num">Encajados / xG</th></tr></thead><tbody>
    ${L.map((r) => `<tr><th scope="row" class="l">${esc(name(r.team))}</th><td class="num">${r.pts}</td><td class="num">${dec(r.xpts, 1)}</td>
      <td class="l"><span class="dbar"><i class="${r.diff >= 0 ? "pos" : "neg"}" style="--w:${(50 * Math.abs(r.diff) / max).toFixed(1)}%"></i></span> <span class="num ${Math.abs(r.diff) >= 3 ? (r.diff > 0 ? "up" : "down") : ""}">${r.diff > 0 ? "+" : r.diff < 0 ? "−" : ""}${dec(Math.abs(r.diff), 1)}</span></td>
      <td class="num">${r.gf} <span class="quiet">/ ${dec(r.xg, 1)}</span></td><td class="num">${r.ga} <span class="quiet">/ ${dec(r.xga, 1)}</span></td></tr>`).join("")}</tbody></table></div>
    <p class="quiet small">«Merecidos»: puntos esperados según las ocasiones de cada partido (xPts de Understat). Quien suma bastante más de lo que merece suele bajar después, y al revés: es de lo que mejor anticipa los cambios de rendimiento.</p>`;
}

/* ---------- Premios ---------- */
function awardsHtml() {
  const a = (DATA.extra || {}).awards || {};
  const block = (title, rows, cols, note) => rows && rows.length ? `<div class="pm-card"><h4>${title}</h4><div class="tbl-wrap"><table class="track"><thead><tr>${cols.map((c, i) => `<th scope="col" class="${i ? "num" : "l"}">${c}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table></div>${note ? `<p class="quiet small">${note}</p>` : ""}</div>` : "";
  return `<div class="pm-grid">
    ${block("Máximo asistente", (a.assists || []).map((r) => `<tr><th scope="row" class="l">${esc(r.name)} <span class="quiet small">${esc(name(r.team))}</span></th><td class="num">${r.assists}</td><td class="num">${dec(r.exp, 0)}</td><td class="num">${pc1(r.p)} %</td></tr>`), ["Jugador", "Ahora", "Al final", "Prob."], "Asistencias que le quedan = goles esperados de su equipo × su parte del xA.")}
    ${block("Zamora (aprox.)", (a.zamora || []).map((r) => `<tr><th scope="row" class="l">${esc(r.name)} <span class="quiet small">${esc(name(r.team))}</span></th><td class="num">${dec(r.ratio, 2)}</td><td class="num">${pc1(r.p)} %</td></tr>`), ["Portero", "Goles/partido", "Prob."], "Portero titular de cada equipo con los goles que encajaría su equipo; no cuenta rotaciones ni lesiones.")}
    ${block("Equipo más goleador", (a.most_goals || []).map((r) => `<tr><th scope="row" class="l">${esc(name(r.team))}</th><td class="num">${dec(r.exp, 0)}</td><td class="num">${pc1(r.p)} %</td></tr>`), ["Equipo", "Goles al final", "Prob."])}
    ${block("Equipo menos goleado", (a.least_conceded || []).map((r) => `<tr><th scope="row" class="l">${esc(name(r.team))}</th><td class="num">${dec(r.exp, 0)}</td><td class="num">${pc1(r.p)} %</td></tr>`), ["Equipo", "Encajados al final", "Prob."])}
  </div>`;
}

/* ---------- Estilo y Elo histórico (ficha de equipo) ---------- */
const STYLE_SHORT = { ppda: "Presión", oppda: "Le dejan el balón", deep: "Llegadas al área", deep_allowed: "Llegadas que concede", npxg: "Ocasiones (xG)", npxga: "Ocasiones que concede" };
function styleHtml(team) {
  const st = (DATA.extra || {}).styles || {};
  const t = (st.teams || {})[team];
  if (!t) return "";
  // en presión, menos PPDA = más presión: se invierte el percentil para que la barra larga sea «más»
  const bars = Object.keys(STYLE_SHORT).map((f) => {
    let p = t.pct[f];
    if (f === "ppda") p = 1 - p;
    return `<div class="sty"><span class="sty__l" title="${esc(st.features[f])}">${STYLE_SHORT[f]}</span><span class="pbar"><i style="width:${(100 * p).toFixed(0)}%"></i></span><span class="num quiet small">${dec(t.values[f], f.startsWith("npx") ? 2 : 1)}</span></div>`;
  }).join("");
  const mu = ((DATA.extra || {}).style_matchups || []).filter((m) => m.style === t.style && m.n >= 20).sort((a, b) => a.ppg - b.ppg);
  const worst = mu[0], best = mu[mu.length - 1];
  return `<h4>Estilo de juego: ${esc(t.style)}</h4><div class="stys">${bars}</div>
    <p class="quiet small">Barras: posición frente al resto de la liga esta temporada (más larga = más).${worst && best && worst !== best ? ` En las tres últimas temporadas, los equipos de este estilo sacaron ${dec(best.ppg, 2)} puntos por partido contra «${esc(best.vs)}» y ${dec(worst.ppg, 2)} contra «${esc(worst.vs)}».` : ""}</p>`;
}
function eloSeasonsHtml(team) {
  const eh = ((DATA.extra || {}).elo_history || {}).seasons || {};
  const s = eh[team];
  if (!s || s.length < 2) return "";
  const best = ((DATA.extra || {}).elo_history || {}).best[team];
  const pts = s.map(([y, e]) => [`${y}-07-01`, e]);
  return `<h4>Fuerza (Elo) por temporada desde ${s[0][0]}</h4>${lineChart(pts, 520, 150, "var(--slip)")}
    <p class="quiet small">Su mejor temporada por Elo: ${best[0]}-${String(best[0] + 1).slice(2)} (${Math.round(best[1])}).</p>`;
}

/* ---------- Jugadores ---------- */
const PL = { team: null, q: "", sel: null };
function shotMap(shots) {
  if (!shots || !shots.length) return '<p class="quiet small">Sin tiros registrados esta temporada.</p>';
  // Understat: x de 0 (portería propia) a 1 (portería rival), y de 0 a 1. Se dibuja medio campo de ataque.
  const W = 300, H = 210;
  const X = (y) => 10 + y * (W - 20), Y = (x) => 10 + (1 - x) * 2 * (H - 20);
  const dots = shots.filter((s) => s[0] >= 0.5).map((s) => {
    const r = 3 + Math.sqrt(s[2] || 0) * 12;
    return `<circle cx="${X(s[1]).toFixed(1)}" cy="${Y(s[0]).toFixed(1)}" r="${r.toFixed(1)}" class="${s[3] === "G" ? "goal" : "miss"}"><title>xG ${dec(s[2] || 0, 2)}${s[3] === "G" ? " · gol" : ""}</title></circle>`;
  }).join("");
  return `<svg class="shotmap" viewBox="0 0 ${W} ${H}" role="img" aria-label="Mapa de tiros">
    <rect x="10" y="10" width="${W - 20}" height="${H - 20}" class="pitch"/>
    <rect x="${X(0.21)}" y="10" width="${X(0.79) - X(0.21)}" height="${Y(0.83) - 10}" class="pitch"/>
    <rect x="${X(0.37)}" y="10" width="${X(0.63) - X(0.37)}" height="${Y(0.945) - 10}" class="pitch"/>
    <line x1="${X(0.45)}" x2="${X(0.55)}" y1="10" y2="10" class="goalline"/>
    ${dots}</svg>
    <p class="quiet small">Cada círculo es un tiro; cuanto más grande, más xG. Rellenos: goles.</p>`;
}
function playerCard(team, p) {
  const f = (v, n = 2) => (v == null ? "–" : dec(v, n));
  return `<div class="pcard"><h4>${esc(p.name)} <span class="quiet small">${esc(name(team))} · ${esc(p.pos)}</span></h4>
    <dl class="facts facts--4">
      <div><dt>Goles</dt><dd>${p.goals} <small class="quiet">(${f(p.xg, 1)} xG)</small></dd></div>
      <div><dt>Asistencias</dt><dd>${p.assists} <small class="quiet">(${f(p.xa, 1)} xA)</small></dd></div>
      <div><dt>Minutos</dt><dd>${p.minutes.toLocaleString("es-ES")} <small class="quiet">${p.games} PJ</small></dd></div>
      <div><dt>Tarjetas</dt><dd>${p.yellow} <small class="quiet">amarillas</small>${p.red ? ` · ${p.red} <small class="quiet">rojas</small>` : ""}</dd></div>
    </dl>
    <p class="quiet small">Por 90 minutos: xG ${f(p.xg90)}, xA ${f(p.xa90)}, participación en jugadas de gol ${f(p.chain90)}. Tiros ${p.shots}, pases clave ${p.key_passes}.${p.pen_taken ? ` Penaltis: ${p.pen_scored} de ${p.pen_taken} (esta temporada y la anterior).` : p.pens ? ` ${p.pens} goles de penalti.` : ""}</p>
    ${shotMap(p.shots_map)}</div>`;
}
function playersHtml() {
  const P = (DATA.extra || {}).players || {};
  const teams = Object.keys(P).sort((a, b) => name(a).localeCompare(name(b), "es"));
  if (!teams.length) return '<p class="quiet">Aparecerán tras la próxima actualización.</p>';
  if (!PL.team || !P[PL.team]) PL.team = teams[0];
  const q = PL.q.trim().toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "");
  const all = q ? teams.flatMap((t) => P[t].map((p) => [t, p])).filter(([, p]) => p.name.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "").includes(q))
    : P[PL.team].map((p) => [PL.team, p]);
  const sel = all.find(([t, p]) => `${t}|${p.name}` === PL.sel) || all[0];
  const rows = all.slice(0, 40).map(([t, p]) => `<tr class="${sel && sel[1] === p ? "is-sel" : ""}" data-p="${esc(`${t}|${p.name}`)}" tabindex="0"><th scope="row" class="l">${esc(p.name)}${q ? ` <span class="quiet small">${esc(name(t))}</span>` : ""}</th><td class="num">${p.goals}</td><td class="num">${dec(p.xg, 1)}</td><td class="num">${p.assists}</td><td class="num">${dec(p.xa, 1)}</td><td class="num">${p.minutes.toLocaleString("es-ES")}</td></tr>`).join("");
  return `<div class="pl-controls"><label class="qsel">Equipo <select id="pl-team">${teams.map((t) => `<option value="${esc(t)}" ${t === PL.team ? "selected" : ""}>${esc(name(t))}</option>`).join("")}</select></label>
      <label class="search"><span class="sr">Buscar jugador</span><input id="pl-q" type="search" placeholder="Buscar jugador…" value="${esc(PL.q)}" autocomplete="off"></label></div>
    <div class="pl-grid"><div class="tbl-wrap"><table class="track pl"><thead><tr><th scope="col" class="l">Jugador</th><th scope="col" class="num">G</th><th scope="col" class="num">xG</th><th scope="col" class="num">A</th><th scope="col" class="num">xA</th><th scope="col" class="num">Min.</th></tr></thead><tbody>${rows || `<tr><td colspan="6" class="quiet">Ningún jugador coincide con «${esc(PL.q)}».</td></tr>`}</tbody></table></div>
    <div>${sel ? playerCard(sel[0], sel[1]) : ""}</div></div>
    <p class="quiet small">Datos de Understat (goles esperados de cada tiro y pase). Los jugadores con más minutos de cada equipo.</p>`;
}
function wirePlayers(root) {
  const t = root.querySelector("#pl-team");
  if (t) t.addEventListener("change", () => { PL.team = t.value; PL.sel = null; PL.q = ""; root.innerHTML = playersHtml(); wirePlayers(root); });
  const q = root.querySelector("#pl-q");
  if (q) q.addEventListener("input", () => { PL.q = q.value; const pos = q.selectionStart; root.innerHTML = playersHtml(); wirePlayers(root); const n = root.querySelector("#pl-q"); n.focus(); n.setSelectionRange(pos, pos); });
  root.querySelectorAll("tr[data-p]").forEach((tr) => {
    const go = () => { PL.sel = tr.dataset.p; root.innerHTML = playersHtml(); wirePlayers(root); };
    tr.addEventListener("click", go);
    tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
  });
}

/* ---------- Árbitros ---------- */
function refereesHtml() {
  const R = (DATA.extra || {}).referees || [];
  if (R.length < 2) return '<p class="quiet">Aparecerán tras la próxima actualización.</p>';
  const lg = R[0].league;
  const rows = R.slice(1);
  const upcoming = Object.fromEntries((DATA.matches || []).filter((m) => m.referee).map((m) => [m.referee.split(" ").slice(-1)[0].toLowerCase(), m]));
  return `<div class="tbl-wrap"><table class="track"><thead><tr><th scope="col" class="l">Árbitro</th><th scope="col" class="num">Partidos</th><th scope="col" class="num">Amarillas</th><th scope="col" class="num">Rojas</th><th scope="col" class="num">Faltas</th><th scope="col" class="num">Goles</th><th scope="col" class="num">Gana el local</th></tr></thead><tbody>
    ${rows.map((r) => { const nx = Object.entries(upcoming).find(([k]) => r.key && r.key.includes(k)); const cls = (v, l) => (v > l * 1.15 ? "up-ink" : v < l * 0.85 ? "down-ink" : ""); return `<tr><th scope="row" class="l">${esc(r.referee)}${nx ? ` <span class="pill">${esc(name(nx[1].home))} – ${esc(name(nx[1].away))}</span>` : ""}</th><td class="num">${r.n} <span class="quiet small">(${r.n_season} este año)</span></td><td class="num ${cls(r.yellow, lg.yellow)}">${dec(r.yellow, 1)}</td><td class="num">${dec(r.red, 2)}</td><td class="num">${dec(r.fouls, 0)}</td><td class="num">${dec(r.goals, 1)}</td><td class="num">${pc1(r.home_win)} %</td></tr>`; }).join("")}
    </tbody></table></div>
    <p class="quiet small">Por partido, en esta temporada y la anterior. Media de la liga: ${dec(lg.yellow, 1)} amarillas, ${dec(lg.red, 2)} rojas, ${dec(lg.fouls, 0)} faltas y ${dec(lg.goals, 1)} goles. En negrita, más de un 15 % por encima de la media en amarillas. La etiqueta marca a quién le toca un partido de la próxima jornada.</p>`;
}

/* ---------- Otras ligas ---------- */
const LG = { code: null };
const LG_ORDER = ["SP2", "LF", "E0", "I1", "D1", "F1"];
function leaguesHtml() {
  const L = (DATA.leagues || {}).leagues || {};
  const codes = LG_ORDER.filter((c) => L[c]);
  if (!codes.length) return '<p class="quiet">Las otras competiciones aparecerán tras la próxima actualización.</p>';
  if (!LG.code || !L[LG.code]) LG.code = codes[0];
  const lg = L[LG.code];
  const zones = Object.keys(lg.zones || {});
  const proj = Object.fromEntries((lg.projection || []).map((p) => [p.team, p]));
  const zcls = (pos) => { for (const z of zones) { const [a, b] = lg.zones[z]; if (pos >= a && pos <= b && z !== "campeón") return z === "descenso" ? "z-rel" : "z-ucl"; } return ""; };
  const table = `<div class="tbl-wrap"><table class="standings"><thead><tr><th scope="col">#</th><th scope="col" class="l">Equipo</th><th scope="col">PJ</th><th scope="col">DG</th><th scope="col">Pts</th><th scope="col" class="sep">Pts al final</th>${zones.map((z) => `<th scope="col">${z.charAt(0).toUpperCase() + z.slice(1)}</th>`).join("")}</tr></thead><tbody>
    ${(lg.table || []).map((t) => { const p = proj[t.team] || {}; return `<tr class="${zcls(t.pos)}"><td class="pos">${t.pos}</td><th scope="row" class="l">${esc(name(t.team))}</th><td>${t.pj}</td><td>${t.dg > 0 ? "+" : ""}${t.dg}</td><td class="pts">${t.pts}</td><td class="sep">${p.exp_pts != null ? Math.round(p.exp_pts) : "–"}</td>${zones.map((z) => pctCell(p[`p_${z}`])).join("")}</tr>`; }).join("")}</tbody></table></div>`;
  const up = (lg.upcoming || []).filter((m) => !m.started).map((m) => `<li class="lg-m"><span class="lg-m__w">${m.kickoff_utc ? fmtTs(m.kickoff_utc) : esc(m.date)}</span><span class="lg-m__t">${esc(name(m.home))} – ${esc(name(m.away))}</span><span class="lg-m__p">${probBar(m.p)}</span></li>`).join("");
  const ev = lg.eval;
  const tr = lg.track || {};
  return `<div class="lg-tabs" role="tablist" aria-label="Competición">${codes.map((c) => `<button type="button" role="tab" aria-selected="${c === LG.code}" data-lg="${c}">${esc(L[c].name)}</button>`).join("")}</div>
    <div class="lg-grid"><div>${table}<p class="quiet small">${lg.remaining} partidos por jugar, simulados con el modelo (incertidumbre incluida). ${LG.code === "LF" ? "Liga F: con el Elo femenino calculado con los resultados de ESPN." : ""}</p></div>
    <div><h4>Próximos partidos</h4>${up ? `<ul class="lg-list">${up}</ul>` : '<p class="quiet small">No hay partidos con cuotas publicadas en los próximos días.</p>'}
    ${ev ? `<p class="quiet small">Medido en la temporada ${ev.season}-${String(ev.season + 1).slice(2)} (${ev.n} partidos, log-loss, menos es mejor): modelo ${dec(ev.modelo, 3)}, mercado ${dec(ev.mercado, 3)}. La probabilidad mezcla ${Math.round(100 * (1 - Math.max(0.1, ev.w_model)))} % mercado y ${Math.round(100 * Math.max(0.1, ev.w_model))} % modelo.</p>` : ""}
    ${tr.n ? `<p class="quiet small">Desde que se sigue: ${tr.n} partidos, acierta el ${pc1(tr.accuracy)} %, log-loss ${dec(tr.log_loss, 3)}.</p>` : ""}</div></div>`;
}
function probBar(p) {
  if (!p) return '<span class="quiet small">sin datos</span>';
  return `<span class="pb3" title="1 ${pc1(p[0])} % · X ${pc1(p[1])} % · 2 ${pc1(p[2])} %"><i style="width:${(100 * p[0]).toFixed(1)}%">${pc1(p[0])}</i><i style="width:${(100 * p[1]).toFixed(1)}%">${pc1(p[1])}</i><i style="width:${(100 * p[2]).toFixed(1)}%">${pc1(p[2])}</i></span>`;
}
function wireLeagues(root) {
  root.querySelectorAll("[data-lg]").forEach((b) => b.addEventListener("click", () => { LG.code = b.dataset.lg; root.innerHTML = leaguesHtml(); wireLeagues(root); }));
}
function cupsHtml() {
  const C = (DATA.leagues || {}).cups || {};
  const blocks = Object.entries(C).filter(([, v]) => v.matches && v.matches.length).map(([lg, v]) => `<div class="pm-card"><h4>${esc(v.name)}</h4><ul class="lg-list">${v.matches.map((m) => `<li class="lg-m"><span class="lg-m__w">${m.state === "post" ? "Final" : m.state === "in" ? '<span class="live-dot"></span> En juego' : m.kickoff_utc ? fmtTs(m.kickoff_utc) : esc(m.date)}</span><span class="lg-m__t">${esc(m.home)} – ${esc(m.away)}${m.score ? ` <b>${m.score[0]}-${m.score[1]}</b>` : ""}</span><span class="lg-m__p">${probBar(m.p)}</span></li>`).join("")}</ul></div>`).join("");
  return blocks ? `<div class="pm-grid">${blocks}</div><p class="quiet small">Copa del Rey, Liga F y los partidos de equipos españoles en Champions, Europa League y Conference de los próximos días. Probabilidad con el Elo europeo (ligas de 25 países y cruces europeos); sin factor campo en finales a partido único.</p>` : '<p class="quiet">No hay partidos de copa ni de Europa en los próximos días.</p>';
}

/* ---------- Quinigol ---------- */
function quinigolHtml() {
  const Q = DATA.quinigol || {};
  const js = Object.values(Q.jornadas || {}).sort((a, b) => (b.temporada - a.temporada) || (b.jornada - a.jornada));
  if (js.length && js[0].rows) return quinigolOfficial(js[0], js, Q.summary);
  return quinigolFromMatches();
}
function quinigolOfficial(j, all, sm) {
  const C = ["0", "1", "2", "M"];
  const cells = (arr, pub, pick, res) => C.map((c, k) => `<td class="num qgc${pick === c ? " is-pick" : ""}${res === c ? " is-real" : ""}">${pc1(arr[k])}${pub ? `<small class="quiet"> ${Math.round(100 * pub[k])}</small>` : ""}</td>`).join("");
  const rows = j.rows.map((r, i) => {
    const t = j.ticket[i] || "";
    const res = r.result || "";
    const st = r.live ? '<span class="live-dot"></span>' : res ? `<b class="qres ${res === t ? "ok" : "ko"}">${r.hg}-${r.ag}</b>` : `<span class="quiet small">${esc(r.when || "")}</span>`;
    return `<tr class="qg-m"><th scope="rowgroup" colspan="6" class="l">${r.n}. ${esc(r.home)} – ${esc(r.away)} <span class="src src--${(SRC_SHORT[r.source] || "x").replace(/[^a-z]/gi, "")}">${esc(SRC_SHORT[r.source] || r.source)}</span> ${st}</th></tr>
      <tr><td class="l small">${esc(r.home)}</td>${cells(r.q.home, r.public && r.public[0], t[0], res[0])}<td class="num" rowspan="2"><b>${t[0] || ""}-${t[1] || ""}</b><br><small class="quiet">${pc1(r.q.p_pick)} %</small></td></tr>
      <tr><td class="l small">${esc(r.away)}</td>${cells(r.q.away, r.public && r.public[1], t[1], res[1])}</tr>`;
  }).join("");
  const hd = j.hits_dist || [];
  const p2 = hd.slice(2).reduce((a, b) => a + b, 0);
  const done = all.filter((x) => x.settled && x.rows);
  return `<p class="qstatus">Jornada ${j.jornada}${j.settled ? ` · terminada: <b>${j.hits} acierto${j.hits === 1 ? "" : "s"}</b>` : j.frozen ? " · en juego" : " · boleto abierto"}${j.late ? " · rellenado con la jornada empezada (no cuenta)" : ""}</p>
    <div class="tbl-wrap"><table class="track qg"><thead><tr><th scope="col" class="l">Equipo</th>${C.map((c) => `<th scope="col" class="num">${c}</th>`).join("")}<th scope="col" class="num">Boleto</th></tr></thead><tbody>${rows}</tbody></table></div>
    <p class="quiet small">Probabilidad (%) de que cada equipo marque 0, 1, 2 o más goles (M); en pequeño, lo que juega la gente. Subrayado: lo que marca el boleto (el resultado conjunto más probable de cada partido); en rojo, lo que pasó. Aciertos esperados ${dec(j.expected_hits, 2)}; probabilidad de cobrar (2 o más) ${pc1(p2)} %.${sm && sm.jornadas ? ` Historial: ${sm.jornadas} jornadas, ${sm.hits} aciertos (esperados ${dec(sm.expected, 1)}), premio en ${sm.with_prize}.` : ""}</p>`;
}
function quinigolFromMatches() {
  const ms = DATA.matches || [];
  const withQ = ms.filter((m) => m.markets && m.markets.quinigol).slice(0, 6);
  if (!withQ.length) return "";
  const pickOf = (arr) => ["0", "1", "2", "M"][arr.indexOf(Math.max(...arr))];
  return `<div class="tbl-wrap"><table class="track qg"><thead><tr><th scope="col" class="l">Partido</th><th scope="col" class="num">Local 0/1/2/M</th><th scope="col" class="num">Visitante 0/1/2/M</th><th scope="col" class="num">Marcador</th></tr></thead><tbody>
    ${withQ.map((m) => { const q = m.markets.quinigol; let best = [0, 0, 0]; q.joint.forEach((r, a) => r.forEach((v, b) => { if (v > best[2]) best = [a, b, v]; })); return `<tr><th scope="row" class="l">${esc(name(m.home))} – ${esc(name(m.away))}</th><td class="num">${q.home.map(pc1).join(" / ")}</td><td class="num">${q.away.map(pc1).join(" / ")}</td><td class="num"><b>${["0", "1", "2", "M"][best[0]]}-${["0", "1", "2", "M"][best[1]]}</b> <span class="quiet small">${pc1(best[2])} %</span></td></tr>`; }).join("")}
    </tbody></table></div>
    <p class="quiet small">Probabilidad (%) de que cada equipo marque 0, 1, 2 o más goles (M), con la matriz de marcadores del modelo. Son los próximos partidos de LaLiga: el boleto oficial del Quinigol puede incluir otros.</p>`;
}

/* ---------- Lo que se ha medido (Aciertos) ---------- */
function measuredHtml() {
  const pc = DATA.projection_calib || {}, lc = DATA.live_calib || {};
  const L = (DATA.leagues || {}).leagues || {};
  const parts = [];
  if (pc.table) {
    const base = pc.table.find((r) => r.sigma === 0), best = pc.table.find((r) => r.sigma === pc.sigma);
    parts.push(`<div class="pm-card"><h4>Proyección de la temporada</h4><p>Simular cada temporada con una incertidumbre del ${Math.round(100 * pc.sigma)} % en la fuerza de cada equipo acierta mejor el puesto final que darla por exacta (${pc.table[0].n} proyecciones de ${pc.seasons[0]} a ${pc.seasons[pc.seasons.length - 1]}, desde las jornadas ${pc.rounds.join(", ")}): log-verosimilitud ${dec(best.loglik, 3)} frente a ${dec(base.loglik, 3)}. El efecto es pequeño.</p></div>`);
  }
  if (lc.tiros_a_puerta) {
    const t = lc.tiros_a_puerta, x = lc.xg;
    parts.push(`<div class="pm-card"><h4>Probabilidad en directo</h4><p>Con ${lc.matches.toLocaleString("es-ES")} partidos (tiros de Understat minuto a minuto): saber los tiros a puerta mejora la predicción de los goles que quedan solo de ${dec(t.nll_solo_marcador, 4)} a ${dec(t.nll_con_dato, 4)}; con xG, a ${dec(x.nll_con_dato, 4)}. Lo que más pesa es el marcador, el minuto y la fuerza previa; un equipo que tira mucho a puerta solo marca algo más de lo previsto.</p></div>`);
  }
  const g = (DATA.polymarket_eval || {}).goals || {};
  const gl = (k, lab) => { const r = g[k]; if (!r || !r.polymarket) return ""; return `<tr><th scope="row" class="l">${lab}</th><td class="num">${dec(r.polymarket, 3)}</td><td class="num">${r.modelo != null ? dec(r.modelo, 3) : "–"}</td><td class="num">${r.cierre_casas != null ? dec(r.cierre_casas, 3) : "–"}</td><td class="num">${r.n}</td></tr>`; };
  const gr = gl("over25", "Más de 2,5 goles") + gl("btts", "Marcan los dos");
  if (gr) parts.push(`<div class="pm-card"><h4>Polymarket en goles</h4><div class="tbl-wrap"><table class="track"><thead><tr><th scope="col" class="l">Mercado</th><th scope="col" class="num">Polymarket</th><th scope="col" class="num">Modelo</th><th scope="col" class="num">Cierre casas</th><th scope="col" class="num">Partidos</th></tr></thead><tbody>${gr}</tbody></table></div><p class="quiet small">Log-loss (menos es mejor) en partidos con al menos 5.000 $ negociados en ese contrato.</p></div>`);
  const ev = Object.values(L).filter((l) => l.eval);
  if (ev.length) {
    parts.push(`<div class="pm-card"><h4>Otras ligas</h4><div class="tbl-wrap"><table class="track"><thead><tr><th scope="col" class="l">Liga</th><th scope="col" class="num">Modelo</th><th scope="col" class="num">Mercado</th><th scope="col" class="num">Partidos</th></tr></thead><tbody>${ev.map((l) => `<tr><th scope="row" class="l">${esc(l.name)}</th><td class="num">${dec(l.eval.modelo, 3)}</td><td class="num">${dec(l.eval.mercado, 3)}</td><td class="num">${l.eval.n}</td></tr>`).join("")}</tbody></table></div><p class="quiet small">Log-loss en la última temporada completa (menos es mejor). En todas, el mercado acierta más que un Dixon-Coles propio.</p></div>`);
  }
  return parts.length ? `<div class="pm-grid">${parts.join("")}</div>` : '<p class="quiet">Aparecerá tras el próximo reentrenamiento semanal.</p>';
}

window.EXTRA = { measuredHtml, previewHtml, marketsHtml, thresholdsHtml, simulatorHtml, wireSimulator, luckHtml, awardsHtml, styleHtml,
  eloSeasonsHtml, playersHtml, wirePlayers, refereesHtml, leaguesHtml, wireLeagues, cupsHtml, quinigolHtml };

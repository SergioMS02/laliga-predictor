"use strict";

const NAMES = {
  "Ath Madrid": "Atlético", "Ath Bilbao": "Athletic", "Sociedad": "Real Sociedad", "Vallecano": "Rayo Vallecano",
  "Espanol": "Espanyol", "La Coruna": "Deportivo", "Santander": "Racing", "Betis": "Real Betis", "Alaves": "Alavés",
  "Malaga": "Málaga", "Leganes": "Leganés", "Cadiz": "Cádiz", "Almeria": "Almería", "Sp Gijon": "Sporting",
  "Celta": "Celta", "Real Madrid": "Real Madrid", "Barcelona": "Barcelona", "Las Palmas": "Las Palmas",
  "Valladolid": "Valladolid", "Oviedo": "Oviedo", "Mirandes": "Mirandés", "Castellon": "Castellón", "Cordoba": "Córdoba",
};
const name = (t) => NAMES[t] || t;
const OUT = ["1", "X", "2"];
const pct = (x) => `${Math.round(100 * x)}`;
const dec = (x, n = 3) => Number(x).toLocaleString("es-ES", { minimumFractionDigits: n, maximumFractionDigits: n });
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const tz = "Europe/Madrid";
const fmtWhen = (iso, date) => {
  if (iso) {
    const d = new Date(iso);
    return d.toLocaleString("es-ES", { timeZone: tz, weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
  }
  return new Date(date + "T12:00:00").toLocaleDateString("es-ES", { weekday: "short", day: "numeric", month: "short" }) + ", hora por confirmar";
};
const fmtTs = (iso) => new Date(iso).toLocaleString("es-ES", { timeZone: tz, day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
const prettify = (s) => Object.keys(NAMES).reduce((acc, k) => acc.split(k).join(NAMES[k]), String(s || ""));

/* ---------- Trazo de lápiz alrededor del resultado más probable ---------- */
function pencilPath(seed) {
  // elipse ligeramente irregular que se pasa de largo, como hecha a mano
  let r = seed * 9301 + 49297;
  const rnd = () => ((r = (r * 9301 + 49297) % 233280) / 233280);
  const cx = 50, cy = 50, rx = 44 + rnd() * 3, ry = 40 + rnd() * 4;
  const start = -Math.PI * (0.62 + rnd() * 0.1);
  const pts = [];
  const turns = 2 * Math.PI * 1.12;
  for (let i = 0; i <= 40; i++) {
    const a = start + (turns * i) / 40;
    const wob = 1 + (rnd() - 0.5) * 0.05;
    pts.push([cx + Math.cos(a) * rx * wob, cy + Math.sin(a) * ry * wob]);
  }
  return "M" + pts.map((p) => p.map((v) => v.toFixed(1)).join(" ")).join(" L");
}

/* ---------- Mini gráfico de tendencia ---------- */
function sparkline(hist, idx) {
  const vals = hist.map((h) => h.p[idx]);
  if (vals.length < 2) return '<svg viewBox="0 0 100 26" aria-hidden="true"><line x1="0" y1="13" x2="100" y2="13" stroke="var(--rule)" stroke-dasharray="3 3"/></svg>';
  const lo = Math.min(...vals) - 0.03, hi = Math.max(...vals) + 0.03;
  const x = (i) => (i / (vals.length - 1)) * 100;
  const y = (v) => 24 - ((v - lo) / (hi - lo)) * 22;
  const d = vals.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ");
  const last = vals[vals.length - 1];
  return `<svg viewBox="0 0 100 26" preserveAspectRatio="none" aria-hidden="true">
    <path d="${d}" fill="none" stroke="var(--slip)" stroke-width="1.8" vector-effect="non-scaling-stroke"/>
    <circle cx="100" cy="${y(last).toFixed(1)}" r="2.6" fill="var(--slip)"/></svg>`;
}

function trendText(hist) {
  if (hist.length < 2) return "Primera predicción";
  const a = hist[0].p[0], b = hist[hist.length - 1].p[0];
  const d = Math.round(100 * (b - a));
  if (d === 0) return `Local estable en ${pct(b)} %`;
  const cls = d > 0 ? "up" : "down";
  return `Local ${pct(a)} → ${pct(b)} % <span class="${cls}">(${d > 0 ? "+" : ""}${d})</span>`;
}

/* ---------- Gráfico de evolución (detalle) ---------- */
function evolutionChart(m) {
  const hist = m.history;
  const W = 520, H = 220, L = 34, R = 34, T = 12, B = 30;
  if (hist.length < 2) {
    return `<p class="quiet">Todavía no hay cambios: esta es la primera predicción del partido (${pct(m.p_home)} / ${pct(m.p_draw)} / ${pct(m.p_away)} %). Cada actualización quedará aquí.</p>`;
  }
  const t = hist.map((h) => new Date(h.ts).getTime());
  const t0 = t[0], t1 = t[t.length - 1] === t0 ? t0 + 1 : t[t.length - 1];
  const x = (v) => L + ((v - t0) / (t1 - t0)) * (W - L - R);
  const y = (p) => T + (1 - p) * (H - T - B);
  let g = "";
  for (const p of [0, 0.25, 0.5, 0.75]) g += `<line class="axis" x1="${L}" x2="${W - R}" y1="${y(p)}" y2="${y(p)}"/><text class="lbl" x="${L - 6}" y="${y(p) + 4}" text-anchor="end">${p * 100}%</text>`;
  const cls = ["l1", "lX", "l2"];
  hist.forEach((h, i) => {
    if (h.lineup && (i === 0 || !hist[i - 1].lineup)) {
      const xl = x(t[i]), right = xl > W / 2;
      g += `<line x1="${xl}" x2="${xl}" y1="${T}" y2="${H - B}" stroke="var(--up)" stroke-dasharray="2 3"/><text class="lbl" x="${xl + (right ? -4 : 4)}" y="${T + 10}" text-anchor="${right ? "end" : "start"}">alineaciones</text>`;
    }
  });
  const ends = [];
  for (let k = 0; k < 3; k++) {
    const d = hist.map((h, i) => `${i ? "L" : "M"}${x(t[i]).toFixed(1)} ${y(h.p[k]).toFixed(1)}`).join(" ");
    g += `<path class="line ${cls[k]}" d="${d}"/>`;
    ends.push({ k, y: y(hist[hist.length - 1].p[k]) + 4 });
  }
  ends.sort((a, b) => a.y - b.y);  // separa etiquetas que se pisan
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 13) ends[i].y = ends[i - 1].y + 13;
  for (const e of ends) g += `<text class="end" x="${W - R + 6}" y="${e.y}" fill="${e.k === 0 ? "var(--slip)" : e.k === 1 ? "var(--ink-soft)" : "var(--ink)"}">${OUT[e.k]}</text>`;
  g += `<text class="lbl" x="${L}" y="${H - 8}">${esc(fmtTs(hist[0].ts))}</text><text class="lbl" x="${W - R}" y="${H - 8}" text-anchor="end">${esc(fmtTs(hist[hist.length - 1].ts))}</text>`;
  return `<div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Evolución de las probabilidades de 1, X y 2">${g}</svg></div>`;
}

/* ---------- Explicación de cambios ---------- */
function reasonsHtml(mv) {
  const i = OUT.indexOf(mv.focus || "1");
  const max = Math.max(1, ...((mv.reasons || []).map((r) => Math.abs(r.effect_pts[i]))));
  const items = (mv.reasons || []).filter((r) => Math.abs(r.effect_pts[i]) >= 0.5).map((r) => {
    const e = r.effect_pts[i];
    const w = Math.min(50, (Math.abs(e) / max) * 50);
    const s = e < 0 ? "neg" : "pos";
    return `<li class="reason"><b class="effect__num ${s}">${e > 0 ? "+" : ""}${dec(e, 1)}</b><div class="effect" aria-hidden="true"><span class="${s}" style="width:${w}%"></span></div>
      <div class="reason__txt"><strong>${esc(r.factor)}</strong>${r.detail ? ` <span>${esc(prettify(r.detail))}</span>` : ""}
      <span class="sr-only">(${e > 0 ? "+" : ""}${e.toFixed(1)} puntos)</span></div></li>`;
  }).join("");
  return items ? `<ul class="reasons">${items}</ul>` : "";
}

function moveHead(mv, withTeams) {
  const i = OUT.indexOf(mv.focus || "1");
  const a = mv.p_from[i], b = mv.p_to[i];
  const label = { "1": `gana ${name(mv.home)}`, X: "empate", "2": `gana ${name(mv.away)}` }[OUT[i]];
  const cls = b < a ? "down" : "up";
  const teams = withTeams ? `${esc(name(mv.home))} – ${esc(name(mv.away))}: ` : "";
  return `<div class="why__head">${teams}${esc(label)} <span class="${cls}">${pct(a)} → ${pct(b)} %</span> <span class="when">${esc(fmtTs(mv.to_ts))}</span></div>`;
}

/* ---------- Matriz de marcadores (Poisson con los goles esperados) ---------- */
function poisson(k, l) { let p = Math.exp(-l); for (let i = 1; i <= k; i++) p *= l / i; return p; }
function scoreGrid(m) {
  const lh = m.dc.xg_home, la = m.dc.xg_away, N = 5;
  let max = 0; const g = [];
  for (let i = 0; i <= N; i++) { g.push([]); for (let j = 0; j <= N; j++) { const v = poisson(i, lh) * poisson(j, la); g[i].push(v); max = Math.max(max, v); } }
  let html = `<table class="scores"><caption>Goles de ${esc(name(m.home))} en filas y de ${esc(name(m.away))} en columnas</caption><tr><th></th>${[...Array(N + 1).keys()].map((j) => `<th scope="col">${j}</th>`).join("")}</tr>`;
  for (let i = 0; i <= N; i++) {
    html += `<tr><th scope="row">${i}</th>`;
    for (let j = 0; j <= N; j++) {
      const v = g[i][j], a = (0.08 + 0.85 * (v / max)).toFixed(2);
      const strong = v / max > 0.55;
      html += `<td style="background:rgba(215,24,42,${a});color:${strong ? "#fff" : "inherit"}" title="${i}-${j}: ${(100 * v).toFixed(1)} %">${v >= 0.02 ? Math.round(100 * v) : ""}</td>`;
    }
    html += "</tr>";
  }
  return html + "</table>";
}

/* ---------- Bajas, árbitro, clima ---------- */
function contextHtml(m) {
  let h = "";
  const abs = Object.entries(m.absences || {}).filter(([, v]) => v && v.length);
  if (m.lineup_confirmed && m.lineups) {
    h += `<h4>Alineaciones oficiales</h4>`;
    for (const [side, team] of [["home", m.home], ["away", m.away]]) {
      h += `<p class="quiet"><strong>${esc(name(team))}:</strong> ${esc((m.lineups[side] || []).join(", "))}</p>`;
    }
  }
  if (abs.length) {
    h += `<h4>Bajas y dudas en las noticias</h4>`;
    for (const [team, list] of abs) {
      h += `<p class="quiet">${esc(name(team))}</p><ul class="chips">${list.map((a) => `<li class="${a.status === "doubtful" ? "doubt" : ""}">${esc(a.player)}${a.status === "doubtful" ? " (duda)" : ""}</li>`).join("")}</ul>`;
    }
  }
  const rows = [];
  if (m.referee) rows.push(["Árbitro", esc(m.referee)]);
  const w = m.weather || {};
  if (w.temp_c != null) rows.push(["Clima", `${Math.round(w.temp_c)} °C, lluvia ${w.precip_mm ?? 0} mm, viento ${Math.round(w.wind_kmh ?? 0)} km/h`]);
  if (m.market) rows.push(["Mercado", `${pct(m.market[0])} / ${pct(m.market[1])} / ${pct(m.market[2])} %`]);
  const c = m.context || {};
  const notes = [];
  if (c.derby) notes.push("derbi");
  if (c.euro_prev_home) notes.push(`${name(m.home)} viene de jugar en Europa o Copa`);
  if (c.euro_prev_away) notes.push(`${name(m.away)} viene de jugar en Europa o Copa`);
  if (c.euro_next_home) notes.push(`${name(m.home)} juega en Europa o Copa justo después`);
  if (c.euro_next_away) notes.push(`${name(m.away)} juega en Europa o Copa justo después`);
  if (c.post_break) notes.push("vuelta de parón internacional");
  if (c.new_coach_home) notes.push(`entrenador nuevo en ${name(m.home)}`);
  if (c.new_coach_away) notes.push(`entrenador nuevo en ${name(m.away)}`);
  if (c.red_prev_home) notes.push(`${name(m.home)} vio una roja en su último partido`);
  if (c.red_prev_away) notes.push(`${name(m.away)} vio una roja en su último partido`);
  if (notes.length) rows.push(["Contexto", esc(notes.join(", "))]);
  if (rows.length) h += `<h4>Contexto del partido</h4><dl class="context">${rows.map(([k, v]) => `<div><dt>${k}:</dt><dd>${v}</dd></div>`).join("")}</dl>`;
  return h;
}

function detailHtml(m) {
  const moves = (m.moves || []).slice().reverse();
  const why = moves.length
    ? `<ul class="why">${moves.map((mv) => `<li>${moveHead(mv, false)}${reasonsHtml(mv)}</li>`).join("")}</ul>`
    : `<p class="quiet">Ningún cambio ha superado aún los ${DATA.threshold} puntos. Cuando ocurra, aquí verás qué lo provocó.</p>`;
  return `<div class="detail__grid">
    <div>
      <h3>Cómo han cambiado las probabilidades</h3>
      ${evolutionChart(m)}
      <h4>Por qué han cambiado</h4>
      ${why}
    </div>
    <div>
      <h3>El partido en números</h3>
      <dl class="facts">
        <div><dt>Goles esperados</dt><dd>${m.dc.xg_home.toFixed(2)} – ${m.dc.xg_away.toFixed(2)}</dd></div>
        <div><dt>Más de 2,5 goles</dt><dd>${pct(m.dc.p_over25)} %</dd></div>
        <div><dt>Marcan los dos</dt><dd>${pct(m.dc.p_btts)} %</dd></div>
      </dl>
      <h4>Marcadores más probables (%)</h4>
      ${scoreGrid(m)}
      ${contextHtml(m)}
    </div>
  </div>`;
}

/* ---------- Fila del boleto ---------- */
function rowHtml(m, n) {
  const p = [m.p_home, m.p_draw, m.p_away];
  const imax = p.indexOf(Math.max(...p));
  const cells = p.map((v, i) => `<div class="cell${i === imax ? " cell--max" : ""}" data-o="${OUT[i]}">
      <div class="cell__fill" style="height:${(100 * v).toFixed(1)}%"></div>
      <span class="cell__pct">${pct(v)}<small>%</small></span>
      ${i === imax ? `<svg class="cell__mark" viewBox="0 0 100 100" aria-hidden="true"><path d="${pencilPath(n + 3)}" vector-effect="non-scaling-stroke"/></svg>` : ""}
    </div>`).join("");
  const status = [];
  if (m.lineup_confirmed) status.push('<span class="badge badge--ok">Alineaciones oficiales</span>');
  const nAbs = Object.values(m.absences || {}).reduce((a, l) => a + (l ? l.length : 0), 0);
  if (nAbs) status.push(`<span class="badge">${nAbs} ${nAbs === 1 ? "baja o duda" : "bajas o dudas"}</span>`);
  if ((m.moves || []).length) status.push(`<span class="badge badge--warn">${m.moves.length} ${m.moves.length === 1 ? "cambio importante" : "cambios importantes"}</span>`);
  if (!status.length) status.push("<span>Sin novedades</span>");
  const id = `d${n}`;
  const label = `${name(m.home)} contra ${name(m.away)}: local ${pct(p[0])} %, empate ${pct(p[1])} %, visitante ${pct(p[2])} %`;
  const quiet = m.history.length < 2 && status.length === 1 && status[0].startsWith("<span>");
  return `<div class="row${quiet ? " row--quiet" : ""}" role="listitem">
    <button class="row__main" aria-expanded="false" aria-controls="${id}" aria-label="${esc(label)}">
      <span class="row__num">${n}</span>
      <span class="match"><span class="match__teams">${esc(name(m.home))}<span class="vs">–</span>${esc(name(m.away))}</span><br>
        <span class="match__when">${esc(fmtWhen(m.kickoff_utc, m.date))}</span></span>
      ${cells}
      <span class="trend">${sparkline(m.history, 0)}<span class="trend__txt">${trendText(m.history)}</span></span>
      <span class="status">${status.join("")}</span>
    </button>
    <div class="detail" id="${id}" role="region" aria-label="Detalle de ${esc(name(m.home))} contra ${esc(name(m.away))}"><div class="detail__inner"></div></div>
  </div>`;
}

/* ---------- Fiabilidad ---------- */
function trustHtml(bt, live) {
  const ov = (bt && bt.overall) || {};
  const labels = { stacker: "Este modelo", dixon_coles: "Solo goles (Dixon-Coles)", elo: "Solo Elo", mercado: "Casas de apuestas (previas)", mercado_cierre: "Casas de apuestas (cierre)", stacker_mercado: "Este modelo + cuotas", frecuencias: "Repetir lo habitual", uniforme: "Al azar" };
  const order = Object.keys(ov).sort((a, b) => ov[a].log_loss - ov[b].log_loss);
  if (!order.length) return '<p class="quiet">El rendimiento aparecerá tras el primer entrenamiento.</p>';
  let t = `<table class="bt"><thead><tr><th scope="col">Modelo</th><th scope="col">Error (log-loss)</th><th scope="col">Acierta el resultado</th></tr></thead><tbody>`;
  for (const k of order) t += `<tr class="${k.startsWith("stacker") ? "ours" : ""}"><td>${labels[k] || k}</td><td>${dec(ov[k].log_loss)}</td><td>${dec(100 * ov[k].accuracy, 1)} %</td></tr>`;
  const n = ov.stacker ? ov.stacker.n : 0;
  const seasons = bt.test_seasons || [];
  t += `</tbody></table><p class="bt-note">${n.toLocaleString("es-ES")} partidos de prueba${seasons.length ? `, temporadas ${seasons[0]}/${String(seasons[0] + 1).slice(2)} a ${seasons[seasons.length - 1]}/${String(seasons[seasons.length - 1] + 1).slice(2)}` : ""}. Menos error es mejor; al azar da 1,099.</p>`;
  return t;
}

function calibChart(cal) {
  if (!cal || !cal.length) return "";
  const W = 360, H = 260, L = 48, B = 34, T = 10, R = 12;
  const x = (v) => L + v * (W - L - R), y = (v) => H - B - v * (H - B - T);
  let g = `<line class="axis" x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}"/><line class="axis" x1="${L}" y1="${T}" x2="${L}" y2="${H - B}"/>`;
  g += `<line class="diag" x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}"/>`;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) g += `<text x="${x(v)}" y="${H - B + 16}" text-anchor="middle">${v * 100}%</text><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`;
  const pts = cal.filter((c) => c.n >= 30);
  g += `<path class="ln" d="${pts.map((c, i) => `${i ? "L" : "M"}${x(c.pred).toFixed(1)} ${y(c.obs).toFixed(1)}`).join(" ")}"/>`;
  g += pts.map((c) => `<circle class="pt" cx="${x(c.pred)}" cy="${y(c.obs)}" r="${Math.max(3, Math.min(8, Math.sqrt(c.n) / 6))}"><title>Predicho ${(100 * c.pred).toFixed(0)} %, ocurrió ${(100 * c.obs).toFixed(0)} % (${c.n} casos)</title></circle>`).join("");
  g += `<text x="${(L + W - R) / 2}" y="${H - 2}" text-anchor="middle">Probabilidad que dio el modelo</text>`;
  g += `<text x="12" y="${(T + H - B) / 2}" text-anchor="middle" transform="rotate(-90 12 ${(T + H - B) / 2})">Lo que ocurrió</text>`;
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Calibración: probabilidad predicha frente a frecuencia real">${g}</svg>`;
}

/* ---------- Arranque ---------- */
let DATA = null;
async function main() {
  try {
    const r = await fetch("data.json", { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    DATA = await r.json();
  } catch (e) {
    document.getElementById("updated").textContent = "No se pudieron cargar las predicciones. Vuelve a intentarlo en unos minutos.";
    return;
  }
  const ms = DATA.matches || [];
  const upd = DATA.generated_at ? `Actualizado el ${fmtTs(DATA.generated_at)}` : "Aún no hay predicciones";
  document.getElementById("updated").textContent = ms.length ? `${upd}. ${ms.length} partidos en los próximos días.` : upd;

  const slip = document.getElementById("slip");
  if (!ms.length) { document.getElementById("empty").hidden = false; slip.hidden = true; }
  slip.innerHTML = `<div class="slip__head" aria-hidden="true"><span class="c">Nº</span><span>Partido</span><span class="c">1</span><span class="c">X</span><span class="c">2</span><span>Evolución local</span><span>Novedades</span></div>`
    + ms.map((m, i) => rowHtml(m, i + 1)).join("");

  slip.querySelectorAll(".row").forEach((row, i) => {
    const btn = row.querySelector(".row__main");
    const inner = row.querySelector(".detail__inner");
    btn.addEventListener("click", () => {
      const open = btn.getAttribute("aria-expanded") === "true";
      if (!open && !inner.dataset.ready) { inner.innerHTML = detailHtml(ms[i]); inner.dataset.ready = "1"; }
      btn.setAttribute("aria-expanded", String(!open));
      row.classList.toggle("open", !open);
    });
  });

  // el lápiz marca el boleto, fila a fila
  slip.querySelectorAll(".cell__mark path").forEach((p, i) => {
    p.style.setProperty("--len", Math.ceil(p.getTotalLength ? p.getTotalLength() * 3 : 400));
    p.style.setProperty("--delay", `${0.15 + i * 0.07}s`);
  });
  requestAnimationFrame(() => slip.classList.add("drawing"));

  const recent = DATA.recent_moves || [];
  if (recent.length) {
    document.getElementById("moves-section").hidden = false;
    document.getElementById("moves-note").textContent = `Cambios de ${DATA.threshold} puntos o más y lo que los provocó.`;
    document.getElementById("moves").innerHTML = recent.map((mv) => `<li>${moveHead(mv, true)}${reasonsHtml(mv)}</li>`).join("");
  }
  document.getElementById("bt-table").innerHTML = trustHtml(DATA.backtest, DATA.live);
  document.getElementById("calib").innerHTML = calibChart((DATA.backtest || {}).calibration);
  const lv = DATA.live || {};
  if (lv.n) document.getElementById("live").textContent =
    `En partidos reales ya jugados desde que está en marcha: ${lv.n} predicciones, error ${dec(lv.log_loss)}, acierta el resultado el ${(100 * lv.accuracy).toFixed(0)} % de las veces.`;
}
main();

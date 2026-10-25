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
  if (d === 0) return `Sin cambios (${pct(b)} %)`;
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

function probsHtml(m) {
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
        <div><dt>Goles esperados</dt><dd>${dec(m.dc.xg_home, 2)} – ${dec(m.dc.xg_away, 2)}</dd></div>
        <div><dt>Más de 2,5 goles</dt><dd>${pct(m.dc.p_over25)} %</dd></div>
        <div><dt>Marcan los dos</dt><dd>${pct(m.dc.p_btts)} %</dd></div>
      </dl>
      <h4>Marcadores más probables (%)</h4>
      ${scoreGrid(m)}
      ${contextHtml(m)}
    </div>
  </div>`;
}

/* ---------- Alineaciones: once dibujado en el campo ---------- */
// profundidad en el campo según la posición (0 portería propia, 3 delantera)
function depthOf(p) {
  const pos = String(p.pos || "").toUpperCase().replace(/[^A-Z]/g, "");
  if (p.role === "G") return 0;
  if (/^(LWB|RWB)$/.test(pos)) return 1.3;
  if (p.role === "D") return 1;
  if (/^(DM|CDM)/.test(pos)) return 1.6;
  if (/^(AM|CAM)/.test(pos)) return 2.5;
  if (/^(LW|RW|LM|RM)$/.test(pos)) return p.role === "F" ? 2.8 : 2.3;
  if (p.role === "F" || /^(CF|ST|SS|F)/.test(pos)) return 3;
  return 2;
}
function sideScore(pos) {
  const p = String(pos || "").toUpperCase();
  if (/^(RB|RW|RM|RWB)$/.test(p)) return 2;
  if (/^(LB|LW|LM|LWB)$/.test(p)) return -2;
  if (/-R$|^R/.test(p)) return 1;
  if (/-L$|^L/.test(p)) return -1;
  return 0;
}
function lines(xi, formation) {
  const gk = xi.filter((p) => p.role === "G").slice(0, 1);
  const rest = xi.filter((p) => !gk.includes(p)).sort((a, b) => depthOf(a) - depthOf(b));
  let counts = formation ? formation.split("-").map(Number).filter((n) => n > 0) : null;
  if (!counts || counts.reduce((a, b) => a + b, 0) !== rest.length) {
    counts = ["D", "M", "F"].map((r) => rest.filter((p) => p.role === r).length).filter((n) => n > 0);
  }
  const out = [gk];
  let i = 0;
  for (const c of counts) { out.push(rest.slice(i, i + c)); i += c; }
  return out.filter((l) => l.length);
}
const PARTICLES = new Set(["de", "van", "von", "di", "da", "dos", "del", "la", "le", "el", "ter", "ten"]);
function shortName(n) {
  const w = String(n).split(" ").filter(Boolean);
  if (w.length <= 1) return n;
  if (PARTICLES.has(w[w.length - 2].toLowerCase())) return w.slice(-2).join(" ");
  if (PARTICLES.has(w[0].toLowerCase())) return w.join(" ");
  return w[w.length - 1];
}
function pitchHtml(m) {
  const sq = m.squads || {};
  const W = 340, H = 520, pad = 18;
  const fieldH = H - 2 * pad, half = fieldH / 2;
  let g = `<rect x="${pad}" y="${pad}" width="${W - 2 * pad}" height="${fieldH}" class="pl"/>
    <line x1="${pad}" y1="${H / 2}" x2="${W - pad}" y2="${H / 2}" class="pl"/>
    <circle cx="${W / 2}" cy="${H / 2}" r="42" class="pl"/>
    <rect x="${W / 2 - 80}" y="${pad}" width="160" height="62" class="pl"/><rect x="${W / 2 - 36}" y="${pad}" width="72" height="22" class="pl"/>
    <rect x="${W / 2 - 80}" y="${H - pad - 62}" width="160" height="62" class="pl"/><rect x="${W / 2 - 36}" y="${H - pad - 22}" width="72" height="22" class="pl"/>`;
  const draw = (side) => {
    const s = sq[side];
    if (!s || !s.xi || !s.xi.length) return "";
    const ls = lines(s.xi, s.formation);
    let h = "";
    ls.forEach((line, li) => {
      // local abajo atacando hacia arriba; visitante arriba atacando hacia abajo
      const depth = li === 0 ? 0.04 : 0.3 + ((li - 1) / Math.max(1, ls.length - 2)) * 0.64;
      const y = side === "home" ? H - pad - 26 - depth * (half - 40) : pad + 26 + depth * (half - 40);
      const sorted = line.slice().sort((a, b) => (sideScore(a.pos) - sideScore(b.pos)) * (side === "home" ? 1 : -1));
      sorted.forEach((p, k) => {
        const x = pad + ((k + 1) / (sorted.length + 1)) * (W - 2 * pad);
        const cls = `pp pp--${side}${p.absent ? " pp--out" : ""}${s.confirmed && !p.usual ? " pp--new" : ""}`;
        const r = 11 + Math.min(5, (p.att || 0) * 4);
        const label = esc(shortName(p.name));
        const tip = `${p.name}${p.absent ? (p.absent === "doubtful" ? " (duda)" : " (baja)") : ""} · titular en el ${Math.round(100 * (p.imp || 0))} % de partidos · xG+xA/90: ${dec(p.att || 0, 2)}`;
        h += `<g class="${cls}"><title>${esc(tip)}</title><circle cx="${x}" cy="${y}" r="${r}"/>
          <text x="${x}" y="${y + 4}" class="pp__n">${esc(p.jersey || (p.role === "G" ? "P" : ""))}</text>
          <text x="${x}" y="${y + r + 12}" class="pp__name">${label}</text>
          ${p.absent ? `<line x1="${x - r}" y1="${y - r}" x2="${x + r}" y2="${y + r}" class="pp__x"/>` : ""}</g>`;
      });
    });
    return h;
  };
  g += draw("away") + draw("home");
  const head = (side, team) => {
    const s = sq[side] || {};
    const kind = s.confirmed ? `Alineación oficial${s.formation ? ` (${s.formation})` : ""}`
      : s.source === "minutos" ? "Once probable según minutos jugados" : s.xi && s.xi.length ? "Once tipo según titularidades" : "Sin datos de alineación";
    const out = (s.out || []).filter((o) => o.name);
    const outTxt = out.length ? `<p class="quiet">${s.confirmed ? "Habituales que no salen de inicio" : "Bajas y dudas"}: ${out.map((o) => esc(o.name) + (o.status === "doubtful" ? " (duda)" : "")).join(", ")}</p>` : "";
    return `<div class="lineup__team lineup__team--${side}"><h4>${esc(name(team))}</h4><p class="quiet">${kind}</p>${outTxt}</div>`;
  };
  if (!(sq.home && sq.home.xi && sq.home.xi.length) && !(sq.away && sq.away.xi && sq.away.xi.length)) {
    return `<p class="quiet">Todavía no hay datos de alineación para este partido. Aparecerán en cuanto se publiquen o haya historial de titulares.</p>`;
  }
  return `<div class="lineup">
    <div class="lineup__pitch"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Alineaciones dibujadas en el campo">${g}</svg></div>
    <div class="lineup__info">${head("away", m.away)}${head("home", m.home)}
      <p class="quiet small">El tamaño de cada círculo indica su aportación ofensiva (xG + xA por 90 minutos). Tachado: baja; borde discontinuo: titular poco habitual.</p></div>
  </div>`;
}

/* ---------- Estadísticas comparadas ---------- */
function statRow(label, a, b, fmt, higherIsBetter = true) {
  if (a == null || b == null) return "";
  const tot = Math.abs(a) + Math.abs(b) || 1;
  const wa = (Math.abs(a) / tot) * 100;
  const better = higherIsBetter ? (a > b ? "a" : b > a ? "b" : "") : (a < b ? "a" : b < a ? "b" : "");
  return `<div class="cmp"><span class="cmp__v ${better === "a" ? "is-best" : ""}">${fmt(a)}</span>
    <span class="cmp__bar" aria-hidden="true"><i style="width:${wa}%"></i></span>
    <span class="cmp__v cmp__v--b ${better === "b" ? "is-best" : ""}">${fmt(b)}</span>
    <span class="cmp__l">${label}</span></div>`;
}
const form5 = (t) => (t && t.last5 ? t.last5.map((g) => `<span class="res res--${g.res}" title="${esc(`${g.venue === "C" ? "vs" : "en"} ${name(g.opp)}: ${g.gf}-${g.ga} (${g.date})`)}">${g.res}</span>`).join("") : "");
function playersTable(t, team) {
  const kp = t && t.key_players;
  if (!kp) return "";
  return `<table class="kp"><caption>${esc(name(team))}: jugadores clave${kp.season !== (DATA.extra || {}).season ? " (temporada pasada)" : ""}</caption>
    <thead><tr><th scope="col">Jugador</th><th scope="col">Goles</th><th scope="col">Asist.</th><th scope="col">xG</th><th scope="col">xA</th></tr></thead><tbody>
    ${kp.players.map((p) => `<tr><td>${esc(p.name)}</td><td>${p.goals ?? "–"}</td><td>${p.assists ?? "–"}</td><td>${p.xg != null ? dec(p.xg, 1) : "–"}</td><td>${p.xa != null ? dec(p.xa, 1) : "–"}</td></tr>`).join("")}
    </tbody></table>`;
}
function statsHtml(m) {
  const ex = DATA.extra || {};
  const A = (ex.teams || {})[m.home], B = (ex.teams || {})[m.away];
  if (!A || !B) return '<p class="quiet">Las estadísticas aparecerán tras la próxima actualización.</p>';
  const f2 = (v) => dec(v, 2), f1 = (v) => dec(v, 1), f0 = (v) => String(Math.round(v)), ord = (v) => `${v}º`;
  const rows = [
    statRow("Posición", A.pos, B.pos, ord, false),
    statRow("Puntos por partido", A.ppg, B.ppg, f2),
    statRow("Goles a favor por partido", A.gf_pg, B.gf_pg, f2),
    statRow("Goles en contra por partido", A.ga_pg, B.ga_pg, f2, false),
    statRow("xG a favor por partido", A.xg_pg, B.xg_pg, f2),
    statRow("xG en contra por partido", A.xga_pg, B.xga_pg, f2, false),
    statRow("Tiros por partido", A.shots_pg, B.shots_pg, f1),
    statRow("Tiros a puerta por partido", A.sot_pg, B.sot_pg, f1),
    statRow("Porterías a cero", A.clean_sheets, B.clean_sheets, f0),
    statRow(`Puntos por partido (${name(m.home)} en casa, ${name(m.away)} fuera)`, A.home_ppg, B.away_ppg, f2),
    statRow("Fuerza (Elo)", A.elo, B.elo, f0),
  ].join("");
  const h2h = (ex.h2h || {})[`${m.home}|${m.away}`] || [];
  const h2hHtml = h2h.length ? `<ul class="h2h">${h2h.map((g) => `<li><span class="quiet">${new Date(g.date).toLocaleDateString("es-ES", { day: "numeric", month: "short", year: "numeric" })}</span> ${esc(name(g.home))} <b>${g.hg}-${g.ag}</b> ${esc(name(g.away))}</li>`).join("")}</ul>`
    : '<p class="quiet">No se han enfrentado en LaLiga en las temporadas con datos.</p>';
  return `<div class="detail__grid">
    <div>
      <div class="cmp__head"><b>${esc(name(m.home))}</b><b>${esc(name(m.away))}</b></div>
      ${rows}
      <div class="forms"><div><span class="quiet">Últimos 5</span> ${form5(A)}</div><div>${form5(B)}</div></div>
    </div>
    <div>
      <h4 class="mt0">Últimos enfrentamientos</h4>${h2hHtml}
      ${playersTable(A, m.home)}${playersTable(B, m.away)}
    </div>
  </div>`;
}

function detailHtml(m, i) {
  const tabs = [["probs", "Probabilidades"], ["lineup", "Alineaciones"], ["stats", "Estadísticas"]];
  return `<div class="dtabs" role="tablist" aria-label="Detalle del partido">${tabs.map(([k, l], j) =>
    `<button role="tab" id="t${i}-${k}" aria-controls="p${i}-${k}" aria-selected="${j === 0}" tabindex="${j === 0 ? 0 : -1}" data-k="${k}">${l}</button>`).join("")}</div>
    <div role="tabpanel" id="p${i}-probs" aria-labelledby="t${i}-probs">${probsHtml(m)}</div>
    <div role="tabpanel" id="p${i}-lineup" aria-labelledby="t${i}-lineup" hidden>${pitchHtml(m)}</div>
    <div role="tabpanel" id="p${i}-stats" aria-labelledby="t${i}-stats" hidden>${statsHtml(m)}</div>`;
}
function wireTabs(root) {
  const btns = [...root.querySelectorAll('.dtabs [role="tab"]')];
  const select = (b) => {
    btns.forEach((x) => {
      const on = x === b;
      x.setAttribute("aria-selected", on); x.tabIndex = on ? 0 : -1;
      root.querySelector(`#${x.getAttribute("aria-controls")}`).hidden = !on;
    });
  };
  btns.forEach((b, j) => {
    b.addEventListener("click", () => select(b));
    b.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        const n = btns[(j + (e.key === "ArrowRight" ? 1 : btns.length - 1)) % btns.length]; select(n); n.focus();
      }
    });
  });
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
  const vb = ((DATA.bets || {}).bets || []).find((x) => x.status === "pendiente" && x.strategy === "valor" && x.home === m.home && x.away === m.away);
  if (vb) status.push(`<span class="badge">Apuesta de valor: ${esc(String(vb.pick))} a ${dec(vb.odds, 2)}</span>`);
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

/* ---------- Clasificación y proyección ---------- */
const pctCell = (v) => {
  if (v == null) return "<td>–</td>";
  const t = v >= 0.995 ? ">99" : v > 0 && v < 0.005 ? "<1" : String(Math.round(100 * v));
  return `<td class="pc" style="--a:${Math.min(1, v).toFixed(3)}"><span>${t}${v > 0 ? " %" : ""}</span></td>`;
};
function standingsHtml() {
  const ex = DATA.extra || {};
  const tab = ex.table || [];
  if (!tab.length) return '<p class="quiet">La clasificación aparecerá tras la próxima actualización.</p>';
  const proj = {};
  ((ex.projection || {}).teams || []).forEach((t) => (proj[t.team] = t));
  const n = (ex.projection || {}).n_sims;
  document.getElementById("proj-note").textContent = n
    ? `Lo que queda de temporada (${ex.projection.remaining} partidos) simulado ${n.toLocaleString("es-ES")} veces con el modelo. Desempate aproximado por diferencia de goles.`
    : "La proyección aparecerá tras la próxima actualización.";
  const zone = (pos) => (pos <= 4 ? "z-ucl" : pos <= 7 ? "z-eu" : pos >= tab.length - 2 ? "z-rel" : "");
  return `<div class="tbl-wrap"><table class="standings">
    <thead><tr><th scope="col">#</th><th scope="col" class="l">Equipo</th><th scope="col">PJ</th><th scope="col">G</th><th scope="col">E</th><th scope="col">P</th>
      <th scope="col">Goles</th><th scope="col">DG</th><th scope="col">Pts</th><th scope="col" class="l">Forma</th>
      <th scope="col" class="sep">Pts al final</th><th scope="col">Liga</th><th scope="col">Champions</th><th scope="col">Europa</th><th scope="col">Descenso</th></tr></thead>
    <tbody>${tab.map((t) => {
      const p = proj[t.team] || {};
      return `<tr class="${zone(t.pos)}"><td class="pos">${t.pos}</td><th scope="row" class="l"><a href="#equipo/${encodeURIComponent(t.team)}">${esc(name(t.team))}</a></th>
        <td>${t.pj}</td><td>${t.g}</td><td>${t.e}</td><td>${t.p}</td><td>${t.gf}:${t.gc}</td><td>${t.dg > 0 ? "+" : ""}${t.dg}</td><td class="pts">${t.pts}</td>
        <td class="l form">${[...(t.form || "")].map((r) => `<span class="res res--${r}">${r}</span>`).join("")}</td>
        <td class="sep">${p.exp_pts != null ? `${Math.round(p.exp_pts)} <small>(${p.pts_p10}–${p.pts_p90})</small>` : "–"}</td>
        ${pctCell(p.p_title)}${pctCell(p.p_ucl)}${pctCell(p.p_europe)}${pctCell(p.p_releg)}</tr>`;
    }).join("")}</tbody></table></div>
    <p class="quiet small">Champions: 4 primeros. Europa: 7 primeros (incluye Champions). Entre paréntesis, el rango de puntos en 8 de cada 10 simulaciones.</p>`;
}
function positionsHtml() {
  const pr = (DATA.extra || {}).projection;
  if (!pr || !pr.teams) return "";
  const k = pr.teams[0].pos_probs.length;
  return `<div class="tbl-wrap"><table class="posgrid"><thead><tr><th scope="col" class="l">Equipo</th>${[...Array(k).keys()].map((i) => `<th scope="col">${i + 1}</th>`).join("")}</tr></thead>
    <tbody>${pr.teams.map((t) => `<tr><th scope="row" class="l">${esc(name(t.team))}</th>${t.pos_probs.map((v, i) =>
      `<td class="${v * 2.2 > 0.6 ? "is-strong" : ""}" style="--a:${Math.min(1, v * 2.2).toFixed(3)}" title="${esc(name(t.team))}, puesto ${i + 1}: ${dec(100 * v, 1)} %"><span>${v >= 0.1 ? Math.round(100 * v) : ""}</span></td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

/* ---------- Equipos ---------- */
function lineChart(points, W = 520, H = 170, color = "var(--slip)") {
  if (!points || points.length < 2) return "";
  const ys = points.map((p) => p[1]);
  const mn = Math.min(...ys), mx = Math.max(...ys);
  const pad = Math.max(3, (mx - mn) * 0.15);
  const lo = mn - pad, hi = mx + pad;
  const L = 40, R = 10, T = 10, B = 22;
  const x = (i) => L + (i / (points.length - 1)) * (W - L - R), y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  let g = "";
  for (const v of [mn, (mn + mx) / 2, mx]) g += `<line class="axis" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text class="lbl" x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${Math.round(v)}</text>`;
  g += `<path class="line" style="stroke:${color}" d="${points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(p[1]).toFixed(1)}`).join(" ")}"/>`;
  g += `<text class="lbl" x="${L}" y="${H - 4}">${esc(points[0][0])}</text><text class="lbl" x="${W - R}" y="${H - 4}" text-anchor="end">hoy</text>`;
  return `<div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Evolución de la fuerza Elo">${g}</svg></div>`;
}
function teamsHtml(sel) {
  const ex = DATA.extra || {};
  const teams = Object.entries(ex.teams || {}).filter(([, t]) => t.elo != null).sort((a, b) => b[1].elo - a[1].elo);
  if (!teams.length) return '<p class="quiet">Los equipos aparecerán tras la próxima actualización.</p>';
  const max = teams[0][1].elo, min = teams[teams.length - 1][1].elo;
  const list = `<ol class="elo">${teams.map(([k, t]) => {
    const w = 18 + ((t.elo - min) / (max - min || 1)) * 82;
    const ch = t.elo_change5;
    return `<li><a href="#equipo/${encodeURIComponent(k)}" class="${k === sel ? "is-sel" : ""}" ${k === sel ? 'aria-current="true"' : ""}>
      <span class="elo__name">${esc(name(k))}</span><span class="elo__bar"><i style="width:${w}%"></i></span>
      <span class="elo__val">${Math.round(t.elo)}${ch != null ? ` <small class="${ch >= 0 ? "up" : "down"}">${ch >= 0 ? "+" : ""}${Math.round(ch)}</small>` : ""}</span></a></li>`;
  }).join("")}</ol>`;
  let panel = '<p class="quiet">Elige un equipo de la lista.</p>';
  const t = sel && ex.teams[sel];
  if (t) {
    const pr = ((ex.projection || {}).teams || []).find((x) => x.team === sel) || {};
    const next = (DATA.matches || []).filter((mm) => mm.home === sel || mm.away === sel);
    const ph = (ex.projection_history || []).map((h) => [h.date, ((h.teams || {})[sel] || {}).exp_pts]).filter((p) => p[1] != null);
    panel = `<h3 class="team__name">${esc(name(sel))}</h3>
      <dl class="facts facts--4">
        <div><dt>Posición</dt><dd>${t.pos}º · ${t.pts} pts</dd></div>
        <div><dt>Balance</dt><dd>${t.g}-${t.e}-${t.p}</dd></div>
        <div><dt>Goles por partido</dt><dd>${t.gf_pg != null ? dec(t.gf_pg, 2) : "–"} / ${t.ga_pg != null ? dec(t.ga_pg, 2) : "–"}</dd></div>
        <div><dt>xG por partido</dt><dd>${t.xg_pg != null ? `${dec(t.xg_pg, 2)} / ${dec(t.xga_pg, 2)}` : "–"}</dd></div>
      </dl>
      ${pr.exp_pts != null ? `<p class="team__proj">Proyección: <b>${Math.round(pr.exp_pts)} puntos</b>, puesto medio ${dec(pr.exp_pos, 1)}. Liga ${pct(pr.p_title)} %, Champions ${pct(pr.p_ucl)} %, descenso ${pct(pr.p_releg)} %.</p>` : ""}
      <h4>Últimos partidos</h4>
      ${t.last5 ? `<ul class="h2h">${t.last5.slice().reverse().map((g) => `<li><span class="res res--${g.res}">${g.res}</span> ${g.venue === "C" ? "vs" : "en"} ${esc(name(g.opp))} <b>${g.gf}-${g.ga}</b></li>`).join("")}</ul>` : ""}
      ${next.length ? `<h4>Próximos partidos</h4><ul class="h2h">${next.map((mm) => {
        const home = mm.home === sel; const pw = home ? mm.p_home : mm.p_away;
        return `<li>${home ? "vs" : "en"} ${esc(name(home ? mm.away : mm.home))}: gana ${pct(pw)} %, empate ${pct(mm.p_draw)} %</li>`;
      }).join("")}</ul>` : ""}
      <h4>Fuerza (Elo) en las dos últimas temporadas</h4>${lineChart(t.elo_hist)}
      <p class="quiet small">En verano todos los equipos se acercan un poco a la media: por eso hay un salto entre temporadas.</p>
      ${ph.length >= 2 ? `<h4>Puntos finales proyectados, día a día</h4>${lineChart(ph.map((p) => [p[0], p[1]]), 520, 150, "var(--ink)")}` : ""}
      ${playersTable(t, sel)}`;
  }
  return `<div class="teams__grid"><div>${list}</div><div class="team">${panel}</div></div>`;
}

/* ---------- Aciertos ---------- */
function trackHtml() {
  const tr = (DATA.extra || {}).track || {};
  if (!tr.n) return '<p class="quiet">Todavía no se ha jugado ningún partido predicho desde que el modelo está en marcha. En cuanto termine la jornada, aquí verás cuántos acertó.</p>';
  const lab = { "1": "1", X: "X", "2": "2" };
  const weeks = tr.weeks || [];
  const maxN = Math.max(...weeks.map((w) => w.n));
  const bars = weeks.map((w) => `<div class="wk" title="Semana del ${w.week}: ${w.correct} de ${w.n}">
      <span class="wk__bar"><i style="height:${(w.n / maxN) * 100}%"></i><b style="height:${(w.correct / maxN) * 100}%"></b></span>
      <span class="wk__l">${new Date(w.week).toLocaleDateString("es-ES", { day: "numeric", month: "short" })}</span></div>`).join("");
  return `<dl class="facts facts--3">
      <div><dt>Partidos predichos</dt><dd>${tr.n}</dd></div>
      <div><dt>Acierta el resultado</dt><dd>${dec(100 * tr.accuracy, 0)} %</dd></div>
      <div><dt>Error (log-loss)</dt><dd>${dec(tr.log_loss)}</dd></div>
    </dl>
    <h4>Aciertos por semana</h4><div class="weeks" aria-label="Aciertos por semana">${bars}</div>
    <p class="quiet small">Barra clara: partidos de la semana. Barra roja: aciertos. Al azar se acertaría 1 de cada 3.</p>
    <div class="tbl-wrap"><table class="track"><thead><tr><th scope="col" class="l">Partido</th><th scope="col">Resultado</th><th scope="col">1</th><th scope="col">X</th><th scope="col">2</th><th scope="col">¿Acertó?</th></tr></thead>
    <tbody>${tr.matches.map((g) => `<tr><th scope="row" class="l">${esc(name(g.home))} – ${esc(name(g.away))} <span class="quiet small">${new Date(g.date).toLocaleDateString("es-ES", { day: "numeric", month: "short" })}</span></th>
      <td><b>${g.score}</b></td>${g.p.map((v, i) => `<td class="${["1", "X", "2"][i] === g.result ? "is-real" : ""}${["1", "X", "2"][i] === g.pick ? " is-pick" : ""}">${pct(v)} %</td>`).join("")}
      <td>${g.correct ? '<span class="ok">Sí</span>' : '<span class="ko">No</span>'}</td></tr>`).join("")}</tbody></table></div>
    <p class="quiet small">Subrayado: lo que predijo como más probable. En rojo: lo que pasó.</p>`;
}
function sourcesHtml() {
  const s = DATA.sources || {};
  const src = s.sources || {};
  const keys = Object.keys(src);
  if (!keys.length) return '<li class="quiet">Sin información todavía.</li>';
  return keys.map((k) => `<li class="${src[k].ok ? "ok" : "ko"}"><span>${src[k].ok ? "Disponible" : "Sin datos"}</span> ${esc(k)}${src[k].rows != null && src[k].ok ? ` <span class="quiet">(${src[k].rows})</span>` : ""}</li>`).join("")
    + (s.at ? `<li class="quiet">Comprobado el ${esc(fmtTs(s.at))}</li>` : "");
}

/* ---------- Apuestas simuladas ---------- */
const eur = (v) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toLocaleString("es-ES", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} €`;
const pickName = (b) => ({ "1": `gana ${name(b.home)}`, X: "empate", "2": `gana ${name(b.away)}` }[String(b.pick)] || b.pick);
function multiLine(series, W = 620, H = 220) {
  const all = series.flatMap((s) => s.points.map((p) => p[1]));
  if (!all.length || series.every((s) => s.points.length < 2)) return "";
  const mn = Math.min(0, ...all), mx = Math.max(0, ...all);
  const pad = Math.max(5, (mx - mn) * 0.1), lo = mn - pad, hi = mx + pad;
  const L = 56, R = 70, T = 10, B = 24;
  const n = Math.max(...series.map((s) => s.points.length));
  const x = (i, len) => L + (len <= 1 ? 0 : (i / (len - 1)) * (W - L - R)), y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  let g = `<line x1="${L}" x2="${W - R}" y1="${y(0)}" y2="${y(0)}" stroke="var(--ink-soft)" stroke-width="1"/><text class="lbl" x="${L - 6}" y="${y(0) + 4}" text-anchor="end">0 €</text>`;
  for (const v of [mn, mx]) if (Math.abs(v) > pad) g += `<line class="axis" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text class="lbl" x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${Math.round(v)} €</text>`;
  const labels = [];
  for (const s of series) {
    if (s.points.length < 1) continue;
    const pts = [[null, 0], ...s.points];
    g += `<path class="line" style="stroke:${s.color}${s.dash ? ";stroke-dasharray:5 4" : ""}" d="${pts.map((p, i) => `${i ? "L" : "M"}${x(i, pts.length).toFixed(1)} ${y(p[1]).toFixed(1)}`).join(" ")}"/>`;
    labels.push({ s, ly: y(pts[pts.length - 1][1]) + 4 });
  }
  labels.sort((a, b) => a.ly - b.ly);
  for (let i = 1; i < labels.length; i++) if (labels[i].ly - labels[i - 1].ly < 15) labels[i].ly = labels[i - 1].ly + 15;
  for (const { s, ly } of labels) g += `<text class="end" x="${W - R + 6}" y="${ly}" fill="${s.color}">${esc(s.label)}</text>`;
  return `<div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Beneficio acumulado">${g}</svg></div>`;
}
function stratCard(key, s, live) {
  if (!s) return "";
  const cls = s.profit > 0 ? "up" : s.profit < 0 ? "down" : "";
  return `<div class="strat"><h3>${key === "favorito" ? "Favorito del modelo" : "Apuesta de valor"}</h3>
    <p class="quiet">${esc(s.label)}.</p>
    <p class="strat__profit ${cls}">${s.bets ? eur(s.profit) : "–"}</p>
    <dl class="strat__facts">
      <div><dt>Apuestas</dt><dd>${s.bets}${live && s.pending ? ` <small>+${s.pending} abiertas</small>` : ""}</dd></div>
      <div><dt>Acertadas</dt><dd>${s.bets ? `${s.won} (${Math.round((100 * s.won) / s.bets)} %)` : "–"}</dd></div>
      <div><dt>Rentabilidad</dt><dd class="${cls}">${s.roi != null ? `${s.roi > 0 ? "+" : ""}${dec(100 * s.roi, 1)} %` : "–"}</dd></div>
      <div><dt>Cuota media</dt><dd>${s.avg_odds != null ? dec(s.avg_odds, 2) : "–"}</dd></div>
    </dl></div>`;
}
function betsHtml() {
  const b = DATA.bets || {};
  const st = b.strategies;
  if (!st) return '<p class="quiet">La cartera empezará a apostar en la próxima actualización, en cuanto haya cuotas de Bet365 para los partidos.</p>';
  const c = b.config || {};
  document.getElementById("bets-note").textContent =
    `Dinero ficticio: ${c.stake} € por apuesta con las cuotas de Bet365, colocada con la última cuota antes del partido y liquidada al terminar. Nada de esto es dinero real.`;
  const series = [
    { label: "Favorito", color: "var(--slip)", points: (st.favorito || {}).curve || [] },
    { label: "Valor", color: "var(--ink)", points: (st.valor || {}).curve || [], dash: true },
  ];
  const chart = multiLine(series);
  const bets = b.bets || [];
  const open = bets.filter((x) => x.status === "pendiente");
  const done = bets.filter((x) => x.status !== "pendiente");
  const stratName = (k) => (k === "favorito" ? "Favorito" : "Valor");
  const openT = open.length ? `<h4>Apuestas abiertas</h4><div class="tbl-wrap"><table class="track"><thead><tr><th scope="col" class="l">Partido</th><th scope="col">Estrategia</th><th scope="col" class="l">Apuesta</th><th scope="col">Cuota</th><th scope="col">Prob. del modelo</th><th scope="col">Valor esperado</th></tr></thead><tbody>
    ${open.map((x) => `<tr><th scope="row" class="l">${esc(name(x.home))} – ${esc(name(x.away))} <span class="quiet small">${esc(fmtWhen(x.kickoff_utc, x.date))}</span></th><td>${stratName(x.strategy)}</td><td class="l">${esc(pickName(x))}</td><td>${dec(x.odds, 2)}</td><td>${pct(x.prob)} %</td><td class="${x.ev >= 0 ? "up" : "down"}">${x.ev >= 0 ? "+" : ""}${dec(100 * x.ev, 0)} %</td></tr>`).join("")}
    </tbody></table></div>` : "";
  const doneT = done.length ? `<h4>Historial</h4><div class="tbl-wrap"><table class="track"><thead><tr><th scope="col" class="l">Partido</th><th scope="col">Estrategia</th><th scope="col" class="l">Apuesta</th><th scope="col">Cuota</th><th scope="col">Resultado</th><th scope="col">Ganancia</th></tr></thead><tbody>
    ${done.map((x) => `<tr><th scope="row" class="l">${esc(name(x.home))} – ${esc(name(x.away))} <span class="quiet small">${new Date(x.date).toLocaleDateString("es-ES", { day: "numeric", month: "short" })}</span></th><td>${stratName(x.strategy)}</td><td class="l">${esc(pickName(x))}</td><td>${dec(x.odds, 2)}</td><td><b>${esc(x.score || "")}</b></td><td class="${x.profit > 0 ? "up" : "down"}">${eur(x.profit)}</td></tr>`).join("")}
    </tbody></table></div>` : '<p class="quiet">Aún no hay apuestas liquidadas: el primer balance llegará cuando terminen los partidos con apuesta.</p>';
  return `<div class="strats">${stratCard("favorito", st.favorito, true)}${stratCard("valor", st.valor, true)}</div>
    ${chart ? `<h4>Beneficio acumulado</h4>${chart}` : ""}${openT}${doneT}`;
}
function betsBtHtml() {
  const bb = DATA.bets_backtest || {};
  const st = bb.strategies;
  if (!st) return '<p class="quiet">Disponible tras el próximo reentrenamiento semanal, cuando se descarguen las cuotas históricas de Bet365.</p>';
  const series = [
    { label: "Favorito", color: "var(--slip)", points: (st.favorito || {}).curve || [] },
    { label: "Valor", color: "var(--ink)", points: (st.valor || {}).curve || [], dash: true },
  ];
  const seasons = [...new Set(Object.values(st).flatMap((s) => s.by_season.map((r) => r.season)))].sort();
  const cell = (s, season) => {
    const r = s && s.by_season.find((x) => x.season === season);
    return r ? `<td class="${r.profit >= 0 ? "up" : "down"}">${eur(r.profit)} <small>(${r.roi > 0 ? "+" : ""}${dec(100 * r.roi, 1)} %)</small></td>` : "<td>–</td>";
  };
  return `<div class="strats">${stratCard("favorito", st.favorito, false)}${stratCard("valor", st.valor, false)}</div>
    <h4>Beneficio acumulado apuesta a apuesta</h4>${multiLine(series)}
    <div class="tbl-wrap"><table class="track"><thead><tr><th scope="col" class="l">Temporada</th><th scope="col">Favorito</th><th scope="col">Valor</th></tr></thead><tbody>
      ${seasons.map((se) => `<tr><th scope="row" class="l">${se}/${String(se + 1).slice(2)}</th>${cell(st.favorito, se)}${cell(st.valor, se)}</tr>`).join("")}
    </tbody></table></div>
    <p class="quiet small">Las casas de apuestas aciertan algo más que el modelo (mira la pestaña Aciertos) y cobran un margen en cada cuota. Por eso lo esperable es perder dinero a la larga; este apartado sirve para comprobarlo con datos, no para apostar.</p>`;
}

/* ---------- Navegación ---------- */
function route() {
  const h = decodeURIComponent(location.hash.slice(1) || "partidos");
  const [view, arg] = h.split("/");
  const v = ["partidos", "clasificacion", "equipos", "aciertos", "apuestas", "equipo"].includes(view) ? view : "partidos";
  const tabv = v === "equipo" ? "equipos" : v;
  document.querySelectorAll(".view").forEach((el) => (el.hidden = el.dataset.view !== tabv));
  document.querySelectorAll(".tabs a").forEach((a) => a.setAttribute("aria-current", a.dataset.tab === tabv ? "page" : "false"));
  if (tabv === "clasificacion") {
    document.getElementById("standings").innerHTML = standingsHtml();
    document.getElementById("positions").innerHTML = positionsHtml();
  }
  if (tabv === "equipos") {
    const teams = Object.entries((DATA.extra || {}).teams || {}).sort((a, b) => (b[1].elo || 0) - (a[1].elo || 0));
    document.getElementById("teams").innerHTML = teamsHtml(arg || (teams[0] || [])[0]);
  }
  if (tabv === "apuestas") {
    document.getElementById("bets").innerHTML = betsHtml();
    document.getElementById("bets-bt").innerHTML = betsBtHtml();
  }
  if (tabv === "aciertos") {
    document.getElementById("track").innerHTML = trackHtml();
    document.getElementById("sources").innerHTML = sourcesHtml();
  }
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
      if (!open && !inner.dataset.ready) { inner.innerHTML = detailHtml(ms[i], i); inner.dataset.ready = "1"; wireTabs(inner); }
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
  window.addEventListener("hashchange", route);
  route();
  const lv = DATA.live || {};
  if (lv.n) document.getElementById("live").textContent =
    `En partidos reales ya jugados desde que está en marcha: ${lv.n} predicciones, error ${dec(lv.log_loss)}, acierta el resultado el ${(100 * lv.accuracy).toFixed(0)} % de las veces.`;
}
main();

"use strict";
/* ======================================================================
   En directo: marcadores, goles, tarjetas, cambios, estadísticas y
   alineaciones de los partidos que se están jugando, directamente desde
   la API pública de ESPN (permite peticiones desde el navegador).
   Se refresca cada 30 s con la pestaña abierta (cada 2 min en segundo
   plano, solo para el contador de la pestaña) y cada 20 s el partido abierto.
   ====================================================================== */
const ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer";
const LEAGUES = [
  ["esp.1", "LaLiga"], ["esp.2", "Segunda"], ["esp.copa_del_rey", "Copa del Rey"],
  ["uefa.champions", "Champions League"], ["uefa.europa", "Europa League"], ["uefa.europa.conf", "Conference League"],
  ["esp.w.1", "Liga F"], ["eng.1", "Premier League"], ["ita.1", "Serie A"], ["ger.1", "Bundesliga"], ["fra.1", "Ligue 1"],
];
const LIVE = { events: [], at: null, open: null, summary: {}, probHist: {}, timer: null, dtimer: null, day: 0, error: null };

const ymd = (d) => d.toLocaleDateString("en-CA", { timeZone: tz }).replace(/-/g, "");
function liveDay(offset) { const d = new Date(); d.setDate(d.getDate() + offset); return d; }

function parseEvent(ev, lg, lgName) {
  const c = ev.competitions[0];
  const side = {};
  c.competitors.forEach((t) => (side[t.homeAway] = t));
  const st = c.status || ev.status;
  const team = (t) => ({ id: t.team.id, name: t.team.shortDisplayName || t.team.displayName, full: t.team.displayName,
    logo: t.team.logo, score: t.score != null && t.score !== "" ? Number(t.score) : null });
  const details = (c.details || []).map((d) => ({
    team: d.team && d.team.id, clock: (d.clock || {}).displayValue || "", goal: !!d.scoringPlay, own: !!d.ownGoal, pen: !!d.penaltyKick,
    yellow: !!d.yellowCard, red: !!d.redCard, who: ((d.athletesInvolved || [])[0] || {}).shortName || ((d.athletesInvolved || [])[0] || {}).displayName || "",
  }));
  return {
    id: ev.id, lg, lgName, date: ev.date, state: st.type.state, completed: st.type.completed,
    clock: st.displayClock, detail: st.type.shortDetail, period: st.period,
    home: team(side.home), away: team(side.away), details,
  };
}

async function fetchLive() {
  const day = ymd(liveDay(LIVE.day));
  const res = await Promise.allSettled(LEAGUES.map(([lg, n]) =>
    fetch(`${ESPN}/${lg}/scoreboard?dates=${day}`, { cache: "no-store" }).then((r) => (r.ok ? r.json() : { events: [] }))
      .then((js) => (js.events || []).map((e) => parseEvent(e, lg, n)))));
  const evs = res.filter((r) => r.status === "fulfilled").flatMap((r) => r.value);
  LIVE.error = res.every((r) => r.status === "rejected") ? "No se ha podido conectar con el servicio de resultados." : null;
  LIVE.events = evs.sort((a, b) => a.date.localeCompare(b.date));
  LIVE.at = new Date();
  return evs;
}

/* ---------- Probabilidad en directo (desde la predicción previa del modelo) ---------- */
function poisPmf(l, n = 10) { const out = []; let p = Math.exp(-l); for (let k = 0; k <= n; k++) { out.push(p); p *= l / (k + 1); } return out; }
function liveProbs(lam, mu, minute, hg, ag, redH = 0, redA = 0) {
  const rem = Math.max(0.01, (95 - Math.min(minute, 94)) / 95);
  const l = lam * rem * Math.pow(0.7, redH) * Math.pow(1.18, redA);
  const m = mu * rem * Math.pow(0.7, redA) * Math.pow(1.18, redH);
  const ph = poisPmf(l), pa = poisPmf(m);
  let w = 0, d = 0, lo = 0;
  for (let x = 0; x < ph.length; x++) for (let y = 0; y < pa.length; y++) {
    const p = ph[x] * pa[y], fh = hg + x, fa = ag + y;
    if (fh > fa) w += p; else if (fh === fa) d += p; else lo += p;
  }
  const s = w + d + lo;
  return [w / s, d / s, lo / s];
}
const plain = (s) => String(s || "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase()
  .replace(/\b(fc|cf|cd|ud|sd|rcd|rc|sad|club|de|real|femenino|women|w)\b/g, " ").replace(/[^a-z0-9 ]/g, " ").replace(/\s+/g, " ").trim();
function sameTeam(a, b) {
  const A = plain(a), B = plain(b);
  if (!A || !B) return false;
  if (A === B) return true;
  const a0 = A.split(" ").find((t) => t.length >= 4) || A.split(" ")[0];
  const b0 = B.split(" ").find((t) => t.length >= 4) || B.split(" ")[0];
  return a0.length >= 4 && b0.length >= 4 && (a0.startsWith(b0) || b0.startsWith(a0));
}
// predicción previa para un partido de ESPN: LaLiga (por id o nombres) o partido del boleto de la quiniela
function prematchFor(e) {
  const pm = Object.values(DATA.prematch || {});
  let hit = pm.find((x) => x.espn_id && String(x.espn_id) === String(e.id));
  if (!hit) hit = pm.find((x) => sameTeam(name(x.home), e.home.full) && sameTeam(name(x.away), e.away.full));
  if (!hit) hit = (DATA.matches || []).map((m) => ({ home: m.home, away: m.away, p: [m.p_home, m.p_draw, m.p_away], xg: [m.dc.xg_home, m.dc.xg_away] }))
    .find((x) => sameTeam(name(x.home), e.home.full) && sameTeam(name(x.away), e.away.full));
  if (hit) return { p: hit.p, xg: hit.xg, src: "modelo" };
  for (const j of Object.values((DATA.quiniela || {}).jornadas || {})) {
    for (const r of j.rows || []) {
      if (sameTeam(r.home, e.home.full) && sameTeam(r.away, e.away.full)) return { p: r.p, xg: [r.lam, r.mu], src: r.source };
    }
  }
  return null;
}
function minuteOf(e) {
  if (e.state === "pre") return 0;
  if (e.state === "post") return 95;
  const m = String(e.clock || "").match(/(\d+)(?:'\s*\+\s*(\d+))?/);
  if (!m) return e.period >= 2 ? 45 : 0;
  return Math.min(95, Number(m[1]) + (m[2] ? Number(m[2]) : 0));
}
function liveProbFor(e) {
  const pm = prematchFor(e);
  if (!pm || !pm.xg || pm.xg[0] == null) return null;
  const reds = (tid) => e.details.filter((d) => d.red && d.team === tid).length;
  if (e.state === "pre") return { now: pm.p, pre: pm.p, src: pm.src };
  const now = liveProbs(pm.xg[0], pm.xg[1], minuteOf(e), e.home.score || 0, e.away.score || 0, reds(e.home.id), reds(e.away.id));
  const h = (LIVE.probHist[e.id] = LIVE.probHist[e.id] || [{ m: 0, p: pm.p }]);
  const mm = minuteOf(e);
  if (!h.length || h[h.length - 1].m !== mm) h.push({ m: mm, p: now });
  return { now, pre: pm.p, src: pm.src };
}

/* ---------- Listado ---------- */
function statusLabel(e) {
  if (e.state === "in") return /half/i.test(e.detail) || e.detail === "HT" ? "Descanso" : `${esc(e.clock || "")}`;
  if (e.state === "post") return /pen/i.test(e.detail) ? "Final (pen.)" : "Final";
  if (/postp|canc|susp/i.test(e.detail)) return "Aplazado";
  return new Date(e.date).toLocaleTimeString("es-ES", { timeZone: tz, hour: "2-digit", minute: "2-digit" });
}
function scorersLine(e, tid) {
  return e.details.filter((d) => d.goal && d.team === tid).map((d) => `${esc(d.who)} ${esc(d.clock)}${d.pen ? " (p)" : ""}${d.own ? " (pp)" : ""}`).join(", ");
}
function liveRow(e) {
  const live = e.state === "in";
  const reds = (tid) => e.details.filter((d) => d.red && d.team === tid).length;
  const rc = (n) => (n ? `<span class="rc" title="${n} roja${n > 1 ? "s" : ""}">${n > 1 ? n : ""}</span>` : "");
  const pr = liveProbFor(e);
  const bar = pr ? `<span class="lv__bar" title="Probabilidad: local ${pct(pr.now[0])} %, empate ${pct(pr.now[1])} %, visitante ${pct(pr.now[2])} %">
      <i style="width:${(100 * pr.now[0]).toFixed(1)}%"></i><i style="width:${(100 * pr.now[1]).toFixed(1)}%"></i><i style="width:${(100 * pr.now[2]).toFixed(1)}%"></i></span>` : "";
  const sc = (t) => (t.score == null || e.state === "pre" ? "" : t.score);
  return `<button class="lv${live ? " lv--live" : ""}${LIVE.open === e.id ? " is-open" : ""}" data-id="${e.id}" aria-expanded="${LIVE.open === e.id}">
    <span class="lv__st${live ? " is-live" : ""}">${statusLabel(e)}</span>
    <span class="lv__t lv__t--h"><span class="lv__n">${esc(e.home.name)}</span>${rc(reds(e.home.id))}${e.home.logo ? `<img src="${esc(e.home.logo)}" alt="" loading="lazy">` : ""}</span>
    <span class="lv__sc">${sc(e.home)}<i>${e.state === "pre" ? "–" : "-"}</i>${sc(e.away)}</span>
    <span class="lv__t lv__t--a">${e.away.logo ? `<img src="${esc(e.away.logo)}" alt="" loading="lazy">` : ""}${rc(reds(e.away.id))}<span class="lv__n">${esc(e.away.name)}</span></span>
    ${bar}
  </button>${LIVE.open === e.id ? `<div class="lvd" id="lvd-${e.id}">${detailLive(e)}</div>` : ""}`;
}
function quinielaLive() {
  const js = Object.values((DATA.quiniela || {}).jornadas || {}).filter((j) => j.rows && j.tickets && !j.settled);
  if (!js.length || !LIVE.events.length) return "";
  const j = js.sort((a, b) => (a.date < b.date ? -1 : 1)).find((x) => x.frozen) || null;
  if (!j) return "";
  let hits = 0, played = 0, live = 0;
  const picks = j.tickets.sencilla.picks;
  j.rows.slice(0, 14).forEach((r, i) => {
    const e = LIVE.events.find((x) => sameTeam(r.home, x.home.full) && sameTeam(r.away, x.away.full));
    const m = (j.matches || [])[i] || {};
    let hg = null, ag = null, st = null;
    if (e && e.state !== "pre") { hg = e.home.score; ag = e.away.score; st = e.state; }
    else if (m.hg != null) { hg = m.hg; ag = m.ag; st = m.status === "fin" ? "post" : "in"; }
    if (hg == null) return;
    const out = hg > ag ? "1" : hg === ag ? "X" : "2";
    if (st === "post") played++; else live++;
    if (picks[i][0] === out) hits++;
  });
  if (!played && !live) return "";
  return `<a class="qlive" href="#quiniela"><b>Quiniela ${j.jornada ? `${j.jornada}ª` : ""}</b> — la apuesta sencilla lleva <b>${hits}</b> aciertos
    (${played} terminados${live ? `, ${live} en juego` : ""})</a>`;
}
function liveListHtml() {
  if (LIVE.error && !LIVE.events.length) return `<p class="quiet">${LIVE.error}</p>`;
  const days = [-1, 0, 1].map((o) => `<button type="button" data-day="${o}" aria-pressed="${LIVE.day === o}">${o === -1 ? "Ayer" : o === 0 ? "Hoy" : "Mañana"}</button>`).join("");
  const nLive = LIVE.events.filter((e) => e.state === "in").length;
  const head = `<div class="lvhead"><div class="chips-filter">${days}</div>
    <span class="quiet small">${nLive ? `<span class="live-dot"></span> ${nLive} en juego · ` : ""}${LIVE.at ? `actualizado a las ${LIVE.at.toLocaleTimeString("es-ES", { timeZone: tz, hour: "2-digit", minute: "2-digit", second: "2-digit" })}` : "cargando…"}</span></div>`;
  if (!LIVE.events.length) return head + (LIVE.at ? `<p class="quiet">No hay partidos de estas competiciones ${LIVE.day === 0 ? "hoy" : LIVE.day < 0 ? "ayer" : "mañana"}.</p>` : "");
  const groups = {};
  LIVE.events.forEach((e) => (groups[e.lgName] = groups[e.lgName] || []).push(e));
  const order = LEAGUES.map((l) => l[1]).filter((n) => groups[n]);
  return head + (LIVE.day === 0 ? quinielaLive() : "") + order.map((n) => `<section class="lvg"><h3>${esc(n)}</h3><div class="lvg__list">${groups[n].map(liveRow).join("")}</div></section>`).join("");
}

/* ---------- Detalle del partido ---------- */
const EVENT_ES = (t) => {
  const s = String(t || "").toLowerCase();
  if (s.includes("own goal")) return ["⚽", "Gol en propia"];
  if (s.includes("penalty") && s.includes("goal")) return ["⚽", "Gol de penalti"];
  if (s.includes("goal")) return ["⚽", "Gol"];
  if (s.includes("red")) return ["🟥", "Roja"];
  if (s.includes("yellow")) return ["🟨", "Amarilla"];
  if (s.includes("substitution")) return ["⇄", "Cambio"];
  if (s.includes("penalty") && (s.includes("miss") || s.includes("saved"))) return ["✖", "Penalti fallado"];
  if (s.includes("var")) return ["▣", "VAR"];
  if (s.includes("halftime") || s.includes("half time")) return ["⏸", "Descanso"];
  if (s.includes("end regular") || s.includes("full time") || s.includes("end of")) return ["⏹", "Final"];
  if (s === "kickoff" || s.includes("kickoff")) return ["▶", "kickoff"];
  return null;
};
const STAT_ES = [["possessionPct", "Posesión (%)"], ["totalShots", "Tiros"], ["shotsOnTarget", "Tiros a puerta"],
  ["wonCorners", "Córners"], ["foulsCommitted", "Faltas"], ["yellowCards", "Amarillas"], ["redCards", "Rojas"],
  ["offsides", "Fueras de juego"], ["saves", "Paradas"], ["accuratePasses", "Pases buenos"], ["passPct", "Acierto en el pase"]];
function detailLive(e) {
  const s = LIVE.summary[e.id];
  const pr = liveProbFor(e);
  const sum = `<div class="lvd__score">
      <div><b>${esc(e.home.full)}</b><span class="quiet small">${scorersLine(e, e.home.id)}</span></div>
      <div class="lvd__big">${e.state === "pre" ? statusLabel(e) : `${e.home.score ?? 0} - ${e.away.score ?? 0}`}<small>${e.state === "pre" ? "" : statusLabel(e)}</small></div>
      <div class="r"><b>${esc(e.away.full)}</b><span class="quiet small">${scorersLine(e, e.away.id)}</span></div></div>`;
  let prob = "";
  if (pr) {
    const h = LIVE.probHist[e.id] || [];
    prob = `<div class="lvd__prob"><h4>Probabilidad ${e.state === "post" ? "final" : "en directo"}</h4>
      <div class="lvd__pb"><span style="width:${(100 * pr.now[0]).toFixed(1)}%">${pct(pr.now[0])} %</span><span style="width:${(100 * pr.now[1]).toFixed(1)}%">${pct(pr.now[1])} %</span><span style="width:${(100 * pr.now[2]).toFixed(1)}%">${pct(pr.now[2])} %</span></div>
      <p class="quiet small">Antes del partido: ${pct(pr.pre[0])} / ${pct(pr.pre[1])} / ${pct(pr.pre[2])} % (${esc(pr.src)}). Se recalcula con el marcador, el minuto y las expulsiones a partir de los goles esperados del modelo.</p>
      ${h.length > 2 ? probChart(h) : ""}</div>`;
  }
  if (!s) return sum + prob + '<p class="quiet small">Cargando detalles…</p>';
  // cronología
  const homeId = e.home.id;
  const evs = (s.keyEvents || []).map((k) => {
    const tr = EVENT_ES(k.type && (k.type.text || k.type.type));
    if (!tr) return "";
    const who = (k.participants || []).map((p) => (p.athlete || {}).displayName).filter(Boolean);
    const sideCls = k.team ? (String(k.team.id) === String(homeId) ? "h" : "a") : "c";
    if (tr[1] === "kickoff") {
      const per = (k.period || {}).number || 1;
      if (per > 2) return "";
      tr[1] = per === 1 ? "Empieza el partido" : "Empieza la segunda parte";
    }
    let txt = tr[1];
    if (tr[1] === "Cambio" && who.length >= 2) txt = `Entra ${who[0]}, sale ${who[1]}`;
    else if (who.length && !/Empieza|Descanso|Final/.test(txt)) txt += `: ${who[0]}`;
    return `<li class="tl tl--${sideCls}"><span class="tl__m">${esc((k.clock || {}).displayValue || "")}</span><span class="tl__i">${tr[0]}</span><span class="tl__t">${esc(txt)}</span></li>`;
  }).filter(Boolean).reverse().join("");
  // estadísticas
  const T = (s.boxscore || {}).teams || [];
  const byHA = {};
  T.forEach((t) => { const ha = String(t.team.id) === String(homeId) ? "h" : "a"; byHA[ha] = Object.fromEntries((t.statistics || []).map((x) => [x.name, x.displayValue])); });
  const num = (v) => parseFloat(String(v || "0").replace(",", "."));
  const stats = byHA.h && byHA.a ? STAT_ES.filter(([k]) => byHA.h[k] != null).map(([k, l]) => {
    let a = num(byHA.h[k]), b = num(byHA.a[k]);
    const fmt = (v) => (k === "passPct" ? `${Math.round(v * (v <= 1 ? 100 : 1))} %` : String(Math.round(v * 10) / 10));
    const tot = a + b || 1;
    return `<div class="cmp"><span class="cmp__v">${fmt(a)}</span><span class="cmp__bar" aria-hidden="true"><i style="width:${(100 * a / tot).toFixed(1)}%"></i></span><span class="cmp__v cmp__v--b">${fmt(b)}</span><span class="cmp__l">${l}</span></div>`;
  }).join("") : '<p class="quiet small">Sin estadísticas todavía.</p>';
  // alineaciones
  const lineup = (s.rosters || []).map((r) => {
    const xi = (r.roster || []).filter((p) => p.starter);
    const subs = (r.roster || []).filter((p) => !p.starter && p.subbedIn);
    if (!xi.length) return "";
    const pl = (p) => `<li><span class="jn">${esc(p.jersey || "")}</span>${esc((p.athlete || {}).displayName || "")}${p.subbedOut ? ' <span class="quiet small">⇄</span>' : ""}</li>`;
    const form = typeof r.formation === "string" ? r.formation : (r.formation || {}).name || "";
    return `<div class="lvd__xi"><h5>${esc(r.team.displayName)}${form ? ` <span class="quiet">(${esc(form)})</span>` : ""}</h5><ol>${xi.map(pl).join("")}</ol>
      ${subs.length ? `<p class="quiet small">Entraron: ${subs.map((p) => esc((p.athlete || {}).displayName || "")).join(", ")}</p>` : ""}</div>`;
  }).join("");
  return `${sum}${prob}<div class="lvd__grid">
      <div><h4>Lo que ha pasado</h4>${evs ? `<ol class="tline">${evs}</ol>` : '<p class="quiet small">Aún no hay incidencias.</p>'}</div>
      <div><h4>Estadísticas</h4>${stats}${lineup ? `<h4>Alineaciones</h4><div class="lvd__xis">${lineup}</div>` : ""}</div>
    </div>`;
}
function probChart(h) {
  const W = 520, H = 120, L = 30, R = 10, T = 8, B = 18;
  const x = (m) => L + (m / 95) * (W - L - R), y = (p) => T + (1 - p) * (H - T - B);
  let g = "";
  for (const p of [0, 0.5, 1]) g += `<line class="axis" x1="${L}" x2="${W - R}" y1="${y(p)}" y2="${y(p)}"/><text class="lbl" x="${L - 4}" y="${y(p) + 4}" text-anchor="end">${p * 100}%</text>`;
  [["l1", 0], ["lX", 1], ["l2", 2]].forEach(([c, k]) => {
    g += `<path class="line ${c}" d="${h.map((q, i) => `${i ? "L" : "M"}${x(q.m).toFixed(1)} ${y(q.p[k]).toFixed(1)}`).join(" ")}"/>`;
  });
  g += `<text class="lbl" x="${L}" y="${H - 4}">0'</text><text class="lbl" x="${x(45)}" y="${H - 4}" text-anchor="middle">45'</text><text class="lbl" x="${W - R}" y="${H - 4}" text-anchor="end">90'</text>`;
  return `<div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Evolución de la probabilidad durante el partido">${g}</svg></div>`;
}

async function loadSummary(e) {
  try {
    const r = await fetch(`${ESPN}/${e.lg}/summary?event=${e.id}`, { cache: "no-store" });
    if (r.ok) LIVE.summary[e.id] = await r.json();
  } catch (err) { /* sin conexión: se reintenta en la próxima actualización */ }
}

/* ---------- Ciclo de actualización ---------- */
function renderLive() {
  const el = document.getElementById("live-list");
  if (!el) return;
  const y = window.scrollY;
  el.innerHTML = liveListHtml();
  window.scrollTo(0, y);
  el.querySelectorAll(".lv").forEach((b) => b.addEventListener("click", async () => {
    const id = b.dataset.id;
    LIVE.open = LIVE.open === id ? null : id;
    renderLive();
    if (LIVE.open) {
      const e = LIVE.events.find((x) => x.id === id);
      await loadSummary(e);
      renderLive();
    }
  }));
  el.querySelectorAll("[data-day]").forEach((b) => b.addEventListener("click", async () => {
    LIVE.day = Number(b.dataset.day); LIVE.open = null; LIVE.events = []; LIVE.at = null; renderLive();
    await fetchLive(); renderLive();
  }));
}
function updateBadge() {
  const n = LIVE.events.filter((e) => e.state === "in").length;
  const a = document.querySelector('.tabs a[data-tab="directo"]');
  if (a) a.classList.toggle("has-live", n > 0);
  const gl = document.getElementById("gl-live");
  if (gl) {
    const live = LIVE.events.filter((e) => e.state === "in");
    gl.querySelector(".gl__v").innerHTML = live.length ? `<span class="live-dot"></span> ${live.length} en juego` : "Resultados al momento";
    gl.querySelector(".gl__s").textContent = live.length
      ? live.slice(0, 2).map((e) => `${e.home.name} ${e.home.score}-${e.away.score} ${e.away.name}`).join(" · ")
      : `${LIVE.events.filter((e) => e.state === "pre").length} partidos más hoy`;
  }
  // marcadores en directo dentro del boleto de la pestaña Partidos
  document.querySelectorAll("[data-espn]").forEach((elm) => {
    const e = LIVE.events.find((x) => x.id === elm.dataset.espn);
    if (e && e.state !== "pre") elm.innerHTML = `<span class="badge badge--live">${e.state === "in" ? '<span class="live-dot"></span>' : ""}${e.home.score}-${e.away.score} ${statusLabel(e)}</span>`;
  });
}
async function tick() {
  const onTab = !document.getElementById("live-list").closest(".view").hidden;
  if (document.hidden && !onTab) return;
  await fetchLive();
  if (LIVE.open && onTab) {
    const e = LIVE.events.find((x) => x.id === LIVE.open);
    if (e) await loadSummary(e);
  }
  updateBadge();
  if (onTab) renderLive();
}
function startLive() {
  clearTimeout(LIVE.timer);
  const gen = (LIVE.gen = (LIVE.gen || 0) + 1);
  const period = () => (document.getElementById("live-list").closest(".view").hidden ? 120000 : 30000);
  const loop = async () => {
    if (gen !== LIVE.gen) return;
    await tick();
    if (gen === LIVE.gen) LIVE.timer = setTimeout(loop, period());
  };
  loop();
}
window.renderLive = renderLive;
window.startLive = startLive;

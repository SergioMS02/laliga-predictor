"""Quiniela oficial (14 partidos + Pleno al 15) rellenada por el modelo.

El boleto de cada jornada se lee de combinacionganadora.com (la web de Loterías bloquea
las peticiones automáticas). Para cada partido:
  · LaLiga: la predicción completa del modelo (la misma de la web).
  · Segunda: probabilidades del mercado (football-data.co.uk) si hay cuotas, porque en
    Segunda el mercado acierta claramente más que cualquier modelo propio; si no, un
    Dixon-Coles ajustado con el histórico de Segunda.
  · Resto (Copa con equipos de otras categorías, Europa, fútbol femenino...): estimación
    aproximada por categoría del equipo (y Elo propio si es de LaLiga), marcada como tal.

Se generan dos boletos: el sencillo (una apuesta) y uno reducido con dobles y triples
elegidos para maximizar la probabilidad de 14 aciertos dentro de un presupuesto de
apuestas. El boleto se congela cuando empieza el primer partido de la jornada y se
liquida con los resultados y la tabla oficial de premios. Todo queda en
predictions/quiniela.json.
"""
from __future__ import annotations

import html as _html
import json
import logging
import math
import re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .config import Settings
from .data import http, results
from .models.dixon_coles import DixonColes, score_matrix
from .teams import canonical

log = logging.getLogger(__name__)

CG_URL = "https://www.combinacionganadora.com/quiniela/"
CG_DATE_URL = CG_URL + "resultados/{date}/"
_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}
SIGNS = ("1", "X", "2")
PLENO_CATS = ("0", "1", "2", "M")
MONTHS = {m: i + 1 for i, m in enumerate(["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
                                          "septiembre", "octubre", "noviembre", "diciembre"])}
# Rating aproximado por categoría (escala Elo) para partidos sin modelo
TIER_RATING = {1: 1500.0, 2: 1370.0, 3: 1170.0, "foreign": 1520.0, "women": 1400.0}
HOME_ADV = 65.0


def cfg(settings: Settings) -> dict:
    c = {"price": 0.75, "max_bets": 16}
    c.update(getattr(settings, "raw", {}).get("quiniela", {}) or {})
    return c


# ---------------------------------------------------------------------------
# Lectura del boleto
# ---------------------------------------------------------------------------
def _txt(s: str | None) -> str:
    if s is None:
        return ""
    return _html.unescape(re.sub(r"<[^>]+>", " ", s)).replace("\xa0", " ").strip()


def _num(s: str | None) -> int | None:
    m = re.search(r"\d+", s or "")
    return int(m.group()) if m else None


def _money(s: str) -> float | None:
    s = _txt(s).replace("€", "").replace(".", "").replace(",", ".").strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse_page(page: str) -> dict:
    """Extrae fecha, jornada, partidos (con resultado si lo hay) y premios de una página de jornada."""
    out = {"date": None, "jornada": None, "prev": None, "next": None, "matches": [], "prizes": {}, "recaudacion": None}
    i = page.find('id="sorteo"')
    body = page[i:] if i >= 0 else page
    links = re.findall(r'quiniela/resultados/(\d{4}-\d{2}-\d{2})/', body[:4000])
    m = re.search(r'fdQuiniela">\s*\w+,\s*(\d{1,2})\s+(\w+)\s+(\d{4})', body)
    if m and m.group(2).lower() in MONTHS:
        out["date"] = f"{int(m.group(3)):04d}-{MONTHS[m.group(2).lower()]:02d}-{int(m.group(1)):02d}"
    if links:
        out["prev"] = links[0]
        out["next"] = links[1] if len(links) > 1 else None
    j = re.search(r'"jornada":"(\d+)"', page)
    out["jornada"] = int(j.group(1)) if j else None
    for row in re.findall(r"<tr[^>]*>\s*<td data-match-num[^>]*>.*?</tr>", body, flags=re.S):
        n = _num(re.search(r"data-match-num[^>]*>(.*?)</td>", row, flags=re.S).group(1))
        h = re.search(r"data-m1\s*>(.*?)</span>", row, flags=re.S)
        a = re.search(r"data-m2\s*>(.*?)</span>", row, flags=re.S)
        r1 = re.search(r"data-r1>(.*?)</span>", row, flags=re.S)
        r2 = re.search(r"data-r2>(.*?)</span>", row, flags=re.S)
        mn = re.search(r"data-min>(.*?)</div>", row, flags=re.S)
        tm = re.search(r'<time datetime="([^"]*)"', row)
        state = _txt(mn.group(1)) if mn else ""
        hg, ag = _num(r1.group(1)) if r1 else None, _num(r2.group(1)) if r2 else None
        status = "fin" if state.lower().startswith("fin") else ("live" if hg is not None else "pending")
        out["matches"].append({"n": n, "home": _txt(h.group(1)) if h else "", "away": _txt(a.group(1)) if a else "",
                               "hg": hg, "ag": ag, "status": status, "when": tm.group(1) if tm else (state or None)})
    rec = re.search(r"<dt>\s*Recaudaci[oó]n\s*</dt>\s*<dd>(.*?)</dd>", page, flags=re.S)
    if rec:
        out["recaudacion"] = _money(rec.group(1))
    pz = re.search(r'class="escrutinioTable".*?</table>', page, flags=re.S)
    if pz:
        for row in re.findall(r"<tr>(.*?)</tr>", pz.group(), flags=re.S):
            cat = re.search(r"data-cat>(.*?)</td>", row, flags=re.S)
            prize = re.search(r"data-prize>(.*?)</td>", row, flags=re.S)
            if cat and prize:
                v = _money(prize.group(1))
                if v is not None:
                    out["prizes"][_txt(cat.group(1))] = v
    return out


def fetch_page(date: str | None = None) -> dict | None:
    url = CG_DATE_URL.format(date=date) if date else CG_URL
    try:
        r = http.get(url, timeout=25, retries=1, headers=_UA)
        p = parse_page(r.text)
        if date and not p["date"]:
            p["date"] = date
        return p if p["matches"] else None
    except Exception as e:
        log.warning("Quiniela no disponible (%s): %s", url, e)
        return None


# ---------------------------------------------------------------------------
# Probabilidades de cada partido
# ---------------------------------------------------------------------------
def _plain(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


_EXTRA_ALIASES = {
    "r oviedo": "Oviedo", "r zaragoza": "Zaragoza", "r sociedad": "Sociedad", "r madrid": "Real Madrid",
    "r betis": "Betis", "r valladolid": "Valladolid", "r racing": "Santander", "r sporting": "Sp Gijon",
    "cd castellon": "Castellon", "deportivo": "La Coruna", "athletic de bilbao": "Ath Bilbao",
    "atletico de madrid": "Ath Madrid", "sporting": "Sp Gijon", "racing": "Santander", "cultural": "Cultural Leonesa",
    "cultural leonesa": "Cultural Leonesa", "ad ceuta": "Ceuta", "ceuta": "Ceuta", "fc andorra": "Andorra",
    "andorra": "Andorra", "real union": "Real Union", "almeria": "Almeria", "cadiz": "Cadiz", "malaga": "Malaga",
    "granada": "Granada", "leganes": "Leganes", "cordoba": "Cordoba", "huesca": "Huesca", "eibar": "Eibar",
    "mirandes": "Mirandes", "burgos": "Burgos", "albacete": "Albacete", "las palmas": "Las Palmas",
    "tenerife": "Tenerife", "valladolid": "Valladolid", "zaragoza": "Zaragoza",
}


def resolve(name: str, laliga: set[str], segunda: set[str]) -> tuple[str, object]:
    """(nombre canónico o original, categoría: 1, 2, 3, 'women' o 'foreign')."""
    raw = name.strip()
    if re.search(r"femenin|\bfem\b|\bwomen\b", raw, flags=re.I):
        return raw, "women"
    p = _plain(raw)
    cand = _EXTRA_ALIASES.get(p)
    if cand is None:
        try:
            cand = canonical(raw, strict=False, fuzzy=True)
        except Exception:
            cand = raw
    if cand in laliga:
        return cand, 1
    if cand in segunda:
        return cand, 2
    pool = list(segunda | laliga)
    if pool:
        best = process.extractOne(p, {t: _plain(t) for t in pool}, scorer=fuzz.token_sort_ratio)
        if best and best[1] >= 88:
            t = best[2]
            return t, 1 if t in laliga else 2
    return raw, "unknown"


def elo_probs(r_home: float, r_away: float) -> list[float]:
    """1X2 aproximado a partir de una diferencia de rating (con ventaja de campo)."""
    dr = r_home + HOME_ADV - r_away
    e = 1 / (1 + 10 ** (-dr / 400))
    draw = max(0.10, 0.29 * (1 - abs(2 * e - 1)) ** 0.7)
    ph = max(0.02, e - draw / 2)
    pa = max(0.02, 1 - e - draw / 2)
    s = ph + draw + pa
    return [ph / s, draw / s, pa / s]


def goals_from_probs(p: list[float]) -> tuple[float, float]:
    """Goles esperados aproximados coherentes con un 1X2 (para el Pleno al 15 sin modelo)."""
    best, err = (1.3, 1.1), 9.0
    for lam in np.arange(0.3, 3.6, 0.1):
        for mu in np.arange(0.3, 3.6, 0.1):
            m = score_matrix(lam, mu, -0.05, 10)
            m = m / m.sum()
            i, j = np.indices(m.shape)
            q = [m[i > j].sum(), m[i == j].sum(), m[i < j].sum()]
            e = sum((a - b) ** 2 for a, b in zip(q, p)) + 0.002 * abs(lam + mu - 2.6)
            if e < err:
                best, err = (float(lam), float(mu)), e
    return best


def pleno_probs(lam: float, mu: float) -> dict:
    m = score_matrix(lam, mu, -0.05, 10)
    m = m / m.sum()
    cat = lambda g: min(g, 3)  # noqa: E731
    probs = {}
    for a in range(m.shape[0]):
        for b in range(m.shape[1]):
            k = (PLENO_CATS[cat(a)], PLENO_CATS[cat(b)])
            probs[k] = probs.get(k, 0.0) + float(m[a, b])
    best = max(probs, key=probs.get)
    top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
    return {"pick": list(best), "p": round(probs[best], 3), "top": [[f"{k[0]}-{k[1]}", round(v, 3)] for k, v in top]}


class SegundaModel:
    """Dixon-Coles con el histórico de Segunda + cuotas del mercado de los próximos partidos."""

    def __init__(self, settings: Settings, online: bool):
        self.settings = settings
        hist = results.load_history(settings, refresh_current=online, online=online, div="SP2")
        self.matches = hist
        cur = results.current_season()
        self.teams = set(hist[hist.season == cur].home) | set(hist[hist.season == cur].away)
        fx = results.fetch_fixtures(settings, online=online, div="SP2")
        self.teams |= set(fx.home) | set(fx.away)
        self.odds = {(r.home, r.away): (r.odds_h, r.odds_d, r.odds_a) for r in fx.itertuples()
                     if not pd.isna(r.odds_h)}
        c = settings["dixon_coles"]
        self.dc = DixonColes(xi=c["xi"], ridge=c["ridge"], window_days=c["window_days"],
                             new_team_attack=-0.2, new_team_defence=0.15, max_goals=10)
        prev = set(hist[hist.season == cur - 1].home) | set(hist[hist.season == cur - 1].away)
        self.dc.fit(hist, pd.Timestamp.now().normalize() + pd.Timedelta(days=1), current_teams=sorted(self.teams),
                    new_teams=self.teams - prev)

    def predict(self, home: str, away: str) -> dict:
        lam, mu = self.dc.expected_goals(home, away)
        p = self.dc.predict(home, away, lam, mu)
        model = [p["p_home"], p["p_draw"], p["p_away"]]
        o = self.odds.get((home, away))
        if o:
            from .models.market import devig_power
            mk = devig_power(np.array([o], float))[0]
            if np.isfinite(mk).all():
                return {"p": [float(x) for x in mk], "lam": lam, "mu": mu, "source": "mercado"}
        return {"p": model, "lam": lam, "mu": mu, "source": "modelo Segunda"}


def _laliga_elo(settings: Settings) -> dict:
    p = settings.predictions_dir / "site_extra.json"
    try:
        teams = json.loads(p.read_text(encoding="utf-8")).get("teams", {})
        return {t: v["elo"] for t, v in teams.items() if v.get("elo")}
    except Exception:
        return {}


def is_european(matches: list[dict], laliga: set[str], segunda: set[str]) -> bool:
    """Jornada de competiciones europeas: muchos partidos sin ningún equipo español conocido."""
    unk = 0
    for m in matches[:14]:
        th = resolve(m["home"], laliga, segunda)[1]
        ta = resolve(m["away"], laliga, segunda)[1]
        unk += th == "unknown" and ta == "unknown"
    return unk >= 5


def match_probs(home: str, away: str, preds: dict, seg: SegundaModel | None, laliga: set[str],
                elo: dict, european: bool = False, market: list[float] | None = None, euro=None) -> dict:
    """Probabilidad 1X2 de un partido del boleto con la mejor fuente disponible.

    Orden: modelo de LaLiga > mercado de apuestas (Betfair, publicado por eduardolosilla.es para
    todos los partidos del boleto) > Segunda (cuotas o Dixon-Coles) > Elo europeo / Liga F > categoría."""
    h, th = resolve(home, laliga, seg.teams if seg else set())
    a, ta = resolve(away, laliga, seg.teams if seg else set())
    if "women" in (th, ta):
        th = ta = "women"
    pr = preds.get((h, a))
    if pr is not None:
        return {"p": [pr["p_home"], pr["p_draw"], pr["p_away"]], "lam": pr["dc"]["xg_home"], "mu": pr["dc"]["xg_away"],
                "source": "modelo LaLiga", "teams": [h, a]}
    if market is not None and len(market) == 3 and all(x is not None for x in market) and sum(market) > 0:
        p = [float(x) / sum(market) for x in market]
        if seg is not None and th == 2 and ta == 2:
            r = seg.predict(h, a)
            lam, mu = r["lam"], r["mu"]
        else:
            lam, mu = goals_from_probs(p)
        return {"p": p, "lam": lam, "mu": mu, "source": "mercado", "teams": [h, a]}
    if seg is not None and th == 2 and ta == 2:
        r = seg.predict(h, a)
        return {**r, "teams": [h, a]}
    if euro is not None:
        fh = euro.find(home, women=th == "women") if th not in (1, 2) else None
        fa = euro.find(away, women=ta == "women") if ta not in (1, 2) else None
        if th == "women" and fh and fa:
            p = elo_probs(fh[1], fa[1])
            lam, mu = goals_from_probs(p)
            return {"p": p, "lam": lam, "mu": mu, "source": "Elo Liga F", "teams": [h, a]}
        if th != "women" and (fh or fa):
            def r_of(t, tier, f):
                if f:
                    return f[1]
                if tier == 1:
                    return elo.get(t, TIER_RATING[1])
                if tier == 2 and seg is not None and t in seg.dc.params.teams:
                    att, de = seg.dc.team_strength(t)
                    return TIER_RATING[2] + 400 * (att - de) / 2.3
                return TIER_RATING["foreign"] if european and tier == "unknown" else TIER_RATING.get(tier, TIER_RATING[3])
            p = elo_probs(r_of(h, th, fh), r_of(a, ta, fa))
            lam, mu = goals_from_probs(p)
            return {"p": p, "lam": lam, "mu": mu, "source": "Elo europeo", "teams": [h, a]}
    if th == "women" and ta == "women":
        p = elo_probs(1400, 1400)
        lam, mu = goals_from_probs(p)
        return {"p": p, "lam": lam, "mu": mu, "source": "estimación genérica (femenino)", "teams": [h, a]}

    def rating(t, tier):
        if tier == 1:
            return elo.get(t, TIER_RATING[1])
        if tier == 2 and seg is not None and t in seg.dc.params.teams:
            att, de = seg.dc.team_strength(t)
            return TIER_RATING[2] + 400 * (att - de) / 2.3
        if tier == "unknown":  # en Europa, un rival europeo; en Copa, un equipo de categoría inferior
            return TIER_RATING["foreign"] if european else TIER_RATING[3]
        return TIER_RATING.get(tier, TIER_RATING[3])
    p = elo_probs(rating(h, th), rating(a, ta))
    lam, mu = goals_from_probs(p)
    if th == 1 and ta == 1 and h in elo and a in elo:  # dos equipos de LaLiga: su Elo real, no una categoría
        return {"p": p, "lam": lam, "mu": mu, "source": "Elo LaLiga", "teams": [h, a]}
    return {"p": p, "lam": lam, "mu": mu, "source": "estimación por categoría", "teams": [h, a]}


# ---------------------------------------------------------------------------
# Boletos
# ---------------------------------------------------------------------------
def poisson_binomial(ps: list[float]) -> list[float]:
    dist = [1.0]
    for p in ps:
        new = [0.0] * (len(dist) + 1)
        for k, v in enumerate(dist):
            new[k] += v * (1 - p)
            new[k + 1] += v * p
        dist = new
    return dist


def best_reduction(probs: list[list[float]], max_bets: int) -> list[list[str]]:
    """Dobles y triples que maximizan P(14) con 2^dobles · 3^triples ≤ max_bets."""
    n = len(probs)
    order = [np.argsort(p)[::-1] for p in probs]
    gains = []
    for p, o in zip(probs, order):
        p1, p2 = p[o[0]], p[o[1]]
        gains.append((math.log(max(p1, 1e-9)), math.log(max(p1 + p2, 1e-9)), 0.0))
    dmax = int(math.log2(max_bets)) if max_bets >= 2 else 0
    tmax = int(math.log(max_bets, 3) + 1e-9) if max_bets >= 3 else 0
    NEG = -1e18
    dp = {(0, 0): (0.0, [])}
    for i in range(n):
        nxt = {}
        for (d, t), (v, ch) in dp.items():
            for kind, (dd, tt) in enumerate(((0, 0), (1, 0), (0, 1))):
                nd, nt = d + dd, t + tt
                if nd > dmax or nt > tmax or (2 ** nd) * (3 ** nt) > max_bets:
                    continue
                val = v + gains[i][kind]
                if val > nxt.get((nd, nt), (NEG, None))[0]:
                    nxt[(nd, nt)] = (val, ch + [kind])
        dp = nxt
    (_, choice) = max(dp.values(), key=lambda x: x[0])
    out = []
    for o, k in zip(order, choice):
        out.append([SIGNS[j] for j in sorted(o[:k + 1])])
    return out


# ---------------------------------------------------------------------------
# Lo que juega la gente (porcentajes de Loterías) y premio esperado
# ---------------------------------------------------------------------------
EL_URL = "https://www.eduardolosilla.es/quiniela"
# reparto aproximado de la recaudación por categoría (normas de La Quiniela; el P15 aparte)
POOL = {14: 0.16, 13: 0.075, 12: 0.075, 11: 0.075, 10: 0.075}
MIN_PRIZE = 1.5           # por debajo de 1,50 € la categoría no cobra
DEFAULT_RECAUDACION = 1_300_000.0


def parse_public(page: str) -> dict:
    """{nº jornada: {"lae": [[%1,%X,%2]...], "real": [[p1,pX,p2]...], "pleno_lae": {...}}} desde el estado
    que eduardolosilla.es incrusta en su página (porcentajes de Loterías y probabilidades de Betfair)."""
    m = re.search(r'<script id="[^"]*state" type="application/json">(.*?)</script>', page, flags=re.S)
    if not m:
        return {}
    raw = m.group(1).replace("&q;", '"').replace("&a;", "&").replace("&l;", "<").replace("&g;", ">").replace("&s;", "'")
    try:
        st = json.loads(raw)
    except ValueError:
        return {}
    out = {}

    def row(r):
        try:
            v = [float(r.get("porc_1_dec") or r["porc_1"]), float(r.get("porc_X_dec") or r["porc_X"]),
                 float(r.get("porc_2_dec") or r["porc_2"])]
            return [x / 100 for x in v] if sum(v) > 0 else None
        except (TypeError, ValueError, KeyError):
            return None
    for key, val in (st.get("probabilidades") or {}).items():
        if not isinstance(val, dict) or "partidos" not in val:
            continue
        n = val.get("numJornada") or _num(key.split("_")[-1])
        parts = val["partidos"]
        out[int(n)] = {k: [row(r) for r in (parts.get(k) or [])[:14]] for k in ("lae", "real", "quinielista")}
        out[int(n)]["updated"] = (val.get("fechaModificacion") or {}).get("lae")
    return out


def fetch_public(online: bool = True) -> dict:
    if not online:
        return {}
    try:
        return parse_public(http.get(EL_URL, timeout=25, retries=1, headers=_UA).text)
    except Exception as e:
        log.warning("Porcentajes de la quiniela no disponibles: %s", e)
        return {}


def simulate_outcomes(probs: list[list[float]], n: int = 8000, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    P = np.asarray(probs, float)
    P = P / P.sum(1, keepdims=True)
    u = rng.random((n, len(P)))
    return (u[:, :, None] > np.cumsum(P, 1)[None, :, :-1]).sum(2)  # 0=1, 1=X, 2=2


def winners_by_outcome(public: list[list[float]], outcomes: np.ndarray, n_cols: float) -> np.ndarray:
    """Acertantes esperados (otras apuestas) con k aciertos, para cada resultado simulado: (n, 15)."""
    Pub = np.asarray(public, float)
    Pub = Pub / Pub.sum(1, keepdims=True)
    hit = Pub[np.arange(Pub.shape[0])[None, :], outcomes]  # (n, 14)
    dist = np.zeros((len(outcomes), Pub.shape[0] + 1))
    dist[:, 0] = 1.0
    for i in range(Pub.shape[0]):
        p = hit[:, i:i + 1]
        dist[:, 1:] = dist[:, 1:] * (1 - p) + dist[:, :-1] * p
        dist[:, 0] = dist[:, 0] * (1 - p[:, 0])
    return dist * n_cols


def prize_matrix(winners: np.ndarray, recaudacion: float) -> np.ndarray:
    """Premio por columna con k aciertos en cada resultado simulado (incluye nuestra propia columna)."""
    pr = np.zeros_like(winners)
    for k, share in POOL.items():
        v = share * recaudacion / (winners[:, k] + 1)
        pr[:, k] = np.where(v >= MIN_PRIZE, v, 0.0)
    return pr


def cover_ev(cover: list[list[str]], outcomes: np.ndarray, prizes: np.ndarray) -> float:
    """Premio medio esperado de un boleto (todas sus columnas)."""
    counts = np.zeros((len(outcomes), len(cover) + 1))
    counts[:, 0] = 1.0
    for i, c in enumerate(cover):
        idx = [SIGNS.index(x) for x in c]
        hit = np.isin(outcomes[:, i], idx).astype(float)[:, None]
        miss = len(c) - hit
        new = counts * miss
        new[:, 1:] += counts[:, :-1] * hit
        counts = new
    return float((counts * prizes[:, :counts.shape[1]]).sum(1).mean())


def best_value_ticket(probs, public, max_bets: int, recaudacion: float, start=None, n_sims: int = 60000) -> dict:
    """Boleto (≤ max_bets apuestas) con el mayor premio esperado.

    Los premios se reparten entre los acertantes, así que un resultado probable que la mayoría no
    juega vale más. Para cada partido se usa q ∝ p · (p / % del público)^β y se elige el reducido
    con más probabilidad de 14 según q. β se escoge entre unos pocos valores con una simulación y el
    premio esperado final se mide con otra simulación independiente (si no, la búsqueda se
    "enamora" del ruido de la simulación y promete rentabilidades imposibles)."""
    P = np.asarray(probs, float)
    Pub = np.clip(np.asarray(public, float), 0.01, None)
    rec_cols = recaudacion / 0.75
    outs_a = simulate_outcomes(probs, n_sims, seed=11)
    prizes_a = prize_matrix(winners_by_outcome(public, outs_a, rec_cols), recaudacion)
    best = None
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
        q = P * (P / Pub) ** beta
        q = q / q.sum(1, keepdims=True)
        cover = best_reduction([list(r) for r in q], max_bets)
        v = cover_ev(cover, outs_a, prizes_a)
        if best is None or v > best[0]:
            best = (v, cover, beta)
    outs_b = simulate_outcomes(probs, int(n_sims * 2.5), seed=23)
    prizes_b = prize_matrix(winners_by_outcome(public, outs_b, rec_cols), recaudacion)
    return {"picks": best[1], "ev": cover_ev(best[1], outs_b, prizes_b), "beta": best[2]}


def ticket_stats(probs: list[list[float]], cover: list[list[str]], price: float) -> dict:
    cov = [sum(p[SIGNS.index(s)] for s in c) for p, c in zip(probs, cover)]
    dist = poisson_binomial(cov)
    n_bets = int(np.prod([len(c) for c in cover]))
    return {"bets": n_bets, "cost": round(n_bets * price, 2), "p14": float(np.prod(cov)),
            "p10plus": float(sum(dist[10:])), "expected_hits": round(float(sum(cov)), 2),
            "dist": [round(x, 5) for x in dist]}


def build_tickets(rows: list[dict], c: dict, public: list | None = None, recaudacion: float | None = None) -> dict:
    probs = [r["p"] for r in rows[:14]]
    single = [[SIGNS[int(np.argmax(p))]] for p in probs]
    reduced = best_reduction(probs, int(c["max_bets"]))
    t = {"sencilla": {"picks": single, **ticket_stats(probs, single, c["price"])},
         "reducida": {"picks": reduced, **ticket_stats(probs, reduced, c["price"])}}
    if public and len(public) >= 14 and all(x for x in public[:14]):
        rec = recaudacion or DEFAULT_RECAUDACION
        outs = simulate_outcomes(probs, 150000, seed=23)
        prizes = prize_matrix(winners_by_outcome(public[:14], outs, rec / 0.75), rec)
        for k in ("sencilla", "reducida"):
            t[k]["ev"] = round(cover_ev(t[k]["picks"], outs, prizes), 3)
        v = best_value_ticket(probs, public[:14], int(c["max_bets"]), rec, start=reduced)
        t["valor"] = {"picks": v["picks"], **ticket_stats(probs, v["picks"], c["price"]), "ev": round(v["ev"], 3),
                      "beta": v["beta"]}
        for k in t:
            t[k]["roi"] = round(t[k]["ev"] / t[k]["cost"] - 1, 4) if t[k]["cost"] else None
    return t


def columns_by_hits(cover: list[list[str]], outcome: list[str | None]) -> dict[int, int]:
    """Nº de columnas del boleto con k aciertos (para aplicar la tabla de premios)."""
    poly = [1]
    for c, o in zip(cover, outcome):
        hit = 1 if o in c else 0
        miss = len(c) - hit
        new = [0] * (len(poly) + 1)
        for k, v in enumerate(poly):
            new[k] += v * miss
            new[k + 1] += v * hit
        poly = new
    return {k: v for k, v in enumerate(poly) if v}


def settle(j: dict) -> dict:
    """Aciertos y premios de una jornada terminada."""
    outcome = []
    for m in j["matches"][:14]:
        if m.get("hg") is None or m.get("ag") is None:
            outcome.append(None)
        else:
            outcome.append("1" if m["hg"] > m["ag"] else "X" if m["hg"] == m["ag"] else "2")
    pl = j["matches"][14] if len(j["matches"]) > 14 else None
    pleno_res = None
    if pl and pl.get("hg") is not None:
        pleno_res = [PLENO_CATS[min(pl["hg"], 3)], PLENO_CATS[min(pl["ag"], 3)]]
    prizes = j.get("prizes") or {}
    res = {"outcome": outcome, "pleno": pleno_res}
    for name, t in j["tickets"].items():
        cols = columns_by_hits(t["picks"], outcome)
        hits = max(cols)
        pleno_ok = pleno_res is not None and j.get("pleno") and pleno_res == j["pleno"]["pick"]
        won = 0.0
        for k, n in cols.items():
            if k >= 10:
                won += n * float(prizes.get(str(k), 0.0) or 0.0)
        if hits == 14 and pleno_ok:
            won += float(prizes.get("P15", 0.0) or 0.0)
        res[name] = {"hits": int(hits), "pleno_ok": bool(pleno_ok), "prize": round(won, 2), "cost": t["cost"],
                     "columns_by_hits": {str(k): int(v) for k, v in cols.items() if k >= 10}}
    return res


# ---------------------------------------------------------------------------
# Actualización
# ---------------------------------------------------------------------------
def _path(settings: Settings):
    return settings.predictions_dir / "quiniela.json"


def load(settings: Settings) -> dict:
    p = _path(settings)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"jornadas": {}}


def _started(page: dict) -> bool:
    return any(m["status"] in ("fin", "live") for m in page["matches"])


def _finished(page: dict) -> bool:
    return len(page["matches"]) >= 14 and all(m["status"] == "fin" for m in page["matches"][:15])


def update(settings: Settings, preds: list[dict] | None = None, online: bool = True,
           pages: list[dict] | None = None, seg: SegundaModel | None = None, public: dict | None = None,
           euro=None) -> dict:
    """Lee el boleto actual (y el siguiente/anteriores pendientes), rellena y liquida."""
    c = cfg(settings)
    state = load(settings)
    jor = state["jornadas"]
    if preds is None:
        lp = settings.predictions_dir / "latest.json"
        preds = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else []
    by_pair = {(p["home"], p["away"]): p for p in preds}
    laliga = set()
    for p in preds:
        laliga |= {p["home"], p["away"]}
    sp1 = settings.raw_dir / f"SP1_{results.season_code(results.current_season())}.csv"
    if sp1.exists():
        cur = pd.read_csv(sp1)
        laliga |= set(cur.home) | set(cur.away)
    if pages is None:
        pages = []
        if online:
            main = fetch_page()
            if main:
                pages.append(main)
                if _started(main) and main.get("next"):  # el siguiente boleto, si ya está publicado
                    nxt = fetch_page(main["next"])
                    if nxt and nxt["date"] != main["date"]:
                        pages.append(nxt)
            # jornadas guardadas aún sin liquidar
            for d, j in jor.items():
                if not j.get("settled") and all(p["date"] != d for p in pages) and d <= datetime.now().strftime("%Y-%m-%d"):
                    pg = fetch_page(d)
                    if pg:
                        pages.append(pg)
    if seg is None and pages:
        try:
            seg = SegundaModel(settings, online=online)
        except Exception as e:
            log.warning("Modelo de Segunda no disponible: %s", e)
    elo = _laliga_elo(settings)
    if public is None:
        public = fetch_public(online) if pages else {}
    if euro is None and pages:
        try:
            from . import eurelo
            tab = eurelo.load(settings)
            if online and (tab.empty or (pd.Timestamp.now() - pd.Timestamp(
                    (settings.processed_dir / "euro_elo.csv").stat().st_mtime, unit="s")).total_seconds() > 20 * 3600):
                tab = eurelo.build(settings, online=True)
            euro = eurelo.Lookup(tab) if len(tab) else None
        except Exception as e:
            log.warning("Elo europeo no disponible: %s", e)
    recs = [j["recaudacion"] for j in jor.values() if j.get("recaudacion")]
    rec_est = float(np.median(recs[-6:])) if recs else DEFAULT_RECAUDACION
    now = datetime.now(timezone.utc).isoformat(timespec="minutes")
    for pg in pages:
        d = pg["date"]
        if not d:
            continue
        j = jor.get(d, {"date": d, "frozen": False, "settled": False})
        j["jornada"] = pg.get("jornada") or j.get("jornada")
        if pg.get("prizes"):
            j["prizes"] = pg["prizes"]
        if pg.get("recaudacion"):
            j["recaudacion"] = pg["recaudacion"]
        pub = public.get(j["jornada"]) if j.get("jornada") else None
        if not j.get("frozen") and _started(pg) and j.get("tickets"):
            j["frozen"], j["frozen_at"] = True, now  # empezó la jornada: el boleto ya no cambia
        if not j.get("frozen"):
            rows = []
            euro_j = is_european(pg["matches"], laliga, seg.teams if seg else set())
            real = (pub or {}).get("real") or []
            for k, m in enumerate(pg["matches"][:15]):
                mk = real[k] if k < len(real) else None
                mp = match_probs(m["home"], m["away"], by_pair, seg, laliga, elo, euro_j, market=mk, euro=euro)
                rows.append({"n": m["n"], "home": m["home"], "away": m["away"], "when": m.get("when"),
                             "p": [round(float(x), 4) for x in mp["p"]], "source": mp["source"],
                             "lam": round(float(mp["lam"]), 3), "mu": round(float(mp["mu"]), 3)})
            if len(rows) >= 14:
                if "rows" not in j and _started(pg):
                    j["late"] = True  # se rellenó con la jornada ya empezada: no cuenta en el balance
                lae = (pub or {}).get("lae") or []
                for k, r in enumerate(rows[:14]):
                    r["public"] = [round(x, 3) for x in lae[k]] if k < len(lae) and lae[k] else None
                j["rows"] = rows
                j["recaudacion_est"] = j.get("recaudacion") or rec_est
                j["tickets"] = build_tickets(rows, c, lae if len(lae) >= 14 and all(lae[:14]) else None,
                                             j["recaudacion_est"])
                if len(rows) >= 15:
                    j["pleno"] = pleno_probs(rows[14]["lam"], rows[14]["mu"])
                j["updated"] = now
            if _started(pg) and j.get("tickets"):
                j["frozen"] = True
                j["frozen_at"] = now
        # resultados (siempre)
        j["matches"] = [{"n": m["n"], "home": m["home"], "away": m["away"], "hg": m["hg"], "ag": m["ag"],
                         "status": m["status"], "when": m.get("when")} for m in pg["matches"][:15]]
        if _finished(pg) and j.get("tickets") and not j.get("settled"):
            j["result"] = settle(j)
            j["settled"] = bool(pg.get("prizes"))  # se reintenta hasta que publiquen los premios
        jor[d] = j
    state["jornadas"] = dict(sorted(jor.items()))
    state["summary"] = summarize(state)
    state["config"] = c
    state["updated"] = now
    _path(settings).write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return state


def summarize(state: dict) -> dict:
    out = {}
    played = [j for j in state["jornadas"].values() if j.get("result") and not j.get("late")]
    names = ["sencilla", "reducida"] + (["valor"] if any("valor" in j["result"] for j in played) else [])
    for name in names:
        rs = [j["result"][name] for j in played if name in j["result"]]
        cost = sum(r["cost"] for r in rs)
        prize = sum(r["prize"] for r in rs)
        hits = [r["hits"] for r in rs]
        out[name] = {"jornadas": len(rs), "cost": round(cost, 2), "prize": round(prize, 2),
                     "balance": round(prize - cost, 2), "avg_hits": round(float(np.mean(hits)), 2) if hits else None,
                     "best": int(max(hits)) if hits else None, "prized": int(sum(h >= 10 for h in hits)),
                     "curve": [], "hist": {str(k): int(sum(h == k for h in hits)) for k in range(15)}}
        acc = 0.0
        for j in played:
            r = j["result"].get(name)
            if r:
                acc += r["prize"] - r["cost"]
                out[name]["curve"].append([j["date"], round(acc, 2)])
    return out


def format_ticket(j: dict) -> str:
    if not j or not j.get("tickets"):
        return ""
    t = j["tickets"]["sencilla"]
    lines = [f"Quiniela jornada {j.get('jornada') or ''} ({j['date']})"]
    for r, pk, rd in zip(j["rows"][:14], t["picks"], j["tickets"]["reducida"]["picks"]):
        lines.append(f"{r['n']:>2}. {r['home']} – {r['away']}: {pk[0]}  (reducida {''.join(rd)})")
    if j.get("pleno"):
        lines.append(f"15. {j['rows'][14]['home']} – {j['rows'][14]['away']}: {'-'.join(j['pleno']['pick'])}")
    lines.append(f"Prob. de 14: sencilla {100 * t['p14']:.4f} % · reducida {100 * j['tickets']['reducida']['p14']:.3f} %")
    return "\n".join(lines)

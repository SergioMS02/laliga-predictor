"""Cliente HTTP compartido con reintentos, timeouts cortos y 'circuit breaker' por host
(si un host falla repetidamente, se deja de intentar durante esta ejecución)."""
from __future__ import annotations

import logging
import time
from collections import Counter
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (X11; Linux x86_64) laliga-predictor/0.1 (+uso personal, no comercial)"
_session: requests.Session | None = None
_fails: Counter = Counter()
MAX_HOST_FAILS = 2


class HostUnavailable(requests.ConnectionError):
    pass


def session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": _UA, "Accept-Language": "es-ES,es;q=0.9,en;q=0.8"})
    return _session


def reset_breaker() -> None:
    _fails.clear()


def _check(url: str) -> str:
    host = urlparse(url).netloc
    if _fails[host] >= MAX_HOST_FAILS:
        raise HostUnavailable(f"{host} no disponible en esta ejecución")
    return host


def get(url: str, params: dict | None = None, timeout: float = 20, retries: int = 2,
        headers: dict | None = None) -> requests.Response:
    host = _check(url)
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = session().get(url, params=params, timeout=timeout, headers=headers)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            _fails[host] = 0
            return r
        except requests.HTTPError as e:
            last = e
            code = e.response.status_code if e.response is not None else 0
            if code and code < 500 and code != 429:
                break  # 4xx: reintentar no sirve
        except requests.RequestException as e:
            last = e
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    if not isinstance(last, requests.HTTPError) or (last.response is not None and last.response.status_code >= 500):
        _fails[host] += 1
    assert last is not None
    raise last


def post_json(url: str, payload: dict, headers: dict | None = None, timeout: float = 60) -> dict:
    host = _check(url)
    try:
        r = session().post(url, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        _fails[host] += 1
        raise

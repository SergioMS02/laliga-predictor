from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("LALIGA_ROOT", Path(__file__).resolve().parents[2]))


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class Settings:
    raw: dict[str, Any]
    root: Path = field(default_factory=lambda: ROOT)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.raw.get(section, {}).get(key, default)

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def state_dir(self) -> Path:
        return self.data_dir / "state"

    @property
    def predictions_dir(self) -> Path:
        return self.root / "predictions"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.state_dir, self.predictions_dir):
            d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def env(name: str) -> str | None:
        v = os.environ.get(name)
        return v if v else None


def _deep_update(raw: dict, upd: dict) -> dict:
    raw = copy.deepcopy(raw)
    for sec, vals in upd.items():
        if isinstance(vals, dict):
            raw.setdefault(sec, {}).update(vals)
    return raw


def load_settings(path: str | Path | None = None, root: Path | None = None,
                  overrides: dict | None = None, use_tuned: bool = True) -> Settings:
    root = Path(root) if root else ROOT
    _load_dotenv(root / ".env")
    cfg_path = Path(path) if path else root / "config.yaml"
    if not cfg_path.exists():
        cfg_path = ROOT / "config.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    tuned = root / "data" / "state" / "tuned.json"
    if use_tuned and raw.get("use_tuned", True) and tuned.exists():
        t = json.loads(tuned.read_text(encoding="utf-8"))
        raw = _deep_update(raw, {k: t[k] for k in ("dixon_coles", "stacker", "stacker_market", "availability") if k in t})
    if overrides:
        raw = _deep_update(raw, overrides)
    s = Settings(raw=raw, root=root)
    s.ensure_dirs()
    return s

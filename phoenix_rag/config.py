"""Load config/config.yaml + config/seeds.yaml and expose them as simple objects."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


@dataclass
class IncludeRule:
    prefix: str
    section_type: str
    priority: int = 3
    exact: bool = False
    note: str = ""


@dataclass
class ExcludeRule:
    prefix: str
    reason: str = ""


@dataclass
class Seeds:
    include: list[IncludeRule] = field(default_factory=list)
    exclude: list[ExcludeRule] = field(default_factory=list)
    section_type_rules: dict[str, list[str]] = field(default_factory=dict)


class Config:
    """Thin wrapper: attribute access to the YAML tree plus resolved paths."""

    def __init__(self, raw: dict[str, Any], seeds: Seeds):
        self.raw = raw
        self.seeds = seeds
        self.root = ROOT
        self.data_dir = ROOT / raw["project"].get("data_dir", "data")
        self.logs_dir = ROOT / raw["project"].get("logs_dir", "logs")
        self.raw_dir = self.data_dir / "raw"
        self.clean_dir = self.data_dir / "clean"
        self.chunks_path = self.data_dir / "chunks.jsonl"
        self.bm25_path = self.data_dir / "bm25.pkl"
        self.chroma_dir = ROOT / raw.get("vector_store", {}).get("path", "data/chroma")
        self.db_path = self.data_dir / "pages.sqlite"

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def crawl(self) -> dict[str, Any]:
        return self.raw["crawl"]

    @property
    def cleaning(self) -> dict[str, Any]:
        return self.raw["cleaning"]

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.raw_dir, self.clean_dir, self.logs_dir, self.chroma_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def chunking(self) -> dict[str, Any]:
        return self.raw["chunking"]

    @property
    def embedding(self) -> dict[str, Any]:
        return self.raw["embedding"]

    @property
    def vector_store(self) -> dict[str, Any]:
        return self.raw["vector_store"]

    @property
    def retrieval(self) -> dict[str, Any]:
        return self.raw["retrieval"]

    @property
    def llm(self) -> dict[str, Any]:
        return self.raw["llm"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw.get("evaluation", {})

    @property
    def messages(self) -> dict[str, Any]:
        return self.raw.get("messages", {})


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_seeds(path: Path | None = None) -> Seeds:
    raw = _load_yaml(path or CONFIG_DIR / "seeds.yaml")
    include = [IncludeRule(**r) for r in raw.get("include", [])]
    exclude = [ExcludeRule(**r) for r in raw.get("exclude", [])]
    # Longer prefixes first so the most specific rule wins.
    include.sort(key=lambda r: (-len(r.prefix), r.priority))
    return Seeds(include=include, exclude=exclude, section_type_rules=raw.get("section_type_rules", {}))


def load_config(config_path: Path | None = None, seeds_path: Path | None = None) -> Config:
    load_dotenv(ROOT / ".env")
    raw = _load_yaml(config_path or CONFIG_DIR / "config.yaml")
    # Allow a few env overrides without editing YAML.
    if os.getenv("LLM_MODEL"):
        raw.setdefault("llm", {})["model"] = os.environ["LLM_MODEL"]
    if os.getenv("EMBEDDING_MODEL"):
        raw.setdefault("embedding", {})["model"] = os.environ["EMBEDDING_MODEL"]
    return Config(raw, load_seeds(seeds_path))

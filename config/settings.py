from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip("\"").strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    data_dir: Path
    output_dir: Path
    table_dir: Path
    fig_dir: Path
    gateway_price_quantile: float
    mentorship_price_quantile: float
    groq_api_key: str | None
    groq_model: str
    max_llm_rows: int
    llm_batch_size: int
    llm_batch_sleep_seconds: int


def get_settings() -> Settings:
    base_dir = Path(__file__).resolve().parents[1]
    _load_env(base_dir / ".env")
    output_dir = base_dir / "output"
    table_dir = output_dir / "tables"
    fig_dir = output_dir / "figures"

    return Settings(
        base_dir=base_dir,
        data_dir=base_dir / "db",
        output_dir=output_dir,
        table_dir=table_dir,
        fig_dir=fig_dir,
        gateway_price_quantile=0.25,
        mentorship_price_quantile=0.75,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        max_llm_rows=int(os.getenv("MAX_LLM_ROWS", "200")),
        llm_batch_size=int(os.getenv("LLM_BATCH_SIZE", "50")),
        llm_batch_sleep_seconds=int(os.getenv("LLM_BATCH_SLEEP_SECONDS", "30")),
    )

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dirs(table_dir: Path, fig_dir: Path) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def fmt_num(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value)}"
    return f"{value:,.2f}" if isinstance(value, float) else str(value)


def fmt_int(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{int(value)}"


def safe_label(primary: object, fallback: object, default: str = "Unknown") -> str:
    def _clean(val: object) -> str | None:
        if not isinstance(val, str):
            return None
        # Normalize common messy whitespace/encoding artifacts from exports.
        # - NBSP (\u00a0) -> space
        # - "Â" (\u00c2) can appear when NBSP is mis-decoded in upstream exports
        val = val.replace("\u00c2", "").replace("\u00a0", " ").strip()
        val = " ".join(val.split())
        if not val or val.lower() in {"nan", "none"}:
            return None
        return val

    primary_clean = _clean(primary)
    if primary_clean:
        return primary_clean
    fallback_clean = _clean(fallback)
    if fallback_clean:
        return fallback_clean
    return default

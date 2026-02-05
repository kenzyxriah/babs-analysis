from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd
from typing import Dict

from analytics.config.settings import Settings


@dataclass
class Context:
    settings: Settings
    data: Dict[str, pd.DataFrame]
    results: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def add_result(self, name: str, df: pd.DataFrame) -> None:
        self.results[name] = df

    def get(self, name: str) -> pd.DataFrame:
        return self.results[name]

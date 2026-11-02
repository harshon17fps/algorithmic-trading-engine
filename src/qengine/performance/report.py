"""Exportable analytics report (JSON, Markdown and CSV artefacts)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TypeAlias

import pandas as pd

MetricValue: TypeAlias = float | int | str | None
MetricSections: TypeAlias = Mapping[str, Mapping[str, MetricValue]]

_PERCENT_KEYS = {
    "total_return", "cagr", "annual_volatility", "max_drawdown", "current_drawdown",
    "win_rate", "alpha", "tracking_error", "avg_risk_free_rate", "benchmark_total_return",
    "benchmark_cagr",
}


def _json_safe(value: MetricValue) -> MetricValue:
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def _fmt(key: str, value: MetricValue) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        if math.isinf(value):
            return "∞" if value > 0 else "-∞"
        if key in _PERCENT_KEYS:
            return f"{value:.2%}"
        return f"{value:,.2f}" if abs(value) >= 1000 else f"{value:.4f}"
    return str(value)


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    metrics: MetricSections
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    fills: pd.DataFrame

    def flat(self) -> dict[str, MetricValue]:
        return {k: v for section in self.metrics.values() for k, v in section.items()}

    def __getitem__(self, key: str) -> MetricValue:
        return self.flat()[key]

    def to_markdown(self, title: str = "Backtest Analytics Report") -> str:
        lines = [f"# {title}", ""]
        for section, values in self.metrics.items():
            lines += [f"## {section}", "", "| Metric | Value |", "|---|---:|"]
            lines += [f"| {k.replace('_', ' ')} | {_fmt(k, v)} |" for k, v in values.items()]
            lines.append("")
        return "\n".join(lines)

    def export(self, directory: str | Path) -> dict[str, Path]:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "json": out / "report.json",
            "markdown": out / "report.md",
            "equity_curve": out / "equity_curve.csv",
            "trades": out / "trades.csv",
            "fills": out / "fills.csv",
        }
        safe = {s: {k: _json_safe(v) for k, v in vals.items()} for s, vals in self.metrics.items()}
        paths["json"].write_text(json.dumps(safe, indent=2, default=str))
        paths["markdown"].write_text(self.to_markdown())
        self.equity_curve.to_csv(paths["equity_curve"])
        self.trades.to_csv(paths["trades"], index=False)
        self.fills.to_csv(paths["fills"], index=False)
        return paths

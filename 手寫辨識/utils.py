"""Utility functions for paths and metrics report writing."""

from __future__ import annotations

from pathlib import Path


def ensure_dirs() -> None:
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)


def write_accuracy_report(content: str, output_path: str = "results/accuracy.txt") -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

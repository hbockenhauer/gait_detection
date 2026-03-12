"""Shared helpers for saving plots into the centralized outputs directory."""

from pathlib import Path


def ensure_plot_dir(path: str | Path) -> Path:
    plot_dir = Path(path)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir

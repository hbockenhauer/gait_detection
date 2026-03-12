"""Shared helpers for lightweight data-loading tasks."""

from pathlib import Path
from typing import Iterable


def list_files(path: str | Path, suffixes: Iterable[str]) -> list[Path]:
    root = Path(path)
    suffix_set = {suffix.lower() for suffix in suffixes}
    return sorted(
        child for child in root.iterdir()
        if child.is_file() and child.suffix.lower() in suffix_set
    )

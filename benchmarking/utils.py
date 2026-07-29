from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """
    Set the random seed for Python, NumPy and PyTorch.

    Use this before every evaluation episode so that repeated runs are easier
    to reproduce.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_directory(path: str | Path) -> Path:
    """Create an output directory and return it as a Path object."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def first_available(
    sources: Iterable[Any],
    possible_names: Iterable[str],
    default: Any = None,
) -> Any:
    """
    Search dictionaries and object attributes for the first available value.

    This is useful because one environment may store position as
    `state.position`, while another may return it as `info["position"]`.
    """
    for source in sources:
        if source is None:
            continue

        for name in possible_names:
            if isinstance(source, dict) and name in source:
                value = source[name]
                if value is not None:
                    return value

            if hasattr(source, name):
                value = getattr(source, name)
                if value is not None:
                    return value

    return default


def to_builtin(value: Any) -> Any:
    """Convert NumPy/PyTorch values into JSON-compatible Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if torch.is_tensor(value):
        return value.detach().cpu().tolist()

    if isinstance(value, dict):
        return {
            str(key): to_builtin(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]

    return str(value)


def observation_to_json(observation: np.ndarray) -> str:
    """Store an observation vector in one compact CSV cell."""
    return json.dumps(
        np.asarray(observation, dtype=np.float32).tolist(),
        separators=(",", ":"),
    )


def write_json(path: str | Path, data: dict) -> None:
    """Write a readable JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            to_builtin(data),
            file,
            indent=2,
            ensure_ascii=False,
        )

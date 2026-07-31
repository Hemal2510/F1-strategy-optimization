from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def to_builtin(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if is_dataclass(value):
        return {k: to_builtin(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_builtin(v) for v in value]
    return str(value)


def json_dumps(value: Any) -> str:
    return json.dumps(to_builtin(value), ensure_ascii=False, separators=(",", ":"))


def observation_to_json(observation: Any) -> str:
    """
    Serialize an observation (numpy array, tensor, list, etc.) to a compact
    JSON string, for storage in a CSV cell (e.g. lap_trace.csv).
    """
    return json_dumps(observation)


def first_present(sources: Iterable[Any], names: Iterable[str], default: Any = None) -> Any:
    for source in sources:
        if source is None:
            continue
        for name in names:
            if isinstance(source, Mapping) and name in source:
                value = source[name]
                if value is not None:
                    return value
            elif hasattr(source, name):
                value = getattr(source, name)
                if value is not None:
                    return value
    return default



def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target



def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(to_builtin(payload), handle, indent=2, ensure_ascii=False)

from __future__ import annotations

import importlib
import inspect
import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping

import numpy as np
import torch


def import_object(path: str) -> Any:
    """Import ``package.module:object`` or ``package.module.object``."""
    if ":" in path:
        module_name, object_name = path.split(":", 1)
    else:
        module_name, object_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def call_with_supported_kwargs(callable_obj: Callable[..., Any], kwargs: Mapping[str, Any]) -> Any:
    signature = inspect.signature(callable_obj)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return callable_obj(**dict(kwargs))
    supported = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return callable_obj(**supported)


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


def first_available(sources: Iterable[Any], names: Iterable[str], default: Any = None) -> Any:
    """Alias for first_present, kept because evaluator.py imports this name."""
    return first_present(sources, names, default)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def ensure_directory(path: str | Path) -> Path:
    """Alias for ensure_dir, kept because evaluator.py imports this name."""
    return ensure_dir(path)


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(to_builtin(payload), handle, indent=2, ensure_ascii=False)

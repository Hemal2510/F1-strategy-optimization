from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    """
    Load and validate the benchmark JSON configuration.

    A normal dictionary is used here because it is easier for a beginner to
    inspect than a larger hierarchy of dataclasses.
    """
    config_path = Path(path).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix.lower() != ".json":
        raise ValueError("This beginner version expects a .json file.")

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    required_fields = [
        "project_name",
        "output_dir",
        "device",
        "seeds",
        "environment",
        "agents",
        "benchmark",
    ]

    for field_name in required_fields:
        if field_name not in config:
            raise ValueError(
                f"Missing required top-level field: '{field_name}'"
            )

    if not isinstance(config["seeds"], list) or not config["seeds"]:
        raise ValueError("'seeds' must be a non-empty list.")

    for required_agent in ("dqn", "qrl"):
        if required_agent not in config["agents"]:
            raise ValueError(
                f"Missing agent configuration: '{required_agent}'"
            )

    # JSON object keys are strings. Convert action IDs back to integers.
    action_names = config["benchmark"].get("action_names", {})
    config["benchmark"]["action_names"] = {
        int(action_id): str(action_name)
        for action_id, action_name in action_names.items()
    }

    config["_config_path"] = str(config_path)
    return config

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


DEFAULT_SEARCH_ROOT = "artifacts"

AGENT_COLORS = {
    "dqn": "#4C78A8",
    "qrl": "#F58518",
    "random": "#72B7B2",
    "rule_based": "#54A24B",
    "real_team": "#B279A2",
}

METRIC_LABELS = {
    "total_reward": "Total reward",
    "final_position": "Final position",
    "position_gain": "Position gain",
    "total_race_time": "Total race time (s)",
    "mean_lap_time": "Mean lap time (s)",
    "mean_lap_delta": "Mean lap delta (s)",
    "pit_count": "Pit count",
    "invalid_action_rate": "Invalid action rate",
    "mean_inference_ms": "Mean inference time (ms)",
    "p95_inference_ms": "P95 inference time (ms)",
}


def agent_color(agent: str) -> str:
    return AGENT_COLORS.get(str(agent).lower(), "#999999")


def find_benchmark_runs(root: str = DEFAULT_SEARCH_ROOT) -> list[Path]:
    """Locate every benchmark output directory under `root` (anything containing manifest.json)."""
    root_path = Path(root)
    if not root_path.exists():
        return []
    found = {p.parent for p in root_path.rglob("manifest.json")}
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_benchmark_config(config_path: str) -> dict:
    """
    Read a benchmark.json (the same file passed to `run_benchmark.py --config`)
    so the dashboard can point straight at its `output_dir` instead of the
    user having to retype/guess the path.
    """
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def output_dir_from_config(config_path: str) -> str:
    """
    Return the `output_dir` declared in benchmark.json, resolved relative to
    the config file's own location (matching how a person would normally run
    `python -m benchmark.run_benchmark --config benchmark.json` from the
    project root: `output_dir` is a plain relative path in that same JSON).
    """
    config = load_benchmark_config(config_path)
    output_dir = config.get("output_dir", "")
    if not output_dir:
        return ""
    candidate = Path(output_dir)
    if candidate.is_absolute():
        return str(candidate)
    # Prefer the path relative to the config file's folder; fall back to
    # relative-to-cwd if that's where it actually exists (matches ensure_dir's
    # behaviour in utils.py, which never resolves to an absolute path).
    beside_config = path.parent / output_dir if (path := Path(config_path)).exists() else None
    if beside_config and beside_config.exists():
        return str(beside_config)
    return output_dir


def column_completeness(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Percentage of non-null values per column. Useful for spotting when the
    real F1StrategyEnv doesn't expose the attribute names
    `evaluator.snapshot_environment` guesses (e.g. `position`, `final_position`),
    which silently leaves those columns empty rather than raising an error.
    """
    rows = []
    for col in columns:
        if col not in df.columns:
            rows.append({"column": col, "present": False, "non_null_pct": 0.0})
            continue
        non_null_pct = float(df[col].notna().mean() * 100) if len(df) else 0.0
        rows.append({"column": col, "present": True, "non_null_pct": round(non_null_pct, 1)})
    return pd.DataFrame(rows)



@st.cache_data(show_spinner=False)
def load_manifest(output_dir: str) -> dict:
    path = Path(output_dir) / "manifest.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_episode_summary(output_dir: str) -> pd.DataFrame:
    return _read_csv(Path(output_dir) / "episode_summary.csv")


@st.cache_data(show_spinner=False)
def load_lap_trace(output_dir: str) -> pd.DataFrame:
    return _read_csv(Path(output_dir) / "lap_trace.csv")


@st.cache_data(show_spinner=False)
def load_overall_metrics(output_dir: str) -> pd.DataFrame:
    return _read_csv(Path(output_dir) / "overall_metrics.csv")


@st.cache_data(show_spinner=False)
def load_track_metrics(output_dir: str) -> pd.DataFrame:
    return _read_csv(Path(output_dir) / "track_metrics.csv")


@st.cache_data(show_spinner=False)
def load_paired_comparison(output_dir: str) -> pd.DataFrame:
    return _read_csv(Path(output_dir) / "paired_comparison.csv")


@st.cache_data(show_spinner=False)
def load_action_disagreement(output_dir: str) -> pd.DataFrame:
    return _read_csv(Path(output_dir) / "action_disagreement.csv")


@st.cache_data(show_spinner=False)
def load_errors(output_dir: str) -> pd.DataFrame:
    return _read_csv(Path(output_dir) / "errors.csv")

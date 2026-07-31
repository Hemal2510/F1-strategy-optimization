from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


# 6 agents with contrasting colors
AGENT_COLORS = {
    "dqn":                  "#E63946",   # Red
    "qrl":                  "#1D7FD4",   # Blue
    "random":               "#9E9E9E",   # Grey
    "always_stay_out":      "#FF9800",   # Amber
    "rule_aware_heuristic": "#4CAF50",   # Green
    "real_driver":          "#9C27B0",   # Purple
}

AGENT_DISPLAY_NAMES = {
    "dqn":                  "DQN",
    "qrl":                  "QRL",
    "random":               "Random",
    "always_stay_out":      "Always Stay Out",
    "rule_aware_heuristic": "Rule Heuristic",
    "real_driver":          "Real Driver",
}

METRIC_LABELS = {
    "total_reward":       "Total Reward",
    "final_position":     "Final Position",
    "position_gain":      "Positions Gained",
    "pit_count":          "Pit Stops",
    "mean_lap_time":      "Mean Lap Time (s)",
}


def agent_color(agent: str) -> str:
    return AGENT_COLORS.get(str(agent).lower(), "#999999")


def agent_label(agent: str) -> str:
    return AGENT_DISPLAY_NAMES.get(str(agent).lower(), agent.upper())


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

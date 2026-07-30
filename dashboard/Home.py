from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow importing benchmarking modules from root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader import AGENT_COLORS, AGENT_DISPLAY_NAMES, METRIC_LABELS, agent_color, agent_label

st.set_page_config(
    page_title="F1 Dashboard",
    page_icon="🏎️",
    layout="wide",
)

# Sidebar filters & controls
st.sidebar.title("Benchmark Controls")

selected_tracks = st.sidebar.multiselect(
    "Tracks",
    options=["Monaco", "Monza", "Silverstone"],
    default=["Monaco", "Monza", "Silverstone"],
)

selected_years = st.sidebar.multiselect(
    "Years",
    options=[2022, 2023, 2024, 2025],
    default=[2022, 2023, 2024, 2025],
)

n_seeds = st.sidebar.slider("No. of Seeds (races per track/year)", min_value=1, max_value=30, value=10)

st.sidebar.markdown("**Agents**")
agent_options = {
    "dqn":                  "🔴 DQN",
    "qrl":                  "🔵 QRL",
    "random":               "⚫ Random",
    "always_stay_out":      "🟠 Always Stay Out",
    "rule_aware_heuristic": "🟢 Rule Heuristic",
    "real_driver":          "🟣 Real Driver",
}
selected_agents = [
    name for name, label in agent_options.items()
    if st.sidebar.checkbox(label, value=(name in ("dqn", "qrl")), key=f"agent_{name}")
]

st.sidebar.divider()
run_clicked = st.sidebar.button("▶ Run Benchmark", type="primary", width="stretch")
load_clicked = st.sidebar.button("📂 Load Latest Results", width="stretch")

# Load saved benchmark results if available
if load_clicked:
    _latest = Path(__file__).resolve().parent.parent / "benchmarking" / "artifacts" / "latest"
    if not _latest.exists() or not (_latest / "manifest.json").exists():
        st.sidebar.error("No saved results found at benchmarking/artifacts/latest/")
    else:
        def _safe_csv(p: Path) -> pd.DataFrame:
            if not p.exists():
                return pd.DataFrame()
            try:
                return pd.read_csv(p)
            except Exception:
                return pd.DataFrame()

        with (_latest / "manifest.json").open(encoding="utf-8") as _f:
            _manifest = json.load(_f)

        _episode_df  = _safe_csv(_latest / "episode_summary.csv")
        _lap_df      = _safe_csv(_latest / "lap_trace.csv")
        _overall     = _safe_csv(_latest / "overall_metrics.csv")
        _track       = _safe_csv(_latest / "track_metrics.csv")
        _paired      = _safe_csv(_latest / "paired_comparison.csv")
        _disagree    = _safe_csv(_latest / "action_disagreement.csv")
        _errors_df   = _safe_csv(_latest / "errors.csv")

        st.session_state["results"] = {
            "episode_df":   _episode_df,
            "lap_df":       _lap_df,
            "overall":      _overall,
            "by_track":     _track,
            "paired":       _paired,
            "disagreement": _disagree,
            "errors":       _errors_df,
            "manifest":     _manifest,
        }
        st.session_state["filters"] = {
            "tracks":  _manifest.get("tracks", []),
            "years":   _manifest.get("years", []),
            "n_seeds": len(_manifest.get("seeds", [])),
            "agents":  _manifest.get("agents", []),
        }
        st.sidebar.success(f"✅ Loaded {_manifest.get('episode_count', len(_episode_df))} episodes from latest run.")

# Dashboard title and main page layout
st.title("F1 Pit-Stop Strategy — Agent Benchmark")
st.caption("SoC 2026 · Quantum Reinforcement Learning for pit strategy optimisation . QC-3")

# Validation and run
if run_clicked:
    errors_pre = []
    if not selected_tracks:
        errors_pre.append("Select at least one track.")
    if not selected_years:
        errors_pre.append("Select at least one year.")
    if not selected_agents:
        errors_pre.append("Select at least one agent.")
    if errors_pre:
        for e in errors_pre:
            st.error(e)
        st.stop()

    total = len(selected_agents) * n_seeds * len(selected_tracks) * len(selected_years)
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def on_progress(current: int, total_eps: int, msg: str):
        progress_bar.progress(current / total_eps)
        status_text.caption(f"Running {current}/{total_eps} — {msg}")

    with st.spinner(f"Running {total} episodes across {len(selected_agents)} agents…"):
        try:
            from benchmarking.runner import run_benchmark
            results = run_benchmark(
                tracks=selected_tracks,
                years=selected_years,
                n_seeds=n_seeds,
                agents=selected_agents,
                progress_callback=on_progress,
            )
            st.session_state["results"] = results
            st.session_state["filters"] = {
                "tracks": selected_tracks,
                "years": selected_years,
                "n_seeds": n_seeds,
                "agents": selected_agents,
            }
            progress_bar.progress(1.0)
            status_text.success(f"✅ Benchmark complete — {len(results['episode_df'])} episodes recorded.")
        except Exception as exc:
            st.error(f"Benchmark failed: {exc}")
            st.stop()

# ---------------------------------------------------------------------------
# Display results if available
# ---------------------------------------------------------------------------
results = st.session_state.get("results")

if results is None:
    st.info(
        "Configure your benchmark in the sidebar and click **▶ Run Benchmark** to start.\n\n"
        "- Pick any combination of tracks, years, seeds, and agents.\n"
        "- Results appear here and on the **Overview**, **Lap Replay**, and **Strategy Divergence** pages."
    )
    st.stop()

episode_df: pd.DataFrame = results["episode_df"]
manifest: dict = results["manifest"]
overall_df: pd.DataFrame = results.get("overall", pd.DataFrame())

# KPI tiles
if manifest:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Agents run", len(manifest.get("agents", [])))
    c2.metric("Seeds", manifest.get("seeds", []) and len(manifest["seeds"]))
    c3.metric("Episodes", manifest.get("episode_count", len(episode_df)))
    c4.metric("Errors", manifest.get("error_count", 0))

st.divider()

# Headline metrics per agent
if not overall_df.empty:
    st.subheader("Headline Results")
    headline = ["total_reward", "final_position", "pit_count"]
    cols = st.columns(len(headline))
    for col, metric in zip(cols, headline):
        with col:
            st.markdown(f"**{METRIC_LABELS.get(metric, metric)}**")
            sub = overall_df[overall_df["metric"] == metric].sort_values("agent")
            for _, row in sub.iterrows():
                st.metric(
                    label=agent_label(row["agent"]),
                    value=f"{row['mean']:.2f}",
                    help=f"std={row['std']:.2f}, n={int(row['count'])}",
                )

st.divider()
st.markdown(
    "### Explore results using the pages in the sidebar\n"
    "- **📊 Overview** — full metrics comparison with confidence intervals\n"
    "- **🔁 Lap Replay** — scrub through any race lap-by-lap\n"
    "- **🔀 Strategy Divergence** — where DQN and QRL disagree"
)

# Raw data in expander
with st.expander("📂 Raw episode data", expanded=False):
    if episode_df.empty:
        st.info("No episode data yet.")
    else:
        st.dataframe(episode_df, width="stretch", height=350)

errors_df: pd.DataFrame = results.get("errors", pd.DataFrame())
if not errors_df.empty:
    with st.expander(f"⚠️ {len(errors_df)} episodes failed", expanded=False):
        st.dataframe(errors_df, width="stretch")

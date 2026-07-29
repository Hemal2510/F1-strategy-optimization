from __future__ import annotations

from pathlib import Path

import streamlit as st

from data_loader import (
    column_completeness,
    find_benchmark_runs,
    load_episode_summary,
    load_errors,
    load_lap_trace,
    load_manifest,
    output_dir_from_config,
)

st.set_page_config(
    page_title="F1 DQN vs QRL Benchmark",
    page_icon="\U0001F3CE\uFE0F",
    layout="wide",
)

st.title("\U0001F3CE\uFE0F F1 Pit-Stop Strategy — DQN vs QRL Benchmark")
st.caption("SoC 2026 · Quantum Reinforcement Learning for pit strategy optimisation")

# ---------------------------------------------------------------------------
# Locate a benchmark run (a folder containing episode_summary.csv, lap_trace.csv,
# manifest.json, etc. — i.e. the `output_dir` from benchmark.json)
# ---------------------------------------------------------------------------
st.sidebar.header("Benchmark run")

config_path = st.sidebar.text_input(
    "Path to benchmark.json",
    value="",
    help=(
        "Point this at the same config file you pass to "
        "`run_benchmark.py --config`. The dashboard will read its `output_dir` "
        "field directly, so there's nothing to retype."
    ),
)

search_root = st.sidebar.text_input(
    "...or a folder to search for runs",
    value="artifacts",
    help="Parent folder that may contain one or more benchmark output_dir folders.",
)

runs = find_benchmark_runs(search_root)
manual_path = st.sidebar.text_input(
    "...or paste an output_dir path directly",
    value="",
)

config_derived_dir = output_dir_from_config(config_path) if config_path else ""

if manual_path:
    output_dir = manual_path
elif config_derived_dir:
    output_dir = config_derived_dir
    st.sidebar.caption(f"Using output_dir from config: `{output_dir}`")
elif runs:
    labels = [str(r) for r in runs]
    output_dir = st.sidebar.selectbox("Detected runs (newest first)", labels)
else:
    output_dir = ""
    st.sidebar.warning("No benchmark runs found yet.")

if config_path and not config_derived_dir:
    st.sidebar.error(
        "Couldn't find that benchmark.json, or it has no `output_dir` field."
    )

if not output_dir or not Path(output_dir).exists():
    st.info(
        "Point the sidebar at your results — either the path to `benchmark.json`, "
        "a folder to search, or the `output_dir` itself (the folder with "
        "`episode_summary.csv`, `lap_trace.csv`, `manifest.json`, etc.).\n\n"
        "No real results yet? Run `python make_sample_data.py` to generate a synthetic "
        "run at `artifacts/sample_run` so you can explore the dashboard right away."
    )
    st.stop()

st.session_state["output_dir"] = output_dir

manifest = load_manifest(output_dir)
episode_df = load_episode_summary(output_dir)
errors_df = load_errors(output_dir)

if manifest:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Experiment", manifest.get("project_name", "—"))
    col2.metric("Agents", ", ".join(manifest.get("agents", [])) or "—")
    col3.metric("Seeds evaluated", len(manifest.get("seeds", [])))
    col4.metric("Episodes logged", manifest.get("episode_count", len(episode_df)))

st.divider()

if episode_df.empty:
    st.warning("`episode_summary.csv` is empty or missing in this output_dir.")
else:
    st.subheader("Episode summary (raw)")
    st.dataframe(episode_df, width='stretch', height=350)

if not errors_df.empty:
    with st.expander(f"\u26A0\uFE0F {len(errors_df)} evaluation errors were logged"):
        st.dataframe(errors_df, width='stretch')

st.divider()

# ---------------------------------------------------------------------------
# Data completeness check — evaluator.snapshot_environment() guesses attribute
# names (e.g. "position", "final_position", "tyre_compound") on env.state /
# env / info. If your F1StrategyEnv uses different names, those columns will
# be silently empty rather than erroring, which would otherwise show up as
# confusing blank charts on the Overview / Lap Replay pages.
# ---------------------------------------------------------------------------
lap_df = load_lap_trace(output_dir)

KEY_EPISODE_COLUMNS = [
    "starting_position", "final_position", "position_gain",
    "total_race_time", "mean_lap_time",
]
KEY_LAP_COLUMNS = [
    "position_before", "position_after", "tyre_compound_after",
    "tyre_age_after", "track_wetness", "lap_time",
]

with st.expander("\U0001F50D Data completeness check", expanded=False):
    st.caption(
        "Percentage of non-null rows for fields that depend on your F1StrategyEnv "
        "exposing matching attribute names in `snapshot_environment`. Anything "
        "well below 100% likely means a rename is needed there, not in the dashboard."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**episode_summary.csv**")
        if not episode_df.empty:
            st.dataframe(
                column_completeness(episode_df, KEY_EPISODE_COLUMNS),
                width='stretch',
                hide_index=True,
            )
    with c2:
        st.markdown("**lap_trace.csv**")
        if not lap_df.empty:
            st.dataframe(
                column_completeness(lap_df, KEY_LAP_COLUMNS),
                width='stretch',
                hide_index=True,
            )

st.divider()
st.markdown(
    """
    ### Use the pages in the sidebar
    - **\U0001F4CA Overview** — aggregate benchmark comparison table + paired significance tests
    - **\U0001F501 Lap Replay** — lap-by-lap side-by-side replay of a single race
    - **\U0001F500 Strategy Divergence** — action disagreement heatmap between agents
    """
)

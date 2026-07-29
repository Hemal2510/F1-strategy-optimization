from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import agent_color, load_lap_trace

st.set_page_config(page_title="Lap Replay", page_icon="\U0001F501", layout="wide")

output_dir = st.session_state.get("output_dir")
if not output_dir:
    st.warning("Pick a benchmark run on the Home page first.")
    st.stop()

st.title("\U0001F501 Lap-by-Lap Replay")

lap_df = load_lap_trace(output_dir)
if lap_df.empty:
    st.warning("`lap_trace.csv` not found or empty.")
    st.stop()

# ---------------------------------------------------------------------------
# Race selector
# ---------------------------------------------------------------------------
race_keys = (
    lap_df[["seed", "track", "year", "race_name"]]
    .drop_duplicates()
    .sort_values(["track", "seed"])
    .reset_index(drop=True)
)
race_labels = [
    f"seed {row.seed} · {row.track} ({row.year})" for row in race_keys.itertuples()
]
race_choice = st.selectbox(
    "Race to replay", options=range(len(race_labels)), format_func=lambda i: race_labels[i]
)
sel = race_keys.iloc[race_choice]

race_df = lap_df[
    (lap_df["seed"] == sel["seed"])
    & (lap_df["track"] == sel["track"])
    & (lap_df["race_name"] == sel["race_name"])
].copy()

agents_in_race = sorted(race_df["agent"].unique())
st.caption(
    f"Comparing **{', '.join(a.upper() for a in agents_in_race)}** on "
    f"seed **{sel['seed']}**, track **{sel['track']}**."
)

# Numeric columns can come back as NaN (not raise) when your F1StrategyEnv
# doesn't expose an attribute name snapshot_environment() is guessing for —
# coerce defensively so a stray non-numeric value can't break a chart.
for col in ("position_after", "position_before", "track_wetness", "tyre_age_after", "reward", "cumulative_reward"):
    if col in race_df.columns:
        race_df[col] = pd.to_numeric(race_df[col], errors="coerce")

# lap_index is a plain loop counter assigned directly by evaluator.run_episode()
# (0, 1, 2, ...) — always populated. current_lap, by contrast, is read from
# snapshot_environment() and depends on the env exposing "current_lap"/"lap"/
# "lap_number"; it can be entirely null if none of those match your env's
# actual attribute names. Use lap_index as the x-axis so replay still works
# even when current_lap doesn't.
position_available = race_df["position_after"].notna().any()
if not position_available:
    st.warning(
        "`position_after` is empty for this race. `evaluator.snapshot_environment()` "
        "looks for `position` / `current_position` / `end_position` on `env.state` — "
        "check your F1StrategyEnv exposes one of those attribute names."
    )

# Optional real-team-strategy overlay — not produced by the current benchmarking
# pipeline (adapters.py only has dqn/qrl/random/rule_based), so this is a hook
# for whenever real timing data is available.
real_df = None
with st.expander("\u2795 Add a real-team-strategy trace (optional)"):
    st.caption(
        "The benchmarking pipeline doesn't currently log real F1 team strategy — "
        "upload a CSV with columns `lap_index,action_name` (optionally `position_after`) "
        "to overlay it here."
    )
    real_upload = st.file_uploader("Real strategy CSV", type="csv", key="real_strategy")
    if real_upload is not None:
        real_df = pd.read_csv(real_upload)

# ---------------------------------------------------------------------------
# Position over laps, with pit-stop markers
# ---------------------------------------------------------------------------
st.subheader("Race position over laps")

if position_available:
    fig_pos = go.Figure()
    for agent in agents_in_race:
        a_df = race_df[race_df["agent"] == agent].sort_values("lap_index")
        fig_pos.add_trace(
            go.Scatter(
                x=a_df["lap_index"],
                y=a_df["position_after"],
                mode="lines+markers",
                name=agent.upper(),
                line=dict(color=agent_color(agent), width=3),
            )
        )
        pit_laps = a_df[a_df["action"] != 0]
        if not pit_laps.empty:
            fig_pos.add_trace(
                go.Scatter(
                    x=pit_laps["lap_index"],
                    y=pit_laps["position_after"],
                    mode="markers",
                    name=f"{agent.upper()} pit stop",
                    marker=dict(
                        color=agent_color(agent), size=13, symbol="diamond",
                        line=dict(width=1, color="black"),
                    ),
                )
            )

    if real_df is not None and "position_after" in real_df.columns:
        fig_pos.add_trace(
            go.Scatter(
                x=real_df["lap_index"],
                y=real_df["position_after"],
                mode="lines+markers",
                name="REAL TEAM",
                line=dict(color=agent_color("real_team"), width=3, dash="dot"),
            )
        )

    fig_pos.update_yaxes(autorange="reversed", title="Position (P1 at top)")
    fig_pos.update_xaxes(title="Lap")
    fig_pos.update_layout(height=420, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_pos, width='stretch')

# ---------------------------------------------------------------------------
# Cumulative reward over laps
# ---------------------------------------------------------------------------
st.subheader("Cumulative reward over laps")

fig_reward = go.Figure()
for agent in agents_in_race:
    a_df = race_df[race_df["agent"] == agent].sort_values("lap_index")
    fig_reward.add_trace(
        go.Scatter(
            x=a_df["lap_index"],
            y=a_df["cumulative_reward"],
            mode="lines",
            name=agent.upper(),
            line=dict(color=agent_color(agent), width=3),
        )
    )
fig_reward.update_layout(height=320, xaxis_title="Lap", yaxis_title="Cumulative reward")
st.plotly_chart(fig_reward, width='stretch')

st.divider()

# ---------------------------------------------------------------------------
# Lap scrubber — side-by-side agent state, with Q-value bars
# ---------------------------------------------------------------------------
st.subheader("Scrub through the race")

max_lap = int(race_df["lap_index"].max())
lap_choice = st.slider("Lap (lap_index)", min_value=0, max_value=max_lap, value=0)

action_cols = [c for c in race_df.columns if c.startswith("q_")]
n_actions = len(action_cols)
action_names_map = (
    race_df[["action", "action_name"]].drop_duplicates().set_index("action")["action_name"].to_dict()
)

cols = st.columns(len(agents_in_race))
for col, agent in zip(cols, agents_in_race):
    with col:
        st.markdown(f"### {agent.upper()}")
        row_df = race_df[(race_df["agent"] == agent) & (race_df["lap_index"] == lap_choice)]
        if row_df.empty:
            st.info("No data for this lap.")
            continue
        row = row_df.iloc[0]

        st.metric("Action taken", row["action_name"])
        if pd.notna(row.get("current_lap")):
            st.caption(f"env-reported lap: {row['current_lap']:.0f}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Tyre", str(row["tyre_compound_after"]))
        m2.metric(
            "Tyre age",
            f"{row['tyre_age_after']:.0f}" if pd.notna(row["tyre_age_after"]) else "—",
        )
        m3.metric(
            "Wetness",
            f"{row['track_wetness']:.2f}" if pd.notna(row["track_wetness"]) else "—",
        )

        pos_after = row["position_after"]
        pos_label = f"P{int(pos_after)}" if pd.notna(pos_after) else "—"
        st.caption(f"Reward this lap: {row['reward']:.3f} · Position: {pos_label}")

        if n_actions and pd.notna(row.get("q_0", np.nan)):
            q_vals = [row[f"q_{i}"] for i in range(n_actions)]
            mask_vals = [bool(row[f"mask_{i}"]) for i in range(n_actions)]
            labels = [action_names_map.get(i, f"action_{i}") for i in range(n_actions)]
            chosen = int(row["action"])
            bar_colors = [
                "#2ecc71" if i == chosen else ("#bdc3c7" if mask_vals[i] else "#e74c3c")
                for i in range(n_actions)
            ]
            fig_q = go.Figure(go.Bar(x=labels, y=q_vals, marker_color=bar_colors))
            fig_q.update_layout(
                height=260,
                title="Q-values (green = chosen, grey = legal, red = illegal)",
                margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig_q, width='stretch')
        else:
            st.caption("No Q-values logged for this agent (non-neural baseline).")

st.divider()

with st.expander("Full lap trace for this race"):
    st.dataframe(
        race_df.sort_values(["agent", "lap_index"]),
        width='stretch',
        height=400,
    )

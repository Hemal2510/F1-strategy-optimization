from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import agent_color, agent_label

st.set_page_config(page_title="Lap Replay", page_icon="🔁", layout="wide")

results = st.session_state.get("results")
if results is None:
    st.warning("Run a benchmark on the Home page first.")
    st.stop()

st.title("🔁 Lap-by-Lap Replay")

lap_df: pd.DataFrame = results.get("lap_df", pd.DataFrame())
if lap_df.empty:
    st.warning("`lap_trace.csv` is empty — no lap data to replay.")
    st.stop()

# Ensure lap_index exists
if "lap_index" not in lap_df.columns:
    for alias in ("current_lap", "lap", "lap_number"):
        if alias in lap_df.columns:
            lap_df["lap_index"] = pd.to_numeric(lap_df[alias], errors="coerce").fillna(0).astype(int)
            break
    else:
        lap_df["lap_index"] = lap_df.index

# Coerce numerics
for col in ("position_after", "position_before", "track_wetness", "tyre_age_after", "reward", "cumulative_reward"):
    if col in lap_df.columns:
        lap_df[col] = pd.to_numeric(lap_df[col], errors="coerce")

# Select which race to display
key_cols = [c for c in ("seed", "track", "year", "race_name") if c in lap_df.columns]
race_keys = lap_df[key_cols].drop_duplicates().sort_values([c for c in ("track", "year", "seed") if c in key_cols]).reset_index(drop=True)

def _race_label(row):
    parts = []
    if hasattr(row, "track"): parts.append(str(row.track))
    if hasattr(row, "year"): parts.append(str(row.year))
    if hasattr(row, "seed"): parts.append(f"seed {row.seed}")
    return " | ".join(parts) if parts else f"Race {row.Index}"

race_labels = [_race_label(r) for r in race_keys.itertuples()]
race_choice = st.selectbox("Race to replay", options=range(len(race_labels)), format_func=lambda i: race_labels[i])
sel = race_keys.iloc[race_choice]

mask = pd.Series(True, index=lap_df.index)
for col in key_cols:
    mask &= lap_df[col] == sel[col]
race_df = lap_df[mask].copy()

agents_in_race = sorted(race_df["agent"].unique()) if "agent" in race_df.columns else []
st.caption(f"Agents: **{', '.join(agent_label(a) for a in agents_in_race)}**")

# Position curve plot over race laps
st.subheader("Race Position over Laps")

position_available = "position_after" in race_df.columns and race_df["position_after"].notna().any()
if position_available:
    fig_pos = go.Figure()
    for agent in agents_in_race:
        a_df = race_df[race_df["agent"] == agent].sort_values("lap_index")
        fig_pos.add_trace(go.Scatter(
            x=a_df["lap_index"], y=a_df["position_after"],
            mode="lines+markers", name=agent_label(agent),
            line=dict(color=agent_color(agent), width=3),
        ))
        pits = a_df[a_df["action"] != 0] if "action" in a_df.columns else pd.DataFrame()
        if not pits.empty:
            fig_pos.add_trace(go.Scatter(
                x=pits["lap_index"], y=pits["position_after"],
                mode="markers", name=f"{agent_label(agent)} pit",
                marker=dict(color=agent_color(agent), size=13, symbol="diamond",
                            line=dict(width=1, color="black")),
            ))
    fig_pos.update_yaxes(autorange="reversed", title="Position (P1 at top)")
    fig_pos.update_xaxes(title="Lap")
    fig_pos.update_layout(height=420, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_pos, width="stretch")
else:
    st.info("Position data not available for this race.")

# Plot cumulative rewards
st.subheader("Cumulative Reward over Laps")

if "cumulative_reward" in race_df.columns:
    fig_rew = go.Figure()
    for agent in agents_in_race:
        a_df = race_df[race_df["agent"] == agent].sort_values("lap_index")
        fig_rew.add_trace(go.Scatter(
            x=a_df["lap_index"], y=a_df["cumulative_reward"],
            mode="lines", name=agent_label(agent),
            line=dict(color=agent_color(agent), width=3),
        ))
    fig_rew.update_layout(height=320, xaxis_title="Lap", yaxis_title="Cumulative Reward")
    st.plotly_chart(fig_rew, width="stretch")

st.divider()

# Lap slider component for step-by-step breakdown
st.subheader("Scrub Through the Race")

max_lap = int(race_df["lap_index"].max()) if not race_df.empty else 0
lap_choice = st.slider("Lap", min_value=0, max_value=max_lap, value=0)

DEFAULT_ACTION_NAMES = {
    0: "stay_out",
    1: "pit_soft",
    2: "pit_medium",
    3: "pit_hard",
    4: "pit_intermediate",
    5: "pit_wet",
}

if "action" in race_df.columns and "action_name" in race_df.columns:
    extracted_map = race_df[["action", "action_name"]].dropna().drop_duplicates().set_index("action")["action_name"].to_dict()
    action_names_map = {**DEFAULT_ACTION_NAMES, **extracted_map}
else:
    action_names_map = DEFAULT_ACTION_NAMES

action_cols = [c for c in race_df.columns if c.startswith("q_")]
n_actions = len(action_cols)

N_COLS = 3
for i in range(0, len(agents_in_race), N_COLS):
    chunk = agents_in_race[i:i + N_COLS]
    cols = st.columns(len(chunk))
    for col, agent in zip(cols, chunk):
        with col:
            st.markdown(f"### {agent_label(agent)}")
            row_df = race_df[(race_df["agent"] == agent) & (race_df["lap_index"] == lap_choice)]
            if row_df.empty:
                st.info("No data for this lap.")
                continue
            row = row_df.iloc[0]

            act_val = row.get("action", "?")
            action_name = row.get("action_name")
            if pd.isna(action_name) or str(action_name).startswith("action_"):
                action_name = action_names_map.get(act_val, f"action {act_val}")

            st.metric("Action taken", str(action_name))
            m1, m2, m3 = st.columns(3)
            m1.metric("Tyre", str(row.get("tyre_compound_after", "—")))
            m2.metric("Tyre age", f"{row['tyre_age_after']:.0f}" if pd.notna(row.get("tyre_age_after")) else "—")
            m3.metric("Wetness", f"{row['track_wetness']:.2f}" if pd.notna(row.get("track_wetness")) else "—")

            pos = row.get("position_after")
            pos_label = f"P{int(pos)}" if pd.notna(pos) else "—"
            rew = row.get("reward", float("nan"))
            st.caption(f"Reward: {rew:.3f}  ·  Position: {pos_label}")

            if n_actions and pd.notna(row.get("q_0", np.nan)):
                q_vals = [row[f"q_{idx}"] for idx in range(n_actions)]
                mask_vals = [bool(row.get(f"mask_{idx}", True)) for idx in range(n_actions)]
                labels = [action_names_map.get(idx, f"action_{idx}") for idx in range(n_actions)]
                chosen = int(row.get("action", -1))
                bar_colors = [
                    "#2ecc71" if idx == chosen else ("#bdc3c7" if mask_vals[idx] else "#e74c3c")
                    for idx in range(n_actions)
                ]
                fig_q = go.Figure(go.Bar(x=labels, y=q_vals, marker_color=bar_colors))
                fig_q.update_layout(height=240, title="Q-values", margin=dict(t=36, b=8))
                st.plotly_chart(fig_q, width="stretch")
            else:
                st.caption("No Q-values (non-neural agent).")

st.divider()
with st.expander("Full lap trace table", expanded=False):
    st.dataframe(race_df.sort_values(["agent", "lap_index"]) if "agent" in race_df.columns else race_df,
                 width="stretch", height=400)

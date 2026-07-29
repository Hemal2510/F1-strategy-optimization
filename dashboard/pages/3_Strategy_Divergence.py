from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import agent_color, agent_label

st.set_page_config(page_title="Strategy Divergence", page_icon="🔀", layout="wide")

results = st.session_state.get("results")
if results is None:
    st.warning("Run a benchmark on the Home page first.")
    st.stop()

filters = st.session_state.get("filters", {})
agents_run = filters.get("agents", [])

if "dqn" not in agents_run or "qrl" not in agents_run:
    st.info("Strategy Divergence requires both **DQN** and **QRL** to be selected in your benchmark run.")
    st.stop()

st.title("🔀 Strategy Divergence — DQN vs QRL")
st.caption("Where and how often do DQN and QRL choose different actions on the same lap?")

disagree_df: pd.DataFrame = results.get("disagreement", pd.DataFrame())
if disagree_df.empty:
    st.warning("No action-disagreement data — re-run the benchmark with both DQN and QRL selected.")
    st.stop()

# Ensure lap_index
if "lap_index" not in disagree_df.columns:
    disagree_df["lap_index"] = disagree_df.index

# ---------------------------------------------------------------------------
# Track filter
# ---------------------------------------------------------------------------
track_options = sorted(disagree_df["track"].dropna().unique()) if "track" in disagree_df.columns else []
if track_options:
    track_choice = st.selectbox("Filter by Track", options=["All"] + list(track_options))
    heat_df = disagree_df if track_choice == "All" else disagree_df[disagree_df["track"] == track_choice]
else:
    heat_df = disagree_df

# ---------------------------------------------------------------------------
# Heatmap: seed × lap
# ---------------------------------------------------------------------------
st.subheader("Disagreement Heatmap (Seed × Lap)")

if "seed" in heat_df.columns and "actions_disagree" in heat_df.columns:
    pivot = heat_df.pivot_table(
        index="seed", columns="lap_index", values="actions_disagree", aggfunc="max"
    ).sort_index()

    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot.values.astype(float),
        x=pivot.columns,
        y=pivot.index.astype(str),
        colorscale=[[0, "#2ecc71"], [1, "#e74c3c"]],
        zmin=0, zmax=1,
        colorbar=dict(title="Disagree", tickvals=[0, 1], ticktext=["Same", "Different"]),
    ))
    fig_heat.update_layout(
        height=max(320, 22 * pivot.shape[0]),
        xaxis_title="Lap", yaxis_title="Seed",
    )
    st.plotly_chart(fig_heat, width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Disagreement rate per lap
# ---------------------------------------------------------------------------
st.subheader("Disagreement Rate per Lap")

rate_by_lap = heat_df.groupby("lap_index")["actions_disagree"].mean().reset_index()
fig_rate = px.line(rate_by_lap, x="lap_index", y="actions_disagree", markers=True)
fig_rate.update_layout(
    height=320, xaxis_title="Lap",
    yaxis_title="Fraction of races where agents disagree",
    yaxis_tickformat=".0%",
)
st.plotly_chart(fig_rate, width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Action distribution
# ---------------------------------------------------------------------------
st.subheader("Action Distribution: DQN vs QRL")

if "action_name_a" in heat_df.columns and "action_name_b" in heat_df.columns:
    counts_a = heat_df["action_name_a"].value_counts(normalize=True).rename("dqn")
    counts_b = heat_df["action_name_b"].value_counts(normalize=True).rename("qrl")
    action_dist = pd.concat([counts_a, counts_b], axis=1).fillna(0).reset_index()
    action_dist = action_dist.rename(columns={"index": "action"})
    action_dist_long = action_dist.melt(id_vars="action", var_name="agent", value_name="share")

    fig_dist = px.bar(
        action_dist_long, x="action", y="share", color="agent", barmode="group",
        color_discrete_map={"dqn": agent_color("dqn"), "qrl": agent_color("qrl")},
    )
    fig_dist.update_layout(height=380, yaxis_tickformat=".0%")
    st.plotly_chart(fig_dist, width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Reward impact of disagreement
# ---------------------------------------------------------------------------
st.subheader("Reward Impact of Disagreement")

if "reward_b_minus_a" in heat_df.columns and "actions_disagree" in heat_df.columns:
    fig_box = px.box(
        heat_df, x="actions_disagree", y="reward_b_minus_a", points="all",
        labels={
            "actions_disagree": "Agents disagreed on action",
            "reward_b_minus_a": "Reward(QRL) − Reward(DQN) that lap",
        },
    )
    fig_box.update_xaxes(tickvals=[False, True], ticktext=["Agreed", "Disagreed"])
    st.plotly_chart(fig_box, width="stretch")
    st.caption(
        "A consistently higher reward delta on disagreement laps suggests QRL's divergent choices are paying off."
    )

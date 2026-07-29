from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import load_action_disagreement

st.set_page_config(page_title="Strategy Divergence", page_icon="\U0001F500", layout="wide")

output_dir = st.session_state.get("output_dir")
if not output_dir:
    st.warning("Pick a benchmark run on the Home page first.")
    st.stop()

st.title("\U0001F500 Strategy Divergence Analysis")
st.caption("Where and how often do DQN and QRL choose different actions on the same lap?")

disagree_df = load_action_disagreement(output_dir)
if disagree_df.empty:
    st.warning(
        "`action_disagreement.csv` not found or empty. It's produced by "
        "`metrics.action_disagreement()` / `build_metric_artifacts()`."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Heatmap: seed x lap, colored by disagreement
# ---------------------------------------------------------------------------
st.subheader("Disagreement heatmap (seed × lap)")

track_options = sorted(disagree_df["track"].dropna().unique())
track_choice = st.selectbox("Track", options=["All"] + list(track_options))
heat_df = disagree_df if track_choice == "All" else disagree_df[disagree_df["track"] == track_choice]

pivot = heat_df.pivot_table(
    index="seed", columns="lap_index", values="actions_disagree", aggfunc="max"
).sort_index()

fig_heat = go.Figure(
    data=go.Heatmap(
        z=pivot.values.astype(float),
        x=pivot.columns,
        y=pivot.index.astype(str),
        colorscale=[[0, "#2ecc71"], [1, "#e74c3c"]],
        zmin=0,
        zmax=1,
        colorbar=dict(
            title="Disagree",
            tickvals=[0, 1],
            ticktext=["Same action", "Different action"],
        ),
    )
)
fig_heat.update_layout(
    height=max(320, 22 * pivot.shape[0]),
    xaxis_title="Lap",
    yaxis_title="Seed",
)
st.plotly_chart(fig_heat, width='stretch')

st.divider()

# ---------------------------------------------------------------------------
# Disagreement rate by lap, averaged over seeds
# ---------------------------------------------------------------------------
st.subheader("Disagreement rate by lap")

rate_by_lap = heat_df.groupby("lap_index")["actions_disagree"].mean().reset_index()
fig_rate = px.line(rate_by_lap, x="lap_index", y="actions_disagree", markers=True)
fig_rate.update_layout(
    height=320,
    xaxis_title="Lap",
    yaxis_title="Fraction of races where agents disagree",
    yaxis_tickformat=".0%",
)
st.plotly_chart(fig_rate, width='stretch')

st.divider()

# ---------------------------------------------------------------------------
# Action distribution comparison
# ---------------------------------------------------------------------------
st.subheader("Action distribution: DQN vs QRL")

counts_a = heat_df["action_name_a"].value_counts(normalize=True).rename("dqn")
counts_b = heat_df["action_name_b"].value_counts(normalize=True).rename("qrl")
action_dist = pd.concat([counts_a, counts_b], axis=1).fillna(0).reset_index()
action_dist = action_dist.rename(columns={"index": "action"})
action_dist_long = action_dist.melt(id_vars="action", var_name="agent", value_name="share")

fig_dist = px.bar(
    action_dist_long,
    x="action",
    y="share",
    color="agent",
    barmode="group",
    color_discrete_map={"dqn": "#4C78A8", "qrl": "#F58518"},
)
fig_dist.update_layout(height=380, yaxis_tickformat=".0%")
st.plotly_chart(fig_dist, width='stretch')

st.divider()

# ---------------------------------------------------------------------------
# Does disagreement correlate with a reward difference?
# ---------------------------------------------------------------------------
st.subheader("Reward impact of disagreement")

fig_box = px.box(
    heat_df,
    x="actions_disagree",
    y="reward_b_minus_a",
    points="all",
    labels={
        "actions_disagree": "Agents disagreed on action",
        "reward_b_minus_a": "Reward(QRL) − Reward(DQN) that lap",
    },
)
fig_box.update_xaxes(tickvals=[False, True], ticktext=["Agreed", "Disagreed"])
st.plotly_chart(fig_box, width='stretch')

st.caption(
    "If disagreement laps show a systematically higher or lower reward delta than "
    "agreement laps, that's a signal about whether QRL's divergent choices are paying off."
)

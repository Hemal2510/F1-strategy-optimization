from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import (
    METRIC_LABELS,
    agent_color,
    load_overall_metrics,
    load_paired_comparison,
    load_track_metrics,
)

st.set_page_config(page_title="Overview · Benchmark Comparison", page_icon="\U0001F4CA", layout="wide")

output_dir = st.session_state.get("output_dir")
if not output_dir:
    st.warning("Pick a benchmark run on the Home page first.")
    st.stop()

st.title("\U0001F4CA Benchmark Comparison")

overall_df = load_overall_metrics(output_dir)
track_df = load_track_metrics(output_dir)
paired_df = load_paired_comparison(output_dir)

if overall_df.empty:
    st.warning(
        "`overall_metrics.csv` not found. Run `metrics.build_metric_artifacts()` "
        "(step 3 of `run_benchmark.py`) first."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Headline KPI cards
# ---------------------------------------------------------------------------
st.subheader("Headline metrics")

headline_metrics = ["total_reward", "final_position", "pit_count"]
cols = st.columns(len(headline_metrics))
for col, metric in zip(cols, headline_metrics):
    with col:
        st.markdown(f"**{METRIC_LABELS.get(metric, metric)}**")
        sub = overall_df[overall_df["metric"] == metric]
        if sub.empty:
            st.caption("No data for this metric in overall_metrics.csv.")
            continue
        for _, row in sub.iterrows():
            st.metric(
                label=row["agent"].upper(),
                value=f"{row['mean']:.2f}",
                help=f"std={row['std']:.2f}, n={int(row['count'])}",
            )

st.divider()

# ---------------------------------------------------------------------------
# Mean metrics with 95% CI, one bar chart per selected metric
# ---------------------------------------------------------------------------
st.subheader("Mean metrics with 95% confidence intervals")

metric_options = list(overall_df["metric"].unique())
chosen_metrics = st.multiselect(
    "Metrics to plot",
    options=metric_options,
    default=[m for m in headline_metrics if m in metric_options],
)

for metric in chosen_metrics:
    sub = overall_df[overall_df["metric"] == metric].sort_values("agent")
    if sub.empty:
        continue

    fig = go.Figure(
        go.Bar(
            x=sub["agent"],
            y=sub["mean"],
            error_y=dict(
                type="data",
                symmetric=False,
                array=sub["ci_high"] - sub["mean"],
                arrayminus=sub["mean"] - sub["ci_low"],
            ),
            marker_color=[agent_color(a) for a in sub["agent"]],
            text=[f"{v:.2f}" for v in sub["mean"]],
            textposition="outside",
        )
    )
    lower_better = bool(sub["lower_is_better"].iloc[0])
    fig.update_layout(
        title=(
            f"{METRIC_LABELS.get(metric, metric)} "
            f"({'lower is better' if lower_better else 'higher is better'})"
        ),
        yaxis_title=METRIC_LABELS.get(metric, metric),
        showlegend=False,
        height=380,
    )
    st.plotly_chart(fig, width='stretch')

st.divider()

# ---------------------------------------------------------------------------
# Paired comparison table
# ---------------------------------------------------------------------------
st.subheader("Paired comparison (matched seed / track / race)")

if paired_df.empty:
    st.info("No paired comparison data available.")
else:
    display_df = paired_df.copy()
    display_df["metric"] = display_df["metric"].map(lambda m: METRIC_LABELS.get(m, m))
    display_df["significant (p<0.05)"] = display_df["wilcoxon_p_value"] < 0.05

    st.dataframe(
        display_df[
            [
                "metric",
                "paired_runs",
                "mean_a",
                "mean_b",
                "mean_b_minus_a",
                "paired_t_p_value",
                "wilcoxon_p_value",
                "significant (p<0.05)",
                "numerical_winner",
            ]
        ].style.format(
            {
                "mean_a": "{:.3f}",
                "mean_b": "{:.3f}",
                "mean_b_minus_a": "{:.3f}",
                "paired_t_p_value": "{:.4f}",
                "wilcoxon_p_value": "{:.4f}",
            }
        ),
        width='stretch',
    )

    st.caption(
        f"agent_a = **{paired_df['agent_a'].iloc[0]}**, agent_b = **{paired_df['agent_b'].iloc[0]}**. "
        "`mean_b_minus_a` is on the raw metric — check `lower_is_better` before reading it as 'better'. "
        "With ~25 seeds, treat p-values as directional, not conclusive."
    )

st.divider()

# ---------------------------------------------------------------------------
# Per-track breakdown
# ---------------------------------------------------------------------------
st.subheader("Breakdown by track")

if track_df.empty:
    st.info("No per-track metrics available.")
else:
    metric_for_track = st.selectbox(
        "Metric",
        options=list(track_df["metric"].unique()),
        key="track_metric",
    )
    sub = track_df[track_df["metric"] == metric_for_track]
    fig = px.bar(
        sub,
        x="track",
        y="mean",
        color="agent",
        barmode="group",
        color_discrete_map={a: agent_color(a) for a in sub["agent"].unique()},
        error_y=sub["ci_high"] - sub["mean"],
        title=f"{METRIC_LABELS.get(metric_for_track, metric_for_track)} by track",
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, width='stretch')

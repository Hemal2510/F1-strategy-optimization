from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from data_loader import METRIC_LABELS, agent_color, agent_label

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")

results = st.session_state.get("results")
if results is None:
    st.warning("Run a benchmark on the Home page first.")
    st.stop()

st.title("📊 Benchmark Overview")

overall_df = results.get("overall")
track_df = results.get("by_track")
paired_df = results.get("paired")
filters = st.session_state.get("filters", {})

if overall_df is None or overall_df.empty:
    st.warning("No aggregated metrics available. Re-run the benchmark.")
    st.stop()

# Bar charts with 95% confidence interval
st.subheader("Mean Metrics with 95% Confidence Interval")

metric_options = [m for m in METRIC_LABELS if m in overall_df["metric"].unique()]
chosen_metrics = st.multiselect(
    "Select metrics to display",
    options=metric_options,
    default=metric_options[:3],
    format_func=lambda m: METRIC_LABELS.get(m, m),
)

for metric in chosen_metrics:
    sub = overall_df[overall_df["metric"] == metric].sort_values("agent")
    if sub.empty:
        continue
    lower_better = bool(sub["lower_is_better"].iloc[0])
    fig = go.Figure(go.Bar(
        x=[agent_label(a) for a in sub["agent"]],
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
    ))
    fig.update_layout(
        title=METRIC_LABELS.get(metric, metric),
        yaxis_title=METRIC_LABELS.get(metric, metric),
        showlegend=False,
        height=380,
    )
    st.plotly_chart(fig, width="stretch")

# Track by track metrics breakdown
if track_df is not None and not track_df.empty:
    st.subheader("Per-Track Breakdown")
    track_metric = st.selectbox(
        "Metric",
        options=metric_options,
        format_func=lambda m: METRIC_LABELS.get(m, m),
    )
    tracks_available = sorted(track_df["track"].unique()) if "track" in track_df.columns else []
    for track in tracks_available:
        sub = track_df[(track_df["metric"] == track_metric) & (track_df["track"] == track)].sort_values("agent")
        if sub.empty:
            continue
        fig_t = go.Figure(go.Bar(
            x=[agent_label(a) for a in sub["agent"]],
            y=sub["mean"],
            marker_color=[agent_color(a) for a in sub["agent"]],
            text=[f"{v:.2f}" for v in sub["mean"]],
            textposition="outside",
        ))
        fig_t.update_layout(title=track, height=300, showlegend=False)
        st.plotly_chart(fig_t, width="stretch")

st.divider()

# Paired comparison in expander
if paired_df is not None and not paired_df.empty:
    with st.expander("🔬 DQN vs QRL paired statistical comparison", expanded=False):
        disp = paired_df.copy()
        disp["metric"] = disp["metric"].map(lambda m: METRIC_LABELS.get(m, m))
        disp["significant (p<0.05)"] = disp["wilcoxon_p_value"] < 0.05
        cols_to_show = [c for c in [
            "metric", "paired_runs", "mean_a", "mean_b",
            "mean_b_minus_a", "paired_t_p_value", "wilcoxon_p_value",
            "significant (p<0.05)", "numerical_winner",
        ] if c in disp.columns]
        st.dataframe(disp[cols_to_show].style.format({
            "mean_a": "{:.3f}", "mean_b": "{:.3f}",
            "mean_b_minus_a": "{:.3f}",
            "paired_t_p_value": "{:.4f}", "wilcoxon_p_value": "{:.4f}",
        }), width="stretch")
        st.caption(
            f"agent_a = **{paired_df['agent_a'].iloc[0]}**, "
            f"agent_b = **{paired_df['agent_b'].iloc[0]}**. "
            "`mean_b_minus_a` is QRL − DQN on the raw metric."
        )

from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_report(
    output_dir: str | Path,
) -> Path:
    """
    Create a Markdown report from the benchmark CSV artifacts.
    """
    output_dir = Path(output_dir)

    episode_df = pd.read_csv(
        output_dir / "episode_summary.csv"
    )

    overall_df = pd.read_csv(
        output_dir / "overall_metrics.csv"
    )

    paired_df = pd.read_csv(
        output_dir / "paired_comparison.csv"
    )

    lines = [
        "# F1 DQN vs QRL Benchmark Report",
        "",
        "## Experiment coverage",
        "",
        f"- Evaluation episodes: **{len(episode_df)}**",
        (
            "- Agents: **"
            + ", ".join(
                sorted(
                    episode_df[
                        "agent"
                    ].dropna().astype(str).unique()
                )
            )
            + "**"
        ),
        f"- Unique seeds: **{episode_df['seed'].nunique()}**",
        (
            "- Tracks: **"
            + ", ".join(
                sorted(
                    episode_df[
                        "track"
                    ].dropna().astype(str).unique()
                )
            )
            + "**"
        ),
        "",
        "## Overall mean metrics",
        "",
    ]

    if overall_df.empty:
        lines.append(
            "No overall metrics were produced."
        )
    else:
        mean_table = overall_df.pivot(
            index="metric",
            columns="agent",
            values="mean",
        )

        lines.append(
            mean_table.to_markdown(
                floatfmt=".4f"
            )
        )

    lines.extend(
        [
            "",
            "## Paired DQN–QRL comparison",
            "",
        ]
    )

    if paired_df.empty:
        lines.append(
            "No paired results were available."
        )
    else:
        report_columns = [
            "metric",
            "paired_runs",
            "mean_a",
            "mean_b",
            "mean_b_minus_a",
            "paired_t_p_value",
            "wilcoxon_p_value",
            "numerical_winner",
        ]

        lines.append(
            paired_df[
                report_columns
            ].to_markdown(
                index=False,
                floatfmt=".4f",
            )
        )

    lines.extend(
        [
            "",
            "## How to interpret the results",
            "",
            "- Higher reward and position gain are normally better.",
            "- Lower final position, race time, lap time and inference time are better.",
            "- Pit count is descriptive; fewer stops are not automatically better.",
            "- A numerical winner only compares sample means.",
            "- A p-value should be read together with effect size and practical race impact.",
            "- Use at least 20–30 paired seeds before making strong conclusions.",
            "- QRL simulator inference time is not the same as real quantum-hardware execution time.",
        ]
    )

    report_path = (
        output_dir / "benchmark_report.md"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return report_path

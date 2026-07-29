from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    stats = None


DEFAULT_METRICS = [
    "total_reward",
    "final_position",
    "position_gain",
    "pit_count",
    "mean_lap_time",
    "invalid_action_rate",
]


LOWER_IS_BETTER = {
    "final_position",
    "mean_lap_time",
    "invalid_action_rate",
}


def approximate_95_ci(
    values: np.ndarray,
) -> tuple[float, float]:
    """
    Calculate an easy-to-understand approximate 95% confidence interval.

    mean ± 1.96 × standard_error
    """
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, np.nan

    if values.size == 1:
        value = float(values[0])
        return value, value

    mean = float(np.mean(values))
    standard_deviation = float(
        np.std(values, ddof=1)
    )
    standard_error = (
        standard_deviation / np.sqrt(values.size)
    )
    margin = 1.96 * standard_error

    return mean - margin, mean + margin


def aggregate_metrics(
    episode_df: pd.DataFrame,
    group_columns: Iterable[str],
) -> pd.DataFrame:
    """
    Calculate descriptive statistics for each group and metric.
    """
    rows: list[dict] = []
    group_columns = list(group_columns)

    for group_key, group_data in episode_df.groupby(
        group_columns,
        dropna=False,
    ):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_values = dict(
            zip(group_columns, group_key)
        )

        for metric in DEFAULT_METRICS:
            values = pd.to_numeric(
                group_data[metric],
                errors="coerce",
            ).to_numpy(dtype=float)

            values = values[np.isfinite(values)]

            if values.size == 0:
                continue

            ci_low, ci_high = approximate_95_ci(
                values
            )

            rows.append(
                {
                    **group_values,
                    "metric": metric,
                    "count": int(values.size),
                    "mean": float(np.mean(values)),
                    "std": (
                        float(np.std(values, ddof=1))
                        if values.size > 1
                        else 0.0
                    ),
                    "median": float(
                        np.median(values)
                    ),
                    "minimum": float(
                        np.min(values)
                    ),
                    "maximum": float(
                        np.max(values)
                    ),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "lower_is_better": (
                        metric in LOWER_IS_BETTER
                    ),
                }
            )

    return pd.DataFrame(rows)


def paired_comparison(
    episode_df: pd.DataFrame,
    agent_a: str = "dqn",
    agent_b: str = "qrl",
) -> pd.DataFrame:
    """
    Compare two agents on matched seed/track/year/race combinations.

    The saved difference is:
        agent_b - agent_a
        QRL - DQN
    """
    pairing_keys = [
        "seed",
        "track",
        "year",
        "race_name",
    ]

    data_a = episode_df[
        episode_df["agent"] == agent_a
    ].copy()

    data_b = episode_df[
        episode_df["agent"] == agent_b
    ].copy()

    paired = data_a.merge(
        data_b,
        on=pairing_keys,
        suffixes=("_a", "_b"),
        how="inner",
    )

    rows: list[dict] = []

    for metric in DEFAULT_METRICS:
        values_a = pd.to_numeric(
            paired[f"{metric}_a"],
            errors="coerce",
        ).to_numpy(dtype=float)

        values_b = pd.to_numeric(
            paired[f"{metric}_b"],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = (
            np.isfinite(values_a)
            & np.isfinite(values_b)
        )

        values_a = values_a[valid]
        values_b = values_b[valid]

        if values_a.size == 0:
            continue

        differences = values_b - values_a

        mean_a = float(np.mean(values_a))
        mean_b = float(np.mean(values_b))
        lower_is_better = (
            metric in LOWER_IS_BETTER
        )

        if np.isclose(mean_a, mean_b):
            winner = "tie"
        elif lower_is_better:
            winner = (
                agent_b
                if mean_b < mean_a
                else agent_a
            )
        else:
            winner = (
                agent_b
                if mean_b > mean_a
                else agent_a
            )

        t_stat = np.nan
        t_p_value = np.nan
        wilcoxon_stat = np.nan
        wilcoxon_p_value = np.nan

        if stats is not None and values_a.size >= 2:
            t_result = stats.ttest_rel(
                values_b,
                values_a,
            )

            t_stat = float(t_result.statistic)
            t_p_value = float(t_result.pvalue)

            try:
                wilcoxon_result = stats.wilcoxon(
                    values_b,
                    values_a,
                )

                wilcoxon_stat = float(
                    wilcoxon_result.statistic
                )
                wilcoxon_p_value = float(
                    wilcoxon_result.pvalue
                )
            except ValueError:
                pass

        rows.append(
            {
                "agent_a": agent_a,
                "agent_b": agent_b,
                "metric": metric,
                "paired_runs": int(
                    values_a.size
                ),
                "mean_a": mean_a,
                "mean_b": mean_b,
                "mean_b_minus_a": float(
                    np.mean(differences)
                ),
                "difference_std": (
                    float(
                        np.std(
                            differences,
                            ddof=1,
                        )
                    )
                    if differences.size > 1
                    else 0.0
                ),
                "paired_t_stat": t_stat,
                "paired_t_p_value": t_p_value,
                "wilcoxon_stat": wilcoxon_stat,
                "wilcoxon_p_value": (
                    wilcoxon_p_value
                ),
                "lower_is_better": (
                    lower_is_better
                ),
                "numerical_winner": winner,
            }
        )

    return pd.DataFrame(rows)


def action_disagreement(
    lap_df: pd.DataFrame,
    agent_a: str = "dqn",
    agent_b: str = "qrl",
) -> pd.DataFrame:
    """
    Align DQN and QRL rows by race and lap index, then mark action differences.
    """
    keys = [
        "seed",
        "track",
        "year",
        "race_name",
        "lap_index",
    ]

    selected_columns = keys + [
        "action",
        "action_name",
        "reward",
        "position_after",
    ]

    data_a = lap_df[
        lap_df["agent"] == agent_a
    ][selected_columns].copy()

    data_b = lap_df[
        lap_df["agent"] == agent_b
    ][selected_columns].copy()

    merged = data_a.merge(
        data_b,
        on=keys,
        suffixes=("_a", "_b"),
        how="inner",
    )

    if merged.empty:
        return merged

    merged["actions_disagree"] = (
        merged["action_a"]
        != merged["action_b"]
    )

    merged["reward_b_minus_a"] = (
        merged["reward_b"]
        - merged["reward_a"]
    )

    return merged


def build_metric_artifacts(
    output_dir: str | Path,
    agent_a: str = "dqn",
    agent_b: str = "qrl",
) -> dict[str, pd.DataFrame]:
    """Read raw CSV files, calculate metrics and save derived artifacts."""
    output_dir = Path(output_dir)

    episode_df = pd.read_csv(
        output_dir / "episode_summary.csv"
    )

    lap_df = pd.read_csv(
        output_dir / "lap_trace.csv"
    )

    overall_df = aggregate_metrics(
        episode_df,
        group_columns=["agent"],
    )

    track_df = aggregate_metrics(
        episode_df,
        group_columns=["agent", "track"],
    )

    paired_df = paired_comparison(
        episode_df,
        agent_a=agent_a,
        agent_b=agent_b,
    )

    disagreement_df = action_disagreement(
        lap_df,
        agent_a=agent_a,
        agent_b=agent_b,
    )

    overall_df.to_csv(
        output_dir / "overall_metrics.csv",
        index=False,
    )

    track_df.to_csv(
        output_dir / "track_metrics.csv",
        index=False,
    )

    paired_df.to_csv(
        output_dir / "paired_comparison.csv",
        index=False,
    )

    disagreement_df.to_csv(
        output_dir / "action_disagreement.csv",
        index=False,
    )

    return {
        "overall": overall_df,
        "by_track": track_df,
        "paired": paired_df,
        "disagreement": disagreement_df,
    }

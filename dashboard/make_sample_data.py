"""
Generate a synthetic benchmark output folder (matching schema.py's columns)
so the Streamlit dashboard can be explored before real checkpoints / the
F1StrategyEnv are wired up.

Usage:
    python make_sample_data.py --output-dir artifacts/sample_run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    stats = None

ACTION_NAMES = {
    0: "stay_out",
    1: "pit_soft",
    2: "pit_medium",
    3: "pit_hard",
    4: "pit_intermediate",
    5: "pit_wet",
}
TRACKS = {"monaco": 78, "monza": 53, "silverstone": 52}
AGENTS = ["dqn", "qrl"]
SEEDS = [1001, 1002, 1003, 1004, 1005]

LOWER_IS_BETTER = {
    "final_position", "total_race_time", "mean_lap_time",
    "mean_lap_delta", "invalid_action_rate", "mean_inference_ms", "p95_inference_ms",
}
DEFAULT_METRICS = [
    "total_reward", "final_position", "position_gain", "total_race_time",
    "mean_lap_time", "mean_lap_delta", "pit_count", "invalid_action_rate",
    "mean_inference_ms", "p95_inference_ms",
]


def simulate_race(rng: np.random.Generator, agent: str, seed: int, track: str, n_laps: int):
    quality = 1.05 if agent == "qrl" else 1.0
    position = int(rng.integers(6, 16))
    starting_position = position
    tyre_age = 0
    compound = "medium"
    cumulative_reward = 0.0
    pit_count = 0
    lap_rows = []

    for lap in range(n_laps):
        wetness = max(0.0, rng.normal(0.05, 0.15)) if rng.random() < 0.15 else 0.0
        tyre_age += 1

        want_pit = tyre_age > rng.integers(16, 24)
        if wetness > 1.0:
            action = 5
        elif wetness > 0:
            action = 4
        elif want_pit and rng.random() < (0.8 if agent == "dqn" else 0.72):
            action = int(rng.choice([1, 2, 3]))
        else:
            action = 0

        if action != 0:
            pit_count += 1
            tyre_age = 0
            compound = {1: "soft", 2: "medium", 3: "hard", 4: "intermediate", 5: "wet"}[action]
            position += int(rng.choice([-1, 0, 1, 2], p=[0.25, 0.35, 0.25, 0.15]))
        else:
            probs = np.array([0.3 * quality, 0.5, 0.2 / quality])
            probs = probs / probs.sum()
            position += int(rng.choice([-1, 0, 1], p=probs))

        position = int(np.clip(position, 1, 20))
        lap_time = 78 + rng.normal(0, 1.2) + (0.02 * tyre_age) + (3 if action != 0 else 0)
        reward = (
            -0.1 * position
            + (0.5 if action != 0 and want_pit else 0.0)
            - (0.05 if action != 0 and not want_pit else 0.0)
        )
        cumulative_reward += reward

        q_values = rng.normal(0, 1, size=6)
        q_values[action] += 1.5
        mask = np.ones(6, dtype=bool)
        if wetness == 0:
            mask[4] = False
            mask[5] = False

        row = {
            "experiment_id": "sample-run",
            "agent": agent,
            "seed": seed,
            "episode_id": f"{agent}-{seed}-{track}",
            "track": track,
            "year": 2026,
            "race_name": f"{track.title()} Grand Prix",
            "lap_index": lap,
            "current_lap": lap,
            "position_before": int(np.clip(position - 1, 1, 20)),
            "position_after": position,
            "tyre_compound_before": compound,
            "tyre_compound_after": compound,
            "tyre_age_before": max(0, tyre_age - 1),
            "tyre_age_after": tyre_age,
            "track_wetness": round(float(wetness), 3),
            "safety_car_flag": bool(rng.random() < 0.03),
            "pit_window": bool(16 <= tyre_age <= 24),
            "gap_to_leader": round(float(position * rng.uniform(1.5, 3.0)), 2),
            "gap_ahead": round(float(rng.uniform(0.5, 3.0)), 2),
            "gap_behind": round(float(rng.uniform(0.5, 3.0)), 2),
            "action": action,
            "action_name": ACTION_NAMES[action],
            "action_valid": True,
            "reward": round(float(reward), 4),
            "cumulative_reward": round(float(cumulative_reward), 4),
            "lap_time": round(float(lap_time), 3),
            "lap_delta": round(float(lap_time - 78), 3),
            "terminated": lap == n_laps - 1,
            "truncated": False,
            "inference_ms": round(float(rng.uniform(0.4, 1.2) * (3.0 if agent == "qrl" else 1.0)), 4),
            "observation": "",
            "next_observation": "",
        }
        for i in range(6):
            row[f"mask_{i}"] = bool(mask[i])
            row[f"q_{i}"] = round(float(q_values[i]), 4)

        lap_rows.append(row)

    final_position = position
    lap_times = pd.Series([r["lap_time"] for r in lap_rows])
    inference = pd.Series([r["inference_ms"] for r in lap_rows])

    summary = {
        "experiment_id": "sample-run",
        "agent": agent,
        "seed": seed,
        "episode_id": f"{agent}-{seed}-{track}",
        "track": track,
        "year": 2026,
        "race_name": f"{track.title()} Grand Prix",
        "completed": True,
        "laps_completed": n_laps,
        "total_reward": round(float(cumulative_reward), 4),
        "starting_position": starting_position,
        "final_position": final_position,
        "position_gain": float(starting_position - final_position),
        "total_race_time": round(float(lap_times.sum()), 3),
        "mean_lap_time": round(float(lap_times.mean()), 3),
        "mean_lap_delta": round(float(lap_times.mean() - 78), 3),
        "pit_count": pit_count,
        "invalid_action_count": 0,
        "invalid_action_rate": 0.0,
        "mean_inference_ms": round(float(inference.mean()), 4),
        "median_inference_ms": round(float(inference.median()), 4),
        "p95_inference_ms": round(float(np.percentile(inference, 95)), 4),
        "parameter_count": 48000 if agent == "dqn" else 5200,
        "checkpoint_path": f"checkpoints/{agent}/sample/final.pt",
    }
    return summary, lap_rows


def aggregate_metrics(episode_df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for group_key, group_data in episode_df.groupby(group_columns, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_values = dict(zip(group_columns, group_key))
        for metric in DEFAULT_METRICS:
            values = pd.to_numeric(group_data[metric], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            se = std / np.sqrt(values.size) if values.size > 1 else 0.0
            rows.append({
                **group_values, "metric": metric, "count": int(values.size),
                "mean": mean, "std": std, "median": float(np.median(values)),
                "minimum": float(np.min(values)), "maximum": float(np.max(values)),
                "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se,
                "lower_is_better": metric in LOWER_IS_BETTER,
            })
    return pd.DataFrame(rows)


def paired_comparison(episode_df: pd.DataFrame, agent_a="dqn", agent_b="qrl") -> pd.DataFrame:
    keys = ["seed", "track", "year", "race_name"]
    a = episode_df[episode_df["agent"] == agent_a]
    b = episode_df[episode_df["agent"] == agent_b]
    paired = a.merge(b, on=keys, suffixes=("_a", "_b"), how="inner")
    rows = []
    for metric in DEFAULT_METRICS:
        va = pd.to_numeric(paired[f"{metric}_a"], errors="coerce").to_numpy(dtype=float)
        vb = pd.to_numeric(paired[f"{metric}_b"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(va) & np.isfinite(vb)
        va, vb = va[valid], vb[valid]
        if va.size == 0:
            continue
        diffs = vb - va
        lower_better = metric in LOWER_IS_BETTER
        mean_a, mean_b = float(np.mean(va)), float(np.mean(vb))
        winner = "tie" if np.isclose(mean_a, mean_b) else (
            agent_b if (mean_b < mean_a) == lower_better else agent_a
        )
        t_stat = t_p = w_stat = w_p = np.nan
        if stats is not None and va.size >= 2:
            t = stats.ttest_rel(vb, va)
            t_stat, t_p = float(t.statistic), float(t.pvalue)
            try:
                w = stats.wilcoxon(vb, va)
                w_stat, w_p = float(w.statistic), float(w.pvalue)
            except ValueError:
                pass
        rows.append({
            "agent_a": agent_a, "agent_b": agent_b, "metric": metric,
            "paired_runs": int(va.size), "mean_a": mean_a, "mean_b": mean_b,
            "mean_b_minus_a": float(np.mean(diffs)),
            "difference_std": float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0,
            "paired_t_stat": t_stat, "paired_t_p_value": t_p,
            "wilcoxon_stat": w_stat, "wilcoxon_p_value": w_p,
            "lower_is_better": lower_better, "numerical_winner": winner,
        })
    return pd.DataFrame(rows)


def action_disagreement(lap_df: pd.DataFrame, agent_a="dqn", agent_b="qrl") -> pd.DataFrame:
    keys = ["seed", "track", "year", "race_name", "lap_index"]
    cols = keys + ["action", "action_name", "reward", "position_after"]
    a = lap_df[lap_df["agent"] == agent_a][cols]
    b = lap_df[lap_df["agent"] == agent_b][cols]
    merged = a.merge(b, on=keys, suffixes=("_a", "_b"), how="inner")
    if merged.empty:
        return merged
    merged["actions_disagree"] = merged["action_a"] != merged["action_b"]
    merged["reward_b_minus_a"] = merged["reward_b"] - merged["reward_a"]
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/sample_run")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summaries, laps = [], []
    for track, n_laps in TRACKS.items():
        for seed in SEEDS:
            for agent in AGENTS:
                s, rows = simulate_race(rng, agent, seed, track, n_laps)
                summaries.append(s)
                laps.extend(rows)

    episode_df = pd.DataFrame(summaries)
    lap_df = pd.DataFrame(laps)

    episode_df.to_csv(out / "episode_summary.csv", index=False)
    lap_df.to_csv(out / "lap_trace.csv", index=False)
    pd.DataFrame([]).to_csv(out / "errors.csv", index=False)

    aggregate_metrics(episode_df, ["agent"]).to_csv(out / "overall_metrics.csv", index=False)
    aggregate_metrics(episode_df, ["agent", "track"]).to_csv(out / "track_metrics.csv", index=False)
    paired_comparison(episode_df).to_csv(out / "paired_comparison.csv", index=False)
    action_disagreement(lap_df).to_csv(out / "action_disagreement.csv", index=False)

    manifest = {
        "experiment_id": "sample-run",
        "project_name": "f1_dqn_qrl_student_benchmark (SAMPLE DATA)",
        "config_path": "",
        "device": "cpu",
        "seeds": SEEDS,
        "agents": AGENTS,
        "episode_count": len(episode_df),
        "lap_row_count": len(lap_df),
        "python_platform": "",
        "torch_version": "",
        "numpy_version": np.__version__,
        "errors": [],
    }
    with (out / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Sample benchmark data written to: {out.resolve()}")


if __name__ == "__main__":
    main()

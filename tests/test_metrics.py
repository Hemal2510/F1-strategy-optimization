import pandas as pd

from benchmarking.metrics import aggregate_metrics, paired_comparisons, strategy_divergence


def episode_frame():
    rows = []
    for seed in [1, 2, 3]:
        rows.append(
            {
                "scenario": "default",
                "seed": seed,
                "track": "Monaco",
                "year": 2025,
                "race_name": "Monaco GP",
                "agent": "dqn",
                "total_reward": 100 + seed,
                "final_position": 7,
                "position_gain": 3,
                "mean_inference_ms": 1.0,
            }
        )
        rows.append(
            {
                "scenario": "default",
                "seed": seed,
                "track": "Monaco",
                "year": 2025,
                "race_name": "Monaco GP",
                "agent": "qrl",
                "total_reward": 103 + seed,
                "final_position": 6,
                "position_gain": 4,
                "mean_inference_ms": 8.0,
            }
        )
    return pd.DataFrame(rows)


def test_aggregate_metrics():
    result = aggregate_metrics(episode_frame(), group_by=("agent",), bootstrap_samples=100)
    assert set(result["agent"]) == {"dqn", "qrl"}
    assert "total_reward" in set(result["metric"])


def test_paired_comparison_direction():
    result = paired_comparisons(episode_frame(), bootstrap_samples=100)
    reward = result[result["metric"] == "total_reward"].iloc[0]
    position = result[result["metric"] == "final_position"].iloc[0]
    latency = result[result["metric"] == "mean_inference_ms"].iloc[0]
    assert reward["numerical_winner"] == "qrl"
    assert position["numerical_winner"] == "qrl"
    assert latency["numerical_winner"] == "dqn"


def test_strategy_divergence():
    rows = []
    for agent, actions in [("dqn", [0, 0, 2]), ("qrl", [0, 1, 2])]:
        for step, action in enumerate(actions):
            rows.append(
                {
                    "scenario": "default",
                    "seed": 1,
                    "track": "Monaco",
                    "year": 2025,
                    "race_name": "Monaco GP",
                    "step_index": step,
                    "agent": agent,
                    "action": action,
                    "action_name": str(action),
                    "reward": 1.0,
                    "position_after": 5,
                    "tyre_compound_after": 1,
                    "safety_car_before": False,
                    "track_wetness_before": 0,
                }
            )
    _, summary = strategy_divergence(pd.DataFrame(rows))
    assert summary.iloc[0]["action_disagreement_rate"] == 1 / 3

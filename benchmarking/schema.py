from __future__ import annotations


DEFAULT_ACTION_NAMES = {
    0: "stay_out",
    1: "pit_soft",
    2: "pit_medium",
    3: "pit_hard",
    4: "pit_intermediate",
    5: "pit_wet",
}


EPISODE_COLUMNS = [
    "experiment_id",
    "agent",
    "seed",
    "episode_id",
    "track",
    "year",
    "race_name",
    "completed",
    "laps_completed",
    "total_reward",
    "starting_position",
    "final_position",
    "position_gain",
    "total_race_time",
    "mean_lap_time",
    "mean_lap_delta",
    "pit_count",
    "invalid_action_count",
    "invalid_action_rate",
    "mean_inference_ms",
    "median_inference_ms",
    "p95_inference_ms",
    "parameter_count",
    "checkpoint_path",
]


LAP_BASE_COLUMNS = [
    "experiment_id",
    "agent",
    "seed",
    "episode_id",
    "track",
    "year",
    "race_name",
    "lap_index",
    "current_lap",
    "position_before",
    "position_after",
    "tyre_compound_before",
    "tyre_compound_after",
    "tyre_age_before",
    "tyre_age_after",
    "track_wetness",
    "safety_car_flag",
    "pit_window",
    "gap_to_leader",
    "gap_ahead",
    "gap_behind",
    "action",
    "action_name",
    "action_valid",
    "reward",
    "cumulative_reward",
    "lap_time",
    "lap_delta",
    "terminated",
    "truncated",
    "inference_ms",
    "observation",
    "next_observation",
]


def lap_columns(action_dim: int = 6) -> list[str]:
    """
    Return the lap CSV columns, including one mask and Q-value column for each
    action.
    """
    columns = list(LAP_BASE_COLUMNS)

    for action_id in range(action_dim):
        columns.append(f"mask_{action_id}")
        columns.append(f"q_{action_id}")

    return columns

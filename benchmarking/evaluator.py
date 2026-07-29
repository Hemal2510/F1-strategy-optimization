from __future__ import annotations

import platform
import time
import uuid
from typing import Any

import numpy as np
import pandas as pd
import torch

from agents.dqn.action_mask import get_action_mask
from env.f1_env import F1StrategyEnv

from .adapters import build_adapters
from .schema import DEFAULT_ACTION_NAMES, EPISODE_COLUMNS, lap_columns
from .utils import (
    ensure_directory,
    first_available,
    observation_to_json,
    set_global_seed,
    write_json,
)


def snapshot_environment(
    env: Any,
    info: dict | None = None,
) -> dict:
    """
    Read important race values from info, env.state or env.

    This function gives the rest of the benchmark one consistent dictionary.
    """
    info = info or {}
    state = getattr(env, "state", None)
    sources = [info, state, env]

    return {
        "track": first_available(
            sources,
            ["track", "track_name", "circuit"],
            "unknown",
        ),
        "year": first_available(
            sources,
            ["year", "season"],
            "unknown",
        ),
        "race_name": first_available(
            sources,
            ["name", "race_name", "event_name"],
            "unknown",
        ),
        "current_lap": first_available(
            sources,
            ["current_lap", "lap", "lap_number"],
        ),
        "position": first_available(
            sources,
            ["position", "current_position", "end_position"],
        ),
        "starting_position": first_available(
            sources,
            ["starting_position", "start_position", "grid_position"],
        ),
        "final_position": first_available(
            sources,
            ["final_position", "end_position"],
        ),
        "tyre_compound": first_available(
            sources,
            ["tyre_compound", "compound"],
        ),
        "tyre_age": first_available(
            sources,
            ["tyre_age", "tyre_life"],
        ),
        "track_wetness": first_available(
            sources,
            ["track_wetness", "wetness"],
        ),
        "safety_car_flag": first_available(
            sources,
            ["safety_car_flag", "safety_car", "sc_flag"],
        ),
        "pit_window": first_available(
            sources,
            ["pit_window", "in_pit_window"],
        ),
        "gap_to_leader": first_available(
            sources,
            ["gap_to_leader"],
        ),
        "gap_ahead": first_available(
            sources,
            ["gap_ahead"],
        ),
        "gap_behind": first_available(
            sources,
            ["gap_behind"],
        ),
        "lap_time": first_available(
            sources,
            ["lap_time", "simulated_lap_time"],
        ),
        "lap_delta": first_available(
            sources,
            ["lap_delta", "delta"],
        ),
    }


def reset_environment(
    env: Any,
    seed: int,
    reset_options: dict,
) -> tuple[np.ndarray, dict]:
    """Reset the environment and support common Gymnasium return styles."""
    try:
        result = env.reset(
            seed=seed,
            options=reset_options or None,
        )
    except TypeError:
        result = env.reset(seed=seed)

    if isinstance(result, tuple) and len(result) == 2:
        observation, info = result
    else:
        observation, info = result, {}

    return (
        np.asarray(observation, dtype=np.float32),
        dict(info or {}),
    )


def run_episode(
    experiment_id: str,
    adapter: Any,
    seed: int,
    environment_kwargs: dict,
    reset_options: dict,
    action_names: dict[int, str],
    save_observations: bool,
) -> tuple[dict, list[dict]]:
    """
    Run one complete race for one agent using one seed.
    """
    set_global_seed(seed)

    # Each agent receives a completely fresh environment instance.
    env = F1StrategyEnv(**environment_kwargs)

    observation, info = reset_environment(
        env,
        seed,
        reset_options,
    )

    initial_state = snapshot_environment(env, info)
    episode_id = (
        f"{adapter.name}-{seed}-{uuid.uuid4().hex[:8]}"
    )

    lap_rows: list[dict] = []
    cumulative_reward = 0.0
    inference_times: list[float] = []
    pit_count = 0
    invalid_action_count = 0
    lap_index = 0
    done = False

    while not done:
        before = snapshot_environment(env, info)

        action_mask = np.asarray(
            get_action_mask(env),
            dtype=bool,
        )

        if action_mask.ndim != 1:
            raise ValueError(
                f"Action mask must be one-dimensional: {action_mask}"
            )

        if not action_mask.any():
            raise RuntimeError("Action mask contains no legal action.")

        start_ns = time.perf_counter_ns()

        decision = adapter.select_action(
            observation=observation,
            action_mask=action_mask,
            env=env,
        )

        inference_ms = (
            time.perf_counter_ns() - start_ns
        ) / 1_000_000.0

        action = int(decision.action)
        action_valid = (
            0 <= action < len(action_mask)
            and bool(action_mask[action])
        )

        if not action_valid:
            invalid_action_count += 1

        if action != 0:
            pit_count += 1

        (
            next_observation,
            reward,
            terminated,
            truncated,
            step_info,
        ) = env.step(action)

        info = dict(step_info or {})
        done = bool(terminated or truncated)

        cumulative_reward += float(reward)
        inference_times.append(inference_ms)

        after = snapshot_environment(env, info)

        row = {
            "experiment_id": experiment_id,
            "agent": adapter.name,
            "seed": seed,
            "episode_id": episode_id,
            "track": initial_state["track"],
            "year": initial_state["year"],
            "race_name": initial_state["race_name"],
            "lap_index": lap_index,
            "current_lap": before["current_lap"],
            "position_before": before["position"],
            "position_after": after["position"],
            "tyre_compound_before": before["tyre_compound"],
            "tyre_compound_after": after["tyre_compound"],
            "tyre_age_before": before["tyre_age"],
            "tyre_age_after": after["tyre_age"],
            "track_wetness": before["track_wetness"],
            "safety_car_flag": before["safety_car_flag"],
            "pit_window": before["pit_window"],
            "gap_to_leader": before["gap_to_leader"],
            "gap_ahead": before["gap_ahead"],
            "gap_behind": before["gap_behind"],
            "action": action,
            "action_name": action_names.get(
                action,
                f"action_{action}",
            ),
            "action_valid": action_valid,
            "reward": float(reward),
            "cumulative_reward": cumulative_reward,
            "lap_time": after["lap_time"],
            "lap_delta": after["lap_delta"],
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "inference_ms": inference_ms,
            "observation": (
                observation_to_json(observation)
                if save_observations
                else ""
            ),
            "next_observation": (
                observation_to_json(next_observation)
                if save_observations
                else ""
            ),
        }

        for action_id in range(len(action_mask)):
            row[f"mask_{action_id}"] = bool(
                action_mask[action_id]
            )

            if decision.q_values is None:
                row[f"q_{action_id}"] = np.nan
            else:
                row[f"q_{action_id}"] = float(
                    decision.q_values[action_id]
                )

        lap_rows.append(row)
        observation = np.asarray(
            next_observation,
            dtype=np.float32,
        )
        lap_index += 1

    final_state = snapshot_environment(env, info)

    starting_position = initial_state["starting_position"]
    if starting_position is None and lap_rows:
        starting_position = lap_rows[0]["position_before"]

    final_position = final_state["final_position"]
    if final_position is None:
        final_position = final_state["position"]

    position_gain = np.nan
    if (
        starting_position is not None
        and final_position is not None
    ):
        position_gain = (
            float(starting_position)
            - float(final_position)
        )

    lap_times = pd.to_numeric(
        pd.Series(
            [row["lap_time"] for row in lap_rows]
        ),
        errors="coerce",
    ).dropna()

    lap_deltas = pd.to_numeric(
        pd.Series(
            [row["lap_delta"] for row in lap_rows]
        ),
        errors="coerce",
    ).dropna()

    inference_array = np.asarray(
        inference_times,
        dtype=float,
    )

    summary = {
        "experiment_id": experiment_id,
        "agent": adapter.name,
        "seed": seed,
        "episode_id": episode_id,
        "track": initial_state["track"],
        "year": initial_state["year"],
        "race_name": initial_state["race_name"],
        "completed": True,
        "laps_completed": len(lap_rows),
        "total_reward": cumulative_reward,
        "starting_position": starting_position,
        "final_position": final_position,
        "position_gain": position_gain,
        "total_race_time": (
            float(lap_times.sum())
            if not lap_times.empty
            else np.nan
        ),
        "mean_lap_time": (
            float(lap_times.mean())
            if not lap_times.empty
            else np.nan
        ),
        "mean_lap_delta": (
            float(lap_deltas.mean())
            if not lap_deltas.empty
            else np.nan
        ),
        "pit_count": pit_count,
        "invalid_action_count": invalid_action_count,
        "invalid_action_rate": (
            invalid_action_count / max(1, len(lap_rows))
        ),
        "mean_inference_ms": float(
            inference_array.mean()
        ),
        "median_inference_ms": float(
            np.median(inference_array)
        ),
        "p95_inference_ms": float(
            np.percentile(inference_array, 95)
        ),
        "parameter_count": adapter.parameter_count,
        "checkpoint_path": str(
            adapter.checkpoint_path or ""
        ),
    }

    try:
        env.close()
    except Exception:
        pass

    return summary, lap_rows


def evaluate(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all enabled agents on all configured seeds."""
    output_dir = ensure_directory(
        config["output_dir"]
    )

    adapters = build_adapters(config)

    experiment_id = (
        config["project_name"]
        + "-"
        + time.strftime("%Y%m%d-%H%M%S")
    )

    action_names = config["benchmark"].get(
        "action_names",
        DEFAULT_ACTION_NAMES,
    )

    episode_summaries: list[dict] = []
    lap_traces: list[dict] = []
    errors: list[dict] = []

    for seed in config["seeds"]:
        for adapter in adapters:
            print(
                f"Evaluating {adapter.name} on seed {seed}..."
            )

            try:
                summary, rows = run_episode(
                    experiment_id=experiment_id,
                    adapter=adapter,
                    seed=int(seed),
                    environment_kwargs=config[
                        "environment"
                    ].get("kwargs", {}),
                    reset_options=config[
                        "environment"
                    ].get("reset_options", {}),
                    action_names=action_names,
                    save_observations=config[
                        "benchmark"
                    ].get(
                        "save_observation_vectors",
                        True,
                    ),
                )

                episode_summaries.append(summary)
                lap_traces.extend(rows)

            except Exception as error:
                errors.append(
                    {
                        "agent": adapter.name,
                        "seed": seed,
                        "error": repr(error),
                    }
                )

                print(
                    f"Failed: {adapter.name}, "
                    f"seed {seed}: {error!r}"
                )

    episode_df = pd.DataFrame(
        episode_summaries
    ).reindex(columns=EPISODE_COLUMNS)

    action_dim = len(action_names)

    lap_df = pd.DataFrame(
        lap_traces
    ).reindex(
        columns=lap_columns(action_dim)
    )

    episode_df.to_csv(
        output_dir / "episode_summary.csv",
        index=False,
    )

    lap_df.to_csv(
        output_dir / "lap_trace.csv",
        index=False,
    )

    pd.DataFrame(errors).to_csv(
        output_dir / "errors.csv",
        index=False,
    )

    manifest = {
        "experiment_id": experiment_id,
        "project_name": config["project_name"],
        "config_path": config["_config_path"],
        "device": config["device"],
        "seeds": config["seeds"],
        "agents": [
            adapter.name
            for adapter in adapters
        ],
        "episode_count": len(episode_df),
        "lap_row_count": len(lap_df),
        "python_platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "errors": errors,
    }

    write_json(
        output_dir / "manifest.json",
        manifest,
    )

    return episode_df, lap_df

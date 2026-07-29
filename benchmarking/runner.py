from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from .adapters import build_adapters_from_list
from .evaluator import run_episode
from .metrics import build_metric_artifacts
from .schema import DEFAULT_ACTION_NAMES, EPISODE_COLUMNS, lap_columns
from .utils import ensure_dir, write_json


AVAILABLE_TRACKS = ["Monaco", "Monza", "Silverstone"]
AVAILABLE_YEARS = [2022, 2023, 2024, 2025]
ALL_AGENTS = ["dqn", "qrl", "random", "always_stay_out", "rule_aware_heuristic", "real_driver"]

OUTPUT_DIR = "benchmarking/artifacts/latest"


def run_benchmark(
    tracks: list[str],
    years: list[int],
    n_seeds: int,
    agents: list[str],
    output_dir: str = OUTPUT_DIR,
    device: str = "cpu",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """
    Run the full benchmark from the dashboard.

    For each (track, year, seed, agent) combination one episode is simulated.
    Results are written to output_dir as CSVs and returned as DataFrames.

    progress_callback(current, total, message) is called after each episode.
    """
    out = ensure_dir(output_dir)

    adapters = build_adapters_from_list(agents, device=device)
    experiment_id = f"benchmark-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    seeds = list(range(1001, 1001 + n_seeds))
    race_combos = [(track, year) for track in tracks for year in years]

    total_episodes = len(adapters) * len(seeds) * len(race_combos)
    current = 0

    episode_summaries: list[dict] = []
    lap_traces: list[dict] = []
    errors: list[dict] = []

    for adapter in adapters:
        for seed in seeds:
            for track, year in race_combos:
                current += 1
                msg = f"{adapter.name} | {track} {year} | seed {seed}"
                if progress_callback:
                    progress_callback(current, total_episodes, msg)

                try:
                    summary, rows = run_episode(
                        experiment_id=experiment_id,
                        adapter=adapter,
                        seed=seed,
                        environment_kwargs={},
                        reset_options={"track": track, "year": year},
                        action_names=DEFAULT_ACTION_NAMES,
                        save_observations=False,
                    )
                    episode_summaries.append(summary)
                    lap_traces.extend(rows)

                except Exception as err:
                    errors.append({
                        "agent": adapter.name,
                        "seed": seed,
                        "track": track,
                        "year": year,
                        "error": repr(err),
                    })

    action_dim = len(DEFAULT_ACTION_NAMES)
    episode_df = pd.DataFrame(episode_summaries).reindex(columns=EPISODE_COLUMNS)
    lap_df = pd.DataFrame(lap_traces).reindex(columns=lap_columns(action_dim))

    episode_df.to_csv(out / "episode_summary.csv", index=False)
    lap_df.to_csv(out / "lap_trace.csv", index=False)
    pd.DataFrame(errors).to_csv(out / "errors.csv", index=False)

    manifest = {
        "experiment_id": experiment_id,
        "tracks": tracks,
        "years": years,
        "seeds": seeds,
        "agents": [a.name for a in adapters],
        "episode_count": len(episode_df),
        "lap_row_count": len(lap_df),
        "error_count": len(errors),
    }
    write_json(out / "manifest.json", manifest)

    metrics = {}
    if not episode_df.empty:
        metrics = build_metric_artifacts(output_dir=str(out))

    return {
        "episode_df": episode_df,
        "lap_df": lap_df,
        "errors": pd.DataFrame(errors),
        "manifest": manifest,
        **metrics,
    }

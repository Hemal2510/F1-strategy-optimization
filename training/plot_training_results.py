from pathlib import Path
import argparse
import csv
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


NUMERIC_COLUMNS = [
    "episode",
    "episode_return",
    "mean_return_50",
    "final_position",
    "mean_position_50",
    "pit_count",
    "mean_pits_50",
    "mean_loss",
    "epsilon",
    "replay_size",
    "total_steps",
]


def load_metrics(csv_path: Path) -> Dict[str, np.ndarray]:
    """Load numeric training metrics from one CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    values = {column: [] for column in NUMERIC_COLUMNS}

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")

        missing_columns = [
            column for column in NUMERIC_COLUMNS if column not in reader.fieldnames
        ]
        if missing_columns:
            raise ValueError(
                f"CSV file is missing columns {missing_columns}: {csv_path}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                for column in NUMERIC_COLUMNS:
                    text = row[column].strip()
                    values[column].append(float(text) if text else np.nan)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid numeric value in {csv_path}, row {row_number}"
                ) from error

    if not values["episode"]:
        raise ValueError(f"CSV file contains no training rows: {csv_path}")

    return {
        column: np.asarray(column_values, dtype=float)
        for column, column_values in values.items()
    }


def finish_and_save(output_path: Path) -> None:
    """Apply common formatting and save one report-quality graph."""
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_single_agent(
    data: Dict[str, np.ndarray],
    agent_name: str,
    output_dir: Path,
) -> None:
    """Create separate training graphs for one agent."""
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = data["episode"]
    safe_name = agent_name.lower().replace(" ", "_")

    # 1. Episode return and 50-episode rolling mean
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, data["episode_return"], alpha=0.35, label="Episode return")
    plt.plot(episodes, data["mean_return_50"], linewidth=2, label="Mean return (50)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"{agent_name} training return")
    finish_and_save(output_dir / f"{safe_name}_training_return.png")

    # 2. Final race position and 50-episode rolling mean
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, data["final_position"], alpha=0.35, label="Final position")
    plt.plot(
        episodes,
        data["mean_position_50"],
        linewidth=2,
        label="Mean position (50)",
    )
    plt.xlabel("Episode")
    plt.ylabel("Finishing position")
    plt.title(f"{agent_name} finishing position")
    plt.gca().invert_yaxis()  # Position 1 is better than position 20.
    finish_and_save(output_dir / f"{safe_name}_finishing_position.png")

    # 3. Pit stops and 50-episode rolling mean
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, data["pit_count"], alpha=0.35, label="Pit stops")
    plt.plot(episodes, data["mean_pits_50"], linewidth=2, label="Mean pits (50)")
    plt.xlabel("Episode")
    plt.ylabel("Pit-stop count")
    plt.title(f"{agent_name} pit-stop behaviour")
    finish_and_save(output_dir / f"{safe_name}_pit_stops.png")

    # 4. Training loss. NaN values are automatically skipped by Matplotlib.
    valid_loss = np.isfinite(data["mean_loss"])
    plt.figure(figsize=(10, 6))
    if np.any(valid_loss):
        plt.plot(
            episodes[valid_loss],
            data["mean_loss"][valid_loss],
            label="Mean episode loss",
        )
    else:
        plt.text(
            0.5,
            0.5,
            "No loss values were recorded",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
        plt.plot([], [], label="Mean episode loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title(f"{agent_name} training loss")
    finish_and_save(output_dir / f"{safe_name}_training_loss.png")

    # 5. Exploration schedule
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, data["epsilon"], label="Epsilon")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title(f"{agent_name} exploration decay")
    finish_and_save(output_dir / f"{safe_name}_epsilon.png")

    # 6. Replay buffer growth
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, data["replay_size"], label="Replay size")
    plt.xlabel("Episode")
    plt.ylabel("Stored transitions")
    plt.title(f"{agent_name} replay-buffer growth")
    finish_and_save(output_dir / f"{safe_name}_replay_buffer.png")


def plot_comparison(
    dqn: Dict[str, np.ndarray],
    qrl: Dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """Create direct DQN-versus-QRL comparison graphs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Mean training return
    plt.figure(figsize=(10, 6))
    plt.plot(dqn["episode"], dqn["mean_return_50"], linewidth=2, label="DQN")
    plt.plot(qrl["episode"], qrl["mean_return_50"], linewidth=2, label="QRL")
    plt.xlabel("Episode")
    plt.ylabel("Mean return (50 episodes)")
    plt.title("DQN vs QRL: training return")
    finish_and_save(output_dir / "comparison_mean_return.png")

    # 2. Mean finishing position
    plt.figure(figsize=(10, 6))
    plt.plot(dqn["episode"], dqn["mean_position_50"], linewidth=2, label="DQN")
    plt.plot(qrl["episode"], qrl["mean_position_50"], linewidth=2, label="QRL")
    plt.xlabel("Episode")
    plt.ylabel("Mean finishing position (50 episodes)")
    plt.title("DQN vs QRL: finishing position")
    plt.gca().invert_yaxis()
    finish_and_save(output_dir / "comparison_mean_position.png")

    # 3. Mean pit-stop count
    plt.figure(figsize=(10, 6))
    plt.plot(dqn["episode"], dqn["mean_pits_50"], linewidth=2, label="DQN")
    plt.plot(qrl["episode"], qrl["mean_pits_50"], linewidth=2, label="QRL")
    plt.xlabel("Episode")
    plt.ylabel("Mean pit stops (50 episodes)")
    plt.title("DQN vs QRL: pit-stop behaviour")
    finish_and_save(output_dir / "comparison_mean_pits.png")

    # 4. Mean loss. Each agent may start learning at a different episode.
    dqn_valid = np.isfinite(dqn["mean_loss"])
    qrl_valid = np.isfinite(qrl["mean_loss"])

    plt.figure(figsize=(10, 6))
    if np.any(dqn_valid):
        plt.plot(
            dqn["episode"][dqn_valid],
            dqn["mean_loss"][dqn_valid],
            label="DQN",
        )
    else:
        plt.plot([], [], label="DQN: no loss recorded")

    if np.any(qrl_valid):
        plt.plot(
            qrl["episode"][qrl_valid],
            qrl["mean_loss"][qrl_valid],
            label="QRL",
        )
    else:
        plt.plot([], [], label="QRL: no loss recorded")

    plt.xlabel("Episode")
    plt.ylabel("Mean episode loss")
    plt.title("DQN vs QRL: training loss")
    finish_and_save(output_dir / "comparison_training_loss.png")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Matplotlib graphs from DQN and QRL training CSV files."
    )
    parser.add_argument(
        "--dqn",
        type=Path,
        default=Path("checkpoints_1/dqn/checkpoints_v1/training_metrics.csv"),
        help="Path to the DQN training_metrics.csv file.",
    )
    parser.add_argument(
        "--qrl",
        type=Path,
        default=Path("checkpoints_1/qrl/checkpoints_qrl_v6/training_metrics.csv"),
        help="Path to the QRL training_metrics.csv file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/training_graphs"),
        help="Directory where PNG graphs will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    dqn_data = load_metrics(args.dqn)
    qrl_data = load_metrics(args.qrl)

    plot_single_agent(dqn_data, "DQN", args.output / "dqn")
    plot_single_agent(qrl_data, "QRL", args.output / "qrl")
    plot_comparison(dqn_data, qrl_data, args.output / "comparison")

    print(f"\nAll graphs saved inside: {args.output.resolve()}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

from .config import load_config
from .evaluator import evaluate
from .metrics import build_metric_artifacts
from .report import generate_report


def main() -> None:
    """
    Run the complete benchmark pipeline.

    1. Load benchmark.json
    2. Evaluate all agents
    3. Calculate metrics
    4. Generate Markdown report
    """
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark trained DQN and QRL "
            "F1 strategy agents."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to benchmark.json",
    )

    arguments = parser.parse_args()
    config = load_config(arguments.config)

    evaluate(config)

    compare_agents = config[
        "benchmark"
    ].get(
        "compare_agents",
        ["dqn", "qrl"],
    )

    build_metric_artifacts(
        output_dir=config["output_dir"],
        agent_a=compare_agents[0],
        agent_b=compare_agents[1],
    )

    report_path = generate_report(
        config["output_dir"]
    )

    print("")
    print("Benchmark completed.")
    print(
        f"Artifacts: {config['output_dir']}"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

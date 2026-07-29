from __future__ import annotations

import argparse

from benchmarking.training_analysis import analyze_training_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DQN/QRL training-history CSV files.")
    parser.add_argument("--input", required=True, help="Combined training_history.csv path.")
    parser.add_argument("--output", default="artifacts/training_metrics.csv")
    parser.add_argument("--rolling-window", type=int, default=50)
    parser.add_argument("--final-window", type=int, default=100)
    parser.add_argument("--threshold-fraction", type=float, default=0.95)
    parser.add_argument("--patience", type=int, default=25)
    args = parser.parse_args()

    result = analyze_training_history(
        args.input,
        args.output,
        rolling_window=args.rolling_window,
        final_window=args.final_window,
        threshold_fraction=args.threshold_fraction,
        patience=args.patience,
    )
    print(result.to_string(index=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

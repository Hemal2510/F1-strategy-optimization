# Quantum RL Agent for F1 Race Strategy Optimization

## Overview
This project optimizes Formula 1 race strategies—such as pit stop timing, tyre compound selection, and pace management—using Reinforcement Learning. It compares a classical **Deep Q-Network (DQN)** agent against a **Hybrid Quantum Reinforcement Learning (QRL)** agent within a custom, data-driven F1 simulation environment built on real FastF1 telemetry and timing data.

---

## Tech Stack
- **Quantum Computing**: [PennyLane](https://pennylane.ai/) (Variational Quantum Circuits with `qml.qnn.TorchLayer`)
- **Deep Reinforcement Learning**: [PyTorch](https://pytorch.org/) (Double DQN, Dueling Architecture, Prioritized Experience Replay, Action Masking)
- **Environment & Data Engine**: Python 3.10+, [Gymnasium](https://gymnasium.farama.org/), [FastF1](https://fastf1.dev/), Pandas, NumPy
- **Dashboard & Visualization**: [Streamlit](https://streamlit.io/), Plotly, Matplotlib

---

## How to Run the Dashboard

The interactive Streamlit dashboard lets you visually analyze benchmark results, compare agents, and inspect step-by-step race replays.

To launch the dashboard:
```bash
streamlit run dashboard/Home.py
```

### Dashboard Pages:
1. **Overview Page (`1_Overview.py`)**: Visualizes aggregate performance metrics (Total Reward, Final Position, Position Gain, Pit Count, Lap Time) with 95% confidence intervals, overall win-rate breakdowns, and paired statistical agent comparisons.
2. **Lap Replay Page (`2_Lap_Replay.py`)**: Interactively steps through race episodes lap-by-lap to inspect driver position, gap to leader, tyre age, and pit stop calls.
3. **Strategy Divergence Page (`3_Strategy_Divergence.py`)**: Highlights specific laps where the QRL agent and DQN agent made differing strategic decisions.

---

## How to Run Evaluation & Benchmarking

You can evaluate trained models against baseline policies (Random, Always Stay Out, Rule-Aware Heuristic, and Real Driver Strategy):

### 1. Run Benchmark Pipeline
To execute automated evaluations across tracks and generate CSV artifacts:
```bash
python -c "from benchmarking.runner import run_benchmark; run_benchmark()"
```
*Artifacts will be saved under `benchmarking/artifacts/latest/` (`overall_metrics.csv`, `track_metrics.csv`, `paired_comparison.csv`, `lap_trace.csv`).*

### 2. Run Evaluation Scripts
- **Evaluate All Agents**:
  ```bash
  python evaluation/evaluate_all.py
  ```
- **Evaluate DQN Agent**:
  ```bash
  python evaluation/evaluate_dqn.py
  ```
- **Evaluate QRL Agent**:
  ```bash
  python evaluation/evaluate_qrl.py
  ```
- **Evaluate on 2025 Season (Not used in training)**:
  ```bash
  python evaluation/evaluate_tracks_2025.py
  ```
---

## Reports & Documentation

For comprehensive details on mathematical formulations, environment mechanics, quantum circuit architecture experiments (v6–v12), and detailed result analysis, please consult the report files in the [`reports/`](reports/) directory:

- **End Evaluation Report**: [`reports/end_eval_report_qc_3.pdf`](reports/end_eval_report_qc_3.pdf)
- **Mid Evaluation Report**: [`reports/mid_eval_report_qc_3.pdf`](reports/mid_eval_report_qc_3.pdf)

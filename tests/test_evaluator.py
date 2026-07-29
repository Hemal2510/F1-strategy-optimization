from types import SimpleNamespace

import numpy as np

from benchmarking.adapters import ActionDecision
from benchmarking.config import ScenarioSpec
from benchmarking.evaluator import run_episode


class DummyAdapter:
    name = "dqn"
    parameter_count = 10
    trainable_parameter_count = 10
    checkpoint_size_mb = 1.0
    checkpoint_sha256 = "abc"

    def decide(self, obs, action_mask):
        q_values = np.asarray([1.0, 0.5, 0.2, 0.1, -0.1, -0.2])
        legal = q_values.copy()
        legal[~action_mask] = -np.inf
        return ActionDecision(0, q_values, legal, 0.5)


class DummyEnv:
    def __init__(self):
        self.max_laps = 3
        self.track = "Test"
        self.year = 2025
        self.name = "Test GP"
        self.state = SimpleNamespace(
            current_lap=0,
            position=10,
            end_position=10,
            tyre_compound=1,
            tyre_age=3,
            track_wetness=0.0,
            safety_car_flag=False,
            pit_window=False,
            gap_to_leader=5.0,
            gap_ahead=1.0,
            gap_behind=1.0,
            lap_time=70.0,
            lap_delta=0.5,
        )

    def reset(self, seed=None, options=None):
        self.state.current_lap = 0
        self.state.position = 10
        self.state.end_position = 10
        return np.zeros(15, dtype=np.float32), {}

    def step(self, action):
        self.state.current_lap += 1
        self.state.tyre_age += 1
        terminated = self.state.current_lap >= self.max_laps
        if terminated:
            self.state.end_position = 9
            self.state.position = 9
        return np.ones(15, dtype=np.float32), 1.0, terminated, False, {}

    def close(self):
        pass


def mask_fn(env):
    return np.asarray([True, False, False, False, False, False])


def test_run_episode():
    summary, rows = run_episode(
        experiment_id="test",
        adapter=DummyAdapter(),
        env_factory=DummyEnv,
        base_env_kwargs={},
        scenario=ScenarioSpec(name="default"),
        action_mask_fn=mask_fn,
        action_names={0: "stay_out", 1: "pit_soft", 2: "pit_medium", 3: "pit_hard", 4: "pit_intermediate", 5: "pit_wet"},
        seed=42,
        max_steps=10,
        save_observations=True,
        warmup_inference_steps=0,
    )
    assert summary["completed"] is True
    assert summary["episode_steps"] == 3
    assert summary["total_reward"] == 3.0
    assert summary["final_position"] == 9
    assert len(rows) == 3
    assert rows[0]["mask_0"] is True
    assert rows[0]["action"] == 0

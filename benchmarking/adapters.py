from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from agents.dqn.dqn_agent import DQNAgent, DQNConfig
from agents.qrl.qrl_agent import QRLAgent, QRLConfig
from agents.baselines import AlwaysStayOutPolicy, RuleAwareHeuristicPolicy
from agents.real_agent import RealDriverPolicy


DQN_CHECKPOINT = "checkpoints/dqn/checkpoints_v2/best.pt"
QRL_CHECKPOINT = "checkpoints/qrl/checkpoints_qrl_v6/latest.pt"


@dataclass
class ActionDecision:
    """Common result returned by every adapter. Neural agents populate q_values; baselines return None."""
    action: int
    q_values: Optional[np.ndarray]


def _build_config(config_class: type, values: dict) -> Any:
    """Build a dataclass config using only fields accepted by that class."""
    accepted_names = {field.name for field in fields(config_class)}
    supported_values = {k: v for k, v in values.items() if k in accepted_names}
    return config_class(**supported_values)


class DQNAdapter:
    """Load the trained DQN and expose a simple evaluation interface."""

    def __init__(self, checkpoint_path: str, device: str, config_overrides: dict) -> None:
        self.name = "dqn"
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"DQN checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        saved_config = dict(checkpoint.get("config", {}))
        saved_config.update(config_overrides)
        saved_config["device"] = str(self.device)

        agent_config = _build_config(DQNConfig, saved_config)
        self.agent = DQNAgent(agent_config)
        self.agent.online_net.load_state_dict(checkpoint["online_net"], strict=True)
        self.agent.online_net.eval()

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.agent.online_net.parameters())

    @torch.no_grad()
    def select_action(self, observation: np.ndarray, action_mask: np.ndarray, env: Any) -> ActionDecision:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.agent.online_net(obs_tensor).squeeze(0)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        legal_q = q_values.masked_fill(~mask_tensor, -1e9)
        action = int(torch.argmax(legal_q).item())
        return ActionDecision(action=action, q_values=q_values.detach().cpu().numpy())


class QRLAdapter:
    """Load the trained hybrid QRL agent using the same adapter interface."""

    def __init__(self, checkpoint_path: str, device: str, config_overrides: dict) -> None:
        self.name = "qrl"
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"QRL checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        saved_config = dict(checkpoint.get("config", {}))
        saved_config.update(config_overrides)
        saved_config["device"] = str(self.device)

        agent_config = _build_config(QRLConfig, saved_config)
        self.agent = QRLAgent(agent_config)
        self.agent.online_net.load_state_dict(checkpoint["online_net"], strict=True)
        self.agent.online_net.eval()

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.agent.online_net.parameters())

    @torch.no_grad()
    def select_action(self, observation: np.ndarray, action_mask: np.ndarray, env: Any) -> ActionDecision:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.agent.online_net(obs_tensor).squeeze(0)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        legal_q = q_values.masked_fill(~mask_tensor, -1e9)
        action = int(torch.argmax(legal_q).item())
        return ActionDecision(action=action, q_values=q_values.detach().cpu().numpy())


class RandomAdapter:
    """Choose randomly from legal actions."""
    name = "random"
    checkpoint_path = None

    @property
    def parameter_count(self) -> int:
        return 0

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray, env: Any) -> ActionDecision:
        legal_actions = np.flatnonzero(action_mask)
        if legal_actions.size == 0:
            raise RuntimeError("No legal action available.")
        action = int(np.random.choice(legal_actions))
        return ActionDecision(action=action, q_values=None)


class AlwaysStayOutAdapter:
    """Never pits — action 0 every lap."""
    name = "always_stay_out"
    checkpoint_path = None

    @property
    def parameter_count(self) -> int:
        return 0

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray, env: Any) -> ActionDecision:
        return ActionDecision(action=0, q_values=None)


class RuleAwareHeuristicAdapter:
    """Pit strategy rule-based heuristic (compound rules + wet weather)."""
    name = "rule_aware_heuristic"
    checkpoint_path = None

    def __init__(self) -> None:
        self._policy = RuleAwareHeuristicPolicy()

    @property
    def parameter_count(self) -> int:
        return 0

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray, env: Any) -> ActionDecision:
        self._policy.reset() if not hasattr(self, "_ep_started") else None
        self._ep_started = True
        action = self._policy.act(env, observation)
        return ActionDecision(action=int(action), q_values=None)


class RealDriverAdapter:
    """Replays the actual driver's historical pit decisions from fastf1 data."""
    name = "real_driver"
    checkpoint_path = None

    def __init__(self) -> None:
        self._policy = RealDriverPolicy()

    @property
    def parameter_count(self) -> int:
        return 0

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray, env: Any) -> ActionDecision:
        action = self._policy.act(env, observation)
        return ActionDecision(action=int(action), q_values=None)


# Map of agent name → factory
_ADAPTER_REGISTRY: dict[str, Any] = {
    "dqn": lambda device: DQNAdapter(
        checkpoint_path=DQN_CHECKPOINT,
        device=device,
        config_overrides={"obs_dim": 15, "action_dim": 6},
    ),
    "qrl": lambda device: QRLAdapter(
        checkpoint_path=QRL_CHECKPOINT,
        device=device,
        config_overrides={"obs_dim": 15, "action_dim": 6, "n_qubits": 8, "device": device},
    ),
    "random": lambda device: RandomAdapter(),
    "always_stay_out": lambda device: AlwaysStayOutAdapter(),
    "rule_aware_heuristic": lambda device: RuleAwareHeuristicAdapter(),
    "real_driver": lambda device: RealDriverAdapter(),
}


def build_adapters(config: dict) -> list[Any]:
    """Create all enabled adapters from a benchmark config dict (CLI path)."""
    device = str(config["device"])
    agent_specs = config["agents"]
    adapters: list[Any] = []

    for name, factory in _ADAPTER_REGISTRY.items():
        spec = agent_specs.get(name, {})
        if spec.get("enabled", False):
            adapters.append(factory(device))

    return adapters


def build_adapters_from_list(agent_names: list[str], device: str = "cpu") -> list[Any]:
    """Create adapters for a given list of agent names (dashboard path)."""
    adapters: list[Any] = []
    for name in agent_names:
        if name not in _ADAPTER_REGISTRY:
            raise ValueError(f"Unknown agent: '{name}'. Available: {list(_ADAPTER_REGISTRY)}")
        adapters.append(_ADAPTER_REGISTRY[name](device))
    return adapters

    """
    Common result returned by every adapter.

    Neural agents return their original Q-values. Baseline agents return None.
    """
    action: int
    q_values: Optional[np.ndarray]


def _build_config(config_class: type, values: dict) -> Any:
    """
    Build a dataclass configuration using only fields accepted by that class.

    This lets us combine checkpoint settings with small JSON overrides.
    """
    accepted_names = {
        field.name
        for field in fields(config_class)
    }

    supported_values = {
        key: value
        for key, value in values.items()
        if key in accepted_names
    }

    return config_class(**supported_values)


class DQNAdapter:
    """Load the trained DQN and expose a simple evaluation interface."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str,
        config_overrides: dict,
    ) -> None:
        self.name = "dqn"
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"DQN checkpoint not found: {self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        saved_config = dict(checkpoint.get("config", {}))
        saved_config.update(config_overrides)
        saved_config["device"] = str(self.device)

        agent_config = _build_config(DQNConfig, saved_config)
        self.agent = DQNAgent(agent_config)

        self.agent.online_net.load_state_dict(
            checkpoint["online_net"],
            strict=True,
        )
        self.agent.online_net.eval()

    @property
    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.agent.online_net.parameters()
        )

    @torch.no_grad()
    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        env: Any,
    ) -> ActionDecision:
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        q_values = self.agent.online_net(
            observation_tensor
        ).squeeze(0)

        mask_tensor = torch.as_tensor(
            action_mask,
            dtype=torch.bool,
            device=self.device,
        )

        legal_q_values = q_values.masked_fill(
            ~mask_tensor,
            -1e9,
        )

        action = int(torch.argmax(legal_q_values).item())

        return ActionDecision(
            action=action,
            q_values=q_values.detach().cpu().numpy(),
        )


class QRLAdapter:
    """Load the trained hybrid QRL agent using the same adapter interface."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str,
        config_overrides: dict,
    ) -> None:
        self.name = "qrl"
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"QRL checkpoint not found: {self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        saved_config = dict(checkpoint.get("config", {}))
        saved_config.update(config_overrides)
        saved_config["device"] = str(self.device)

        agent_config = _build_config(QRLConfig, saved_config)
        self.agent = QRLAgent(agent_config)

        self.agent.online_net.load_state_dict(
            checkpoint["online_net"],
            strict=True,
        )
        self.agent.online_net.eval()

    @property
    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.agent.online_net.parameters()
        )

    @torch.no_grad()
    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        env: Any,
    ) -> ActionDecision:
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        q_values = self.agent.online_net(
            observation_tensor
        ).squeeze(0)

        mask_tensor = torch.as_tensor(
            action_mask,
            dtype=torch.bool,
            device=self.device,
        )

        legal_q_values = q_values.masked_fill(
            ~mask_tensor,
            -1e9,
        )

        action = int(torch.argmax(legal_q_values).item())

        return ActionDecision(
            action=action,
            q_values=q_values.detach().cpu().numpy(),
        )


class RandomAdapter:
    """Choose randomly from legal actions. Useful as a basic baseline."""

    def __init__(self) -> None:
        self.name = "random"
        self.checkpoint_path = None

    @property
    def parameter_count(self) -> int:
        return 0

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        env: Any,
    ) -> ActionDecision:
        legal_actions = np.flatnonzero(action_mask)

        if legal_actions.size == 0:
            raise RuntimeError("No legal action is available.")

        action = int(np.random.choice(legal_actions))
        return ActionDecision(action=action, q_values=None)


class RuleBasedAdapter:
    """
    A simple student-level rule-based strategy.

    It does not claim to be an optimal F1 strategy. It exists to provide a
    transparent non-learning baseline.
    """

    def __init__(self, tyre_age_threshold: int = 18) -> None:
        self.name = "rule_based"
        self.checkpoint_path = None
        self.tyre_age_threshold = int(tyre_age_threshold)

    @property
    def parameter_count(self) -> int:
        return 0

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        env: Any,
    ) -> ActionDecision:
        state = env.state

        wetness = float(getattr(state, "track_wetness", 0.0))
        tyre_age = int(getattr(state, "tyre_age", 0))
        current_lap = int(getattr(state, "current_lap", 0))
        max_laps = int(getattr(env, "max_laps", current_lap + 1))
        laps_remaining = max(0, max_laps - current_lap)

        if wetness >= 1.5:
            preferences = [5, 4, 0]
        elif wetness > 0:
            preferences = [4, 5, 0]
        elif tyre_age < self.tyre_age_threshold:
            preferences = [0]
        elif laps_remaining > 28:
            preferences = [3, 2, 1, 0]
        elif laps_remaining > 12:
            preferences = [2, 3, 1, 0]
        else:
            preferences = [1, 2, 3, 0]

        for action in preferences:
            if bool(action_mask[action]):
                return ActionDecision(
                    action=action,
                    q_values=None,
                )

        legal_actions = np.flatnonzero(action_mask)
        return ActionDecision(
            action=int(legal_actions[0]),
            q_values=None,
        )


def build_adapters(config: dict) -> list[Any]:
    """Create all enabled adapters from benchmark.json."""
    adapters: list[Any] = []
    device = str(config["device"])
    agent_specs = config["agents"]

    dqn_spec = agent_specs["dqn"]
    if dqn_spec.get("enabled", True):
        adapters.append(
            DQNAdapter(
                checkpoint_path=dqn_spec["checkpoint"],
                device=device,
                config_overrides=dqn_spec.get(
                    "config_overrides",
                    {},
                ),
            )
        )

    qrl_spec = agent_specs["qrl"]
    if qrl_spec.get("enabled", True):
        adapters.append(
            QRLAdapter(
                checkpoint_path=qrl_spec["checkpoint"],
                device=device,
                config_overrides=qrl_spec.get(
                    "config_overrides",
                    {},
                ),
            )
        )

    random_spec = agent_specs.get("random", {})
    if random_spec.get("enabled", False):
        adapters.append(RandomAdapter())

    rule_spec = agent_specs.get("rule_based", {})
    if rule_spec.get("enabled", False):
        adapters.append(
            RuleBasedAdapter(
                tyre_age_threshold=rule_spec.get(
                    "tyre_age_threshold",
                    18,
                )
            )
        )

    return adapters

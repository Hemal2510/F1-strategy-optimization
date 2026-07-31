from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import random
import numpy as np
import torch
import torch.nn.functional as F

from agents.qrl.hybrid_model import hybridQuantumNetwork
from agents.dqn.replay_buffer import PrioritizedReplayBuffer


@dataclass
class QRLConfig:
    obs_dim: int = 15
    action_dim: int = 6
    n_qubits: int = 8

    gamma: float = 0.99
    lr: float = 5e-3

    batch_size: int = 32
    replay_capacity: int = 100_000
    learning_starts: int = 2_000
    train_every: int = 1

    tau: float = 0.005
    gradient_clip: float = 10.0

    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 80_000

    reward_scale: float = 0.01

    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100_000

    seed: int = 42
    device: str = "cpu"


class QRLAgent:
    def __init__(self, config: QRLConfig):
        self.config = config
        self.device = torch.device(config.device)

        self._set_seeds(config.seed)

        self.online_net = hybridQuantumNetwork(config.obs_dim, config.action_dim, config.n_qubits).to(self.device)
        self.target_net = hybridQuantumNetwork(config.obs_dim, config.action_dim, config.n_qubits).to(self.device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(self.online_net.parameters(), lr=config.lr)

        self.replay = PrioritizedReplayBuffer(
            capacity=config.replay_capacity,
            obs_dim=config.obs_dim,
            alpha=config.per_alpha,
            beta_start=config.per_beta_start,
            beta_frames=config.per_beta_frames,
            device=str(self.device)
        )

        self.total_steps = 0

    def _set_seeds(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def epsilon(self) -> float:
        progress = min(1.0, self.total_steps / self.config.epsilon_decay_steps)
        epsilon = self.config.epsilon_start + progress * (
            self.config.epsilon_final - self.config.epsilon_start
        )
        return epsilon

    @torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        evaluation: bool = False,
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        epsilon = 0.0 if evaluation else self.epsilon()

        if random.random() < epsilon:
            if action_mask is None:
                return random.randrange(self.config.action_dim)
            allowed_actions = np.flatnonzero(action_mask)
            return int(np.random.choice(allowed_actions))

        obs_tensor = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        q_values = self.online_net(obs_tensor).squeeze(0)

        if action_mask is not None:
            mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
            q_values = q_values.masked_fill(~mask_tensor, -1e9)

        return int(torch.argmax(q_values).item())

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_action_mask: Optional[np.ndarray] = None,
    ) -> None:
        scaled_reward = reward * self.config.reward_scale
        self.replay.add(obs=obs, action=action, reward=scaled_reward, next_obs=next_obs, done=done, next_action_mask=next_action_mask)

    def train_step(self) -> Optional[Dict[str, Any]]:
        if len(self.replay) < self.config.learning_starts:
            return None

        if self.total_steps % self.config.train_every != 0:
            return None

        batch = self.replay.sample(self.config.batch_size)

        q_values = self.online_net(batch.obs)
        current_q = q_values.gather(dim=1, index=batch.actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_online = self.online_net(batch.next_obs)
            next_q_online = next_q_online.masked_fill(~batch.next_action_mask, -1e9)
            next_actions = next_q_online.argmax(dim=1)
            
            next_q = self.target_net(batch.next_obs).gather(
                dim=1, index=next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q = batch.rewards + self.config.gamma * (1.0 - batch.dones) * next_q

        td_errors = target_q - current_q

        elementwise_loss = F.smooth_l1_loss(current_q, target_q, reduction="none")
        loss = (elementwise_loss * batch.weights).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.gradient_clip)

        self.optimizer.step()

        self.replay.update_priorities(
            indices=batch.indices,
            td_errors=td_errors.detach().cpu().numpy(),
        )

        self._soft_update_target_network()

        return {
            "loss": float(loss.item()),
            "mean_q": float(current_q.detach().mean().item()),
            "mean_target_q": float(target_q.detach().mean().item()),
            "epsilon": float(self.epsilon()),
        }

    def _soft_update_target_network(self) -> None:
        with torch.no_grad():
            for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
                target_param.data.mul_(1.0 - self.config.tau)
                target_param.data.add_(self.config.tau * online_param.data)

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "config": self.config.__dict__,
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }

        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)

        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)

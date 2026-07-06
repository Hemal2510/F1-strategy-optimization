from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor
    weights: torch.Tensor
    indices: np.ndarray
    next_action_mask: torch.Tensor   # NEW


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        eps: float = 1e-6,
        device: str = "cpu",
        action_dim: int = 6,  # NEW
    ):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        self.device = device

        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

        self.priorities = np.zeros((self.capacity,), dtype=np.float32)

        self.pos = 0
        self.size = 0
        self.frame = 1

        self.next_action_masks = np.ones((self.capacity, self.action_dim), dtype=bool)  # NEW
        
    def __len__(self) -> int:
        return self.size

    def add(self, obs, action, reward, next_obs, done, next_action_mask):  # NEW arg
        if self.size == 0:
            max_priority = 1.0
        else:
            max_priority = self.priorities[:self.size].max()

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)

        self.priorities[self.pos] = max_priority

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        self.next_action_masks[self.pos] = next_action_mask  # NEW

    def _beta(self) -> float:
        progress = min(1.0, self.frame / self.beta_frames)
        beta = self.beta_start + progress * (1.0 - self.beta_start)

        self.frame += 1

        return beta

    def sample(self, batch_size: int) -> ReplayBatch:

        if self.size < batch_size:
            raise ValueError(
                f"Not enough samples in replay buffer. "
                f"Current size: {self.size}, batch_size: {batch_size}"
            )

        priorities = self.priorities[:self.size]

        scaled_priorities = priorities ** self.alpha
        probabilities = scaled_priorities / scaled_priorities.sum()

        indices = np.random.choice(
            self.size,
            size=batch_size,
            replace=False,
            p=probabilities,
        )

        beta = self._beta()

        weights = (self.size * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()

        batch = ReplayBatch(
            obs=torch.as_tensor(self.obs[indices], dtype=torch.float32, device=self.device),
            actions=torch.as_tensor(self.actions[indices], dtype=torch.long, device=self.device),
            rewards=torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
            next_obs=torch.as_tensor(self.next_obs[indices], dtype=torch.float32, device=self.device),
            dones=torch.as_tensor(self.dones[indices], dtype=torch.float32, device=self.device),
            weights=torch.as_tensor(weights, dtype=torch.float32, device=self.device),
            indices=indices,next_action_mask=torch.as_tensor(
                self.next_action_masks[indices], dtype=torch.bool, device=self.device
            ),
        )

        return batch

    def update_priorities(self, indices, td_errors):

        td_errors = np.asarray(td_errors, dtype=np.float32)

        new_priorities = np.abs(td_errors) + self.eps

        self.priorities[indices] = new_priorities
from pathlib import Path
from collections import deque
import numpy as np
import torch

from env.f1_env import F1StrategyEnv
from agents.dqn.dqn_agent import DQNAgent, DQNConfig
from agents.dqn.action_mask import get_action_mask


def set_global_seed(seed: int) -> None:

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_dqn(
    total_episodes: int = 1500,
    seed: int = 42,
    checkpoint_dir: str = "checkpoints/dqn/checkpoints_v1",
):


    set_global_seed(seed)

    env = F1StrategyEnv()

    config = DQNConfig(
        obs_dim=15,
        action_dim=6,

        gamma=0.99,
        lr=1e-4,

        batch_size=128,
        replay_capacity=100_000,
        learning_starts=2_000,
        train_every=1,

        tau=0.005,
        gradient_clip=10.0,

        epsilon_start=1.0,
        epsilon_final=0.05,
        epsilon_decay_steps=80_000,

        reward_scale=0.01,

        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_frames=100_000,

        seed=seed,
    )

    agent = DQNAgent(config)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    recent_returns = deque(maxlen=50)
    recent_positions = deque(maxlen=50)
    recent_pits = deque(maxlen=50)

    best_mean_return = -float("inf")

    for episode in range(1, total_episodes + 1):
        obs, info = env.reset(seed=seed + episode)

        done = False
        episode_return = 0.0
        episode_loss_values = []
        pit_count = 0

        while not done:
            agent_mask = get_action_mask(env)
            action = agent.select_action(obs, evaluation=False, action_mask = agent_mask,)

            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            if action != 0:
                pit_count += 1

            agent.store(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
            )

            agent.total_steps += 1

            metrics = agent.train_step()

            if metrics is not None:
                episode_loss_values.append(metrics["loss"])

            obs = next_obs
            episode_return += reward

        final_position = env.state.end_position

        recent_returns.append(episode_return)
        recent_positions.append(final_position)
        recent_pits.append(pit_count)

        mean_return_50 = float(np.mean(recent_returns))
        mean_position_50 = float(np.mean(recent_positions))
        mean_pits_50 = float(np.mean(recent_pits))

        mean_loss = float(np.mean(episode_loss_values)) if episode_loss_values else 0.0

        print(
            f"Episode {episode:04d} | "
            f"Return {episode_return:9.2f} | "
            f"Mean50 Return {mean_return_50:9.2f} | "
            f"Final P{final_position:02d} | "
            f"Mean50 Pos {mean_position_50:5.2f} | "
            f"Pits {pit_count:02d} | "
            f"Mean50 Pits {mean_pits_50:5.2f} | "
            f"Loss {mean_loss:.5f} | "
            f"Epsilon {agent.epsilon():.3f} | "
            f"Replay {len(agent.replay)} | "
            f"{env.track} {env.year} {env.name}"
        )

        if episode % 50 == 0:
            latest_path = checkpoint_dir / "latest.pt"
            agent.save(str(latest_path))
            print(f"Saved latest checkpoint: {latest_path}")

        if len(recent_returns) == recent_returns.maxlen:
            if mean_return_50 > best_mean_return:
                best_mean_return = mean_return_50
                best_path = checkpoint_dir / "best.pt"
                agent.save(str(best_path))
                print(f"Saved best checkpoint: {best_path} | Mean50 Return: {best_mean_return:.2f}")

    final_path = checkpoint_dir / "final.pt"
    agent.save(str(final_path))

    print(f"Training complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    train_dqn(
        total_episodes=1500,
        seed=42,
        checkpoint_dir="checkpoints/dqn/checkpoints_v1",
    )
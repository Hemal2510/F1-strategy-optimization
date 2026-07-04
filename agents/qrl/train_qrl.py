from pathlib import Path
from collections import deque
import numpy as np
import torch

from env.f1_env import F1StrategyEnv
from agents.qrl.qrl_agent import QRLAgent, QRLConfig
from agents.dqn.action_mask import get_action_mask

"""
Main QRL training loop for F1StrategyEnv.
 
This is a near-line-for-line mirror of train_dqn.py -- same environment,
same action masking, same seeding, same checkpointing/logging strategy.
The only swap is DQNAgent/DQNConfig -> QRLAgent/QRLConfig.
"""
"""
A note on runtime: quantum circuit simulation (default.qubit, 8 qubits)
is meaningfully slower per forward pass than the classical MLP. Before
launching a full 1500-episode run, do a short smoke test (e.g.
total_episodes=20) to confirm the pipeline runs end-to-end and to get a
feel for time-per-episode on your machine.
"""
def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_qrl(
    total_episodes: int = 1500,
    seed: int = 42,
    checkpoint_dir: str = "checkpoints/qrl/checkpoints_qrl_v5",
):
    set_global_seed(seed)
 
    env = F1StrategyEnv()
 
    config = QRLConfig(
        obs_dim=15,
        action_dim=6,
        n_qubits=8,
 
        gamma=0.99,
        lr=5e-3,
 
        batch_size=32,
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
 
    agent = QRLAgent(config)
 
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
 
    recent_returns = deque(maxlen=50)#stores rewards
    recent_positions = deque(maxlen=50)#stores final positions
    recent_pits = deque(maxlen=50)#stores pit stops
 
    best_mean_return = -float("inf")#best score so far
 
    for episode in range(1, total_episodes + 1):
        obs, info = env.reset(seed=seed + episode)
 
        done = False
        episode_return = 0.0#total reward for the episode
        episode_loss_values = []#stores loss values for the episode
        pit_count = 0#count of pit stops in the episode
 
        while not done:
            agent_mask = get_action_mask(env)
            action = agent.select_action(obs, evaluation=False, action_mask=agent_mask)
 
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
 
            if action != 0:
                pit_count += 1
 
            agent.store(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
            agent.total_steps += 1#increment total steps taken by the agent
 
            metrics = agent.train_step()#train the agent and get loss metrics
            if metrics is not None:
                episode_loss_values.append(metrics["loss"])
 
            obs = next_obs
            episode_return += reward
 
        final_position = env.state.end_position#final position of the agent at the end of the episode
 
        recent_returns.append(episode_return)
        recent_positions.append(final_position)
        recent_pits.append(pit_count)
 
        mean_return_50 = float(np.mean(recent_returns))#mean return over the last 50 episodes
        mean_position_50 = float(np.mean(recent_positions))#mean final position over the last 50 episodes
        mean_pits_50 = float(np.mean(recent_pits))#mean pit stops over the last 50 episodes
        mean_loss = float(np.mean(episode_loss_values)) if episode_loss_values else 0.0#mean loss over the episode
 
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
 
        #save best model based on mean return over the last 50 episodes
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
    # Smoke test first -- swap total_episodes back to 1500 once you've
    # confirmed it runs end-to-end and you have a sense of episode time.
    train_qrl(
        total_episodes=1500,
        seed=42,
        checkpoint_dir="checkpoints/qrl/checkpoints_qrl_v5",
    )
 
import numpy as np 

from env.f1_env import F1StrategyEnv
from agents.dqn.dqn_agent import DQNAgent, DQNConfig
from agents.dqn.action_mask import get_action_mask

def evaluate_checkpoint(
        checkpoint_path: str = "checkpoints/dqn/checkpoints_v2/final.pt",
        episodes: int = 20,
        seed: int = 10000,
):
    env = F1StrategyEnv()

    agent = DQNAgent(DQNConfig(obs_dim=15, action_dim=6, seed = seed,))
    agent.load(checkpoint_path)
    agent.online_net.eval()
    agent.target_net.eval()

    returns = []
    final_positions = []
    pit_counts = []
    

    action_names ={
        0: "stay out",
        1: "pit soft",
        2: "pit medium",
        3: "pit hard",
        4: "pit intermediate",
        5: "pit wet",
            
        }
    
    for episode in range(1, episodes + 1):
        obs,info = env.reset(seed = seed + episode)

        done = False
        episode_return = 0.0
        pit_count = 0
        actions = []

        while not done:
            action_mask = get_action_mask(env)
            action = agent.select_action(obs, evaluation= True, action_mask= action_mask)

            actions.append(action)

            if action != 0:
                pit_count += 1
            
            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            episode_return += reward
        
        returns.append(episode_return)
        final_positions.append(env.state.end_position)
        pit_counts.append(pit_count)

        print(
            f"Episode {episode:03d} |"
            f"Return {episode_return:.2f} |"
            f"Final P{env.state.end_position} |"
            f"Pits {pit_count} |"
            f"{env.track} {env.year} {env.name}"
        )

        print("Actions:", [action_names[a] for a in actions])

    print("\nSummary")
    print("----------")
    print("Checkpoint:", checkpoint_path)
    print("Mean return: ", np.mean(returns))
    print("Mean final position:", np.mean(final_positions))
    print("Mean pit count: ", np.mean(pit_counts))
    print("Best final position: ", np.min(final_positions))
    print("Worst final position: ", np.max(final_positions))

if __name__ == "__main__":
    evaluate_checkpoint(
        checkpoint_path="checkpoints/dqn/checkpoints_v2/final.pt",
        episodes = 20,
        seed = 10_000,
    )


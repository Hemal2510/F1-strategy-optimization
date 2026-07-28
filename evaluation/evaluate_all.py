import numpy as np

from env.f1_env import F1StrategyEnv
from agents.dqn.dqn_agent import DQNAgent, DQNConfig
from agents.qrl.qrl_agent import QRLAgent, QRLConfig
from agents.dqn.action_mask import get_action_mask
from agents.baselines import RandomPolicy, AlwaysStayOutPolicy, RuleAwareHeuristicPolicy
from agents.real_agent import RealDriverPolicy

# Checkpoint paths
BEST_DQN_CHECKPOINT = "checkpoints/dqn/checkpoints_v2/best.pt"
BEST_QRL_CHECKPOINT = "checkpoints/qrl/checkpoints_qrl_v6/latest.pt"

EPISODES = 20
SEED     = 10_000
N_QUBITS = 8

ACTION_NAMES = {
    0: "stay out",
    1: "pit soft",
    2: "pit medium",
    3: "pit hard",
    4: "pit intermediate",
    5: "pit wet",
}


class DQNPolicy:
    name = "DQN (best – v2/best.pt)"

    def __init__(self):
        config = DQNConfig(obs_dim=15, action_dim=6, seed=SEED)
        self.agent = DQNAgent(config)
        self.agent.load(BEST_DQN_CHECKPOINT)
        self.agent.online_net.eval()
        self.agent.target_net.eval()

    def reset(self):
        pass

    def act(self, env, obs):
        mask = get_action_mask(env)
        return self.agent.select_action(obs, evaluation=True, action_mask=mask)


class QRLPolicy:
    name = "QRL (best – v6/latest.pt)"

    def __init__(self):
        config = QRLConfig(obs_dim=15, action_dim=6, n_qubits=N_QUBITS, seed=SEED)
        self.agent = QRLAgent(config)
        self.agent.load(BEST_QRL_CHECKPOINT)
        self.agent.online_net.eval()
        self.agent.target_net.eval()

    def reset(self):
        pass

    def act(self, env, obs):
        mask = get_action_mask(env)
        return self.agent.select_action(obs, evaluation=True, action_mask=mask)


def evaluate_policy(policy, episodes=EPISODES, seed=SEED):
    env = F1StrategyEnv()
    returns, positions, pit_counts, violations, action_histories = [], [], [], [], []

    for ep in range(1, episodes + 1):
        policy.reset()
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        pits = 0
        actions = []

        while not done:
            action = policy.act(env, obs)
            actions.append(action)
            if action != 0:
                pits += 1
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward

        returns.append(ep_return)
        positions.append(env.state.end_position)
        pit_counts.append(pits)
        is_wet   = any(w > 0 for w in env.track_wetness)
        dry_used = env.compounds_used & {0, 1, 2}
        violations.append(int(not is_wet and len(dry_used) < 2))
        action_histories.append(actions)

    return {
        "policy":          policy.name,
        "mean_return":     float(np.mean(returns)),
        "mean_position":   float(np.mean(positions)),
        "best_position":   int(np.min(positions)),
        "worst_position":  int(np.max(positions)),
        "mean_pits":       float(np.mean(pit_counts)),
        "rule_violations": int(np.sum(violations)),
        "sample_actions":  action_histories[0],
    }


def print_summary(s):
    print(f"\n{'='*60}")
    print(f"  Policy : {s['policy']}")
    print(f"{'='*60}")
    print(f"  Mean return       : {s['mean_return']:.2f}")
    print(f"  Mean position     : {s['mean_position']:.2f}")
    print(f"  Best position     : {s['best_position']}")
    print(f"  Worst position    : {s['worst_position']}")
    print(f"  Mean pit stops    : {s['mean_pits']:.2f}")
    print(f"  Rule violations   : {s['rule_violations']}/{EPISODES}")
    sample = [ACTION_NAMES[a] for a in s["sample_actions"][:15]]
    print(f"  Sample actions    : {sample} ...")


def main():
    policies = [
        RandomPolicy(),
        AlwaysStayOutPolicy(),
        RuleAwareHeuristicPolicy(),
        DQNPolicy(),
        QRLPolicy(),
        RealDriverPolicy(),
    ]

    summaries = []
    for p in policies:
        print(f"\nEvaluating: {p.name} ...")
        s = evaluate_policy(p)
        print_summary(s)
        summaries.append(s)

    ranked = sorted(summaries, key=lambda x: x["mean_return"], reverse=True)
    print("\n\n" + "=" * 95)
    print("FINAL COMPARISON TABLE  (ranked by mean return, descending)")
    print("=" * 95)
    print(
        f"{'Policy':<35} | {'Mean Return':>11} | {'Mean Pos':>8} | "
        f"{'Best Pos':>8} | {'Worst Pos':>9} | {'Mean Pits':>9} | {'Rule Viol':>9}"
    )
    print("-" * 95)
    for s in ranked:
        print(
            f"{s['policy'][:35]:<35} | {s['mean_return']:11.2f} | "
            f"{s['mean_position']:8.2f} | {s['best_position']:8d} | "
            f"{s['worst_position']:9d} | {s['mean_pits']:9.2f} | "
            f"{s['rule_violations']:9d}"
        )
    print()


if __name__ == "__main__":
    main()
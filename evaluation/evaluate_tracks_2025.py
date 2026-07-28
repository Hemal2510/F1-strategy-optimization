import numpy as np

from env.f1_env import F1StrategyEnv
from agents.dqn.action_mask import get_action_mask

# Import all policy classes from evaluate_all to avoid duplication
from evaluation.evaluate_all import (
    RandomPolicy,
    AlwaysStayOutPolicy,
    RuleAwareHeuristicPolicy,
    DQNPolicy,
    QRLPolicy,
    RealDriverPolicy,
)

# ── Configuration ──────────────────────────────────────────────────────────────

YEAR    = 2025
SEED    = 42
TRACKS  = ["Monaco", "Monza", "Silverstone"]
DRIVERS = ["VER", "LEC", "HAM"]


# ══════════════════════════════════════════════════════════════════════════════
# Single-episode runner
# ══════════════════════════════════════════════════════════════════════════════

def run_single_episode(policy, track: str, year: int, driver: str, seed: int) -> dict:
    """
    Run one episode with the environment pinned to (track, year, driver).

    Returns
    -------
    dict with keys: policy, ep_return, start_pos, end_pos, pits, rule_violated
    """
    env = F1StrategyEnv()

    policy.reset()
    obs, _ = env.reset(
        seed=seed,
        options={"track": track, "year": year, "driver": driver},
    )

    # Capture grid / starting position right after reset (before any steps)
    start_pos = env.state.start_position

    done       = False
    ep_return  = 0.0
    pits       = 0

    while not done:
        action = policy.act(env, obs)
        if action != 0:
            pits += 1
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_return += reward

    end_pos = env.state.end_position

    is_wet        = any(w > 0 for w in env.track_wetness)
    dry_used      = env.compounds_used & {0, 1, 2}
    rule_violated = int(not is_wet and len(dry_used) < 2)

    return {
        "policy":        policy.name,
        "ep_return":     float(ep_return),
        "start_pos":     int(start_pos),
        "end_pos":       int(end_pos),
        "pits":          int(pits),
        "rule_violated": rule_violated,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Printing
# ══════════════════════════════════════════════════════════════════════════════

COL_W = 95  # total table width

def print_driver_table(track: str, driver: str, results: list):
    """Print a formatted results table for one (track, driver) combination."""
    header = f"  Track: {track}  |  Driver: {driver}  |  Year: {YEAR}  "
    print("\n\n" + "=" * COL_W)
    print(header.center(COL_W))
    print("=" * COL_W)
    print(
        f"{'Policy':<35} | {'Return':>9} | {'Start Pos':>9} | "
        f"{'End Pos':>7} | {'Pits':>4} | {'Rule Viol':>9}"
    )
    print("-" * COL_W)
    # Rank by return descending
    for r in sorted(results, key=lambda x: x["ep_return"], reverse=True):
        viol_str = "YES" if r["rule_violated"] else "no"
        print(
            f"{r['policy'][:35]:<35} | {r['ep_return']:9.2f} | "
            f"{r['start_pos']:9d} | {r['end_pos']:7d} | "
            f"{r['pits']:4d} | {viol_str:>9}"
        )
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def build_policies():
    return [
        RandomPolicy(),
        AlwaysStayOutPolicy(),
        RuleAwareHeuristicPolicy(),
        DQNPolicy(),
        QRLPolicy(),
        RealDriverPolicy(),
    ]


def main():
    print(f"\n{'='*COL_W}")
    print(f"  F1 STRATEGY EVALUATION  —  {YEAR} Season  "
          f"|  Drivers: {', '.join(DRIVERS)}".center(COL_W))
    print(f"{'='*COL_W}")

    for track in TRACKS:
        for driver in DRIVERS:
         
            policies = build_policies()   # fresh instances per (track, driver)
            results  = []

            for policy in policies:

                try:
                    r = run_single_episode(
                        policy, track, YEAR, driver,
                        seed=SEED,
                    )
                    results.append(r)
                except Exception as e:
                    results.append({
                        "policy":        policy.name,
                        "ep_return":     float("nan"),
                        "start_pos":     -1,
                        "end_pos":       -1,
                        "pits":          -1,
                        "rule_violated": 0,
                    })

            print_driver_table(track, driver, results)


if __name__ == "__main__":
    main()

# Baseline strategy policies for evaluation comparison

class RandomPolicy:
    # Randomly picks actions every lap (sanity check baseline)
    name = "Random"

    def reset(self):
        pass

    def act(self, env, obs):
        return env.action_space.sample()


class AlwaysStayOutPolicy:
    # Baseline that never pits
    name = "Always Stay Out"

    def reset(self):
        pass

    def act(self, env, obs):
        return 0


class RuleAwareHeuristicPolicy:
    # Simple rule-based heuristic: handles wet weather & compound rules
    name = "Rule Aware Heuristic"

    def __init__(self):
        self.last_pit_lap = -999

    def reset(self):
        self.last_pit_lap = -999

    def act(self, env, obs):
        cur = env.state.current_lap
        mx  = env.max_laps
        age = env.state.tyre_age
        cmp = env.state.tyre_compound
        wet = env.state.track_wetness
        remaining = mx - cur

        # Don't pit back-to-back
        if cur - self.last_pit_lap <= 3:
            return 0

        # Wet weather rules
        if wet >= 1.5 and cmp != 4:
            self.last_pit_lap = cur
            return 5  # wet tyre

        if 0 < wet < 1.5 and cmp != 3:
            self.last_pit_lap = cur
            return 4  # inter tyre

        if wet == 0:
            dry_used = env.compounds_used & {0, 1, 2}

            # Enforce 2 dry compound rule near end of race
            if len(dry_used) < 2 and remaining <= 8:
                self.last_pit_lap = cur
                return 3 if cmp != 2 else 2

            # Pit when tyres get old
            if cur >= int(0.45 * mx) and age >= 18:
                self.last_pit_lap = cur
                return 3 if cmp != 2 else 2

        return 0

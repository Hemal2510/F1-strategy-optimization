from data.real_strategy_data import RealStrategyLoader


class RealDriverPolicy:
    # Replays the actual driver's real pit decisions from fastf1 data
    name = "Real Driver"

    def __init__(self):
        self._loader  = RealStrategyLoader()
        self._actions = {}
        self._race_id = None

    def reset(self):
        self._actions = {}
        self._race_id = None

    def act(self, env, obs) -> int:
        race_id = (env.track, env.year, env.name)

        # Load data once per episode when race info changes
        if race_id != self._race_id:
            self._race_id = race_id
            track, year, driver = race_id
            try:
                self._actions = self._loader.get_driver_actions(track, year, driver)
            except Exception as e:
                print(f"[RealDriverPolicy] Could not load strategy for {driver} @ {track} {year}: {e}")
                self._actions = {}

        lap = env.state.current_lap
        return self._actions.get(lap, 0)

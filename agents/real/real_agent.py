
from env.data.real_strategy_data import RealStrategyLoader


class RealDriverPolicy:
  

    name = "Real Driver"

    def __init__(self):
        self._loader  = RealStrategyLoader()
        self._actions = {}          # {lap_int: action_int}  for current episode
        self._race_id = None        # (track, year, driver)  for current episode

   

    def reset(self):
        """
        Clear the cached action sequence.  The real sequence will be loaded
        lazily on the first ``act()`` call of the new episode (because env
        details like track/year/driver are only available after env.reset()).
        """
        self._actions = {}
        self._race_id = None

    def act(self, env, obs) -> int:
        """
        Return the action the real driver took on this lap.

        Parameters
        ----------
        env : F1StrategyEnv
            The live environment instance (used to read track, year, driver,
            and the current lap number).
        obs : np.ndarray
            Current observation (unused — the real driver ignores the obs).

        Returns
        -------
        int
            The action taken in real life, or 0 (stay out) if unknown.
        """
        race_id = (env.track, env.year, env.name)

        # Lazy-load whenever the race identity changes (once per episode)
        if race_id != self._race_id:
            self._race_id = race_id
            track, year, driver = race_id
            try:
                self._actions = self._loader.get_driver_actions(track, year, driver)
            except Exception as e:
                # Graceful fallback: if data fetch fails, always stay out
                print(f"[RealDriverPolicy] Could not load strategy for "
                      f"{driver} @ {track} {year}: {e}")
                self._actions = {}

        lap = env.state.current_lap
        return self._actions.get(lap, 0)

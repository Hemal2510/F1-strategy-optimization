"""
real_strategy_data.py
----------------------
Extracts the real pit-stop strategy (as action-space integers) for a specific
driver in a specific race using FastF1 lap data.

Action encoding (matches F1StrategyEnv):
    0 → stay out
    1 → pit → Soft      (compound id 0)
    2 → pit → Medium    (compound id 1)
    3 → pit → Hard      (compound id 2)
    4 → pit → Intermediate (compound id 3)
    5 → pit → Wet       (compound id 4)

A pit stop is detected when a lap's `PitInTime` is non-NaN.  The compound
fitted is read from the *following* lap's `Compound` column (the pit-out lap).

Usage:
    loader = RealStrategyLoader()
    actions = loader.get_driver_actions("Monaco", 2024, "VER")
    # → {1: 0, 2: 0, ..., 27: 2, ...}  (lap → action)
"""

import os
import pickle
import fastf1
import pandas as pd


COMPOUND_MAP = {
    "SOFT":         0,
    "MEDIUM":       1,
    "HARD":         2,
    "INTERMEDIATE": 3,
    "INTER":        3,
    "I":            3,
    "WET":          4,
    "HEAVY_WET":    4,
    "W":            4,
}

TRACKS = {
    "Monaco":      "Monaco Grand Prix",
    "Monza":       "Italian Grand Prix",
    "Silverstone": "British Grand Prix",
}


class RealStrategyLoader:
    """
    Loads and caches the real pit-stop action sequence for a driver in a race.

    Results are cached in ``env/data/processed_cache/real_{track}_{year}_{driver}.pkl``
    so that subsequent calls are instant (no FastF1 network/cache hit needed).
    """

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Reuse the same FastF1 cache that my_data.py uses
        fastf1_cache = os.path.join(current_dir, "my_cache")
        os.makedirs(fastf1_cache, exist_ok=True)   # create if missing
        fastf1.Cache.enable_cache(fastf1_cache)
        self._processed_cache_dir = os.path.join(current_dir, "processed_cache")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_driver_actions(self, track: str, year: int, driver: str) -> dict:
        """
        Return a dict mapping ``{lap_number (int): action (int)}`` for every
        lap of the race.  Missing laps default to 0 (stay out) when looked up.

        Parameters
        ----------
        track  : one of "Monaco", "Monza", "Silverstone"
        year   : calendar year (e.g. 2024)
        driver : 3-letter FastF1 abbreviation (e.g. "VER", "HAM")
        """
        cache_key = f"real_{track}_{year}_{driver}"
        cache_file = os.path.join(self._processed_cache_dir, f"{cache_key}.pkl")

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        actions = self._extract_actions(track, year, driver)

        os.makedirs(self._processed_cache_dir, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(actions, f)

        return actions

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_actions(self, track: str, year: int, driver: str) -> dict:
        """
        Core extraction logic.  Returns {lap_int: action_int}.
        """
        session = fastf1.get_session(year, TRACKS[track], "R")
        session.load(laps=True, telemetry=False, weather=False, messages=False)

        laps = session.laps.copy()
        driver_laps = laps[laps["Driver"] == driver].sort_values("LapNumber").reset_index(drop=True)

        if driver_laps.empty:
            # Driver not found — return empty dict (all laps → stay out)
            return {}

        max_lap = int(driver_laps["LapNumber"].max())
        actions = {}

        for idx, row in driver_laps.iterrows():
            lap_num = int(row["LapNumber"])
            pitted  = pd.notna(row["PitInTime"])

            if pitted:
                # The new compound is on the very next lap for this driver
                next_laps = driver_laps[driver_laps["LapNumber"] == lap_num + 1]
                if not next_laps.empty:
                    new_compound_str = str(next_laps.iloc[0]["Compound"]).upper().strip()
                    compound_id      = COMPOUND_MAP.get(new_compound_str, None)
                    if compound_id is not None:
                        actions[lap_num] = compound_id + 1  # 1-indexed pit action
                    else:
                        actions[lap_num] = 0  # unknown compound → stay out
                else:
                    actions[lap_num] = 0  # no next lap → stay out
            else:
                actions[lap_num] = 0  # no pit → stay out

        return actions

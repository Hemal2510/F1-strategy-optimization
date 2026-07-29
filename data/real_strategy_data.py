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
    # Extracts real pit stop sequence from FastF1 telemetry/laps

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fastf1_cache = os.path.join(current_dir, "fastf1_cache")
        os.makedirs(fastf1_cache, exist_ok=True)
        fastf1.Cache.enable_cache(fastf1_cache)
        self._processed_cache_dir = os.path.join(current_dir, "processed_cache")

    def get_driver_actions(self, track: str, year: int, driver: str) -> dict:
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

    def _extract_actions(self, track: str, year: int, driver: str) -> dict:
        session = fastf1.get_session(year, TRACKS[track], "R")
        session.load(laps=True, telemetry=False, weather=False, messages=False)

        laps = session.laps.copy()
        driver_laps = laps[laps["Driver"] == driver].sort_values("LapNumber").reset_index(drop=True)

        if driver_laps.empty:
            return {}

        max_lap = int(driver_laps["LapNumber"].max())
        actions = {}

        for idx, row in driver_laps.iterrows():
            lap_num = int(row["LapNumber"])
            pitted  = pd.notna(row["PitInTime"])

            if pitted:
                next_laps = driver_laps[driver_laps["LapNumber"] == lap_num + 1]
                if not next_laps.empty:
                    new_compound_str = str(next_laps.iloc[0]["Compound"]).upper().strip()
                    compound_id      = COMPOUND_MAP.get(new_compound_str, None)
                    if compound_id is not None:
                        actions[lap_num] = compound_id + 1
                    else:
                        actions[lap_num] = 0
                else:
                    actions[lap_num] = 0
            else:
                actions[lap_num] = 0

        return actions

import os
import pickle
from data.data import F1TrackDataLoader


class Env_data:
    def __init__(self, track, year):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(current_dir, "processed_cache")
        cache_file = os.path.join(cache_dir, f"{track}_{year}.pkl")
        
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.data = pickle.load(f)
        else:
            loader = F1TrackDataLoader()
            self.data = loader.get_race_data(track, year)
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(self.data, f)


from env.tyre_model import TyreModel
from env.traffic_model import TrafficModel
from env.pit_model import PitModel


class RaceBackend:
    def __init__(self, data):

        self.fastest_lap = float("inf")

        self.base_track_time = data["base_time"]
        # Stored to update field-fastest lap from historical drivers each step
        self.lap_times_dict = data["lap_times"]

        self.tyre_model = TyreModel(data)
        self.traffic_model = TrafficModel()
        self.pit_model  = PitModel(data)

    def simulated_lap_time(
        self,
        current_lap: int,
        tyre_compound: int,
        tyre_age: int,
        pitted: bool,
        gap_ahead: float,
        safety_car: int = 0,   # 0=green, 1=VSC, 2=full SC
        track_wetness: int = 0,
        noise: float = 0.0,
    ) -> tuple[float, float]:
        
        base_time = self.base_track_time[tyre_compound]

        # Fuel penalty removed: historical drivers' replayed lap times already embed
        # fuel-load effects from real race data. Adding a penalty only to the agent
        # creates an asymmetry that disadvantages it on early laps.

        tyre_degradation_penalty = self.tyre_model.degradation(
            tyre_compound,
            tyre_age
        )

        # If Intermediate/Wet tyres are run on a dry track, their degradation penalty increases dramatically
        if track_wetness == 0 and tyre_compound in [3, 4]:
            tyre_degradation_penalty *= 4.0

        # Determine weather mismatch penalties
        # 0: DRY, 1: DAMP, 2: WET
        weather_penalty = 0.0
        if track_wetness == 0:  # Dry track
            if tyre_compound == 3:  # Intermediate on dry
                weather_penalty = 8.0
            elif tyre_compound == 4:  # Wet on dry
                weather_penalty = 15.0
        elif track_wetness == 1:  # Damp track
            if tyre_compound in [0, 1, 2]:  # Dry tyres on damp
                weather_penalty = 15.0
            elif tyre_compound == 4:  # Wet on damp
                weather_penalty = 4.0
        elif track_wetness == 2:  # Wet track
            if tyre_compound in [0, 1, 2]:  # Dry tyres on wet
                weather_penalty = 40.0
            elif tyre_compound == 3:  # Intermediate on wet
                weather_penalty = 5.0

        pit_loss = 0.0
        if pitted: pit_loss = self.pit_model.get_loss(safety_car)



        traffic_loss = self.traffic_model.get_traffic_loss(gap_ahead)

        lap_time = (
            base_time
            + tyre_degradation_penalty
            + pit_loss
            + traffic_loss
            + weather_penalty
            + noise
        )

        if safety_car == 2:    # Full Safety Car: ~35% slower (pitlane speed limiter pace)
            lap_time *= 1.35
        elif safety_car == 1:  # Virtual Safety Car: ~18% slower (no overtaking, slow zones)
            lap_time *= 1.18

     
        hist_times = self.lap_times_dict.get(current_lap, {})
        for t in hist_times.values():
            if 60.0 < t < 200.0:
                self.fastest_lap = min(self.fastest_lap, t)
        if not pitted:
        
            self.fastest_lap = min(self.fastest_lap, lap_time)

        lap_delta = lap_time - self.fastest_lap

        return lap_time, lap_delta
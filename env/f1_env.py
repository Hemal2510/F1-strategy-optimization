import gymnasium as gym
import numpy as np

from env.race_state import RaceState
from env.reward import RewardCalculator
from env.race_backend import RaceBackend
from env.race_session import RaceSession
from data.data_for_env import Env_data


class F1StrategyEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        self.reward_calc = RewardCalculator()
        self.available_races = (
            ("Monaco", 2024), ("Monza", 2024), ("Silverstone", 2024),
            ("Monaco", 2023), ("Monza", 2023), ("Silverstone", 2023),
            ("Monaco", 2022), ("Monza", 2022), ("Silverstone", 2022)
        )

        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(15,), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options:
            self.track = options.get("track")
            self.year  = options.get("year")
            self.name  = options.get("driver", None)
        else:
            idx = int(self.np_random.integers(0, len(self.available_races)))
            self.track, self.year = self.available_races[idx]
            self.name = None  

        self.env_data = Env_data(self.track, self.year)
        self.race_backend = RaceBackend(self.env_data.data)
        self.race_session = RaceSession(self.env_data.data)

        self.max_laps       = self.env_data.data["max_laps"]
        self.total_players  = self.env_data.data["total_drivers"]
        self.safety_car_times = self.env_data.data["safety_car"]
        self.track_wetness  = self.env_data.data["track_wetness"]

        if self.name is None:
            lap_times_data  = self.env_data.data["lap_times"]
            starting_grid   = self.env_data.data["starting_grid"]
            min_laps        = int(self.max_laps * 0.95)
            eligible = [
                d for d in starting_grid
                if sum(
                    1 for lap in range(1, self.max_laps + 1)
                    if d in lap_times_data.get(lap, {})
                ) >= min_laps
            ]
            if not eligible:
                eligible = starting_grid
            idx = int(self.np_random.integers(0, len(eligible)))
            self.name = eligible[idx]

        # Starting compound from race data
        starting_compounds  = self.env_data.data.get("starting_compounds", {})
        initial_tyre_compound = starting_compounds.get(self.name, 1)
        self.compounds_used = {initial_tyre_compound}

        self.race_session.step(0, 0, self.name)
        arranged_cars = self.race_session.get_agent_state(self.name)
        gap_ahead = arranged_cars['gap_ahead']

        # Lap 1 simulation
        noise = self.np_random.normal(0, 0.15)
        lap_time, lap_delta = self.race_backend.simulated_lap_time(
            1,
            initial_tyre_compound,
            1,
            False,
            gap_ahead,
            self.safety_car_times[1],
            track_wetness=self.track_wetness[1],
            noise=noise
        )

        self.race_session.step(1, lap_time, self.name)
        initial_state = self.race_session.get_agent_state(self.name)

        self.state = RaceState(
            current_lap=1,
            lap_time=lap_time,
            lap_delta=lap_delta,
            start_position=arranged_cars["current_position"],
            end_position=initial_state['current_position'],
            tyre_compound=initial_tyre_compound,
            tyre_age=1,
            gap_leader=initial_state["gap_leader"],
            gap_ahead=initial_state["gap_ahead"],
            gap_behind=initial_state["gap_behind"],
            safety_car=self.safety_car_times[1],
            track_wetness=self.track_wetness[1],
        )
        return self._get_obs(), {}

    def _get_obs(self):
        tyre = [0, 0, 0, 0, 0]
        tyre[self.state.tyre_compound] = 1

        compound    = self.state.tyre_compound
        current_deg = self.race_backend.tyre_model.degradation(compound, self.state.tyre_age)
        curve       = self.race_backend.tyre_model.degradation_curves[compound]
        ref_age     = min(max(curve.keys()), self.max_laps // 2)
        ref_deg     = curve[ref_age]
        tyre_wear   = min(current_deg / max(ref_deg, 0.01), 1.0)

        # Check 2 dry compound rule
        dry_compounds_used = self.compounds_used & {0, 1, 2}
        compounds_rule_met = float(len(dry_compounds_used) >= 2)

        obs = np.array([
            self.state.current_lap / self.max_laps,
            self.state.end_position / self.total_players,
            max(0.0, 1.0 - self.state.lap_delta / 10.0),
            tyre[0],
            tyre[1],
            tyre[2],
            tyre[3],
            tyre[4],
            tyre_wear,
            min(self.state.gap_leader / 40, 1),
            min(self.state.gap_ahead / 10, 1),
            min(self.state.gap_behind / 10, 1),
            self.state.safety_car / 2.0,
            self.state.track_wetness / 2.0,
            compounds_rule_met,
        ], dtype=np.float32)

        return obs

    def step(self, action):
        pitted = False
        previous_tyre_age = self.state.tyre_age

        if action != 0:
            pitted = True
            self.state.tyre_compound = action - 1
            self.state.tyre_age = 0
            self.compounds_used.add(self.state.tyre_compound)
        
        self.state.current_lap += 1
        self.state.tyre_age += 1
        self.state.safety_car = self.safety_car_times[self.state.current_lap]
        track_wetness = self.track_wetness[self.state.current_lap]
        self.state.track_wetness = track_wetness

        noise = self.np_random.normal(0, 0.15)

        lap_time, lap_delta = self.race_backend.simulated_lap_time(
            self.state.current_lap,
            self.state.tyre_compound,
            self.state.tyre_age,
            pitted,
            self.state.gap_ahead,
            self.state.safety_car,
            track_wetness=track_wetness,
            noise=noise
        )

        self.race_session.step(self.state.current_lap, lap_time, self.name)
        state_now = self.race_session.get_agent_state(agent_name=self.name)

        prev_gap_leader = self.state.gap_leader

        self.state.gap_leader = state_now['gap_leader']
        self.state.gap_ahead = state_now['gap_ahead']
        self.state.gap_behind = state_now['gap_behind']
        self.state.start_position = self.state.end_position
        self.state.end_position = state_now['current_position']
        
        self.state.lap_time = lap_time
        self.state.lap_delta = lap_delta

        terminated = False
        if self.state.current_lap >= self.max_laps:
            terminated = True

        pit_loss = self.race_backend.pit_model.get_loss(self.state.safety_car) if pitted else 0.0
        reward = self.reward_calc.compute(
            self.state.lap_delta,
            self.state.start_position,
            self.state.end_position,
            self.state.gap_leader,
            prev_gap_leader,
            pitted=pitted,
            pit_loss=pit_loss
        )
        if pitted:
            reward -= 5.0
            if previous_tyre_age <= 5:
                reward -= 20.0

        # Enforce 2 dry compound rule at end of race
        if terminated:
            is_wet_race = any(w > 0 for w in self.track_wetness)
            dry_used = self.compounds_used & {0, 1, 2}
            if not is_wet_race and len(dry_used) < 2:
                reward -= 150.0

            F1_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
            reward += F1_POINTS.get(self.state.end_position, 0)

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        tyre_chart = {
            0: 'soft',
            1: 'medium',
            2: 'hard',
            3: 'intermediate',
            4: 'wet'
        }
        print(f"""
Lap: {self.state.current_lap}
Lap Time: {self.state.lap_time}
Start Position: {self.state.start_position}
End Position: {self.state.end_position}
Tyre: {tyre_chart[self.state.tyre_compound]}
Tyre Age: {self.state.tyre_age}
Gap Leader: {self.state.gap_leader:.3f}
""")

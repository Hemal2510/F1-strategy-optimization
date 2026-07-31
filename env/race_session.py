class Driver:
    def __init__(self, name):
        self.name = name
        self.total_race_time = 0.0
        self.current_position = 0
        
        # State variables for the RL Agent
        self.gap_to_leader = 0.0
        self.gap_ahead = float('inf')  # Infinity if P1
        self.gap_behind = float('inf') # Infinity if P20

class RaceSession:
    def __init__(self, data):

        self.lap_times_dict = data["lap_times"]

        starting_grid_list = data["starting_grid"]
        grid_stagger_time = 1e-5
        self.drivers = []

        for grid_pos, name in enumerate(starting_grid_list):
            driver = Driver(name)
            
            # Apply the grid stagger so the initial sort works perfectly
            driver.total_race_time = grid_pos * grid_stagger_time
            self.drivers.append(driver)
        
      

    def step(self, lap_num , agent_time , agent_name):
        self.lap_times_dict[lap_num][agent_name]= agent_time

        # 1. Update Total Race Time
        for driver in self.drivers:
            if (driver.name in self.lap_times_dict[lap_num]):
                driver.total_race_time += self.lap_times_dict[lap_num][driver.name]
            else:
                driver.total_race_time += 999

               

        # 2. Sort the grid based on the lowest total time
        self.drivers.sort(key=lambda d: d.total_race_time)

        # 3. Recalculate Positions and Gaps
        leader_time = self.drivers[0].total_race_time

        for i, driver in enumerate(self.drivers):
            # Update Position (Index + 1)
            driver.current_position = i + 1
            
            # Gap to Leader
            driver.gap_to_leader = driver.total_race_time - leader_time

            # Gap Ahead
            if i == 0:
                driver.gap_ahead = float('inf') # P1 has free air
            else:
                time_ahead = self.drivers[i-1].total_race_time
                driver.gap_ahead = driver.total_race_time - time_ahead

            # Gap Behind
            if i == len(self.drivers) - 1:
                driver.gap_behind = float('inf') # Last place has no one behind
            else:
                time_behind = self.drivers[i+1].total_race_time
                driver.gap_behind = time_behind - driver.total_race_time

    def get_grid_state(self):
        """Helper to view the current standings"""
        standings = []
        for d in self.drivers:
            standings.append({
                "Pos": d.current_position,
                "Driver": d.name,
                "Gap Ahead": round(d.gap_ahead, 3) if d.gap_ahead != float('inf') else "-",
                "Gap Leader": round(d.gap_to_leader, 3)
            })
        return standings
    
    def get_agent_state(self, agent_name):
        agent_driver = next((d for d in self.drivers if d.name == agent_name), None)
        
        if not agent_driver:
            raise ValueError(f"Driver '{agent_name}' not found in the simulation.")
        
        # Cap infinity to a max value (e.g., 99.9) to prevent neural network NaN errors
        max_gap_cap = 99.9 
        
        gap_ahead = agent_driver.gap_ahead if agent_driver.gap_ahead != float('inf') else max_gap_cap
        gap_behind = agent_driver.gap_behind if agent_driver.gap_behind != float('inf') else max_gap_cap

        return {
            "current_position": agent_driver.current_position,
            "gap_leader": agent_driver.gap_to_leader,
            "gap_ahead": gap_ahead,
            "gap_behind": gap_behind
        }
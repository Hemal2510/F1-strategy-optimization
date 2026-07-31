from dataclasses import dataclass

@dataclass
class RaceState:
    current_lap: int
    lap_time: float
    lap_delta : float

    start_position : int
    end_position: int

    tyre_compound : int
    tyre_age: int
    
    gap_leader : float
    gap_ahead: float
    gap_behind: float

    safety_car: int = 0   # 0=Green, 1=Virtual Safety Car, 2=Full Safety Car
    track_wetness: int = 0   # 0=DRY, 1=DAMP, 2=WET

    
   

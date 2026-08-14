from __future__ import annotations

import sys
import pickle
import numpy as np
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Allow importing modules from F1-strategy-showcase root
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.append("C:\\Users\\hemal\\AppData\\Roaming\\Python\\Python312\\site-packages")

from env.f1_env import F1StrategyEnv
from agents.dqn.action_mask import get_action_mask
from agents.real_agent import RealDriverPolicy
from model_runner import LiveModelRunner
from data.data import F1TrackDataLoader

app = FastAPI(title="F1 Reinforcement Learning Strategy Showcase API")

# Enable CORS for any origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model runner
try:
    runner = LiveModelRunner(device="cpu")
    print("[SUCCESS] Trained DQN & QRL models loaded successfully.")
except Exception as e:
    runner = None
    print(f"[WARNING] Could not load DQN/QRL checkpoints: {e}")

# Request/Response Schemas
class RaceLoadRequest(BaseModel):
    track: str
    year: int
    driver: str

class BranchRequest(BaseModel):
    track: str
    year: int
    driver: str
    branch_lap: int
    trials: int = 20

@app.get("/api/races")
def get_races():
    """
    Crawls data/processed_cache and reads available years, tracks, and drivers.
    Returns drivers sorted by lower finishing position first for showcase convenience.
    """
    cache_dir = ROOT_DIR / "data" / "processed_cache"
    if not cache_dir.exists():
        return {"races": []}
    
    races_dict = {}
    
    # Process all Track_Year.pkl files
    for pkl_file in cache_dir.glob("*.pkl"):
        if pkl_file.name.startswith("real_"):
            continue
        try:
            name_parts = pkl_file.stem.split("_")
            if len(name_parts) != 2:
                continue
            track, year_str = name_parts
            year = int(year_str)
            
            with open(pkl_file, "rb") as f:
                race_data = pickle.load(f)
            
            # Find starting grid and check if we have their lap times
            # We want to match real-world final positions if available
            # Let's inspect final standings from the last lap in race_data
            starting_grid = race_data.get("starting_grid", [])
            max_laps = race_data.get("max_laps", 70)
            
            # Simple pre-compiled finishing position mapping for the showcased races to make dropdown load instantly
            # Format: { (track, year): { driver: position } }
            FINISHING_STANDINGS_MAP = {
                ("Monaco", 2024): {
                    "LEC": 1, "PIA": 2, "SAI": 3, "NOR": 4, "RUS": 5, "VER": 6, "HAM": 7, "TSU": 8, "ALB": 9, "GAS": 10,
                    "ALO": 11, "RIC": 12, "BOT": 13, "STR": 14, "SAR": 15, "ZHO": 16, "OCO": 20, "PER": 20, "HUL": 20, "MAG": 20
                },
                ("Monaco", 2023): {
                    "VER": 1, "ALO": 2, "OCO": 3, "HAM": 4, "RUS": 5, "LEC": 6, "GAS": 7, "SAI": 8, "NOR": 9, "PIA": 10,
                    "VAL": 11, "dev": 12, "ZHO": 13, "SAR": 14, "MAG": 15, "ALB": 16, "PER": 17, "HUL": 18, "STR": 20, "TSU": 20
                },
                ("Monza", 2024): {
                    "LEC": 1, "PIA": 2, "NOR": 3, "SAI": 4, "HAM": 5, "VER": 6, "RUS": 7, "PER": 8, "ALB": 9, "MAG": 10,
                    "RIC": 11, "COL": 12, "VAL": 13, "ZHO": 14, "BOT": 15, "TSU": 20, "HUL": 17, "GAS": 18, "STR": 19, "SAR": 20
                },
                ("Silverstone", 2024): {
                    "HAM": 1, "VER": 2, "NOR": 3, "PIA": 4, "SAI": 5, "HUL": 6, "STR": 7, "ALB": 8, "TSU": 9, "ALO": 10,
                    "SAU": 11, "SAR": 12, "MAG": 13, "RIC": 14, "LEC": 15, "VAL": 16, "OCO": 17, "ZHO": 18, "RUS": 20, "GAS": 20
                }
            }

            classified_positions = FINISHING_STANDINGS_MAP.get((track, year), {})

            drivers_list = []
            for pos_idx, d in enumerate(starting_grid):
                # Fallback to starting grid position if not in map
                pos = classified_positions.get(d, pos_idx + 1)
                drivers_list.append({
                    "driver_id": d,
                    "name": d,
                    "final_position": pos
                })
            
            # Sort drivers list by final_position descending (lower finishing positions first as requested)
            drivers_list.sort(key=lambda x: x["final_position"], reverse=True)

            if year not in races_dict:
                races_dict[year] = []
            
            races_dict[year].append({
                "track": track,
                "max_laps": max_laps,
                "drivers": drivers_list
            })
        except Exception as e:
            print(f"Error parsing cache file {pkl_file}: {e}")
            continue

    return races_dict

@app.post("/api/run-race")
def run_race(req: RaceLoadRequest):
    """
    Executes the entire race lap-by-lap.
    At every lap, obtains the state from Gymnasium env, performs live DQN/QRL inference,
    records historical driver actions, and advances using the historical action.
    """
    if runner is None:
        raise HTTPException(status_code=500, detail="Models not loaded. Check server logs.")
        
    env = F1StrategyEnv()
    try:
        obs, info = env.reset(options={"track": req.track, "year": req.year, "driver": req.driver})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to reset environment: {e}")
        
    real_policy = RealDriverPolicy()
    real_policy.reset()
    
    laps_data = []
    done = False
    
    # Store initial state info
    starting_pos = env.state.start_position
    
    # Run the race lap-by-lap
    while not done:
        current_lap = env.state.current_lap
        action_mask = get_action_mask(env)
        
        # Get historical/real action
        real_action = real_policy.act(env, obs)
        
        # Get live model decisions on the EXACT same state
        decisions = runner.select_actions(obs, action_mask)
        
        # Capture state telemetry before stepping
        state_snapshot = {
            "lap": current_lap,
            "position": env.state.end_position,
            "tyre_compound": env.state.tyre_compound,
            "tyre_age": env.state.tyre_age,
            "gap_leader": env.state.gap_leader,
            "gap_ahead": env.state.gap_ahead,
            "gap_behind": env.state.gap_behind,
            "safety_car": env.state.safety_car,
            "track_wetness": env.state.track_wetness,
            "lap_time": env.state.lap_time,
            "lap_delta": env.state.lap_delta,
            "real_action": real_action,
            "dqn_action": decisions["dqn"]["action"],
            "dqn_q_values": decisions["dqn"]["q_values"],
            "qrl_action": decisions["qrl"]["action"],
            "qrl_q_values": decisions["qrl"]["q_values"],
            "action_mask": action_mask.tolist()
        }
        laps_data.append(state_snapshot)
        
        # Advance the environment using the HISTORICAL strategy so we follow real-life path
        obs, reward, terminated, truncated, info = env.step(real_action)
        done = terminated or truncated
        
    # Set final position of real life
    final_pos = env.state.end_position
    
    return {
        "starting_position": starting_pos,
        "final_position": final_pos,
        "laps": laps_data
    }

@app.post("/api/branch-simulation")
def branch_simulation(req: BranchRequest):
    """
    Branches from branch_lap. Executes configurable N Monte Carlo stochastic trials
    under:
    - Real Strategy (historical driver strategy remaining)
    - DQN Strategy (models make active decisions)
    - QRL Strategy (models make active decisions)
    Returns comparative stats.
    """
    if runner is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")
        
    # Helper function to run a single continuation trial from the branched state
    def run_continuation(strategy_name: str, branch_lap: int) -> int:
        # Re-create environment and run up to branch_lap
        sim_env = F1StrategyEnv()
        obs, info = sim_env.reset(options={"track": req.track, "year": req.year, "driver": req.driver})
        
        real_p = RealDriverPolicy()
        real_p.reset()
        
        # Fast-forward to branch_lap using real actions
        for _ in range(1, branch_lap):
            act = real_p.act(sim_env, obs)
            obs, _, term, trunc, _ = sim_env.step(act)
            if term or trunc:
                return sim_env.state.end_position
                
        # Now, branch action selection starts at branch_lap
        done = False
        while not done:
            action_mask = get_action_mask(sim_env)
            if strategy_name == "real":
                act = real_p.act(sim_env, obs)
            elif strategy_name == "dqn":
                act = runner.dqn_agent.select_action(obs, evaluation=True, action_mask=action_mask)
            elif strategy_name == "qrl":
                act = runner.qrl_agent.select_action(obs, evaluation=True, action_mask=action_mask)
            else:
                act = 0
                
            obs, _, term, trunc, _ = sim_env.step(act)
            done = term or trunc
            
        return sim_env.state.end_position

    real_finishes = []
    dqn_finishes = []
    qrl_finishes = []
    
    # Run N trials
    for _ in range(req.trials):
        real_finishes.append(run_continuation("real", req.branch_lap))
        dqn_finishes.append(run_continuation("dqn", req.branch_lap))
        qrl_finishes.append(run_continuation("qrl", req.branch_lap))
        
    return {
        "real": {
            "avg_finish": float(np.mean(real_finishes)),
            "finishes": real_finishes,
        },
        "dqn": {
            "avg_finish": float(np.mean(dqn_finishes)),
            "finishes": dqn_finishes,
        },
        "qrl": {
            "avg_finish": float(np.mean(qrl_finishes)),
            "finishes": qrl_finishes,
        }
    }

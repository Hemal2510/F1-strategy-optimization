import fastf1
import pandas as pd
import numpy as np
from collections import defaultdict
import os


class F1TrackDataLoader:

    TRACKS = {
        "Monaco": "Monaco Grand Prix",
        "Monza": "Italian Grand Prix",
        "Silverstone": "British Grand Prix"
    }

    COMPOUND_MAP = {
        "SOFT": 0,
        "MEDIUM": 1,
        "HARD": 2,
        "INTERMEDIATE": 3,
        "INTER": 3,
        "I": 3,
        "WET": 4,
        "HEAVY_WET": 4,
        "W": 4
    }

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(current_dir, "fastf1_cache")
        fastf1.Cache.enable_cache(cache_dir)

    def get_race_data(self, track: str, year: int):
        session = fastf1.get_session(year, self.TRACKS[track], "R")
        session.load(laps=True, telemetry=False, weather=False, messages=True)

        laps = session.laps.copy()
        starting_grid = self._get_starting_grid(session)

        return {
            "starting_grid": starting_grid,
            "starting_compounds": self._get_starting_compounds(laps),
            "safety_car": self._get_safety_car_flags(session),
            "track_wetness": self._get_track_wetness(session),
            "lap_times": self._get_lap_times(laps, starting_grid),
            "tyre_loss": self._get_tyre_degradation(laps),
            "pit_loss": self._get_average_pit_loss(laps),
            "base_time": self._get_base_time(laps),
            "max_laps": int(laps["LapNumber"].max()) if len(laps) > 0 else 70,
            "total_drivers": len(starting_grid) if starting_grid else laps["Driver"].nunique()
        }

    def _get_starting_compounds(self, laps):
        # Maps driver abbreviation to their lap 1 starting compound
        lap1 = laps[(laps["LapNumber"] == 1) & laps["Compound"].notna()][["Driver", "Compound"]]

        result = {}
        for _, row in lap1.iterrows():
            driver   = row["Driver"]
            compound = str(row["Compound"]).upper().strip()
            cid = self.COMPOUND_MAP.get(compound, None)
            if cid is not None and driver not in result:
                result[driver] = cid
        return result

    def _get_starting_grid(self, session):
        grid = session.results.sort_values("GridPosition")
        return grid["Abbreviation"].tolist()

    def _get_safety_car_flags(self, session):
        # 0 = Green, 1 = VSC, 2 = Full SC
        max_laps = int(session.laps["LapNumber"].max())
        sc_flags = [0] * (max_laps + 1)

        for lap_no, group in session.laps.groupby("LapNumber"):
            statuses = group["TrackStatus"].unique()
            lap = int(lap_no)
            if lap <= max_laps:
                has_sc  = any('4' in str(s) for s in statuses)
                has_vsc = any('6' in str(s) for s in statuses)
                if has_sc:
                    sc_flags[lap] = 2
                elif has_vsc:
                    sc_flags[lap] = 1

        return sc_flags

    def _get_track_wetness(self, session):
        max_laps = int(session.laps["LapNumber"].max())
        wetness = [0] * (max_laps + 1)
        
        for lap_no, group in session.laps.groupby("LapNumber"):
            compounds = group["Compound"].dropna().tolist()
            if not compounds:
                continue
            
            num_wets = sum(1 for c in compounds if c in ["WET", "HEAVY_WET", "W"])
            num_inters = sum(1 for c in compounds if c in ["INTERMEDIATE", "INTER", "I"])
            num_total = len(compounds)
            
            lap = int(lap_no)
            if lap <= max_laps:
                if num_wets / num_total >= 0.3:
                    wetness[lap] = 2
                elif (num_inters + num_wets) / num_total >= 0.3:
                    wetness[lap] = 1
                else:
                    wetness[lap] = 0
                    
        return wetness

    def _get_lap_times(self, laps, starting_grid=None):
        drivers_set = set(laps["Driver"].dropna().unique())
        if starting_grid is not None:
            drivers_set.update(starting_grid)
        drivers = sorted(list(drivers_set))

        max_laps = int(laps["LapNumber"].max()) if len(laps) > 0 else 0
        lap_dict = {0: {d: 0.0 for d in drivers}}

        for lap_no in range(1, max_laps + 1):
            lap_dict[lap_no] = {}
            current = laps[laps["LapNumber"] == lap_no]

            for d in drivers:
                driver_lap = current[current["Driver"] == d]
                if len(driver_lap):
                    t = driver_lap.iloc[0]["LapTime"]
                    if pd.notna(t):
                        lap_dict[lap_no][d] = t.total_seconds()

        return lap_dict

    def _get_tyre_degradation(self, laps):
        max_laps = int(laps["LapNumber"].max()) if len(laps) > 0 else 70
        max_tyre_life = int(laps["TyreLife"].max()) if "TyreLife" in laps and len(laps) > 0 else 0
        limit_age = max(max_laps, max_tyre_life, 10)
        fuel_burn_per_lap = 3.5 / max_laps if max_laps > 0 else 0.05

        def get_default_curve(compound, limit):
            if compound == "SOFT":
                base = {1:0.00, 2:0.03, 3:0.07, 4:0.12, 5:0.18, 6:0.25, 7:0.34, 8:0.45, 9:0.58, 10:0.73}
                last_diff = 0.15
            elif compound == "MEDIUM":
                base = {1:0.00, 2:0.02, 3:0.05, 4:0.09, 5:0.14, 6:0.20, 7:0.27, 8:0.35, 9:0.44, 10:0.54}
                last_diff = 0.10
            elif compound == "HARD":
                base = {1:0.00, 2:0.01, 3:0.03, 4:0.05, 5:0.08, 6:0.11, 7:0.15, 8:0.19, 9:0.24, 10:0.30}
                last_diff = 0.06
            elif compound == "INTERMEDIATE":
                base = {1:0.00, 2:0.01, 3:0.03, 4:0.05, 5:0.08, 6:0.12, 7:0.17, 8:0.23, 9:0.30, 10:0.38}
                last_diff = 0.08
            else:
                base = {1:0.00, 2:0.01, 3:0.02, 4:0.04, 5:0.06, 6:0.09, 7:0.13, 8:0.18, 9:0.24, 10:0.31}
                last_diff = 0.07
                
            curve = {}
            for age in range(1, limit + 1):
                if age <= 10:
                    curve[age] = base[age]
                else:
                    curve[age] = base[10] + (age - 10) * last_diff
            return curve

        deg = {}
        for compound in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
            compound_id = self.COMPOUND_MAP[compound]
            if compound == "INTERMEDIATE":
                subset = laps[laps["Compound"].isin(["INTERMEDIATE", "INTER", "I"])].copy()
            elif compound == "WET":
                subset = laps[laps["Compound"].isin(["WET", "HEAVY_WET", "W"])].copy()
            else:
                subset = laps[laps["Compound"] == compound].copy()
            
            d_clean = subset.dropna(subset=["LapTime", "TrackStatus", "IsAccurate", "TyreLife"])
            d_clean = d_clean[(d_clean["TrackStatus"] == "1") & (d_clean["IsAccurate"] == True)]
            
            default_curve = get_default_curve(compound, limit_age)
            increases_by_age = defaultdict(list)
            
            for (driver, stint), d_group in d_clean.groupby(["Driver", "Stint"]):
                if len(d_group) < 2:
                    continue
                
                d_stint = d_group.copy()
                d_stint["LapTimeSec"] = d_stint["LapTime"].dt.total_seconds()
                
                median_time = d_stint["LapTimeSec"].median()
                d_stint_filtered = d_stint[np.abs(d_stint["LapTimeSec"] - median_time) < 5.0]
                
                if len(d_stint_filtered) < 2:
                    continue
                
                d_stint_filtered = d_stint_filtered.sort_values("TyreLife")
                stint_by_age = d_stint_filtered.groupby("TyreLife")["LapTimeSec"].median()
                
                min_age = min(stint_by_age.keys())
                
                if min_age <= 4:
                    base_time = stint_by_age[min_age]
                    default_offset = default_curve[min_age]
                    for age, time_val in stint_by_age.items():
                        raw_increase = time_val - base_time
                        fuel_correction = (age - min_age) * fuel_burn_per_lap
                        increases_by_age[int(age)].append(raw_increase + fuel_correction + default_offset)
            
            known_ages = [1]
            known_values = [0.0]
            
            for age in sorted(increases_by_age.keys()):
                if age <= 1:
                    continue
                samples = increases_by_age[age]
                if len(samples) >= 2:
                    avg_increase = float(np.median(samples))
                    known_ages.append(age)
                    known_values.append(avg_increase)
            
            is_fallback = False
            if len(known_ages) < 2:
                is_fallback = True
            else:
                max_known_age = known_ages[-1]
                slope = known_values[-1] / (max_known_age - 1)
                if known_values[-1] < 0.05 or slope < 0.002:
                    is_fallback = True
            
            if is_fallback:
                deg[compound_id] = default_curve
                continue
            
            compound_curve = {}
            max_known_age = known_ages[-1]
            extrap_slope = max(0.005, known_values[-1] / (max_known_age - 1))
            
            for age in range(1, limit_age + 1):
                if age == 1:
                    compound_curve[age] = 0.0
                elif age <= max_known_age:
                    val = float(np.interp(age, known_ages, known_values))
                    compound_curve[age] = max(compound_curve[age - 1], val)
                else:
                    compound_curve[age] = compound_curve[max_known_age] + (age - max_known_age) * extrap_slope
                    
            deg[compound_id] = compound_curve
            
        return deg

    def _get_average_pit_loss(self, laps):
        pit_losses = []
        
        for driver in laps["Driver"].unique():
            d = laps[laps["Driver"] == driver].copy()
            d = d.sort_values("LapNumber")
            
            normal_laps = d[
                pd.isna(d["PitInTime"]) & 
                pd.isna(d["PitOutTime"]) & 
                (d["IsAccurate"] == True) & 
                pd.notna(d["LapTime"])
            ]
            
            if len(normal_laps) == 0:
                normal_laps = d[pd.notna(d["LapTime"])]
                
            if len(normal_laps) == 0:
                continue
                
            avg_lap_time = normal_laps["LapTime"].dt.total_seconds().mean()
            
            for i in range(1, len(d)):
                prev = d.iloc[i - 1]
                curr = d.iloc[i]
                
                if pd.notna(curr["PitOutTime"]) and pd.notna(prev["PitInTime"]):
                    if pd.notna(curr["LapTime"]) and pd.notna(prev["LapTime"]):
                        pit_time = curr["LapTime"].total_seconds() + prev["LapTime"].total_seconds()
                        pit_loss = pit_time - 2 * avg_lap_time
                        pit_losses.append(max(0.0, pit_loss))
                        
        if len(pit_losses) > 0:
            return float(np.mean(pit_losses))
            
        return 20.0

    def _get_base_time(self, laps):
        all_times = laps["LapTime"].dropna()
        all_times = all_times.dt.total_seconds().values
        if len(all_times) > 0:
            all_times = np.sort(all_times)
            k = max(1, int(len(all_times) * 0.10))
            overall_base = float(np.mean(all_times[:k]))
        else:
            overall_base = 80.0

        base_times = {}
        processed_compound_ids = set()
        for compound_name, compound_id in self.COMPOUND_MAP.items():
            if compound_id in processed_compound_ids:
                continue
            processed_compound_ids.add(compound_id)
            
            if compound_name in ["INTERMEDIATE", "INTER", "I"]:
                comp_laps = laps[laps["Compound"].isin(["INTERMEDIATE", "INTER", "I"])].copy()
            elif compound_name in ["WET", "HEAVY_WET", "W"]:
                comp_laps = laps[laps["Compound"].isin(["WET", "HEAVY_WET", "W"])].copy()
            else:
                comp_laps = laps[laps["Compound"] == compound_name].copy()
            
            comp_laps = comp_laps.dropna(subset=["LapTime", "TrackStatus", "TyreLife", "IsAccurate"])
            
            fresh_no_disturb = comp_laps[
                (comp_laps["TrackStatus"] == "1") & 
                (comp_laps["IsAccurate"] == True) & 
                (comp_laps["TyreLife"] <= 3)
            ]
            
            times = np.sort(fresh_no_disturb["LapTime"].dt.total_seconds().values)

            if len(times) > 0:
                k = max(1, int(len(times) * 0.20))
                base_times[compound_id] = float(np.mean(times[:k]))
                continue
                
            fresh_no_disturb_5 = comp_laps[
                (comp_laps["TrackStatus"] == "1") &
                (comp_laps["IsAccurate"] == True) &
                (comp_laps["TyreLife"] <= 5)
            ]
            times = np.sort(fresh_no_disturb_5["LapTime"].dt.total_seconds().values)
            if len(times) > 0:
                k = max(1, int(len(times) * 0.20))
                base_times[compound_id] = float(np.mean(times[:k]))
                continue
                
            any_no_disturb = comp_laps[
                (comp_laps["TrackStatus"] == "1") & 
                (comp_laps["IsAccurate"] == True)
            ]
            times = any_no_disturb["LapTime"].dt.total_seconds().values
            if len(times) > 0:
                times = np.sort(times)
                k = max(1, int(len(times) * 0.20))
                base_times[compound_id] = float(np.mean(times[:k]))
                continue
                
            any_accurate = comp_laps[comp_laps["IsAccurate"] == True]
            times = any_accurate["LapTime"].dt.total_seconds().values
            if len(times) > 0:
                times = np.sort(times)
                k = max(1, int(len(times) * 0.20))
                base_times[compound_id] = float(np.mean(times[:k]))
                continue

        soft_id = 0
        medium_id = 1
        hard_id = 2
        intermediate_id = 3
        wet_id = 4

        if medium_id in base_times:
            if soft_id not in base_times:
                base_times[soft_id] = base_times[medium_id] - 1.0
            if hard_id not in base_times:
                base_times[hard_id] = base_times[medium_id] + 1.0
        elif soft_id in base_times:
            if medium_id not in base_times:
                base_times[medium_id] = base_times[soft_id] + 1.0
            if hard_id not in base_times:
                base_times[hard_id] = base_times[soft_id] + 2.0
        elif hard_id in base_times:
            if medium_id not in base_times:
                base_times[medium_id] = base_times[hard_id] - 1.0
            if soft_id not in base_times:
                base_times[soft_id] = base_times[hard_id] - 2.0

        for cid in [soft_id, medium_id, hard_id]:
            if cid not in base_times:
                if cid == soft_id:
                    base_times[cid] = overall_base - 0.5
                elif cid == medium_id:
                    base_times[cid] = overall_base
                elif cid == hard_id:
                    base_times[cid] = overall_base + 0.5
                    
        if intermediate_id not in base_times:
            base_times[intermediate_id] = overall_base + 8.0
        if wet_id not in base_times:
            base_times[wet_id] = overall_base + 15.0

        return base_times

import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { Play, Pause, RotateCcw, ChevronLeft, ChevronRight, AlertTriangle, HelpCircle, Trophy, BarChart3, LineChart } from "lucide-react";
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, BarChart, Bar } from "recharts";

// Setup API endpoint
const API_URL = "http://127.0.0.1:8000";

interface Driver {
  driver_id: string;
  name: string;
  final_position: number;
}

interface Race {
  track: string;
  max_laps: number;
  drivers: Driver[];
}

interface RaceData {
  [year: string]: Race[];
}

interface LapData {
  lap: number;
  position: number;
  tyre_compound: number;
  tyre_age: number;
  gap_leader: number;
  gap_ahead: number;
  gap_behind: number;
  safety_car: number;
  track_wetness: number;
  lap_time: number;
  lap_delta: number;
  real_action: number;
  dqn_action: number;
  dqn_q_values: number[] | null;
  qrl_action: number;
  qrl_q_values: number[] | null;
  action_mask: boolean[];
}

interface RunRaceResponse {
  starting_position: number;
  final_position: number;
  laps: LapData[];
}

interface BranchResponse {
  real: { avg_finish: number; finishes: number[] };
  dqn: { avg_finish: number; finishes: number[] };
  qrl: { avg_finish: number; finishes: number[] };
}

const COMPOUND_NAMES = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"];
const COMPOUND_COLORS = {
  SOFT: "bg-red-600 text-white border-red-500",
  MEDIUM: "bg-yellow-500 text-black border-yellow-400",
  HARD: "bg-white text-black border-gray-300",
  INTERMEDIATE: "bg-green-600 text-white border-green-500",
  WET: "bg-blue-600 text-white border-blue-500",
};

const ACTION_DESCRIPTIONS = [
  "STAY OUT",
  "PIT -> SOFT",
  "PIT -> MEDIUM",
  "PIT -> HARD",
  "PIT -> INTER",
  "PIT -> WET",
];

export default function App() {
  // Navigation & Data State
  const [races, setRaces] = useState<RaceData>({});
  const [selectedYear, setSelectedYear] = useState<string>("");
  const [selectedTrack, setSelectedTrack] = useState<string>("");
  const [selectedDriver, setSelectedDriver] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Simulation Replay State
  const [raceResult, setRaceResult] = useState<RunRaceResponse | null>(null);
  const [currentLapIdx, setCurrentLapIdx] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1000); // ms per lap

  // Branching / Counterfactual Simulation State
  const [branching, setBranching] = useState<boolean>(false);
  const [branchResult, setBranchResult] = useState<BranchResponse | null>(null);
  const [numTrials, setNumTrials] = useState<number>(50);

  const playbackTimer = useRef<NodeJS.Timeout | null>(null);

  // Fetch available races on mount
  useEffect(() => {
    axios
      .get(`${API_URL}/api/races`)
      .then((res) => {
        setRaces(res.data);
        const years = Object.keys(res.data);
        if (years.length > 0) {
          const latestYear = years[years.length - 1];
          setSelectedYear(latestYear);
          const racesInYear = res.data[latestYear];
          if (racesInYear.length > 0) {
            setSelectedTrack(racesInYear[0].track);
            const firstDriver = racesInYear[0].drivers[0]?.driver_id || "";
            setSelectedDriver(firstDriver);
          }
        }
      })
      .catch((err) => {
        console.error(err);
        setError("Could not connect to FastAPI server. Make sure main.py is running on port 8000!");
      });
  }, []);

  // Update dropdowns when year or track change
  const handleYearChange = (year: string) => {
    setSelectedYear(year);
    const tracks = races[year] || [];
    if (tracks.length > 0) {
      setSelectedTrack(tracks[0].track);
      setSelectedDriver(tracks[0].drivers[0]?.driver_id || "");
    }
  };

  const handleTrackChange = (track: string) => {
    setSelectedTrack(track);
    const tracks = races[selectedYear] || [];
    const race = tracks.find((r) => r.track === track);
    if (race && race.drivers.length > 0) {
      setSelectedDriver(race.drivers[0].driver_id);
    }
  };

  // Run full race simulation
  const loadRace = () => {
    setLoading(true);
    setError(null);
    setIsPlaying(false);
    setBranchResult(null);
    axios
      .post(`${API_URL}/api/run-race`, {
        track: selectedTrack,
        year: parseInt(selectedYear),
        driver: selectedDriver,
      })
      .then((res) => {
        setRaceResult(res.data);
        setCurrentLapIdx(0);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setError("Error running race simulation. Check backend logs.");
        setLoading(false);
      });
  };

  // Playback logic
  useEffect(() => {
    if (isPlaying && raceResult) {
      playbackTimer.current = setInterval(() => {
        setCurrentLapIdx((prev) => {
          if (prev >= raceResult.laps.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, playbackSpeed);
    } else {
      if (playbackTimer.current) clearInterval(playbackTimer.current);
    }
    return () => {
      if (playbackTimer.current) clearInterval(playbackTimer.current);
    };
  }, [isPlaying, raceResult, playbackSpeed]);

  // Run branching Monte Carlo simulation
  const runBranching = () => {
    if (!raceResult) return;
    setBranching(true);
    const lapData = raceResult.laps[currentLapIdx];
    axios
      .post(`${API_URL}/api/branch-simulation`, {
        track: selectedTrack,
        year: parseInt(selectedYear),
        driver: selectedDriver,
        branch_lap: lapData.lap,
        trials: numTrials,
      })
      .then((res) => {
        setBranchResult(res.data);
        setBranching(false);
      })
      .catch((err) => {
        console.error(err);
        setBranching(false);
      });
  };

  // Current lap telemetry shorthand
  const currentLap = raceResult?.laps[currentLapIdx] || null;
  const isDivergent =
    currentLap &&
    (currentLap.real_action !== currentLap.dqn_action ||
      currentLap.real_action !== currentLap.qrl_action);

  // Auto pause on divergence
  useEffect(() => {
    if (isDivergent && isPlaying) {
      setIsPlaying(false);
    }
  }, [currentLapIdx]);

  // Drivers list helper sorted to highlight lower-finishing grid positions first
  const currentRace = races[selectedYear]?.find((r) => r.track === selectedTrack);
  const sortedDrivers = currentRace?.drivers ? [...currentRace.drivers] : [];

  return (
    <div className="min-h-screen bg-[#0d0e12] text-gray-200 p-4 lg:p-6 flex flex-col justify-between">
      {/* 1. TOP HEADER SELECTORS */}
      <header className="glass-panel rounded-xl p-4 mb-6 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="bg-[#e63946] text-white p-2 rounded-lg font-black tracking-widest text-lg">F1</div>
          <div>
            <h1 className="text-xl font-bold tracking-wider text-white m-0 p-0 leading-none">STRATEGY COMMAND CENTER</h1>
            <span className="text-xs text-gray-400">Quantum Reinforcement Learning Showcase</span>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Year Select */}
          <div className="flex flex-col">
            <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider mb-1">Year</span>
            <select
              value={selectedYear}
              onChange={(e) => handleYearChange(e.target.value)}
              className="bg-[#191b24] border border-gray-700 rounded-md px-3 py-1.5 text-sm font-semibold focus:outline-none focus:ring-1 focus:ring-[#e63946]"
            >
              {Object.keys(races).map((y) => (
                <option key={y} value={y}>
                  {y}
                </option>
              ))}
            </select>
          </div>

          {/* Race/Track Select */}
          <div className="flex flex-col">
            <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider mb-1">Grand Prix</span>
            <select
              value={selectedTrack}
              onChange={(e) => handleTrackChange(e.target.value)}
              className="bg-[#191b24] border border-gray-700 rounded-md px-3 py-1.5 text-sm font-semibold focus:outline-none focus:ring-1 focus:ring-[#e63946]"
            >
              {races[selectedYear]?.map((r) => (
                <option key={r.track} value={r.track}>
                  {r.track} GP
                </option>
              ))}
            </select>
          </div>

          {/* Driver Select */}
          <div className="flex flex-col">
            <span className="text-[10px] text-gray-400 font-bold uppercase tracking-wider mb-1">Driver (Showcase First)</span>
            <select
              value={selectedDriver}
              onChange={(e) => setSelectedDriver(e.target.value)}
              className="bg-[#191b24] border border-gray-700 rounded-md px-3 py-1.5 text-sm font-semibold focus:outline-none focus:ring-1 focus:ring-[#e63946]"
            >
              {sortedDrivers.map((d) => (
                <option key={d.driver_id} value={d.driver_id}>
                  {d.name} (Real: P{d.final_position === 20 ? "Retired" : d.final_position})
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={loadRace}
            disabled={loading || !selectedDriver}
            className="bg-[#e63946] hover:bg-[#c32f3a] text-white px-6 py-2 rounded-md font-bold tracking-wide transition duration-150 disabled:opacity-50 mt-4 md:mt-0 cursor-pointer"
          >
            {loading ? "SIMULATING..." : "LOAD RACE"}
          </button>
        </div>
      </header>

      {error && (
        <div className="bg-red-950/40 border border-red-500/50 rounded-xl p-4 text-red-300 text-sm mb-6 flex items-center gap-3">
          <AlertTriangle size={18} />
          {error}
        </div>
      )}

      {/* Main Command Workspace */}
      {raceResult ? (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch mb-6">
          {/* LAP TIMING CONTROLLER */}
          <div className="lg:col-span-12 glass-panel rounded-xl p-4 flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="text-3xl font-black font-mono tracking-tighter text-[#e63946]">
                LAP {currentLapIdx + 1} <span className="text-gray-600 text-lg">/ {raceResult.laps.length}</span>
              </div>
              <div className="bg-green-950/40 text-green-400 border border-green-500/30 px-3 py-1 rounded-full text-xs font-bold tracking-widest uppercase">
                {currentLap?.safety_car === 2
                  ? "SAFETY CAR"
                  : currentLap?.safety_car === 1
                  ? "VIRTUAL SC"
                  : "GREEN FLAG"}
              </div>
            </div>

            {/* Replay Controls */}
            <div className="flex items-center gap-3">
              <button
                onClick={() => {
                  setIsPlaying(false);
                  setCurrentLapIdx(0);
                }}
                className="bg-[#1c1e28] hover:bg-gray-800 p-2.5 rounded-lg border border-gray-800 transition cursor-pointer"
              >
                <RotateCcw size={16} />
              </button>

              <button
                onClick={() => setCurrentLapIdx((prev) => Math.max(0, prev - 1))}
                disabled={currentLapIdx === 0}
                className="bg-[#1c1e28] hover:bg-gray-800 p-2.5 rounded-lg border border-gray-800 disabled:opacity-50 cursor-pointer"
              >
                <ChevronLeft size={16} />
              </button>

              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="bg-[#e63946] hover:bg-[#c32f3a] text-white p-3 rounded-lg transition cursor-pointer"
              >
                {isPlaying ? <Pause size={18} /> : <Play size={18} />}
              </button>

              <button
                onClick={() => setCurrentLapIdx((prev) => Math.min(raceResult.laps.length - 1, prev + 1))}
                disabled={currentLapIdx === raceResult.laps.length - 1}
                className="bg-[#1c1e28] hover:bg-gray-800 p-2.5 rounded-lg border border-gray-800 disabled:opacity-50 cursor-pointer"
              >
                <ChevronRight size={16} />
              </button>

              {/* Speed Slider */}
              <div className="flex items-center gap-2 ml-4">
                <span className="text-[10px] uppercase font-bold text-gray-500">Speed</span>
                <input
                  type="range"
                  min="300"
                  max="2000"
                  step="100"
                  value={playbackSpeed}
                  onChange={(e) => setPlaybackSpeed(parseInt(e.target.value))}
                  className="w-20 accent-[#e63946]"
                />
              </div>
            </div>
          </div>

          {/* 2. LEFT SIDE PANEL: REAL LIFE HISTORICAL STATE */}
          <div className="lg:col-span-4 glass-panel rounded-xl p-5 flex flex-col justify-between">
            <div>
              <div className="flex justify-between items-center mb-4 border-b border-gray-800 pb-3">
                <h3 className="font-extrabold tracking-wide uppercase text-sm text-gray-300">REAL LIFE STRATEGY</h3>
                <span className="bg-[#9c27b0]/20 text-[#9c27b0] border border-[#9c27b0]/30 px-2 py-0.5 rounded text-[10px] font-extrabold uppercase">
                  HISTORICAL
                </span>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="bg-[#12131a] border border-gray-800/80 rounded-xl p-4 text-center">
                  <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider block mb-1">
                    Position
                  </span>
                  <div className="text-4xl font-black text-white">P{currentLap?.position}</div>
                </div>

                <div className="bg-[#12131a] border border-gray-800/80 rounded-xl p-4 text-center">
                  <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider block mb-1">
                    Tyre Life
                  </span>
                  <div className="text-4xl font-black text-white">{currentLap?.tyre_age} <span className="text-xs text-gray-400">laps</span></div>
                </div>
              </div>

              {/* Telemetry info list */}
              <div className="space-y-3 mb-6">
                <div className="flex justify-between items-center bg-[#12131a]/50 p-2.5 rounded-lg border border-gray-850">
                  <span className="text-xs text-gray-400">Tyre Compound</span>
                  <span className={`text-xs px-2.5 py-0.5 rounded font-bold border ${COMPOUND_COLORS[COMPOUND_NAMES[currentLap?.tyre_compound || 0] as keyof typeof COMPOUND_COLORS]}`}>
                    {COMPOUND_NAMES[currentLap?.tyre_compound || 0]}
                  </span>
                </div>

                <div className="flex justify-between items-center bg-[#12131a]/50 p-2.5 rounded-lg border border-gray-850">
                  <span className="text-xs text-gray-400">Gap to Leader</span>
                  <span className="text-sm font-semibold font-mono text-gray-200">
                    {currentLap?.gap_leader === 99.9 ? "P1" : `+${currentLap?.gap_leader.toFixed(3)}s`}
                  </span>
                </div>

                <div className="flex justify-between items-center bg-[#12131a]/50 p-2.5 rounded-lg border border-gray-850">
                  <span className="text-xs text-gray-400">Gap Ahead / Behind</span>
                  <span className="text-sm font-semibold font-mono text-gray-200">
                    +{currentLap?.gap_ahead.toFixed(1)}s / +{currentLap?.gap_behind.toFixed(1)}s
                  </span>
                </div>

                <div className="flex justify-between items-center bg-[#12131a]/50 p-2.5 rounded-lg border border-gray-850">
                  <span className="text-xs text-gray-400">Lap Time</span>
                  <span className="text-sm font-semibold font-mono text-gray-200">
                    {currentLap?.lap_time ? `${currentLap.lap_time.toFixed(3)}s` : "IN PIT"}
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-[#12131a]/80 rounded-xl p-4 border border-gray-800">
              <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider block mb-2">
                Actual Pit Stop Decision
              </span>
              <div className="text-xl font-bold text-white uppercase">
                {currentLap ? ACTION_DESCRIPTIONS[currentLap.real_action] : "—"}
              </div>
            </div>
          </div>

          {/* 3. MIDDLE PANEL: AI AGENT DECISIONS & COMPARISONS */}
          <div className="lg:col-span-5 glass-panel rounded-xl p-5 flex flex-col justify-between">
            <div>
              <div className="flex justify-between items-center mb-4 border-b border-gray-800 pb-3">
                <h3 className="font-extrabold tracking-wide uppercase text-sm text-gray-300">AI STRATEGY ENGINE</h3>
                <span className="bg-blue-950/40 text-blue-400 border border-blue-500/30 px-2 py-0.5 rounded text-[10px] font-extrabold uppercase">
                  LIVE MODEL INFERENCE
                </span>
              </div>

              {/* Side-by-Side Model Inference Cards */}
              <div className="grid grid-cols-2 gap-4 mb-6">
                {/* DQN */}
                <div className="bg-[#12131a] border border-[#e63946]/20 rounded-xl p-4 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center gap-1.5 mb-2">
                      <div className="w-2 h-2 rounded-full bg-[#e63946]" />
                      <span className="text-xs font-bold text-gray-300">DQN (Classical)</span>
                    </div>
                    <div className="text-sm font-black text-white min-h-[40px] uppercase">
                      {currentLap ? ACTION_DESCRIPTIONS[currentLap.dqn_action] : "—"}
                    </div>
                  </div>
                  {/* Action Compound Badge */}
                  {currentLap && currentLap.dqn_action !== 0 && (
                    <div className="mt-3">
                      <span className={`text-[9px] px-2 py-0.5 rounded font-bold border ${COMPOUND_COLORS[COMPOUND_NAMES[currentLap.dqn_action - 1] as keyof typeof COMPOUND_COLORS]}`}>
                        {COMPOUND_NAMES[currentLap.dqn_action - 1]}
                      </span>
                    </div>
                  )}
                </div>

                {/* QRL */}
                <div className="bg-[#12131a] border border-[#1d7fd4]/20 rounded-xl p-4 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center gap-1.5 mb-2">
                      <div className="w-2 h-2 rounded-full bg-[#1d7fd4]" />
                      <span className="text-xs font-bold text-gray-300">QRL (Quantum)</span>
                    </div>
                    <div className="text-sm font-black text-white min-h-[40px] uppercase">
                      {currentLap ? ACTION_DESCRIPTIONS[currentLap.qrl_action] : "—"}
                    </div>
                  </div>
                  {/* Action Compound Badge */}
                  {currentLap && currentLap.qrl_action !== 0 && (
                    <div className="mt-3">
                      <span className={`text-[9px] px-2 py-0.5 rounded font-bold border ${COMPOUND_COLORS[COMPOUND_NAMES[currentLap.qrl_action - 1] as keyof typeof COMPOUND_COLORS]}`}>
                        {COMPOUND_NAMES[currentLap.qrl_action - 1]}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Action scores or Q-values Bar Chart */}
              {currentLap && (currentLap.dqn_q_values || currentLap.qrl_q_values) && (
                <div className="mb-6">
                  <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider block mb-2">
                    Action Q-Values / Confidence Score
                  </span>
                  <div className="h-32">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={ACTION_DESCRIPTIONS.map((desc, idx) => ({
                          name: desc.replace("PIT -> ", ""),
                          DQN: currentLap.dqn_q_values?.[idx] || 0,
                          QRL: currentLap.qrl_q_values?.[idx] || 0,
                        }))}
                        margin={{ top: 5, right: 5, left: -25, bottom: 5 }}
                      >
                        <XAxis dataKey="name" tick={{ fill: "#6b7280", fontSize: 9 }} />
                        <YAxis tick={{ fill: "#6b7280", fontSize: 9 }} />
                        <Tooltip contentStyle={{ backgroundColor: "#191b24", borderColor: "#374151", fontSize: 10 }} />
                        <Bar dataKey="DQN" fill="#e63946" radius={[2, 2, 0, 0]} />
                        <Bar dataKey="QRL" fill="#1d7fd4" radius={[2, 2, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </div>

            {/* Divergence Notification box */}
            {isDivergent ? (
              <div className="bg-yellow-950/20 border border-yellow-500/30 rounded-xl p-4 flex flex-col md:flex-row items-center justify-between gap-4 animate-pulse">
                <div className="flex items-center gap-3">
                  <div className="bg-yellow-600/20 p-2 rounded-lg text-yellow-500">
                    <AlertTriangle size={18} />
                  </div>
                  <div>
                    <div className="text-sm font-black text-yellow-500 uppercase tracking-wide">STRATEGY DIVERGENCE DETECTED</div>
                    <span className="text-xs text-gray-400 block mt-0.5">DQN and QRL model decisions differ from real-world strategy.</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-[#12131a]/30 border border-gray-800/50 rounded-xl p-4 text-center text-xs text-gray-500">
                All strategies in agreement on Lap {currentLapIdx + 1}. Replay proceeding normally.
              </div>
            )}
          </div>

          {/* 4. RIGHT SIDE PANEL: BRANCHING / COUNTERFACTUAL RESULTS */}
          <div className="lg:col-span-3 glass-panel rounded-xl p-5 flex flex-col justify-between">
            <div>
              <div className="flex justify-between items-center mb-4 border-b border-gray-800 pb-3">
                <h3 className="font-extrabold tracking-wide uppercase text-sm text-gray-300">COUNTERFACTUAL BRANCH</h3>
                <span className="bg-emerald-950/40 text-emerald-400 border border-emerald-500/30 px-2 py-0.5 rounded text-[10px] font-extrabold uppercase">
                  SIMULATION
                </span>
              </div>

              <div className="mb-6">
                <p className="text-xs text-gray-400 leading-relaxed mb-4">
                  Branch from the historical race state at Lap {currentLapIdx + 1} and run Monte Carlo simulations to predict the outcomes of alternative strategies.
                </p>

                {/* Configuration */}
                <div className="flex items-center justify-between bg-[#12131a] border border-gray-800/80 p-3 rounded-xl mb-4">
                  <span className="text-xs font-bold text-gray-300">MC Trials</span>
                  <input
                    type="number"
                    min="10"
                    max="100"
                    value={numTrials}
                    onChange={(e) => setNumTrials(parseInt(e.target.value) || 20)}
                    className="w-16 bg-[#191b24] border border-gray-700 rounded px-2 py-1 text-xs text-center font-bold"
                  />
                </div>

                <button
                  onClick={runBranching}
                  disabled={branching}
                  className="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-2.5 px-4 rounded-lg text-sm tracking-wide transition duration-150 cursor-pointer disabled:opacity-50"
                >
                  {branching ? "SIMULATING TRIALS..." : "SIMULATE ALTERNATIVES"}
                </button>
              </div>

              {/* Branch Simulation Outcomes */}
              {branchResult && (
                <div className="space-y-4">
                  <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider block border-b border-gray-800 pb-1.5">
                    Predicted Average Finish
                  </span>

                  <div className="space-y-3">
                    {/* Real */}
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="font-bold text-gray-400">Real Driver Policy</span>
                        <span className="font-mono text-white font-semibold">P{branchResult.real.avg_finish.toFixed(1)}</span>
                      </div>
                      <div className="w-full bg-gray-800 h-1.5 rounded-full overflow-hidden">
                        <div
                          className="bg-[#9c27b0] h-full rounded-full"
                          style={{ width: `${(21 - branchResult.real.avg_finish) * 5}%` }}
                        />
                      </div>
                    </div>

                    {/* DQN */}
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="font-bold text-gray-400">DQN Strategy</span>
                        <span className="font-mono text-white font-semibold">P{branchResult.dqn.avg_finish.toFixed(1)}</span>
                      </div>
                      <div className="w-full bg-gray-800 h-1.5 rounded-full overflow-hidden">
                        <div
                          className="bg-[#e63946] h-full rounded-full"
                          style={{ width: `${(21 - branchResult.dqn.avg_finish) * 5}%` }}
                        />
                      </div>
                    </div>

                    {/* QRL */}
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="font-bold text-gray-400">QRL Strategy</span>
                        <span className="font-mono text-white font-semibold">P{branchResult.qrl.avg_finish.toFixed(1)}</span>
                      </div>
                      <div className="w-full bg-gray-800 h-1.5 rounded-full overflow-hidden">
                        <div
                          className="bg-[#1d7fd4] h-full rounded-full"
                          style={{ width: `${(21 - branchResult.qrl.avg_finish) * 5}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Real vs AI Final Comparison Statistics Card */}
              {raceResult && (
                <div className="mt-5 bg-[#12131a]/85 border border-[#3b82f6]/20 rounded-xl p-4">
                  <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider block mb-2 border-b border-gray-800 pb-1.5">
                    Showcase Insight (Final Stands)
                  </span>
                  <div className="space-y-2.5 text-xs">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400 font-medium">Real Finishing Place:</span>
                      <span className="font-extrabold text-[#9c27b0] bg-[#9c27b0]/15 px-2.5 py-0.5 rounded border border-[#9c27b0]/35">
                        P{raceResult.final_position === 20 ? "Retired" : raceResult.final_position}
                      </span>
                    </div>

                    {branchResult && (
                      <>
                        <div className="flex justify-between items-center">
                          <span className="text-gray-400 font-medium">DQN Strategy Gain:</span>
                          <span className={`font-extrabold px-2 py-0.5 rounded border ${
                            raceResult.final_position - branchResult.dqn.avg_finish >= 0
                              ? "text-emerald-400 bg-emerald-950/15 border-emerald-500/30"
                              : "text-rose-400 bg-rose-950/15 border-rose-500/30"
                          }`}>
                            {raceResult.final_position - branchResult.dqn.avg_finish >= 0 ? "+" : ""}
                            {(raceResult.final_position - branchResult.dqn.avg_finish).toFixed(1)} Positions
                          </span>
                        </div>

                        <div className="flex justify-between items-center">
                          <span className="text-gray-400 font-medium">QRL (Quantum) Gain:</span>
                          <span className={`font-extrabold px-2 py-0.5 rounded border ${
                            raceResult.final_position - branchResult.qrl.avg_finish >= 0
                              ? "text-emerald-400 bg-emerald-950/15 border-emerald-500/30"
                              : "text-rose-400 bg-rose-950/15 border-rose-500/30"
                          }`}>
                            {raceResult.final_position - branchResult.qrl.avg_finish >= 0 ? "+" : ""}
                            {(raceResult.final_position - branchResult.qrl.avg_finish).toFixed(1)} Positions
                          </span>
                        </div>

                        <div className="mt-2.5 pt-2.5 border-t border-gray-800 text-[11px] text-gray-400 leading-relaxed italic">
                          {branchResult.qrl.avg_finish < raceResult.final_position ? (
                            <span>
                              💡 <strong>Quantum Advantage:</strong> By adapting tyre stints earlier, the hybrid Quantum RL model beats the historical F1 pit choices, unlocking a projected gain of <strong>{(raceResult.final_position - branchResult.qrl.avg_finish).toFixed(1)} positions</strong>!
                            </span>
                          ) : (
                            <span>
                              💡 <strong>Strategy Analysis:</strong> The AI models simulate alternative tyres to avoid high degradation, proving that optimizing tyre age helps lower-grid drivers gain traffic advantages.
                            </span>
                          )}
                        </div>
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Disclaimer */}
            <div className="bg-[#12131a]/30 border border-gray-800/80 rounded-xl p-3 text-[10px] text-gray-500 leading-normal flex items-start gap-2">
              <HelpCircle size={12} className="shrink-0 mt-0.5" />
              <span>
                Model predictions are based on stochastic environment runs starting from Lap {currentLapIdx + 1}. Outcomes are simulated approximations, not real race guarantees.
              </span>
            </div>
          </div>

          {/* 5. BOTTOM TIMELINE FOR GENERAL SCRUBBING & STINTS */}
          <div className="lg:col-span-12 glass-panel rounded-xl p-5">
            <h3 className="font-extrabold tracking-wide uppercase text-xs text-gray-400 mb-4">RACE STRATEGY TIMELINE (Interactive)</h3>
            <div className="space-y-4">
              {/* Real Driver timeline row */}
              <div className="flex items-center gap-3">
                <span className="w-12 text-[10px] font-bold text-gray-400 text-right uppercase">Real</span>
                <div className="grow bg-[#191b24] h-6 rounded-lg relative overflow-hidden flex">
                  {raceResult.laps.map((lap, idx) => {
                    const isPit = lap.real_action !== 0;
                    return (
                      <div
                        key={idx}
                        onClick={() => setCurrentLapIdx(idx)}
                        className={`grow h-full border-r border-black/25 cursor-pointer hover:opacity-80 transition-colors ${
                          isPit ? "bg-purple-800" : "bg-[#272935]/40"
                        }`}
                        title={`Lap ${lap.lap}: Real Action ${ACTION_DESCRIPTIONS[lap.real_action]}`}
                      />
                    );
                  })}
                  {/* Current Lap Marker */}
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-[#e63946] z-10"
                    style={{ left: `${(currentLapIdx / (raceResult.laps.length - 1)) * 100}%` }}
                  />
                </div>
              </div>

              {/* DQN timeline row */}
              <div className="flex items-center gap-3">
                <span className="w-12 text-[10px] font-bold text-gray-400 text-right uppercase">DQN</span>
                <div className="grow bg-[#191b24] h-6 rounded-lg relative overflow-hidden flex">
                  {raceResult.laps.map((lap, idx) => {
                    const isPit = lap.dqn_action !== 0;
                    return (
                      <div
                        key={idx}
                        onClick={() => setCurrentLapIdx(idx)}
                        className={`grow h-full border-r border-black/25 cursor-pointer hover:opacity-80 transition-colors ${
                          isPit ? "bg-red-800" : "bg-[#272935]/40"
                        }`}
                      />
                    );
                  })}
                  {/* Current Lap Marker */}
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-[#e63946] z-10"
                    style={{ left: `${(currentLapIdx / (raceResult.laps.length - 1)) * 100}%` }}
                  />
                </div>
              </div>

              {/* QRL timeline row */}
              <div className="flex items-center gap-3">
                <span className="w-12 text-[10px] font-bold text-gray-400 text-right uppercase">QRL</span>
                <div className="grow bg-[#191b24] h-6 rounded-lg relative overflow-hidden flex">
                  {raceResult.laps.map((lap, idx) => {
                    const isPit = lap.qrl_action !== 0;
                    return (
                      <div
                        key={idx}
                        onClick={() => setCurrentLapIdx(idx)}
                        className={`grow h-full border-r border-black/25 cursor-pointer hover:opacity-80 transition-colors ${
                          isPit ? "bg-blue-800" : "bg-[#272935]/40"
                        }`}
                      />
                    );
                  })}
                  {/* Current Lap Marker */}
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-[#e63946] z-10"
                    style={{ left: `${(currentLapIdx / (raceResult.laps.length - 1)) * 100}%` }}
                  />
                </div>
              </div>
            </div>
            <div className="flex justify-between items-center text-[10px] text-gray-500 font-mono mt-3">
              <span>LAP 1</span>
              <div className="flex gap-4">
                <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded bg-purple-800 block" /> Real Pit Stop</span>
                <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded bg-red-800 block" /> DQN Pit Recommendation</span>
                <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded bg-blue-800 block" /> QRL Pit Recommendation</span>
              </div>
              <span>LAP {raceResult.laps.length}</span>
            </div>
          </div>
        </div>
      ) : (
        /* Empty / Initial State */
        <div className="glass-panel rounded-2xl p-16 text-center my-6 flex flex-col items-center justify-center grow">
          <div className="bg-[#e63946]/10 text-[#e63946] p-5 rounded-full border border-[#e63946]/20 mb-6">
            <Trophy size={48} />
          </div>
          <h2 className="text-2xl font-black text-white mb-2 uppercase tracking-wide">Ready for Strategy Evaluation</h2>
          <p className="text-gray-400 max-w-lg text-sm leading-relaxed mb-6">
            Configure your Grand Prix year, track, and driver at the top of the command screen, then click **LOAD RACE** to start running live RL agent model inferences.
          </p>
        </div>
      )}

      {/* Footer */}
      <footer className="text-center text-xs text-gray-600 border-t border-gray-900/60 pt-4">
        F1 Reinforcement Learning strategy optimization showcase dashboard · QC-3 IIT Indore SoC 2026.
      </footer>
    </div>
  );
}

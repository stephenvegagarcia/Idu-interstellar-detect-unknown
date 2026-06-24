import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const Dashboard = () => {
  const [state, setState] = useState({
    quantum: { bell_state: 0, coeff_00: 0, coeff_11: 0, fidelity: 0 },
    solar: { flux_value: 0, flux_trend: 'QUIET', activity_index: 0, corona_lock: 0 },
    particles: [],
    nasa_image_url: null,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(`${API_URL}/api/state`);
        setState(response.data);
        setError(null);
      } catch (err) {
        setError(`Connection Error: ${err.message}`);
        console.error('Failed to fetch state:', err);
      } finally {
        setLoading(false);
      }
    };

    // Initial fetch
    fetchData();

    // Set up polling (update every 500ms)
    const interval = setInterval(fetchData, 500);
    return () => clearInterval(interval);
  }, []);

  const getTrendColor = (trend) => {
    if (trend.includes('FLARE (X)')) return '#ff3232';
    if (trend.includes('FLARE (M)')) return '#ff6b35';
    if (trend.includes('ACTIVE (C)')) return '#ffa500';
    if (trend.includes('ACTIVE (B)')) return '#ffff00';
    return '#00ff80';
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-blue-950 to-slate-950 text-white">
      {/* Header */}
      <div className="border-b border-cyan-500/30 bg-black/40 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <h1 className="text-3xl font-bold text-cyan-400">🚀 INTERSTELLAR DETECTION</h1>
          <p className="text-sm text-gray-400">Quantum Simulation • Solar Monitor • Comet Detection</p>
        </div>
      </div>

      {/* Status Bar */}
      <div className="bg-black/60 border-b border-cyan-500/20 px-6 py-2 text-xs text-gray-400">
        {error ? (
          <span className="text-red-400">⚠️ {error}</span>
        ) : loading ? (
          <span className="text-yellow-400">⏳ Initializing systems...</span>
        ) : (
          <span className="text-green-400">✓ All systems online • Last update: {new Date().toLocaleTimeString()}</span>
        )}
      </div>

      {/* Main Content Grid */}
      <div className="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Panel 1: Quantum State */}
        <div className="lg:col-span-1 bg-slate-900/50 border border-cyan-500/30 rounded-lg p-6 backdrop-blur-sm hover:border-cyan-500/60 transition">
          <h2 className="text-lg font-bold text-cyan-400 mb-4">⚛️ QUANTUM STATE</h2>
          
          {/* Bell State Visualization */}
          <div className="mb-6">
            <div className="text-sm text-gray-400 mb-2">Bell Fidelity</div>
            <div className="relative h-2 bg-slate-800 rounded-full overflow-hidden mb-2">
              <div
                className="h-full bg-gradient-to-r from-cyan-500 to-magenta-500 transition-all duration-300"
                style={{ width: `${state.quantum.bell_state * 100}%` }}
              />
            </div>
            <div className="text-2xl font-bold text-cyan-400">{(state.quantum.bell_state * 100).toFixed(1)}%</div>
          </div>

          {/* Algorithm Display */}
          <div className="bg-black/40 border border-magenta-500/30 rounded p-3 mb-4 text-center">
            <div className="text-xs text-gray-400 mb-1">Algorithm</div>
            <div className="font-mono text-sm text-magenta-400">|φ⁺⟩ = 1/√2 (|00⟩ + |11⟩)</div>
          </div>

          {/* Amplitudes */}
          <div className="space-y-3">
            <div>
              <div className="text-xs text-gray-400 mb-1">|00⟩ Coefficient</div>
              <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-400 transition-all"
                  style={{ width: `${state.quantum.coeff_00 * 100}%` }}
                />
              </div>
              <div className="text-sm text-blue-400 mt-1">{state.quantum.coeff_00.toFixed(3)}</div>
            </div>

            <div>
              <div className="text-xs text-gray-400 mb-1">|11⟩ Coefficient</div>
              <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-red-400 transition-all"
                  style={{ width: `${state.quantum.coeff_11 * 100}%` }}
                />
              </div>
              <div className="text-sm text-red-400 mt-1">{state.quantum.coeff_11.toFixed(3)}</div>
            </div>
          </div>
        </div>

        {/* Panel 2: Solar Telemetry */}
        <div className="lg:col-span-1 bg-slate-900/50 border border-cyan-500/30 rounded-lg p-6 backdrop-blur-sm hover:border-cyan-500/60 transition">
          <h2 className="text-lg font-bold text-cyan-400 mb-4">☀️ SOLAR TELEMETRY</h2>

          {/* X-ray Flux */}
          <div className="mb-6">
            <div className="text-sm text-gray-400 mb-2">X-Ray Flux (W/m²)</div>
            <div className="text-2xl font-mono text-yellow-400">{state.solar.flux_value.toExponential(2)}</div>
          </div>

          {/* Status Alert */}
          <div
            className="rounded p-3 mb-4 border text-center"
            style={{
              backgroundColor: getTrendColor(state.solar.flux_trend) + '20',
              borderColor: getTrendColor(state.solar.flux_trend),
              color: getTrendColor(state.solar.flux_trend),
            }}
          >
            <div className="text-xs text-gray-400 mb-1">Activity Status</div>
            <div className="font-bold">{state.solar.flux_trend}</div>
          </div>

          {/* Activity Index */}
          <div>
            <div className="text-xs text-gray-400 mb-1">Activity Index</div>
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden mb-2">
              <div
                className="h-full bg-gradient-to-r from-green-400 via-yellow-400 to-red-400 transition-all"
                style={{ width: `${state.solar.activity_index * 100}%` }}
              />
            </div>
            <div className="text-sm text-orange-400">{(state.solar.activity_index * 100).toFixed(1)}%</div>
          </div>
        </div>

        {/* Panel 3: Particle Count */}
        <div className="lg:col-span-1 bg-slate-900/50 border border-cyan-500/30 rounded-lg p-6 backdrop-blur-sm hover:border-cyan-500/60 transition">
          <h2 className="text-lg font-bold text-cyan-400 mb-4">✨ PARTICLES</h2>

          <div className="text-3xl font-bold text-green-400 mb-4">
            {state.particles.length.toLocaleString()}
          </div>

          {/* Particle Type Breakdown */}
          <div className="space-y-2 text-sm">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Nucleus (|00⟩)</span>
              <span className="text-blue-400 font-mono">
                {state.particles.filter(p => p.type === 'nucleus').length}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Ion Tail (|11⟩)</span>
              <span className="text-red-400 font-mono">
                {state.particles.filter(p => p.type === 'ion_tail').length}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Corona</span>
              <span className="text-green-400 font-mono">
                {state.particles.filter(p => p.type === 'corona').length}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* NASA Image Panel */}
      {state.nasa_image_url && (
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="bg-slate-900/50 border border-cyan-500/30 rounded-lg p-6 backdrop-blur-sm">
            <h2 className="text-lg font-bold text-cyan-400 mb-4">🛰️ NASA SDO - AIA 171 (REALITY)</h2>
            <div className="rounded overflow-hidden border border-cyan-500/20 max-h-96">
              <img
                src={state.nasa_image_url}
                alt="NASA Solar Dynamics Observatory"
                className="w-full h-auto object-cover"
              />
            </div>
            <p className="text-xs text-gray-400 mt-2">Updated: {new Date().toLocaleString()}</p>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="border-t border-cyan-500/30 bg-black/40 backdrop-blur-md mt-8 py-4">
        <div className="max-w-7xl mx-auto px-6 text-center text-xs text-gray-500">
          <p>Interstellar Detection System | Backend API: {API_URL}</p>
          <p className="mt-1">Quantum Engine • Solar Monitor • NASA Uplink • NOAA Data</p>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;

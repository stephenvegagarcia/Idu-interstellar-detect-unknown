# 🚀 Interstellar Detection System - Integration Guide

This project integrates multiple astronomical and quantum simulation engines into a unified API-driven system suitable for **Loveable** web app builder.

## Project Structure

```
Idu-interstellar-detect-unknown/
├── backend/
│   ├── main.py                 # FastAPI server
│   ├── requirements.txt        # Python dependencies
│   └── engines/
│       ├── quantum_engine.py   # QuTiP Bell state simulator
│       ├── solar_engine.py     # Solar particle generation
│       ├── comet_engine.py     # 3I/ATLAS comet detection
│       ├── nasa_uplink.py      # NASA SDO image fetching
│       └── noaa_fetcher.py     # Real-time solar flux data
├── 3d atlas.py                 # Original pygame solar comet viz
├── Thesun3d.py                 # Original pygame solar dual monitor
└── [other original files]
```

## Features

### 1. **Quantum Engine**
- Bell State Entanglement Simulation: `|φ⁺⟩ = 1/√2 (|00⟩ + |11⟩)`
- Uses QuTiP for real quantum calculations (with fallback emulation)
- Configurable noise injection from solar flux

### 2. **Solar Simulator**
- Real-time 3D particle generation for solar visualization
- Fibonacci sphere distribution for realistic surface rendering
- Corona expansion based on quantum stability
- Solar flare detection and highlighting

### 3. **Comet Detection (3I/ATLAS)**
- Quantum-state-based particle system
- Nucleus (|00⟩ state) and ion tail (|11⟩ state) visualization
- Coma cloud generation

### 4. **NASA SDO Integration**
- Live AIA 171 Angstrom images (solar corona)
- Multiple wavelength support (94, 131, 171, 193, 211, 304, 335, 1600, 1700)
- HMI magnetic field images

### 5. **NOAA Space Weather Data**
- Real-time X-ray flux monitoring
- Solar activity classification (Quiet, B, C, M, X flares)
- Activity index normalization for visualization

## Installation & Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the API Server

```bash
python main.py
```

Server runs on `http://localhost:8000`

### 3. Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## REST API Endpoints

### Core State Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/api/state` | Current simulation state (all data) |
| `GET` | `/api/quantum` | Quantum state only |
| `GET` | `/api/solar` | Solar telemetry |
| `GET` | `/api/particles` | 3D particle data |
| `GET` | `/api/nasa/image` | Latest NASA image URL |

### Configuration Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/quantum/configure` | Set quantum noise level |
| `POST` | `/api/simulation/reset` | Reset all simulations |

### WebSocket Endpoints

| Endpoint | Purpose | Update Rate |
|----------|---------|------------|
| `/ws/stream` | Real-time state streaming | 100ms (~10Hz) |
| `/ws/particles` | Particle data for 3D viz | 33ms (~30Hz) |

## Example API Usage

### Get Current State
```bash
curl http://localhost:8000/api/state
```

Response:
```json
{
  "quantum": {
    "bell_state": 0.78,
    "coeff_00": 0.55,
    "coeff_11": 0.55,
    "fidelity": 0.78,
    "timestamp": "2025-06-24T12:34:56.789012"
  },
  "solar": {
    "flux_value": 1.5e-6,
    "flux_trend": "ACTIVE (C)",
    "activity_index": 0.42,
    "corona_lock": 0.58,
    "timestamp": "2025-06-24T12:34:56.789012"
  },
  "particles": [
    {
      "x": -12.3,
      "y": 45.6,
      "z": 78.9,
      "color": [255, 180, 0],
      "radius": 3,
      "type": "surface"
    }
  ],
  "nasa_image_url": "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0171.jpg",
  "timestamp": "2025-06-24T12:34:56.789012"
}
```

### Configure Quantum Noise
```bash
curl -X POST "http://localhost:8000/api/quantum/configure?flux_noise=0.5"
```

### WebSocket Stream (JavaScript)
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Quantum State:', data.quantum);
  console.log('Solar Data:', data.solar);
};
```

## Integration with Loveable

### Step 1: Deploy Backend
```bash
# Use any hosting (Heroku, Render, AWS, etc.)
gunicorn -w 4 -b 0.0.0.0:8000 backend.main:app
```

### Step 2: Frontend Components (for Loveable)

#### 3D Visualization (Three.js)
```javascript
// Display particles from /api/particles
import * as THREE from 'three';

const geometry = new THREE.BufferGeometry();
// Add particles as points
const material = new THREE.PointsMaterial({size: 2});
const points = new THREE.Points(geometry, material);
scene.add(points);
```

#### Real-time Dashboard
```javascript
// Use WebSocket /ws/stream for live updates
setInterval(async () => {
  const response = await fetch('/api/state');
  const state = await response.json();
  
  // Update UI with quantum, solar, NASA data
}, 100);
```

#### NASA Image Viewer
```javascript
// Fetch from /api/nasa/image
// Display as full-screen image panel
```

## Original Files

The project includes the original PyGame-based visualizations:

- **3d atlas.py**: 3I/ATLAS comet detector with quantum visualization
- **Thesun3d.py**: Dual-panel solar monitor (simulation vs. reality)
- **Solar defense.py**: Solar defense simulation
- **[Other files]**: Various detection and analysis scripts

These can still be run independently or integrated with the API backend.

## Dependencies Overview

| Package | Purpose |
|---------|---------|
| `fastapi` | Web API framework |
| `uvicorn` | ASGI server |
| `qutip` | Quantum computing simulations |
| `numpy` | Numerical computing |
| `scipy` | Scientific computing |
| `skyfield` | Astronomy calculations |
| `pygame` | Original visualization (optional) |
| `requests` | HTTP requests for NASA/NOAA |
| `pillow` | Image processing |

## Advanced Features

### Multi-wavelength Solar Imaging
```python
# Get all available AIA wavelengths
nasa_uplink.get_aia_wavelengths()
# Returns URLs for 94, 131, 171, 193, 211, 304, 335, 1600, 1700 Å
```

### Historical Trend Analysis
```python
# Get 6-hour flux trends
noaa_fetcher.get_historical_trends(hours=6)
# Returns timestamps and flux values for graphing
```

### Quantum State Export
```python
# Get detailed quantum state for analysis
/api/quantum
# Returns bell_state, coefficients, fidelity, simulation method
```

## Performance Considerations

- **Particle Generation**: ~1000 particles per frame @ 60 FPS
- **NASA Image Update**: 5-minute cache (configurable)
- **NOAA Data Update**: 60-second polling (configurable)
- **WebSocket Rate**: 30 FPS for particles, 10 FPS for telemetry

## Future Enhancements

- [ ] Add machine learning predictions for solar flares
- [ ] Implement multi-wavelength analysis
- [ ] Add historical data archive
- [ ] Create alert system for X-class flares
- [ ] Support for other solar instruments (HMI, EVE)
- [ ] Real-time data export (CSV, JSON)
- [ ] Custom simulation parameter presets

## Support

For issues with:
- **NASA Data**: Check https://sdo.gsfc.nasa.gov/
- **NOAA Data**: Check https://www.swpc.noaa.gov/
- **QuTiP**: See http://qutip.org/

---

**Happy detecting!** 🌞🔭✨

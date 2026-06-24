"""
Loveable Frontend Integration Guide
Complete setup for connecting your Interstellar Detection API to Loveable
"""

# ============================================================================
# LOVEABLE SETUP INSTRUCTIONS
# ============================================================================

## Step 1: Create a New Loveable Project

1. Go to https://loveable.dev
2. Sign in or create an account
3. Click "Create New Project"
4. Choose "React" template
5. Name it: "Interstellar Detection Dashboard"

## Step 2: Install Dependencies

In your Loveable project, add to `package.json`:

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.6.0",
    "three": "^r128",
    "react-three-fiber": "^8.13.0",
    "@react-three/drei": "^9.88.0",
    "recharts": "^2.10.0",
    "tailwindcss": "^3.3.0"
  }
}
```

## Step 3: Configure API Endpoint

Create `.env.local` in your Loveable project:

```
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
```

Or for production (after deploying backend):

```
REACT_APP_API_URL=https://your-backend-domain.com
REACT_APP_WS_URL=wss://your-backend-domain.com
```

## Step 4: Copy Frontend Components

Copy these React components into your `src/components` folder:

- `Dashboard.jsx` - Main dashboard layout
- `QuantumVisualizer.jsx` - Quantum state display
- `SolarMonitor.jsx` - Solar data & NASA image
- `ParticleViewer3D.jsx` - 3D particle visualization
- `TelemetryPanel.jsx` - Real-time telemetry

## Step 5: Update Main App.jsx

```jsx
import Dashboard from './components/Dashboard';
import './App.css';

function App() {
  return <Dashboard />;
}

export default App;
```

---

## Component Documentation

### Dashboard.jsx
Main component that orchestrates all subcomponents and manages global state.

**Features:**
- Connects to FastAPI backend
- WebSocket real-time streaming
- Component layout (3 panels)
- Global state management

### QuantumVisualizer.jsx
Displays quantum simulation state with Bell state visualization.

**Props:**
- `bellState: number` - Fidelity value (0-1)
- `coeff00: number` - |00⟩ amplitude
- `coeff11: number` - |11⟩ amplitude

### SolarMonitor.jsx
Shows solar telemetry and NASA SDO image.

**Props:**
- `fluxValue: number` - Current X-ray flux
- `fluxTrend: string` - Activity classification
- `nasaImageUrl: string` - URL to latest image
- `activityIndex: number` - Normalized activity (0-1)

### ParticleViewer3D.jsx
3D visualization using Three.js & react-three-fiber.

**Props:**
- `particles: array` - Array of particle objects
- `type: string` - "solar" or "comet"

### TelemetryPanel.jsx
Real-time data display with charts and metrics.

**Props:**
- `data: object` - Simulation state

---

## Running Locally

### Terminal 1 - Start Backend:
```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

### Terminal 2 - Start Loveable Frontend:
```bash
cd loveable-project
npm start
```

### Terminal 3 (Optional) - Run Original Pygame Visualizations:
```bash
python "3d atlas.py"
# or
python Thesun3d.py
```

---

## Deployment

### Deploy Backend (Render.com example):

```bash
# 1. Push to GitHub
git push

# 2. Create Render service
# - Connect GitHub repo
# - Runtime: Python 3.9+
# - Build: pip install -r backend/requirements.txt
# - Start: gunicorn -w 4 -b 0.0.0.0:8000 backend.main:app

# 3. Set environment variables if needed
```

### Deploy Frontend (Vercel example):

```bash
# 1. In Loveable, click "Export"
# 2. Choose "Vercel"
# 3. Update REACT_APP_API_URL to production backend

# Or manually:
npm run build
# Deploy 'build' folder to Vercel
```

---

## API Endpoints Quick Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health check |
| `/api/state` | GET | Full simulation state |
| `/api/quantum` | GET | Quantum data only |
| `/api/solar` | GET | Solar telemetry |
| `/api/particles` | GET | 3D particle data |
| `/api/nasa/image` | GET | NASA image URL |
| `/ws/stream` | WS | Real-time updates |
| `/ws/particles` | WS | Particle stream |

---

## Troubleshooting

### Connection Issues

**Error: "Failed to fetch from API"**
- Check backend is running: `http://localhost:8000/docs`
- Verify CORS is enabled in `backend/main.py`
- Check firewall settings

**WebSocket won't connect**
- Ensure backend WebSocket routes are enabled
- Browser must support WebSocket
- Check for proxy issues (corporate networks)

### Performance Issues

**3D visualization laggy**
- Reduce particle count in backend
- Lower WebSocket update frequency
- Use particle culling (only render visible particles)

**NASA images not loading**
- Check internet connection
- NASA servers may be down temporarily
- Try accessing directly: https://sdo.gsfc.nasa.gov/

---

## Advanced Customization

### Change Update Frequencies

In `backend/main.py`:

```python
# Change WebSocket update rate (in seconds)
await asyncio.sleep(0.1)  # Currently 100ms (~10Hz)

# For faster updates:
await asyncio.sleep(0.033)  # ~30Hz

# For slower updates:
await asyncio.sleep(0.5)   # ~2Hz
```

### Modify Particle Generation

In `backend/engines/solar_engine.py`:

```python
# Increase particle count
num_surface_points = 1000  # Default: 500

# Change corona expansion
expansion = (1.0 - quantum_stability) * 200  # Default: 100
```

### Custom Color Schemes

In frontend components:

```jsx
const colors = {
  ATLAS_GREEN: '#00ff80',      // Change to any hex color
  ION_BLUE: '#3264ff',
  NUCLEUS_WHITE: '#dcdcff',
  // ...
};
```

---

## Support & Resources

- **Loveable Docs**: https://docs.loveable.dev
- **React Docs**: https://react.dev
- **Three.js Docs**: https://threejs.org/docs
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **NASA SDO**: https://sdo.gsfc.nasa.gov

---

**Ready to launch? Start with the Dashboard.jsx component!** 🚀

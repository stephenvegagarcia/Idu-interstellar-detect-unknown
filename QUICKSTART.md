# Quick Start Guide

## Getting Started in 3 Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python run.py
# or
python app.py
```

### 3. Open Your Browser
Navigate to: **http://localhost:5000**

---

## What You'll See

### Home Page
- Grid of 9 different astronomical visualizations
- Click any card to explore

### Visualization Page
- **Run Visualization**: Starts the program (opens in new window)
- **Stop Visualization**: Closes the running program
- **View Code**: Shows the Python source code
- **Status Indicator**: Shows if the program is running

---

## Available Visualizations

1. **Star Mapper - Live Sky** - Real-time star mapping
2. **Quantum-Enhanced Sky Watcher** - Anomaly detection
3. **Binary Transmitter** - Signal transmission simulation
4. **Entropy-Driven Atlas Observation** - Real telescope data
5. **Mas View** - Mass observation
6. **Radio Waves Simulation** - Quantum radio waves
7. **Solar Defense** - Solar activity monitoring
8. **The Sun 3D** - 3D solar visualization
9. **3D Atlas** - 3D cosmic atlas

---

## Troubleshooting

### Flask Not Found
```bash
pip install Flask
```

### Display Issues
Some visualizations need a graphical display. On Linux servers:
```bash
sudo apt-get install xvfb
export DISPLAY=:0
```

### Port Already in Use
Edit `app.py` and change the port:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

---

## Features

✅ Web-based control panel  
✅ Start/stop visualizations from browser  
✅ View source code inline  
✅ Monitor running status  
✅ Responsive design  
✅ Cosmic-themed UI  

---

## System Requirements

- Python 3.8+
- Modern web browser
- Display system (for Pygame/OpenCV windows)
- 2GB+ RAM recommended

---

For detailed information, see [README.md](README.md)

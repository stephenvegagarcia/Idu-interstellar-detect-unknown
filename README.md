# IDU - Interstellar Detect Unknown

A Flask web application that provides a unified interface for running and managing various astronomical visualization and simulation tools.

## Overview

This project combines multiple astronomical visualization scripts into a single Flask web application, making it easy to launch and manage different cosmic simulations from a web browser.

## Features

- **Web-based Interface**: Easy-to-use web interface to access all visualizations
- **Multiple Visualizations**: Nine different astronomical simulations and visualizations
- **Process Management**: Start and stop visualizations from the web interface
- **Source Code Viewer**: View the source code of any visualization
- **Real-time Status**: Monitor the running status of each visualization

## Available Visualizations

1. **Star Mapper - Live Sky** (`Code.py`)
   - Real-time star mapping using Skyfield and Hipparcos catalog
   - Live GPS integration for location-based sky views
   - Interactive star identification

2. **Quantum-Enhanced Sky Watcher** (`Code2.py`)
   - Advanced anomaly detection using quantum computing
   - Comet tracking with real-time data
   - Data analysis with Pandas and NumPy

3. **Binary Transmitter** (`Cod 3.py`)
   - Binary signal transmission visualization
   - Quantum sensor simulation
   - Defense shield visualization

4. **Entropy-Driven Atlas Observation** (`Mas2.py`)
   - Real telescope data from SIMBAD
   - Quantum clarity factors
   - Entropy-driven zoom control

5. **Radio Waves Simulation** (`Radio waves.py`)
   - Quantum radio wave propagation
   - QuTiP quantum simulations
   - Space signal modeling

6. **Solar Defense** (`Solar defense.py`)
   - Solar activity monitoring
   - Defense system visualization

7. **The Sun 3D** (`Thesun3d.py`)
   - Three-dimensional solar visualization

8. **Mas View** (`Mas view.py`)
   - Mass observation and analysis

9. **3D Atlas** (`3d atlas.py`)
   - Three-dimensional cosmic atlas

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/stephenvegagarcia/Idu-interstellar-detect-unknown.git
   cd Idu-interstellar-detect-unknown
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the Flask application**
   ```bash
   python app.py
   # or
   python run.py
   
   # For development with debug mode (NOT for production):
   python run.py --debug
   ```

2. **Access the web interface**
   - Open your web browser
   - Navigate to `http://localhost:5000`

3. **Launch a visualization**
   - Click on any visualization card from the home page
   - Click "Launch Visualization" to view details
   - Click "Run Visualization" to start the application
   - The visualization will open in a new window

4. **Stop a visualization**
   - Click "Stop Visualization" to close the running application

5. **View source code**
   - Click "View Code" to see the source code of any visualization

## Project Structure

```
Idu-interstellar-detect-unknown/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/            # HTML templates
│   ├── base.html        # Base template
│   ├── index.html       # Home page
│   ├── view.html        # Visualization view page
│   └── about.html       # About page
├── static/              # Static files
│   └── css/
│       └── style.css    # Application styles
└── [visualization files]
    ├── Code.py
    ├── Code2.py
    ├── Cod 3.py
    ├── Mas2.py
    ├── Mas view.py
    ├── Radio waves.py
    ├── Solar defense.py
    ├── Thesun3d.py
    └── 3d atlas.py
```

## Technologies Used

- **Flask**: Web framework
- **Pygame**: Graphics and visualization
- **OpenCV**: Computer vision and image processing
- **NumPy & Pandas**: Data analysis
- **Skyfield**: Astronomical calculations
- **QuTiP**: Quantum computing simulations
- **Astropy**: Astronomical data analysis

## API Endpoints

The application provides several REST API endpoints:

- `GET /` - Home page
- `GET /view/<viz_id>` - Visualization detail page
- `POST /api/run/<viz_id>` - Start a visualization
- `POST /api/stop/<viz_id>` - Stop a visualization
- `GET /api/status/<viz_id>` - Check visualization status
- `GET /api/code/<viz_id>` - Get source code
- `GET /about` - About page

## Requirements

All dependencies are listed in `requirements.txt`:

- Flask 3.0.0
- Pygame 2.5.2
- NumPy 1.24.3
- Pandas 2.0.3
- OpenCV-Python 4.8.1.78
- Skyfield 1.46
- Plyer 2.1.0
- QuTiP 4.7.3
- SciPy 1.11.4
- scikit-image 0.21.0
- Astropy 5.3.4
- Requests 2.31.0

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

2. **Display Issues**
   - Some visualizations require a display (X11 on Linux)
   - On headless servers, use virtual displays (Xvfb)

3. **Permission Errors**
   - Ensure proper file permissions
   - Run with appropriate user privileges

4. **Port Already in Use**
   - Change the port in `app.py`: `app.run(port=5001)`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

See the LICENSE file for details.

## Author

Stephen Vega Garcia

## Acknowledgments

- Skyfield for astronomical calculations
- QuTiP for quantum computing simulations
- The open-source community for various libraries and tools

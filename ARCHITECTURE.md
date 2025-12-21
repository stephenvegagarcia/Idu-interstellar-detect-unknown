# Flask Application Architecture

## Overview
This Flask web application integrates all the astronomical visualization Python scripts into a single, easy-to-use web interface.

## Architecture Components

### 1. Backend (Flask)
**File:** `app.py`

#### Key Features:
- **Process Management**: Manages subprocess execution of each visualization
- **REST API**: Provides endpoints for controlling visualizations
- **Code Viewer**: Serves source code for inspection
- **Status Monitoring**: Tracks running processes

#### API Endpoints:
```
GET  /                      - Home page with all visualizations
GET  /view/<viz_id>         - Individual visualization control page
GET  /about                 - About page
POST /api/run/<viz_id>      - Start a visualization
POST /api/stop/<viz_id>     - Stop a visualization  
GET  /api/status/<viz_id>   - Get visualization status
GET  /api/code/<viz_id>     - Get source code
```

### 2. Frontend (HTML/CSS/JS)

#### Templates:
- **base.html**: Base template with navigation and layout
- **index.html**: Grid view of all visualizations
- **view.html**: Individual visualization control panel
- **about.html**: Project information

#### Styling:
- **style.css**: Cosmic-themed responsive design
- Dark space background with cyan accents
- Animated status indicators
- Responsive grid layout
- Modal dialogs for code viewing

### 3. Visualization Scripts

Each Python script runs independently:
- **Code.py**: Star mapper with live GPS
- **Code2.py**: Quantum sky watcher
- **Cod 3.py**: Binary transmitter
- **Mas2.py**: Atlas observation
- **Mas view.py**: Mass view
- **Radio waves.py**: Radio wave simulation
- **Solar defense.py**: Solar defense system
- **Thesun3d.py**: 3D Sun visualization
- **3d atlas.py**: 3D cosmic atlas

## How It Works

### Starting a Visualization
1. User clicks "Launch Visualization" on home page
2. Browser navigates to `/view/<viz_id>`
3. User clicks "Run Visualization"
4. JavaScript sends POST to `/api/run/<viz_id>`
5. Flask starts subprocess with `subprocess.Popen()`
6. Process ID stored in `running_processes` dict
7. Visualization window opens on user's desktop
8. Status updates via polling `/api/status/<viz_id>`

### Stopping a Visualization
1. User clicks "Stop Visualization"
2. JavaScript sends POST to `/api/stop/<viz_id>`
3. Flask terminates subprocess
4. Process removed from tracking dict
5. Status indicator updates to "Stopped"

### Viewing Code
1. User clicks "View Code"
2. JavaScript sends GET to `/api/code/<viz_id>`
3. Flask reads source file
4. Returns code as JSON
5. Modal displays formatted code

## Process Management

### Running Processes Dictionary
```python
running_processes = {
    'viz_id': subprocess.Popen object,
    ...
}
```

### Cleanup on Shutdown
The `cleanup_processes()` function ensures all subprocesses are terminated when Flask shuts down.

## Security Considerations

1. **No Remote Execution**: Scripts only run on local machine
2. **Read-Only Code Access**: Source code is read but not executed through API
3. **Process Isolation**: Each visualization runs in separate subprocess
4. **Local Host Only**: Default binding to localhost (change for network access)

## Customization

### Adding New Visualizations
1. Add Python file to repository
2. Update `VISUALIZATIONS` dict in `app.py`:
```python
VISUALIZATIONS = {
    'new_viz': {'name': 'New Visualization', 'file': 'newviz.py'},
    ...
}
```

### Changing Port
Edit `app.py` or `run.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Custom Styling
Modify `static/css/style.css` to change colors, layout, etc.

## Dependencies

### Flask App Dependencies:
- Flask 3.0.0
- Werkzeug 3.1.0

### Visualization Dependencies:
- Pygame (graphics)
- OpenCV (image processing)
- NumPy, Pandas (data)
- Skyfield (astronomy)
- QuTiP (quantum)
- And more (see requirements.txt)

## Development vs Production

### Development (Current)
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

### Production (Recommended)
Use a WSGI server like Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## File Structure
```
.
├── app.py                    # Flask application
├── run.py                    # Startup script
├── requirements.txt          # Dependencies
├── README.md                 # Main documentation
├── QUICKSTART.md            # Quick start guide
├── ARCHITECTURE.md          # This file
├── templates/               # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── view.html
│   └── about.html
├── static/                  # Static files
│   └── css/
│       └── style.css
└── [visualization files]    # Individual Python scripts
```

## Browser Compatibility

Tested on:
- Chrome/Chromium
- Firefox
- Safari
- Edge

## Future Enhancements

Potential additions:
- WebSocket support for real-time updates
- Screenshot/video capture from visualizations
- Embedded visualization (if possible with Pygame)
- User authentication
- Visualization parameters configuration
- Saved presets
- Logs viewing
- Performance metrics

## Troubleshooting

### Common Issues:

1. **Visualization won't start**
   - Check dependencies installed
   - Verify Python script has no syntax errors
   - Check system has display capability

2. **Port conflict**
   - Change port in app.py
   - Kill process using port 5000

3. **Can't stop visualization**
   - Process may have crashed
   - Use system process manager
   - Restart Flask app

## Contributing

When adding new features:
1. Test thoroughly
2. Update documentation
3. Follow existing code style
4. Add to requirements.txt if needed

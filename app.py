"""
Flask Web Application for Interstellar Detection Visualizations
This app provides web interface to run various astronomical visualization scripts
"""
from flask import Flask, render_template, jsonify
import subprocess
import os

app = Flask(__name__)

# Dictionary to store running processes
running_processes = {}

# Available visualizations
VISUALIZATIONS = {
    'code': {'name': 'Star Mapper - Live Sky', 'file': 'Code.py'},
    'code2': {'name': 'Quantum-Enhanced Sky Watcher', 'file': 'Code2.py'},
    'code3': {'name': 'Binary Transmitter', 'file': 'Cod 3.py'},
    'mas2': {'name': 'Entropy-Driven Atlas Observation', 'file': 'Mas2.py'},
    'mas_view': {'name': 'Mas View', 'file': 'Mas view.py'},
    'radio_waves': {'name': 'Radio Waves Simulation', 'file': 'Radio waves.py'},
    'solar_defense': {'name': 'Solar Defense', 'file': 'Solar defense.py'},
    'thesun3d': {'name': 'The Sun 3D', 'file': 'Thesun3d.py'},
    '3d_atlas': {'name': '3D Atlas', 'file': '3d atlas.py'}
}

@app.route('/')
def index():
    """Main page showing all available visualizations"""
    return render_template('index.html', visualizations=VISUALIZATIONS)

@app.route('/view/<viz_id>')
def view_visualization(viz_id):
    """View page for a specific visualization"""
    if viz_id not in VISUALIZATIONS:
        return "Visualization not found", 404
    
    viz = VISUALIZATIONS[viz_id]
    return render_template('view.html', viz_id=viz_id, viz_name=viz['name'])

@app.route('/api/run/<viz_id>', methods=['POST'])
def run_visualization(viz_id):
    """API endpoint to run a visualization script"""
    if viz_id not in VISUALIZATIONS:
        return jsonify({'error': 'Visualization not found'}), 404
    
    viz_file = VISUALIZATIONS[viz_id]['file']
    
    # Stop any existing process for this visualization
    if viz_id in running_processes:
        stop_visualization(viz_id)
    
    try:
        # Start the visualization in a subprocess
        process = subprocess.Popen(
            ['python', viz_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        running_processes[viz_id] = process
        
        return jsonify({
            'status': 'started',
            'message': f'{VISUALIZATIONS[viz_id]["name"]} is now running'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop/<viz_id>', methods=['POST'])
def stop_visualization(viz_id):
    """API endpoint to stop a running visualization"""
    if viz_id in running_processes:
        process = running_processes[viz_id]
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        del running_processes[viz_id]
        return jsonify({
            'status': 'stopped',
            'message': f'{VISUALIZATIONS[viz_id]["name"]} has been stopped'
        })
    return jsonify({'status': 'not_running'}), 404

@app.route('/api/status/<viz_id>')
def get_status(viz_id):
    """Check if a visualization is running"""
    is_running = viz_id in running_processes and running_processes[viz_id].poll() is None
    return jsonify({
        'running': is_running,
        'viz_name': VISUALIZATIONS[viz_id]['name'] if viz_id in VISUALIZATIONS else None
    })

@app.route('/api/code/<viz_id>')
def get_code(viz_id):
    """Get the source code of a visualization"""
    if viz_id not in VISUALIZATIONS:
        return jsonify({'error': 'Visualization not found'}), 404
    
    viz_file = VISUALIZATIONS[viz_id]['file']
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), viz_file)
    
    try:
        with open(file_path, 'r') as f:
            code = f.read()
        return jsonify({'code': code})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

def cleanup_processes():
    """Clean up all running processes on shutdown"""
    for viz_id in list(running_processes.keys()):
        stop_visualization(viz_id)

if __name__ == '__main__':
    import sys
    # Only enable debug mode if explicitly requested
    debug_mode = '--debug' in sys.argv
    try:
        app.run(debug=debug_mode, host='0.0.0.0', port=5000)
    finally:
        cleanup_processes()

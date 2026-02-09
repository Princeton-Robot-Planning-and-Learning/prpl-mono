"""Flask backend for visualizing concrete states from bilevel planning graph."""

import base64
import io
import pickle
from pathlib import Path
import traceback


import numpy as np
import kinder
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

import matplotlib # Used for generating state visualizations
matplotlib.use('Agg')  # Use non-interactive backend for Flask

# Load Flask app, and enable CORS for React frontend
# Default port 5001 (5000 is used by MacOS)
app = Flask(__name__)
CORS(app) 

# Global state data dictionary, to be loaded from pickle file
STATE_DATA = {}
env_visualization = None
ENV_CONFIG = {
    'env_name': 'kinder/Obstruction2D-o1-v0',
    'num_obstructions': 1
}

# Default pickle path (can be overridden via API)
# Default path: webapp/data/bpg.pkl
DEFAULT_PICKLE_PATH = Path(__file__).parent.parent / 'data' / 'bpg.pkl'
DEFAULT_PICKLE_DIR = DEFAULT_PICKLE_PATH.parent

@app.route('/api/load_pickle', methods=['POST'])
def load_pickle():
    """Load the pickle file containing state data."""
    global STATE_DATA
    global env_visualization
    
    try:
        data = request.get_json()
        pickle_path_str = data['pickle_path']
        
        # Try multiple path resolution strategies
        pickle_path = None
        # Record attempted paths for debugging purposes
        attempted_paths = []
        
        # Strategy 1: Use as absolute path
        potential_path = Path(pickle_path_str)
        attempted_paths.append(str(potential_path))
        if potential_path.exists():
            pickle_path = potential_path
        
        # Strategy 2: Treat as relative to default directory
        if pickle_path is None:
            potential_path = DEFAULT_PICKLE_DIR / pickle_path_str
            attempted_paths.append(str(potential_path))
            if potential_path.exists():
                pickle_path = potential_path
        
        # Strategy 3: Search for filename in default directory
        if pickle_path is None:
            filename = Path(pickle_path_str).name
            potential_path = DEFAULT_PICKLE_DIR / filename
            attempted_paths.append(str(potential_path))
            if potential_path.exists():
                pickle_path = potential_path
        
        # If still not found, return error with attempted paths
        if pickle_path is None:
            return jsonify({
                'error': f'Pickle file not found: {pickle_path_str}',
                'attempted_paths': attempted_paths
            }), 404
        
        # Load the pickle file
        with open(pickle_path, 'rb') as f:
            STATE_DATA = pickle.load(f)
        
        num_states = len(STATE_DATA)
        node_ids = list(STATE_DATA.keys())[:5]  # Show first 5 node IDs
        
        print(f"Successfully loaded {num_states} states from {pickle_path}")

        # Create environment with render mode
        print(f"Creating environment: {ENV_CONFIG['env_name']}")

        kinder.register_all_environments()
        # Initialize environment for visualization
        # Only create a new env_visualization if we load another pickle
        env_visualization = kinder.make(ENV_CONFIG['env_name'], render_mode="rgb_array")
        
        return jsonify({
            'success': True,
            'num_states': num_states,
            'sample_node_ids': node_ids,
            'message': f'Loaded {num_states} states from {pickle_path.name}',
            'full_path': str(pickle_path)
        })
    
    except KeyError:
        return jsonify({'error': 'pickle_path is required in request body'}), 400
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/configure_env', methods=['POST'])
def configure_env():
    """Configure the environment settings."""
    global ENV_CONFIG
    
    try:
        data = request.get_json()
        if 'env_name' in data:
            ENV_CONFIG['env_name'] = data['env_name']
        if 'num_obstructions' in data:
            ENV_CONFIG['num_obstructions'] = data['num_obstructions']
        
        return jsonify({
            'success': True,
            'config': ENV_CONFIG
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualize_state', methods=['POST'])
def visualize_state():
    """Generate visualization for a concrete state node."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        node_id = data.get('node_id')
        if not node_id:
            return jsonify({'error': 'node_id is required'}), 400
        
        print(f"Received visualization request for node_id: {node_id}")
        
        # Check if we have state data loaded
        if not STATE_DATA:
            print("ERROR: No state data loaded")
            return jsonify({'error': 'No state data loaded. Call /api/load_pickle first.'}), 400
        
        # Check if node_id exists
        if node_id not in STATE_DATA:
            print(f"ERROR: Node ID not found: {node_id}")
            return jsonify({'error': f'Node ID not found: {node_id}'}), 404
        
        # Get the state
        state = STATE_DATA[node_id]

        # Reset environment with the state
        # .reset() loads the state into the environment
        print("Resetting environment with state...")
        options = {'init_state': state}
        obs, _ = env_visualization.reset(options=options)
        
        # Render the environment
        print("Rendering environment...")
        rgb_array = env_visualization.render()
        
        # Convert numpy array to PIL Image
        if isinstance(rgb_array, np.ndarray):
            image = Image.fromarray(rgb_array.astype('uint8'))
        else:
            image = Image.fromarray(np.array(rgb_array).astype('uint8'))
        
        print(f"Image created: {image.size}")
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                
        # Get image dimensions
        width, height = image.size
        
        print(f"Successfully generated visualization for {node_id}")
        
        return jsonify({
            'success': True,
            'node_id': node_id,
            'image': f'data:image/png;base64,{img_base64}',
            'width': width,
            'height': height,
            'state_str': str(state)
        })
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print("ERROR in visualize_state:")
        print(error_trace)
        return jsonify({
            'error': str(e),
            'traceback': error_trace
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'state_data_loaded': len(STATE_DATA) > 0,
        'num_states': len(STATE_DATA),
        'env_config': ENV_CONFIG
    })


if __name__ == '__main__':
    print('Starting Flask backend...')
    print(f'Environment: {ENV_CONFIG["env_name"]}')
    print(f'Default pickle path: {DEFAULT_PICKLE_PATH}')
    print()
    
    print('Endpoints:')
    print('POST /api/load_pickle - Load state data from pickle file')
    print('POST /api/configure_env - Configure environment settings')
    print('POST /api/visualize_state - Generate state visualization')
    print('GET  /api/health - Health check')
    print()
    print('Running on http://localhost:5001')
    print('=' * 60)
    app.run(debug=True, port=5001)

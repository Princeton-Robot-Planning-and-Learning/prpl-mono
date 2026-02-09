# Bilevel Planning Visualization Backend

This directory contains the Flask-based backend for visualizing concrete states generated from the bilevel planning graph.

## Overview

The backend is responsible for:
- Loading planning graph data from pickle files.
- Managing an instance of the `kinder` environment for generating visualizations.
- Dynamically rendering environment states into images for the frontend.
- Providing a REST API for state inspection and visualization.

Dynamic rendering is used because pre-rendering every possible state in a large planning graph is computationally expensive and storage-intensive.

## Prerequisites

- Python 3.11
- [uv](https://github.com/astral-sh/uv) (recommended)

## Installation

It is recommended to use `uv` for package management. From this directory, run:

```bash
uv pip install -r requirements.txt
```

> [!IMPORTANT]
> This project depends on the local `kinder` package in the monorepo root. Ensure you have the monorepo dependencies correctly set up.

## Running the Backend

Start the Flask server on port 5001:

```bash
python app.py
```

The server will be available at `http://localhost:5001`.

## Generating Visualization Data

To use the visualizer, you need to export a Bilevel Planning Graph (BPG) and its associated concrete states from your planning script.

The following example demonstrates how to export a planning graph using the `export_graph_with_pickle` method:

```python
from pathlib import Path
import kinder
from kinder_bilevel_planning import create_bilevel_planning_models

# 1. Initialize environment and approach
kinder.register_all_environments()
env = kinder.make("kinder/Obstruction2D-o1-v0")
env_models = create_bilevel_planning_models(
    "obstruction2d", env.observation_space, env.action_space
)

# ... [Initialize your approach here] ...

# 2. Run the planner on a problem instance
obs, _ = env.reset(seed=100)
problem = approach._observation_to_planning_problem(obs)
plan, bpg = approach._planner.run(problem, timeout=100)

# 3. Export the graph and state data for the visualizer
save_dir = Path('webapp/data')
save_dir.mkdir(exist_ok=True)

bpg.export_graph_with_pickle(
    json_path = save_dir / "bpg_graph.json",
    pickle_path = save_dir / "bpg_state_data.pkl",
    final_state = plan.states[-1] if plan else None,
)
```

After generating these files, you can load the `.pkl` file via the backend API or UI to visualize the nodes.

## API Documentation

- **`POST /api/load_pickle`**: Load planning data from a specified `.pkl` file.
- **`POST /api/visualize_state`**: Generate a base64-encoded PNG for a specific `node_id`.
- **`GET /api/health`**: Check the status of the backend and whether data is loaded.

## Frontend Integration

The React-based frontend is located in the parent `webapp/` directory. To run the full application:

1. Start this backend (`python app.py`).
2. Navigate to the parent directory (`cd ..`) and run `npm install`.
3. Run `npm run dev` to start the frontend development server. The server will be available at `http://localhost:3000`.

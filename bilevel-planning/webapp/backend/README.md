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

## API Documentation

- **`POST /api/load_pickle`**: Load planning data from a specified `.pkl` file.
- **`POST /api/visualize_state`**: Generate a base64-encoded PNG for a specific `node_id`.
- **`GET /api/health`**: Check the status of the backend and whether data is loaded.

## Frontend Integration

The React-based frontend is located in the parent `webapp/` directory. To run the full application:

1. Start this backend (`python app.py`).
2. Navigate to the parent directory (`cd ..`) and run `npm install`.
3. Run `npm run dev` to start the frontend development server. The server will be available at `http://localhost:3000`.

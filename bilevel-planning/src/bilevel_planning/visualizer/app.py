"""Flask backend for visualizing concrete states from a bilevel planning graph.

This module is environment-agnostic: it does not import or construct any
simulation environment. Callers supply a ``render_state_fn`` that maps a
concrete state (as stored in the pickled bilevel planning graph) to an RGB
image. Launch the webapp by importing ``run_webapp`` from your own script,
where you own the env construction.
"""

# pylint: disable=global-statement

import base64
import io
import pickle
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS  # type: ignore[import-untyped]
from PIL import Image  # type: ignore[import-untyped]

RenderStateFn = Callable[[Any], np.ndarray]

STATE_DATA: dict = {}


def _resolve_pickle_path(
    pickle_path_str: str, data_dir: Path
) -> tuple[Path | None, list[str]]:
    attempted: list[str] = []
    for candidate in (
        Path(pickle_path_str),
        data_dir / pickle_path_str,
        data_dir / Path(pickle_path_str).name,
    ):
        attempted.append(str(candidate))
        if candidate.exists():
            return candidate, attempted
    return None, attempted


def create_app(render_state_fn: RenderStateFn, data_dir: Path) -> Flask:
    """Build the Flask app with the given state renderer."""
    app = Flask(__name__)
    CORS(app)

    @app.route("/api/load_pickle", methods=["POST"])
    def load_pickle():
        global STATE_DATA
        try:
            data = request.get_json()
            pickle_path_str = data["pickle_path"]
            pickle_path, attempted_paths = _resolve_pickle_path(
                pickle_path_str, data_dir
            )
            if pickle_path is None:
                return (
                    jsonify(
                        {
                            "error": f"Pickle file not found: {pickle_path_str}",
                            "attempted_paths": attempted_paths,
                        }
                    ),
                    404,
                )

            with open(pickle_path, "rb") as f:
                STATE_DATA = pickle.load(f)

            num_states = len(STATE_DATA)
            node_ids = list(STATE_DATA.keys())[:5]
            return jsonify(
                {
                    "success": True,
                    "num_states": num_states,
                    "sample_node_ids": node_ids,
                    "message": f"Loaded {num_states} states from {pickle_path.name}",
                    "full_path": str(pickle_path),
                }
            )

        except KeyError:
            return jsonify({"error": "pickle_path is required in request body"}), 400
        except Exception as e:  # pylint: disable=broad-exception-caught
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    @app.route("/api/visualize_state", methods=["POST"])
    def visualize_state():
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            node_id = data.get("node_id")
            if not node_id:
                return jsonify({"error": "node_id is required"}), 400

            if not STATE_DATA:
                return (
                    jsonify(
                        {"error": "No state data loaded. Call /api/load_pickle first."}
                    ),
                    400,
                )

            if node_id not in STATE_DATA:
                return jsonify({"error": f"Node ID not found: {node_id}"}), 404

            state = STATE_DATA[node_id]
            rgb_array = render_state_fn(state)
            image = Image.fromarray(np.asarray(rgb_array).astype("uint8"))

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            width, height = image.size

            return jsonify(
                {
                    "success": True,
                    "node_id": node_id,
                    "image": f"data:image/png;base64,{img_base64}",
                    "width": width,
                    "height": height,
                    "state_str": str(state),
                }
            )

        except Exception as e:  # pylint: disable=broad-exception-caught
            return (
                jsonify({"error": str(e), "traceback": traceback.format_exc()}),
                500,
            )

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "status": "healthy",
                "state_data_loaded": len(STATE_DATA) > 0,
                "num_states": len(STATE_DATA),
            }
        )

    return app


def run_webapp(
    render_state_fn: RenderStateFn,
    data_dir: Path | str | None = None,
    port: int = 5001,
    debug: bool = True,
) -> None:
    """Run the visualization Flask server.

    Callers supply ``render_state_fn``, which takes a concrete state from the
    loaded pickle and returns an HxWx3 uint8 RGB array. The webapp itself has
    no knowledge of the underlying environment.

    ``data_dir`` is the directory searched when the frontend requests a pickle
    by bare filename. Defaults to ``./webapp/data`` under the current working
    directory.

    Planned follow-ups (tracked here and in related visualizer files):
      * Replace the JSON + pickle pair with a single pickle artifact and add
        a ``/api/graph`` endpoint so the frontend fetches the topology from
        the backend instead of uploading JSON itself.
      * Add a ``/api/set_renderer`` endpoint that accepts Python source for
        ``render_state_fn``, ``exec``s it, and caches the resulting callable.
        Lets users launch the webapp with no bespoke script and write their
        render function in a browser editor pane. Localhost-only — not for
        hosted deployment.
      * Move the React frontend under ``src/bilevel_planning/visualizer/
        frontend/`` so all visualizer code lives under one name.
    """
    resolved_data_dir = (
        Path(data_dir) if data_dir is not None else Path.cwd() / "webapp" / "data"
    )
    app = create_app(render_state_fn, resolved_data_dir)
    print(f"Starting Flask backend on http://localhost:{port}")
    print(f"Pickle data directory: {resolved_data_dir}")
    app.run(debug=debug, port=port)

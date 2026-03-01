"""Flask server for the Blockly visual programming interface."""

import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, Response, request
from PIL import Image

from kinder_blockly.executor import execute_program, render_initial_frame

STATIC_DIR = Path(__file__).parent / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")


@app.route("/")
def index() -> Response:
    """Serve the main page."""
    return app.send_static_file("index.html")


def _encode_frame(frame: "np.ndarray[Any, Any]") -> str:
    """Encode a numpy RGB frame as a base64 JPEG string."""
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.route("/reset", methods=["POST"])
def reset_env() -> Response:
    """Reset the environment and return the initial frame."""
    data = request.get_json() or {}
    seed = data.get("seed", 0)
    try:
        frame = render_initial_frame(seed=seed)
        return Response(
            json.dumps({"frame": _encode_frame(frame)}),
            status=200,
            mimetype="application/json",
        )
    except Exception as exc:  # pylint: disable=broad-except
        return Response(
            json.dumps({"error": str(exc)}),
            status=500,
            mimetype="application/json",
        )


@app.route("/run", methods=["POST"])
def run_program() -> Response:
    """Execute a Blockly program and return rendered frames."""
    data = request.get_json()
    if data is None:
        return Response(
            json.dumps({"error": "Invalid JSON"}),
            status=400,
            mimetype="application/json",
        )

    program = data.get("program", {})
    seed = data.get("seed", 0)

    frames_b64: list[str] = []
    try:
        for frame in execute_program(program, seed=seed):
            frames_b64.append(_encode_frame(frame))
    except Exception as exc:  # pylint: disable=broad-except
        return Response(
            json.dumps({"error": str(exc), "frames": frames_b64}),
            status=500,
            mimetype="application/json",
        )

    return Response(
        json.dumps({"success": True, "frames": frames_b64}),
        status=200,
        mimetype="application/json",
    )


def main() -> None:
    """Run the development server."""
    app.run(host="127.0.0.1", port=5000, debug=True)


if __name__ == "__main__":
    main()

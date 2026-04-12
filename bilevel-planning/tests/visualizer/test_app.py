"""Smoke tests for the visualizer Flask backend.

Uses Flask's in-process test client so these tests run inside pytest without starting a
real server.
"""

import base64
import io
import pickle
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.visualizer import app as visualizer_app
from bilevel_planning.visualizer.app import create_app


@pytest.fixture(autouse=True)
def _reset_state_data():
    """Clear the module-level STATE_DATA cache before each test.

    ``app.py`` stores loaded pickle contents in a module global so the Flask
    routes can see them across requests. That's fine for a running backend
    but leaks between tests; reset it so each test starts fresh.
    """
    visualizer_app.STATE_DATA = {}
    yield
    visualizer_app.STATE_DATA = {}


def _constant_color_renderer(state: np.ndarray) -> np.ndarray:
    """Render each state as a small solid-color patch."""
    color = np.asarray(state, dtype=np.uint8).reshape(-1)[:3]
    if color.size < 3:
        color = np.pad(color, (0, 3 - color.size))
    return np.broadcast_to(color, (8, 8, 3)).astype(np.uint8)


def _write_demo_pickle(tmp_path: Path) -> tuple[Path, list[str]]:
    """Build a small BPG, export its state pickle, return the path and node ids."""
    bpg: BilevelPlanningGraph = BilevelPlanningGraph()
    states = [
        np.array([220, 40, 40], dtype=np.uint8),
        np.array([40, 90, 210], dtype=np.uint8),
        np.array([50, 180, 70], dtype=np.uint8),
    ]
    for s in states:
        bpg.add_state_node(s)
    bpg.add_abstract_state_node("start")
    bpg.add_abstract_state_node("end")
    bpg.add_state_abstractor_edge(states[0], "start")
    bpg.add_state_abstractor_edge(states[-1], "end")
    bpg.add_abstract_action_edge("start", "go", "end")
    for a, b in zip(states[:-1], states[1:]):
        bpg.add_action_edge(a, "step", b)

    pickle_path = tmp_path / "demo.pkl"
    bpg.export_state_data_pickle(pickle_path)

    # Load the pickle we just wrote to discover the node ids without
    # touching BilevelPlanningGraph's private hashing method.
    with open(pickle_path, "rb") as f:
        state_data = pickle.load(f)
    return pickle_path, list(state_data.keys())


def test_health_endpoint(tmp_path: Path):
    """``/api/health`` reports an empty backend before any pickle is loaded."""
    app = create_app(_constant_color_renderer, data_dir=tmp_path)
    client = app.test_client()
    resp = client.get("/api/health")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["status"] == "healthy"
    assert payload["state_data_loaded"] is False
    assert payload["num_states"] == 0


def test_load_and_visualize_roundtrip(tmp_path: Path):
    """Loading a pickle and visualizing a node returns a decodable PNG."""
    pickle_path, node_ids = _write_demo_pickle(tmp_path)

    app = create_app(_constant_color_renderer, data_dir=tmp_path)
    client = app.test_client()

    # Load the pickle by bare filename (exercises data_dir resolution).
    resp = client.post("/api/load_pickle", json={"pickle_path": pickle_path.name})
    assert resp.status_code == 200, resp.get_json()
    payload = resp.get_json()
    assert payload["success"] is True
    assert payload["num_states"] == len(node_ids)

    # Visualize one of the nodes and confirm the PNG decodes correctly.
    target = node_ids[0]
    resp = client.post("/api/visualize_state", json={"node_id": target})
    assert resp.status_code == 200, resp.get_json()
    payload = resp.get_json()
    assert payload["success"] is True
    assert payload["node_id"] == target
    assert payload["image"].startswith("data:image/png;base64,")

    png_b64 = payload["image"].split(",", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(png_b64)))
    assert image.size == (8, 8)
    assert image.mode in ("RGB", "RGBA")


def test_visualize_unknown_node_returns_404(tmp_path: Path):
    """Asking for a node id that isn't in the loaded pickle returns 404."""
    pickle_path, _ = _write_demo_pickle(tmp_path)

    app = create_app(_constant_color_renderer, data_dir=tmp_path)
    client = app.test_client()
    client.post("/api/load_pickle", json={"pickle_path": pickle_path.name})

    resp = client.post("/api/visualize_state", json={"node_id": "x:doesnotexist"})
    assert resp.status_code == 404

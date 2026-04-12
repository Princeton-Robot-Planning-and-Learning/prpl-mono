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
def _reset_module_state():
    """Clear the module-level caches before each test.

    ``app.py`` stores the loaded bundle in two module globals so the Flask
    routes can see them across requests. That's fine for a running backend
    but leaks between tests; reset both so each test starts fresh.
    """
    visualizer_app.GRAPH_DATA = {}
    visualizer_app.STATE_DATA = {}
    yield
    visualizer_app.GRAPH_DATA = {}
    visualizer_app.STATE_DATA = {}


def _constant_color_renderer(state: np.ndarray) -> np.ndarray:
    """Render each state as a small solid-color patch."""
    color = np.asarray(state, dtype=np.uint8).reshape(-1)[:3]
    if color.size < 3:
        color = np.pad(color, (0, 3 - color.size))
    return np.broadcast_to(color, (8, 8, 3)).astype(np.uint8)


def _write_demo_bundle(tmp_path: Path) -> tuple[Path, list[str]]:
    """Build a small BPG, export the bundle, return the path and node ids."""
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

    bundle_path = tmp_path / "demo.pkl"
    bpg.export(bundle_path, final_state=states[-1])

    # Load the pickle we just wrote to discover the node ids without
    # touching BilevelPlanningGraph's private hashing method.
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle_path, list(bundle["states"].keys())


def test_health_endpoint(tmp_path: Path):
    """``/api/health`` reports an empty backend before any pickle is loaded."""
    app = create_app(_constant_color_renderer, data_dir=tmp_path)
    client = app.test_client()
    resp = client.get("/api/health")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["status"] == "healthy"
    assert payload["graph_loaded"] is False
    assert payload["num_states"] == 0


def test_graph_endpoint_requires_load(tmp_path: Path):
    """``/api/graph`` returns 400 until a bundle has been loaded."""
    app = create_app(_constant_color_renderer, data_dir=tmp_path)
    client = app.test_client()
    resp = client.get("/api/graph")
    assert resp.status_code == 400


def test_load_graph_and_visualize_roundtrip(tmp_path: Path):
    """Loading a bundle exposes both the topology and per-node rendering."""
    bundle_path, node_ids = _write_demo_bundle(tmp_path)

    app = create_app(_constant_color_renderer, data_dir=tmp_path)
    client = app.test_client()

    resp = client.post("/api/load_pickle", json={"pickle_path": bundle_path.name})
    assert resp.status_code == 200, resp.get_json()
    payload = resp.get_json()
    assert payload["success"] is True
    assert payload["num_states"] == len(node_ids)

    # After load, /api/graph serves the topology.
    resp = client.get("/api/graph")
    assert resp.status_code == 200
    graph = resp.get_json()
    assert set(graph.keys()) >= {"nodes", "edges", "plan", "config"}

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
    """Asking for a node id that isn't in the loaded bundle returns 404."""
    bundle_path, _ = _write_demo_bundle(tmp_path)

    app = create_app(_constant_color_renderer, data_dir=tmp_path)
    client = app.test_client()
    client.post("/api/load_pickle", json={"pickle_path": bundle_path.name})

    resp = client.post("/api/visualize_state", json={"node_id": "x:doesnotexist"})
    assert resp.status_code == 404


def test_load_pickle_rejects_wrong_shape(tmp_path: Path):
    """A pickle that isn't a ``{'graph':..., 'states':...}`` bundle is rejected."""
    bad_path = tmp_path / "bad.pkl"
    with open(bad_path, "wb") as f:
        pickle.dump({"only_states": {}}, f)

    app = create_app(_constant_color_renderer, data_dir=tmp_path)
    client = app.test_client()
    resp = client.post("/api/load_pickle", json={"pickle_path": bad_path.name})
    assert resp.status_code == 400

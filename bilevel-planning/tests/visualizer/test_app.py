"""Smoke tests for the visualizer Flask backend.

Uses Flask's in-process test client so these tests run inside pytest without starting a
real server. The bundle and renderer are loaded via the same helpers the launcher uses
(``load_bundle_from_path`` / ``load_renderer_from_path``).
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
from bilevel_planning.visualizer.app import (
    create_app,
    load_bundle_from_path,
    load_renderer_from_path,
)

# Source for a renderer file; turns the (3,) state arrays from
# _write_demo_bundle into a small solid-color patch so the PNG round-trip is
# easy to assert against.
RENDERER_SOURCE = """
import numpy as np
def render_state(state):
    color = np.asarray(state, dtype=np.uint8).reshape(-1)[:3]
    if color.size < 3:
        color = np.pad(color, (0, 3 - color.size))
    return np.broadcast_to(color, (8, 8, 3)).astype(np.uint8)
"""


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Clear the module-level caches before and after each test.

    ``app.py`` stores the loaded bundle and the current renderer in module
    globals so the Flask routes can see them across requests. Reset all
    three so each test starts fresh.
    """
    visualizer_app.GRAPH_DATA = {}
    visualizer_app.STATE_DATA = {}
    visualizer_app.RENDER_FN = None
    yield
    visualizer_app.GRAPH_DATA = {}
    visualizer_app.STATE_DATA = {}
    visualizer_app.RENDER_FN = None


def _write_demo_bundle(tmp_path: Path) -> tuple[Path, list[str]]:
    """Build a small BPG, write its bundle to disk, return the path and node ids."""
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

    bundle_path = tmp_path / "bundle.pkl"
    bpg.export(bundle_path, final_state=states[-1])
    node_ids = list(pickle.loads(bundle_path.read_bytes())["states"].keys())
    return bundle_path, node_ids


def _write_renderer(tmp_path: Path, source: str = RENDERER_SOURCE) -> Path:
    renderer_path = tmp_path / "renderer.py"
    renderer_path.write_text(source, encoding="utf-8")
    return renderer_path


def test_health_endpoint_reports_empty_boot():
    """``/api/health`` reports an empty backend before anything is loaded."""
    app = create_app()
    client = app.test_client()
    resp = client.get("/api/health")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["status"] == "healthy"
    assert payload["graph_loaded"] is False
    assert payload["num_states"] == 0
    assert payload["renderer_ready"] is False


def test_graph_endpoint_requires_load():
    """``/api/graph`` returns 400 until a bundle has been loaded."""
    app = create_app()
    client = app.test_client()
    resp = client.get("/api/graph")
    assert resp.status_code == 400


def test_preload_and_visualize_roundtrip(tmp_path: Path):
    """Preload a bundle + renderer from disk, then render a node end-to-end."""
    bundle_path, node_ids = _write_demo_bundle(tmp_path)
    renderer_path = _write_renderer(tmp_path)

    num_states = load_bundle_from_path(bundle_path)
    assert num_states == len(node_ids)
    load_renderer_from_path(renderer_path)

    app = create_app()
    client = app.test_client()

    # Health reflects the preloaded state.
    health = client.get("/api/health").get_json()
    assert health["graph_loaded"] is True
    assert health["renderer_ready"] is True
    assert health["num_states"] == len(node_ids)

    # /api/graph serves the topology.
    resp = client.get("/api/graph")
    assert resp.status_code == 200
    graph = resp.get_json()
    assert set(graph.keys()) >= {"nodes", "edges", "plan", "config"}

    # Visualization succeeds and the PNG decodes.
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
    load_bundle_from_path(bundle_path)
    load_renderer_from_path(_write_renderer(tmp_path))

    app = create_app()
    client = app.test_client()
    resp = client.post("/api/visualize_state", json={"node_id": "x:doesnotexist"})
    assert resp.status_code == 404


def test_load_bundle_rejects_wrong_shape(tmp_path: Path):
    """A pickle that isn't a ``{'graph':..., 'states':...}`` bundle is rejected."""
    bad_path = tmp_path / "bad.pkl"
    bad_path.write_bytes(pickle.dumps({"only_states": {}}))
    with pytest.raises(ValueError, match="visualizer bundle"):
        load_bundle_from_path(bad_path)


def test_load_renderer_rejects_file_without_entrypoint(tmp_path: Path):
    """A renderer file that doesn't define ``render_state`` is rejected."""
    renderer_path = _write_renderer(tmp_path, source="x = 1\n")
    with pytest.raises(ValueError, match="render_state"):
        load_renderer_from_path(renderer_path)


def test_load_renderer_propagates_exec_errors(tmp_path: Path):
    """A renderer file that raises during exec surfaces the error."""
    renderer_path = _write_renderer(tmp_path, source="raise RuntimeError('nope')\n")
    with pytest.raises(RuntimeError, match="nope"):
        load_renderer_from_path(renderer_path)

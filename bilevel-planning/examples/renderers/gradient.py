"""Renderer for the simple_two_state bundle.

Maps each concrete state ``(i,)`` to a solid color along a red-to-green
gradient, so clicking the chain ``c0 -> c1 -> c2 -> c3 -> c4`` walks visibly
from red to green. Pass this file to the visualizer with ``--renderer``.
"""

import numpy as np


def render_state(state):
    """Return a 256x256x3 uint8 image whose color encodes the state index."""
    (index,) = state
    t = index / 4.0
    color = np.array([int(255 * (1 - t)), int(255 * t), 0], dtype=np.uint8)
    return np.broadcast_to(color, (256, 256, 3)).astype(np.uint8)

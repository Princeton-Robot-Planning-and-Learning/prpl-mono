"""Utilities."""

import hashlib
import os
import textwrap
import time
from contextlib import contextmanager
from dataclasses import fields
from functools import cached_property
from pathlib import Path
from typing import Any, Collection, Generator

import graphviz
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from prpl_utils.structs import Image

_NOT_FOUND = object()


class _DISABLED_cached_property_until_field_change(cached_property):
    """Decorator that caches a property in a dataclass until any field is
    changed.

    This descriptor is currently disabled because it does not play well
    with pylint. For example, see
    https://stackoverflow.com/questions/74523859/

    It is left here in case future versions of python / pylint have better
    support for custom property-like descriptors.
    """

    def __get__(self, instance, owner=None):
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property_until_field_change instance "
                "without calling __set_name__ on it."
            )
        try:
            cache = instance.__dict__
        except AttributeError:  # not all objects have __dict__
            msg = (
                f"No '__dict__' attribute on {type(instance).__name__!r} "
                f"instance to cache {self.attrname!r} property."
            )
            raise TypeError(msg) from None
        field_key = f"_cached_property_until_field_change_{self.attrname}_field"
        cur_field_vals = tuple(getattr(instance, f.name) for f in fields(instance))
        last_field_vals = cache.get(field_key, _NOT_FOUND)
        prop_key = f"_cached_property_until_field_change_{self.attrname}_property"
        if cur_field_vals == last_field_vals:
            return cache[prop_key]
        # Fields were updated, so we need to recompute and update both caches.
        new_prop_val = self.func(instance)
        cache[field_key] = cur_field_vals
        cache[prop_key] = new_prop_val
        return new_prop_val


def fig2data(fig: plt.Figure) -> Image:
    """Convert matplotlib figure into Image."""
    fig.canvas.draw()
    return np.array(fig.canvas.renderer.buffer_rgba())  # type: ignore


def wrap_angle(angle: float) -> float:
    """Wrap an angle in radians to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def get_signed_angle_distance(target: float, source: float) -> float:
    """Given two angles between [-pi, pi], get the smallest signed angle d s.t.

    source + d = target.
    """
    assert -np.pi <= source <= np.pi
    assert -np.pi <= target <= np.pi
    a = target - source
    return (a + np.pi) % (2 * np.pi) - np.pi


def draw_dag(edges: Collection[tuple[str, str]], outfile: Path) -> None:
    """Draw a DAG using graphviz."""
    if not outfile.parent.exists():
        os.makedirs(outfile.parent)
    intermediate_dot_file = outfile.parent / outfile.stem
    assert not intermediate_dot_file.exists()
    dot = graphviz.Digraph(format=outfile.suffix[1:])
    nodes = {e[0] for e in edges} | {e[1] for e in edges}
    for node in nodes:
        dot.node(node)
    for node1, node2 in edges:
        dot.edge(node1, node2)
    dot.render(outfile.stem, directory=outfile.parent)
    os.remove(intermediate_dot_file)
    print(f"Wrote out to {outfile}")


def consistent_hash(obj: Any) -> int:
    """A hash function that is consistent between sessions, unlike hash().

    This function assumes that `repr(obj)` is a precise and unambiguous representation
    of `obj`. In practice, this may not be true for some objects. For example, numpy
    arrays can be ambiguous, so we handle them as a special case within this function,
    since they are common in our code.
    """
    # See note in docstring.
    if isinstance(obj, np.ndarray):
        obj_bytes = (
            obj.tobytes()
            + str(obj.shape).encode("utf-8")
            + str(obj.dtype).encode("utf-8")
        )
    else:
        obj_str = repr(obj)
        obj_bytes = obj_str.encode("utf-8")
    hash_hex = hashlib.sha256(obj_bytes).hexdigest()
    hash_int = int(hash_hex, 16)
    # Mimic Python's built-in hash() behavior by returning a 64-bit signed int.
    # This makes it comparable to hash()'s output range.
    return hash_int if hash_int < 2**63 else hash_int - 2**6


def render_textbox_on_image(
    img: Image,
    text: str,
    text_color: tuple[int, ...] = (255, 255, 255, 255),
    left_offset_frac: float = 0.2,
    right_offset_frac: float = 0.2,
    top_offset_frac: float = 0.05,
    bottom_offset_frac: float = 0.85,
    max_chars_per_line: int | None = None,
    textbox_color: tuple[int, ...] | None = None,
) -> Image:
    """Add a textbox on an image."""
    if max_chars_per_line is not None:
        text = "\n".join(textwrap.wrap(text, width=max_chars_per_line))

    img_height, img_width = img.shape[:2]
    x = left_offset_frac * img_width
    width = (1 - (left_offset_frac + right_offset_frac)) * img_width
    y = top_offset_frac * img_height
    height = (1 - (bottom_offset_frac + top_offset_frac)) * img_height
    text_x = x + width / 2
    text_y = y + height / 2

    pil_img = PILImage.fromarray(img)  # type: ignore
    draw = ImageDraw.Draw(pil_img)

    if textbox_color is not None:
        draw.rectangle([(x, y), (x + width, y + height)], fill=textbox_color)

    font_size = 100
    font = ImageFont.load_default(font_size)
    while font_size > 1:
        font = font.font_variant(size=font_size)  # type: ignore
        bb = draw.multiline_textbbox(
            (text_x, text_y), text, font=font, anchor="mm", font_size=font_size
        )
        if bb[0] >= x and bb[1] >= y and bb[2] <= x + width and bb[3] <= y + height:
            break
        font_size -= 1

    draw.text(
        (text_x, text_y),
        text,
        font=font,
        fill=text_color,
        anchor="mm",
        font_size=font_size,
    )

    return np.array(pil_img)


def sample_seed_from_rng(rng: np.random.Generator) -> int:
    """Sample a random seed that can be used to seed another rng."""
    return int(rng.integers(0, 2**31 - 1))


def create_rng_from_rng(rng: np.random.Generator) -> np.random.Generator:
    """Create another RNG by sampling a seed from a current rng.

    Example use case: we want to call a deterministic function that uses an rng
    but the number of times that the rng is used internally to the function may
    vary (e.g., due to wall-clock timeouts). We shouldn't use the original rng
    if it will later be used by other parts of the code because it would be
    not deterministic after the function call.
    """
    seed = sample_seed_from_rng(rng)
    return np.random.default_rng(seed)


@contextmanager
def timer() -> Generator[dict[str, float], None, None]:
    """A context manager that logs the time taken to execute the block of code.

    Hat tip to Rohan Banerjee.
    """
    result: dict[str, float] = {}
    start_time = time.perf_counter()
    yield result
    end_time = time.perf_counter()
    result["time"] = end_time - start_time

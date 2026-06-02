"""Render a Vega "analytic IK on random targets" demo (the ssik showcase).

Each round draws a random reachable target pose for each arm -- shown as a glowing
RGB coordinate-frame marker -- and snaps the arm to ssik's exact solution, the
point of analytic IK: any target, solved directly with no initialization. The
arms sweep between targets only for the animation. Renders through Blender (the
markers use emissive shapes so they stay vivid) and writes a looping GIF plus an
mp4.

Standalone (not a pytest test): the Blender render takes several minutes. Run::

    python scripts/render_ssik_targets.py            # Blender (default)
    python scripts/render_ssik_targets.py pybullet   # fast preview

Requires the optional ssik extra: `pip install "prpl_kinematics[ssik]"`.
"""

from __future__ import annotations

import sys

import imageio.v2 as imageio
import numpy as np
import pybullet
from PIL import Image
from spatialmath import SE3

from prpl_kinematics.geometry.shapes import BoxShape, Shape
from prpl_kinematics.planning.configuration_space import ConfigurationSpace
from prpl_kinematics.robots import make_vega
from prpl_kinematics.tree.joints import FixedJoint
from prpl_kinematics.tree.kinematic_tree import Configuration, Edge, KinematicTree, Node
from prpl_kinematics.visualization import (
    BlenderRenderer,
    CameraParams,
    PyBulletRenderer,
    Renderer,
)

ROUNDS = 10
MOVE, HOLD, FPS = 10, 6, 24
LEFT = [f"L_arm_j{i}" for i in range(1, 8)]
RIGHT = [f"R_arm_j{i}" for i in range(1, 8)]
_SIDES = (("left", LEFT), ("right", RIGHT))
_AXES = ((1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.3, 1.0, 1.0))
_CAMERA = CameraParams(
    target=(0.5, 0.0, 0.90), distance=1.85, yaw=80.0, pitch=-10.0, width=560, height=420
)


def add_target_marker(
    tree: KinematicTree, name: str, length: float = 0.13, width: float = 0.010
) -> None:
    """Add a glowing RGB coordinate-frame marker (one node, three emissive bars)."""
    sizes = [(length, width, width), (width, length, width), (width, width, length)]
    origins = [SE3(length / 2, 0, 0), SE3(0, length / 2, 0), SE3(0, 0, length / 2)]
    visuals: list[Shape] = [
        BoxShape(size=s, origin=o, color=c, emissive=True)
        for s, o, c in zip(sizes, origins, _AXES)
    ]
    tree.add_node(Node(name, visuals=visuals))
    tree.add_edge(Edge(tree.root, name, FixedJoint(name=f"{name}_fix", origin=SE3())))


def lerp(
    spaces: dict[str, ConfigurationSpace],
    home: Configuration,
    start: Configuration,
    goal: Configuration,
    alpha: float,
) -> Configuration:
    """Joint-space interpolation of both arms between two configurations."""
    config = dict(home)
    for side, _ in _SIDES:
        space = spaces[side]
        a, b = space.to_vector(start), space.to_vector(goal)
        config.update(space.to_configuration(a + (b - a) * alpha))
    return config


def make_renderer(backend: str, tree: KinematicTree) -> Renderer:
    """A loaded Blender (high-fidelity) or PyBullet (preview) renderer."""
    if backend == "blender":
        renderer: Renderer = BlenderRenderer(samples=128)
    else:
        renderer = PyBulletRenderer(pybullet.connect(pybullet.DIRECT))
    renderer.load(tree)
    return renderer


def write_gif(path: str, frames: list[np.ndarray]) -> None:
    """Write a looping GIF with one global palette and no dither (no shimmer)."""
    sample = np.concatenate(frames[:: max(1, len(frames) // 24)], axis=0)
    palette = Image.fromarray(sample).quantize(
        colors=255, method=Image.Quantize.MEDIANCUT
    )
    quantized = [
        Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE)
        for f in frames
    ]
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=1000 / FPS,
        loop=0,
        disposal=1,
        optimize=False,
    )


def main() -> None:
    """Build the scene, solve random targets with ssik, render, and write the GIF."""
    backend = sys.argv[1] if len(sys.argv) > 1 else "blender"
    robot = make_vega(ik="ssik")
    tree = robot.tree
    add_target_marker(tree, "L_tgt")
    add_target_marker(tree, "R_tgt")
    spaces = {"left": robot.groups["left_arm"], "right": robot.groups["right_arm"]}

    rng = np.random.default_rng(7)
    rounds: list[tuple[dict[str, SE3], Configuration]] = []
    seed: dict[str, Configuration] = {
        "left": dict(robot.home),
        "right": dict(robot.home),
    }
    while len(rounds) < ROUNDS:
        targets: dict[str, SE3] = {}
        solutions: dict[str, Configuration] = {}
        ok = True
        for side, joints in _SIDES:
            space = spaces[side]
            lo = np.array([tree.joint(n).lower_limits[0] for n in joints])
            hi = np.array([tree.joint(n).upper_limits[0] for n in joints])
            sample = space.to_configuration(lo + (hi - lo) * rng.random(7))
            target = tree.forward_kinematics(
                robot.manipulators[side].ee_frame, {**dict(robot.home), **sample}
            )
            solution = robot.manipulators[side].ik.solve(target, seed[side])
            if solution is None:
                ok = False
                break
            targets[side], solutions[side] = target, solution
        if not ok:
            continue
        seed = solutions
        config: Configuration = {
            **dict(robot.home),
            **{j: solutions["left"][j] for j in LEFT},
            **{j: solutions["right"][j] for j in RIGHT},
        }
        rounds.append((targets, config))
    print(f"solved {len(rounds)} random target pairs")

    renderer = make_renderer(backend, tree)
    frames: list[np.ndarray] = []
    prev = rounds[-1][1]  # loop seamlessly: sweep into round 0 from the last pose
    for targets, config in rounds:
        tree.attach("L_tgt", tree.root, targets["left"])
        tree.attach("R_tgt", tree.root, targets["right"])
        seq = [
            lerp(spaces, robot.home, prev, config, (k + 1) / MOVE) for k in range(MOVE)
        ]
        seq += [config] * HOLD
        frames += [np.asarray(f)[..., :3] for f in renderer.render_frames(seq, _CAMERA)]
        prev = config

    mp4, gif = f"ssik_targets_{backend}.mp4", f"ssik_targets_{backend}.gif"
    imageio.mimsave(mp4, frames, fps=FPS)  # type: ignore[arg-type]
    write_gif(gif, frames)
    print(f"wrote {gif} / {mp4} ({len(frames)} frames)")


if __name__ == "__main__":
    main()

"""Register environments and expose them through make()."""

import os
import sys
from pathlib import Path

import gymnasium
from gymnasium.envs.registration import register


def register_all_environments() -> None:
    """Add all benchmark environments to the gymnasium registry."""
    # NOTE: ids must start with "prbench/" to be properly registered.

    # Detect headless mode (no DISPLAY) and set OSMesa if needed
    if not os.environ.get("DISPLAY"):
        if sys.platform == "darwin":
            os.environ["MUJOCO_GL"] = "glfw"
            os.environ["PYOPENGL_PLATFORM"] = "glfw"
        else:
            os.environ["MUJOCO_GL"] = "osmesa"
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    # ******* Geom2D Environments *******
    # Obstructions2D environment with different numbers of obstructions.
    num_obstructions = [0, 1, 2, 3, 4]
    for num_obstruction in num_obstructions:
        _register(
            id=f"prbench/Obstruction2D-o{num_obstruction}-v0",
            entry_point="prbench.envs.geom2d.obstruction2d:Obstruction2DEnv",
            kwargs={"num_obstructions": num_obstruction},
        )

    # ClutteredRetrieval2D environment with different numbers of obstructions.
    num_obstructions = [1, 10, 25]
    for num_obstruction in num_obstructions:
        _register(
            id=f"prbench/ClutteredRetrieval2D-o{num_obstruction}-v0",
            entry_point="prbench.envs.geom2d.clutteredretrieval2d:ClutteredRetrieval2DEnv",  # pylint: disable=line-too-long
            kwargs={"num_obstructions": num_obstruction},
        )

    # ClutteredStorage2D environment with different numbers of blocks.
    num_blocks = [1, 3, 7, 15]
    for num_block in num_blocks:
        _register(
            id=f"prbench/ClutteredStorage2D-b{num_block}-v0",
            entry_point="prbench.envs.geom2d.clutteredstorage2d:ClutteredStorage2DEnv",
            kwargs={"num_blocks": num_block},
        )

    # Motion2D environment with different numbers of passages.
    num_passages = [0, 1, 2, 3, 4, 5]
    for num_passage in num_passages:
        _register(
            id=f"prbench/Motion2D-p{num_passage}-v0",
            entry_point="prbench.envs.geom2d.motion2d:Motion2DEnv",
            kwargs={"num_passages": num_passage},
        )

    # StickButton2D environment with different numbers of buttons.
    num_buttons = [1, 2, 3, 5, 10]
    for num_button in num_buttons:
        _register(
            id=f"prbench/StickButton2D-b{num_button}-v0",
            entry_point="prbench.envs.geom2d.stickbutton2d:StickButton2DEnv",
            kwargs={"num_buttons": num_button},
        )

    # PushPullHook2D environment
    register(
        id="prbench/PushPullHook2D-v0",
        entry_point="prbench.envs.geom2d.pushpullhook2d:PushPullHook2DEnv",
    )

    # ******* Dynamic2D Environments *******

    # DynObstruction2D environment with different numbers of obstructions.
    num_obstructions = [0, 1, 2, 3]
    for num_obstruction in num_obstructions:
        _register(
            id=f"prbench/DynObstruction2D-o{num_obstruction}-v0",
            entry_point="prbench.envs.dynamic2d.dyn_obstruction2d:DynObstruction2DEnv",
            kwargs={"num_obstructions": num_obstruction},
        )

    # DynPushPullStick2D environment with different numbers of obstructions.
    num_obstructions = [0, 1, 5]
    for num_obstruction in num_obstructions:
        _register(
            id=f"prbench/DynPushPullHook2D-o{num_obstruction}-v0",
            entry_point="prbench.envs.dynamic2d.dyn_pushpullhook2d:DynPushPullHook2DEnv",
            kwargs={"num_obstructions": num_obstruction},
        )

    # DynPushT environment
    _register(
        id="prbench/DynPushT-t1-v0",
        entry_point="prbench.envs.dynamic2d.dyn_pushtee:DynPushTEnv",
        kwargs={"num_tee": 1},
    )

    # DynScoopPour environment with different numbers of small objects
    num_objects = [10, 20, 30, 50]
    for num_object in num_objects:
        num_circles = num_object // 2
        num_squares = num_object - num_circles
        _register(
            id=f"prbench/DynScoopPour-o{num_object}-v0",
            entry_point="prbench.envs.dynamic2d.dyn_scooppour:DynScoopPourEnv",
            kwargs={"num_small_circles": num_circles, "num_small_squares": num_squares},
        )

    # ******* Geom3D Environments *******

    # Motion3D environment.
    _register(
        id="prbench/Motion3D-v0",
        entry_point="prbench.envs.geom3d.motion3d:Motion3DEnv",
    )

    # Obstructions3D environment with different numbers of obstructions.
    num_obstructions = [0, 1, 2, 3, 4]
    for num_obstruction in num_obstructions:
        _register(
            id=f"prbench/Obstruction3D-o{num_obstruction}-v0",
            entry_point="prbench.envs.geom3d.obstruction3d:Obstruction3DEnv",
            kwargs={"num_obstructions": num_obstruction},
        )

    # ******* Dynamic3D Environments *******

    # Tasks with different scenes and object counts
    tasks_root = Path(__file__).parent / "envs" / "dynamic3d" / "tasks"

    for task_config in tasks_root.iterdir():
        config_name = task_config.stem
        robot = {"tidybot": "TidyBot3D", "rby1a": "RBY1A3D"}[config_name.split("-")[0]]
        scene_type = config_name.split("-")[1]
        num_task_objects = int(config_name.split("-")[2][1:])
        register(
            id=f"prbench/{robot}-{scene_type}-o{num_task_objects}-v0",
            entry_point=f"prbench.envs.dynamic3d.tidybot3d:{robot}Env",
            kwargs={
                "scene_type": scene_type,
                "num_objects": num_task_objects,
                "task_config_path": str(task_config),
            },
        )


def _register(id: str, *args, **kwargs) -> None:  # pylint: disable=redefined-builtin
    """Call register(), but only if the environment id is not already registered.

    This is to avoid noisy logging.warnings in register(). We are assuming that envs
    with the same id are equivalent, so this is safe.
    """
    if id not in gymnasium.registry:
        register(id, *args, **kwargs)


def make(*args, **kwargs) -> gymnasium.Env:
    """Create a registered environment from its name."""
    return gymnasium.make(*args, **kwargs)


def get_all_env_ids() -> set[str]:
    """Get all known benchmark environments."""
    return {env for env in gymnasium.registry if env.startswith("prbench/")}

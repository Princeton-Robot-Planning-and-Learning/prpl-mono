"""Scene loader utilities for loading different types of MuJoCo scene XMLs."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


class SceneLoader:
    """Factory class for loading different scene types."""

    @staticmethod
    def load_scene(scene_config: dict[str, Any], model_base_path: Path) -> str:
        """Load scene XML based on configuration.

        Args:
            scene_config: Scene configuration dictionary with keys:
                - type: "simple" or "mimiclabs"
                - xml_path: (optional) path to scene XML file
                - lab: (optional, for mimiclabs) lab number (2-8)
            model_base_path: Base path to models directory

        Returns:
            XML string of the loaded scene
        """
        scene_type = scene_config.get("type", "simple")

        if scene_type == "mimiclabs":
            return MimicLabsSceneLoader.load(scene_config)
        if scene_type == "simple":
            return SimpleSceneLoader.load(scene_config, model_base_path)
        raise ValueError(f"Unknown scene type: {scene_type}")


class SimpleSceneLoader:
    """Loader for simple ground scenes."""

    @staticmethod
    def load(scene_config: dict[str, Any], model_base_path: Path) -> str:
        """Load a simple ground scene.

        Args:
            scene_config: Scene configuration with optional "xml_path" key
            model_base_path: Base path to models directory

        Returns:
            XML string of the scene
        """
        # Use provided xml_path or default to ground_scene.xml
        xml_filename = scene_config.get("xml_path", "ground_scene.xml")

        # If it's not an absolute path, make it relative to model_base_path
        if not Path(xml_filename).is_absolute():
            xml_path = model_base_path / xml_filename
        else:
            xml_path = Path(xml_filename)

        with open(xml_path, "r", encoding="utf-8") as f:
            return f.read()


class MimicLabsSceneLoader:
    """Loader for MimicLabs realistic background scenes."""

    @staticmethod
    def load(scene_config: dict[str, Any]) -> str:
        """Load a MimicLabs scene with proper path resolution.

        Args:
            scene_config: Scene configuration with keys:
                - lab: lab number (2-8), or
                - xml_path: relative path to scene XML

        Returns:
            XML string with absolute paths for assets
        """
        # Resolve mimiclabs assets directory
        # Path(__file__) is at: prpl-mono/prbench/src/prbench/envs/dynamic3d/scene_loader.py
        # Need to go up to prpl-mono root
        # parent levels: dynamic3d -> envs -> prbench -> src -> prbench (pkg) -> prpl-mono
        repo_root = Path(__file__).parent.parent.parent.parent.parent.parent
        mimiclabs_assets = (
            repo_root
            / "third_party"
            / "mimiclabs"
            / "mimiclabs"
            / "mimiclabs"
            / "assets"
        )

        # Determine scene XML path
        if "xml_path" in scene_config:
            scene_xml_path = mimiclabs_assets / scene_config["xml_path"]
        elif "lab" in scene_config:
            lab_num = scene_config["lab"]
            scene_xml_path = (
                mimiclabs_assets / f"scenes/mimiclabs_scenes/lab{lab_num}.xml"
            )
        else:
            raise ValueError(
                "MimicLabs scene config must specify either 'lab' or 'xml_path'"
            )

        if not scene_xml_path.exists():
            raise FileNotFoundError(f"MimicLabs scene not found at: {scene_xml_path}")

        # Load scene XML
        with open(scene_xml_path, "r", encoding="utf-8") as f:
            xml_string = f.read()

        # Parse and update asset paths to be absolute
        tree = ET.fromstring(xml_string)

        # Set meshdir and texturedir to absolute paths
        # NOTE: mimiclabs XML files already include "meshes/" and "textures/" in file paths
        # so meshdir/texturedir should point to mimiclabs_scenes directory
        scenes_dir = mimiclabs_assets / "scenes" / "mimiclabs_scenes"
        meshdir = scenes_dir
        texturedir = scenes_dir

        # Update or create compiler section
        compiler = tree.find("compiler")
        if compiler is None:
            # Create compiler section and insert it at the beginning
            compiler = ET.Element("compiler")
            tree.insert(0, compiler)

        # Set absolute paths
        compiler.set("meshdir", str(meshdir.resolve()))
        if texturedir.exists():
            compiler.set("texturedir", str(texturedir.resolve()))

        return ET.tostring(tree, encoding="unicode")

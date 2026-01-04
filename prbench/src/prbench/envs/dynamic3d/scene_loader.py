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
        # Path(__file__) is at: prbench/src/prbench/envs/dynamic3d/scene_loader.py
        # MimicLabs scenes are stored in models/assets/mimiclabs_scenes/
        # (similar to how RoboCasa objects are in models/assets/robocasa_objects/)
        mimiclabs_scenes_dir = (
            Path(__file__).parent / "models" / "assets" / "mimiclabs_scenes"
        )

        # Check if mimiclabs_scenes directory exists
        if not mimiclabs_scenes_dir.exists():
            raise FileNotFoundError(
                f"MimicLabs scenes directory not found at: {mimiclabs_scenes_dir}\n"
                f"Please run: python scripts/download_mimiclabs_assets.py"
            )

        # Determine scene XML path
        if "xml_path" in scene_config:
            scene_xml_path = mimiclabs_scenes_dir / scene_config["xml_path"]
        elif "lab" in scene_config:
            lab_num = scene_config["lab"]
            scene_xml_path = mimiclabs_scenes_dir / f"lab{lab_num}.xml"
        else:
            raise ValueError(
                "MimicLabs scene config must specify either 'lab' or 'xml_path'"
            )

        if not scene_xml_path.exists():
            raise FileNotFoundError(
                f"MimicLabs scene not found at: {scene_xml_path}\n"
                f"Please run: python scripts/download_mimiclabs_assets.py"
            )

        # Load scene XML
        with open(scene_xml_path, "r", encoding="utf-8") as f:
            xml_string = f.read()

        # Parse and update asset paths to be absolute
        tree = ET.fromstring(xml_string)

        # Set meshdir and texturedir to absolute paths
        # NOTE: mimiclabs XML files already include "meshes/" and "textures/" in file paths
        # so meshdir/texturedir should point to mimiclabs_scenes directory
        meshdir = mimiclabs_scenes_dir
        texturedir = mimiclabs_scenes_dir

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

        # De-duplicate assets within the scene XML
        # Some mimiclabs scenes have duplicate material/texture/mesh definitions
        # NOTE: Deduplicate per asset type (tag), not globally, since MuJoCo allows
        # same name for different types (e.g., texture and material both named "X")
        asset_section = tree.find("asset")
        if asset_section is not None:
            seen_names: dict[str, set[str]] = {}  # tag -> set of names
            assets_to_remove = []
            for asset_elem in asset_section:
                asset_tag = asset_elem.tag
                asset_name = asset_elem.get("name")
                if asset_name:
                    if asset_tag not in seen_names:
                        seen_names[asset_tag] = set()
                    if asset_name in seen_names[asset_tag]:
                        # Mark for removal (duplicate within same type)
                        assets_to_remove.append(asset_elem)
                    else:
                        seen_names[asset_tag].add(asset_name)

            # Remove duplicate assets
            for asset_elem in assets_to_remove:
                asset_section.remove(asset_elem)

        return ET.tostring(tree, encoding="unicode")

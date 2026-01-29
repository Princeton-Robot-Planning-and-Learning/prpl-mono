"""Imported object classes for objects loaded from external model files."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from prbench.envs.dynamic3d.objects.robocasa_objects import RoboCasaObject
from prbench.envs.dynamic3d.objects.utils import euler_to_quat


class ReplicaObject(RoboCasaObject):
    """Replica object class with support for pos_z and euler options.

    This class extends RoboCasaObject with additional options for positioning and
    rotating the object, which is useful for replica objects like furniture.
    """

    def _create_xml_element(self) -> ET.Element:
        """Create the XML Element for this Replica object.

        Extends the base class to support pos_z and euler options for
        positioning and rotating the object.

        Returns:
            ET.Element representing the object body
        """
        # Find the worldbody in the model
        worldbody = self.model_root.find("worldbody")
        if worldbody is None:
            raise ValueError(
                f"No worldbody found in model.xml for {self.object_type_name}"
            )

        # Find the object body (should be nested inside worldbody)
        # Structure is typically: worldbody -> body -> body[name="object"]
        object_body = None
        for body in worldbody.iter("body"):
            if body.attrib.get("name") == "object":
                object_body = body
                break

        if object_body is None:
            raise ValueError(
                f"No body with name='object' found in model.xml for "
                f"{self.object_type_name}"
            )

        # Create a new body element with our object name
        new_body = ET.Element("body", name=self.name)

        # Add freejoint for position/orientation control
        ET.SubElement(new_body, "freejoint", name=self.joint_name)

        # Get optional pos_z and euler from options (replica-specific)
        pos_z = self.options.get("pos_z", 0.0)
        euler = self.options.get("euler", None)

        # Copy all geom elements from the object body and update mesh references
        for geom in object_body.findall("geom"):
            new_geom = ET.Element("geom", geom.attrib.copy())
            new_geom.text = geom.text
            new_geom.tail = geom.tail

            # Update mesh reference to use the prefixed name
            if "mesh" in geom.attrib:
                original_mesh = geom.attrib["mesh"]
                new_geom.attrib["mesh"] = f"{self.name}_{original_mesh}"

            # Update material reference to use the prefixed name
            if "material" in geom.attrib:
                original_material = geom.attrib["material"]
                new_geom.attrib["material"] = f"{self.name}_{original_material}"

            # Apply pos_z offset if specified
            if pos_z != 0.0:
                # Get existing pos or default to "0 0 0"
                existing_pos = new_geom.attrib.get("pos", "0 0 0")
                pos_parts = [float(p) for p in existing_pos.split()]
                pos_parts[2] += pos_z  # Add to z component
                new_geom.attrib["pos"] = f"{pos_parts[0]} {pos_parts[1]} {pos_parts[2]}"

            # Apply euler rotation if specified (in degrees: [roll, pitch, yaw])
            if euler is not None:
                quat_str = euler_to_quat(euler)
                new_geom.attrib["quat"] = quat_str

            new_body.append(new_geom)

        return new_body

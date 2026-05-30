"""Scene builder and renderer that runs inside Blender's Python interpreter.

Invoked as ``blender --background --python _blender_script.py -- <job.json>`` by
:class:`prpl_kinematics.visualization.blender_renderer.BlenderRenderer`. It reads
a job describing each node's shapes, the per-frame node world poses, and the
camera, builds the scene once, and renders one PNG per frame into the job's
output directory. ``bpy`` only exists inside Blender, so this module is never
imported by the package itself (it is excluded from type-checking and linting).
"""

import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector


def pose_matrix(pose):
    """A 4x4 rigid transform from ``[x, y, z, qx, qy, qz, qw]``."""
    px, py, pz, qx, qy, qz, qw = pose
    rotation = Quaternion((qw, qx, qy, qz)).to_matrix().to_4x4()
    return Matrix.Translation((px, py, pz)) @ rotation


def scale_matrix(scale):
    """A 4x4 diagonal scale matrix from a 3-vector."""
    return Matrix.Diagonal((scale[0], scale[1], scale[2], 1.0))


def principled_material(name, color, roughness=0.45, metallic=0.0):
    """A Principled-BSDF material with the given base color and finish."""
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return material


def clear_scene():
    """Remove the default objects and meshes from the startup scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)


def import_mesh(path):
    """Import a mesh file and return it as a single mesh object at identity.

    Joins the imported objects and bakes the importer's own world transform (e.g.
    glTF's Y-up to Z-up conversion) into the vertices, so the caller fully controls
    placement via ``matrix_world``.
    """
    before = {obj.name for obj in bpy.data.objects}
    extension = os.path.splitext(path)[1].lower()
    if extension == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif extension == ".stl":
        bpy.ops.wm.stl_import(filepath=path)
    elif extension == ".dae":
        bpy.ops.wm.collada_import(filepath=path)
    elif extension in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif extension == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    else:
        raise ValueError(f"unsupported mesh format: {extension}")

    new_names = [obj.name for obj in bpy.data.objects if obj.name not in before]
    mesh_names = [n for n in new_names if bpy.data.objects[n].type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for name in new_names:
        bpy.data.objects[name].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects[mesh_names[0]]
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.select_all(action="DESELECT")
    for name in mesh_names:
        bpy.data.objects[name].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects[mesh_names[0]]
    if len(mesh_names) > 1:
        bpy.ops.object.join()
    result_name = mesh_names[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for name in new_names:
        if name != result_name and name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    return bpy.data.objects[result_name]


def make_primitive(spec):
    """Create a unit Blender primitive for a box, cylinder, or sphere spec."""
    kind = spec["kind"]
    if kind == "box":
        bpy.ops.mesh.primitive_cube_add(size=1.0)
    elif kind == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=1.0)
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.shade_smooth()
    color = spec.get("color", (0.55, 0.58, 0.62, 1.0))
    obj.data.materials.append(principled_material("primitive", color, roughness=0.5))
    return obj


def primitive_scale(spec):
    """The unit-primitive scale that yields the spec's real dimensions."""
    kind = spec["kind"]
    if kind == "box":
        return spec["size"]
    if kind == "cylinder":
        return [spec["radius"], spec["radius"], spec["length"]]
    return [spec["radius"], spec["radius"], spec["radius"]]


def build_objects(shapes):
    """Create one Blender object per shape; return objects and geometry scales."""
    objects = {}
    geometry_scale = {}
    for spec in shapes:
        if spec["kind"] == "mesh":
            obj = import_mesh(spec["file"])
            bpy.ops.object.shade_smooth()
            if "color" in spec:  # an explicit color overrides any embedded material
                obj.data.materials.clear()
                obj.data.materials.append(
                    principled_material("robot", spec["color"], roughness=0.4)
                )
            elif not obj.data.materials:
                obj.data.materials.append(
                    principled_material("robot", (0.9, 0.9, 0.92, 1.0), roughness=0.4)
                )
            geometry_scale[spec["id"]] = spec["scale"]
        else:
            obj = make_primitive(spec)
            geometry_scale[spec["id"]] = primitive_scale(spec)
        objects[spec["id"]] = obj
    return objects, geometry_scale


def setup_world(color):
    """Show ``color`` as the camera background while keeping ambient light low.

    The camera sees the full-strength background color, but lighting rays see a dim
    version, so the visible background stays a soft color without a bright environment
    flooding (and desaturating) the scene.
    """
    rgba = (color[0], color[1], color[2], 1.0)
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputWorld")
    mix = tree.nodes.new("ShaderNodeMixShader")
    light_path = tree.nodes.new("ShaderNodeLightPath")
    visible = tree.nodes.new("ShaderNodeBackground")
    visible.inputs[0].default_value = rgba
    visible.inputs[1].default_value = 1.0
    ambient = tree.nodes.new("ShaderNodeBackground")
    ambient.inputs[0].default_value = rgba
    ambient.inputs[1].default_value = 0.25  # dim fill so colors stay saturated
    tree.links.new(light_path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    tree.links.new(ambient.outputs["Background"], mix.inputs[1])
    tree.links.new(visible.outputs["Background"], mix.inputs[2])
    tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])


def add_ground_plane():
    """Add a large neutral floor so objects cast grounding shadows."""
    bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0, 0, 0))
    plane = bpy.context.view_layer.objects.active
    plane.data.materials.append(
        principled_material("ground", (0.8, 0.8, 0.82, 1.0), roughness=0.9)
    )


def setup_lighting():
    """A soft area key light plus a fill sun."""
    key = bpy.data.lights.new("Key", type="AREA")
    key.energy = 90.0
    key.size = 3.0
    key_obj = bpy.data.objects.new("Key", key)
    bpy.context.collection.objects.link(key_obj)
    key_obj.location = (1.5, -1.5, 2.5)
    key_obj.rotation_euler = (math.radians(35), 0.0, math.radians(45))
    sun = bpy.data.lights.new("Sun", type="SUN")
    sun.energy = 1.2
    sun_obj = bpy.data.objects.new("Sun", sun)
    bpy.context.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (math.radians(50), math.radians(15), math.radians(-30))


def setup_camera(camera):
    """Place a camera orbiting ``target`` by yaw/pitch/distance, looking at it.

    Mirrors PyBullet's yaw-pitch-roll convention (Z up) so views are comparable.
    """
    target = Vector(camera["target"])
    yaw = math.radians(camera["yaw"])
    pitch = math.radians(camera["pitch"])
    distance = camera["distance"]
    eye = target + Vector(
        (
            distance * math.cos(pitch) * math.sin(yaw),
            -distance * math.cos(pitch) * math.cos(yaw),
            -distance * math.sin(pitch),
        )
    )
    camera_data = bpy.data.cameras.new("Camera")
    camera_data.sensor_fit = "VERTICAL"
    camera_data.angle_y = math.radians(camera["fov"])
    camera_obj = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera_obj)
    camera_obj.location = eye
    camera_obj.rotation_euler = (target - eye).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera_obj


def setup_render(job):
    """Configure the Cycles engine, sampling, and output resolution."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = job["samples"]
    scene.cycles.device = "CPU"
    scene.cycles.use_denoising = True
    # Faithful color; the default AgX desaturates saturated materials. With the
    # low-ambient world and moderate lamps below, exposure stays in range.
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_x = job["camera"]["width"]
    scene.render.resolution_y = job["camera"]["height"]
    scene.render.image_settings.file_format = "PNG"


def main():
    """Read the job, build the scene, and render one PNG per frame."""
    argv = sys.argv[sys.argv.index("--") + 1 :]
    with open(argv[0], encoding="utf-8") as handle:
        job = json.load(handle)

    clear_scene()
    setup_world(job["background_color"])
    setup_render(job)
    if job["ground_plane"]:
        add_ground_plane()
    setup_lighting()
    setup_camera(job["camera"])
    objects, geometry_scale = build_objects(job["shapes"])

    for index, frame in enumerate(job["frames"]):
        for spec in job["shapes"]:
            obj = objects[spec["id"]]
            obj.matrix_world = (
                pose_matrix(frame["poses"][spec["node"]])
                @ pose_matrix(spec["origin"])
                @ scale_matrix(geometry_scale[spec["id"]])
            )
        bpy.context.scene.render.filepath = os.path.join(
            job["output_dir"], "frame_%04d.png" % index
        )
        bpy.ops.render.render(write_still=True)


main()

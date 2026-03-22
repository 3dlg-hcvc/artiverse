# isort: off
import blenderproc as bproc
import argparse
import json
import math
import os
import subprocess
import tempfile

import bpy
import imageio
import numpy as np
import trimesh
from mathutils import Matrix, Vector

import sys
sys.path.append("./")
import external.pygltftoolkit as pygltk


def _init_scene(res_x=768, res_y=768, ambient_strength=0.08, hdri_path=None):
    bproc.init()
    bproc.renderer.set_output_format(file_format="PNG", enable_transparency=True)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)

    scene = bpy.context.scene
    render = scene.render
    # Prefer EEVEE for speed and compatibility
    render.engine = 'BLENDER_EEVEE_NEXT'
    render.resolution_x = int(res_x)
    render.resolution_y = int(res_y)
    render.resolution_percentage = 100
    render.image_settings.file_format = 'PNG'
    render.image_settings.color_mode = 'RGBA'
    render.film_transparent = True

    # Basic EEVEE tuning for quality/speed
    eevee = scene.eevee
    eevee.taa_render_samples = 64
    eevee.taa_samples = 16
    eevee.use_gtao = True
    eevee.gtao_distance = 0.2
    eevee.gtao_factor = 1.0
    eevee.gtao_quality = 0.25
    eevee.use_ssr = True
    eevee.use_ssr_refraction = True
    eevee.ssr_quality = 0.5
    eevee.ssr_max_roughness = 0.5
    eevee.shadow_cube_size = '512'
    eevee.shadow_cascade_size = '1024'
    eevee.use_soft_shadows = True
    eevee.use_bloom = True
    eevee.bloom_threshold = 0.8
    eevee.bloom_intensity = 0.05

    # World background
    if bpy.context.scene.world:
        world = bpy.context.scene.world
    else:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    world_nodes = world.node_tree.nodes
    world_links = world.node_tree.links

    if hdri_path and os.path.exists(hdri_path):
        # HDRI environment — wipe default nodes and build a proper env texture graph
        world_nodes.clear()
        env_node = world_nodes.new('ShaderNodeTexEnvironment')
        env_node.image = bpy.data.images.load(hdri_path)
        bg_node = world_nodes.new('ShaderNodeBackground')
        bg_node.inputs['Strength'].default_value = float(ambient_strength)
        output_node = world_nodes.new('ShaderNodeOutputWorld')
        world_links.new(env_node.outputs['Color'], bg_node.inputs['Color'])
        world_links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
    else:
        if "Background" in world_nodes:
            world_nodes["Background"].inputs["Strength"].default_value = float(ambient_strength)
            world_nodes["Background"].inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)


def _add_lighting(light_intensity_scale=0.3, light_size_scale=1.8, center=None):
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])

    # Clear existing lights
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()

    # Symmetric 4-point setup aimed at model center: top key + left/right fills + bottom bounce
    light_data = [
        # Key: top-front center
        {"energy": 4000 * light_intensity_scale, "location": (cx + 0.0, cy - 1.5, cz + 4.0), "size": (4.0 * light_size_scale, 4.0 * light_size_scale)},
        # Left fill (symmetric with right)
        {"energy": 2000 * light_intensity_scale, "location": (cx - 3.5, cy + 0.0, cz + 1.5), "size": (3.0 * light_size_scale, 3.0 * light_size_scale)},
        # Right fill (symmetric with left)
        {"energy": 2000 * light_intensity_scale, "location": (cx + 3.5, cy + 0.0, cz + 1.5), "size": (3.0 * light_size_scale, 3.0 * light_size_scale)},
        # Bounce: subtle fill from below
        {"energy": 800 * light_intensity_scale, "location": (cx + 0.0, cy + 0.0, cz - 3.0), "size": (5.0 * light_size_scale, 5.0 * light_size_scale)},
    ]
    for i, d in enumerate(light_data):
        light = bpy.data.lights.new(name=f"Light_{i}", type='AREA')
        light.energy = d["energy"]
        light.shape = 'RECTANGLE'
        light.size = d["size"][0]
        light.size_y = d["size"][1]
        obj = bpy.data.objects.new(f"LightObj_{i}", light)
        obj.location = Vector(d["location"])
        bpy.context.scene.collection.objects.link(obj)
        # Aim light at model center so it actually illuminates the object
        direction = Vector((cx, cy, cz)) - Vector(d["location"])
        rot_quat = direction.to_track_quat('-Z', 'Y')
        obj.rotation_euler = rot_quat.to_euler()


def _create_materials():
    # Highlight material (opaque magenta)
    highlight_mat_name = "highlight_magenta_mat"
    if highlight_mat_name not in bpy.data.materials:
        hl_mat = bpy.data.materials.new(highlight_mat_name)
        hl_mat.use_nodes = True
        hl_mat.blend_method = "OPAQUE"
        pbsdf = next(n for n in hl_mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        pbsdf.inputs["Base Color"].default_value = (1.0, 0.0, 1.0, 1.0)
        pbsdf.inputs["Alpha"].default_value = 1.0
        pbsdf.inputs["Roughness"].default_value = 0.2
    else:
        hl_mat = bpy.data.materials[highlight_mat_name]

    # Gray semi-transparent material for context
    gray_mat_name = "gray_semitransparent_mat"
    if gray_mat_name not in bpy.data.materials:
        gray_mat = bpy.data.materials.new(gray_mat_name)
        gray_mat.use_nodes = True
        gray_mat.blend_method = "BLEND"
        pbsdf_g = next(n for n in gray_mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        pbsdf_g.inputs["Base Color"].default_value = (0.6, 0.6, 0.6, 1.0)
        pbsdf_g.inputs["Alpha"].default_value = 0.6
        pbsdf_g.inputs["Roughness"].default_value = 0.5
    else:
        gray_mat = bpy.data.materials[gray_mat_name]

    # Blue-ish material to indicate parent of the highlighted part
    parent_mat_name = "parent_blue_mat"
    if parent_mat_name not in bpy.data.materials:
        parent_mat = bpy.data.materials.new(parent_mat_name)
        parent_mat.use_nodes = True
        parent_mat.blend_method = "BLEND"
        pbsdf_p = next(n for n in parent_mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        pbsdf_p.inputs["Base Color"].default_value = (0.2, 0.45, 1.0, 1.0)
        pbsdf_p.inputs["Alpha"].default_value = 0.85
        pbsdf_p.inputs["Roughness"].default_value = 0.25
    else:
        parent_mat = bpy.data.materials[parent_mat_name]

    # Enable backface culling for all
    for m in [hl_mat, gray_mat, parent_mat]:
        m.use_backface_culling = True

    return hl_mat, gray_mat, parent_mat


def _compute_bbox_from_objects(objs):
    all_world = []
    for obj in objs:
        bobj = obj if isinstance(obj, bpy.types.Object) else obj.blender_obj
        for corner in bobj.bound_box:
            world_coord = bobj.matrix_world @ Vector(corner)
            all_world.append([world_coord.x, world_coord.y, world_coord.z])
    if not all_world:
        return None
    arr = np.array(all_world)
    return np.min(arr, axis=0), np.max(arr, axis=0)


def _bbox_center_from_objects(objs):
    bb = _compute_bbox_from_objects(objs)
    if bb is None:
        return np.array([0.0, 0.0, 0.0])
    return (np.array(bb[0]) + np.array(bb[1])) / 2.0


def _load_segmented_parts(object_dir, highlight_pid, arts_by_pid, force_highlight=False):
    model_id = object_dir.rstrip("/").split("/")[-1]
    glb_path = f"{object_dir}/{model_id}.segmented.glb"

    gltf = pygltk.load(glb_path, annotated=True)

    tmp_dir = f"./_temp/{model_id}"
    os.makedirs(tmp_dir, exist_ok=True)

    highlight_mat, gray_mat, parent_mat = _create_materials()

    # Parent empties per pid for easy articulation
    part_empty_by_pid = {}
    imported_by_pid = {}
    parent_by_pid = {}

    # Determine which parts should be rendered:
    # 1. Parts that have articulations (and are not fixed/free only)
    # 2. The highlighted part (if specified) - for context even if fixed/free
    # 3. Parts that are parents/children of articulated parts (for connectivity)
    # 4. All parts for general context (but only articulated ones get highlighted)
    
    articulated_pids = set()
    for pid, arts in arts_by_pid.items():
        # Check if this part has any non-fixed articulations
        has_non_fixed = any(art.get("type", "fixed") not in ["fixed", "free"] for art in arts)
        if has_non_fixed:
            articulated_pids.add(pid)
    
    # Start with all unique pids for full context
    unique_pids = list(np.unique(gltf.segmentation_map))
    render_pids = set(int(pid) for pid in unique_pids)
    
    # Add parent/child parts for connectivity (we'll determine this from flattened_graph)
    connectivity_pids = set()
    try:
        gltf2 = getattr(gltf, "gltf2", None)
        extras = getattr(gltf2, "extras", None) if gltf2 is not None else None
        flattened = extras.get("flattened_graph") if isinstance(extras, dict) else None
        if isinstance(flattened, dict):
            for sid, node in flattened.items():
                try:
                    cid = int(sid)
                    parent_id = node.get("parent", -1)
                    # If this part or its parent is articulated, include both
                    if cid in articulated_pids or (parent_id != -1 and parent_id in articulated_pids):
                        connectivity_pids.add(cid)
                        if parent_id != -1:
                            connectivity_pids.add(parent_id)
                except Exception:
                    continue
    except Exception:
        pass
    
    render_pids.update(connectivity_pids)

    # Export and import only the parts we need to render
    unique_pids = list(np.unique(gltf.segmentation_map))
    for pid in unique_pids:
        if int(pid) not in render_pids:
            continue  # Skip parts that don't need to be rendered
            
        mesh = trimesh.Trimesh(vertices=gltf.vertices, faces=gltf.faces[gltf.segmentation_map == pid])
        obj_path = f"{tmp_dir}/{pid}.obj"
        mesh.export(obj_path)
        imported_objs = bproc.loader.load_obj(obj_path)

        # Create parent empty
        empty = bpy.data.objects.new(f"Part_{pid}", None)
        bpy.context.scene.collection.objects.link(empty)

        # Material assignment and parenting
        is_highlight = False
        if highlight_pid is not None and int(pid) == int(highlight_pid):
            # Only highlight if this part has non-fixed articulations
            part_arts = arts_by_pid.get(int(pid), [])
            has_non_fixed = any(art.get("type", "fixed") not in ["fixed", "free"] for art in part_arts)
            is_highlight = has_non_fixed or bool(force_highlight)
            
        for bp in imported_objs:
            bobj = bp.blender_obj
            mesh_data = bobj.data
            mesh_data.materials.clear()
            if is_highlight:
                mesh_data.materials.append(highlight_mat)
                bobj.active_material = highlight_mat
            else:
                mesh_data.materials.append(gray_mat)
                bobj.active_material = gray_mat

            # Parent while keeping world transform
            bobj.parent = empty
            bobj.matrix_parent_inverse = empty.matrix_world.inverted()

        part_empty_by_pid[int(pid)] = empty
        imported_by_pid[int(pid)] = [bp.blender_obj for bp in imported_objs]

    # Apply parent-child relationships from connectivity graph (if available)
    try:
        gltf2 = getattr(gltf, "gltf2", None)
        extras = getattr(gltf2, "extras", None) if gltf2 is not None else None
        flattened = extras.get("flattened_graph") if isinstance(extras, dict) else None
        if isinstance(flattened, dict):
            for sid, node in flattened.items():
                try:
                    cid = int(sid)
                except Exception:
                    continue
                parent_id = node.get("parent", -1)
                if parent_id is None or int(parent_id) == -1:
                    continue
                if cid in part_empty_by_pid and int(parent_id) in part_empty_by_pid:
                    child_empty = part_empty_by_pid[cid]
                    parent_empty = part_empty_by_pid[int(parent_id)]
                    child_empty.parent = parent_empty
                    child_empty.matrix_parent_inverse = parent_empty.matrix_world.inverted()
                    parent_by_pid[cid] = int(parent_id)
    except Exception as e:
        # Optional warning only in debug builds
        if getattr(bpy.app, "debug", False):
            print(f"[WARN] Connectivity parenting failed: {e}")

    # Highlight immediate parent of highlighted pid in blue (fixed or articulated).
    if highlight_pid is not None:
        parent_pid = parent_by_pid.get(int(highlight_pid))
        if parent_pid is not None and int(parent_pid) in imported_by_pid and int(parent_pid) != int(highlight_pid):
            for bobj in imported_by_pid[int(parent_pid)]:
                mesh_data = bobj.data
                mesh_data.materials.clear()
                mesh_data.materials.append(parent_mat)
                bobj.active_material = parent_mat

    # Compute bboxes
    full_bbox = (np.min(gltf.vertices, axis=0), np.max(gltf.vertices, axis=0))
    highlight_bbox = None
    if highlight_pid is not None and int(highlight_pid) in imported_by_pid:
        hb = _compute_bbox_from_objects(imported_by_pid[int(highlight_pid)])
        if hb is not None:
            highlight_bbox = hb

    return part_empty_by_pid, imported_by_pid, parent_by_pid, highlight_bbox, full_bbox


def _extract_pid_from_blender_obj(blender_obj):
    for key in ["scenePid", "pid", "id"]:
        if key in blender_obj:
            try:
                return int(blender_obj[key])
            except Exception:
                pass
    return None


def _collect_descendant_meshes(root_obj):
    meshes = []
    stack = [root_obj]
    seen = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur.type == "MESH":
            meshes.append(cur)
        for ch in cur.children:
            stack.append(ch)
    return meshes


def _object_depth(obj):
    d = 0
    cur = obj
    while cur.parent is not None:
        d += 1
        cur = cur.parent
    return d


def _load_textured_segmented_parts(object_dir, highlight_pid, arts_by_pid):
    del arts_by_pid  # not used in textured mode
    model_id = object_dir.rstrip("/").split("/")[-1]
    glb_path = f"{object_dir}/{model_id}.segmented.glb"

    gltf = pygltk.load(glb_path, annotated=True)

    objects_before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=glb_path, merge_vertices=True)
    objects_after = set(bpy.context.scene.objects)
    imported_objects = list(objects_after - objects_before)

    part_empty_by_pid = {}
    imported_by_pid = {}
    parent_by_pid = {}

    # Use imported GLB hierarchy nodes as articulation drivers; this preserves child geometry motion.
    pid_candidates = {}
    for obj in imported_objects:
        pid = _extract_pid_from_blender_obj(obj)
        if pid is None:
            continue
        pid_candidates.setdefault(pid, []).append(obj)

    for pid, objs in pid_candidates.items():
        # Choose the highest object in hierarchy for this pid.
        driver = sorted(objs, key=_object_depth)[0]
        part_empty_by_pid[pid] = driver
        imported_by_pid[pid] = _collect_descendant_meshes(driver)

    # Use flattened_graph for pid parenting to match highlight-mode kinematic behavior.
    # Unlike the previous textured path, we enforce actual Blender parent links so
    # child parts (e.g., handles) inherit drawer transforms during articulation.
    try:
        gltf2 = getattr(gltf, "gltf2", None)
        extras = getattr(gltf2, "extras", None) if gltf2 is not None else None
        flattened = extras.get("flattened_graph") if isinstance(extras, dict) else None
        if isinstance(flattened, dict):
            for sid, node in flattened.items():
                try:
                    cid = int(sid)
                except Exception:
                    continue
                parent_id = node.get("parent", -1)
                if parent_id is None or int(parent_id) == -1:
                    continue
                if cid in part_empty_by_pid and int(parent_id) in part_empty_by_pid:
                    child_driver = part_empty_by_pid[cid]
                    parent_driver = part_empty_by_pid[int(parent_id)]

                    # Re-parent with world transform preserved.
                    child_driver.parent = parent_driver
                    child_driver.matrix_parent_inverse = parent_driver.matrix_world.inverted()

                    parent_by_pid[cid] = int(parent_id)
    except Exception as e:
        if getattr(bpy.app, "debug", False):
            print(f"[WARN] Textured flattened_graph parenting failed: {e}")

    full_bbox = (np.min(gltf.vertices, axis=0), np.max(gltf.vertices, axis=0))
    highlight_bbox = None
    if highlight_pid is not None and int(highlight_pid) in imported_by_pid:
        hb = _compute_bbox_from_objects(imported_by_pid[int(highlight_pid)])
        if hb is not None:
            highlight_bbox = hb
    return part_empty_by_pid, imported_by_pid, parent_by_pid, highlight_bbox, full_bbox


MTYPE_LOOKUP = {
    "revolute": "revolute",
    "rotation": "revolute",
    "prismatic": "prismatic",
    "translation": "prismatic",
    "screw": "screw",
    "cylindrical": "cylindrical",
    "universal": "universal",
    "continuous": "continuous",
    "fixed": "fixed",
    "free": "fixed",
}


def _parse_articulations(art_json_path):
    with open(art_json_path, 'r') as f:
        data = json.load(f)
    raw_arts = data.get("articulations", [])

    arts_by_pid = {}
    for art in raw_arts:
        pid = int(art.get("pid"))
        atype = str(art.get("type", "")).lower()
        atype = MTYPE_LOOKUP.get(atype, atype)
        # Support both generic keys ("axis"/"origin") and universal-specific aliases ("axis1"/"origin1").
        axis = np.array(art.get("axis", art.get("axis1", [0, 0, 1])), dtype=np.float64)
        origin = np.array(art.get("origin", art.get("origin1", [0, 0, 0])), dtype=np.float64)
        rmin = art.get("rangeMin", art.get("rotRangeMin", art.get("rotation_rangeMin", None)))
        rmax = art.get("rangeMax", art.get("rotRangeMax", art.get("rotation_rangeMax", None)))

        entry = {
            "type": atype,
            "axis": axis,
            "origin": origin,
            "rangeMin": rmin,
            "rangeMax": rmax,
        }
        # Optional cylindrical/prismatic translation ranges (commonly used by export tools)
        if art.get("prismatic_rangeMin") is not None:
            entry["prismatic_rangeMin"] = art.get("prismatic_rangeMin")
        if art.get("prismatic_rangeMax") is not None:
            entry["prismatic_rangeMax"] = art.get("prismatic_rangeMax")
        # Optional universal fields (axis2/origin2/range2) - support if present
        if art.get("axis2") is not None:
            entry["axis2"] = np.array(art.get("axis2"), dtype=np.float64)
        if art.get("origin2") is not None:
            entry["origin2"] = np.array(art.get("origin2"), dtype=np.float64)
        if art.get("rot2RangeMin") is not None:
            entry["rot2RangeMin"] = art.get("rot2RangeMin")
        if art.get("rot2RangeMax") is not None:
            entry["rot2RangeMax"] = art.get("rot2RangeMax")

        arts_by_pid.setdefault(pid, []).append(entry)

    # Heuristic: if a pid has both revolute and prismatic entries and no explicit composite type,
    # keep both entries so we can render as cylindrical by default.
    return arts_by_pid


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def _rotation_matrix_axis_angle(axis, theta):
    # Rodrigues' formula
    axis = _normalize(axis)
    kx, ky, kz = axis
    c = math.cos(theta)
    s = math.sin(theta)
    v = 1.0 - c
    R = Matrix(((
        c + kx*kx*v,      kx*ky*v - kz*s, kx*kz*v + ky*s, 0.0,
    ), (
        ky*kx*v + kz*s,   c + ky*ky*v,    ky*kz*v - kx*s, 0.0,
    ), (
        kz*kx*v - ky*s,   kz*ky*v + kx*s, c + kz*kz*v,    0.0,
    ), (
        0.0,              0.0,            0.0,            1.0,
    )))
    return R


def _translation_matrix(vec3):
    T = Matrix.Identity(4)
    T[0][3] = float(vec3[0])
    T[1][3] = float(vec3[1])
    T[2][3] = float(vec3[2])
    return T


def _compose_about_origin(M0, axis, origin, angle):
    # Compute T(origin) * R(axis, angle) * T(-origin) * M0
    T1 = _translation_matrix(-np.asarray(origin))
    R = _rotation_matrix_axis_angle(axis, angle)
    T2 = _translation_matrix(np.asarray(origin))
    return T2 @ (R @ (T1 @ M0))


def _apply_world_matrix(obj, M):
    obj.matrix_world = M
    # Correct Blender API usage: Matrix.decompose() returns (loc, rot, scale)
    loc, rot, sca = M.decompose()
    obj.rotation_mode = 'QUATERNION'
    obj.location = loc
    obj.rotation_quaternion = rot
    obj.scale = sca


def _compute_camera_static(full_bbox, fov_degrees=60.0, azimuth_offset_deg=0.0):
    center = (np.array(full_bbox[0]) + np.array(full_bbox[1])) / 2.0
    radius = 0.5 * np.linalg.norm(np.array(full_bbox[1]) - np.array(full_bbox[0]))
    fov_rad = math.radians(fov_degrees)
    distance = radius / math.sin(fov_rad/2.0)
    distance *= 1.2

    cam_data = bpy.data.cameras.new(name="Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    cam_data.lens_unit = 'FOV'
    cam_data.angle = fov_rad

    base_offset = np.array([0.0, -distance, 0.2 * radius], dtype=np.float64)
    theta = math.radians(float(azimuth_offset_deg))
    rot_z = np.array([
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta), math.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    cam_offset = rot_z @ base_offset
    cam_obj.location = Vector((center[0] + cam_offset[0], center[1] + cam_offset[1], center[2] + cam_offset[2]))
    # Look at center
    direction = Vector(center) - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()
    return cam_obj, center, radius, distance


def _compute_turntable_positions(center, radius_or_dist, frames, elevation_ratio=0.2):
    # radius_or_dist used as orbit radius
    elev = elevation_ratio * radius_or_dist
    theta = np.linspace(0, 2*np.pi, frames, endpoint=False)
    positions = []
    for t in theta:
        x = radius_or_dist * math.cos(t)
        y = radius_or_dist * math.sin(t)
        z = elev
        positions.append(center + np.array([x, y, z]))
    return positions


def _compute_turntable_positions_with_offset(center, radius_or_dist, frames, elevation_ratio=0.2, azimuth_offset_deg=0.0):
    elev = elevation_ratio * radius_or_dist
    theta = np.linspace(0, 2*np.pi, frames, endpoint=False)
    theta += math.radians(float(azimuth_offset_deg))
    positions = []
    for t in theta:
        x = radius_or_dist * math.cos(t)
        y = radius_or_dist * math.sin(t)
        z = elev
        positions.append(center + np.array([x, y, z]))
    return positions


def _set_camera_lookat(cam_obj, target):
    direction = Vector(target) - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()


def _compute_bbox_extent_along_axis(full_bbox, axis):
    a = _normalize(np.asarray(axis))
    pmin = np.asarray(full_bbox[0])
    pmax = np.asarray(full_bbox[1])
    extent = abs(np.dot(pmax - pmin, a))
    return float(extent)


def _value_at_frame_pingpong(range_min, range_max, frame, total_frames):
    if range_min is None or range_max is None:
        range_min, range_max = 0.0, 1.0
    if total_frames <= 1:
        return float(range_min)
    # Smooth ping-pong with endpoints included.
    t = (frame % total_frames) / float(max(1, total_frames - 1))
    p = 0.5 * (1.0 - math.cos(2.0 * math.pi * t))
    return range_min + (range_max - range_min) * p


def _prepare_baselines(part_empty_by_pid):
    baselines = {}
    for pid, empty in part_empty_by_pid.items():
        baselines[pid] = empty.matrix_world.copy()
    return baselines


def _collect_composite_types(arts_by_pid):
    # Identify per-pid type composition
    comp = {}
    for pid, arts in arts_by_pid.items():
        types = [a["type"] for a in arts]
        if "screw" in types:
            comp[pid] = "screw"
        elif "cylindrical" in types:
            comp[pid] = "cylindrical"
        elif "universal" in types:
            comp[pid] = "universal"
        else:
            # derive: both revolute and prismatic present -> cylindrical
            if (any(t == 'revolute' for t in types) and any(t == 'prismatic' for t in types)):
                comp[pid] = "cylindrical"
            elif any(t == 'revolute' for t in types):
                comp[pid] = "revolute"
            elif any(t == 'prismatic' for t in types):
                comp[pid] = "prismatic"
            else:
                comp[pid] = types[0] if types else 'fixed'
    return comp


def _fmt_vec(v):
    try:
        arr = np.asarray(v).astype(float).ravel()
        return f"[{arr[0]:.6f}, {arr[1]:.6f}, {arr[2]:.6f}]"
    except Exception:
        return str(v)


def _matrix_scale_factor(M):
    try:
        _, _, sca = M.decompose()
        vals = [abs(float(sca[0])), abs(float(sca[1])), abs(float(sca[2]))]
        valid = [v for v in vals if v > 1e-8]
        return float(np.mean(valid)) if valid else 1.0
    except Exception:
        return 1.0


def _get_range(a, min_keys, max_keys, default_min, default_max):
    rmin = None
    rmax = None
    for k in min_keys:
        if a.get(k) is not None:
            rmin = a.get(k)
            break
    for k in max_keys:
        if a.get(k) is not None:
            rmax = a.get(k)
            break
    if rmin is None:
        rmin = default_min
    if rmax is None:
        rmax = default_max
    return float(rmin), float(rmax)


def _compute_articulated_matrix(baseline_M, arts, comp_type, frame, n_frames, full_bbox, prismatic_origin_fallback=None):
    M = baseline_M.copy()
    scale_mult = _matrix_scale_factor(M)

    def _first_of(t):
        for a in arts:
            if a["type"] == t:
                return a
        return None

    if comp_type == 'fixed' or not arts:
        return M

    if comp_type == 'revolute' or comp_type == 'continuous':
        a = _first_of('revolute') or _first_of('rotation') or arts[0]
        dmax = 2.0 * math.pi if comp_type == 'continuous' else (math.pi / 2.0)
        rmin, rmax = _get_range(a, ["rangeMin"], ["rangeMax"], 0.0, dmax)
        if comp_type == 'continuous':
            # Full cycle, monotonic rotation for cleaner spinner-like joints.
            angle = rmin + (rmax - rmin) * ((frame % n_frames) / float(max(1, n_frames)))
        else:
            angle = _value_at_frame_pingpong(rmin, rmax, frame, n_frames)
        axis = a.get("axis", np.array([0, 0, 1], dtype=np.float64))
        origin = a.get("origin", np.array([0, 0, 0], dtype=np.float64))
        return _compose_about_origin(M, axis, origin, angle)

    if comp_type == 'prismatic':
        a = _first_of('prismatic') or _first_of('translation') or arts[0]
        dmax = 0.25 * _compute_bbox_extent_along_axis(full_bbox, a.get("axis", [0, 0, 1]))
        rmin, rmax = _get_range(
            a,
            ["prismatic_rangeMin", "rangeMin"],
            ["prismatic_rangeMax", "rangeMax"],
            0.0,
            dmax,
        )
        value = _value_at_frame_pingpong(rmin, rmax, frame, n_frames)
        axis = _normalize(a.get("axis", np.array([0, 0, 1], dtype=np.float64)))
        T = _translation_matrix(axis * (value * scale_mult))
        return T @ M

    if comp_type == 'screw':
        aR = _first_of('revolute') or arts[0]
        aT = _first_of('prismatic') or aR
        rmin, rmax = _get_range(aR, ["rangeMin"], ["rangeMax"], 0.0, math.pi / 2.0)
        angle = _value_at_frame_pingpong(rmin, rmax, frame, n_frames)
        p = (angle - rmin) / (rmax - rmin + 1e-12)
        tmin, tmax = _get_range(
            aT,
            ["prismatic_rangeMin", "rangeMin"],
            ["prismatic_rangeMax", "rangeMax"],
            0.0,
            0.25 * _compute_bbox_extent_along_axis(full_bbox, aT.get("axis", [0, 0, 1])),
        )
        trans_value = tmin + (tmax - tmin) * p
        axis = aR.get("axis", np.array([0, 0, 1]))
        origin = aR.get("origin", np.array([0, 0, 0]))
        M2 = _compose_about_origin(M, axis, origin, angle)
        T = _translation_matrix(_normalize(aT.get("axis", axis)) * (trans_value * scale_mult))
        return T @ M2

    if comp_type == 'cylindrical':
        aR = _first_of('revolute') or arts[0]
        aT = _first_of('prismatic') or arts[0]
        only_cyl_entry = all(a.get("type") == "cylindrical" for a in arts)
        rot_default_max = 2.0 * math.pi if only_cyl_entry else (math.pi / 2.0)
        rmin, rmax = _get_range(aR, ["rangeMin"], ["rangeMax"], 0.0, rot_default_max)
        tmin, tmax = _get_range(
            aT,
            ["prismatic_rangeMin", "rangeMin"],
            ["prismatic_rangeMax", "rangeMax"],
            0.0,
            0.25 * _compute_bbox_extent_along_axis(full_bbox, aT.get("axis", [0, 0, 1])),
        )

        # Apply both motions simultaneously for smooth cylindrical behavior.
        angle = rmin + (rmax - rmin) * ((frame % n_frames) / float(max(1, n_frames)))
        value = _value_at_frame_pingpong(tmin, tmax, frame, n_frames)
        axis_r = aR.get("axis", np.array([0, 0, 1]))
        origin_r = aR.get("origin", np.array([0, 0, 0]))
        M_rot = _compose_about_origin(M, axis_r, origin_r, angle)
        axis_t = _normalize(aT.get("axis", axis_r))
        T = _translation_matrix(axis_t * (value * scale_mult))
        return T @ M_rot

    if comp_type == 'universal':
        a = arts[0]
        axis1 = _normalize(a.get("axis", np.array([0, 0, 1])))
        origin1 = a.get("origin", np.array([0, 0, 0]))
        axis2_raw = a.get("axis2", a.get("ref", np.array([1, 0, 0])))
        axis2 = _normalize(axis2_raw)
        if np.linalg.norm(axis2) < 1e-8:
            axis2 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(axis1, axis2))) > 0.98:
            # Keep universal axes distinct for stable visualization.
            axis2 = _normalize(np.cross(axis1, np.array([0.0, 0.0, 1.0])))
            if np.linalg.norm(axis2) < 1e-8:
                axis2 = _normalize(np.cross(axis1, np.array([0.0, 1.0, 0.0])))
        origin2 = a.get("origin2", origin1)
        r1min, r1max = _get_range(
            a,
            ["rangeMin", "rot1RangeMin"],
            ["rangeMax", "rot1RangeMax"],
            -math.pi,
            math.pi,
        )
        r2min, r2max = _get_range(
            a,
            ["rot2RangeMin", "range2Min", "rotation2_rangeMin"],
            ["rot2RangeMax", "range2Max", "rotation2_rangeMax"],
            -math.pi,
            math.pi,
        )
        # Sequential universal animation:
        # phase-1: animate axis1 across full min/max cycle while axis2 stays at baseline (r2min)
        # phase-2: animate axis2 across full min/max cycle while axis1 stays at baseline (r1min)
        half = max(1, n_frames // 2)
        if (frame % n_frames) < half:
            ang1 = _value_at_frame_pingpong(r1min, r1max, frame % half, half)
            M1 = _compose_about_origin(M, axis1, origin1, ang1)
            return _compose_about_origin(M1, axis2, origin2, r2min)
        else:
            ang2 = _value_at_frame_pingpong(r2min, r2max, (frame - half) % half, half)
            M1 = _compose_about_origin(M, axis1, origin1, r1min)
            return _compose_about_origin(M1, axis2, origin2, ang2)

    # Fallback
    a0 = arts[0]
    if a0["type"] == 'revolute':
        rmin = a0.get("rangeMin", 0.0)
        rmax = a0.get("rangeMax", math.pi/2.0)
        angle = _value_at_frame_pingpong(rmin, rmax, frame, n_frames)
        return _compose_about_origin(M, a0.get("axis", [0, 0, 1]), a0.get("origin", [0, 0, 0]), angle)
    if a0["type"] == 'prismatic':
        rmin, rmax = _get_range(
            a0,
            ["prismatic_rangeMin", "rangeMin"],
            ["prismatic_rangeMax", "rangeMax"],
            0.0,
            0.25 * _compute_bbox_extent_along_axis(full_bbox, a0.get("axis", [0, 0, 1])),
        )
        value = _value_at_frame_pingpong(rmin, rmax, frame, n_frames)
        T = _translation_matrix(_normalize(a0.get("axis", [0, 0, 1])) * (value * scale_mult))
        return T @ M
    return M


def _render_frames(output_dir, start_frame, end_frame):
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for f in range(start_frame, end_frame + 1):
        bpy.context.scene.frame_current = f
        out_path = os.path.join(output_dir, f"frame_{f:04d}.png")
        bpy.context.scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        paths.append(out_path)
    return paths


def _encode_videos(frame_paths, fps, output_dir, highlight_pid):
    os.makedirs(output_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        list_file = os.path.join(temp_dir, "frames.txt")
        with open(list_file, 'w') as f:
            for p in frame_paths:
                f.write(f"file '{os.path.abspath(p)}'\n")
                f.write(f"duration {1.0/float(fps)}\n")
            f.write(f"file '{os.path.abspath(frame_paths[-1])}'\n")

        webm_path = os.path.join(output_dir, f"{highlight_pid}.webm")

        webm_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", list_file,
            "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "2M", "-crf", "30", "-deadline", "good", "-cpu-used", "2",
            webm_path
        ]
        subprocess.run(webm_cmd)

        print(f"Saved: WEBM={webm_path}")

        # Remove PNG frames now that the video is created
        for p in frame_paths:
            if os.path.exists(p) and p.endswith(".png"):
                os.remove(p)


def _rotate_x(angle_rad):
    return Matrix.Rotation(angle_rad, 4, 'X')


def _apply_y_up_to_z_up_on_articulations(arts_by_pid):
    R = _rotate_x(math.pi / 2.0)
    R3 = Matrix(((R[0][0], R[0][1], R[0][2]), (R[1][0], R[1][1], R[1][2]), (R[2][0], R[2][1], R[2][2])))

    def rot_vec3(v):
        v3 = Vector((float(v[0]), float(v[1]), float(v[2])))
        out = R3 @ v3
        return np.array([out[0], out[1], out[2]], dtype=np.float64)

    def rot_point3(p):
        p4 = Vector((float(p[0]), float(p[1]), float(p[2]), 1.0))
        out = R @ p4
        return np.array([out[0], out[1], out[2]], dtype=np.float64)

    for pid, alist in arts_by_pid.items():
        for a in alist:
            if a.get("axis") is not None:
                a["axis"] = rot_vec3(a["axis"]).astype(np.float64)
            if a.get("origin") is not None:
                a["origin"] = rot_point3(a["origin"]).astype(np.float64)
            if a.get("axis2") is not None:
                a["axis2"] = rot_vec3(a["axis2"]).astype(np.float64)
            if a.get("origin2") is not None:
                a["origin2"] = rot_point3(a["origin2"]).astype(np.float64)


def _set_textured_material_double_sided():
    for mat in bpy.data.materials:
        try:
            mat.use_backface_culling = False
            if hasattr(mat, "use_backface_culling_shadow"):
                mat.use_backface_culling_shadow = False
            if hasattr(mat, "show_transparent_back"):
                mat.show_transparent_back = True
        except Exception:
            continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_input_path', type=str, required=True, help='Path to model directory containing <id>.glb and segmentation')
    parser.add_argument('--art_json_path', type=str, required=True, help='Path to articulations JSON file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for frames/videos')
    parser.add_argument('--highlight_pid', type=int, default=None, help='Part id to highlight')
    parser.add_argument('--frames', type=int, default=64, help='Total frames to render')
    parser.add_argument('--fps', type=int, default=15, help='Frames per second for video')
    parser.add_argument('--turntable', action='store_true', help='Enable camera turntable during animation')
    parser.add_argument('--num_views', type=int, default=16, help='Number of distinct camera positions if turntable is used')
    parser.add_argument('--camera_dist', type=float, default=1.7, help='Base camera distance for turntable')
    parser.add_argument('--res', type=int, default=768, help='Square render resolution')
    parser.add_argument('--light_scale', type=float, default=0.4, help='Lighting intensity scale')
    parser.add_argument('--ambient_strength', type=float, default=0.08, help='Ambient world background strength')
    parser.add_argument('--light_size_scale', type=float, default=1.8, help='Scale multiplier for light size/radius')
    parser.add_argument('--animate_mode', type=str, choices=['single','all'], default=None, help='Animate only highlighted part (single) or all articulated parts (all)')
    parser.add_argument('--render_mode', type=str, choices=['highlight', 'textured'], default='highlight', help='Highlight mode uses magenta/gray part materials; textured keeps original GLB materials')
    parser.add_argument('--azimuth_offset_deg', type=float, default=0.0, help='Camera azimuth offset in degrees for side-view rendering')
    parser.add_argument('--debug', action='store_true', help='Print debug info for axes/origins and applied transforms')
    parser.add_argument('--articulations_y_up', action='store_true', default=True, help='Treat articulation axes/origins as Y-up and convert to Blender Z-up')
    parser.add_argument('--force_highlight', action='store_true', help='Highlight highlighted pid even if it has no non-fixed articulation')
    _default_hdri = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brown_photostudio_01_4k.exr")
    parser.add_argument('--hdri_path', type=str, default=_default_hdri, help='Path to HDRI .exr for world environment lighting')
    args = parser.parse_args()

    print(f"[INFO] Loading model from {args.model_input_path}")
    _init_scene(res_x=args.res, res_y=args.res, ambient_strength=args.ambient_strength, hdri_path=args.hdri_path)

    arts_by_pid = _parse_articulations(args.art_json_path)
    if args.articulations_y_up:
        if args.debug:
            print("[DEBUG] Applying Y-up -> Z-up conversion to articulation axes/origins (rotate +90deg around X)")
        _apply_y_up_to_z_up_on_articulations(arts_by_pid)
    
    if args.render_mode == "highlight":
        part_empty_by_pid, imported_by_pid, parent_by_pid, highlight_bbox, full_bbox = _load_segmented_parts(
            args.model_input_path,
            args.highlight_pid,
            arts_by_pid,
            force_highlight=args.force_highlight,
        )
    else:
        part_empty_by_pid, imported_by_pid, parent_by_pid, highlight_bbox, full_bbox = _load_textured_segmented_parts(
            args.model_input_path,
            args.highlight_pid,
            arts_by_pid,
        )
        _set_textured_material_double_sided()

    # Add lighting aimed at the actual model center (requires bbox from loaded geometry)
    scene_center = (np.array(full_bbox[0]) + np.array(full_bbox[1])) / 2.0
    _add_lighting(args.light_scale, light_size_scale=args.light_size_scale, center=scene_center)
    if args.highlight_pid is not None and int(args.highlight_pid) not in part_empty_by_pid:
        print(f"[WARN] highlighted pid={args.highlight_pid} not found in loaded objects for render_mode={args.render_mode}")
    composites = _collect_composite_types(arts_by_pid)
    baselines = _prepare_baselines(part_empty_by_pid)

    # Prismatic origin fallback: part centers for missing/zero origins
    prismatic_origin_by_pid = {}
    for pid, alist in arts_by_pid.items():
        for a in alist:
            if a["type"] == 'prismatic':
                o = a.get("origin", None)
                if o is None or (isinstance(o, (list, np.ndarray)) and np.allclose(o, 0.0)):
                    if pid in imported_by_pid:
                        prismatic_origin_by_pid[pid] = _bbox_center_from_objects(imported_by_pid[pid])
                        if args.debug:
                            print(f"[DEBUG][pid={pid}] prismatic origin fallback -> part center {_fmt_vec(prismatic_origin_by_pid[pid])}")

    # Camera setup
    cam_obj, center, radius, dist = _compute_camera_static(
        full_bbox,
        azimuth_offset_deg=args.azimuth_offset_deg,
    )
    if args.turntable:
        orbit_radius = args.camera_dist if args.camera_dist > 0 else dist
        cam_positions = _compute_turntable_positions_with_offset(
            np.array(center),
            orbit_radius,
            args.frames,
            elevation_ratio=0.2,
            azimuth_offset_deg=args.azimuth_offset_deg,
        )
    else:
        cam_positions = None

    # Determine animate set - only animate parts that have non-fixed articulations
    animate_mode = args.animate_mode
    if animate_mode is None:
        animate_mode = 'single' if args.highlight_pid is not None else 'all'
    
    if animate_mode == 'single' and args.highlight_pid is not None:
        # Only animate the highlighted part if it has non-fixed articulations
        highlight_arts = arts_by_pid.get(int(args.highlight_pid), [])
        has_non_fixed = any(art.get("type", "fixed") not in ["fixed", "free"] for art in highlight_arts)
        animate_pids = {int(args.highlight_pid)} if has_non_fixed else set()
    else:
        # Animate all parts that have non-fixed articulations
        animate_pids = set()
        for pid, arts in arts_by_pid.items():
            if pid in part_empty_by_pid:  # Only consider loaded parts
                has_non_fixed = any(art.get("type", "fixed") not in ["fixed", "free"] for art in arts)
                if has_non_fixed:
                    animate_pids.add(pid)

    if args.debug:
        print(f"[DEBUG] animate_mode={animate_mode} animate_pids={sorted(list(animate_pids))}")
        if args.highlight_pid is not None and int(args.highlight_pid) in arts_by_pid:
            print(f"[DEBUG] articulations for highlighted pid={args.highlight_pid}: {json.dumps(arts_by_pid[int(args.highlight_pid)], default=lambda o: o.tolist() if hasattr(o,'tolist') else o)}")

    # If single mode and highlighted pid is static (no non-fixed articulation), skip rendering entirely
    if animate_mode == 'single' and (args.highlight_pid is None or int(args.highlight_pid) not in animate_pids) and not args.force_highlight:
        print(f"[SKIP] pid={args.highlight_pid} has no non-fixed articulations; skipping render and encoding")
        return

    # Render frames with animated articulations
    os.makedirs(args.output_dir, exist_ok=True)
    start_frame = 1
    end_frame = args.frames

    for f in range(start_frame, end_frame + 1):
        # Apply articulations to parts that should be animated
        # Blender's parenting system will automatically handle child transforms
        for pid, empty in part_empty_by_pid.items():
            arts = arts_by_pid.get(pid, [])
            comp_type = composites.get(pid, 'fixed')
            if pid in animate_pids and comp_type != 'fixed':
                # Apply articulation to this part
                M_art = _compute_articulated_matrix(baselines[pid], arts, comp_type, f - start_frame, args.frames, full_bbox, prismatic_origin_by_pid.get(pid))
                _apply_world_matrix(empty, M_art)
            else:
                # Reset to baseline only if this is a root part (no parent)
                if pid not in parent_by_pid:
                    _apply_world_matrix(empty, baselines[pid])

        if cam_positions is not None:
            cam_obj.location = Vector(cam_positions[f - start_frame])
            _set_camera_lookat(cam_obj, center)

        bpy.context.scene.frame_current = f
        out_path = os.path.join(args.output_dir, f"frame_{f:04d}.png")
        bpy.context.scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)

    # Encode outputs
    frame_paths = [os.path.join(args.output_dir, f"frame_{f:04d}.png") for f in range(start_frame, end_frame + 1)]
    _encode_videos(frame_paths, args.fps, args.output_dir, args.highlight_pid)

    # Save .blend for inspection
    # blend_path = os.path.join(args.output_dir, "motion.blend")
    # bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"[DONE] Saved outputs at {args.output_dir}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

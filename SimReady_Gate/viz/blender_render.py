"""Render shots.json frames with Blender Cycles (run inside Blender 3.x, headless).

  blender -b --python viz/blender_render.py -- --shots out/shots.json --list gate --out out/blender_gate \
          [--panel K] [--start 0 --end N] [--width 1920 --height 1080] [--samples 128] [--gpu 0] [--hdri studio.exr]

Bodies come from `shots["assets"][name]` = {"obj": path, "c_model": [3], "fixed": bool}; the OBJ is
the solver's own mesh in the body's native frame, so a frame pose (center, R, s) places it as
  world = center + R (s (v - c_model))  =  T(center - s R c_model) R S(s) v.
Materials are rebuilt from `<obj>.materials.json` (OmniPBR inputs) as Principled BSDF node trees.
No text is drawn. Bodies listed under "red" get a red-tinted material, "hidden" ones are skipped.
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--shots", required=True); ap.add_argument("--list", default="gate"); ap.add_argument("--out", required=True)
ap.add_argument("--panel", type=int, default=-1); ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=-1)
ap.add_argument("--width", type=int, default=1920); ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--samples", type=int, default=128); ap.add_argument("--gpu", type=int, default=-1)
ap.add_argument("--hdri", default=""); ap.add_argument("--hdri-strength", type=float, default=0.9)
ap.add_argument("--world-gray", type=float, default=0.75)
ap.add_argument("--sun", type=float, default=3.0); ap.add_argument("--floor-z", type=float, default=None)
ap.add_argument("--fov-deg", type=float, default=45.0); ap.add_argument("--red-mix", type=float, default=0.65)
ap.add_argument("--exposure", type=float, default=0.0); ap.add_argument("--stride", type=int, default=1)
a = ap.parse_args(argv)

shots = json.load(open(a.shots))
frames = [s for s in shots[a.list] if a.panel < 0 or s.get("panel", -1) == a.panel]
end = len(frames) if a.end < 0 else min(a.end, len(frames))
os.makedirs(a.out, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = a.samples
scene.cycles.use_adaptive_sampling = True; scene.cycles.adaptive_threshold = 0.02
scene.cycles.use_denoising = True
try:
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
except Exception:  # noqa: BLE001
    pass
scene.cycles.max_bounces = 6; scene.cycles.caustics_reflective = False; scene.cycles.caustics_refractive = False
scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = a.width, a.height, 100
scene.render.image_settings.file_format = "PNG"; scene.render.image_settings.color_mode = "RGB"
scene.render.film_transparent = False
try:
    scene.view_settings.view_transform = "Filmic"; scene.view_settings.look = "Medium Contrast"
except TypeError:                     # this Blender build ships without the Filmic OCIO config
    scene.view_settings.view_transform = "Standard"
scene.view_settings.exposure = a.exposure
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "CUDA"; prefs.get_devices()
n_gpu = 0
for i, d in enumerate([d for d in prefs.devices if d.type == "CUDA"]):
    d.use = (a.gpu < 0) or (i == a.gpu); n_gpu += int(d.use)
scene.cycles.device = "GPU" if n_gpu else "CPU"
print("[blender] cycles device", scene.cycles.device, "gpus enabled", n_gpu, flush=True)

# --- world: HDRI + sun -------------------------------------------------------------------------------
world = bpy.data.worlds.new("World"); scene.world = world; world.use_nodes = True
nt = world.node_tree; nt.nodes.clear()
bg = nt.nodes.new("ShaderNodeBackground"); outw = nt.nodes.new("ShaderNodeOutputWorld")
if a.hdri and os.path.exists(a.hdri):
    env = nt.nodes.new("ShaderNodeTexEnvironment"); env.image = bpy.data.images.load(a.hdri)
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = a.hdri_strength
else:
    g = a.world_gray; bg.inputs["Color"].default_value = (g, g, g, 1.0); bg.inputs["Strength"].default_value = 1.0
nt.links.new(bg.outputs["Background"], outw.inputs["Surface"])
if a.sun > 0:
    sun_data = bpy.data.lights.new("Sun", "SUN"); sun_data.energy = a.sun; sun_data.angle = math.radians(4.0)
    sun = bpy.data.objects.new("Sun", sun_data); scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(35.0), math.radians(12.0), math.radians(-60.0))

# --- materials ----------------------------------------------------------------------------------------
_images = {}


def image(path, colorspace):
    key = (path, colorspace)
    if key not in _images:
        im = bpy.data.images.load(path)
        for name in ((colorspace, "Linear") if colorspace == "Non-Color" else (colorspace,)):
            try:
                im.colorspace_settings.name = name; break
            except TypeError:               # builds without the full OCIO config only know Linear / sRGB
                continue
        _images[key] = im
    return _images[key]


def build_material(mat, rec, red=False, box_project=False):
    """Rebuild `mat`'s node tree from an OmniPBR record; red=True tints the base colour;
    box_project=True samples textures by world-space box projection (meshes without UVs)."""
    mat.use_nodes = True; nt = mat.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial"); bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    col = rec.get("diffuse_color") or [0.6, 0.6, 0.6]
    tint = rec.get("diffuse_tint") or [1.0, 1.0, 1.0]
    base_src = None
    uvmap = None
    ts = rec.get("texture_scale")
    if ts or rec.get("texture_translate"):
        uvmap = nt.nodes.new("ShaderNodeUVMap"); mapping = nt.nodes.new("ShaderNodeMapping")
        nt.links.new(uvmap.outputs["UV"], mapping.inputs["Vector"])
        if ts:
            mapping.inputs["Scale"].default_value = (float(ts[0]), float(ts[1]), 1.0)
        tt = rec.get("texture_translate")
        if tt:
            mapping.inputs["Location"].default_value = (float(tt[0]), float(tt[1]), 0.0)
        uvmap = mapping
    box_src = None
    if box_project:
        tc = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (2.0, 2.0, 2.0)          # one texture tile per 0.5 m
        nt.links.new(tc.outputs["Object"], mp.inputs["Vector"]); box_src = mp.outputs["Vector"]
    def tex(path, colorspace):
        n = nt.nodes.new("ShaderNodeTexImage"); n.image = image(path, colorspace)
        if box_project:
            n.projection = "BOX"; n.projection_blend = 0.2; nt.links.new(box_src, n.inputs["Vector"])
        elif uvmap is not None:
            nt.links.new(uvmap.outputs["Vector"], n.inputs["Vector"])
        return n
    if rec.get("diffuse_texture") and os.path.exists(rec["diffuse_texture"]):
        t = tex(rec["diffuse_texture"], "sRGB")
        mix = nt.nodes.new("ShaderNodeMixRGB"); mix.blend_type = "MULTIPLY"; mix.inputs["Fac"].default_value = 1.0
        mix.inputs["Color2"].default_value = (tint[0], tint[1], tint[2], 1.0)
        nt.links.new(t.outputs["Color"], mix.inputs["Color1"]); base_src = mix.outputs["Color"]
    else:
        bsdf.inputs["Base Color"].default_value = (col[0] * tint[0], col[1] * tint[1], col[2] * tint[2], 1.0)
    if red:
        mixr = nt.nodes.new("ShaderNodeMixRGB"); mixr.blend_type = "MIX"; mixr.inputs["Fac"].default_value = a.red_mix
        mixr.inputs["Color2"].default_value = (0.85, 0.12, 0.10, 1.0)
        if base_src is not None:
            nt.links.new(base_src, mixr.inputs["Color1"])
        else:
            mixr.inputs["Color1"].default_value = bsdf.inputs["Base Color"].default_value
        base_src = mixr.outputs["Color"]
    if base_src is not None:
        nt.links.new(base_src, bsdf.inputs["Base Color"])
    rough = rec.get("roughness"); bsdf.inputs["Roughness"].default_value = float(rough) if isinstance(rough, (int, float)) else 0.5
    met = rec.get("metallic"); bsdf.inputs["Metallic"].default_value = float(met) if isinstance(met, (int, float)) else 0.0
    if rec.get("enable_orm") and rec.get("orm_texture") and os.path.exists(rec["orm_texture"]):
        t = tex(rec["orm_texture"], "Non-Color"); sep = nt.nodes.new("ShaderNodeSeparateRGB")
        nt.links.new(t.outputs["Color"], sep.inputs["Image"])
        nt.links.new(sep.outputs["G"], bsdf.inputs["Roughness"]); nt.links.new(sep.outputs["B"], bsdf.inputs["Metallic"])
    else:
        if rec.get("roughness_texture") and os.path.exists(rec["roughness_texture"]):
            t = tex(rec["roughness_texture"], "Non-Color"); nt.links.new(t.outputs["Color"], bsdf.inputs["Roughness"])
        if rec.get("metallic_texture") and os.path.exists(rec["metallic_texture"]):
            t = tex(rec["metallic_texture"], "Non-Color"); nt.links.new(t.outputs["Color"], bsdf.inputs["Metallic"])
    if rec.get("normal_texture") and os.path.exists(rec["normal_texture"]):
        t = tex(rec["normal_texture"], "Non-Color"); nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(t.outputs["Color"], nm.inputs["Color"]); nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def import_body(name, asset):
    """Import the OBJ (no axis conversion), one object, materials rebuilt from the JSON."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.obj(filepath=asset["obj"], use_split_objects=False, use_split_groups=False, use_image_search=False,
                             axis_forward="Y", axis_up="Z")
    new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not new:
        raise RuntimeError(f"nothing imported from {asset['obj']}")
    obj = new[0]
    if len(new) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in new:
            o.select_set(True)
        bpy.context.view_layer.objects.active = obj; bpy.ops.object.join()
    obj.name = name
    info = json.load(open(os.path.splitext(asset["obj"])[0] + ".materials.json"))
    recs = info.get("materials", {})
    no_uv = int(info.get("n_uv", 0)) == 0          # the importer may add an empty UV layer; trust the export
    originals, reds = [], []
    for slot in obj.material_slots:
        m = slot.material
        if m is None:
            continue
        base = m.name.split(".")[0]
        rec = recs.get(base, {})
        build_material(m, rec, red=False, box_project=no_uv)
        r = bpy.data.materials.new(m.name + "_red"); build_material(r, rec, red=True, box_project=no_uv)
        originals.append(m); reds.append(r)
    obj.data.use_auto_smooth = True; obj.data.auto_smooth_angle = math.radians(35.0)
    for p in obj.data.polygons:
        p.use_smooth = True
    obj["c_model"] = list(asset["c_model"])
    return obj, originals, reds


bodies = {}
always_hidden = set(shots.get("always_hidden", []))
for name, asset in shots["assets"].items():
    if name in always_hidden:
        continue                                   # ground sheets: the floor plane stands in
    if not asset.get("obj") or not os.path.exists(asset["obj"]):
        print("[blender] no obj for", name, flush=True); continue
    bodies[name] = import_body(name, asset)
print("[blender] imported", len(bodies), "bodies", flush=True)

floor_z = a.floor_z if a.floor_z is not None else shots.get("floor_z")
if floor_z is not None:
    bpy.ops.mesh.primitive_plane_add(size=40.0, location=(0.0, 0.0, floor_z))
    fl = bpy.context.active_object; fl.name = "floor"
    fm = bpy.data.materials.new("floor"); build_material(fm, {"diffuse_color": [0.62, 0.62, 0.60], "roughness": 0.8}); fl.data.materials.append(fm)

# --- camera --------------------------------------------------------------------------------------------
cam_spec = shots.get("camera_settle" if a.list == "settle" else "camera", {})
target = Vector(cam_spec.get("target", (0.4, 0.0, 0.05))); dist = cam_spec.get("dist", 1.3)
el, az = math.radians(cam_spec.get("elev_deg", 32.0)), math.radians(cam_spec.get("azim_deg", -35.0))
az_end = math.radians(cam_spec["azim_deg_end"]) if "azim_deg_end" in cam_spec else az
dist_end = cam_spec.get("dist_end", dist)
def eye_at(u):
    """camera position at progress u in [0, 1] (orbit / dolly when the spec asks for it)"""
    a_ = az + (az_end - az) * u; d_ = dist + (dist_end - dist) * u
    return target + d_ * Vector((math.cos(el) * math.cos(a_), math.cos(el) * math.sin(a_), math.sin(el)))
eye = eye_at(0.0)
cam_data = bpy.data.cameras.new("cam"); cam_data.sensor_fit = "VERTICAL"; cam_data.angle_y = math.radians(a.fov_deg)
cam = bpy.data.objects.new("cam", cam_data); scene.collection.objects.link(cam); scene.camera = cam
cam.location = eye
tgt = bpy.data.objects.new("target", None); tgt.location = target; scene.collection.objects.link(tgt)
c = cam.constraints.new("TRACK_TO"); c.target = tgt; c.track_axis = "TRACK_NEGATIVE_Z"; c.up_axis = "UP_Y"


def place(name, center, R, s):
    obj = bodies[name][0]
    cm = Vector(obj["c_model"])
    R4 = Matrix(((R[0][0], R[0][1], R[0][2], 0.0), (R[1][0], R[1][1], R[1][2], 0.0), (R[2][0], R[2][1], R[2][2], 0.0), (0.0, 0.0, 0.0, 1.0)))
    t = Vector(center) - s * (R4.to_3x3() @ cm)
    obj.matrix_world = Matrix.Translation(t) @ R4 @ Matrix.Scale(s, 4)


# fixed bodies at their identity pose (their OBJ frame is the child's local frame): pose from shots
for name, asset in shots["assets"].items():
    if name in bodies and asset.get("pose"):
        c0, R0 = asset["pose"]; place(name, c0, R0, 1.0)

for f in range(a.start, end, a.stride):
    shot = frames[f]
    cam.location = eye_at(f / max(len(frames) - 1, 1))
    for name, (cen, R, s) in shot["poses"].items():
        if name in bodies:
            place(name, cen, R, float(s))
    red, hidden = set(shot.get("red", [])), set(shot.get("hidden", []))
    for name, (obj, originals, reds) in bodies.items():
        obj.hide_render = name in hidden
        for k, slot in enumerate(obj.material_slots):
            if k < len(originals):
                slot.material = reds[k] if name in red else originals[k]
    tag = f"p{a.panel}_" if a.panel >= 0 else "f_"
    scene.render.filepath = os.path.join(a.out, f"{tag}{f:05d}.png")
    bpy.ops.render.render(write_still=True)
    print(f"[blender] frame {f}/{end} -> {scene.render.filepath}", flush=True)
print("[blender] done", flush=True)

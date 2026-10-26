"""Render shots.json frames with Blender Cycles (run inside Blender 4.5, headless).

  blender -b --python viz/blender_render.py -- --shots out/shots.json --list gate --out out/blender_gate \
          [--panel K] [--start 0 --end N] [--width 1920 --height 1080] [--samples 128] [--gpu 0] [--device OPTIX] [--hdri studio.exr]

Bodies come from `shots["assets"][name]` = {"obj": path, "c_model": [3], "fixed": bool}; the OBJ is
the solver's own mesh in the body's native frame, so a frame pose (center, R, s) places it as
  world = center + R (s (v - c_model))  =  T(center - s R c_model) R S(s) v.
Materials are rebuilt from `<obj>.materials.json` (build_material) as Principled BSDF node trees.
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
ap.add_argument("--device", choices=("OPTIX", "CUDA"), default="OPTIX")
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
scene.cycles.use_denoising = True; scene.cycles.denoiser = "OPENIMAGEDENOISE"; scene.cycles.denoising_use_gpu = True
scene.render.use_persistent_data = True          # frames differ only in poses and tints: keep meshes and images
scene.cycles.max_bounces = 6; scene.cycles.caustics_reflective = False; scene.cycles.caustics_refractive = False
scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = a.width, a.height, 100
scene.render.image_settings.file_format = "PNG"; scene.render.image_settings.color_mode = "RGB"
scene.render.film_transparent = False
scene.view_settings.view_transform = "Filmic"; scene.view_settings.look = "Medium Contrast"
scene.view_settings.exposure = a.exposure
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = a.device; prefs.get_devices()
n_gpu = 0
for d in prefs.devices:
    d.use = False                                # the GPUs asked for only, no CPU in the render
for i, d in enumerate([d for d in prefs.devices if d.type == a.device]):
    d.use = (a.gpu < 0) or (i == a.gpu); n_gpu += int(d.use)
if not n_gpu:
    raise RuntimeError(f"no {a.device} device {a.gpu} (--gpu -1 takes them all)")
scene.cycles.device = "GPU"
print("[blender] cycles", a.device, [d.name for d in prefs.devices if d.use], flush=True)

# --- world: HDRI + sun -------------------------------------------------------------------------------
world = bpy.data.worlds.new("World"); scene.world = world; world.use_nodes = True
nt = world.node_tree; nt.nodes.clear()
bg = nt.nodes.new("ShaderNodeBackground"); outw = nt.nodes.new("ShaderNodeOutputWorld")
if a.hdri:
    if not os.path.exists(a.hdri):
        raise RuntimeError(f"no HDRI at {a.hdri}")
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
        im = bpy.data.images.load(path); im.colorspace_settings.name = colorspace
        _images[key] = im
    return _images[key]


# A material record (viz/usd_to_obj.py, viz/mjcf_to_obj.py) names every value the render reads, as its shader model
# evaluates it (usd_to_obj._resolve_reads); a value missing or a key the render does not know is refused, so that
# nothing is guessed and nothing an export records goes unseen.
TEXTURES = ("diffuse_texture", "roughness_texture", "metallic_texture", "orm_texture", "normal_texture", "ao_texture")
ALWAYS = ("diffuse_tint", "albedo_brightness", "albedo_add", "albedo_desaturation", "ao_to_diffuse", "ao_source", "roughness",
          "roughness_source", "metallic", "metallic_source", "texture_combine", "specular", "ior", "texture_scale",
          "texture_translate")
NORMAL = ("normal_formula", "normal_factor", "flip_tangent_u", "flip_tangent_v")
GLASS = ("thin_walled", "opacity")
OPTIONAL = ("diffuse_color", "roughness_texture_influence", "metallic_texture_influence", "ao", "glass", *GLASS, *TEXTURES,
            *(t + "_colorspace" for t in TEXTURES), *NORMAL)
INFO = ("name", "prim", "mdl", "mdl_file", "model", "model_defaults", "not_parameters", "unbound", "display_color", "mujoco",
        "specular_level", "specular_constant", "enable_orm", "project_uvw", "world_or_object", "uv_projected", "evaluated_as")
THIN_ROUGHNESS = 0.03       # thin-walled glass up to this frosting: a transmitted lobe under 1e-3 rad wide, rendered straight
CHANNEL = {"r": "Red", "g": "Green", "b": "Blue"}


def file_of(rec, key):
    """The file a material record names for `key`; it must be on disk."""
    path = rec[key]
    if not os.path.exists(path):
        raise RuntimeError(f"material {rec.get('name')}: its {key} {path} is not on disk")
    return path


def build_material(mat, rec, red=False, has_uv=True):
    """Rebuild `mat`'s node tree from a material record: the diffuse colour (texture with its albedo adjustment, or
    constant) times the tint and the occlusion by ao_to_diffuse; roughness and metallic from their source, met with the
    constant as texture_combine says; the normal from the map by the model's formula, its tangent axes flipped as
    flip_tangent_u / _v say (MDL flips for DirectX maps, Blender's tangents are OpenGL's); F0 as Blender's IOR level and
    IOR. Texture coordinates are uv * texture_scale + texture_translate. red=True tints the base colour. A record that
    misses a value the render reads, holds a key it does not know, names a file not on disk, or puts a texture on a
    mesh without texture coordinates (`has_uv`), is refused."""
    name = rec.get("name")
    if rec.get("missing") or rec.get("unknown"):
        raise RuntimeError(f"material {name} cannot be rendered as authored: {rec.get('missing') or rec.get('unknown')}")
    unknown = set(rec) - set(ALWAYS) - set(OPTIONAL) - set(INFO)
    if unknown:
        raise RuntimeError(f"material {name}: keys the render does not know: {sorted(unknown)}")
    texs = [t for t in TEXTURES if rec.get(t)]
    need = [*ALWAYS, *(t + "_colorspace" for t in texs), *(() if rec.get("diffuse_texture") else ("diffuse_color",)),
            *(NORMAL if rec.get("normal_texture") else ()), *(("ao",) if rec.get("ao_source") == "constant" else ()),
            *(GLASS if rec.get("glass") else ())]
    if rec.get("texture_combine") == "lerp":
        need += [k + "_texture_influence" for k in ("roughness", "metallic") if rec.get(k + "_source") != "constant"]
    absent = [k for k in need if k not in rec]
    if absent:
        raise RuntimeError(f"material {name}: no {absent}")
    if texs and not has_uv:
        raise RuntimeError(f"material {name}: a texture on a mesh without texture coordinates")
    if rec.get("project_uvw") and not rec.get("uv_projected"):
        raise RuntimeError(f"material {name}: projected texture coordinates that the export did not write")
    if rec["texture_combine"] not in ("lerp", "replace"):
        raise RuntimeError(f"material {name}: texture_combine {rec['texture_combine']!r}")
    sx, sy = (float(x) for x in rec["texture_scale"])
    if rec.get("normal_texture") and (sx <= 0 or sy <= 0):
        raise RuntimeError(f"material {name}: a mirrored normal map (texture_scale {rec['texture_scale']})")
    thin = bool(rec.get("glass")) and bool(rec["thin_walled"])
    if thin and (rec["roughness_source"] != "constant" or float(rec["roughness"]) > THIN_ROUGHNESS):
        raise RuntimeError(f"material {name}: thin-walled glass of frosting roughness {rec['roughness']} ({rec['roughness_source']})")
    mat.use_nodes = True; nt = mat.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial"); bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    uv = nt.nodes.new("ShaderNodeUVMap"); mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (sx, sy, 1.0)
    mapping.inputs["Location"].default_value = (float(rec["texture_translate"][0]), float(rec["texture_translate"][1]), 0.0)
    nt.links.new(uv.outputs["UV"], mapping.inputs["Vector"])
    textures = {}

    def tex(key):
        """The colour output of the record's texture `key`, decoded as the record says (sRGB or raw)."""
        if key not in textures:
            n = nt.nodes.new("ShaderNodeTexImage")
            n.image = image(file_of(rec, key), {"sRGB": "sRGB", "raw": "Non-Color"}[rec[key + "_colorspace"]])
            nt.links.new(mapping.outputs["Vector"], n.inputs["Vector"])
            textures[key] = n.outputs["Color"]
        return textures[key]

    def channels(key):
        sep = nt.nodes.new("ShaderNodeSeparateColor"); nt.links.new(tex(key), sep.inputs["Color"])
        return sep.outputs

    def lookup(key):
        """(socket, None) when `key`'s source reads a texture, else (None, the number a lookup of no texture gives)."""
        src = rec[key + "_source"]
        if src in ("zero", "one"):
            return None, float(src == "one")
        if src in ("orm_r", "orm_g", "orm_b"):
            return channels("orm_texture")[CHANNEL[src[-1]]], None
        if src in ("texture_r", "texture_g", "texture_b"):
            return channels(key + "_texture")[CHANNEL[src[-1]]], None
        if src == "texture_rgb":
            return tex(key + "_texture"), None
        if src == "texture_max":
            ch, m = channels(key + "_texture"), None
            for a_, b_ in (("Red", "Green"), (None, "Blue")):
                n = nt.nodes.new("ShaderNodeMath"); n.operation = "MAXIMUM"
                nt.links.new(ch[a_] if m is None else m, n.inputs[0]); nt.links.new(ch[b_], n.inputs[1]); m = n.outputs["Value"]
            return m, None
        raise RuntimeError(f"material {name}: {key}_source {src!r}")

    def scalar(key, socket):
        """roughness or metallic: the constant, or a lookup met with the constant (lerp by the influence, or replace)."""
        c = float(rec[key])
        if rec[key + "_source"] == "constant":
            socket.default_value = c
            return
        sock, val = lookup(key)
        w = float(rec[key + "_texture_influence"]) if rec["texture_combine"] == "lerp" else 1.0
        if sock is None:
            socket.default_value = (1 - w) * c + w * val
        elif w == 1.0:
            nt.links.new(sock, socket)
        else:                                  # c + w (s - c)
            m = nt.nodes.new("ShaderNodeMath"); m.operation = "MULTIPLY_ADD"
            m.inputs[1].default_value = w; m.inputs[2].default_value = (1 - w) * c
            nt.links.new(sock, m.inputs[0]); nt.links.new(m.outputs["Value"], socket)

    tint = [float(x) for x in rec["diffuse_tint"]]
    if rec.get("diffuse_texture"):             # base::file_texture: texture x albedo_brightness + albedo_add, then desaturated
        base = tex("diffuse_texture")
        b, add, d = float(rec["albedo_brightness"]), float(rec["albedo_add"]), float(rec["albedo_desaturation"])
        if b != 1.0 or add != 0.0:
            ma = nt.nodes.new("ShaderNodeVectorMath"); ma.operation = "MULTIPLY_ADD"
            ma.inputs[1].default_value = (b, b, b); ma.inputs[2].default_value = (add, add, add)
            nt.links.new(base, ma.inputs[0]); base = ma.outputs["Vector"]
        if d != 0.0:                           # toward its luminance (Blender's RGB to BW: Rec. 709, as MDL's luminance)
            bw = nt.nodes.new("ShaderNodeRGBToBW"); nt.links.new(base, bw.inputs["Color"])
            mx = nt.nodes.new("ShaderNodeMixRGB"); mx.inputs["Fac"].default_value = d
            nt.links.new(base, mx.inputs["Color1"]); nt.links.new(bw.outputs["Val"], mx.inputs["Color2"]); base = mx.outputs["Color"]
        mul = nt.nodes.new("ShaderNodeMixRGB"); mul.blend_type = "MULTIPLY"; mul.inputs["Fac"].default_value = 1.0
        mul.inputs["Color2"].default_value = (tint[0], tint[1], tint[2], 1.0)
        nt.links.new(base, mul.inputs["Color1"]); base_src = mul.outputs["Color"]
    else:
        col = [float(x) for x in rec["diffuse_color"]]
        rgb = nt.nodes.new("ShaderNodeRGB"); rgb.outputs["Color"].default_value = (col[0] * tint[0], col[1] * tint[1], col[2] * tint[2], 1.0)
        base_src = rgb.outputs["Color"]
    w_ao = float(rec["ao_to_diffuse"])
    if w_ao != 0.0:                            # base x lerp(1, ao, ao_to_diffuse): MDL's multiply blend is Blender's
        ao_sock, ao_val = (None, float(rec["ao"])) if rec["ao_source"] == "constant" else lookup("ao")
        aom = nt.nodes.new("ShaderNodeMixRGB"); aom.blend_type = "MULTIPLY"; aom.inputs["Fac"].default_value = w_ao
        nt.links.new(base_src, aom.inputs["Color1"])
        if ao_sock is not None:
            nt.links.new(ao_sock, aom.inputs["Color2"])
        else:
            aom.inputs["Color2"].default_value = (ao_val, ao_val, ao_val, 1.0)
        base_src = aom.outputs["Color"]
    if red:
        mixr = nt.nodes.new("ShaderNodeMixRGB"); mixr.blend_type = "MIX"; mixr.inputs["Fac"].default_value = a.red_mix
        mixr.inputs["Color2"].default_value = (0.85, 0.12, 0.10, 1.0)
        nt.links.new(base_src, mixr.inputs["Color1"]); base_src = mixr.outputs["Color"]
    nt.links.new(base_src, bsdf.inputs["Base Color"])
    scalar("roughness", bsdf.inputs["Roughness"]); scalar("metallic", bsdf.inputs["Metallic"])
    bsdf.inputs["Specular IOR Level"].default_value = float(rec["specular"])    # F0 = F0(IOR) x 2 x level
    bsdf.inputs["IOR"].default_value = float(rec["ior"])
    surface, normal = bsdf.outputs["BSDF"], None
    if rec.get("glass") and not thin:          # OmniGlass: the glass colour tints what is transmitted
        bsdf.inputs["Transmission Weight"].default_value = 1.0
    if rec.get("normal_texture"):
        # The map's colour c is turned into the colour c' whose tangent-space normal 2c' - 1 Blender's Normal Map node
        # (strength 1, the UV map's tangents) normalizes into the model's normal: tangent components 0.5 + s f (c - 0.5)
        # (s = -1 for a flipped axis, f the factor); the normal component as is ("scale_xy", OmniPBRBase.mdl), or
        # |f (2c - 1)| + 1 - f in colour ("base_mdl", base.mdl's tangent_space_normal_texture).
        f, ch = float(rec["normal_factor"]), channels("normal_texture")
        comb = nt.nodes.new("ShaderNodeCombineColor")
        for c, flip in (("Red", rec["flip_tangent_u"]), ("Green", rec["flip_tangent_v"])):
            k = -f if flip else f
            m = nt.nodes.new("ShaderNodeMath"); m.operation = "MULTIPLY_ADD"
            m.inputs[1].default_value = k; m.inputs[2].default_value = 0.5 - 0.5 * k
            nt.links.new(ch[c], m.inputs[0]); nt.links.new(m.outputs["Value"], comb.inputs[c])
        if rec["normal_formula"] == "scale_xy":
            nt.links.new(ch["Blue"], comb.inputs["Blue"])
        elif rec["normal_formula"] == "base_mdl":
            m1 = nt.nodes.new("ShaderNodeMath"); m1.operation = "MULTIPLY_ADD"
            m1.inputs[1].default_value = 2 * f; m1.inputs[2].default_value = -f
            m2 = nt.nodes.new("ShaderNodeMath"); m2.operation = "ABSOLUTE"
            m3 = nt.nodes.new("ShaderNodeMath"); m3.operation = "MULTIPLY_ADD"
            m3.inputs[1].default_value = 0.5; m3.inputs[2].default_value = 1.0 - 0.5 * f
            nt.links.new(ch["Blue"], m1.inputs[0]); nt.links.new(m1.outputs["Value"], m2.inputs[0])
            nt.links.new(m2.outputs["Value"], m3.inputs[0]); nt.links.new(m3.outputs["Value"], comb.inputs["Blue"])
        else:
            raise RuntimeError(f"material {name}: normal_formula {rec['normal_formula']!r}")
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(comb.outputs["Color"], nm.inputs["Color"]); normal = nm.outputs["Normal"]
        nt.links.new(normal, bsdf.inputs["Normal"])
    if thin:
        # MDL thin-walled glass (OmniGlass_Opacity.mdl): fresnel_layer(ior) of a GGX reflection over a GGX transmission
        # that does not refract, the same from both faces. The reflection is a Glossy BSDF; the transmission leaves the
        # way it came whatever the normal, a Transparent BSDF tinted by the glass colour; the Fresnel weight is taken at
        # the IOR on both faces (Blender's Fresnel node inverts it on back faces, so they are given 1 / IOR).
        ior = float(rec["ior"])
        glossy = nt.nodes.new("ShaderNodeBsdfAnisotropic"); glossy.distribution = "GGX"
        glossy.inputs["Roughness"].default_value = float(rec["roughness"])
        clear = nt.nodes.new("ShaderNodeBsdfTransparent"); nt.links.new(base_src, clear.inputs["Color"])
        geo = nt.nodes.new("ShaderNodeNewGeometry"); eta = nt.nodes.new("ShaderNodeMath"); eta.operation = "MULTIPLY_ADD"
        eta.inputs[1].default_value = 1.0 / ior - ior; eta.inputs[2].default_value = ior
        nt.links.new(geo.outputs["Backfacing"], eta.inputs[0])
        fres = nt.nodes.new("ShaderNodeFresnel"); nt.links.new(eta.outputs["Value"], fres.inputs["IOR"])
        if normal is not None:
            nt.links.new(normal, glossy.inputs["Normal"]); nt.links.new(normal, fres.inputs["Normal"])
        mix = nt.nodes.new("ShaderNodeMixShader"); nt.links.new(fres.outputs["Fac"], mix.inputs["Fac"])
        nt.links.new(clear.outputs["BSDF"], mix.inputs[1]); nt.links.new(glossy.outputs["BSDF"], mix.inputs[2])
        surface = mix.outputs["Shader"]
    if rec.get("glass") and float(rec["opacity"]) != 1.0:      # the cutout: what is not there lets light through
        hole = nt.nodes.new("ShaderNodeBsdfTransparent"); cut = nt.nodes.new("ShaderNodeMixShader")
        cut.inputs["Fac"].default_value = float(rec["opacity"])
        nt.links.new(hole.outputs["BSDF"], cut.inputs[1]); nt.links.new(surface, cut.inputs[2]); surface = cut.outputs["Shader"]
    nt.links.new(surface, out.inputs["Surface"])
    return mat


def import_body(name, asset):
    """Import the OBJ (no axis conversion), one object, materials rebuilt from the JSON."""
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=asset["obj"], use_split_objects=False, use_split_groups=False, forward_axis="Y", up_axis="Z")
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
    has_uv = int(info["n_uv"]) > 0                  # the importer may add an empty UV layer; trust the export
    originals, reds = [], []
    for slot in obj.material_slots:
        m = slot.material
        if m is None:
            continue
        base = m.name.split(".")[0]
        if base not in recs:
            raise RuntimeError(f"{name}: material {m.name} is not in the materials.json of {asset['obj']}")
        rec = recs[base]                           # 'default': the exporter's record of faces without a material
        build_material(m, rec, red=False, has_uv=has_uv)
        r = bpy.data.materials.new(m.name + "_red"); build_material(r, rec, red=True, has_uv=has_uv)
        originals.append(m); reds.append(r)
    obj.data.shade_smooth(); obj.data.set_sharp_from_angle(angle=math.radians(35.0))
    obj["c_model"] = list(asset["c_model"])
    return obj, originals, reds


bodies = {}
always_hidden = set(shots.get("always_hidden", []))
for name, asset in shots["assets"].items():
    if name in always_hidden:
        continue                                   # ground sheets: the floor plane stands in
    if not asset.get("obj") or not os.path.exists(asset["obj"]):
        raise RuntimeError(f"no OBJ for {name}: {asset.get('obj')}")
    bodies[name] = import_body(name, asset)
print("[blender] imported", len(bodies), "bodies", flush=True)

floor_z = a.floor_z if a.floor_z is not None else shots.get("floor_z")
if floor_z is not None:
    bpy.ops.mesh.primitive_plane_add(size=40.0, location=(0.0, 0.0, floor_z))
    fl = bpy.context.active_object; fl.name = "floor"
    fm = bpy.data.materials.new("floor"); build_material(fm, {"name": "floor", "diffuse_color": [0.62, 0.62, 0.60], "diffuse_tint": [1.0, 1.0, 1.0], "albedo_brightness": 1.0,
                         "albedo_add": 0.0, "albedo_desaturation": 0.0, "roughness": 0.8, "roughness_source": "constant", "metallic": 0.0,
                         "metallic_source": "constant", "texture_combine": "replace", "ao_source": "one", "ao_to_diffuse": 0.0,
                         "specular": 0.5, "ior": 1.5, "texture_scale": [1.0, 1.0], "texture_translate": [0.0, 0.0]}); fl.data.materials.append(fm)

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

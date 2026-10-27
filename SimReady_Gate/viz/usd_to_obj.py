"""Export a RoboLab object/fixture USD to OBJ (+ a materials JSON) for rendering.

The geometry is exactly what the solver sees: the same prim traversal and local-to-world
composition as `simready.io.usd_io.load_object_usd`, so the render mesh and the collision
mesh are the same vertices in the same (native default-prim) frame. UVs (`primvars:st`) and
per-face material assignment (GeomSubsets) are carried into the OBJ. Each material is written to
`<name>.materials.json` (MTL cannot express it): for the shader models the export carries (MODELS:
OmniPBR, SimPBR, OmniGlass, UsdPreviewSurface, and library materials that are OmniPBR with other
defaults), every parameter that changes what the model shows, authored or at the model's default,
with its texture files and how they are decoded. A parameter the render does not carry set off its
default is refused; so are another shader and a texture that is not on disk (check_renderable).

Usage: python viz/usd_to_obj.py <file.usd> <out_dir> [--name NAME]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import simready  # noqa: E402,F401
from simready.io.usd_io import _gprim_mesh, _is_geom, _matrix_np  # noqa: E402

ROBOLAB_MATERIALS = os.path.join(os.environ.get("ROBOLAB_DIR", str(Path(__file__).resolve().parents[2] / "ext" / "RoboLab")), "assets", "materials")

# Every shader model the export carries, as its definition declares it: each parameter (for an MDL model the signature
# in the model's .mdl file, for UsdPreviewSurface its specification) with the record key it becomes and its default.
# Key None: the render does not carry the parameter; at its default it changes nothing the model shows, at another value
# the material is refused. INERT: the parameter changes nothing the model shows for what the export reads (the comment
# says why). GATES: the switch a parameter is read under; while the switch keeps its default, the parameter changes
# nothing. An authored input that is no parameter of the model is not read by the model either; the record lists it
# under not_parameters.
INERT = "inert"
_MONO_AVERAGE = 1                   # base::mono_average (base.mdl: mono_alpha 0, mono_average 1, mono_luminance 2, ...)
OMNIPBR = {   # OmniPBR.mdl 2.1.2, evaluated by OmniPBR_ClearCoat.mdl 2.1.1
    "diffuse_color_constant": ("diffuse_color", (0.2, 0.2, 0.2)), "diffuse_texture": ("diffuse_texture", None),
    "albedo_desaturation": ("albedo_desaturation", 0.0), "albedo_add": ("albedo_add", 0.0),
    "albedo_brightness": ("albedo_brightness", 1.0), "diffuse_tint": ("diffuse_tint", (1.0, 1.0, 1.0)),
    "reflection_roughness_constant": ("roughness", 0.5),
    "reflection_roughness_texture_influence": ("roughness_texture_influence", 0.0),
    "reflectionroughness_texture": ("roughness_texture", None), "metallic_constant": ("metallic", 0.0),
    "metallic_texture_influence": ("metallic_texture_influence", 0.0), "metallic_texture": ("metallic_texture", None),
    "specular_level": ("specular_level", 0.5), "enable_ORM_texture": ("enable_orm", False), "ORM_texture": ("orm_texture", None),
    "ao_to_diffuse": ("ao_to_diffuse", 0.0), "ao_texture": ("ao_texture", None),
    "enable_emission": (None, False), "emissive_color": (None, (1.0, 0.1, 0.1)), "emissive_color_texture": (None, None),
    "emissive_mask_texture": (None, None), "emissive_intensity": (None, 40.0),
    "enable_opacity": (None, False), "opacity_texture": (None, None), "opacity_constant": (None, 1.0),
    "enable_opacity_texture": (None, False), "opacity_mode": (None, _MONO_AVERAGE), "opacity_threshold": (None, 0.0),
    # adds roughness only from what an Omniverse-packed normal map stores (OmniPBRBase.mdl); a map in [0, 1] stores none
    "geometry_normal_roughness_strength": (INERT, 1.0),
    "bump_factor": ("normal_factor", 1.0), "normalmap_texture": ("normal_texture", None),
    "detail_bump_factor": (None, 0.3), "detail_normalmap_texture": (None, None),
    "flip_tangent_u": ("flip_tangent_u", False), "flip_tangent_v": ("flip_tangent_v", True),
    "project_uvw": ("project_uvw", False), "world_or_object": ("world_or_object", False), "uv_space_index": (None, 0),
    "texture_translate": ("texture_translate", (0.0, 0.0)), "texture_rotate": (None, 0.0),
    "texture_scale": ("texture_scale", (1.0, 1.0)), "detail_texture_translate": (None, (0.0, 0.0)),
    "detail_texture_rotate": (None, 0.0), "detail_texture_scale": (None, (1.0, 1.0)),
    "round_edges_radius": (None, 0.0), "round_edges_roundness": (None, 1.0), "round_edges_across_materials": (None, False),
}
OMNIPBR_GATES = {
    **dict.fromkeys(("emissive_color", "emissive_color_texture", "emissive_mask_texture", "emissive_intensity"), "enable_emission"),
    **dict.fromkeys(("opacity_texture", "opacity_constant", "enable_opacity_texture", "opacity_mode", "opacity_threshold"), "enable_opacity"),
    **dict.fromkeys(("detail_bump_factor", "detail_texture_translate", "detail_texture_rotate", "detail_texture_scale"),
                    "detail_normalmap_texture"),
    **dict.fromkeys(("round_edges_roundness", "round_edges_across_materials"), "round_edges_radius")}
SIMPBR = {   # SimPBR.mdl, evaluated by SimPBR_Model.mdl
    "diffuse_color_constant": ("diffuse_color", (0.2, 0.2, 0.2)), "diffuse_texture": ("diffuse_texture", None),
    "diffuse_tint": ("diffuse_tint", (1.0, 1.0, 1.0)), "enable_transmission": (None, False), "opacity_ratio": (None, 1.0),
    "opacity_tint": (None, (1.0, 1.0, 1.0)), "opacity_multiplier": (None, 1.0),
    "reflection_roughness_constant": ("roughness", 0.5),
    "reflection_roughness_texture_influence": ("roughness_texture_influence", 0.0),
    "reflectionroughness_texture": ("roughness_texture", None), "anisotropy_constant": (None, 0.0),
    "anisotropy_texture_influence": (None, 0.0), "anisotropy_texture": (None, None), "metallic_constant": ("metallic", 0.0),
    "metallic_texture_influence": ("metallic_texture_influence", 0.0), "metallic_texture": ("metallic_texture", None),
    "specular_constant": ("specular_constant", 1.0), "specular_texture": (None, None),
    "enable_ORM_texture": ("enable_orm", True), "ORM_texture": ("orm_texture", None),
    "enable_clearcoat": (None, False), "clearcoat_tint": (None, (1.0, 1.0, 1.0)), "clearcoat_transparency": (None, 1.0),
    "clearcoat_reflection_roughness": (None, 0.0), "clearcoat_weight": (None, 1.0), "clearcoat_flatten": (None, 1.0),
    "clearcoat_ior": (None, 1.56), "clearcoat_bump_factor": (None, 1.0), "clearcoat_normal_input": (None, (0.0, 0.0, 0.0)),
    "clearcoat_normalmap_texture": (None, None), "clearcoat_flip_tangent_u": (None, False), "clearcoat_flip_tangent_v": (None, True),
    "enable_retroreflection": (None, False), "retroreflection_tint": (None, (0.182, 0.8, 0.0)), "normal_reflectivity": (None, 0.5),
    "grazing_reflectivity": (None, 1.0), "ao_to_diffuse": ("ao_to_diffuse", 0.0), "ao_input": ("ao", 1.0),
    "ao_texture": ("ao_texture", None), "enable_emission": (None, False), "emissive_color": (None, (1.0, 0.1, 0.1)),
    "emissive_mask_input": (None, (1.0, 1.0, 1.0)), "emissive_mask_texture": (None, None), "emissive_intensity": (None, 40.0),
    "enable_emissive_flipbook": (None, False), "flipbook_dim": (None, (1.0, 1.0)), "target_fps": (None, 1),
    "enable_opacity": (None, False), "alpha_constant": (None, 1.0), "opacity_texture": (None, None),
    "enable_opacity_cutout": (None, False), "alpha_cutout_cutoff": (None, 1.0),
    "bump_factor": ("normal_factor", 1.0), "normal_input": (None, "state::normal()"), "normalmap_texture": ("normal_texture", None),
    "flip_tangent_u": ("flip_tangent_u", False), "flip_tangent_v": ("flip_tangent_v", True),
    "project_uvw": ("project_uvw", False), "world_or_object": ("world_or_object", False),
    "texture_translate": ("texture_translate", (0.0, 0.0)), "texture_scale": ("texture_scale", (1.0, 1.0)),
    "texture_speed": (INERT, (0.0, 0.0)),                   # SimPBR.mdl moves the texture by 1e-8 x texture_speed
}
SIMPBR_GATES = {
    **dict.fromkeys(("opacity_ratio", "opacity_tint", "opacity_multiplier"), "enable_transmission"),
    "anisotropy_texture_influence": "anisotropy_texture",
    **dict.fromkeys(("clearcoat_tint", "clearcoat_transparency", "clearcoat_reflection_roughness", "clearcoat_weight",
                     "clearcoat_flatten", "clearcoat_ior", "clearcoat_bump_factor", "clearcoat_normal_input",
                     "clearcoat_normalmap_texture", "clearcoat_flip_tangent_u", "clearcoat_flip_tangent_v"), "enable_clearcoat"),
    **dict.fromkeys(("retroreflection_tint", "normal_reflectivity", "grazing_reflectivity"), "enable_retroreflection"),
    **dict.fromkeys(("emissive_color", "emissive_mask_input", "emissive_mask_texture", "emissive_intensity",
                     "enable_emissive_flipbook", "flipbook_dim", "target_fps"), "enable_emission"),
    **dict.fromkeys(("alpha_constant", "opacity_texture", "enable_opacity_cutout", "alpha_cutout_cutoff"), "enable_opacity")}
OMNIPBR_OPACITY = {   # OmniPBR_Opacity.mdl: OmniPBR's parameters but its own tangent flips, round edges and roughness from
    # the normal map, opacity on by default; it calls OmniPBR_ClearCoat_Opacity.mdl, which is not on disk and is taken to
    # evaluate as OmniPBR_ClearCoat.mdl does, with that file's tangent flips (_resolve_reads)
    **{k: v for k, v in OMNIPBR.items() if k not in ("enable_opacity", "opacity_constant", "opacity_threshold", "flip_tangent_u",
                                                     "flip_tangent_v", "geometry_normal_roughness_strength", "round_edges_radius",
                                                     "round_edges_roundness", "round_edges_across_materials")},
    "enable_opacity": ("enable_opacity", True), "opacity_constant": ("opacity_constant", 1.0),
    "opacity_threshold": ("opacity_threshold", 0.0),
    "enable_normalmap_texture": (INERT, False), "enable_detail_normalmap_texture": (INERT, False),   # not passed on
}
OMNIPBR_OPACITY_GATES = {
    **{k: v for k, v in OMNIPBR_GATES.items() if v not in ("enable_opacity", "round_edges_radius")},
    **dict.fromkeys(("opacity_texture", "opacity_mode"), "enable_opacity_texture")}
OMNIGLASS = {   # OmniGlass.mdl, evaluated by OmniGlass_Opacity.mdl
    "absorption_coeff": (INERT, (0.0, 0.0, 0.0)), "use_glass_color_texture": (INERT, False),   # not passed on by OmniGlass.mdl
    "glass_color": ("diffuse_color", (1.0, 1.0, 1.0)), "glass_color_texture": ("diffuse_texture", None),
    "frosting_roughness": ("roughness", 0.0), "glass_ior": ("ior", 1.491), "thin_walled": ("thin_walled", False),
    # the absorption coefficient of the glass's volume: at its default it takes under 1 % over 10 cm of glass
    "depth": (None, 0.001),
    "roughness_texture_influence": ("roughness_texture_influence", 1.0), "roughness_texture": ("roughness_texture", None),
    "reflection_color_texture": (None, None), "reflection_color": (None, (1.0, 1.0, 1.0)),
    "normal_map_texture": ("normal_texture", None), "normal_map_strength": ("normal_factor", 1.0),
    "geometry_normal_roughness_strength": (INERT, 1.0),     # as OmniPBR's
    "flip_tangent_u": ("flip_tangent_u", False), "flip_tangent_v": ("flip_tangent_v", True),
    # a cutout opacity other than 1, enabled or not, makes the glass thin-walled (OmniGlass_Opacity.mdl, _resolve_reads)
    "enable_opacity": ("enable_opacity", False), "cutout_opacity": ("cutout_opacity", 1.0), "cutout_opacity_texture": (None, None),
    "cutout_opacity_mono_source": (None, 0), "opacity_threshold": ("opacity_threshold", 0.0),
    "project_uvw": ("project_uvw", False), "world_or_object": ("world_or_object", False), "uv_space_index": (None, 0),
    "texture_translate": ("texture_translate", (0.0, 0.0)), "texture_rotate": (None, 0.0),
    "texture_scale": ("texture_scale", (1.0, 1.0)),
}
OMNIGLASS_GATES = {"cutout_opacity_mono_source": "cutout_opacity_texture"}
CARPAINT = {   # vMaterials 2 Carpaint_Solid.mdl (NVIDIA), its own parameters; carried only while its flake layer does not show
    "base_color": ("cp_base_color", (0.002125, 0.051269, 0.046665)), "ground_coat_influence": ("cp_ground_coat_influence", 0.9),
    "ground_coat_brightness": ("cp_ground_coat_brightness", 0.24), "edge_darkening": ("cp_edge_darkening", 0.96),
    "flakes_tint": ("cp_flakes_tint", (0.016807, 0.171441, 0.417885)), "flakes_density": ("cp_flakes_density", 0.29),
    "spread": (INERT, 0.12), "flakes_roughness": (INERT, 0.46), "flakes_size": (INERT, 0.1),   # read by the flake layer only
    "candy_flip": ("cp_candy_flip", 0.0), "flake_layer_visibility": ("cp_flake_layer_visibility", 0.0),
    "roughness_texture": ("coat_roughness_texture", None), "clearcoat_rough": ("cp_clearcoat_rough", 0.0),
    "roughness_variation": ("cp_roughness_variation", 0.0), "clearcoat_ior": ("cp_clearcoat_ior", 1.6),
    "clearcoat_visibility": ("cp_clearcoat_visibility", 1.0), "orange_peel_strength": (None, 0.0), "orange_peel_size": (None, 1.0),
    "texture_translate": ("cp_texture_translate", (0.0, 0.0)), "texture_rotate": (None, 0.0), "texture_scale": ("cp_texture_scale", (1.0, 1.0)),
    "roundcorners_enable": (None, False), "roundcorners_radius_mm": (None, 1.5), "roundcorners_across_materials": (None, False),
    "uv_space_index": (None, 0),
}
CARPAINT_GATES = {"orange_peel_size": "orange_peel_strength", "roundcorners_radius_mm": "roundcorners_enable",
                  "roundcorners_across_materials": "roundcorners_enable"}
PREVIEW = {   # UsdPreviewSurface, its specification
    "diffuseColor": ("diffuse_color", (0.18, 0.18, 0.18)), "emissiveColor": (None, (0.0, 0.0, 0.0)),
    "useSpecularWorkflow": (None, 0), "specularColor": (None, (0.0, 0.0, 0.0)), "metallic": ("metallic", 0.0),
    "roughness": ("roughness", 0.5), "clearcoat": (None, 0.0), "clearcoatRoughness": (None, 0.01), "opacity": (None, 1.0),
    "opacityMode": (None, "transparent"), "opacityThreshold": (None, 0.0), "ior": ("ior", 1.5),
    "normal": (None, (0.0, 0.0, 1.0)), "displacement": (None, 0.0), "occlusion": (None, 1.0),
}
PREVIEW_GATES = {"specularColor": "useSpecularWorkflow", "clearcoatRoughness": "clearcoat", "opacityMode": "opacity",
                 "opacityThreshold": "opacity"}
# the UsdPreviewSurface inputs that may read a UsdUVTexture: the record key of its file and the outputs it may read
PREVIEW_TEXTURES = {"diffuseColor": ("diffuse_texture", ("rgb",)), "roughness": ("roughness_texture", ("r", "g", "b")),
                    "metallic": ("metallic_texture", ("r", "g", "b")), "normal": ("normal_texture", ("rgb",))}
MODELS = {"OmniPBR": (OMNIPBR, OMNIPBR_GATES), "OmniPBR_Opacity": (OMNIPBR_OPACITY, OMNIPBR_OPACITY_GATES),
          "SimPBR": (SIMPBR, SIMPBR_GATES), "OmniGlass": (OMNIGLASS, OMNIGLASS_GATES), "Carpaint_Solid": (CARPAINT, CARPAINT_GATES),
          "UsdPreviewSurface": (PREVIEW, PREVIEW_GATES)}
# what the render reads that a model has no parameter for, at the value that model's evaluation amounts to
NEUTRAL_ALBEDO = {"albedo_brightness": 1.0, "albedo_add": 0.0, "albedo_desaturation": 0.0}
TEXTURE_KEYS = ("diffuse_texture", "roughness_texture", "metallic_texture", "orm_texture", "normal_texture", "ao_texture")
# faces with no bound material (and no display colour): neutral grey, marked as such
UNBOUND = {"unbound": True, "diffuse_color": [0.5, 0.5, 0.5], "diffuse_tint": [1.0, 1.0, 1.0], **NEUTRAL_ALBEDO,
           "roughness": 0.5, "roughness_source": "constant", "metallic": 0.0, "metallic_source": "constant",
           "texture_combine": "replace", "ao_source": "one", "ao_to_diffuse": 0.0, "specular": 0.5, "ior": 1.5,
           "texture_scale": [1.0, 1.0], "texture_translate": [0.0, 0.0]}


def _shader_of(material):
    """The shader a renderer reads for the material's surface: its MDL surface (Omniverse), or without one its
    universal surface (a UsdPreviewSurface); None for a material without a surface."""
    sh = material.ComputeSurfaceSource("mdl")[0]
    return sh if sh else None


def _plain(v):
    """A USD or MDL value as plain Python: a vector as a tuple of floats; a number, bool or token as it is."""
    if isinstance(v, (bool, int, float, str)):
        return v
    return tuple(float(x) for x in v)


def _same(a, b):
    """Whether two parameter values are the same (numbers at float32, the precision USD and MDL store them in)."""
    if isinstance(a, tuple) or isinstance(b, tuple):
        return isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (bool, str, dict)) or isinstance(b, (bool, str, dict)) or a is None or b is None:
        return a == b
    return bool(np.float32(a) == np.float32(b))


def _depth_channels(path):
    """Bits per channel and number of channels of a PNG or JPEG file, from its header."""
    with open(path, "rb") as f:
        head = f.read(32)
    if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
        return (8, 3) if head[25] == 3 else (head[24], {0: 1, 2: 3, 4: 2, 6: 4}[head[25]])    # 3: a palette of 8-bit colours
    if head[:3] == b"\xff\xd8\xff":
        from PIL import Image
        with Image.open(path) as im:
            return 8, len(im.getbands())
    raise ValueError(f"{path}: an image format whose depth the export does not read")


def _colorspace(declared, path, rule):
    """How a texture file's values are decoded, "sRGB" or "raw": as declared (USD colorSpace sRGB or raw, an MDL
    texture_2d's gamma_srgb or gamma_linear), or for "auto" as the shader's definition decides it for the file: MDL
    (gamma_default) reads an 8-bit image as sRGB; UsdUVTexture reads an 8-bit image of 3 or 4 channels as sRGB and any
    other as raw. An MDL "auto" on another image, or another declaration, is refused."""
    if declared in ("sRGB", "raw"):
        return declared
    if declared != "auto":
        raise ValueError(f"{path}: colour space {declared!r} is not carried into the export")
    bits, channels = _depth_channels(path)
    if rule == "UsdUVTexture":
        return "sRGB" if bits == 8 and channels in (3, 4) else "raw"
    if bits == 8 and channels in (3, 4):
        return "sRGB"
    raise ValueError(f"{path}: an MDL texture of colour space auto on a {bits}-bit image of {channels} channels")


def _texture(asset, declared, rec, rule):
    """A texture parameter's value: {"file", "colorspace"} for a file on disk, None for no texture. A file it names
    that is not on disk is recorded as missing (fetch it) and taken as no texture."""
    if not asset.path:
        return None
    if not asset.resolvedPath or not os.path.isfile(asset.resolvedPath):
        rec.setdefault("missing", []).append(asset.path)
        return None
    return {"file": asset.resolvedPath, "colorspace": _colorspace(declared or "auto", asset.resolvedPath, rule)}


def _producing(inp):
    """The attribute that gives a shader input its value: its own, or through connections the material's interface
    input that does (UsdShade's value-producing attribute), or another shader's output; None when nothing does."""
    attrs = inp.GetValueProducingAttributes()
    if len(attrs) > 1:
        raise ValueError(f"{inp.GetAttr().GetPath()}: an input with more than one source is not carried into the export")
    return attrs[0] if attrs else None


def _authored(shader, rec, params):
    """parameter -> value of what an MDL shader authors, or its material's interface gives it (a texture as _texture
    gives it). An input that is no parameter of the model is not read (its value is None); a parameter that reads
    another shader is refused: the export reads values, not shading networks."""
    from pxr import Sdf, UsdShade
    out = {}
    for inp in shader.GetInputs():
        n, attr = inp.GetBaseName(), _producing(inp)
        if attr is None:
            continue
        if n not in params:
            out[n] = None
            continue
        if UsdShade.Output.IsOutput(attr):
            raise ValueError(f"{shader.GetPath()}: {n} reads another shader, which is not carried into the export")
        v = attr.Get()
        if v is not None:
            out[n] = _texture(v, attr.GetColorSpace(), rec, "MDL") if isinstance(v, Sdf.AssetPath) else _plain(v)
    return out


def _carry(rec, where, model, values):
    """Write the values of one shader's parameters into its record (`values`: what is authored, or set by a library
    material's call; every other parameter takes the model's default); refuse a value the render does not carry."""
    params, gates = MODELS[model]
    extra = sorted(n for n in values if n not in params)
    if extra:
        rec["not_parameters"] = extra
    full = {n: values.get(n, d) for n, (k, d) in params.items()}
    for n, (key, default) in params.items():
        if key == INERT or (n in gates and _same(full[gates[n]], params[gates[n]][1])):
            continue
        v = full[n]
        if key is None:
            if not _same(v, default):
                raise ValueError(f"{where}: {model} {n} = {v} is not carried into the export")
        elif isinstance(v, dict):                     # a texture on disk
            rec[key], rec[key + "_colorspace"] = v["file"], v["colorspace"]
        elif v is not None:
            rec[key] = list(v) if isinstance(v, tuple) else v
    rec["model_defaults"] = [n for n, (k, d) in params.items() if n not in values and k not in (None, INERT)]


def _mdl_call_args(txt):
    """The `name: value` arguments of the ::OmniPBR::OmniPBR( ... ) call in an .mdl text, as raw strings; None when
    the material does not call OmniPBR."""
    import re
    m = re.search(r"::OmniPBR::OmniPBR\s*\(", txt)
    return _call_args(txt, m.end()) if m else None


def _call_args(txt, start):
    """The `name: value` arguments of the call whose opening parenthesis ends at `start`, as raw strings (comments
    removed)."""
    import re
    i, depth = start, 1
    while depth:
        c = txt[i]
        depth += c == "("
        depth -= c == ")"
        i += 1
    body = re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", txt[start:i - 1], flags=re.S))
    args, depth, cur = {}, 0, ""
    for c in body + ",":
        if c == "," and depth == 0:
            if cur.strip():
                k, v = cur.split(":", 1)
                args[k.strip()] = v.strip()
            cur = ""
            continue
        depth += c == "("
        depth -= c == ")"
        cur += c
    return args


def _mdl_literal(raw, default, mdl_dir, rec):
    """One argument of a library material's OmniPBR call, as a value of its parameter's type (the parameter's default
    gives the type): texture_2d("file", ::tex::gamma_*) or texture_2d(), true or false, a number, or a constructor such
    as color(0.2f) or float2(1.f, 2.f), a single value standing for every component."""
    import re
    raw = raw.strip()
    if raw.startswith("texture_2d"):
        m = re.fullmatch(r'texture_2d\(\s*(?:"([^"]*)"\s*(?:,\s*::tex::(gamma_\w+)\s*)?)?\)', raw)
        if m is None:
            raise ValueError(f"the MDL texture {raw!r} is not read by the export")
        if not m.group(1):
            return None
        path = os.path.normpath(os.path.join(mdl_dir, m.group(1)))
        if not os.path.isfile(path):
            rec.setdefault("missing", []).append(path)
            return None
        gamma = {"gamma_srgb": "sRGB", "gamma_linear": "raw", "gamma_default": "auto", None: "auto"}[m.group(2)]
        return {"file": path, "colorspace": _colorspace(gamma, path, "MDL")}
    if isinstance(default, bool):
        if raw not in ("true", "false"):
            raise ValueError(f"the MDL literal {raw!r} is not true or false")
        return raw == "true"
    mono = re.fullmatch(r"(?:::)?base::mono_(alpha|average|luminance|maximum)", raw)
    if mono:
        return ("alpha", "average", "luminance", "maximum").index(mono.group(1))
    m = re.fullmatch(r"(?:color|float[234]?|int[234]?)\s*\((.*)\)", raw, flags=re.S)
    items = [x.strip() for x in (m.group(1) if m else raw).split(",")]
    if not all(re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?f?", x) for x in items):
        raise ValueError(f"the MDL literal {raw!r} is not a number or a constructor of numbers")
    vals = [float(x.rstrip("f")) for x in items]
    if isinstance(default, tuple):
        if len(vals) == 1:
            vals = vals * len(default)
        if len(vals) != len(default):
            raise ValueError(f"the MDL literal {raw!r} has {len(vals)} components, not {len(default)}")
        return tuple(vals)
    if len(vals) != 1:
        raise ValueError(f"the MDL literal {raw!r} is not one number")
    return int(vals[0]) if isinstance(default, int) else vals[0]


def _library_values(shader, rec):
    """A library material (Base/Wood/Oak.mdl, ...): an .mdl that declares `export material <name>(*)` as a call of
    OmniPBR, so that its parameters are OmniPBR's with the call's arguments as defaults. Returns those arguments as
    values, or None when the .mdl is not on disk or is not such a material."""
    import re
    src = shader.GetPrim().GetAttribute("info:mdl:sourceAsset")
    asset = src.Get() if src else None
    if asset is None:
        return None
    path = asset.resolvedPath
    if not path and "/Materials/Base/" in asset.path:       # the Omniverse base library by URL: RoboLab ships it locally
        path = os.path.join(ROBOLAB_MATERIALS, "Base", asset.path.split("/Materials/Base/", 1)[1])
    if not path or not path.endswith(".mdl") or not os.path.isfile(path):
        return None
    txt = open(path, errors="ignore").read()
    args = _mdl_call_args(txt)
    if args is None or not re.search(r"export\s+material\s+\w+\s*\(\s*\*\s*\)", txt):
        return None
    rec["mdl_file"] = path
    out = {}
    for name, raw in args.items():
        if name not in OMNIPBR:
            raise ValueError(f"{path}: {name} is no parameter of OmniPBR")
        out[name] = _mdl_literal(raw, OMNIPBR[name][1], os.path.dirname(path), rec)
    return out


def _carpaint_values(shader, rec):
    """A vMaterials 2 car paint: the material the shader names in Carpaint_Solid.mdl, the base material or one of the
    file's presets (`export material <name>(*) = Carpaint_Solid(...)`), as values over the base material's defaults
    (the default coat texture is the file's own), with the USD shader's inputs over them."""
    import re
    prim = shader.GetPrim()
    asset, sub = prim.GetAttribute("info:mdl:sourceAsset").Get(), prim.GetAttribute("info:mdl:sourceAsset:subIdentifier").Get()
    path = asset.resolvedPath
    if not path or not os.path.isfile(path):
        rec["missing"] = [asset.path]
        return {}
    txt, mdl_dir = open(path, errors="ignore").read(), os.path.dirname(path)
    sig = re.search(r"texture_2d\s+roughness_texture\s*=\s*(texture_2d\([^)]*\))", txt)
    out = {"roughness_texture": _mdl_literal(re.sub(r"/\*.*?\*/", "", sig.group(1)), None, mdl_dir, rec)}
    if sub != "Carpaint_Solid":
        m = re.search(r"export\s+material\s+" + re.escape(sub) + r"\s*\(\s*\*\s*\)\s*(?:\[\[.*?\]\])?\s*=\s*Carpaint_Solid\s*\(", txt, flags=re.S)
        if m is None:
            raise ValueError(f"{path}: no material {sub} that is a Carpaint_Solid")
        for name, raw in _call_args(txt, m.end()).items():
            if name not in CARPAINT:
                raise ValueError(f"{path}: {name} is no parameter of Carpaint_Solid")
            out[name] = _mdl_literal(raw, CARPAINT[name][1], mdl_dir, rec)
    rec["mdl_file"], rec["preset"] = path, sub
    return {**out, **_authored(shader, rec, CARPAINT)}


GLTF_READ = {"base_color_factor", "base_color_texture", "metallic_factor", "roughness_factor", "alpha_mode", "alpha_cutoff", "base_alpha",
             "emissive_strength"}


def _read_gltf(rec, shader):
    """NVIDIA's gltf/pbr.mdl gltf_material, read as the glTF 2.0 metallic-roughness model: base colour = factor x the
    base colour texture (sRGB, through a gltf_texture_lookup: texture coordinate set 0, uv * scale + offset, no rotation,
    repeating), metallic and roughness their factors, opaque. Its alpha settings are not read in opaque mode, nor its
    emissive strength without an emissive colour; any other authored input is refused."""
    from pxr import UsdShade
    for inp in shader.GetInputs():
        n, attr = inp.GetBaseName(), _producing(inp)
        if attr is None:
            continue
        if n not in GLTF_READ:
            raise ValueError(f"{shader.GetPath()}: glTF {n} is not carried into the export")
        if n == "base_color_texture":
            if not UsdShade.Output.IsOutput(attr):
                raise ValueError(f"{shader.GetPath()}: a base colour texture that no texture lookup gives")
            look = UsdShade.Shader(attr.GetPrim())
            sub = look.GetPrim().GetAttribute("info:mdl:sourceAsset:subIdentifier")
            if not sub or sub.Get() != "gltf_texture_lookup":
                raise ValueError(f"{look.GetPath()}: not a gltf_texture_lookup")
            val = {i.GetBaseName(): (_producing(i).Get() if _producing(i) is not None else None) for i in look.GetInputs()}
            if (val.get("tex_coord_index") or 0) != 0 or (val.get("rotation") or 0.0) != 0.0 or {val.get("wrap_s", 10497), val.get("wrap_t", 10497)} != {10497}:
                raise ValueError(f"{look.GetPath()}: a texture lookup other than set 0, unrotated, repeating, is not carried into the export")
            t = _texture(val["texture"], "sRGB", rec, "MDL") if val.get("texture") is not None else None
            if t is not None:
                rec["diffuse_texture"], rec["diffuse_texture_colorspace"] = t["file"], t["colorspace"]
            rec["texture_scale"] = [float(x) for x in (val.get("scale") or (1.0, 1.0))]
            rec["texture_translate"] = [float(x) for x in (val.get("offset") or (0.0, 0.0))]
            continue
        v = attr.Get()
        if n == "alpha_mode" and int(v) != 0:
            raise ValueError(f"{shader.GetPath()}: glTF alpha mode {v} (not opaque) is not carried into the export")
        if n == "base_color_factor":
            rec["diffuse_tint"] = [float(x) for x in v]
        elif n in ("metallic_factor", "roughness_factor"):
            rec[n.split("_")[0]] = float(v)
    rec.setdefault("diffuse_tint", [1.0, 1.0, 1.0]); rec.setdefault("metallic", 1.0); rec.setdefault("roughness", 1.0)   # glTF defaults
    if not rec.get("diffuse_texture"):
        rec["diffuse_color"] = rec.pop("diffuse_tint"); rec["diffuse_tint"] = [1.0, 1.0, 1.0]
    rec.setdefault("texture_scale", [1.0, 1.0]); rec.setdefault("texture_translate", [0.0, 0.0])


def _model_of(shader):
    """The model of a shader: a name of MODELS when its definition is one the export carries (UsdPreviewSurface, or the
    MDL file and material of OmniPBR, OmniPBR_Opacity, SimPBR or OmniGlass), "library" for another MDL material, else
    None."""
    if shader.GetIdAttr().Get() == "UsdPreviewSurface":
        return "UsdPreviewSurface"
    prim = shader.GetPrim()
    sub = prim.GetAttribute("info:mdl:sourceAsset:subIdentifier")
    src = prim.GetAttribute("info:mdl:sourceAsset")
    path = src.Get().path if src and src.Get() is not None else ""
    sub = sub.Get() if sub else None
    if sub in ("OmniPBR", "OmniPBR_Opacity", "SimPBR", "OmniGlass") and os.path.basename(path) == sub + ".mdl":
        return sub
    if sub == "gltf_material" and path.endswith("gltf/pbr.mdl"):
        return "gltf_material"
    if os.path.basename(path) == "Carpaint_Solid.mdl":
        return "Carpaint_Solid"                # its base material or a preset (the subIdentifier)
    return "library" if path.endswith(".mdl") else None


def _preview_coordinates(tex):
    """(scale, translation) of a UsdUVTexture's coordinates: the primvar st, read directly or through a UsdTransform2d
    (out = in * scale + translation); a rotation, another primvar or coordinates not read from a primvar are refused."""
    from pxr import UsdShade

    def value(node, name, default):
        i = node.GetInput(name)
        return _plain(i.Get()) if i and i.Get() is not None else default

    st = tex.GetInput("st")
    if not (st and st.HasConnectedSource()):
        raise ValueError(f"{tex.GetPath()}: texture coordinates not read from a primvar are not carried into the export")
    node = UsdShade.Shader(st.GetConnectedSource()[0].GetPrim())
    scale, translation = (1.0, 1.0), (0.0, 0.0)
    if node.GetIdAttr().Get() == "UsdTransform2d":
        if not _same(value(node, "rotation", 0.0), 0.0):
            raise ValueError(f"{node.GetPath()}: a rotated texture is not carried into the export")
        scale, translation = value(node, "scale", (1.0, 1.0)), value(node, "translation", (0.0, 0.0))
        inner = node.GetInput("in")
        if not (inner and inner.HasConnectedSource()):
            raise ValueError(f"{node.GetPath()}: texture coordinates not read from a primvar are not carried into the export")
        node = UsdShade.Shader(inner.GetConnectedSource()[0].GetPrim())
    if node.GetIdAttr().Get() != "UsdPrimvarReader_float2" or str(value(node, "varname", "")) != "st":
        raise ValueError(f"{node.GetPath()}: texture coordinates other than the primvar st are not carried into the export")
    return tuple(scale), tuple(translation)


def _read_preview(rec, shader):
    """A UsdPreviewSurface: its constants, and for diffuseColor, roughness, metallic and normal the UsdUVTexture each
    may read: its file, the output read, how the file is decoded, and the transform of its coordinates (one for all of a
    material's textures). A texture's scale and bias must be the identity (for the normal: 2 and -1, which make a
    tangent-space normal of the file's values), and it must repeat."""
    from pxr import UsdShade
    values, coords = {}, set()
    for inp in shader.GetInputs():
        n, attr = inp.GetBaseName(), _producing(inp)
        if attr is None or n not in PREVIEW:
            if attr is not None:
                values[n] = None                    # no input of UsdPreviewSurface: not read
            continue
        if not UsdShade.Output.IsOutput(attr):
            if attr.Get() is not None:
                values[n] = _plain(attr.Get())
            continue
        if n not in PREVIEW_TEXTURES:
            raise ValueError(f"{shader.GetPath()}: {n} reads another shader, which is not carried into the export")
        key, outputs = PREVIEW_TEXTURES[n]
        out = UsdShade.Output(attr).GetBaseName()
        tex = UsdShade.Shader(attr.GetPrim())
        if tex.GetIdAttr().Get() != "UsdUVTexture" or out not in outputs:
            raise ValueError(f"{shader.GetPath()}: {n} reads {out} of {tex.GetPath()}, which is not carried into the export")

        def value(name, default):
            i = tex.GetInput(name)
            a_ = _producing(i) if i else None
            if a_ is not None and UsdShade.Output.IsOutput(a_):
                raise ValueError(f"{tex.GetPath()}: {name} reads another shader, which is not carried into the export")
            return _plain(a_.Get()) if a_ is not None and a_.Get() is not None else default

        identity = ((2.0, 2.0, 2.0), (-1.0, -1.0, -1.0)) if n == "normal" else ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
        if not (_same(value("scale", (1.0, 1.0, 1.0, 1.0))[:3], identity[0]) and _same(value("bias", (0.0, 0.0, 0.0, 0.0))[:3], identity[1])):
            raise ValueError(f"{tex.GetPath()}: a scale or bias of the texture is not carried into the export")
        if value("wrapS", "useMetadata") != "repeat" or value("wrapT", "useMetadata") != "repeat":
            raise ValueError(f"{tex.GetPath()}: a texture that does not repeat is not carried into the export")
        fa = _producing(tex.GetInput("file")) if tex.GetInput("file") else None
        f = fa.Get() if fa is not None and not UsdShade.Output.IsOutput(fa) else None
        if f is None or not f.path:
            raise ValueError(f"{tex.GetPath()}: a texture without a file")
        t = _texture(f, value("sourceColorSpace", "auto"), rec, "UsdUVTexture")
        if t is not None:
            rec[key], rec[key + "_colorspace"] = t["file"], t["colorspace"]
            if n in ("roughness", "metallic"):
                rec[n + "_source"] = "texture_" + out
        coords.add(_preview_coordinates(tex))
    _carry(rec, shader.GetPath(), "UsdPreviewSurface", values)
    if len(coords) > 1:
        raise ValueError(f"{shader.GetPath()}: textures with different coordinate transforms are not carried into the export")
    scale, translation = coords.pop() if coords else ((1.0, 1.0), (0.0, 0.0))
    rec["texture_scale"], rec["texture_translate"] = list(scale), list(translation)


def _resolve_reads(rec, model):
    """What the render reads that the model derives from its parameters, as the model's evaluation defines it.
    `<key>_source` says where roughness, metallic and occlusion come from: "orm_r", "orm_g", "orm_b" (a channel of the
    ORM texture), "texture_r", "texture_g", "texture_b" (a channel of the key's texture), "texture_max" (the largest of
    its channels), "texture_rgb" (its colour: occlusion), "zero" or "one" (what a lookup of no texture gives), or
    "constant" (no lookup: the constant itself). `texture_combine` says how a lookup meets the constant: "lerp" by the
    key's texture influence (OmniPBR_ClearCoat.mdl, SimPBR.mdl, OmniGlass_Opacity.mdl) or "replace" (UsdPreviewSurface).
    `specular` is Blender's IOR level and `ior` its IOR: OmniPBR's F0 is 0.08 x specular_level, SimPBR's 0.03 x
    specular_constant (SimPBR_Model.mdl), both at IOR 1.5; glass and UsdPreviewSurface take F0 from their IOR.
    `normal_formula` names how a normal map's values make the normal: "scale_xy" (OmniPBRBase.mdl and
    UsdPreviewSurface: the tangent components times the factor) or "base_mdl" (base.mdl's tangent_space_normal_texture:
    every component times the factor, the normal component |z| + 1 - factor). Glass is thin-walled (transmission
    without refraction) when it says so or its cutout opacity is not 1, and `opacity` is the cutout it renders with."""
    if model == "OmniPBR_Opacity":
        on, value, threshold = rec.pop("enable_opacity"), rec.pop("opacity_constant"), rec.pop("opacity_threshold")
        cutout = (value if threshold == 0 else float(value >= threshold)) if on else 1.0      # as OmniPBR_ClearCoat.mdl
        if cutout != 1.0:
            raise ValueError(f"{rec['prim']}: OmniPBR_Opacity cuts the surface out (opacity {cutout}), not carried into the export")
        rec["flip_tangent_u"], rec["flip_tangent_v"] = False, True
        rec["evaluated_as"] = "OmniPBR_ClearCoat.mdl 2.1.1 (OmniPBR_ClearCoat_Opacity.mdl, which OmniPBR_Opacity.mdl calls, is not on disk)"
    if model in ("OmniPBR", "OmniPBR_Opacity", "SimPBR"):
        absent = "constant" if model == "SimPBR" else "zero"      # a lookup of no texture: OmniPBR 0, SimPBR the constant
        orm = bool(rec["enable_orm"])
        for key, ch in (("roughness", "g"), ("metallic", "b")):
            if orm:
                rec[key + "_source"] = "orm_" + ch if rec.get("orm_texture") else "zero"
            else:
                rec[key + "_source"] = "texture_r" if rec.get(key + "_texture") else absent
        if orm:
            rec["ao_source"] = "orm_r" if rec.get("orm_texture") else "zero"
        elif model != "SimPBR":
            rec["ao_source"] = "texture_rgb" if rec.get("ao_texture") else "zero"
        else:
            rec["ao_source"] = "texture_r" if rec.get("ao_texture") else "constant"
        rec["texture_combine"], rec["ior"] = "lerp", 1.5
        if model != "SimPBR":
            rec["specular"], rec["normal_formula"] = rec["specular_level"], "scale_xy"
        else:
            rec["specular"], rec["normal_formula"] = 0.375 * rec["specular_constant"], "base_mdl"
            rec.update(NEUTRAL_ALBEDO)
    elif model == "OmniGlass":
        rec["roughness_source"] = "texture_max" if rec.get("roughness_texture") else "constant"   # get_float: mono_maximum
        on, value, threshold = rec.pop("enable_opacity"), rec.pop("cutout_opacity"), rec.pop("opacity_threshold")
        rec["thin_walled"] = bool(rec["thin_walled"]) or value != 1.0
        rec["opacity"] = (value if threshold == 0 else float(value >= threshold)) if on else 1.0
        rec.update(NEUTRAL_ALBEDO, diffuse_tint=[1.0, 1.0, 1.0], metallic=0.0, metallic_source="constant", ao_source="one",
                   ao_to_diffuse=0.0, texture_combine="lerp", specular=0.5, glass=True, normal_formula="scale_xy")
    elif model == "gltf_material":                                # F0 0.04 (IOR 1.5); no occlusion
        rec.update(NEUTRAL_ALBEDO, roughness_source="constant", metallic_source="constant", texture_combine="replace", ao_source="one",
                   ao_to_diffuse=0.0, specular=0.5, ior=1.5)
    elif model == "Carpaint_Solid":
        v = lambda k: rec.pop("cp_" + k)
        if v("flakes_density") > 0 and v("flake_layer_visibility") != 0:
            raise ValueError(f"{rec['prim']}: a car paint whose flake layer shows is not carried into the export")
        base, edge, flip, flakes = v("base_color"), v("edge_darkening"), v("candy_flip"), v("flakes_tint")
        rough, var = v("clearcoat_rough"), v("roughness_variation")       # histogram_range(texture, var, rough)
        low = min(max(1 - min((1 - rough) + 0.5 * var, 2 * (1 - rough)), 0.0), 1.0)
        high = min(max(min(rough + 0.5 * var, 2 * rough), 0.0), 1.0)
        gc = 0.95 * v("ground_coat_brightness") + 0.05
        rec["carpaint"] = {"base_color": list(base), "ground_coat": {"weight": 0.5 * v("ground_coat_influence"), "tint": [gc] * 3,
                                                                     "roughness": 0.64},                 # GGX alpha 0.4096
                           "base_tint": {"normal": list(base), "grazing": [1 - edge] * 3, "exponent": 1.0},
                           "outer_tint": {"normal": [1.0] * 3, "grazing": [1 - edge] * 3, "exponent": 1.0},
                           "coat": {"ior": v("clearcoat_ior"), "weight": v("clearcoat_visibility"), "roughness_range": [low, high],
                                    "tint": [(1 - flip) + flip * t for t in flakes]}}
        scale = v("texture_scale")
        rec["texture_scale"], rec["texture_translate"] = [1.0 / x for x in scale], list(v("texture_translate"))   # vmat_transform divides
        rec["evaluated_as"] = ("an approximation, as chosen: the layers, weights, colours, IOR and the coat's roughness map as "
                               "Carpaint_Solid.mdl has them; its simple_glossy_bsdf, microfacet_ggx_vcavities_bsdf and "
                               "directional_factor as Blender's GGX, Fresnel and facing ratio, which have no exact equivalent")
    else:                                                          # UsdPreviewSurface
        for key in ("roughness", "metallic"):
            rec.setdefault(key + "_source", "constant")
        rec.update(NEUTRAL_ALBEDO, diffuse_tint=[1.0, 1.0, 1.0], ao_source="one", ao_to_diffuse=0.0, texture_combine="replace",
                   specular=0.5, normal_formula="scale_xy", normal_factor=1.0, flip_tangent_u=False, flip_tangent_v=False)
    if not rec.get("normal_texture"):
        for key in ("normal_formula", "normal_factor", "flip_tangent_u", "flip_tangent_v"):
            rec.pop(key, None)


def _material_record(material, layer_dir, cache):
    key = material.GetPath().pathString
    if key in cache:
        return key
    rec = {"name": key.strip("/").replace("/", "_"), "prim": key}
    sh = _shader_of(material)
    model = _model_of(sh) if sh is not None else None
    if sh is not None:
        src = sh.GetPrim().GetAttribute("info:mdl:sourceAsset")
        rec["mdl"] = src.Get().path if src and src.Get() is not None else None
    if model == "UsdPreviewSurface":
        _read_preview(rec, sh)
    elif model == "Carpaint_Solid":
        _carry(rec, sh.GetPath(), model, _carpaint_values(sh, rec))
    elif model == "gltf_material":
        _read_gltf(rec, sh)
    elif model in MODELS:
        _carry(rec, sh.GetPath(), model, _authored(sh, rec, MODELS[model][0]))
    elif model == "library":
        lib = _library_values(sh, rec)
        if lib is None:
            model = None
        else:                             # the USD shader's inputs are OmniPBR's parameters, over the library's arguments
            model = "OmniPBR"
            _carry(rec, sh.GetPath(), model, {**lib, **_authored(sh, rec, OMNIPBR)})
    if model is None:
        rec["unknown"] = f"a shader the export does not carry ({rec.get('mdl')})"
    else:
        rec["model"] = model
        _resolve_reads(rec, model)
    if rec.get("project_uvw"):            # _gather writes the projection's texture coordinates (_cubic_uv)
        rec["uv_projected"] = "cubic, " + ("world" if rec.get("world_or_object") else "object") + " space"
    cache[key] = rec
    return key


def _cubic_uv(P):
    """MDL's cubic projection (base.mdl, coordinate_projection with projection_cubic, compute_cubic_transform) of one
    triangle P (3 x 3, in the projection's space): the texture coordinates of its corners. The face's geometric normal,
    turned to point away from the origin, picks the plane: x -> (y, z), -x -> (-y, z), y -> (-x, z), -y -> (x, z),
    z -> (x, y), -z -> (-x, y), x before y before z on a tie. MDL picks the plane at every shading point from the
    shading normal; one plane per face is exact for flat faces."""
    n = np.cross(P[1] - P[0], P[2] - P[0])
    c = P.mean(0)
    if c @ n < min(float(np.linalg.norm(c)) * 1e-9, 1e-5):
        n = -n
    a = np.abs(n)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    if a[0] >= a[1] and a[0] >= a[2]:
        uv = (y, z) if n[0] > 0 else (-y, z)
    elif a[1] >= a[0] and a[1] >= a[2]:
        uv = (-x, z) if n[1] > 0 else (x, z)
    else:
        uv = (-x, y) if n[2] < 0 else (x, y)
    return np.stack(uv, 1)


def _display_colour_record(prim, mats):
    """A prim without a material that authors one display colour: a record of that colour (roughness and metallic as
    UNBOUND). A colour that varies over the prim is refused; without one the prim's faces are unbound (None)."""
    from pxr import UsdGeom
    pv = UsdGeom.Gprim(prim).GetDisplayColorPrimvar()
    if not pv.HasAuthoredValue():
        return None
    vals = pv.ComputeFlattened()
    if pv.GetInterpolation() != "constant" or len(vals) != 1:
        raise ValueError(f"{prim.GetPath()}: a display colour that varies over the prim is not carried into the export")
    key = prim.GetPath().pathString + ".displayColor"
    mats[key] = dict(UNBOUND, unbound=False, display_color=True, prim=prim.GetPath().pathString,
                     name=key.strip("/").replace("/", "_"), diffuse_color=[float(x) for x in vals[0]])
    return key


def _gather(root, xcache, usd_path, to_local=None):
    """Collect triangles, uvs and per-face materials below `root`, in world coordinates of its
    stage or, with to_local=(Rc, tc), in the frame Rc^T (x - tc) (the scene child's frame with
    its scale baked in, the frame simready.io.usd_io.load_scene_usda starts a body mesh from)."""
    from pxr import Usd, UsdGeom, UsdShade
    V, VT, F, FT, FM = [], [], [], [], []      # vertices, uvs, tri vertex idx, tri uv idx (or -1), tri material key
    mats = {}
    v_off = vt_off = 0
    n_prims = 0
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if not _is_geom(prim) or UsdGeom.Imageable(prim).GetPurposeAttr().Get() == "guide":
            continue
        layer_dir = os.path.dirname(prim.GetPrimStack()[0].layer.realPath or usd_path) if prim.GetPrimStack() else os.path.dirname(usd_path)
        Rm, tm = _matrix_np(xcache.GetLocalToWorldTransform(prim))
        bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        mesh_mat = _material_record(bound, layer_dir, mats) if bound and bound.GetPrim().IsValid() else None
        if prim.GetTypeName() == "Mesh":
            m = UsdGeom.Mesh(prim)
            pts = np.asarray(m.GetPointsAttr().Get() or [], dtype=np.float64)
            counts = list(m.GetFaceVertexCountsAttr().Get() or [])
            idx = list(m.GetFaceVertexIndicesAttr().Get() or [])
            if len(pts) == 0 or not counts:
                continue
            # uv
            uv_vals, uv_mode = None, None
            pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
            if pv and pv.HasValue():
                flat = np.asarray(pv.ComputeFlattened(), dtype=np.float64)
                interp = pv.GetInterpolation()
                if interp == "faceVarying" and len(flat) == len(idx):
                    uv_vals, uv_mode = flat, "fv"
                elif interp in ("vertex", "varying") and len(flat) == len(pts):
                    uv_vals, uv_mode = flat, "vtx"
                else:
                    raise ValueError(f"{prim.GetPath()}: texture coordinates with {interp} interpolation ({len(flat)} values for "
                                     f"{len(pts)} points, {len(idx)} face corners) are not carried into the export")
            # per-face material from GeomSubsets
            face_mat = {}
            for sub in UsdGeom.Subset.GetAllGeomSubsets(m):
                if sub.GetFamilyNameAttr().Get() not in (None, "", "materialBind"):
                    continue
                b = UsdShade.MaterialBindingAPI(sub.GetPrim()).ComputeBoundMaterial()[0]
                if not (b and b.GetPrim().IsValid()):
                    continue
                k = _material_record(b, layer_dir, mats)
                for fi in (sub.GetIndicesAttr().Get() or []):
                    face_mat[int(fi)] = k
            if mesh_mat is None and len(face_mat) < len(counts):
                mesh_mat = _display_colour_record(prim, mats)      # faces no subset binds show the prim's display colour
            world = (Rm @ pts.T).T + tm
            space = {False: pts, True: world}                  # a projection's object or world space
            if to_local is not None:
                Rc, tc = to_local; world = (Rc.T @ (world - tc).T).T
            V.append(world)
            if uv_vals is not None:
                VT.append(uv_vals)
            vt_end = vt_off + (len(uv_vals) if uv_vals is not None else 0)
            proj = []                                           # texture coordinates of projected faces
            k0 = 0
            for fi, c in enumerate(counts):
                poly = idx[k0:k0 + c]
                mk = face_mat.get(fi, mesh_mat)
                rec = mats.get(mk) if mk is not None else None
                for j in range(1, c - 1):
                    tri = (poly[0], poly[j], poly[j + 1])
                    F.append([t + v_off for t in tri])
                    if rec is not None and rec.get("project_uvw"):
                        proj.append(_cubic_uv(space[bool(rec.get("world_or_object"))][list(tri)]))
                        FT.append([vt_end + 3 * (len(proj) - 1) + q for q in range(3)])
                    elif uv_mode == "fv":
                        FT.append([k0 + 0 + vt_off, k0 + j + vt_off, k0 + j + 1 + vt_off])
                    elif uv_mode == "vtx":
                        FT.append([t + vt_off for t in tri])
                    elif rec is not None and any(rec.get(t) for t in TEXTURE_KEYS):
                        raise ValueError(f"{prim.GetPath()}: material {rec['prim']} has textures, the faces no texture coordinates")
                    else:
                        FT.append([-1, -1, -1])
                    FM.append(mk)
                k0 += c
            if proj:
                VT.append(np.vstack(proj))
            v_off += len(pts)
            vt_off = vt_end + 3 * len(proj)
        else:
            got = _gprim_mesh(prim)
            if got is None:
                continue
            pts, tri = got
            if mesh_mat is None:
                mesh_mat = _display_colour_record(prim, mats)
            world = (Rm @ pts.T).T + tm
            space = {False: pts, True: world}
            if to_local is not None:
                Rc, tc = to_local; world = (Rc.T @ (world - tc).T).T
            V.append(world)
            rec = mats.get(mesh_mat) if mesh_mat is not None else None
            proj = []
            for t in tri:
                F.append([int(x) + v_off for x in t]); FM.append(mesh_mat)
                if rec is not None and rec.get("project_uvw"):
                    proj.append(_cubic_uv(space[bool(rec.get("world_or_object"))][[int(x) for x in t]]))
                    FT.append([vt_off + 3 * (len(proj) - 1) + q for q in range(3)])
                elif rec is not None and any(rec.get(k) for k in TEXTURE_KEYS):
                    raise ValueError(f"{prim.GetPath()}: material {rec['prim']} has textures, the faces no texture coordinates")
                else:
                    FT.append([-1, -1, -1])
            if proj:
                VT.append(np.vstack(proj))
                vt_off += 3 * len(proj)
            v_off += len(pts)
        n_prims += 1
    if not V:
        return None
    V = np.vstack(V); VT = np.vstack(VT) if VT else np.zeros((0, 2))
    return V, VT, F, FT, FM, mats, n_prims


def _write(out_dir, name, usd_path, gathered):
    V, VT, F, FT, FM, mats, n_prims = gathered
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    obj_path = out_dir / f"{name}.obj"
    short = {mk: f"{name}_m{i}" for i, mk in enumerate(mats)}       # Blender keeps 63 bytes of a name; USD paths are longer
    if any(len(v.encode()) > 63 for v in short.values()):
        raise ValueError(f"the material names of {name} are longer than Blender keeps")
    with open(obj_path, "w") as f:
        f.write(f"# exported from {usd_path}\nmtllib {name}.mtl\n")
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in VT:
            f.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
        # group faces by material so `usemtl` switches are few
        order = sorted(range(len(F)), key=lambda i: (FM[i] or ""))
        cur = object()
        for i in order:
            mk = FM[i]
            if mk != cur:
                f.write(f"usemtl {short[mk] if mk in mats else 'default'}\n"); cur = mk
            a, b, c = F[i]; ta, tb, tc = FT[i]
            if ta >= 0:
                f.write(f"f {a+1}/{ta+1} {b+1}/{tb+1} {c+1}/{tc+1}\n")
            else:
                f.write(f"f {a+1} {b+1} {c+1}\n")
    records = {short[mk]: dict(rec, name=short[mk]) for mk, rec in mats.items()}
    if any(mk not in mats for mk in FM):
        records["default"] = dict(UNBOUND, name="default")
    with open(out_dir / f"{name}.mtl", "w") as f:
        for n, rec in records.items():
            col = rec.get("diffuse_color") or [1.0, 1.0, 1.0]
            f.write(f"newmtl {n}\nKd {col[0]:.4f} {col[1]:.4f} {col[2]:.4f}\n")
            if rec.get("diffuse_texture"):
                f.write(f"map_Kd {rec['diffuse_texture']}\n")
    info = {"usd": usd_path, "name": name, "n_prims": n_prims, "n_verts": int(len(V)), "n_tris": int(len(F)),
            "n_uv": int(len(VT)), "aabb": [V.min(0).tolist(), V.max(0).tolist()], "materials": records}
    json.dump(info, open(out_dir / f"{name}.materials.json", "w"), indent=1)
    check_renderable(info)
    return info


def check_renderable(info: dict) -> None:
    """Raise unless every material of an export says what its surface looks like: a texture it names must be on disk
    (fetch it), and a material whose shader and inputs say nothing (an unsupported shader) is refused. The export's
    files are written first, so the materials JSON shows what is missing."""
    bad = {n: m.get("missing") or m["unknown"] for n, m in info["materials"].items() if m.get("missing") or m.get("unknown")}
    if bad:
        raise RuntimeError(f"{info['name']} ({info['usd']}): materials that cannot be rendered as authored: {bad}")


def _rotate_gathered(g, R):
    """The gathered geometry with its vertices turned by R (a resting-frame rotation)."""
    if R is None:
        return g
    V, VT, F, FT, FM, mats, n_prims = g
    return (np.asarray(V) @ np.asarray(R, dtype=np.float64).T, VT, F, FT, FM, mats, n_prims)


def export(usd_path, out_dir, name=None, rotation=None):
    """One object/fixture USD in its own stage frame (matches load_object_usd); `rotation`
    turns the mesh into the body's resting frame (matches body_from_object_usd)."""
    from pxr import Usd, UsdGeom
    usd_path = str(usd_path)
    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()
    g = _gather(root, UsdGeom.XformCache(), usd_path)
    if g is None:
        raise ValueError(f"no geometry in {usd_path}")
    return _write(out_dir, name or root.GetName(), usd_path, _rotate_gathered(g, rotation))


def export_scene(usda_path, out_dir, world_prim=None):
    """Every child of the scene's world prim as its own OBJ in the body frame load_scene_usda gives
    it (its asset's frame, then its resting frame) with the child's scale baked in; returns
    {child_name: info} with `c_model` (AABB centre of the local mesh), `center` and `rotation`
    matching load_scene_usda's Body for that child."""
    from pxr import Usd, UsdGeom
    from simready.io.usd_io import _rotation_from_affine, asset_frame_rotation, child_source
    from simready.io.asset_rest import rest_rotation
    usda_path = str(usda_path)
    stage = Usd.Stage.Open(usda_path)
    root = stage.GetPrimAtPath(world_prim) if world_prim else stage.GetDefaultPrim()
    if not root or not root.IsValid():
        root = stage.GetPseudoRoot().GetChildren()[0]
    xcache = UsdGeom.XformCache()
    out = {}
    for child in root.GetChildren():
        if not child.IsA(UsdGeom.Xformable) and not child.GetChildren():
            continue
        Rc_aff, tc = _matrix_np(xcache.GetLocalToWorldTransform(child))
        Rc = _rotation_from_affine(Rc_aff)
        g = _gather(child, xcache, usda_path, to_local=(Rc, tc))
        if g is None:
            continue
        srcpath, payload_targets = child_source(child, usda_path)
        R_frame = asset_frame_rotation(child)
        if R_frame is not None:                # same asset frame as load_scene_usda
            g = _rotate_gathered(g, R_frame)
            Rc = Rc @ R_frame.T
        R_rest = rest_rotation(srcpath or child.GetName()) if payload_targets else rest_rotation(child.GetName())
        if R_rest is not None:                 # same resting frame as load_scene_usda
            g = _rotate_gathered(g, R_rest)
            Rc = Rc @ R_rest.T
        info = _write(out_dir, child.GetName(), usda_path, g)
        lo, hi = np.asarray(info["aabb"])
        c_model = 0.5 * (lo + hi)
        info.update({"c_model": c_model.tolist(), "center": (tc + Rc @ c_model).tolist(), "rotation": Rc.tolist(), "prim": str(child.GetPath())})
        json.dump(info, open(Path(out_dir) / f"{child.GetName()}.materials.json", "w"), indent=1)
        out[child.GetName()] = info
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("usd"); ap.add_argument("out_dir"); ap.add_argument("--name"); ap.add_argument("--scene", action="store_true")
    a = ap.parse_args()
    if a.scene:
        res = export_scene(a.usd, a.out_dir)
        for k, info in res.items():
            print(f"{k:22s} tris={info['n_tris']:7d} uv={info['n_uv']:7d} mats={len(info['materials'])} tex={sum(1 for m in info['materials'].values() if m.get('diffuse_texture'))} aabb={np.round(info['aabb'], 3).tolist()}")
    else:
        info = export(a.usd, a.out_dir, a.name)
        print(json.dumps({k: v for k, v in info.items() if k != "materials"}), "\nmaterials:", len(info["materials"]),
              [(m["name"], bool(m.get("diffuse_texture")), m.get("diffuse_color")) for m in info["materials"].values()][:6])

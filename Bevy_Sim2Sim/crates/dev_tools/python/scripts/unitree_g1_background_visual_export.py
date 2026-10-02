#!/usr/bin/env python3
"""Export a bounded original static-task background for render comparison only.

The original background pose, startup deactivations and native environment
translation are explicit. Geometry and textures retain their source identities.
Only authored basic PBR inputs are mapped; MDL shader/default/effect and lighting
parity are not claimed. This does not add background physics or run a simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

from unitree_g1_task_asset_query import digest
from unitree_g1_task_visual_export import geometry

USD_SHA = "7e14dcfd948591b8fdae61d41b412097b39490022dfc25aab9b90b6884509051"
WAREHOUSE_MDL_SHA = "2e2aef644544b619f5df253eacbaed941be89bc1018fb58cbb47d0e8ac1d4192"
DEACTIVATE = ("BackgroundAssets/boxes/jetson_orin_06", "BackgroundAssets/boxes/jetson_orin_03",
              "BackgroundAssets/boxes/hesai_box_06")


def display_color_surface(prim):
    """Retain authored uniform Gprim colors on otherwise unbound surfaces.

    No replacement color is inferred from screenshots. Nonuniform colors,
    procedural shaders and translucent primvars remain unsupported.
    """
    color = UsdGeom.Gprim(prim).GetDisplayColorPrimvar()
    values = color.ComputeFlattened() if color and color.HasAuthoredValue() else None
    if not values:
        raise ValueError(f"Unbound background surface has no authored displayColor: {prim.GetPath()}")
    first = tuple(map(float, values[0]))
    if any(tuple(map(float, value)) != first for value in values):
        raise ValueError("Nonuniform unbound displayColor needs vertex-color mapping")
    if any(not 0 <= value <= 1 for value in first):
        raise ValueError("Unbound displayColor exceeds its finite color bound")
    color_space = Usd.ColorSpaceAPI.ComputeColorSpaceName(color.GetAttr(), None)
    if color_space not in ('', 'lin_rec709_scene', 'lin_rec709'):
        raise ValueError(f'Unbound authored color space needs explicit conversion: {color_space}')
    interpolation = color.GetInterpolation()
    expected = {'constant': 1, 'vertex': len(UsdGeom.Mesh(prim).GetPointsAttr().Get() or []),
                'uniform': len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get() or []),
                'faceVarying': len(UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get() or [])}.get(interpolation)
    if expected != len(values):
        raise ValueError('Unbound authored displayColor interpolation/count mismatch')
    opacity = UsdGeom.Gprim(prim).GetDisplayOpacityPrimvar()
    opacity_values = opacity.ComputeFlattened() if opacity and opacity.HasAuthoredValue() else []
    if any(float(value) != 1 for value in opacity_values):
        raise ValueError('Translucent unbound displayOpacity is unsupported')
    return {'source_material': 'usd_displayColor:' + str(prim.GetPath()),
            'base_color': list(first), 'base_color_space': 'linear_rec709',
            'albedo_add': 0.,
            'roughness': .5, 'metallic': 0., 'albedo': None, 'normal': None, 'orm': None,
            'uv_scale': [1., 1.], 'normal_flip_tangent_v': False,
            'authored_shader_inputs': {'source_fallback': 'authored uniform Gprim displayColor; no bound material',
                'displayColor_interpolation': interpolation, 'displayColor_count': str(len(values)),
                'displayOpacity_count': str(len(opacity_values)),
                'roughness_metallic_scope': 'renderer approximation; original unbound shading parity unproven'}}


def surface(prim, usd, geometry_prim=None):
    bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    if not bound:
        return display_color_surface(geometry_prim if geometry_prim is not None else prim)
    shaders = [UsdShade.Shader(p) for p in Usd.PrimRange(bound.GetPrim()) if p.IsA(UsdShade.Shader)]
    if len(shaders) != 1:
        raise ValueError("Background shader graph requires a separate material audit")
    shader = shaders[0]

    def value(name, fallback):
        input_ = shader.GetInput(name)
        result = input_.Get() if input_ else None
        return fallback if result is None else result

    def texture(name):
        asset = value(name, None)
        if asset is None:
            return None
        if not hasattr(asset, "resolvedPath"):
            raise ValueError("Connected background textures require a separate audit")
        path = Path(asset.resolvedPath or (usd.parent / asset.path)).resolve()
        if not path.is_file() or path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            raise ValueError(f"Original background texture is unavailable: {path}")
        if path.stat().st_size > 128 * 1024 * 1024:
            raise ValueError(f"Background texture exceeds 128 MiB byte bound: {path}")
        return {"path": str(path), "sha256": digest(path)}

    # This original UE4 graph uses different input names from OmniPBR. Reading
    # OmniPBR defaults here silently replaced three textured brown boxes with
    # gray surfaces. Map only the inspected byte-bound graph, not generic MDL.
    source = shader.GetPrim().GetAttribute("info:mdl:sourceAsset").Get()
    if source is not None and Path(source.path).name == "MI_LampCeilingA.mdl":
        mdl = Path(source.resolvedPath or (usd.parent / source.path)).resolve()
        if (not mdl.is_file() or digest(mdl) != WAREHOUSE_MDL_SHA
                or shader.GetPrim().GetAttribute("info:mdl:sourceAsset:subIdentifier").Get() != "MI_LampCeilingA"):
            raise ValueError("Warehouse MDL graph identity changed")
        required = {"U_Tiling", "V_Tiling", "MainNormalInput", "AlbedoTexture",
                    "Desaturation", "BaseColor_Tint", "MergeMapInput", "RoughnessMin", "RoughnessMax"}
        if {i.GetBaseName() for i in shader.GetInputs()} != required:
            raise ValueError("Warehouse MDL input contract changed")
        tint = list(map(float, value("BaseColor_Tint", None)))
        desaturation = float(value("Desaturation", None))
        roughness = [float(value("RoughnessMin", None)), float(value("RoughnessMax", None))]
        tiling = [float(value("U_Tiling", None)), float(value("V_Tiling", None))]
        if (len(tint) != 4 or tint[3] != 1. or any(not 0. <= v <= 1. for v in tint)
                or not 0. <= desaturation <= 1. or not 0. <= roughness[0] <= roughness[1] <= 1.
                or any(not 0. < v <= 1024. for v in tiling)):
            raise ValueError("Warehouse MDL parameters exceed inspected finite mapping")
        textures = {key: texture(name) for key, name in (
            ("albedo", "AlbedoTexture"), ("normal", "MainNormalInput"), ("orm", "MergeMapInput"))}
        if any(v is None for v in textures.values()):
            raise ValueError("Warehouse MDL requires all three original textures")
        return {"source_material": str(bound.GetPath()), "base_color": tint[:3],
                "base_color_space": "linear_rec709", "albedo_add": 0.,
                "roughness": 1., "metallic": 1., **textures,
                "uv_scale": tiling, "normal_flip_tangent_v": False,
                "warehouse_mdl": {"source_mdl_sha256": WAREHOUSE_MDL_SHA,
                                  "albedo_desaturation": desaturation, "roughness_min_max": roughness},
                "authored_shader_inputs": {i.GetBaseName(): str(i.Get()) for i in shader.GetInputs()}}

    if value("enable_opacity", False) or value("enable_emission", False):
        raise ValueError("Active background opacity/emission needs a separate mapping")
    albedo = texture("diffuse_texture")
    if (float(value('albedo_brightness', 1.)) != 1.
            or float(value('albedo_desaturation', 0.)) != 0.):
        raise ValueError('Nondefault albedo brightness/desaturation needs a separate mapping')
    albedo_add = float(value('albedo_add', 0.)) if albedo else 0.
    if not -1. <= albedo_add <= 1.:
        raise ValueError('Authored albedo_add exceeds its finite mapping bound')
    color = value("diffuse_tint", (1., 1., 1.)) if albedo else value("diffuse_color_constant", value("diffuse_tint", (.5, .5, .5)))
    color_input = shader.GetInput("diffuse_tint" if albedo or not shader.GetInput("diffuse_color_constant") else "diffuse_color_constant")
    color_space = Usd.ColorSpaceAPI.ComputeColorSpaceName(color_input.GetAttr(), None) if color_input else ""
    if color_space not in ("", "lin_rec709_scene", "lin_rec709"):
        raise ValueError(f"Background authored color space requires an explicit conversion: {color_space}")
    return {
        "source_material": str(bound.GetPath()), "base_color": list(map(float, color)),
        "base_color_space": "linear_rec709",
        "albedo_add": albedo_add,
        "roughness": float(value("reflection_roughness_constant", .5)),
        "metallic": float(value("metallic_constant", 0.)), "albedo": albedo,
        "normal": texture("normalmap_texture"),
        "orm": texture("ORM_texture") if value("enable_ORM_texture", False) else None,
        "uv_scale": list(map(float, value("texture_scale", (1., 1.)))),
        "normal_flip_tangent_v": bool(value("flip_tangent_v", False)),
        "authored_shader_inputs": {i.GetBaseName(): str(i.Get()) for i in shader.GetInputs()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mobile-background", type=Path, help="byte-bound native T2 physics definition; enables mobile visuals")
    args = parser.parse_args()
    mobile = args.mobile_background is not None
    physics_sha = digest(args.mobile_background) if mobile else None
    if mobile:
        physical = json.loads(args.mobile_background.read_text())
        if physical["schema"] != "native_g1_released_t2_background_v1" or physical["source_background_sha256"] != USD_SHA or physical["physics_parity_qualified"]:
            raise ValueError("Mobile visual source does not match the original T2 background")
    if digest(args.usd) != USD_SHA:
        raise ValueError("Original background USD identity changed")
    stage = Usd.Stage.Open(str(args.usd))
    stage.SetEditTarget(stage.GetSessionLayer())
    root = stage.GetDefaultPrim()
    disabled = []
    for relative in (() if mobile else DEACTIVATE):
        prim = stage.GetPrimAtPath(str(root.GetPath()) + "/" + relative)
        if not prim.IsValid():
            raise ValueError("Original static background deactivation path disappeared")
        stage.OverridePrim(prim.GetPath()).SetActive(False)
        disabled.append(str(prim.GetPath()))
    # Original LibraryBackground pose (4.420,1.408,-.795), followed by the
    # same disclosed +.795 native environment translation as the physics world.
    offset = Gf.Matrix4d().SetTranslate(Gf.Vec3d(4.420, 1.408, 0.))
    region_min, region_max = ([-1.5,-3.5,0.],[2.5,1.5,2.5]) if mobile else ([-.25,-1.2,0.],[1.6,1.2,2.5])
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    cache = UsdGeom.XformCache()
    meshes, skipped, omitted = [], [], []

    def append_mesh(prim, material_prim, transform, selected=None, owner_path=None):
        try:
            material = surface(material_prim, args.usd, prim)
        except ValueError as error:
            omitted.append({"path": str(material_prim.GetPath()), "reason": str(error),
                            "face_indices": selected})
            return
        # Geometry validation remains fatal. Unsupported materials are disclosed
        # coverage gaps; invalid source geometry is never silently accepted.
        meshes.append({"path": str(material_prim.GetPath()), "material": material,
                       "source_face_subset": selected, **({"owner_path":owner_path} if mobile else {}), **geometry(prim, transform, selected)})
    for prim in Usd.PrimRange(root):
        if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube)):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility() == "invisible" or imageable.ComputePurpose() in ("proxy", "guide"):
            continue
        box = bounds.ComputeWorldBound(prim).ComputeAlignedBox()
        lo = offset.Transform(box.GetMin()); hi = offset.Transform(box.GetMax())
        if any(hi[i] < region_min[i] or lo[i] > region_max[i] for i in range(3)):
            skipped.append(str(prim.GetPath()))
            continue
        if prim.IsA(UsdGeom.Mesh):
            uv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
            if not uv or not uv.ComputeFlattened():
                # MDL procedural projection is not the authored-UV mapping used
                # by this bounded importer. Disclose the gap, never invent UVs.
                omitted.append({"path": str(prim.GetPath()), "reason": "no authored st; MDL projection unimplemented"})
                continue
        transform = cache.GetLocalToWorldTransform(prim) * offset
        if transform.GetDeterminant() <= 0:
            raise ValueError("Background reflection needs a separate winding audit")
        owner_path = None
        if mobile:
            owner = prim
            while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = owner.GetParent()
            if owner:
                # Keep the authored owner scale baked into body-local mesh
                # vertices. Owner poses are supplied by actual native frames.
                owner_transform = Gf.Transform(cache.GetLocalToWorldTransform(owner))
                owner_rigid = Gf.Matrix4d(1)
                owner_rigid.SetRotate(owner_transform.GetRotation())
                owner_rigid.SetTranslateOnly(owner_transform.GetTranslation())
                transform = cache.GetLocalToWorldTransform(prim) * owner_rigid.GetInverse()
                owner_path = str(owner.GetPath()).replace(str(root.GetPath()), '/World/envs/env_0/galileo_locomanip', 1)
                if owner_path not in {body['path'] for body in physical['bodies'] if body['dynamic']}:
                    raise ValueError('Original dynamic visual owner is absent from frozen physics')
        subsets = UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets()
        if subsets:
            covered = set()
            for subset in subsets:
                selected = list(map(int, subset.GetIndicesAttr().Get()))
                if covered.intersection(selected):
                    raise ValueError("Overlapping source material subsets")
                covered.update(selected)
                append_mesh(prim, subset.GetPrim(), transform, selected, owner_path)
            all_faces = set(range(len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get())))
            remaining = sorted(all_faces - covered)
            if remaining:
                omitted.append({"path": str(prim.GetPath()), "reason": "unbound faces outside original material subsets",
                                "face_indices": remaining})
        else:
            append_mesh(prim, prim, transform, owner_path=owner_path)
    if not meshes or len(meshes) > (256 if mobile else 128) or digest(args.usd) != USD_SHA:
        raise ValueError("Background geometry is empty, unbounded or modified")
    if sum("/TaskAssets/shelf/" in mesh["path"] for mesh in meshes) != 3:
        raise ValueError("All three original task shelf surfaces are required")
    document = {
        "schema": "native_g1_mobile_background_visual_v1" if mobile else "native_g1_static_background_visual_v1", "units": "metres_z_up",
        "source_usd_sha256": USD_SHA, "exporter_sha256": digest(Path(__file__)),
        "source_arena_commit": "8b4a3a47fc53de23e8205089d71109a2e2348acd",
        "source_deactivated_prims": disabled,
        "source_background_translation": [4.420, 1.408, -.795],
        "native_environment_translation": [0., 0., .795],
        "selection_region_min": region_min, "selection_region_max": region_max,
        "mesh_selection": "whole source meshes intersecting disclosed mobile-task region" if mobile else "whole source meshes intersecting disclosed static-task region",
        "material_scope": "authored basic PBR inputs; MDL defaults/effects and lighting parity unproven",
        "source_renderer_parity_proven": False, "physics_integrations": 0,
        "background_physics_registered": False, "skipped_outside_region": skipped,
        "omitted_unmapped_geometry": omitted,
        "meshes": meshes,
    }
    if mobile:
        document["source_t2_background_sha256"] = physics_sha
    with args.output.open("x") as output:
        json.dump(document, output, separators=(",", ":"), allow_nan=False)
        output.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output),
                      "meshes": len(meshes), "bytes": args.output.stat().st_size,
                      "physics_integrations": 0, "qualified": False}))


if __name__ == "__main__":
    main()

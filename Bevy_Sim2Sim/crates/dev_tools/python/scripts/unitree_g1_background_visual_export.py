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

from pxr import Gf, Usd, UsdGeom, UsdShade

from unitree_g1_task_asset_query import digest
from unitree_g1_task_visual_export import geometry

USD_SHA = "7e14dcfd948591b8fdae61d41b412097b39490022dfc25aab9b90b6884509051"
DEACTIVATE = ("BackgroundAssets/boxes/jetson_orin_06", "BackgroundAssets/boxes/jetson_orin_03",
              "BackgroundAssets/boxes/hesai_box_06")


def surface(prim, usd):
    bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    if not bound:
        raise ValueError(f"Background surface has no bound source material: {prim.GetPath()}")
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

    if value("enable_opacity", False) or value("enable_emission", False):
        raise ValueError("Active background opacity/emission needs a separate mapping")
    albedo = texture("diffuse_texture")
    color = value("diffuse_tint", (1., 1., 1.)) if albedo else value("diffuse_color_constant", value("diffuse_tint", (.5, .5, .5)))
    color_input = shader.GetInput("diffuse_tint" if albedo or not shader.GetInput("diffuse_color_constant") else "diffuse_color_constant")
    color_space = Usd.ColorSpaceAPI.ComputeColorSpaceName(color_input.GetAttr(), None) if color_input else ""
    if color_space not in ("", "lin_rec709_scene", "lin_rec709"):
        raise ValueError(f"Background authored color space requires an explicit conversion: {color_space}")
    return {
        "source_material": str(bound.GetPath()), "base_color": list(map(float, color)),
        "base_color_space": "linear_rec709",
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
    args = parser.parse_args()
    if digest(args.usd) != USD_SHA:
        raise ValueError("Original background USD identity changed")
    stage = Usd.Stage.Open(str(args.usd))
    stage.SetEditTarget(stage.GetSessionLayer())
    root = stage.GetDefaultPrim()
    disabled = []
    for relative in DEACTIVATE:
        prim = stage.GetPrimAtPath(str(root.GetPath()) + "/" + relative)
        if not prim.IsValid():
            raise ValueError("Original static background deactivation path disappeared")
        stage.OverridePrim(prim.GetPath()).SetActive(False)
        disabled.append(str(prim.GetPath()))
    # Original LibraryBackground pose (4.420,1.408,-.795), followed by the
    # same disclosed +.795 native environment translation as the physics world.
    offset = Gf.Matrix4d().SetTranslate(Gf.Vec3d(4.420, 1.408, 0.))
    region_min, region_max = [-.25, -1.2, 0.], [1.6, 1.2, 2.5]
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    cache = UsdGeom.XformCache()
    meshes, skipped, omitted = [], [], []

    def append_mesh(prim, material_prim, transform, selected=None):
        try:
            material = surface(material_prim, args.usd)
        except ValueError as error:
            omitted.append({"path": str(material_prim.GetPath()), "reason": str(error),
                            "face_indices": selected})
            return
        # Geometry validation remains fatal. Unsupported materials are disclosed
        # coverage gaps; invalid source geometry is never silently accepted.
        meshes.append({"path": str(material_prim.GetPath()), "material": material,
                       "source_face_subset": selected, **geometry(prim, transform, selected)})
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
        subsets = UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets()
        if subsets:
            covered = set()
            for subset in subsets:
                selected = list(map(int, subset.GetIndicesAttr().Get()))
                if covered.intersection(selected):
                    raise ValueError("Overlapping source material subsets")
                covered.update(selected)
                append_mesh(prim, subset.GetPrim(), transform, selected)
            all_faces = set(range(len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get())))
            remaining = sorted(all_faces - covered)
            if remaining:
                omitted.append({"path": str(prim.GetPath()), "reason": "unbound faces outside original material subsets",
                                "face_indices": remaining})
        else:
            bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            if not bound:
                omitted.append({"path": str(prim.GetPath()), "reason": "no source-bound material or face subset"})
                continue
            append_mesh(prim, prim, transform)
    if not meshes or len(meshes) > 128 or digest(args.usd) != USD_SHA:
        raise ValueError("Background geometry is empty, unbounded or modified")
    if sum("/TaskAssets/shelf/" in mesh["path"] for mesh in meshes) != 3:
        raise ValueError("All three original task shelf surfaces are required")
    document = {
        "schema": "native_g1_static_background_visual_v1", "units": "metres_z_up",
        "source_usd_sha256": USD_SHA, "exporter_sha256": digest(Path(__file__)),
        "source_arena_commit": "8b4a3a47fc53de23e8205089d71109a2e2348acd",
        "source_deactivated_prims": disabled,
        "source_background_translation": [4.420, 1.408, -.795],
        "native_environment_translation": [0., 0., .795],
        "selection_region_min": region_min, "selection_region_max": region_max,
        "mesh_selection": "whole source meshes intersecting disclosed static-task region",
        "material_scope": "authored basic PBR inputs; MDL defaults/effects and lighting parity unproven",
        "source_renderer_parity_proven": False, "physics_integrations": 0,
        "background_physics_registered": False, "skipped_outside_region": skipped,
        "omitted_unmapped_geometry": omitted,
        "meshes": meshes,
    }
    with args.output.open("x") as output:
        json.dump(document, output, separators=(",", ":"), allow_nan=False)
        output.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output),
                      "meshes": len(meshes), "bytes": args.output.stat().st_size,
                      "physics_integrations": 0, "qualified": False}))


if __name__ == "__main__":
    main()

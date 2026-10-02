#!/usr/bin/env python3
"""CPU export of byte-bound original task visuals, authored normals and UVs.

Geometry uses source Z-up metres in the dynamic object's local rigid frame.
No physics integrations or MDL shader execution occur. Supported original glTF
and OmniPBR surface inputs are recorded explicitly; full renderer parity is not
claimed. Original texture bytes stay in their separately hashed local cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdShade

from unitree_g1_task_asset_query import ASSETS, digest


def surface(prim, usd):
    bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    if not bound:
        raise ValueError(f"Missing original visual material: {prim.GetPath()}")
    shaders = [UsdShade.Shader(p) for p in Usd.PrimRange(bound.GetPrim()) if p.IsA(UsdShade.Shader)]
    original = next((s for s in shaders if s.GetInput("diffuse_texture") or s.GetInput("base_color_factor")), None)
    if not original:
        raise ValueError(f"Unsupported original visual surface: {bound.GetPath()}")

    def value(name, fallback):
        input_ = original.GetInput(name)
        result = input_.Get() if input_ else None
        return fallback if result is None else result

    def texture(input_):
        if not input_:
            return None
        asset = input_.Get()
        if asset is None:
            connection = input_.GetConnectedSource()
            if not connection:
                return None
            input_ = UsdShade.Shader(connection[0].GetPrim()).GetInput("texture")
            asset = input_.Get() if input_ else None
        if asset is None:
            return None
        path = Path(asset.resolvedPath or (usd.parent / asset.path)).resolve()
        if not path.is_file():
            raise ValueError(f"Original texture dependency missing: {path}")
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg") or path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("Unsupported or oversized original texture")
        return {"path": str(path), "sha256": digest(path)}

    gltf = bool(original.GetInput("base_color_factor"))
    color = value("base_color_factor" if gltf else "diffuse_tint", (1, 1, 1))
    # OmniPBR's diffuse_color_constant is used only when no diffuse texture exists.
    albedo = texture(original.GetInput("base_color_texture" if gltf else "diffuse_texture"))
    if albedo is None and not gltf:
        color = value("diffuse_color_constant", color)
    if value("enable_emission", False) or value("enable_opacity", False):
        raise ValueError("Unsupported active source emission/opacity")
    if not albedo:
        raise ValueError("Expected original albedo texture is missing")
    return {
        "source_material": str(bound.GetPath()), "base_color": list(map(float, color)),
        "roughness": float(value("roughness_factor" if gltf else "reflection_roughness_constant", value("roughness", 0.5))),
        "metallic": float(value("metallic_factor" if gltf else "metallic_constant", value("metallic", 0.0))),
        "albedo": albedo, "normal": texture(original.GetInput("normalmap_texture")),
        "orm": texture(original.GetInput("ORM_texture")) if value("enable_ORM_texture", False) else None,
        "uv_scale": list(map(float, value("texture_scale", (1, 1)))),
        "normal_flip_tangent_v": bool(value("flip_tangent_v", False)),
        "analytic_cube_uv": prim.IsA(UsdGeom.Cube),
    }


def at(values, interpolation, vertex, corner, face, expected):
    count = {"constant": 1, "uniform": expected[0], "vertex": expected[1], "varying": expected[1], "faceVarying": expected[2]}.get(interpolation)
    if count is None or len(values) != count:
        raise ValueError(f"Authored attribute interpolation/count mismatch: {interpolation}/{len(values)}/{count}")
    return values[{"constant": 0, "uniform": face, "vertex": vertex, "varying": vertex, "faceVarying": corner}[interpolation]]


def geometry(prim, matrix, selected_faces=None):
    fallback_vertices = 0
    if prim.IsA(UsdGeom.Cube):
        half = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
        points = [Gf.Vec3d(x * half, y * half, z * half) for x, y, z in
                  [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]]
        faces = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
        indices = [i for f in faces for i in f]
        counts = [4] * 6
        normals = [Gf.Cross(points[f[1]]-points[f[0]], points[f[2]]-points[f[0]]).GetNormalized() for f in faces]
        normal_interpolation = "uniform"
        uvs = [(0.,0.),(1.,0.),(1.,1.),(0.,1.)] * 6
        uv_interpolation = "faceVarying"
        orientation = "rightHanded"
    else:
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = list(mesh.GetFaceVertexCountsAttr().Get())
        indices = list(mesh.GetFaceVertexIndicesAttr().Get())
        normals = list(mesh.GetNormalsAttr().Get() or [])
        normal_interpolation = mesh.GetNormalsInterpolation()
        uv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        uvs = uv.ComputeFlattened() if uv else None
        uv_interpolation = uv.GetInterpolation() if uv else None
        orientation = mesh.GetOrientationAttr().Get()
        if not normals or not uvs or mesh.GetHoleIndicesAttr().Get():
            raise ValueError("Missing original mesh normals/UVs or unsupported holes")
    if sum(counts) != len(indices) or min(counts) < 3 or orientation not in ("rightHanded", "leftHanded"):
        raise ValueError("Invalid original visual faces")
    # The frozen plate contains 36 vertex normals with zero/denormal length.
    # Preserve every triangle/UV and all valid authored normals. Only these
    # invalid source normals receive area-weighted adjacent source-face normals.
    bad = [i for i, n in enumerate(normals) if Gf.Vec3d(*n).GetLength() < 1e-12]
    if bad:
        if normal_interpolation != "vertex":
            raise ValueError("Invalid non-vertex source normals need a separate audit")
        sums = {i: Gf.Vec3d(0) for i in bad}
        largest = {i: Gf.Vec3d(0) for i in bad}
        offset = 0
        for count in counts:
            face = indices[offset:offset+count]
            for c in range(1, count-1):
                tri = (face[0], face[c], face[c+1])
                p, a, b = [Gf.Vec3d(*points[i]) for i in tri]
                normal = Gf.Cross(a-p, b-p)
                if orientation == "leftHanded":
                    normal = -normal
                for i in tri:
                    if i in sums:
                        sums[i] += normal
                        if normal.GetLength() > largest[i].GetLength():
                            largest[i] = normal
            offset += count
        for i, normal in sums.items():
            if normal.GetLength() <= 1e-12:
                # Opposite adjacent faces can cancel at source non-manifold
                # seams. The largest original supporting face is unambiguous.
                normal = largest[i]
            if normal.GetLength() <= 1e-12:
                raise ValueError("Invalid source normal has no supporting original face")
            normals[i] = normal.GetNormalized()
        fallback_vertices = len(bad)
    normal_matrix = matrix.GetInverse().GetTranspose()
    expected = (len(counts), len(points), len(indices))
    if selected_faces is not None:
        selected_faces = set(selected_faces)
        if not selected_faces or any(i < 0 or i >= len(counts) for i in selected_faces):
            raise ValueError("Invalid original material face subset")
    vertices, out_normals, out_uvs, triangles, unique = [], [], [], [], {}
    offset = 0
    for face, count in enumerate(counts):
        if selected_faces is not None and face not in selected_faces:
            offset += count
            continue
        corners = []
        for c in range(offset, offset + count):
            vertex = indices[c]
            if vertex < 0 or vertex >= len(points):
                raise ValueError("Original visual index out of range")
            p = matrix.Transform(Gf.Vec3d(*points[vertex]))
            n = normal_matrix.TransformDir(Gf.Vec3d(*at(normals, normal_interpolation, vertex, c, face, expected))).GetNormalized()
            uv = at(uvs, uv_interpolation, vertex, c, face, expected)
            key = tuple(map(float, p)) + tuple(map(float, n)) + tuple(map(float, uv))
            if key not in unique:
                unique[key] = len(vertices)
                vertices.append(list(key[:3])); out_normals.append(list(key[3:6])); out_uvs.append(list(key[6:8]))
            corners.append(unique[key])
        for c in range(1, count - 1):
            tri = [corners[0], corners[c], corners[c+1]]
            if orientation == "leftHanded":
                tri[1], tri[2] = tri[2], tri[1]
            triangles.append(tri)
        offset += count
    return {"points": vertices, "normals": out_normals, "uvs": out_uvs, "triangles": triangles,
            "source_normal_interpolation": normal_interpolation, "source_uv_interpolation": uv_interpolation,
            "source_normal_fallback_vertices": fallback_vertices}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--query-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.query) != args.query_sha256:
        raise ValueError("Source query identity changed")
    query = json.loads(args.query.read_text())
    if not query.get("all_asset_queries_succeeded") or query.get("physics_integrations") != 0:
        raise ValueError("Incomplete original source query")
    objects = []
    for asset in query["assets"]:
        usd = Path(asset["local_path"]); expected, scale = ASSETS[asset["name"]]
        if digest(usd) != expected or asset["usd_sha256"] != expected:
            raise ValueError("Original visual USD bytes changed")
        stage = Usd.Stage.Open(str(usd)); stage.SetEditTarget(stage.GetSessionLayer())
        root = stage.GetDefaultPrim(); root.GetAttribute("xformOp:scale").Set(Gf.Vec3f(*scale))
        cache = UsdGeom.XformCache(); meshes = []
        for prim in Usd.PrimRange(root):
            if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube)):
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable.ComputeVisibility() == "invisible" or imageable.ComputePurpose() in ("guide", "proxy"):
                continue
            matrix = cache.GetLocalToWorldTransform(prim)
            if matrix.GetDeterminant() <= 0:
                raise ValueError("Visual frame orientation needs an explicit reflection audit")
            meshes.append({"path": str(prim.GetPath()), "material": surface(prim, usd), **geometry(prim, matrix)})
        if not meshes or digest(usd) != expected:
            raise ValueError("Empty visuals or modified original USD")
        objects.append({"kind": asset["name"], "usd_sha256": expected, "source_scale_override": list(scale), "meshes": meshes})
    result = {"schema": "native_g1_task_visual_v2", "units": "metres_z_up",
              "source_query_sha256": args.query_sha256, "exporter_sha256": digest(Path(__file__)),
              "source_renderer_parity_proven": False, "objects": objects}
    with args.output.open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False); out.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output), "bytes": args.output.stat().st_size,
                      "triangles": {o["kind"]: sum(len(m["triangles"]) for m in o["meshes"]) for o in objects}, "qualified": False}))


if __name__ == "__main__":
    main()

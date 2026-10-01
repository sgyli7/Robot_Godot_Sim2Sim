#!/usr/bin/env python3
"""Export original cooked convex parts and explicit queried mass for Rapier.

This CPU conversion does not recook the visual mesh or replace a bowl with one
convex hull. All geometry remains source Z-up metres. Source runtime mismatch
and unqualified physics remain explicit; no source task success is implied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

from unitree_g1_task_asset_query import ASSETS, digest


def vector(value):
    return [float(v) for v in value]


def material(prim):
    bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")[0]
    if not bound or not bound.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
        raise ValueError("Task collider lacks its original bound physics material")
    api = UsdPhysics.MaterialAPI(bound.GetPrim())
    return {"static_friction": float(api.GetStaticFrictionAttr().Get()),
            "dynamic_friction": float(api.GetDynamicFrictionAttr().Get()),
            "restitution": float(api.GetRestitutionAttr().Get())}


def polygons_to_triangles(hull):
    triangles = []
    for polygon in hull["polygons"]:
        start, count = polygon["index_base"], polygon["num_vertices"]
        if count < 3 or start < 0 or start + count > len(hull["indices"]):
            raise ValueError("Invalid source cooked polygon")
        face = hull["indices"][start:start + count]
        for index in range(1, count - 1):
            triangles.append([face[0], face[index], face[index + 1]])
    return triangles


def export(query):
    if not query.get("all_asset_queries_succeeded") or not query.get("closed_timeline_was_stopped") or query["physics_integrations"] != 0:
        raise ValueError("Incomplete source asset query")
    result = []
    for item in query["assets"]:
        name = item["name"]
        expected, scale = ASSETS[name]
        usd = Path(item["local_path"])
        if item["usd_sha256"] != expected or digest(usd) != expected:
            raise ValueError("Source asset bytes changed after query")
        # All four original bodies have identity unscaled local rigid frames.
        # Scaled geometry and already-scaled query COM/inertia use this frame.
        expected_matrix = [[scale[r] if r == c and r < 3 else 1.0 if r == c == 3 else 0.0 for c in range(4)] for r in range(4)]
        if any(abs(a - b) > 1e-6 for row, expected_row in zip(item["rigid_body_matrix"], expected_matrix) for a, b in zip(row, expected_row)):
            raise ValueError("Body frame changed; explicit rebase audit required")
        stage = Usd.Stage.Open(str(usd))
        stage.SetEditTarget(stage.GetSessionLayer())
        root = stage.GetDefaultPrim()
        root.GetAttribute("xformOp:scale").Set(Gf.Vec3f(*scale))
        cache = UsdGeom.XformCache()
        bodies = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(bodies) != 1:
            raise ValueError("Original object rigid-body coverage changed")
        ccd_attr = bodies[0].GetAttribute("physxRigidBody:enableCCD")
        ccd_enabled = bool(ccd_attr.Get()) if ccd_attr else False
        parts, materials = [], []
        meshes = {m["path"].removeprefix(f"/Objects/{name}"): m for m in item["collision_meshes"]}
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            relative = str(prim.GetPath()).removeprefix(str(root.GetPath()))
            materials.append(material(prim))
            matrix = cache.GetLocalToWorldTransform(prim)
            if prim.IsA(UsdGeom.Mesh):
                cooked = meshes.pop(relative)
                if cooked["source_approximation"] != "convexDecomposition":
                    raise ValueError("Unexpected source task collision approximation")
                if any(abs(float(matrix[r][c]) - cooked["mesh_to_object_matrix"][r][c]) > 1e-6 for r in range(4) for c in range(4)):
                    raise ValueError("CPU/source mesh transform changed")
                for hull in cooked["hulls"]:
                    points = [vector(matrix.Transform(Gf.Vec3d(*p))) for p in hull["vertices_mesh_local"]]
                    parts.append({"points": points, "triangles": polygons_to_triangles(hull)})
            elif prim.IsA(UsdGeom.Cube):
                half = UsdGeom.Cube(prim).GetSizeAttr().Get() / 2
                points = [vector(matrix.Transform(Gf.Vec3d(x * half, y * half, z * half))) for x, y, z in
                          [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]]
                faces = [[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
                parts.append({"points": points, "triangles": [[f[0], f[i], f[i+1]] for f in faces for i in (1,2)]})
            else:
                raise ValueError("Unsupported original task collision primitive")
        if meshes or not parts or not materials or any(m != materials[0] for m in materials):
            raise ValueError("Task collision/material coverage incomplete")
        # Hull callback order is not reproducible; retain every part and sort its
        # unchanged geometry for stable file ordering, without merging/smoothing.
        parts.sort(key=lambda part: json.dumps(part, sort_keys=True, separators=(",", ":")))
        mass = item["mass_query"]
        x, y, z, w = mass["principal_axes_xyzw"]
        result.append({"kind": name, "usd_sha256": expected, "source_scale_override": list(scale),
                       "source_ccd_enabled": ccd_enabled,
                       "mass_kg": mass["mass_kg"], "center_of_mass": mass["center_of_mass"],
                       "principal_inertia": mass["inertia"], "principal_axes_wxyz": [w,x,y,z],
                       "material": materials[0], "convex_parts": parts})
        if digest(usd) != expected:
            raise ValueError("Exporter modified original asset bytes")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--query-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.query) != args.query_sha256:
        raise ValueError("Source query byte identity changed")
    query = json.loads(args.query.read_text())
    result = {"schema": "native_g1_task_objects_v2", "units": "metres_kilograms_radians_z_up",
              "source_query_sha256": args.query_sha256, "exporter_sha256": digest(Path(__file__)),
              "source_runtime_build": query["runtime_build"], "physics_parity_qualified": False,
              "objects": export(query)}
    with args.output.open("x") as output:
        json.dump(result, output, separators=(",", ":"), allow_nan=False)
        output.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output),
                      "parts": {o["kind"]: len(o["convex_parts"]) for o in result["objects"]}, "qualified": False}))


if __name__ == "__main__":
    main()

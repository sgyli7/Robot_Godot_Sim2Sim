#!/usr/bin/env python3
"""Query original task collision cooking and mass, without advancing physics.

Default invocation only verifies cached byte identities. --cook-source uses the
already-installed Isaac runtime in a fresh process. This is an asset diagnostic,
not an original task rollout, and records the T1 runtime mismatch explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import sys
import time


ASSETS = {
    "t1_apple": ("2e0e0462c4345b340e1c6040c11abe1818437ced1db228a53ec0944001bb46d8", (0.009,) * 3),
    "t1_plate": ("286238c8f957e3267fa21a0003b320868130f903a4b1d991a4ffeb49faaea5a8", (0.5,) * 3),
    "t2_box": ("50dc139612086b9483770a1abc17dc600445aa4f85323d74fa97069f7c2eb4ed", (1.0,) * 3),
    "t2_bin": ("b9ffec2e70fd009863a3fa8bd699aca808403522eafb259d5638135e63506999", (4.0, 2.0, 1.0)),
}
API_SOURCES = [
    "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/mass_inertia_queries.html",
    "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physx/docs/api/python.html",
]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def values(value) -> list[float]:
    result = [float(x) for x in value]
    if not all(math.isfinite(x) for x in result):
        raise ValueError("Non-finite source query value")
    return result


def save(output, receipt):
    output.seek(0)
    output.truncate()
    json.dump(receipt, output, indent=2, allow_nan=False)
    output.write("\n")
    output.flush()


def query(args, receipt, output):
    # SimulationApp must precede imports of simulation-dependent modules.
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "disable_viewport_updates": True})
    try:
        import omni.timeline
        import omni.usd
        from omni.physx import get_physx_cooking_interface, get_physx_property_query_interface
        from omni.physx.bindings._physx import PhysxCollisionRepresentationResult, PhysxPropertyQueryResult
        from pxr import Gf, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdUtils

        if omni.timeline.get_timeline_interface().is_playing():
            raise ValueError("Asset query requires a stopped timeline")
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, "Z")
        UsdPhysics.Scene.Define(stage, "/PhysicsScene")
        cache = UsdGeom.XformCache()
        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        if stage_id < 0:
            raise ValueError("Source stage has no valid Kit cache identity")
        version = Path('/isaac-sim/VERSION')
        if not version.is_file():
            version = Path(sys.modules['isaacsim'].__file__).parent / "VERSION"
        receipt["runtime_build"] = version.read_text().strip()
        receipt['t1_matching_runtime_verified'] = (
            receipt['runtime_build'] == '6.0.0-rc.22+release.33481.407f3ea1.gl')
        try:
            receipt["isaacsim_package_version"] = importlib.metadata.version("isaacsim")
        except importlib.metadata.PackageNotFoundError:
            receipt["isaacsim_package_version"] = None
            receipt["runtime_distribution"] = "standalone_image; exact VERSION build retained"
        receipt["openusd_version"] = list(Usd.GetVersion())

        for item in receipt["assets"]:
            name = item["name"]
            print(f"G1_ASSET_QUERY compose {name}", flush=True)
            prim = stage.DefinePrim(f"/Objects/{name}")
            prim.GetReferences().AddReference(item["local_path"])
            scale_attr = prim.GetAttribute("xformOp:scale")
            if not scale_attr:
                raise ValueError("Original default prim lacks its scale operation")
            # Override the source op, never multiply the task scale twice.
            scale_attr.Set(Gf.Vec3f(*ASSETS[name][1]))
            cache.Clear()
            subtree = list(Usd.PrimRange(prim))
            bodies = [p for p in subtree if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(bodies) != 1:
                raise ValueError(f"Expected one dynamic source body: {name}")
            body = bodies[0]
            api = UsdPhysics.RigidBodyAPI(body)
            if not api.GetRigidBodyEnabledAttr().Get() or api.GetKinematicEnabledAttr().Get():
                raise ValueError("Original task object must be dynamic")
            item["composed_root_scale"] = values(scale_attr.Get())
            item["rigid_body_path"] = str(body.GetPath())
            item["rigid_body_matrix"] = [values(row) for row in cache.GetLocalToWorldTransform(body)]
            item["collision_meshes"] = []

            for mesh in subtree:
                if not mesh.IsA(UsdGeom.Mesh) or not mesh.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                data = {"path": str(mesh.GetPath()), "source_approximation": str(UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get()),
                        "mesh_to_object_matrix": [values(row) for row in cache.GetLocalToWorldTransform(mesh)], "hulls": []}

                def cooked(result, convexes):
                    if result != PhysxCollisionRepresentationResult.RESULT_VALID:
                        data["error"] = str(result)
                        return
                    for convex in convexes:
                        polygons = [{"index_base": int(p.index_base), "num_vertices": int(p.num_vertices), "plane": values(p.plane)} for p in convex.polygons]
                        data["hulls"].append({"vertices_mesh_local": [values(v) for v in convex.vertices], "indices": [int(i) for i in convex.indices], "polygons": polygons})

                get_physx_cooking_interface().request_convex_collision_representation(
                    stage_id=stage_id, collision_prim_id=PhysicsSchemaTools.sdfPathToInt(mesh.GetPath()),
                    run_asynchronously=False, on_result=cooked,
                )
                if data.get("error") or not data["hulls"]:
                    raise ValueError(f"Source collision cooking failed: {data}")
                item["collision_meshes"].append(data)

            finished = [False]

            def mass_received(response):
                if response.result != PhysxPropertyQueryResult.VALID:
                    item["mass_error"] = str(response.result)
                    return
                item["mass_query"] = {"mass_kg": float(response.mass), "inertia": values(response.inertia),
                                      "center_of_mass": values(response.center_of_mass), "principal_axes_xyzw": values(response.principal_axes),
                                      "type": str(response.type), "body_path": str(PhysicsSchemaTools.intToSdfPath(response.path_id))}

            def collider_received(response):
                data = {"result": str(response.result), "path": str(PhysicsSchemaTools.intToSdfPath(response.path_id))}
                if response.result == PhysxPropertyQueryResult.VALID:
                    data.update(volume=float(response.volume), aabb_local_min=values(response.aabb_local_min), aabb_local_max=values(response.aabb_local_max))
                item.setdefault("collider_queries", []).append(data)

            get_physx_property_query_interface().query_prim(
                stage_id=stage_id, prim_id=PhysicsSchemaTools.sdfPathToInt(body.GetPath()),
                timeout_ms=30_000, rigid_body_fn=mass_received, collider_fn=collider_received,
                finished_fn=lambda: finished.__setitem__(0, True),
            )
            deadline = time.monotonic() + 35
            while not finished[0] and time.monotonic() < deadline:
                app.update()
            if not finished[0] or item.get("mass_error") or "mass_query" not in item:
                raise ValueError(f"Source mass query failed: {name}")
            if item["mass_query"]["mass_kg"] <= 0 or min(item["mass_query"]["inertia"]) <= 0:
                raise ValueError(f"Nonpositive source dynamic mass or inertia: {name}")
            if omni.timeline.get_timeline_interface().is_playing():
                raise ValueError("Unexpected timeline start during asset query")
            item["query_succeeded"] = True
            save(output, receipt)
            print(f"G1_ASSET_QUERY complete {name} mass={item['mass_query']['mass_kg']} hulls={sum(len(m['hulls']) for m in item['collision_meshes'])}", flush=True)
        receipt["all_asset_queries_succeeded"] = True
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        # Persist query results even if Kit shutdown exits the interpreter.
        receipt["closed_timeline_was_stopped"] = not omni.timeline.get_timeline_interface().is_playing()
        receipt["shutdown_mode"] = "documented_SimulationApp_immediate_exit_after_flushed_receipt"
        save(output, receipt)
        # No replicator or running timeline exists in this process. The installed
        # Kit build's graceful viewport teardown asserts on a busy task group.
        # Use its documented immediate exit and preserve diagnostic failure status.
        exit_code = 0 if receipt.get("all_asset_queries_succeeded") and "error" not in receipt else 1
        supports_exit_code = 'exit_code' in inspect.signature(app.close).parameters
        receipt['shutdown_exit_code_parameter_supported'] = supports_exit_code
        receipt['requested_exit_code'] = exit_code
        save(output, receipt)
        if supports_exit_code:
            app.close(wait_for_replicator=False, skip_cleanup=True, exit_code=exit_code)
        else:
            # Pinned 6.0's public close has no exit_code keyword. Preserve the
            # flushed outcome and use its supported shutdown signature.
            app.close(wait_for_replicator=False, skip_cleanup=True)
            if exit_code:
                raise SystemExit(exit_code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cook-source", action="store_true")
    args = parser.parse_args()
    receipt = {"scope": "original_task_object_cooking_and_mass_only", "qualified": False, "source_task_rollout_verified": False,
               "physics_integrations": 0, "timeline_started": False, "api_sources": API_SOURCES,
               "t1_matching_runtime_verified": False, "t2_matching_runtime_verified": False, "t1_expected_runtime": "6.0.0-dev2", "exporter_sha256": digest(Path(__file__)), "assets": []}
    with args.output.open("x") as output:
        try:
            inventory = json.loads(args.inventory.read_text())
            for name, (expected, scale) in ASSETS.items():
                item = next(r for r in inventory["assets"] if r["name"] == name)
                path = Path(item["local_path"])
                if digest(path) != expected:
                    raise ValueError(f"Original task object bytes changed: {name}")
                receipt["assets"].append({"name": name, "local_path": str(path), "usd_sha256": expected,
                                          "source_url": item["url"], "task_scale_override": list(scale), "query_succeeded": False})
            save(output, receipt)
            if args.cook_source:
                query(args, receipt, output)
        except BaseException as error:
            receipt["error"] = repr(error)
            save(output, receipt)
            raise
        finally:
            save(output, receipt)


if __name__ == "__main__":
    main()

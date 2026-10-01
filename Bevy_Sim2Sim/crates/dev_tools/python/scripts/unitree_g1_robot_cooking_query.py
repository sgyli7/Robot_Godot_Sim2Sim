#!/usr/bin/env python3
"""Query frozen G1 PhysX convex cooking without running a physics timeline.

The installed source runtime and original mesh approximation are recorded.
This is geometry evidence only, not a matched T1 rollout or native qualification.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

from unitree_g1_task_asset_query import digest, save, values


USD_SHA256 = "a7a2bab76981d19a1d76adecdfffec9b52afa34df9ba8e288ccedf410d3ce6bd"


def query(args, receipt, output):
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "disable_viewport_updates": True})
    try:
        import omni.timeline
        import omni.usd
        from omni.physx import get_physx_cooking_interface
        from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
        from pxr import Gf, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdUtils

        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_playing():
            raise ValueError("Robot cooking query requires a stopped timeline")
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, "Z")
        UsdPhysics.Scene.Define(stage, "/PhysicsScene")
        robot = stage.DefinePrim("/Robot")
        robot.GetReferences().AddReference(str(args.usd))
        cache = UsdGeom.XformCache()
        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        if stage_id < 0:
            raise ValueError("Source stage has no valid cache identity")
        receipt["runtime_build"] = (Path(sys.modules["isaacsim"].__file__).parent / "VERSION").read_text().strip()
        receipt["isaacsim_package_version"] = importlib.metadata.version("isaacsim")
        receipt["openusd_version"] = list(Usd.GetVersion())
        subtree = list(Usd.PrimRange(robot))
        bodies = {str(p.GetPath()): p for p in subtree if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        if len(bodies) != 53:
            raise ValueError("Original robot body topology changed")
        for prim in subtree:
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                raise ValueError("Disabled collider must not be exported as physical")
            owner = prim
            while owner and str(owner.GetPath()) not in bodies:
                owner = owner.GetParent()
            if not owner or not prim.IsA(UsdGeom.Mesh):
                raise ValueError("Query requires an owned original collision mesh")
            approximation = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
            if approximation != "convexHull":
                raise ValueError("Original robot collision approximation changed")
            relative = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(owner).GetInverse()
            item = {"body_name": owner.GetName(), "body_path": str(owner.GetPath()),
                    "collision_path": str(prim.GetPath()), "approximation": approximation,
                    "mesh_to_body_matrix": [values(row) for row in relative], "hulls": []}
            receipt["collisions"].append(item)

            def cooked(result, convexes):
                if result != PhysxCollisionRepresentationResult.RESULT_VALID:
                    item["error"] = str(result)
                    return
                for convex in convexes:
                    mesh_vertices = [values(v) for v in convex.vertices]
                    item["hulls"].append({"vertices_mesh_local": mesh_vertices,
                        "vertices_body_local": [values(relative.Transform(Gf.Vec3d(*v))) for v in mesh_vertices],
                        "indices": [int(i) for i in convex.indices],
                        "polygons": [{"index_base": int(p.index_base), "num_vertices": int(p.num_vertices),
                                      "plane": values(p.plane)} for p in convex.polygons]})

            get_physx_cooking_interface().request_convex_collision_representation(
                stage_id=stage_id, collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),
                run_asynchronously=False, on_result=cooked,
            )
            if item.get("error") or len(item["hulls"]) != 1:
                raise ValueError(f"Expected one original cooked convex hull: {item['collision_path']}")
            if timeline.is_playing():
                raise ValueError("Unexpected physics timeline start")
            save(output, receipt)
            print(f"G1_ROBOT_COOK body={item['body_name']} vertices={len(item['hulls'][0]['vertices_body_local'])}", flush=True)
        if len(receipt["collisions"]) != 52:
            raise ValueError("Original collision coverage changed")
        receipt["all_robot_queries_succeeded"] = True
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        receipt["closed_timeline_was_stopped"] = not omni.timeline.get_timeline_interface().is_playing()
        save(output, receipt)
        app.close(wait_for_replicator=False, skip_cleanup=True,
                  exit_code=0 if receipt.get("all_robot_queries_succeeded") and "error" not in receipt else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cook-source", action="store_true")
    args = parser.parse_args()
    receipt = {"schema": "g1_robot_physx_cooking_query_v1", "qualified": False,
               "source_task_rollout_verified": False, "t1_matching_runtime_verified": False,
               "t1_expected_runtime": "6.0.0-dev2", "physics_integrations": 0,
               "timeline_started": False, "exporter_sha256": digest(Path(__file__)),
               "usd_path": str(args.usd), "usd_sha256": USD_SHA256, "collisions": []}
    started = time.monotonic()
    with args.output.open("x") as output:
        try:
            if digest(args.usd) != USD_SHA256:
                raise ValueError("Frozen original robot USD identity changed")
            save(output, receipt)
            if args.cook_source:
                query(args, receipt, output)
        except BaseException as error:
            receipt["error"] = repr(error)
            raise
        finally:
            receipt["wall_seconds"] = time.monotonic() - started
            save(output, receipt)


if __name__ == "__main__":
    main()

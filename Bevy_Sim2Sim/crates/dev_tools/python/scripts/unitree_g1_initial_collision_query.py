"""Read existing original-source collision geometry at reset; never step or write USD."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from unitree_g1_task_asset_query import digest, values


def query_initial_collisions(raw, measured, output_path):
    """Use measured dynamic body poses and original composed static transforms.

    The caller verifies complete physical state and integration counters before
    and after this synchronous query. Results are acceptance-only, never policy
    observations. No query support, collision or report schema is enabled here.
    """
    import numpy as np
    import omni.usd
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from pxr import Gf, PhysicsSchemaTools, UsdGeom, UsdPhysics, UsdUtils

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.XformCache()
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    if stage_id < 0:
        raise ValueError('Initial geometry query requires the actual SDK stage')
    robot = raw.scene['robot']
    poses = measured(robot.data.body_link_pose_w)
    if poses.shape != (53, 7) or len(robot.body_names) != 53:
        raise ValueError('Original T2 measured body topology changed')
    body_poses = dict(zip(robot.body_names, poses.tolist(), strict=True))
    background_views = {}

    def pose_matrix(pose):
        matrix = Gf.Matrix4d(1)
        matrix.SetRotate(Gf.Quatd(pose[6], Gf.Vec3d(*pose[3:6])))
        matrix.SetTranslateOnly(Gf.Vec3d(*pose[:3]))
        return matrix

    def physics_attributes(prim):
        return {str(attr.GetName()): str(attr.Get()) for attr in prim.GetAttributes()
                if str(attr.GetName()).startswith(('physics:', 'physx'))}

    def body_owner(prim):
        owner = prim
        while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
            owner = owner.GetParent()
        return owner

    def transform(prim):
        owner = body_owner(prim)
        if owner:
            relative = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(owner).GetInverse()
            path = str(owner.GetPath())
            if '/Robot/' in path:
                if owner.GetName() not in body_poses:
                    raise ValueError(f'Unknown original T2 body: {path}')
                pose = body_poses[owner.GetName()]
            elif '/brown_box/' in path:
                pose = measured(raw.scene['brown_box'].data.root_link_pose_w).tolist()
            elif '/blue_sorting_bin/' in path:
                pose = measured(raw.scene['blue_sorting_bin'].data.root_link_pose_w).tolist()
            elif '/galileo_locomanip/' in path:
                # The original background also owns a dynamic tabletop and
                # power drill. Read their actual tensors rather than treating
                # their authored transforms as current physical state.
                if path not in background_views:
                    view = raw.sim.physics_manager.get_physics_sim_view().create_rigid_body_view(path)
                    pose = measured(view.get_transforms()).tolist()
                    if len(pose) != 7:
                        raise ValueError(f'Unexpected background body transform: {path}')
                    background_views[path] = (view, pose)
                pose = background_views[path][1]
            else:
                raise ValueError(f'Unmeasured dynamic collider owner: {path}')
            return relative * pose_matrix(pose), {'body_path': path, 'measured_body_pose_xyzw': pose,
                'shape_to_body_matrix': [values(row) for row in relative]}
        return cache.GetLocalToWorldTransform(prim), {'static_composed_usd_transform': True}

    def bounds(vertices):
        return [np.min(vertices, axis=0).tolist(), np.max(vertices, axis=0).tolist()]

    def authored_vertices(prim):
        if prim.IsA(UsdGeom.Mesh):
            return UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if prim.IsA(UsdGeom.Cube):
            half = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
            return [Gf.Vec3d(*xyz) for xyz in itertools.product((-half, half), repeat=3)]
        return None

    result = {'schema': 'g1_initial_source_collision_query_v1', 'qualified': False,
        'scope': 'actual SDK reset shapes and measured body poses; no physics or USD mutation',
        'query_sha256': digest(Path(__file__)), 'scene_physics_manager': str(type(raw.sim.physics_manager)),
        'sdk_counters_at_query': [int(raw._sim_step_counter), int(raw.sim._physics_step_count)],
        'robot_body_poses_xyzw': body_poses, 'robot_config_prim_path': robot.cfg.prim_path,
        'colliders': [], 'unsupported': [], 'disabled_colliders': [],
        'box_pose_xyzw': measured(raw.scene['brown_box'].data.root_link_pose_w).tolist(),
        'scene_query_support_enabled': raw.cfg.sim.enable_scene_query_support,
        'api_source': 'https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physx/docs/api/python.html',
        'limitation': 'Cooking API representation, not live contact attribution; triangle broadphase uses authored vertices. Sensor forces are separate evidence.'}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith('/World/envs/env_0/') or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            result['disabled_colliders'].append(path)
            continue
        tf, owner_info = transform(prim)
        item = {'path': path, 'type': str(prim.GetTypeName()), **owner_info,
                'shape_to_world_matrix': [values(row) for row in tf],
                'physics_attributes': physics_attributes(prim), 'hulls': []}
        result['colliders'].append(item)
        vertices = authored_vertices(prim)
        if vertices is not None:
            points = [values(tf.Transform(Gf.Vec3d(*v))) for v in vertices]
            item['authored_world_aabb'] = bounds(points)
            if '/brown_box/' in path:
                result['box_world_vertices'] = points
                result['box_world_aabb'] = item['authored_world_aabb']
        elif prim.GetTypeName() in ('Sphere', 'Capsule'):
            item['analytic_attributes'] = {key: str(prim.GetAttribute(key).Get()) for key in ('radius', 'height', 'axis')
                                            if prim.GetAttribute(key)}
        else:
            result['unsupported'].append(path)
    if 'box_world_aabb' not in result:
        raise ValueError('Original source box collision was not found')
    box_bounds = result['box_world_aabb']
    for item in result['colliders']:
        bb = item.get('authored_world_aabb')
        overlaps = bool(bb and all(min(bb[1][i], box_bounds[1][i]) > max(bb[0][i], box_bounds[0][i])
                                  for i in range(3)))
        item['authored_aabb_overlaps_box'] = overlaps
        # All robot hulls are exported to cover initial articulated poses;
        # background cooking is bounded to original AABB candidates.
        if not ('/Robot/' in item['path'] or overlaps) or item['type'] != 'Mesh':
            continue
        prim = stage.GetPrimAtPath(item['path'])
        approximation = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
        item['approximation'] = approximation
        tf = Gf.Matrix4d(item['shape_to_world_matrix'])
        if approximation == 'none':
            mesh = UsdGeom.Mesh(prim)
            item['authored_triangle_world_vertices'] = [values(tf.Transform(Gf.Vec3d(*v))) for v in mesh.GetPointsAttr().Get()]
            item['face_vertex_indices'] = list(mesh.GetFaceVertexIndicesAttr().Get())
            item['face_vertex_counts'] = list(mesh.GetFaceVertexCountsAttr().Get())
            continue
        if approximation not in ('convexHull', 'convexDecomposition'):
            raise ValueError(f'Unimplemented original approximation: {approximation}')
        def cooked(status, convexes):
            item['cooking_result'] = str(status)
            if status != PhysxCollisionRepresentationResult.RESULT_VALID:
                return
            for convex in convexes:
                local = [values(v) for v in convex.vertices]
                world = [values(tf.Transform(Gf.Vec3d(*v))) for v in local]
                item['hulls'].append({'vertices_shape_local': local, 'vertices_world': world,
                    'world_aabb': bounds(world), 'indices': [int(i) for i in convex.indices],
                    'polygons': [{'index_base': int(p.index_base), 'num_vertices': int(p.num_vertices),
                                  'plane_shape_local': values(p.plane)} for p in convex.polygons]})
        get_physx_cooking_interface().request_convex_collision_representation(
            stage_id=stage_id, collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),
            run_asynchronously=False, on_result=cooked)
        if not item['hulls']:
            raise ValueError(f'Actual source collision cooking failed: {item["path"]}')
        print(f'G1_INITIAL_COLLISION path={item["path"]} hulls={len(item["hulls"])}', flush=True)
    result['measured_background_body_poses_xyzw'] = {path: pose for path, (_, pose) in background_views.items()}
    if any(not np.array_equal(measured(view.get_transforms()), np.asarray(pose))
           for view, pose in background_views.values()):
        raise ValueError('Initial query changed background body poses')
    with output_path.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    return {'path': str(output_path), 'sha256': digest(output_path),
            'enabled_colliders': len(result['colliders']), 'unsupported_colliders': result['unsupported'],
            'cooked_hulls': sum(len(item['hulls']) for item in result['colliders'])}

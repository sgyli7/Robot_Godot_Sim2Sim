"""Read all released T2 background owners/shapes; never write or step physics."""
from pathlib import Path
import json
import time

from unitree_g1_task_asset_query import digest, values


def query_background(raw, measured, output_path):
    import numpy as np
    import omni.usd
    from omni.physx import get_physx_cooking_interface, get_physx_property_query_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult, PhysxPropertyQueryResult
    from pxr import Gf, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.XformCache()
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    if stage_id < 0:
        raise ValueError('Released background requires actual cached SDK stage')
    prefix = '/World/envs/env_0/galileo_locomanip/'
    views = {}
    default = raw.cfg.sim.physics_material
    result = {'schema': 'g1_released_t2_background_query_v1', 'qualified': False,
        'query_sha256': digest(Path(__file__)), 'source_runtime': Path('/isaac-sim/VERSION').read_text().strip(),
        'physics_integrations': 0, 'usd_writes': 0, 'default_material_cfg': default.to_dict(),
        'colliders': [], 'dynamic_owners': {}, 'disabled': [], 'completed': False,
        'isolated_box_query_scene_integrations': 0, 'isolated_box_query_body_count': 0}

    def save():
        output_path.write_text(json.dumps(result, indent=2, allow_nan=False, default=str) + '\n')

    def owner_of(prim):
        owner = prim
        while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
            owner = owner.GetParent()
        return owner

    def pose_matrix(pose):
        matrix = Gf.Matrix4d(1)
        matrix.SetRotate(Gf.Quatd(pose[6], Gf.Vec3d(*pose[3:6])))
        matrix.SetTranslateOnly(Gf.Vec3d(*pose[:3]))
        return matrix

    def body_info(owner):
        path = str(owner.GetPath())
        if path not in views:
            api = UsdPhysics.RigidBodyAPI(owner)
            if not api.GetRigidBodyEnabledAttr().Get() or api.GetKinematicEnabledAttr().Get():
                raise ValueError('Unexpected disabled/kinematic background owner')
            view = raw.sim.physics_manager.get_physics_sim_view().create_rigid_body_view(path)
            if view.count != 1 or list(view.prim_paths) != [path]:
                raise ValueError('Background owner view identity changed')
            pose = measured(view.get_transforms()).copy()
            velocity = measured(view.get_velocities()).copy()
            mass = float(measured(view.get_masses()).reshape(-1)[0])
            com = measured(view.get_coms()).copy()
            inertia = measured(view.get_inertias()).copy()
            materials = measured(view.get_material_properties()).copy()
            if (pose.shape != (7,) or com.shape != (7,) or inertia.shape != (9,)
                    or mass <= 0 or not all(np.isfinite(x).all() for x in (pose, velocity, com, inertia, materials))):
                raise ValueError('Invalid actual background body parameters')
            result['dynamic_owners'][path] = {'pose_xyzw': pose.tolist(), 'velocity_world': velocity.tolist(),
                'mass_kg': mass, 'com_pose_xyzw': com.tolist(), 'inertia_body_frame_column_major': inertia.tolist(),
                'compiled_shape_materials': materials.tolist(), 'ccd_enabled': bool(owner.GetAttribute('physxRigidBody:enableCCD').Get()),
                'linear_damping': float(owner.GetAttribute('physxRigidBody:linearDamping').Get() or 0.),
                'angular_damping': float(owner.GetAttribute('physxRigidBody:angularDamping').Get() or 0.)}
            views[path] = (view, pose, velocity)
        return path, result['dynamic_owners'][path]

    def material(prim):
        bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')[0]
        if bound and bound.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
            api = UsdPhysics.MaterialAPI(bound.GetPrim())
            combine = lambda name, fallback: str(bound.GetPrim().GetAttribute(name).Get() or fallback)
            return {'static_friction': float(api.GetStaticFrictionAttr().Get()),
                'dynamic_friction': float(api.GetDynamicFrictionAttr().Get()),
                'restitution': float(api.GetRestitutionAttr().Get()),
                'friction_combine': combine('physxMaterial:frictionCombineMode', 'average'),
                'restitution_combine': combine('physxMaterial:restitutionCombineMode', 'average'),
                'origin': str(bound.GetPath())}
        return {'static_friction': float(default.static_friction), 'dynamic_friction': float(default.dynamic_friction),
            'restitution': float(default.restitution), 'friction_combine': default.friction_combine_mode,
            'restitution_combine': default.restitution_combine_mode, 'origin': 'actual scene default physics material cfg'}

    # Property queries require a RigidBodyAPI, absent on released static boxes.
    # Query copied collision geometry in an unattached, never-integrated stage.
    # The live source stage and every physical state remain untouched.
    box_stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(box_stage, UsdGeom.GetStageMetersPerUnit(stage))
    UsdGeom.SetStageUpAxis(box_stage, UsdGeom.GetStageUpAxis(stage))
    box_cache_id = UsdUtils.StageCache.Get().Insert(box_stage).ToLongInt()
    save()
    try:
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(prefix) or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                result['disabled'].append(path); continue
            owner = owner_of(prim)
            if owner:
                owner_path, info = body_info(owner)
                # Physical poses have no scale. Preserve the tabletop's .7 Z
                # scale in collision vertices rather than cancelling it with
                # the full authored owner-matrix inverse.
                authored = Gf.Transform(cache.GetLocalToWorldTransform(owner))
                authored_rigid = Gf.Matrix4d(1)
                authored_rigid.SetRotate(authored.GetRotation())
                authored_rigid.SetTranslateOnly(authored.GetTranslation())
                measured_rigid = pose_matrix(info['pose_xyzw'])
                info['reset_authored_to_measured_rigid_matrix_max_abs'] = max(
                    abs(float(authored_rigid[r][c]) - float(measured_rigid[r][c]))
                    for r in range(4) for c in range(4))
                # Background bodies are not reset by the task factory. Retain
                # their actual measured reset pose/velocity, including startup
                # settling, rather than snapping them back to authored poses.
                relative = cache.GetLocalToWorldTransform(prim) * authored_rigid.GetInverse()
                tf = relative * pose_matrix(info['pose_xyzw'])
                info['authored_scale_baked_into_shapes'] = values(authored.GetScale())
            else:
                owner_path = None; tf = cache.GetLocalToWorldTransform(prim)
                relative = tf
            item = {'path': path, 'owner_path': owner_path, 'type': str(prim.GetTypeName()),
                'shape_to_world_matrix': [values(row) for row in tf],
                'shape_to_owner_matrix': [values(row) for row in relative],
                'material': material(prim), 'hulls': []}
            result['colliders'].append(item)
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                approximation = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
                item['approximation'] = approximation
                if approximation == 'none':
                    item.update(vertices_shape_local=[values(p) for p in mesh.GetPointsAttr().Get()],
                        face_vertex_indices=list(mesh.GetFaceVertexIndicesAttr().Get()),
                        face_vertex_counts=list(mesh.GetFaceVertexCountsAttr().Get()))
                elif approximation == 'boundingCube':
                    # The convex API explicitly rejects original box fits.
                    # Retain SDK box pose/bounds/volume, without substituting an
                    # authored mesh AABB for that physical representation.
                    responses = []; finished = [False]
                    number = result['isolated_box_query_body_count']
                    body = UsdGeom.Xform.Define(box_stage, '/body_' + str(number))
                    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
                    copy = UsdGeom.Mesh.Define(box_stage, str(body.GetPath()) + '/shape')
                    copy.CreatePointsAttr(mesh.GetPointsAttr().Get())
                    copy.CreateFaceVertexCountsAttr(mesh.GetFaceVertexCountsAttr().Get())
                    copy.CreateFaceVertexIndicesAttr(mesh.GetFaceVertexIndicesAttr().Get())
                    copy.AddTransformOp().Set(tf)
                    UsdPhysics.CollisionAPI.Apply(copy.GetPrim())
                    UsdPhysics.MeshCollisionAPI.Apply(copy.GetPrim()).CreateApproximationAttr('boundingCube')
                    result['isolated_box_query_body_count'] += 1
                    def collider(response):
                        value = {'result': str(response.result),
                            'path': str(PhysicsSchemaTools.intToSdfPath(response.path_id))}
                        if response.result == PhysxPropertyQueryResult.VALID:
                            value.update(aabb_local_min=values(response.aabb_local_min),
                                aabb_local_max=values(response.aabb_local_max),
                                local_position=values(response.local_pos), local_rotation=values(response.local_rot),
                                volume=float(response.volume))
                        responses.append(value)
                    get_physx_property_query_interface().query_prim(stage_id=box_cache_id,
                        prim_id=PhysicsSchemaTools.sdfPathToInt(body.GetPath()), timeout_ms=1000,
                        collider_fn=collider, finished_fn=lambda: finished.__setitem__(0, True))
                    deadline = time.monotonic() + 2
                    while not finished[0] and time.monotonic() < deadline:
                        time.sleep(.005)  # release GIL only; no Kit/app/physics update
                    item['box_property_query'] = responses
                    item['box_property_query_finished'] = finished[0]
                    item['box_property_query_origin'] = 'isolated original mesh/approximation/world-transform copy, unattached stage'
                    item['vertices_shape_local'] = [values(p) for p in mesh.GetPointsAttr().Get()]
                    if (not finished[0] or len(responses) != 1 or responses[0].get('volume', 0) <= 0
                            or responses[0]['path'] != str(copy.GetPath())):
                        raise ValueError('Original fitted-box property query unavailable: ' + path + ' ' + str(responses))
                else:
                    if approximation not in ('convexHull', 'convexDecomposition', 'boundingCube'):
                        raise ValueError('Unsupported original background collision: ' + approximation)
                    def cooked(status, convexes):
                        item['cooking_status'] = str(status)
                        if status == PhysxCollisionRepresentationResult.RESULT_VALID:
                            for convex in convexes:
                                item['hulls'].append({'vertices_shape_local': [values(v) for v in convex.vertices],
                                    'indices': list(convex.indices), 'polygons': [{'index_base': int(p.index_base),
                                        'num_vertices': int(p.num_vertices)} for p in convex.polygons]})
                    get_physx_cooking_interface().request_convex_collision_representation(
                        stage_id=stage_id, collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),
                        run_asynchronously=False, on_result=cooked)
                    if not item['hulls']:
                        raise ValueError('Original background cooking unavailable: ' + path + ' ' + str(item.get('cooking_status')))
            elif prim.IsA(UsdGeom.Cube):
                item['size'] = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
            elif prim.GetTypeName() == 'Plane':
                item['plane_attributes'] = {name: str(prim.GetAttribute(name).Get()) for name in ('axis', 'width', 'length')}
            else:
                raise ValueError('Unsupported original background primitive: ' + path)
            if len(result['colliders']) % 25 == 0:
                save(); print('G1_BACKGROUND_QUERY colliders=' + str(len(result['colliders'])), flush=True)
        if len(result['colliders']) != 251 or len(views) != 2:
            raise ValueError('Released background collider/body coverage changed')
        result['completed'] = True
    except BaseException as error:
        result['error'] = repr(error); raise
    finally:
        UsdUtils.StageCache.Get().Erase(box_stage)
        result['dynamic_state_max_abs_change'] = max((max(float(np.abs(measured(view.get_transforms()) - pose).max()),
            float(np.abs(measured(view.get_velocities()) - velocity).max())) for view, pose, velocity in views.values()), default=0.)
        save()
    if result['dynamic_state_max_abs_change'] != 0:
        raise ValueError('Background reading changed actual dynamic state')
    return {'path': str(output_path), 'sha256': digest(output_path), 'colliders': len(result['colliders']),
        'dynamic_owners': len(views), 'cooked_hulls': sum(len(x['hulls']) for x in result['colliders']), 'completed': True}

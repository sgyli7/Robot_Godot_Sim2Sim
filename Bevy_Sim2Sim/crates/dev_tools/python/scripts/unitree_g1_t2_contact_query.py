"""Zero-step actual SDK contact/motor inputs; no write or inference."""
import json
from unitree_g1_task_asset_query import digest
from pathlib import Path


def make_box_hand_contact_view(raw):
    """Read existing reporting bodies via tensors; never enable a schema."""
    from pxr import PhysxSchema
    import omni.usd

    path = '/World/envs/env_0/brown_box/Geometry/brown_box'
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(path).HasAPI(PhysxSchema.PhysxContactReportAPI):
        raise ValueError('Original box must already report contacts; no API is added')
    hands = [name for name in raw.scene['robot'].body_names if 'hand_' in name]
    if len(hands) != 16:
        raise ValueError('Released hand body coverage changed')
    filters = ['/World/envs/env_0/Robot/' + name for name in hands]
    view = raw.sim.physics_manager.get_physics_sim_view().create_rigid_contact_view(
        path, filter_patterns=filters, max_contact_data_count=1024)
    if view.sensor_count != 1 or view.filter_count != 16:
        raise ValueError('Actual box/hand contact view coverage changed')
    return view, {'sensor_paths': list(view.sensor_paths),
                  'filter_paths': view.filter_paths, 'hand_body_names': hands,
                  'sensor_count': view.sensor_count, 'filter_count': view.filter_count,
                  'capacity': view.max_contact_data_count, 'schema_writes': 0,
                  'scope': 'actual last 200Hz source solve only, acceptance-only tensor observation'}


def sample_box_hand_contacts(view, dt):
    import numpy as np
    import warp as wp

    def array(value):
        if hasattr(value, 'torch'):
            value = value.torch
        elif isinstance(value, wp.array):
            value = wp.to_torch(value)
        return value.detach().cpu().numpy()

    net = array(view.get_net_contact_forces(dt))
    matrix = array(view.get_contact_force_matrix(dt))
    force, point, normal, separation, count, start = map(array, view.get_contact_data(dt))
    points = []
    for index in range(view.filter_count):
        size, offset = int(count[0, index]), int(start[0, index])
        if size < 0 or offset < 0 or offset + size > view.max_contact_data_count:
            raise ValueError('Contact tensor data exceeded bounded capacity')
        for i in range(offset, offset + size):
            points.append({'hand_filter_index': index, 'normal_force_n': float(force[i, 0]),
                           'position_world': point[i].tolist(), 'normal_world': normal[i].tolist(),
                           'separation_m': float(separation[i, 0])})
    if not np.isfinite(net).all() or not np.isfinite(matrix).all():
        raise ValueError('Nonfinite actual source contact force')
    return {'source_solve_dt_s': dt, 'box_net_force_world_n': net[0].tolist(),
            'box_hand_force_matrix_world_n': matrix[0].tolist(),
            'box_hand_contact_points': points}


def query_contacts(raw, measured, path):
    import numpy as np
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics
    robot=raw.scene['robot'];views={'robot':robot.root_physx_view,
        'brown_box':raw.scene['brown_box'].root_physx_view,
        'blue_sorting_bin':raw.scene['blue_sorting_bin'].root_physx_view}
    result={'schema':'g1_released_t2_actual_contact_parameters_v1','qualified':False,
        'query_sha256':digest(Path(__file__)),'physics_integrations':0,'usd_writes':0,
        'runtime_build':Path('/isaac-sim/VERSION').read_text().strip(),'views':{},'explicit_actuators':{},'authored_background_offsets':[]}
    for name,view in views.items():
        fields={key:np.asarray(measured(getattr(view,method)())).tolist() for key,method in
            [('contact_offsets','get_contact_offsets'),('rest_offsets','get_rest_offsets'),('materials','get_material_properties')]}
        result['views'][name]={'prim_paths':list(view.prim_paths),'max_shapes':view.max_shapes,**fields}
    for name,actuator in robot.actuators.items():
        result['explicit_actuators'][name]={'joint_names':list(actuator.joint_names),
            **{k:np.asarray(measured(getattr(actuator,k))).tolist() for k in ('stiffness','damping','effort_limit','velocity_limit')}}
    stage=omni.usd.get_context().get_stage()
    for p in stage.Traverse():
        if str(p.GetPath()).startswith('/World/envs/env_0/galileo_locomanip/') and p.HasAPI(UsdPhysics.CollisionAPI):
            api=PhysxSchema.PhysxCollisionAPI(p)
            result['authored_background_offsets'].append({'path':str(p.GetPath()),
                'contact_offset_input':api.GetContactOffsetAttr().Get(),'rest_offset_input':api.GetRestOffsetAttr().Get(),
                'scope':'authored/schema input, not measured effective static shape offset'})
    # Some schema offset inputs use -inf as an automatic/default sentinel.
    # Preserve that explicitly; it is not a finite physical offset or JSON NaN.
    def encode(value):
        if isinstance(value,float) and not np.isfinite(value):
            return {'sdk_non_finite_value':str(value)}
        if isinstance(value,dict):return {k:encode(v) for k,v in value.items()}
        if isinstance(value,list):return [encode(v) for v in value]
        return value
    path.write_text(json.dumps(encode(result),indent=2,allow_nan=False)+'\n')
    return {'path':str(path),'sha256':digest(path),'physics_integrations':0,'usd_writes':0}

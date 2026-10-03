"""Read existing SDK contact reporting; independent placement evidence only."""
from __future__ import annotations


def make_apple_contact_view(raw, apple_name, plate_name):
    from pxr import PhysxSchema
    import omni.usd

    apple = list(raw.scene[apple_name].root_physx_view.prim_paths)
    plate = list(raw.scene[plate_name].root_physx_view.prim_paths)
    robot_names = list(raw.scene['robot'].body_names)
    if len(apple) != 1 or len(plate) != 1 or len(robot_names) != 53:
        raise ValueError('Original T1 contact-body coverage changed')
    if not omni.usd.get_context().get_stage().GetPrimAtPath(apple[0]).HasAPI(
            PhysxSchema.PhysxContactReportAPI):
        raise ValueError('Original apple must already report contacts; no schema is added')
    filters = plate + ['/World/envs/env_0/Robot/' + name for name in robot_names]
    view = raw.sim.physics_manager.get_physics_sim_view().create_rigid_contact_view(
        apple[0], filter_patterns=filters, max_contact_data_count=4096)
    if view.sensor_count != 1 or view.filter_count != 54:
        raise ValueError('Original apple/plate/robot contact view coverage changed')
    return view, {'sensor_paths': list(view.sensor_paths), 'filter_paths': view.filter_paths,
                  'robot_body_names': robot_names, 'plate_filter_index': 0,
                  'sensor_count': 1, 'filter_count': 54, 'capacity': 4096,
                  'schema_writes': 0, 'extra_integrations': 0,
                  'scope': 'last original200Hz solve, acceptance-only observation'}


def sample_apple_contacts(view, dt):
    import numpy as np
    import warp as wp

    def array(value):
        if hasattr(value, 'torch'):
            value = value.torch
        elif isinstance(value, wp.array):
            value = wp.to_torch(value)
        return value.detach().cpu().numpy()

    matrix = array(view.get_contact_force_matrix(dt))
    force, point, normal, separation, count, start = map(array, view.get_contact_data(dt))
    if matrix.shape != (1, 54, 3) or not np.isfinite(matrix).all():
        raise ValueError('Invalid original apple contact-force matrix')
    points = []
    for index in range(54):
        size, offset = int(count[0, index]), int(start[0, index])
        if size < 0 or offset < 0 or offset + size > view.max_contact_data_count:
            raise ValueError('Source contact report exceeded capacity')
        for i in range(offset, offset + size):
            sample = {'filter_index': index, 'normal_force_n': float(force[i, 0]),
                      'point_world': point[i].tolist(), 'normal_world': normal[i].tolist(),
                      'separation_m': float(separation[i, 0])}
            if not np.isfinite([sample['normal_force_n'], sample['separation_m'],
                                *sample['point_world'], *sample['normal_world']]).all():
                raise ValueError('Nonfinite original contact report')
            points.append(sample)
    return {'source_solve_dt_s': dt, 'apple_filter_force_matrix_world_n': matrix[0].tolist(),
            'contact_points': points}

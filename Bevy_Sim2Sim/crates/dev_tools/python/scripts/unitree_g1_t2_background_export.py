#!/usr/bin/env python3
"""CPU-only conversion of the byte-bound released T2 zero-step query.

Retains all original collision nodes and measured dynamic mass/inertia. Static
polygon triangulation is an explicit fan diagnostic, not cooked PhysX parity.
Fitted boxes use SDK properties from an isolated, never-integrated geometry
copy. No live world, asset, model, pose or acceptance threshold is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ARENA = '8b4a3a47fc53de23e8205089d71109a2e2348acd'
BACKGROUND = '7e14dcfd948591b8fdae61d41b412097b39490022dfc25aab9b90b6884509051'
RUNTIME = '6.0.0-rc.22+release.33481.407f3ea1.gl'
PREFIX = '/World/envs/env_0/galileo_locomanip/'
BOX_TRIANGLES = [[0,3,2],[0,2,1],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                 [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
BOX_SIGNS = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                      [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(q):
    q = np.array(q, dtype=float)
    if q.shape != (4,) or abs(np.linalg.norm(q)-1) > 1e-5:
        raise ValueError('Invalid measured quaternion')
    x,y,z,w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def transform(points, matrix):
    p, m = np.array(points, dtype=float), np.array(matrix, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or m.shape != (4,4) or not np.isfinite(p).all() or not np.isfinite(m).all():
        raise ValueError('Invalid source coordinates')
    return p @ m[:3,:3] + m[3,:3]


def fan(indices, counts):
    if sum(counts) != len(indices) or any(c not in (3,4) for c in counts):
        raise ValueError('Unsupported original static polygon topology')
    start, triangles = 0, []
    for count in counts:
        face = indices[start:start+count]; start += count
        triangles.extend([[face[0],face[i],face[i+1]] for i in range(1,count-1)])
    return triangles


def export(query, receipt):
    if (query['schema'] != 'g1_released_t2_background_query_v1' or not query['completed']
            or query['qualified'] or query['source_runtime'] != RUNTIME
            or query['physics_integrations'] != 0 or query['usd_writes'] != 0
            or query['dynamic_state_max_abs_change'] != 0
            or query['isolated_box_query_scene_integrations'] != 0
            or len(query['colliders']) != 251 or len(query['dynamic_owners']) != 2
            or receipt['source_profile'] != 'release_0_2_1' or receipt['arena_commit'] != ARENA
            or receipt['task_assets_sha256']['background'] != BACKGROUND
            or receipt['actual_sdk_integrations'] != 0 or receipt['policy_calls'] != 0
            or receipt['background_owner_query']['physical_state_max_abs_change'] != 0):
        raise ValueError('Incomplete, foreign or mutated original query')
    bodies = [{'path': PREFIX+'native_static_group', 'dynamic':False,
               'root_pose':{'position':[0,0,0], 'rotation_wxyz':[1,0,0,0]},
               'linear_velocity':[0,0,0], 'angular_velocity':[0,0,0],
               'mass_kg':0, 'center_of_mass':[0,0,0], 'principal_inertia':[0,0,0],
               'principal_axes_wxyz':[1,0,0,0], 'ccd_enabled':False,
               'linear_damping':0, 'angular_damping':0, 'colliders':[]}]
    owners = {}
    for path, item in sorted(query['dynamic_owners'].items()):
        pose, com = item['pose_xyzw'], item['com_pose_xyzw']
        r = rotation(com[3:]); inertia = np.array(item['inertia_body_frame_column_major']).reshape((3,3),order='F')
        diagonal = r.T @ inertia @ r
        if np.max(np.abs(diagonal - np.diag(np.diag(diagonal)))) > 1e-5 or min(np.diag(diagonal)) <= 0:
            raise ValueError('Measured COM axes/inertia frame mismatch')
        body = {'path':path, 'dynamic':True,
                'root_pose':{'position':pose[:3], 'rotation_wxyz':[pose[6],*pose[3:6]]},
                'linear_velocity':item['velocity_world'][:3], 'angular_velocity':item['velocity_world'][3:],
                'mass_kg':item['mass_kg'], 'center_of_mass':com[:3], 'principal_inertia':np.diag(diagonal).tolist(),
                'principal_axes_wxyz':[com[6],*com[3:6]], 'ccd_enabled':item['ccd_enabled'],
                'linear_damping':item['linear_damping'], 'angular_damping':item['angular_damping'], 'colliders':[]}
        bodies.append(body); owners[path] = body
    ground = None
    audit = {'triangulation':'authored static polygon fan; SDK cooked triangle parity unverified',
             'physics_parity_qualified':False, 'fitted_box_max_authored_enclosure_excess_m':0.,
             'fitted_box_max_relative_volume_error':0., 'fitted_boxes':0, 'cooked_convex_parts':0,
             'static_triangles':0, 'degenerate_static_triangles_removed':0}
    paths = set()
    for item in query['colliders']:
        if item['path'] in paths or not item['path'].startswith(PREFIX):
            raise ValueError('Duplicate/foreign original collider')
        paths.add(item['path'])
        material = {k:item['material'][k] for k in ('static_friction','dynamic_friction','restitution','friction_combine','restitution_combine')}
        if item['type'] == 'Plane':
            m = np.array(item['shape_to_world_matrix'])
            if ground or item['plane_attributes']['axis'] != 'Z' or np.max(np.abs(m[:3,:3]-np.eye(3))) > 1e-6:
                raise ValueError('Original ground plane changed')
            ground = {'path':item['path'], 'height_source':float(m[3,2]), 'material':material}; continue
        body = owners[item['owner_path']] if item['owner_path'] else bodies[0]
        matrix = item['shape_to_owner_matrix']
        parts = []
        approximation = item.get('approximation')
        if approximation == 'boundingCube':
            response = item['box_property_query'][0]
            m = np.array(item['shape_to_world_matrix'])
            half = (np.array(response['aabb_local_max'])-np.array(response['aabb_local_min']))*.5*np.linalg.norm(m[:3,:3],axis=1)
            r = rotation(response['local_rotation']); center = np.array(response['local_position'])
            actual = transform(item['vertices_shape_local'], m)
            excess = float(np.max(np.abs((actual-center)@r)-half))
            volume_error = abs(float(8*np.prod(half))-response['volume'])/response['volume']
            # SDK transform decomposition itself differs slightly on the authored
            # non-orthogonal transforms. Report that measured difference without
            # enlarging, inflating or snapping the returned physical box.
            if excess > .00015 or volume_error > 1e-4 or min(half) <= 0:
                raise ValueError('SDK fitted-box frame/volume audit failed: '+item['path'])
            audit['fitted_box_max_authored_enclosure_excess_m'] = max(audit['fitted_box_max_authored_enclosure_excess_m'],excess)
            audit['fitted_box_max_relative_volume_error'] = max(audit['fitted_box_max_relative_volume_error'],volume_error)
            points = (BOX_SIGNS*half)@r.T+center
            if item['owner_path']:
                pose = query['dynamic_owners'][item['owner_path']]['pose_xyzw']
                points = (points-np.array(pose[:3]))@rotation(pose[3:])
            parts = [{'points':points.tolist(), 'triangles':BOX_TRIANGLES}]
            audit['fitted_boxes'] += 1
        elif approximation in ('convexHull','convexDecomposition'):
            for hull in item['hulls']:
                triangles = []
                for polygon in hull['polygons']:
                    start, n = polygon['index_base'],polygon['num_vertices']
                    if n<3 or start<0 or start+n>len(hull['indices']):raise ValueError('Invalid cooked face')
                    face=hull['indices'][start:start+n]
                    triangles.extend([[face[0],face[i],face[i+1]] for i in range(1,n-1)])
                if np.linalg.det(np.array(matrix)[:3,:3])<0:triangles=[t[::-1] for t in triangles]
                parts.append({'points':transform(hull['vertices_shape_local'],matrix).tolist(),'triangles':triangles})
                audit['cooked_convex_parts'] += 1
        elif approximation == 'none':
            if item['owner_path']:raise ValueError('Original dynamic non-convex mesh unsupported')
            points = transform(item['vertices_shape_local'],matrix)
            triangles = fan(item['face_vertex_indices'],item['face_vertex_counts'])
            if any(min(t)<0 or max(t)>=len(points) for t in triangles):raise ValueError('Static triangle index out of range')
            valid = []
            for t in triangles:
                a,b,c=points[t]
                if len(set(t))<3 or np.linalg.norm(np.cross(b-a,c-a))<1e-14:
                    audit['degenerate_static_triangles_removed'] += 1; continue
                valid.append(t[::-1] if np.linalg.det(np.array(matrix)[:3,:3])<0 else t)
            parts = [{'points':points.tolist(),'triangles':valid}];audit['static_triangles']+=len(valid)
        elif item['type']=='Cube':
            parts=[{'points':transform(BOX_SIGNS*item['size']/2,matrix).tolist(),'triangles':BOX_TRIANGLES}]
        else:raise ValueError('Unsupported original source collision')
        body['colliders'].append({'path':item['path'],'geometry':'triangle_mesh' if approximation=='none' else 'convex_parts',
                                  'material':material,'parts':parts})
    if not ground or audit['fitted_boxes']!=130 or sum(len(b['colliders']) for b in bodies)!=250:
        raise ValueError('Released scene coverage incomplete')
    return bodies,ground,audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('query','source-receipt','background-usd','output'):parser.add_argument('--'+key,type=Path,required=True)
    for key in ('query-sha256','source-receipt-sha256'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    if digest(args.query)!=args.query_sha256 or digest(args.source_receipt)!=args.source_receipt_sha256 or digest(args.background_usd)!=BACKGROUND:
        raise ValueError('Input identity changed')
    query=json.loads(args.query.read_text());receipt=json.loads(args.source_receipt.read_text())
    if receipt['background_owner_query']['sha256']!=args.query_sha256:raise ValueError('Source receipt does not bind actual query')
    bodies,ground,audit=export(query,receipt)
    out={'schema':'native_g1_released_t2_background_v1','units':'metres_kilograms_radians_z_up',
         'source_arena_commit':ARENA,'source_background_sha256':BACKGROUND,'source_runtime_build':RUNTIME,
         'source_query_sha256':args.query_sha256,'source_receipt_sha256':args.source_receipt_sha256,
         'exporter_sha256':digest(Path(__file__)),'physics_parity_qualified':False,
         'bodies':bodies,'existing_ground_plane':ground,'audit':audit}
    with args.output.open('x') as handle:json.dump(out,handle,separators=(',',':'),allow_nan=False);handle.write('\n')
    print(json.dumps({'output':str(args.output),'sha256':digest(args.output),'bodies':len(bodies),'colliders':250,'audit':audit}))

if __name__=='__main__':main()

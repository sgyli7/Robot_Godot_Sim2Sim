#!/usr/bin/env python3
"""Static object localization from actual RGB, public labels and original self FK.

This declared classical tool accepts no simulator object pose, contact, root
world position, saved action or learned-policy output. It proposes no actuation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from unitree_g1_mobile_vision import DEFINITION_SHA256, root_from_camera, original_self_body_frames, rotation
from unitree_g1_static_label_fit import apple_label, register as register_public_label

SIDE_LAYOUT = {33:([-.030,0.,.014],[2**-.5,0.,-2**-.5,0.]),34:([.002,-.031,.014],[2**-.5,2**-.5,0.,0.]),35:([.033,0.,.014],[2**-.5,0.,2**-.5,0.]),36:([.002,.031,.014],[2**-.5,-2**-.5,0.,0.]),37:([.002,0.,-.017],[0.,1.,0.,0.])}
LAYOUT = {31: ('t1_apple', .02, [.002, 0., .046]), 32: ('t1_plate', .06, [0., 0., .0255])}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observation(path):
    if Path(path).stat().st_size > 128 * 1024:
        raise ValueError('Static self observation exceeds byte budget')
    value = json.loads(Path(path).read_text())
    if (set(value) != {'schema', 'stamp', 'camera', 'measured_joints'}
            or value['schema'] != 'g1_static_marker_observation_v1'):
        raise ValueError('Only static RGB calibration and original self sensors are admitted')
    stamp = value['stamp']
    if (set(stamp) != {'episode_id', 'frame_id', 'sim_time_ns', 'captured_at_unix_ms'}
            or any(type(stamp[k]) is not int for k in stamp)
            or any(stamp[k] <= 0 for k in ['episode_id', 'frame_id', 'captured_at_unix_ms'])
            or stamp['sim_time_ns'] < 0 or stamp['sim_time_ns'] % 20_000_000):
        raise ValueError('Invalid static frame/episode/time identity')
    state = value['measured_joints']
    if set(state) != {'positions', 'velocities', 'root_rotation_wxyz', 'root_angular_velocity_body', 'root_velocity_source'}:
        raise ValueError('Foreign self-state fields')
    for name, count in [('positions',43),('velocities',43),('root_angular_velocity_body',3),('root_velocity_source',3)]:
        a = np.asarray(state[name],dtype=np.float64)
        if a.shape != (count,) or not np.isfinite(a).all():
            raise ValueError(f'Invalid original self sensor:{name}')
    rotation(state['root_rotation_wxyz'])
    camera = value['camera']
    camera_fields = {'fx','fy','cx','cy','vertical_fov_radians','near_m','far_m'}
    if set(camera) not in (camera_fields, camera_fields | {'mount_profile'}):
        raise ValueError('Foreign camera fields')
    if camera.get('mount_profile', 'arena_ego') not in ('arena_ego', 'auxiliary_grip_overview', 'static_placement_overview'):
        raise ValueError('Unsupported published static camera mount')
    expected = {'fx':458.1245526,'fy':458.1245526,'cx':320.,'cy':240.,'near_m':.1,'far_m':5.}
    if not all(np.isfinite(camera[k]) for k in camera_fields) or any(abs(camera[k]-v)>1e-4 for k,v in expected.items()):
        raise ValueError('Original static pinhole calibration changed')
    if abs(camera['vertical_fov_radians'] - 2*np.arctan(240/camera['fy'])) > 1e-5:
        raise ValueError('Original static field of view changed')
    return value


class PinnedPublicStaticVisionAssets:
    """Verified public assets prepared before image capture; no task-world data.

    Every request rechecks the byte identities. Only public model loading is
    cached; FK still uses each frame's measured joints and changed files reject.
    """
    def __init__(self, definition_path, fiducial_path, fiducial_sha256):
        self._definition_path = Path(definition_path).resolve(strict=True)
        self._fiducial_path = Path(fiducial_path).resolve(strict=True)
        self._fiducial_sha256 = fiducial_sha256
        definition_path, fiducial_path = self._definition_path, self._fiducial_path
        if Path(definition_path).stat().st_size > 64*1024*1024 or sha(definition_path) != DEFINITION_SHA256:
            raise ValueError('Original public robot definition identity changed')
        if Path(fiducial_path).stat().st_size > 16*1024 or sha(fiducial_path) != fiducial_sha256:
            raise ValueError('Static printed calibration identity changed')
        calibration = json.loads(Path(fiducial_path).read_text())
        fields={'schema','dictionary','layout_profile','png_paths','png_sha256','calibration_version','marker_mounts_source_m'}
        multiface=calibration.get('layout_profile')=='static_apple_plate_multi_face'
        if (set(calibration) != (fields|{'static_apple_side_markers'} if multiface else fields)
                or calibration['schema'] != 'g1_task_fiducials_v1'
                or calibration['dictionary'] != 'DICT_4X4_50'
                or calibration['layout_profile'] not in ('static_apple_plate','static_apple_plate_multi_face')
                or calibration['calibration_version'] != (3 if multiface else 2)
                or calibration['marker_mounts_source_m'] != [LAYOUT[31][2],LAYOUT[32][2]]
                or len(calibration['png_paths']) != 2 or len(calibration['png_sha256']) != 2):
            raise ValueError('Foreign static label profile')
        self.layout={i:(kind,size,mount,np.eye(3)) for i,(kind,size,mount) in LAYOUT.items()}
        sides=calibration.get('static_apple_side_markers',[])
        if multiface:
            if len(sides)!=len(SIDE_LAYOUT):raise ValueError('Incomplete public multi-face layout')
            for side,(marker_id,(mount,q)) in zip(sides,SIDE_LAYOUT.items()):
                if (set(side)!={'marker_id','png_path','png_sha256','center_source_m','rotation_wxyz'}
                        or side['marker_id']!=marker_id or side['center_source_m']!=mount
                        or not np.allclose(side['rotation_wxyz'],q,atol=1e-7,rtol=0)):
                    raise ValueError('Public multi-face marker mount changed')
                self.layout[marker_id]=('t1_apple',.02,mount,rotation(q))
        for name, expected in zip(calibration['png_paths'],calibration['png_sha256']):
            p = Path(name)
            if not p.is_absolute():p=Path(fiducial_path).parent/p
            if p.stat().st_size > 1024*1024 or sha(p) != expected:
                raise ValueError('Public printed pixels changed')
        for side in sides:
            path=Path(side['png_path'])
            if not path.is_absolute():path=Path(fiducial_path).parent/path
            if path.stat().st_size>1024*1024 or sha(path)!=side['png_sha256']:raise ValueError('Public side marker pixels changed')
        self._definition = json.loads(Path(definition_path).read_text())
        self._files = [(self._definition_path, DEFINITION_SHA256, 64*1024*1024),
                       (self._fiducial_path, fiducial_sha256, 16*1024)]
        for name, expected in zip(calibration['png_paths'], calibration['png_sha256']):
            path = Path(name)
            if not path.is_absolute():
                path = self._fiducial_path.parent/path
            self._files.append((path, expected, 1024*1024))
        for side in sides:
            path=Path(side['png_path'])
            if not path.is_absolute():path=self._fiducial_path.parent/path
            self._files.append((path,side['png_sha256'],1024*1024))

    def verify_request(self, definition_path, fiducial_path, fiducial_sha256):
        if (Path(definition_path).resolve(strict=True) != self._definition_path
                or Path(fiducial_path).resolve(strict=True) != self._fiducial_path
                or fiducial_sha256 != self._fiducial_sha256):
            raise ValueError('Static request changed its prepared public asset identity')
        for path, expected, limit in self._files:
            if path.stat().st_size > limit or sha(path) != expected:
                raise ValueError('Prepared static public asset bytes changed')

    def root_camera(self, measured_positions, camera_profile='arena_ego'):
        if camera_profile == 'static_placement_overview':
            camera = root_from_camera(self._definition, measured_positions)
            head = original_self_body_frames(self._definition, measured_positions)[19]
            camera[:3, 3] += head[:3, :3] @ np.array([0., 0., .15])
            camera[:3, :3] = camera[:3, :3] @ cv2.Rodrigues(np.array([-25*np.pi/180, 0., 0.]))[0]
            return camera
        return root_from_camera(self._definition, measured_positions, camera_profile)


def localize(image_path, observation_path, definition_path, fiducial_path, fiducial_sha256,
             *, public_assets=None):
    value = observation(observation_path)
    if public_assets is None:
        public_assets = PinnedPublicStaticVisionAssets(definition_path, fiducial_path, fiducial_sha256)
    else:
        public_assets.verify_request(definition_path, fiducial_path, fiducial_sha256)
    if Path(image_path).stat().st_size > 16*1024*1024:
        raise ValueError('Static RGB exceeds finite byte budget')
    image = cv2.imread(str(image_path),cv2.IMREAD_COLOR)
    if image is None or image.shape != (480,640,3):
        raise ValueError('Static actual camera requires640x480PNG')
    camera_profile = value['camera'].get('mount_profile', 'arena_ego')
    root_camera=public_assets.root_camera(value['measured_joints']['positions'], camera_profile)
    c=value['camera'];K=np.array([[c['fx'],0,c['cx']],[0,c['fy'],c['cy']],[0,0,1.]])
    parameters=cv2.aruco.DetectorParameters();parameters.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
    detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),parameters)
    corners,ids,rejected=detector.detectMarkers(image);detections=[];seen=set()
    # The apple's full public pattern validates/refines accepted and rejected
    # contours. Plate localization retains the existing original corner path.
    refined=apple_label(image,corners,ids,rejected)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    candidates=[]
    for p,i in zip(corners,[] if ids is None else ids.flatten()):
        marker_id=int(i)
        if marker_id==31:continue
        if marker_id in SIDE_LAYOUT and marker_id in public_assets.layout:
            fit=register_public_label(gray,p.reshape(4,2),marker_id=marker_id)
            if fit is None:continue
            candidates.append((fit[0],marker_id,fit[1]))
        else:candidates.append((p,marker_id,None))
    if refined is not None:candidates.append((refined[0],31,refined[1]))
    for points,marker_id,registration in candidates:
        marker_id=int(marker_id)
        if marker_id not in public_assets.layout:continue
        if marker_id in seen:raise ValueError('Duplicate static target identity in actual RGB')
        seen.add(marker_id);kind,size,mount,mount_rotation=public_assets.layout[marker_id];pixels=points.reshape(4,2).astype(np.float64)
        edge=float(np.linalg.norm(pixels-np.roll(pixels,-1,axis=0),axis=1).min());h=size/2
        object_points=np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]])
        ok,rvecs,tvecs,_=cv2.solvePnPGeneric(object_points,pixels,K,None,flags=cv2.SOLVEPNP_IPPE_SQUARE)
        candidates=[]
        if ok:
            for rv,tv in zip(rvecs,tvecs):
                r=cv2.Rodrigues(rv)[0];t=tv.reshape(3)
                projected=cv2.projectPoints(object_points,rv,tv,K,None)[0].reshape(4,2)
                error=float(np.sqrt(np.mean(np.sum((projected-pixels)**2,axis=1))))
                if .1<t[2]<5. and float(r[:,2]@t)<0:
                    pose=np.eye(4);pose[:3,:3]=r;pose[:3,3]=t;candidates.append((error,pose))
        if not candidates or edge<8:continue
        error,pose=min(candidates,key=lambda x:x[0])
        if error>1:continue
        object_marker=np.eye(4);object_marker[:3,3]=mount;object_marker[:3,:3]=mount_rotation
        root_object=root_camera@pose@np.linalg.inv(object_marker)
        detection={'marker_id':marker_id,'object_kind':kind,'corners_px':pixels.tolist(),
            'minimum_edge_px':edge,'reprojection_rms_px':error,'root_from_object':root_object.tolist()}
        if registration is not None:detection['pixel_registration']=registration
        detections.append(detection)
    # The original calibration has one physical marker per object and retains
    # its exact output. Multi-face calibration jointly uses all admitted apple
    # faces; a largest-contour heuristic cannot resolve their depth uncertainty.
    if len(public_assets.layout)>2:
        apples=[d for d in detections if d['object_kind']=='t1_apple']
        others=[d for d in detections if d['object_kind']!='t1_apple']
        if len(apples)>1:
            joint=multi_face_apple_pose(apples,public_assets,root_camera,K)
            detections=others+([] if joint is None else [joint])
        else:detections=others+apples
    return {'schema':'g1_static_actual_rgb_localization_v1','observation':value['stamp'],
        'source':'actual_rgb_printed_label_pnp_and_original_self_FK',
        'image_sha256':sha(image_path),'input_sha256':sha(observation_path),
        'definition_sha256':DEFINITION_SHA256,'fiducial_sha256':fiducial_sha256,
        'camera_profile':camera_profile,'detections':detections,'rejected_marker_candidates':len(rejected),
        'world_or_contact_truth_input':False,'actuation_proposed':False,'task_qualified':False}


def multi_face_apple_pose(entries, public_assets, root_camera, K):
    """Joint rigid PnP of public noncoplanar printed faces, using pixels only."""
    points=[];pixels=[]
    for entry in entries:
        marker_id=entry['marker_id'];_,size,mount,R=public_assets.layout[marker_id];h=size/2
        quad=np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]])
        points.extend((quad@R.T+mount).tolist());pixels.extend(entry['corners_px'])
    points=np.array(points,np.float64);pixels=np.array(pixels,np.float64)
    ok,rvecs,tvecs,_=cv2.solvePnPGeneric(points,pixels,K,None,flags=cv2.SOLVEPNP_SQPNP)
    solutions=[]
    if ok:
        for rv,tv in zip(rvecs,tvecs):
            R=cv2.Rodrigues(rv)[0];t=tv.reshape(3)
            depths=(points@R.T+t)[:,2]
            if not np.all((depths>.1)&(depths<5.)):continue
            front=True
            for entry in entries:
                _,_,mount,mountR=public_assets.layout[entry['marker_id']]
                if float((R@mountR)[:,2]@(R@np.array(mount)+t))>=0:front=False
            if not front:continue
            projected=cv2.projectPoints(points,rv,tv,K,None)[0].reshape(-1,2)
            error=float(np.sqrt(np.mean(np.sum((projected-pixels)**2,axis=1))))
            if error>1.:continue
            pose=np.eye(4);pose[:3,:3]=R;pose[:3,3]=t;solutions.append((error,root_camera@pose))
    if not solutions:return None
    error,pose=min(solutions,key=lambda q:q[0])
    primary=next((e for e in entries if e['marker_id']==31),min(entries,key=lambda e:e['marker_id']))
    result=dict(primary)
    result.update(root_from_object=pose.tolist(),reprojection_rms_px=error,
        minimum_edge_px=min(e['minimum_edge_px'] for e in entries),
        physical_marker_ids_used=sorted(e['marker_id'] for e in entries),
        pose_method='public_nonplanar_multi_face_pattern_pnp_v1')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['image','observation','definition','fiducials','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--fiducial-sha256',required=True);a=p.parse_args()
    result=localize(a.image,a.observation,a.definition,a.fiducials,a.fiducial_sha256)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'detections':len(result['detections']),'task_qualified':False}),flush=True)

if __name__=='__main__':main()

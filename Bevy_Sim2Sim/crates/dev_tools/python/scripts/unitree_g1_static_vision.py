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
from unitree_g1_mobile_vision import DEFINITION_SHA256, root_from_camera, rotation

LAYOUT = {31: ('t1_apple', .02, [.002, 0., .046]), 32: ('t1_plate', .06, [0., 0., .0045])}


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
    if set(camera) != {'fx','fy','cx','cy','vertical_fov_radians','near_m','far_m'}:
        raise ValueError('Foreign camera fields')
    expected = {'fx':458.1245526,'fy':458.1245526,'cx':320.,'cy':240.,'near_m':.1,'far_m':5.}
    if not all(np.isfinite(v) for v in camera.values()) or any(abs(camera[k]-v)>1e-4 for k,v in expected.items()):
        raise ValueError('Original static pinhole calibration changed')
    if abs(camera['vertical_fov_radians'] - 2*np.arctan(240/camera['fy'])) > 1e-5:
        raise ValueError('Original static field of view changed')
    return value


def localize(image_path, observation_path, definition_path, fiducial_path, fiducial_sha256):
    value = observation(observation_path)
    if sha(definition_path) != DEFINITION_SHA256 or Path(definition_path).stat().st_size > 64*1024*1024:
        raise ValueError('Original public robot definition identity changed')
    if Path(fiducial_path).stat().st_size > 16*1024 or sha(fiducial_path) != fiducial_sha256:
        raise ValueError('Static printed calibration identity changed')
    calibration = json.loads(Path(fiducial_path).read_text())
    if (set(calibration) != {'schema','dictionary','layout_profile','png_paths','png_sha256'}
            or calibration['schema'] != 'g1_task_fiducials_v1'
            or calibration['dictionary'] != 'DICT_4X4_50'
            or calibration['layout_profile'] != 'static_apple_plate'
            or len(calibration['png_paths']) != 2 or len(calibration['png_sha256']) != 2):
        raise ValueError('Foreign static label profile')
    for name, expected in zip(calibration['png_paths'],calibration['png_sha256']):
        p = Path(name)
        if not p.is_absolute():p=Path(fiducial_path).parent/p
        if p.stat().st_size > 1024*1024 or sha(p) != expected:
            raise ValueError('Public printed pixels changed')
    if Path(image_path).stat().st_size > 16*1024*1024:
        raise ValueError('Static RGB exceeds finite byte budget')
    image = cv2.imread(str(image_path),cv2.IMREAD_COLOR)
    if image is None or image.shape != (480,640,3):
        raise ValueError('Static actual camera requires640x480PNG')
    definition=json.loads(Path(definition_path).read_text())
    root_camera=root_from_camera(definition,value['measured_joints']['positions'])
    c=value['camera'];K=np.array([[c['fx'],0,c['cx']],[0,c['fy'],c['cy']],[0,0,1.]])
    parameters=cv2.aruco.DetectorParameters();parameters.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
    detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),parameters)
    corners,ids,rejected=detector.detectMarkers(image);detections=[];seen=set()
    for points,marker_id in zip(corners,[] if ids is None else ids.flatten()):
        marker_id=int(marker_id)
        if marker_id not in LAYOUT:continue
        if marker_id in seen:raise ValueError('Duplicate static target identity in actual RGB')
        seen.add(marker_id);kind,size,mount=LAYOUT[marker_id];pixels=points.reshape(4,2).astype(np.float64)
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
        object_marker=np.eye(4);object_marker[:3,3]=mount
        root_object=root_camera@pose@np.linalg.inv(object_marker)
        detections.append({'marker_id':marker_id,'object_kind':kind,'corners_px':pixels.tolist(),
            'minimum_edge_px':edge,'reprojection_rms_px':error,'root_from_object':root_object.tolist()})
    return {'schema':'g1_static_actual_rgb_localization_v1','observation':value['stamp'],
        'source':'actual_rgb_printed_label_pnp_and_original_self_FK',
        'image_sha256':sha(image_path),'input_sha256':sha(observation_path),
        'definition_sha256':DEFINITION_SHA256,'fiducial_sha256':fiducial_sha256,
        'camera_profile':'arena_ego','detections':detections,'rejected_marker_candidates':len(rejected),
        'world_or_contact_truth_input':False,'actuation_proposed':False,'task_qualified':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['image','observation','definition','fiducials','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--fiducial-sha256',required=True);a=p.parse_args()
    result=localize(a.image,a.observation,a.definition,a.fiducials,a.fiducial_sha256)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'detections':len(result['detections']),'task_qualified':False}),flush=True)

if __name__=='__main__':main()

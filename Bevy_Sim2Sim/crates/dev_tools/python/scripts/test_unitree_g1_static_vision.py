import hashlib,json,math,tempfile,unittest
from pathlib import Path
import cv2,numpy as np
from unitree_g1_static_vision import observation,localize

class StaticObservationTests(unittest.TestCase):
    def fixture(self,root):
        value={'schema':'g1_static_marker_observation_v1','stamp':{'episode_id':1,'frame_id':1,'sim_time_ns':20000000,'captured_at_unix_ms':1},'camera':{'fx':458.1245526,'fy':458.1245526,'cx':320.,'cy':240.,'near_m':.1,'far_m':5.,'vertical_fov_radians':2*math.atan(240/458.1245526)},'measured_joints':{'positions':[0.]*43,'velocities':[0.]*43,'root_rotation_wxyz':[1.,0.,0.,0.],'root_angular_velocity_body':[0.]*3,'root_velocity_source':[0.]*3}}
        path=Path(root)/'observation.json';path.write_text(json.dumps(value));return path,value

    def test_world_truth_or_foreign_fields_cannot_enter_self_observation(self):
        with tempfile.TemporaryDirectory() as root:
            path,value=self.fixture(root);self.assertEqual(observation(path)['schema'],value['schema'])
            for field in ('task_objects','root_position_source','contacts','saved_action'):
                bad=dict(value);bad[field]={};path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):observation(path)

    def test_original_time_sensor_and_camera_contracts_are_required(self):
        with tempfile.TemporaryDirectory() as root:
            path,value=self.fixture(root)
            for mutation in [lambda x:x['stamp'].update(sim_time_ns=1),lambda x:x['camera'].update(fx=457.),lambda x:x['measured_joints']['positions'].__setitem__(0,float('nan')),lambda x:x['measured_joints'].update(velocities=[0.]*42)]:
                bad=json.loads(json.dumps(value));mutation(bad);path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):observation(path)

    def test_plain_actual_rgb_yields_no_object_pose_or_actuation(self):
        base=Path(__file__).resolve().parents[4];labels=base/'assets/g1_fiducials/static_apple_plate.json'
        definition=Path('/home/ethan/models/unitree_g1/homie_v2/g1_physics.json')
        if not definition.exists():self.skipTest('byte-bound original public model is not installed')
        with tempfile.TemporaryDirectory() as root:
            path,_=self.fixture(root);image=Path(root)/'rgb.png';cv2.imwrite(str(image),np.ones((480,640,3),np.uint8)*127)
            result=localize(image,path,definition,labels,hashlib.sha256(labels.read_bytes()).hexdigest())
            self.assertEqual(result['detections'],[]);self.assertFalse(result['actuation_proposed']);self.assertFalse(result['task_qualified'])

if __name__=='__main__':unittest.main()

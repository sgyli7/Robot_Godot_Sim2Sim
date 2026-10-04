import hashlib,json,math,tempfile,unittest
from pathlib import Path
import cv2,numpy as np
from unitree_g1_static_vision import PinnedPublicStaticVisionAssets,observation,localize
from unittest.mock import patch

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

    def test_only_published_camera_mounts_are_admitted(self):
        with tempfile.TemporaryDirectory() as root:
            path,value=self.fixture(root)
            value['camera']['mount_profile']='auxiliary_grip_overview'
            path.write_text(json.dumps(value))
            self.assertEqual(observation(path)['camera']['mount_profile'],'auxiliary_grip_overview')
            value['camera']['mount_profile']='static_placement_overview'
            path.write_text(json.dumps(value))
            self.assertEqual(observation(path)['camera']['mount_profile'],'static_placement_overview')
            value['camera']['mount_profile']='custom_pose'
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):observation(path)
            value['camera']['mount_profile']='auxiliary_grip_overview'
            value['camera']['root_from_camera']=[[1.,0.,0.,0.]]
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):observation(path)

    def test_stale_or_changed_print_mount_cannot_reinterpret_an_observation(self):
        base=Path(__file__).resolve().parents[4];labels=base/'crates/dev_tools/python/fixtures/g1_fiducials/static_apple_plate.json'
        definition=Path('/home/ethan/models/unitree_g1/homie_v2/g1_physics.json')
        if not definition.exists():self.skipTest('byte-bound original public model is not installed')
        with tempfile.TemporaryDirectory() as root:
            path,_=self.fixture(root);image=Path(root)/'rgb.png';cv2.imwrite(str(image),np.ones((480,640,3),np.uint8)*127)
            for mutation in [lambda x:x.update(calibration_version=1),lambda x:x['marker_mounts_source_m'][1].__setitem__(2,.0045)]:
                bad=json.loads(labels.read_text());mutation(bad);stale=Path(root)/'labels.json';stale.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError,'Foreign static label profile'):
                    localize(image,path,definition,stale,hashlib.sha256(stale.read_bytes()).hexdigest())

    def test_plain_actual_rgb_yields_no_object_pose_or_actuation(self):
        base=Path(__file__).resolve().parents[4];labels=base/'crates/dev_tools/python/fixtures/g1_fiducials/static_apple_plate.json'
        definition=Path('/home/ethan/models/unitree_g1/homie_v2/g1_physics.json')
        if not definition.exists():self.skipTest('byte-bound original public model is not installed')
        with tempfile.TemporaryDirectory() as root:
            path,_=self.fixture(root);image=Path(root)/'rgb.png';cv2.imwrite(str(image),np.ones((480,640,3),np.uint8)*127)
            result=localize(image,path,definition,labels,hashlib.sha256(labels.read_bytes()).hexdigest())
            self.assertEqual(result['detections'],[]);self.assertFalse(result['actuation_proposed']);self.assertFalse(result['task_qualified'])

    def test_prepared_public_model_uses_each_current_joint_state(self):
        base=Path(__file__).resolve().parents[4];labels=base/'crates/dev_tools/python/fixtures/g1_fiducials/static_apple_plate.json'
        definition=Path('/home/ethan/models/unitree_g1/homie_v2/g1_physics.json')
        if not definition.exists():self.skipTest('byte-bound original public model is not installed')
        assets=PinnedPublicStaticVisionAssets(definition,labels,hashlib.sha256(labels.read_bytes()).hexdigest())
        q=[0.]*43;before=assets.root_camera(q);q[12]=.1
        self.assertGreater(np.max(abs(before-assets.root_camera(q))),.01)

    def test_prepared_asset_does_not_reparse_model_or_accept_changed_identity(self):
        base=Path(__file__).resolve().parents[4];labels=base/'crates/dev_tools/python/fixtures/g1_fiducials/static_apple_plate.json'
        definition=Path('/home/ethan/models/unitree_g1/homie_v2/g1_physics.json')
        if not definition.exists():self.skipTest('byte-bound original public model is not installed')
        identity=hashlib.sha256(labels.read_bytes()).hexdigest()
        assets=PinnedPublicStaticVisionAssets(definition,labels,identity)
        with tempfile.TemporaryDirectory() as root:
            path,_=self.fixture(root);image=Path(root)/'rgb.png';cv2.imwrite(str(image),np.ones((480,640,3),np.uint8)*127)
            with patch('unitree_g1_static_vision.json.loads',wraps=json.loads) as parse:
                result=localize(image,path,definition,labels,identity,public_assets=assets)
                self.assertEqual(parse.call_count,1)  # Current observation, never the model again.
            self.assertEqual(result['detections'],[])
            with self.assertRaisesRegex(ValueError,'prepared public asset identity'):
                localize(image,path,definition,labels,'0'*64,public_assets=assets)

    def test_changed_bytes_after_preparation_reject_instead_of_using_cache(self):
        base=Path(__file__).resolve().parents[4];source=base/'crates/dev_tools/python/fixtures/g1_fiducials/static_apple_plate.json'
        definition=Path('/home/ethan/models/unitree_g1/homie_v2/g1_physics.json')
        if not definition.exists():self.skipTest('byte-bound original public model is not installed')
        with tempfile.TemporaryDirectory() as root:
            calibration=json.loads(source.read_text())
            calibration['png_paths']=[str((source.parent/p).resolve()) for p in calibration['png_paths']]
            labels=Path(root)/'labels.json';labels.write_text(json.dumps(calibration))
            identity=hashlib.sha256(labels.read_bytes()).hexdigest()
            assets=PinnedPublicStaticVisionAssets(definition,labels,identity)
            labels.write_text(labels.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'asset bytes changed'):
                assets.verify_request(definition,labels,identity)

if __name__=='__main__':unittest.main()

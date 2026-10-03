"""Actual bounded CPU-port tests; synthetic pixels, no robot/model/task trial."""
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
from unitree_g1_static_vision import sha


class StaticWorkerTests(unittest.TestCase):
    def setUp(self):
        self.definition=Path('/home/ethan/models/unitree_g1/homie_v2/g1_physics.json')
        if not self.definition.exists():self.skipTest('pinned public model not installed')
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.directory=self.root/'frame';self.directory.mkdir()
        cv2.imwrite(str(self.directory/'ego.png'),np.ones((480,640,3),np.uint8)*127)
        self.stamp={'episode_id':12,'frame_id':1,'sim_time_ns':0,'captured_at_unix_ms':1}
        value={'schema':'g1_static_marker_observation_v1','stamp':self.stamp,
            'camera':{'fx':458.1245526,'fy':458.1245526,'cx':320.,'cy':240.,'near_m':.1,'far_m':5.,
                      'vertical_fov_radians':2*math.atan(240/458.1245526)},
            'measured_joints':{'positions':[0.]*43,'velocities':[0.]*43,
                'root_rotation_wxyz':[1.,0.,0.,0.],'root_angular_velocity_body':[0.]*3,
                'root_velocity_source':[0.]*3}}
        (self.directory/'observation.json').write_text(json.dumps(value))
        self.request={'schema':'g1_persistent_static_marker_request_v1',
                      'directory':str(self.directory),'observation':self.stamp}

    def worker(self, episode=12):
        scripts=Path(__file__).parent;labels=scripts.parents[3]/'assets/g1_fiducials/static_apple_plate.json'
        env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        child=subprocess.Popen([sys.executable,str(scripts/'unitree_g1_static_vision_worker.py'),
            '--definition',str(self.definition),'--fiducials',str(labels),'--fiducial-sha256',sha(labels),
            '--output-root',str(self.root),'--episode',str(episode)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
        def cleanup():
            if child.poll() is None:child.terminate()
            child.communicate(timeout=5)
        self.addCleanup(cleanup)
        ready=self.read(child);self.assertEqual(ready['event'],'ready');return child

    def read(self, child):
        self.assertTrue(select.select([child.stdout],[],[],5)[0],'bounded worker response')
        return json.loads(child.stdout.readline())

    def send(self, child, request):
        child.stdin.write(json.dumps(request)+'\n');child.stdin.flush();return self.read(child)

    def test_tick_zero_accepted_then_repeated_frame_rejected(self):
        child=self.worker();result=self.send(child,self.request)
        self.assertEqual(result['event'],'result');self.assertEqual(result['observation'],self.stamp)
        self.assertEqual(result['result']['detections'],[]);self.assertFalse(result['actuation_proposed'])
        self.assertEqual(self.send(child,self.request)['event'],'error')
        self.assertEqual(child.wait(timeout=5),1)

    def test_reset_worker_cannot_accept_previous_episode(self):
        child=self.worker(13)
        self.assertEqual(self.send(child,self.request)['event'],'error')
        self.assertEqual(child.wait(timeout=5),1)
        self.assertFalse((self.directory/'localization.json').exists())

    def test_foreign_truth_fields_reject_before_localization(self):
        child=self.worker();request=dict(self.request);request['task_objects']={}
        self.assertEqual(self.send(child,request)['event'],'error')
        self.assertEqual(child.wait(timeout=5),1)
        self.assertFalse((self.directory/'localization.json').exists())


if __name__=='__main__':unittest.main()

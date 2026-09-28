"""Actual gates and atomic CPU authorizers; never run a learner or reserve real budget."""
import copy
import hashlib
import json
import multiprocessing
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from bevy_microduck_tools.authorization import LearningRequest, Rejection, authorize
from bevy_microduck_tools.serialization import identity, sha256_file, write_json
from bevy_microduck_tools.source_adapter import JOINT_ORDER
from bevy_microduck_tools.training import train_bounded
from bevy_microduck_tools.workflow import STAGES, TrainingBudget, admit, bind_candidate, bind_evidence


def permission(ledger, candidate_id='test_only_candidate', *, iterations=1, max_runs=1):
    return {'schema': 'microduck_root_review_v2', 'authorization_id': str(uuid.uuid4()),
            'candidate_id': candidate_id, 'reviewer_role':'root_gpt', 'decision':'approve',
            'reviewed_at':time.time()-1, 'expires_at':time.time()+120, 'authorized_stages': ['learning'], 'learning_allowed': True,
            'learning_limit': {'iterations': iterations, 'max_wall_seconds': 180, 'seed': 1000001,
                               'gpus': 1, 'max_runs': max_runs, 'budget_ledger_path': str(ledger.resolve())}}


def authorizer_probe(ledger, review, digest, barrier, queue):
    # This process only competes for a temporary ledger; no scientific worker exists.
    try:
        barrier.wait(timeout=10)
        request = LearningRequest(1, 10, 1000001, 1, ledger)
        run = TrainingBudget(Path(ledger)).reserve_authorized('test_only', review['candidate_id'], review, digest, request)
        queue.put({'reserved': True, 'run_id': run, 'learner_executed': False})
    except Rejection as error:
        queue.put({'reserved': False, 'reason': str(error), 'learner_executed': False})


class AuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.ledger = self.root / 'test_only_budget.json'
        self.request = LearningRequest(1, 180, 1000001, 1, str(self.ledger))
        self.review = permission(self.ledger)
        self.digest = hashlib.sha256(json.dumps(self.review).encode()).hexdigest()

    def test_capability_and_limit_refusals(self):
        cases = []
        for key, value in [('schema', 'microduck_root_review_v1'), ('authorization_id', 'not-a-uuid'),
                           ('authorized_stages', ['source_rollout']), ('authorized_stages', []),
                           ('authorized_stages', ['learning', 'learning']), ('authorized_stages', ['unknown']),
                           ('learning_allowed', False), ('learning_allowed', 1), ('learning_limit', None)]:
            review = copy.deepcopy(self.review); review[key] = value; cases.append((key, review))
        review = copy.deepcopy(self.review); del review['authorized_stages']; cases.append(('missing stage', review))
        for key, value in [('iterations', True), ('max_runs', 0), ('max_wall_seconds', float('nan')),
                           ('max_wall_seconds', float('inf')), ('budget_ledger_path', 'relative.json')]:
            review = copy.deepcopy(self.review); review['learning_limit'][key] = value; cases.append((key, review))
        review = copy.deepcopy(self.review); del review['learning_limit']['seed']; cases.append(('missing seed', review))
        for name, review in cases:
            with self.subTest(case=name), self.assertRaises(Rejection): authorize(review, 'learning', self.request)
        for values in [(2,180,1000001,1,self.ledger), (1,181,1000001,1,self.ledger),
                       (1,180,1000002,1,self.ledger), (1,180,1000001,2,self.ledger),
                       (1,180,1000001,1,self.root/'alternate.json')]:
            with self.subTest(request=values), self.assertRaises(Rejection):
                authorize(self.review, 'learning', LearningRequest(*values[:4], str(values[4])))
        with self.assertRaises(Rejection): authorize(self.review, 'learning')
        with self.assertRaises(Rejection): LearningRequest(True,180,1000001,1,str(self.ledger))
        for stage in ('selection','export','target_validation','release'):
            with self.subTest(stage=stage), self.assertRaises(Rejection): authorize(self.review, stage)
        capture = copy.deepcopy(self.review)
        capture.update(authorized_stages=['source_rollout'], learning_allowed=False, learning_limit=None)
        authorize(capture,'source_rollout')
        with self.assertRaises(Rejection): authorize(capture,'learning',self.request)
        self.assertFalse(self.ledger.exists())

    def test_failed_reservation_consumes_and_review_copy_cannot_replenish(self):
        budget = TrainingBudget(self.ledger)
        run = budget.reserve_authorized('test_only',self.review['candidate_id'],self.review,self.digest,self.request)
        budget.finish(run, .01, 'failed')
        before = self.ledger.read_bytes()
        for review, digest, candidate in [(copy.deepcopy(self.review),self.digest,self.review['candidate_id']),
                                         (self.review,'a'*64,self.review['candidate_id']),
                                         ({**self.review,'candidate_id':'other'},self.digest,'other')]:
            with self.assertRaises(Rejection): budget.reserve_authorized('test_only',candidate,review,digest,self.request)
        self.assertEqual(before,self.ledger.read_bytes())
        self.assertEqual(json.loads(before)['runs'][0]['authorization']['request']['iterations'],1)
        self.assertFalse(list(self.root.glob('*.partial')))

    def test_cumulative_iteration_limit_independent_of_run_limit(self):
        review = permission(self.ledger,iterations=2,max_runs=3)
        request = LearningRequest(1,10,1000001,1,str(self.ledger))
        budget = TrainingBudget(self.ledger)
        for _ in range(2): budget.reserve_authorized('test_only',review['candidate_id'],review,self.digest,request)
        with self.assertRaisesRegex(Rejection,'cumulative'): budget.reserve_authorized('test_only',review['candidate_id'],review,self.digest,request)

    def test_corrupt_or_ambiguous_ledger_is_unchanged_and_rejected(self):
        for data in ('{"runs":', '{"runs":{}}', '{"runs":[{}]}'):
            self.ledger.write_text(data)
            with self.assertRaises(Rejection): TrainingBudget(self.ledger).reserve_authorized('test_only',self.review['candidate_id'],self.review,self.digest,self.request)
            self.assertEqual(self.ledger.read_text(),data)
        self.ledger.unlink()
        budget = TrainingBudget(self.ledger)
        budget.reserve_authorized('test_only',self.review['candidate_id'],self.review,self.digest,self.request)
        data = json.loads(self.ledger.read_text())
        data['runs'].append(copy.deepcopy(data['runs'][0])); write_json(self.ledger,data)
        before=self.ledger.read_bytes()
        with self.assertRaisesRegex(Rejection,'Duplicate'): budget.reserve_authorized('test_only',self.review['candidate_id'],self.review,self.digest,self.request)
        self.assertEqual(before,self.ledger.read_bytes())

    def test_atomic_expiration_and_copied_ledger_refuse(self):
        expired={**self.review,'expires_at':time.time()-1}
        with self.assertRaisesRegex(Rejection,'current root'):
            TrainingBudget(self.ledger).reserve_authorized('test_only',self.review['candidate_id'],expired,self.digest,self.request)
        self.assertFalse(self.ledger.exists())
        budget=TrainingBudget(self.ledger)
        budget.reserve_authorized('test_only',self.review['candidate_id'],self.review,self.digest,self.request)
        data=json.loads(self.ledger.read_text())
        data['runs'][0]['state']='unknown';write_json(self.ledger,data)
        with self.assertRaisesRegex(Rejection,'Unknown'):budget.reserve('test_only','other',1000001,1)

    def test_two_real_cpu_processes_only_one_reserves(self):
        context=multiprocessing.get_context('spawn')
        barrier=context.Barrier(2); queue=context.Queue()
        children=[context.Process(target=authorizer_probe,args=(str(self.ledger),self.review,self.digest,barrier,queue)) for _ in range(2)]
        try:
            for child in children: child.start()
            results=[queue.get(timeout=15) for _ in children]
            for child in children: child.join(timeout=5); self.assertEqual(child.exitcode,0)
        finally:
            for child in children:
                if child.is_alive(): child.kill(); child.join(timeout=2)
            queue.close(); queue.join_thread()
        self.assertEqual(sum(result['reserved'] for result in results),1)
        self.assertTrue(all(not result['learner_executed'] for result in results))
        self.assertEqual(len(json.loads(self.ledger.read_text())['runs']),1)
        print('TEST_ONLY two CPU authorizers: exactly one reserve; zero learners')


class TrainingEntryRefusalTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name);self.ledger=self.root/'ledger.json'
        names=('source_inventory','adoption','compiled_robot','bam_parameters','checkpoint','onnx','evaluator','source_trajectory','source_video')
        artifacts={name:self.root/f'{name}.json' for name in names}
        for name,path in artifacts.items():write_json(path,{'test_only':name})
        write_json(artifacts['adoption'],{'skill':'test_only_standing'})
        write_json(artifacts['source_inventory'],{'root':str(self.root),'test_only':True})
        header={'kind':'header','canonical_actuator_order':list(JOINT_ORDER),'actuator_order':list(JOINT_ORDER),'body_order':['world','robot/trunk_base'],'auto_reset':False,'source_substeps':1}
        rows=[header]+[{'kind':kind,'physics_tick':tick,'body_positions':[[0,0,0],[0,0,.12]],'body_quaternions':[[1,0,0,0]]*2,'contacts':[],'reason':'time_limit'} for kind,tick in [('reset_initial',0),('physics_step',1),('terminal',1),('pre_reset',1),('reset_after_terminal',1)]]
        artifacts['source_trajectory'].write_text('\n'.join(json.dumps(row) for row in rows))
        self.candidate=bind_candidate(artifacts,{'source_substeps':1,'required_skills':['standing']})
        self.evidence=[]
        for stage in STAGES[:3]:
            path=self.root/f'{stage}_report.json';write_json(path,{'candidate_id':self.candidate['candidate_id'],'status':'passed','capture':{'natural_task_terminal':True}})
            self.evidence.append(bind_evidence(self.candidate,stage,path,['test_only_complete_lifecycle']))
        self.review=permission(self.ledger,self.candidate['candidate_id'])
        self.review.update(reviewer_role='root_gpt',decision='approve',reviewed_at=time.time()-1,expires_at=time.time()+60,evidence_identity=identity(self.evidence),observations=[{'time_seconds':0,'data_basis':'explicit test fixture','method':'unit test','disposition':'test only'}],anomalies_checked=['test only'],video_artifacts=[{'path':str(artifacts['source_video']),'sha256':sha256_file(artifacts['source_video'])}])
        self.paths={name:self.root/f'{name}.json' for name in ('candidate','evidence','review')}
        write_json(self.paths['candidate'],self.candidate);write_json(self.paths['evidence'],self.evidence)

    def call(self,**kw):
        return train_bounded(self.paths['candidate'],self.paths['evidence'],self.paths['review'],source_root=self.root,output=self.root/'never_created',ledger=kw.pop('ledger',self.ledger),seed=kw.pop('seed',1000001),**kw)

    def test_full_entry_refuses_before_reserve_or_spawn(self):
        variants=[]
        for name,updates in [('legacy',{'schema':'microduck_root_review_v1'}),('capture',{'authorized_stages':['source_rollout'],'learning_allowed':False,'learning_limit':None}),('false',{'learning_allowed':False}),('missingstage',{'authorized_stages':None}),('unknownstage',{'authorized_stages':['learning','unknown']}),('missinglimits',{'learning_limit':None}),('expired',{'expires_at':time.time()-1}),('changedcandidate',{'candidate_id':'other'}),('changedbasis',{'evidence_identity':'other'}),('emptyvideo',{'video_artifacts':[]})]:
            variants.append((name,{**self.review,**updates},{}))
        for name,request in [('2updates',{'iterations':2}),('181seconds',{'wall_seconds':181}),('wrongseed',{'seed':1000002}),('2gpus',{'gpus':2}),('nan',{'wall_seconds':float('nan')}),('bool',{'iterations':True}),('alternateledger',{'ledger':self.root/'other_ledger.json'})]:
            variants.append((name,self.review,request))
        for name,review,request in variants:
            write_json(self.paths['review'],review)
            with self.subTest(case=name),patch('bevy_microduck_tools.training.TrainingBudget.reserve_authorized') as reserve,patch('bevy_microduck_tools.training._run_child') as spawn:
                with self.assertRaises(Rejection):self.call(**request)
                reserve.assert_not_called();spawn.assert_not_called()
            self.assertFalse(self.ledger.exists());self.assertFalse((self.root/'never_created').exists())
        write_json(self.paths['review'],self.review)
        with patch('bevy_microduck_tools.training._run_child') as spawn:
            # Deliberately corrupt an evidence file rather than merely editing its summary.
            Path(self.evidence[0]['path']).write_bytes(b'changed evidence')
            with self.assertRaises(Rejection):self.call()
            spawn.assert_not_called();self.assertFalse(self.ledger.exists())

    def test_corrupt_ledger_rejected_before_process_and_never_rewritten(self):
        write_json(self.paths['review'],self.review)
        self.ledger.write_text('{"runs":')
        with patch('bevy_microduck_tools.training.verify_candidate_runtime'),patch('bevy_microduck_tools.training._run_child') as spawn:
            with self.assertRaisesRegex(Rejection,'Corrupt'):self.call()
            spawn.assert_not_called()
        self.assertEqual(self.ledger.read_text(),'{"runs":')

    def test_cli_requires_complete_learning_request_and_does_not_consume_on_inspection(self):
        from bevy_microduck_tools.cli import main
        write_json(self.paths['review'],self.review)
        args=['gate','--candidate',str(self.paths['candidate']),'--evidence',str(self.paths['evidence']),
              '--review',str(self.paths['review']),'--stage','learning']
        with self.assertRaises(SystemExit) as context:main(args)
        self.assertEqual(context.exception.code,2);self.assertFalse(self.ledger.exists())
        with patch('bevy_microduck_tools.source_runtime_identity.verify_candidate_runtime'):
            self.assertEqual(main(args+['--iterations','1','--wall-seconds','180','--seed','1000001','--gpus','1','--ledger',str(self.ledger)]),0)
        self.assertFalse(self.ledger.exists())

    def test_provided_review_is_checked_even_for_initial_capture_stage(self):
        review={**self.review,'evidence_identity':identity(self.evidence[:2])}
        with self.assertRaisesRegex(Rejection,'does not authorize'):
            admit(self.candidate,'source_rollout',self.evidence[:2],review)
        # Initial capture without any supplied review remains permitted.
        admit(self.candidate,'source_rollout',self.evidence[:2])

    def test_public_scientific_helpers_reject_learning_before_any_work(self):
        from bevy_microduck_tools.cli import audit, _audit_capture, _audit_authorized_learning
        from bevy_microduck_tools.trajectory import capture_source, _capture_source
        for value in (1,2,-1,True):
            with self.subTest(updates=value),patch('bevy_microduck_tools.cli.load_source') as load:
                with self.assertRaises(Rejection):audit(self.root,self.root/'must_not_exist','standing',learning_iterations=value)
                with self.assertRaises(Rejection):capture_source(None,None,None,self.root/'must_not_exist',learning_iterations=value)
                load.assert_not_called()
        with patch('bevy_microduck_tools.cli.load_source') as load:
            with self.assertRaises(Rejection):_audit_capture(self.root,self.root/'must_not_exist','standing',learning_iterations=1)
            with self.assertRaises(Rejection):_capture_source(type('Config',(),{'seed':1000001})(),None,None,self.root/'must_not_exist',learning_iterations=1)
            with self.assertRaises(Rejection):_audit_authorized_learning(self.root,self.root/'must_not_exist','standing',binding={},substeps=1,seed=1000001,device='cuda:0',iterations=1,checkpoint=Path('none'))
            load.assert_not_called()
        self.assertFalse((self.root/'must_not_exist').exists());self.assertFalse(self.ledger.exists())

    def test_private_worker_requires_actual_consumption_and_is_claimed_once(self):
        from bevy_microduck_tools.cli import _audit_authorized_learning
        from bevy_microduck_tools.workflow import verify_claimed_learning
        request=LearningRequest(1,180,1000001,1,str(self.ledger))
        digest='c'*64
        run=TrainingBudget(self.ledger).reserve_authorized('test_only_standing',self.candidate['candidate_id'],self.review,digest,request)
        binding={'ledger_path':str(self.ledger),'run_id':run,'candidate_id':self.candidate['candidate_id'],
                 'review_sha256':digest,'request':request.record(),'candidate_path':str(self.paths['candidate']),
                 'candidate_file_sha256':sha256_file(self.paths['candidate'])}
        checkpoint=Path(self.candidate['artifacts']['checkpoint']['path'])
        with self.assertRaisesRegex(Rejection,'claim'):verify_claimed_learning(binding,iterations=1,seed=1000001,checkpoint=checkpoint)
        with patch('bevy_microduck_tools.cli._audit_capture',return_value={'test_only':True,'actual_learning_updates':0}) as capture:
            result=_audit_authorized_learning(self.root,self.root/'unused','test_only_standing',binding=binding,substeps=1,seed=1000001,device='cuda:0',iterations=1,checkpoint=checkpoint)
            capture.assert_called_once();self.assertEqual(result['actual_learning_updates'],0)
        verify_claimed_learning(binding,iterations=1,seed=1000001,checkpoint=checkpoint)
        with patch('bevy_microduck_tools.cli._audit_capture') as capture:
            with self.assertRaisesRegex(Rejection,'claimed'):_audit_authorized_learning(self.root,self.root/'unused','test_only_standing',binding=binding,substeps=1,seed=1000001,device='cuda:0',iterations=1,checkpoint=checkpoint)
            capture.assert_not_called()
        with self.assertRaises(Rejection):verify_claimed_learning(binding,iterations=2,seed=1000001,checkpoint=checkpoint)
        with self.assertRaises(Rejection):verify_claimed_learning({**binding,'review_sha256':'a'*64},iterations=1,seed=1000001,checkpoint=checkpoint)
        TrainingBudget(self.ledger).finish(run,.001,'failed')
        with self.assertRaises(Rejection):verify_claimed_learning(binding,iterations=1,seed=1000001,checkpoint=checkpoint)
        self.assertFalse((self.root/'unused').exists())

    def test_failed_launch_consumes_permission_and_second_call_never_launches(self):
        write_json(self.paths['review'],self.review)
        with patch('bevy_microduck_tools.training.verify_candidate_runtime'),patch('bevy_microduck_tools.training._run_child',return_value={'status':'failed','reason':'test-only no child'}) as spawn:
            result=self.call()
            self.assertEqual(result['status'],'failed');spawn.assert_called_once()
        with patch('bevy_microduck_tools.training.verify_candidate_runtime'),patch('bevy_microduck_tools.training._run_child') as spawn:
            with self.assertRaisesRegex(Rejection,'consumed'):self.call()
            spawn.assert_not_called()
        row=json.loads(self.ledger.read_text())['runs'][0]
        self.assertEqual(row['state'],'failed');self.assertEqual(row['authorization']['request']['iterations'],1)


if __name__=='__main__':unittest.main()

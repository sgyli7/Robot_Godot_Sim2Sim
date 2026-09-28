"""CPU filesystem/metadata guards; no science worker, environment or real budget."""
import copy
import dataclasses
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from bevy_microduck_tools import source_runtime_identity as runtime
from bevy_microduck_tools.authorization import Rejection
from bevy_microduck_tools.serialization import identity, sha256_file, write_json
from bevy_microduck_tools.workflow import VerifiedRuntimeIdentity, require_verified_runtime


@unittest.skipUnless(importlib.util.find_spec('packaging'), 'Scientific metadata parser not installed')
class ActualFileIdentityTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)
        self.package=self.root/'identity_test_science';self.package.mkdir()
        self.module=self.package/'__init__.py';self.module.write_text('VALUE = 1\n')
        self.native=self.package/'native.so';self.native.write_bytes(b'native-not-loaded-0001')
        self.data=self.package/'motor.json';self.data.write_text('{"gain":1}\n')
        self.header=self.package/'native.hpp';self.header.write_text('constexpr int gain=1;\n')
        self.dist=self.root/'identity_test_distribution-1.0.dist-info';self.dist.mkdir()
        (self.dist/'METADATA').write_text('Metadata-Version: 2.1\nName: identity-test-distribution\nVersion: 1.0\n')
        self.record=self.dist/'RECORD'
        self.record.write_text('\n'.join(f'{p.relative_to(self.root)},sha256=FAKE,0' for p in [self.module,self.native,self.data,self.header,self.dist/'METADATA'])+'\n')
        sys.path.insert(0,str(self.root));self.addCleanup(lambda:sys.path.remove(str(self.root)))
        sys.modules.pop('identity_test_science',None)
        self.addCleanup(lambda:sys.modules.pop('identity_test_science',None))
        importlib.invalidate_caches()
        distributions,missing=runtime._distribution_closure(['identity-test-distribution'])
        self.assertEqual(missing,[])
        imports=runtime._origins(['identity_test_science'])
        roots=sorted({p for row in distributions for p in row['package_roots']})
        explicit=sorted({p for row in distributions for p in row['explicit_files']}|{os.path.abspath(sys.executable)})
        self.receipt={'schema':runtime.SCHEMA,'source_root':str(self.root),'required_distributions':['identity-test-distribution'],
            'python':runtime._python(),'resolution':runtime._resolution(),'distributions':distributions,
            'package_roots':roots,'explicit_files':explicit,
            'files':[runtime.file_record(Path(p)) for p in runtime._files(roots,explicit)],'imports':imports,'native_libraries':[],
            'coverage':{'installed_file_sets_complete':True,'recursive_dependencies_complete':True,
                'cpu_import_origins_complete':True,'native_profile_complete':False,'complete':False,
                'missing_dependencies':[],'limitations':['Test-only CPU package; native file is hashed but never loaded']}}
        self.seal(self.receipt)

    @staticmethod
    def seal(receipt):
        receipt['identity']=identity({k:v for k,v in receipt.items() if k!='identity'})

    def verify(self):
        # Native loaded-profile coverage is exercised by the separate real 7.56GB
        # scientific receipt. These tests isolate real temporary filesystem bytes.
        with patch.object(runtime,'_native_maps',return_value=[]):
            return runtime.verify_source_runtime(self.receipt,source_root=self.root,require_complete=False)

    def test_actual_metadata_and_complete_declared_byte_sets(self):
        result=self.verify();self.assertFalse(result['complete'])
        self.assertIn(str(self.header),[row['path'] for row in self.receipt['files']])
        self.assertNotIn(str(self.record),[row['path'] for row in self.receipt['files']])
        with self.assertRaisesRegex(Rejection,'incomplete'):
            runtime.verify_source_runtime(self.receipt,source_root=self.root)

    def test_same_version_python_native_data_and_header_changes_rejected(self):
        for path in (self.module,self.native,self.data,self.header):
            before=path.read_bytes();stamp=path.stat()
            path.write_bytes(bytes([before[0]^1])+before[1:])
            os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
            with self.subTest(path=path.name),self.assertRaisesRegex(Rejection,'bytes/origin'):
                self.verify()
            path.write_bytes(before)

    def test_added_and_missing_nonbytecode_files_rejected(self):
        added=self.package/'unlisted_parameter.json';added.write_text('new parameter')
        with self.assertRaisesRegex(Rejection,'file set'):self.verify()
        added.unlink();data=self.data.read_bytes();self.data.unlink()
        with self.assertRaises(Rejection):self.verify()
        self.data.write_bytes(data)

    def test_record_hash_forgery_cannot_hide_actual_byte_change(self):
        self.record.write_text(self.record.read_text().replace('FAKE','FORGED'))
        self.verify()  # RECORD hashes are not trusted and are not file identities.
        self.native.write_bytes(b'native-not-loaded-0002')
        with self.assertRaisesRegex(Rejection,'bytes/origin'):self.verify()

    def test_record_new_root_and_metadata_version_change_rejected(self):
        second=self.root/'another_science';second.mkdir();(second/'__init__.py').write_text('X=1')
        record=self.record.read_text();self.record.write_text(record+'another_science/__init__.py,,\n')
        with self.assertRaisesRegex(Rejection,'closure'):self.verify()
        self.record.write_text(record)
        metadata=self.dist/'METADATA';metadata.write_text(metadata.read_text().replace('Version: 1.0','Version: 2.0'))
        with self.assertRaisesRegex(Rejection,'closure'):self.verify()

    def test_bytecode_changes_ignored_and_symlink_directory_refused(self):
        cache=self.package/'__pycache__';cache.mkdir(exist_ok=True);(cache/'extra.pyc').write_bytes(b'ignored')
        (self.package/'unused.pyo').write_bytes(b'ignored');self.verify()
        external=self.root/'outside';external.mkdir();(external/'science.py').write_text('X=2')
        (self.package/'linked').symlink_to(external,target_is_directory=True)
        with self.assertRaisesRegex(Rejection,'Directory symlink'):self.verify()

    def test_symlink_retarget_and_missing_target_rejected(self):
        target=self.root/'external_parameter';target.write_text('original')
        link=self.package/'linked_parameter';link.symlink_to(target)
        self.receipt['files']=[runtime.file_record(Path(p)) for p in runtime._files(self.receipt['package_roots'],self.receipt['explicit_files'])]
        self.seal(self.receipt);self.verify()
        replacement=self.root/'other_parameter';replacement.write_text('original')
        link.unlink();link.symlink_to(replacement)
        with self.assertRaisesRegex(Rejection,'bytes/origin'):self.verify()
        replacement.unlink()
        with self.assertRaisesRegex(Rejection,'Broken'):self.verify()

    def test_wrong_python_module_source_and_library_resolution_rejected(self):
        with patch.object(runtime,'_python',return_value={**self.receipt['python'],'version':'different'}):
            with self.assertRaisesRegex(Rejection,'Python'):self.verify()
        with patch.object(runtime,'_origins',return_value=[]):
            with self.assertRaisesRegex(Rejection,'import origin'):self.verify()
        with patch.dict(os.environ,{'LD_PRELOAD':'unbound-library'}):
            with self.assertRaisesRegex(Rejection,'resolution'):self.verify()
        with self.assertRaisesRegex(Rejection,'root changed'):
            runtime.verify_source_runtime(self.receipt,source_root=self.root/'other',require_complete=False)
        with patch.object(runtime,'_native_maps',return_value=['/unbound/native.so']):
            with self.assertRaisesRegex(Rejection,'unbound library'):
                runtime.verify_source_runtime(self.receipt,source_root=self.root,require_complete=False)

    def test_native_map_requires_actual_device_inode_and_not_deleted(self):
        stat=self.native.stat()
        device=f'{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}'
        line=f'1000-2000 r-xp 00000000 {device} {stat.st_ino} {self.native}\n'
        with patch.object(Path,'read_text',return_value=line):
            self.assertEqual(runtime._native_maps(),[str(self.native)])
        for changed in (line.replace(device,'ff:ff'),line.replace(str(stat.st_ino),str(stat.st_ino+1)),line.rstrip()+' (deleted)\n'):
            with self.subTest(line=changed),patch.object(Path,'read_text',return_value=changed),self.assertRaises(Rejection):runtime._native_maps()

    def test_strict_unknown_fields_and_coverage_cannot_be_promoted(self):
        cases=[]
        changed=copy.deepcopy(self.receipt);changed['unknown']=True;cases.append(changed)
        changed=copy.deepcopy(self.receipt);changed['coverage']['complete']=True;cases.append(changed)
        changed=copy.deepcopy(self.receipt);changed['coverage']['native_profile_complete']=1;cases.append(changed)
        changed=copy.deepcopy(self.receipt);changed['package_roots']=[{}];cases.append(changed)
        changed=copy.deepcopy(self.receipt);changed['distributions'][0]['unknown']=True;cases.append(changed)
        changed=copy.deepcopy(self.receipt);changed['explicit_files']=[];cases.append(changed)
        for changed in cases:
            self.seal(changed)
            with self.subTest(receipt=changed['identity']),self.assertRaises(Rejection):runtime.validate_receipt(changed)
        changed=copy.deepcopy(self.receipt);changed['files'][0]['sha256']='0'*64
        with self.assertRaisesRegex(Rejection,'payload changed'):runtime.validate_receipt(changed)


class ProductionEntryIdentityTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)
        self.candidate=self.root/'candidate.json';write_json(self.candidate,{'candidate_id':'test_only','artifacts':{}})
        self.evidence=self.root/'evidence.json';write_json(self.evidence,[])
        self.review=self.root/'review.json';write_json(self.review,{})

    def test_missing_receipt_refused_parent_before_reserve_or_process(self):
        from bevy_microduck_tools.training import train_bounded
        with patch('bevy_microduck_tools.training.admit'),patch('bevy_microduck_tools.training.TrainingBudget.reserve_authorized') as reserve,patch('bevy_microduck_tools.training._run_child') as process:
            with self.assertRaisesRegex(Rejection,'lacks complete source runtime'):
                train_bounded(self.candidate,self.evidence,self.review,source_root=self.root,output=self.root/'never',ledger=self.root/'ledger',seed=1000001)
            reserve.assert_not_called();process.assert_not_called()
        self.assertFalse((self.root/'ledger').exists());self.assertFalse((self.root/'never').exists())

    def test_worker_guard_rejection_before_source_compile_or_capture(self):
        from bevy_microduck_tools.cli import _audit_capture
        # The consumed-run guard is independently tested in authorization tests;
        # here the actual missing-runtime rejection occurs before source loading.
        binding={'candidate_path':str(self.candidate),'candidate_file_sha256':sha256_file(self.candidate)}
        with patch('bevy_microduck_tools.cli.verify_claimed_learning'),patch('bevy_microduck_tools.workflow._learning_binding_request'),patch('bevy_microduck_tools.cli.load_source') as load,patch('bevy_microduck_tools.cli.export_compiled') as compile_model,patch('bevy_microduck_tools.trajectory._capture_source') as capture:
            with self.assertRaisesRegex(Rejection,'lacks complete source runtime'):
                _audit_capture(self.root,self.root/'never','standing',learning_iterations=1,_learning_binding=binding)
            load.assert_not_called();compile_model.assert_not_called();capture.assert_not_called()
        self.assertFalse((self.root/'never').exists())

    def test_verified_token_bound_to_manifest_receipt_pid_and_run(self):
        receipt=self.root/'runtime.json';receipt.write_text('test-only-token-bytes')
        candidate={'candidate_id':'test_only','artifacts':{'source_runtime_identity':{'path':str(receipt),'sha256':sha256_file(receipt)}}}
        write_json(self.candidate,candidate)
        binding={'candidate_id':'test_only','run_id':'unit-only','candidate_path':str(self.candidate),'candidate_file_sha256':sha256_file(self.candidate)}
        token=VerifiedRuntimeIdentity('test_only','unit-only',str(self.root),os.getpid(),'test-only',sha256_file(receipt))
        require_verified_runtime(token,binding,self.root)
        for bad in (None,dataclasses.replace(token,process_id=-1),dataclasses.replace(token,run_id='other'),dataclasses.replace(token,source_root='/other')):
            with self.assertRaises(Rejection):require_verified_runtime(bad,binding,self.root)
        receipt.write_text('changed')
        with self.assertRaisesRegex(Rejection,'receipt changed'):require_verified_runtime(token,binding,self.root)
        receipt.write_text('test-only-token-bytes');self.candidate.write_text(self.candidate.read_text()+' ')
        with self.assertRaisesRegex(Rejection,'manifest changed'):require_verified_runtime(token,binding,self.root)
        with self.assertRaises(dataclasses.FrozenInstanceError):token.runtime_identity='changed'

    def test_one_full_private_preflight_token_reaches_capture_without_second_hash(self):
        from bevy_microduck_tools.cli import _audit_capture
        receipt=self.root/'runtime.json';receipt.write_text('test-only-no-scientific-qualification')
        candidate={'candidate_id':'test_only','artifacts':{'source_runtime_identity':{'path':str(receipt),'sha256':sha256_file(receipt)}}}
        write_json(self.candidate,candidate)
        binding={'candidate_id':'test_only','run_id':'unit-only','candidate_path':str(self.candidate),'candidate_file_sha256':sha256_file(self.candidate)}
        class StopBeforeAnyScience(Exception):pass
        def capture(*args,**kw):
            require_verified_runtime(kw['_verified_runtime'],binding,self.root)
            raise StopBeforeAnyScience()
        # Actual private orchestration/token validation, with science-only helpers
        # isolated. Real complete-byte re-reading is measured by the full receipt.
        with patch('bevy_microduck_tools.cli.verify_claimed_learning'),patch('bevy_microduck_tools.workflow._learning_binding_request'),patch.object(runtime,'verify_candidate_runtime',return_value={'identity':'test-only'}) as full,patch('bevy_microduck_tools.cli.load_source',return_value={}),patch('bevy_microduck_tools.cli.inventory',return_value={}),patch('bevy_microduck_tools.cli.make_skill_cfg',return_value=(types.SimpleNamespace(),{'family':'leg','timing_plan':{}})),patch('bevy_microduck_tools.cli.export_compiled'),patch('bevy_microduck_tools.trajectory._capture_source',side_effect=capture):
            with self.assertRaises(StopBeforeAnyScience):
                _audit_capture(self.root,self.root/'only-cpu-json','standing',learning_iterations=1,_learning_binding=binding)
            full.assert_called_once()


if __name__=='__main__':unittest.main()

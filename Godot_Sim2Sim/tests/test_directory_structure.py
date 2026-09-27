"""Directory policy tests use temporary Git repositories, never game runtimes."""

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("structure_check", ROOT / "scripts/check_godot_structure.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class TemporaryRepository(unittest.TestCase):
    role = "lab"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="directory policy ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q", "--initial-branch=main")
        self.old = "godot/old_folder/OldController.gd" if self.role == "lab" else "vehicles/Old_Vehicle/runtime/Drive.gd"
        self.write(self.old, "old implementation\n")
        self.write("docs/directory_inventory.json", "{}\n")
        self.git("add", ".")
        self.git("commit", "-qm", "fixture baseline")
        self.base = self.git("rev-parse", "HEAD").strip()
        self.policy = {
            "schema_version": 1,
            "repository_role": self.role,
            "baseline_commit": self.base,
            "legacy_paths": [self.old, "docs/directory_inventory.json"],
            "compatibility_additions": [],
            "source_records": [],
        }
        self.save_policy()

    def git(self, *args):
        env = dict(os.environ, GIT_AUTHOR_NAME="Directory tests", GIT_AUTHOR_EMAIL="tests@example.invalid",
                   GIT_COMMITTER_NAME="Directory tests", GIT_COMMITTER_EMAIL="tests@example.invalid")
        return subprocess.check_output(["git", "-C", str(self.root), *args], env=env, text=True)

    def write(self, value, text="fixture\n"):
        path = self.root / value
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def save_policy(self):
        self.write("docs/directory_inventory.json", json.dumps(self.policy))

    def check(self, **kwargs):
        return CHECKER.check(self.root, **kwargs)


class LabStructureTests(TemporaryRepository):
    def test_existing_fix_and_legacy_folder_are_not_blanket_exempt(self):
        self.write(self.old, "normal fix\n")
        self.assertEqual(self.check()[1], [])
        new = "godot/old_folder/new_controller.gd"
        self.write(new)
        candidates, errors, _ = self.check()
        self.assertEqual(candidates, [new])
        self.assertEqual(len(errors), 1)

    def test_legacy_record_cannot_exempt_an_uncommitted_new_file(self):
        new = "godot/old_folder/unregistered.gd"
        self.write(new)
        self.policy["legacy_paths"].append(new)
        self.save_policy()
        self.assertEqual(len(self.check()[1]), 1)

    def test_legacy_record_cannot_exempt_a_new_renamed_destination(self):
        destination = "godot/old_folder/renamed.gd"
        self.git("mv", self.old, destination)
        self.policy["legacy_paths"].append(destination)
        self.save_policy()
        self.assertEqual(len(self.check()[1]), 1)

    def test_new_and_staged_runtime_data_resources_and_sidecars(self):
        values = [
            "godot/game/scripts/modules/robot/drive.gd",
            "godot/game/scripts/modules/robot/drive.gd.uid",
            "godot/game/scripts/core/math_util.gd",
            "godot/game/dynamic_assets/game_data/policy.onnx",
            "godot/game/dynamic_assets/settings/joint_map.json",
            "godot/game/dynamic_assets/game_data/sample.csv",
            "godot/game/shaders/surface.gdshaderinc",
            "godot/game/arts/entity/models/robot.glb",
            "godot/game/arts/entity/models/robot.bin",
            "godot/game/arts/ui/fonts/family.ttc",
            "godot/game/arts/ui/fonts/family.ttc.import",
            "godot/game/arts/ui/fonts/LICENSE.txt",
            "godot/game/scenes/dev/fixture.json",
            "godot/addons/dev_tools/probe.gd",
            "godot/game/scenes/frontend/menu.tscn",
            "godot/game/instances/entity/robot.tscn",
            "godot/i18n_assets/en.po",
        ]
        for value in values:
            target = self.write(value)
            if "/arts/" in value and not value.endswith((".import", "LICENSE.txt")):
                self.policy["source_records"].append({"path": value, "repository": "https://example.invalid/art",
                    "revision": "delivery-v1", "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        self.policy["source_records"].append({"path": "godot/game/arts/ui/fonts/family.ttc.import",
            "repository": "https://example.invalid/art", "revision": "delivery-v1",
            "sha256": hashlib.sha256((self.root / "godot/game/arts/ui/fonts/family.ttc.import").read_bytes()).hexdigest()})
        self.save_policy()
        self.git("add", values[0])
        self.assertEqual(set(self.check()[0]), set(values))
        self.assertEqual(self.check()[1], [])

    def test_wrong_role_type_and_names_are_rejected(self):
        for value in [
            "godot/game/drive.gd",
            "godot/game/scenes/dev/probe.gd",
            "godot/game/scripts/core/BadName.gd",
            "godot/game/scripts/modules/robot/runtime.py",
            "godot/game/arts/entity/models/robot.blend",
            "godot/game/arts/unknown/textures/image.png",
            "godot/game/arts/ui/fonts/font.exe",
            "godot/addons/random_vendor/library.so",
        ]:
            self.write(value)
        self.assertEqual(len(self.check()[1]), 8)

    def test_lab_art_delivery_needs_an_explicit_source(self):
        self.write("godot/game/arts/entity/models/robot.glb")
        self.assertEqual(len(self.check()[1]), 1)

    def test_renamed_legacy_file_must_validate_destination(self):
        destination = "godot/another_old/drive.gd"
        (self.root / destination).parent.mkdir()
        self.git("mv", self.old, destination)
        self.assertIn(destination, self.check()[0])
        self.assertEqual(len(self.check()[1]), 1)

    def test_precise_compatibility_record_only_exempts_that_file(self):
        value = "godot/old_folder/compat.gd"
        self.write(value)
        self.policy["compatibility_additions"] = [{"path": value, "reason": "Existing caller needs the old entry until migration."}]
        self.save_policy()
        self.assertEqual(self.check()[1], [])
        self.write("godot/old_folder/another.gd")
        self.assertEqual(len(self.check()[1]), 1)
        self.policy["compatibility_additions"][0]["reason"] = ""
        self.save_policy()
        with self.assertRaises(ValueError):
            self.check()

    def test_import_record_preserves_upstream_name_and_checks_new_bytes(self):
        value = "godot/plugins/upstream/ForeignLibrary.so"
        target = self.write(value)
        self.policy["source_records"] = [{"path": value, "repository": "https://example.invalid/upstream",
            "revision": "v1.0", "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}]
        self.save_policy()
        self.assertEqual(self.check()[1], [])
        self.write(value, "different bytes\n")
        self.assertEqual(len(self.check()[1]), 1)

    def test_missing_peer_and_historical_divergence_are_only_notes(self):
        self.assertEqual(self.check(peer_root=self.root / "absent")[1], [])
        self.assertTrue(self.check(peer_root=self.root / "absent")[2])
        peer = self.root / "peer"
        peer.mkdir()
        self.write("godot/game/scripts/core/math.gd", "current source\n")
        (peer / "copy.gd").write_text("compatibility variant\n")
        self.policy["comparisons"] = [{"lab_path": "godot/game/scripts/core/math.gd", "art_path": "copy.gd",
            "lab_sha256": "0" * 64, "art_sha256": "1" * 64}]
        self.save_policy()
        _, errors, notes = self.check(peer_root=peer)
        self.assertEqual(errors, [])
        self.assertTrue(any("偏离" in note for note in notes))

    def test_explicit_base_and_invalid_baseline(self):
        self.assertEqual(self.check(base="HEAD")[1], [])
        with self.assertRaises(ValueError):
            self.check(base="missing-test-ref")

    def test_lab_subtree_from_repository_root_excludes_unity(self):
        nested = self.root / "Godot_Sim2Sim"
        nested.mkdir()
        for name in ("godot", "docs"):
            (self.root / name).rename(nested / name)
        self.git("add", "-A")
        self.git("commit", "-qm", "fixture repository layout")
        self.policy["baseline_commit"] = self.git("rev-parse", "HEAD").strip()
        self.write("Godot_Sim2Sim/docs/directory_inventory.json", json.dumps(self.policy))
        self.write("Godot_Sim2Sim/godot/game/scripts/core/math.gd")
        self.write("Unity_Sim2Sim/UnrelatedName.cs")
        candidates, errors, _ = CHECKER.check(self.root)
        self.assertEqual(candidates, ["godot/game/scripts/core/math.gd"])
        self.assertEqual(errors, [])

    def test_no_parent_or_directory_wide_exception(self):
        for value in ["../escape.gd", "godot/old_folder/", "/absolute.gd"]:
            self.policy["compatibility_additions"] = [{"path": value, "reason": "test"}]
            self.save_policy()
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.check()

    def test_cli_exit_codes_and_invalid_local_inventory(self):
        def run():
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as stderr:
                code = CHECKER.main(["--repo-root", str(self.root)])
            return code, stderr.getvalue()
        self.assertEqual(run()[0], 0)
        self.write("godot/old_folder/new.gd")
        self.assertEqual(run()[0], 1)
        for invalid in [[], {"schema_version": 1, "repository_role": "lab", "legacy_paths": None},
                        {"schema_version": 1, "repository_role": "lab", "source_records": ["bad record"]}]:
            self.write("docs/directory_inventory.json", json.dumps(invalid))
            code, error = run()
            self.assertEqual(code, 2)
            self.assertNotIn("Traceback", error)


class ArtStructureTests(TemporaryRepository):
    role = "art"

    def test_asset_tools_and_existing_identifier_are_allowed(self):
        for value in [
            "vehicles/Old_Vehicle/source/build_mesh.py",
            "vehicles/Old_Vehicle/source/preview.gd",
            "vehicles/Old_Vehicle/source/body.blend",
            "vehicles/Old_Vehicle/assets/body.glb",
            "vehicles/Old_Vehicle/assets/LICENSE.txt",
            "vehicles/Old_Vehicle/themes/snow/texture.png",
            "scripts/export_asset.py",
            "tests/check_geometry.py",
            "docs/delivery.md",
        ]:
            self.write(value)
        self.assertEqual(self.check()[1], [])

    def test_old_runtime_fixes_pass_new_runtime_delivery_needs_record(self):
        self.write(self.old, "normal fix\n")
        self.assertEqual(self.check()[1], [])
        values = ["vehicles/Old_Vehicle/runtime/drive.gd", "integration/sai60/extra.gd", "vehicles/Old_Vehicle/assets/policy.onnx"]
        for value in values:
            self.write(value)
        self.assertEqual(len(self.check()[1]), 3)
        self.policy["compatibility_additions"] = [{"path": values[0], "reason": "Registered temporary old entry for current delivery."}]
        self.save_policy()
        self.assertEqual(len(self.check()[1]), 2)

    def test_optional_peer_inventory_failures_remain_advisory(self):
        with tempfile.TemporaryDirectory(prefix="directory peer ") as folder:
            peer = Path(folder)
            (peer / "docs").mkdir()
            inventory = peer / "docs/directory_inventory.json"
            for invalid in ["not JSON", "[]", '{"comparisons": null}', '{"comparisons": [null]}']:
                inventory.write_text(invalid)
                with self.subTest(invalid=invalid):
                    _, errors, notes = self.check(peer_root=peer)
                    self.assertEqual(errors, [])
                    self.assertTrue(notes)
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        code = CHECKER.main(["--repo-root", str(self.root), "--peer-root", str(peer)])
                    self.assertEqual(code, 0)


class ForestForwardingTests(unittest.TestCase):
    def test_arguments_cwd_and_exit_status_with_a_stub_in_a_temporary_copy(self):
        with tempfile.TemporaryDirectory(prefix="forest forwarding ") as folder:
            root = Path(folder)
            scripts = root / "scripts"
            original = root / "godot/scripts"
            scripts.mkdir()
            original.mkdir(parents=True)
            shutil.copy2(ROOT / "scripts/setup_forest_vendor.sh", scripts)
            stub = original / "setup_forest_vendor.sh"
            stub.write_text("#!/usr/bin/env python3\nimport json, os, sys\nfrom pathlib import Path\nPath(os.environ['FOREST_TEST_OUTPUT']).write_text(json.dumps({'args':sys.argv[1:], 'cwd':os.getcwd()}))\nsys.exit(int(os.environ['FOREST_TEST_EXIT']))\n")
            stub.chmod(0o755)
            caller = root / "caller with spaces"
            caller.mkdir()
            output = root / "output.json"
            for status, arguments in [(0, []), (42, ["--flag", "argument with spaces", "", "$(literal)", "*.glb"])]:
                env = dict(os.environ, FOREST_TEST_OUTPUT=str(output), FOREST_TEST_EXIT=str(status))
                with self.subTest(status=status, arguments=arguments):
                    result = subprocess.run([str(scripts / "setup_forest_vendor.sh"), *arguments], cwd=caller, env=env)
                    self.assertEqual(result.returncode, status)
                    self.assertEqual(json.loads(output.read_text()), {"args": arguments, "cwd": str(caller)})


if __name__ == "__main__":
    unittest.main()

"""Verify source-process preparation retains virtual-environment identity."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import venv


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/unitree_g1_source_process_trials.py"
SPEC = importlib.util.spec_from_file_location("g1_source_process_trials", SCRIPT)
TRIALS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRIALS)


class SourceProcessIdentityTests(unittest.TestCase):
    def test_symlinked_venv_python_retains_its_package_environment(self):
        scratch = SCRIPT.parents[4] / ".scratch/g1_python_tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            root = Path(temporary)
            environment = root / "env"
            venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
            interpreter = environment / "bin/python"
            self.assertTrue(interpreter.is_symlink())
            site = environment / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
            for name in ("isaacsim", "torch", "warp-lang", "onnxruntime"):
                metadata = site / f"{name.replace('-', '_')}-0.0.1.dist-info"
                metadata.mkdir(parents=True)
                (metadata / "METADATA").write_text(f"Name: {name}\nVersion: 0.0.1\n")
            harness = root / "harness.py"
            harness.write_text("raise RuntimeError('prepare must never execute this harness')\n")
            output = root / "prepared"
            args = SimpleNamespace(
                python=interpreter, harness=harness, arena_source=root, lab_source=root,
                homie_assets=root, output_dir=output, trials=2, ticks=150, seed=42,
                timeout_seconds=85,
            )
            TRIALS.prepare(args)
            plan = json.loads((output / "plan.json").read_text())
            self.assertEqual(plan["python_runtime"]["prefix"], str(environment))
            self.assertEqual(set(plan["python_runtime"]["versions"].values()), {"0.0.1"})
            self.assertFalse((output / "execution_claim.json").exists())


if __name__ == "__main__":
    unittest.main()

"""Reject fabricated origin/update counts and detect actual model buffer changes."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from bevy_microduck_tools.provenance import describe_origin, model_state, restored_origin
from bevy_microduck_tools.serialization import sha256_file

try:
    import torch
except ImportError:
    torch = None


class OriginTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.checkpoint = self.root / "checkpoint.pt"
        self.checkpoint.write_bytes(b"test-only sealed checkpoint, not a policy")
        self.failure = self.root / "failed.json"
        self.failure.write_text(json.dumps({"status": "failed", "run_id": "test_run", "parent_candidate_id": "test_parent",
            "known_completed_ppo_updates": 1, "artifacts": [{"path": str(self.checkpoint), "sha256": sha256_file(self.checkpoint)}]}))
        self.origin = {"schema": "restored_checkpoint_origin_input_v1", "originating_run_id": "test_run",
            "parent_candidate_id": "test_parent", "known_prior_ppo_iterations": 1,
            "checkpoint": {"path": str(self.checkpoint), "sha256": sha256_file(self.checkpoint)},
            "originating_failed_run": {"path": str(self.failure), "sha256": sha256_file(self.failure)},
            "expected_checkpoint_state": {"saved_runner_iteration_index": 0, "optimizer_internal_steps": [],
                                           "normalizer_counts": {"actor": 24, "critic": 24}}}
        self.path = self.root / "origin.json"
        self.store()

    def store(self):
        self.path.write_text(json.dumps(self.origin))

    def test_prior_update_is_bound_to_sealed_run_not_iteration_index(self):
        self.assertEqual(restored_origin(self.path, self.checkpoint)["known_prior_ppo_iterations"], 1)
        self.origin["known_prior_ppo_iterations"] = 2
        self.store()
        with self.assertRaisesRegex(ValueError, "prior updates"):
            restored_origin(self.path, self.checkpoint)

    def test_changed_checkpoint_and_origin_manifest_are_rejected(self):
        self.checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint differs"):
            restored_origin(self.path, self.checkpoint)
        self.origin["checkpoint"]["sha256"] = sha256_file(self.checkpoint)
        self.store()
        with self.assertRaisesRegex(ValueError, "not sealed"):
            restored_origin(self.path, self.checkpoint)
        self.failure.write_text('{}')
        with self.assertRaisesRegex(ValueError, "run identity changed"):
            restored_origin(self.path, self.checkpoint)

    @unittest.skipIf(torch is None, "requires actual Torch source environment")
    def test_zero_updates_restored_actor_is_not_fresh_or_untrained(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.obs_normalizer = torch.nn.Module()
                self.obs_normalizer.register_buffer("count", torch.tensor(24))
                self.obs_normalizer.register_buffer("mean", torch.ones(61))
                self.weight = torch.nn.Parameter(torch.ones(14))
        runner = SimpleNamespace(current_learning_iteration=0,
            alg=SimpleNamespace(actor=Model(), critic=Model(), optimizer=SimpleNamespace(state={})))
        description = describe_origin(runner, self.checkpoint, self.origin, 0)
        self.assertEqual(description["initialization"], "restored_checkpoint")
        self.assertEqual(description["known_prior_ppo_iterations"], 1)
        self.assertEqual(description["ppo_iterations_requested_this_invocation"], 0)
        self.assertEqual(description["known_cumulative_ppo_iterations_if_completed"], 1)
        before = model_state(runner)
        runner.alg.actor.obs_normalizer.mean[0] += 1
        self.assertNotEqual(before, model_state(runner))
        runner.alg.actor.obs_normalizer.count += 1
        with self.assertRaisesRegex(ValueError, "Actual loaded checkpoint"):
            describe_origin(runner, self.checkpoint, self.origin, 0)


if __name__ == "__main__":
    unittest.main()

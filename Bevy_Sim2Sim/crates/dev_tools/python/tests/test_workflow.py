"""Behavioral rejection tests exercise changed bytes and lifecycle evidence."""

import json
import tempfile
import time
import unittest
from pathlib import Path

from bevy_microduck_tools.evaluation import evaluate_raw
from bevy_microduck_tools.serialization import identity, sha256_file, write_json
from bevy_microduck_tools.source_adapter import JOINT_ORDER, delay_plan
from bevy_microduck_tools.timing import TimedHistory
from bevy_microduck_tools.workflow import Rejection, TrainingBudget, admit, bind_candidate, bind_evidence, verify_candidate, verify_review


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        names = ("source_inventory", "adoption", "compiled_robot", "bam_parameters", "checkpoint", "onnx", "evaluator")
        artifacts = {}
        for name in names:
            artifacts[name] = self.root / f"{name}.json"
            artifacts[name].write_text(json.dumps({"real_test_bytes": name}))
        self.candidate = bind_candidate(artifacts, {"required_skills": ["standing", "sprint"]})

    def report(self, stage):
        path = self.root / f"{stage}_report.json"
        write_json(path, {"candidate_id": self.candidate["candidate_id"], "status": "passed"})
        return bind_evidence(self.candidate, stage, path, ["nonempty_behavioral_check"])

    def test_first_stages_do_not_require_all_skill_completion(self):
        evidence = []
        for stage in ("adoption", "source_compile", "source_rollout"):
            admit(self.candidate, stage, evidence)
            evidence.append(self.report(stage))
        with self.assertRaises(Rejection):
            admit(self.candidate, "learning", evidence)

    def test_changed_weight_invalidates_candidate(self):
        Path(self.candidate["artifacts"]["checkpoint"]["path"]).write_bytes(b"different weight")
        with self.assertRaisesRegex(Rejection, "artifact changed"):
            verify_candidate(self.candidate)

    def test_empty_coverage_rejected(self):
        with self.assertRaisesRegex(Rejection, "empty"):
            bind_evidence(self.candidate, "source_compile", self.root / "compiled_robot.json", [])

    def test_review_expiration_and_changed_basis_rejected(self):
        evidence = [self.report("adoption")]
        now = time.time()
        review = {"candidate_id": self.candidate["candidate_id"], "reviewer_role": "root_gpt", "decision": "approve",
                  "reviewed_at": now - 10, "expires_at": now - 1, "evidence_identity": identity(evidence),
                  "observations": [{"time_seconds": 0, "data_basis": "frame0", "method": "test fixture", "disposition": "test"}],
                  "anomalies_checked": ["all"], "video_artifacts": []}
        with self.assertRaisesRegex(Rejection, "expired"):
            verify_review(self.candidate, review, evidence, stage="learning", now=now)
        review["expires_at"] = now + 10
        evidence[0]["coverage"].append("changed")
        with self.assertRaisesRegex(Rejection, "changed since review"):
            verify_review(self.candidate, review, evidence, stage="learning", now=now)

    def test_per_run_and_aggregate_budget_and_training_partition(self):
        budget = TrainingBudget(self.root / "budget.json")
        with self.assertRaises(Rejection):
            budget.reserve("standing", "candidate", 1000001, 7201)
        with self.assertRaises(Rejection):
            budget.reserve("standing", "candidate", 3000001, 10)
        for index in range(6):
            budget.reserve("standing", "candidate", 1000001 + index, 7200)
        with self.assertRaisesRegex(Rejection, "exhausted"):
            budget.reserve("standing", "candidate", 1000007, 1)

    def test_clock_causality_and_reset(self):
        history = TimedHistory("HOME")
        history.append(0.0, "a")
        history.append(1/60, "b")
        self.assertEqual(history.read(1/60, .020), "HOME")
        self.assertEqual(history.read(2/60, .020), "a")
        self.assertEqual(history.read(2/60, .015), "b")
        history.reset("RESET")
        self.assertEqual(history.read(10, .020), "RESET")

    def test_motor_distribution_is_not_uniform_in_mapped_ticks(self):
        plan = delay_plan(3, 6, .005, 1/60)
        self.assertEqual(plan["mapped_ticks"], [1, 2, 2, 2])
        self.assertEqual(plan["probabilities"], [.25] * 4)
        self.assertEqual(delay_plan(0, 1, .020, 1/60, 64)["update_period_ticks"], 77)

    def raw(self):
        path = self.root / "raw.jsonl"
        header = {"kind": "header", "canonical_actuator_order": list(JOINT_ORDER),
                  "actuator_order": list(JOINT_ORDER), "body_order": ["world", "robot/trunk_base"],
                  "auto_reset": False, "source_substeps": 1}
        def row(kind, tick=0):
            return {"kind": kind, "physics_tick": tick, "body_positions": [[0, 0, 0], [0, 0, .12]],
                    "body_quaternions": [[1, 0, 0, 0]]*2, "contacts": [], "reason": "horizon"}
        rows = [header, row("reset_initial"), row("physics_step", 1), row("terminal", 1),
                row("pre_reset", 1), row("reset_after_terminal", 1)]
        path.write_text("\n".join(json.dumps(item) for item in rows))
        return path, rows

    def test_terminal_missing_rejected_even_when_summary_passes(self):
        path, rows = self.raw()
        self.assertEqual(evaluate_raw(path)["episodes_in_denominator"], 1)
        path.write_text("\n".join(json.dumps(row) for row in rows if row["kind"] != "terminal"))
        with self.assertRaisesRegex(Rejection, "Missing"):
            evaluate_raw(path)

    def test_wrong_mapping_rejected(self):
        path, rows = self.raw()
        rows[0]["actuator_order"][0], rows[0]["actuator_order"][1] = rows[0]["actuator_order"][1], rows[0]["actuator_order"][0]
        path.write_text("\n".join(json.dumps(row) for row in rows))
        with self.assertRaisesRegex(Rejection, "mapping"):
            evaluate_raw(path)

    def test_reset_forward_contact_is_not_an_integrated_impulse_or_new_terminal(self):
        path, rows = self.raw()
        rows[0].update(schema="microduck_4d_raw_v2", integration_dt=1/60)
        for row in rows[1:]:
            row.update(episode_epoch=0, contact_interval=[0, 1/60] if row["kind"] == "physics_step" else None,
                       termination_measurement={"belongs_to_current_episode": row["kind"] != "reset_after_terminal",
                                                "computed_at_current_physics_tick": row["kind"] != "reset_after_terminal"},
                       policy_input_and_output_at_call=None)
        reset = rows[-1]
        reset["episode_epoch"] = 1
        reset["contacts"] = [{"distance": 0, "force_contact_frame": [1, 0, 0, 0, 0, 0], "impulse_contact_frame": None}]
        def store():
            path.write_text("\n".join(json.dumps(item) for item in rows))
        store()
        self.assertEqual(evaluate_raw(path)["episodes_in_denominator"], 1)
        reset["contact_interval"] = [0, 1/60]
        store()
        with self.assertRaisesRegex(Rejection, "falsely claims"):
            evaluate_raw(path)
        reset["contact_interval"] = None
        reset["termination_measurement"]["belongs_to_current_episode"] = True
        store()
        with self.assertRaisesRegex(Rejection, "previous episode"):
            evaluate_raw(path)


if __name__ == "__main__":
    unittest.main()

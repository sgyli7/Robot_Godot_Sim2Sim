"""Exercise negative cases against copies of a real captured candidate."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .evaluation import evaluate_raw
from .serialization import identity, write_json
from .workflow import Rejection, bind_candidate, bind_evidence, verify_candidate, verify_review


def check_rejections(candidate_path: Path, output: Path) -> dict:
    candidate = json.loads(candidate_path.read_text())
    verify_candidate(candidate)
    if output.exists() and any(output.iterdir()):
        raise Rejection("Negative cases require a new isolated directory")
    output.mkdir(parents=True, exist_ok=True)
    trajectory = Path(candidate["artifacts"]["source_trajectory"]["path"])
    baseline = evaluate_raw(trajectory)
    artifacts = {key: Path(record["path"]) for key, record in candidate["artifacts"].items()}
    copy = output / "copied_checkpoint.pt"
    shutil.copyfile(artifacts["checkpoint"], copy)
    artifacts["checkpoint"] = copy
    test_candidate = bind_candidate(artifacts, {**candidate["contract"], "purpose": "negative test fixture only"})
    results = {}
    def rejected(name, callback):
        try:
            callback()
        except Rejection as error:
            results[name] = {"rejected": True, "reason": str(error)}
        else:
            raise AssertionError(f"Negative case incorrectly accepted: {name}")
    original = copy.read_bytes()
    copy.write_bytes(original + b"\x00tampered")
    rejected("changed_checkpoint", lambda: verify_candidate(test_candidate))
    copy.write_bytes(original)
    rows = [json.loads(line) for line in trajectory.read_text().splitlines()]
    missing = output / "missing_terminal.jsonl"
    missing.write_text("\n".join(json.dumps(row) for row in rows if row["kind"] != "terminal"))
    rejected("missing_terminal", lambda: evaluate_raw(missing))
    wrong = output / "wrong_mapping.jsonl"
    rows[0]["actuator_order"][0], rows[0]["actuator_order"][1] = rows[0]["actuator_order"][1], rows[0]["actuator_order"][0]
    wrong.write_text("\n".join(json.dumps(row) for row in rows))
    rejected("wrong_mapping", lambda: evaluate_raw(wrong))
    rejected("empty_coverage", lambda: bind_evidence(test_candidate, "source_compile", candidate_path, []))
    now = time.time()
    # This intentionally expired, explicitly isolated fixture is never a real
    # approval record and is not written to the captured candidate directory.
    expired = {"candidate_id": test_candidate["candidate_id"], "reviewer_role": "root_gpt", "decision": "approve",
               "test_fixture_only": True, "reviewed_at": now - 2, "expires_at": now - 1,
               "evidence_identity": identity([]), "observations": [], "anomalies_checked": [], "video_artifacts": []}
    rejected("expired_review", lambda: verify_review(test_candidate, expired, [], stage="learning", now=now))
    report = {"parent_candidate_id": candidate["candidate_id"], "status": "negative_checks_passed",
              "test_only": True, "qualification": False, "actual_baseline": baseline, "cases": results}
    write_json(output / "rejection_report.json", report)
    return report

"""Bind restored weights to measured update history without interpreting iteration indices."""

from __future__ import annotations

import json
from pathlib import Path

from .serialization import identity, sha256_file


def restored_origin(path: Path, checkpoint: Path) -> dict:
    origin = json.loads(path.read_text())
    if origin.get("schema") != "restored_checkpoint_origin_input_v1":
        raise ValueError("Unknown checkpoint-origin schema")
    record = origin["checkpoint"]
    if Path(record["path"]).resolve() != checkpoint.resolve() or sha256_file(checkpoint) != record["sha256"]:
        raise ValueError("Restored checkpoint differs from origin identity")
    failure_record = origin["originating_failed_run"]
    failure_path = Path(failure_record["path"])
    if sha256_file(failure_path) != failure_record["sha256"]:
        raise ValueError("Originating measured run identity changed")
    failure = json.loads(failure_path.read_text())
    if failure.get("status") != "failed" or failure["run_id"] != origin["originating_run_id"]:
        raise ValueError("Checkpoint origin is not the declared retained failed run")
    if failure["parent_candidate_id"] != origin["parent_candidate_id"]:
        raise ValueError("Checkpoint origin parent mismatch")
    if failure["known_completed_ppo_updates"] != origin["known_prior_ppo_iterations"]:
        raise ValueError("Known prior updates differ from measured run evidence")
    if not any(Path(item["path"]).resolve() == checkpoint.resolve() and item["sha256"] == record["sha256"]
               for item in failure["artifacts"]):
        raise ValueError("Restored checkpoint was not sealed in the measured run")
    for artifact in failure["artifacts"]:
        source = Path(artifact["path"])
        if not source.is_file() or sha256_file(source) != artifact["sha256"]:
            raise ValueError("Originating failed-run artifact changed or missing")
    return origin


def model_state(runner) -> dict:
    """Fingerprint actual model tensors, including empirical normalizer buffers."""
    import hashlib
    result = {}
    for role, module in (("actor", runner.alg.actor), ("critic", runner.alg.critic)):
        tensors = {}
        for name, tensor in module.state_dict().items():
            value = tensor.detach().contiguous().cpu()
            tensors[name] = {"dtype": str(value.dtype), "shape": list(value.shape),
                             "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()}
        result[role] = {"tensors": tensors, "state_identity": identity(tensors),
                        "normalizer_count": int(module.obs_normalizer.count.detach().cpu())}
    result["optimizer_internal_steps"] = sorted({int(value["step"].detach().cpu())
                                                 for value in runner.alg.optimizer.state.values() if "step" in value})
    result["saved_runner_iteration_index"] = runner.current_learning_iteration
    return result


def describe_origin(runner, checkpoint: Path | None, origin: dict | None, updates: int) -> dict:
    actual = model_state(runner)
    if origin is not None:
        if checkpoint is None:
            raise ValueError("Checkpoint origin provided without restored weights")
        expected = origin["expected_checkpoint_state"]
        if (actual["saved_runner_iteration_index"] != expected["saved_runner_iteration_index"] or
                actual["optimizer_internal_steps"] != expected["optimizer_internal_steps"] or
                {role: actual[role]["normalizer_count"] for role in ("actor", "critic")} != expected["normalizer_counts"]):
            raise ValueError("Actual loaded checkpoint state differs from proven origin")
    prior = 0 if checkpoint is None else (origin["known_prior_ppo_iterations"] if origin is not None else None)
    return {"schema": "checkpoint_origin_v1", "initialization": "fresh_initialization" if checkpoint is None else "restored_checkpoint",
            "checkpoint": None if checkpoint is None else {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)},
            "verified_origin": origin, "actual_loaded_model_state": actual,
            "known_prior_ppo_iterations": prior, "ppo_iterations_requested_this_invocation": updates,
            "known_cumulative_ppo_iterations_if_completed": None if prior is None else prior + updates,
            "iteration_index_is_not_update_count": True}

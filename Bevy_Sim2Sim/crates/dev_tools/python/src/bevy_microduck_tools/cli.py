"""Run with the pinned upstream Python environment; artifacts belong in .scratch.

python -m bevy_microduck_tools.cli audit --source /path/to/pollen-robotics-microduck_rl
  --output /path/to/Bevy_Sim2Sim/.scratch/source_audit --skill standing --steps 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model_export import export_bam_reference, export_compiled
from .serialization import sha256_file, write_json
from .source_adapter import SKILL_TASKS, inventory, load_source, make_skill_cfg
from .workflow import STAGES, Rejection, TrainingBudget, admit, bind_candidate, bind_evidence, verify_claimed_learning, verify_claimed_runtime
from .authorization import LearningRequest


def _audit_capture(source_root: Path, output: Path, skill: str, *, steps: int | None = 60,
          substeps: int = 1, seed: int = 1000001, device: str = "cuda:0", video: bool = False,
          learning_iterations: int = 0, resume: Path | None = None,
          resume_origin: Path | None = None, _learning_binding: dict | None = None,
          _profile_binding: dict | None = None, _profile_manifest: dict | None = None,
          _profile_token=None, _phase_journal=None) -> dict:
    runtime_verified=None
    if learning_iterations:
        verify_claimed_learning(_learning_binding, iterations=learning_iterations, seed=seed, checkpoint=resume)
        runtime_verified=verify_claimed_runtime(_learning_binding, source_root)
    if _profile_binding is not None:
        from .profile_discovery import require_claimed_worker
        if _profile_manifest is None or _phase_journal is None or learning_iterations != 0:
            raise Rejection("Private zero-update source profile requires consumed identity and phase journal")
        require_claimed_worker(_profile_binding, _profile_token, _profile_manifest)
        if (skill != "standing" or substeps != 1 or seed != 1000001 or device != "cuda:0" or
                steps is not None or not video or resume is None or resume_origin is None):
            raise Rejection("Discovery source lifecycle arguments differ from fixed standing profile")
    from .evaluation import evaluate_raw
    from .trajectory import _capture_source
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evidence directories are immutable; use a new output directory per candidate run")
    source = load_source(source_root)
    output.mkdir(parents=True, exist_ok=True)
    inventory_path = output / "source_inventory.json"
    write_json(inventory_path, inventory(source))
    cfg, adoption = make_skill_cfg(source, skill, substeps=substeps, seed=seed)
    if _profile_manifest is not None and (
            source["commit"] != _profile_manifest["source_commit"] or
            adoption["sha256"] != _profile_manifest["adoption_sha256"] or
            adoption["family"] != _profile_manifest["profile"]["model_family"]):
        raise Rejection("Actual compiled source configuration differs from preflighted profile inputs")
    adoption_path = output / "adoption.json"
    write_json(adoption_path, adoption)
    if _phase_journal is not None:
        _phase_journal.mark("source_compile", source_commit=source["commit"], adoption_sha256=adoption["sha256"])
    definition = export_compiled(cfg, output, family=adoption["family"], timing_plan=adoption["timing_plan"])
    if _phase_journal is not None:
        _phase_journal.mark("source_compile_completed", compiled_definition_sha256=sha256_file(output / "compiled_robot.json"))
    capture = _capture_source(cfg, adoption, source, output, steps=steps, device=device, video=video,
                             learning_iterations=learning_iterations, resume=resume, resume_origin=resume_origin, _learning_binding=_learning_binding, _verified_runtime=runtime_verified,
                             _profile_manifest=_profile_manifest, _profile_token=_profile_token, _phase_journal=_phase_journal,
                             _profile_manifest_path=Path(_profile_binding["manifest_path"]) if _profile_binding is not None else None)
    evaluation = evaluate_raw(output / "source_trajectory.jsonl")
    rows = [json.loads(line) for line in (output / "source_trajectory.jsonl").read_text().splitlines()]
    policy_cases = []
    for row in rows:
        trace = row.get("policy_input_and_output_at_call")
        if row["kind"] == "policy_call" and trace and trace["phase"] == "inference":
            policy_cases.append({"input": trace["input_61"], "expected_output": trace["output_14"],
                                 "physics_tick": trace["physics_tick_before_step"], "time_seconds": trace["time_seconds"]})
    if not policy_cases:
        raise ValueError("No actual policy-call inputs/outputs were captured")
    write_json(output / "policy_inference_reference.json", {"schema": "torch_policy_call_reference_v1", "dtype": "float32",
               "model_sha256": sha256_file(output / "source_initial.onnx"), "cases": policy_cases,
               "method": "actual inference input cloned before Torch policy call and output cloned before env.step",
               "normalizer_embedded": True, "skill_qualified": False})
    contract = {"schema_version": "microduck_pollen_bam60_v1", "skill": skill,
                "required_skills": list(SKILL_TASKS) + ["sprint"],
                "model_sha256": sha256_file(output / "source_initial.onnx"),
                "robot_definition_sha256": sha256_file(output / "compiled_robot.json"),
                "actuator_order": definition["joint_order"], "home": capture["home"],
                "action_scale": capture["action_scale"], "physics_hz": 60, "policy_hz": 60,
                "source_substeps": substeps, "normalizer_embedded": True,
                "delay_seconds": {"motor_source_samples": [.015, .020, .025, .030],
                                  "joint_velocity": .020, "imu_source_samples": [0.0, .020],
                                  "imu_resample_period": 1.28},
                "status": "bounded learned source diagnostic" if learning_iterations else
                          ("restored checkpoint zero-update source diagnostic" if resume is not None else "fresh untrained source diagnostic"),
                "checkpoint_origin": capture["checkpoint_origin"],
                "timing_plan": adoption["timing_plan"]}
    runtime_keys = ("schema_version", "model_sha256", "robot_definition_sha256", "actuator_order", "home", "action_scale",
                    "physics_hz", "policy_hz", "normalizer_embedded", "delay_seconds")
    write_json(output / "policy_contract.json", {key: contract[key] for key in runtime_keys})
    write_json(output / "policy_runtime.json", {"platform": "linux_aarch64", "execution_provider": "CPU",
               "onnx_runtime_version": "1.30.0", "onnx_api_version": 28, "ort_crate": "2.0.0-rc.13",
               "native_library_sha256": "64e903a43a041240fd6bcffe0ac6d4fea47ef87bf24b9d097801bd00a9612a4b",
               "execution": "single CPU thread; sequential execution", "verified_here": "Torch export/ONNX schema only; Rust CpuPolicy verification recorded separately"})
    artifacts = {"source_inventory": inventory_path, "adoption": adoption_path,
                                "compiled_robot": output / "compiled_robot.json", "bam_parameters": output / "bam_params.json",
                                "checkpoint": output / "source_initial.pt", "onnx": output / "source_initial.onnx",
                                "source_trajectory": output / "source_trajectory.jsonl",
                                "observation_action_reference": output / "observation_action_reference.json",
                                "policy_contract": output / "policy_contract.json",
                                "policy_runtime": output / "policy_runtime.json",
                                "policy_inference_reference": output / "policy_inference_reference.json",
                                "checkpoint_origin": output / "checkpoint_origin.json",
                                "policy_freeze": output / "policy_freeze.json",
                                "home_pose_oracle": output / "home_pose_oracle.json",
                                "effective_xml": output / "effective_robot.xml",
                                "effective_assets": output / "effective_assets.json",
                                "evaluator": Path(__file__).with_name("evaluation.py")}
    if resume is not None:
        artifacts["restored_checkpoint_input"] = resume
    if resume_origin is not None:
        artifacts["restored_origin_input"] = resume_origin
        origin = json.loads(resume_origin.read_text())
        artifacts["originating_failed_run_identity"] = Path(origin["originating_failed_run"]["path"])
    if _profile_manifest is not None:
        artifacts["policy_post_export_freeze"] = output / "policy_post_export_freeze.json"
        artifacts["jit_producer_journal"] = output / "jit_producer_journal.json"
    if video:
        artifacts.update(source_video=output / "source_video.mp4", source_video_frames=output / "source_video.json")
    artifacts.update({f"tool_{path.stem}": path for path in Path(__file__).parent.glob("*.py")})
    artifacts.update({f"upstream_{index}": source_root / name for index, name in enumerate(source["files"])})
    assets_manifest = json.loads((output / "effective_assets.json").read_text())
    artifacts.update({f"vfs_{index}": Path(record["path"]) for index, record in enumerate(assets_manifest["assets"].values())})
    from .source_runtime_identity import collect_source_runtime
    runtime_identity_path=output / "source_runtime_identity.json"
    # This CPU collector keeps full lazy/native coverage false. A later approved
    # source-profile collector must close that gap before this candidate can learn.
    if _profile_manifest is None:
        write_json(runtime_identity_path, collect_source_runtime(source_root))
    else:
        # Parent and worker already re-read all v9 installed bytes. Preserve
        # that exact incomplete receipt; final actual-use scope is a separate
        # post-child artifact, never a silent global v9 coverage promotion.
        runtime_identity_path.write_bytes(Path(_profile_manifest["runtime_receipt"]["path"]).read_bytes())
    artifacts["source_runtime_identity"]=runtime_identity_path
    candidate = bind_candidate(artifacts, contract)
    write_json(output / "candidate.json", candidate)
    evidence = []
    for stage, report, coverage in [
        ("adoption", {"registry_tasks": len(inventory(source)["registered_tasks"]), "source_commit": source["commit"]},
         ["actual_registry", "rewards", "domain_randomization", "curriculum", "runner_save_resume_export"]),
        ("source_compile", {"counts": definition["counts"], "compiled_sha256": definition["sha256"]},
         ["full_body", "inertial_frames", "14_actuator_mapping", "effective_bam", "home"]),
    ]:
        admit(candidate, stage, evidence)
        report.update(candidate_id=candidate["candidate_id"], status="passed", stage=stage,
                      skill_qualified=False, review_complete=False)
        path = output / f"{stage}_evidence.json"
        write_json(path, report)
        evidence.append(bind_evidence(candidate, stage, path, coverage))
    natural = capture["natural_task_terminal"]
    complete = natural and video and evaluation["complete_first_source_episode"] and evaluation["physics_frames"] > 0
    stage = "source_rollout"
    report = {"candidate_id": candidate["candidate_id"], "stage": stage,
              "status": "passed" if complete else "diagnostic_only",
              "capture": capture, "independent_evaluation": evaluation,
              "skill_qualified": False, "review_complete": False,
              "reason": "complete natural source episode captured" if complete else
                        "bounded horizon or missing real video; complete first-episode gate remains unmet"}
    path = output / "source_rollout_evidence.json"
    write_json(path, report)
    if complete:
        admit(candidate, stage, evidence)
        coverage = ["true_task_terminal", "pre_reset", "per_integration_full_state", "contacts_and_impulses",
                    "source_timebase", "checkpoint_restore", "onnx_export", "corresponding_real_video"]
        evidence.append(bind_evidence(candidate, stage, path, coverage))
    else:
        write_json(output / "diagnostic_coverage.json", {"candidate_id": candidate["candidate_id"],
                   "status": "diagnostic_only", "coverage": ["horizon_truncation_pre_reset" if not natural else
                   "true_task_terminal", "source_timebase", "checkpoint_restore", "onnx_export"],
                   "promotion_allowed": False})
    write_json(output / "evidence.json", evidence)
    return {"candidate_id": candidate["candidate_id"], "stage": "source_rollout" if complete else "source_compile",
            "skill_qualified": False, "learning_allowed": False, "reason": "root GPT video/temporal review pending",
            "capture": capture, "evaluation": evaluation}


def audit(source_root: Path, output: Path, skill: str, *, steps: int | None = 60,
          substeps: int = 1, seed: int = 1000001, device: str = "cuda:0", video: bool = False,
          learning_iterations: int = 0, resume: Path | None = None,
          resume_origin: Path | None = None) -> dict:
    """Public source capture performs zero learning updates only."""
    if type(learning_iterations) is not int or learning_iterations != 0:
        raise Rejection("Public audit cannot learn; use the authorized bounded training entry")
    return _audit_capture(source_root, output, skill, steps=steps, substeps=substeps, seed=seed,
                          device=device, video=video, learning_iterations=0, resume=resume, resume_origin=resume_origin)


def _audit_authorized_learning(source_root: Path, output: Path, skill: str, *,
                               binding: dict, substeps: int, seed: int, device: str,
                               iterations: int, checkpoint: Path) -> dict:
    """Private worker claims the parent's actual consumed ledger run once."""
    if not isinstance(binding, dict) or "ledger_path" not in binding:
        raise Rejection("Private learning requires the consumed parent run binding")
    TrainingBudget(Path(binding["ledger_path"])).claim_learning_worker(binding)
    verify_claimed_learning(binding, iterations=iterations, seed=seed, checkpoint=checkpoint)
    parent = json.loads(Path(binding["candidate_path"]).read_text())
    adoption = json.loads(Path(parent["artifacts"]["adoption"]["path"]).read_text())
    inventory_record = json.loads(Path(parent["artifacts"]["source_inventory"]["path"]).read_text())
    if (skill != adoption["skill"] or substeps != parent["contract"]["source_substeps"] or
            source_root.resolve() != Path(inventory_record["root"]).resolve()):
        raise Rejection("Private learning source/profile differs from the consumed parent")
    if device.startswith("cuda:") is False or binding["request"]["gpus"] != 1:
        raise Rejection("Private learning requires the single-GPU parent execution")
    return _audit_capture(source_root, output, skill, steps=None, substeps=substeps, seed=seed,
                          device=device, video=True, learning_iterations=iterations, resume=checkpoint,
                          _learning_binding=binding)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit", help="Real registry, BAM compilation, short source rollout, save/resume/export")
    audit_parser.add_argument("--source", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    audit_parser.add_argument("--skill", choices=list(SKILL_TASKS), default="standing")
    audit_parser.add_argument("--steps", type=int, default=60)
    audit_parser.add_argument("--full-episode", action="store_true", help="Run until actual source terminated/time_limit without reducing its duration")
    audit_parser.add_argument("--substeps", type=int, default=1)
    audit_parser.add_argument("--seed", type=int, default=1000001)
    audit_parser.add_argument("--device", default="cuda:0")
    audit_parser.add_argument("--video", action="store_true", help="Render actual source frames with their raw timestamps")
    audit_parser.add_argument("--resume", type=Path)
    audit_parser.add_argument("--resume-origin", type=Path)
    runtime_parser = commands.add_parser("runtime-identity", help="CPU installed-byte identity; does not certify lazy native/GPU coverage")
    runtime_parser.add_argument("--source", type=Path, required=True)
    runtime_parser.add_argument("--output", type=Path, required=True)
    inventory_parser = commands.add_parser("inventory")
    inventory_parser.add_argument("--source", type=Path, required=True)
    inventory_parser.add_argument("--output", type=Path, required=True)
    bam_parser = commands.add_parser("bam-reference", help="Reproducible authoritative Torch BAM oracle")
    bam_parser.add_argument("--source", type=Path, required=True)
    bam_parser.add_argument("--output", type=Path, required=True)
    bam_parser.add_argument("--cases", type=int, default=512)
    bam_parser.add_argument("--seed", type=int, default=20260928)
    gate = commands.add_parser("gate", help="The shared entry check for train/resume/select/export/target/release")
    gate.add_argument("--candidate", type=Path, required=True)
    gate.add_argument("--evidence", type=Path, required=True)
    gate.add_argument("--review", type=Path)
    gate.add_argument("--stage", choices=STAGES, required=True)
    for name, kind in (("iterations", int), ("wall-seconds", float), ("seed", int), ("gpus", int), ("ledger", Path)):
        gate.add_argument(f"--{name}", type=kind)
    train = commands.add_parser("train", help="Gate, bound and measure one/two actual upstream PPO updates or checkpoint resume")
    for name in ("candidate", "evidence", "review", "source", "output", "ledger"):
        train.add_argument(f"--{name}", type=Path, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--iterations", type=int, default=1)
    train.add_argument("--wall-seconds", type=float, default=180)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--gpus", type=int, default=1)
    restore = commands.add_parser("restore", help="Bound a fresh environment zero-update natural inference episode from a proven checkpoint")
    for name in ("origin", "source", "output", "ledger"):
        restore.add_argument(f"--{name}", type=Path, required=True)
    restore.add_argument("--seed", type=int, required=True)
    restore.add_argument("--wall-seconds", type=float, default=180)
    restore.add_argument("--device", default="cuda:0")
    rejections = commands.add_parser("rejection-check", help="Tamper isolated copies of an actual source candidate")
    rejections.add_argument("--candidate", type=Path, required=True)
    rejections.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "audit":
        result = audit(args.source, args.output, args.skill, steps=None if args.full_episode else args.steps, substeps=args.substeps,
                       seed=args.seed, device=args.device, video=args.video, resume=args.resume, resume_origin=args.resume_origin)
    elif args.command == "runtime-identity":
        from .source_runtime_identity import collect_source_runtime, verify_source_runtime
        if args.output.exists():
            raise Rejection("Runtime identity receipt paths are immutable; use a new output")
        receipt=collect_source_runtime(args.source)
        write_json(args.output,receipt)
        result=verify_source_runtime(receipt,source_root=args.source,require_complete=False)
    elif args.command == "inventory":
        result = inventory(load_source(args.source))
        write_json(args.output, result)
        result = {"registered_tasks": len(result["registered_tasks"]), "sha256": result["sha256"]}
    elif args.command == "bam-reference":
        source = load_source(args.source)
        cfg, _ = make_skill_cfg(source, "standing")
        reference = export_bam_reference(cfg, args.output, cases=args.cases, seed=args.seed)
        result = {"cases": len(reference["cases"]), "sha256": reference["sha256"],
                  "bam_source_sha256": reference["bam_source_sha256"]}
    elif args.command == "restore":
        from .training import restore_bounded
        result = restore_bounded(args.origin, source_root=args.source, output=args.output,
                                ledger=args.ledger, seed=args.seed, wall_seconds=args.wall_seconds, device=args.device)
    elif args.command == "train":
        from .training import train_bounded
        result = train_bounded(args.candidate, args.evidence, args.review, source_root=args.source,
                               output=args.output, ledger=args.ledger, seed=args.seed,
                               iterations=args.iterations, wall_seconds=args.wall_seconds, device=args.device, gpus=args.gpus)
    elif args.command == "rejection-check":
        from .rejections import check_rejections
        result = check_rejections(args.candidate, args.output)
    else:
        candidate = json.loads(args.candidate.read_text())
        evidence = json.loads(args.evidence.read_text())
        review = json.loads(args.review.read_text()) if args.review else None
        request = None
        if args.stage == "learning":
            if any(getattr(args, name) is None for name in ("iterations", "wall_seconds", "seed", "gpus", "ledger")):
                parser.error("learning gate requires iterations/wall-seconds/seed/gpus/ledger; inspection does not consume authorization")
            request = LearningRequest(args.iterations, args.wall_seconds, args.seed, args.gpus, str(args.ledger.resolve()))
        elif any(getattr(args, name) is not None for name in ("iterations", "wall_seconds", "seed", "gpus", "ledger")):
            parser.error("learning request fields only apply to stage learning")
        admit(candidate, args.stage, evidence, review, request=request)
        if args.stage=="learning":
            from .source_runtime_identity import verify_candidate_runtime
            inventory_root=json.loads(Path(candidate["artifacts"]["source_inventory"]["path"]).read_text())["root"]
            verify_candidate_runtime(candidate,Path(inventory_root))
        result = {"inspection_only": True, "authorization_consumed": False, "admitted": args.stage, "candidate_id": candidate["candidate_id"]}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

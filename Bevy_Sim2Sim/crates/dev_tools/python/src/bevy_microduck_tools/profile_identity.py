"""Finite, byte-addressed inputs for one zero-update source discovery profile."""
from __future__ import annotations

import os
import hashlib
import subprocess
import sys
import sysconfig
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .authorization import Rejection
from .profile_observer import _platform, _real_file
from .serialization import identity, sha256_file

SCHEMA = "microduck_source_profile_inputs_v2"
MODE = "zero_update_discovery"
PROFILE = {"skill": "standing", "physics_hz": 60, "policy_hz": 60,
           "source_substeps": 1, "env_count": 1, "seed": 1000001,
           "device": "cuda:0", "model_family": "leg_allcollisions"}
RESTORED_CHECKPOINT_SHA256 = "9af38b14aa725b4f0c4061f55ecc70e605b1552c296cc7f25d0089c494881c05"
SELECTED_ENV = ("PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONHOME",
                "HOME", "TMPDIR", "XDG_CACHE_HOME", "CUBLAS_WORKSPACE_CONFIG",
                "PYTHONHASHSEED", "PYTHONSAFEPATH", "PYTHONNOUSERSITE",
                "CUDA_MODULE_LOADING", "CUDA_CACHE_PATH", "CUDA_CACHE_DISABLE",
                "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_LAUNCH_BLOCKING",
                "PYTORCH_CUDA_ALLOC_CONF", "TORCH_CUDA_ARCH_LIST",
                "MUJOCO_GL", "WARP_CACHE_PATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
EXECUTION_ENV_PREFIXES = ("CUDA_", "WARP_", "MUJOCO_", "TORCH_", "PYTORCH_", "NVIDIA_", "LD_", "PYTHON_")


def _execution_env_hashes(environment=None) -> dict[str, str]:
    environment = os.environ if environment is None else environment
    return {key: hashlib.sha256(value.encode()).hexdigest() for key, value in sorted(environment.items())
            if key.startswith(EXECUTION_ENV_PREFIXES) or key in {
                "PATH", "HOME", "TMPDIR", "XDG_CACHE_HOME", "CUBLAS_WORKSPACE_CONFIG",
                "PYTHONHASHSEED", "PYTHONSAFEPATH", "PYTHONNOUSERSITE",
                "OMP_NUM_THREADS", "MKL_NUM_THREADS"}}


def private_cache_environment(manifest: dict, base=None) -> dict[str, str]:
    result = dict(os.environ if base is None else base)
    root = Path(manifest["private_cache_root"])
    result["WARP_CACHE_PATH"] = str(root / "warp")
    result["CUDA_CACHE_PATH"] = str(root / "cuda")
    return result


def _regular_catalog(paths) -> list[dict]:
    records = []
    for path in sorted({str(Path(value).absolute()) for value in paths}):
        if _platform(path):
            raise Rejection("Platform interfaces are observations, never claimed as hashed file inputs")
        records.append(_real_file(path))
    return records


def _stdlib_paths() -> list[Path]:
    root = Path(sysconfig.get_path("stdlib"))
    if not root.is_dir():
        raise Rejection("Active Python stdlib root missing")
    result = []
    for path in root.rglob("*"):
        if "site-packages" in path.parts or "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
            continue
        if path.is_symlink() and path.is_dir():
            raise Rejection(f"Unbound stdlib directory symlink: {path}")
        if path.is_file():
            result.append(path)
    return result


def make_inputs(*, source_root: Path, runtime_receipt: Path, origin_path: Path,
                output_root: Path, source_files: list[Path], additional_files: list[Path],
                source_commit: str, adoption_sha256: str, nonce: str | None = None,
                include_stdlib: bool = True) -> dict:
    """Build provisional inputs on CPU. This does not reserve/execute science."""
    from .provenance import restored_origin
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    receipt = runtime_receipt.resolve()
    origin_path = origin_path.resolve()
    try:
        import json
        v9 = json.loads(receipt.read_text())
        origin_raw = json.loads(origin_path.read_text())
    except (OSError, ValueError) as error:
        raise Rejection(f"Cannot read adopted receipt/origin: {error}") from error
    checkpoint = Path(origin_raw["checkpoint"]["path"])
    origin = restored_origin(origin_path, checkpoint)
    if (origin.get("skill") != PROFILE["skill"] or origin.get("source_substeps") != 1 or
            origin["checkpoint"]["sha256"] != RESTORED_CHECKPOINT_SHA256 or
            origin["checkpoint"]["sha256"] != sha256_file(checkpoint)):
        raise Rejection("Origin differs from adopted zero-update standing profile")
    if (v9.get("schema") != "microduck_source_runtime_identity_v1" or
            v9.get("source_root") != str(source_root) or
            v9.get("coverage", {}).get("native_profile_complete") is not False):
        raise Rejection("Discovery must start from exact incomplete v9 installed-file receipt")
    if not isinstance(adoption_sha256, str) or len(adoption_sha256) != 64:
        raise Rejection("Source adoption SHA required before profile discovery")
    tools = sorted(Path(__file__).parent.glob("*.py"))
    wheels = [Path(path).absolute() for path in additional_files if Path(path).suffix == ".whl"]
    if len(wheels) != 1:
        raise Rejection("Exactly one frozen tool wheel must be declared")
    own = list(source_files) + list(additional_files) + [checkpoint, origin_path, receipt,
          Path(sys.executable), Path("/usr/bin/strace")]
    if include_stdlib:
        own += _stdlib_paths()
    private_root = output_root / "private_cache"
    child_environment = dict(os.environ)
    child_environment.update(WARP_CACHE_PATH=str(private_root / "warp"),
                             CUDA_CACHE_PATH=str(private_root / "cuda"))
    payload = {"schema": SCHEMA, "mode": MODE, "nonce": nonce or str(uuid.uuid4()),
               "source_root": str(source_root), "source_commit": source_commit,
               "profile": dict(PROFILE), "adoption_sha256": adoption_sha256,
               "runtime_receipt": {"path": str(receipt), "sha256": sha256_file(receipt),
                                   "identity": v9["identity"], "native_profile_complete": False},
               "checkpoint_origin": {"path": str(origin_path), "sha256": sha256_file(origin_path),
                                     "checkpoint_path": str(checkpoint.resolve()),
                                     "checkpoint_sha256": sha256_file(checkpoint),
                                     "known_prior_ppo_iterations": origin["known_prior_ppo_iterations"]},
               "python_executable": str(Path(sys.executable).absolute()),
               "python_version": sys.version, "stdlib_root": sysconfig.get_path("stdlib"),
               "adopted_source_files": sorted(str(path.absolute()) for path in source_files),
               "tool_python_files": sorted(str(path.absolute()) for path in tools),
               "tool_wheel": str(wheels[0]),
               "resolution": {key: os.environ.get(key) for key in SELECTED_ENV},
               "execution_environment_sha256": _execution_env_hashes(),
               "child_resolution": {key: child_environment.get(key) for key in SELECTED_ENV},
               "child_execution_environment_sha256": _execution_env_hashes(child_environment),
               "private_cache_root": str(private_root),
               "cache_policy": {"warp": "empty_run_private_cache", "cuda": "empty_run_private_cache",
                                "adopted_seed_files": [], "cold_compile_may_exceed_180s": True},
               "declared_files": _regular_catalog(own), "output_root": str(output_root),
               "parent_preflight_max_seconds": 120, "source_child_hard_seconds": 180,
               "covered_modes_requested": ["source_compile", "env_graph", "deterministic_inference",
                                           "natural_terminal_reset", "save_resume_export", "real_video"],
               "not_executed": ["PPO_stochastic_collection", "PPO_backward", "Adam_step", "learn_reset"],
               "qualification": "provisional input identity, zero updates; requires actual-use observer and root review"}
    payload["input_id"] = identity(payload)
    return payload


def validate_inputs(manifest: dict) -> None:
    fields = {"schema", "mode", "nonce", "source_root", "source_commit", "profile",
              "adoption_sha256", "runtime_receipt", "checkpoint_origin", "python_executable",
              "python_version", "stdlib_root", "adopted_source_files", "tool_python_files", "tool_wheel",
              "resolution", "execution_environment_sha256", "child_resolution",
              "child_execution_environment_sha256", "private_cache_root", "cache_policy",
              "declared_files", "output_root",
              "parent_preflight_max_seconds", "source_child_hard_seconds", "covered_modes_requested",
              "not_executed", "qualification", "input_id"}
    if not isinstance(manifest, dict) or set(manifest) != fields or manifest["schema"] != SCHEMA or manifest["mode"] != MODE:
        raise Rejection("Unknown or incomplete zero-update source input schema")
    if manifest["profile"] != PROFILE or manifest["parent_preflight_max_seconds"] != 120 or manifest["source_child_hard_seconds"] != 180:
        raise Rejection("Discovery profile or hard bounds changed")
    try:
        if str(uuid.UUID(manifest["nonce"])) != manifest["nonce"]:
            raise ValueError()
    except (TypeError, ValueError, AttributeError):
        raise Rejection("Discovery run requires a canonical unique nonce") from None
    if identity({key: value for key, value in manifest.items() if key != "input_id"}) != manifest["input_id"]:
        raise Rejection("Source profile input manifest changed")
    if (manifest["runtime_receipt"].get("native_profile_complete") is not False or
            manifest["checkpoint_origin"].get("checkpoint_sha256") != RESTORED_CHECKPOINT_SHA256 or
            manifest["checkpoint_origin"].get("known_prior_ppo_iterations") != 1 or
            manifest["not_executed"] != ["PPO_stochastic_collection", "PPO_backward", "Adam_step", "learn_reset"]):
        raise Rejection("Discovery cannot masquerade as native-complete or learning-covered profile")
    private_root = str(Path(manifest["output_root"]) / "private_cache")
    if (manifest["private_cache_root"] != private_root or
            manifest["cache_policy"] != {"warp": "empty_run_private_cache", "cuda": "empty_run_private_cache",
                                          "adopted_seed_files": [], "cold_compile_may_exceed_180s": True}):
        raise Rejection("Discovery cache must be new, run-private and unseeded")
    child_resolution = {**manifest["resolution"], "WARP_CACHE_PATH": private_root + "/warp",
                        "CUDA_CACHE_PATH": private_root + "/cuda"}
    if manifest["child_resolution"] != child_resolution:
        raise Rejection("Child cache resolution is not bound to unique run output")
    records = manifest["declared_files"]
    if not isinstance(records, list) or not records or [row["path"] for row in records] != sorted({row["path"] for row in records}):
        raise Rejection("Declared actual-byte catalog missing, duplicated or unordered")
    for row in records:
        if set(row) != {"path", "resolved_path", "device_major", "device_minor", "inode", "size", "sha256"}:
            raise Rejection("Declared file lacks exact path, device, inode and actual SHA")
        if not Path(row["path"]).is_absolute() or len(row["sha256"]) != 64:
            raise Rejection("Declared file path or digest invalid")
    catalog_paths = {row["path"] for row in records}
    for name in ("adopted_source_files", "tool_python_files"):
        group = manifest[name]
        if (not isinstance(group, list) or not group or group != sorted(set(group)) or
                any(not Path(path).is_absolute() or path not in catalog_paths for path in group)):
            raise Rejection(f"Incomplete declared {name} source set")
    if manifest["tool_wheel"] not in catalog_paths or not manifest["tool_wheel"].endswith(".whl"):
        raise Rejection("Frozen tool wheel bytes absent from declared catalog")
    for name in ("source_root", "output_root", "python_executable", "stdlib_root"):
        if not isinstance(manifest[name], str) or not Path(manifest[name]).is_absolute():
            raise Rejection(f"Discovery {name} must be absolute")
    for field in ("execution_environment_sha256", "child_execution_environment_sha256"):
        if (not isinstance(manifest[field], dict) or
                any(not isinstance(key, str) or not isinstance(value, str) or len(value) != 64
                    for key, value in manifest[field].items())):
            raise Rejection("Invalid hashed execution environment")


@dataclass(frozen=True)
class VerifiedDeclaredInputs:
    process_id: int
    nonce: str
    input_id: str
    mode: str
    source_root: str
    manifest_sha256: str
    native_profile_complete: bool = False
    runtime_verified: bool = False
    verification_phase: str = "parent"


def verify_declared_inputs(manifest: dict, *, manifest_path: Path | None = None,
                           require_runtime: bool = True, for_child: bool = False) -> VerifiedDeclaredInputs:
    """Full byte re-read once per process, before any source compile/environment."""
    validate_inputs(manifest)
    if manifest["python_executable"] != str(Path(sys.executable).absolute()) or manifest["python_version"] != sys.version:
        raise Rejection("Actual interpreter differs from profile input")
    expected_resolution = manifest["child_resolution"] if for_child else manifest["resolution"]
    expected_hashes = manifest["child_execution_environment_sha256"] if for_child else manifest["execution_environment_sha256"]
    if expected_resolution != {key: os.environ.get(key) for key in SELECTED_ENV}:
        raise Rejection("Actual library/interpreter resolution environment changed")
    if expected_hashes != _execution_env_hashes():
        raise Rejection("Actual CUDA/Warp/MuJoCo/Torch execution environment changed")
    receipt = Path(manifest["runtime_receipt"]["path"])
    if sha256_file(receipt) != manifest["runtime_receipt"]["sha256"]:
        raise Rejection("Installed-file receipt bytes changed")
    if require_runtime:
        import json
        from .provenance import restored_origin
        from .source_runtime_identity import verify_source_runtime
        checkpoint_path = Path(manifest["checkpoint_origin"]["checkpoint_path"])
        origin_path = Path(manifest["checkpoint_origin"]["path"])
        if (sha256_file(checkpoint_path) != RESTORED_CHECKPOINT_SHA256 or
                sha256_file(origin_path) != manifest["checkpoint_origin"]["sha256"]):
            raise Rejection("Restored checkpoint or its proven origin changed")
        proven_origin = restored_origin(origin_path, checkpoint_path)
        if proven_origin["known_prior_ppo_iterations"] != 1:
            raise Rejection("Actual retained source checkpoint update provenance changed")
        source_receipt = json.loads(receipt.read_text())
        verified = verify_source_runtime(source_receipt, source_root=Path(manifest["source_root"]), require_complete=False)
        if verified["complete"] or verified["identity"] != manifest["runtime_receipt"]["identity"]:
            raise Rejection("Discovery cannot use an altered or promoted v9 source receipt")
        root = Path(manifest["source_root"])
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
        adopted = sorted(str((root / name).absolute()) for name in tracked if name and
                         (name.startswith("src/") or name in {"pyproject.toml", "uv.lock", "scripts/train.py"}))
        if commit != manifest["source_commit"] or adopted != manifest["adopted_source_files"]:
            raise Rejection("Pinned actual upstream commit or adopted source file set changed")
        tools = sorted(str(path.absolute()) for path in Path(__file__).parent.glob("*.py"))
        if tools != manifest["tool_python_files"]:
            raise Rejection("Discovery package Python module set changed")
        with zipfile.ZipFile(manifest["tool_wheel"]) as wheel:
            member_names = {name for name in wheel.namelist() if name.startswith("bevy_microduck_tools/") and name.endswith(".py")}
            expected_members = {f"bevy_microduck_tools/{Path(path).name}" for path in tools}
            if member_names != expected_members:
                raise Rejection("Frozen wheel module set differs from actual imported tool files")
            for path in tools:
                if wheel.read(f"bevy_microduck_tools/{Path(path).name}") != Path(path).read_bytes():
                    raise Rejection(f"Actual imported workflow module differs from frozen wheel: {path}")
        stdlib_actual = {str(path.absolute()) for path in _stdlib_paths()}
        declared_paths = {row["path"] for row in manifest["declared_files"]}
        if not stdlib_actual <= declared_paths or sysconfig.get_path("stdlib") != manifest["stdlib_root"]:
            raise Rejection("Python stdlib nonbytecode source set changed or is incomplete")
    for record in manifest["declared_files"]:
        if _real_file(record["path"]) != record:
            raise Rejection(f"Declared input changed: {record['path']}")
    digest = sha256_file(manifest_path) if manifest_path is not None else identity(manifest)
    return VerifiedDeclaredInputs(os.getpid(), manifest["nonce"], manifest["input_id"], MODE,
                                  manifest["source_root"], digest, False, require_runtime,
                                  "child" if for_child else "parent")


def require_zero_token(token: VerifiedDeclaredInputs, manifest: dict, *, manifest_path: Path | None = None) -> None:
    validate_inputs(manifest)
    digest = sha256_file(manifest_path) if manifest_path is not None else identity(manifest)
    if (not isinstance(token, VerifiedDeclaredInputs) or token.process_id != os.getpid() or
            token.nonce != manifest["nonce"] or token.input_id != manifest["input_id"] or
            token.mode != MODE or token.source_root != manifest["source_root"] or
            token.manifest_sha256 != digest or token.native_profile_complete is not False):
        raise Rejection("Science flow lacks exact same-process zero-update verified input token")

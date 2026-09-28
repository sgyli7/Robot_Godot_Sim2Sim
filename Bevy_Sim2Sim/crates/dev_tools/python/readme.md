# MicroDuck development tools

The package consumes the pinned Pollen source registry and scientific environment.
It is not part of the standalone game. `uv.lock` locks this small tools package;
the adopted upstream `uv.lock` and live dependency versions are recorded separately.

Build with `uv build --project crates/dev_tools/python`. Run scientific commands
with the adopted source `.venv/bin/python` and `PYTHONPATH=crates/dev_tools/python/src`.
The ordinary Python entry point is `python -m bevy_microduck_tools.cli`.

- `inventory --source SOURCE --output FILE` records all 33 actual registered
  MicroDuck tasks, complete rewards, randomization, curricula and runner classes.
- `audit --source SOURCE --output NEW_DIRECTORY --skill standing --full-episode
  --video` compiles the full robot, records every real source integration and
  natural terminal before reset, saves/restores the real runner, exports ONNX,
  and records actual policy-call input/output pairs and corresponding source video.
  `--steps N` is a diagnostic horizon; it cannot satisfy the first-episode gate.
- `bam-reference --source SOURCE --output FILE --cases 512` exports reproducible
  authoritative Torch motor/friction vectors, including electrical saturation,
  high-speed and strict directional-tie cases.
- `rejection-check --candidate FILE --output NEW_DIRECTORY` exercises changed
  weights, missing terminal, wrong mapping, empty coverage and expired review on
  isolated copies of a real candidate.
- `gate --candidate FILE --evidence FILE --review FILE --stage learning --iterations 1
  --wall-seconds 180 --seed 1000001 --gpus 1 --ledger FILE` checks
  current content identities, predecessor evidence, explicit root GPT stage authorization
  and the complete learning request. Gate inspection does not consume authorization.
- `train --candidate FILE --evidence FILE --review FILE --source SOURCE --output
  NEW_DIRECTORY --ledger FILE --seed 1000001 --iterations 1 --wall-seconds 180`
  uses that same gate, resumes the actual upstream checkpoint/optimizer, executes
  one or two real upstream PPO updates in an isolated process, and records measured
  GPU time. It reads a bounded IPC locator while waiting, verifies the child's
  atomically published result file, and kills the worker at a hard deadline of
  at most 180 seconds. Full trajectory/video reports never travel through the pipe.
  This first implementation deliberately
  supports only structural short training; long training is not implemented yet.

All evidence belongs in `.scratch`. Each source/candidate run requires an unused
directory. Source, tools, effective config, VFS assets, compiled robot, checkpoint,
ONNX, evaluator, trajectory and video bytes participate in candidate identity.
An explicit `superseded.json` prevents promotion of an old candidate. Every new
model requires independent evaluation and a corresponding fresh root GPT review.

The timestamp adapter uses actual environment integration counters. Motor reset
write/forward returns known HOME minus encoder bias without entering history.
The first true policy tick records its actual target at time zero. Repeated reads
and appends at the same physical timestamp are idempotent. Original delay seconds
and their sampling probabilities are preserved; quantization and schedule changes
are recorded. Source substeps may be requested with `--substeps`; target Rapier
single-step 60 Hz still requires separate validation.

The diagnostic evaluator reads raw state rather than reward. It verifies full
body/lifecycle/contact capture and computes basic physical metrics. It does not
yet implement the ten independent skill oracles; no diagnostic or short learning
run is a qualified skill. Sprint is explicitly rejected until its independent
recipe is fixed. Final release additionally requires complete ten-skill results,
held-out coverage, three independent training seeds, regression, safety, paired
target validation and a root GPT review of actual temporal data and real video.

Raw schema v2 gives only actual integration frames an executed contact interval
and impulse. Reset/forward snapshots keep instantaneous forces, explicitly stamped
manager-result caches, last real inference calls, and the BAM warmup feedback/load
epochs. This collection labeling does not clear or modify the native physical state.

`compiled_robot.json` contains the BAM-edited compiled fields. Its internal
`sha256` is a canonical payload identity; runtime `robot_definition_sha256` is the
SHA256 of the actual file bytes. `effective_robot.xml` requires the captured VFS
manifest and resource bytes; XML by itself is not claimed to be reproducible.
Structure reconstruction checks `nq` and `ngeom`; rounded XML is not claimed to
reproduce every floating-point compiled field bit for bit. Actual `jnt_stiffness`,
`qpos_spring`, and `exclude_signature` arrays are exported, not assumed to be zero.
`home_pose_oracle.json` comes from source MuJoCo `mj_forward`, without integration.
`policy_contract.json` is the strict Rust runtime contract; other source metadata
is kept separately. `policy_inference_reference.json` contains actual pre-step
61D Torch inputs and 14D outputs, with normalization embedded in the ONNX actor.

Run tests with `python -m unittest discover -s crates/dev_tools/python/tests -v`.
The source adapter tests use the installed Torch/mjlab implementation on CPU and
skip when that scientific environment is absent. They are interface checks, not
robot or policy qualification.

Review v2 requires an immutable UUID, explicit `authorized_stages`, an exact boolean
`learning_allowed`, and complete learning limits (iterations, wall time, seed, GPU
count, maximum runs, and the resolved shared ledger path). Historical approvals
remain evidence but grant no new action. A training entry validates the actual
request before reserve/spawn. Under one file lock, budget and review consumption
are atomically committed; failed or timed-out launches still consume permission.
Copies, changed formatting, alternate ledgers, and repeated calls cannot replenish
an authorization. This runner currently supports one indexed CUDA device only.

PPO collection and post-learning reset/evaluation use matching PyTorch inference
mode. The original BAM reset clears its own previous computed motor torque; the
original reward reset takes current native actuator feedback. No arbitrary cache
cloning or fabricated zero feedback is used. CPU production-function fixtures
verify these named boundaries; the combined robot learn/reset lifecycle still
requires a separately authorized real run. The tools package performs no such run
as part of its build or tests.

Explicit review permissions govern promotion stages: learning, selection, export,
target validation, GPT promotion review and release. Initial adoption, compilation
and zero-update capture do not require a prior review. If a review is supplied to
any stage, its authorization is checked for that exact stage.

Public `audit` and `capture_source` reject every nonzero learning request before
source/config/environment work. Learning is reachable only through the private
bounded worker, with the parent's consumed run ID, candidate/review/request and
manifest-byte identity. A worker atomically claims that run once, verifies the
source/profile/checkpoint, and the private capture checks the active claim before
constructing an environment. Direct private calls without a valid consumed claim
are rejected. These are workflow boundaries, not a sandbox against arbitrary
Python monkeypatching or someone modifying the trusted ledger on disk.

`runtime-identity --source SOURCE --output NEW_RECEIPT.json` reads the actual
installed scientific files on CPU. The strict receipt binds Python, imported
module origins, recursive active distribution dependencies, independent complete
package-root file sets, native loaded paths, headers and parameter/data bytes.
Installed RECORD entries locate roots; their claimed hashes never replace actual
SHA256 reads. Bytecode is excluded. Directory symlinks without explicit recursive
coverage are refused. Same-version replacements, additional/missing files,
changed import/library paths and malformed/incomplete receipts fail closed.

Learning requires this receipt as a candidate artifact. Its current CPU collector
declares `native_profile_complete=false` and `complete=false`: full lazy/JIT/native
source-profile coverage has not been demonstrated. It cannot qualify any existing
or new candidate for learning. The parent checks complete actual bytes before
budget reserve/spawn; the private worker checks them once before its first
scientific compile/environment/runner flow. An immutable process/run/manifest-bound
token carries that check into the same flow without re-reading all files. This
requires a trusted environment with no concurrent writes; it is not a permanent
filesystem lock. Initial zero-update capture collects evidence without granting
promotion. Runtime-check CPU wall time is not evidence of GPU learning or updates.

# MicroDuck source workflow tools

Version `0.1.9` adds a read-only CPU v3 identity and review check. It binds
the v2 input file bytes, a freshly sealed finite negative-lookup envelope,
an exact scope plan and an acyclic review request. The review check validates
exact request bytes and expiry without claiming or consuming approval.
Single-use authorization is **not implemented**; the earlier scratch claim
prototype failed independent directory, lock-inode and deadline audits and
was removed from the shipped package. The public discovery command remains
disabled. No review of this CPU identity authorizes source or learning work.
The retired v2 private profile worker also rejects immediately, including
direct module invocation. The historical `reserve_discovery` Python function
remains callable for CPU budget-accounting fixtures and can write a ledger if
explicitly invoked; it does not make the retired profile worker available or
grant a positive profile. Do not use it with a real review or budget ledger.

Version `0.1.8` closes a zero-update discovery budget in one locked ledger
transaction. Every requested outcome is recorded only as `budget_exhausted`
with its requested status and an explicit uncertainty reason: a clock check
before `replace`/`fsync` cannot prove that the durable commit preceded the
deadline. The stored budget charge is conservative, not a post-commit GPU
measurement. CPU fixtures delay the commit across its deadline and inject
failures around replacement and directory fsync; no reader observes a
completed or undercharged failure state. The actual source `discover` entry
remains disabled. A future
v3 needs a separate, externally supervised positive-completion protocol.
Historical negative-lookup witnesses created after an old trace are diagnostic
only; witness drift makes them unusable for a new run.

Version `0.1.7` adds a **CPU-only finite negative-lookup parser**. A rule must
name an exact pathname, read-only syscall arguments, errno, phase and maximum
count. The sealer records the real first missing component or non-symlink
`readlinkat` target, including parent directories and intermediate symlinks;
the parser checks those witnesses again after the trace. Unlisted or changed
failures remain diagnostic. The contract is not yet bound to a source input
manifest or external systemd scope, so the public `discover` command rejects
before reading a manifest, reserving a budget or starting a worker. This
package does not approve source capture, qualify native dependencies or grant
PPO learning. The learning gate still rejects the incomplete v9 receipt.

The CPU API is `seal_contract(spec)` followed by
`parse_trace(trace_dir, command_exitcode=..., negative_contract=sealed)`.
`spec` has schema `microduck_negative_lookup_spec_v1` and a finite `rules`
list. Each rule contains `rule_id`, exact `path`, `syscall`, `errno`, `phase`,
`dirfd="AT_FDCWD"`, exact `qualifier`, and `max_count`. The sealed JSON must
be saved before any future observed run and bound to a new source input ID;
the current package only demonstrates the parser with CPU fixtures and an
older diagnostic trace. Even matching all 80 failures in that old trace leaves
17 unrelated diagnostics, so that old trace is not made complete by this tool.

The following v11 command examples are retained for historical interface
reference only. Running `discover` with this package always rejects, as does
the retired private profile worker.

The profile is fixed to official source commit
`5946fd9cdbc58956424420153e51975af3b30d77`, standing,
`leg_allcollisions`, MuJoCo physics/policy 60 Hz, one source substep, one world,
training-partition seed `1000001`, `cuda:0`, a complete natural first episode,
real video and the proven once-updated checkpoint origin. Discovery invokes
**zero** `runner.learn`, `alg.update`, backward or Adam steps. Those modes and
learn→reset remain `not_executed` in its receipt.

There are two separate commands for a later expressly approved source run:

```bash
bevy-microduck-profile prepare \
  --source /home/ethan/Projects/Sai_Lab/upstream/pollen-robotics-microduck_rl \
  --runtime-receipt /ABSOLUTE/PATH/v9_actual_source_runtime_cpu/source_runtime_identity.json \
  --origin /ABSOLUTE/PATH/standing_source60_restored_zero_update_v7_20260927T201811Z_origin_input.json \
  --wheel /ABSOLUTE/PATH/bevy_microduck_tools-0.1.6-py3-none-any.whl \
  --manifest /ABSOLUTE/UNIQUE/PATH/profile_inputs.json \
  --output-root /ABSOLUTE/UNIQUE/PATH/source_run

bevy-microduck-profile discover \
  --manifest /ABSOLUTE/PATH/profile_inputs.json \
  --review /ABSOLUTE/PATH/NEW_ROOT_ZERO_UPDATE_DISCOVERY_REVIEW.json \
  --ledger /ABSOLUTE/PATH/training_budget_ledger.json \
  --output-root /ABSOLUTE/UNIQUE/PATH/source_run --wall-seconds 180
```

These commands are **documentation for root review**, not a request to run
them. `prepare` uses CPU imports/configuration and reads actual adopted source,
wheel, interpreter, stdlib and checkpoint bytes. The later `discover` entry
rechecks all declared bytes in a 120-second CPU parent preflight **before**
any resource reservation, then atomically consumes an exact one-use root
`microduck_root_discovery_review_v2`. Review fields explicitly bind input ID,
nonce, zero-update mode, seed, one GPU, 180-second bound and ledger path;
`learning_allowed` must be false. A failed child still spends the authorization
and its measured reserved-GPU wall time. There is no retry implied by failure.

The private child claims that same consumed run, rechecks declared bytes once
before first source compile/environment/runner use, and holds a same-PID
zero-update token. The manifest also binds `HOME`, `TMPDIR`,
`XDG_CACHE_HOME`, `PYTHONHASHSEED`, `PYTHONSAFEPATH`,
`PYTHONNOUSERSITE` and `CUBLAS_WORKSPACE_CONFIG` in the execution
environment. A passive `strace` process-group observer captures file, exec,
memory and `fstat` events for short-lived native libraries and children.
Phase snapshots capture imports and native maps; Warp build/load hooks record
actual generated or adopted blobs without changing the source solver, rewards,
observations, actions or integration order. All observer and video processes
share one monotonic budget: observer closure stops by 165 seconds, sealing by
175 seconds, leaving five seconds for the ledger at a 180-second request.
Any discovered overrun is recorded as
`budget_exhausted`; no completed discovery may use an over-limit result.
This is a fail-closed accounting bound, not a hard real-time guarantee against
an uninterruptible filesystem operation or stalled ledger `fsync`/lock. A
reviewer requiring a literal process-return guarantee below 180 seconds must
keep actual discovery unauthorized until a separately supervised execution
and ledger protocol is implemented.

Actual-use records distinguish `post_use_current_bytes_with_observed_open_inode`
from weaker `post_use_current_bytes_without_observed_open_inode` and exec-path
identity. Successful open, stat and readlink metadata and failed candidate
lookups are retained with syscall, phase and actual path. An unbound failed
lookup, missing opened inode, unexplained successful metadata or unknown
traced syscall makes the trace diagnostic and prevents candidate creation.
Normal Python/loader ENOENT lookups currently have no predeclared negative
contract, so a first actual profile run may produce useful evidence yet
remain incomplete. Discovery cannot claim pre-use blocking for syscall
observations.
`/dev`, `/proc` and `/sys` are platform-interface observations, not fake
content hashes; successful `/proc` metadata is diagnostic because aliases
can refer to ordinary files. Unknown origins, missing/partial traces, unverified producer
recipes and opaque driver/JIT internals stay provisional for root review.
Successful natural capture may create a **new** `candidate_profile.json`, whose
source runtime v9 receipt remains incomplete and whose dependency receipt
always has `native_profile_complete=false`, `learning_allowed=false`. Source
force components in the raw trajectory are phase-stamped and separately
include `qfrc_bias`, `qfrc_constraint`, own friction, applied actuator and 14
joint IDs; the CPU fixtures below do not exercise their robot values.

This workflow trusts the pinned interpreter and a maintenance window with no
concurrent edits to the scientific venv, source, declared files, cache seeds or
driver libraries. Same-process tokens and byte checks are not a permanent
filesystem lock or a defense against malicious Python/native code. A root
review of a discovered profile would still be separate from plant/IO/BAM,
video/trajectory, Rapier transfer, skill and future learning-mode decisions.

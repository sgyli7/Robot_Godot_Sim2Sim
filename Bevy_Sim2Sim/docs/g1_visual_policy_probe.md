# Native G1 offline visual policy probe

This development binary sends one unchanged saved native ego PNG and its original
stamp through the existing matched `StaticPolicyClient` or `MobilePolicyClient`.
It validates the native 53-body/43-joint camera pairing using the shared
`snapshot_from_capture` guard, maps measured joints by the frozen 43 names, and
uses the decoded RGB pixels. It starts no service, renderer, physics or executor.

The root agent registers the module and binary behind `rendering_preview`, then
builds/tests using its coordinated warm Cargo target. This source-only delivery
has not compiled, executed tests, contacted a service or loaded a model.

```bash
/absolute/path/to/g1_visual_policy_probe \
  /absolute/capture/ego_640x480.png \
  /absolute/capture/ego_stamp.json \
  /absolute/existing_parent/NEW_policy_receipt.json \
  --profile mobile_box \
  --endpoint http://127.0.0.1:5558/infer \
  --offline-diagnostic
```

All three flags are required. Use `--profile static_apple` with the static
service's explicit endpoint. `http://localhost:PORT/infer` is mapped to literal
`127.0.0.1` without DNS; the existing client rejects other hosts, credentials,
queries, fragments and paths. No live mode or instruction override exists.
Profile mapping, model revision, reference instruction, 40/50-step horizon and
20 ms action period remain attached to their original contract. The mobile
profile uses the published `gn1_6` revision, never the old main checkpoint.

The requested total HTTP timeout is 30 seconds. The existing transport disables
proxying, redirects and retries and also places the total deadline on the
request body. The probe calls `infer` exactly once after successful validation
and client construction. **Integration prerequisite:** base `6fbf817` caps that
client at 20 seconds; the root agent must extend the shared validated cap to
30 seconds before the actual run. The probe saves `InvalidTimeout` and exits
nonzero until then, rather than silently choosing a different deadline/client.

The receipt is reserved with `create_new` before flag, file or pairing validation;
its parent must already exist. Existing evidence is never overwritten. Ordinary
validation/transport/reply failures save a failure receipt and return nonzero.
PNGs are bounded at 16 MiB and stamps at 2 MiB. The typed native stamp's complete
original field set is required; service schemas, missing/extra fields, duplicate
fields, false native pairing and invalid acquisition chronology are rejected.

Episode, render frame, simulation time, capture sequence and acquisition wall
time stay unchanged. The original capture sequence is this request's sequence.
Receipts include the full original native stamp/self-state, sent policy
observation, complete actual returned chunk, input/RGB/canonical-probe hashes,
binary and compiled client/observation/action/frame-guard hashes, expected model
source identity, call/result counts, HTTP elapsed time and original image age at
submission/return. The canonical probe hash is not mislabeled as a private HTTP
wire-body digest. Clients validate the reply's exact profile/revision/stamp/
sequence/horizon/20 ms and every finite action field; no permissive physical
limits are invented. Server weight bytes and raw HTTP-attempt counters are not
exposed by the clients, so the receipt does not claim they were independently
verified. The full saved observation contains RGB bytes; allow space for its
bounded JSON representation.

Cold latency can consume an entire action horizon: static 40×20 ms is 0.8 s;
mobile 50×20 ms is 1 s. The existing mobile CUDA direct-inference measurement
was 1.0406 s, already longer than that horizon. The receipt records horizon
consumption without restamping or scheduling any reply. Saved-frame diagnostic
success does not qualify live scheduling, standing, grasping or either task:
`qualified=false`, `live_admitted=false`, `physics_integrations=0`,
`executor_actions=0`, and `physical_limits_checked=false` always remain explicit.

Pure tests cover original stamp schema rejection, the actual 43→31 joint seam
and unchanged identity, and mandatory offline flags. Native pairing/pose/clock
guards reuse the shared decision diagnostic's tests; neither those fixtures nor
these tests are real model or physics qualification.

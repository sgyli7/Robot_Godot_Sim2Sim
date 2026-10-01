#!/usr/bin/env bash
# Exact frozen Homie standing regression. A failed qualification exits nonzero.
# Usage: ./reproduce_g1_stand.sh /absolute/path/to/new_report.json
set -euo pipefail
if [[ $# != 1 || "$1" != /* ]]; then
    echo 'usage: reproduce_g1_stand.sh /absolute/path/to/new_report.json' >&2
    exit 2
fi
g1_workspace=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
g1_output=$1
mkdir -p -- "$(dirname -- "$g1_output")"
if [[ -e "$g1_output" ]]; then
    echo "refusing to overwrite existing evidence: $g1_output" >&2
    exit 2
fi
export G1_MODEL_DIR=/home/ethan/models/unitree_g1/homie_v2
export G1_DEFINITION_SHA256=571cb2558c137dccafa2d18adda5021f0885e0f10abf6d61edd62f1c6e8f13bd
export G1_ORT=/home/ethan/models/unitree_g1/homie_v2/tools_env/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.30.0
export G1_ORT_SHA256=8a218e1e446880131115c368ec7005969a5022af9a15a26173de648fef0b7789
G1_CODE_COMMIT=$(git -C "$g1_workspace" rev-parse HEAD)
export G1_CODE_COMMIT
export G1_T0_OUTPUT="$g1_output"
# Save provenance even when the intentionally red test exits 101.
{
    git -C "$g1_workspace" status --short
    git -C "$g1_workspace" rev-parse HEAD
    sha256sum "$G1_MODEL_DIR/g1_physics.json" "$G1_MODEL_DIR/stand.onnx" "$G1_MODEL_DIR/walk.onnx" "$G1_ORT"
    sha256sum "$g1_workspace/crates/modules/simulation/src/g1/assembly.rs" "$g1_workspace/crates/modules/simulation/src/g1/runner.rs"
} > "${g1_output}.source.txt"
export CARGO_BUILD_JOBS=2
export CARGO_TARGET_DIR=/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/target
exec flock /tmp/sai-g1-cargo.lock cargo test --manifest-path "$g1_workspace/Cargo.toml" \
    -p simulation_minigame real_homie_stand_qualification --locked --offline -- --ignored --nocapture

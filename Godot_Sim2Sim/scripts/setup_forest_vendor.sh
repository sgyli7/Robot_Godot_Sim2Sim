#!/usr/bin/env bash
set -euo pipefail
FOREST_TOOLS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$FOREST_TOOLS_ROOT/godot/scripts/setup_forest_vendor.sh" "$@"

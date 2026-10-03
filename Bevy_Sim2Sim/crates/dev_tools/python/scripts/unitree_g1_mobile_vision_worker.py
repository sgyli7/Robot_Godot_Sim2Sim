#!/usr/bin/env python3
"""Single-episode CPU worker around the pinned RGB localizer.

Loads dependencies before requesting an image. Input stays actual RGB and the
named self observation; no task/world/contact frame is accepted by this port.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(value):
    print(json.dumps(value, allow_nan=False, separators=(",", ":")), flush=True)


def main():
    started = time.monotonic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vision-script", required=True, type=Path)
    ap.add_argument("--vision-sha256", required=True)
    ap.add_argument("--definition", required=True, type=Path)
    ap.add_argument("--geometry", required=True, type=Path)
    ap.add_argument("--fiducials", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--episode", required=True, type=int)
    args = ap.parse_args()
    if digest(args.vision_script) != args.vision_sha256 or args.episode <= 0:
        raise ValueError("foreign localizer/episode")
    spec = importlib.util.spec_from_file_location("g1_bound_rgb_localizer", args.vision_script)
    vision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vision)
    public_assets = vision.PinnedPublicVisionAssets(args.definition, args.geometry, args.fiducials)
    # Release's CPU geometry dependencies are loaded during initial preparation.
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull
    linprog([1.], bounds=[(0., 1.)], method="highs")
    root = args.output_root.resolve(strict=True)
    emit({"schema": "g1_persistent_marker_worker_v1", "event": "ready",
          "episode_id": args.episode, "pid": os.getpid(),
          "vision_script_sha256": digest(args.vision_script),
          "worker_sha256": digest(__file__), "startup_seconds": time.monotonic()-started})
    previous_frame = previous_sim = requests = 0
    while True:
        line = sys.stdin.buffer.readline(8193)
        if not line:
            break
        if len(line) > 8192 or not line.endswith(b"\n"):
            raise ValueError("request exceeds its single-line byte budget")
        request = json.loads(line)
        if set(request) != {"schema", "directory", "observation", "placement_view_only"}:
            raise ValueError("unknown request fields")
        obs = request["observation"]
        if (request["schema"] != "g1_persistent_marker_request_v1"
                or set(obs) != {"episode_id", "frame_id", "sim_time_ns", "captured_at_unix_ms"}
                or obs["episode_id"] != args.episode or obs["frame_id"] <= previous_frame
                or obs["sim_time_ns"] <= previous_sim or obs["sim_time_ns"] % 20_000_000
                or not obs["captured_at_unix_ms"] or type(request["placement_view_only"]) is not bool
                or requests >= 12):
            raise ValueError("foreign, repeated or unbounded request identity")
        directory = Path(request["directory"]).resolve(strict=True)
        if directory.parent != root:
            raise ValueError("request is outside this episode's output root")
        image, observation = directory/"ego.png", directory/"observation.json"
        if image.is_symlink() or observation.is_symlink():
            raise ValueError("observation files must be owned regular files")
        if json.loads(observation.read_text())["stamp"] != obs:
            raise ValueError("request changed its actual observation")
        previous_frame, previous_sim = obs["frame_id"], obs["sim_time_ns"]
        requests += 1
        began = time.monotonic()
        try:
            result = vision.localize(image, observation, args.definition, args.geometry,
                                     None, False, args.fiducials, request["placement_view_only"],
                                     public_assets=public_assets)
            with (directory/"localization.json").open("x") as output:
                json.dump(result, output, indent=2, allow_nan=False)
                output.write("\n")
            elapsed = time.monotonic()-began
            with (directory/"localization_timing.json").open("x") as output:
                json.dump({"owned_pid": os.getpid(), "observation": obs,
                           "localization_seconds": elapsed, "persistent_cpu_worker": True,
                           "world_or_contact_truth_input": False}, output, indent=2)
                output.write("\n")
            emit({"schema": "g1_persistent_marker_worker_v1", "event": "result",
                  "observation": obs, "localization_seconds": elapsed,
                  "result": result})
        except Exception as error:
            emit({"schema": "g1_persistent_marker_worker_v1", "event": "error",
                  "observation": obs, "error": str(error)})
            break


if __name__ == "__main__":
    main()

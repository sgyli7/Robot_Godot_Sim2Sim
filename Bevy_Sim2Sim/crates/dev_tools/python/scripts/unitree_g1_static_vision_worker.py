#!/usr/bin/env python3
"""Bounded single-episode CPU port for the static RGB/self-sensor localizer.

Dependencies and verified public assets prepare before a camera request. This
distinct static protocol cannot be substituted for a mobile VLA/vision contract.
"""
import argparse
import hashlib
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
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("definition", "fiducials", "output-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--fiducial-sha256", required=True)
    parser.add_argument("--episode", type=int, required=True)
    args = parser.parse_args()
    if args.episode <= 0:
        raise ValueError("invalid static worker episode")
    from unitree_g1_static_vision import PinnedPublicStaticVisionAssets, localize
    assets = PinnedPublicStaticVisionAssets(args.definition, args.fiducials, args.fiducial_sha256)
    root = args.output_root.resolve(strict=True)
    emit({"schema":"g1_persistent_static_marker_worker_v1", "event":"ready",
          "episode_id":args.episode, "pid":os.getpid(), "worker_sha256":digest(__file__),
          "localizer_sha256":digest(Path(localize.__code__.co_filename)),
          "startup_seconds":time.monotonic()-started, "task_qualified":False})
    previous_frame, previous_sim, requests = 0, -1, 0
    while True:
        line = sys.stdin.buffer.readline(8193)
        if not line:
            return 0
        obs = None
        try:
            if len(line) > 8192 or not line.endswith(b"\n"):
                raise ValueError("static request exceeds one-line byte budget")
            request = json.loads(line)
            if set(request) != {"schema", "directory", "observation"}:
                raise ValueError("foreign static request fields")
            obs = request["observation"]
            if (request["schema"] != "g1_persistent_static_marker_request_v1"
                    or set(obs) != {"episode_id", "frame_id", "sim_time_ns", "captured_at_unix_ms"}
                    or any(type(v) is not int for v in obs.values())
                    or obs["episode_id"] != args.episode or obs["frame_id"] <= previous_frame
                    or obs["sim_time_ns"] <= previous_sim or obs["sim_time_ns"] % 20_000_000
                    or obs["captured_at_unix_ms"] <= 0 or requests >= 12):
                raise ValueError("foreign, repeated or unbounded static observation")
            directory = Path(request["directory"]).resolve(strict=True)
            if directory.parent != root:
                raise ValueError("static request outside owned episode directory")
            image, observation = directory/"ego.png", directory/"observation.json"
            if image.is_symlink() or observation.is_symlink():
                raise ValueError("static input must be owned regular files")
            if observation.stat().st_size > 128*1024 or json.loads(observation.read_text())["stamp"] != obs:
                raise ValueError("request changed actual static observation")
            previous_frame, previous_sim = obs["frame_id"], obs["sim_time_ns"]
            requests += 1
            began = time.monotonic()
            result = localize(image, observation, args.definition, args.fiducials,
                              args.fiducial_sha256, public_assets=assets)
            elapsed = time.monotonic()-began
            with (directory/"localization.json").open("x") as output:
                json.dump(result, output, indent=2, allow_nan=False)
                output.write("\n")
            emit({"schema":"g1_persistent_static_marker_worker_v1", "event":"result",
                  "observation":obs, "localization_seconds":elapsed, "result":result,
                  "world_or_contact_truth_input":False, "actuation_proposed":False})
        except Exception as error:
            emit({"schema":"g1_persistent_static_marker_worker_v1", "event":"error",
                  "observation":obs, "error":str(error), "actuation_proposed":False})
            return 1


if __name__ == "__main__":
    raise SystemExit(main())

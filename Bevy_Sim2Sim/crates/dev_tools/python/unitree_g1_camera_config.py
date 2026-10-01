"""Prepare byte-bound native Homie G1 camera diagnostic configuration.

This only checks cached files and writes a new JSON file. It does not download,
start inference, launch rendering or qualify a physical task.
"""

import argparse
import hashlib
import json
from pathlib import Path


ARTIFACTS = {
    "g1_physics.json": "571cb2558c137dccafa2d18adda5021f0885e0f10abf6d61edd62f1c6e8f13bd",
    "g1_visuals.json": "98711070da898c75089b66ab35c799900cacb1e2980c537f9fcc1da8c8e9297d",
    "stand.onnx": "f645da599d4ca3d29ed273c8f4712620bb680d34977469ca3aeabe5bb9631c18",
    "walk.onnx": "7c82255b6905ffcc4468fa7f8ddcf7b70db168cf1042107ccab887cb6a8e5407",
}
ORT_SHA256 = "8a218e1e446880131115c368ec7005969a5022af9a15a26173de648fef0b7789"


def prepare(model_dir: Path, ort: Path, output: Path) -> None:
    model_dir, ort = model_dir.absolute(), ort.absolute()
    for path, expected in [
        *((model_dir / name, digest) for name, digest in ARTIFACTS.items()),
        (ort, ORT_SHA256),
    ]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact SHA256 mismatch: {path}")
    config = {
        "runner": {
            "episode_id": 1,
            "definition": str(model_dir / "g1_physics.json"),
            "definition_sha256": ARTIFACTS["g1_physics.json"],
            "ort_library": str(ort),
            "ort_sha256": ORT_SHA256,
            "stand_model": str(model_dir / "stand.onnx"),
            "walk_model": str(model_dir / "walk.onnx"),
            "root_pose": {"position": [0, 0, 0.78], "rotation_wxyz": [1, 0, 0, 0]},
            "robot_contact_friction": 0.5,
            "floor_contact_friction": 1,
        },
        "visual_path": str(model_dir / "g1_visuals.json"),
        "visual_sha256": ARTIFACTS["g1_visuals.json"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Prepared unqualified Homie camera diagnostic: {output.absolute()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--ort", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.model_dir, args.ort, args.output)


if __name__ == "__main__":
    main()

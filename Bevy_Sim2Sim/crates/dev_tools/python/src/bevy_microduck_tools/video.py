"""Render the real MuJoCo state; preserve the raw frame-to-time mapping."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .serialization import sha256_file, write_json


class SourceVideo:
    def __init__(self, model, path: Path, *, physics_dt: float, root_body: int):
        import mujoco
        binary = shutil.which("ffmpeg")
        if binary is None:
            raise RuntimeError("Real video capture requires ffmpeg")
        self.path = path
        self.dt = physics_dt
        self.root_body = root_body
        self.renderer = mujoco.Renderer(model, height=480, width=640)
        self.camera = mujoco.MjvCamera()
        self.camera.distance = .65
        self.camera.azimuth = 135
        self.camera.elevation = -20
        self.process = subprocess.Popen([binary, "-y", "-loglevel", "error", "-f", "rawvideo",
                                        "-pixel_format", "rgb24", "-video_size", "640x480",
                                        "-framerate", str(1 / physics_dt), "-i", "pipe:0",
                                        "-an", "-vcodec", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(path)],
                                       stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.frames = []

    def capture(self, data, tick: int):
        self.camera.lookat[:] = data.xpos[self.root_body]
        self.renderer.update_scene(data, camera=self.camera)
        self.process.stdin.write(self.renderer.render().tobytes())
        self.frames.append({"frame_index": len(self.frames), "physics_tick": tick, "time_seconds": tick * self.dt})

    def close(self) -> dict:
        self.process.stdin.close()
        errors = self.process.stderr.read().decode()
        code = self.process.wait(timeout=30)
        self.renderer.close()
        if code or not self.frames or not self.path.is_file():
            raise RuntimeError(f"Real source video capture failed: {errors}")
        report = {"path": str(self.path.resolve()), "sha256": sha256_file(self.path),
                  "method": "MuJoCo renderer on actual captured post-integration qpos; no synthetic animation",
                  "fps": 1 / self.dt, "frames": self.frames,
                  "limits": "source renderer; target Rapier rendering and behavior require separate evidence"}
        write_json(self.path.with_suffix(".json"), report)
        return report

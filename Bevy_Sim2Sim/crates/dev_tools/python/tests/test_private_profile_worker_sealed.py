"""The retired profile child must reject before touching any source or output."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from bevy_microduck_tools.authorization import Rejection
from bevy_microduck_tools.profile_discovery import _worker


def _paths(tmp_path):
    return (tmp_path / "absent_binding.json", tmp_path / "worker_result.json",
            tmp_path / "source_output", tmp_path / "phase.jsonl")


def test_direct_private_worker_rejects_before_any_file_read_or_write(tmp_path):
    with pytest.raises(Rejection, match="Private v2 source worker disabled"):
        _worker(*_paths(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_private_worker_module_rejects_before_any_file_read_or_write(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "bevy_microduck_tools.profile_worker",
         *(str(path) for path in _paths(tmp_path))],
        capture_output=True, text=True, timeout=5,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=False)
    assert completed.returncode != 0
    assert "Private v2 source worker disabled" in completed.stderr
    assert list(tmp_path.iterdir()) == []

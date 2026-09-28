"""Passive file/exec observer for a finite source profile.

The trace is evidence of use, not a sandbox or a pre-use policy.  In particular,
paths only hashed after the run have a weaker identity than a JIT blob bound
immediately before ``load_cuda``. Unbound failures leave the profile open.
"""
from __future__ import annotations

import ast
import json
import hashlib
import os
import re
import signal
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .authorization import Rejection
from .serialization import identity, write_json

TRACE_ARGUMENTS = ("-ff", "-yy", "-v", "-ttt", "-s", "256", "-e",
                   "trace=%file,%process,%memory,fstat", "-e", "abbrev=execve,execveat")
_STAMP = re.compile(r"^(\d+\.\d+)\s+(.*)$")
_FD = re.compile(r"(?:=\s*|fstat\()\d+<([^>]+)>")
_STAT = re.compile(r"st_dev=makedev\(0x([0-9a-f]+),\s*0x([0-9a-f]+)\).*?st_ino=(\d+)")
_EXEC = re.compile(r'^(?:execve|execveat)\("((?:\\.|[^"\\])*)"')
_SYSCALL = re.compile(r"^([a-z][a-z0-9_]*)\(")
_QUOTED = re.compile(r'"(?:\\.|[^"\\])*"')
_ARG_FD = re.compile(r"(?:AT_FDCWD|\d+)<(/[^>]+)>")
_RETURN_FD = re.compile(r"\)\s+=\s+(\d+)<(/[^>]+)>")
_FD_FIRST = re.compile(r"^(?:fstat|fstatfs)\((\d+)<(/[^>]+)>")
_FD_ANY = re.compile(r"(\d+)<(/[^>]+)>")
_FAILED = re.compile(r"\)\s+=\s+-1\s+([A-Z][A-Z0-9_]+)")
_SUCCESS = re.compile(r"\)\s+=\s+(?:\d+|0x[0-9a-f]+)(?:<[^>]+>)?(?:\s|$)")
_CHILD_PID = re.compile(r"\)\s+=\s+(\d+)(?:\s|$)")
_MODE = re.compile(r"st_mode=(S_IF[A-Z]+)")
_SIZE = re.compile(r"st_size=(\d+)")
_FILE_OPEN = {"open", "openat", "openat2", "creat"}
_FILE_METADATA = {"stat", "lstat", "fstat", "newfstatat", "statx", "statfs",
                  "fstatfs", "access", "faccessat", "faccessat2", "readlink", "readlinkat"}
_FILE_MUTATION = {"rename", "renameat", "renameat2", "unlink", "unlinkat", "mkdir",
                  "mkdirat", "rmdir", "link", "linkat", "symlink", "symlinkat",
                  "chmod", "fchmod", "fchmodat", "chown", "lchown", "fchown",
                  "fchownat", "truncate", "ftruncate", "utime", "utimes",
                  "utimensat", "mknod", "mknodat", "setxattr", "lsetxattr",
                  "fsetxattr", "removexattr", "lremovexattr", "fremovexattr"}
_TRACE_NONFILE = {"clone", "clone3", "fork", "vfork", "wait4", "waitid",
                  "exit", "exit_group", "munmap", "mprotect", "mremap", "brk",
                  "madvise", "mlock", "munlock", "msync", "mincore", "membarrier"}


class _DeadlineExceeded(Rejection):
    pass


def _remaining(deadline: float | None) -> float:
    if deadline is None:
        return float("inf")
    left = deadline - time.monotonic()
    if left <= 0:
        raise _DeadlineExceeded("Full observer deadline expired before evidence closure")
    return left


def _checked_sha256_file(path: Path, deadline: float | None) -> str:
    digest = hashlib.sha256()
    # O_NONBLOCK makes a path replaced by a FIFO fail immediately, including
    # after the caller's preliminary stat. Never open an arbitrary file before
    # checking the descriptor's actual type.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise Rejection(f"Cannot hash nonregular observed file: {path}")
        while block := source.read(1024 * 1024):
            _remaining(deadline)
            digest.update(block)
    _remaining(deadline)
    return digest.hexdigest()


def _read_regular_text(path: Path, deadline: float | None, *, max_bytes: int = 64 * 1024 * 1024) -> str:
    """Read a child-authored journal without blocking on a replacement FIFO."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as source:
        actual = os.fstat(source.fileno())
        if not stat.S_ISREG(actual.st_mode) or actual.st_size > max_bytes:
            raise Rejection(f"Journal is nonregular or exceeds bounded size: {path}")
        chunks = []
        total = 0
        while block := source.read(1024 * 1024):
            _remaining(deadline)
            total += len(block)
            if total > max_bytes:
                raise Rejection(f"Journal grew beyond bounded size: {path}")
            chunks.append(block)
    _remaining(deadline)
    return b"".join(chunks).decode("utf-8")


def _trace_paths(call: str, *, descriptor: bool = False, count: int = 1) -> list[str]:
    """Resolve strace's quoted path arguments; never guess an unknown cwd."""
    if descriptor:
        match = _ARG_FD.search(call)
        return [match[1] if match else ""]
    values = []
    for match in _QUOTED.finditer(call):
        if call[match.end():].startswith("..."):
            return []
        try:
            path = ast.literal_eval(match[0])
        except (ValueError, SyntaxError):
            return []
        if not isinstance(path, str):
            return []
        if path.startswith("/"):
            values.append(os.path.normpath(path))
            if len(values) == count:
                break
            continue
        base = list(_ARG_FD.finditer(call[:match.start()]))
        if not base:
            return []
        values.append(os.path.normpath(os.path.join(base[-1][1], path)))
        if len(values) == count:
            break
    return values


def _trace_lines(path: Path, deadline: float | None):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(fd, "r", errors="replace") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise Rejection(f"Trace is not a regular file: {path}")
        for line in source:
            _remaining(deadline)
            yield line.rstrip("\n")


def _real_file(path: str, *, deadline: float | None = None) -> dict:
    """Read actual current bytes; do not pretend this was an atomic open-time read."""
    name = Path(path)
    try:
        before = name.stat()
        if not stat.S_ISREG(before.st_mode):
            raise Rejection(f"Observed source is not a regular file: {path}")
        digest = _checked_sha256_file(name, deadline)
        after = name.stat()
    except (OSError, ValueError) as error:
        raise Rejection(f"Observed file unavailable for actual-byte identity: {path}: {error}") from error
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise Rejection(f"Observed file changed during hashing: {path}")
    return {"path": str(name.absolute()), "resolved_path": str(name.resolve()),
            "device_major": os.major(after.st_dev), "device_minor": os.minor(after.st_dev),
            "inode": after.st_ino, "size": after.st_size, "sha256": digest}


def _platform(path: str) -> bool:
    if path.startswith(("/proc/", "/sys/")) or path in ("/proc", "/sys"):
        return True
    if path.startswith("/dev/") or path == "/dev":
        try:
            mode = Path(path).stat().st_mode
        except OSError:
            return False
        return stat.S_ISCHR(mode) or stat.S_ISBLK(mode) or stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)
    return False


def parse_trace(trace_dir: Path, *, command_exitcode: int, declared: dict[str, dict] | None = None,
                generated: dict[str, dict] | None = None, phases: list[dict] | None = None,
                deadline: float | None = None, negative_contract: dict | None = None) -> dict:
    """Parse full trace files, including short-lived children and unloaded libraries.

    The caller must close/stop the process group first.  Hashing is post-use;
    ``fstat`` supplies the actual opened device/inode where emitted by libc.
    """
    declared = declared or {}
    generated = generated or {}
    from .negative_lookup import NegativeMatcher
    matcher = NegativeMatcher(negative_contract) if negative_contract is not None else None
    traces = sorted(trace_dir.glob("trace.[0-9]*"))
    if not traces:
        raise Rejection("Missing syscall trace: observer cannot certify a source profile")
    origins: dict[str, dict] = {}
    executable_events = []
    platform_events = []
    unresolved = []
    phase_events = []
    failed_candidates = []
    failed_candidate_details = []
    negative_lookup_events = []
    successful_metadata = []
    filesystem_mutations = []
    open_selections = []
    opened_descriptors: dict[tuple[int, int], dict] = {}
    syscall_inventory: dict[str, int] = {}
    expected_child_pids: set[int] = set()
    complete = command_exitcode == 0
    if any(not path.name.removeprefix("trace.").isdecimal() for path in traces):
        complete = False
        unresolved.append({"reason": "trace directory contains malformed per-PID trace name"})
    phase_points = sorted((float(item["wall_seconds"]), item["phase"]) for item in (phases or []))

    def phase_for(stamp):
        active = "before_first_science_phase"
        for at, phase in phase_points:
            if at > stamp:
                break
            active = phase
        return active

    def origin_for(path: str, pid: int, phase: str) -> dict:
        event = origins.setdefault(path, {"path": path, "pids": set(), "phases": set(),
                                          "actual_opened_devices": set(), "metadata_devices": set(),
                                          "metadata_modes": set(), "readlink_targets": set(),
                                          "metadata_stats": [],
                                          "event_count": 0, "read_opened": False,
                                          "write_opened": False, "metadata_unbound": False})
        event["pids"].add(pid)
        event["phases"].add(phase)
        event["event_count"] += 1
        return event

    for trace in traces:
        pid_text = trace.name.removeprefix("trace.")
        if not pid_text.isdecimal():
            continue
        pid = int(pid_text)
        initial_trace_sha256 = _checked_sha256_file(trace, deadline)
        saw_line = False
        saw_clean_exit = False
        for line_number, line in enumerate(_trace_lines(trace, deadline), 1):
            saw_line = True
            saw_clean_exit |= line.endswith("+++ exited with 0 +++")
            parsed = _STAMP.match(line)
            if not parsed:
                if line and not line.startswith(("+++", "---")):
                    complete = False
                    unresolved.append({"pid": pid, "reason": "unparsed trace record", "trace": str(trace)})
                continue
            stamp, call = float(parsed[1]), parsed[2]
            phase = phase_for(stamp)
            name_match = _SYSCALL.match(call)
            if name_match is None:
                if call.startswith(("+++", "---")):
                    continue
                complete = False
                unresolved.append({"pid": pid, "phase": phase, "reason": "unparsed or unfinished syscall",
                                   "call": call[:256]})
                continue
            name = name_match[1]
            syscall_inventory[name] = syscall_inventory.get(name, 0) + 1
            if name in {"clone", "clone3", "fork", "vfork"}:
                child = _CHILD_PID.search(call)
                failure = _FAILED.search(call)
                if failure or child is None:
                    complete = False
                    unresolved.append({"pid": pid, "phase": phase, "syscall": name,
                                       "reason": "failed or unknown child creation result",
                                       "call": call[:256]})
                elif int(child[1]) > 0:
                    expected_child_pids.add(int(child[1]))
                continue
            if name in _TRACE_NONFILE:
                continue
            if name == "mmap":
                failure = _FAILED.search(call)
                success = _SUCCESS.search(call)
                descriptors = _ARG_FD.findall(call)
                # A mapped filename may itself contain MAP_ANONYMOUS. An
                # annotated ordinary FD always wins over flag text.
                anonymous = not descriptors and ("MAP_ANONYMOUS" in call or "MAP_ANON" in call)
                if anonymous:
                    if not success:
                        complete = False
                        unresolved.append({"pid": pid, "phase": phase, "syscall": name,
                                           "reason": "failed or unknown anonymous mapping result",
                                           "call": call[:256]})
                    continue
                if not descriptors or not success:
                    complete = False
                    if failure:
                        failed_candidates.append({"pid": pid, "phase": phase,
                                                  "syscall": name,
                                                  "path": descriptors[-1] if descriptors else None,
                                                  "errno": failure[1], "wall_seconds": stamp})
                    unresolved.append({"pid": pid, "phase": phase, "syscall": name,
                                       "reason": "file mapping lacks resolved FD or successful result",
                                       "call": call[:256]})
                    continue
                path = descriptors[-1]
                if " (deleted)" in path or not path.startswith("/"):
                    complete = False
                    unresolved.append({"pid": pid, "phase": phase, "syscall": name,
                                       "reason": "mapped FD target deleted or nonabsolute", "call": call[:256]})
                elif _platform(path):
                    platform_events.append({"pid": pid, "phase": phase, "interface": path,
                                            "syscall": name, "wall_seconds": stamp})
                else:
                    origin_for(path, pid, phase)["read_opened"] = True
                    descriptor = _FD_ANY.findall(call)
                    if not descriptor:
                        complete = False
                        unresolved.append({"pid": pid, "phase": phase, "syscall": name,
                                           "reason": "mapped ordinary file lacks numeric FD"})
                    elif (pid, int(descriptor[-1][0])) not in opened_descriptors:
                        complete = False
                        unresolved.append({"pid": pid, "phase": phase, "syscall": name,
                                           "path": path, "reason": "mapped FD lacks traced open identity"})
                continue
            if name not in _FILE_OPEN | _FILE_METADATA | _FILE_MUTATION | {"execve", "execveat"}:
                complete = False
                unresolved.append({"pid": pid, "phase": phase, "reason": "unknown traced syscall",
                                   "syscall": name, "call": call[:256]})
                continue
            count = 2 if name in {"rename", "renameat", "renameat2", "link", "linkat",
                                  "symlink", "symlinkat"} else 1
            paths = _trace_paths(call, descriptor=name in {"fstat", "fstatfs", "fchmod",
                                                             "fchown", "ftruncate", "fsetxattr",
                                                             "fremovexattr", "fchdir"}, count=count)
            if len(paths) != count or any(not path.startswith("/") for path in paths):
                complete = False
                if _FAILED.search(call):
                    failed_candidate_details.append({"pid": pid, "phase": phase, "trace": str(trace.resolve()),
                        "trace_sha256": initial_trace_sha256, "line_number": line_number,
                        "wall_seconds": stamp, "raw_call": call,
                        "matched_predeclared_rule": False,
                        "reason": "cannot resolve exact pathname or descriptor"})
                unresolved.append({"pid": pid, "phase": phase, "syscall": name,
                                   "reason": "cannot resolve exact pathname or descriptor",
                                   "call": call[:256]})
                continue
            failure = _FAILED.search(call)
            if failure:
                detail = {"pid": pid, "phase": phase, "trace": str(trace.resolve()),
                          "trace_sha256": initial_trace_sha256, "line_number": line_number,
                          "wall_seconds": stamp, "raw_call": call, "syscall": name,
                          "errno": failure[1], "paths_v11_lexical": paths,
                          "matched_predeclared_rule": False}
                for path in paths:
                    failed_candidates.append({"pid": pid, "phase": phase, "syscall": name,
                                              "path": path, "errno": failure[1],
                                              "wall_seconds": stamp})
                if matcher is not None and len(paths) == 1:
                    try:
                        matched = matcher.match(call, phase=phase)
                    except Rejection as error:
                        detail["reason"] = str(error)
                        unresolved.append({"pid": pid, "phase": phase, "syscall": name,
                                           "paths": paths, "errno": failure[1],
                                           "reason": f"negative lookup rule rejected: {error}"})
                        complete = False
                    else:
                        detail["matched_predeclared_rule"] = True
                        detail["rule_id"] = matched["rule_id"]
                        negative_lookup_events.append({**matched, "pid": pid, "phase": phase,
                            "wall_seconds": stamp, "trace": str(trace.resolve()),
                            "trace_sha256": initial_trace_sha256, "line_number": line_number})
                    failed_candidate_details.append(detail)
                    continue
                # Without a sealed finite rule this remains diagnostic only.
                detail["reason"] = "no finite negative lookup contract"
                failed_candidate_details.append(detail)
                complete = False
                unresolved.append({"pid": pid, "phase": phase, "syscall": name,
                                   "reason": "unbound failed pathname candidate",
                                   "paths": paths, "errno": failure[1]})
                continue
            if not _SUCCESS.search(call):
                complete = False
                unresolved.append({"pid": pid, "phase": phase, "syscall": name,
                                   "reason": "unknown syscall result", "call": call[:256]})
                continue
            if name in {"execve", "execveat"}:
                executable_events.append({"pid": pid, "phase": phase, "path": paths[0],
                                          "wall_seconds": stamp})
                continue
            if name in _FILE_MUTATION:
                filesystem_mutations.append({"pid": pid, "phase": phase, "syscall": name,
                                              "paths": paths, "wall_seconds": stamp})
                continue
            path = paths[0]
            if _platform(path) and name not in _FILE_OPEN:
                if (path.startswith("/proc/") and name in _FILE_METADATA) or "/fd/" in path:
                    complete = False
                    unresolved.append({"pid": pid, "phase": phase, "syscall": name,
                                       "path": path, "reason": "platform metadata or fd alias may select ordinary file"})
                platform_events.append({"pid": pid, "phase": phase, "interface": path,
                                        "syscall": name, "wall_seconds": stamp})
                continue
            if name in _FILE_OPEN:
                returned = _RETURN_FD.search(call)
                if returned is None:
                    complete = False
                    unresolved.append({"pid": pid, "phase": phase, "syscall": name,
                                       "path": path, "reason": "successful open lacks resolved FD target"})
                    continue
                used_path = returned[2].removesuffix(" (deleted)")
                if not used_path.startswith("/") or " (deleted)" in returned[2]:
                    complete = False
                    unresolved.append({"pid": pid, "phase": phase, "syscall": name,
                                       "path": path, "reason": "opened FD target missing or deleted"})
                    continue
                if _platform(path) and not _platform(used_path):
                    complete = False
                    unresolved.append({"pid": pid, "phase": phase, "syscall": name,
                                       "path": path, "opened_path": used_path,
                                       "reason": "platform fd alias selected ordinary file"})
                    continue
                if _platform(used_path):
                    platform_events.append({"pid": pid, "phase": phase, "interface": used_path,
                                            "syscall": name, "wall_seconds": stamp})
                    continue
                open_selections.append({"pid": pid, "phase": phase, "syscall": name,
                                        "candidate": path, "opened_path": used_path,
                                        "fd": int(returned[1]),
                                        "wall_seconds": stamp})
                selected_open = open_selections[-1]
                opened_descriptors[(pid, int(returned[1]))] = selected_open
                event = origin_for(used_path, pid, phase)
                quoted_arguments = list(_QUOTED.finditer(call))
                flag_tail = call[quoted_arguments[-1].end():call.rfind(") =")] if quoted_arguments else ""
                write_only = name == "creat" or ("O_WRONLY" in flag_tail and "O_RDWR" not in flag_tail)
                selected_open["read_opened"] = not write_only
                event["write_opened"] = event["write_opened"] or write_only or "O_RDWR" in flag_tail
                event["read_opened"] = event["read_opened"] or not write_only
                phase_events.append((stamp, phase))
                continue
            event = origin_for(path, pid, phase)
            stat_result = _STAT.search(call)
            if stat_result:
                current_identity = (int(stat_result[1], 16), int(stat_result[2], 16),
                                    int(stat_result[3]))
                if name in {"fstat", "fstatfs"}:
                    event["actual_opened_devices"].add(current_identity)
                    first_fd = _FD_FIRST.search(call)
                    selected_open = opened_descriptors.get((pid, int(first_fd[1]))) if first_fd else None
                    if selected_open is None or selected_open["opened_path"] != path:
                        complete = False
                        unresolved.append({"pid": pid, "phase": phase, "syscall": name,
                                           "path": path, "reason": "fstat FD lacks matching traced open"})
                    else:
                        selected_open["observed_identity"] = list(current_identity)
                else:
                    event["metadata_devices"].add(current_identity)
                mode = _MODE.search(call)
                if mode:
                    event["metadata_modes"].add(mode[1])
                size = _SIZE.search(call)
                if size:
                    event["metadata_stats"].append({"mode": mode[1] if mode else None,
                                                     "size": int(size[1]),
                                                     "identity": current_identity})
                else:
                    complete = False
                    unresolved.append({"pid": pid, "path": path,
                                       "reason": "successful stat lacks observed size"})
            elif name in {"readlink", "readlinkat"}:
                quoted = _QUOTED.findall(call)
                if len(quoted) < 2 or call.split(quoted[1], 1)[1].startswith("..."):
                    complete = False
                    unresolved.append({"pid": pid, "path": path,
                                       "reason": "successful readlink target truncated or absent"})
                else:
                    try:
                        target = ast.literal_eval(quoted[1])
                    except (ValueError, SyntaxError):
                        complete = False
                        unresolved.append({"pid": pid, "path": path,
                                           "reason": "cannot decode successful readlink target"})
                    else:
                        event["readlink_targets"].add(target)
            else:
                event["metadata_unbound"] = True
                complete = False
                unresolved.append({"pid": pid, "path": path, "syscall": name,
                                   "reason": "successful metadata lacks observed device/inode or target"})
            successful_metadata.append({"pid": pid, "phase": phase, "syscall": name,
                                        "path": path, "wall_seconds": stamp,
                                        "observed_identity": list(current_identity) if stat_result else None})
            phase_events.append((stamp, phase))
        if not saw_line or not saw_clean_exit:
            complete = False
            unresolved.append({"trace": str(trace), "reason": "missing clean process exit or killed process"})
        if _checked_sha256_file(trace, deadline) != initial_trace_sha256:
            complete = False
            unresolved.append({"trace": str(trace), "reason": "trace bytes changed during negative lookup parsing"})

    if matcher is not None:
        try:
            matcher.verify_after()
        except Rejection as error:
            complete = False
            unresolved.append({"reason": f"negative lookup witness changed after trace: {error}"})

    traced_pids = {int(path.name.removeprefix("trace.")) for path in traces}
    for child_pid in sorted(expected_child_pids - traced_pids):
        complete = False
        unresolved.append({"pid": child_pid, "reason": "successful child creation lacks child trace"})

    actual = []
    outputs = []
    directories = []
    unknown = []
    for selection in open_selections:
        if selection["read_opened"] and "observed_identity" not in selection:
            complete = False
            unresolved.append({"pid": selection["pid"], "path": selection["opened_path"],
                               "fd": selection["fd"],
                               "reason": "read-open FD lacks its own observed device/inode"})
        try:
            selected = str(Path(selection["candidate"]).resolve(strict=True))
            opened = str(Path(selection["opened_path"]).resolve(strict=True))
        except OSError as error:
            complete = False
            unresolved.append({"path": selection["candidate"],
                               "reason": f"successful open candidate unavailable after use: {error}"})
            continue
        if selected != opened:
            complete = False
            unresolved.append({"path": selection["candidate"], "opened_path": selection["opened_path"],
                               "reason": "candidate path no longer resolves to opened FD target"})
    for event in executable_events:
        _remaining(deadline)
        path = event["path"]
        if not path.startswith("/") or _platform(path):
            unresolved.append({"path": path, "reason": "relative or platform executable origin"})
            complete = False
            continue
        try:
            record = _real_file(path, deadline=deadline)
        except Rejection as error:
            unresolved.append({"path": path, "reason": str(error)})
            complete = False
            continue
        expected = declared.get(path) or declared.get(record["resolved_path"])
        event["current_file"] = record
        event["identity_timing"] = "post_use_current_bytes_exec_path_only"
        event["classification"] = "declared_executable" if expected is not None else "unknown_executable"
        if expected is None:
            unknown.append(path)
        elif expected["sha256"] != record["sha256"]:
            unresolved.append({"path": path, "reason": "declared executable bytes changed"})
            complete = False
    for path, event in sorted(origins.items()):
        _remaining(deadline)
        if " (deleted)" in path:
            complete = False
            unresolved.append({"path": path, "reason": "used file was deleted before identity check"})
            continue
        metadata = sorted(event["metadata_devices"])
        if Path(path).is_dir():
            directory_stat = Path(path).stat()
            current_tuple = (os.major(directory_stat.st_dev), os.minor(directory_stat.st_dev), directory_stat.st_ino)
            if any(item != current_tuple for item in event["metadata_devices"] | event["actual_opened_devices"]):
                complete = False
                unresolved.append({"path": path, "reason": "successful directory metadata changed"})
            if any(item["size"] != directory_stat.st_size or item["mode"] != "S_IFDIR"
                   for item in event["metadata_stats"]):
                complete = False
                unresolved.append({"path": path, "reason": "successful directory metadata size/type changed"})
            if event["metadata_unbound"] and not metadata and not event["actual_opened_devices"]:
                complete = False
                unresolved.append({"path": path, "reason": "successful directory lookup lacks observed inode"})
            directories.append({"path": path, "resolved_path": str(Path(path).resolve()),
                                "device_major": os.major(directory_stat.st_dev), "device_minor": os.minor(directory_stat.st_dev),
                                "inode": directory_stat.st_ino, "identity_timing": "post_use_directory_metadata_only",
                                "observed_metadata_devices": [list(item) for item in metadata],
                                "pids": sorted(event["pids"]), "phases": sorted(event["phases"])})
            if path not in declared:
                unknown.append(path)
            continue
        if event["readlink_targets"]:
            try:
                current_target = os.readlink(path)
            except OSError as error:
                complete = False
                unresolved.append({"path": path, "reason": f"successful readlink target unavailable: {error}"})
            else:
                if event["readlink_targets"] != {current_target}:
                    complete = False
                    unresolved.append({"path": path, "reason": "successful readlink target changed"})
        if event["metadata_unbound"] and not metadata and not event["actual_opened_devices"] and not event["readlink_targets"]:
            complete = False
            unresolved.append({"path": path, "reason": "successful metadata lookup lacks observed inode or target"})
        if metadata and Path(path).is_symlink():
            link_stat = Path(path).lstat()
            link_tuple = (os.major(link_stat.st_dev), os.minor(link_stat.st_dev), link_stat.st_ino)
        else:
            link_tuple = None
        if event["write_opened"] and not event["read_opened"]:
            try:
                current = _real_file(path, deadline=deadline)
            except Rejection:
                outputs.append({"path": path, "status": "ephemeral_write_only_output", "pids": sorted(event["pids"])})
                if metadata or event["actual_opened_devices"]:
                    complete = False
                    unresolved.append({"path": path, "reason": "successful metadata for vanished output cannot be rechecked"})
            else:
                outputs.append({**current, "status": "write_only_output", "pids": sorted(event["pids"])})
                current_tuple = (current["device_major"], current["device_minor"], current["inode"])
                if any(item != current_tuple for item in event["metadata_devices"] | event["actual_opened_devices"]):
                    complete = False
                    unresolved.append({"path": path, "reason": "write-only output inode differs from observed success"})
            continue
        try:
            current = _real_file(path, deadline=deadline)
        except Rejection as error:
            complete = False
            unresolved.append({"path": path, "reason": str(error)})
            continue
        opened = sorted(event["actual_opened_devices"])
        current_tuple = (current["device_major"], current["device_minor"], current["inode"])
        for item in event["metadata_stats"]:
            observed = Path(path).lstat() if item["mode"] == "S_IFLNK" else Path(path).stat()
            actual_mode = "S_IFREG" if stat.S_ISREG(observed.st_mode) else (
                "S_IFLNK" if stat.S_ISLNK(observed.st_mode) else (
                "S_IFDIR" if stat.S_ISDIR(observed.st_mode) else None))
            if item["size"] != observed.st_size or item["mode"] != actual_mode:
                complete = False
                unresolved.append({"path": path, "reason": "successful metadata size/type changed"})
        if any(item != current_tuple and item != link_tuple for item in metadata):
            complete = False
            unresolved.append({"path": path, "reason": "successful metadata device/inode differs from hashed file"})
        if opened and any(item != current_tuple for item in opened):
            complete = False
            unresolved.append({"path": path, "reason": "actual open device/inode differs from hashed file"})
        if event["read_opened"] and not opened:
            complete = False
            unresolved.append({"path": path, "reason": "actual opened device/inode absent; post-use stat is insufficient"})
        expected = declared.get(path) or declared.get(current["resolved_path"])
        produced = generated.get(path) or generated.get(current["resolved_path"])
        classification = "declared_input" if expected is not None else ("generated_input" if produced else "unknown_input")
        if expected is not None and any(current.get(key) != expected.get(key) for key in ("sha256", "size")):
            complete = False
            unresolved.append({"path": path, "reason": "declared bytes changed"})
        if produced is not None and any(current[key] != produced.get(key) for key in current):
            complete = False
            unresolved.append({"path": path, "reason": "generated blob changed"})
        if classification == "unknown_input":
            unknown.append(path)
        actual.append({**current, "classification": classification,
                       "identity_timing": "post_use_current_bytes_with_observed_open_inode" if opened else
                                          "post_use_current_bytes_without_observed_open_inode",
                       "actual_opened_devices": [list(item) for item in opened],
                       "observed_metadata_devices": [list(item) for item in metadata],
                       "readlink_targets": sorted(event["readlink_targets"]),
                       "pids": sorted(event["pids"]), "phases": sorted(event["phases"]),
                       "event_count": event["event_count"]})
    return {"schema": "microduck_actual_file_use_v2", "trace_files":
            [{"path": str(path.resolve()), "sha256": _checked_sha256_file(path, deadline)} for path in traces],
            "trace_complete": complete, "posthoc_actual_use": actual,
            "unknown_inputs": sorted(set(unknown)), "unresolved": unresolved,
            "write_only_outputs": outputs, "directory_lookups": directories,
            "platform_interfaces": platform_events, "exec_events": executable_events,
            "failed_candidates": failed_candidates, "successful_metadata": successful_metadata,
            "failed_candidate_details": failed_candidate_details,
            "negative_lookup_contract_id": matcher.contract["contract_id"] if matcher else None,
            "negative_lookup_events": negative_lookup_events,
            "filesystem_mutations": filesystem_mutations, "open_selections": open_selections,
            "syscall_inventory": syscall_inventory,
            "phase_event_count": len(phase_events),
            "qualification": "provisional posthoc observation; no pre-use blocking or learning capability"}


def _terminate_traced_tree(trace_dir: Path, expected_tracer_pid: int) -> dict:
    """Kill current traced PIDs by pidfd, including descendants that called setsid.

    A numeric PID is never signalled unless /proc still identifies its tracer as
    strace.  pidfd prevents reuse between that check and SIGKILL.
    """
    attempted = []
    uncertain = []
    for trace in sorted(trace_dir.glob("trace.[0-9]*")):
        pid = int(trace.name.removeprefix("trace."))
        try:
            status = Path(f"/proc/{pid}/status").read_text()
        except FileNotFoundError:
            continue
        match = re.search(r"^TracerPid:\s*(\d+)$", status, re.MULTILINE)
        if match is None or int(match[1]) == 0:
            continue
        tracer_pid = int(match[1])
        try:
            if tracer_pid != expected_tracer_pid:
                uncertain.append({"pid": pid, "reason": "tracee belongs to a different tracer PID"})
                continue
            tracer_exe = os.readlink(f"/proc/{tracer_pid}/exe")
            if not tracer_exe.endswith("/strace"):
                uncertain.append({"pid": pid, "reason": "tracee has unexpected tracer"})
                continue
            handle = os.pidfd_open(pid)
            try:
                # PIDFD identity is stable if the numeric PID is recycled.
                signal.pidfd_send_signal(handle, signal.SIGKILL)
            finally:
                os.close(handle)
            attempted.append(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, AttributeError) as error:
            uncertain.append({"pid": pid, "reason": f"cannot signal exact tracee: {error}"})
    return {"pidfd_killed_tracees": attempted, "uncertain_tracees": uncertain}


def _incomplete_observation(reason: str) -> dict:
    return {"schema": "microduck_actual_file_use_v2", "trace_files": [],
            "trace_complete": False, "posthoc_actual_use": [],
            "unknown_inputs": [], "unresolved": [{"reason": reason}],
            "write_only_outputs": [], "directory_lookups": [],
            "platform_interfaces": [], "exec_events": [],
            "failed_candidates": [], "failed_candidate_details": [], "successful_metadata": [],
            "negative_lookup_contract_id": None, "negative_lookup_events": [],
            "filesystem_mutations": [], "open_selections": [], "syscall_inventory": {},
            "phase_event_count": 0,
            "qualification": "incomplete trace; diagnostic only"}


def _postprocess_observation(trace: Path, *, process_returncode: int, status: str,
                             declared: dict[str, dict] | None, generated: dict[str, dict] | None,
                             phases: list[dict] | None, phase_path: Path | None,
                             generated_path: Path | None, teardown: dict,
                             deadline: float) -> tuple[dict, dict | None]:
    _remaining(deadline)
    if phase_path is not None:
        if phases is not None:
            raise Rejection("Observer phases must come from one actual journal source")
        try:
            phases = [json.loads(line) for line in _read_regular_text(phase_path, deadline).splitlines()]
            _remaining(deadline)
        except (OSError, ValueError) as error:
            phases = []
            phase_error = f"Missing or invalid actual source phase journal: {error}"
        else:
            phase_error = None if phases else "Actual source phase journal is empty"
    else:
        phase_error = None
    producer_meta = None
    if generated_path is not None:
        if generated is not None:
            raise Rejection("Generated inputs must come from one actual child journal")
        try:
            producer_journal = json.loads(_read_regular_text(generated_path, deadline))
            _remaining(deadline)
            if (set(producer_journal) != {"schema", "generated", "loaded", "scope"} or
                    producer_journal["schema"] != "microduck_profile_producer_journal_v1" or
                    not isinstance(producer_journal["generated"], dict) or
                    not isinstance(producer_journal["loaded"], list)):
                raise ValueError("invalid producer journal schema")
            generated = producer_journal["generated"]
            for loaded in producer_journal["loaded"]:
                _remaining(deadline)
                path = loaded["path"]
                record = generated[path]
                current = _real_file(path, deadline=deadline)
                if (record["recipe_identity"] != loaded["recipe_identity"] or
                        any(current[key] != loaded[key] for key in
                            ("resolved_path", "device_major", "device_minor", "inode", "size", "sha256"))):
                    raise ValueError(f"actual loaded JIT blob changed: {path}")
            generated_error = None
        except (OSError, ValueError, KeyError, TypeError, Rejection) as error:
            if isinstance(error, _DeadlineExceeded):
                raise
            generated = {}
            generated_error = f"Missing, incomplete or changed actual producer journal: {error}"
        producer_meta = {"path": str(generated_path.resolve()),
                         "sha256": _checked_sha256_file(generated_path, deadline)
                         if generated_path.is_file() else None}
    else:
        generated_error = None
    parsed = parse_trace(trace, command_exitcode=process_returncode if status == "completed" else -1,
                         declared=declared, generated=generated, phases=phases, deadline=deadline)
    for reason in teardown["uncertain_tracees"]:
        parsed["trace_complete"] = False
        parsed["unresolved"].append(reason)
    for reason in (phase_error, generated_error):
        if reason is not None:
            parsed["trace_complete"] = False
            parsed["unresolved"].append({"reason": reason})
    _remaining(deadline)
    return parsed, producer_meta


def observe_command(command: list[str], output: Path, *, wall_seconds: float,
                    env: dict[str, str] | None = None, declared: dict[str, dict] | None = None,
                    generated: dict[str, dict] | None = None, phases: list[dict] | None = None,
                    phase_path: Path | None = None, generated_path: Path | None = None,
                    deadline: float | None = None) -> dict:
    """Observe child and postprocessing under the caller's one monotonic deadline."""
    if (not command or not all(isinstance(item, str) for item in command) or
            not 0 < wall_seconds <= 180 or not Path("/usr/bin/strace").is_file()):
        raise Rejection("Invalid or unavailable bounded syscall observer")
    if output.exists():
        raise Rejection("Observer output directory must be unique")
    output.mkdir(parents=True)
    trace = output / "trace"
    trace.mkdir()
    stdout_path = output / "stdout.txt"
    stderr_path = output / "stderr.txt"
    started = time.monotonic()
    deadline = started + wall_seconds if deadline is None else min(deadline, started + wall_seconds)
    status = "failed"
    teardown = {"pidfd_killed_tracees": [], "uncertain_tracees": []}
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        _remaining(deadline)
        process = subprocess.Popen(["/usr/bin/strace", *TRACE_ARGUMENTS, "-o", str(trace / "trace"), *command],
                                   stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        try:
            process.wait(timeout=_remaining(deadline))
            status = "completed" if process.returncode == 0 else "failed"
        except (subprocess.TimeoutExpired, _DeadlineExceeded):
            status = "budget_exhausted"
        finally:
            if process.poll() is None:
                teardown = _terminate_traced_tree(trace, process.pid)
            # strace is group leader; tracees and short-lived render/exec children
            # normally inherit the group. pidfd covers traced setsid children.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=max(.01, min(5, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                status = "budget_exhausted"
                teardown["uncertain_tracees"].append({"pid": process.pid,
                    "reason": "strace process did not exit within bounded teardown"})
    producer_meta = None
    try:
        if status == "budget_exhausted":
            raise _DeadlineExceeded("Source child consumed the full observer budget")
        parsed, producer_meta = _postprocess_observation(
            trace, process_returncode=process.returncode, status=status, declared=declared,
            generated=generated, phases=phases, phase_path=phase_path,
            generated_path=generated_path, teardown=teardown, deadline=deadline)
    except _DeadlineExceeded as error:
        status = "budget_exhausted"
        parsed = _incomplete_observation(str(error))
    except (OSError, ValueError, Rejection) as error:
        status = "failed"
        parsed = _incomplete_observation(f"Observer postprocess failed closed: {error}")
    if time.monotonic() >= deadline:
        status = "budget_exhausted"
        parsed["trace_complete"] = False
        parsed["unresolved"].append({"reason": "Full observer deadline reached before report write"})
    try:
        stdout_hash = _checked_sha256_file(stdout_path, deadline) if status != "budget_exhausted" else None
        stderr_hash = _checked_sha256_file(stderr_path, deadline) if status != "budget_exhausted" else None
    except _DeadlineExceeded:
        status = "budget_exhausted"
        parsed = _incomplete_observation("Full observer deadline reached while hashing captured output")
        stdout_hash = None
        stderr_hash = None
    report = {"schema": "microduck_observer_run_v2", "command": command,
              "status": status, "exitcode": process.returncode,
              "teardown": teardown,
              "producer_journal": producer_meta,
              "elapsed_seconds": time.monotonic() - started,
              "stdout_sha256": stdout_hash, "stderr_sha256": stderr_hash,
              **parsed}
    write_json(output / "observer_report.json", report)
    if time.monotonic() >= deadline:
        report["status"] = "budget_exhausted"
        report["trace_complete"] = False
        report["unresolved"].append({"reason": "Full observer deadline crossed while sealing report"})
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "observer_report.json", report)
    return report


class ProducerJournal:
    """Bind actual generated source/blobs before reuse. No producer is inferred."""

    def __init__(self):
        self.produced: dict[str, dict] = {}
        self.loaded: list[dict] = []

    def record_generated(self, path: Path, *, producer: str, recipe: dict) -> dict:
        if not isinstance(producer, str) or not producer or not isinstance(recipe, dict) or not recipe:
            raise Rejection("Generated blob requires a named, nonempty producer recipe")
        if any(type(key) is not str for key in recipe):
            raise Rejection("Producer recipe keys must be strings")
        try:
            frozen_recipe = json.loads(json.dumps(recipe, sort_keys=True, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise Rejection(f"Producer recipe is not complete stable JSON: {error}") from error
        record = {**_real_file(str(path)), "producer": producer,
                  "recipe": frozen_recipe, "recipe_identity": identity(frozen_recipe)}
        self.produced[str(path.resolve())] = record
        return record

    def before_load(self, path: Path, *, expected_producer: str | None = None) -> dict:
        current = _real_file(str(path))
        record = self.produced.get(str(path.resolve()))
        if (record is None or any(current[key] != record[key] for key in current) or
                identity(record["recipe"]) != record["recipe_identity"] or
                (expected_producer is not None and record["producer"] != expected_producer)):
            raise Rejection("JIT input missing or changed before load; no recipe can be inferred")
        event = {**current,
                 "producer": record["producer"], "recipe": record["recipe"],
                 "recipe_identity": record["recipe_identity"],
                 "load_phase": "before_native_loader"}
        self.loaded.append(event)
        return event

    def transfer(self, source: Path, destination: Path, *, operation: str) -> dict | None:
        """Track official Warp temp→cache migration without inferring from basename."""
        previous = self.produced.get(str(source.resolve()))
        if previous is None:
            return None
        current = _real_file(str(destination))
        if current["sha256"] != previous["sha256"] or current["size"] != previous["size"]:
            raise Rejection("Warp migrated binary differs from its recorded generated bytes")
        recipe = {"operation": operation, "source_path": previous["path"],
                  "source_sha256": previous["sha256"], "source_device_major": previous["device_major"],
                  "source_device_minor": previous["device_minor"], "source_inode": previous["inode"],
                  "source_recipe_identity": previous["recipe_identity"],
                  "origin_producer": previous["producer"]}
        return self.record_generated(destination, producer="warp.cache_migration", recipe=recipe)


class PhaseJournal:
    """Flush actual phase, loaded Python origins and mapped native files."""

    def __init__(self, path: Path):
        if path.exists():
            raise Rejection("Use a unique phase journal path")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.events: list[dict] = []

    def mark(self, phase: str, **details) -> dict:
        if not isinstance(phase, str) or not phase or type(details) is not dict:
            raise Rejection("Invalid source lifecycle phase")
        import sys
        modules = sorted({str(Path(module.__file__).absolute()) for module in sys.modules.values()
                          if getattr(module, "__file__", None) and not str(module.__file__).endswith((".pyc", ".pyo"))})
        from .source_runtime_identity import _native_maps
        event = {"phase": phase, "wall_seconds": time.time(), "monotonic_seconds": time.monotonic(),
                 "pid": os.getpid(), "module_origins": modules, "native_maps": _native_maps(),
                 "details": details}
        self.events.append(event)
        with self.path.open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return event


@contextmanager
def warp_observation(journal: PhaseJournal, producer: ProducerJournal):
    """Wrap official Warp load/build functions once, preserving calls and options.

    New blobs get a provisional recipe. Pre-existing cache inputs are labelled
    adopted/unproven; this observer never turns them into an accepted producer.
    """
    import warp._src.build as build
    import warp._src.context as context
    original_build = build.build_cuda
    original_load = build.load_cuda
    original_safe_rename = build.safe_rename
    original_replace = os.replace
    original_module_load = context.Module.load
    current_recipe = []

    def compile_cuda(*args, **kwargs):
        input_path = Path(args[0] if args else kwargs["cu_path"])
        output_path = Path(args[2] if len(args) > 2 else kwargs["output_path"])
        def link_inputs(name):
            values = kwargs.get(name) or ()
            return [{"sha256": hashlib.sha256(bytes(value)).hexdigest(), "size": len(value),
                     "kind": name} for value in values]
        import warp
        recipe = {"input_cu": _real_file(str(input_path)),
                  "keyword_options": {key: str(value) for key, value in sorted(kwargs.items())
                                      if key not in {"cu_path", "ltoirs", "fatbins"}},
                  "positional_options": [str(value) for value in args[1:]],
                  "linked_ltoir": link_inputs("ltoirs"), "linked_fatbins": link_inputs("fatbins"),
                  "warp_native_include_root": str(Path(warp.__file__).parent / "native"),
                  "llvm_cuda": bool(warp.config.llvm_cuda),
                  "precompiled_headers": bool(warp.config.use_precompiled_headers),
                  "kernel_cache_dir": str(warp.config.kernel_cache_dir),
                  "warp_module": dict(current_recipe[-1]) if current_recipe else None,
                  "external_compiler_closure_pending_actual_trace": True}
        result = original_build(*args, **kwargs)
        if output_path.exists():
            producer.record_generated(output_path, producer="warp.build_cuda", recipe=recipe)
            journal.mark("warp_build_cuda", output=str(output_path.resolve()), recipe=recipe)
        return result

    def load_cuda(path, device):
        name = Path(path)
        current = _real_file(str(name))
        previous = producer.produced.get(str(name.resolve()))
        if previous is not None:
            producer.before_load(name, expected_producer=previous["producer"])
            origin = "generated_or_bound_seed"
        else:
            origin = "adopted_preexisting_unproven"
        journal.mark("warp_load_cuda_before", blob=current, origin=origin,
                     warp_module=current_recipe[-1] if current_recipe else None)
        return original_load(path, device)

    def safe_rename(source, destination):
        source_root = Path(source)
        output_root = Path(destination)
        before = [(Path(path), record) for path, record in producer.produced.items()
                  if Path(path).is_relative_to(source_root)]
        result = original_safe_rename(source, destination)
        for path, _ in before:
            target = output_root / path.relative_to(source_root)
            if target.is_file():
                migrated = producer.transfer(path, target, operation="warp.safe_rename_directory")
                if migrated is not None:
                    journal.mark("warp_cache_migration", output=migrated)
        return result

    def replace(source, destination):
        previous = producer.produced.get(str(Path(source).resolve()))
        result = original_replace(source, destination)
        if previous is not None and Path(destination).is_file():
            migrated = producer.transfer(Path(source), Path(destination), operation="warp.os_replace_file")
            if migrated is not None:
                journal.mark("warp_cache_migration", output=migrated)
        return result

    def module_load(module, *args, **kwargs):
        try:
            options = {key: str(value) for key, value in sorted(module.options.items())}
        except (AttributeError, TypeError, ValueError) as error:
            raise Rejection(f"Cannot bind actual Warp module recipe: {error}") from error
        recipe = {"name": module.name, "module_hash": "pending_original_load",
                  "hash_source": "actual Module.load return; observer does not call hash generator first", "options": options,
                  "device_argument": str(args[0]) if args else str(kwargs.get("device"))}
        current_recipe.append(recipe)
        journal.mark("warp_module_load_begin", recipe=recipe)
        try:
            result = original_module_load(module, *args, **kwargs)
            actual_hash = getattr(result, "module_hash", None)
            if isinstance(actual_hash, bytes):
                recipe["module_hash"] = actual_hash.hex()
            journal.mark("warp_module_load_end", recipe=recipe, return_type=type(result).__name__)
            return result
        finally:
            current_recipe.pop()

    build.build_cuda = compile_cuda
    build.load_cuda = load_cuda
    build.safe_rename = safe_rename
    os.replace = replace
    context.Module.load = module_load
    try:
        yield
    finally:
        context.Module.load = original_module_load
        build.load_cuda = original_load
        build.build_cuda = original_build
        build.safe_rename = original_safe_rename
        os.replace = original_replace

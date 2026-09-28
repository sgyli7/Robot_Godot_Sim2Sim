"""Finite, pre-sealed witnesses for expected read-only pathname failures.

This is a CPU evidence parser, not a syscall sandbox.  In particular, equal
before/after directory snapshots do not exclude an intervening mutation.
"""
from __future__ import annotations

import ast
import hashlib
import os
import re
import stat
from pathlib import Path

from .authorization import Rejection
from .serialization import identity

SPEC_SCHEMA = "microduck_negative_lookup_spec_v1"
CONTRACT_SCHEMA = "microduck_negative_lookup_contract_v1"
MAX_RULES = 4096
MAX_EVENTS = 16384
MAX_DIRECTORY_ENTRIES = 100000
_QUOTED = re.compile(r'"(?:\\.|[^"\\])*"')
_CALL = re.compile(r"^([a-z][a-z0-9_]*)\(")
_RESULT = re.compile(r"\)\s+=\s+-1\s+([A-Z][A-Z0-9_]+)(?:\s+\([^)]*\))?$")
_DIRFD = re.compile(r"AT_FDCWD(?:<(/[^>]*)>)?,\s*")
_ADDRESS = r"0x[0-9a-f]+"
_QUALIFIERS = {
    "faccessat": {"R_OK": "ENOENT"},
    "openat": {"O_RDONLY": "ENOENT", "O_RDONLY|O_CLOEXEC": "ENOENT"},
    "newfstatat": {"0": "ENOENT"},
    "readlinkat": {"1023": "EINVAL", "4096": "EINVAL"},
}


def _exact_path(path: str) -> str:
    if (not isinstance(path, str) or not path.startswith("/") or path == "/" or
            "\x00" in path or "//" in path or len(os.fsencode(path)) > 4096 or
            any(ord(char) > 127 for char in path) or
            any(char in path for char in "*?[]") or
            path in ("/proc", "/sys", "/dev") or
            path.startswith(("/proc/", "/sys/", "/dev/"))):
        raise Rejection("Negative lookup needs one exact non-platform absolute ASCII pathname")
    return path


def _node(path: str) -> dict:
    value = os.lstat(path)
    return {"path": path, "device": value.st_dev, "inode": value.st_ino,
            "mode": value.st_mode, "mtime_ns": value.st_mtime_ns,
            "ctime_ns": value.st_ctime_ns, "nlink": value.st_nlink}


def _directory(path: str) -> dict:
    before = _node(path)
    if not stat.S_ISDIR(before["mode"]):
        raise Rejection(f"Negative lookup ancestor is not a directory: {path}")
    names = [os.fsencode(item) for item in os.listdir(path)]
    if len(names) > MAX_DIRECTORY_ENTRIES:
        raise Rejection(f"Negative lookup parent directory exceeds bounded inventory: {path}")
    names.sort()
    digest = hashlib.sha256()
    for name in names:
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
    after = _node(path)
    if before != after:
        raise Rejection(f"Negative lookup directory changed during witness: {path}")
    return {"identity": after, "entry_count": len(names), "names_sha256": digest.hexdigest()}


def _witness(path: str, errno: str) -> dict:
    """Resolve original components in kernel order; do not lexically normpath.

    Every visited directory and symlink is recorded.  A symlink's target is
    spliced into the pending components before a later '..' is interpreted.
    """
    _exact_path(path)
    pending = path.split("/")[1:]
    resolved: list[str] = []
    visited: list[dict] = []
    links = 0
    while pending:
        component = pending.pop(0)
        if component in ("", "."):
            continue
        if component == "..":
            if resolved:
                resolved.pop()
            visited.append({"operation": "parent", "resolved": "/" + "/".join(resolved)})
            continue
        parent = "/" + "/".join(resolved)
        parent_record = _directory(parent)
        visited.append({"operation": "directory", "record": parent_record})
        candidate = os.path.join(parent, component)
        try:
            current = _node(candidate)
        except FileNotFoundError:
            if errno != "ENOENT":
                raise Rejection("Readlink EINVAL witness has a missing pathname") from None
            return {"kind": "first_missing_component", "raw_path": path,
                    "first_missing": candidate, "unresolved_suffix": pending,
                    "visited": visited}
        visited.append({"operation": "component", "record": current})
        final = not pending
        if stat.S_ISLNK(current["mode"]) and not (errno == "EINVAL" and final):
            links += 1
            if links > 40:
                raise Rejection("Negative lookup exceeds bounded symlink resolution")
            target = os.readlink(candidate)
            if not target or "\x00" in target or any(ord(char) > 127 for char in target):
                raise Rejection("Unrepresentable negative lookup symlink target")
            visited.append({"operation": "symlink", "path": candidate, "target": target})
            if target.startswith("/"):
                resolved = []
                pending = target.split("/")[1:] + pending
            else:
                pending = target.split("/") + pending
            continue
        if errno == "EINVAL" and final:
            if stat.S_ISLNK(current["mode"]):
                raise Rejection("Readlink EINVAL target is actually a symlink")
            return {"kind": "existing_non_symlink", "raw_path": path,
                    "resolved_path": candidate, "leaf": current, "visited": visited}
        if not stat.S_ISDIR(current["mode"]) and not final:
            raise Rejection("Negative lookup would produce ENOTDIR, not ENOENT")
        resolved.append(component)
    raise Rejection("Negative lookup resolved to an existing pathname")


def _context() -> dict:
    try:
        status = Path("/proc/self/status").read_text()
        fsuid = next(line.split()[4] for line in status.splitlines() if line.startswith("Uid:"))
        fsgid = next(line.split()[4] for line in status.splitlines() if line.startswith("Gid:"))
        mount_ns = os.readlink("/proc/self/ns/mnt")
    except (OSError, StopIteration, IndexError) as error:
        raise Rejection(f"Cannot bind negative lookup resolution context: {error}") from error
    return {"uid": os.getuid(), "euid": os.geteuid(), "gid": os.getgid(),
            "egid": os.getegid(), "fsuid": int(fsuid), "fsgid": int(fsgid),
            "cwd": os.getcwd(), "mount_namespace": mount_ns, "root": _node("/")}


def _validate_rule(rule: dict) -> None:
    keys = {"rule_id", "path", "syscall", "errno", "phase", "dirfd", "qualifier", "max_count"}
    if not isinstance(rule, dict) or set(rule) != keys:
        raise Rejection("Negative lookup rule has unknown or missing fields")
    if (not isinstance(rule["rule_id"], str) or
            not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", rule["rule_id"]) or
            not isinstance(rule["phase"], str) or
            not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", rule["phase"]) or
            rule["dirfd"] != "AT_FDCWD" or
            type(rule["max_count"]) is not int or not 1 <= rule["max_count"] <= MAX_EVENTS):
        raise Rejection("Invalid finite negative lookup rule identity, phase, dirfd or count")
    _exact_path(rule["path"])
    if (not all(isinstance(rule[key], str) for key in ("syscall", "errno", "qualifier")) or
            _QUALIFIERS.get(rule["syscall"], {}).get(rule["qualifier"]) != rule["errno"]):
        raise Rejection("Negative lookup syscall, qualifier or errno is not an exact read-only shape")


def seal_contract(spec: dict) -> dict:
    """CPU-only: freeze explicit paths and actual pre-run filesystem witnesses."""
    if not isinstance(spec, dict) or set(spec) != {"schema", "rules"} or spec["schema"] != SPEC_SCHEMA:
        raise Rejection("Unknown negative lookup specification")
    rules = spec["rules"]
    if not isinstance(rules, list) or not 0 < len(rules) <= MAX_RULES:
        raise Rejection("Negative lookup requires a bounded nonempty exact rule list")
    seen_ids = set()
    seen_shapes = set()
    sealed = []
    for rule in rules:
        _validate_rule(rule)
        shape = tuple(rule[key] for key in ("path", "syscall", "errno", "phase", "dirfd", "qualifier"))
        if rule["rule_id"] in seen_ids or shape in seen_shapes:
            raise Rejection("Negative lookup rule ID or syscall shape is ambiguous")
        seen_ids.add(rule["rule_id"])
        seen_shapes.add(shape)
        try:
            witness = _witness(rule["path"], rule["errno"])
        except OSError as error:
            raise Rejection(f"Negative lookup pre-run witness unavailable: {error}") from error
        sealed.append({**rule, "witness": witness})
    if sum(row["max_count"] for row in sealed) > MAX_EVENTS:
        raise Rejection("Negative lookup event bound exceeded")
    sealed.sort(key=lambda row: row["rule_id"])
    payload = {"schema": CONTRACT_SCHEMA, "context": _context(), "rules": sealed,
               "scope": "finite_cpu_posthoc_parser_only", "native_profile_complete": False,
               "learning_allowed": False}
    payload["contract_id"] = identity(payload)
    return payload


def verify_contract(contract: dict) -> None:
    if (not isinstance(contract, dict) or
            set(contract) != {"schema", "context", "rules", "scope", "native_profile_complete",
                              "learning_allowed", "contract_id"} or
            contract["schema"] != CONTRACT_SCHEMA or
            contract["scope"] != "finite_cpu_posthoc_parser_only" or
            contract["native_profile_complete"] is not False or
            contract["learning_allowed"] is not False or
            identity({key: value for key, value in contract.items() if key != "contract_id"}) != contract["contract_id"]):
        raise Rejection("Negative lookup sealed identity or status changed")
    if contract["context"] != _context():
        raise Rejection("Negative lookup resolution context changed")
    rules = contract["rules"]
    if not isinstance(rules, list) or not 0 < len(rules) <= MAX_RULES:
        raise Rejection("Negative lookup sealed rule count invalid")
    ids = []
    shapes = set()
    for row in rules:
        if not isinstance(row, dict) or set(row) != {
                "rule_id", "path", "syscall", "errno", "phase", "dirfd", "qualifier", "max_count", "witness"}:
            raise Rejection("Negative lookup sealed rule shape changed")
        rule = {key: value for key, value in row.items() if key != "witness"}
        _validate_rule(rule)
        ids.append(rule["rule_id"])
        shape = tuple(rule[key] for key in ("path", "syscall", "errno", "phase", "dirfd", "qualifier"))
        try:
            current_witness = _witness(rule["path"], rule["errno"])
        except OSError as error:
            raise Rejection(f"Negative lookup post-run witness unavailable: {error}") from error
        if shape in shapes or current_witness != row["witness"]:
            raise Rejection("Negative lookup witness changed or rule is ambiguous")
        shapes.add(shape)
    if ids != sorted(set(ids)):
        raise Rejection("Negative lookup sealed rule order/ID changed")
    if sum(row["max_count"] for row in rules) > MAX_EVENTS:
        raise Rejection("Negative lookup sealed event bound changed")


def parse_failed_call(call: str) -> dict:
    """Parse only four explicit, read-only failed syscall argument shapes."""
    name_match = _CALL.match(call)
    result = _RESULT.search(call)
    if name_match is None or result is None or name_match[1] not in _QUALIFIERS:
        raise Rejection("Failed syscall is outside the finite negative lookup parser")
    name = name_match[1]
    pathname = _QUOTED.search(call)
    if pathname is None:
        raise Rejection("Failed syscall has no complete quoted pathname")
    dirfd = _DIRFD.fullmatch(call[name_match.end():pathname.start()])
    if dirfd is None:
        raise Rejection("Failed syscall has unknown dirfd or relative resolution")
    try:
        raw_path = ast.literal_eval(pathname[0])
    except (ValueError, SyntaxError) as error:
        raise Rejection("Failed syscall pathname has unknown strace escaping") from error
    _exact_path(raw_path)
    if call[pathname.end():].startswith("..."):
        raise Rejection("Failed syscall pathname was truncated")
    tail = call[pathname.end():result.start()]
    if name == "openat":
        match = re.fullmatch(r",\s*(O_RDONLY(?:\|O_CLOEXEC)?)", tail)
    elif name == "faccessat":
        match = re.fullmatch(r",\s*(R_OK)", tail)
    elif name == "newfstatat":
        match = re.fullmatch(r",\s*(?:" + _ADDRESS + r"|NULL),\s*(0)", tail)
    else:
        match = re.fullmatch(r",\s*" + _ADDRESS + r",\s*([1-9][0-9]*)", tail)
    if match is None:
        raise Rejection("Failed syscall has unknown flags, buffer or argument shape")
    qualifier = match[1]
    if name == "readlinkat":
        address = re.search(_ADDRESS, tail)
        if address is None or int(address[0], 16) == 0:
            raise Rejection("Readlink EINVAL requires a nonzero buffer address")
    if _QUALIFIERS[name].get(qualifier) != result[1]:
        raise Rejection("Failed syscall has unapproved errno or qualifier")
    return {"syscall": name, "errno": result[1], "path": raw_path,
            "dirfd": "AT_FDCWD", "dirfd_annotation": dirfd[1], "qualifier": qualifier,
            "arguments": call[name_match.end():result.start()], "raw_call": call}


class NegativeMatcher:
    def __init__(self, contract: dict):
        verify_contract(contract)
        self.contract = contract
        self.counts = {row["rule_id"]: 0 for row in contract["rules"]}

    def match(self, call: str, *, phase: str) -> dict:
        parsed = parse_failed_call(call)
        if parsed["dirfd_annotation"] is not None and parsed["dirfd_annotation"] != self.contract["context"]["cwd"]:
            raise Rejection("Failed syscall cwd annotation differs from sealed resolution context")
        matches = [row for row in self.contract["rules"] if all(
            row[key] == parsed[key] for key in ("path", "syscall", "errno", "dirfd", "qualifier")) and
            row["phase"] == phase]
        if len(matches) != 1:
            raise Rejection("Failed syscall has no unique predeclared negative lookup rule")
        row = matches[0]
        self.counts[row["rule_id"]] += 1
        if self.counts[row["rule_id"]] > row["max_count"]:
            raise Rejection("Negative lookup exceeded predeclared per-path event count")
        return {**parsed, "rule_id": row["rule_id"], "occurrence": self.counts[row["rule_id"]],
                "witness_sha256": identity(row["witness"]), "contract_id": self.contract["contract_id"]}

    def verify_after(self) -> None:
        verify_contract(self.contract)

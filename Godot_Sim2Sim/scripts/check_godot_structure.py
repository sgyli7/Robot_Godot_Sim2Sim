#!/usr/bin/env python3
"""Opt-in, read-only checks for new paths; never prepare or launch either project."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


DOMAINS = {"app", "effects", "entity", "environment", "game_play", "ui"}
SCENES = {".tscn", ".tres"}
SHADERS = {".gdshader", ".gdshaderinc", ".glsl"}
DATA = {".json", ".onnx", ".csv", ".yaml", ".yml", ".txt", ".bin"}
ART_TYPES = {
    "models": {".glb", ".gltf", ".fbx", ".obj", ".bin", ".mtl"},
    "textures": {".png", ".jpg", ".jpeg", ".webp", ".tga", ".exr", ".hdr", ".svg"},
    "materials": {".tres"},
    "animations": {".tres", ".glb", ".gltf", ".bin"},
    "audio": {".wav", ".mp3", ".ogg"},
    "fonts": {".ttf", ".otf", ".ttc"},
    "videos": {".ogv", ".mp4", ".webm"},
}
MEDIA = set().union(*ART_TYPES.values())
ASSET_SOURCE = MEDIA | {".blend", ".zip", ".xml", ".stl", ".usd", ".usda", ".usdc"}
AUTHORING = {".py", ".sh", ".gd", ".tscn", ".tres", ".json", ".csv", ".yaml", ".yml", ".txt"}
DOCUMENTS = {".md", ".json", ".txt", ".pdf"} | MEDIA
SNAKE = re.compile(r"_?[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
LICENSE = re.compile(r"(?:LICENSE|NOTICE|COPYING)(?:[._-].*)?$", re.IGNORECASE)


def git(root, *args):
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, check=False
    )
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("清单路径必须是非空相对路径")
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or p.as_posix() != value or value == ".":
        raise ValueError(f"清单路径不是精确文件路径: {value}")
    return value


def load_policy(root):
    if (root / "Godot_Sim2Sim/docs/directory_inventory.json").is_file():
        root = root / "Godot_Sim2Sim"
    path = root / "docs/directory_inventory.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(policy, dict):
        raise ValueError("清单根必须是 JSON object")
    if policy.get("schema_version") != 1 or policy.get("repository_role") not in {"lab", "art"}:
        raise ValueError("清单 schema_version 或 repository_role 无效")
    for key in ("legacy_paths", "compatibility_additions", "source_records"):
        if not isinstance(policy.get(key, []), list):
            raise ValueError(f"清单 {key} 必须是数组")
    for value in policy.get("legacy_paths", []):
        relative_path(value)
    exceptions = {}
    for key in ("compatibility_additions", "source_records"):
        for item in policy.get(key, []):
            if not isinstance(item, dict):
                raise ValueError(f"清单 {key} 必须包含精确文件记录")
            value = relative_path(item["path"])
            required = ("reason",) if key == "compatibility_additions" else ("repository", "revision", "sha256")
            if any(not isinstance(item.get(field), str) or not item[field].strip() for field in required):
                raise ValueError(f"{key}: {value} 缺少 {', '.join(required)}")
            if key == "source_records" and not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
                raise ValueError(f"source_records: {value} 的 SHA256 无效")
            if value in exceptions:
                raise ValueError(f"重复登记: {value}")
            exceptions[value] = (key, item)
    return root, policy, exceptions


def changed_paths(root, base):
    git_root = Path(git(root, "rev-parse", "--show-toplevel").decode().strip())
    git(root, "rev-parse", "--verify", f"{base}^{{commit}}")
    # Compare the whole worktree against base, including staged paths. Renames
    # must validate the destination even when the source is a legacy file.
    fields = git(git_root, "diff", "--name-status", "-z", "--find-renames", base, "--").split(b"\0")
    changed = set()
    index = 0
    while index < len(fields) and fields[index]:
        status = fields[index].decode()
        index += 1
        path = fields[index].decode("utf-8", errors="surrogateescape")
        index += 1
        if status.startswith(("R", "C")):
            path = fields[index].decode("utf-8", errors="surrogateescape")
            index += 1
        if status.startswith(("A", "R", "C")):
            changed.add(path)
    changed.update(
        value.decode("utf-8", errors="surrogateescape")
        for value in git(git_root, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
        if value
    )
    prefix = root.relative_to(git_root).as_posix()
    return sorted(
        path if prefix == "." else path[len(prefix) + 1:]
        for path in changed
        if prefix == "." or path.startswith(prefix + "/")
    )


def base_file(path):
    # Keep sidecars subject to the same placement rule as their primary asset.
    while path.suffix in {".uid", ".import"}:
        path = path.with_suffix("")
    return path


def name_error(path, existing_dirs=()):
    for parent in path.parents:
        if parent.as_posix() == "." or parent.as_posix() in existing_dirs:
            continue
        if not SNAKE.fullmatch(parent.name):
            return f"新目录名应为 snake_case: {parent.name}"
    if not LICENSE.fullmatch(path.name) and path.name != "README.md":
        if not SNAKE.fullmatch(path.stem):
            return f"新文件主名应为 snake_case: {path.name}"
    return None


def lab_error(value):
    path = base_file(PurePosixPath(value))
    parts = path.parts
    # Python, research, native source and maintenance documents have their own
    # layout. This checker does not apply res:// rules outside the Godot root.
    if parts[0] != "godot":
        return None
    parts = parts[1:]
    if not parts:
        return "Godot 根应为目录"
    if len(parts) == 1 and parts[0] in {
        "project.godot", "export_presets.cfg", ".gitignore", ".gitattributes", ".editorconfig"
    }:
        return None
    error = name_error(PurePosixPath(*parts))
    if error:
        return error
    ext = path.suffix
    license_file = bool(LICENSE.fullmatch(path.name))
    allowed = False
    if parts[:2] == ("addons", "dev_tools") and len(parts) >= 3:
        allowed = ext in {".gd", ".cfg", ".json"} | SCENES | SHADERS | MEDIA or license_file
    elif parts[0] in {"addons", "plugins"}:
        return "第三方插件/原生库需 source_records 精确来源登记"
    elif parts[0] == "i18n_assets" and len(parts) >= 2:
        allowed = ext in {".csv", ".po", ".pot", ".translation"} or license_file
    elif parts[:3] == ("game", "scripts", "core") and len(parts) >= 4:
        allowed = ext == ".gd"
    elif parts[:3] == ("game", "scripts", "modules") and len(parts) >= 5:
        allowed = ext == ".gd"
    elif parts[:2] == ("game", "shaders") and len(parts) >= 3:
        allowed = ext in SHADERS or license_file
    elif parts[:2] == ("game", "instances") and len(parts) >= 4 and parts[2] in DOMAINS:
        allowed = ext in SCENES
    elif parts[:2] == ("game", "scenes"):
        if parts == ("game", "scenes", "init.tscn"):
            allowed = True
        elif len(parts) >= 4 and parts[2] in {"frontend", "levels"}:
            allowed = ext == ".tscn"
        elif len(parts) >= 4 and parts[2] == "dev":
            allowed = ext in SCENES | DATA | MEDIA or license_file
    elif parts[:2] == ("game", "dynamic_assets") and len(parts) >= 4:
        category = parts[2]
        if category in {"game_data", "game_play", "settings"}:
            allowed = ext in DATA | {".tres"} or license_file
        elif category == "scenes":
            allowed = ext in SCENES or license_file
        elif category == "shaders":
            allowed = ext in SHADERS or license_file
    elif parts[:2] == ("game", "arts") and len(parts) >= 5:
        if parts[2] in DOMAINS and parts[3] in ART_TYPES:
            allowed = ext in ART_TYPES[parts[3]] or license_file
            if allowed and not license_file:
                return "Lab 美术部署副本需 source_records 精确来源、版本和 SHA256"
    return None if allowed else "路径/类型不在新 Godot 落点中；旧布局兼容需精确登记"


def art_error(value, existing_dirs):
    path = base_file(PurePosixPath(value))
    parts = path.parts
    if value in {"AGENTS.md", "CLAUDE.md", "README.md", "art_engineering_rules.md"}:
        return None
    if parts[:2] == (".cursor", "rules") and len(parts) == 3 and path.suffix == ".mdc":
        return None if SNAKE.fullmatch(path.stem) else "规则入口主名应为 snake_case"
    error = name_error(path, existing_dirs)
    if error:
        return error
    # Runtime delivery remains possible with a precise source/compatibility
    # record; language alone cannot distinguish preview tools from gameplay.
    allowed = False
    if len(parts) >= 2:
        if parts[0] == "docs":
            allowed = path.suffix in DOCUMENTS or bool(LICENSE.fullmatch(path.name))
        elif parts[0] in {"scripts", "tests"}:
            allowed = path.suffix in AUTHORING | ASSET_SOURCE or bool(LICENSE.fullmatch(path.name))
    if len(parts) >= 4 and parts[0] == "vehicles":
        category = parts[2]
        if category == "source":
            allowed = path.suffix in AUTHORING | ASSET_SOURCE or bool(LICENSE.fullmatch(path.name))
        elif category in {"assets", "themes"}:
            allowed = path.suffix in ASSET_SOURCE | {".json", ".tres"} or bool(LICENSE.fullmatch(path.name))
        elif category == "docs":
            allowed = path.suffix in DOCUMENTS or bool(LICENSE.fullmatch(path.name))
    return None if allowed else "新运行交付/旧布局需精确登记；新制作内容按 Art 目录规则落点"


def peer_notes(root, policy, peer_root):
    if not peer_root:
        return []
    peer = Path(peer_root).expanduser().resolve()
    if not peer.is_dir():
        return ["可选 peer 不存在，跳过比较；本地任务可继续"]
    if policy["repository_role"] == "lab":
        lab, art = root, peer
    else:
        lab, art = peer, root
    if (lab / "Godot_Sim2Sim").is_dir():
        lab = lab / "Godot_Sim2Sim"
    notes = []
    comparisons = policy.get("comparisons", [])
    if not comparisons and (lab / "docs/directory_inventory.json").is_file():
        # Read the single canonical comparison inventory, without requiring a
        # peer checkout for normal Art authoring or copying its data into Art.
        try:
            peer_policy = json.loads((lab / "docs/directory_inventory.json").read_text(encoding="utf-8"))
            if not isinstance(peer_policy, dict):
                raise ValueError("peer 清单根不是 JSON object")
            comparisons = peer_policy.get("comparisons", [])
        except (OSError, ValueError):
            notes.append("可选 peer 清单不可读，跳过副本比较；本地任务可继续")
    if not comparisons:
        notes.append("详细比较记录在 Lab 清单；本地路径检查不依赖 peer")
    if not isinstance(comparisons, list):
        return notes + ["可选 peer 副本记录无效（仅提示），跳过比较"]
    for item in comparisons:
        try:
            paths = [lab / relative_path(item["lab_path"]), art / relative_path(item["art_path"])]
            recorded_hashes = [item["lab_sha256"], item["art_sha256"]]
        except (ValueError, KeyError, TypeError):
            notes.append("可选副本比较记录无效（仅提示），跳过此项")
            continue
        if not all(p.is_file() for p in paths):
            notes.append(f"副本缺失（仅提示）: {item['lab_path']} ↔ {item['art_path']}")
            continue
        try:
            hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        except OSError:
            notes.append(f"副本不可读（仅提示）: {item['lab_path']} ↔ {item['art_path']}")
            continue
        if hashes != recorded_hashes:
            notes.append(f"副本已偏离记录基线（仅提示）: {item['lab_path']} ↔ {item['art_path']}")
    return notes


def check(root, base=None, peer_root=None):
    root, policy, exceptions = load_policy(root.resolve())
    revision = base or policy["baseline_commit"]
    candidates = changed_paths(root, revision)
    # ls-tree paths are relative to the current scope, including when Lab is a
    # subtree. A new file cannot become legacy simply by editing the inventory.
    committed = {
        value.decode("utf-8", errors="surrogateescape")
        for value in git(root, "ls-tree", "-r", "--name-only", "-z", revision, "--").split(b"\0")
        if value
    }
    legacy = set(policy.get("legacy_paths", [])) & committed
    existing_dirs = {parent.as_posix() for p in legacy for parent in PurePosixPath(p).parents}
    errors = []
    for value in candidates:
        # Only exact stored paths are legacy, never an entire directory prefix.
        if value in legacy:
            continue
        if value in exceptions:
            kind, record = exceptions[value]
            if kind == "source_records":
                target = root / value
                if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != record["sha256"]:
                    errors.append(f"{value}: 新副本 SHA256 与来源登记不符")
            continue
        error = lab_error(value) if policy["repository_role"] == "lab" else art_error(value, existing_dirs)
        if error:
            errors.append(f"{value}: {error}")
    notes = peer_notes(root, policy, peer_root)
    summary = policy.get("comparison_summary")
    if summary:
        notes.insert(0, f"记录基线的机器人副本: {summary['robot_same_path_equal']} 一致 / {summary['robot_same_path_diverged']} 分叉（非实时扫描）。")
    return candidates, errors, notes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--base", help="已提交比较点；默认使用清单 baseline_commit")
    parser.add_argument("--peer-root", help="可选另一仓库检出根；分叉只提示")
    args = parser.parse_args(argv)
    try:
        candidates, errors, notes = check(args.repo_root, args.base, args.peer_root)
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"无法读取结构检查输入: {error}", file=sys.stderr)
        return 2
    for note in notes:
        print(f"提示: {note}")
    for error in errors:
        print(f"不符合新增路径规则: {error}", file=sys.stderr)
    print(f"检查新增/搬移文件 {len(candidates)} 个，路径问题 {len(errors)} 个。已有修改、历史分叉不阻断。")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

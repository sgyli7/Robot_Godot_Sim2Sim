//! Static checks for directory layout, file placement, crate layering, and `.gitignore`.
//!
//! The report covers only rules that can be decided from paths and Cargo metadata.
//! It does not claim that rendering, physics, runtime semantics, release asset
//! dependencies, or loaded-config immutability have passed.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;
use serde::Serialize;

/// Report scope. A pass here is not a pass for gameplay, rendering, or physics.
pub const SCOPE: &str = "static_engineering_structure";

const NOT_GUARANTEED: &[&str] = &[
    "release_package_excludes_dev_scenes",
    "production_code_and_assets_do_not_depend_on_dev_assets",
    "config_fields_stay_immutable_after_load",
    "rendering",
    "physics",
    "runtime_semantics",
    "rust_identifier_and_comment_style",
    "runtime_asset_loader_support_and_source_conversion",
];

const REQUIRED_IGNORES: &[&str] = &[
    "target/",
    "build/",
    "logs/",
    ".codegraph/",
    ".scratch/",
    ".claude/",
];

const ASSET_ROOTS: &[&str] = &["game", "i18n_assets", "third_party"];
const GAME_DIRS: &[&str] = &["arts", "instances", "shaders", "scenes", "dynamic_assets"];
const DOMAINS: &[&str] = &["app", "effects", "entity", "environment", "game_play", "ui"];
const SCENE_DIRS: &[&str] = &["frontend", "levels", "dev"];
const DYNAMIC_DIRS: &[&str] = &["game_data", "game_play", "scenes", "settings", "shaders"];
const ART_TYPES: &[&str] = &[
    "models",
    "textures",
    "materials",
    "animations",
    "audio",
    "fonts",
    "videos",
];
const FORBIDDEN_DIR_NAMES: &[&str] = &["gameplay", "art", "video", "sprites", "localization"];

/// One static failure. `path` is relative to the requested project root.
#[derive(Debug, Serialize)]
pub struct Failure {
    /// Project-relative path that failed, or `.` when the root itself is unusable.
    pub path: String,
    /// Why that path failed the static rule.
    pub reason: String,
}

/// JSON report returned by the checker. `passed` means the static checks passed.
#[derive(Debug, Serialize)]
pub struct StructureReport {
    /// Report schema. Increment when the JSON fields change.
    pub schema_version: u32,
    /// Fixed scope label so a pass cannot be read as a runtime or rendering pass.
    pub scope: &'static str,
    /// `true` only when `failures` is empty.
    pub passed: bool,
    /// Number of static checks that were actually executed.
    pub check_count: u32,
    /// Failed checks, each with a path and a reason.
    pub failures: Vec<Failure>,
    /// Rules this checker does not decide. A pass does not cover them.
    pub not_guaranteed: &'static [&'static str],
}

/// Build a one-failure report when the project root cannot be inspected.
pub fn configuration_report(reason: impl Into<String>) -> StructureReport {
    StructureReport {
        schema_version: 1,
        scope: SCOPE,
        passed: false,
        check_count: 1,
        failures: vec![Failure {
            path: ".".to_string(),
            reason: reason.into(),
        }],
        not_guaranteed: NOT_GUARANTEED,
    }
}

/// Serialize `report` as pretty JSON. Serialization failure becomes a configuration report.
pub fn to_json(report: &StructureReport) -> String {
    match serde_json::to_string_pretty(report) {
        Ok(json) => json,
        Err(error) => {
            let fallback =
                configuration_report(format!("configuration: JSON serialization failed: {error}"));
            serde_json::to_string_pretty(&fallback)
                .unwrap_or_else(|_| "{\"passed\":false}".to_string())
        }
    }
}

/// Check `project_root`. Missing manifests and metadata errors stay failures.
pub fn check(project_root: &Path) -> StructureReport {
    let Ok(root) = project_root.canonicalize() else {
        return configuration_report(format!(
            "configuration: project root `{}` is unavailable",
            project_root.display()
        ));
    };
    if !root.is_dir() {
        return configuration_report(format!(
            "configuration: project root `{}` is not a directory",
            root.display()
        ));
    }
    let mut ctx = Ctx {
        root,
        checks: 0,
        failures: Vec::new(),
    };
    let modules = disk_modules(&ctx.root);
    walk(&mut ctx, &modules, PathBuf::new());
    check_gitignore(&mut ctx);
    check_cargo(&mut ctx, &modules);
    ctx.finish()
}

struct Ctx {
    root: PathBuf,
    checks: u32,
    failures: Vec<Failure>,
}

impl Ctx {
    fn record(&mut self, path: &Path, ok: bool, reason: impl AsRef<str>) {
        self.checks += 1;
        if !ok {
            self.failures.push(Failure {
                path: display_path(&self.root, path),
                reason: reason.as_ref().to_string(),
            });
        }
    }

    fn finish(self) -> StructureReport {
        StructureReport {
            schema_version: 1,
            scope: SCOPE,
            passed: self.failures.is_empty(),
            check_count: self.checks,
            failures: self.failures,
            not_guaranteed: NOT_GUARANTEED,
        }
    }
}

fn display_path(root: &Path, path: &Path) -> String {
    let relative = path.strip_prefix(root).unwrap_or(path);
    let text = relative.to_string_lossy().replace('\\', "/");
    if text.is_empty() {
        ".".to_string()
    } else {
        text
    }
}

fn disk_modules(root: &Path) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let Ok(entries) = fs::read_dir(root.join("crates/modules")) else {
        return names;
    };
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_dir() {
            continue;
        }
        if let Some(name) = entry.file_name().to_str() {
            names.insert(name.to_string());
        }
    }
    names
}

fn walk(ctx: &mut Ctx, modules: &BTreeSet<String>, rel: PathBuf) {
    let absolute = if rel.as_os_str().is_empty() {
        ctx.root.clone()
    } else {
        ctx.root.join(&rel)
    };
    let entries = match fs::read_dir(&absolute) {
        Ok(entries) => entries,
        Err(error) => {
            ctx.record(
                &rel,
                false,
                format!("configuration: cannot read directory: {error}"),
            );
            return;
        }
    };
    let mut children: Vec<_> = entries.flatten().collect();
    children.sort_by_key(|entry| entry.file_name());
    for entry in children {
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            ctx.record(
                &rel.join(entry.file_name()),
                false,
                "file name is not valid UTF-8",
            );
            continue;
        };
        let child = if rel.as_os_str().is_empty() {
            PathBuf::from(&name)
        } else {
            rel.join(&name)
        };
        let Ok(file_type) = entry.file_type() else {
            ctx.record(&child, false, "configuration: cannot inspect file type");
            continue;
        };
        if should_skip(&child) {
            continue;
        }
        if file_type.is_symlink() {
            ctx.record(
                &child,
                false,
                "symbolic links are outside the static structure check",
            );
            continue;
        }
        if file_type.is_dir() {
            if should_skip(&child) {
                continue;
            }
            match directory_failure(&child, modules) {
                Some(reason) => ctx.record(&child, false, reason),
                None => ctx.record(&child, true, ""),
            }
            walk(ctx, modules, child);
            continue;
        }
        if file_type.is_file() {
            check_file(ctx, modules, &child, &name);
        }
    }
}

fn should_skip(rel: &Path) -> bool {
    if rel.starts_with(Path::new("assets/third_party")) {
        return true;
    }
    let Some(first) = rel.components().next() else {
        return false;
    };
    let first = first.as_os_str().to_str().unwrap_or("");
    if matches!(
        first,
        "target"
            | ".git"
            | ".scratch"
            | "build"
            | "logs"
            | ".codegraph"
            | "third_party"
            | "plugins"
    ) {
        return true;
    }
    rel.components().any(|component| {
        matches!(
            component.as_os_str().to_str().unwrap_or(""),
            "target" | ".git" | ".scratch" | "__pycache__" | ".pytest_cache" | ".venv"
        )
    })
}

fn directory_failure(rel: &Path, modules: &BTreeSet<String>) -> Option<String> {
    let name = rel.file_name()?.to_str()?;
    if FORBIDDEN_DIR_NAMES.contains(&name) {
        return Some(format!("directory name `{name}` is a forbidden spelling"));
    }
    if under_src_crates_assets(rel) && is_ai_directory(name) {
        return Some(
            "AI configuration directories are only allowed at the project root".to_string(),
        );
    }
    if let Some(reason) = literal_directory_failure(rel, name, modules) {
        return Some(reason);
    }
    if !exempt_directory_naming(rel) && !is_snake_case(name) {
        return Some(format!("directory name `{name}` must be ASCII snake_case"));
    }
    None
}

fn literal_directory_failure(rel: &Path, name: &str, modules: &BTreeSet<String>) -> Option<String> {
    let parent = rel.parent()?;
    if parent.as_os_str().is_empty() {
        return None;
    }
    let allowed = if parent == Path::new("assets") {
        Some(ASSET_ROOTS)
    } else if parent == Path::new("assets/game") {
        Some(GAME_DIRS)
    } else if parent == Path::new("assets/game/arts")
        || parent == Path::new("assets/game/instances")
    {
        Some(DOMAINS)
    } else if parent == Path::new("assets/game/scenes") {
        Some(SCENE_DIRS)
    } else if parent == Path::new("assets/game/dynamic_assets") {
        Some(DYNAMIC_DIRS)
    } else if parent == Path::new("assets/game/dynamic_assets/settings") {
        if name == "core" || name == "project" || modules.contains(name) {
            return None;
        }
        return Some(format!(
            "settings module `{name}` is not crates/modules/<module>, core, or project"
        ));
    } else if let Ok(rest) = parent.strip_prefix(Path::new("assets/game/arts")) {
        let mut parts = rest.components();
        let domain = parts.next()?.as_os_str().to_str().unwrap_or("");
        if parts.next().is_none() && DOMAINS.contains(&domain) {
            if ART_TYPES.contains(&name) {
                return None;
            }
            return Some(format!(
                "directory `{name}` is not an arts type folder under `{domain}`"
            ));
        }
        return None;
    } else {
        return None;
    };
    if allowed.is_some_and(|names| names.contains(&name)) {
        None
    } else {
        Some(format!(
            "directory `{name}` is not allowed in `{}`",
            parent.display()
        ))
    }
}

fn exempt_directory_naming(rel: &Path) -> bool {
    rel.starts_with("docs")
        || rel.starts_with(".cursor")
        || rel.starts_with(".claude")
        || rel.starts_with(".codex")
        || rel.starts_with(".agents")
        || rel.starts_with("crates/dev_tools/python")
}

fn under_src_crates_assets(rel: &Path) -> bool {
    rel.starts_with("src") || rel.starts_with("crates") || rel.starts_with("assets")
}

fn is_ai_directory(name: &str) -> bool {
    matches!(name, ".cursor" | ".claude" | ".codex" | ".agents")
}

fn is_forbidden_ai_file(name: &str) -> bool {
    name == "AGENTS.md"
        || name == "CLAUDE.md"
        || name.contains("规范")
        || name.contains("RULE")
        || name.starts_with("CODING_STYLE")
}

#[derive(Clone, Copy)]
enum AssetKind {
    Rust,
    Wgsl,
    ScnRon,
    Ron,
    Model,
    Texture,
    Audio,
    Font,
    Video,
    Localization,
    Sidecar,
    Python,
    Onnx,
    Json,
}

fn classify_name(name: &str) -> Option<AssetKind> {
    if name.ends_with(".scn.ron") {
        return Some(AssetKind::ScnRon);
    }
    let (_, extension) = name.rsplit_once('.')?;
    Some(match extension {
        "rs" => AssetKind::Rust,
        "wgsl" => AssetKind::Wgsl,
        "ron" => AssetKind::Ron,
        "glb" | "gltf" | "fbx" | "obj" => AssetKind::Model,
        "png" | "jpg" | "tga" | "exr" | "psd" => AssetKind::Texture,
        "wav" | "mp3" | "ogg" => AssetKind::Audio,
        "ttf" | "otf" => AssetKind::Font,
        "mp4" => AssetKind::Video,
        "csv" | "po" | "ftl" => AssetKind::Localization,
        "bin" | "mtl" | "meta" => AssetKind::Sidecar,
        "py" => AssetKind::Python,
        "onnx" => AssetKind::Onnx,
        "json" => AssetKind::Json,
        _ => return None,
    })
}

fn check_file(ctx: &mut Ctx, modules: &BTreeSet<String>, rel: &Path, name: &str) {
    if under_src_crates_assets(rel) {
        ctx.record(
            rel,
            !is_forbidden_ai_file(name),
            "AI configuration files are forbidden under src, crates, and assets",
        );
    }
    if rel.parent() == Some(Path::new("assets")) {
        ctx.record(rel, false, "assets root must not contain files");
    }
    let Some(kind) = classify_name(name) else {
        if rel.starts_with("assets") {
            ctx.record(
                rel,
                false,
                "unlisted self-developed asset extension is forbidden",
            );
        }
        return;
    };
    ctx.record(
        rel,
        placement_allowed(rel, kind, modules),
        "self-developed file is outside its allowed directory",
    );
    if matches!(kind, AssetKind::Sidecar) {
        ctx.record(
            rel,
            sidecar_has_primary(&ctx.root.join(rel)),
            "asset sidecar has no corresponding primary asset/reference",
        );
    }
    let primary_name = if name.ends_with(".meta") {
        name.strip_suffix(".meta").unwrap_or(name)
    } else {
        name
    };
    let stem = file_stem(primary_name);
    if is_top_level_scene(rel, kind) {
        ctx.record(
            rel,
            scene_name_ok(stem),
            "top-level scene name must be 2-3 snake_case segments, or init/loading",
        );
    } else {
        ctx.record(
            rel,
            is_snake_case(stem)
                || (matches!(kind, AssetKind::Python)
                    && matches!(name, "__init__.py" | "__main__.py")),
            "file name must be ASCII snake_case",
        );
    }
}

fn file_stem(name: &str) -> &str {
    name.strip_suffix(".scn.ron")
        .unwrap_or_else(|| name.rsplit_once('.').map(|(stem, _)| stem).unwrap_or(name))
}

fn is_top_level_scene(rel: &Path, kind: AssetKind) -> bool {
    matches!(kind, AssetKind::ScnRon)
        && (rel.starts_with("assets/game/scenes")
            || rel.starts_with("assets/game/dynamic_assets/scenes"))
}

fn placement_allowed(rel: &Path, kind: AssetKind, modules: &BTreeSet<String>) -> bool {
    match kind {
        AssetKind::Rust => rust_source_allowed(rel),
        AssetKind::Wgsl => {
            has_file_after(rel, "assets/game/shaders")
                || in_dynamic_category(rel)
                || in_scene_dev(rel)
        }
        AssetKind::ScnRon => {
            entity_template(rel)
                || top_level_scene_path(rel)
                || in_dynamic_category(rel)
                || in_scene_dev(rel)
        }
        AssetKind::Ron => {
            under_art_bucket(rel, "materials")
                || under_art_bucket(rel, "animations")
                || has_file_after(rel, "assets/i18n_assets")
                || in_dynamic_category(rel)
                || in_scene_dev(rel)
        }
        AssetKind::Model => {
            under_art_bucket(rel, "models") || in_dynamic_category(rel) || in_scene_dev(rel)
        }
        AssetKind::Texture => {
            under_art_bucket(rel, "textures") || in_dynamic_category(rel) || in_scene_dev(rel)
        }
        AssetKind::Audio => {
            under_art_bucket(rel, "audio") || in_dynamic_category(rel) || in_scene_dev(rel)
        }
        AssetKind::Font => {
            under_art_bucket(rel, "fonts") || in_dynamic_category(rel) || in_scene_dev(rel)
        }
        AssetKind::Video => {
            under_art_bucket(rel, "videos") || in_dynamic_category(rel) || in_scene_dev(rel)
        }
        AssetKind::Localization => has_file_after(rel, "assets/i18n_assets"),
        AssetKind::Sidecar => {
            under_art_bucket(rel, "models")
                || under_art_bucket(rel, "textures")
                || under_art_bucket(rel, "materials")
                || under_art_bucket(rel, "animations")
                || under_art_bucket(rel, "audio")
                || under_art_bucket(rel, "fonts")
                || under_art_bucket(rel, "videos")
                || in_dynamic_category(rel)
                || in_scene_dev(rel)
        }
        AssetKind::Python => has_file_after(rel, "crates/dev_tools/python"),
        AssetKind::Onnx => onnx_in_existing_module(rel, modules),
        AssetKind::Json => {
            onnx_in_existing_module(rel, modules)
                || has_file_after(rel, "assets/i18n_assets")
                || has_file_after(rel, "crates/dev_tools/python")
        }
    }
}

fn sidecar_has_primary(path: &Path) -> bool {
    let Some(parent) = path.parent() else {
        return false;
    };
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    if let Some(primary) = name.strip_suffix(".meta") {
        return parent.join(primary).is_file() && classify_name(primary).is_some();
    }
    let Ok(entries) = fs::read_dir(parent) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let primary = entry.path();
        match primary.extension().and_then(|ext| ext.to_str()) {
            Some("gltf") if name.ends_with(".bin") => {
                let Ok(bytes) = fs::read(&primary) else {
                    return false;
                };
                let Ok(document) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
                    return false;
                };
                document["buffers"].as_array().is_some_and(|buffers| {
                    buffers
                        .iter()
                        .any(|buffer| buffer["uri"].as_str() == Some(name))
                })
            }
            Some("obj") if name.ends_with(".mtl") => {
                fs::read_to_string(primary).is_ok_and(|text| {
                    text.lines().any(|line| {
                        line.trim()
                            .strip_prefix("mtllib ")
                            .is_some_and(|files| files.split_whitespace().any(|file| file == name))
                    })
                })
            }
            _ => false,
        }
    })
}

fn rust_source_allowed(rel: &Path) -> bool {
    if rel == Path::new("src/main.rs") || rel == Path::new("build.rs") {
        return true;
    }
    let mut parts = rel.components();
    if parts.next().and_then(|part| part.as_os_str().to_str()) != Some("crates") {
        return false;
    }
    let rest: Vec<&str> = match parts.next().and_then(|part| part.as_os_str().to_str()) {
        Some("core") => parts
            .map(|part| part.as_os_str().to_str().unwrap_or(""))
            .collect(),
        Some("dev_tools") => parts
            .map(|part| part.as_os_str().to_str().unwrap_or(""))
            .collect(),
        Some("modules") => {
            let Some(module) = parts.next().and_then(|part| part.as_os_str().to_str()) else {
                return false;
            };
            if !is_snake_case(module) {
                return false;
            }
            parts
                .map(|part| part.as_os_str().to_str().unwrap_or(""))
                .collect()
        }
        _ => return false,
    };
    match rest.as_slice() {
        ["build.rs"] => true,
        [folder, tail @ ..] if is_cargo_code_dir(folder) && !tail.is_empty() => true,
        _ => false,
    }
}

fn is_cargo_code_dir(name: &str) -> bool {
    matches!(name, "src" | "tests" | "examples" | "benches")
}

fn has_file_after(rel: &Path, prefix: &str) -> bool {
    rel.strip_prefix(Path::new(prefix))
        .is_ok_and(|rest| rest.components().next().is_some())
}

fn in_dynamic_category(rel: &Path) -> bool {
    let Ok(rest) = rel.strip_prefix(Path::new("assets/game/dynamic_assets")) else {
        return false;
    };
    let mut parts = rest.components();
    let Some(category) = parts.next().and_then(|part| part.as_os_str().to_str()) else {
        return false;
    };
    DYNAMIC_DIRS.contains(&category) && parts.next().is_some()
}

fn in_scene_dev(rel: &Path) -> bool {
    has_file_after(rel, "assets/game/scenes/dev")
}

fn under_art_bucket(rel: &Path, bucket: &str) -> bool {
    let Ok(rest) = rel.strip_prefix(Path::new("assets/game/arts")) else {
        return false;
    };
    let mut parts = rest.components();
    let Some(domain) = parts.next().and_then(|part| part.as_os_str().to_str()) else {
        return false;
    };
    let Some(kind) = parts.next().and_then(|part| part.as_os_str().to_str()) else {
        return false;
    };
    DOMAINS.contains(&domain) && kind == bucket && parts.next().is_some()
}

fn entity_template(rel: &Path) -> bool {
    let Ok(rest) = rel.strip_prefix(Path::new("assets/game/instances")) else {
        return false;
    };
    let mut parts = rest.components();
    let Some(domain) = parts.next().and_then(|part| part.as_os_str().to_str()) else {
        return false;
    };
    DOMAINS.contains(&domain) && parts.next().is_some()
}

fn top_level_scene_path(rel: &Path) -> bool {
    if rel == Path::new("assets/game/scenes/init.scn.ron")
        || rel == Path::new("assets/game/scenes/loading.scn.ron")
    {
        return true;
    }
    let Ok(rest) = rel.strip_prefix(Path::new("assets/game/scenes")) else {
        return false;
    };
    let mut parts = rest.components();
    let Some(dir) = parts.next().and_then(|part| part.as_os_str().to_str()) else {
        return false;
    };
    SCENE_DIRS.contains(&dir) && parts.next().is_some()
}

fn onnx_in_existing_module(rel: &Path, modules: &BTreeSet<String>) -> bool {
    let Ok(rest) = rel.strip_prefix(Path::new("assets/game/dynamic_assets/game_data")) else {
        return false;
    };
    let mut parts = rest.components();
    let Some(module) = parts.next().and_then(|part| part.as_os_str().to_str()) else {
        return false;
    };
    module == "robot" && modules.contains(module) && parts.next().is_some()
}

fn is_snake_case(name: &str) -> bool {
    let mut parts = name.split('_');
    let Some(first) = parts.next() else {
        return false;
    };
    if !snake_segment(first, true) {
        return false;
    }
    parts.all(|part| snake_segment(part, false))
}

fn snake_segment(part: &str, first: bool) -> bool {
    let mut chars = part.chars();
    let Some(lead) = chars.next() else {
        return false;
    };
    let lead_ok = if first {
        lead.is_ascii_lowercase()
    } else {
        lead.is_ascii_lowercase() || lead.is_ascii_digit()
    };
    lead_ok && chars.all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit())
}

fn scene_name_ok(stem: &str) -> bool {
    if stem == "init" || stem == "loading" {
        return true;
    }
    let segments = stem.split('_').count();
    (2..=3).contains(&segments) && is_snake_case(stem)
}

fn check_gitignore(ctx: &mut Ctx) {
    let text = match fs::read_to_string(ctx.root.join(".gitignore")) {
        Ok(text) => text,
        Err(error) => {
            ctx.record(
                Path::new(".gitignore"),
                false,
                format!("configuration: .gitignore is missing or unreadable: {error}"),
            );
            return;
        }
    };
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();
    for entry in REQUIRED_IGNORES {
        ctx.record(
            Path::new(".gitignore"),
            lines.iter().any(|line| line == entry),
            format!(".gitignore is missing required entry `{entry}`"),
        );
    }
    let keeps_rules = lines.contains(&"!.cursor/rules/") && lines.contains(&"!.cursor/rules/**");
    ctx.record(
        Path::new(".gitignore"),
        lines.iter().any(|line| *line == ".cursor/*") && keeps_rules,
        ".gitignore must ignore .cursor/* and keep .cursor/rules",
    );
}

fn check_cargo(ctx: &mut Ctx, modules: &BTreeSet<String>) {
    let manifest_exists = ctx.root.join("Cargo.toml").is_file();
    ctx.record(
        Path::new("Cargo.toml"),
        manifest_exists,
        "configuration: root Cargo.toml is missing",
    );
    ctx.record(
        Path::new("Cargo.lock"),
        ctx.root.join("Cargo.lock").is_file(),
        "configuration: root Cargo.lock is missing",
    );
    require_member_manifest(ctx, "crates/core");
    require_member_manifest(ctx, "crates/dev_tools");
    for module in modules {
        require_member_manifest(ctx, &format!("crates/modules/{module}"));
    }
    if !manifest_exists {
        return;
    }
    let metadata = match run_metadata(&ctx.root) {
        Ok(metadata) => metadata,
        Err(reason) => {
            ctx.record(Path::new("Cargo.toml"), false, reason);
            return;
        }
    };
    analyze_metadata(ctx, metadata);
}

fn require_member_manifest(ctx: &mut Ctx, rel_dir: &str) {
    let dir = Path::new(rel_dir);
    if !ctx.root.join(dir).is_dir() {
        ctx.record(
            dir,
            false,
            format!("configuration: required crate directory `{rel_dir}` is missing"),
        );
        return;
    }
    let manifest = dir.join("Cargo.toml");
    ctx.record(
        &manifest,
        ctx.root.join(&manifest).is_file(),
        format!("configuration: member Cargo.toml is missing at `{rel_dir}/Cargo.toml`"),
    );
}

fn run_metadata(root: &Path) -> Result<Metadata, String> {
    let output = Command::new("cargo")
        .args([
            "metadata",
            "--no-deps",
            "--locked",
            "--offline",
            "--format-version",
            "1",
            "--manifest-path",
        ])
        .arg(root.join("Cargo.toml"))
        .current_dir(root)
        .output()
        .map_err(|error| format!("configuration: failed to run cargo metadata: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "configuration: cargo metadata failed: {}",
            truncate(stderr.trim(), 800)
        ));
    }
    serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("configuration: cargo metadata JSON could not be parsed: {error}"))
}

fn truncate(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn analyze_metadata(ctx: &mut Ctx, metadata: Metadata) {
    let meta_root = match metadata.workspace_root.canonicalize() {
        Ok(path) => path,
        Err(error) => {
            ctx.record(
                Path::new("Cargo.toml"),
                false,
                format!("configuration: cargo metadata workspace root is unavailable: {error}"),
            );
            return;
        }
    };
    if meta_root != ctx.root {
        ctx.record(
            Path::new("Cargo.toml"),
            false,
            "configuration: cargo metadata workspace root does not match the requested project root",
        );
        return;
    }
    let member_ids: HashSet<&str> = metadata
        .workspace_members
        .iter()
        .map(String::as_str)
        .collect();
    let mut members = Vec::new();
    for package in &metadata.packages {
        if !member_ids.contains(package.id.as_str()) {
            continue;
        }
        let Ok(manifest_path) = package.manifest_path.canonicalize() else {
            ctx.record(
                Path::new("Cargo.toml"),
                false,
                format!(
                    "configuration: manifest `{}` from cargo metadata is unavailable",
                    package.manifest_path.display()
                ),
            );
            continue;
        };
        let Ok(rel) = manifest_path.strip_prefix(&ctx.root) else {
            ctx.record(
                Path::new("Cargo.toml"),
                false,
                format!(
                    "configuration: manifest `{}` is outside the requested project root",
                    manifest_path.display()
                ),
            );
            continue;
        };
        if rel.starts_with("third_party")
            || rel.starts_with("plugins")
            || rel.starts_with("assets/third_party")
        {
            continue;
        }
        let Some(layer) = classify_member(rel) else {
            ctx.record(
                rel,
                false,
                "workspace member is outside the runtime/core/dev layout",
            );
            continue;
        };
        if let Layer::Business(module) = &layer {
            ctx.record(
                rel,
                is_snake_case(module),
                format!("runtime module `{module}` must be ASCII snake_case"),
            );
            ctx.record(
                rel,
                package.name == format!("{module}_minigame"),
                format!("crate name must be `{module}_minigame`"),
            );
        } else if let Some(expected) = expected_name(&layer) {
            ctx.record(
                rel,
                package.name == expected,
                format!("crate name must be `{expected}`"),
            );
        }
        members.push(Member {
            name: package.name.clone(),
            manifest: rel.to_path_buf(),
            manifest_dir: manifest_path
                .parent()
                .unwrap_or(manifest_path.as_path())
                .to_path_buf(),
            layer,
            features: package.features.clone(),
            deps: package.dependencies.clone(),
        });
    }

    expect_workspace_member(ctx, &members, Path::new("crates/core/Cargo.toml"));
    expect_workspace_member(ctx, &members, Path::new("crates/dev_tools/Cargo.toml"));
    if let Ok(entries) = fs::read_dir(ctx.root.join("crates/modules")) {
        let mut module_dirs: Vec<_> = entries.flatten().collect();
        module_dirs.sort_by_key(|entry| entry.file_name());
        for entry in module_dirs {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if !file_type.is_dir() {
                continue;
            }
            let Some(module) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            let manifest = PathBuf::from("crates/modules")
                .join(module)
                .join("Cargo.toml");
            if ctx.root.join(&manifest).is_file() {
                expect_workspace_member(ctx, &members, &manifest);
            }
        }
    }

    for member in &members {
        let mut problems = Vec::new();
        for dep in &member.deps {
            let Some(target) = resolve_internal(dep, &members) else {
                continue;
            };
            if !layer_edge_allowed(&member.layer, &target.layer) {
                problems.push(format!("{} depends on {}", member.name, target.name));
            }
        }
        ctx.record(
            &member.manifest,
            problems.is_empty(),
            format!("forbidden crate dependency: {}", problems.join("; ")),
        );
    }
    match find_cycle(&business_graph(&members)) {
        Some(cycle) => {
            let path = members
                .iter()
                .find(|member| member.name == cycle[0])
                .map(|member| member.manifest.clone())
                .unwrap_or_else(|| PathBuf::from("Cargo.toml"));
            ctx.record(
                &path,
                false,
                format!(
                    "business crate dependencies form a cycle: {}",
                    cycle.join(" -> ")
                ),
            );
        }
        None => ctx.record(Path::new("Cargo.toml"), true, ""),
    }
    check_root_dev_tools(ctx, &members);
}

fn expect_workspace_member(ctx: &mut Ctx, members: &[Member], manifest: &Path) {
    let found = members.iter().any(|member| member.manifest == manifest);
    ctx.record(
        manifest,
        found,
        format!(
            "`{}` is not a workspace member",
            display_path(&ctx.root, manifest)
        ),
    );
}

fn classify_member(rel: &Path) -> Option<Layer> {
    if rel == Path::new("Cargo.toml") {
        return Some(Layer::Root);
    }
    if rel == Path::new("crates/core/Cargo.toml") {
        return Some(Layer::Core);
    }
    if rel == Path::new("crates/dev_tools/Cargo.toml") {
        return Some(Layer::Dev);
    }
    let rest = rel.strip_prefix(Path::new("crates/modules")).ok()?;
    let mut parts = rest.components();
    let module = parts.next()?.as_os_str().to_str()?.to_string();
    let file = parts.next()?.as_os_str().to_str()?;
    if file == "Cargo.toml" && parts.next().is_none() {
        Some(Layer::Business(module))
    } else {
        None
    }
}

fn expected_name(layer: &Layer) -> Option<&'static str> {
    match layer {
        Layer::Core => Some("common_minigame"),
        Layer::Dev => Some("dev_tools_minigame"),
        Layer::Root | Layer::Business(_) => None,
    }
}

fn resolve_internal<'a>(dep: &Dependency, members: &'a [Member]) -> Option<&'a Member> {
    if let Some(path) = &dep.path {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
        return members
            .iter()
            .find(|member| member.manifest_dir == canonical);
    }
    if dep.source.is_some() {
        return None;
    }
    members.iter().find(|member| member.name == dep.name)
}

fn layer_edge_allowed(from: &Layer, to: &Layer) -> bool {
    !matches!(
        (from, to),
        (Layer::Core, Layer::Business(_))
            | (Layer::Core, Layer::Dev)
            | (Layer::Business(_), Layer::Dev)
    )
}

fn business_graph(members: &[Member]) -> BTreeMap<String, Vec<String>> {
    let mut graph = BTreeMap::new();
    for member in members {
        if matches!(member.layer, Layer::Business(_)) {
            graph.insert(member.name.clone(), Vec::new());
        }
    }
    for member in members {
        if !matches!(member.layer, Layer::Business(_)) {
            continue;
        }
        let Some(edges) = graph.get_mut(&member.name) else {
            continue;
        };
        for dep in &member.deps {
            let Some(target) = resolve_internal(dep, members) else {
                continue;
            };
            if matches!(target.layer, Layer::Business(_)) {
                edges.push(target.name.clone());
            }
        }
        edges.sort();
        edges.dedup();
    }
    graph
}

fn find_cycle(graph: &BTreeMap<String, Vec<String>>) -> Option<Vec<String>> {
    let mut color: BTreeMap<&str, u8> = graph.keys().map(|name| (name.as_str(), 0)).collect();
    let mut stack = Vec::new();
    for node in graph.keys() {
        if color[node.as_str()] == 0 {
            if let Some(cycle) = visit(graph, &mut color, &mut stack, node) {
                return Some(cycle);
            }
        }
    }
    None
}

fn visit<'a>(
    graph: &'a BTreeMap<String, Vec<String>>,
    color: &mut BTreeMap<&'a str, u8>,
    stack: &mut Vec<&'a str>,
    node: &'a str,
) -> Option<Vec<String>> {
    color.insert(node, 1);
    stack.push(node);
    if let Some(nexts) = graph.get(node) {
        for next in nexts {
            match color.get(next.as_str()).copied().unwrap_or(0) {
                1 => {
                    let start = stack.iter().position(|name| *name == next).unwrap_or(0);
                    let mut cycle: Vec<String> = stack[start..]
                        .iter()
                        .map(|name| (*name).to_string())
                        .collect();
                    cycle.push(next.clone());
                    return Some(cycle);
                }
                0 => {
                    if let Some(cycle) = visit(graph, color, stack, next) {
                        return Some(cycle);
                    }
                }
                _ => {}
            }
        }
    }
    stack.pop();
    color.insert(node, 2);
    None
}

fn check_root_dev_tools(ctx: &mut Ctx, members: &[Member]) {
    let Some(root) = members
        .iter()
        .find(|member| matches!(member.layer, Layer::Root))
    else {
        ctx.record(
            Path::new("Cargo.toml"),
            false,
            "configuration: workspace has no root package, so dev_tools cannot be checked",
        );
        return;
    };
    let Some(dev) = members
        .iter()
        .find(|member| matches!(member.layer, Layer::Dev))
    else {
        ctx.record(
            Path::new("Cargo.toml"),
            false,
            "root dev_tools dependency cannot be checked because crates/dev_tools is not a workspace member",
        );
        return;
    };
    let normals: Vec<&Dependency> = root
        .deps
        .iter()
        .filter(|dep| {
            dep.kind.is_none() && resolve_internal(dep, std::slice::from_ref(dev)).is_some()
        })
        .collect();
    if normals.len() != 1 {
        ctx.record(
            &root.manifest,
            false,
            "root package must declare one normal optional dependency on dev_tools_minigame, and it must not be a default feature",
        );
        return;
    }
    let dep = normals[0];
    let mut problems = Vec::new();
    if !dep.optional {
        problems.push("it is not optional".to_string());
    }
    let feature_name = dep.rename.as_deref().unwrap_or(dep.name.as_str());
    if default_enables(&root.features, feature_name) || default_enables(&root.features, &dev.name) {
        problems.push("it is enabled by default features".to_string());
    }
    ctx.record(
        &root.manifest,
        problems.is_empty(),
        format!(
            "root dev_tools dependency must stay optional and out of default features: {}",
            problems.join("; ")
        ),
    );
}

fn default_enables(features: &BTreeMap<String, Vec<String>>, dependency: &str) -> bool {
    let mut stack = vec!["default".to_string()];
    let mut seen = BTreeSet::new();
    while let Some(feature) = stack.pop() {
        if !seen.insert(feature.clone()) {
            continue;
        }
        let Some(items) = features.get(&feature) else {
            continue;
        };
        for item in items {
            if feature_item_enables(item, dependency) {
                return true;
            }
            if !item.starts_with("dep:") && !item.contains('/') {
                stack.push(item.clone());
            }
        }
    }
    false
}

fn feature_item_enables(item: &str, dependency: &str) -> bool {
    if item == dependency || item == format!("dep:{dependency}") {
        return true;
    }
    let Some((package, _)) = item.split_once('/') else {
        return false;
    };
    !package.ends_with('?') && package == dependency
}

struct Member {
    name: String,
    manifest: PathBuf,
    manifest_dir: PathBuf,
    layer: Layer,
    features: BTreeMap<String, Vec<String>>,
    deps: Vec<Dependency>,
}

enum Layer {
    Root,
    Core,
    Business(String),
    Dev,
}

#[derive(Deserialize)]
struct Metadata {
    packages: Vec<Package>,
    workspace_members: Vec<String>,
    workspace_root: PathBuf,
}

#[derive(Deserialize)]
struct Package {
    name: String,
    id: String,
    manifest_path: PathBuf,
    #[serde(default)]
    dependencies: Vec<Dependency>,
    #[serde(default)]
    features: BTreeMap<String, Vec<String>>,
}

#[derive(Clone, Deserialize)]
struct Dependency {
    name: String,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    optional: bool,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    path: Option<PathBuf>,
    #[serde(default)]
    rename: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct Fixture {
        root: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let scratch_root =
                Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.scratch/structure_fixtures");
            let root = scratch_root.join(format!(
                "bevy_sim2sim_structure_{}_{}_{}",
                std::process::id(),
                nanos,
                n
            ));
            fs::create_dir_all(&root).unwrap();
            Self { root }
        }

        fn write(&self, rel: &str, body: &str) {
            let path = self.root.join(rel);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(path, body).unwrap();
        }

        fn generate_lock(&self) {
            let output = Command::new("cargo")
                .args(["generate-lockfile", "--offline", "--manifest-path"])
                .arg(self.root.join("Cargo.toml"))
                .current_dir(&self.root)
                .output()
                .expect("run cargo generate-lockfile");
            assert!(
                output.status.success(),
                "generate-lockfile failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let name = self
                .root
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("");
            let scratch_root =
                Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.scratch/structure_fixtures");
            if self.root.starts_with(scratch_root) && name.starts_with("bevy_sim2sim_structure_") {
                let _ = fs::remove_dir_all(&self.root);
            }
        }
    }

    const GITIGNORE: &str = "\
target/
build/
logs/
.codegraph/
.scratch/
.claude/
.cursor/*
!.cursor/rules/
!.cursor/rules/**
";

    fn write_common_tree(fixture: &Fixture) {
        fixture.write("src/main.rs", "fn main() {}\n");
        fixture.write("crates/core/src/lib.rs", "");
        fixture.write("crates/core/Cargo.toml", CORE_MANIFEST);
        fixture.write("crates/dev_tools/src/lib.rs", "");
        fixture.write("crates/dev_tools/Cargo.toml", DEV_MANIFEST);
        fixture.write(
            "crates/dev_tools/python/src/bevy_microduck_tools/__init__.py",
            "",
        );
        fixture.write("crates/dev_tools/python/tests/test_flow.py", "");
        fixture.write(
            "crates/dev_tools/python/pyproject.toml",
            "[project]\nname = \"bevy-microduck-tools\"\n",
        );
        fixture.write("crates/dev_tools/python/build/keep.txt", "");
        fixture.write(".gitignore", GITIGNORE);
        fixture.write("README.md", "fixture\n");
        fixture.write("AGENTS.md", "See bevy_engineering_rules.md\n");
        fixture.write("CLAUDE.md", "See bevy_engineering_rules.md\n");
        fixture.write("docs/notes.md", "notes\n");
        fixture.write("bevy_engineering_rules.md", "rules\n");
        fixture.write("assets/game/shaders/ink_line.wgsl", "");
        fixture.write("assets/game/scenes/init.scn.ron", "");
        fixture.write("assets/game/scenes/loading.scn.ron", "");
        fixture.write("assets/game/scenes/frontend/main_menu.scn.ron", "");
        fixture.write("assets/game/dynamic_assets/settings/robot/tuning.ron", "");
        fixture.write("assets/game/dynamic_assets/settings/core/time_step.ron", "");
        fixture.write("assets/game/dynamic_assets/settings/project/window.ron", "");
        fixture.write("assets/game/dynamic_assets/game_data/balance.ron", "");
        fixture.write("assets/game/dynamic_assets/game_data/robot/policy.onnx", "");
        fixture.write("assets/game/dynamic_assets/game_data/robot/policy.json", "");
        fixture.write("assets/game/arts/entity/models/duck_body.glb", "");
        fixture.write("assets/game/instances/entity/duck_body.scn.ron", "");
    }

    const CORE_MANIFEST: &str = "\
[package]
name = \"common_minigame\"
version.workspace = true
edition.workspace = true
";

    const DEV_MANIFEST: &str = "\
[package]
name = \"dev_tools_minigame\"
version.workspace = true
edition.workspace = true

[dependencies]
common_minigame.workspace = true
";

    const ROBOT_MANIFEST: &str = "\
[package]
name = \"robot_minigame\"
version.workspace = true
edition.workspace = true

[dependencies]
common_minigame.workspace = true
";

    fn root_manifest(default_features: &str, with_station: bool) -> String {
        let station_member = if with_station {
            "    \"crates/modules/station\",\n"
        } else {
            ""
        };
        let station_dep = if with_station {
            "station_minigame = { path = \"crates/modules/station\" }\n"
        } else {
            ""
        };
        format!(
            "\
[workspace]
members = [
    \".\",
    \"crates/core\",
    \"crates/modules/robot\",
{station_member}    \"crates/dev_tools\",
]
resolver = \"3\"

[workspace.package]
version = \"0.1.0\"
edition = \"2024\"

[workspace.dependencies]
common_minigame = {{ path = \"crates/core\" }}
robot_minigame = {{ path = \"crates/modules/robot\" }}
{station_dep}dev_tools_minigame = {{ path = \"crates/dev_tools\" }}

[package]
name = \"fixture_game\"
version.workspace = true
edition.workspace = true

[features]
default = [{default_features}]
dev_tools = [\"dep:dev_tools_minigame\"]

[dependencies]
common_minigame.workspace = true
robot_minigame.workspace = true
dev_tools_minigame = {{ workspace = true, optional = true }}
"
        )
    }

    fn write_robot(fixture: &Fixture, manifest: &str) {
        fixture.write("crates/modules/robot/src/lib.rs", "");
        fixture.write("crates/modules/robot/Cargo.toml", manifest);
    }

    fn legal_fixture() -> Fixture {
        let fixture = Fixture::new();
        write_common_tree(&fixture);
        write_robot(&fixture, ROBOT_MANIFEST);
        fixture.write("Cargo.toml", &root_manifest("", false));
        fixture.generate_lock();
        fixture
    }

    fn failure_text(report: &StructureReport) -> String {
        report
            .failures
            .iter()
            .map(|failure| format!("{} {}", failure.path, failure.reason))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn legal_fixture_passes() {
        let fixture = legal_fixture();
        let report = check(&fixture.root);
        let text = failure_text(&report);
        assert!(report.passed, "{text}");
        assert!(report.check_count > 0);
        let json: serde_json::Value = serde_json::from_str(&to_json(&report)).unwrap();
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["scope"], SCOPE);
        assert_eq!(json["passed"], true);
        assert!(json["check_count"].as_u64().unwrap() > 0);
        assert!(json["failures"].as_array().unwrap().is_empty());
        let skipped = json["not_guaranteed"].as_array().unwrap();
        assert!(skipped.iter().any(|item| {
            item.as_str()
                .unwrap()
                .contains("do_not_depend_on_dev_assets")
        }));
        assert!(
            skipped
                .iter()
                .any(|item| item.as_str().unwrap().contains("immutable"))
        );
        assert!(!fixture.root.join("structure_report.json").exists());
    }

    #[test]
    fn illegal_asset_location_is_rejected() {
        let fixture = legal_fixture();
        fixture.write("assets/game/bad_sprite.png", "");
        let report = check(&fixture.root);
        let text = failure_text(&report);
        assert!(!report.passed, "{text}");
        assert!(
            report
                .failures
                .iter()
                .any(|failure| failure.path.contains("assets/game/bad_sprite.png")),
            "{text}"
        );
    }

    #[test]
    fn unknown_settings_module_is_rejected() {
        let fixture = legal_fixture();
        fixture.write(
            "assets/game/dynamic_assets/settings/missing_mod/tune.ron",
            "",
        );
        let report = check(&fixture.root);
        let text = failure_text(&report);
        assert!(!report.passed, "{text}");
        assert!(
            report
                .failures
                .iter()
                .any(|failure| failure.path.contains("settings/missing_mod")),
            "{text}"
        );
    }

    #[test]
    fn nested_agents_file_is_rejected() {
        let fixture = legal_fixture();
        fixture.write("crates/core/AGENTS.md", "nested\n");
        let report = check(&fixture.root);
        let text = failure_text(&report);
        assert!(!report.passed, "{text}");
        assert!(
            report.failures.iter().any(|failure| {
                failure.path.contains("crates/core/AGENTS.md") && failure.reason.contains("AI")
            }),
            "{text}"
        );
    }

    #[test]
    fn business_dependency_cycle_is_rejected() {
        let fixture = Fixture::new();
        write_common_tree(&fixture);
        fixture.write("crates/modules/robot/Cargo.toml", ROBOT_MANIFEST);
        fixture.write("crates/modules/robot/src/lib.rs", "");
        fixture.write(
            "crates/modules/station/Cargo.toml",
            "\
[package]
name = \"station_minigame\"
version.workspace = true
edition.workspace = true

[dependencies]
common_minigame.workspace = true
robot_minigame.workspace = true
",
        );
        fixture.write("crates/modules/station/src/lib.rs", "");
        fixture.write("Cargo.toml", &root_manifest("", true));
        fixture.generate_lock();
        // Cargo refuses to generate a lock for a cyclic package graph. Establish
        // the legal fixture first, then introduce the cycle for the checker.
        fixture.write(
            "crates/modules/robot/Cargo.toml",
            &format!("{ROBOT_MANIFEST}\nstation_minigame.workspace = true\n"),
        );
        let report = check(&fixture.root);
        let text = failure_text(&report);
        assert!(!report.passed, "{text}");
        assert!(
            report.failures.iter().any(|failure| {
                let reason = failure.reason.to_ascii_lowercase();
                reason.contains("cycle") || reason.contains("cyclic")
            }),
            "{text}"
        );
    }

    #[test]
    fn root_dev_tools_default_feature_is_rejected() {
        let fixture = Fixture::new();
        write_common_tree(&fixture);
        write_robot(&fixture, ROBOT_MANIFEST);
        fixture.write("Cargo.toml", &root_manifest("\"dev_tools\"", false));
        fixture.generate_lock();
        let report = check(&fixture.root);
        let text = failure_text(&report);
        assert!(!report.passed, "{text}");
        assert!(
            report.failures.iter().any(|failure| {
                failure.path == "Cargo.toml" && failure.reason.contains("default")
            }),
            "{text}"
        );
    }

    #[test]
    fn missing_manifest_is_configuration_failure() {
        let fixture = Fixture::new();
        let report = check(&fixture.root);
        let text = failure_text(&report);
        assert!(!report.passed, "{text}");
        assert!(report.check_count >= 1);
        assert!(
            report.failures.iter().any(|failure| {
                failure.reason.contains("configuration") && failure.reason.contains("Cargo.toml")
            }),
            "{text}"
        );
        assert!(
            report
                .not_guaranteed
                .iter()
                .any(|item| item.contains("immutable"))
        );
    }

    #[test]
    fn onnx_outside_module_directory_is_rejected() {
        let fixture = legal_fixture();
        fixture.write("assets/game/shaders/policy.onnx", "");
        let report = check(&fixture.root);
        let text = failure_text(&report);
        assert!(!report.passed, "{text}");
        assert!(
            report
                .failures
                .iter()
                .any(|failure| failure.path.contains("assets/game/shaders/policy.onnx")),
            "{text}"
        );
    }

    #[test]
    fn unlisted_asset_extension_and_orphan_sidecar_are_rejected() {
        let fixture = legal_fixture();
        fixture.write("assets/game/arts/entity/models/duck_mesh.xyz", "");
        fixture.write("assets/game/arts/entity/models/duck_mesh.bin", "");
        let report = check(&fixture.root);
        for suffix in ["duck_mesh.xyz", "duck_mesh.bin"] {
            assert!(
                report
                    .failures
                    .iter()
                    .any(|failure| failure.path.ends_with(suffix)),
                "{}",
                failure_text(&report)
            );
        }
    }

    #[test]
    fn documented_video_localization_and_referenced_model_sidecars_are_allowed() {
        let fixture = legal_fixture();
        fixture.write("assets/game/arts/environment/videos/station_intro.mp4", "");
        fixture.write("assets/i18n_assets/en_us.csv", "");
        fixture.write("assets/i18n_assets/zh_cn.po", "");
        fixture.write("assets/i18n_assets/ja_jp.ftl", "");
        fixture.write(
            "assets/game/arts/entity/models/duck_mesh.gltf",
            "{\"buffers\":[{\"uri\":\"duck_buffer.bin\"}]}\n",
        );
        fixture.write("assets/game/arts/entity/models/duck_buffer.bin", "");
        fixture.write(
            "assets/game/arts/entity/models/duck_frame.obj",
            "mtllib duck_material.mtl\n",
        );
        fixture.write("assets/game/arts/entity/models/duck_material.mtl", "");
        fixture.write("assets/game/arts/entity/models/duck_body.glb.meta", "");
        let report = check(&fixture.root);
        assert!(report.passed, "{}", failure_text(&report));
    }

    #[test]
    fn robot_artifact_exception_does_not_extend_to_other_modules() {
        let fixture = legal_fixture();
        fixture.write(
            "crates/modules/station/Cargo.toml",
            "[package]\nname = \"station_minigame\"\nversion.workspace = true\nedition.workspace = true\n[dependencies]\ncommon_minigame.workspace = true\n",
        );
        fixture.write("crates/modules/station/src/lib.rs", "");
        fixture.write("Cargo.toml", &root_manifest("", true));
        fixture.generate_lock();
        fixture.write(
            "assets/game/dynamic_assets/game_data/station/policy.onnx",
            "",
        );
        fixture.write(
            "assets/game/dynamic_assets/game_data/station/policy.json",
            "",
        );
        let report = check(&fixture.root);
        for suffix in ["station/policy.onnx", "station/policy.json"] {
            assert!(
                report
                    .failures
                    .iter()
                    .any(|failure| failure.path.ends_with(suffix)),
                "{}",
                failure_text(&report)
            );
        }
    }

    #[test]
    fn python_source_is_development_only_and_canonical_dunder_names_are_allowed() {
        let fixture = legal_fixture();
        fixture.write(
            "crates/dev_tools/python/src/bevy_microduck_tools/__main__.py",
            "",
        );
        fixture.write(
            "crates/dev_tools/python/src/bevy_microduck_tools/__pycache__/cache.pyc",
            "",
        );
        fixture.write("docs/用户提供的 Bevy 规划.md", "");
        assert!(check(&fixture.root).passed);
        fixture.write("crates/modules/robot/src/train.py", "");
        let report = check(&fixture.root);
        assert!(
            report
                .failures
                .iter()
                .any(|failure| failure.path.ends_with("robot/src/train.py")),
            "{}",
            failure_text(&report)
        );
    }

    #[test]
    fn weak_default_dependency_feature_does_not_activate_optional_dependency() {
        let features = BTreeMap::from([(
            "default".to_string(),
            vec!["dev_tools_minigame?/tracing".to_string()],
        )]);
        assert!(!default_enables(&features, "dev_tools_minigame"));
        let strong = BTreeMap::from([(
            "default".to_string(),
            vec!["dev_tools_minigame/tracing".to_string()],
        )]);
        assert!(default_enables(&strong, "dev_tools_minigame"));
    }

    #[test]
    fn cursor_rules_need_parent_and_contents_unignored() {
        let fixture = legal_fixture();
        fixture.write(".gitignore", &GITIGNORE.replace("!.cursor/rules/\n", ""));
        let report = check(&fixture.root);
        assert!(
            report
                .failures
                .iter()
                .any(|failure| failure.path == ".gitignore" && failure.reason.contains("cursor")),
            "{}",
            failure_text(&report)
        );
    }

    #[test]
    fn third_party_workspace_member_remains_outside_self_developed_layer_rules() {
        let fixture = legal_fixture();
        fixture.write(
            "third_party/vendor/Cargo.toml",
            "[package]\nname = \"vendor\"\nversion = \"1.0.0\"\nedition = \"2024\"\n",
        );
        fixture.write("third_party/vendor/src/lib.rs", "");
        let manifest = root_manifest("", false).replace(
            "    \"crates/dev_tools\",\n",
            "    \"crates/dev_tools\",\n    \"third_party/vendor\",\n",
        );
        fixture.write("Cargo.toml", &manifest);
        fixture.generate_lock();
        let before = fs::read(fixture.root.join("Cargo.lock")).unwrap();
        let report = check(&fixture.root);
        assert!(report.passed, "{}", failure_text(&report));
        assert_eq!(fs::read(fixture.root.join("Cargo.lock")).unwrap(), before);
    }
}

---
description: Bevy 工程的目录结构、资产落点、crate 分层与 Rust 命名规范。新建或移动任何文件、新建模块、改目录名、提交 Git 前必须遵守。
globs:
alwaysApply: true
---

# 工程结构与命名规范

本文件是本工程唯一的结构与命名规范，从第一次提交起生效，无存量豁免。违反任一条即失败，无「先报告后修」通道。
只约束自研内容，`third_party/**`、`assets/third_party/**`、`plugins/**` 与第三方 Cargo 依赖不受约束。
AI 配置只允许在工程根：`.cursor/` `.claude/` `.codex/` `.agents/` `AGENTS.md` `CLAUDE.md`。`src/**`、`crates/**`、`assets/**` 下禁新建这些目录，以及文件名命中 `*规范*` `*RULE*` `AGENTS.md` `CLAUDE.md` `CODING_STYLE*` 的文本文件。
跨工具规则入口固定为：Codex 使用 `AGENTS.md`，Claude 使用 `CLAUDE.md`，Cursor 使用 `.cursor/rules/**`。这些入口必须引用本文件，不得复制规范正文或另建冲突版本。

## 目录

```text
<project>/
  Cargo.toml                  # workspace 与游戏入口包
  Cargo.lock                  # 依赖锁文件
  src/main.rs                 # 游戏启动与模块装配入口
  crates/                     # 自研库，一级见「crate 分层」
    core/
    modules/<module>/
    dev_tools/
  assets/                     # 默认资产根
    game/                     # 自研资产
      arts/                   # 美术媒体源资产，领域 / 类型两级
      instances/              # 外置实体模板，按领域分
      shaders/                # 着色器代码
      scenes/                 # 外置顶层场景
      dynamic_assets/         # 按需加载的配置、数据与场景
    i18n_assets/              # 本地化资源
    third_party/              # 第三方资产
  plugins/                    # 原生库与平台库
  third_party/                # 第三方源码
```

自研入库代码与资产只允许落在树中对应目录及其子目录，树外新增 → 失败。`assets/` 根禁出现文件。Cargo、Git、AI 入口与格式配置保留各自固定位置；各 crate 的 `src/`、`tests/`、`examples/`、`benches/` 与构建文件按 Cargo 布局归属。

下列位置的子目录名必须完全等于表内字面，按需创建，不预建空目录：

| 位置 | 允许的目录名 |
|---|---|
| `assets/game/arts/` 与 `assets/game/instances/` | `app` `effects` `entity` `environment` `game_play` `ui` |
| `assets/game/scenes/` | `frontend` `levels` `dev` |
| `assets/game/dynamic_assets/` | `game_data` `game_play` `scenes` `settings` `shaders` |

禁用拼写：`gameplay` `art` `video` `sprites` `localization`。
目录名与 `<module>` 名入库后冻结，改名只走独立 commit。

## 文件落点

| 类型 | 唯一合法落点 |
|---|---|
| 运行时 `.rs` | `crates/core/src/**`、`crates/modules/<module>/src/**`；启动装配为 `src/main.rs` |
| 编辑器与开发工具 `.rs` | `crates/dev_tools/src/**` |
| `.wgsl` | `assets/game/shaders/**` |
| 外置实体模板 `.scn.ron` | `assets/game/instances/<domain>/**` |
| `.glb` `.gltf`；制作源 `.fbx` `.obj` | `assets/game/arts/<domain>/models/**` |
| `.png` `.jpg` `.tga` `.exr`；制作源 `.psd` | `assets/game/arts/<domain>/textures/**` |
| 外置材质描述 `.ron` | `assets/game/arts/<domain>/materials/**` |
| 外置动画描述 `.ron` | `assets/game/arts/<domain>/animations/**` |
| `.wav` `.mp3` `.ogg` | `assets/game/arts/<domain>/audio/**` |
| `.ttf` `.otf` | `assets/game/arts/<domain>/fonts/**` |
| 视频制作源 `.mp4` | `assets/game/arts/<domain>/videos/**` |
| 外置顶层场景 `.scn.ron` | `assets/game/scenes/init.scn.ron`，或 `assets/game/scenes/<dir>/**` |
| 配置与数据 `.ron` | `assets/game/dynamic_assets/settings/**`、`assets/game/dynamic_assets/game_data/**` |
| 本地化 `.ron` `.json` `.csv` `.po` `.ftl` | `assets/i18n_assets/**`，只使用项目已采用的格式 |

- 表中未列出的自研资产扩展名禁止新增；主资产必需的 `.bin`、`.mtl` 等附属文件与已使用的 `.meta` 加载设置随其所属文件维护。
- `assets/game/dynamic_assets/settings/<module>/` 的 `<module>` 必须已存在于 `crates/modules/`，或为 `core`、`project`；`project` 只放引擎级配置。与模块同批新建时，crate 声明与配置必须在同一 commit，禁止只建配置空壳。
- 明确作为按需加载内容的资产落 `assets/game/dynamic_assets/<分类>/**`；开发验证资产落 `assets/game/scenes/dev/**`。这两类专用落点优先于上表。
- glTF 内嵌的材质、动画与场景随模型保存；代码生成的实体模板留在所属业务 crate。只有外置内容适用对应资产落点。
- 运行时格式须由项目当前启用的加载器支持；普通 `.ron` 数据须有对应解析逻辑，制作源文件须转换后使用。

## crate 分层

内层禁止引用外层：公共层（业务无关）→ 业务层（只做 Runtime）→ 开发层（Editor 与 Dev 工具）。

| 层 | 目录 | crate 名 |
|---|---|---|
| 公共层 | `crates/core/` | `common_minigame` |
| 业务层 | `crates/modules/<module>/` | `<module>_minigame` |
| 开发层 | `crates/dev_tools/` | `dev_tools_minigame`，仅开发构建启用 |

- 上表每个目录必须在其根含对应 `Cargo.toml`，纳入根 workspace；自研库代码必须归属对应 crate。
- Rust 模块路径归属对应 crate：`common_minigame::*` / `<module>_minigame::*` / `dev_tools_minigame::*`。
- Cargo 依赖出现「公共层 → 业务层」「公共层 → 开发层」「业务层 → 开发层」→ 失败。
- 业务 crate 之间允许单向引用，须在所属 `Cargo.toml` 显式声明；出现环 → 失败。跨模块共用接口下沉到公共层。
- 根包负责游戏启动与模块装配；开发工具依赖为 optional，通过非默认 feature 及条件编译启用。

## 命名

自研目录名与文件主名只允许 `A-Z a-z 0-9 _`，禁空格、中文、连字符 `-`、其它符号；扩展名、`.scn.ron` 等复合后缀与 Cargo、工具固定文件名除外。

| 目标 | 写法 |
|---|---|
| 自研目录、文件主名、crate、Rust mod、`<module>`、`<domain>` | `all_lower_with_underscore`，正则 `^[a-z][a-z0-9]*(_[a-z0-9]+)*$` |
| `const`、`static` | `SCREAMING_SNAKE_CASE`，如 `MAX_SPEED` |
| `struct`、`enum`、枚举变体、`trait`、类型别名 | `UpperCamelCase`，如 `Transform`、`TimerMode::Once` |
| 函数（含 system）、方法、字段、局部变量、参数 | `snake_case`，如 `from_translation`、`elapsed_secs`；可见性由 Rust 关键字表达 |

- `<module>` 与 `<domain>` 用业务名词。
- 游戏启动入口为根包 `src/main.rs`；首个状态或场景由启动代码选择。
- 外置顶层场景主名正则 `^[a-z][a-z0-9]*(_[a-z0-9]+){1,2}$`（`_` 分隔 2～3 段）。单段例外为 `init`、`loading`；实体模板按普通文件命名。
- 禁止发明表外前缀。
- Rust 排版采用 Bevy 源码的 rustfmt 风格：4 空格缩进、LF、同名字段初始化简写；由根 `rustfmt.toml` 和 `cargo fmt` 统一，`.editorconfig` 与其保持一致。

## Rust 约束

- `match` 必须覆盖所有可能输入；已穷尽枚举变体时，不额外要求 `_` 分支。
- 配置资产加载完成后，其配置字段禁止被运行时业务代码赋值；需运行时改写的数据另设字段。
- 类型、函数、字段的文档注释用 `///`，模块说明用 `//!`，普通注释用 `//`；文档使用 Markdown，注释禁止只含数字。

## 构建

- 发布构建不启用开发工具 feature，正式资源包排除 `assets/game/scenes/dev/**`；其余正式场景允许同时打包。
- 正式代码与资产不得依赖开发资产；发布时按实际加载路径部署所需 `assets/**` 内容。

## Git

- 新增或移动资产时，资产、已有加载设置元数据与引用更新必须在同一 commit。
- 文本场景、配置、着色器与清单保留文本格式；二进制资产按实际内容处理。根与成员 `Cargo.toml`、根 `Cargo.lock` 随工程入库。
- `.gitignore` 必须含 `target/`、`build/`、`logs/`、`.codegraph/`、`.scratch/`、`.claude/`；忽略 `.cursor/` 中的本机内容，但保留 `.cursor/rules/**`。
- Agent 临时产出物禁止入库：review 报告、临时脚本、抓取的证据、日志。只允许共享工程根的 `AGENTS.md`、`CLAUDE.md` 与 `.cursor/rules/**` 作为跨工具规则入口；入口不得复制规范正文。

---
description: Godot 工程的目录结构、资产落点、模块分层与 GDScript 命名规范。新建或移动任何文件、新建模块、改目录名、提交 Git 前必须遵守。
globs:
alwaysApply: true
---

# 工程结构与命名规范

本文件是本工程唯一的结构与命名规范，从第一次提交起生效，无存量豁免。违反任一条即失败，无「先报告后修」通道。
只约束自研内容，`addons/` 下的第三方包与 `plugins/**` 不受约束；自研 `addons/dev_tools/` 仍受约束。
AI 配置只允许在工程根：`.cursor/` `.claude/` `.codex/` `.agents/` `AGENTS.md` `CLAUDE.md`。代码与资产目录下禁新建这些目录，以及文件名命中 `*规范*` `*RULE*` `AGENTS.md` `CLAUDE.md` `CODING_STYLE*` 的文本文件。
跨工具规则入口固定为：Codex 使用 `AGENTS.md`，Claude 使用 `CLAUDE.md`，Cursor 使用 `.cursor/rules/**`。这些入口必须引用本文件，不得复制规范正文或另建冲突版本。

## 目录

```text
<project>/                    # 工程根，包含 project.godot；res:// 指向这里
  project.godot               # 项目配置
  export_presets.cfg          # 导出配置
  game/                       # 自研代码与资产
    scripts/                  # GDScript，一级见「模块分层」
    arts/                     # 美术媒体源资产，领域 / 类型两级
    instances/                # 可复用实例场景，按领域分
    shaders/                  # 着色器代码
    scenes/                   # 顶层场景
    dynamic_assets/           # 运行时按路径加载的资产
  i18n_assets/                # 本地化资源
  addons/                     # 第三方插件与资源包
    dev_tools/                # 自研编辑器与开发工具
  plugins/                    # 原生扩展与平台库
```

自研入库代码与资产只允许落在树中对应目录及其子目录，树外新增 → 失败。`game/` 根禁出现文件。引擎、Git、AI 入口与格式配置保留各自固定位置。

下列位置的子目录名必须完全等于表内字面，按需创建，不预建空目录：

| 位置 | 允许的目录名 |
|---|---|
| `game/arts/` 与 `game/instances/` | `app` `effects` `entity` `environment` `game_play` `ui` |
| `game/scenes/` | `frontend` `levels` `dev` |
| `game/dynamic_assets/` | `game_data` `game_play` `scenes` `settings` `shaders` |

禁用拼写：`gameplay` `art` `video` `sprites` `localization`。
目录名与 `<module>` 名入库后冻结，改名只走独立 commit。

## 文件落点

| 类型 | 唯一合法落点 |
|---|---|
| 运行时 `.gd` | `game/scripts/core/**`、`game/scripts/modules/<module>/**` |
| 编辑器与开发工具 `.gd` | `addons/dev_tools/**` |
| `.gdshader` `.gdshaderinc` `.glsl` | `game/shaders/**` |
| 可复用实例 `.tscn` | `game/instances/<domain>/**` |
| `.glb` `.gltf` `.fbx` `.obj` | `game/arts/<domain>/models/**` |
| `.png` `.jpg` `.tga` `.exr` `.svg`；制作源 `.psd` | `game/arts/<domain>/textures/**` |
| 材质 `.tres` | `game/arts/<domain>/materials/**` |
| 动画及动画库 `.tres` | `game/arts/<domain>/animations/**` |
| `.wav` `.mp3` `.ogg` | `game/arts/<domain>/audio/**` |
| `.ttf` `.otf` | `game/arts/<domain>/fonts/**` |
| 运行时视频 `.ogv`；制作源 `.mp4` | `game/arts/<domain>/videos/**` |
| 顶层场景 `.tscn` | `game/scenes/init.tscn`，或 `game/scenes/<dir>/**` |
| 配置与数据 `.tres` | `game/dynamic_assets/settings/**`、`game/dynamic_assets/game_data/**` |
| 本地化 `.csv` `.po` `.pot` | `i18n_assets/**` |

- 表中未列出的自研资产扩展名禁止新增；引擎生成的 `.uid`、`.import`、项目与插件配置，以及主资产必需的 `.bin`、`.mtl` 等附属文件随其所属文件维护。
- `game/dynamic_assets/settings/<module>/` 的 `<module>` 必须已存在于 `game/scripts/modules/`，或为 `core`、`project`；`project` 只放引擎级配置。与模块同批新建时，模块代码必须与配置在同一 commit，禁止只建配置空壳。
- 明确作为动态加载内容的资产落 `game/dynamic_assets/<分类>/**`；开发验证资产落 `game/scenes/dev/**`。这两类专用落点优先于上表。
- `.tscn` 内嵌的子资源随场景保存；外置资源按实际用途落点。模型自带的缓冲、材质和纹理保留有效相对引用。

## 模块分层

内层禁止引用外层：公共层（业务无关）→ 业务层（只做 Runtime）→ 开发层（Editor 与 Dev 工具）。

| 层 | 目录 | 边界 |
|---|---|---|
| 公共层 | `game/scripts/core/` | 公共脚本 |
| 业务层 | `game/scripts/modules/<module>/` | 每个目录一个业务模块 |
| 开发层 | `addons/dev_tools/` | 编辑器与开发工具，排除出发布内容 |

- 自研 `.gd` 必须归属上表某层；模块依赖按脚本类型与资源引用检查。
- 出现「公共层 → 业务层」「公共层 → 开发层」「业务层 → 开发层」引用 → 失败。
- 业务模块之间允许单向引用，通过脚本类型或资源路径显式表达；出现环 → 失败。跨模块共用接口下沉到公共层。

## 命名

自研目录名与文件主名只允许 `A-Z a-z 0-9 _`，禁空格、中文、连字符 `-`、其它符号；扩展名与引擎、工具固定文件名除外。

| 目标 | 写法 |
|---|---|
| 自研目录、文件主名、`<module>`、`<domain>` | `all_lower_with_underscore`，正则 `^[a-z][a-z0-9]*(_[a-z0-9]+)*$` |
| 类、声明的 `class_name`、枚举类型、节点名 | `UpperCamelCase` |
| 常量、枚举成员 | `CONSTANT_CASE`，如 `MAX_SPEED`；绑定脚本类型的常量使用类型名 |
| 对外方法、属性、普通变量、参数 | `snake_case` |
| 内部方法、内部字段 | `_snake_case` |
| `@export` 配置字段 | `snake_case`，如 `max_speed` |
| 信号 | `snake_case`，用已发生的事件命名，如 `health_changed` |
| 引擎回调 | 保留引擎要求的名称 |

- `<module>` 与 `<domain>` 用业务名词。
- 启动场景 `game/scenes/init.tscn` 必须设为 `project.godot` 的 `application/run/main_scene`。
- 其余顶层场景主名正则 `^[a-z][a-z0-9]*(_[a-z0-9]+){1,2}$`（`_` 分隔 2～3 段）。单段例外只有 `loading`；实例场景按普通文件命名。
- 禁止发明表外前缀。
- GDScript 排版采用 Godot 官方脚本风格：UTF-8、LF、Tab 缩进，函数之间空两行，每行一条语句；工程根 `.editorconfig` 与此保持一致。
- 脚本按声明、信号、枚举、常量、字段、方法组织；字段按静态、导出、普通、`@onready` 排列；实例方法中引擎回调在前，对外方法其次，内部方法在后。

## GDScript 约束

- `@export` 配置字段禁止被运行时代码赋值；需运行时改写的另设非导出字段。
- 类、函数、字段的文档注释用 `##`，普通注释用 `#`；类说明位于 `class_name` / `extends` 之后；注释禁止只含数字。

## 构建

- 发布导出排除 `game/scenes/dev/**` 与 `addons/dev_tools/**`；其余正式场景允许同时纳入导出。
- 正式入口场景及 Autoload 的静态依赖闭包中，禁止出现开发内容与 `game/dynamic_assets/**`；动态资产通过运行时加载，并显式纳入导出范围。静态依赖包含场景、资源引用和 `preload` 引用。

## Git

- 新增或移动资产时，源文件、已有 `.uid`、`.import` 与引用更新必须在同一 commit；移动保留原 UID。
- 场景与外置资源使用 `.tscn`、`.tres` 文本格式；文本文件与二进制资产按实际内容处理。`project.godot`、`export_presets.cfg` 及需保留的 `.uid`、`.import` 随工程入库。
- `.gitignore` 必须含 `.godot/`、生成的 `*.translation`、`build/`、`logs/`、`.codegraph/`、`.scratch/`、`.claude/`；忽略 `.cursor/` 中的本机内容，但保留 `.cursor/rules/**`。
- Agent 临时产出物禁止入库：review 报告、临时脚本、抓取的证据、日志。只允许共享工程根的 `AGENTS.md`、`CLAUDE.md` 与 `.cursor/rules/**` 作为跨工具规则入口；入口不得复制规范正文。

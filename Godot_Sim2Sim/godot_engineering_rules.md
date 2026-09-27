# Godot 工程结构与命名规则

本文件是 Godot 自研结构的维护源。维护根是 `Sai_Lab/Godot_Sim2Sim`，资源根是其中的 `godot`（`res://`）。Python、研究文档、原生源码与 Unity 不套用 Godot 资源落点。跨仓库职责以 [统一约定](docs/repository_ownership.md) 为准。

## 适用与迁移

- 新增和搬移自研文件按本规则；既有文件原位修复保持原布局，不要求顺带迁移、改名或格式化。
- 已提交存量精确路径及基线范围见 `docs/directory_inventory.json`；不是整个旧目录的永久豁免。新增旧布局兼容文件登记精确路径、原因到 `compatibility_additions`。
- 第三方、外部交付和部署副本通过 `source_records` 登记精确文件、来源仓库、版本和 SHA256；保留来源名称及许可，不靠任意目录获得豁免。
- 实验基线、冻结源码及证据原位保留。当前 Tick 与运行相关迁移暂缓，正常业务授权不因存量目录问题失效。
- AI 入口在维护根使用 AGENTS.md、CLAUDE.md、.cursor/rules；入口引用本文和目录导航，不复制正文。Godot 代码/资产目录内不另建规则入口。

## 新内容目录

按需创建，不预建空目录；`game` 根不放文件。

| 内容 | 落点（相对 res://） |
|---|---|
| 公共 GDScript | `game/scripts/core/**` |
| 业务 GDScript | `game/scripts/modules/<module>/**` |
| 编辑器/开发验证脚本及工具资源 | `addons/dev_tools/**` |
| 可复用场景 | `game/instances/<domain>/**` |
| 正式顶层场景 | `game/scenes/frontend/**`、`game/scenes/levels/**`；可选 `game/scenes/init.tscn` |
| 开发验证场景、fixture、资产 | `game/scenes/dev/**`；验证脚本仍在 dev_tools |
| shader 代码 | `game/shaders/**`；按路径动态加载的 shader 可在 dynamic_assets/shaders |
| 运行数据/配置/模型与策略部署 | `game/dynamic_assets/{game_data,game_play,scenes,settings,shaders}/**` |
| 必需的美术导出/第三方资产副本 | `game/arts/<domain>/{models,textures,materials,animations,audio,fonts,videos}/**` |
| 本地化 | `i18n_assets/**` |
| 第三方插件/原生平台库 | `addons/<package>/**`、`plugins/**`；登记导入来源 |

`<domain>` 允许 app、effects、entity、environment、game_play、ui；`<module>` 使用业务名词。制作源、模型编辑和贴图制作去 Sai_Art，Lab 只保留有来源的部署副本。JSON 配置、绑定映射和部署数据有正式落点，不因格式不是 tres 而禁止。

## 文件类型与名称

- 自研新目录及文件主名用 snake_case。类/枚举类型使用 UpperCamelCase，常量用 CONSTANT_CASE，方法/属性/信号用 snake_case；内部字段和方法按既有公开接口约定使用前导下划线。
- 引擎固定名称、上游/导出模型名、关节/刚体/骨架/节点路径和资产附属文件按交付约定保留；特殊新增文件通过精确来源或兼容登记维护，不批量改名。
- `.gd` 按 Runtime/开发职责归属；`@tool` 本身不表示纯开发内容，地形碰撞等运行代码仍属于业务模块。
- 支持 gdshader/gdshaderinc/glsl；tscn/tres；glb/gltf/fbx/obj 与 bin/mtl 等附属文件；png/jpg/jpeg/webp/tga/exr/hdr/svg；wav/mp3/ogg；ttf/otf/ttc；ogv/mp4/webm；json/onnx/csv/yaml/yml/txt；本地化 csv/po/pot。类型按实际用途进入上表相应位置。
- 模型、字体及交付资产所需 LICENSE/NOTICE/COPYING 等许可文档随资产维护。已有 uid/import 随主资产移动并保留值；本轮不修改忽略规则或生成/补齐 UID。
- 新 GDScript 使用 UTF-8、LF、Tab 缩进、每行一条语句和官方声明顺序；既有脚本不因目录整理而全量格式化。配置与运行状态按职责区分，保留初始化/交付接口。

## 依赖与现有流程

依赖方向为开发工具 → 业务模块 → 业务无关公共层。公共层不引用业务或开发工具，Runtime 不引用纯开发内容；模块之间显式引用并避免环。业务相关共享能力保持明确业务归属，不全部塞入 core。

保留现有 main_scene 及 feature override，不强制改成 init。已有启动、组装、动态加载、物理/策略协议和资源查找行为原样保留。新增静态/动态资源必须核查依赖与导出范围；已有导出过滤、UID 忽略和 shader 合并作为单独验证后的迁移项。

## 检查与提交

从维护根运行 `python3 scripts/check_godot_structure.py`；`--base` 可选已提交比较点。工具只做新增/搬移文件的路径、类型、命名和精确登记检查，已有内容修改/布局及历史副本分叉不失败。依赖、语义、导出和物理行为仍按任务完成适用验证。

源文件、已有 uid/import 与调用者更新在同批变更中审阅。持久目录导航、来源清单及结果说明放 docs；临时报告、抓取日志和临时脚本使用忽略的本机位置。检查不接入现有启动、训练、导出、hook 或 CI。

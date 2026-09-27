# Godot_Sim2Sim 目录导航

新增文件、目录迁移或跨仓库任务先读 [职责约定](repository_ownership.md)，Godot 新文件的具体落点见 [工程规则](../godot_engineering_rules.md)。下面描述当前布局，不表示已完成运行迁移。

## 当前维护根

`Sai_Lab/Godot_Sim2Sim` 是维护根，`godot` 是资源根（`res://`）。Python、原生源码与研究文档不套用 Godot 资产规则。Unity_Sim2Sim 保持其自己的规则和目录。

| 当前位置 | 用途 / 维护方式 |
|---|---|
| `src/mjcf2godot` | 模型转换工具；生成场景和绑定数据，不手改输出替代转换源 |
| `src/sim2sim` | Python 集成、参考后端、部署与研究工具；原位维护 |
| `scripts` / `tests` | 外层准备、验证、工具入口 / Python 验证；森林统一入口也在 scripts |
| `configs` / `robots` / `policies` | 运行配置、机器人来源、外置策略输入；遵守各自 provenance 与忽略规则 |
| `native` | 原生推理源码和依赖记录；构建产品与依赖缓存继续忽略 |
| `integrations/leviathan` | 载具后端与游戏覆盖层；存在替换基础 hub 脚本的同名文件，暂保留 |
| `integrations/leviathan003` | 历史可导入交付副本；部分脚本与 Art support 相同，按清单来源处理 |
| `docs` / `artifacts` | 持久说明、结果、冻结实验与证据；不是可批量删除的临时目录 |
| 根目录 README、CONTEXT、HANDOFF、研究/复现说明 | 导航、领域术语、复现约定；持久文档原位保留 |
| 根目录启动脚本、pyproject、uv.lock、版本/许可文件 | 现有入口、环境和许可；本轮保持执行行为 |
| `results`、`dist`、运行/虚拟环境/构建缓存 | 生成或本机产物；不能成为新增能力的维护源 |

## 当前 Godot 内容

| 当前位置 | 用途 / 本轮状态 |
|---|---|
| 根场景、physics_server、play_hud、project/export 配置 | 物理、参考服务、展示和多入口配置；保持原路径 |
| `standalone` / `sai` | 本地策略部署与控制；Tick/协议相关，暂缓迁移 |
| `hub` | 场景切换、机器人接入、UI、回放/探测；与集成覆盖层分别登记 |
| `atelier` / `science_station` | 程序化场景、碰撞、镜头和视觉资源混合；暂缓拆分 |
| `visuals` | 机器人样式、shader 与角色映射；保留当前接口及分叉 |
| `scenes` / `research` / `spikes` / `tests` | 可运行场景、研究场景、物理探针和验证脚本；已有调用者，保留入口 |
| `scripts` | 森林准备实现原位保留；外层新入口只转发 |
| `native` | GDExtension 描述与本机生成库；保持加载路径 |
| `generated` / `runtime_assets` / vendor 链接 | 转换、部署或第三方输入；使用对应生成/准备源 |

新自研 Godot 文件按 `game` 布局按需创建，不预建空目录。新 `.gd` 按职责进入 core、业务模块或 dev_tools；艺术制作源去 Art，Lab 的必要资产副本记录版本。旧目录的正常修复不触发整批迁移。

## 实际组装与副本

维修站准备复制 Godot 树，组合 Sai 上游适配与策略包，写出 `results/workshop-hub/runtime`；生成目录不是另一份源码。Leviathan 组装先复制基础目录再应用集成覆盖，覆盖层的 hub/sai/scene_grab 不能当作基础文件的等价副本。

Sainiverse 启动调用 Art `run.py`。Art 交付机器人快照，再根据开关读取 Lab 已准备机器人，叠加 Sai 60 Hz 与样式兼容代码。实际比较的提交、文件哈希和原快照/覆盖差异见 [清单](directory_inventory.json)，数字只描述该次提交，不代表实时工作目录状态。

## 手动检查

```bash
# 从 Godot_Sim2Sim 执行；只读，不启动引擎或准备资源
python3 scripts/check_godot_structure.py
# 可选跨仓库比较；两个根都由调用者明确选择
python3 scripts/check_godot_structure.py --peer-root /path/to/Sai_Art
python3 scripts/check_godot_structure.py --repo-root /path/to/Sai_Art
```

默认与清单中的已提交基线比较。后续单个任务可用 `--base <任务起点提交>` 仅检查该任务新增/搬移的路径，避免重复检查基线之后已经交付的文件。相对于比较点的已有文件内容修改跳过；旧布局、冻结证据、历史副本分叉和缺失可选 peer 不成为失败项。新增/搬移路径违规返回非零。检查未接入启动、导出、训练、hook 或 CI；未实现语义/物理正确性验证。

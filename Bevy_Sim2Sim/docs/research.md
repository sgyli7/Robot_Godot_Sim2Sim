# Bevy Sim2Sim 调研与验证记录

更新日期：2026-09-28。本文保存 MicroDuck 开发需要的来源、契约、检查摘要及待验证风险。实施顺序见 [计划](implementation_plan.md)，接续环境见 [交接入口](handoff.md)。

证据分为「源码/契约核查」「本会话局部实测」「项目待验证」。局部结果不能自动升级为 Bevy 机器人闭环或平台验收通过。

## 依赖与版本选择

| 项目 | 首个实现配置 | 核查范围 |
|---|---|---|
| Rust / Bevy | Rust 1.95，Bevy `=0.19.1` | 官方清单/MSRV；尚未本项目编译 |
| Bevy Rapier | `bevy_rapier3d =0.36.0`，官方 f32 绑定 | 发布清单锁定 `rapier3d =0.35.0-glamx0.2` |
| ORT Rust | `ort =2.0.0-rc.13`，关闭默认特性，启用 `std/load-dynamic/api-28` | RC13 API、环境初始化、Tensor/Session 源码 |
| ONNX Runtime | 固定 1.29，CPU、intra/inter=1、SEQUENTIAL、ALL 图优化 | Python 原生库及 API28 ABI 局部实测；Rust 集成待验证 |

### Bevy/Rapier

参考 [Bevy 0.19.1 清单](https://github.com/bevyengine/bevy/blob/v0.19.1/Cargo.toml)、[发布绑定清单](https://docs.rs/crate/bevy_rapier3d/0.36.0/source/Cargo.toml)、[发布源码包](https://static.crates.io/crates/bevy_rapier3d/bevy_rapier3d-0.36.0.crate) 和 [绑定所用 Rapier 源码包](https://static.crates.io/crates/rapier3d/rapier3d-0.35.0-glamx0.2.crate)。

以发布包为依赖事实来源：同名 monorepo 标签中的 Bevy 绑定使用 git 主线，其 Rapier 版本也可能不同。新版 MJCF loader 的类型不能直接混入当前绑定；初版从有效定义显式转换机器人。

物理状态以 Rapier 为真值；通过 Bevy 生命周期同步。版本被选为实现起点，仍需小型关节和完整 MicroDuck 验证其实际精度与稳定性。

### ONNX

RC13 默认 feature 包含自动下载和较旧 API 选择，因此使用显式 feature 及运行库路径。初始化过程在创建 Session 前调用 `init_from(path)` 并提交环境，禁用遥测；库版本、文件和加载失败可诊断。

参考 [ORT RC13 清单](https://github.com/pykeio/ort/blob/v2.0.0-rc.13/Cargo.toml)、[环境初始化](https://github.com/pykeio/ort/blob/v2.0.0-rc.13/src/environment.rs)、[Session 选项](https://github.com/pykeio/ort/blob/v2.0.0-rc.13/src/session/builder/impl_options.rs)、[Tensor 创建](https://github.com/pykeio/ort/blob/v2.0.0-rc.13/src/value/impl_tensor/create.rs)。

缓存模型输入输出名称，检查 float32、批次和契约维度。固定输入缓冲区可使用借用 Tensor，输出须在复用缓冲区前复制到有明确所有权的动作。控制边界同步推理，保持无额外异步策略延迟。

目标平台首先使用 CPU provider。逐平台核对 ONNX Runtime 1.29 官方分发的实际架构；若缺少对应发行包，用固定版本源码构建 CPU 库并记录构建选项。CUDA/CoreML/DirectML 的依赖支持不代表本项目已通过，也不作为初版 CPU 基线的替代。

## MicroDuck 来源索引

下表路径相对于仓库；源文件继续保留在原工程，导入到 Bevy 时记录来源、哈希和转换版本。

| 内容 | 来源 |
|---|---|
| 有效策略 profile、权重与 manifest | [microduck_sprint_v1](../../Godot_Sim2Sim/src/sim2sim/assets/microduck_sprint_v1/) |
| 冻结 ONNX 样本 | [self_test.json](../../Godot_Sim2Sim/src/sim2sim/assets/microduck_sprint_v1/self_test.json) |
| 与 profile 哈希匹配的 spec | [robot_spec.json](../../Godot_Sim2Sim/docs/sprint_input_diagnosis_20260914/evidence/robot_spec.json) |
| 当前控制参数 | [microduck_ball_stand_fix.json](../../Godot_Sim2Sim/robots/microduck_ball_stand_fix.json) |
| 观测及动作转换 | [policy_contract.gd](../../Godot_Sim2Sim/godot/standalone/policy_contract.gd) |
| 外层控制及电流限制 | [driver.gd](../../Godot_Sim2Sim/godot/standalone/driver.gd) |
| 关节组装、力矩及观测器 | [physics_server.gd](../../Godot_Sim2Sim/godot/physics_server.gd) |
| 物理定义转换与碰撞校准 | [convert.py](../../Godot_Sim2Sim/src/mjcf2godot/convert.py) |
| 原始腿式 XML 与 mesh | [Legged](../../Unity_Sim2Sim/TuanjieProject/Assets/MicroDuck/Generated/MuJoCo/Legged/) |
| 模型与原有风格入口 | [microduck visuals](../../Godot_Sim2Sim/godot/visuals/microduck/) |
| 单关节场景 | [pd_hinge.gd](../../Godot_Sim2Sim/godot/spikes/pd_hinge.gd)、[pd_hinge.tscn](../../Godot_Sim2Sim/godot/spikes/pd_hinge.tscn) |
| 现有 Godot 运行说明 | [README](../../Godot_Sim2Sim/README.md)、[REPRODUCING](../../Godot_Sim2Sim/REPRODUCING.md) |

### 固定哈希与时序

| 文件 | SHA256 |
|---|---|
| Walk_Godot.onnx | `27ebbf83d63e5e59f125ceb158414cf7d2574990676d73c75b9c9870bcefed6e` |
| Sprint_Godot.onnx | `a402791e30e8ec9ffba5a288a2ebc377e43e7b6925b49986a3c523c686665770` |
| profile 指定 robot_spec | `c0553f0fed271ced99c607b050d091bc9c26d17116b1cb3d0dca8721c04f3ecc` |

有效 profile 为物理 200 Hz、decimation 4。其 provenance 有后续验收记录，manifest 仍含较早实验标签；结合来源时序读取，保留权重及原始契约。历史 provenance 是已有实验记录，本次不将其计为当前机器重跑通过。

### 关节顺序

14 个策略执行器按原定义依次为：

```text
left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle,
neck_pitch, head_pitch, head_yaw, head_roll,
right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle
```

腿式原始定义包含 world 共 16 bodies、15 joints、20 nv、14 nu；轮式共 20 bodies、19 joints、24 nv、14 nu。INIT/STAND/SIT/FOLD 初态来自原始 XML；已有转换 manifest 的 keyframes 不一定完整。

## 物理与观测契约

### 坐标、质量和碰撞

使用右手正交变换 `B(x,y,z)=(x,z,-y)`。已有 Unity 转换产物采用不同、带反射的基，导入时不能把其坐标直接作为中立定义。

刚体以惯性质心坐标系表示，保留主惯量与惯性姿态；完整 mass properties 配合零 collider density，防止重复加质量。关节轴及两端锚点分别落在局部惯性坐标系，保留零位与限位。策略机体姿态通过惯性变换还原，线速度沿用当前质心观测语义。

主动关节 armature `.0018`；轮式被动关节 `.0001`。当前 Godot/Jolt 的处理是子体对角惯量补偿及条件数限制，Rapier 原生关节 armature 与其动力学不同，二者不叠加。通过单关节公式与完整机器人回归检查迁移影响。

有效脚底是校准后的轮廓挤出体：STAND 最低点附近 2 mm 带、最多 16 边、8 mm 厚、零 margin。轮式有效轮胎碰撞采用 PCA 半径球体。保留实际生成结果，变更形状必须单独回归。

当前 Godot 摩擦从每刚体 1.0 起取几何最大值，rough=false 时组合取 Min；Rapier 默认摩擦及 Average 组合不同。显式配置接触属性，保留原碰撞组及父/祖父碰撞排除，无休眠。参考 [Godot PhysicsMaterial](https://docs.godotengine.org/en/stable/classes/class_physicsmaterial.html)。

### 力矩

原 spec 的 ±.96 Nm 还受到运行电流 `1.75 A` 收紧，实际限幅约 ±.6405236195572268 Nm。先限制 PD，再减 `.053*qd` 和 `.0048*tanh(qd/.05)`，父子反向施力。

Rapier 的用户 torque 会持久存在且累加，逐步清零后聚合；避免与 ExternalForce 上传重复。通过多体雅可比把外部刚体力矩投影到自由度。原生 damping 与显式被动力避免重复计算。

### 观测及动作

61 维依次为 gyro3、projected gravity3、q-home14、qd14、last_action14、command13。q 按父子机体旋转与关节静止姿态求 twist；qd 使用 wrapped 角差/步长后做 5 ms EMA。gyro 使用世界坐标下的惯性姿态最短弧差分，先做时间常数 5 ms 的 EMA，再转为机体局部；5 ms 步长下 alpha 约 `.6321205588`。顺序见 physics_server.gd 的 `_update_kinematic_vel` 与状态输出路径。

坐标中间计算使用 f64，模型输入 f32。q 先转换 f32 再减 home；动作乘法、加法分别按 f32 舍入，不合并为 FMA。last_action 是上一轮原始 ONNX 输出；技能切换按契约保留，重置归零。外层控制和指令平滑先于观测，HUD 平滑仅用于显示。

直接改用 Rapier 原生速度会改变观测器，因此初版保留有效观测器，原生速度另记为诊断量。

## 组合风险与处理顺序

| 风险 | 已核查事实 | 项目验证与处理 |
|---|---|---|
| 发布版本与源码类型不同 | 绑定发布包与仓库主线、新版 MJCF loader 依赖不同 | 锁定发布依赖，显式转换机器人，检查 Cargo.lock |
| 小尺度接触容差 | Rapier 默认米尺度误差/预测距离远大于当前脚底容差 | 5 ms、米单位、`.0002 m` 起始容差、零预测、禁接触缓存复用，测试实际脚底并冻结通过参数 |
| 多体数值稳定性 | 绑定所用版本早于后续近奇异 Coriolis 能量保护；部分接触修复已存在 | 实测无重力能量和限位；复现后移植已定位的最小兼容修复 |
| 限位与 motor 的数值路径 | 相关有效质量/CFM 路径有版本差异 | 初版外部 PD 力矩驱动，保留限位专项回归 |
| armature 初始化 | 关节树自由度位置由 assembly_id 决定 | SyncBackend 后获取新句柄，按实际自由度写入并验证 |
| 重置/关节变化 | 后端关节生命周期与 ECS 延迟命令有关 | 重建关节图，等待同步，初始化非零 q、速度及历史；检查 20 次重置数量 |
| 时钟和掉帧 | Bevy 固定频率默认值和 Virtual 最大 delta 会影响步数 | 显式 200 Hz，记录步数、欠账、clipped 时间，检测暂停与停顿 |
| 本地推理库 | ABI 通过不等于 Rust Session、Tensor 生命周期通过 | 检查实际库、初始化、冻结样本、真实观测与错误路径 |
| 图形平台 | Metal 上存在 0.19 材质/预处理问题报告 | 首次图形探针包含实际 robot mesh 和实例化；复现后验证 CPU 预处理路径并记录测量配置 |
| 资产就绪 | Asset 就绪与 shader pipeline 就绪不同 | 启动屏障及超时/失败状态；从其他 cwd 启动 |
| 无窗口验证 | MinimalPlugins 不自动包含所需 Asset/Transform 能力 | 显式装配必要插件，复用固定步；离屏图形单独检验 |

Rapier 调查参考 [0.35.3 多体实现](https://github.com/dimforge/rapier/blob/v0.35.3/src/dynamics/joint/multibody_joint/multibody.rs)、[变更记录](https://github.com/dimforge/rapier/blob/v0.35.3/CHANGELOG.md)；用于定位后续修复，项目实际依赖仍以发布绑定为准。上游 #372/#810 有回归测试，不能仅凭问题标题认定仍未修复；接触修复 #968/#970 已在所查发布源码中。

单关节基准可取 `Izz=.0001`、armature `.0018`、tau `.001`、dt `.005`，首步角速度增量约 `.00263158`；未含 armature 时约 `.05`。该数值用于识别 armature 是否真正进入自由度质量矩阵。

重置非零 q 时，先完成根姿态与正向运动学，按现有关节位移应用目标差值，再更新刚体；清空广义和刚体速度/用户力。完成初始化前不插入隐式物理 settle 步。实际公共 API 路径须随已锁定源码和编译验证。

## 视觉与地图来源

原有机器人 shader 位于上方来源索引的 visuals 目录：`style.gd`、`enamel.gdshader`、`ink.gdshader`、`lens.gdshader`。

分段光照参考原式 `.12 + .48*smoothstep(.03,.13,NdotL) + .26*smoothstep(.55,.65,NdotL)`，迁移时校准 Godot 与 Bevy 灯光单位及曝光。显式设置 Tonemapping None 对照当前线性模式。

排线使用原局部坐标、主法线及 fwidth 衰减；已有批次网格包含烘焙变换，拆分 primitive 时要保留参考空间。保留角度加权 42° 法线、像素宽度描边和距离衰减；机器人描边由 style 覆盖为 `.60 px`，排线强度覆盖为 `.03`，须连同配置读取。镜片为不描边的涂层材质，原金属度/粗糙度约 `.35/.19`，需要对照反射及光圈细节。

科学站地面负数 mod/hash 的语义须在 WGSL 中保持；光照和抗锯齿参数以实际启动入口为准，现有 hub MSAA4 与独立 atelier profile MSAA8 不可混用。

后续地图入口： [科学站](../../Godot_Sim2Sim/docs/science-station.md)、[维修站](../../Godot_Sim2Sim/docs/workshop-hub.md)、[视觉说明](../../Godot_Sim2Sim/docs/visual_style.md)。这些是导航，验收需对应当前代码的实际截图与录像。

Bevy 参考：[ExtendedMaterial 示例](https://bevy.org/examples/shaders/extended-material/)、[0.18→0.19 迁移指南](https://bevy.org/learn/migration-guides/0-18-to-0-19/)、[Metal 问题 #25595](https://github.com/bevyengine/bevy/issues/25595)。后者有 M3 Pro 报告，本机 M5 和 DGX 尚未执行 Bevy 图形复现。

## 已执行检查

以下为本会话 2026-09-27 至 28 的局部检查摘要，交接前再次执行 ORT 1.29 数值复核。原始终端输出未作为项目原始证据包入库；接手后重新产生的日志和完整回放放 `.scratch/`。

| 检查 | 结果 | 能证明的范围 |
|---|---|---|
| MicroDuck Python ONNX CPU | ORT 1.28 与 1.29 配置检查；Walk/Sprint 各 136 样本，共 272 通过，重复输出一致 | 推理数值；每模型 128 随机+8 边界，real_count=0 |
| 最大数值误差 | Walk 约 `1.1772e-6`，Sprint 约 `1.2815e-6`，小于 `1e-5` | 已用冻结样本；不是实际物理观测覆盖 |
| 原生 ABI | ARM64 ORT 1.29 dylib 加载，GetApi(28) 返回非空 | 原生 API 可取；不是 Rust 绑定通过 |
| 原生依赖 | 所查 macOS 库依赖 Apple 系统库及 libc++ | 当前检查的文件；其他平台待检查 |
| 原始机器人定义 | MuJoCo 3.10 编译腿式/轮式，短步推进有限 | 先在内存修正 XML 编码声明和 meshdir，不能认定原文件直接可载入 |
| Godot/Jolt hinge | 200 Hz，目标 ±.4 各 200 步，q 约 ±.29595；零目标接近零，正常退出 | 单关节 TCP 力矩探针；完整机器人未运行通过 |
| 现有纯检查 | 本会话曾执行 8 项局部静态/契约检查 | 摘要未保留完整命令列表，接手需按具体测试重新验证 |
| 完整 Godot 启动 | 缺生成 robot.tscn、robot_spec 和所需原生运行资产 | 完整基线尚未准备成功 |
| Bevy/Rust | 无 Cargo 实现，未发现本机工具链 | 无构建、运行或图形通过结论 |

仅推理微基准曾得到 Walk/Sprint P50 约 14.125/21.625 µs（ORT1.28、当前 Mac）。它不包含物理、渲染或完整控制，也不是 Godot/Bevy 性能对比。

### 可复核的现有入口

在仓库根检查权重和 spec 哈希：

```bash
shasum -a 256 Godot_Sim2Sim/src/sim2sim/assets/microduck_sprint_v1/Walk_Godot.onnx
shasum -a 256 Godot_Sim2Sim/src/sim2sim/assets/microduck_sprint_v1/Sprint_Godot.onnx
shasum -a 256 Godot_Sim2Sim/docs/sprint_input_diagnosis_20260914/evidence/robot_spec.json
```

Linux 可用 `sha256sum` 替换 `shasum -a 256`。这些只验证文件身份。

已有单关节检查入口为 `Godot_Sim2Sim/src/sim2sim/spikes.py` 的 S3。在已准备 Godot/MuJoCo 依赖的 Linux 环境可从该子工程运行：

```bash
uv run sim2sim-spikes --only s3
```

S3 使用不同目标和步数并包含 MuJoCo 比较；上表 TCP 探针没有证明 S3 全套通过。Mac 的既有 spawn helper 含 Linux CPU affinity 调用，需核对平台适配后使用。

以下命令已在交接前从仓库根运行，复核 ORT 1.29 的模型身份与 272 条输出，依赖由 uv 环境提供。容差采用现有 [native_policy_probe.gd](../../Godot_Sim2Sim/godot/tests/native_policy_probe.gd) 的 `1e-5`；self_test JSON 提供模型、观测和期望动作：

```bash
uv run --no-project --python 3.12 --with numpy --with onnxruntime==1.29.0 python - <<'PY'
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

root = Path("Godot_Sim2Sim/src/sim2sim/assets/microduck_sprint_v1")
fixtures = json.loads((root / "self_test.json").read_text())
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

for case in fixtures["cases"]:
    model = root / Path(case["path"]).name
    assert hashlib.sha256(model.read_bytes()).hexdigest() == case["sha256"]
    actor = ort.InferenceSession(
        str(model), options, providers=["CPUExecutionProvider"]
    )
    outputs = []
    for observation in case["observations"]:
        action = actor.run(None, {
            actor.get_inputs()[0].name: np.asarray(observation, np.float32)[None]
        })[0][0]
        assert action.shape == (14,) and np.isfinite(action).all()
        outputs.append(action)
    error = float(np.max(np.abs(
        np.asarray(outputs) - np.asarray(case["actions"], np.float32)
    )))
    assert error < 1e-5
    print({
        "skill": case["skill"], "samples": len(outputs),
        "real_count": case["real_count"], "max_abs_error": error,
        "runtime": ort.__version__, "passed": True
    })
PY
```

新增真实物理观测轨迹必须独立保存并检查。Rust 复核实现属于计划第一阶段。

## 平台与效率证据要求

| 平台 | 依赖线索 | 本项目状态 |
|---|---|---|
| Apple Silicon macOS | Bevy/Metal、ORT ARM64 原生库有对应能力；本机完成 ORT 局部检查 | Bevy 构建、图形、闭环、长期运行未验证 |
| DGX Spark Linux ARM64 | CPU 推理与 Vulkan 为首个验证路径，需检查发行库、驱动及环境 | 真机构建、启动、完整回放及稳定性未验证 |
| Windows | 需核验实际架构、MSVC、DLL 与图形后端 | 本项目未验证 |

记录依赖支持、真实先行案例和本项目实测三个层次。现有上游案例不替代本项目运行。

效率比较使用 [实施计划中的测量流程](implementation_plan.md#后续验证与效率测量)。CPU 进程时间除以墙钟时间可大于 100%；RSS 各平台单位需明确转换，统一内存 GPU 数据单独标注。采样期间将追踪缓冲后再输出，报告所用 feature、资源包、图形配置、物理参数及重复运行情况。

开发效率以实际任务耗时记录；目前没有可用的跨引擎开发速度提升结论。

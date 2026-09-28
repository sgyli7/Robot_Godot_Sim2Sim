# MuJoCo / Rapier 逐关节受力对照：BAM 入口仍关闭

更新日期：2026-09-28。对两种冻结的完整 MicroDuck 本体，各做四种同状态、同 `1/60 s` 的单步对照。Rapier 侧由 `paired_force_probe` 从唯一 `SimulationWorld` 读取真实求解观测；MuJoCo 侧由 `compare_paired_force.py` 加载原始编译 `.mjb`，使用相同源 `qpos`/`qvel`、零控制力矩，读取 **`mj_step` 此步求解后、再次 `mj_forward` 前** 的 `qfrc_bias` 和 `qfrc_constraint`。本组全部驱动 DOF 的步前 `mj_forward` 与该步求解外载差为零。脚本逐路核对 14 个 actuator 的源 joint/DOF、输入文件 SHA、自由根速度坐标映射和实际步长。此处没有策略推理、科学站地面、BAM 控制或技能验收。

| 本体 / 初始状态 | 最大驱动 DOF 外载差 (N·m) | 最大步末关节位置差 (rad) | 最大步末速度差 (rad/s) |
| --- | ---: | ---: | ---: |
| 腿式 / 原位零速 | `3.795e-7` | `1.230e-8` | `3.118e-8` |
| 腿式 / 原位非零关节速度 | `3.795e-7` | `2.138e-8` | `1.301e-7` |
| 腿式 / 原位非零根体及关节速度 | `3.902e-6` | `2.044e-8` | `2.374e-7` |
| 腿式 / `head_roll` 超上限 `0.02 rad` | **`0.035109`** | **`0.015000`** | **`0.299982`** |
| 轮式 / 原位零速 | `3.795e-7` | `1.230e-8` | `1.309e-7` |
| 轮式 / 原位非零关节速度 | `3.795e-7` | `2.381e-8` | `2.075e-7` |
| 轮式 / 原位非零根体及关节速度 | `3.902e-6` | `1.874e-8` | `2.566e-7` |
| 轮式 / `head_roll` 超上限 `0.02 rad` | **`0.035111`** | **`0.015000`** | **`0.299982`** |

原位非零根体与关节速度时，Rapier 的被驱动关节惯性投影最大约 `7.15e-5 N·m`，因此动态组并非仅重复静态重力对照。前三组只支持**无约束冲量的本体重力/惯性项**在这些冻结状态下数值接近；不能据此打开 `JointFeedbackFrame.previous_solve_load`。

超限组给出了明确反例。腿式源 DOF 14、轮式源 DOF 16 的 MuJoCo 上限约 `0.436332 rad`，输入为 `0.456332 rad`。MuJoCo 在该步产生约 `−0.03511 N·m` 的 `qfrc_constraint`，步末仍为约 `0.451333 rad`、`−0.299982 rad/s`。Rapier 步末直接到约 `0.436333 rad`、速度为零；当前原生观测中的**最终**通用关节冲量为零，故候选外载漏掉这一约束力。数值不同不只是观测字段缺失，**关节限位动力学本身也不同**。在源限位可能触发的轨迹上，把此候选直接交给 BAM 会造成错误的负载历史。

特性 `sim2sim_limit_row_trace` 下，在新的直连 `SimulationWorld` 中采集同一条 `head_roll` 限位行三个阶段，直接证实非零中间冲量为何没有出现在最终观测。两种本体的行 RHS、坐标及速度一致；以下冲量分别列出：

| 求解检查点 | 坐标 (rad) | 广义速度 (rad/s) | 行 RHS (rad/s) | 腿式 / 轮式原始行冲量 (N·m·s) |
| --- | ---: | ---: | ---: | ---: |
| 带位置偏差求解后 | `0.456332326` | `−1.199977636` | `1.199977636` | `0.002340687` / `0.002340817` |
| 位置积分后 | `0.436332703` | `−1.199977636` | `1.199977636` | `0.002340687` / `0.002340817` |
| 无偏差稳定化求解后 | `0.436332703` | `0` | `0` | `0` / `0` |

这条上限行的原始正冲量在广义坐标上方向为负；最终无偏差阶段在已经完成位置积分后将其累计值抵消。原有 Rapier 报告去掉新增追踪字段后，与未开启追踪的 JSON 逐项相同。阶段记录证明了**Rapier 内部的观测时机**，并没有证明其限位力等价于 MuJoCo；尤其不能将带偏差阶段的 `0.00234 N·m·s` 简单除以时间步长替代 MuJoCo 的 `qfrc_constraint`。两份本机追踪报告为 `.scratch/paired_force_v1/leg_limit_trace_v1.json`（SHA256 `70f01c6777eae6d5dee8e0095c6cae3372eabeaec79415ae9e2c848acc6d27a1`）和 `roller_limit_trace_v1.json`（SHA256 `2af89f18526da1994cbb2297857c8b60bb1368d49cd552f5826c67fc7ef5aa27`）。

两种本体的 MuJoCo 单步均无自接触，当前直连 Rapier 探针各有 5 对活跃自接触候选，法向冲量全零。这次力对照不使用已审计的 `SourceCollisionWorld` 过滤器，也不消耗它的有限积分预算；这 5 对仍证明直连世界的源自碰撞规则未等价。科学站地面、道具、摩擦和碰撞冲量均没有得到对照。

## 复现和后续门禁

Rust 探针在 `crates/dev_tools/src/bin/paired_force_probe.rs`，MuJoCo 对照脚本在 `crates/dev_tools/python/scripts/compare_paired_force.py`。本机原始 JSON 位于 `.scratch/paired_force_v1/`，含每个状态的 `*_rapier.json` 和 `*_comparison.json`；八组报告哈希汇总为 `manifest.json`，其 SHA256 为 `3507bdf2b52ae4fa614902ccb8337d7d70617cf366e39441bdc095f29233a3d7`。探针必须用 `sim2sim_observation` feature 编译，命令格式为：

```text
cargo run --locked -p dev_tools_minigame --features sim2sim_observation \
  --bin paired_force_probe -- \
  COMPILED_DEFINITION DEFINITION_SHA QPOS QPOS_SHA [QVEL QVEL_SHA] REPORT.json

python crates/dev_tools/python/scripts/compare_paired_force.py \
  SOURCE.mjb MJB_SHA QPOS QPOS_SHA REPORT.json COMPARISON.json \
  [--qvel QVEL --qvel-sha256 QVEL_SHA]

# 只为定向限位行调查开启，仍使用同一个 paired_force_probe 参数：
cargo run --locked -p dev_tools_minigame --features sim2sim_limit_row_trace \
  --bin paired_force_probe -- \
  COMPILED_DEFINITION DEFINITION_SHA QPOS QPOS_SHA REPORT_WITH_TRACE.json
```

这批证据绑定腿式 `.mjb` SHA `832e1f08a1e328d8498be565874691348b89bc13041ff0d9eebf3f06c6bdc47c`、轮式 `.mjb` SHA `ea54c7c2b0aa8fbc3e9a0e3a431214650cbd4bdbd934d6707bb0aa52ac475191`，以及两个带原碰撞凸包的编译定义 SHA `eb7a276a5199c757181c60681de54a7eb3e490431929d1fb043f8fb25a964299` / `a4524cb0b63a64fd88f3fff1375c8fac375d2ec88254545409ca6b6eaca768fd`。各 `qpos`/`qvel` SHA 写在报告中，运行时逐一核对。

下一门禁是确定与源相容的限位法则，并以同状态、同控制的受力和步末运动对照验收；不能只把位置钳制量伪装为 MuJoCo `qfrc_constraint`。随后要在同一个真实世界中核对源自碰撞过滤、站体/地面和道具接触，覆盖摩擦、限位、接触的负载历史，再接入上一步实际施加的 actuator torque 与 BAM。生产 `previous_solve_load` 继续明确为 `Unavailable`，九个旧 ONNX 的视频仍仅为 P-only、60/60 零样本诊断。

非零执行器输入与 oracle 限位力的后续同态对照见 [同力矩单步对照](paired_actuation_comparison.md)：14 路实际广义力矩投影接近，但本来就在运动的状态仍有明确的步末速度差，故生产 BAM 门禁不变。

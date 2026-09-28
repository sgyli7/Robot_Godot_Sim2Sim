# 两条关节软限位同时激活：内部 PGS 收敛诊断

更新日期：2026-09-28。在[单行](source_limit_row_probe.md)、[非零力矩](source_limit_actuated_probe.md)及[连续八步](source_limit_multistep_probe.md)实验后，此诊断首次让**两个**源软限位在同一步同时产生非零力。腿式和轮式都从 `right_hip_pitch`、`right_knee` 上限各超出 `0.02 rad`、其余关节原位、全零速度／执行器力矩开始。源 MuJoCo 3.10.0 的真实 `mj_forward` 与 `mj_step` 均观察到恰好两条限位行、零接触；两条约束行的有效逆质量非对角耦合比约为腿式 `0.0665`、轮式 `0.0776`。目标只使用默认关闭、显式选择并限定两份 SHA 定义的源限位行诊断。

Rapier `num_solver_iterations` 始终为 **1**，故每案例仍只有一次真实 `1/60 s` 积分，未加入时间子步。实验只改变该步求解中的 `num_internal_pgs_iterations` 为 1、2、4、8、16；生产设置未改变。`crates/dev_tools/python/scripts/compare_coupled_source_limits.py` 从 SHA 锁定的 `.mjb` 与 qpos 构造两个超限关节，逐案例运行原生诊断和已有 `compare_paired_force.py`，核对两侧实际求解力、步末角度／角速度、行时序和所有接触冲量。

| 本体 | 内部 PGS 轮数 | 两关节最大限位力差 | 两关节最大步末角度差 | 两关节最大步末角速度差 |
| --- | ---: | ---: | ---: | ---: |
| 腿式 | 1 | `8.338e−6 N·m` | `2.831e−4 rad` | `6.747e−5 rad/s` |
| 腿式 | 2 | `3.532e−8 N·m` | `1.145e−6 rad` | `2.827e−7 rad/s` |
| 腿式 | 4 | `3.532e−8 N·m` | `4.822e−8 rad` | `3.125e−7 rad/s` |
| 轮式 | 1 | `1.324e−5 N·m` | `3.265e−4 rad` | `1.060e−4 rad/s` |
| 轮式 | 2 | `3.809e−8 N·m` | `1.742e−6 rad` | `2.859e−7 rad/s` |
| 轮式 | 4 | `4.285e−8 N·m` | `6.914e−8 rad` | `2.898e−7 rad/s` |

8 或 16 轮没有进一步降低表中的最大角度差。四轮在两种本体均通过预设门槛：两行力差 `<1e−6 N·m`、角度差 `<1e−6 rad`、角速度差 `<1e−5 rad/s`。1 轮两侧角度误差均 `>1e−4 rad`，作为明确负控。针对腿式 `right_hip_pitch`，1 轮时 Rapier 带偏差求解后速度为 `−0.31764638 rad/s`，位置已经按此速度积分；随后无偏差阶段将最终速度修正到 `−0.30072656 rad/s`，却不能回改已经积分的位置。四轮时积分前后速度都约 `−0.30065879 rad/s`，接近 MuJoCo 步末 `−0.30065909 rad/s`。这说明本例的主要位置偏差来自**两个耦合行在位置积分前迭代不足**，而不是增加物理时间频率的需要。

源侧两案例均零接触。目标侧各有 4 对机器人自碰撞窄相候选，但每一案例所有接触法向／切向观测冲量均为零；仍不能声称接触拓扑等价。脚本将每份模型、源 `.mjb`、初始 qpos/qvel、目标原始报告和源/目标比较报告绑定 SHA。腿／轮初始 qpos SHA 分别为 `60caab636a662633040674f5bc434ad3620af75eb9593ccc4ba1b1ce43e805d0`、`cb73871ba3a0c57f4dd172102ae4c80141452b3a0e723293c65e74a4d7f43a08`；汇总 `.scratch/source_limit_two_row_v1/coupled_comparison.json` 连续两次重跑逐字节相同，SHA 为 `e638a4a43d258c952a6bfb89ba0c189ee579cdafe94360a5f3075c375710e370`。旧单行 42 组在扩展显式诊断范围后也保持原汇总 SHA `4668e6c8…`。

复现命令（Python 环境须提供 MuJoCo 3.10.0）：

```sh
cargo build --locked -p dev_tools_minigame --features sim2sim_source_limit_probe,sim2sim_plain_mass_probe --bin paired_force_probe
python crates/dev_tools/python/scripts/compare_coupled_source_limits.py
```

这仅覆盖两个冻结关节、一次无真实接触冲量的零力矩求解。它支持把内部 PGS 轮数作为后续完整本体物理验收变量，但没有直接改变生产配置，也不证明其他多行组合、接触／摩擦耦合、长程轨迹、BAM 上一步负载或九项技能。

# Homie_v2 的 50 Hz native motor 候选

`G1RunnerConfig.actuator_backend` 显式选择 `external_explicit_pd` 或
`native_force_based`。旧 JSON 省略该字段时保留显式 PD。
两者使用同一 Homie_v2 输入、动作时序和源关节参数；候选仅改变执行器离散后端。
不把原来的 200/50 Hz 源配置宣称为已经等价迁移。

native 候选由同一个后台 G1 owner 为 43 个关节设置 ForceBased motor 的
位置目标、原 stiffness/damping/effort caps；仅调用空外力矩列表的既有积分
边界。每 Tick 一次 20 ms 积分。`G1Step.applied_torques` 是外施 PD 努力，
native 时为 43 个零，不能当作真实 motor 努力；实际 motor 行证据来自独立
默认关闭的 `sim2sim_motor_row_trace` 仪器。

修复单位旋转维护后的冷对照最大 150 Tick：native 完成 3 秒、最大水平漂移
0.1106023 m，未观察到跌倒或源关节限位越界；显式 PD 漂移 0.3765884 m，
第 58 Tick 出现手指限位越界。两者首个 action/targets 逐位一致。该结果只
支持进一步验证，不是长期站立、任务、实时运行或接触材料等价的声明。

有限 body 诊断包含 stand、turn、walk、stop、reach，最多 650 Tick；只输入
公开的诊断身体命令，不构成视觉任务闭环。环境固定模型、ORT 和源码身份后：

```bash
export G1_ACTUATOR_BACKEND=native_force_based
export G1_T0_OUTPUT=/absolute/path/to/new/body_diagnostic.json
cargo test --locked --offline -p simulation_minigame --lib \
  g1::runner::tests::real_homie_t0_diagnostic -- \
  --ignored --exact --test-threads=1 --nocapture
```

还需设置 `G1_MODEL_DIR`、`G1_DEFINITION_SHA256`、`G1_ORT`、
`G1_ORT_SHA256` 和完整 `G1_CODE_COMMIT`。使用固定集成源码路径和有界外部
执行器记录二进制及输入 hash，不共享兄弟工作树的 Cargo target。

独立的 `g1::runner::tests::real_homie_released_stand30_diagnostic` 固定为
1500 次零导航命令，采用发布版 T2 的初始根位置经地板平移后
`[0,.18,.795]`、机器人及地板摩擦 .5、上肢零目标和 .75 m 高度目标。
它只用平地，省略源场景背景及任务物件，不等于 T2 场景稳定性。
共享 USD 导出的本体身份仍独立保留原导出提交；发布版 USD 和控制器
已经通过原始字节及源代码对照，没有重标旧导出为新源场景。
保持现有 Homie 原生候选的每 Tick 一次 20 ms 积分、1 次 solver 时间步、
1 次非积分 PGS、1 次 CCD 上限及零附加时间步，不启用预测限位或额外 PGS。
输出记录完整逐 Tick 自状态、命令、实际网络输出及关节目标，并检查
全部原配关节限位（数值容差 1e-4 rad），记录倾斜、水平漂移及实际计数。
测试通过只意味着生成了真实诊断结果；跌倒或漂移不会被计为任务成功。

设置同上的模型、源码身份环境变量以及全新 `G1_T0_OUTPUT` 后：

```bash
G1_ACTUATOR_BACKEND=native_force_based \
cargo test --locked --offline -p simulation_minigame --lib \
  g1::runner::tests::real_homie_released_stand30_diagnostic -- \
  --ignored --exact --test-threads=1 --nocapture
```

同名 `.jsonl` 保存逐 Tick 记录；禁止覆盖或重复使用旧结果路径。

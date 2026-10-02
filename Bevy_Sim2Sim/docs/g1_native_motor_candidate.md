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

首个该配置的 1500 Tick 诊断完成，积分、目标更新及 Homie 推理各 1500 次；
最低 upright 为 .996896，未跌倒。零导航下最大水平漂移 .888487 m，
因此没有通过静止站立资格。手指最大原限位外误差 .000189262 rad，
高于本次记录的 1e-4 容差，但未超过既有源端诊断的 .001 rad 容差。
这些容差分别保留，不能把小幅数值越界误报成大范围关节失控。
该诊断运行 3.07 秒墙钟、30 秒仿真，不能用来宣称真实 1× 时间。

下一项有界对照为 `unitree_g1_released_body_stand.py`，只在已经核对的
Arena .2.1 / SDK 6.0 容器中运行一次原配 Homie 平地站立 1500 个控制。
初始根位置、关节姿态、地面 .5 材料及身体命令匹配上述原生配置，
保持原配 PhysX/IdealPD 200/50，记录源端漂移及实际编译后的 mass/COM/inertia。
它不是 T2 源场景或原生任务验收，不进行频率、增益和训练扫圈。

该源端对照实际完成 1500 个 Homie 控制 / 6000 次原配积分。初始 43 个
关节状态与原生逐值相同；源端最大水平漂移 .122958 m、末端水平速度
.000681916 m/s，原生分别为 .888487 m 与 .147537 m/s。源端也有初始
漂移，不能凭其未跌倒宣称严格静止资格。53 个编译后身体与冻结本体
数据的最大质量误差 2.384e-7 kg、质心误差零、身体帧惯量误差
3.841e-9 kg*m²，包含四个质量默认值；这未测量 Rapier 的有效组装惯量。

只允许一次显式 `G1_HOMIE_STAND_PGS_DIAGNOSTIC=4` 对照：同一忽略测试
把 1 次非积分 PGS 改为 4 次，其他值拒绝。它增加同一积分边界内部
的代数求解次数，不改变 dt、控制周期、增益或时间积分次数。只在测试
内部修改，不改变公开 runner 默认；输出明确标记开发诊断。

该单变量 4 PGS 对照同样完成 1500 次积分、控制更新和网络调用；初始
43 关节/根状态及第一帧真实网络输出与 1 PGS 基线相同。最大漂移降至
.0891116 m，末端水平速度 .00133388 m/s，最低 upright .999086，
没有超过固定 1e-4 rad 容差的原限位越界。它说明本次原生漂移对代数
求解收敛敏感，尚不能将作用归因于某个 motor 或接触行，也不是公开默认、
源任务场景、携物走路或 1× 时间资格。该站立迭代次数对照结束，不继续
试 8/16 次；下一步是原配 T2 背景、物件和动作的同一世界迁移。

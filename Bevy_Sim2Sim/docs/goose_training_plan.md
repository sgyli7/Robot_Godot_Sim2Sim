# Goose V0.1｜Bevy Sim2Sim 长期训练计划

版本：2026-10-02。实施分支：`codex/goose50_training`。

当前仍在 M0-S／M0-T，GPU／优化器使用为零。按用户确认，主线是 **统一 MJCF → 原生 MuJoCo 基线 → mjlab／RSL-RL 基础训练 → Rapier／Bevy 迁移验收**。成熟栈的原生模型接线已完成；当前[模型整理检查点](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/native_geometry_checkpoint.json)关闭了收益不足的包含删除和崩溃的整脚分解，修正了源／目标凸包导出的真实差异，尚未晋升任务本体。下一项按连通分量复用成熟几何工具，冻结可复核的任务 MJCF。Bevy 性能子 agent 的缓冲改动经[独立对照](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/independent_performance_cycle02_review.json)将相同短程序的物理步均值约从 **157 ms 降到 53 ms**，尚未达到持续 50 Hz；第三周期在修正的导出上重新测量。已有自定义 CPU 接触／离散分支已收口为历史诊断，不再作为默认训练内核扩展。所有自研代码按 [唯一工程规范](/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/bevy_engineering_rules.md)落点，实验产物只保存在项目备份目录。

## 1. 目标与当前基线

本任务交付三条主线，优先完成移动和跌倒恢复，再完成拾物运输闭环。

| 主线 | 最终行为 |
| --- | --- |
| 移动 | 站稳，前后、横移、转向、启停；通过室内接缝、低障碍、缓坡和小台阶 |
| 跌倒恢复 | 前倒、后倒、左右侧倒、真实扰动跌倒中自主起身，站稳后继续移动 |
| 拾物 | 用户在 Bevy 选定物体，接近、弯腰用嘴叼起、站直、携带、放到指定位置 |

- 物理、策略、力矩更新均为 **50 Hz**；每 Tick 一次推理、一次 20 ms 积分，无隐藏时间子步。离线可快于实时，模拟步长不变。
- 首版载荷 **100、200、300 g**，目标位姿由 Bevy 提供，不纳入自主视觉识别。
- 不训练独立坐站、踢球、轮滑、特殊翻滚；姿态变化可作为三条主线的内部动作。
- 共享 DGX Spark，先小实验筛选，每批长训最多两小时。

输入是 `/home/ethan/Projects/Sai_Rotbots/artifacts/Goose_V0.1/goose_460_training_checkpoint_20261002.zip`。16,314 个清单文件哈希全部匹配；源装配 460 件、33 个运行刚体、18 主动轴、14 被动坐标、10.430762603 kg、65 观测／18 动作。原证据只有 0.2 秒冒烟和 RSL 初始化，没有优化器更新或合格策略。

权威为包内 `training_checkpoint_contract.json` 和 `stage_one.py`，不使用历史 53／10 的 `control.py`、`runtime.py`、`rsl.py`。零碰撞 training_reference 只用于参数/FK核对，不能作为任务训练本体。

MD 经验落实为：先核本体、实际驱动和观测时效，再调奖励；冷启动与热切换分别测；恢复必须包含继续移动；拾物按实体结果验收；训练完成和候选晋升分开。参见 [MD 冻结交付状态](/home/ethan/ProjectBackups/2026-10-01/Sai_Lab/microduck_player_package_015/DELIVERY_STATUS.md)。

## 2. 技术路线与接口

### 50 Hz 本体

原包物理 0.1 ms、力矩 5 ms、策略 20 ms，必须建立新合同，不覆写原模型身份。

1. `goose_460_full50_v1` 保留 33 体和全部软垫参数。源 MuJoCo 隐式积分，目标 Rapier 原生滑动关节和隐式弹簧。限速、PD、名义颈部重力前馈、延迟、2 秒力矩热代理和 350 W 正机械功率预算均按 20 ms 更新。
2. full50 失败后，测试 `goose_460_condensed50_v1`：12 个微小脚垫质量用并轴定理合入脚体，保留每脚六个接触区域、位置、面积、支撑轮廓、1.5 mm 行程及载荷—压缩曲线，以柔顺接触表达。
3. 凝聚版本保持总质量、COM、完整惯量、接触合力与力矩映射；静态、低频、冲击误差分别报告，不宣称高频瞬态等价。
4. 两份均失败时记录 M0 未通过，禁止长训；继续围绕已定位的物理阻断推进，不以增加训练预算掩盖问题。

物理诊断另有 `goose_460_full50_be_v2`：保留原 33 体及物理 K/C/A，在一次原生约束求解内使用 `M+hC+h²K` 的线性弹簧后向欧拉离散、双端预测限位和真实嘴销。它是显式版本化的 **CPU 实验**，非线性速度力仍显式计算，尚未替代本计划的源训练积分路线。整机接触、跨引擎、GPU 兼容和物理准入未通过前，不授予训练资格；原 v1 模型及失败收据保持冻结。

另有 `goose_460_full50_discrete_v3`：由冻结 full50 v1 派生，仅选择 MuJoCo **3.13.0** 的原生 `discrete` 并关闭异常自动重置，保留原 33 体、物理参数和 65／18 驱动接口。开发运行时显式锁定项目 **Python 3.12**，默认旧运行时拒绝其合同。此 CPU 候选的脚垫行程和接触检查失败；尚无 Rapier、Warp 或训练资格。新求解器的数值含义与既有 CPU 实验分别记录，不替换批准的训练链路。

Rapier 的 `num_solver_iterations=1`、每体 `additional_solver_iterations=0`、`max_ccd_substeps=1`。只能调整内部 PGS 收敛轮数（预定 4/8/16/32），不能增加实际积分子步。弹簧采用 ForceBased SI 单位，碰撞体不重复增加质量。

嘴部采用实际四杆闭合约束，电机只驱动 `beak_input_rotor`，与 head_roll 产生相反反力，约束将力传给 jaw/coupler。禁止逐 Tick 写 qpos、FK 搬动物件或焊接附着辅助抓取。

任务用碰撞代理保留脚、嘴、壳及运动干涉关键表面、来源和过滤映射；不填实空心结构，不扩大自碰撞排除范围凑成功。原邻接过滤、显式嘴销配合排除与碰撞 masks 均可追溯。

### 训练与目标接入

**MuJoCo 50 Hz → mjlab／MuJoCo Warp + RSL-RL PPO → Rapier 零样本评估 → 必要的有界目标微调 → Bevy CPU ONNX 独立验收。**

- 当前执行顺序先完成源端模型和成熟链路。复用 MicroDuck 的 `MjSpec`／`EntityCfg`、任务配置／注册、原生 VecEnv、上游 PPO、复载与导出组织方式；Goose 只增加必要的模型、65／18 控制适配及任务项，不另写训练器、通用环境框架或求解器。MicroDuck 的 BAM、轴序、身体尺度和奖励数值不直接移植。
- 统一 MJCF 从已核验交付派生：显式质量与完整惯量、关节坐标、18 轴驱动、嘴闭环及来源映射只有一份权威；视觉细节与任务碰撞代理分开。原约 15,715 凸块不能直接作为成熟批量训练模型。代理保留空腔及脚／嘴／外壳关键接触面，经几何与载荷短检查再采用。
- 源端优先原生 `mj_step` 和上游已支持的积分器／约束。v2–v6 的自定义 CPU 修正仅用于解释历史失败；未经独立兼容检查不进入 mjlab／Warp，不为保留某个局部方案继续改内核。
- 明确设定 `timestep=0.02`、`decimation=1`，力矩与策略同频；不能沿用常见的 5 ms×4 配置。依赖首先对齐 MicroDuck 实际采用的 mjlab 1.3.0、Warp 1.12.0 与 RSL-RL 5.0.1，再锁定解析得到的兼容 MuJoCo／Warp 后端；3.13 CPU 诊断环境不等于训练依赖。
- 物理准入分为 **M0-S 源端** 和 **M0-T 目标端**。源端物理及 GPU 小批量链路通过后，可先做有界源端站立／移动基础训练，保留 `target_qualified=false`。源策略形成后执行零样本目标评估；M0-T 未过不做目标训练或授予 Bevy 能力资格。两侧和最终行为验收门槛保持不变，不用源成绩代替迁移成绩。
- GPU 小批量验证完整惯量、约束、驱动、接触容量和步进，才扩大并行数；CUDA 网络不等于 GPU 物理。
- 新增 Rapier 持久无头 batch worker 与 RSL VecEnv，训练和游戏复用物理步、驱动、观测与重置。
- 分别报告源端、零样本迁移、目标适配后、Bevy 游戏四层成绩。
- 复用现有单世界、事件队列、物理线程和位姿发布，Goose 独立 profile；默认 MicroDuck 60 Hz 和 61／14 合同保持兼容，不继承其 BAM、尺寸或轴序。
- 正式游戏不依赖 Python。

### 三个 Actor 和任务接口

- 移动 **65→18**，零速度承担站立；恢复 **65→18**。
- 拾物 **82→18**：基础 65＋物体相对机身位置 3＋相对姿态四元数 4＋放置点相对机身位置 3＋六阶段 one-hot 6＋目标有效性 1。
- 阶段为接近、弯腰对准、合嘴、抬起、携带、放置；TaskGoal 提供物体身份、位姿、放置目标、有效性，训练/游戏坐标、布局、归一化一致。
- 18 轴顺序和目标动作语义以原合同为准。额外真实质量、接触力、地形可用于 Critic 和评价，不暗中混入 Actor。
- 显式状态机每 Tick 只选择一个 Actor；目标失效中止拾物，跌倒恢复优先；物体持有状态依据真实接触判断，无瞬移补救。

## 3. 里程碑与训练课程

| 里程碑 | 工作 | 退出依据 |
| --- | --- | --- |
| M0 物理接入 | M0-S：统一 MJCF、原生源端逐轴／嘴／足底／接触短检查；M0-T：目标端同版本复核 | 分别报告源／目标资格；整体 M0 仍要求同版本两侧通过 |
| M1 链路 | 优先锁源端依赖、小 batch GPU rollout、一次真实 PPO 更新、复载与 ONNX 对照；目标 batch 后续按需要接入 | 源端物理已通过，真实积分和优化器更新、有限数值与完整收据；目标资格单列 |
| M2 平地 | 站立→低速前后→横移→转向→启停反向→混合指令 | 独立移动验收，首个可操作 Bevy 候选 |
| M3 恢复 | 四类倒地→随机初态→真实扰动→继续移动 | 完整链通过，自然跌倒与直接倒地分别报告 |
| M4 地形 | 5/10/20 mm、正反缓坡、小台阶、方向变化与停止 | 各档达标，终点稳定 |
| M5 拾物 | 空载探地→近夹取→三重量→随机位置→运输转向→放置 | 各重量完整物体任务通过 |
| M6 集成 | 热切换、持物异常、用户输入、科学站实体、性能、连续运行 | 冻结游戏与三主线报告 |

M2/M3 共享站立基础并行；M4 需要稳定移动与恢复；拾物可达性从 M0 排查，完整学习在移动基础合格后展开。

移动奖励覆盖速度/yaw 跟踪、路线进度、终点稳定，并约束滑移、摆动、限位、力矩与功率。恢复覆盖姿态改善、稳定支撑、保持和继续移动，允许声明的身体/腿部接地，取消旧 nonfoot、低高度、低 upright 的立即终止。拾物覆盖接近夹持点、真实接触、离地、保持、运输、放置；低头本身不是成功。

逐体读取随机化范围，名义条件开始逐步加难；热切换保留前驱策略姿态、历史动作、驱动状态和相位。

## 4. 固定验收标准

结果出现前冻结协议。失败报告未通过，不移动出生点、改阈值或拼接重置后的轨迹。

| 项目 | 标准 |
| --- | --- |
| 物理 | 20 冷重置、逐轴/脚底/嘴部，无非有限状态、警告、异常能量增长或穿透逃逸；实际 dt 与次数符合 50 Hz |
| 足底 | 每区域 5/10/20/30 N＋行程末端，源/目标压缩差≤0.05 mm、曲线误差≤10%；释放耗散/冲击另报 |
| 站立 | 60 秒无跌倒，漂移≤5 cm |
| 移动 | 前进0.3、后退0.15、横移0.1 m/s、转向0.6 rad/s；200独立案例整体≥95% |
| 输入停止 | 短长按、松键、反向、失焦、暂停恢复；停止1秒内≤0.04 m/s，随后3秒漂移≤5 cm |
| 地形 | 5/10/20 mm、±5°坡、20 mm台阶，每类≥50例、≥90%；不授予连续楼梯资格 |
| 恢复 | 四方向各50例，10秒起身、稳定3秒、再移动1 m；整体≥95%、每类≥90%；真实扰动另测 |
| 拾物 | 三重量、圆柱夹持段/带把手物体；底部离地≥80 mm、保持5秒、携带2 m并转向、放置误差≤100 mm；每重量≥60例、≥90% |
| 集成 | 同世界移动→跌倒→恢复→移动→拾物→运输→放置；20固定程序全过，30分钟实际运行 |

恢复还要求直立度≥0.95、COM高度≥名义站立85%，不能瞬时竖起就交接。开发、训练、验收集独立；正式评估独立进程、确定性 ONNX、auto_reset=false。先核同状态观测/动作目标/实际力矩，再比较接触任务，不要求跨引擎长轨迹逐位一致。

性能要求持续50 Hz、无积累欠账，物理＋推理P95≤16 ms；渲染独立并插值，目标机实测画质、帧率与交互延迟。

## 5. 执行、排期与交付

每轮：**单一可检验问题→冻结版本与指标→短对照→≤两小时训练→独立评估→晋升或保留失败。**

- 首轮5–15分钟 pilot 测吞吐、内存和接触容量，再定 batch/更新；记录真实物理样本、优化器更新和资源消耗。
- G1 与 Goose 平等共享 DGX：仅 GPU PPO 开始前和释放时简短通知预计窗口并排队；不停止他人任务、不日常跨聊天问进度。等待用低频调度。
- 两次受控实验无改善就回查物理、初态、观测、终止和奖励分项，避免局部死循环。
- 保存模型/合同/代码/依赖/配置/seed/策略哈希；候选显式指定，不用 latest。M0-S 和源 GPU 链路未过禁止源端长训，M0-T 未过禁止目标训练；数值/约束/容量异常停止该批。
- 每关交付通过/失败/适用范围、录像、算力和下一实验，基于真实吞吐滚动估时。
- 当前实施聊天的 `Goose ProjectManager` heartbeat 每两小时检查产出与主线进度，不新建聊天。用户授权的单一 `bevy_performance` 子 agent 持续处理目标端性能；每周期有界测量、对照与回归，完成后闲置，下一周期再派工。性能工作不得改碰撞形状／过滤、求解配置、物理参数或观测／动作合同来凑速度。

当前M0实施中，尚未启动PPO。原始产物归档 `/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/`；本页为唯一计划和状态入口。最终交付冻结策略、本体/合同、Bevy包、复现命令、三主线报告及未通过清单。制造/电气/实物载荷仍由机器人工程侧维护；仿真资格不自动等于实物资格。

当前优先项是成熟源训练链路的可运行最小闭环：**复用并锁定依赖 → 统一任务 MJCF／碰撞代理 → 原生源端准入 → 小批量 PPO 与导出 → 首个站立／移动源策略 → 目标迁移**。Bevy 性能并行推进。每周期只选能解锁下一阶段的一个问题；局部实验达到诊断目的即收口，连续两次无改善就回到模型或成熟流程，不追加求解器分支。下方历史检查点中的“下一批”只记录当时决定，以本段现行顺序为准。

## 6. 实施检查点：M0 未通过

以下为真实积分或明确标记的静态诊断，不能授予策略资格。原 v1 模型、合同、收据及执行源码已冻结；后续实验使用新的目录和源码哈希，工具拒绝覆写已有收据。

| 项目 | 已取得的证据 | 结论 |
| --- | --- | --- |
| 源 full50 | 20 次冷重置共尝试 220 次积分；出现 QACC 警告及原生时间重置，嘴闭合和脚垫曲线失败 | 原结构不能直接开始训练 |
| 源 condensed50 | 20×100＝2000 次真实 20 ms 积分无警告；质量、COM、完整惯量及逐轴映射通过；嘴闭合和未校准脚底接触仍失败 | 数值冒烟通过，M0 未通过 |
| Rapier full50 | 接入 33 体、32 个树关节、12 个原生被动弹簧、实际输入转子及嘴销闭合约束；每 Tick 一次原生积分 | 已有目标侧运行证据，完整控制器和验收尚未完成 |
| 碰撞与性能 | 全部 15,727 个来源几何保持不变，按刚体／mask／摩擦打包为 33 个 compound collider；优化构建 PGS 4/8/16/32 四组分别完成 18/19/29/20 Tick 后触发 5 秒预算，嘴销最大误差 6.24/6.08/5.32/3.28 mm | 打包降低宽相开销，未达到性能或闭合门槛；不继续重复增加 PGS |
| 足底独立诊断 | 原生约束行的隐式 Kelvin 映射，96 个小夹具案例全部完成；静态曲线吻合，无释放能量增长 | 只覆盖无摩擦、已接触初态；整机、飞行落地、1.5 mm 行程末端、Rapier 和 Warp 均未授予资格 |
| 嘴部隔离诊断 | 12 个初始角的销距误差≤1.15e−16 m；固定头、无重力／接触时动态销距最大 3.69 mm，抵抗载荷时最大 8.17 mm | 几何映射正确；原软约束在 20 ms 下是独立阻断 |
| 嘴部离散约束对照 | 将关节阻尼与原生销／限位约束放到相同离散矩阵中求解；10 组固定头部案例、1000 次 20 ms 积分全部通过局部门槛；覆盖 20 N 载荷和 ±4.4 Nm 输入，销距最大 0.000143 mm、角关系最大 1.253e−5 rad、限位最大 1.311e−7 rad | 找到局部有效方案；尚无整机、碰撞物体、Rapier 或 GPU 资格 |
| Rapier 同嘴夹具对照 | 保留四杆几何、质量、完整惯量、物理 armature／阻尼／摩擦，分别完成相同十组程序各 1000 次积分；默认平面基最大销距 28.026 mm，沿连杆的等价平面基降至 0.000120 mm，角关系最大 5.621e−6 rad；PGS 4 已保持闭合，PGS 8/16/32 未消除限位问题 | 定位平面约束行收敛问题；新平面基全部通过闭合分项，但仅 2/10 组完整局部通过，限位最大越界 0.153929 rad；未修改整机装配或授予 M0 资格 |
| Rapier 预测限位 | 沿连杆平面基、PGS 4，显式启用原生双侧预测限位；十组程序、1000 次 20 ms 积分全部通过局部检查，最大销距 0.000120 mm、角关系 5.081e−6 rad、限位越界 1.863e−8 rad；两个物理端点均可到达 | 单关节显式选择，默认关闭；整机尚未启用，不继承 M0 资格 |
| 嘴部耗散与反力 | 源端五个初始角×四个初始速度，共 2000 次无驱动积分，无异常物理动能增长；目标端已记录积分前销／限位行，开启记录前后 5000 项轨迹字段完全一致；十组共享初态首步销反力最大差 1.217e−4 Nm、角度最大差 1.181e−4 rad | 固定头、无物体接触；反力为差异报告，未设置事后验收阈值，也不要求长轨迹逐位一致 |
| 地面可达性 | 2401 个静态蹲弯组合中 53 个夹持点在地面以上 5–50 mm，100/200/300 g 点载荷静力筛查通过 | 只是 FK、静力和几何支撑筛查；未验证碰撞、夹紧、真实起身或物体运输 |
| 观测时效 | 已证实 `mj_step` 后派生姿态仍为积分前状态；新运行时仅刷新运动学和 COM，关闭数值异常自动重置；17 项 Python 回归通过 | 65／82 布局不变，后续收据显式记录运行时 v2 和力／姿态的时间边界 |

证据入口：

- [源端冻结汇总](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/run_summary.json)。
- [Rapier 优化构建 PGS 对照](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/target_pgs_release/matrix.json)，是无 Actor、无 PD 的规定力矩诊断；P95 仅代表这些短轨迹，不能替代最终游戏验收。
- [足底隐式接触诊断](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/kelvin_contact_native.json)、[嘴部隔离诊断](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/jaw_isolated/jaw_receipt.json)。
- [工程结构检查](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/engineering_structure_checkpoint.json)：486 项通过。运行时代码在 `crates/modules/{robot,simulation}/src/goose/`，开发工具在 `crates/dev_tools/`，临时模型、脚本、日志和轨迹全部在备份目录。

本轮 Rust 回归为 14 项 robot 单元、21 项 simulation 单元和 2 项原生 armature／质量检查通过；另有 10 项依赖外部冻结夹具的既有检查未运行。Rust 格式检查和 Git 空白检查通过。这些代码检查不替代本体资格。

新增 [嘴部隐式约束收据](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/jaw_implicit_verified/receipt.json) 和 [最终执行源码](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/attempts/jaw_implicit_verified_current_code/mouth_constraints.py)。诊断没有 Actor、PD 或优化器更新，没有初始化后的关节坐标写入。21 项 Python 回归及 [489 项结构检查](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/engineering_structure_jaw_checkpoint.json)通过；临时日志在测试临时目录或备份目录。

局部对照发现：原生隐式积分器在约束力求解后修改速度增量，固定头部也会因关节阻尼而失去闭合。实验方案统一使用 `M_effective = M_physical + dt*C` 求解阻尼和约束；原生 Euler 仅负责一次状态推进，关闭二次阻尼处理。另用真实边界距离生成预测限位参考，激活余量不缩小实际行程。新增的对角量是数值离散矩阵，必须和物理 armature 分开记录，禁止直接当作目标引擎的真实惯量。非线性速度力仍为显式评估；此方案目前属于实验积分器，没有替换获批的整机候选或 GPU 路线。

Rapier 证据：[同代码、PGS 4 的两种平面基对照](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/jaw_rapier_verified_basis/comparison.json)、[沿连杆平面基完整轨迹](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/jaw_rapier_verified_basis/coupler_axis.json)、[PGS 收敛对照](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/jaw_rapier_coupler_basis/matrix.json)。最终执行源码及实际编译的 `third_party/rapier3d` 已一起冻结在 `attempts/jaw_rapier_verified_basis_code/`；收据核对源码、二进制和依赖哈希。早期辅助文件记录的是 registry 源码哈希，不能当作实际编译依赖证明；[依赖来源说明](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/jaw_rapier_verified_basis/native_dependency_provenance.json)明确实际路径。

开发工具位于 `crates/dev_tools/src/bin/goose_jaw_probe.rs`，用 `--pin-basis source_axes|coupler_axis` 和 `--limit-mode original|predictive` 显式选对照；默认保留原平面基及原限位。整机装配暂未启用这两项实验选择。工具每 Tick 清理旧力队列，经统一步进接口提交输入转子力矩和载荷力臂，核对一次力矩更新、一次 20 ms 积分。启用 `sim2sim_limit_row_trace` 后记录积分前的原始销冲量及签名广义冲量；最终稳定化后的值仍单独保留，不混用时间边界。

本检查点的 21 项 simulation 单元及 2 项 armature／完整惯量检查通过，10 项依赖外部冻结夹具的既有检查未运行；[492 项工程结构检查](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/engineering_structure_rapier_jaw_checkpoint.json)、Rust 格式和 Git 空白检查通过。所有实验输出仍在备份目录，未占用 GPU。

本轮证据：[原限位／预测限位同代码对照](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/jaw_rapier_predictive_stop/comparison.json)、[最终目标端局部收据及积分前反力](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/jaw_rapier_frozen_checkpoint/receipt.json)、[源端被动耗散](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/jaw_source_passive_energy/receipt.json)、[同初态首步反力比较](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/jaw_source_target_reaction/receipt.json)、[只读记录不改变轨迹](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/jaw_rapier_integration_reaction/trace_physics_regression.json)。源端动能扣除了数值矩阵中的 `dt*C`，保留真实 armature；20 组最大单步增量 1.278e−15 J，处于预先固定的数值容差内。

本检查点的默认 simulation／armature／预测限位检查共 27 项通过，10 项外部夹具检查未运行；启用源限位及原生行记录的 5 项检查通过，原生后端 multibody 范围的 21 项检查通过。[495 项结构检查](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/engineering_structure_predictive_checkpoint.json)、格式及空白检查通过。实验源码、原生依赖、模型和轨迹按实际执行哈希冻结；未使用 GPU。

下一检查点的首个问题：**原 33 体模型的线性脚垫弹簧，能否与阻尼和约束一起在一次 20 ms 积分中隐式求解？** 先在脚垫夹具固定 5/10/20/30 N、释放及行程末端程序，验证物理 `K/C` 与数值矩阵分离，再做新的整机实验候选和 20 次冷重置。新的数值离散方法须独立版本化，原 v1 模型和收据保留；没有源／目标同版本物理准入和 GPU 小批量证据前不开始 PPO。嘴部仅在接入整机或物体接触后出现新失败时回到局部诊断。

### 脚垫与整机实验检查点

上述线性脚垫问题已有真实收据：原 12 个脚垫各执行固定 5/10/20/30 N、30 N 释放、正反 40 N 端点和无外力初速度八种程序。源与目标各 **96/96 组通过、9,600 次 20 ms 积分**；[源端](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/pad_source_verified/receipt.json)、[目标端](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/pad_rapier_verified/receipt.json)、[逐组比较](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/pad_source_target_comparison/receipt.json)。最大压缩差 **3.877e−10 m**，远小于 0.05 mm 门槛。无初始化后的 qpos 写入、隐藏子步、Actor 或优化器更新。此处父脚固定、没有地面接触，不等同于整机足底接触、摩擦或冲击资格。

源端的编译项为 `A_effective=A_physical+h*C+h²*K`、`C_effective=C+h*K`，真实质量、完整惯量、物理 K/C/A 单独记录；目标端直接使用物理参数，避免重复加入数值项。单个最近端点限位在 40 N 首步超载时曾穿过另一端 **0.205 mm**；[失败对照](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/pad_implicit_metric_pilot/receipt.json)保留。两条固定 tendon 原生单侧行共同表达原区间，保持 1.5 mm 行程，源端最大越界约 1.71e−10 m。

新的 [33 体源端实验候选](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/goose_460_full50_be_v2/source_receipt.json)通过 20 次冷重置，共 2,000 Tick，以及惯量、逐轴和嘴部局部检查；实际物理加控制器 P95 **3.82 ms**。这只证明有限数值和时序：零动作仍会倒地，碰撞最大穿透约 **12.26 mm**，发生在倒地后的头壳与地面。[接触与物理能量诊断](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/full50_be_v2_contact_energy/receipt.json)另存；能量计算扣除数值对角项，但未计接触柔顺储能，不据此授予全局耗散资格。

[同版本目标端 20 次检查](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/goose_460_full50_be_v2/target_receipt.json) **未通过**：共 800 Tick，每次均达到原定 5 秒预算，物理 P95 **295.68 ms**，最大销点误差 **2.379 mm**。固定头夹具的销点精度不能继承到整机；源端采用零动作 PD，目标端是声明的 rotor 力矩程序，二者也不是整机轨迹对照。[物理导出核对](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/goose_460_full50_be_v2/plant_export_receipt.json)确认目标使用的身体、完整惯量、K/C/A、几何及排除项与父版本逐项一致。

本检查点 Python 30 项、Goose 合同 5 项、simulation/armature/预测限位 27 项通过；10 项依赖外部冻结夹具的既有 simulation 检查未运行。[510 项结构检查](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/engineering_structure_pad_candidate_checkpoint.json)、格式及空白检查通过。实际执行源码和原生依赖哈希冻结在备份目录；GPU 使用为零，M0/M1 未通过。

补充 [目标端接触轨迹](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/full50_be_v2_target_contacts/receipt.json)：前 12 Tick 的关节坐标、速度和销点距离与原收据逐项完全一致。**该旧探针误把原始流形当作求解流形，接触计数与“没有嘴部求解接触”的结论撤回**；对应纠正清单另存，见下文最新接触检查点。物理轨迹和实际执行源码继续保留。

下一周期以 **整机接触与任务碰撞代理** 为主线：先对齐同初态下的接触对、距离和约束，再建立保留空腔、脚底、嘴部及干涉关键面的代理，分别量化源／目标载荷、冲击、销点闭合和吞吐。原几何失败保留，不扩张排除项、不延长预算凑过关，不再重复已通过的固定头嘴部或脚垫微实验。之后才进行同世界重置、实际控制器和 GPU 小批量准入。

其后依次验证任务碰撞代理和足底完整接触实现；新代理须保留空腔、脚嘴关键表面及原碰撞过滤，报告来源映射和几何误差。足底需补摩擦、脱离后落地、行程末端和源／目标曲线对照，再接入相同版本整机与 GPU 小批量。未通过这些门槛前不占用 DGX 做 PPO。

### 整机耦合诊断检查点

[无地面诊断对照](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/full50_coupling_comparison/receipt.json)保留全部原始身体、自碰撞几何和过滤：零重力与自由落体的最大嘴销误差分别为 **0.768 mm**、**0.711 mm**。**旧接触计数及“零求解接触”结论已撤回**，因为求解器使用聚类后的 `solver_manifolds()`。只保留嘴部预测限位的对照，销点误差完全一致，主动坐标和速度最大差分别为 4.64e−12 rad、2.33e−10 rad/s；不据此修改正式限位合同。

只读记录的 `M*WJ-J` 相对残差约 1e−7 至 7.6e−7，没有触发原生能量守卫回退；但第 12、15 Tick 积分前销点行的速度残差分别为 **0.00959 m/s**、**−0.03838 m/s**，与随后 0.192、0.768 mm 的位置漂移吻合。现有证据尚未定位产生残差的具体约束更新。[扩展限位记录](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/full50_coupling_limit_trace/receipt.json)加入原生 Jacobian、加权 Jacobian、速度、RHS、CFM、矩阵和限位行，状态与原探针逐项完全一致；[实际执行源码与二进制](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/attempts/full50_coupling_limit_trace_code/verified_snapshot.json)全部冻结。PGS8 的最大误差反而增至 **2.680 mm**，停止扩大迭代数扫描。

碰撞代理的两个短试验均未晋升：同体同过滤盒体合并只从 15,727 个叶几何降至 15,705；单个前躯壳重新 CoACD 得到 **758** 个凸体，多于原来的 **547**，后续空腔抽样验证达到 300 秒预算，按 [失败终态](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/coacd_torso_shell_pilot/terminal_status.json)保存，不重启、不采用结果。没有运行时几何、物理参数、Actor 合同或碰撞排除项变更，GPU 与优化器使用为零。

下一轮限定两个可检验问题：用逐次原生约束更新记录定位销点行残差首次出现的位置；从 CAD 已有壳体面和关键接触区域构造更少几何的代理，先做静态距离与空腔对照。每个实验预先限定时间与输出，未改善时保留失败并回到诊断，不重复固定头微实验或增加 PGS 扫描。M0/M1 仍未通过。

本轮启用只读记录的 simulation、完整惯量和预测限位回归 **31 项通过，10 项外部夹具检查未运行**；Rust 格式和 Git 空白检查通过。运行时代码、开发工具与实验产物继续遵循 [唯一工程规范](/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/bevy_engineering_rules.md)，[本轮结构检查](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/engineering_structure_coupling_checkpoint.json)作为提交门槛。原先平铺的 `goose_contract.rs` 已归入 `crates/modules/robot/src/goose/contract.rs`，不另设工程规则入口。

### 求解接触统计纠正与同状态几何检查点

最新权威入口为 [接触检查点及撤回清单](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/contact_manifold_checkpoint.json)。原探针过滤使用 `has_any_active_contact()`，计数却读取原始 `pair.manifolds`；开启聚类时，实际求解器使用 `pair.solver_manifolds()`。旧计数为零不代表没有求解接触。修正后零重力首 Tick 有 **2,401 个求解候选接触点**，包含正距离的预测点，不能把该数量直接当作非零力或真实相撞数量。旧收据保持原样，并分别附上 `contact_reporting_correction.json`。

逐次原生行更新显示：嘴部两条约束行在自身求解后残差约 1e−9 至 1e−8 m/s；较大的速度改变量出现在随后的接触求解阶段，再进入下一轮关节求解。关闭 Rapier 接触复用没有改变 20 Tick 的状态。源端以相同初态、零重力、无地面、rotor **0.24 Nm**、其他执行器零力矩运行 [20 Tick 对照](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/full50_source_prescribed_zero_gravity/receipt.json)，没有嘴部求解接触，最大销点误差 **2.98e−9 m**。

[同状态几何核对](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/full50_same_state_geometry/receipt.json)定位到 `beak_input_bush_candidate_collision_005/007` 与 `beak_motor_catalog_case_collision_000`：目标端原始流形报告 **−1.846/−5.586 mm**，但身体位姿差仅 26–56 nm；MuJoCo、独立双精度凸体相交检查、目标端原生 f32 的独立距离／接触查询，都确认这两块凸体相距约 **8.075 mm**。原生 f32 的新建叶流形也得到正距离。[原生叶查询收据](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/full50_native_leaf_query/receipt.json)保留全部证据。因此问题已缩小到复合体持久流形路径的一致性，尚未定位库内具体触发条件；不能以关节质量矩阵误差或硬件碰撞作结论。

保留原几何与过滤，仅在开发探针把预测距离从默认 **20 mm** 改为 **2 mm**，得到 [新受控短试验](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/full50_prediction_two_mm/receipt.json)：零重力 20 Tick 最大销点误差 **0.120 µm**。记录功能开启／关闭时坐标、速度和销点误差完全一致；[关闭记录的构建](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/full50_two_mm_feature_off/receipt.json)物理 P95 **9.80 ms**。这只覆盖无地面的规定力矩程序，没有 Actor、用户输入或全技能验收；2 mm 参数仍是声明的诊断覆盖项，未晋升正式运行合同。

带地面 100 Tick 的 [PGS4 试验](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/full50_two_mm_floor_pilot/receipt.json)仍不合格：P95 **105.00 ms**，最大销点误差 **2.103 mm**，最大报告穿透 **44.36 mm**，来自右脚垫与地面。因接触路径已改变，额外执行一次 [PGS32 对照](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/full50_two_mm_floor_pgs32/receipt.json)：P95 **29.71 ms**、销点误差 **0.233 mm**、最大报告穿透 **27.09 mm**，仍未达到物理及性能门槛。停止该分支的冷重置长批与迭代扫描，不据短试验宣告 50 Hz 或 M0 合格。

下一周期以 **足底与整机任务碰撞代理** 为单一主线：在同状态下先核对脚垫—地面接触距离、实际冲量及限位响应，再做保留空腔和脚嘴关键面的几何简化，用源／目标对照验证载荷与冲击。保持真实质量、完整惯量、K/C/A、原碰撞过滤及 65／18、82／18 Actor 合同。源与目标同版物理准入前不开始 PPO；M0/M1 均未通过，GPU 使用为零。上一轮归类为有证据推进，本轮同样产出了改变下一行动的真实收据。

本检查点 **31 项回归通过、10 项外部夹具检查未运行**，开启／关闭记录的真实整机轨迹对照通过；[510 项工程结构检查](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/engineering_structure_contact_manifold_checkpoint.json)、Rust 格式和 Git 空白检查通过。所有临时脚本、模型、收据、轨迹与按执行哈希冻结的源码仍在项目备份目录。

### 可移动足底与 CCD 对照检查点

新增开发夹具使用原始右脚体和六个脚垫的质量、完整惯量、几何、K/C/A 与 1.5 mm 行程。脚体仅受竖直导向约束，地面为固定 0.2 m 厚箱体；取消摩擦，重力为零，向真实脚体施加 5／10／20／30 N 每接触区及半机重载荷。释放和一次声明的冲击另列程序。夹具总物理质量 **0.292980652 kg**；外力模拟静态载荷，没有给脚体添加半机质量，因此冲击不代表完整机器人的落地资格。每程序 100 Tick，每 Tick 一次真实 20 ms 积分。

[源端收据](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/guided_sole_source/receipt.json)与 [目标端初始收据](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/guided_sole_target_pgs4/receipt.json)各覆盖 7 个程序／700 次积分。目标静载总弹簧承载相对误差小于 **0.001%**，但不能据此授予接触资格。源端保留原生柔顺法向接触，目标保留原生法向接触，两者接触规律未匹配。源端从原始 0.5 mm 间隙起步时没有提前生成法向约束；20／30 N 档越过有限厚度地面并继续下落，失败轨迹完整保留。

冲击程序暴露了独立于弹簧局部求解的问题。[CCD 开启记录](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/guided_sole_ccd_on_witness/receipt.json)中，Rapier 自动高速 CCD 在积分后单独裁剪脚垫刚体位置，首 Tick 的刚体位置与原生关节链位置差 **74.35 mm**，随后穿透逃逸。[关闭 CCD 的唯一对照](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/guided_sole_ccd_off_control/receipt.json)中，该位置差为 **零**，没有发生相同逃逸，但首次接触仍穿透 **69.35 mm**。六个非冲击程序的实际轨迹完全相同，新增记录与初始记录的实际轨迹也完全相同。这只确认该夹具的 CCD 裁剪会破坏关节几何一致性，不声称已经解决整机穿透。

Goose50 现允许显式关闭 CCD 做单次积分诊断，禁止大于一次 CCD／求解时间步；默认仍为原值，MicroDuck 的步进规则保持原值。此选择尚未晋升正式候选。下一次对照检查整机关节几何在关闭 CCD 后是否一致，并建立源／目标同版本的 20 ms 法向接触试验；不继续盲扫 PGS。任务碰撞代理继续保持空腔、脚嘴关键面及原自碰撞过滤。**M0 未通过，PPO／GPU 使用仍为零。**

结果、范围和下一问题集中在 [足底检查点](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/guided_sole_checkpoint.json)。运行时代码位于 `crates/modules/simulation/src/`；新增 Rust、Python 夹具分别位于 `crates/dev_tools/src/bin/`、`crates/dev_tools/python/src/bevy_microduck_tools/goose/`；临时模型、日志、收据与冻结源码均位于项目备份目录。模拟库 **22 项通过、10 项外部夹具未运行**，Goose Python **30 项通过**，工程结构、Rust 格式及 Git 空白检查通过。

### 整机 CCD 配对与预测法向接触检查点

[关闭 CCD 的整机对照](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/full50_floor_ccd_off/receipt.json)和 [同代码开启 CCD 的控制组](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/full50_floor_ccd_on_control/receipt.json)各运行 100 Tick，保留 33 体、全部原始几何／过滤、PGS32 与 2 mm 预测距离。两组的实际身体位姿、主动坐标、速度、嘴销误差完全相同，刚体—原生关节链位置差均为零。物理 P95 分别 **6.96／24.93 ms**，只代表这次本机规定力矩程序；关闭 CCD 没有改善该程序的 **0.233 mm** 嘴销误差。脚垫实际最低点的最大地面穿透为 **4.43 mm**；原始接触流形的最大负距离来自嘴与躯干，不能将其当作足底穿透。此前可移动足底的 CCD 失配不能外推为这份整机轨迹的根因。

新增显式 `predictive_rigid` **无摩擦法向接触实验**：源端以 2 mm 接触搜索带，用实际有符号距离建立 `v_next ≥ −distance / 20 ms` 的原生单边行；目标端保留原生接触求解，声明接触系数、零容许穿透、关闭接触复用与 CCD、PGS32。两者仍有有限正则化及 Newton／PGS 差别，不声称完整接触法则已等价。真实质量、完整惯量、K/C/A、六个原始脚垫和初态未改变；没有积分后的坐标改写，也没有额外时间子步。默认 v2 源端／游戏接触行为不采用这个局部实验。

[源端](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/guided_sole_predictive_source/receipt.json)与 [目标端](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/guided_sole_predictive_target/receipt.json)各完成 7 个程序、700 次积分。源端四档静载已不穿过有限厚度地面；末 25 Tick 的逐脚垫压缩差小于 **0.000005 mm**，稳态总载荷曲线误差小于 **0.00003%**。加载瞬态的压缩差仍达 **1.26 mm**，冲击首 Tick 仍有约 **69.35 mm** 穿透；不得用稳态成绩代替动态、冲击或整机准入。首 Tick 法向约束新增三档载荷回归，确认实际间隙、单次原生积分和未改写坐标；Goose Python **33 项通过**。

证据、实际执行源码及未通过项集中在 [预测法向接触检查点](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/predictive_normal_checkpoint.json)。下一主线转回任务碰撞代理与复合体持久流形的一致性，保留空腔、脚嘴关键表面和原自碰撞过滤，不再做局部嘴部或 PGS 扫描。**M0 未通过，未启动 PPO／GPU，也未晋升新运行候选。**

### 流形重建与占用覆盖代理检查点

开发探针新增 `--manifold-cache fresh`，使用原生 Parry 查询接口逐次重建原始流形及工作区，实际记录 **20,669 次**查询。此模式同时丢弃原始流形的热启动数据；后续 Rapier 接触聚类与匹配保留，不能将结果解释成单一缓存机制的隔离证明。默认运行行为未改变，全部几何、质量、惯量、弹簧和原碰撞过滤保留。

[重建流形的 100 Tick 整机对照](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/full50_fresh_manifolds/receipt.json)中，最大报告负距离从约 **27.09 mm** 降至 **17.90 mm**，但最大嘴销误差从 **0.233 mm** 增至 **0.290 mm**，脚垫实际地面穿透从 **4.43 mm** 增至 **6.91 mm**。[同代码、关闭接触复用的缓存控制组](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/full50_fresh_manifolds_cached_control/receipt.json)与前一轮控制状态完全相同。流形重建未解决 M0 问题，不晋升这个模式，也不继续重复该扫描。执行源码、二进制及实际 Parry 依赖源码均已冻结；原生复合体回归确认旧负距离被丢弃，新查询与独立查询一致。

[严格占用覆盖代理 pilot](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/convex_union_proxy_pilot_verified/receipt.json)针对同一躯干前左壳体的 **547 个**原始凸块，在 **45.62 秒**内仅合并 **1 对**，得到 **546 个**凸块。合并先做占用见证与体积拒绝，再对所有原始面组合求线性规划，检查合并凸包中是否存在同时位于两块之外的区域；记录归一化平面深度的数值容差，不宣称欧氏 Hausdorff 或完整 CAD 资格。来源成员、碰撞掩码和摩擦保留，几何试验未装入运行候选。该严格简化路线的收益不足，停止扩大相同搜索；首个脚本因类型守卫与实际 `convex_mesh` 命名不符，在处理前失败，原脚本和失效说明另行保留。

权威入口为 [流形与代理检查点](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/manifold_proxy_checkpoint.json)。下一受控问题是同一 multibody 内两侧接触的有效逆质量：用实际的有符号 J／WJ 和同一隐式矩阵核对两侧求和与合并行的差别，再决定是否改求解器。目前只是待验证假设，不是已证实根因。停止局部嘴部、PGS、流形缓存和严格几何合并的重复扫描。**M0 仍未通过，PPO／GPU 使用为零。**

### 共享关节链接触响应检查点

[原生接触行核对](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/full50_shared_contact_mass_baseline/audit.json)使用积分前的实际有符号 J/WJ，20 Tick 共记录 **3,456 条**共享关节链接触行，全部与原生所有权清单匹配。原系数采用两侧独立响应之和，与该和的最大相对差约 **2.14e−7**；同一广义速度上的真实响应还包含交叉项。**567 条**带有非零冲量，其中 **283 条**的合并响应比独立之和低至少 10%。部分法向行的合并响应近乎抵消，仍累计约 **0.91 N·s** 的冲量。此结论定位了接触行系数缺口，不等于确认整机全部失败的根因。

[关闭记录的同代码控制组](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/full50_shared_contact_mass_trace_off/receipt.json)与开启记录的 20 Tick 状态逐项相同。刚体几何记录按唯一身体身份比较，避免将 HashMap 的列表顺序误当物理差异；关节位置、速度、销点误差、时钟和足底几何完全相同。实际执行代码和二进制均已冻结。

新增显式 `--shared-owner-contact-mode combined` 实验：共享链的第一侧行置零，第二侧使用有符号合并行及一次原生隐式质量响应，保持双侧缓冲布局。使用原生相对 Jacobian 的既有数值抵消守卫，清除不可作用行的热启动；原碰撞检测、过滤、几何、真实质量、K/C/A 与 20 ms 单次积分保留。此模式默认关闭，属于数值合同覆盖项。三刚体解析夹具复现不可运动法向的独立响应和为 2/3、合并响应为零；另一可运动单接触夹具验证一次原生求解能消除闭合速度。记录读取完成的广义速度，不把积分前刚体速度缓存当作求解后的状态。

[同代码原生控制](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/full50_combined_contact_native_control/receipt.json)与 [合并行 pilot](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/full50_combined_contact_pilot/receipt.json)各执行 20 Tick、PGS32、2 mm 预测距离及关闭 CCD。合并行后非退化行的系数—实际响应误差小于 **1.50e−7**，近零响应行不再累计冲量；但嘴销最大误差 **0.0752→0.0820 mm**，足底实际最大穿透 **4.43→8.43 mm**，仍未通过。记录开启时物理 P95 **8.14／10.48 ms**，仅代表该短规定力矩程序。默认控制的物理状态与先前基线完全相同。**不晋升合并模式，不扩大整机长批，不继续扫描 PGS。**

汇总与下一动作见 [共享接触检查点](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/shared_contact_checkpoint.json)。下一周期回到本计划已规定的凝聚足底路线：先以原脚体质量与六接触区测 5/10/20/30 N 的真实压缩、释放及 1.5 mm 末端；已有源端直接格式 `solref` 明确未校准，不能把参数数值相同当作物理 K/C 等价。源端载荷证据先于目标接触实现，保持总质量、COM、完整惯量及合力／力矩映射，不授予高频动态等价。新实现须单独冻结版本，原 v1/v2 不覆写。

本检查点原生关节范围 **23 项通过（含两项新增物理回归）**，simulation **25 项通过、10 项外部夹具检查未运行**；[519 项结构检查](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/engineering_structure_shared_contact_checkpoint.json)、格式及空白检查通过。临时收据、轨迹、日志和冻结源码全部在项目备份目录。**M0 仍未通过，GPU／优化器使用为零。**

### 凝聚足底源端物理载荷检查点

[原生接触平衡检查](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/condensed_source_native_balance/receipt.json)确认：凝聚 v1 的直接格式 `solref` 不能当作 N/m 接触刚度。按 5／10／20／30 N 每区的 `load/K` 设置初始压缩后，实际平衡总载荷分别约为目标的 **70／94／301／352 倍**；每档另用独立冷启动验证平衡载荷。原模型和失败证据保留，不修改 v1。

新的开发工具 `condensed_contact.py` 在 MuJoCo 3.10 的一次原生 20 ms 积分中设置物理法向行：每区 K/C 按实际接触点分配，`R=1/[h(h*Kp+Cp)]`，`aref=−v/h−Kp*gap/[h(h*Kp+Cp)]`，同步原生 island 行。使用实际距离，不改写积分后的坐标，不改变脚体质量、COM 或完整惯量。[源端 pilot](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/condensed_source_physical_rows/receipt.json)完成 **8 个程序、800 次积分**，无求解警告；5／10／20／30 N 的稳态压缩符合物理 K，40 N 超载的最大 1.5 mm 行程越界约 **5.93e−11 m**。释放与声明的冲击轨迹另存。

此结果仅覆盖原凝聚右脚、六接触区和无摩擦竖直导向。六个移高 1.5 mm 的背衬盒只提供实验末端约束，尚无整机 CAD 碰撞资格；搜索带根据该夹具最大自由位移推导。冲击使用真实脚体质量和声明的外力，不代表完整机器人落地惯性；倾斜、摩擦、合力／力矩映射及高频等价均未授予资格。工具明确拒绝整机、多自由度和摩擦接触。

**下一批只实施同参数的 Rapier 法向行及八程序配对**，预先保留 0.05 mm 压缩差和 10% 静载曲线门槛，动态、冲击及背衬误差分别报告，再决定是否进入整机候选。M0 仍未通过，PPO／GPU 为零。权威入口为 [凝聚足底检查点](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/condensed_contact_checkpoint.json)；[实际执行源码快照](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/attempts/condensed_source_physical_rows_code/verified_snapshot.json)共 26 个文件逐项哈希验证。

新增开发代码和回归分别位于 `crates/dev_tools/python/src/bevy_microduck_tools/goose/` 与 `crates/dev_tools/python/tests/`；运行时合同继续位于 `crates/modules/robot/src/goose/contract.rs`。本检查点 **51 项 Goose Python 回归、525 项工程结构检查及 Git 空白检查通过**。提交前按 [唯一工程规范](/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/bevy_engineering_rules.md)复核；全部实验产物仍在项目备份目录。

### 凝聚足底源／目标配对检查点

开发工具 [goose_condensed_sole_probe.rs](/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/crates/dev_tools/src/bin/goose_condensed_sole_probe.rs)读取同一冻结凝聚本体的质量、COM、完整惯量、六盒几何与 K/C。原生后端新增显式 `sim2sim-physical-normal-contact` feature，默认为关闭；每接触点的物理柔顺系数和恢复力保留到代数松弛阶段，不改变紧凑的 scalar／SIMD 接触点布局。

[原 PGS32 配对](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/condensed_native_contact_comparison.json)虽通过静载曲线门槛，加载瞬态压缩差仍达 **0.217–1.50 mm**，不晋升。单点解析回归确认物理系数正确；多点残差可由原有耦合行的 PGS 更新独立复现，未扩大迭代扫描。

受限的 `guided_block` 对照用原生 J/WJ 同时求解原法向行的单调互补方程，保留逐点冲量和归属；64 次有界二分是既有求解阶段中的代数工作，仍只积分一次 20 ms。背衬使用实际 gap/h 和与源端相同的有限正则化。求解器拒绝多自由度、多所有者、摩擦和不平行法向，**不能接入整机**。首个实现取错切向块偏移，被行布局守卫拦截；[失败代码、二进制和终态](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/attempts/condensed_target_block_layout_failure/failure.json)保留，随后按原生 normal 行布局修正。

[最终独立比较](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/condensed_guided_block_comparison.json)覆盖相同八程序、每引擎 **800 次积分**；固定 0.05 mm 压缩和 10% 静载门槛全部通过。全程最大压缩差 **0.000022 mm**，目标静载总力误差 ≤**0.00012%**，行程最大越界 **0.000018 mm**；释放后物理能量无增长。原生 epoch 连续 1–100，实际步长记录符合 50 Hz。目标局部夹具物理 P95 ≤**0.117 ms**，不外推整机性能。物理清单比较允许最多两个 f64 解析 ULP，原始输入文件 SHA256 完全相同。

权威入口为 [目标端凝聚足底检查点](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/condensed_contact_target_checkpoint.json)；[最终执行快照](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/attempts/condensed_target_guided_block_code/verified_snapshot.json) **159 文件**核对执行哈希和二进制。原生关节范围 **27 项通过**，simulation **25 项通过、10 项外部夹具未运行**；格式、空白及结构检查通过。原候选、失败收据和默认运行行为保留，GPU／优化器为零，**M0 未通过**。

下一批转向**无导向足底的完整合力／力矩和倾斜接触**：建立新的实验版本，保留完整六自由度及惯量，先验证耦合法向行在名义半机重与 ±3° pitch／roll 初态下的响应；禁止把单自由度求解器套到整机或继续重复导向载荷实验。无导向、摩擦与几何背衬资格完成后，再接入同版本 21 体候选的冷重置、驱动和嘴部约束准入。

### 六自由度足底合力／力矩检查点

源端新增开发模块 `condensed_free_contact.py`，目标端新增开发工具 `goose_free_sole_probe.rs`，仍使用同一冻结右脚的质量、COM、完整惯量与六个原始接触盒。局部基础近似按实际点数分配 K/C，并根据脚垫轴与地面法向的余弦平方作虚功投影；它不等于原六个共享滑动坐标的精确凝聚，不继承原脚垫高频动态资格。脚体保留完整六自由度；零重力、零摩擦，名义半机重外力施加于真实踝部原点。初始高度只在初始化时按旋转后的最低盒角点设为 0.5 mm 间隙，积分后不改写位姿。

新的 `experimental_free_normal_block` 默认关闭，限定同一六自由度所有者、固定对侧、零摩擦和最多 96 条原生行。使用原生有符号 J/WJ 构造互补方程，保留实际非对称响应，采用有界主动集和 LU；未收敛或残差超限即失败。它在既有有偏与松弛求解阶段末尾分别进行代数求解，仍只作一次 20 ms 原生积分，不能套入整机。单步合力／力矩回归最初遗漏 Rapier 默认 0.1 角阻尼；[失败夹具源码](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/attempts/free_normal_first_analytic_failure/failure.json)保留，清零与源端一致后解析回归通过。

[独立比较](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/condensed_free_comparison_v2.json)覆盖平放、±3° pitch、±3° roll，共每引擎 **500 次积分**。五组稳态载荷相对误差 ≤**3.73e−7**，目标力矩平衡残差 ≤**1.85e−6 N·m**，未越过 1.5 mm 行程；平放全程压缩差约 **0.00034 mm**。倾斜全程最大压缩差分别 **0.1024／0.1296／0.1085／0.1085 mm**，超过这批预先保留的 0.05 mm 局部门槛。动态合力差最大 **0.6543 N**，力矩差最大 **0.01106 N·m**，分别报告，不用稳态成绩覆盖瞬态。两条侧倾轨迹的平方四元数范数误差约 **2.265e−6／2.205e−6**，超出现有接口的 2e−6 容差；开发轨迹保存原值，没有绕过 Actor 合同或归一化物理状态。**拒绝本次局部动态晋升，M0 未通过，GPU／优化器为零。**

执行代码、二进制与比较器见 [17 文件执行快照](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/attempts/condensed_free_v2_executed_code/verified_snapshot.json)。Python **55 项通过**，原生关节 **29 项通过**，simulation **25 项通过、10 项外部夹具未运行**。下一受控问题只隔离偏心 COM 自由体的惯性响应：以同一物理角速度比较无接触初态与第二个 Tick，再决定接触或积分公式是否需要调整；单位旋转保持须在原生积分表示内处理，禁止积分后搬正身体。不继续扫描 PGS，不重复导向载荷，不把这份右脚夹具当作摩擦、左脚、背衬 CAD 或整机资格。

### 六自由度 BE 接触顺序检查点

[无接触惯量与接触矩阵审查](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/free_inertia_contact_metric_audit.json)排除了偏心 COM 响应不一致的假设。五个程序各两次真实积分，完成后的 COM 速度最大差 **7.12e−8 m/s**，目标未触发能量回退；同一初态、同一坐标基下的物理质量矩阵和速度相关加速度矩阵最大绝对差均为 **1.89e−8**。居中 COM 两组仅作声明的诊断控制，没有改写候选质量或惯量。

五个冻结接触初态的单 Tick 对照确认：源端 v2 接触力满足隐式更新前的 Euler 质量度量，逐点残差 ≤**1.51e−13 N**；用实际完成的隐式速度核对相同 K/C，倾斜四组残差达 **0.620–0.760 N**。实际速度与原生隐式矩阵重建误差 ≤**4.72e−16 m/s**。这是该夹具接触求解和后续速度更新的边界不一致，不能据此推断所有整机失败的根因。

新版本 `goose_free_condensed_be_contact_v3` 延续既有 BE 实验：物理 K/C 接触行隐式求解，非线性速度力显式计算，关闭原生 Euler 的二次阻尼处理，每 Tick 一次原生 20 ms 积分。目标通过显式 `free_block_plain` 选择既有物理质量矩阵控制；默认 v2、游戏和训练路线未采用此选择。它仍是 **CPU 局部实验**，没有替代计划中的源端隐式／GPU 准入。原 6DOF 所有者、零摩擦、固定对侧等求解守卫保持不变。

[v3 独立比较](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/condensed_free_comparison_be_v3.json)覆盖平放和 ±3° pitch／roll，每引擎 **500 次积分**，五组通过原定 0.05 mm 压缩差、10% 静载曲线及 2e−6 平方四元数范数门槛。最大压缩差 **0.000023 mm**，合力差 **3.53e−5 N**，力矩差 **3.95e−6 N·m**；静载相对误差 ≤**2.99e−7**，单位旋转误差 ≤**1.20e−7**，行程无越界。完整质量、COM、惯量和原六盒输入保留。相同代码运行默认 v2 的 **500 帧全部字段与旧失败轨迹完全相同**。

[额外接触力精度检查](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/free_be_completed_force_law_v3.json)独立重建了各 12,000 条柔顺接触行。源端实际完成速度的 K/C 力残差 ≤**2.27e−14 N**；目标端由原始流形几何与完成速度重建的残差最大 **1.45e−4 N**，超过该检查事先设定的 **1e−4 N**。失败保留，未提高门槛，未确认其成因；原始流形距离与实际求解行的精度还需区分。这项失败不改写已完成的压缩／静载配对结果，也不授予逐点精确力律或整机资格。

权威入口为 [BE 足底检查点](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/condensed_free_be_contact_checkpoint.json)，执行代码、依赖、二进制与比较器见 [29 文件快照](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/attempts/condensed_free_be_v3_executed_code/verified_snapshot.json)。Python **60 项**、原生关节 **29 项**、默认 simulation **22 项**通过；10 项外部夹具未运行。[537 项工程结构检查](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/engineering_structure_free_be_checkpoint.json)、Rust 格式及 Git 空白检查通过。开发代码遵循 `crates/dev_tools/` 落点，临时脚本、模型、日志和快照全部位于项目备份目录。

**下一批转回凝聚整机 M0 接入**：先冻结原共享滑动脚垫的合力／力矩映射误差、摩擦与行程合同，再建立同版本 21 体源／目标候选并做冷重置、驱动、嘴约束检查。右脚无摩擦夹具没有左脚、背衬 CAD、原滑动脚垫动态等价或整机资格，不绕过自由度守卫；停止继续扫导向夹具、PGS 或局部微小精度。**M0 未通过，GPU／PPO／优化器使用为零。**

### 原滑动脚垫与共同压缩映射检查点

新的原结构源端夹具保留自由脚体和六个真实滑动脚垫的质量、完整惯量、K/C 及 1.5 mm 行程，用已测 BE 矩阵与预测地面行执行平放、±3° pitch／roll 五程序，共 **500 次 20 ms 积分**，无数值警告或行程越界。与 v3 独立点弹簧比较，侧倾过程最大零压缩支撑面高度差为 **0.08205 mm**，超过保留的 0.05 mm 门槛。v3 自身跨引擎吻合不等于原物理机构映射合格，拒绝将它装入整机。

源端开发模块 [shared_pad_contact.py](/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/crates/dev_tools/python/src/bevy_microduck_tools/goose/shared_pad_contact.py)引入版本 `goose_shared_pad_condensed_contact_v4`：每个原区域保留一个共同压缩状态，六个区域分别使用原 K/C、历史压缩和行程。接触点通过同一滑动方向与压缩坐标耦合；虚拟对角 `h*C+h²*K` 是数值离散量，不重复加入已经凝聚到脚体的脚垫物理质量。原上下行程约束与接触行在同一有界互补求解中处理，没有用独立点弹簧分摊共同压缩。

这份显式 CPU 实验以原生 MuJoCo 质量矩阵和光滑加速度建立求解，向原生积分器提交 `qacc`，每 Tick 仅调用一次 `mj_Euler`；没有直接改写 qpos／qvel、积分后位姿修复或时间子步。共同压缩历史只在该 Tick 成功积分后提交。计算内核可以表达一般广义质量矩阵，但运行夹具仍明确拒绝整机、摩擦和未声明约束，不能据此授予 21 体资格。

[独立映射比较](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/shared_pad_mapping_comparison_v4.json)中，v4 相同五程序再完成 **500 次积分**；零压缩支撑面高度最大差 **0.000212 mm**，内部压缩最大差 **0.000215 mm**，通过 0.05 mm 门槛。合力差最大 **0.001187 N**；以共同世界原点比较的力矩差最大 **2.779e−5 N·m**，描述性报告，不设置事后力矩阈值。静载总力误差通过 10% 门槛，原始质量、零压缩 COM 和完整惯量经独立并轴计算核对。实际原脚垫会随压缩移动，后续仍须分别报告动态惯量、低频及冲击误差，不继承高频等价资格。

权威入口为 [共同压缩映射检查点](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/shared_pad_mapping_checkpoint.json)。执行源码、模型和比较器见 [19 文件快照](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/attempts/shared_pad_mapping_v4_executed_code/verified_snapshot.json)；**67 项 Goose Python 回归通过**，包含共同压缩、释放历史、行程反力、偏心 COM 的完整质量响应及一次原生积分检查。全部新增源码和测试位于 `crates/dev_tools/python/` 对应目录，临时材料继续在项目备份目录。

**下一批实施 Rapier 同法则的共同压缩与行程耦合**，保持物理质量不重复、每 Tick 一次积分和原接触点冲量归属；按相同五程序先配对，再补左脚、摩擦、四档载荷、释放／冲击与背衬映射。随后把完整法则接入同版 21 体整机及全部原生嘴／限位约束。源端这五个名义半机重程序没有覆盖这些剩余项目；不重复独立点弹簧、PGS 或局部精度扫描。**M0 未通过，PPO／GPU 仍为零。**

### 共同压缩 Rapier 配对检查点

开发工具 [goose_free_sole_probe.rs](/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/crates/dev_tools/src/bin/goose_free_sole_probe.rs)新增显式 `free_shared` 模式，使用与源端 v4 相同的六个共同压缩状态、物理 K/C 和原 1.5 mm 行程。Rapier 原生接触行的实际有符号 J/WJ 与虚拟压缩、上下行程反力共同求解；不重复脚垫质量，不增加背衬盒，不改写刚体或几何位姿。既有有偏与松弛阶段各做一次代数求解，每 Tick 仍只作一次原生 20 ms 积分；积分成功后只提交一次用于位置积分的有偏压缩状态，重复提交会被拒绝。实验默认关闭，并保留六自由度、两链接、无其他关节约束、固定对侧、零摩擦及轴投影大于 0.5 的守卫。

[独立配对结果](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/shared_free_pad_paired_comparison_v4.json)覆盖平放、±3° pitch／roll，源端和目标端各 **500 次积分**。最大支撑面高度差 **0.0000243 mm**、内部压缩差 **0.0000143 mm**，通过既定 0.05 mm 门槛；目标静载相对误差 ≤**2.99e−7**，原始平方四元数范数误差 ≤**1.20e−7**，行程无越界。有偏与松弛候选的压缩差 ≤**2.23e−11 m**，互补残差 ≤**5.95e−10 m/s**。合力差最大 **3.224e−5 N**、共同世界原点力矩差最大 **2.553e−4 N·m**，均为描述性结果，没有设置事后力矩门槛。局部物理 P95 ≤**0.0624 ms**，不外推整机性能。

首个只读比较器使用时间严格相等，因源端累加时间与目标 Tick 乘步长之间 **1.33e−15 s** 的尾差失败；[原比较器与失败说明](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/attempts/shared_pad_time_evaluator_failure/failure.json)保留。最终核对使用既有 **1e−12 s** 源端时间容差，另严格检查积分数、原生 epoch 和实际步长，没有修改物理验收阈值。[默认路径控制](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/shared_pad_default_control_comparison.json)的 **500 帧全部字段**与原 v2 失败轨迹完全一致。

最新权威入口为 [共同压缩源／目标检查点](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/shared_pad_target_checkpoint.json)；[185 文件执行快照](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/attempts/shared_pad_target_v4_executed_code/verified_snapshot.json)核对实际源码、原生 fork、依赖、二进制和比较器哈希，并保存复现命令。原生关节 **30 项通过**，默认 simulation **22 项通过、10 项外部夹具未运行**，未改变的源端 Python **67 项通过**；**543 项结构检查**、Rust 格式及 Git 空白检查通过。临时日志、模型、轨迹、脚本和快照全部在项目备份目录。

**下一批补完整足底准入，再进入同版 21 体源／目标接入。** 原左／右脚物理清单、四档载荷、行程末端、释放／冲击及摩擦／接触方向耦合均需实测；不能删除局部守卫来宣称整机支持。当前只晋升为 `LOCAL_PAIR_ONLY`，没有整机、GPU、Warp 或高频动态等价资格。**M0 未通过，PPO／GPU／优化器使用为零。**

### 左右足底载荷、行程与释放／冲击检查点

[预先冻结的 16 程序协议](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/shared_full_foot_protocol_v1.json)分别读取原左右脚的质量、COM、完整惯量及六接触区，未用镜像副本代替左脚。每脚覆盖 5／10／20／30 N 每区、40 N 超载、30 N 加载后释放、30 mm 初始间隙与 −0.4 m/s 竖直冲击，以及 3° 侧倾冲击；外力施加于原接触区中心的平均位置，身体不导向、不保持。每程序 150 次积分，原滑动脚垫源端、凝聚源端与 Rapier 最终各 **2,400 次 20 ms 积分**，共 **7,200 次**，无数值失败。开发探针接受显式冻结协议并逐 Tick 保存轨迹，原五程序入口保留。

首批冲击夹具把 Rapier 自由关节的局部平移速度当作世界速度，3° 侧倾初态多出了约 0.020934 m/s 横向速度；压缩检查没有捕获这项输入错误。[首批轨迹、错误说明、实际源码及二进制](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/attempts/shared_full_foot_initial_velocity_failure/failure.json)保留，明确拒绝其冲击资格。修复采用已验证的逆旋转坐标变换；最终比较额外核对实际原生身体 Jacobian 的世界初速度与源端一致，未修改冻结物理门槛。

[最终独立比较](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/shared_full_foot_comparison_corrected_v1.json)中，16 程序通过冻结的压缩、四档静载、行程、旋转与卸载压缩检查。源／目标最大支撑面高度差 **0.00170 mm**，内部压缩差 **0.000106 mm**；原机构与凝聚模型在四档静载程序中的最大支撑面差 **0.0000419 mm**，曲线误差 ≤**0.0125%**。40 N 超载达到原 1.5 mm 末端并产生上限反力；最大越界 **1.27e−10 m**。释放末尾三种实现的压缩均接近零。源／目标合力差最大 **9.16e−5 N**、共同世界原点力矩差最大 **0.001279 N·m**，分别报告；局部物理 P95 ≤**0.1212 ms**，不外推整机。

原机构与凝聚模型的内部压缩差最大 **0.000748 mm**，但卸载后的自由飞行支撑面位置差达到 **0.9393 mm**；该动态差异明确保留，不授予原机构动态轨迹或高频等价。释放能量记录使用积分前物理质量度量下的完成速度，加已提交的弹簧能量，最大相邻增长 **1.17e−10 J**；这项描述性记录不等于完整积分后物理能量准入。

最新入口为 [左右足底检查点](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/shared_full_foot_checkpoint.json)，状态为 `FRICTIONLESS_FOOT_MATRIX_ONLY`；[234 文件执行快照](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/attempts/shared_full_foot_corrected_executed_code/verified_snapshot.json)保存实际源码、模型、依赖、协议、比较器和二进制。[控制结果](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/shared_full_foot_final_control.json)确认最终二进制的默认 500 帧全部字段与原失败轨迹一致；源端只补初态元数据，原结构和凝聚各 2,400 帧物理字段与首批完全一致。发布构建、格式、Git 空白及 **543 项结构检查**通过；本轮未修改原生物理核心或 Actor 合同。

**下一批实施摩擦及接触方向耦合，再接入整机关节／嘴销／限位。** 当前仍是分开的六自由度无摩擦足体，不能表达任意倒地接触或移动物体对侧，也不能通过删除守卫直接装入 21 体。**M0 未通过，PPO／GPU／优化器使用为零。**


### MuJoCo 3.13 原生 discrete 源端检查点

MuJoCo **3.13.0** 新增原生 `discrete`，让线性刚度、阻尼与约束在同一个有效惯量度量中求解，值得用原结构做一次有界对照；它仍有自己的接触、限位和非线性数值含义。依据 [固定版本更新说明](https://mujoco.readthedocs.io/en/3.13.0/changelog.html)和 [计算说明](https://mujoco.readthedocs.io/en/3.13.0/computation/index.html)，不由“支持积分器”推断物理或 GPU 准入。

源码位于 [native_discrete_runtime.py](/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/crates/dev_tools/python/src/bevy_microduck_tools/goose/native_discrete_runtime.py)。新候选 `goose_460_full50_discrete_v3` 保持原 33 体、18 主动轴、14 被动坐标、10.430762603 kg、全部 K/C/armature、四杆几何、摩擦和碰撞过滤。模型 SHA 为 `1de4fb3b98d04dae6b1788b571bfa14dfa93ca9c0a7a3befdb11a9c9ed68e606`，合同 SHA 为 `857be912c635df35f93a87615ff235c0b8d0300034dae9fcfb9ca37b67a507b9`；模型、依赖、轨迹和脚本全部在备份目录。新旧引擎之间 **55 组编译物理输入与 qpos0 FK** 数组哈希完全一致，身体、关节和电机名称及顺序一致。

[项目 Python 3.12 的独立评价](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/native_discrete313_py312_evaluation_v1.json)核对 **96 个脚垫程序、9,600 次积分**和 **20 次名义冷重置、2,000 次整机积分**。每 Tick 只有一次真实 20 ms 积分，整机同时只有一次力矩更新；没有 Actor、优化器、初始化后的坐标改写或自动重置。5／10／20／30 N 的静态压缩相对误差最大 **1.27e−16**，无数值警告。但孤立端点程序最大越界 **1.713 mm**，整机脚垫最大越界 **2.411 mm**，均超过原行程要求；继承既有端点容差，没有放宽门槛。因此 **源端物理未通过，M0 未通过**。

整机嘴销最大距离 **0.4782 mm**。运行时记录的接触属于积分前时间边界；另在独立数据实例中重放六个已保存的完成姿态，只重算运动学和碰撞，没有添加积分或改动原轨迹。最大的 **28.571 mm** 是颈部壳体与头部上壳代理之间的自碰撞；对应来源方法为 `source_convex_hull` 与 `local_surface_clusters_5mm_approximate`。须核对原 CAD 的空腔及真实干涉面，不扩大排除名单。采样姿态的地面接触穿透和自碰撞分别保留。全 CAD 的源端物理加控制器 P95 约 **27.70 ms**，仅为此 CPU 诊断的描述，不外推 Bevy 或策略性能；2 秒、重复同一名义出生也不授予站立或随机恢复资格。

首批隔离环境误用了 Python 3.13.15，与项目 `>=3.12,<3.13` 不兼容，记录及实际执行源码保留。最终锁定 **Python 3.12.14、MuJoCo 3.13.0、NumPy 2.5.3** 后复跑，**116 份 JSONL 物理轨迹哈希与首批完全一致**。运行时增加 Python 与引擎版本检查。旧引擎回归 **69 项通过、3 项新引擎专属检查跳过**；正确环境的新版本检查 **4 项通过、1 项旧引擎专属检查跳过**。首个新环境宽范围测试触发 3 项旧 MuJoCo 3.10 ABI 的主动拒绝，日志保留，未取消旧版本锁或宣称跨版本兼容。

当前权威入口为 [原生 discrete 检查点](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/native_discrete313_checkpoint.json)，状态为 `SOURCE_NATIVE_DISCRETE_LIMITS_CONTACT_UNQUALIFIED`。下一批先对 **原生限位激活／约束参数**做一次版本化短对照，并检查已定位碰撞代理的来源及真实表面；保留原物理 K/C、质量和完整惯量，不重复精度、PGS 或局部嘴部扫描。共同压缩路线的摩擦与任意接触方向仍是未通过项，保留其守卫。源／目标同版本整机物理准入、真实 GPU 小批量和 PPO 更新之前，训练链路继续锁定。**GPU、PPO、优化器更新为零，长期目标尚未完成。**


### 预测限位与精确约束对角检查点

源端开发模块 [predictive_discrete.py](/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/crates/dev_tools/python/src/bevy_microduck_tools/goose/predictive_discrete.py)为原 32 个标量关节的两个真实端点建立原生约束行。每 Tick 使用一次前向预览、一次修正代数求解和 **一次 20 ms 原生 Euler 推进**；预览不推进时间，关闭 Euler 的二次阻尼，没有逐 Tick 写入 qpos／qvel。约束修正在 native effective metric 建立后执行。固定版本 [原生前向源码](https://github.com/google-deepmind/mujoco/blob/3.13.0/src/engine/engine_forward.c)支持这一阶段边界；这些公共阶段的 CPU 组合不构成 Warp 兼容声明。

[推进映射对照](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/discrete_predictive_final_parity_v1/receipt.json)在关闭预测限位修正的控制路径中，对八个原脚垫程序及十个保存整机初态进行 **1,620 次真实积分**，qpos／qvel／qacc 与原生 `mj_step` 的差均为 **0**。额外代数求解没有增加时间子步；相同 65／18 驱动、延迟、热代理和功率限制保留。传感器回归另核对前向预览产生的 RNE 缓存被最终约束解正确失效，不将机器人没有的力传感器当作本体验收证据。

近似对角候选 v4 的行程已改善，但整机关节仍有 **3.743e−5 rad** 越界、嘴销 **0.1792 mm** 误差、P95 **57.48 ms**。同一 100 Tick 控制中只开启 `diagexact`，闭合和收敛明显改善，继而冻结独立候选 `goose_460_full50_discrete_limits_exact_v5`。该选项在当前有效惯量的 factored backbone 上计算约束空间对角；不包括额外 tendon／actuator／flex 耦合，见 [3.13 选项定义](https://mujoco.readthedocs.io/en/3.13.0/XMLreference.html#option-flag)。本机器人新增限位 tendon 的 K/C 为零，原驱动仍为直接力矩。

[最终独立评价](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/discrete_exact_limits_evaluation_v1.json)核对 **96 个脚垫程序、9,600 次积分**和 **20 次名义冷重置、2,000 次整机积分**。全部有限、无警告；32 组编译物理数组与原生 v3 一致，包括质量、完整惯量、物理 K/C/armature、几何、摩擦和碰撞过滤。原关节区间保留，限位行表达和求解选项明确版本化。孤立脚垫最大行程越界 **4.208e−11 m**，整机滑动轴 **1.032e−10 m**，旋转轴 **7.272e−9 rad**；四档静载曲线相对误差 **1.27e−16**。嘴销最大距离 **0.696 μm**，源物理加原控制器 P95 **7.16 ms**。模型 SHA `a8f0abe37bcab6883c6e713de1d4c3a443e41a0fa6389a29591497428f1891d1`，合同 SHA `04f0381c5f46bafc12acd4825a964385784d790fc1b675bf2faf75d876f32076`。

独立进程重算 100 个完成姿态的 FK／碰撞，没有新增积分或修改轨迹。足底地面最大穿透仍为 **13.048 mm**，自碰撞最大 **3.432 mm**，最深配对为右胫骨惰轮叉板与右脚装饰件；两者分别保留；**源端接触与 M0 未通过**。末帧直立度约 0.9992、COM 高度约 0.28393 m，只是重复名义出生的 2 秒无 Actor 诊断，不能授予 60 秒站立或移动／恢复资格。此前颈部壳体来源审查确认它是厂商尺寸圆柱包络而非厂商 CAD，不能凭猜测开空腔、删碰撞体或扩大排除名单。

两个新地面接触实验各 **100 次积分**，改变地面接触法则，未修改冻结 v5 或授予资格。只加强已激活接触时穿地达 **18.389 mm**；运行时改写检测余量的短对照曾记录 **0.328 mm／21.19 ms**。[原收据](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/diagnostics/discrete_ground_predictive_activation_diagnostic_v2/receipt.json)保留，但后续独立审查发现 `body_margin` 广相缓存未更新，只检测到 48 个地面接触；正确编译的 10 mm 余量检测到 608 个，独立刷新缓存后接触签名与编译版本一致。因此这组较好数字**不能作为完整编译接触配置的证据**，由下方 v6 结果替代其推论。

本轮旧引擎 Goose 回归 **70 项通过、7 项新引擎检查跳过**，新引擎专项 **4 项通过、1 项旧引擎检查跳过**；[552 项结构检查](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/engineering_structure_discrete_exact_checkpoint.json)通过。代码和测试位于 `crates/dev_tools/python/`，运行时合同仍为 [robot/src/goose/contract.rs](/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/crates/modules/robot/src/goose/contract.rs)。实验模型、脚本、日志、收据和执行快照全部在项目备份目录。**GPU、PPO、优化器更新为零，长期目标尚未完成。**

### 接触实验收口与成熟源链路切换

`goose_460_full50_discrete_ground_v6` 将 10 mm 检测余量写入 XML 后编译，保持真实表面为接触边界，仅修改静态地面的约束参考与正则；原椭圆摩擦锥比例、刚体和驱动参数保留。新守卫拒绝缓存不一致的配置。[收口检查点](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/ground_discrete_diagnostic_checkpoint.json)核对 **2,500 帧实际积分轨迹及哈希**：v5 控制 100、v6 名义冷重置 2,000、三种冲击 300、缓存守卫后复核 100。

v6 冷重置地面穿透 **2.504 mm**、源 CPU 物理加控制器 P95 **180.17 ms**；冲击最大 **3.333 mm／120.40 ms**。已测法向非黏附、摩擦锥及线性化完成速度下的切向耗散检查通过，但不能由此推断完整能量、原物理等价或物理准入。守卫前后的名义物理轨迹逐值一致。历史 v5 约 13 mm 地面和 3.43 mm 自碰撞来自 **MuJoCo CPU**，不当作 Bevy 游戏实测。Rapier 构造已采用逐块 `convex_hull` 和 compound；MuJoCo 常规 mesh 也使用凸包碰撞，见 [MuJoCo mesh 说明](https://mujoco.readthedocs.io/en/3.13.0/XMLreference.html#asset-mesh)。问题不是未打开 convex。

本分支到此冻结为 CPU 诊断，不继续增加自定义地面法则或扫参数。新原生参考复用已存在的凝聚 MJCF 和质量合并，只采用成熟栈支持的 `implicitfast`，保持 `training_release=false`；先接通 `MjSpec`／`EntityCfg` 原生模型，再整理可训练碰撞代理与源端物理准入。流程依据 [MicroDuck 源配置](/home/ethan/ProjectBackups/2026-10-01/Sai_Lab/microduck_official_velstand_source_001/source_5946/pyproject.toml)、[mjlab 架构](https://mujocolab.github.io/mjlab/main/source/architecture_overview.html)和 [官方自定义机器人示例](https://github.com/mujocolab/anymal_c_velocity)，只增加 Goose 必需的适配。

用户授权的 Bevy 性能子 agent 首轮 [报告](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_bevy_performance_001/cycle_01_report.md)使用相同前 20 Tick 的两次测量：原生 step 均值 **210.93 ms**，约束组装约 **80.1%**，快照仅 **0.215 ms**。计时开／关完整状态、力矩、接触逐值一致，5 项原生回归通过；只增加开发探针可选计时，未修改运行时算法。下一性能周期跟随主线冻结的新本体／代理，避免无收益的小优化。**M0-S／M0-T 尚未通过，PPO／GPU／优化器使用为零。**

### 原生 mjlab 模型接线检查点

[mjlab_baseline.py](/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/crates/dev_tools/python/src/bevy_microduck_tools/goose/mjlab_baseline.py)复用冻结的 21 体凝聚 MJCF，通过上游 `MjSpec`／`EntityCfg`／`XmlActuatorCfg` 接入；场景平面交给任务场景，XML 电机保持 effort 输入。按实际电机目标绑定 18 轴，第六轴为 `beak_input_rotor`，不把嘴输出关节误作驱动轴。65 维观测及目标动作控制器仍沿用 Goose 合同，尚未接入 PPO 任务注册。

已锁定 Python 3.12、mjlab 1.3.0、MuJoCo 3.10.0、MuJoCo Warp 3.8.1、Warp 1.12.0、RSL-RL 5.0.1、Torch 2.9.1。依赖与锁文件归属现有开发工具 Python 包；本地验证采用隔离 CPU 环境，DGX 的 aarch64 Torch 指向官方 cu129 索引，尚未使用 GPU。旧 CPU 3.13 实验保留独立环境，不能通过改依赖混入原生训练链路。

[接线检查点](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/mjlab_baseline_001/bootstrap_checkpoint.json)中，新参考的 22 组物理数组与原生凝聚父模型一致，上游 Entity 封装后的 18 组身体、关节、机构和驱动数组一致，电机名称／顺序不变。质量仍为 **10.430762603 kg**，21 体、18 主动轴、2 被动坐标。沿用 Goose 控制器完成 **10 次 20 ms 原生积分及 10 次力矩更新**，65 维观测有限，无数值警告。这只有 0.2 秒接线证据，未授予站立或源端物理资格；原碰撞几何和未校准足底仍待处理。

最终 **76 项 Goose Python 回归通过、11 项专属新引擎检查跳过**；5 项 Bevy 原生回归、Rust 格式和工程结构检查通过。下一项明确是**从已核验来源整理任务碰撞代理并冻结一份原生训练 MJCF**，保持总质量／完整惯量、支撑轮廓、嘴接触与空腔；完成有界源端物理筛查后接上游任务管理和一次 PPO smoke。不上新的通用框架，不继续扩展 CPU 地面修正。

### 成熟碰撞工具的有界筛选

[碰撞代理试验检查点](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/native_proxy_pilot_checkpoint.json)保存 26 份证据及六份实际执行源码。复用已安装的 CoACD 1.0.14，使用[官方实尺度模式](https://github.com/SarahWeiii/CoACD)，按米声明凹度阈值，不自行实现分解器。每个零件先冻结来源、参数、种子、CPU 线程、墙钟预算和几何检查；不修改运行模型、物理参数或碰撞过滤。

- 右脚饰件：原 279 块 → 184 块，分解约 4.26 秒；4,384 个源表面采样的最大缺口约 **0.106 mm**，4,511 个有源表面间隙的空腔见证均未填堵。双精度布尔检查的多占／缺失体积约为源体积的 **1.451%／0.0177%**。只通过本零件的局部几何筛选，未取得整机碰撞或性能资格。
- 右胫骨叉板：原 100 块 → 167 块，分解约 57.81 秒；局部几何筛选通过，但块数增加，拒绝作为性能候选。不能以成熟工具名称保证全量重跑会提速。
- 左前躯干壳：普通 QEM 的 20,000 三角减面未保留闭合正体积；Manifold 保拓扑简化后仍有 **0.164 mm** 采样缺口，超过提前冻结的 0.1 mm 门槛，布尔体积恒等式也未达到冻结容差。均拒绝，未进入分解或模型晋升；一次只读数组接口失败及仅增加可写副本的修复另存。

独立审查发现通用 Trimesh 布尔封装会将顶点转换为 float32，薄脚环的体积恒等式误差约为源体积的 0.591%；改用原生 Manifold `Mesh64` 后误差降至约 4.33e−20 m³。原收据保持不变，采用单独的[双精度复核](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/coacd_native_proxy_001/double_precision_audit.json)，不把体积和有限空腔采样当作全局几何证明。

本轮没有新增积分、策略或 GPU 使用，也没有冻结新的全身碰撞模型。下一周期回到**来源刚体归属、任务接触表面与剩余碰撞成本**，形成可复核的模型简化方案；保留空腔、运动干涉、脚和嘴见证，复用成熟几何工具，不继续扫外壳减面参数或全量原样重跑 CoACD。M0-S／M0-T 未通过。

### Bevy 性能子 agent 第二周期

性能子 agent 继续复用同一聊天中的任务，按用户授权长期保留，每轮有界工作后闲置。第二轮只修改原生 Rapier 接触 Jacobian 缓冲增容：nalgebra 0.35.0 的 mutable resize 会克隆整个旧向量，改为转移所有权后调用同一个 resize，保留旧前缀和新增零值。几何、过滤、求解参数、驱动、观测及每 Tick 一次 20 ms 积分均保持原值。

带相同实际冲量签名工具的前 20 Tick 配对中，**原生 step 均值 156.62→52.45 ms；约束组装均值 114.70→10.79 ms；探针物理墙钟 P95 950.16→80.46 ms**。主线[独立复核](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/independent_performance_cycle02_review.json)的总物理墙钟均值为 **157.08→52.89 ms**，与原生 step 区分。初次未加冲量签名的另一对照为约 191.95→52.49 ms，说明冷启动及主机负载影响计时；不把不同程序的数值拼成同一成绩。

前后、计时开／关、重复程序的实际关节、身体、力矩及接触字段一致；只按唯一 body 名规范化无序几何读出。20 Tick 的真实法向／摩擦冲量、热启动及力臂 SHA 签名完全一致，每 Tick 有 30–74 个非零冲量点。默认 simulation 回归 **28 项通过**（22 单元、2 armature、4 预测限位），10 项依赖外部冻结夹具的既有检查未运行；工程结构 **564 项通过**。七次配对与重复试验共 **140 次实际积分**，完整证据见[周期报告](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_bevy_performance_001/cycle_02/cycle_02_report.md)及[核验收据](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_bevy_performance_001/cycle_02/cycle_02_audit.json)。这是性能改动回归，不授予原候选物理资格。没有 Actor、PPO 或 GPU，仍未达到 16 ms／持续 50 Hz 门槛。

本周期到此收口，不继续追逐下一条内核热点。下一次性能周期根据新碰撞候选或新的可检验瓶颈派工，保留固定基线、真实物理结果对照与默认功能回归；无安全收益时不采用改动。

### 任务几何整理与原生凸包导出修正

同一刚体、相同接触参数下，仅删除全部顶点被保留凸包包含的碰撞体，15,715→15,585 块，减少 **130 块**。[独立核验](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/native_containment_001/conformance/receipt.json)确认 24 组物理数组、15,670 个保留命名几何及 130 个包含关系；质量和驱动不变，脚垫和夹持面保持。父模型与候选各完成 100 次真实积分／力矩更新，无警告，但两者的零动作程序均未站稳；2 秒短程序的 P95 几乎没有改善。几何占据集合相同也不授予接触力或长轨迹等价资格。此候选只归档，不作为主要优化路线。

右脚刚体的 1,049 个既有源凸包通过 Manifold `Mesh64` 合并耗时约 **0.24 秒**，六个足底区域未参与合并。对全部合并网格直接运行 CoACD 后在约 **279 秒**发生原生崩溃（退出码 −11），无分解或模型晋升。[拓扑审查](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/native_body_union_001/topology_audit.json)发现 **186 个连通分量、128 个凸分量**。此轮关闭，不原样重跑；下一轮采用工具支持的逐连通分量流程，只做一个有界可检验筛选，保留空腔和任务接触见证。合并对象是既有碰撞包络，不能代替原 CAD 忠实度审查。

[native_geometry.py](/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/crates/dev_tools/python/src/bevy_microduck_tools/goose/native_geometry.py)修正了迁移导出：MuJoCo 3.10 的网格保留原顶点，但碰撞支持点按编译凸包图选择；`maxhullvert` 会限制凸包，少于 10 顶点时仍使用原顶点。依据[固定版本原生实现](https://github.com/google-deepmind/mujoco/blob/3.10.0/src/engine/engine_collision_convex.c)，导出只采用实际支持点，保留原顶点顺序和编译后坐标变换，显式锁定引擎和导出版本。真实 64→8 顶点夹具的原生距离查询确认：旧全顶点导出会在源端仍分离的位置产生碰撞；新导出与源查询一致，同时覆盖不限顶点及小网格回退。

新的[冻结导出](/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/native_support_export_002/receipt.json)仍有 **15,727** 个碰撞体；1,398 个网格减少支持顶点，另有 337 个网格的原顶点超出源凸包超过 20 nm，最大平面距离约 **2.976 mm**，其余为冗余顶点删除。导出 SHA `c470d10e4349d3f27e5f9d7e469e72bd112c4d3f1653d709dd143f23771448d8`；质量、关节、物理系数、摩擦、过滤和坐标变换不变。原始 XML、合同、旧导出和历史对照保持冻结。旧目标性能结果继续属于旧导出，不能继承为新导出的物理资格；目标端必须按新哈希复核。该修正也不解释全部既有穿透。

最终 **81 项 Goose 回归通过、11 项新引擎专项跳过，570 项结构检查通过**。本轮源端共 200 次真实积分，没有 Actor／PPO／GPU；M0-S／M0-T 仍未通过。性能子 agent 第三周期仅在新导出上做有界计时与物理结果回归；主线继续任务 MJCF，不扩展自定义求解器或继续扫描整脚分解参数。

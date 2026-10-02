# Goose V0.1｜Bevy Sim2Sim 长期训练计划

版本：2026-10-02。实施分支：`codex/goose50_training`。

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

Rapier 的 `num_solver_iterations=1`、每体 `additional_solver_iterations=0`、`max_ccd_substeps=1`。只能调整内部 PGS 收敛轮数（预定 4/8/16/32），不能增加实际积分子步。弹簧采用 ForceBased SI 单位，碰撞体不重复增加质量。

嘴部采用实际四杆闭合约束，电机只驱动 `beak_input_rotor`，与 head_roll 产生相反反力，约束将力传给 jaw/coupler。禁止逐 Tick 写 qpos、FK 搬动物件或焊接附着辅助抓取。

任务用碰撞代理保留脚、嘴、壳及运动干涉关键表面、来源和过滤映射；不填实空心结构，不扩大自碰撞排除范围凑成功。原邻接过滤、显式嘴销配合排除与碰撞 masks 均可追溯。

### 训练与目标接入

**MuJoCo 50 Hz → mjlab／MuJoCo Warp + RSL-RL PPO → Rapier 零样本评估 → 必要的有界目标微调 → Bevy CPU ONNX 独立验收。**

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
| M0 物理接入 | 两种本体、逐轴、嘴约束、软底载荷、碰撞代理；同步地面可达性与带载静力筛查 | 同版本至少一份源/目标均过物理门槛 |
| M1 链路 | 锁依赖、小 batch GPU rollout、真实 PPO 更新、复载、ONNX 对照、Rapier batch | 真实积分和优化器更新、有限数值与完整收据 |
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
- 保存模型/合同/代码/依赖/配置/seed/策略哈希；候选显式指定，不用 latest。M0未过禁止长训，数值/约束/容量异常停止该批。
- 每关交付通过/失败/适用范围、录像、算力和下一实验，基于真实吞吐滚动估时。
- 当前实施聊天的 `Goose ProjectManager` heartbeat 每两小时检查阶段产出和失焦，不新建聊天；首检查点是计划、50 Hz合同和首次M0真实证据，下个检查点是门槛矩阵、失败归因和下一候选。

当前M0实施中，尚未启动PPO。原始产物归档 `/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/`；本页为唯一计划和状态入口。最终交付冻结策略、本体/合同、Bevy包、复现命令、三主线报告及未通过清单。制造/电气/实物载荷仍由机器人工程侧维护；仿真资格不自动等于实物资格。

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

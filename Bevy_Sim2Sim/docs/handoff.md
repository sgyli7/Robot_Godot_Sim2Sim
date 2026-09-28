# Bevy_Sim2Sim 实施交接

更新日期：2026-09-28。用户已授权逐步实现 [当前实施计划](implementation_plan.md)，并行推进 Rust 工程、Sim2Sim 模型与验收链、科学站和墨比斯渲染。

## 阅读顺序

1. [唯一工程规则](../bevy_engineering_rules.md)。
2. [当前实施计划](implementation_plan.md)：全部目标、版本、接口语义、预算与最终门禁。
3. [历史研究](research.md)：此前 Godot200 Hz调查，仅供定位旧代码与失败经验。
4. [运行说明](../README.md)：仅以实际验证后的命令和功能为准。

## 当前授权与范围

MuJoCo训练、Rapier部署；完整Pollen/mjlab本体与BAM，目标端物理和策略60 Hz且不隐藏高频时间子步。实现九项技能、Sprint、腿/轮切换、完整科学站；嘴部真实抓放、维修站、SaiRobot及其他平台真机验收后置。旧Godot200/50策略不能视为新引擎合格模型。

用户允许按规则自主决策，疑问集中写入临时问题记录。根GPT亲自审核MJCF来源的完整三维时间轨迹与对应真实视频；代码测试或评估器通过不能代替此审查。

## 工程与状态

当前建立隔离工作树并行实施，尚无完整游戏或新技能通过结论。固定步、单次 Rapier 积分和 armature 原语已经实际测试；BAM 内核与上游 Torch 512 组输入核对通过。两种完整本体已经在同一原生 Rapier 世界中实际组装：腿式 15 个物理 body／14 个 hinge／11 个碰撞 geom，轮式 19／18／13，含四个被动轮。HOME 与非零根姿态／关节姿态的位姿、质量、质心、完整惯性张量和被动参数已做零积分对照。

实际验证发现零质量 collider 的非单位局部变换会触发小尺度惯性张量重新对角化误差。保留原形状和 geom 变换为单子项 compound、让 collider 局部变换保持单位后，导入张量与源端差降到约 `2.6e-11`；没有改动后端物理公式。单子项 primitive 接触几何已测，但完整机器人接触／CCD 等价仍未验收。独立 MuJoCo 原生编译对照另发现自行重建凸包存在约 `4.6e-6 m` 的支持面差异，未通过预置 `1e-6` 门槛。现直接导入原生凸包拓扑，四组腿／轮姿态的实际支持面采样最大差约 `1.42e-7 m`；六种缺失／损坏凸包均实际拒绝且原有世界对象数量保持不变。原失败记录保留，采样对照不替代完整接触动力学验收。

已提供只读 `RobotPoseFrame` 与 `RobotAssembly.pose_frame`，把唯一世界快照的完整、带 generation 的 body handles 映射到源 body IDs，不运行第二套 FK 或物理。四组实际初始化均输出完整帧，缺失、重复、错模型、非有限、非单位旋转及实际删除重建后的旧句柄均拒绝。完整碰撞过滤、外载观测和控制生命周期尚未接通，结构构建不能作为目标 plant 合格依据。

源碰撞 eligibility 独立原生 oracle 已完成：256 个非对称掩码组合、8 个腿／轮与 world 案例和 6 个拓扑案例，共完整枚举 29,396 个候选 pair、记录 638 个原生接触。所有球真实共位且半径为正，全部原视觉 geom 编号保留；完整重放逐字节相同。不可变 typed profile、实际 JSON／RON 回读和身份篡改拒绝已完成。封闭的 `SourceCollisionWorld` 已接入完整 body／collider／joint generation registry、冷安装的 contact／intersection hook 和逐步准入；两种原完整本体的 7,156 个 source pair、103 个 eligible pair，以及删除重建和错误输入已做零积分核对。

第一批运行验证另真实进入 Rapier pipeline 118 次，逐次 before／after 落盘，61 项预设门禁中 58 项通过、3 项未通过。腿式和轮式原几何都观察到非零 warmstart；各自 8 次历史积分后删除重建，与新世界的 6 步 body／joint／contact 位串相同。16 项首步 CCD 只是在球体机制替身上通过；4 项原本体 post-pipeline 故障正确将已发生的积分计入账本并使 token 失效。两项静止球体的允许接触虽有真实 hook 和 active manifold，实际冲量仍为零，故未满足正冲量门槛；睡眠热挂负控也未达到预想的跳过前提。测试因此按严格门禁 exit 101，原失败证据保留。根独立复核原账本、17 项冻结源码身份和实际结果。独立补充批在相向速度的球体替身上再进入 8 次，两种插入顺序的允许接触均有真实 hook 和约 `0.5236` 的法向冲量，mask／父子拒绝均无 active contact；根复核新旧账本连续且旧失败未重分类。累计 126／128 次，余下 2 次保留；这不证明原几何受力、睡眠热挂或完整 plant。生产环境、地面、prop 和 sensor 尚不允许登记，不能视为完整接触／CCD 或目标 plant 合格。

源端 v6 已由根 GPT 重新亲审完整自然首回合的 76 帧录像、全部本体时序和 240 项文件身份，75 次真实 CPU ONNX 推理对照通过。准入与五项反例均实际执行。仅授权的一次 PPO 更新已实际发生，但随后重置遇到 inference-mode tensor 错误，整次结构 smoke 仍记录失败，真实预算约 41.18 GPU 秒；原错误与已更新 checkpoint 保留。v7 在全新环境恢复该权重，零追加学习，以约 39.69 GPU 秒完成首个自然回合：62 次积分／真实推理、63 帧，全部策略／归一化／优化器状态不变。根 GPT 已亲看全部帧和本体时序、独立核对 246 项文件身份；62 次真实 CPU ONNX 推理误差最大约 `1.49e-7`，通过 `1e-5` 门槛。策略仍在约 1.03 秒自然倒下，没有合格技能。

v8 已补齐严格的审查范围、学习许可、单次更新限额和原子单次授权消费；失败与超时不返还许可。BAM 与 MDP 缓存的 inference-mode 重置边界已统一修正并通过 CPU fixture，真实合并 learn→reset 生命周期仍未重新执行。v9 将实际安装包和依赖字节纳入准入：129 个 distribution、42,960 个非 bytecode 文件、约 7.56 GB，parent 与 worker 在首次科学调用前实际复读；52 项源 CPU 测试通过，安装包 45 项通过、7 项科学依赖测试明确跳过。根另核对 101 项封存证据和 15 个模块的源码／wheel 字节。当前依赖 receipt 的 `complete`／`native_profile_complete` 仍为 false，不能据此开始学习；实际 CUDA lazy library／JIT／外部程序闭包仍待固定 profile 采集。v7 的 request_changes 审查、失败记录和原预算账本保留，不复用旧候选资格。

v10 的固定零更新 profile 工具已通过 CPU 测试并冻结 126 项输入身份，但独立只读审查发现追踪器可能忽略成功元数据／失败候选路径查找，180 秒限额也没有覆盖子进程退出后的解析与封存。因此 v10 未运行一次实际 source/GPU 发现，也未取得完整依赖闭包；其 `native_profile_complete=false`、`learning_allowed=false` 不变。v11 工具已修复追踪与越时误报并独立核对 37 项冻结产物、31 项快照文件；最终安装包 CPU 测试 81 项通过、7 项科学环境用例跳过。但任意文件系统／账本阻塞下的字面 180 秒硬返回仍无法证明，正常失败路径查找也缺少冻结的负查找契约，故根不批准 v11 实际 source/GPU 发现或学习。不得复用 v10 的单次许可或把旧 `source_rollout_evidence.json` 的局部 `passed` 解读成科学闭包合格。

v12 在 CPU 假进程树上分别验证了精确 cgroup leaf 终止与 systemd 的 5 秒自主截止；leaf 迅速被回收，因此没有字面读到 `populated=0`，也未验证 GPU 上下文清空。有限负查找工具对旧 v10 trace 的 80 次已声明只读失败作后验诊断，45 条规则逐条匹配，但仍有 17 项未解决，且其中 8 条当前目录见证已漂移，历史契约不能用于新运行。v12 的 94 项冻结文件与 33 项快照、20 个模块的 wheel／源码身份经根与独立审查；安装包测试 96 通过、7 跳过。独立审查发现预算封口的两次提交间可短暂写出错误完成态。v13 改成 CPU-only 单事务失败闭合：发现任务无论请求何种终态，均只可持久化 `budget_exhausted`，实测下界与保守占用预算分字段，通用账本 `finish` 不能绕过此门禁。根复核 45 项冻结产物、34 项快照和 20 个 live／wheel／sdist／installed 模块，独立重跑 13 项定向测试；安装包全量记录为 109 通过、7 跳过、62 个子测试通过。v11/v12 冻结身份未改。v13 仍没有正向完成协议，公共 `discover_source` 入口硬拒绝，`native_profile_complete=false`、`learning_allowed=false`。没有新增 source、MuJoCo、GPU 或 PPO 运行，也没有消费真实账本／review。下一阶段须把预封的负查找、scope 与一次性许可绑定新 v3 输入身份，并解决余下 trace 诊断；见 `.scratch/model_workflow/root_v13_cpu_review_v1.json`。

0.1.9 新增只读的 v3 CPU 输入身份和 review 核对，把旧输入的实际字节、有限负查找契约及精确但尚不执行的 scope 计划绑定；核对结果明确 `single_use_enforced=false`、`source_execution_allowed=false`、`learning_allowed=false`。先前的本地单次领取试验被独立审查复现目录替换、锁 inode 替换和审查到期竞态，已从交付包删除。随后独立审查又找到旧私有 `profile_worker` 可经有效 v2 绑定触发 source；该执行体及模块入口现均立即硬拒。保留的 `reserve_discovery` 仅供历史 CPU 记账测试，显式调用仍可能写账本，不能作为正向许可。新安装包完整测试为 124 通过、7 跳过、62 个子测试通过；没有新的 source、MuJoCo、CUDA/GPU 或 PPO 运行，也未消费真实预算。现阶段仍需可信的一次性许可、完整负查找与真实依赖闭包，才能重新讨论发现任务。

Rapier `0.35.3` 的本地 vendor 仅增加默认关闭的只读观测功能，原始发布包身份与许可证保留。阶段 A 的 152 条物理记录在开关功能前后逐位相同。阶段 B 已在记录的接触原语中完成原生 normal／两个 tangent 行的独立完整性核对：开关前后 1713 条物理位串相同，2831 次实际步调用、1561 次 owner-step 检查。根已审读收集器和独立选择审计并重新核对位串哈希；parallel／多个时间子步及不完整数据仍拒绝发布。此原语结论不能提供完整 BAM 外载或整机动力学资格。

用户授权渲染线充当地编，科学站 v4 已冻结为新的空间基线：主广场整理为面向主舱的院落，四个功能区各补四组设施，连续沙地取代色斑和机械网格，缓坡按全宽检查净空与支撑。六个真实 GPU 机位均成功输出并由根 GPT 亲审，四项共享几何检查和四种受控故障退出通过。v5 的六处导向文字经独立 typed RON 与允许变更集合复核；根和场景 Agent 均亲审六张真实 1080p 画面，接受入口导向和局部标题改善。观测区标题仍有灯具局部遮挡，小铭牌仍偏小。v6 利用已有设施表面增加五处功能图形，三块样本架盖板改为紫色；场景 Agent 独立审计原 811,452 个表面三角形、2,496 个 collider payload 不变（仅盖板 36 个三角形变更材质），22 项渲染库测试通过、2 项忽略。根冻结源码与资产，整项目离线构建、六个真实 1080p GPU 机位成功并亲审，接受局部功能辨识改善。v5→v6 总览仅 331 个像素变化，因此另做纯 shader 空间层次：11×9 m 活动核心外围的低对比环带／角标和连续远景沙色退晕。独立 v6 资产副本六机位 GPU 均通过，根亲审后把唯一 WGSL 改动并入 live；其余六份资产逐字节不变，live 总览与候选 PNG 精确同 SHA。布局身份仍为 `windpass_courtyard_v6`，视觉证据另绑定新 shader SHA `ce11569f…`。这是有限的铺地与远近色层改善，并未改变地平线实体轮廓。原地质资产的碰撞导出已恢复，不能沿用旧版碰撞资格；新场景相关下游评价与录像须重新生成。真实机器人通行、URI 画风、交互、运动抗锯齿和性能仍待验收。

v8 E3 地编候选现已晋升工作树：拒绝 D 版总览中叠在旧紫岩上方的北台地，仅保留西侧原创岩脊，并用贴地短色标和低对比通行带强调样本、维修等功能区。原有 2,496 条 collider 记录及 POSITION/INDEX payload、811,452 个表面三角、GLB BIN 前缀和布局 RON 均不变；新增 2 个显示与碰撞同索引的 mesh、88 个非零面积三角。场景 GLB／manifest／WGSL 的 SHA 分别为 `91ced303…`／`50b30290…`／`3e2cfb21…`。根亲审六张 1920×1080 候选图，正式资产晋升后又实际跑六机位，全部 exit 0 且 PNG 与候选逐字节相同；渲染库 22 项测试通过、2 项按其原要求忽略，319 项工程结构检查通过。当前资产的真实同世界机器人 GPU 首帧另单独 exit 0，1920×1080 画面经根亲审，世界有 16 body、2,489 collider、14 joint 且零积分。独立 Rapier 小球探针在西岩脊冠顶与沙地接缝观察到接触；2m 高落差无 CCD 首撞的原始负距离约 7.16cm，低落差对照约 0.72cm，不能认定机器人有稳定足点。当前布局身份仍是 `windpass_courtyard_v6`，但视觉／场景资产身份已改变；所有旧场景相关评价与录像须重新生成。该轮改善了局部功能识别，广场空旷感仍只获得增量改善；证据见 `.scratch/visuals/v8_matte_candidate_agent/physical_e/revision3/`。

在 E3 之后，场景线将旧科学站缺乏建筑布局的问题作为地编任务，否决遮住主舱和标牌的入口门廊，接受西侧与主舱屋面相接的侧翼楼 v10B。它新增 19 个显示与静态碰撞同索引的封闭 mesh、228 个非零面积三角；旧 2,498 条源 collider payload、811,540 个表面三角、地形和布局 RON 不变。侧翼南门净宽 1.30m、净高 2.10m，室内地板高于旧沙地 4cm；屋面与原主舱仅在旧 mesh 1483 处有意重叠，外表面高差 7.5cm。新 GLB／manifest／WGSL SHA 分别为 `f240c9e7…`／`5cf06e01…`／`382cac2b…`。根按旧／新哈希防漂移晋升三项资产，结构检查 328 项通过，渲染库 22 项通过、2 项忽略，正式六机位 GPU 原图与候选逐字节一致。正式同世界零步和真实机器人 GPU 首帧均 exit 0：2,497 个站体静态碰撞体，16 body、2,508 collider、14 joint，零物理积分；根亲审首帧。局部 Rapier 小球从沙地进入新地板，最大观测穿透约 0.44mm，但这不证明整机能够穿门、全局通路或完整接触。中庭和东侧空泊位仍需整体设计；全部验证边界和失败稿留在 `.scratch/visuals/v10_masterplan_candidate/revision_b/` 与 `.scratch/visuals/v10_root_live_validation/`。布局身份仍是 `windpass_courtyard_v6`，场景资产身份已变，相关下游评价和录像须重新生成。

机器人视觉接线已实现，接收同编译模型的视觉分组／材质／原生角点法线和只读位姿帧。源腿式／轮式分别有 70／76 个可见实例、约 80 万三角形；不以碰撞 flags 猜测可见部件，也不把碰撞副本重复显示。开发工具 `robot_initialization_preview` 使用非默认 `rendering_preview` feature，从唯一真实 Rapier 世界的初始化快照取姿态，实际输出四张 1920×1080 GPU 截图，均 exit 0 并由根亲审。每个腿／轮姿态分别逐一核对全部 397,126／400,805 个原视觉顶点及完整三角形索引，最大世界位置差约 `5.47e-7 m`，通过原 `1e-6 m` 门槛。证据、二进制及完整源码／资产副本冻结在 `.scratch/root_robot_gpu_initialization_v1`。

该开发工具现在另有 `--zero-step-only` 同世界预检：在 v7 基线上先导入 2,476 个静态碰撞体，再组装腿式机器人 15 个动态体；实际得到 16 个 body、2,487 个 collider、14 个 joint 和完整 15-body 初始位姿，积分、力矩更新、策略推理与 GPU 调用均为零。20 个可移动道具碰撞体明确暂未导入。源地形的两个零面积三角 `[7443, 7612]` 原样保留并标为风险；Rapier triangle mesh 的索引和顶点 f32 位串、固定体姿态、构造后的凸包点位串在机器人组装前后检查。独立审查发现固定体平移原先只作数值零比较，根改为 f32 位串检查并重新跑过三项测试、无窗预检与 1920×1080 真实 GPU 首帧，亲审机器人在科学站院落中的完整可视姿态。最终输入、两次原始报告、画面及根复核封存在 `.scratch/foundation_engineering/same_world_pose_bits_final_v1/`，独立代码审查在 `.scratch/foundation_engineering/same_world_first_frame_zero_step_v1/independent_code_review.md`。v8 E3 资产晋升后同一工具重新导入当前 2,478 个站体静态碰撞体，得到 16 个 body、2,489 个 collider、14 个 joint；live 与候选的零步报告逐字节相同，原两个退化三角仍在，20 个 prop collider 仍延后，积分和 GPU 调用仍为零。收据在 `.scratch/visuals/v8_matte_candidate_agent/physical_e/revision3/root_live_zero_step/`。此前四张 GPU 截图发生在站体碰撞导入之前；新的单帧仍不证明站体接触、材质、坡道通行、CCD、运动或完整目标物理。

显示诊断另定位了原渲染规则差异：[MuJoCo 3.10 classic renderer](https://github.com/google-deepmind/mujoco/blob/3.10.0/src/render/classic/render_context.c#L229-L267) 对零 UV mesh 的某些角点提交面法线，旧导入器直接使用保存的角点法线。现在保留原始 arrays 与全部三角形，在显示阶段按该规则选择有效法线；去重键也保留同顶点在不同面的锐边。独立编译的官方 C helper oracle 与三组真实模型共 3,760,632 个角点比较，最大分量差为 0；这属于源码规则核对，不是官方发行渲染二进制或像素相等证明。八张固定几何／位姿／机位的真实 GPU 对照均成功并由根亲审，嘴部、颈部和脚部碎线减少，腰髋部仍有局部高对比斑块。诊断证据和根审在 `.scratch/root_robot_normal_ablation_v1`。规则仅适用于当前原生导出证明零 UV 的两种模型；每次仍为一个世界、零积分、零策略推理，无机器人 GPU 运动、完整 URI 画风或持续性能通过结论。

源仓库 README 对代码声明 Apache-2.0，对 3D 模型另声明 Creative Commons BY-SA-NC。后续机器人资源打包须保留这份来源与原声明，不能把所有模型归入自研代码许可证；当前初始化输入仍为开发证据，尚未发布资源包。

根Cargo入口装配业务模块；机器人模块不依赖引擎，simulation管理唯一Rapier世界，rendering负责科学站与画面，开发工具仅非默认feature启用。持久文档按用户指定留在docs，运行说明为README；其余代码资产遵守唯一工程规则。

用户既有授权允许ONNX及配套原始JSON契约放在assets/game/dynamic_assets/game_data/<module>/。原始日志、数据、截图、录像、临时审查及问题放.scratch，不提交。

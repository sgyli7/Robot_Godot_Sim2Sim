# Goose Move｜源端鲁棒能力与 Sim2Sim 推进计划

版本：2026-10-10。分支：`codex/goose_move`。唯一现行入口；历史记录中的下一批不作为派工。

**平地全向是基线，先补齐，再完成源端地形、真实带载与抗冲击，最后推进 Sim2Sim。** 完全倒地起身、地面拾物归独立任务；Move负责持物后的运动和接续。不等待实体采购/制造，不主动联系硬件、G1或其他Agent。

## 当前成果与下一步

**188学习得到抗扰增量，193已冻结为限定前进域候选；原142/149仍是默认基线。** 同一192条开发冲量中，0.5/1/3N·s各64/64恢复并停止；原3N·s行走后向组5/8提高到8/8。193只有原命令0.4/0/0选择学习后的188，其余均值逐位保留原142；单个65→18 ONNX，没有动作Teacher、冲撞真值或事件计时选策略。正式独立保留集、完整Move及Bevy资格仍未取得。

| 能力 | 最新实际证据 | 尚缺与处置 |
| --- | --- | --- |
| 侧移/接续 | 142纯左/右及行进左右四条真冷开发程序通过；147四条侧移接直行/行进转弯/停止通过，全部保留。192新组合左右侧移→0.4直行→停止也通过，23秒连续无重置 | 左右互切、原地转向、后退、60秒Idle和正式200例仍欠；151/153/154/155失败路线收口，不整体重训覆盖原权重 |
| 抗扰学习 | 188原生PPO主批512次更新、10,240Adam、6,291,456GPU积分、351秒，实际经历1,558次20ms冲量，其中764次向后；190与192每档64/64通过 | 属同一开发库改善，不能当成独立正式保留集或真实物体/带载全资格。188完整平地11条仅2条通过，禁止全局替换 |
| 快慢切换 | 078成果保留。192组合真实33秒运行物理和停止通过，走速0.471/0.467、跑速0.709m/s | 跑步偏航0.109995rad/s超过0.1，整条未通过；没有与原142同程序配对，不称作已经证明的回退 |
| 20mm地形 | 215实际学习后216起伏0.4档走3.302m并停止，物理通过；218父策略同场景穿透失败。160/161的5mm开发一例及182/184踏入片段保留 | 216速度/侧漂超标；台阶0.3/0.4均入口姿态失败，四条完整地形均未过。219整脚间隙与中心间隙有明显差距，先核对测量/课程；不续训215同配置或速度网格 |
| 真实物体撞击 | 原142的186有24例实际接触、23例恢复停止，3例为行走先蹭静物。189学习188有22次接触恢复、2例未撞中 | 189实际撞击强度改变，不能据此认定解决等强度强撞；186最大物体接触穿透13.06mm另计，没有授予物体接触精度/完整S2 |
| 真实载荷 | 165无重叠的拾取后初态，100/200/300g圆柱依靠原嘴驱动与接触，各有6秒Idle保持；无附着/维持力 | 167/171运输未通过。175真实100g512PPO后177仍滑脱；185固定嘴目标也失败。停止同配置续训及嘴目标网格；把手、500g、地形载荷组合仍欠 |

**现行顺序：215已完成、216/218独立回放及217连续录像已收口；候选不晋升，不继续同配置训练。下一批先核对台阶入口“整脚包络离地”与原生足部传感/奖励的差距，再决定一个有界课程。** 既有142/149仍默认，193仍只作限定前进抗扰候选。原地转向197/204/210路线关闭，不复开步法/速度/奖励/学习率网格。完整Move、地形、载荷和Bevy资格均未取得。

| 本批实际结果 | 结论 |
| --- | --- |
| 215地形前进小试/主批 | 小试32×16：16PPO、320Adam、49,152积分、14.1秒；主批128×512：512PPO、10,240Adam、6,291,456积分、391.6秒。Actor65、Critic实测275，1,088,884个前进命令世界Tick；原奖励/本体/驱动/50/200Hz保持，原生新Critic/Adam、之后只复载自身快照。仍有3,476次终止，末课程平均0.6797，最高出现3 |
| 216起伏20mm档0.4 | 真冷连续16秒，实际前进3.302m并停止，四子步物理门槛通过；沿轨迹实测表面高差16.676mm。平均vx0.513742、vy0.084439、yaw0.076823，速度与侧漂超标，行为失败 |
| 216台阶20mm档0.4/0.3 | 第296/287Tick分别触发直立度0.944589/0.949829，仍在入口区域；前进约0.462/0.446m，不称为跨越完成 |
| 216起伏0.3与平地13条 | 起伏第542Tick穿透5.885mm失败。平地13条物理/停止通过，只有四条侧移与双向行进转弯六条完整通过；受训0.4直行速度/侧漂超标，0.7偏航0.110467也超标。原策略文件未覆盖，不由候选误差推断默认基线退步 |
| 218冻结父160同场景对照 | 编译本体/驱动/碰撞/材料和真实冷初态逐项相同。父160起伏第522Tick穿透8.250mm；台阶第479Tick直立0.948006、入口未通过。新候选起伏物理结果改善、台阶更早失败，完整行为均未合格；两策略还含不同Idle均值与路线，不能归因于EMA或PPO单因素 |
| 219只读整脚测量 | 同一台阶真实qpos，375次私有FK、6,750次地形射线，0积分/推理/PPO。215两脚共49个“脚中心射线间隙≥20mm、实际凸足底最低顶点间隙<20mm”的足帧，中心与最低点最大差30.766mm。该指标不是原生六射线环传感器的精确复现；支持核对整脚有效离地，未证明唯一失败根因或硬件不可行 |

215只使用自己的160/142神经父策略，没有动作Teacher、地形真值Actor输入、额外施力或姿态写入；限定前进0.3/0.4学习，域外均值与142误差≤2.38e-7，不伪称逐位一致。嘴轴沿冻结父网络均值，无探索/学习残差，17维Gaussian概率与上游一致，公共动作仍18轴。215是多个已声明调整的能力开发批，不是EMA单因素对照。

213确认当前0.2秒奖励EMA记忆未直接进入Critic；同样当前传感器/命令、不同历史可产生不同即时奖励，这是函数层面的状态信息证据，不能当成唯一学习瓶颈。214默认关闭的只读入口加入6个EMA值与reset就绪标志，13项测试及实际配置差异审计通过，默认配置/Actor/奖励/模型/驱动/终止保持；平地Critic69→76、地形268→275。GPU维度和真实更新由215验明，尚未证明它单独改善策略质量。

原地转向210主批512PPO/6,291,456积分，终止从204的4,719降到1,982；211原有8条方向通过，新±0.3rad/s原地目标实际约+0.00166/−0.03424，后退也未合格，212两条实际16秒录像保留。208/209仅嘴轴噪声因果对照已支持局部探索修正，但其他姿态/膝部失败仍在。197/204/210均关闭，不以减少终止或训练量宣称能力完成。

[217起伏穿越/停止实际连续16秒，速度与侧漂未合格](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/forward_terrain_actual_video_217/rough20mm_cross_stop_unqualified/continuous.mp4)、[217台阶入口失败实际连续5.92秒](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/forward_terrain_actual_video_217/steps20mm_entry_failure/continuous.mp4)。均已检查关键帧，同一真实世界、无重置/拼接/姿态修复；渲染0积分/0推理。

[215实际训练收据](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/terrain_forward_memory_course_215/forward_memory128x512/receipt.json)、[218父策略同场景对照](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/terrain_parent_matched_comparison_218/comparison_receipt.json)、[219整脚与中心间隙测量](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/terrain_sole_clearance_readonly_219/receipt.json)、[214配置与测试收据](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/reward_memory_config_admission_214/receipt.json)。S1缺口继续记录；S2/S3开发不等待正式S1，真实带载/抗撞及全部旧成果不遗漏。

194起伏回放第547Tick的表面射线越过有限场景边缘而无命中，原失败收据保存；不是已证明的策略倒地。195独立加长上游原生平地出口，没有添加隐藏无限支撑面，原20mm格子/出生/材料不变。编译审计确认仅四个地形边框的位姿/尺寸变化，机器人、关节、驱动和接触材料数组不变；195不继承194结果，也不由数量或片段授予穿越能力。

[193具名策略清单](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/source_forward_robust_candidate_193/candidate_selection.json)，ONNX SHA256 `b33557c0d950a8d8424ee03e0a27353a462dba9bdee9d313d4244e327157fab0`；默认仍为[149](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/learned_sides_hot_transition_freeze_149/baseline_selection.json)。[192实际冲量收据](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/forward_robust_composite_eval_192/learned191_points192/receipt.json)、[195编译场景差异审计及四条失败](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/finite_terrain_exit_check_195/compiled_scene_difference_audit.json)、[176完整记录](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/source_robust_progress_176/report.md)。

[新学习后的3N·s行走后向高位推扰，连续17秒](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/learned_impact_and_side_video_196/walking_3ns_rearward_high/continuous.mp4)、[新组合左移→直行→停止，连续23秒](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/learned_impact_and_side_video_196/left_side_to_walk_stop/continuous.mp4)、[快慢切换偏航未合格，连续33秒](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/learned_impact_and_side_video_196/walk_run_switch_unqualified/continuous.mp4)。三个实际轨迹已检查关键帧，渲染0积分/0推理，无重置/拼接/姿态修复。视频旧`arm`字段沿用评价器来源名，实际策略身份以收据中的ONNX路径/哈希为准；当前三条实际均为191/193同一权重。

**147新增四条同世界接续开发程序全部通过：左右侧移分别接直行/行进转弯，再停止。** 原142神经策略、14叶本体、数字驱动与验收门槛未改。每条真冷Idle4秒→侧移7秒→接续7秒→停止5秒，共23秒、1,150Tick，命令切换保留真实姿态、前次动作、驱动目标、相位与热历史；全程无重置。

| 接续程序 | 接续vx m/s | 接续yaw rad/s | 同时跟踪比例 | 停止1秒速度 m/s | 后3秒漂移 mm | 运动/物理/停止 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 左移→直行 | 0.433935 | +0.026457 | 1.0000 | 0.009668 | 2.433 | 通过 |
| 右移→直行 | 0.437120 | +0.031012 | 1.0000 | 0.008871 | 2.345 | 通过 |
| 左移→行进左转 | 0.431134 | +0.288540 | 0.7000 | 0.007813 | 1.809 | 通过 |
| 右移→行进右转 | 0.446927 | -0.250486 | 0.6533 | 0.008596 | 2.332 | 通过 |

147累计18,400真实GPU积分、4,600世界CPU ONNX推理；四条独立进程开发程序通过，不授予正式200例、任意热切换或完整Move/Bevy资格。入口：[149附加证据清单](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/learned_sides_hot_transition_freeze_149/baseline_selection.json)、[完整报告](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/learned_sides_hot_transition_freeze_149/report.md)。[左移→行进左转→停止连续录像](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/learned_sides_direct_transition_video_148/left_to_turn/continuous.mp4)、[右移→行进右转→停止连续录像](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/learned_sides_direct_transition_video_148/right_to_turn/continuous.mp4)。

**142此前已通过四条独立侧移开发程序，继续冻结保留。** 实际CPU ONNX控制原14叶MuJoCo/Warp世界，50Hz策略/数字驱动、200Hz物理。每条真冷Idle4秒→移动7秒→停止5秒，800Tick连续无重置。

| 程序 | 实际vx m/s | 实际vy m/s | 实际yaw rad/s | 持续同时跟踪比例 | 运动/物理/停止 |
| --- | ---: | ---: | ---: | ---: | --- |
| 纯左移 | -0.016115 | +0.095927 | +0.002533 | 0.6433 | 通过 |
| 纯右移 | -0.009078 | -0.088034 | +0.065512 | 0.6067 | 通过 |
| 行进左移 | +0.420214 | +0.091990 | -0.030741 | 0.6600 | 通过 |
| 行进右移 | +0.416720 | -0.099388 | +0.039808 | 0.7800 | 通过 |

12,800真实GPU积分、3,200世界ONNX推理，时钟误差4.582e-7秒。右侧持续比例裕量较小，不能据此授予鲁棒性。正式200例、跨方向热切换及Bevy资格仍未完成。

入口：[142冻结神经策略](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/learned_sides_freeze_142/learned_sides142.onnx)，SHA256 `8c1c5c889635dbdae5fdd862a48baea4159a7cd7d2a1a5cc824ae444f40a36f8`；[报告](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/learned_sides_freeze_142/report.md)、[清单](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/learned_sides_freeze_142/manifest.json)。

[16秒纯左移连续实录](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/learned_side_student_video_140/learned142_left/continuous.mp4)、[16秒纯右移连续实录](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/learned_side_student_video_140/learned142_right/continuous.mp4)。运行时使用自己的神经权重，没有动作Teacher、姿态修复或额外施力。

138原生RSL学生实际控制蒸馏学会左侧；141精确恢复原生Adam的一次续训学会右侧、左侧回退。因此142只读合成已通过的138左侧和141右侧，按原Actor命令选择均值，单个65→18 ONNX每Tick一次调用、内部两个冻结神经分支。两个父策略的归一化/动作单位/原非纯侧移均值相同；800实际输入选择均值逐位保持，Torch/ONNX最大差2.086e-7。不能将该只读合成快照当单头PPO恢复点。135动作Teacher和所有旧策略独立保留。

144完整镜像串联未过：左→右切换后1.90秒触发直立度0.946224；另一条右→左和侧移→直行通过，但直行再切纯右移0.44秒后触发0.944563。此前穿地≤2.562mm、自碰0、关节越限≤0.000985mrad。评价停在姿态门槛，未证明已完全倒地；未执行的后续阶段不记作独立失败。146只改输入缓变，固定0.8/0.4/0.6每秒的vx/vy/yaw上限，两条仍失败且更早触发姿态门槛，不晋升、不继续缓变参数网格。本周期共29,640真实GPU积分、7,410世界推理，0PPO/0Adam；八条实际轨迹、计数、边界历史与连续录像全部保留。

151已完成135动作Teacher接续对照并失败；153连续混合、154固定制动未解开接续，155原生残差全向训练的156八条独立行为程序也未通过。它们均不替换142，不再作为待执行批次。源码门控、训练域及具体失败状态仍须凭证据区分，不能由失败猜测积分、本体DOF或单一根因。

## 目标与阶段

空载平地走路 **0.4m/s**、跑步 **0.7m/s**。复杂地形/带载先争取0.4m/s，困难时可 **0.3m/s**；随机错落台阶 **20mm必达**，50mm独立挑战。轻中等撞击允许减速、停下、补偿迈步，**3秒内恢复原指令且不完全倒地**。真实带载与复杂地形必须联合通过；100/200/300g逐档，500g挑战，上限由组合实测确定。

| 阶段 | 工作与退出依据 | 当前范围 |
| --- | --- | --- |
| S0 源端准入 | 14叶动态、时钟、容量、GPU小批量和原生训练链路 | 057/058及后续实际运行支持有界源训练；新场景另验 |
| S1 平地全向 | 行进转弯、原地转向、行进侧移、纯侧移、后退、XYZ、Idle、启停、快慢切换；正式平地验收 | 078快慢/双向行进转弯保留；110/116行进侧移、142双向纯/行进侧移、147双侧接直行/行进转弯/停止开发通过。左右互切及直行切纯侧移仍有缺口；原地转向/后退/完整热切换及正式200例仍欠。078原开发63/240、真冷5/11；单独快跑yaw0.1033>0.1失败保留 |
| S2 地形/抗冲击 | 随机起伏、±5°坡、5/10/20mm错落台阶；高度/间距/横向错位/方向随机；推/擦碰/真实撞击先单项后组合 | 原078的5/20mm起伏各一例保留；160/161新增5mm错落台阶穿越/停止一例；170名义COM24/24保留，180不同作用点每冲量档64例达分档门槛。179/181、182/184地形未晋升，184有实际20mm踏入片段；186真实碰撞23/24开发通过；188/190固定冲量每档64/64，但完整移动回归失败，不整体选用；192组合抗扰及左右接直行停止通过，快慢程序跑步偏航失败；195四条20mm新出口场景全部未过。215/216起伏物理/停止一例通过但速度与侧漂未过；台阶入口仍失败，219整脚测量差距已记录。完整台阶/坡面/真实撞击正式库仍欠 |
| S3 真实带载 | 冻结指定圆柱夹持段/把手物体，先保持，再Idle/前后/侧移/转弯/停止，随后与S2联合 | 109原初态完整柄部重叠15.954mm，撤回其载荷极限解释；165/166/167真实100/200/300g圆柱Idle6秒开发通过。167/171运输未过，169观测掩码对照关闭；175真实100g主PPO完成，177独立运输失败，185原嘴目标固定对照也滑脱，均未晋升；把手/500g/地形组合欠账 |
| S4 源端冻结 | 独立通过平地、地形、冲击、载荷及组合；冻结全部合同/策略/场景/适用范围、连续录像 | 待S1—S3 |
| T1 Rapier/Bevy 50/200 | S4后零样本接收及必要后训练，再键位/相机、实际窗口验收、可玩基线 | 未目标训练，不继承源端资格 |
| T2 50/50完整迁移 | T1后独立50/50后训练，保留50/200，重测完整矩阵 | 频率资格分别授予 |

## 固定实现与验收

- 本体`goose_torso_shells14_rigid_v1`：21机器人刚体、18主动轴、2被动轴、14实际碰撞叶、10.430690821kg，刚性足底；原质量/惯量、轴顺序、自碰排除及驱动限制保持。Actor **65→18**，地形/物体/接触真值仅供Critic、任务管理和验收。
- 源端 **50Hz策略/数字驱动、200Hz物理**：20ms一次推理/驱动包，四次真实5ms积分；17位置/1嘴力矩原语义保持，不在子步额外更新数字PD/限速/热历史。奖励/折扣/延迟/终止按物理时间，记录真实计数。复用统一MJCF、原生MuJoCo/Warp、mjlab/RSL-RL。
- 地形/载荷/撞击版本化。区分自碰、地形、持物、障碍；穿透依据真实表面，不能按z=0或把所有非零geom归为自碰。指定物体尺寸/COM/夹持点/摩擦训练前冻结；靠真实嘴部驱动与接触，无附着、搬运或维持力，掉落/拖行/额外支撑失败。
- Bevy明确200Hz物理/50Hz控制，保留默认50Hz档。W/S前后、A/D转向、Q/E侧移、Shift加速；输入缓变、失焦停止、暂停恢复、跟随/环绕/缩放相机。学Godot工作流，不加载MicroDuck的61→14权重；速度受通过范围约束。
- 生产代码只归对应crate；临时脚本、模型、日志、收据、构建、录像、快照全在项目备份。具名候选，不加载/晋升latest，不改原工作树或Standup，不创建/唤醒Agent、新聊天、PM或goal。

| 项目 | 固定标准 |
| --- | --- |
| 物理完整性 | 冷启动、有限数值、时钟/容量/能量与四子步峰值；平地地形穿透≤5mm、自碰≤0.1mm、关节越限≤0.1mrad、直立度≥0.95 |
| 平地 | 60秒站立漂移≤5cm；原0.4/0.7、转向/停止门槛；200独立案例≥95%；240开发库及11真冷程序作回归 |
| 运动/停止 | 平均目标比例0.8—1.2，未指令XY≤0.04m/s、yaw≤0.1rad/s；指令轴同时≥0.5目标的Tick比例≥0.6；停止1秒≤0.04m/s，随后3秒漂移≤5cm |
| 地形 | 每类别≥50例、≥90%，含转向/终点停止；组合步行≥0.3m/s；50mm另报 |
| 抗冲击 | 0.5/1/3N·s水平冲量后真实物体接触复测；方向/位置/Idle/移动，每档≥50例、≥90%，3秒内恢复且不完全倒地；带载另测保持 |
| 带载组合 | 每授予重量×地形≥60例、≥90%；Idle、携带2m、双向转弯/侧移/停止，真实物体全过程保持 |
| 正式评价 | 独立进程、确定性ONNX、固定验收种子，无自动重置；开发/训练/验收集分开，保存全部失败、实际计数和连续录像 |

撞击允许短暂倾斜，不把平地直立阈值直接当撞击失败。源端、零样本迁移、后训练、游戏成绩分开，仿真不授予实体夹力/热/电气/制造资格。

## 执行节奏与保留证据

每批冻结一个问题→短pilot→最多两小时训练→独立评价→具名保留/失败。连续两次受控实验未改善就回到模型/观测/测量/课程或成熟流程，不无限网格。一个GPU批次，未结束不重复训练/评估/渲染；数值、约束或容量错误停止留证据。代码/依赖/配置/种子/hash、PPO/蒸馏/Adam/物理及资源分别记账。两小时`goose-move`续跑保持启用，无变化不频繁轮询，不改Standup任务。

- [135动作Teacher接收](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/qualified_side_teacher_replay_135/report.md)、[138/139学生实际回放](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/native_onpolicy_side_distillation_138/report.md)、[141续训及左侧回退](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/native_side_student_continuation_141/report.md)、[142神经侧移冻结](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/learned_sides_freeze_142/report.md)。
- [144/146失败及147接续增量独立复核](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/learned_sides_hot_transition_freeze_149/independent_counts_check.json)、[原左→右连续失败录像](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/learned_sides_hot_switch_video_145/left_first/continuous.mp4)、[原串联右→左→直行→右移连续失败录像](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/learned_sides_hot_switch_video_145/right_first/continuous.mp4)。输入缓变146未改善，独立保留，未改现行142策略或控制器。
- 原078、110、116及023/028/041/069等不覆盖。112/116/127失败纯侧移PPO、129/131/133未过Teacher、136/137学生失败及103关闭诊断全部保留，不重开Critic/归一化/Gaussian/LR/速度容差网格。
- [原078双向行进转弯实录](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/native_own_move_distillation_continuation_078/cold_bilateral_turn_video078/continuous.mp4)、[20mm随机起伏真实穿越](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/actual_twenty_mm_video_130/video/continuous.mp4)。20mm起伏均速0.404849m/s、真实高差19.773765mm、行进1.65956m；停止1秒0.010329m/s、后3秒漂移1.024mm、物理通过，仅一例，不是20mm错落台阶资格。
- 更早实验、误测撤回、资源及合同身份见[精简前完整快照](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/plan_before_learned_sides142_entry_20261010.md)及其历史链接，仅追溯、不派工。GitHub主页继续展示MicroDuck、Goose、G1的实际GIF，来源见[媒体清单](sim2sim_results_gallery.md)。

最终仍需冻结策略/场景/本体/合同、可移植复现包、完整能力和失败清单、连续录像与Bevy运行包。当前交付源端具名侧移增量，完整全向、鲁棒性及目标端资格未完成。

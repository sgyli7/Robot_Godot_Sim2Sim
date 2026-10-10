# Goose Move｜源端鲁棒能力与 Sim2Sim 推进计划

版本：2026-10-10。分支：`codex/goose_move`。唯一现行入口；历史记录中的下一批不作为派工。

**平地全向是基线，先补齐，再完成源端地形、真实带载与抗冲击，最后推进 Sim2Sim。** 完全倒地起身、地面拾物归独立任务；Move负责持物后的运动和接续。不等待实体采购/制造，不主动联系硬件、G1或其他Agent。

## 当前成果与下一步

**176阶段汇总：180不同作用位置的冲量测试达到分档门槛；179的181独立地形评价失败保留，当前运行182原生地形Critic对照。** 平地142/149、078等冻结选择保持，以下是增量，未取得完整S1—S3或Bevy资格。

| 能力 | 最新实际证据 | 尚缺 |
| --- | --- | --- |
| 地形 | 160原生地形课程完成512次PPO；161在5mm错落台阶连续穿越并停止一例通过，均速0.475m/s。旧142在20mm起伏穿越达到0.338m/s | 160在20mm台阶第296Tick触发直立度0.947587；旧142的20mm起伏停止失败。坡道、转向和正式类别库仍未过 |
| 抗撞 | 180：Idle/行走、四方向、八类躯干作用点加冻结随机偏移，每档64例。0.5/1N·s各64/64，3N·s为61/64（95.31%）；成功例1.02—2.04秒恢复并最终停止，652,800次GPU积分；170旧名义COM24/24保留 | 3N·s行走中向后推的子组仅5/8，三例关节越限失败，保留局限；真实物体撞击及带载组合未完成。满足冲量层数量/成功率门槛，不授予完整S2 |
| 真实保持 | 165完整物体无重叠的拾取后初态；原142依靠嘴驱动和真实接触，100/200/300g各有6秒Idle保持证据，无附着/额外支撑 | 167/171侧移、直行、行进转弯运输仍失败。100g最长1.792m后失去夹持并嘴部越限，不是完成2m运输 |
| 带载训练 | 175原生MuJoCo/Warp＋RSL-RL，真实100g完成512次PPO、10,240次Adam、6,291,456次GPU积分，427.7秒；实际前进样本660,964世界Tick | 177真冷回放11.88秒后滑脱并嘴部越限，非指令偏航−0.448rad/s；不晋升或覆盖平地基线，不继续相同PPO网格 |

现行证据：[176汇总及复现说明](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/source_robust_progress_176/report.md)、[175真实载荷训练收据](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/native_real_load_sampler_175/real100g128x512/receipt.json)、[177独立策略清单](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/native_real_load_evaluation_177/baseline_selection.json)。[5mm台阶连续录像](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/robust_terrain_payload_impact_video_172/terrain5mm/continuous.mp4)、[3N·s行走抗撞连续录像](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/robust_terrain_payload_impact_video_172/impact3Ns/continuous.mp4)、[20mm台阶失败](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/robust_terrain_payload_impact_video_172/terrain20mm_failure/continuous.mp4)、[100g运输失败](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/robust_terrain_payload_impact_video_172/payload100g_failure/continuous.mp4)。均为实际模型/初态/连续轨迹，未重新积分或拼接。

**当前顺序：接回182原生地形Critic对照并独立验收；随后补真实物体撞击及平地剩余基础能力。** 179原生20mm主批完成512次PPO/6,291,456积分；181四条均未通过：起伏0.3m/s能走完停止但偏航0.238rad/s超标，台阶0.3/0.4m/s受阻，平地回归穿透超标。178单次允许降速也未解台阶，停止速度及相同PPO网格。发现计划要求的地形/接触Critic真值尚未接入；182复用上游原生地形扫描、足高、腾空时间、接触和接触力，Critic268、Actor仍65，原本体/驱动/奖励/门槛不改。首次仅原生Actor转移，明确使用新的Critic/Adam；不伪称完整PPO复载。32世界pilot完成16次PPO/49,152积分，Actor转移ONNX均值误差8.94e-8；主批复载182自身完整PPO/Adam后128×512运行中。此对照检验新的成熟训练接线，不预先认定遗漏就是唯一瓶颈。

[180冲量分档及弱点报告](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/multi_point_horizontal_impact_180/qualification_summary.json)、[3N·s后退补偿恢复连续17秒](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/latest_terrain_impact_video_183/impact3Ns/continuous.mp4)、[181台阶失败连续8.4秒](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/latest_terrain_impact_video_183/terrain20mm_failed/continuous.mp4)、[177真实带载失败连续11.88秒](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/native_real_load_evaluation_177/continuous_video/continuous.mp4)。平地左右热切换、原地转向、后退与正式库继续欠账，不遗漏；S2/S3开发不等待正式S1。151动作Teacher、153混合、154固定制动与155全向残差PPO均已独立失败收口，不继续同头蒸馏/参数网格。下方142/149记录为仍有效的既有基线，不是本轮派工。

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
| S2 地形/抗冲击 | 随机起伏、±5°坡、5/10/20mm错落台阶；高度/间距/横向错位/方向随机；推/擦碰/真实撞击先单项后组合 | 原078的5/20mm起伏各一例保留；160/161新增5mm错落台阶穿越/停止一例；170名义COM24/24保留，180不同作用点每冲量档64例达分档门槛。179/181地形失败未晋升；182原生地形Critic对照中；20mm错落台阶、坡面、真实碰撞/完整地形库仍欠 |
| S3 真实带载 | 冻结指定圆柱夹持段/把手物体，先保持，再Idle/前后/侧移/转弯/停止，随后与S2联合 | 109原初态完整柄部重叠15.954mm，撤回其载荷极限解释；165/166/167真实100/200/300g圆柱Idle6秒开发通过。167/171运输未过，169观测掩码对照关闭；175真实100g主PPO完成，177独立运输失败；把手/500g/地形组合欠账 |
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

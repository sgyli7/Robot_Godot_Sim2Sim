# Goose V0.1｜Move 训练计划

版本：2026-10-09。独立分支：`codex/goose_move`。

**当前状态：取得部分源端能力，完整任务未完成。** 本聊天只负责Move，不推进起身或拾物，不联系其他聊天。用户新授权的唯一独立子Agent专项调查物理频率，其实验不改主线。目标保持：走路0.4m/s、Shift跑步0.7m/s，前后／侧移／双向原地转向／移动中转向／启停及任意切换，操作与MicroDuck三轴组合对齐。不能以请求速度、单段录像或开发小集代替这些资格。最新225的可选左右弧线增量保留176全部12个通过项，原16条开发程序14条通过，扩展集17/25通过；不覆盖143/176或授予完整资格。

**最新用户授权：另建200Hz源端快跑候选，尝试达到0.7m/s。** 策略及数字驱动目标/历史仍50Hz，四次真实5ms积分；原50Hz默认、Actor基线及Bevy目标保持。该追加速度课程独立于四频相同课程的因果实验，不把新增课程成绩归入频率单变量比较。根主线自行实施，串行使用GPU。

**当前派工：200Hz高速奖励宽度单参数短对照已完成，不晋升或覆盖旧基线。** 同148父学习状态、seed4122、512世界、fixed5e-5与0/.4/.55/.7课程，各256更新，仅线速度奖励std从.15改为上游默认.5。新候选200Hz物理48/48、行为0/48，0.4实速.3344、0.7实速.3617m/s，60秒段.3620；.4移动区间8/8达开发门槛，但停止均失败，全部停止仅16/48通过。50Hz迁移物理12/48、行为0/48。初始学习状态、出生与预算相同，首24Tick逐位物理回归未通过，失败证据保留，不作奖励宽度单变量因果结论。下一问题是已学约1.2Hz步频与目标限速的关系：具名小幅步频增量先短筛，不原样加长本轮；保持实际驱动/物理限制，不改硬件。髋偏航部分饱和与停止/迁移缺口分别记录。30秒连续录像、全部轨迹及资源已冻结，GPU已释放给独立频率研究。

代码工作树为`/home/ethan/Projects/TempWorktree/Sai_Lab/goose_move/Bevy_Sim2Sim`。临时模型、脚本、录像、收据及构建均落[Move备份目录](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001)。原工作树和恢复分支不改写。

## 当前具名入口与实际能力

[收尾清单](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/source_move_closeout_177/manifest.json)固定模型、11份外部模型资源、合同、Actor、控制器、冻结源码和收据SHA；[收尾报告](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/source_move_closeout_177/report.md)记录范围、失败和复现方式。禁止加载latest或覆盖旧基线。

| 候选 | 实测及退出依据 |
| --- | --- |
| 默认保留143 | 约0.14／0.20m/s与±0.3rad/s，13条开发程序11条通过；左弧线及完整串接未通过。停止选择只依赖真实用户命令历史 |
| 较快开发176 | 约0.14／0.27m/s；37秒慢→快→左→右→停无重置通过，左右转约+0.334／−0.343rad/s。16条开发程序12条通过；不是默认替换或完整MD资格 |
| 可选左右弧线229 | 保留176原12个通过项，新增冷左弧线及45秒慢→快→左→右→右弧线→停；原16条14条通过，扩展集17/25。29秒慢走→左右弧线→停通过，任意热切换与完整串接仍失败 |
| 原目标 | 0.4／0.7m/s、后退／侧移、完整左右弧线、任意停走切换、正式200例及Bevy资格均未通过 |

[最佳37秒连续录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/hot_turn_continuous_visuals_172/slow_fast_left_right_stop_37s_continuous.mp4)、[29秒快慢切换](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/speed_increment_visuals_159/slow14_fast27_slow14_stop_29s_continuous.mp4)、[41.10秒完整失败录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/hot_turn_failure_visuals_179/right_start_after_stop_failure_41_10s_continuous.mp4)。画面使用真实保存轨迹，不拼接、不回撤；摄像机与世界视图同时显示，保留失败及门槛数据。

## 本轮代码与验证

[控制模块](/home/ethan/Projects/TempWorktree/Sai_Lab/goose_move/Bevy_Sim2Sim/crates/dev_tools/python/src/bevy_microduck_tools/goose/move_supervisor.py)保留原停止选择器，另提供具名`goose_hot_turn_phase_entry_v2`。仅直接非零运动切进纯左／右转时重启策略相位到清单参考值；零指令、冷启动不重启。原1.2Hz时钟每个20ms提交仍推进一次。它修改的是明确版本化的控制相位，不改身体位置、关节位置、动作历史或物理，也不继承原候选资格。

175核对21,800条真实命令／相位记录，8次相位事件和167次状态序列化恢复一致；12条未触发新逻辑的程序与原156所有保存字段逐位一致。176使用实际生产模块重新执行21,800真实CPU积分／推理／私有FK查询，全部轨迹与173逐位一致，[接线回归收据](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/production_hot_turn_integration_176/integration_regression.json)。0.27候选60秒行走后的停止速度0.0489m/s仍超过0.04门槛；转向后接弧线也仍失败，所以保留原默认。178低速版本复核保住原11/13通过项，却在新冷快→左转中失败；不授予任意热切换资格。

## 最新物理诊断与独立对照

[197收口报告](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/gain_calibration_closeout_197/report.md)固定本轮代码、模型、训练与失败收据。182实际GPU训练记录到饱和踝滚转在一个20ms步内由0.279rad到1.044rad、速度38.26rad/s；185的原生CPU单步复现。该版本原生执行器在力矩饱和时不保留隐式增益导数，不能仅凭启用隐式位置驱动认为所有轴在50Hz下都已校准。签名制动已启用，此事件并非遗漏反向制动。

具名开发本体`goose_task_proxy_11_discrete_ankle_kp10_v1`仅将左右踝滚转位置增益20→10Nm/rad，阻尼、力矩、速度、惯量、几何、接触、积分与公共65→18合同均保持；原本体和策略不改写。189实际8世界×50GPU步站立筛查通过，不等于完整M0-S。190与182同父策略、种子、奖励、课程和探索量对照，各完成3,145,728GPU步／128PPO：关节越限终止39,803→1,203，接触终止42,267→22,624，但这些是可重叠的终止原因计数，不是能力资格。

191对16/64/128三个检查点独立执行8,100CPU步；24个方向案例均完成物理区间，侧移、后退和左右弧线均未通过。64检查点只有前进段达到开发指标，尚无停止闭环资格。194足底刚度和195踝俯仰增益各12事件单步对照未解决脚底穿透；两条调参路线收口，不继续网格搜索或同条件加长训练。[真实GPU踝越限连续录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/saturated_ankle_failure_visuals_192/actual_saturated_ankle_failure_continuous.mp4)保留全部59个20ms样本，属于带探索噪声的训练失败，不冒充确定性ONNX能力。

196用当前生产模块、原模型和原Actor重跑21,800CPU积分／ONNX调用／私有FK；16程序全部保存字段与176逐位一致，原12/16通过项保留，[回归收据](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/protected_move_baseline_regression_196/regression.json)。193的增益机制、身份拒绝和原模型测试11项通过。当前证据定位了一个实际驱动失效机制，但尚未证明它是全部移动瓶颈，也不能据此断言结构无法行走。

## 本轮驱动与MicroDuck课程收口

[218阶段报告](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/target_window_closeout_218/report.md)记录真实驱动修正与未通过项。具名`goose_rate_constrained_torque_window_v1`将原生位置目标限制在实际正负力矩预算窗口，并与原20ms目标限速／关节区间相交；交集为空时保留原限速目标并计数。原本体、增益、力矩、惯量、碰撞、求解、50Hz和公共动作历史保持；该开发控制器没有进入默认选择器或继承原合同资格。211单步消除10个踝越限事件，但12个深接触事件只通过4个；212的1,950CPU步保住两个冷转加停止程序，前进出现偏航；213仅8世界×50GPU站立步通过，不等于完整M0-S。

214与182同父策略、原B物理、种子、探索、奖励和课程，完成3,145,728GPU步／128PPO；关节终止39,803→244、接触42,267→18,480，原因可重叠，不能称成功率。215的8,100独立CPU步未取得新方向。216保持该控制器及奖励，复用MD九方向采样，另完成3,145,728GPU步／128PPO；217三检查点共23,934CPU步，64仅一个0.2指令前进加停止的小程序通过，实速0.18499m/s；侧移、后退、纯转向、完整弧线与三轴组合未通过，0.4／0.7的六个案例全部失败。新Actor不晋升，固定前进及该MD采样续训收口。

[命令／动作审计](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/target_window_closeout_218/command_action_audit.json)核对24,576个保存训练Tick，九方向真实出现，PPO裁剪输出与提交动作历史逐位一致；未发现按非前进指令强制切回站立的分支。198／203／207／210侧移Teacher、200／201速度伺服、209反射惯量和202奖励精度路线未获得资格，均收口；代码、真实轨迹和失败保留在备份目录。相关接触失败尚未解决，不能断言DOF不足，也不再用同条件加长训练代替诊断。

## 最新左右弧线增量与边界

[229阶段报告](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/mirrored_arc_closeout_229/report.md)固定可选`goose_move_mirrored_arc_v1`。只镜像自己的Goose右弧线网络并声明左弧线编码标定，不使用MD权重。原65→18、真实动作历史、原本体与50Hz物理／驱动保持。223的具名相位规则只在纯转向或直线快档直接进入前进弧线时重启；慢走、弧线互转、零指令与冷启动不重启。222的广泛重启规则破坏弧线串接，未采用。

225用[实际可选模块](/home/ethan/Projects/TempWorktree/Sai_Lab/goose_move/Bevy_Sim2Sim/crates/dev_tools/python/src/bevy_microduck_tools/goose/move_arc_entry.py)执行28,490CPU积分／ONNX调用／私有FK，与223的525个保存字段逐位一致，4次相位事件和219次暂停状态恢复，23项测试通过。冷左实际vx0.14145m/s、yaw+0.09824rad/s，冷右vx0.14346、yaw−0.09946。近门槛通过不能外推任意切换。

[29秒左右弧线连续录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/arc_progress_continuous_visuals_228/camera_revision_v2/slow_left_arc_right_arc_stop_29s_continuous.mp4)、[45秒快慢／转向／右弧线连续录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/arc_progress_continuous_visuals_228/camera_revision_v2/slow_fast_turns_right_arc_stop_45s_continuous.mp4)。它们是原生源端真实轨迹及碰撞代理可视化，无拼接、回撤或中途重置，不是Bevy游戏资格。

完整六段串接、纯转向后切左弧线、60秒快档停止及原0.4／0.7、后退、侧移仍失败。224停止平滑两档未改善，227扩大自己学得的矢状步幅在68Tick自碰／穿地；两条路线收口，未改默认、硬件或验收门槛。[1.36秒步幅失败录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/arc_progress_continuous_visuals_228/camera_revision_v2/sagittal_stride_collision_failure_1_36s_continuous.mp4)。本轮151,146CPU积分，零新增主线GPU/PPO/Adam；旧资源账本v25保留。

## 独立物理频率调查：用户新增授权

唯一子Agent`integration_frequency_audit`在[独立备份](/home/ethan/ProjectBackups/2026-10-08/Sai_Lab/goose_integration_frequency_audit_001)核查真实源码与50/100/200/500Hz，策略决策统一50Hz。冻结ONNX跨频与真实失败回放只用于数值和旧策略适应性，不能证明高频环境的可学习性。

用户进一步授权四频率独立PPO：同本体、驱动约束、奖励、MD九方向课程、初态银行及三个共同种子，从新网络／归一化／探索／优化器初始化出发。统一模拟时间、控制决策数及PPO更新预算；先以500Hz小链路吞吐冻结预算，再记录各组真实GPU积分数、Adam及耗时。策略／目标限速／驱动历史／相位／奖励／折扣／课程与终止的时间单位保持20ms；原生隐式PD子步内状态依赖与数字控制提交区分说明。

先在各自训练频率独立测试，再全部部署到50Hz对照，保留学习曲线、越限、穿地、自碰、接触、速度、任务成功与首个失败。源端50Hz迁移结果不能直接授予Rapier／Bevy资格。不得降低标准、覆盖已有基线或修改主线频率／训练参数。最终分别回答50Hz对学习的影响、高频独立学习新增能力及50Hz迁移保留程度；有限预算内未学会不能证明能力不可能。

## 200Hz快速移动候选：当前追加工作

[候选计划](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/plan.md)和[冻结协议](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/protocol.json)声明`goose_move_native_discrete_200hz_fast_v1`。保留原B几何、质量、惯量、驱动增益、限制、接触和求解设置；原XML/合同不覆写，具名运行时只将物理步长覆盖为5ms。原生隐式PD保持同一连续反馈公式，子步内力矩随状态变化；prepare/commit、限速、热、相位、奖励和折扣仍每20ms一次。

首轮父策略固定为本Goose独立200Hz/seed3101完整512更新模型，恢复Actor/Critic、归一化、高斯与Adam；使用新任务出生银行及计时，不声称精确世界接续或继承资格。复用冻结mjlab/Warp/RSL原生训练循环和MicroDuck适配奖励，另声明前进/停止课程：0.14/0.27→0.27/0.4→0.4/0.55→0.4/0.55/0.7。停止桶25%，命令保持4–6秒；不改奖励公式或权重。

[CPU预检](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/cpu_preflight_receipt.json)核对四阶段各10万次命令采样、模型/驱动/奖励/观测/终止设置及完整学习状态装载；父Torch/ONNX最大差2.384e−7。该预检零积分、零PPO，不代替物理准入。GPU接线8世界/1更新后执行512世界/256更新pilot；每批最多两小时。以独立当前速度8秒+4秒停止的8个出生全通过作为开发扩档依据，冷启动另测；连续两批不改善即收口。

候选在200Hz自身环境及原50Hz物理部署两层评估，另测0.4→0.7→0.4切换和60秒跑后停止。保留每子步物理峰值、实测速度与连续失败录像，不用请求0.7或训练奖励冒充0.7实际能力，开发小集不授予正式200例或Bevy资格。

[首轮收据与报告](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/first_pilot_closeout_001/report.md)：8世界/1更新接线通过，512世界/256更新pilot完成12,582,912真实GPU积分、3,145,728控制决策、5,120 Adam更新，204.3秒。训练首/末32更新的速度RMSE约0.197/0.199m/s，未改善。200Hz独立六程序×8银行出生物理48/48通过、行为0/48；50Hz物理16/48、行为0/48。0.7指令稳态实速约0.0000235m/s，脚底高度变化仅微米量级，属于站立局部解。

[连续12秒失败录像](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/videos/first_pilot_run70_200hz_001/continuous.mp4)是实际固定出生0轨迹，未拼接、回滚、修复或换Actor；渲染零积分、零推理。首轮没有在0.7命令分布训练，不能把其0.7测试失败当作0.7训练已失败，也不能证明高频无法学习。它没有覆盖143/176/229。

独立评估修正另名v3：手工新命令须重新读取原无历史Actor组，缓存compute(False)不能代表新指令；NumPy结果须显式转换JSON。原评估与两次失败保留（0积分、32积分），训练、物理、案例及门槛保持。评价刚体能量未计armature，不作完整能量结论。首轮含接线和评估共13,226,944 GPU积分、257 PPO/5,140 Adam；正式200例、真冷启动和Bevy均未授予资格。

[第二份冻结协议](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/fast148_run70_002/protocol_v3.json)固定自己的148/updates512.pt（0eb265c3…），CPU装载及Torch/ONNX最大差5.22e-8通过。保留Actor/Critic/高斯/归一化/Adam及原fixed5e-5配方；原生PPO、奖励和200Hz本体与首轮保持。新命令桶为0/.4/.55/.7各25%。v3仅修正v2接线证明中父频率的文字，原v2收据保留；实际父权重与学习/物理语义未改。

[第二轮报告](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/second_pilot_closeout_002/report.md)与[103文件冻结清单](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/second_pilot_closeout_002/manifest.json)记录512世界×256更新、12,582,912真实GPU积分、3,145,728控制决策、5,120 Adam及241.25秒。200Hz父策略物理8/48→新Actor48/48，0.4实速0.2685→0.3028m/s；0.7平均0.2917、60秒段0.2806m/s，仍偏航。48例停止阶段均通过，但行为0/48。0.27实速0.2300→0.1255，不能声称保留全部低速能力。50Hz物理仅2/48；其余46例首次失败都含穿地>5mm，0.4/0.7各八例分别在0.66/0.80秒结束，未迁移成功。

[30秒0.4→0.7→0.4→停止连续录像](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/videos/speed_switch_from148_200hz_002/continuous.mp4)、[12秒0.7指令失败录像](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/videos/run70_from148_200hz_002/continuous.mp4)来自完整真实保存轨迹，无重置/拼接/回滚/换Actor。1,200次离线私有FK确认确实交替迈步；三速度档主要节奏约1.17Hz，0.7的摆幅没有成比例增加。这是已学策略的描述，不是相位硬限制或硬件最大速度证明。本轮没有同条件50Hz续训臂，不将增益因果归于升频；直接run70暴露也不代表中间课程晋升。

[第三轮报告](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/reward_signal_003/report.md)记录实际完整配置树仅std变化、成熟配置出处、真实奖励密度×20ms接线及严格回归失败。[实际PPO收据](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/runtime/runs/fast200_run70_from148_sigma05_pilot_003/receipt.json)为12,582,912 GPU/256PPO/5,120Adam/248.59秒。源端快档比第二轮约提高.07m/s，但仍有偏航；部分短停止1秒残速约.10m/s，不能以移动段通过替代整条资格。训练首/末32更新XY RMSE .2868/.2857，奖励跨宽度不直接比较能力。

[第三轮30秒连续录像](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/videos/sigma05_speed_switch_003/continuous.mp4)为1,500控制Tick/1,501帧/50fps，CPU llvmpipe渲染零积分/推理；已抽帧核查。[关节/驱动诊断](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/reward_signal_003/saved_joint_drive_diagnostic.json)显示髋/膝俯仰主周期仍约1.17Hz，快档俯仰端点速度已接近名义4.189rad/s目标限速；髋偏航端点静态峰值95%占约8.67%/14.71%。端点/静态界不足以排除子步或动态饱和，也不能证明硬件最大速度。本轮总13,230,096 GPU/257PPO/5,140Adam，三份快速实验共39,799,120 GPU/771PPO/15,420Adam，与独立四频研究分账。原65→18、默认50Hz和生产模块保持。

第二轮含参考/接线/两层评估13,342,080 GPU、257 PPO/5,140 Adam；与首轮合计26,569,024 GPU、514 PPO/10,280 Adam，[两轮独立账本](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/second_pilot_closeout_002/combined_local_resource_ledger.json)。不重计148或子研究。独立频率研究12臂已经训练完成，320,864,256 GPU、6,144 PPO/122,880 Adam；目前进行自身频率、共同50Hz与真冷启动测试，正式学习/迁移结论尚未收口。银行与训练共享、确定性Torch已核对ONNX；本轮不授予完整M0-S、正式200例、ONNX独立运行或Bevy资格。

## 已收口训练路线

主线仍为统一MJCF、原生MuJoCo、mjlab／Warp／RSL-RL，然后独立Rapier／Bevy迁移。MD016的输入、合同和已有奖励源码已核对；学习其工作流和任务定义，不把61→14权重装到Goose65→18，也不声称复现无法追溯的原完整训练。

实际训练停止覆盖不足已定位：155的8个保存世界97,976Tick只出现7次移动转零、保持中位4Tick；恢复成熟2–4秒命令重采样后，162的92,160Tick有51次、中位75Tick。覆盖改善，但独立行为未通过，不覆盖148。165三轴课程完成11,796,480GPU积分／480PPO／9,600Adam，655.65秒；五检查点75条程序共47,329CPU积分，未获得合格新方向，[收口结论](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/continuous_axes_independent_eval_166/closure.json)。155未改善续训、147／158左弧线、151提步频及165同条件路线均已收口，不重复盲加长。

[累计资源账本v25](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/completed_move_ppo_resource_ledger_v25.json)记录40个完成的短主批、547,356,672GPU积分、23,232PPO、464,640Adam及约6.04小时批次计时；这是多个策略分支，不是单个已收敛训练或仅本轮十小时的统计。155中断批另列，链路、CPU、搜索与录像不混计。

## 固定合同与后续门槛

当前本体为`goose_task_proxy_11_discrete_mjlab160_v1`：21机器人刚体、18主动轴与2被动轴、11碰撞叶、10.430690821kg；65Actor／69Critic／18动作，真实动作历史和主动轴顺序保持。电脑能力训练使用刚性足底，不追加软底课程。

- 默认及Bevy目标的物理、驱动、策略50Hz，decimation=1；每Tick一次真实20ms积分、一次所选Actor调用，无隐藏子步。用户授权的独立频率研究及具名200Hz源端候选显式记录额外积分，不冒充该默认合同。控制状态需随暂停保存。
- 开发物理门槛固定：地面穿透≤5mm、自碰撞≤0.1mm、限位误差≤0.1mrad、直立度≥0.95。速度平均比0.8–1.2，组合动作同时满足比例≥0.6；停止1秒速度≤0.04m/s，随后3秒漂移≤5cm。不改变门槛凑通过。
- 独立评估采用确定性ONNX、关闭自动重置，保存全部失败。源端、零样本目标、目标微调和游戏成绩分别报告。
- 完整M0-S、M0-T仍未授予资格；当前训练是有界开发pilot。每个新批先做小链路和实际吞吐检查，固定版本／问题／指标；未改善两次回到模型或成熟工作流，不追加同条件长训。
- 后续仍须补齐后退／侧移、两方向弧线、长程停止与任意切换，再验收原0.4／0.7及游戏。先利用已有Goose移动与纯转向成功轨迹做组合对照，区分接触／控制与策略适应的差异；不重启上述同条件续训或失败Teacher。没有新批等待另一Agent或实体制造。

旧逐批历史完整归档于[收尾前计划快照](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/source_move_closeout_177/plan_before_closeout.md)，仅为证据，不是当前派工。此文件继续作为唯一现行计划入口。

# Goose V0.1｜Bevy Sim2Sim 长期训练计划

版本：2026-10-07。实施分支：`codex/goose50_training`。

**最新移动目标（2026-10-06）：** 前进走路 **0.4 m/s**、跑步 **0.7 m/s**，PC按住Shift＋前进加速，松Shift回走路，松前进停止；Shift单独不移动。学习Godot真实输入顺序、转弯中松Shift、反复切换、失焦和暂停恢复的回归经验，不能只给训练速度上限而漏掉游戏切换。优先由同一公共65→18移动Actor通过现有三维速度指令覆盖两档，不新增私有模式观测、不重置物理/驱动/历史。后退0.15、横移0.1m/s、yaw0.6rad/s及左右±π掉头保留。旧0.3及+.06只属历史/中间课程，不是最终目标；两档持续实际COM速度、路径、步态、启停及切换分别验收，不能以请求值或瞬时峰值授能力。自然鹅论文与Cassie官方成绩只作生物/成熟工作流参考，不当Goose速度可达证明。

**现行人类指令：继续推进，同时先把“训练方法还是机械设计阻断移动/起身”定清楚。** 不主动联系、读取或监控硬件／暂停G1，不新聊天/PM。移动、恢复、转向持续并行；“转板”已明确为左右转向/原地掉头，不新增板上运动。既有资产/恢复研究agent的审查已完成并闲置；当前主实施负责恢复与集成，既有`bevy_feasibility`负责移动/转向，GPU有界串行，不以子agent为等待理由。十小时目标已超期，完整移动与恢复仍未交付，目标不缩小。临时产物严格ProjectBackups，生产仅对应crate，本文为唯一入口。

**当前执行入口（最新人类纠正）：先复现实际成功的完整工作流，再逐项适配Goose；停止把成熟函数/组件来源当成熟能力继承。** root亲自负责恢复与集成，已重跑冻结MicroDuck起身12例及原任务短训练链路；既有`bevy_feasibility`独立负责移动/转向，已收口Goose前进桶失败批次，正在重跑MicroDuck实际成功的Walking工作流，持续并行。准备、CPU回放并行，GPU有界串行，不以agent为等待理由。恢复下一项是据已复现任务生命周期明确关节/驱动角色与真实高度测点，先适配一个完整合法前倒起身＋3秒站稳，再扩四向/扰动/继续移动。模型、公共65→18、原真实驱动和50Hz/1保留；MD14索引/角度表、115mm虚拟根目标、61维normalizer不能直接搬入。当前失败tracking参考不合格，不再作为可直接晋升的专家；保留其实际收据。Gaussian单位、静止起点、纯前进桶及旧seed/奖励/网络/solver网格均收口。十小时目标超期，Goose完整走跑/转向/四向恢复仍active、未交付，不归咎未证实的硬件缺陷。

**2026-10-07成功工作流复现（本批最新）：** 冻结MD native010/BAM/50Hz、原Stand61→14、12原案例（站/坐/前倒/后倒×yaw0/±.5）、500Tick/例、确定性独立ONNX且无reset，实际12/12通过原门槛，12例均站稳≥3秒；零yaw前倒原恢复时间2.18秒。亲自重跑原同版本Pollen StandUp任务/20奖励配置、重置/命令/Actor导入与normalizer，25Critic预热＋1真实Actor更新（26PPO/520Adam/159744CPU物理），复载动作误差0、导出最大1.90735e−6；更新后独立再12/12。14实际函数源码与19训练模块哈希通过，Actor确实更新且normalizer逐位不变。共171744新增CPU积分、GPU物理0但策略/PPO实用GPU；12512显式ORT调用、2静态FK，展示0积分/FK/推理。原200次训练与原始预训练Actor没有从零重训；成功的是MD目标端短链路，不是Goose资格或MuJoCo长训。新实际前/后倒录像已完成并检查0.02/2.18/9.98秒。

[原成功工作流实证报告](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_microduck_recovery_reproduction_001/report.md) · [新实际成功录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_microduck_recovery_reproduction_001/actual_microduck_front_back_reproduction.mp4) · [训练链路/哈希对照](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_microduck_recovery_reproduction_001/workflow_reproduction.json) · [Goose语义移植边界](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_microduck_recovery_reproduction_001/compatibility_audit.json)。

**Goose上一并行实验已收口：** root原静止起点修复后仍288Tick/hold0；移动原生纯前进桶292主更新后独立0/21，持续walk.07850/run.08195m/s，站漂367.43mm/floor11.67mm，走跑与双向转向均未交付。两项未晋升、不继续同分支。原奖励/物理/模型与失败证据保留；下一主线按完整成功工作流重建适配。root本轮与已收口移动批＋MD恢复短链路的新合计GPU物理2580964、CPU232627、448PPO/8960Adam，父/旧参考/历史3.52亿不重复计数；另在运行的MD移动复现待其独立账本完成后计入。

[移动292最终独立报告](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_native_velocity_workflow_admission_001/report.md) · [移动实际失败录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_native_velocity_workflow_admission_001/final_actual_walk_run_failure.mp4) · [本周期唯一资源合并](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_microduck_recovery_reproduction_001/parallel_cycle_ledger.json)。

**2026-10-07历史并行接棒周期（已收口，不是下一派工）：** root的`goose_native_tracking_start_resume_v1`完整复载上一教师Actor113/Critic302/learnable Gaussian/Adam/adaptive LR和原clock3072，仅改为安装原生start课程；原冷前倒qpos/zero qvel逐次核对，不投影、不扩限、不追加frame0到合格RSI集合。small100GPU/1PPO＋main786432/128PPO/110.85秒；2916个原静止出生全部实际步进，最长309Tick、参考477后曝光0。独立原前倒32→288Tick（.64→5.76秒）仍COM最高.16004m/up最高.34556/hold0，floor23.336mm/self2.253mm/soft range.094943rad。起始采样不足是真实缺口，但补齐后不足以解释或解决全部失败，不晋升、不追加同课程。原cold嘴/转子/连杆微小软范围偏差与原路径不合格仍披露，不授M0-S或公共65恢复。移动的独立方案为安装G1原生`rel_forward_envs=.2`，保持原稳定父、14奖励、Gaussian/Adam和绝对clock12288/扩档16896；主292真实1794048GPU/5840Adam，small后总1794144/293PPO/5860Adam，GPU已释放；原292结果已完成，最终0/21及实际曝光见顶部当前报告。

恢复已看实际4.98/12.50秒保存画面，教师仍低位趴撑，未完成收腿和COM抬升。安装原生终止函数逐Tick回放288态完全一致：第288Tick为头部参考Z差.251468m超过原.25m；实际5mm地穿第5Tick、自接触第28Tick先发生，终止和正式资格分开。上游奖励/终止在唯一forward刷新前读取身体运动学（50Hz时晚一实际20ms），使用reference_before；当前公共观测与独立资格按刷新后实际态计算。首次用post-state重构终止的探针错误且断言失败，288私有FK归档；修正原生pre-state顺序又288FK逐Tick通过，0新增积分/推理/优化，不据此判唯一训练原因或改框架顺序。恢复本周期合计786820GPU/129PPO/2580Adam/484ONNX样本、288独立FK/475展示FK/576诊断FK；不重复计入此前8605538GPU。

[原静止起点实际报告](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_native_tracking_start_resume_001/report.md) · [新实际三栏录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_native_tracking_start_resume_001/actual_start_resume_comparison.mp4) · [原生失败分段](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_native_tracking_start_resume_001/actual_native_failure_phase.json) · [本周期独立资源账本](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_native_tracking_start_resume_001/resource_ledger.json)。

**2026-10-07单位公式实验已收口：** 两线保留原生Gaussian可学习性，仅用G1`.25*effort/stiffness`换算`.25*peak/(kp*action_scale)`初始化；原18公共目标/确定性初mean、SI、奖励/50Hz不变，没有std网格。恢复small100GPU/1PPO＋main786432/128PPO/172.51秒，后段覆盖76.27%、最长642Tick、raw越界53/14155776；独立原前倒仅32Tick/.64秒、hold0、floor19.887mm/self12.651mm，不晋升或加同族轮数。移动small96＋main786432/128PPO/142.43秒完成；真实92与002真实92配对、128另测，均只有站立1/21，走速.01660→.07770→.09389m/s仍不足.4且偏航大；run.00213→.00083→.00764仍不足.7。128仍在192扩档前，.7例属未曝光范围泛化，不能冒充跑步课程完成。raw/terminated/truncated记录完整。两线所有GPU已结束，移动做CPU录像与指令域审计，root自己负责恢复下一问题；不拿对方结果代替本线判断。

**已核定的RSI起始缺口（历史，不是当前待修）：** 原生cold birth8076、后续有实际步8073（末3次reset未再积分），原frame0实际出生0；最早eligible15/.30秒相比frame0关节最多差.5749rad、root线速度(.141,.081,.558)m/s、最大关节速度2.129rad/s，原始qvel0。它不是原静止出生的近似复刻。本周期已用真正原静止start课程补齐，但独立起身仍失败；原GPU动作可到3秒稳定，全部路径接触/软范围仍未合格，下一轮须回到合格参考与反馈执行，而非继续盲加起点/后段样本。原frame0软范围小差仍如实记录，不投影、不改门槛或偷改出生。公共学生与其他方向待真实教师能力成立后再推进，不能把113参考教师当最终65。移动指令归一化域审计已完成，下一原生纯前进桶单项方案在本周期推进。

[恢复单位公式实际报告](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_native_tracking_action_units_001/report.md) · [原静止出生覆盖缺口](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_native_tracking_action_units_001/cold_rest_coverage_audit.json) · [实际失败三栏录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_native_tracking_action_units_001/actual_physical_gaussian_comparison.mp4)。

[移动单位公式三端点独立报告](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_native_action_unit_adaptation_001/report.md) · [本并行周期实际资源合并账本](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_native_tracking_action_units_001/parallel_cycle_integrations_ledger.json)。本周期仅七份新批次合计GPU8605538/CPU114889、1406实际PPO/28120Adam；旧父策略/旧参考与3.52亿历史不重复计入，FK/绘图/私有控制重构不作实际积分。

**移动指令域审计完成（0积分）：** 同一合法cold65（与原保存obs逐位相同）下，父Actor归一化计数12813312、命令.4/.7经`(x-mean)/(std+.01)`为10.2478/18.9681；新128为6.3152/11.7771。Y/yaw初始化后yaw±.6已约±1.877，实际课程也有yaw曝光，但同cold态新128 yaw+.6的18动作RMS响应仅.001368。输入外推和弱转向响应是确定性政策计算事实，不是实际运动或唯一原因；不能只改normalizer便宣称解决。4批/24 CPU Actor样本、12显式normalizer批、2cold私有FK、0GPU/积分/优化，全部模型状态逐位未变。下一移动方案据成熟步态/课程与支持序列证据决定，不扩大该公式/同父噪声网格。

[移动指令域零积分审计](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_native_action_unit_adaptation_001/command_domain_audit/report.md)。

**已核对的动作事实：** 恢复std1直接套原映射对应静态P请求1.47–21.67倍峰值（不是实际力矩）；teacher/RSI固定三块34214连续样本目标重构误差0，末块多颈/腿轴限速>90%，多轴实际绑定当前制动/热/功率cap。移动四块442368轴值没有裁剪边界，raw具体值未存但在这些样本可推出未越界；亲本std0.01332–0.03244、末0.01290–0.03228，合并slew11.97%、原峰命中0.593%。不能把两线失败归成同一个饱和问题，也不把相关性说成唯一因果。这两份审计新增积分/推理/优化0，私有控制重构34214另记。

[恢复单位与实际动作审计](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_reference_gpu_execution_audit_001/tracking_action_unit_actual_audit.json) · [移动独立动作审计](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_walk_run_commands_002/action_unit_audit/README.md)。

**冻结输入：** `goose_task_proxy_11_rigid_braking_v1`，MJCF SHA `3dd030f0475efd101ab129288e7f04f5694067757d85e0429933cecb9c5268e4`、合同SHA `d3fd7ba668678e1e1237d00bd7fac77e5d0625d77cc65bea85d3579ffb17061c`。21机器人刚体／18真实电机／2被动轴／11叶／10.430690821kg。每腿6主动轴；公共65→18、拾物82→18。9凸mesh＋2刚性足box；原鞋上部遗漏44.76%不继承实物资格。质量/完整惯量/轴位/嘴四杆/过滤/限速/力矩/热/350W保留；50Hz、每Tick一次20ms实际驱动/积分/当前Actor更新、decimation1，无隐藏子步/出生后实际root/q/qd/drive/history写入。电脑训练不考虑软底。

**最新定责：已经大量训练，也有低速双足运动证据，尚无完整合格能力。现有证据不支持“DOF不足/大壳把正常走路卡死”。Lab任务/观测/参考适配已有明确错误，责任由Lab承担；不能把3.52亿不同实验样本当一条成熟配置持续训练，也不能因此判PPO或硬件绝对不可能。** 尾段/原生Critic预热两组真实GPU与独立CPU/GPU回放均失败，薄恢复适配已收口；最新完整检查点native速度转接亦变成近静止，移动/转向均失败；原稳定候选保留。完整移动、恢复尚未通过；下述历史续训与LR代码修复仍保留，不重复计入本轮。

| 检查 | 事实与适用边界 |
|---|---|
| 到底训过多少走路 | 11份不同pilot收据、driver真实物理计数及同现行model哈希核验：351,565,824GPU世界积分、9,084PPO、181,680Adam；最高单批72,204,288。不重复cycle/parent/独立评估，不是一条恒定任务连续3.52亿训练。此前仅拿460,800起身pilot讨论全部训练量，范围不足，撤回该暗示 |
| 是否整个结构不能迈步 | 原guided512确定性ONNX独立20秒：前进1.372m、横偏−0.793m，upright≥.9959、两脚离地120/281Tick；足底11.822mm、heading−.9745rad仍失败。该Actor60秒站漂1.164mm且5mm足底门过。最终Actor源CPU/GPU20秒1.648/1.662m行为相符，但偏航/漂移仍失败。不是完整移动/Bevy资格 |
| 有效DOF | 每腿hip yaw/roll/pitch+knee pitch+ankle pitch/roll；名义足姿Jacobian两侧rank6。编码器knee0已实际弯59.01°，非直膝奇异；原物理弯曲约47.56–162.15°不能全伸直，但原低速轨迹没近边界，不据此增轴或扩范围 |
| 重心 | 名义SI COM高282.08mm、双脚轮廓余量58.27mm；在单脚轮廓外约46mm，需要侧向移重。完全不移重的假设ankle-roll7.8Nm超过4/2Nm，且COM在支撑面外；不是换8Nm电机就可静稳的理由 |
| 壳卡腿 | 1235近名义直立保存姿态的实际contact为self0/非脚触地0；guided20秒24抽样亦0。倾倒/深蹲有torso-shin干涉；真实槽口最大联合摆角未由11凸叶/安装过滤认证。大cutoff geomDistance与实际contact矛盾的负值单列未获资格，不冒充真实碰撞 |
| 最划算改哪里 | 先补移重→卸载→抬脚→落脚的训练/控制闭环，保留6轴。机械重点候选是ankle-roll持续能力/传动及头颈减重/可下移重件；上颈/头2.442kg、占23.4%、COM高531mm。hip-yaw持续能力随后核。尚未消融，不新需求/不直接切壳/加电机；必要时具名微调并记录最终统一交付 |
| 电机是否普遍不够 | 原最终guide膝超过连续设计约22.4/30.1%Tick，左膝≥95%峰5.1%；hip/anklepitch多数未长期峰饱和。连续设计额不是实物热资格，不用整体350W或偏航直接归咎全部电机 |
| 速度测点 | 同实际状态whole-COM Jacobian审计：root vs COM水平速度RMS差.124–.164m/s；COM速度RMSE仍.167/.170m/s且偏航/穿入保留。测点适配确有问题，换指标不能让错误步态合格 |
| 恢复采样错误 | 原Lab自加5mm普通终止两PPO3072样本结束1266次；原生50mm开发逃逸9次。修后300更新460,800GPU真实优化，但0段完成参考、2078头高终止/1逃逸。正式独立5mm/self资格不变 |
| 起身是否结构全局不可能 | 同原公共18动作626Tick可末150Tick真实正确站稳，但全路径10.0978mm地穿/jaw.08258rad越界；未合格。头高参考来自真实FK，投影joint目标播放不能代替原控制提前量/反馈或证明RL不可能 |
| 接管诊断 | 原前175Tick保全历史并逐元素等于基线：双脚49.53/52.93N、self0、地穿1.395mm、轴越界0、但up.121/COM166mm并未站稳。只换冻结300Actor451次ONNX后保持0、9Tick后self。冷起与该入口都失败，该配置收口，不仅靠取消头高终止或无限续训 |

**最新支撑与GPU证据（已收口）：** 原guided512同公共动作1000CPU回放，q/qvel/力矩/观测逐位相同；39段持续离地时另一脚平均载荷中位数95.85%体重，self/nonfoot0。71步/30入口超过5mm；最大11.822mm来自支撑再接触，Tick956最低点+0.0222mm/预速度向上、实际该脚力0，20ms后−8.2919mm，当时腿力矩/转速未近上限。39次首次落地只有2次立即超5mm，不能全归因抬脚前没移重或首次下降速度。动态COM投影不当静稳失败证书。

同原512检查点、现有private-FK修正、原奖励/物理/驱动的两组各256世界×24×128PPO，只改普通训练终止50mm/5mm：50mm组1280回合平均597.55Tick；5mm组61524回合平均12.724Tick/0.254秒、最长105，无完整超时。后者破坏已有步态，分支收口，不继续门槛/seed/噪声/权重搜索。开发组独立站立60秒COM漂5.16mm/地穿2.653mm、停止速度.01536m/s/漂2.70mm；走路20秒前1.883m/横.285m/yaw.426rad/地穿9.712mm，移动未过。嘴闭环软范围另报；1e−6是本短实验预声明诊断标志，不当用户新增站立门槛。严格组移动1.84秒跌倒，站漂15.10cm，未晋升。

**最新续训对照（收口）：** 同原512/50mm任务/seed109/256×24×128，只修原生adaptive标量恢复；786432新GPU/128PPO/2560Adam，独立4600CPU/4600ONNX。学习率已正确恢复7.59375e−5，但独立20秒前进1.713m/横偏.545m/yaw.771rad/地穿12.206mm；未修控制组1.883/.285/.426/9.712mm。超5mm步从57→38不能抵消极值与路径变差；没有能力改善，不继续LR/seed/权重搜索。修复保留，因为真实缺陷仍需修。站60秒漂6.17mm/地穿2.732mm、停止速度.01862m/s/后3秒漂2.437mm；全走停地穿14.060mm，不能晋升移动。嘴/被动软范围诊断另报，完整目标不缩小。

**策略记忆对照（已收口）：** `goose_recovery_gru65_v1` 与同条件fresh MLP各256世界×24×128更新完成，新增1572864GPU/256PPO/5120Adam；独立5200CPU/5296 robot ONNX（含96导出对照）。同Critic92/初始权重、原reward/权重/PPO/出生28/plant/drive/50mm普通开发逃逸，仅安装的原生RSL GRU256×1与MLP对照；公共65物理量和18动作保留，ONNX具名显式h_in/h_out(1×1×256)仅来自公共观测，每Tick携带。3项native reset/序列Torch-ONNX-JIT/复载测试过，合成测试另13 ONNX/0积分/0Adam。两组原四出生28–31均保持0；GRU原后倒up偶尔.9999，但COM最高.1983m，达不到名义高度85%/真实脚支撑/3秒保持。已看实际四方向前期/末尾失败录像；不晋升、不接Bevy stateless loader，不追加memory-size/seed/reward网格或同配置长训。

**已核定的采样缺口：** 两组各786432训练样本中，到参考站稳后段477的样本均0，实际hold全0；最长原冷前倒回合control249Tick/GRU287Tick，3508/3553终止除各1次接触外全tracking失败，无超时。不是“没训练”，是当前冷生课程没有覆盖站稳后段；不能由此宣告硬件全局不可能。原普通5mm终止组、恢复300Actor/投影/handoff和solver/margin/MPPI旧家族仍收口。

**力矩绑定核对：** 五条原真实pre-state/target/thermal/tau重构误差≤1.97e−14Nm，0实际积分/网络/优化器、仅私有构造forward2。旧300在175–259段85Tick有slew，其中72Tick实际当前tau会被它改变，81Tick热限额绑定；纯代数去slew反事实最大差20.259Nm，不当合法轨迹/换电机依据。新GRU同episode段slew0/85/力矩差0却仍不起身，且轨迹不同，不能把slew认作唯一根因；walking全20秒无slew，不能把其穿入继续归咎这个过滤器。原公共动作相应只有2Tick受slew、热0；原成功终点也不是完整路径资格。

**参考状态尾段课程与Critic预热已收口：** 名义E/near426合法筛查后，原512独立CPU保持648/637Tick，地穿2.114/2.277mm；纯零目标E亦可稳住。lean351嘴轴越界.000255478rad被排除，没有投影/改门槛/搜索替代出生。原author参考尾段477/426索引已冻结，名义COM采用真实reference/runtime .284075930m；旧Q使用.279758m，其最高约.198m仍失败，两者不改旧资格。直接尾段128PPO覆盖参考终点525132worldTicks，但训练hold最大2；原生Critic-only32＋joint96覆盖237414、最大1。两份独立ONNX在合法锚点保持6/9和0/0，四原出生28–31均0，均拒绝。不能只补晚段采样后继续加训；不展开更早倒地初态或warmup/LR/noise/seed/奖励网格。Critic预热Actor/normalizer/std全状态逐位未变、Actor Adam状态0，链路确实工作；96与128joint更新不同，因此失败不证明Critic是/不是唯一根因。

**失败位置已进一步限定：** GPU确定性ONNX也复现原策略648/637、新策略6/9、预热0/0保持，不只是CPU迁移。原20Tick/256world接线q误差8.38µm、tau.000954Nm、脚力.0768N；长不稳轨迹有接触分歧，逐项保留实际误差。原source21体/18轴/11叶/驱动/50Hz未改，近站立试验不能判定缺DOF或要求CAD改动。作者原AST/权重在同实际near426轨迹离线评分129.2695原／113.6332新，没有发现这组的奖励排序逆转，亦不证明奖励完整。相同65观测上整体动作RMS变.083641，仅替换normalizer变.004705，主要变动在网络更新；不是反事实物理轨迹。后1秒新策略angular-speed不合格412/600Tick，feet-normal不合格0，持续摆动已由实际录像检查。此前“更晚状态没采到”仍是事实，但不足以解释全部失败。

**本轮资源与失败透明：** 两个真实256world×24×128pilot共1572864训练GPU、256PPO／5120Adam，其中32／640为Critic-only，joint224／4480；桥接5120、独立GPU3900、失败时间断言143，GPU合计1582027；CPU13000、robot ONNX17143＋合成3，离线/录像均无积分。GPU评估首稿误将float32累计时钟与十进制理想时间绝对比较，143步计入成本且traceback保留；当时内存轨迹未保存，不能用于资格。修复保持逐Tick真20ms和独立float32累加检查，另报十进制累计偏差，不改物理/门槛。两个setup错误亦归档；所有作业已结束，未唤醒agent或联系硬件/G1。历史样本不重复入账。

**最新全指令native续训已收口（2026-10-06）：** 先核历史账本，command expansion21,086,208GPU和native14/COM-point全指令训练确已做过，没有把它们说成未训。新具名`goose_native_velocity_resume_v1`改用原guided512完整65/69/Adam＋已有adaptive LR修复＋受保护unusedY/yaw初始化，接安装mjlab1.3原生14奖励；原数字/函数、50Hz及本体不变，Goose名字/上部原standing .05/足传感器和已有torso rigid-body COM测点适配显式冻结。与旧mean-Markov1024/fresh Critic/Adam分支不是单变量因果比较，不称完整G1工作流。1000个原实际65观测的zero/forward输出初始化前后逐位一致，完整Adam逐位保留；回收LR7.59375e−5。两次和最终128次真实PPO/ONNX检查通过；256×24×128=786432GPU、128PPO/2560Adam、115.20秒，全部完成rollout保留，没有长训。

**本批独立结果：** 原/新60秒站立漂2.29/14.95mm且5mm/self门通过；同.06m/s20秒原前1.373m但横−.801/yaw−.974/地穿11.822mm，新仅前.01216m。新.3m/s前.2883m/地穿14.837mm；后退−.00774m；双横移都约+.01m；双yaw实际−.0755/−.08945rad，±π原地掉头均未完成。只站立通过开发项；单独停止速度通过不替代失败的移动→停止完整链。10个固定完整指令案例＋原低速同command对照均保存，无重置拼接，无晋升；200移动/四类恢复/自然扰动/继续1m/Bevy仍未过。已看实际6panel录像3.52/15.92秒，确认新策略主要静稳，未凭reward上涨授能力。

**本批验证与错误透明：** 2profile×2world×20Tick=80CPU，同原实际18动作下质量/完整惯量/几何/轴/材料/过滤/四杆/电机/options、q/qvel/qacc/warmstart/ctrl/clock、target/thermal/tau/history、contact/efc合力和65观测逐位一致；9项奖励来源/实际native分档/command transfer/复载检查通过。中途把native函数默认walking_threshold .5误当实际任务配置的口头判断已撤回：实际显式.05，实际native类回归确认0与前/后/侧/yaw用不同分档，未据该错误改参数。initial RSL API setup失败0积分/更新与synthetic fixture首次错误均归档；初稿native日志误落Sai_Lab根已确认归属后完整移到备份，后续cwd备份。合计本轮CPU15680、GPU786432、128PPO/2560Adam、robot ONNX15600＋3导出API/189实际样本；私有FK/录像不作积分，旧资源不重复。全部作业完成，未联系硬件/G1、唤醒agent或改硬件。完整协议、未通过清单、已看录像、原失败及资源澄清在[本批报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_velocity_resume_001/report.md)。不追加同族父策略/LR/seed/噪声/奖励参数网格或同配置长训。

**最新接收结果（2026-10-06）：** 同一刚性21体/18电机/20树坐标/11叶已接回Rapier，具名 `goose_rigid_native_plant_v1` 只授机械接收身份。冷出生保留源绝对q并正确偏移限位/反馈，0额外root lift；真实CONNECT两端局部销点与三平移约束保留。20次独立冷进程一致，初始65观测误差≤5.961×10⁻⁸、逐体位置≤2.753×10⁻⁸m、完整惯量相对≤7.969×10⁻⁷。旧sampled-PD在此合同下明确拒绝。19项回归、4输入拒绝路径通过。

**共同外力诊断已收口：** 两端均施相同18真实电机外力，源端显式关闭position actuation；这同时去掉源隐式速度导数，因此不是Actor迁移，不能把倒下归给原PPO/机械或把所有差异归给接触。正确PGS4下，目标零力矩40Tick最大入地27.051mm、原guide前100Tick外力84.315mm；对应源2.222/11.802mm。目标自碰撞仍超过门槛。零重力嘴40Tick销点误差≤20.896微米。短场景物理边界P95为0.219–0.901ms，不再是旧51ms问题，但不含Actor/渲染/查询，不授游戏性能。初稿漏接PGS4实跑PGS1的180步及全部失败保留，修正未引入时间子步。新CPU源180＋目标380（初稿180、正确180、冷20），另通用时钟/弹簧fixture107；GPU/PPO/真实Actor推理0。最终100帧实际位姿并排录像已看；渲染失败与0积分账本单列。

[本批接收/动态失败报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_rapier_rigid_intake_001/report.md) · [实际共同外力接触录像](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_rapier_rigid_intake_001/actual_common_effort_contact.mp4) · [具名初态身份核对](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_rapier_rigid_intake_001/named_identity_receipt.json) · [新资源账本](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_rapier_rigid_intake_001/resource_ledger.json)。完整源/目标接触门、移动/转向/恢复与Bevy均未通过；原稳定源策略保留，不据裸力矩倒地切壳/加轴。

**本批原生目标驱动接回完成（2026-10-06）：** `goose_rapier_force_based_drive_v1`保留18目标/slew/名义颈前馈/签名制动/热/350W与65含义，17上游ForceBased位置电机＋真实嘴转子；读取实际内部电机冲量/dt而非未写回的公开字段。600纯源控制端口误差≤2.85e−14；500步观测修改前后原状态/冲量逐位相同。自由root四元数漂移在159真实Tick触发原守卫，积分内维持单位旋转修复（不放宽门槛/增加步/搬姿态），数学红8/绿5000；100Tick正常Actorroot差0.206微米，但旧失败裸力矩轨迹后段可差42.6mm，不冒充全部物理逐位一致。55测试通过、10专用旧fixture保持ignored、5建世界前拒绝。

**最新实际能力仍未达标：** 原Actor目标端60秒不倒，但最大漂移72.8mm>50；前进0.06的20秒位移(+.5914,−.0263)m、偏航+.1289rad、P95物理+推理约.181ms，最大入地.956mm/self0；两脚完整离地>2mm为0/0，不授合格抬脚步行。后退/横移/双向转向均响应不足。原guided512冻结课程确实只含站立/+.06前进、y/yaw范围0，不能把未覆盖指令单归硬件/迁移；旧完整指令native128失败分支仍关闭。20独立冷Actor×20Tick＋18轴双向小脉冲×30Tick，56短案例全过，仅名义小幅准入，不等于完整M0-T/跌倒资格。同原实际源动作1000Tick保留真实目标原生驱动回放，目标位移(−.0438,−.4505)m、最大入地2.756mm/self0，足离地0/8 vs原源120/281；差异在Actor推理之外仍存在，尚不能唯一分解电机/接触贡献。源reported actuator_force和目标最终impulse/dt不宣称同种平均力矩。新增CPU世界积分10619/真实ONNX7359、GPU/PPO/Adam0；模拟控制/数学单测/渲染另列，旧样本不重复。

[本批报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_rapier_native_drive_001/report.md) · [实际正常速度瓶颈录像](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_rapier_native_drive_001/actual_native_target_bottleneck.mp4) · [同动作实际回放](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_rapier_native_drive_001/source_actual_actions_metrics.json) · [资源账本](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_rapier_native_drive_001/resource_ledger.json)。未联系硬件/G1、唤醒agent或改机械。冻结代码/二进制与复现入口均备份。

**最新无地面电机定责（2026-10-06）：** 具名隔离环境取消地面/重力，保留21体/18电机/11叶/完整SI/armature/阻尼/过滤/CONNECT；无Actor/任务前馈/限速/热，仅17原kp/kd、同绝对冷q±.002rad及固定原峰cap、真实转子0，不能当完整驱动资格。35案例（17轴双向＋零基线）均源/目标50Tick、有限/时钟/无接触，源各轴最大响应.001813–.002237rad，目标最大2.323e−6rad。单变量同时取消两端关节frictionloss，右髋pitch目标从<4e−8rad恢复到末.001999994rad；正式摩擦没有改。源friction rowR2.48614和dmin.9有限阻抗，目标只搬峰值、默认1e6Hz关节CFM约6.33e−11形成近硬静摩擦，数值合同缺失明确由Lab负责。去摩擦后首步源/目标.060717/.036404rad/s仍差约40%，与implicitfast/ForceBased离散差异一致，不能将剩余差唯一全归一个项。已看实际响应曲线；不据此判断DOF不足、切壳或改电机。CPU新3601（源1801/目标1800）/GPU/ONNX/PPO0，pilot复用只入账一次；3建世界前拒绝通过，启动/编译/精确JSON断言失败均保留且0积分。首pilot旧binary未单独复制，仅当时hash/源码/轨迹保留；新最终代码/binary冻结。原生Actor上一批10619/7359不重复计。本批无需外部交付。

[无地面定责报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_motor_isolation_001/diagnosis.md) · [真实响应曲线](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_motor_isolation_001/response_comparison.png) · [本批资源](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_motor_isolation_001/resource_ledger.json)。完整移动/双向转向/±π掉头、四向恢复与Bevy仍未过。

**最新独立摩擦阻抗候选（2026-10-06）：** `goose_rapier_independent_joint_friction_impedance_v1`只分离干摩擦CFM配置入口，复用原生摩擦约束/同一求解器，源dmin.9映射为(1-d)/d=.111111；真实峰值、SI、PD、限位参数、嘴闭环、碰撞/过滤和20ms不变，默认None旧50Tick状态/速度/努力/快照逐位一致。目标R采用当前增强逆质量，源为名义invweight0/不同参考加速度，不宣称精确等价。20关节逐名完整/hash/公式绑定；新plant SHA948136e93f4db5733bf1f4a5be24bc531fe2c085a39027f94064a148f2a4a312，旧输入未覆写。34双向小目标全恢复，源/目标末位置差最大3.3063e−5rad；35短隔离均无接触/self/finite/时钟问题。50测试/4建世界前拒绝过，10旧专用fixture ignored。

**但Actor技能没有实质改善：** 新零速60秒漂80.7mm（旧72.8，两者失败）；前进20秒.5976m/横−.0290/yaw+.1176，最大地穿.969mm/self0，完整足离地>2mm仍0/0。无头P95物理+推理.183ms，不是游戏资格。摩擦缺口不足解释全部步行失败，不晋升M0-T/Bevy，不展开CFM网格。新增目标CPU5900/真实ONNX4100，源/GPU/PPO0；默认50＋候选1750（pilot复用一次）＋Actor4100，原源1750和旧批次不重复。Tuple/测试API编译、numpy bool序列化、目录/Git cwd错误全保留无物理重跑，统计修复离线0步；原35完整物理结果先保存。未唤醒agent/联系硬件G1/改机械。

[本批报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_friction_impedance_001/report.md) · [原Actor实际结果](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_friction_impedance_001/actor_metrics.json) · [抬脚失败](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_friction_impedance_001/foot_diagnostic.json) · [本批资源](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_friction_impedance_001/resource_ledger.json)。完整移动/左右转向/±π掉头、四向恢复未过。

**最新位置离散候选已收口（2026-10-06）：** `pre_position_implicit_velocity_v1`仅用上游ForceBased velocity motor表达真实当前q位置项＋原kd隐式D，公共18位置动作/原kp/kd/同摩擦/峰/限位/驱动/20ms/SI不变，真实嘴转子不变，不是精确MuJoCo力矩等价。None模式新50Tick完整物理trace与父逐位相同；右髋+.002位置RMS源差66.97→23.47µrad，速度RMS反而略差，不扩35轴或网格。新plant SHA60a9e98925f751afcba2d549961ec0f838d1537460cc592e1e0ea1379645e383。原Actor60秒不倒/漂39.19mm，该单例漂移过；前进20秒(+.06178,−.55992)m/yaw−2.35389，floor≤.964mm/self0/up≥.99597，两脚完整离地>2mm从0/0变4/101，但严重偏航失败。48测试/2建世界前拒绝过、10旧fixture ignored；新CPU4200/ONNX4100、GPU/PPO/Adam0；旧源和父样本不重计。已保存完整20秒实际原速并排视频，视觉检查249/499/999帧，绘图0积分。

[本批报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_velocity_discretization_001/report.md) · [实际驱动对照录像](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_velocity_discretization_001/actual_discretization_comparison.mp4) · [资源](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_velocity_discretization_001/resource_ledger.json)。两轮局部目标驱动适配未交付步行，停止扩大Rapier内核匹配，候选具名诊断保留，不替换默认。

**最新源端原生奖励实际审计（2026-10-06）：** 两真实GPU世界、相同+.06命令/原SI/公共驱动/65/18/50Hz/原生14函数权重、不训练，原guided512位移(+1.50185,−.65205)m/yaw−.56797/foot12.455mm，native128近静止位移(−.01080,−.00257)m/yaw−.0145/floor3.266mm。原奖励率4.25081 vs5.74174，后10秒4.22878 vs5.74521，近静止更高；差值主要角速度+.93867、线速度+.28261、pose+.26429。旧步态本身摆动/偏航/穿入失败，不能据此要求奖励不合格动作、宣告全空间最优停止或唯一根因。stock速度std=.5m/s使理想停止仍得+.06跟踪98.57%，单位适配问题待受控验证；stock air_time权重原0非接线漏项，walking_threshold实际.05不复用撤回假设。完整1000×2×14原manager加权率与total校验保存（manager净化前原值未独立截获，不冒充逐函数有限证明）。新增GPU2000/ONNX2000，CPU/PPO/Adam0，全部结束，无硬件/G1/agent等待；完整20秒实际源录像存证；首次漏开碰撞显示组空画面保存，修复仅绘图组；渲染失败＋修复私有FK共2000/0积分/0额外Actor，已视觉检查最终249/499/999帧。

[本批奖励审计](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_reward_scale_audit_001/report.md) · [实际分项](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_reward_scale_audit_001/metrics.json) · [实际源端对照录像](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_reward_scale_audit_001/actual_reward_ranking.mp4)。未升级依赖，不把公式/评分替代资格。

**最新低速容差配对已收口（2026-10-06）：** 同guided512完整Actor65/Critic69/Adam、seed127、+.06/20%站立、原生14奖励/权重/原SI/驱动/20ms，仅std=.5 vs .03。完整配置恢复std后相同，两组各256×24×128真实PPO。独立20秒前进仅34.48/32.91mm、完整足离地0/0和0/0、地穿16.746/15.175mm；站60秒漂79.19/20.62mm但后者地穿10.011mm；停止后速度.05509/.06554m/s。均失败，不继续std/seed/权重网格或同配置长训。首Actor/Critic/Adam和首动作逐位相同，但GPU首Tickqpos已差5.126e−5（混合单位），任何优化前首24Tick最大差.08833，未冻结全部pre求解器初态，不把变化全归因于容差。新增1572864GPU/256PPO/5120Adam、9200CPU、9578ONNX含378导出；评价FK9206/绘图1000均非积分；两组全部作业结束。

[配对实际报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_forward_scale_pilot_001/report.md) · [两组实际失败录像](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_forward_scale_pilot_001/actual_scale_pilot_failure.mp4) · [资源](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_forward_scale_pilot_001/resource_ledger.json)。固定资格不变，原稳定Actor保留。

**最新原生BC闭环已收口（2026-10-06）：** 安装RSL5.0.1 Distillation/MLP256×3/Storage、原MSE/Adam、原生normalizer一次拟合、seed131，626实际pre65→18离线1000原生更新/3.849秒，0新GPU物理/0PPO。初预检发现旧规划首previous-action为measured-q而非零；原626零历史回放的q/qvel/tau/target/thermal/time/afterobs逐位等于原轨迹，据真实回放对齐首preobs，后625条原pre逐位一致；失败保留0积分。训练RMSE .008781、最大误差.638855（row125头pitch），末段RMSE .000940；626 ONNX导出样本最大误差7.749e−7。独立原冷前倒28实际626Tick，hold0、COM最高.18901m/up.50279，地穿28.477mm/self13.246mm，Tick11自碰。首obs逐位匹配，但首动作RMSE .003484后逐步偏离；单路径覆盖不足的证据，不证明唯一原因/65不可学/机械不可能。不继续拟合轮数/网络/seed网格，原动作路径本身亦未过资格。

[原生BC报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_recovery_native_distillation_001/report.md) · [实际12.52秒起身对照录像](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_recovery_native_distillation_001/actual_BC_failure.mp4) · [独立收据](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_recovery_native_distillation_001/independent_receipt.json)。新增CPU626/ONNX626；另导出1batch/626样本，1000原生Adam离线呈现626000数据行，不当新增物理样本。评价FK626/失败相机及修复绘图FK1252、4显式cold-forward均非积分。Torch SM提示与ONNX旧导出警告记录，实际更新与对照通过，未升级依赖；所有作业结束。

**成熟跟踪资产已接通：** 原冷出生＋626真实状态转为安装mjlab1.3/BeyondMimic MotionLoader的627帧50fps、21个nonworld body-link世界姿态/速度、20树坐标（含2被动轴）、18真实motor不变。全部六字段由原Loader逐位加载验证；世界速度为body-link原点Jacobian，不混COM，原converter/command同测点已核。0积分/0PPO/0Adam、私有FK627；参考不投影/不改动，仍未合格。当时只是格式准入；本日后续教师实际训练结果见当前入口与恢复线。[资产及边界报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_mature_tracking_asset_001/report.md) · [原Loader接收收据](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_mature_tracking_asset_001/receipt.json)。

**恢复线当前问题：** 安装mjlab/BeyondMimic训练专用Actor113/Critic302反馈教师已经实现；保留原9奖励函数/权重/std与原生RSL、原18轴driver/20ms/decimation1。fixed-start完成small104GPU/1PPO及主786432GPU/128PPO，独立16Tick在0.32秒终止，COM最高0.17255m、地穿25.37mm、保持0；25488终止中25415含原生末端位置，最高参考330帧，未到477站稳段。Admitted-RSI只在真实冷重置调用原生writer，running与clip结束禁止写实际状态；原生adaptive提议映射到241个既有几何/限位合格MotionLoader帧，分布改变与13bin中的5空bin明确报告，不能修复原参考资格。它完成small100GPU/1PPO、主786432GPU/128PPO；59.56%样本进入站稳段参考，但独立原前倒28在96Tick/1.92秒终止，COM最高0.15818m、地穿24.30mm/self18.40mm、保持0，未改善完整起身。两次失败后不追加相同seed/权重/网络/轮数网格，原CPU参考GPU执行审计已完成：626实际Tick，9.54–12.52秒保持150Tick，0普通终止；最大地穿15.86mm、关节越界0.08598rad，目标序列误差≤4.748e−8。该单路径不是结构绝对不可起身，教师执行/学习差距仍由Lab解决；下一步核对成熟奖励/参考/持久drive状态的兼容性，不盲加样本。公共学生采集尚未启动。原参考全路径仍有10.10mm地穿及关节越界；627帧中21地穿/379关节超限（可重叠），仅241出生可用。参考格式、覆盖率、PPO和导出成功不能当恢复成功。M0-S完整接触未过不长训，M0-T未过不目标训练/Bevy；完整200移动、左右转向/±π掉头、四向/自然跌倒起身3秒保持并继续1m仍未交付。没有硬件或agent外部等待，不主动联系/读取硬件与暂停G1，不新聊天/PM。必要机械具名微调最终统一交付。

[fixed-start真实报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_tracking_teacher_001/report.md) · [RSI实际报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_tracking_rsi_001/report.md) · [两教师实际失败录像](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_tracking_rsi_001/actual_teacher_rsi_failure.mp4) · [原动作GPU可执行性报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_reference_gpu_execution_audit_001/report.md) · [实际CPU/GPU起身对照](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_reference_gpu_execution_audit_001/actual_CPU_GPU_reference.mp4)。

**移动线本周期结果：** v1实际load继承common_step12288，绝对4608课程门导致本地74Tick就扩档，376主PPO保全失败，不当正确两阶段课程。v2强制具名course_start_step取真实metadata（文件iter510不等于512），并首物理步前核对load；前192更新1179648GPUworldTick无run-band，后320更新1966080GPUworldTick有68358个run-band，原生reset在阈值后1Tick更新范围。完整512主PPO/10240Adam/492.17秒，small1另计；ONNX差1.788e−7。独立21例均失败：walk0.000776、run0.062513m/s，地穿15.783/16.604mm；站60秒漂155.60mm/地穿16.242mm/yaw−1.441；±π掉头未到，所有输入/Shift/停止组合不授能力。课程接线/曝光通过不等于技能通过。001+002总账CPU54502/GPU5458624/PPO890/Adam17800（含失败/设置/导出/实际复验，勿叠加目录累计再次计数）。现有稳定guided512保留，失败v1/v2不晋升、不作为后续父。

[移动v1完整失败与计数缺陷](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_walk_run_commands_001/completion.json) · [修复v2完整收据](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_walk_run_commands_002/completion.json) · [实际走跑/切换失败录像](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_walk_run_commands_002/actual_walk_run_failure.mp4)。

[尾段完整报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_recovery_state_course_001/diagnosis.md) · [Critic预热报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_recovery_value_warmup_001/diagnosis.md) · [实际尾段/四方向失败录像](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_recovery_state_course_001/actual_tail_course_comparison.mp4) · [预热实际录像](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_recovery_value_warmup_001/actual_valuewarm_comparison.mp4) · [联合唯一账本](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_recovery_state_course_001/resource_ledger.json)。冻结payload/manifest见两目录；完整资格均未通过。

**成熟工作流核对：** HumanUP本身没有Actor参考clock，却有10帧卷积history+RMA/DAgger；当前只复用奖励函数/权重与mjlab PPO，不能叫完整HumanUP。Goose持久target slew/thermal未在65 Actor或92 Critic中，上一请求动作不是已实现target；原公共626动作15Tick限速，而失败300Actor644/650Tick限速。实际抬升阶段175–259原动作2/85 vs坏Actor85/85有滞后。这是待验证关联，坏动作也会造成滞后；不据此宣布必须RNN、改限速或归罪机械。此次GRU属于原生RSL策略记忆受控试验，明确不冒充作者的CNN/RMA。

**机械修改规则：** 现行无新CAD或合同改动，也无硬件交付等待。若实际指定动作被几何/力矩阻断，才起具名最小增量，记录质量/惯量/轴位/范围/来源差分并同初态/动作验证；不主动找原硬件聊天，最终一次汇总给工程。现有低速运动不授予流畅/快走/硬件资格。

**成熟链路和门槛：** 统一MJCF、原生MuJoCo、mjlab/Warp/RSL-RL与MicroDuck组织。名义源数值/真实小GPU/PPO链路已过不是完整接触域M0-S；不在对应源门未过时长训。M0-T未过不目标训练/Bevy资格。训练与独立评估分开，200全指令移动、四方向/自然跌倒、3秒保持/恢复后1m/游戏集成未通过仍未完成。原函数来源验证不能冒充完整成熟工作流。

**前一轮已冻结资源（不重复计入）：** 新GPU1,573,264（训练1,572,864＋检查400含漏冷重置的200失败步），256PPO/5120训练Adam；CPU8784（支撑1000＋独立7784），7784ONNX。单元红/绿回归另5次Adam仅生成小测试状态；真实检查点64条Torch前向/绘图/400录像FK均0积分。脚本缺冷reset、归档cfg缺class和fixture绕过构造器等失败均保留，不伪装物理失败，临时日志已按规范归备份。历史2053CPU/451ONNX、3.52亿训练及其他历史不重复。该前一轮所有进程已结束；新恢复对照亦已结束，其他GPU进程未动。

**本周期合计新资源：** GPU2359296/384PPO/7680训练Adam；CPU9800/robot ONNX9896，测试另13合成ONNX/0Adam；private_eval FK9803、render1440、绑定构造forward2均不是额外积分。全部本周期GPU/CPU评估/绘图/测试进程终止，没有跨聊天联系或修改他人进程。不存在等待硬件或资源的阻断。旧3.52亿、前一轮资源和复用旧链路不重复计入。

[本周期记忆对照报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_recovery_memory_pilot_001/diagnosis.md)（385 payload，manifest `0e4261c46b35be0cfb99730cbb26025916eb1b02e08381941123814f905e1ef2`） · [实际四方向对照录像](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_recovery_memory_pilot_001/actual_memory_comparison.mp4) · [实际力矩核对](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_drive_binding_audit_001/diagnosis.md)（6 payload，manifest `19c1e8d75a20dd3debe0201a46bf30c672969cf8f8097f76184c063c3627bb5f`）。续训恢复对照226 payload manifest `51c51729b8076ae2413a4a93e8a6eb021ea816540b567041e815d7974097e2df`。

[续训恢复新对照报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_adaptive_resume_pilot_001/diagnosis.md) · [恢复工作流002](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_recovery_workflow_audit_002/review.md)（0积分/训练/FK） · [记忆对照预声明协议](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_recovery_memory_pilot_001/protocol.json)（预声明SHA ad3340641b96f1751b1bdb697743e5372e20582f2a7174364e6aa5ad984e7ee5）。两组独立均失败、分支已收口；下一批以本顶部课程问题为准。

[支撑相位报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_walking_support_phase_001/diagnosis.md)（manifest `7371e389d408ccd82c1f4a95942302a645069471d6658b61611c66d4b688a47a`） · [GPU对照与恢复修复](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_guided_contact_domain_pilot_001/diagnosis.md)（manifest `a1dd581de3310a713960eaa7a1a9e1b14845ecfddb05d8b20d93bb77677c89b1`） · [新实际失败对照录像](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_guided_contact_domain_pilot_001/actual_contact_domain_comparison.mp4)。13/481份冻结payload哈希核验，历史下一批不覆盖本顶部。

[完整移动/机构定责报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_locomotion_method_hardware_audit_001/diagnosis.md) · [历史训练账本](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_locomotion_method_hardware_audit_001/historical_training_ledger.json) · [腿部独立审查](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_leg_clearance_audit_001/README.md) · [真实旧低速移动录像](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_gpu_guided_velocity_001/learned_walk20s_update512_labeled.mp4) · [接管实际收据](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_recovery_handoff_diagnosis_001/receipt.json) · [原300恢复四方向失败录像](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_tracking_termination_admission_001/four_direction_failure.mp4)。历史下一批不覆盖本顶部。

## 1. 目标与当前基线

本任务交付三条主线，优先完成移动和跌倒恢复，再完成拾物运输闭环。

| 主线 | 最终行为 |
| --- | --- |
| 移动 | 站稳；前进走路0.4m/s、跑步0.7m/s与Shift切换；后退、横移、转向、启停；通过室内接缝、低障碍、缓坡和小台阶 |
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

1. 当前电脑训练合同为独立具名 `goose_task_proxy_11_rigid_braking_v1`，从已核验004派生，21体／18主动轴／11叶（9原凸网格＋2内接原生足box）；总质量、COM、完整惯量、轴位和自碰撞过滤保持。足box的遗漏上部形状与有界制动修正均在顶部说明，不继承原完整形状或控制资格。
2. 足底使用普通原生刚体接触，不使用软底弹簧、压缩曲线或四点替换。软底静载与行程资格不再阻挡电脑能力训练；本期不要求仿真实物软底。
3. 原生implicitfast和位置驱动采用独立合同：17轴原生隐式位置驱动，嘴部仍驱动真实输入转子；原动作目标、限速、力矩限额、名义颈部前馈、延迟、热代理与350W功率预算显式保留。它不继承旧显式PD的实际力矩身份。
4. 原33体微脚垫、凝聚软底、自定义BE／discrete和材料诊断路线只保留历史失败，不作为当前训练派工；源与目标资格分别验收。

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
| M2 平地 | 站立→低速前后→横移→转向→启停反向→走路0.4/跑步0.7与切换→混合指令 | 两档持续速度、切换和独立移动验收；首个可操作 Bevy 候选 |
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
| 电脑刚性足底 | 原生接触、真实支撑面和滑移／穿透结果；取消软底压缩、曲线及行程验收前置条件 |
| 站立 | 60 秒无跌倒，漂移≤5 cm |
| 移动 | 前进走路0.4、跑步0.7、后退0.15、横移0.1 m/s、转向0.6 rad/s与左右±π掉头；200独立案例整体≥95%，两档及切换分别报告；稳定前进段≥20秒，实际COM前进均速分别≥0.4/0.7m/s，同时报告速度分布、位移、路径/航向和步态，不用指令/峰值代替 |
| 输入停止 | 短长按、Shift先/W先、单独Shift、走↔跑重复切换、两档带转向、转向中松Shift、松W、反向、失焦、暂停恢复；停止1秒内≤0.04 m/s，随后3秒漂移≤5 cm；切换保全实际姿态/驱动/动作历史，不靠重置过关 |
| 地形 | 5/10/20 mm、±5°坡、20 mm台阶，每类≥50例、≥90%；不授予连续楼梯资格 |
| 恢复 | 四方向各50例，10秒起身、稳定3秒、再移动1 m；整体≥95%、每类≥90%；真实扰动另测 |
| 拾物 | 三重量、圆柱夹持段/带把手物体；底部离地≥80 mm、保持5秒、携带2 m并转向、放置误差≤100 mm；每重量≥60例、≥90% |
| 集成 | 同世界移动→跌倒→恢复→移动→拾物→运输→放置；20固定程序全过，30分钟实际运行 |

恢复还要求直立度≥0.95、COM高度≥名义站立85%，不能瞬时竖起就交接。开发、训练、验收集独立；正式评估独立进程、确定性 ONNX、auto_reset=false。先核同状态观测/动作目标/实际力矩，再比较接触任务，不要求跨引擎长轨迹逐位一致。

性能要求持续50 Hz、无积累欠账，物理＋推理P95≤16 ms；渲染独立并插值，目标机实测画质、帧率与交互延迟。

## 5. 执行、排期与交付

每轮：**单一可检验问题→冻结版本与指标→短对照→≤两小时训练→独立评估→晋升或保留失败。**

- 首轮5–15分钟 pilot 测吞吐、内存和接触容量，再定 batch/更新；记录真实物理样本、优化器更新和资源消耗。
- 当前用户已明确G1暂停且禁止主动联系其他聊天：GPU前仅检查实际占用，不发G1/硬件协调消息、不停止他人进程。最新人类要求本聊天移动/恢复并行，能力子agent只做必要的内部排期/交付，GPU串行；不得把它误解为禁止当前已授权的能力并行。
- 两次受控实验无改善就回查物理、初态、观测、终止和奖励分项，避免局部死循环。
- 保存模型/合同/代码/依赖/配置/seed/策略哈希；候选显式指定，不用 latest。M0-S 和源 GPU 链路未过禁止源端长训，M0-T 未过禁止目标训练；数值/约束/容量异常停止该批。
- 每关交付通过/失败/适用范围、录像、算力和下一实验，基于真实吞吐滚动估时。
- 当前实施聊天的 `Goose ProjectManager` heartbeat 每两小时检查产出与主线进度，不新建聊天。原用户授权的 `bevy_performance` 子 agent 当前闲置，不按历史周期自动启动。最新人类要求移动/恢复持续并行：root负责恢复、既有bevy_feasibility负责移动/转向，其他闲置agent不自动唤醒。性能工作不得改碰撞形状／过滤、求解配置、物理参数或观测／动作合同来凑速度。

已实际启动源端有界PPO，当前结果以页面顶部最新检查点为准。最早M0产物归档 `/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/`；本页为唯一计划和状态入口。最终交付冻结策略、本体/合同、Bevy包、复现命令、三主线报告及未通过清单。制造/电气/实物载荷仍由机器人工程侧维护；仿真资格不自动等于实物资格。

完成当前可行性诊断后，长期路线仍为**统一MJCF／冻结碰撞代理 → 原生源端准入 → 成熟小批量PPO与导出 → 站立／移动／恢复源策略 → 目标迁移**。当前不唤醒Bevy性能agent；移动/恢复准备与评估并行，GPU有界串行。每条能力线每周期只选能解锁下一阶段的一个问题；局部实验达到诊断目的即收口，连续两次无改善就回到模型或成熟流程，不追加求解器分支。历史检查点中的“下一批”只记录当时决定，以本页顶部现行任务为准。

现行首要工作以页面顶部为准；不重新启动已收口的300Actor/投影参考配置或扩大局部搜索。源/目标/正式三主线资格仍分别报告，完整目标未达。

参考状态初始化课程依据：[DeepMimic §Training / Reference State Initialization](https://arxiv.org/html/1804.02717v3) 与[作者reset实现](https://github.com/xbpeng/DeepMimic/blob/master/DeepMimicCore/scenes/SceneImitate.cpp)。仅采纳reset阶段的参考状态课程思想，不采纳运行时写真实姿态、不同物理步长或作者控制器合同。

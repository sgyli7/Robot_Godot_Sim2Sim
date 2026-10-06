# Goose V0.1｜Bevy Sim2Sim 长期训练计划

版本：2026-10-06。实施分支：`codex/goose50_training`。

**现行人类指令：继续推进，同时先把“训练方法还是机械设计阻断移动/起身”定清楚。** 不主动联系、读取或监控硬件／暂停G1，不新聊天/PM。用户最新允许移动、恢复、转向并行；“转板”已明确为左右转向/原地掉头，不新增板上运动。复用本聊天既有 `goose_asset_audit` 与 `recovery_planning_research` 做有界独立审查；最新恢复工作流002已完成并闲置；主实施继续负责移动/转向与集成，GPU有界串行，不以子agent为等待理由。十小时目标已超期，完整移动与恢复仍未交付，目标不缩小。临时产物严格ProjectBackups，生产仅对应crate，本文为唯一入口。

**冻结输入：** `goose_task_proxy_11_rigid_braking_v1`，MJCF SHA `3dd030f0475efd101ab129288e7f04f5694067757d85e0429933cecb9c5268e4`、合同SHA `d3fd7ba668678e1e1237d00bd7fac77e5d0625d77cc65bea85d3579ffb17061c`。21机器人刚体／18真实电机／2被动轴／11叶／10.430690821kg。每腿6主动轴；公共65→18、拾物82→18。9凸mesh＋2刚性足box；原鞋上部遗漏44.76%不继承实物资格。质量/完整惯量/轴位/嘴四杆/过滤/限速/力矩/热/350W保留；50Hz、每Tick一次20ms实际驱动/积分/当前Actor更新、decimation1，无隐藏子步/出生后实际root/q/qd/drive/history写入。电脑训练不考虑软底。

**最新定责：已经大量训练，也有低速双足运动证据，尚无完整合格能力。现有证据不支持“DOF不足/大壳把正常走路卡死”。Lab任务/观测/参考适配已有明确错误，责任由Lab承担；不能把3.52亿不同实验样本当一条成熟配置持续训练，也不能因此判PPO或硬件绝对不可能。** 本轮尾段/原生Critic预热两组真实GPU训练和源CPU/GPU独立回放均失败，薄恢复适配已收口，原稳定候选保留。完整移动、恢复尚未通过；下述历史续训与LR代码修复仍保留，不重复计入本轮。

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

**现行唯一下一问题：** 返回已存在的上游**全指令移动／左右转向／停止**工作流：`make_mature_velocity_cfg(forward_only=False)`、RSL native PPO和原512的65/69模型及完整优化器状态，显式恢复正确adaptive LR；复用现有受保护的`initialize_unused_lateral_yaw`初始化原零方差/未学过的command列。先证明零command/原forward输出逐项不变，cold/真实GPU两更新与ONNX一致，再一份有界全指令pilot；不再把恢复Critic92 fresh transfer拿来冒充移动训练，也不新增奖励函数/借改出生授资格。原移动接触域仍未合格，所以不长训/不接Bevy资格。恢复路线暂时收口这组薄HumanUP tail适配，后续按完整成熟工作流与合法运动先验接回；当前不等待硬件。完整移动、左右转向/原地掉头、四方向＋自然恢复＋继续1m仍未完成，目标不缩小。

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
| 电脑刚性足底 | 原生接触、真实支撑面和滑移／穿透结果；取消软底压缩、曲线及行程验收前置条件 |
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
- 当前用户已明确G1暂停且禁止跨agent通信：GPU前仅检查实际占用，不发协调消息、不停止他人进程。后续人类新指令才能恢复跨聊天协调。
- 两次受控实验无改善就回查物理、初态、观测、终止和奖励分项，避免局部死循环。
- 保存模型/合同/代码/依赖/配置/seed/策略哈希；候选显式指定，不用 latest。M0-S 和源 GPU 链路未过禁止源端长训，M0-T 未过禁止目标训练；数值/约束/容量异常停止该批。
- 每关交付通过/失败/适用范围、录像、算力和下一实验，基于真实吞吐滚动估时。
- 当前实施聊天的 `Goose ProjectManager` heartbeat 每两小时检查产出与主线进度，不新建聊天。原用户授权的 `bevy_performance` 子 agent 当前闲置，不按历史周期自动启动。最新人类允许能力并行，当前两只有界审查子agent已收口，其余不自动唤醒。性能工作不得改碰撞形状／过滤、求解配置、物理参数或观测／动作合同来凑速度。

已实际启动源端有界PPO，当前结果以页面顶部最新检查点为准。最早M0产物归档 `/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/`；本页为唯一计划和状态入口。最终交付冻结策略、本体/合同、Bevy包、复现命令、三主线报告及未通过清单。制造/电气/实物载荷仍由机器人工程侧维护；仿真资格不自动等于实物资格。

完成当前可行性诊断后，长期路线仍为**统一MJCF／冻结碰撞代理 → 原生源端准入 → 成熟小批量PPO与导出 → 站立／移动／恢复源策略 → 目标迁移**。当前不唤醒Bevy性能agent或并发训练。每周期只选能解锁下一阶段的一个问题；局部实验达到诊断目的即收口，连续两次无改善就回到模型或成熟流程，不追加求解器分支。历史检查点中的“下一批”只记录当时决定，以本页顶部现行任务为准。

现行首要工作以页面顶部为准；不重新启动已收口的300Actor/投影参考配置或扩大局部搜索。源/目标/正式三主线资格仍分别报告，完整目标未达。

参考状态初始化课程依据：[DeepMimic §Training / Reference State Initialization](https://arxiv.org/html/1804.02717v3) 与[作者reset实现](https://github.com/xbpeng/DeepMimic/blob/master/DeepMimicCore/scenes/SceneImitate.cpp)。仅采纳reset阶段的参考状态课程思想，不采纳运行时写真实姿态、不同物理步长或作者控制器合同。

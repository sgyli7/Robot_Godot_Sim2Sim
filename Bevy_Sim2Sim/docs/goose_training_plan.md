# Goose V0.1｜Move 训练计划

版本：2026-10-09。独立分支：`codex/goose_move`。

**029停止续训及中段复核完成：不晋升，GPU已释放。** [报告](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/arc_stop_continuation_029/report.md)。同028课程/奖励/驱动，仅增加256PPO；最终真冷3/5改善，但弧线23/24→0/24，中段128也仅1/24，因此不替换028或023。实际13,280,912 GPU积分/256PPO/5120Adam；执行及复核均complete，禁止重启或自动同路线续训。真实轨迹的原角跟踪奖励对退步弧线也下降，未证明奖励漏洞或唯一PPO根因。主线保留023速度与028转向，下一有界课程推进完整行进转向±0.6并保留原弧线/直行/停止/冷启动回归，之后原地转、行进侧移、纯侧移及后退；正式自由移动和Bevy50仍未完成。历史“下一批”均不是现行派工。

**当前状态：取得部分源端能力，完整任务未完成。** 本聊天只负责Move，不推进起身或拾物，不联系其他聊天。物理频率独立研究已结束；新授权媒体子Agent只制作主页历史实录GIF，不干扰训练。目标保持：走路0.4m/s、Shift跑步0.7m/s，前后／侧移／双向原地转向／移动中转向／启停及任意切换，操作与MicroDuck三轴组合对齐。不能以请求速度、单段录像或开发小集代替这些资格。最新229的可选左右弧线增量保留176全部12个通过项，原16条开发程序14条通过，扩展集17/25通过；不覆盖143/176或授予完整资格。

**最新用户授权：另建200Hz源端快跑候选，尝试达到0.7m/s。** 策略及数字驱动目标/历史仍50Hz，四次真实5ms积分；原50Hz默认、Actor基线及Bevy目标保持。该追加速度课程独立于四频相同课程的因果实验，不把新增课程成绩归入频率单变量比较。根主线自行实施，串行使用GPU。

**上一批012保留：009—011诊断、012受控PPO及015原失败程序回归已完成。** [012报告](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/symmetry_ppo_comparison_012/report.md)：两臂各512世界×128PPO；对称组快档实速.5896、60秒.5906m/s、横漂-.0190，同预算控制组.5771、横漂-.0813。200Hz物理各48/48；完整行为控制0/48、对称9/48，停止控制48/48、对称16/48。对称组慢档.5437超出0.4验收范围，61秒站立漂移6.56cm失败；50Hz MuJoCo迁移各0/48。保存新Actor，不覆盖005/007或原143/176/229；不得把开发容差内的0.59称为已达到0.7或Bevy资格。

**012已经结束，禁止重启历史execute_queued012.py或继续等待其旧PID。** [实际执行收据](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/symmetry_ppo_comparison_012/execution_receipt.json)记录12阶段全部完成，排队900.24秒后串行训练和评估，无需继续轮询。012保持200Hz物理/50Hz策略及数字驱动、1.5Hz相位、本体/奖励/课程不变；唯一干预是RSL原生mirror-loss及既有native坐标适配，系数.1为本次实验值。两臂Actor/Critic/Adam及保存初态逐位对齐；不声称后续GPU轨迹逐位确定性。每Tick部署单个自身Actor，不使用MD权重或镜像推理平均。

**上一批已收口：017奖励滤波对照、018CPU回归及019录像均完成。** [017报告](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/reward_filter_comparison_017/report.md)记录两臂各512世界×128PPO，同父012完整学习状态/本体/200Hz物理/50Hz策略及数字驱动/1.5Hz相位/奖励函数及权重/课程/种子，只有共享运动奖励滤波.8→.2秒变化。控制组快档.6050、60秒.6055m/s，但冷快启动6.90秒自碰失败；对照快档.5748、慢档.5035，冷物理4/4但慢档仍超速。两臂开发物理48/48、完整24/48、停止48/48；50Hz MuJoCo迁移0/48，61秒站立均因稳定期速度超.04失败。不能将控制组的停止改善归因于缩短滤波，或将热快段增速晋升为完整Move/Bevy资格。execution_receipt已经12阶段complete，禁止重启execute017.py、重复训练或继续等待旧PID。

**020精度对照已完成，执行器关闭；021CPU回归和022录像已完成。** [020报告](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/speed_precision_comparison_020/report.md)：同父017tau02、两臂各512世界×128PPO，唯一线速度Gaussian宽度.5→既有Goose配置.15。精度臂快档.64323、60秒.64287m/s，对照.57444/.57520；横漂.01384→.00373。两臂200Hz物理48/48、完整24/48、停止48/48，真冷物理4/4。精度臂慢档.49979仍超速，61秒站速度.04008797>.04未过；控制长站通过。50Hz MuJoCo完整均0/48，不是实际Rapier成绩。CPU实际ONNX快档.64332/.57642，慢档均失败；33秒真实冷慢→快→慢→停[连续录像](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/videos/truecold_precision020_switch_022/continuous.mp4)保留失败。不覆盖旧基线或晋升为完整Move/Bevy。首个小链路实际成功后的收据字段错误已记录并显式恢复，无重复训练；最终12阶段complete，禁止重启020原/恢复执行器或等待旧PID。

**023收敛批、024CPU回归及025/026实际录像完成，GPU释放，无待执行训练。** [023报告](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/precision_convergence_023/report.md)记录同父020std015、冻结trainer/cfg原样、512世界×512PPO、200Hz物理/50Hz策略及数字驱动的有界续训。慢档.41362、快档.72117、60秒快档.72410m/s，原速度目标取得源端开发证据；开发物理48/48、完整46/48、停止48/48，快慢热切换8/8。[33秒真实冷慢→快→慢→停录像](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/videos/truecold_convergence023_switch_025/continuous.mp4)全程通过，无重置/Teacher/换Actor。两条warm27偏航未过。GPU真冷整体物理3/4、完整2/4；另一个直接快→停在13.10秒自碰.39751mm，保留[失败连续录像](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/videos/truecold_convergence023_stop_failure_026/continuous.mp4)；61秒站立漂移5.474cm未过。CPU ONNX慢.41119/快.72442、两原案例及4次停止通过，不能代替GPU红案例。50Hz MuJoCo迁移0/48，未授予游戏资格或替换旧候选。execute023五阶段complete，禁止重启或等待旧PID；不再直线增速续训或宽度/滤波/相位网格。

**当前派工：行进左右转 > 原地左右转 > 行进侧移 > 纯侧移。** 用户确认快慢速已取得开发进展，从023完整Actor/Critic/Adam继续独立200Hz源端课程；先冻结023为GitHub速度基线，并以原0.4／0.7直行、快慢切换与停止为回归，能力增量未通过不得覆盖023。只改变命令课程，保留模型、驱动、相位、成熟奖励函数／权重及PPO配方；先评估原策略新命令，再小链路和有界pilot。主页并列展示MicroDuck、Goose、G1真实GIF，完整历史成果继续保留。

**028行进转弯pilot完成，旧023不覆盖。** [028报告](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/moving_arc_curriculum_028/report.md)：只把课程扩为0.4下±0.3弧线，模型／驱动／相位／成熟奖励／PPO配方保持；512世界×128更新。弧线从父023的0/24升为23/24，原快慢直行与切换24/24保住，开发物理48/48、完整47/48、停止48/48。真冷5程序物理4/5、完整0/5：行进双向转阶段通过，但停止1秒.05318>.04；冷慢停／快慢停也失败，快停自碰与长站问题保留。028保存增量，023继续作为GitHub冻结速度基线；下一批有界处理冷停止，不改门槛或自动晋升。execute028五阶段complete，禁止重复启动。最终±0.6与后续原地转／侧移仍待训练。

**GitHub阶段发布完成。** 标签 `goose_move023_source200_20261009` 固定源码／模型哈希／完整33秒录像；main主页直接贴入MicroDuck六张与Goose两张实录GIF，G1全部原片及说明保持。main仅文档与展示媒体提交，不合并Move分支覆盖其他课题代码；权重与原始收据按备份规范保持。

**027实际Rapier50接收已收口：M0-T未通过。** [接收报告](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/rapier50_intake_027/report.md)：初始观测一致、Actor差4.85e-8、数字驱动差4.45e-14；首Tick嘴铰链越限1.47026mrad。两项同动作预测嘴限位诊断仍失败，8PGS第三Tick输入转子越限8.66474mrad，不继续局部网格。旧20Tick回归140组字段逐位一致；53Rust＋6Python通过，10外部fixture测试ignored。本轮67CPU积分／62ONNX，GPU/PPO0。源023保持，不因M0-T失败暂停源端转弯能力；目标PPO与游戏资格仍待准入。

**016定位速度分档与停止反馈。** 原17秒冷慢档两次全保存字段逐位一致，实速.54645m/s超0.4门槛，物理/停止通过。实际Actor命令槽6:9与目标逐位一致；仅变命令.4→.7时输出公共动作RMS变化.01595，不支持“命令完全丢失”。[实际离线探针](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/speed_response_diagnostic_016/offline_findings.json)用保存q/qdot的私有FK/COM速度调用原奖励：机器人停止1秒已约.02—.03m/s，但.8秒滤波下停止奖励半高约2.60—2.64秒，.2秒约.92—.98秒；此为奖励测量延迟证据，不是PPO改善证明。宽度.5在慢档超速下仍给4.65/5的跟踪密度，保留为下一假设。本轮不改其宽度、相位或驱动。

009—011已收口：[诊断报告](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/move_asymmetry_diagnostic_009/report.md)支持策略偏置参与横漂，调试平均损害停止，未采用。015原17秒CPU冷快档程序仅更换为012自身ONNX：3,400积分/850推理，原物理/移动/停止门槛通过，实速.5939、横漂-.0169、停止1秒.0310m/s；单例不授予完整能力。[33秒新候选真冷慢→快→慢→停连续录像](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/videos/truecold_mirror012_switch_014/continuous.mp4)保留慢档超速失败。

**上一批006/007保留：005不被007覆盖。** 006 CPU ONNX/GPU真冷四程序各物理4/4、完整0/4，61秒漂移约6cm；007从005同配方续训256PPO，.7实速.4971→.5453、60秒.5453，但横漂约-.0598、停止48/48→37/48、完整8/48→2/48，未晋升。007真冷完整1/4，61秒漂移7.21cm失败；50Hz迁移0/48，没有Bevy资格。原143/176/229及005保持；不存在相同配方50Hz续训对照，不把增速单独归因于频率。禁止自动续同配方或做相位网格。

代码工作树为`/home/ethan/Projects/TempWorktree/Sai_Lab/goose_move/Bevy_Sim2Sim`。临时模型、脚本、录像、收据及构建均落[Move备份目录](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001)。原工作树和恢复分支不改写。

## 当前具名入口与实际能力

[收尾清单](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/source_move_closeout_177/manifest.json)固定模型、11份外部模型资源、合同、Actor、控制器、冻结源码和收据SHA；[收尾报告](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/source_move_closeout_177/report.md)记录范围、失败和复现方式。禁止加载latest或覆盖旧基线。

| 候选 | 实测及退出依据 |
| --- | --- |
| 默认保留143 | 约0.14／0.20m/s与±0.3rad/s，13条开发程序11条通过；左弧线及完整串接未通过。停止选择只依赖真实用户命令历史 |
| 较快开发176 | 约0.14／0.27m/s；37秒慢→快→左→右→停无重置通过，左右转约+0.334／−0.343rad/s。16条开发程序12条通过；不是默认替换或完整MD资格 |
| 可选左右弧线229 | 保留176原12个通过项，新增冷左弧线及45秒慢→快→左→右→右弧线→停；原16条14条通过，扩展集17/25。29秒慢走→左右弧线→停通过，任意热切换与完整串接仍失败 |
| 新200Hz源候选005 | 0.4指令实速.4504m/s，8个支持出生开发移动+停止通过；0.7实速.4971、60秒.4968，速度/横漂仍未通过；50Hz迁移0/48，无Bevy资格 |
| 仅存200Hz速度增量007 | .4实速.4621、.7及60秒约.5453m/s；物理48/48、完整行为2/48、停止37/48，未晋升。真冷慢走单例通过，长站失败；50Hz迁移0/48 |
| 新200Hz对称辅助012 | 快档.5896、60秒.5906m/s，横漂-.0190；物理48/48、完整9/48、停止16/48。慢档.5437超速、长站失败；50Hz MuJoCo迁移0/48，不替换旧基线 |
| 017同配方控制tau08 | 快档.6050、60秒.6055m/s；开发完整24/48、停止48/48，但冷快启动6.90秒自碰失败、初始停止失败，真冷物理3/4。慢档.5421超速；不晋升 |
| 023源端快慢开发 | 慢档.41362、快档.72117、60秒.72410m/s；200Hz物理48/48、完整46/48、停止48/48，快慢热切换8/8。GPU直接快停自碰、61秒站漂移及50Hz迁移失败；未获完整全向/Bevy资格 |
| 020精度开发std015 | 快档.64323、60秒.64287m/s，横漂.00373；物理48/48、完整24/48、停止48/48。慢档.49979超速、长站速度门槛与50Hz完整迁移未过，不晋升 |
| 020同预算std05控制 | 快档.57444、慢档.49409；真冷61秒站立通过，物理48/48、完整24/48，保留独立候选 |
| 017滤波对照tau02 | 快档.5748、60秒.5736m/s，慢档.5035仍超速；开发完整24/48、停止48/48，真冷物理4/4、完整1/4。牺牲快档速度，长站与50Hz迁移失败；仅作为较稳开发候选 |
| 原完整目标 | 新200Hz的0.4开发小集已通过；0.7、后退／侧移、完整任意弧线/切换、独立正式200例及Bevy资格仍未完成 |

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

**专项调查已完成，子Agent闲置，不再排队训练。** [完整独立报告](/home/ethan/ProjectBackups/2026-10-08/Sai_Lab/goose_integration_frequency_audit_001/report.md)包含12个新初始化臂（四频×三种子，每臂512 PPO）及42筛查：局部同状态/动作踝越限50Hz .5439rad→200Hz 0，匹配预算物理终止78,529→5,593；这证明局部数值/探索收益，未识别为Move唯一或主导瓶颈。四频自身完整移动均0/672，课程内动作也未通过；.4/.7在该MD9课程外，不能据此判断其不可学。部分开发真冷站收益在MuJoCo50保留，完整Move/真实Bevy收益未证明。根005/007自己的热学习状态与高速课程单独计账，不纳入此公平比较。主线保留50Hz及旧基线，200Hz作为具名源候选；不默认采用500Hz或再唤醒该研究Agent。

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

## 200Hz具名步态适应与源端进展

[CPU004报告](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/cadence_probe_004/report.md)冻结自己的003 ONNX，只有相位1.2→1.5Hz增量、物理仍200Hz/策略50Hz，两个支持出生×.4/.7×两相位共19,200真实CPU积分。物理8/8完成；直接提相位使.7实速.3604→.2966m/s、yaw+.1257→-.3228rad/s，故不作为直接提速补丁。CPU参考和GPU近似复现已报告，未完整保存GPU warmstart/接触历史，不声称逐位完整跨引擎等价。

[005协议](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/cadence_train_005/protocol.json)在该具名相位下实际适应训练；仅新增相位，其余模型/驱动/求解/奖励/课程保持父003版本。真实8世界/1更新小链路后完成512世界×256 PPO/12,582,912 GPU/5,120Adam/206.30秒。[学习状态审计](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/cadence_train_005/learning_state_audit.json)核对Actor/Critic/Adam初始与003父逐位一致，世界和课程时钟重置；无同父同额外预算的1.2Hz续训对照，不作单参数因果结论。XY RMSE首/末32更新.2910→.2255。

[005报告](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/cadence_train_005/report.md)与[实测统计](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/cadence_train_005/metrics.json)记录源端物理48/48、行为8/48，0.4前进+停止8/8通过，0.7仍不足并横漂；全部48条停止通过。0.55实速.4851但横漂-.0437也失败，0.27实速.3603超档；该候选不继承低速能力。50Hz迁移48例全部首次穿地失败，.7在.66s、.4在2.34s结束；不调门槛或自动修复。评估为确定性Torch+已验证ONNX导出对照，尚不是全程独立ONNX runtime、真冷启动或正式200例资格。

[30秒连续实测录像](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/videos/cadence15_speed_switch_005/continuous.mp4)为1,500Tick/1,501帧、CPU llvmpipe零积分/推理，固定银行出生0、无拼接/重置/回滚。[保存轨迹诊断](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/cadence_train_005/saved_drive_diagnostic.json)核对真实相位及髋/膝主频1.5Hz，.7段髋俯仰近目标限速约11.6%/13.5%、膝约17.5%/10.8%，右髋偏航末子步近当时预算约7.1%；这些局部统计不证明硬件速度上限或全子步饱和情况。

[本批账本](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/cadence_train_005/resource_ledger.json)共13,203,696真实GPU积分/257PPO/5,140Adam；[四批根高速合计](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/cadence_train_005/combined_local_resource_ledger.json)53,002,816GPU/1,028PPO/20,560Adam，独立频率研究与旧主线不混计。005具名源码、模型/合同、检查点、ONNX、录像及失败收据按清单冻结，原143/176/229与三个旧高速批保持。

[求解统计解释更正](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/cadence_probe_004/root_solver_stage_erratum.json)说明旧v3 solver_niter是在控制末尾forward求解后读取，不能称为积分子步迭代峰值；数值、轨迹、训练和旧冻结文件保持。本批收据已明确采样阶段。当前顶部派工优先于此处历史下一步。

## 200Hz真冷检查与同配方续训

[006报告](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/cold_entry_006/report.md)记录全零关节/速度/时间/驱动目标/动作/热/力矩/相位的独立CPU ONNX与GPU检查，从首Tick用自身005 Actor，无支持银行或教师。CPU25,600真实积分/6,400 ONNX调用，GPU48,800积分；四程序各物理4/4但完整行为0/4。冷慢走段约.449m/s通过，冷初始停止和61秒静止漂移约6cm失败，.7约.497m/s仍失败。[33秒实际冷启动连续录像](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/videos/truecold_speed_switch_006/continuous.mp4)保留起步/快慢切换/停止和原失败门槛。

[007报告](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/continuation_007/report.md)固定父005完整Actor/Critic/高斯/归一化/Adam初始逐位一致，相同奖励/课程/1.5Hz相位及全部物理/驱动参数。实际8世界小链路768 GPU/20 Adam通过，512世界×256更新12,582,912 GPU/5,120 Adam/208.75秒；ONNX最大差2.384e-7。首/末32更新XY RMSE .21279→.20632，yaw .13827→.14591，不能以训练奖励代替行为。

[独立指标](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/continuation_007/metrics.json)记录007源端物理48/48、完整2/48；.4实速.46207，.7及60秒约.54525，横漂-.05977m/s；停止37/48，父005原48/48。007GPU真冷单个.4加停止通过，实速.46486；冷.7仅.54103且停止失败，61秒静止漂移7.214cm失败。银行与训练共享、四条冷程序不是成功率样本；正式200例、完整M0-S/Move和Bevy均未授予资格。

50Hz迁移0/48，慢走.64秒、快跑.70秒首次穿地，峰值约6.72–9.08mm。005作为较稳源候选保留，007只存增速检查点，不覆盖低速/转向或降低门槛。[30秒007实际连续录像](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/videos/cadence15_same_recipe_switch_007_v3/continuous.mp4)固定同一Actor/出生，不拼接、回滚或修复。[本轮资源账本](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/continuation_007/resource_ledger.json)006/007合计13,298,032 GPU/257 PPO/5,140 Adam；五根快速批与冷检查累计66,300,848 GPU/1,285 PPO/25,700 Adam，CPU44,800，独立频率研究分账。

本批速度有增量但停止退步，收口自动同配方续训。下一受控问题先定位横漂/停止损失与实际动作/驱动状态的关系；不把.545当硬件极限或频率因果证明。独立四频研究12训练臂与42两层评估、正式三问结论及冻结已单独交付。本轮GPU工作自然结束，未干扰后来观察到的其他比较进程。

## 200Hz受控对称训练012与原失败回归015

[冻结协议](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/symmetry_ppo_comparison_012/protocol.json)固定两臂同父007、种子4123、课程0/.4/.55/.7、512世界/128PPO，只有已安装RSL原生mirror-loss变化，不修改奖励或伪造物理转移。每臂6,291,456实际GPU积分、1,572,864控制决策、2,560 Adam；分别107.59/121.65秒，另有小链路和独立评价。[学习状态审计](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/symmetry_ppo_comparison_012/pilot_learning_state_audit.json)核对初始Actor/Critic/Adam和保存世界字段逐位一致；新ONNX最大误差控制1.192e-7、对称8.941e-8。

[实际统计](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/symmetry_ppo_comparison_012/metrics.json)记录对称组0.7段均速.5896、横漂-.0190、偏航+.0008；控制组.5771、-.0813、+.1370。对称组0.7移动段8/8符合原比例容差，整条5/8；60秒整条4/8。速度均值尚未达到字面0.7，0.4超速与停止回归不允许晋升。对称组训练随机探索的关节/自碰终止原因225/91，控制67/38，原因可重叠；不声称全探索安全改善。单种子、相关训练出生仍不是正式独立成功率。

[015CPU回归](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/mirror_candidate_regression_015/own012_original_redcase_001/receipt.json)在原冷快档程序中只更换Actor，实际3,400积分/850CPU ONNX调用，物理及行为通过原容差，实速.5939。CPU与GPU均值分别保留。014连续录像来自实际GPU真冷33秒程序，1,650Tick/1,651帧，软件渲染零新增积分/推理，无拼接/回撤/修补。

[012账本](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/symmetry_ppo_comparison_012/resource_ledger.json)含小链路/两组pilot/所有GPU测试共13,918,144 GPU、258 PPO/5,160 Adam，阶段计时626.97秒。009—011及015新增20,400 CPU积分；[根高速合计](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/symmetry_ppo_comparison_012/combined_local_resource_ledger.json)80,218,992 GPU、1,543 PPO/30,860 Adam、65,200 CPU，与旧Move和独立频率研究分账。两臂50Hz MuJoCo迁移0/48，实际Bevy未测；具名输入、候选、失败、依赖及视频冻结，不自动加载latest。

## 017奖励反馈受控对照与018/019回归

[017协议](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/reward_filter_comparison_017/protocol.json)、[配置预检](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/reward_filter_comparison_017/config_preflight_receipt.json)及[学习状态审计](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/reward_filter_comparison_017/pilot_learning_state_audit.json)记录只有同一共享奖励滤波参数三处.8→.2秒变化，两个128PPO均实际完成，初始Actor/Critic/Adam及世界字段逐位对齐。每组6,291,456 GPU积分、1,572,864控制决策、2,560 Adam，115.20/112.01秒；ONNX误差均8.941e-8。奖励/gamma/终止保持20ms，现有MD适配默认.2秒不被声称等于所有历史MD检查点配方。

[018原程序CPU ONNX回归](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/filter_candidate_regression_018/own017_tau02_001/receipt.json)实际6,800积分/1,700推理，两条物理及四段停止均通过；慢档.50257仍红、CLI3，快档.57328仅符合原开发容差，不授予字面0.7。控制组[33秒真冷慢→快→慢→停录像](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/videos/truecold_tau08_switch_019/continuous.mp4)保留初始停止与慢档失败，快段约.6050；1,650真实Tick/1,651帧，软件渲染零积分/推理，无拼接/回撤/修补/换Actor，抽帧已核查。

[017资源](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/reward_filter_comparison_017/resource_ledger.json)含小链路、两组pilot和原门槛两层评估，共13,921,552 GPU、258 PPO/5,160 Adam，阶段计时619.52秒；016+018另13,600 CPU积分。[根高速合计](/home/ethan/ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/reward_filter_comparison_017/combined_local_resource_ledger.json)94,140,544 GPU、1,801 PPO/36,020 Adam、78,800 CPU，与旧Move和独立频率研究分账。新速度增量不继承旧低速、转向或Bevy资格，所有失败及旧基线保持。

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

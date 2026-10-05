# Goose V0.1｜Bevy Sim2Sim 长期训练计划

版本：2026-10-06。实施分支：`codex/goose50_training`。

**现行人类指令（2026-10-06）：先把起身失败的责任与阻断点查清，再继续能力训练。** 已停止当前PPO，GPU释放；不启动新的奖励、噪声或训练网格。主实施独立负责诊断，不联系、读取或监控原硬件/G1聊天，不新建聊天/PM、不派或唤醒子agent。通过“固定失败回放 → 几何/支撑/驱动力矩审查 → 无神经网络真实控制基线”分开策略、训练接线与工程可达性；现已找到原前倒冷出生的源仿真起身→保持→移动开发路径，但5mm与四方向资格未通过；不将现有Actor失败归结为硬件不可能，不用更多PPO替代尚缺的接触/可达性验证。电脑端刚性足底、20ms一次积分/控制、原真实驱动/功率/热约束与5mm独立门槛保持；诊断反事实配置只作诊断，不继承正式资格。此前十小时集中推进窗口已过，结果见下文；拾物仍后移。用户最新已授权本聊天对确实影响推进的硬件做具名微调并统一记录，最终再交硬件工程；现有H001保留作对照，不自动晋升。

**现行最小因果结论与派工（2026-10-06）：** 首次落脚超标的接触激活因素已确认。仅17次真实CPU积分，同完整0.12s状态/原驱动历史下：原动作下一20ms穿入10.840mm；只保持旧目标9.840mm仍失败；仅加5mm脚发现gap，状态/动作/目标/力矩/热逐值不变；仅加5mm脚active margin后穿入3.404mm、脚承托28.09→67.85N，动作目标与实际驱动力矩逐值不变。独立原出生7Tick重复一致。5mm active margin明确只作诊断，它改变正间隙下的受力边界，不直接采用或继承资格；17积分加前序110,025共110,042，GPU/PPO/Adam/ONNX均0。**下一项为原生接触方案的具名独立准入与完整动作闭环复核**，检查真实接触间隙/承托、冲击/摩擦和CPU/GPU一致性，先取得合法起身/保持/1m再恢复学习；不再盲搜旧姿态族或奖励/增益网格。原策略和训练实施失败责任已确认，局部结构干涉不等于全局不可达；严格动态存在性仍未全部闭合，不把局部20ms改善称为完整修复。原合同/生产/硬件均未改，不联系或唤醒其他聊天/agent。[最新因果报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_contact_activation_cause_001/diagnosis.md)、[同状态结果图](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_contact_activation_cause_001/same_state_contact_cause.png)、[实际收据](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_contact_activation_cause_001/causal_contact_tick_receipt.json)。

**前序集中归因与决策（2026-10-06，历史证据）：** 已确认当前策略四方向起身失败，且此前训练接线/奖励实施与未先闭合动态可达性的顺序由主实施负责；没有证据把失败判为PPO算法不适用或整机结构全局不可达。本轮整段原生轨迹有界检查110,025真实CPU积分，GPU/PPO/Adam/ONNX均0。SLSQP把所选参数族地面峰值27.324→10.840mm，末态仅脚承托；TRF无实质改善；微分进化10.492mm但新增3.045mm代理自碰撞且丢失仅脚支撑；先收腿路线19.117mm/自碰撞6.709mm。所有候选严格拒绝，初始/选中轨迹独立逐值复现。此有限参数族收口，不扩seed、代数、驱动增益或奖励网格。原开发起身/保持/移动闭环仍超标，不作合法教师；严格动态存在性尚未全部定论，不写“结构没问题”或“结构起不来”。**下一问题回到源端动态接触/驱动准入**，利用已保存超标瞬间与合法站立段，区分动作时序问题与20ms原生接触/伺服适用边界，先取得可复核因果修正再恢复能力学习，不再盲搜完整姿态族。不得因末态或优化器success晋升；50Hz、原驱动/热/350W、5mm与完整任务范围保持。生产代码/硬件本轮未改，不联系其他聊天/agent。[当前归因报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_trajectory_feasibility_001/diagnosis.md)、[实际失败录像](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_trajectory_feasibility_001/native_transition_failures.mp4)、[逐帧审查与收据](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_native_trajectory_feasibility_001/trajectory_evidence_audit.json)。下文前序“下一问题”只作历史记录，不覆盖本段派工。

**前序接触因果对照（2026-10-06，历史证据）：** 本轮851真实CPU积分（执行/独立重复582＋规划269），GPU/PPO/Adam/ONNX均0。同原前倒出生保持关节目标50Tick仅.696mm穿入；原动作两遍逐值复跑均21.427mm，与原生接触距离相同，说明动作引入接触冲击，不能归因于出生即不可维持或测量误差。一Tick原生目标投影执行7Tick≤4.5mm后未找到下一合法目标，未完成承托；头+.6目标对照175Tick穿入20.878mm并新增躯干/下嘴代理自碰撞7.750mm，均拒绝、不扩局部参数网格。回到上游物理方案，隔离3.13原生积分器同50动作：implicitfast13.818mm、discrete12.537mm（50Tick独立重复一致），仅解除REFSAFE反而24.566mm，不采用、不继承旧3.10/GPU/Bevy资格。原训练环境实测仍3.10，原CAD/SI/21体/18轴/11叶和驱动/热/功率未改。下一问题改为整段承托转换的原生轨迹可行性检查，包含动量及驱动历史，退出一步贪心/单关节调整，不自造物理求解器或奖励网格；完整5mm起身→150Tick保持→1m教师路径未过之前，恢复PPO保持停止；四方向/自然跌倒和完整移动仍独立验收，不预先授予资格。[最新因果报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_contact_phase_admission_001/diagnosis.md)、[接触真实录像](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_contact_phase_admission_001/contact_action_vs_native_solver.mp4)、[资源收据](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_contact_phase_admission_001/cycle_ledger.json)。

**原前倒首次连续闭环已证明（2026-10-06）：** 本轮6,577真实CPU积分、GPU/PPO/Adam均0、ONNX1,099次。原21体/18轴/11叶、实际驱动/热/350W和50Hz保持，出生后根/关节/速度/驱动历史写入0。只改颈目标−.6→+.6rad改善身体倾角、COM进入真实脚接触区；仅抬头和脚方向反馈仍不能平衡。将原全身重力平衡解转换为原18维位置动作后，头/身体载荷0、脚102.325N，连续196Tick仅脚承托；原连续额定静力需104.89%，此姿态只作短时峰值过渡、不作长期停留资格。83有限原范围姿态及静力路径核对后，实际连续展开到站立；静力LP补回原DOF摩擦反力，未改变实际物理。原前倒→连续150Tick站稳→N512热切换走1.0007m首次实际通过，首个149Tick回放未偷算成功。去掉等待后，首次瞬间站立指标8.68s、持续150Tick段9.54s开始/12.52s完成，再走1.001424m，19.90s闭环。独立完整995Tick q/施加力矩/action逐值一致并保存积分前65维观测；不是新恢复神经Actor或四方向资格。未采用髋下移、扩大颈范围或关闭碰撞，代理自碰撞0，旧深屈膝CAD坏姿态未经过。

**剩余准入问题集中到接触阶段：** 整条开发路径最大地面几何穿入21.427mm，发生于最初.26s头部；脚法向阶段8.388mm、真实仅脚平衡1.617mm、抬身2.080mm、站立.976mm、N512移动13.055mm。5mm仍false，CPU静态/开发证据不继承GPU/Bevy/硬件资格。直接平滑冷启动替代失去承托；只减慢初段颈请求速度至1rad/s使穿入升到27.104mm，也未通过，不保留、不开增益/速度网格。下一问题仅做状态与接触相位协调的合法首次落脚，复用已证明的仅脚承托→站立段；保留其余倒地方向/自然跌倒及完整移动扩验。严格5mm完整路径未过不恢复PPO；不因一例宣称全机/制造可达性，也不再等待硬件修改。[本轮动态因果报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_wholebody_pitch_transition_001/diagnosis.md)、[原策略失败与实际起身闭环录像](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_wholebody_pitch_transition_001/front_getup_hold_move1m.mp4)、[资源/独立复现收据](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_wholebody_pitch_transition_001/cycle_ledger.json)。

**前序承托/结构诊断（2026-10-06，历史证据）：** 新增1,625真实CPU积分，GPU/PPO/Adam/ONNX均0，原21体/18轴/11叶、驱动/热/350W及50Hz不变。500Tick原前倒同前缀+踝反馈使脚底法向Z≈1，但头40.15N/脚62.13N、COM越出实际脚接触区32.52mm；不是仅脚朝向问题。另1,125Tick同出生+原几何关节路径+踝反馈，末态身体42.46N/头16.70N/脚43.17N、直立度.214、COM131.66mm，0站稳，两条轨迹5mm均失败。固定机身/腿位的头颈有限工作区无承托目标；颈负范围−.6→−1.2rad反事实仅改善.512mm，不采用。上述固定末态/踝朝向/静态关节参考路线收口，不开增益或奖励网格。

**真实局部干涉已证实，不能判死全机（前序证据）：** 在该失败末态固定机身/髋/上身收膝至1.25rad，右前壳体与右胫部叉片实际源CAD局部双向表面交叉，42,214次边/三角查询得到1,000条可重复交叉记录；不是1,000个碰撞对象，也不是整机/内部件认证。原CAD、本体、合同不改。该查询的+5.243mm是COM对潜在脚底投影裕度，脚实际离地约27mm，明确更正为无承托入口。所选直接收腿路线因此无效；既不能靠关自碰撞，也不能据此宣称所有路径必须改硬件。当时待验证的身体倾角与承托转换已有上文本轮前倒动态正例，但严格5mm完整路径尚未证明；原倒地→站稳150Tick→走1m且5mm达标前保持PPO停止，不恢复跨聊天联系或子agent派工。当前策略失败及此前奖励/接线漏洞已确证，PPO算法不适用和原结构全局不可达均未证实；完整移动/恢复/Bevy仍未通过。[前序集中报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_support_mode_control_001/diagnosis.md)、[真实失败录像](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_support_mode_control_001/support_mode_failures.mp4)、[CAD局部交叉图](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_support_mode_control_001/cad_local_interference.png)、[实际收据](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_support_mode_control_001/cycle_ledger.json)。

**前序接触转换诊断（2026-10-06，历史证据）：** 新增1946真实CPU积分、GPU/PPO/ONNX均0。原前倒直接拉90°支撑目标，腿跟踪误差<.019rad但头俯仰差1.785rad；仅头峰值4→8Nm/连续2→4Nm的独立反事实改善头跟踪，仍未起身，不作换电机结论。两个具名头颈/45°目标原驱动对照仍失败，直接关节拉姿态路线关闭。248有限FK查询另算，静态可行不替代动态。原卡住全状态的简化接触LP不可行，但真实原生一步头载荷31.69→18.66N、足载荷70.14→125.81N、COM+1.65mm、机身转回.59°，有局部动作余量；简化LP不作全结构不可达证明。连续三步主回放与原生规划分支q/力矩/特征误差0，第四次因局部头载荷≤5N诊断限制停止，这不是正式恢复规则。独立连续203Tick精确重放再保持150Tick，COM最高176.69mm，最终脚/头各约51N、脚底法向Z最低.403，仍0站稳；解除限制也没有完整成功。该一步贪心方案关闭，不恢复PPO、不扩硬件/奖励网格。结论：策略与当前控制方法未解决真实倒地→脚底承托的动态过渡，主实施此前训练顺序不充分；不能判死硬件，也不能承诺更多PPO能解决。下一步只验证允许非单调动作、临时身体接地、包含速度及驱动历史的完整原生动态路径，保持3秒再走1m且5mm达标后才恢复恢复训练。[集中因果报告](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_fallen_support_transition_001/diagnosis.md)、[实际瓶颈录像](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_fallen_support_transition_001/fallen_support_bottleneck.mp4)、[资源收据](/home/ethan/ProjectBackups/2026-10-06/Sai_Lab/goose_fallen_support_transition_001/cycle_ledger.json)。

**策略与局部力学证据进一步区分：** 只读查询、0新增积分。434更新Actor四方向末100Tick平均脚载荷分别16.03/0/3.94/16.27N，全机102.33N，脚底部分或全部朝上、身体仍接地，当前策略没有有效脚底支撑。独立无神经网络关节路径4秒末态则由头与脚边缘承托，COM投影越出仅脚接触支撑区79.28mm、脚接触维度3；保持该接触模式时不能无头支撑静态平衡，必须变脚位/重心或动量。这个局部必要条件解释当前控制失败，不是所有起身路径不可能。两组查询与实际载荷分别保存，不混用Actor和诊断控制世界。

**能力状态（2026-10-05 23:12，UTC+8）：完整移动、真实四方向倒地起身、Bevy迁移均未通过，十小时窗口目标未达成。** 现有源端开发成果为站立、低速前进、低速左转及低位屈膝起身闭环；不把它们称为完整能力。保留原N512：站立60秒漂移1.16mm、20秒行走1.372m，但航向偏约0.975rad。12°/80mm屈膝出生四方向独立ONNX共3,263真实CPU积分均能起身、连续保持3秒并热切换实际行走1m，仍不是真实倒地资格。[低位恢复实际录像](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_progressive_bounded_recovery_001/crouch80_back12_getup_hold_move1m.mp4)。

**H001授权及已完成对照：** 用户要求影响推进的硬件问题由本聊天直接微调并统一记录，最终交付再带给硬件工程；不恢复跨聊天联系。已生成独立 `goose_task_proxy_11_hip_lower15_v1`，左右完整髋腿模块在躯干坐标下移15mm，原SI/模型不覆写。当前作为具名对照保留，不自动替代旧本体或继承资格。原四方向倒地的同策略新旧本体动态对照已完成：各4例×650Tick，共5,200真实CPU积分，两侧起身/保持/继续移动成功均0，最长稳定保持0Tick。本轮安装微调收口，不继续扩大硬件范围或开重复奖励参数批；不能据此断言新版重训或本体恢复不可能。该周期提出的动作发现建议已由顶部最新“先诊断、后训练”顺序覆盖；本体变体保持对照身份。

**当前起身因果审查（2026-10-06 00:35）：** 已停止PPO、释放GPU。具名434更新策略在原四方向倒地各650Tick全部0稳定保持；前倒第二遍状态/力矩/动作逐值一致。原驱动无神经网络零动作在两种80mm/12°低位侧倾抬升并保持150Tick，前后低位不通过，全部5mm未过。15个标准双足静力配置包含真实质量/COM、Jacobian、嘴闭环反力与内接保守摩擦锥：无碰撞0–1.0rad屈膝段需要≤30.46%峰值／57.11%连续力矩，1.1rad起躯干/胫部代理重叠。原结构还存在+90°双足平地、COM在支撑面内的静态候选，需63.88%峰值／119.77%连续力矩；不是长期可停留姿态。147个连接到站立的几何/静力点通过，实际原驱动回放未通过。新增诊断夹具几何精确触地时实际0接触/0足力，第一20ms COM下落3.88mm；仅在这个合成夹具冷初始化给0.85mm接触深度，实际8接触/106N支撑，第一Tick变化+0.158mm，仍未站稳。正式倒地出生未移动，出生后状态写入0。该修正夹具路径能抬到COM260.64mm但直立度最多.715，随后跌落，不能作教师成功数据。原生contact名义solref5ms被REFSAFE/dt20ms限定至少40ms；同策略改名义40ms的650Tick状态/力矩/动作/深度逐值相同，不能据此认定接触时间常数就是全部起身根因。本周期10,402CPU真实积分、GPU/PPO均0，初版摩擦“保守”标签错误已另存修正、主要静力数值不变。结论：策略任务失败及局部模型障碍成立，全倒地动态可达性尚未证明，不能判死硬件或PPO。下一步只做原倒地→实际脚底承托→反馈稳定抬升的控制可行性诊断，不再开训练/奖励网格。[完整因果报告](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_recovery_cause_audit_001/diagnosis.md)、[真实对照录像](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_recovery_cause_audit_001/four_fallen_failure_cause_audit_final.mp4)、[收据](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_recovery_cause_audit_001/cycle_ledger.json)。

**HumanUP发现短批已停止：** `goose_humanup_discovery_v1` 作者23项原函数/原权重AST一致，具名本体测量/轴适配；原生RSL MLP/PPO、65→18/Critic73，fresh归一化/Adam、作者探索角量/站立出生混合/高度正则课程，未复制作者1ms×20或外力，也不是全RMA架构复现。112CPU/1PPO/20Adam接线、ONNX差1.79e−7；1024GPU准入40,960积分加6CPU对照、力矩差6.08e−7。原GPU进程exit143无正常终止收据，日志证实至少440更新/10,813,440积分，不编造丢失的最后部分。从明确model400（实际401PPO/8020Adam）原生恢复Actor/Critic/归一化/Adam/课程，再完成33PPO/660Adam/815,104积分后收到用户诊断优先纠正，主动安全停止，保留434更新Actor；原分支401之后的训练资源另计，未混合策略。最终四方向独立检查0/4，未晋升。本体/驱动/50Hz不变；不重启该批。[冻结协议](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_humanup_discovery_001/protocol.json)、[停止与资源收据](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_humanup_discovery_001/cycle_ledger.json)。

**本体尺度适配短批已完成关闭：** FK审查38姿态、0积分，Goose hip-knee-ankle路径235.544mm、G1为640.933mm。原100mm足高目标相当于腿长42.45%对15.60%；原线速度std=.5使静止在.1m/s横移指令仍得96.08%跟踪项。`goose_upstream_dimensioned_velocity_v1`保留上游14原函数与原数值权重，具名改变三项任务参数：两项脚高目标36.750mm，线速度std=.098425m/s。不是单变量因果实验，未声称参数不匹配就是全部根因。完成417PPO／8,340Adam／20,496,384GPU积分／512.54秒；最终11,488独立CPU积分无开发行为项通过，另4,887确定性GPU积分/0优化器也失败，左移早期终止，5mm与完整能力不通过。收口、不晋升、不再沿此配方盲试。[完整收据](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_upstream_dimensioned_velocity_001/cycle_ledger.json)、[实际失败录像](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_upstream_dimensioned_velocity_001/motion_failure_modes_final.mp4)。

**H001虚拟安装布局增量：** 原1.2rad俯仰抬腿的右躯干/胫部间隙−9.65mm，新版+5.33mm；左右0–1.2rad共50点回归原模型红、新模型绿。完整1.8rad扫描仍有约32.9mm干涉；50mm下移只作几何对照，没有生成或采用第二模型。21体/18主动轴/11叶/10.430690821kg、局部惯量、驱动、范围、过滤、求解和50Hz均保持；名义站立COM因布置变化由279.758变为286.454mm。出生由新结构计算，脚/腿世界位置与原版相等，后续不写状态。5,100真实CPU积分覆盖20冷重置、站立60s、小动作、嘴约束与N51220s前进，数值守卫全过；站立漂移.0805mm/穿入.8545mm。旧N512前进XY(1.7055,−.4763)m，对照原版同命令/策略/1000Tick为(1.3721,−.7931)m；新增移动对照1,000积分、恢复对照5,200积分，本周期共11,300。动态穿入12.344mm，未达5mm；恢复对照两侧0/4，未证明倒地恢复收益，本微调验证已关闭。未在新版启动GPU/PPO，M0-S/T及行为资格仍false。原CAD见证点邻域的双向实际三角表面查询未发现壳↔叉片交叉，不能把凸包重叠直接归责为真实壳体缺陷，也不能据局部阴性宣称整机无干涉。原硬件CAD未写入，安装件/BOM未制造认证；这是独立虚拟布局而非已完成实物改造。[增量完整记录](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_leg_clearance_adjustment_001/hardware_delta_001.md)、[冻结收据](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_leg_clearance_adjustment_001/cycle_ledger.json)。

**冻结物理与成熟链路：** `goose_task_proxy_11_rigid_braking_v1`，21机器人刚体／18主动轴／2被动坐标／11实际凸叶／10.430690821kg，9原凸网格＋2内接刚性脚box；遗漏44.76%脚体上部凸体积，不继承完整鞋形资格。模型SHA256 `3dd030f0475efd101ab129288e7f04f5694067757d85e0429933cecb9c5268e4`，合同SHA256 `d3fd7ba668678e1e1237d00bd7fac77e5d0625d77cc65bea85d3579ffb17061c`。真实嘴四杆、原SI／轴／关节行程／力矩／热／350W正做功限额保持。17原生隐式位置驱动＋1嘴转子力矩，制动有界保留。原生MuJoCo3.10／mjlab1.3／Warp1.12／RSL5.0.1 PPO，2048GPU世界；Actor65→18、Critic私有信息明确记录。物理、驱动、当前Actor均50Hz，每Tick一次20ms真实积分/推理、decimation1，无隐藏子步或状态/动作覆盖。动态接触5mm资格仍失败，50mm仅逃逸停批界线。

**移动最新完成批：** `goose_mean_velocity_markov_v1`保持成熟原始Actor与完整Adam迁移，Critic69追加3维私有完成Tick平均速度为72。2048GPU准入81,920积分＋6CPU同状态检查，实际力矩差1.03e−6、Actor观测差1.79e−7。完成1469PPO／29,380Adam／72,204,288GPU积分／1,653.964秒；独立128／256／512／1024／最终各运行16项检查，最后一批10,967CPU积分，原5mm和完整移动标准未通过。1024低速左转平均0.143rad/s；最终低速前进平均约0.088m/s、左转0.125rad/s、60秒站立漂移2.97mm，后退／横移／右转接近原地。现存实际转向[ONNX回放录像](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_mean_velocity_markov_001/motion_1024_yaw_left_course.mp4)，只是源端开发。[完整批收据](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_mean_velocity_markov_001/cycle_ledger.json)。

**自写奖励批已停止：** `goose_direction_exploration_v1`完成248PPO／4,960Adam／12,189,696GPU积分／299.04秒，独立最终移动评估尚无完整通过；探索噪声单独不足以证明核心根因。`goose_foot_supported_recovery_v1`完成615PPO／12,300Adam／30,232,576GPU积分／689.71秒，五次共325,000确定性GPU探针四方向成功仍为0。实际失败录像与保存状态表明：躯干直立约0.912、COM约0.16m、两脚法向约−0.71／−0.76、躯干仍接地，却能拿足承重奖励。这是已确认的任务定义漏洞，未证明PPO算法、积分器或硬件不可能起身。自写的sole-support修补只有CPU草案，按用户纠正没有训练，不作为下一派工。[四方向失败录像](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_foot_supported_recovery_001/four_fallen_failure_128.mp4)。

**原版奖励与参考点对照均已收口：** `goose_upstream_native_velocity_v1`直接使用mjlab1.3 `unitree_g1_flat_env_cfg`全部14项原函数和原数值权重/std/脚抬升目标，只作具名轴/传感器接线与上身standing .05系数扩展。完成885PPO／17,700Adam／43,499,520GPU积分／1,075.27秒；独立128／512／最终各12,000CPU积分，最终只有站立行为单项通过（漂移48.62mm），5mm及完整移动均未过，未晋升。[原批收据](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_upstream_native_velocity_001/cycle_ledger.json)、[实际失败录像](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_upstream_native_velocity_001/motion_failure_modes_512.mp4)。另一个同种子/初始Actor/原奖励数学对照 `goose_upstream_torso_com_frame_velocity_v1`仅以原函数跟踪torsoCOM速度，完成896PPO／17,920Adam／44,040,192GPU积分／1,075.86秒；512及最终各12,000CPU积分无行为项通过，最终站立漂移419.20mm。CAD根原点距躯干质心约330mm、摇摆速度差确实存在，但改参考点没有改善，不认定它是停滞根因。[参考点批收据](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_upstream_com_frame_velocity_001/cycle_ledger.json)。两批原冻结配置/失败数据保留，不继续奖励网格。

**已确认并隔离的训练接线漏洞：** 旧穿透检查调用原生FK时就地刷新xpos/xquat/xipos，却不更新cvel/subtree_com，导致reward-before-forward读取混合时刻数据；24次CPU真实积分证实位置变化约3.7mm、四元数变化约.023，而速度派生数组不变。旧CPU收据的 `upstream_derived_data_lag_preserved=true`声明据此纠正，原收据不覆写，纠正在两批ledger中列明。这不是已证明的全部停滞根因。[原边界审查](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_reward_boundary_audit_001/receipt.json)。第一只读候选v1保留了奖励快照，但终止改读滞后深度；50mm当前读数守卫在53,248GPU积分／1PPO／20Adam后停止该批，失败收据保留，没有继续长训。[v1收口](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_upstream_consistent_snapshot_velocity_001/cycle_ledger.json)。当前v2改为独立、不积分的WarpData，只复制当前qpos/mocap并调用上游FK，保持原终止检查的新鲜深度，又不改写实时奖励数组；私有Data零约束/接触容量只用于FK，不改真实物理容量或求解配置。192CPU积分同初态/动作回归，状态、力矩、目标、热、完成Tick穿透差均0；24个奖励边界派生数组无改写，私有FK深度与完成Tick原生读数一致。另112CPU积分/1PPO/20Adam、1000原观察均值完全一致、Torch/ONNX最大差1.49e−7。2048GPU准入81,920积分＋6CPU同状态探针，力矩差7.97e−7、Actor观察差1.19e−7。已完成861PPO／17,220Adam／42,319,872真实GPU积分／1,037.84秒，数值有限无失败；14原函数/原权重、模型/驱动/65→18/50Hz不变，独立5mm验收不变。[v2冻结协议](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_upstream_consistent_snapshot_velocity_002/protocol.json)、[v2回归](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_upstream_consistent_snapshot_velocity_002/snapshot_regression.json)。

**生产接线修复验证：** 已将私有FK深度检查落回 `crates/dev_tools/python/src/bevy_microduck_tools/goose/foot_curriculum.py`，新增奖励边界回归在旧函数上实际失败（位置数组差5.355mm），修复后相关11项测试通过、工程结构676项通过。只修复查询副作用，不改奖励数值/模型/驱动/Actor合同；原训练批和原策略哈希不覆写。[生产修复收据](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_reward_boundary_audit_001/production_fix_receipt.json)。v2的128／512／最终各12,000独立CPU积分，无行为项通过；最终同一确定性ONNX还独立运行5,400真实GPU积分、优化器0、无自动重置，GPU同样缺少有效命令跟踪（前进.3指令RMSE .320、后退.15指令RMSE .176），冷启动Actor观察CPU/GPU差5.96e−8。因此不能把停滞仅归为ONNX或CPU迁移，接线修复也不是已证明的全部根因。该短基线收口、不晋升、不继续同奖励网格，保留原N512。下一有界问题是成熟任务数值的本体尺度适配与动作可达性/真实示例组织，避免再开同类盲试。[当前完整批收据](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_upstream_consistent_snapshot_velocity_002/cycle_ledger.json)。

**姿态表征与采样规划调研完成：** 用户授权的单一research子agent已交付并闲置，研究阶段0积分/0GPU，不再重复唤醒。[完整调研](/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/docs/goose_recovery_planning_research.md)核对OnlineSMC、MJPC、Hydrax、HoST、HumanUP及动作token/latent作者实现。结论：先证明真实倒地→可接管入口的动态边，再收集自身成功轨迹用成熟跟踪/PPO；姿态表征不能替代速度、接触、驱动状态或可执行性。先侧滚等恢复内部动作符合本期范围，不训练MD独立特殊翻滚，不预建Transformer，不声称PPO不可能。

**主实施有界采样与数据接线：** 复用MJPC作者Python `Policy` 样条组织，实际rollout仍走原Goose一步驱动，分支复制 `mjSTATE_INTEGRATION`加外部完整驱动状态；同状态下一Tick状态/观察/目标/力矩差0。最近低位入口启发式明确是本地搜索评分，不是作者起身奖励，不进入PPO。两次仅改变时域1.28→3.2秒，分别52,407与107,615真实CPU积分，共160,022、GPU0；原四方向倒地出生的24个候选tail各批均无起身保持/继续移动成功，关闭这套最近入口/五结点样条方案，不扩大搜索网格，不据此断言本体或规划思路不可能。[采样1](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_recovery_sampling_001/cycle_ledger.json)、[采样2](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_recovery_sampling_002/cycle_ledger.json)。另从已知4条实际低位起身轨迹导出736帧原SI/50Hz/wxyz NPZ，实际由同版本mjlab MotionLoader读取验证；新增积分0，仅为开发参考，不是全倒地示例，不继承tracking默认私有Actor/5ms×4。[数据收据](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_recovery_motion_corpus_001/receipt.json)。

**范围支线已收口：** 原关节范围的100mm落地屈膝几何可达，继续深折出现躯干—胫部碰撞。此前有界审查202场景、0积分／0GPU，扩大hip/knee及144种落点变体无深姿态完整通过；未生成新模型、合同、动作尺度或loader，原哈希不变。不继续扩大范围/碰撞过滤，不把FK当动态证据。[收口记录](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_recovery_range_audit_001/README.md)。

**采样失败的连续复现：** 对第二次采样四方向各选一条具名best计划，原倒地出生后连续执行3.2秒原动作样条→原起身ONNX，无中途状态/驱动重置，共3,240CPU积分和2,600ONNX调用；端点完整状态及tail状态/力矩与保存记录差0，四方向仍均失败。完成帧COM约149–160mm，身体仍低位、脚没有形成有效站立支撑；该观察不证明所有动作或硬件都不可行。没有新增搜索或训练。[连续回放收据](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_recovery_sampling_replay_001/receipt.json)、[四方向实际失败录像](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_recovery_sampling_replay_001/recovery_sampling_failure_sequences.mp4)。

**目标端与执行边界：** 当前所有训练是源端开发。Bevy性能第十周期在世界构造前拒绝004身份，实际积分0；未接回具名21体/刚性接触/驱动合同，不具备迁移或吞吐资格。性能agent保持闲置。[目标收据](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_bevy_performance_001/performance_cycle10/report.md)。每批≤两小时，先pilot再决定续训；两次受控无改善就回到模型或成熟流程。保存显式PT/ONNX、合同、配置、依赖、哈希、真实积分/Adam、失败轨迹；独立确定性ONNX/no autoreset验收，源端/目标端/最终资格分别报告。全部临时脚本、模型、日志、构建和快照落项目备份目录；生产代码只落对应crate。此前细节保留[20:05历史快照](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_ten_hour_execution_001/training_plan_checkpoint_1225.md)，历史“下一批”不是现行派工。

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
- 当前实施聊天的 `Goose ProjectManager` heartbeat 每两小时检查产出与主线进度，不新建聊天。用户授权的单一 `bevy_performance` 子 agent 持续处理目标端性能；每周期有界测量、对照与回归，完成后闲置，下一周期再派工。性能工作不得改碰撞形状／过滤、求解配置、物理参数或观测／动作合同来凑速度。

已实际启动源端有界PPO，当前结果以页面顶部最新检查点为准。最早M0产物归档 `/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/`；本页为唯一计划和状态入口。最终交付冻结策略、本体/合同、Bevy包、复现命令、三主线报告及未通过清单。制造/电气/实物载荷仍由机器人工程侧维护；仿真资格不自动等于实物资格。

完成当前可行性诊断后，长期路线仍为**统一MJCF／冻结碰撞代理 → 原生源端准入 → 成熟小批量PPO与导出 → 站立／移动／恢复源策略 → 目标迁移**。当前不唤醒Bevy性能agent或并发训练。每周期只选能解锁下一阶段的一个问题；局部实验达到诊断目的即收口，连续两次无改善就回到模型或成熟流程，不追加求解器分支。历史检查点中的“下一批”只记录当时决定，以本页顶部现行任务为准。

现行首要工作为**原倒地到脚底支撑再站稳的完整动态可行性诊断**；50Hz、真实接触、嘴部机制和正式行为指标不变，软底专项不再作为电脑训练要求。未完成存在性证明不再投入恢复PPO；失败学习策略不能因更新次数或loss晋升，不以训练回合重置代替独立无reset评价。

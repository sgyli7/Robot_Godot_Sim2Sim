# Goose V0.1｜Move 训练计划

版本：2026-10-07。独立实施分支：`codex/goose_move`。

**现行范围：本聊天独立负责 Move。** 用户已将起身工作拆到其他聊天；本分支只负责站立、前后横移、转向、启停、0.4m/s稳定走路、接近0.7m/s尽可能快的跑步以及走跑模式切换。保留三小时最佳实际连续录像交付窗口（截至2026-10-07 16:57:01 UTC），按最新目标在十小时内推进可检验移动版本。未达到的能力直接报告；不等待其他聊天，不联系硬件/G1，不派额外Agent。

本分支代码工作树：`/home/ethan/Projects/TempWorktree/Sai_Lab/goose_move/Bevy_Sim2Sim`。原工作树与恢复分支不改写；本分支临时实验、收据、录像集中在[Move产物](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001)。继承的历史状态另存[原计划快照](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/inherited_training_plan.md)。

**当前实际结果（2026-10-07 22:39 UTC）。** 本聊天十小时目标窗口截至2026-10-08 01:09:36 UTC。原B本体、18轴驱动和65Actor接口保持，原始移动9份保护hash不覆盖。0.4/0.7m/s、0.6rad/s转向与完整走跑切换均未完成；本分支不承担起身或拾物。

**现行顺序。** 保留第028/512直行策略、原站立策略和左右转向策略；从实际冷启动后50Tick站立状态开始原生GPU PPO移动课程，继续提速。独立接收同时检查世界直线位移、偏航、物理门槛和停止；机身前向速度高但绕圈的候选不晋升。前进、左右转向与热切换分别保留资格，训练更新不自动晋升。下面历史“解锁问题”是当时诊断，不是当前派工。

- **独立Actor左右转向已形成。** 第019候选左转连续20秒，累计+5.146rad，活动段平均+0.256rad/s；第022空间镜像右转连续20秒，累计−5.213rad，平均−0.265rad/s。两侧12秒转向＋4秒停止也完成，末秒平面速度约0.026/0.024m/s。原生物理门槛不变；第022的半周期镜像失败，不采用。它们尚非0.6rad/s、原地半径或200案例正式资格，不能据此替换前进策略。[左转实测](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/actor_turn_evaluation_019/evaluate_trained/receipt.json)、[右转实测](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/mirror_turn_actor_022/evaluation_pure_spatial/evaluate_trained/receipt.json)、[左转21秒连续视频](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/turn_actor_visuals_024/left_turn_actor_21s_continuous.mp4)、[右转21秒连续视频](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/turn_actor_visuals_024/right_turn_actor_21s_continuous.mp4)。两份视频已检查，只有实际轨迹渲染，1062次私有渲染FK不计物理积分。
- **020髋yaw驱动对照已收口。** 两个具名原生伺服候选，累计15090CPU/8290ONNX；kp60旧阻尼仍可移动，但直线进展低于原S，原生临界阻尼候选穿地失败。不采用、不扩展电机网格；本体B及硬件结构不改。
- **023 GPU PPO已完成，未改善。** 同状态动作差5.96e−8的命令归一化迁移保留原步态；统计被冻结，原生动作平滑/角速度权重具名调整，公式未重写。18,874,368GPU积分/768PPO/15,360Adam，561.4秒，数值链路通过。第128/256与最终独立CPU均退回站立，不晋升；原S仍约0.0665m/s。归一化不是已证实的唯一瓶颈，继续重训同课程已收口。[训练收据](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_moving_command_transfer_023/pilot/receipt.json)、[最终实测](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_command_transfer_evaluation_023/evaluate_trained/receipt.json)。025独立前进测试确认左转Actor不能兼任前进，四例104Tick穿地，保留分工。
- **直行第028/512已冻结。** 第028原生PPO18,874,368GPU积分／768PPO／15,360Adam后，独立第031实测选中512检查点：原冷出生→原站立Actor真实50Tick→移动Actor连续22秒，平均机身前向0.10953m/s，世界X前进2.233m／Y−0.406m，终点偏航+0.256rad，最大穿地3.956mm、最低直立度0.977。第033另进程全字段逐位一致，未放宽直立门槛。零指令由原站立Actor承担；移动Actor单独冷出生失败，不能称单Actor完整移动资格。停止1秒速度0.04121m/s略超过0.04门槛，后3秒漂移4.1mm；冷态0.7指令112Tick穿地失败，走跑热切换虽完成却未达目标速度。[冻结清单](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/move_low_speed_baseline_034/manifest.json)、[连续22秒录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/move_low_speed_baseline_034/forward_ppo512_22s_continuous.mp4)、[独立实测](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/protected_stand_moving_probe_031/evaluate_512/receipt.json)。
- **更快但绕圈的最终028不采用。** 其机身前向约0.137m/s，22秒偏航−4.855rad，世界X最终−0.307m；不能据此前进或晋升。继续训练明确从直行512恢复，权重、Gaussian、Adam和归一化均保留。
- **026奖励预算审计已完成；037部分前缀有证据、完整回放失败。** 026确认原小速度课程将迈步排在低代价站立之下；仅校准安装栈14项原生公式的系数后，第028形成更好的实际步态。037将转向跟踪权重0.1→0.5，真实GPU记录的150..489Tick迈步得分4.156–4.225，高于最强静止上界3.345；第490Tick动作带开环回放触发终止，3920真实GPU积分，不称完整物理通过。该审计只支持具名奖励预算选择，闭环仍须独立检查。[037实际收据](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/actual_forward_reward_audit_037/receipt.json)。
- **第036现行训练课程。** 第035已真实执行8世界×50Tick／400GPU积分与400Actor推理，获得同合同完整历史支撑状态，5次ONNX核对通过。第036使用它作真正训练重置，保留episode_age50、相位、动作、速度、驱动、热状态；部署仍从原冷出生连续执行原站立Actor50Tick。移动Actor只优化0.4/0.55任务，原生倾倒终止从70°对齐正式直立度0.95；不改变任何本体、碰撞、接触、驱动参数。先小链路，后至多900秒有界开发pilot；未通过不长训。[035收据](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/actual_stand_prefix_bank_035/receipt.json)、[036冻结协议](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_actual_prefix_resume_036/protocol.json)。
- **局部探针已收口。** 029提高转向反馈、030热切换动作渐变均未普遍改善；032两种原生接触直参数候选与原B不等价，7451CPU／651ONNX，无晋升。它们不再追加网格，也不修改当前训练本体。

- **持续真实步态已形成。** 原11叶／21体本体由原18轴连续执行12次交替步，68.2秒前进约0.69m，平均约0.011m/s；逐Tick无穿透／自碰撞／限位失败，另进程3410Tick全部保存字段逐位一致。它是IK控制参考，尚非独立策略或高速能力。[录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/visuals_002/ik_twelve_steps_68_2s_continuous.mp4)、[独立收据](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ik_repeated_replay_002/receipt.json)。修复了私有Mink忽略原生5对装配排除和误用统一0.6rad/s规划限速的接线错误；物理过滤和真实速度限制未改变。
- **左右转向参考均完成。** 每侧从原冷站立连续37.6秒、1880Tick，真实航向分别+0.573／−0.539rad；另进程逐位回放一致。仍是慢速IK参考，不授予0.6rad/s或独立Actor转向资格。[左转收据](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ik_turn_replay_008_left/receipt.json)、[右转收据](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ik_turn_replay_008_right/receipt.json)、[左转连续录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/visuals_turn_003/left_turn_37_6s_continuous.mp4)、[右转连续录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/visuals_turn_003/right_turn_37_6s_continuous.mp4)。
- **保留实际移动Actor。** 第一批源PPO的确定性低速评估在0.12指令下实际约0.0665m/s，但有侧移／偏航；该模型继续显式保护。第二批7,864,320 GPU积分回归成站立；第三批仅将原生线速度奖励容差0.5→0.06m/s，同样7,864,320积分，没有改善，失败均保留。停止继续盲调奖励／std／seed。
- **模仿输出误差不等于闭环能力。** 原生RSL MLP拟合实际12步参考，6000Adam后导出对照误差4.17e−7；先前一次6000Adam导出失败也记录。独立冷启动闭环在29Tick失稳，不能晋升。[训练／导出收据](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/native_actor_cloning_001/training_receipt.json)、[实际闭环](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/native_actor_evaluation_001/evaluate_trained/receipt.json)。
- **局部物理对照收口。** LIPM提速参考在支撑切换前丢失接触并穿地约10.8mm；仅2mm足margin仍失败，原生implicitfast在嘴铰链首Tick越限，原生12腿dampratio=1仍未解锁加速。原本体和合格慢参考保持，以上候选不晋升、不进入GPU长训。快版静态参考最早失败为左踝侧摆约0.00058rad越限；不能将它说成硬件完全不可行，也不修改验收门槛。

**已结束的受控对照。** 仅实际参考出生的第4批（7,864,320 GPU积分）冷启动7Tick失败；从保护S恢复原Adam、50%冷站立／50%实际参考的第5批（同样7,864,320积分）仍收敛站立。两者均不晋升，停止沿该低速课程追加训练。[参考批](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_reference_start_003/pilot/receipt.json)、[混合批](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_mixed_reference_004/pilot/receipt.json)、[混合批独立实测](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_mixed_evaluation_004/evaluate_trained/receipt.json)。踝侧摆kp20→100的小对照完成两步后5.06mm穿地；加原生临界阻尼反而更差，两候选均不进入训练、不替换原模型／合同，不扩展电机参数网格。

**当前解锁问题：成熟任务的默认姿态接线。** 16原GPU世界在相同物理下连续执行真实慢步／站立，54,560次积分，14项奖励分解确认：原CAD零腿姿作为奖励目标时，慢步得分低于站立。上游G1资产使用弯腿站姿；本模型的零位是编码器/CAD参考。现仅将原生`variable_posture`的奖励局部目标校准到第150Tick实际支撑腿姿，保留上6轴零位、公式、权重、真实冷出生与驱动合同；恢复原生速度容差0.5m/s。保存状态只做原生公式反事实复算，原姿态项最大复算误差1.79e−7，校准后慢步得分高于站立约0.18–0.22，但这不是策略资格。[实际奖励审计](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/native_reward_audit_005/receipt.json)、[具名腿姿](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/preferred_posture_calibration_006/calibration.json)、[反事实收据](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/preferred_posture_calibration_006/receipt.json)。15项原生奖励回归通过。较宽的整文件测试另有24项旧mjlab1.3.0夹具初始化失败（当前执行为1.6.0），不报告它们通过；当前栈的奖励、混合reset和时钟共38项分别验证。

**第6批已完成，尚未解锁目标。** 8世界链路592GPU／50CPU积分、1PPO／20Adam通过；1024×24×512完成12,582,912真实GPU积分、512PPO／10,240Adam，336.4秒。第128更新独立实测在0.4指令下前进约0.066m/s，7.18秒时5.90mm穿地；最终策略近乎站立，均不晋升。[训练收据](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_preferred_posture_006/pilot/receipt.json)、[128实测](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_preferred_evaluation_006_128/evaluate_128/receipt.json)、[最终实测](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_preferred_evaluation_006_final/evaluate_trained/receipt.json)。38项当前栈回归通过；相关代码提交`86c0ed9`。较宽测试的24项旧1.3.0环境夹具失败另存，不算通过。

**动态步态搜索已收口。** 160世界的原生差分进化第009批3,256,000GPU积分，GPU最优约0.042m/s，但独立CPU第202Tick出现0.48mm自碰撞；第010批改为在完整状态私有原生碰撞查询后再选择，另3,256,000GPU积分，GPU最优约0.029m/s。第010候选独立CPU从原冷站立连续631Tick，第631Tick穿地5.58mm，不晋升，不扩展参数网格。[009独立失败](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/periodic_gait_cpu_replay_009/receipt.json)、[010搜索收据](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/gpu_periodic_gait_search_010/receipt.json)、[010独立失败](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/periodic_gait_cpu_replay_010/receipt.json)。新鲜归一化的离线BC另外10,000Adam后原冷闭环第6Tick失败，停止追加BC/MSE/seed对照；实际初始65观测与Teacher初始观测逐位一致，未发现该输入接线错误。

**第011批已完成，未解锁移动。** 新鲜原生MLP/Critic/Adam，真实第150Tick支撑状态仅用于训练重置，动作初始均值由真实支撑动作提供，上6轴在策略内固定；原生速度std0.5、entropy0.01，原本体与18轴驱动不变。8世界链路992GPU积分／1PPO／20Adam通过；1024×24×768完成18,874,368GPU积分／768PPO／15,360Adam，701.9秒，未发生批级数值失败。第128/256/512及最终策略均在独立CPU执行原冷→150Tick真实屈腿→确定性ONNX，累计75,098CPU积分／65,498ONNX／75,098私有FK。启动与站立可连续完成，前进仍近乎0；转向亦未通过，不替换保护S。启动前缀是实际控制器，不能将这批称为单独Actor冷启动资格。[训练](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_nominal_supported_posture_011/pilot/receipt.json)、[最终独立评估](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_supported_start_evaluation_011/evaluate_trained/receipt.json)。

**当前解锁问题：完整原生动作学习单位。** 对真实轨迹直接复算原生动作平滑项，原公共输出单位下的代价相比G1原生学习单位，真实慢参考低约460倍；第011批首/末24Tick样本低约294/371倍。仅把初始Gaussian标准差换算，未同步保留均值/Adam学习坐标与奖励输入坐标，不构成完整原生任务单位复用。现第013批使用项目已有`initialize_native_action_units`、`PublicActionScale`、`PublicUnitsGaussian`与`native_coordinate_action_rate_l2`，保持65→18公共输出、真实驱动、本体、命令、重置、所有原生奖励权重和其他参数；修正学习坐标并明确命名新候选，不继承旧策略资格。[单位审计](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/native_rate_units_audit_012/receipt.json)、[新冻结协议](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_complete_native_units_013/protocol.json)。先做小链路与ONNX对照，再最多900秒小pilot；改善与否仍由同一独立原冷连续评估决定，不能凭单位复算宣称移动已解决。 第013批8世界992GPU／1PPO／20Adam链路通过；完整pilot18,874,368GPU／768PPO／15,360Adam，683.6秒。第128/512及最终独立评估合计59,250CPU／52,050ONNX，仍站立，停止扩展这一支撑站姿冷启动路线。18项动作单位回归在显式绑定当前合同后全部通过；先前漏绑合同的18项跳过不计通过。后续回到实际保护S移动策略，先做函数保持的原生单位转换和同状态对照，再继续推进，不能把静止候选替换它。

**第014批未解锁移动；保护S实际功能保住。** 原移动Actor、Critic、归一化和Adam按具名函数等价单位迁移，16状态原动作最大差8.94e−8；另进程从原冷姿连续22秒仍实测约0.0665m/s。768更新/18,874,368GPU积分后，早期128与最终策略均退为近乎站立；最终独立15,304CPU/ONNX/FK，跑步与切换还有自碰撞失败，不晋升。停止追加同奖励课程。[完整训练收据](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_moving_native_resume_014/pilot/receipt.json)、[保护功能对照](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_moving_native_evaluation_014/evaluate_initial/receipt.json)、[最终实测](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_moving_native_evaluation_014/evaluate_trained/receipt.json)。

**BC启动边界对照015已收口。** 原冷→150Tick真实控制前缀后，旧BC可连续完成，但没有继续迈步；新鲜校准BC依旧在Actor接管后失稳。累计6034CPU积分/5134ONNX，不能把离线小误差视为闭环能力，不再加BC/MSE轮次。[实际收据](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/bc_continuous_start_probe_015)。

**已完成016：保留已有闭环步态的有界校准搜索。** 160个原本体GPU世界，最多900秒，以已保留的S Actor为固定反馈，搜索10个命令映射、步幅和陀螺平衡参数；安装的SciPy差分进化筛选真实直线COM进展、偏航、持续合法时间。原冷出生、每Tick一次Actor/20ms积分，完整状态私有接触查询，失败世界永久失去评分资格。零校准必须保留原功能，最佳候选导出65→18后独立CPU验证长程与停止，提升前不替换S。此路线与已收口的纯周期动作族不同，不是新的PPO奖励调参；目标0.4/0.7与转向资格仍独立验收。[冻结协议](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/gpu_moving_actor_calibration_016/protocol.json)。

**016/017闭环校准路线未晋升。** 016完成3,256,000GPU积分，GPU短窗约0.089m/s；独立CPU约0.096m/s，但第428Tick穿地6.22mm。016结束时收据写入缺失父ONNX哈希键失败，原日志/协议保留，积分与推理计数从37代真实Tick重建并独立验证导出误差2.38e−7；丢失的终止重置计数记不可恢复，不补零。017只在Actor中处理原相位的整数谐波，完成3,206,368GPU积分，最优仍选原1.2Hz；独立第303Tick穿地5.68mm，约0.098m/s。两候选不替换S，停止扩展前进校准/步频网格。[016独立失败](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/moving_actor_calibration_evaluation_016/evaluate_trained/receipt.json)、[016收据重建](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/gpu_moving_actor_calibration_016/receipt_reconstructed.json)、[017搜索](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/gpu_moving_actor_cadence_017/receipt.json)、[017独立失败](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/moving_actor_cadence_evaluation_017/evaluate_trained/receipt.json)、[016实际连续失败录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/moving_candidate_visuals_016/calibrated_S_failure_8_56s_continuous.mp4)。

**现行解锁问题019/020：转向与原低速步态的实际偏航摆动。** 双向转向019共用S状态反馈，每个参数在左右两个原冷GPU世界实际检验；一Tick一次Actor，按两方向最差实际角度/位移/完整合法时间选择，不用准备好的转向出生。020只测试两组原生髋偏航位置伺服增益，力矩/速度/本体/接触/求解均保留，显式另名候选与合同，先20次冷重置、60秒实际站立，再用原S完整连续评估；收益不足则收口。原低速S实际根部偏航角速度RMS约0.827rad/s，与录像姿态有限差分0.821一致；GPU同动作0.827一致，未发现这处角速度框架接线错误。奖励审计018完成2000真实GPU积分，保留14项原生分解；其站立对照使用原S站立输出，仍有原始动作抖动，不能把它当最优低代价站立。BC纯复制前动作的只读对照误差比已训BC更大，不能证明“复制前动作”就是失败主因，不因此追加BC轮次。[018分解](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/moving_reward_budget_audit_018/receipt.json)、[角速度核对](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/moving_reward_budget_audit_018/angular_kinematic_consistency_audit.json)、[019协议](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/gpu_actor_turn_search_019/protocol.json)、[020协议](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/hip_yaw_servo_probe_020/protocol.json)。

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

原始溯源输入是 `/home/ethan/Projects/Sai_Rotbots/artifacts/Goose_V0.1/goose_460_training_checkpoint_20261002.zip`。16,314 个清单文件哈希全部匹配；源装配 460 件、33 个运行刚体、18 主动轴、14 被动坐标、10.430762603 kg、65 观测／18 动作。原证据只有 0.2 秒冒烟和 RSL 初始化，没有优化器更新或合格策略。

权威为包内 `training_checkpoint_contract.json` 和 `stage_one.py`，不使用历史 53／10 的 `control.py`、`runtime.py`、`rsl.py`。零碰撞 training_reference 只用于参数/FK核对，不能作为任务训练本体。

MD 经验落实为：先核本体、实际驱动和观测时效，再调奖励；冷启动与热切换分别测；恢复必须包含继续移动；拾物按实体结果验收；训练完成和候选晋升分开。参见 [MD 冻结交付状态](/home/ethan/ProjectBackups/2026-10-01/Sai_Lab/microduck_player_package_015/DELIVERY_STATUS.md)。

## 2. 技术路线与接口

### 50 Hz 本体

原包物理 0.1 ms、力矩 5 ms、策略 20 ms，必须建立新合同，不覆写原模型身份。

1. 当前源端为独立具名 `goose_task_proxy_11_discrete_mjlab160_v1`，从已核验004及后续刚性诊断派生，21体／18主动轴／11叶（9原凸网格＋2内接原生足box）；总质量、COM、完整惯量、轴位和自碰撞过滤保持。足box的遗漏上部形状与有界制动修正保存在历史版本与合同，不继承原完整形状或控制资格。
2. 足底使用普通原生刚体接触，不使用软底弹簧、压缩曲线或四点替换。软底静载与行程资格不再阻挡电脑能力训练；本期不要求仿真实物软底。
3. 当前上游 MuJoCo 3.15 `discrete` 积分及位置驱动采用独立合同：17轴原生隐式位置驱动，嘴部仍驱动真实输入转子；原动作目标、限速、力矩限额、名义颈部前馈、延迟、热代理与350W功率预算显式保留。它不继承旧显式PD的实际力矩身份。
4. 原33体微脚垫、凝聚软底、自定义BE／discrete和材料诊断路线只保留历史失败，不作为当前训练派工；源与目标资格分别验收。

Rapier 每 Tick 只进行一次 20 ms 积分，CCD 不引入额外时间步。内层约束迭代与物理积分分别计数；历史 PGS 参数诊断不作为新网格派工。弹簧采用 ForceBased SI 单位，碰撞体不重复增加质量。

嘴部采用实际四杆闭合约束，电机只驱动 `beak_input_rotor`，与 head_roll 产生相反反力，约束将力传给 jaw/coupler。禁止逐 Tick 写 qpos、FK 搬动物件或焊接附着辅助抓取。

任务用碰撞代理保留脚、嘴、壳及运动干涉关键表面、来源和过滤映射；不填实空心结构，不扩大自碰撞排除范围凑成功。原邻接过滤、显式嘴销配合排除与碰撞 masks 均可追溯。

### 训练与目标接入

**MuJoCo 50 Hz → mjlab／MuJoCo Warp + RSL-RL PPO → Rapier 零样本评估 → 必要的有界目标微调 → Bevy CPU ONNX 独立验收。**

- 当前执行顺序先完成源端模型和成熟链路。复用 MicroDuck 的 `MjSpec`／`EntityCfg`、任务配置／注册、原生 VecEnv、上游 PPO、复载与导出组织方式；Goose 只增加必要的模型、65／18 控制适配及任务项，不另写训练器、通用环境框架或求解器。MicroDuck 的 BAM、轴序、身体尺度和奖励数值不直接移植。
- 统一 MJCF 从已核验交付派生：显式质量与完整惯量、关节坐标、18 轴驱动、嘴闭环及来源映射只有一份权威；视觉细节与任务碰撞代理分开。原约 15,715 凸块不能直接作为成熟批量训练模型。代理保留空腔及脚／嘴／外壳关键接触面，经几何与载荷短检查再采用。
- 源端优先原生 `mj_step` 和上游已支持的积分器／约束。v2–v6 的自定义 CPU 修正仅用于解释历史失败；未经独立兼容检查不进入 mjlab／Warp，不为保留某个局部方案继续改内核。
- 明确设定 `timestep=0.02`、`decimation=1`，力矩与策略同频；不能沿用常见的 5 ms×4 配置。当前执行依赖为 mjlab 1.6.0、上游 MuJoCo／MuJoCo Warp 3.15、warp-lang 1.15 与 RSL-RL 5.4.2，实际加载路径及代码哈希另存收据；历史 1.3／3.10 环境不与当前身份混用。
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
- 当前用户已明确G1暂停且禁止主动联系其他聊天：GPU前仅检查实际占用，不发G1/硬件协调消息、不停止他人进程。本聊天root独立负责Move与评估，不承担其他聊天的恢复派工，GPU有界串行。
- 两次受控实验无改善就回查物理、初态、观测、终止和奖励分项，避免局部死循环。
- 保存模型/合同/代码/依赖/配置/seed/策略哈希；候选显式指定，不用 latest。M0-S 和源 GPU 链路未过禁止源端长训，M0-T 未过禁止目标训练；数值/约束/容量异常停止该批。
- 每关交付通过/失败/适用范围、录像、算力和下一实验，基于真实吞吐滚动估时。
- 本 Move 聊天的后续唤醒仅检查移动线；不修改原恢复聊天的自动任务，不新建聊天。原用户授权的 `bevy_performance` 子 agent 当前闲置，不按历史周期自动启动。移动与恢复由用户拆分的两个聊天分别负责，本分支只推进移动，不按历史派工唤醒其他Agent。性能工作不得改碰撞形状／过滤、求解配置、物理参数或观测／动作合同来凑速度。

已实际启动源端有界PPO，当前结果以页面顶部最新检查点为准。最早M0产物归档 `/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/goose_50hz_m0_001/`；本页为唯一计划和状态入口。最终交付冻结策略、本体/合同、Bevy包、复现命令、三主线报告及未通过清单。制造/电气/实物载荷仍由机器人工程侧维护；仿真资格不自动等于实物资格。

完成当前可行性诊断后，长期路线仍为**统一MJCF／冻结碰撞代理 → 原生源端准入 → 成熟小批量PPO与导出 → 站立／移动／恢复源策略 → 目标迁移**。本分支不唤醒恢复或Bevy性能agent；移动工作独立推进。每条能力线每周期只选能解锁下一阶段的一个问题；局部实验达到诊断目的即收口，连续两次无改善就回到模型或成熟流程，不追加求解器分支。历史检查点中的“下一批”只记录当时决定，以本页顶部现行任务为准。

现行首要工作以页面顶部为准；不重新启动已收口的300Actor/投影参考配置或扩大局部搜索。源/目标/正式三主线资格仍分别报告，完整目标未达。

参考状态初始化课程依据：[DeepMimic §Training / Reference State Initialization](https://arxiv.org/html/1804.02717v3) 与[作者reset实现](https://github.com/xbpeng/DeepMimic/blob/master/DeepMimicCore/scenes/SceneImitate.cpp)。仅采纳reset阶段的参考状态课程思想，不采纳运行时写真实姿态、不同物理步长或作者控制器合同。

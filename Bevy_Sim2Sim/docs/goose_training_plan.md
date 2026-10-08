# Goose V0.1｜Move 训练计划

版本：2026-10-08。独立实施分支：`codex/goose_move`。

**现行范围：本聊天独立负责 Move。** 用户最新目标为十小时内让 Goose 像 MicroDuck 一样自由移动，先完成快慢档、前后与侧移、原地转向和移动中转弯的统一操作版本。起身、拾物与其他聊天不在本分支派工中。原0.4/0.7m/s目标仍列为未通过速度档，不用请求速度或单项录像冒充实际能力；当前首先完成三轴组合覆盖与热切换。只推进本分支，不联系硬件/G1，不派额外Agent。
本分支代码工作树：`/home/ethan/Projects/TempWorktree/Sai_Lab/goose_move/Bevy_Sim2Sim`。原工作树与恢复分支不改写；本分支临时实验、收据、录像集中在[Move产物](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001)。继承的历史状态另存[原计划快照](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/inherited_training_plan.md)。

**当前执行与最新证据（2026-10-08，148/153/155）。** 148从实际0.24m/s的141/512继续同一原生PPO课程，完成12582912 GPU积分、512PPO、10240Adam、870.36秒，无批级数值失败。153独立五检查点14780真实CPU积分；末512的8/20秒速度为0.2716/0.2703m/s，冷启动、停止和再启动通过物理/停止检查，但0.4速度门槛仍未通过。当前155继续这一有改善的课程，不改本体、奖励、Gaussian或输入编码；156把0.27快档接回受保护的慢档/双向转向组合，正在独立检查热切换，未检查通过前不替换143。[148训练收据](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_forward04_continuation_148/pilot/receipt.json)、[153独立评估](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/forward_continuation_eval_153/independent_comparison.json)。

151步频时钟对照收口：原1.2Hz的3150Tick状态、观测、动作、力矩和接触与145逐位一致；1.8/2.4Hz降低速度并穿地失败，总4740CPU积分，不投入GPU续训。147左弧线PPO完成12582912GPU积分，但149末检查点同时运动比例仅0.5533/0.5733，部分停止失败，不晋升。154实测更强的前进Actor在后退/侧移指令下仍近乎静止，0.7指令第78Tick自碰撞失败；2028CPU积分不能证明硬件不可行。开发工具[停止选择器](/home/ethan/Projects/TempWorktree/Sai_Lab/goose_move/Bevy_Sim2Sim/crates/dev_tools/python/src/bevy_microduck_tools/goose/move_supervisor.py)已用143的14731条真实命令/角色记录验证等价与暂停序列化，新增积分为零，该验证不授予物理资格。[151收口](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/phase_clock_ramp_preflight_151/closure.json)、[152选择器回归](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/production_stop_selector_regression_152/receipt.json)。以下为历史证据，不作为新的派工。

**现行成果与派工（2026-10-08，143/141）。** 143具名开发组合保住约0.14/0.20m/s两档、约±0.3rad/s纯转、右弧线及启停；13条独立程序共15381真实CPU积分，11条通过原冻结开发门槛。慢→快→慢、左转→停→右转→停、60秒直行后停止、左右各20秒转向后停止均通过。零指令时显式按上一次非零用户指令类别选择停止Actor（平移0.25动作历史系数、纯转0.75），不读取速度/穿透等验收指标；公开65→18、每Tick一个网络/一次20ms积分不变。左弧线同时运动比例0.5867仍低于0.6；全动作连续串接第1781Tick穿地5.329mm，失败保留。0.4/0.7、后退/侧移、200例总体及Bevy资格均未通过，不能将此开发组合称为完整MicroDuck对齐。[143独立收据](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/command_history_stop_supervisor_143/evaluate_supervised136/receipt.json)、[快慢切换连续录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/mode_switch_visuals_144/slow_fast_slow_stop_29s_continuous.mp4)、[左右转向连续录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/mode_switch_visuals_144/left_stop_right_stop_25s_continuous.mp4)。

141以保留133/128函数和原生PPO固定学习率5e-5，完成12582912 GPU积分/512 PPO/10240 Adam、756.91秒；145五检查点独立15750 CPU积分，末512的8/20秒前进约0.2413/0.2399m/s、停止及再走物理完成，仍未达到0.4功能门槛。这个方向有实际改善，下一批148继续同课程，不重复改变Gaussian或编码器；147同时从原051的有效左弧线函数进入原生PPO，待GPU依次执行与独立验收。保护143/112/119及旧原件，不加载latest。131/138两次侧移反馈参数搜索均只得到毫米每秒级位移，收口；137仅是实际站姿的静态FK可达性证据，不能据此授予侧移或硬件不可行结论。源端开发小链不继承完整M0-S，目标M0-T与Bevy仍未授予资格。

**当前实际结果（2026-10-08，自由移动重定向）。** 原保护直行051/128约0.14351m/s，原左右转向和固定组合不覆盖用户自由操作。078已核验实际MicroDuck016 Walking ONNX与合同SHA、输入源码：W/S前后、Q/E侧移、A/D转向，三个轴可同时输入；MD原训练版本不可准确追溯，不声称完整源复现，更不把61/14权重装到Goose65/18。

**现行顺序。** 暂停本轮尚未执行的IK扩展。078按直接机身三轴、不覆写玩家yaw、同一Actor处理松键停止，对四份保护Actor各10例实测，共17605真实CPU积分/ONNX/FK；无一份满足统一自由移动，023虽然40项中的自己10项物理完成，但几乎原地，不能记能力通过。新生产 `make_free_move_cfg` 复用安装的UniformVelocityCommand独立三轴采样，保留原站立50Tick及完整reset状态库存、原11叶/21体/18轴驱动、65Actor/69Critic与50Hz；没有新增奖励公式，仅明确适配原生线/角速度误差尺度.15m/s/.3rad/s。命令与旧固定世界航向任务分开命名，旧配置、策略和证据不覆盖。

**统一移动未通过；当前保护112的走停再走，119从有效转向函数初始化原生PPO。** 081完成18,874,368真实GPU积分/768PPO/15,360Adam、683.15秒；四检查点各11条直接机身指令与连续热切换独立实测，共29,702CPU积分，43条物理完成，但没有一份Actor满足冻结功能门槛。第256前进约0.11696m/s，后退、侧移和转向几乎原地；后期退回站立，不晋升、不覆盖保护基线。源端开发小链不是完整M0-S或Bevy资格，Goal保持未完成。

084复用已冻结MicroDuck工作流的0.2秒已完成运动EMA、平面线速度及完整角速度跟踪公式；仅把MD单刚体速度测量改接Goose整机COM原生传感器，不使用MD权重，不继承其训练原始版本资格。原11叶/21体/18轴、50Hz/一次积分、65Actor/69Critic、原动作、物理及稳定性惩罚不改。34项当前栈回归通过，八世界链192GPU积分/1PPO/20Adam，24实际奖励Tick与冻结MD函数最大差4.77e-7。084主批18,874,368GPU/768PPO/15,360Adam、679.47秒；独立30,013CPU积分，42/44条物理完成、功能门槛零通过。086改为单轴左侧移课程，同数量GPU更新、806.80秒；四检查点29,678CPU积分，41/44条物理完成、功能门槛零通过，最终侧移约0.001m/s。两次广范围对照和该单侧课程均已收口，不重复同条件重训。保护051/036/028、站立及019/022，不自动晋升latest；旧误标签留勘误。

**保留能力与瓶颈复核（088–092）。** 088只在具名ONNX中混合当前动作和公开上一动作18维；站立的0.5系数将60秒角速度RMS约0.237降至0.112rad/s、漂移3.01降至1.51cm，但前进和转向减弱，不采用移动平滑。090另命名全腿阻尼四倍候选，5610CPU积分；站立358Tick穿地5.40mm，前进降至约0.020m/s，不采用，原本体B不改。

089确认保护051在直接机身指令[0.4,0,0]下实际约0.143m/s，能同时前进左/右转；8秒活动加4秒同Actor停止的实际转速约+0.075/−0.157rad/s、停止通过，不能记0.4m/s或0.6rad/s资格。091加长到20秒及连续变向，4874真实CPU积分另进程逐字段逐位一致；活动段保持前进/转向，但长段后的停止直立度失败，完整热切换未通过。原078低输入组合未启动该步态，不能据此抹去已有移动中转弯。

092完成12,582,912GPU积分/512PPO/10,240Adam，549.39秒；独立四检查点29,669CPU积分，只有39/48条物理完成，功能零通过，前进均值128/256/384/512依次约0.088/0.063/0.047/0.040m/s。不晋升、不覆盖051。094具名0.8/1.8/2.4Hz原生相位合同对照共13,566CPU积分，1.8Hz仅小幅改善且偏航侧漂，2.4Hz穿地失败，不采用；099只在具名ONNX中反向相位/导数的后退假设共2623CPU积分，未出现合格后退，关闭该路线。

**接线、奖励与现行解锁问题（096–103）。** 096用原S与051在两端实际闭环，5476GPU/5498CPU积分；同状态Actor65最大误差2.98e-7，20秒前进均值GPU/CPU约0.142657/0.142532m/s。未发现明显Actor接线错误，长段停止仍失败，不要求长期接触轨迹逐位一致，也不据此授予Bevy资格。093保存实际连续左右弧线及热切停止失败录像，零额外积分；已查看完整轨迹抽帧，停止末段前倾是真实失败。

097审计092保存的98,304奖励Tick，原地1秒位移≤2cm的90个样本中，线速度项均值0.009727、腾空项0.611111；该项在任意移动指令下支付0.05–0.5秒腾空，不要求实际位移。它是本Goose足课程适配启用，安装的G1原生weight为0；不是永久悬脚奖励，也不是已证明唯一失败原因。098只恢复此原生weight0，完成12,582,912GPU/512PPO/10,240Adam、581.43秒；四检查点独立共34,000CPU积分/48条物理完成，但功能零通过，128前进约0.018m/s，后续几乎原地，收口，不声称修复了运动。

100在原051保护权重上显式把新vx编码成原2×vx；仅校准Actor/Critic的冻结第6列编码器，并计入RSL的epsilon，不改变公共65/69输入、m/s命令与真实速度奖励。原0.4输入步态对应新0.2训练目标，这是密集课程初始化，不是授予实际0.2速度，更不替代0.4/0.7目标。沿用MD0.2秒EMA及原生air_time=0，增加冻结MD精度阶段的command_stop(weight2/std0.035)，三个轴全零才支付，共享一次已完成步滤波；不用MD权重，不改本体/物理/驱动。53项当前栈回归通过；八世界链192GPU/1PPO/20Adam、24Tick三项MD公式误差4.77e-7，初始Actor函数误差1.19e-7。主批完成12,582,912GPU/512PPO/10,240Adam、458.76秒；四检查点独立43,340CPU积分，56/92条物理完成、功能零通过。128/256/384/512前进约0.155/0.148/0.171/0.137m/s，仍有无指令偏航与停止失败，不晋升。两个依赖路径启动错误均在模型/身份检查阶段、零积分中止并保留收据，未计入训练。

**当前103：实际Gaussian探索破坏了已保留步态。** 102只用原051、原本体、共同随机种子及原冷站立50Tick；共12,830CPU积分，确定性8/8物理完成，原继承Gaussian0/8，七例穿地、一例自碰撞，四分之一噪声6/8完成、另外两例在停止末段直立度失败。它测的是训练探索暴露，不是八份确定性重复就授予95%功能资格，也不证明唯一瓶颈。旧064仅固定原sigma值、没有降低其数值，不与此混淆。

103保持100全部课程/奖励/编码器、原051均值权重/Critic/Adam/物理，只将18轴原生log-sigma显式加log(0.25)，随后仍由原生PPO学习；不是保留原Gaussian函数。主批12,582,912GPU/512PPO/10,240Adam、349.81秒，有限失败重置22,872次，100同量为60,613次；该下降不是完整资格。四检查点101独立44,757CPU积分，60/92条物理完成，第256/384各一个直行程序通过开发功能门槛，其余方向未过。384的0.2指令实际0.190872m/s、偏航−0.018862rad/s，停止1秒0.031394m/s、随后3秒漂移8.20mm；已具名哈希保护，不用最后512替代。104三份实际连续对照录像已生成并查看原噪声失败抽帧，零额外积分。累计台账407,371,776GPU样本仅为25份已结束短谱系统计，不是一份收敛策略或能力证明。

**105已收口：复用MD等概率九方向，未补出缺失方向；保留103/384。** 生产适配只替换命令采样：原MD九模板、每桶1/9、幅度0.75–1.25；保留Goose真实50Tick站立前缀、历史/重置、原生2–4秒命令计时，65/69/18、50Hz、本体/驱动/奖励与父Actor/Critic/Adam及已降低Gaussian均保持，不再把vx编码器缩放第二次。42项回归通过；八世界链192GPU/1PPO/20Adam，四次实际CUDA随机采样与冻结MD类体完全一致、24Tick奖励误差4.77e-7。主批已完成12,582,912GPU/512PPO/10,240Adam、521.71秒；四检查点独立45,962CPU积分，14/16/16/16条物理完成，只有旧直行功能通过，侧移/后退/原地转向未补出，收口；原0.4/0.7仍未过。

106已将103/384另进程复验：20/60秒直行分别约0.189915/0.189823m/s，完整物理合法，但长段停止仍未过；8秒走→停→再走时第二段未继续移动，快慢热切也未过。两进程各6800积分、记录全部字段逐位一致，13600积分不授予连续自由移动资格。当前Pose实际walking_threshold=0.01、腿宽0.35rad，不误拿上游G1默认0.5/.05阈值来诊断本配置，也不照抄更窄MD0.12作为放宽。

**112–120现行收口与解锁问题。** 112只在零指令选原Goose站立策略的0.5动作历史混合，非零仍是逐位相同的103/384输出；一个65→18 ONNX If每Tick只执行所选网络，不改本体、驱动、历史或时间步。20/60秒直行后停止与走停再走三条开发程序通过；两独立进程各6800积分、全部记录字段逐位一致，具名保护。113原输入0.14/0.20m/s分别实际约0.14165/0.19087m/s，3250CPU积分五例均物理完成；快慢热切换、200例总体资格和原0.4/0.7仍未过。实际25秒走停再走录像已生成并检查，渲染没有新增物理或推理。

109/110使用安装的RSL-RL Distillation、RolloutStorage和原生MSE单位视图，四份实际合法Goose轨迹2580个标签，120更新/5160Adam、38.18秒、零新增物理；五检查点独立29,003CPU积分，未补出转向，关闭。111从保护父权重运行原生在线蒸馏，八世界链240GPU/1更新/2Adam，教师与原ONNX最大差5.96e-8；主批3,932,160真实GPU/512蒸馏更新/1024Adam、457.51秒，每Tick只应用学生动作，教师另计为训练标签推理。114四检查点独立41,555CPU积分，只保留前进，转向仍未过，关闭；它不是PPO样本或完全M0-S资格。

116用一个公开ONNX条件执行已保留Goose直行/左右转向/零指令站立网络，14745CPU积分，单独0.14/0.20前进及左右转向的真实行为仍在，不能把组合权重当完整热切换资格。117十二组仅髋偏航输出的有界反馈对照，15600CPU积分、物理全程完成但移动转弯未过，关闭、不改引擎。118对116失败轨迹继续执行原Actor和原命令作诊断：2150实际Tick只有一个Tick超5mm（5.108mm），随后自行恢复；首个验收失败保留，未授予资格。

119现行问题是直接从有效转向函数进入标准原生PPO，避免再依靠不成功的蒸馏。以原S隐藏权重、已验证019正转向仿射控制和原生ELU非负恒等通道初始化标准RSL MLP；公开65→18、所有单位及本体不变，内部71个编码量只包含原有gyroX/gyroZ/yaw指令的正负拆分，没有新增传感器或私有Actor输入。58个实际正转状态的施加动作与冻结108最大差1.79e-7，ONNX均值差2.09e-7，零优化/零物理；它只准入正向纯转角色，零指令由原站立策略显式处理，不能冒称全域等价。更换首层后必须重新建立原生Adam并验证所有参数引用，不继承旧优化器；新native sigma为0.25，不声称保留原教师Gaussian。先120独立闭环与GPU小链，再最多900秒正转pilot；旧直行/转向保护件不覆盖。

[112保护清单](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/source_move_stop112_protected_manifest.json)、[走停再走录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/source_move_stop_visuals_115/source112_forward_stop_restart_25s_continuous.mp4)、[119函数初始化收据](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/exact_turn_native_initialization_119/build_receipt.json)。

[103/384保护清单](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/forward_q103384_protected_manifest.json)、[105冻结九方向协议](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_md_balanced_commands_105/protocol.json)、[106独立长段复验](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/improved_forward_baseline_106/independent_replay_receipt.json)。

[102实际探索暴露](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/source_gaussian_gait_audit_102/comparison.json)、[103冻结协议](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_native_gaussian_reduced_103/protocol.json)。

[096实际接线核验](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/cpu_gpu_protected_actor_audit_096/receipt.json)、[097奖励审计](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/actual_reward_objective_audit_097/receipt.json)、[100冻结协议](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_md_command_stop_precision_100/protocol.json)、[连续弧线录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/protected_move_visuals_093/protected051_left_arc8s_stop4s_continuous.mp4)。

[三轴对齐/实测](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/microduck_command_alignment_078/comparison.json)、[081协议](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_native_free_move_081/protocol.json)、[冻结功能评估](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/free_move_independent_evaluation_082/comparison.json)、[084协议](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_md_motion_free_move_084/protocol.json)。历史“解锁问题”不是当前派工。

- **第051/128独立复验直行。** 原冷出生→原站立Actor真实50Tick→新Actor，22秒实际世界前向0.14350695m/s，位移X+2.941m/Y−0.402m、偏航−0.05615rad，最大穿地3.548mm、最低直立度0.983；8秒移动后停止1秒速度0.02074m/s，后3秒漂移4.69mm。五例另进程4350积分/4350ONNX/4350私有FK全部记录字段逐位一致。0.7指令实际仅0.105m/s且绕偏，停止1秒0.05296m/s未过；不授予跑步。原生航向目标只通过公开yaw指令提供，Actor仍65维。[冻结清单](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/whole_com_straight_baseline_056/manifest.json)。
- **第053/054组合程序已保护。** 零指令在前进后选择原站立Actor，在转向后保留该转向Actor处理自己的停止；各留1秒实际中性控制，身体/动作/驱动历史不重置。左右两条27秒连续程序各1350Tick通过物理门槛；快切左右405Tick穿地6.08mm失败。另进程3105Tick逐字段逐位一致，不授予完整切换或0.6rad/s资格。[组合清单](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/source_move_capabilities_054/manifest.json)、[左转组合录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/source_move_capabilities_visuals_055/move_stop_left_turn_move_stop_27s_continuous.mp4)、[右转组合录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/source_move_capabilities_visuals_055/move_stop_right_turn_move_stop_27s_continuous.mp4)。录像只渲染实际连续轨迹，零额外积分。
- **第044已结束，未改善；第051只修正速度测量点。** 044和051各18,874,368GPU积分/768PPO/15,360Adam，约678/680秒。044后期速度下降，不采用；051早期128取得上述直行改善，256变慢，不自动选最终权重。049独立实际轨迹核对原奖励测的是2.215kg刚性躯干COM，验收测的是10.431kg整机COM，横向速度点差RMS约0.290m/s；这不证明它是唯一瓶颈。051使用上游原生subtreelinvel并旋转到相同bodyframe，初态CPU/GPU读数误差1.21e−8，不增加Actor/Critic通道。生产代码新增显式可选适配，旧配置保留，46项当前栈奖励/动作单位/reset测试通过。049初版离线复算遗漏垂向平方项，现改为直接调用安装的原生函数，旧近似留档；GPU训练始终调用完整原生公式。原生角速度跟踪也包含roll/pitch平方，不能解释成纯yaw奖励。[完整公式审计](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/whole_com_velocity_consistency_049/receipt.json)、[051训练](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_native_whole_com_move_051/pilot/receipt.json)、[058课程协议](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_whole_com_speed_course_058/protocol.json)。

- **后续受控结果和实际资源。** 058完成18,874,368GPU/768PPO/15,360Adam，687.9秒；各检查点直行低于保护基线，最终停止460Tick穿地5.09mm，不采用。060将新直行接回转向时，右路线通过，左路线1155Tick在停止时穿地5.15mm，不替换已通过的054组合版本。061只读统计051的77695个结束事件：58319含接触深度终止，Gaussian标准差多数轴后期增长；它提示采样问题而不是证明硬件不可行。062的新原学习单位MSE回归与动作单位共22项通过；其早期128/256尚未改善。运行前已完成的12份本Move主PPO收据合计187,170,816实际GPU积分/8576PPO/171520Adam，未含058、链路、DE和历史训练，不能把样本数当资格。[058独立结果](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/speed_course_evaluation_059/evaluate_trained/receipt.json)、[探索统计](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/training_domain_exploration_audit_061/receipt.json)、[主PPO账本](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/completed_move_ppo_resource_ledger.json)。

- **064与068闭环已完成。** 各18,874,368GPU/768PPO/15,360Adam，651.3/688.0秒。064最终世界前向0.14753m/s，但偏航−0.323rad，比056更差，跑步指令实际0.08407m/s；暂不晋升。068第256退回近乎原地，最终世界前向−0.0263m/s，停止463Tick穿地5.49mm，不采用。066三组幅度校准分别在56..87Tick触发原物理门槛，搜索未启动；它们不证明硬件不可行。[064独立评估](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/fixed_exploration_evaluation_065/evaluate_trained/receipt.json)、[068评估](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/fresh_value_evaluation_069/evaluate_trained/receipt.json)。
- **070/071观测诊断。** 070只分析064保存的8世界147456样本，122段至少8秒未失败片段中最快约0.12882m/s；不覆盖全部1024世界、不证明硬件速度极限。071在两个私有冻结状态只改变根线速度0.5m/s，65观测及18动作逐位相同；已核安装原生G1 Actor含线速度输入，当前Goose的瞬时MLP没有。此差异还不是唯一瓶颈的因果证明。072显式追加3维的Teacher初始原65功能误差8.94e−8、速度快照CPU/GPU误差1.62e−7，公开基线不改。[采样审计](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/actual_rollout_speed_audit_070/receipt.json)、[状态别名证据](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/velocity_observability_audit_071/receipt.json)、[Teacher协议](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_private_velocity_teacher_072/protocol.json)。

- **072/074已收口。** 各18,874,368GPU/768PPO/15,360Adam，671.2/696.7秒。072私有68Teacher增加速度3，初始原65功能与Adam保持，但最终直行世界前向仅0.00407m/s，不证明观测是唯一问题，也不授予公开Actor资格。074使用上游5帧历史，65基础流不变、内部MLP325、历史260携带状态，零新增真实特权输入；初始任意历史功能误差5.96e−8，最新帧与当前65逐位一致。最终直行0.07365m/s，跑步停止460Tick穿地14.20mm，不采用。这两份候选没有替换原模型、驱动或公开基线。
- **窗口收尾证据。** 本Move累计18份完成主PPO短训合计300,417,024实际GPU积分、13184PPO、263680Adam、10811.7秒GPU主训练；包含明确标注的私有Teacher，不包括链路/DE/CPU/render/继承训练，不是一份策略连续300M收敛证明。原9份和两份具名基线中7个策略/检查点条目hash全匹配。077提供新目录复现入口；直行五例新进程4350Tick及组合三例3105Tick全部字段逐位一致。停止失败视频连续9.26秒，最后463Tick穿地5.49mm，未删失败帧。**十小时目标没有完成，Goal保持未完成；不得把本收尾报告当最终能力交付。** [窗口收据](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/move_delivery_guard_077/report.md)、[实际瓶颈录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/move_bottleneck_visuals_076/fresh_value068_stop_floor_failure_continuous.mp4)。

- **独立Actor左右转向已形成。** 第019候选左转连续20秒，累计+5.146rad，活动段平均+0.256rad/s；第022空间镜像右转连续20秒，累计−5.213rad，平均−0.265rad/s。两侧12秒转向＋4秒停止也完成，末秒平面速度约0.026/0.024m/s。原生物理门槛不变；第022的半周期镜像失败，不采用。它们尚非0.6rad/s、原地半径或200案例正式资格，不能据此替换前进策略。[左转实测](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/actor_turn_evaluation_019/evaluate_trained/receipt.json)、[右转实测](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/mirror_turn_actor_022/evaluation_pure_spatial/evaluate_trained/receipt.json)、[左转21秒连续视频](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/turn_actor_visuals_024/left_turn_actor_21s_continuous.mp4)、[右转21秒连续视频](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/turn_actor_visuals_024/right_turn_actor_21s_continuous.mp4)。两份视频已检查，只有实际轨迹渲染，1062次私有渲染FK不计物理积分。
- **020髋yaw驱动对照已收口。** 两个具名原生伺服候选，累计15090CPU/8290ONNX；kp60旧阻尼仍可移动，但直线进展低于原S，原生临界阻尼候选穿地失败。不采用、不扩展电机网格；本体B及硬件结构不改。
- **023 GPU PPO已完成，未改善。** 同状态动作差5.96e−8的命令归一化迁移保留原步态；统计被冻结，原生动作平滑/角速度权重具名调整，公式未重写。18,874,368GPU积分/768PPO/15,360Adam，561.4秒，数值链路通过。第128/256与最终独立CPU均退回站立，不晋升；原S仍约0.0665m/s。归一化不是已证实的唯一瓶颈，继续重训同课程已收口。[训练收据](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_moving_command_transfer_023/pilot/receipt.json)、[最终实测](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_command_transfer_evaluation_023/evaluate_trained/receipt.json)。025独立前进测试确认左转Actor不能兼任前进，四例104Tick穿地，保留分工。
- **第036/128成为新的低速直行基线。** 第036完整有界pilot完成18,874,368GPU积分／768PPO／15,360Adam，673.6秒，数值链路通过。独立选择128检查点：真实原冷出生→原站立Actor50Tick→连续22秒，前向0.13633m/s、世界X+2.763m／Y+0.619m、偏航+0.363rad；最大穿地3.419mm、最低直立度0.978。8秒移动后停止1秒速度0.01950m/s，随后3秒漂移4.5mm；0.7指令连续22秒实际仅0.11468m/s，热走跑切换完成但未达速。五例另进程4350Tick全部记录字段逐位相同，原0.1095版本保留。128是具名源端组合策略，仍非0.4/0.7或Bevy资格。[新基线清单](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/straight_move_baseline_041/manifest.json)、[新22秒连续录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/straight_move_visuals_043/straight_native036_128_22s_continuous.mp4)。
- **第036后期不晋升。** 第256停止穿地，第512／最终机身前向约0.157／0.174m/s，却累计偏航−7.82／−6.47rad，世界前进近零；不以奖励、更新数或机身速度代替直线能力。040替换停止Actor未改善一秒门槛，042双相位探针穿地，两路线均收口，无模型或合同改写。[后期独立实测](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/actual_prefix_evaluation_039/evaluate_trained/receipt.json)。
- **第044历史原生航向目标课程。** 直接复用安装栈UniformVelocityCommand和已有GooseHeadingVelocityCommand，前向均匀0.4..0.7m/s、世界航向目标0、原生航向增益1／yaw限±0.6，恢复上游G1转向跟踪权重2；14项原生公式、原模型／驱动／65→18均保持。它是显式目标命令，不将世界航向偷偷加Actor。训练重置沿用035真实50Tick完整支撑历史，部署仍实际冷启动；重置后先执行原生命令转换再生成观测。045独立验证旧128直接接航向目标仍在热切换停止时穿地，故不将旧策略能力当作新课程成绩。先小链路验证再至多900秒pilot。[044协议](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_native_heading_move_044/protocol.json)。

- **直行第028/512已冻结。** 第028原生PPO18,874,368GPU积分／768PPO／15,360Adam后，独立第031实测选中512检查点：原冷出生→原站立Actor真实50Tick→移动Actor连续22秒，平均机身前向0.10953m/s，世界X前进2.233m／Y−0.406m，终点偏航+0.256rad，最大穿地3.956mm、最低直立度0.977。第033另进程全字段逐位一致，未放宽直立门槛。零指令由原站立Actor承担；移动Actor单独冷出生失败，不能称单Actor完整移动资格。停止1秒速度0.04121m/s略超过0.04门槛，后3秒漂移4.1mm；冷态0.7指令112Tick穿地失败，走跑热切换虽完成却未达目标速度。[冻结清单](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/move_low_speed_baseline_034/manifest.json)、[连续22秒录像](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/move_low_speed_baseline_034/forward_ppo512_22s_continuous.mp4)、[独立实测](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/protected_stand_moving_probe_031/evaluate_512/receipt.json)。
- **更快但绕圈的最终028不采用。** 其机身前向约0.137m/s，22秒偏航−4.855rad，世界X最终−0.307m；不能据此前进或晋升。继续训练明确从直行512恢复，权重、Gaussian、Adam和归一化均保留。
- **026奖励预算审计已完成；037部分前缀有证据、完整回放失败。** 026确认原小速度课程将迈步排在低代价站立之下；仅校准安装栈14项原生公式的系数后，第028形成更好的实际步态。037将转向跟踪权重0.1→0.5，真实GPU记录的150..489Tick迈步得分4.156–4.225，高于最强静止上界3.345；第490Tick动作带开环回放触发终止，3920真实GPU积分，不称完整物理通过。该审计只支持具名奖励预算选择，闭环仍须独立检查。[037实际收据](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/actual_forward_reward_audit_037/receipt.json)。
- **第036训练课程（已结束）。** 第035已真实执行8世界×50Tick／400GPU积分与400Actor推理，获得同合同完整历史支撑状态，5次ONNX核对通过。第036使用它作真正训练重置，保留episode_age50、相位、动作、速度、驱动、热状态；部署仍从原冷出生连续执行原站立Actor50Tick。移动Actor只优化0.4/0.55任务，原生倾倒终止从70°对齐正式直立度0.95；不改变任何本体、碰撞、接触、驱动参数。先小链路，后至多900秒有界开发pilot；未通过不长训。[035收据](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/actual_stand_prefix_bank_035/receipt.json)、[036冻结协议](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/ppo_actual_prefix_resume_036/protocol.json)。
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

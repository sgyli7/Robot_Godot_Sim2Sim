# Goose 倒地恢复：姿态表征与采样规划调研

日期：2026-10-05。范围：有界文献与作者实现审查，服务于冻结 Goose 的真实倒地恢复。本文没有运行训练、物理 rollout、GPU 或硬件试验，没有改变生产代码、模型或 65→18 合同，没有提交 Git。文中的采样参数与实验门槛是后续建议，不是已完成结果。

## 结论与当前建议

**这个 idea 有可信技术先例，但优先验证的是“真实物理下能否转移到可起身状态”，不是“能否把大量姿态 tokenize”。** 物理采样可以帮助寻找先侧滚、调整脚、再起身的非单调动作；学习到的动作表征可以在已有可行技能后压缩搜索。姿态相似、低维 latent、Transformer 预测或离散 token 本身都不证明接触、速度、驱动力矩与能量约束下的可达性。

当前最低实施成本的候选链路是：**原 Goose 50 Hz CPU 驱动与物理 → 少量样条动作的 Predictive Sampling/CEM 离线搜索 → 保存真实成功轨迹 → 用现有 mjlab/RSL 链路学习恢复并独立 ONNX 回放。** 先证明一条倒地到可起身入口的连续路径，再考虑是否需要更复杂表征。恢复阶段可以包含滚到有利姿态；不建立 MD 独立特殊翻滚技能，也不先实现通用规划框架或新的奖励网格。

这条建议可以减少探索的组织成本，**本轮没有证据量化它比成熟 PPO 省多少样本，也没有证明 Goose 四方向均存在满足当前合同的路径**。HoST、HumanUP、DeepMimic、AMP 和物理动作 token 的成功，反而说明应先核对成熟 RL 的动作发现、初态组织与动作跟踪，不能由当前失败推出 PPO 不可能。

## Goose 已知事实与不能越过的边界

以下来自本轮允许只读的 [主线计划](/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/docs/goose_training_plan.md)、[范围审查](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_recovery_range_audit_001/README.md) 和 [失败帧审查](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_foot_supported_recovery_001/failure_frame_audit_128.json)。

| 项目 | 当前冻结条件及研究含义 |
| --- | --- |
| 本体 | `goose_task_proxy_11_rigid_braking_v1`；21 机器人刚体、18 主动轴、2 被动坐标、11 实际凸叶、10.430690821 kg。9 原凸网格加 2 内接刚性足底 box；不继承完整鞋形或硬件资格。 |
| 驱动 | 17 原生隐式位置驱动加 1 嘴转子力矩；原行程、轴、力矩、限速、热代理、350 W 正做功上限、延迟与有界制动保留。搜索输出必须走原 18 动作语义，不能用无约束关节 PD 或直接写 `ctrl` 替代。 |
| 时间 | 物理、驱动、Actor 均 50 Hz；一次 Tick 一次 20 ms 积分，decimation=1，无隐藏子步。离线可以快于实时，不能把步长缩小后宣称符合源合同。 |
| 网络 | 恢复 Actor 65→18。规划器训练时的完整状态、参考轨迹与 Critic 私有信息可明确记录；不能把姿态 token、phase、目标姿态或规划分支悄悄追加进 Actor。 |
| 接触 | 动态接触 5 mm 资格仍失败，50 mm 只是逃逸停批界线。任何规划成功也必须重新满足原动态门槛，不能借静态碰撞结果绕过。 |
| 能力 | 全套移动、真实四方向倒地起身、迁移均未通过。现有低位起身只是局部控制证据。 |

模型 SHA256：`3dd030f0475efd101ab129288e7f04f5694067757d85e0429933cecb9c5268e4`；合同 SHA256：`d3fd7ba668678e1e1237d00bd7fac77e5d0625d77cc65bea85d3579ffb17061c`。

范围审查给出的原范围落地 100 mm crouch，COM 为 209.868 mm；21 个站立到该点的 FK 样本无实际自穿透。但那次审查是 **0 次物理积分**，不证明动态保持、连续可行路径或从倒地可达。更深姿态的 torso↔shin 冲突说明“更收腿”不能无限推进；有限范围扩展失败也不能证明所有动态动作都不可能。

失败帧 JSON 的 24 行来自完成状态上的静态接触查询，`new_integrations=0`，没有实际足底力。前倒及两侧倒的后期帧躯干直立度约 0.912、COM 约 0.160 m，却有脚底朝向错误及 torso↔shin 接触；后倒约 0.901、COM 约 0.140 m，脚姿态不同。查看 [tick500 图](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_foot_supported_recovery_001/four_fallen_128_tick500.png) 与这些记录，只能支持“竖起躯干没有完成承重起身”，不能从接触标签推断足底承担了多少力。

## 表征、搜索与控制需要分开

令完整状态为 `x=(q, qvel, actuator/driver history, thermal/power state, …)`，原驱动和一次 20 ms 积分为 `F(x,a)`。姿态只是 `q` 的一部分。同一姿态可以有不同的角动量、滑动速度、接触求解状态与驱动余量，因此具有不同的下一步。

* **姿态 tokenizer**：压缩或分类几何姿态。若从任意 FK 姿态训练，只学到形状分布，没有学到 `F`。
* **物理动作 latent/token**：低层控制器 `π(a|当前状态,z)` 已通过物理训练学会实现某类动作；`z` 选择行为。可执行性来自低层训练与当前状态覆盖，不来自离散化操作。
* **规划器**：用真实 `F` 或经核验模型比较候选转移；接触图节点只是索引，边必须经动态执行证明。
* **最终恢复 Actor**：在既定 65 输入下输出原 18 动作。离线教师拥有的信息不自动成为部署输入。

MuJoCo 官方说明了完整积分状态与只复制 `qpos/qvel` 的区别。[状态与控制文档](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#state-and-control)、[`mjSTATE_INTEGRATION`](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjstate) 是分支复制的基础；Goose 在 MuJoCo 外部的延迟、限速、热、功率及 previous-action 状态仍需另行复制。不能每个候选 reset 到同一个姿态、清掉速度或热状态，得到漂亮轨迹后再把它叫真实可达转移。

## 原始工作与可复用实现

### VQ token 与连续 latent

[NCP 作者项目](https://tencent-roboticsx.github.io/NCP/)、[论文](https://arxiv.org/html/2308.07200)、[作者代码](https://github.com/Tencent-RoboticsX/NCP) 的 VQ 编码连接的是经物理模仿训练的状态条件控制器，再学习 categorical prior 并训练高层。它支持“用动作 token 组织已有物理技能”；不支持“枚举姿态 token 就自动得到可达图”。其人形动作库、PD 动作和观测均不等于 Goose。

[PLT 作者项目](https://jinseokbae.github.io/plt/)、[实现](https://github.com/jinseokbae/plt) 将身体分区，使用残差量化及连续细化；仍依赖物理动作专家与在线蒸馏。其分腿、分臂的 humanoid 表征不能直接套到 Goose 颈嘴、大躯干与短腿上；跨部分重组也需要重新检验接触。

[ASE 作者项目](https://research.nvidia.com/labs/toronto-ai/ASE/)、[论文](https://arxiv.org/html/2205.01906v1)、[代码](https://github.com/nv-tlabs/ASE) 提供连续技能 latent，由物理模仿与技能区分目标训练低层，高层再选择 latent。连续空间便于搜索，但插值后的动作、离开数据覆盖的恢复状态都没有可行保证。

**Goose 判断（推论）：** 当前没有自身成功的全倒地恢复动作库，先建立 VQ/Transformer 或 ASE 类低层会增加预训练、数据与重定向成本。若只有几十条真实成功轨迹，直接样条动作、有限动作片段或简单连续基底通常更便宜；即使用 PCA/VAE，仍须逐条真实 rollout。只有在已有大量、多接触模式的可执行动作与可靠低层后，token/latent 才值得作为高层动作空间。

### 真实物理采样、CEM/MPPI 与高层 MCTS

[Online Motion Synthesis Using Sequential Monte Carlo 作者页](https://mediatech.aalto.fi/publications/graphics/OnlineSMC/)、[原文](https://mediatech.aalto.fi/publications/graphics/OnlineSMC/preprint.pdf) 是直接相关先例：在物理世界中采样样条控制，树形保留多条候选，出现滚转与起身。它用的是 tree-based SMC，不能等同 UCT/MCTS。论文 ODE、身体尺度、力矩与接触条件不同，本轮作者页未找到可直接复用的源码。

[MuJoCo MPC 作者仓库](https://github.com/google-deepmind/mujoco_mpc)、[论文](https://arxiv.org/abs/2212.00541) 有 Predictive Sampling、CEM 等实现。最小可读入口是 [Python CPU 样条采样示例](https://github.com/google-deepmind/mujoco_mpc/blob/ff572a21e7c2bf9fda62e1862a758da7e9a8719b/python/mujoco_mpc/demos/predictive_sampling/predictive_sampling.py)，不是必须引入整套 C++ residual、GUI 或 gRPC。示例直接写 `data.ctrl`，并只复制部分起始状态；应复用其采样组织，把 rollout 接回既有 Goose 原驱动一步函数，而不是直接复制其控制通路。[CEM 实现](https://github.com/google-deepmind/mujoco_mpc/blob/ff572a21e7c2bf9fda62e1862a758da7e9a8719b/mjpc/planners/cross_entropy/planner.cc) 可用于更新样条参数分布。

[Hydrax 作者实现](https://github.com/vincekurtz/hydrax) 已有 [G1 起身 MPPI/CMA 示例](https://github.com/vincekurtz/hydrax/blob/bd53d43d414e19557d38ee6c72613bf2e39fe834/examples/humanoid_standup.py)：128 样本、4 个随机模型、0.6 s horizon、4 knots；当前还有实验性 Warp 后端。该示例的真实 CPU 世界步长为 0.01 s，异步版本改为 0.005 s，不能用其策略频率替代 Goose 50 Hz 物理验证。G1 的驱动、碰撞和约束不同，原关节确有力矩范围，不应误称无力矩限制；但示例并未实现 Goose 全套功率/热/驱动合同。[任务 cost](https://github.com/vincekurtz/hydrax/blob/bd53d43d414e19557d38ee6c72613bf2e39fe834/hydrax/tasks/humanoid_standup.py) 中 `sum(square(rotate(unit_z,quat)))` 对单位四元数是常量，这是源码数学审查结论，不能把这一项当已验证的直立误差。

| 搜索方式 | 当前 Goose 的适用位置 | 主要风险与实施成本 |
| --- | --- | --- |
| Predictive Sampling/CEM，短样条 rollout | 最先测试少量左右滚转/收脚候选；无需动作库即可搜索原 18 动作 | 短 horizon 容易看不到起身收益；接触切换不光滑，需保留模式与真实 tail 验证。CPU 可做有界试验，吞吐尚未测。 |
| MPPI / MPPI-CMA | 后续滚动规划、已有较好初始控制时 | 多峰左右滚转被加权平均可能相互抵消。必须回放最终平均动作或选中的样本，不能只看样本集最低 cost。Hydrax 适配比纯 CPU 小采样器重。 |
| 高层 MCTS | 已有少量可靠状态条件动作片段后，搜索多段接触转换 | 若每个 token 都还需学习如何执行，MCTS 没有解决底层困难。分支节点应保存完整状态；起步用小图 BFS/beam 足够，无需大树。 |
| 逐 Tick 的大 token 树 / 原始整段动作枚举 | 不作为当前第一步 | 10 s × 50 Hz × 18 已有 9000 连续参数；量化姿态不会消除真实物理积分或长时信用分配。 |

可以把已验证的 0.2–0.8 s 动作片段作为高层动作，这是规划建议而非现成 Goose 技能。搜索时允许 COM 或直立度暂时下降，以完成滚转、卸载脚或转移支撑；不能每个 Tick 都以“更高、更直”淘汰唯一可行路线。左右模式应分开初始化和筛选，避免均值变成无效中间动作。

### 接触/姿态图加成熟低层控制

[LAAS 多接触规划实现](https://github.com/loco-3d/multicontact-locomotion-planning) 把接触选择、质心/动量优化、全身控制与末端轨迹分开；[HPP-RBPRM](https://github.com/humanoid-path-planner/hpp-rbprm) 提供接触可达性规划，后者仓库已归档。它们是可信的分层先例，但 HPP/Pinocchio/TSID 等依赖、目标人形与接触模型都增加适配成本，不是 Goose 起身即插即用项目。

**Goose 判断（推论）：** 可先使用很小的接触状态图，不引入这些全套框架。节点例如“前/后/左/右倒地”“躯干承托且脚正在重新定向”“可接管的脚支撑低位状态”“站立”。节点保存多个实际状态；边只有经原物理执行成功后才加入。静态无碰撞和几何接触可用于剪枝，不能当作边的证据。接触力、滑动、被动嘴机构和 torso↔shin 冲突仍需动态检查。

### DeepMimic、AMP、HoST 与 HumanUP

| 原始方法与来源 | 真正借鉴的部分 | 对 Goose 的限制 |
| --- | --- | --- |
| [DeepMimic 作者页](https://xbpeng.github.io/projects/DeepMimic/index.html)、[论文](https://xbpeng.github.io/projects/DeepMimic/DeepMimic_2018.pdf)、[代码](https://github.com/xbpeng/DeepMimic) | 示例跟踪、参考状态初始化，以及从示例获得较易探索的中间状态；包括非人形角色 | 原方法含 phase 输入、PD 目标与形态专属示例。参考初始化不是正式倒地恢复，提前终止规则也不能直接禁止恢复所需躯干触地。 |
| [AMP 作者页](https://xbpeng.github.io/projects/AMP/)、[论文](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf)、[代码](https://github.com/nv-tlabs/ASE) | 通过动作转移判别器学习风格，无需逐帧 phase 跟踪；论文有恢复动作与任务连续执行 | 数据转移判别分数不是动力学证明。缺少 Goose 成功动作时，失败低坐姿可能变成错误先验。少量单任务轨迹下，先上 discriminator 未必最省成本。 |
| [HoST 论文](https://arxiv.org/html/2502.08378v2)、[作者代码](https://github.com/InternRobotics/HoST) | 无演示 PPO；righting/rising/standing 的奖励组织、多 Critic 与逐步撤除拉力课程 | 高度阶段不是预先找到的可达姿态图。源码 dt=0.005、decimation=4；初期虚拟向上力 200 N 不能成为 Goose 的可行路径证据。源实现不是当前单次 20 ms。 |
| [HumanUP 论文](https://arxiv.org/html/2502.12152v2)、[作者代码](https://github.com/RunpeiDong/HumanUP) | 动作发现→较慢、受约束跟踪；先转换到适合起身方向再起身的组织方式 | 发现阶段较宽松碰撞/控制正则，后阶段恢复完整条件；dt=0.001、decimation=20。简化阶段动作可能无法转入原条件，Goose 应从开始就保原驱动与碰撞。 |

HumanUP 的“两个训练阶段”与“rollover/get-up 两种动作的组合”应分别理解。它确实验证了先改变倒地朝向再起身的意义；不能据此认定没有人形手臂的 Goose 必须先仰卧，也不能照搬躯干高度奖：Goose CAD root 接近地面，root z 不等于人体 pelvis/base 高度。

本轮核对的 HumanUP [跟踪代码](https://github.com/RunpeiDong/HumanUP/blob/7516e0f27e6f4d1e7365cf64ea577a78247bd8cb/simulation/legged_gym/legged_gym/envs/g1track/g1waist_track.py) 读取关节/头高轨迹并插值至 8 s，以 episode 时钟形成训练参考；实际 `compute_observations` 没有显式 phase/目标轨迹输入，但使用历史观测与私有适配 latent，动作 23 维。因此可以借其训练时参考组织，不能说它原样满足 65→18，也不能误说所有两阶段跟踪都必须向 Actor 加 phase。

已有 [MimicKit 作者实现](https://github.com/xbpeng/MimicKit) 整理 DeepMimic/AMP/ASE，但主要列出 IsaacGym、IsaacLab、Newton 后端，迁移到这里并非最低成本。当前同版本 [mjlab v1.3.0 MotionLoader](https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/tasks/tracking/mdp/commands.py) 已可读 NPZ 的 `joint_pos/joint_vel/body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w`，适合未来的 Goose 自身真实轨迹。其 [默认 tracking 配置](https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/tasks/tracking/tracking_env_cfg.py) 给 Actor 加 motion command/anchor，使用通用位置动作与 0.005×4 时间配置；应只复用数据装载和成熟跟踪项组织，保留原 Goose Actor、驱动与时间合同。

## 从前/后/侧倒怎样转到起身入口

首先定义一个**可接管状态集合 B**：从 `x∈B` 出发，现有或经成熟方案获得的恢复控制器，在原 50 Hz 条件下真实到达合格站立。B 不是一个直立躯干图片，也不是一个 FK crouch 点。应由已成功低位起身轨迹的完整状态及其小范围扰动回放得到；脚位、脚方向、关节速度、躯干角动量和驱动余量都在定义内。有些可行入口需要动量，不应强行要求先静止保持再起身。

当前 80 mm 低位起身与 100 mm FK 点只能作为 B 的种子；四方向倒地到 B 的路径尚未被证明。用 B 的真实 tail 成功判断候选终点，比只优化 COM/直立度或姿态距离更直接。

| 倒地方向 | 可以搜索的恢复内转换；全部属于待验证候选 |
| --- | --- |
| 前倒 | 同时比较直接收脚路径和“左/右侧身→半仰卧或其他可用承托→脚重新定向→B”。不预先指定仰卧一定更优。 |
| 后倒 | 比较身体摇动、重置脚朝向再向 B 转移；若 torso↔shin 卡住，允许先转侧身解除几何冲突。当前后倒低 COM 不能仅靠继续竖直躯干处理。 |
| 左/右侧倒 | 分别搜索距实际 B 较近的前/后方向；必要时卸载一只脚、重定向脚底再转移负载。两侧轴符号与约束不相同，不能简单把动作数值取反称为镜像技能。 |

颈部和头部可以按原驱动参与姿态与惯性调整；任何新增承托设想都须核对原几何、允许接触和机构，不把嘴假定成人形支撑臂。搜索后仍需连续执行整条路径，不在中途设置 `qpos`、清速度、重置驱动或跳到 B。

## 后续最小实验所需输入

| 必需输入 | 为什么本轮现有材料不足以替代 |
| --- | --- |
| 冻结模型/合同与既有原一步驱动接口；轴序、18 动作映射、限幅与完整状态保存恢复 | 外部项目的原始 `ctrl`/PD 接口不同；复制 qpos/qvel 不包含全部驱动状态。需要同状态下一步的观测、目标和实际力矩一致性核验。 |
| 四方向真实倒地的完整积分状态，加原驱动/延迟/热/功率状态 | 截图和静态接触 JSON 只能定位症状，不能用于准确分支。必须保留触地前后的速度。 |
| 成功低位起身/站立的确定性 ONNX、完整 50 Hz 状态与动作记录 | 用来测 B；已知局部能力不能自动推导到倒地。需记录真正接管后的成功和失败。 |
| 搜索轨迹：65 Actor 观测、18 原动作、完整 q/qvel、实际力矩/功率/热、动态接触与力 | 私有数据用于离线教师、审查和 Critic，不添加部署输入。仅记录“姿态关键帧”不足以训练或核验。 |
| 同本体跟踪数据：mjlab NPZ 所需数组、关节/刚体顺序、四元数约定、每帧 0.02 s | 只从原物理成功轨迹生成；人体 mocap、FK 插值和放宽合同的轨迹都不能直接充当合格示例。被动坐标与嘴机构继续由原动力学决定。 |

## 有界试验与退出门槛（建议，尚未执行）

1. **先核 branch 与 B。** 完整状态分支后，用同一个 18 动作核下一 Tick 观测、目标、实际力矩与状态；不能减少原约束或改步长。用已有低位起身记录测局部接管集合。若连 B 的 tail 都不可靠，停止“到 B”的搜索，先解决成熟恢复控制的局部接管。
2. **固定一个小规模采样配置。** 可以从 0.8–1.6 s horizon、3–5 样条 knots、左右两个模式开始；例如每方向总计 128 样本×4 更新×60 ticks，四方向为 122,880 次积分，另计 tail。样本在左右模式间分配，不额外乘一轮参数网格。建议首轮上限为 200,000 次真实积分或 30 分钟 CPU，先到者停止；实际吞吐未测，不承诺完成时间或在线 50 Hz 可用性。
3. **筛选优先级固定。** 原合同和动态接触门槛优先；在通过候选中比较 B 接管成功与可行转移进展，允许暂时降低高度/直立度。保存最优实际样本，也独立验证任何 CEM/MPPI 平均后的控制。若短 horizon 没有跨阶段进展，记录阻塞的接触/脚位/力矩原因；不能靠持续加采样量或调奖励网格掩盖。
4. **先证明完整路径，再考虑训练。** 每方向至少一条原始倒地→内部调整→B→合格站立的连续独立回放，才能声称该类有存在性证据；缺类明确缺类。再用每方向至少 10 个未参与搜索的扰动状态测试，建议每类至少 9/10 且所有成功轨迹均满足原约束，作为收集教学数据的开发门槛。它不是正式恢复验收。连续两次受控尝试无新可行边或预算耗尽即收口，保留失败证据，不推出硬件不可能。
5. **只在有真实成功数据后蒸馏/跟踪。** 优先现有 mjlab/RSL 跟踪组织或 HumanUP 的两阶段组织，继续 65→18；检查相近的 65 观测是否对应互相冲突的教师动作。若必须依赖规划 phase/分支输入才能区分动作，停止并报告合同限制，不能隐式加入 phase/history/token。只有动作库规模及多模式复杂度确有需要时，才评价 VQ/连续技能预训练。
6. **正式验收独立执行。** 原计划为四方向各 50 例、10 s 起身、直立度≥0.95、COM≥名义站立 85%、保持 3 s 再移动 1 m，整体≥95%、每类≥90%；真实扰动另测。独立进程、确定性 ONNX、`auto_reset=false`，训练/开发/验收分开。动态 5 mm 门槛与迁移评估继续成立。低位起身、单条规划成功或教师轨迹都不能授予真实恢复与完整移动资格。

因此推荐的下一决策点很小：**能否在原条件下找到至少一条“真实倒地→可接管入口”的边？** 能，则有数据决定要不要跟踪与小图搜索；不能，则准确报告当前预算覆盖的接触障碍、局部控制不足或 horizon 限制，主线继续成熟实现核对，不把研究转成新的框架项目。

## 本轮可复用代码的固定版本与成本排序

| 项目 | 本轮核对版本 | 优先级 |
| --- | --- | --- |
| mjlab tracking | 官方 `v1.3.0`，与当前主线版本一致 | 优先借 MotionLoader 与跟踪组织；默认观测、动作、时间配置须逐项审查。 |
| MuJoCo MPC | `ff572a21e7c2bf9fda62e1862a758da7e9a8719b`，Apache-2.0 | 优先借 CPU 样条采样/CEM 算法组织；rollout 必须走 Goose 原一步接口。 |
| HumanUP | `7516e0f27e6f4d1e7365cf64ea577a78247bd8cb`，Apache-2.0 | 优先借发现→受约束跟踪及轨迹日志流程；不用放宽碰撞作为合格证据。 |
| HoST | `70bb580949a336a920833700e4b5dc3bf7fe87ce`，MIT | 主线成熟恢复审查；多 Critic/课程是实现成本，辅助拉力最终必须为零。 |
| Hydrax | `bd53d43d414e19557d38ee6c72613bf2e39fe834`，MIT | 已有起身采样示例与实验性 Warp；JAX/MJX 及完整驱动接线增加适配成本，暂非最小入口。 |
| NCP / PLT / ASE | 作者公开代码，本文来源链接 | 方法可复用，预训练与本体动作数据成本高；当前不建议作为第一步。 |
| LAAS 多接触工具 | 作者公开代码，本文来源链接 | 高信任结构参考；依赖与全身模型适配较重，当前用小图即可。 |

外部只读 clone、原文 PDF/text 和两份 mjlab 版本源码位于 [本轮临时资料目录](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_recovery_planning_research_001)。本轮阅读核对与源码数学分析不等于复现实验；**新增物理积分 0、GPU 使用 0、合格 Goose 恢复路径 0**。

## 主实施随后完成的有界验证

本节在研究交付后补记；前文“研究阶段0积分”不含此后的实际试验。复用MJPC作者Policy样条组织并接回原Goose驱动，在原四方向倒地冷出生测试1.28秒与3.2秒两个时域，总计160,022次真实CPU积分、GPU0。完整分支状态的下一Tick状态、观察、力矩与目标差均0；两批各24个候选接管tail没有一条完成起身保持和继续移动。端点启发式是本地最近低位入口距离，不是已复现的作者起身任务。左右模式分开，但有限样条/样本及重复端点仍限制覆盖。按两次无成功的退出规则，收口这套具体方案，不推出姿态规划无效或Goose不可能恢复。

实际收据：[1.28秒试验](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_recovery_sampling_001/receipt.json)、[3.2秒试验](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_recovery_sampling_002/receipt.json)。已知4条低位起身真实记录另导出736帧，并由同版本mjlab MotionLoader实际读取；它们仍是开发参考，动态5mm条件未过，不是全倒地演示或已启动的跟踪训练。[数据接线收据](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_recovery_motion_corpus_001/receipt.json)。

主实施又用独立进程连续复现四个具名best计划：真实倒地出生→3.2秒样条控制→原起身ONNX，中途没有状态/驱动重置，新增3,240CPU积分、2,600ONNX调用，端点与tail状态/力矩差0，四方向仍全部失败。这验证了这几条失败路径的连续性，没有再次搜索或授予可达性。[连续复现收据](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_recovery_sampling_replay_001/receipt.json)、[实际失败录像](/home/ethan/ProjectBackups/2026-10-05/Sai_Lab/goose_recovery_sampling_replay_001/recovery_sampling_failure_sequences.mp4)。

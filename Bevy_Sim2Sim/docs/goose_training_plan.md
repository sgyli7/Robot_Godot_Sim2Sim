# Goose V0.1｜Move 训练计划

版本：2026-10-08。独立分支：`codex/goose_move`。

**当前状态：取得部分源端能力，完整任务未完成。** 本聊天只负责Move，不推进起身或拾物，不联系其他聊天，不派Agent。目标保持：走路0.4m/s、Shift跑步0.7m/s，前后／侧移／双向原地转向／移动中转向／启停及任意切换，操作与MicroDuck三轴组合对齐。不能以请求速度、单段录像或开发小集代替这些资格。最新196回归确认原176的16条程序全部保存字段逐位一致，仍为12/16通过；188增益候选独立保存，不替换143/176。

代码工作树为`/home/ethan/Projects/TempWorktree/Sai_Lab/goose_move/Bevy_Sim2Sim`。临时模型、脚本、录像、收据及构建均落[Move备份目录](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001)。原工作树和恢复分支不改写。

## 当前具名入口与实际能力

[收尾清单](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/source_move_closeout_177/manifest.json)固定模型、11份外部模型资源、合同、Actor、控制器、冻结源码和收据SHA；[收尾报告](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/source_move_closeout_177/report.md)记录范围、失败和复现方式。禁止加载latest或覆盖旧基线。

| 候选 | 实测及退出依据 |
| --- | --- |
| 默认保留143 | 约0.14／0.20m/s与±0.3rad/s，13条开发程序11条通过；左弧线及完整串接未通过。停止选择只依赖真实用户命令历史 |
| 较快开发176 | 约0.14／0.27m/s；37秒慢→快→左→右→停无重置通过，左右转约+0.334／−0.343rad/s。16条开发程序12条通过；不是默认替换或完整MD资格 |
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

## 训练路线与已收口对照

主线仍为统一MJCF、原生MuJoCo、mjlab／Warp／RSL-RL，然后独立Rapier／Bevy迁移。MD016的输入、合同和已有奖励源码已核对；学习其工作流和任务定义，不把61→14权重装到Goose65→18，也不声称复现无法追溯的原完整训练。

实际训练停止覆盖不足已定位：155的8个保存世界97,976Tick只出现7次移动转零、保持中位4Tick；恢复成熟2–4秒命令重采样后，162的92,160Tick有51次、中位75Tick。覆盖改善，但独立行为未通过，不覆盖148。165三轴课程完成11,796,480GPU积分／480PPO／9,600Adam，655.65秒；五检查点75条程序共47,329CPU积分，未获得合格新方向，[收口结论](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/continuous_axes_independent_eval_166/closure.json)。155未改善续训、147／158左弧线、151提步频及165同条件路线均已收口，不重复盲加长。

[累计资源账本](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/completed_move_ppo_resource_ledger.json)记录37个完成的短主批、537,919,488GPU积分、22,848PPO、456,960Adam及约5.93小时批次计时；这是多个策略分支，不是单个已收敛训练或仅本轮十小时的统计。155中断批另列，链路、CPU、搜索与录像不混计。

## 固定合同与后续门槛

当前本体为`goose_task_proxy_11_discrete_mjlab160_v1`：21机器人刚体、18主动轴与2被动轴、11碰撞叶、10.430690821kg；65Actor／69Critic／18动作，真实动作历史和主动轴顺序保持。电脑能力训练使用刚性足底，不追加软底课程。

- 物理、驱动、策略50Hz，decimation=1；每Tick一次真实20ms积分、一次所选Actor调用，无隐藏子步。控制状态需随暂停保存。
- 开发物理门槛固定：地面穿透≤5mm、自碰撞≤0.1mm、限位误差≤0.1mrad、直立度≥0.95。速度平均比0.8–1.2，组合动作同时满足比例≥0.6；停止1秒速度≤0.04m/s，随后3秒漂移≤5cm。不改变门槛凑通过。
- 独立评估采用确定性ONNX、关闭自动重置，保存全部失败。源端、零样本目标、目标微调和游戏成绩分别报告。
- 完整M0-S、M0-T仍未授予资格；当前训练是有界开发pilot。每个新批先做小链路和实际吞吐检查，固定版本／问题／指标；未改善两次回到模型或成熟工作流，不追加同条件长训。
- 后续剩余重点为后退／侧移、两方向弧线、长程停止及任意热切换，再验收原0.4／0.7目标、正式独立案例和游戏输入。没有新批在等待另一Agent或实体制造。

旧逐批历史完整归档于[收尾前计划快照](/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_move_continuous_001/source_move_closeout_177/plan_before_closeout.md)，仅为证据，不是当前派工。此文件继续作为唯一现行计划入口。

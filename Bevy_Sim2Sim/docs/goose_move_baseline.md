# Goose Move023｜快慢速移动基线

2026-10-09。冻结标签：`goose_move023_source200_20261009`；实现分支：[codex/goose_move](https://github.com/sgyli7/Sai_Lab/tree/codex/goose_move/Bevy_Sim2Sim)。这个基线用于保留已取得的源端移动能力，后续转弯／侧移候选须通过速度与停止回归才能替换它。

## 连续实录

![Goose站立、慢走、快跑、慢走、停止的33秒连续实录](../assets/game/arts/game_play/textures/goose_move023_walk_run_stop.gif)

[完整33秒原片](../assets/game/arts/game_play/videos/goose_move023_continuous.mp4)来自实际保存物理轨迹，无重置、回滚、拼接、Teacher或换Actor。顺序为站立5秒→慢走8秒→快跑8秒→慢走8秒→停止4秒；该次慢走约 **0.413m/s**、快跑约 **0.729m/s**。GIF仅降低尺寸／采样率，保持原速。

## 已取得的证据与范围

| 项目 | 实际结果 |
| --- | --- |
| 独立开发集慢走0.4指令 | 平均0.41362m/s，8/8 |
| 快跑0.7指令 | 平均0.72117m/s，8/8；60秒快段平均0.72410m/s |
| 48条源端开发程序 | 物理48/48、完整行为46/48、停止48/48；两个低速warm27偏航失败 |
| 快慢热切换 | 8/8；同一Actor并保留身体、动作历史、驱动热状态和相位 |
| 真冷GPU程序 | 物理3/4、完整2/4；直接快→停在13.10秒自碰越界，61秒站立漂移5.474cm未过 |
| 独立CPU实际ONNX | 两条原冷案例物理通过；慢走0.41119、快跑0.72442m/s，4段停止通过 |
| Rapier／Bevy50目标接收 | 未通过：027首Tick嘴铰链越限；未执行目标快跑程序 |

**转弯、侧移、后退及任意组合尚未在023完成验收。** 旧低速转向基线保留，不能据其录像宣称023已具有全向能力。正式200独立案例、Shift／失焦／暂停输入和完整Bevy资格仍待完成。源端和目标引擎成绩独立记录。

## 冻结身份

- 自身Actor **65→18**，不使用MicroDuck权重。21机器人刚体、18主动轴、2被动坐标、11碰撞叶，约10.43069kg；电脑训练采用刚性足底。
- MuJoCo3.15原生DISCRETE＋mjlab1.6／MuJoCo Warp／RSL-RL PPO。此具名源候选物理 **200Hz**、策略与数字驱动 **50Hz**：一个20ms策略Tick含四次真实5ms积分，目标限速／历史／热／奖励和PPO时间语义仍每20ms提交。运行相位 **1.5Hz** 明确由 `cadence_runtime005.py` 覆盖原合同1.2Hz；原合同文件不改写。
- Bevy默认合同仍50Hz、每Tick一次20ms真实积分；200Hz源端成绩不自动授予其资格。
- 023续训从具名020std015完整Actor／Critic／Gaussian／归一化／Adam恢复，512世界×512PPO更新；实际25,165,824 GPU积分、6,291,456控制决策、10,240 Adam更新、约499秒。训练完成后另行独立评估，非自动晋升。

| 制品 | SHA256 |
| --- | --- |
| 自身ONNX | `7ad792bfe5a0587251ed5a5d789d2915b7f30849c0d3ab33992ae77ccc4f7186` |
| 完整学习状态PT | `612c17eceacf91d0141f0ad8fa77ac1811df7c7e4323af08cec129cb08efee4b` |
| 源MJCF | `c2f0f55e3048691fd6a1618c6990fba596880f6711f740eca17d6f81765878cb` |
| 原控制合同 | `8b17aa7d7316c3f36d6949fe79a6392cc6c216f0ea7aa4b82c100c3337116794` |
| 33秒完整MP4 | `0d62c3c1d463121ac391c886117e29548ae3794b7eb5aed4de8c69c2f2962cda` |
| 023的470文件冻结清单 | `60f746d83abd58dce5554a85a2db6b317181926d11c5d26c75258df22226d022` |

源权重、模型、训练执行脚本与原始收据按项目规范保存在 `ProjectBackups/2026-10-09/Sai_Lab/goose_move_200hz_fast_001/`；另有同日 `goose_move_publication_001/goose_move023_source200_baseline.tar.gz` 与逐文件哈希。源码仓库保留实现、具名身份与展示录像，不自动下载或选择latest权重；克隆仓库不等于拥有完整训练环境／权重。只依赖此页的模型哈希不能运行训练。

## 下一批

按照用户优先级推进：**移动中左右转→原地左右转→移动中平移→纯平移**。028先从023完整学习状态加入行进左右弧线命令，冻结模型／控制／相位及既有成熟奖励；保留0.4／0.7直行、快慢切换和停止回归。初级课程±0.3rad/s是开发台阶，最终原转向目标±0.6rad/s保持。详细计划见 [Move训练计划](https://github.com/sgyli7/Sai_Lab/blob/codex/goose_move/Bevy_Sim2Sim/docs/goose_training_plan.md)。

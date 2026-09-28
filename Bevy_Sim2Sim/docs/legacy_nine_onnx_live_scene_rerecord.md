# 九个原始 ONNX 在 v10B 科学站上的 60/60 复录

日期：2026-09-28。本页记录 **v10B 场景**的一次 60/60 零样本、P-only 诊断视频复录。v18AJ 场景资产晋升后，本页视频的场景身份已过期，不能当作新 live 场景录像；[当前 v18AJ 复录](legacy_nine_onnx_aj_v7_rerecord.md)已完成。它不是 BAM 物理、站体接触或九项技能验收。原始 ONNX 仍来自 200 Hz 物理 / 50 Hz 策略训练；本次没有重新训练。

先前的九段视频绑定旧场景 SHA，不能作为当前 live 场景的媒体。复录前把 live `assets` 冻结到 `.scratch/legacy_nine_native_probe/live_55b73c8a_retake/assets/`，并核对了模型、初始 qpos、外观、九个 ONNX 和 ONNX Runtime 1.30.0 ARM64 库的既有哈希。九次物理运行与九次 GPU 捕获均读取这份相同的冻结资产。复录结束时，冻结资产与 live 的 GLB、场景清单、布局、两个 shader 和字体哈希仍完全相同。

| 场景文件 | SHA-256 |
| --- | --- |
| `science_station.glb` | `f240c9e76f15eaf7724182fcaf2e7f00cd3e9a3814d5a959638374d3f3b882ac` |
| `science_station.ron` | `5cf06e012fc323652928a52e113cc323e83f33c564f8647b5ec53010459f3388` |
| `science_station_layout.ron` | `55b73c8a62b7f85e605daf03394532e49f5742dba1f3c045e03adc5be1a6464d` |

每段独立建立一个 Rapier 世界，导入两足或轮足完整本体及 2,497 个科学站静态碰撞体；每个 60 Hz tick 用原生 ORT CPU 对对应原始 ONNX 推理一次、写入力矩并在同一个世界积分一次，然后读取完整机器人姿态。九段共 **2,400 次 ORT 推理、2,400 次 Rapier 积分**。渲染器按完成的连续姿态逐帧截取真实 1920×1080 GPU 画面，30 fps 共 **1,209 帧**。带模型名和诊断边界标签的合辑为 **40.30 秒**；编码后重新解码验证了 1,209 帧。

| 原始 ONNX | 积分 / 推理 | 视频帧 | 实际可见行为 |
| --- | ---: | ---: | --- |
| `alpha_walking` | 360 / 360 | 181 | 起初有抬脚和位移，随后侧倾倒地；未形成稳定步行。 |
| `alpha_stand` | 240 / 240 | 121 | 起初直立，约第 97 步明显倾斜，末段仍斜靠地面；不是合格站立。 |
| `alpha_sitstand` | 480 / 480 | 241 | 身体下折，约第 318 步后明显倾倒；未见完成的坐下再站起。 |
| `alpha_ground_pick` | 240 / 240 | 121 | 身体探低后失稳，末段头身靠近地面；没有实体抓取物交互。 |
| `ball_kick_left` | 180 / 180 | 91 | 腿部摆动后迅速侧翻；场中没有实体球，不能验证踢球。 |
| `ball_kick_right` | 180 / 180 | 91 | 腿部动作后迅速倒地；场中没有实体球，不能验证踢球。 |
| `roller` | 360 / 360 | 181 | 轮足本体移动约半米后翻倒；不是稳定滚行。 |
| `roller_crouch` | 180 / 180 | 91 | 身体降低、末段仍大致直立；尚未验收下蹲高度与接触。 |
| `roulade` | 180 / 180 | 91 | 可见翻滚和位移；翻滚本来要求大角度旋转，单凭直立指标不能判定技能成功。 |

表中“约第 N 步”只表示根体局部上方向与世界上方向点积首次低于 0.5 的诊断时刻，不是技能判据。根体水平净位移来自姿态序列；可见画面另核查了九段代表帧。源报告的 `passed=true` **只表示请求的推理和积分步数完成**，所有报告的 `skill_qualified=false`。

控制法则仍是 `τ = clamp(0.55 × (target − q), ±0.6405236)`，`Kd=0`。BAM 上一步外载、延迟队列、动态摩擦和目标物理门禁尚未闭合。源自碰撞过滤与站体接触等价未验收；20 个动态道具碰撞体未导入，因此画面里与箱子等静态显示物重叠也不证明物理交互。

本地媒体和原始证据均在 `.scratch/legacy_nine_native_probe/live_55b73c8a_retake/`，不入 Git：

- 合辑：`nine_original_onnx_live_scene_60_60_p_only_diagnostic.mp4`，SHA-256 `66511a76f35ea715cf4bd96ef318f0effd738c68f08ca299595c163a44c2ab9f`。
- 总收据：`video_delivery_manifest.json`；含当前场景/渲染资产哈希、九个原 ONNX 哈希、报告与视频 SHA、逐段步数和帧数。
- 每段物理报告：`cpu/<model>.json`；每段 GPU 捕获、PNG 序列和视频：`gpu/<model>_frames/`、`gpu/<model>.mp4`。
- 复现脚本与批次收据：`record_cpu.py`、`record_gpu.py`、`deliver_video.py`、`cpu_batch_receipt.json`、`gpu_batch_receipt.json`。

**场景过期规则：** 只要 live 科学站任一相关资产在地编晋升后改变，本次视频就不再代表新的 live 场景；须以新资产身份重新执行九次物理报告和 GPU 捕获，不能只换背景重编码。

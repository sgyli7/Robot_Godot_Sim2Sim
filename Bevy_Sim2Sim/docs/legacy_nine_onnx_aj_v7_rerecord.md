# 九个原始 ONNX 在 v18AJ 科学站的 60/60 诊断复录

日期：2026-09-28。本页记录 **v18AJ live 科学站**的一次完整复录。历史 v10B 录像见 `legacy_nine_onnx_live_scene_rerecord.md`；其场景身份已经过期。这里的九个 ONNX 均为原始 200 Hz 物理 / 50 Hz 策略模型，本次未经重训，按实验性 60 Hz 物理 / 60 Hz 推理运行。每段独立建立一个 Rapier 世界，每个 tick 执行一次原生 ONNX Runtime CPU 推理和一次真实物理积分，并从连续姿态逐帧捕获 Bevy GPU 画面。控制器是简化的 P-only 诊断律，并非 BAM。

正式 v18AJ 场景的六机位视觉、零步同世界和局部接触门禁先于本次 GPU 录制通过。录制前冻结完整 10 文件 `assets` 树；每段捕获后以及交付时复核 live 与冻结资产哈希。主要身份如下，其余字体、辅助布局和 ink shader 哈希见本地 `video_delivery_manifest.json`。

| Live 资产 | SHA-256 |
| --- | --- |
| `science_station.glb` | `69da91fad18d2eadb2f46806f7ee963de090f6089f752f3865399dbf068ed3a2` |
| `science_station.ron` | `e0b0fe1aad9872968a4961ce27cac7b1b0f0d49cded49a65f64701df413715c7` |
| `science_station_layout.ron` | `b166c951511b51e7a2d2060f6934fe3469f77da11d15d30ecb41a5c7a2414988` |
| `station_enamel.wgsl` | `0c049e8bfda7458232cd28e84641adb2ea1837cfdf9694b732de731635ac6f0c` |

模型定义、两种初始 qpos、两套机器人外观、九个原始 ONNX 和本机 ARM64 ONNX Runtime 1.30.0 动态库也都在运行前按 SHA-256 校验。诊断程序二进制 SHA 为 `157b3b436584d3953f9c0d556d145839982b87c361bf711c4da696269bdc35b3`，视频捕获二进制 SHA 为 `e21d6e75b67ae74945f6d4b266fe0dedaef388fece7429634a684ed53fd54d4b`。九段分别导入 2,553 个科学站静态碰撞体；共完成 **2,400 次 ORT 推理、2,400 次 Rapier 积分**。30 fps GPU 捕获共 **1,209 帧**，合辑 **40.30 秒**；合辑编码后重新解码确认 1,209 帧。

| 原始 ONNX | 物理步 / 推理 | 视频帧 | 逐段画面观察 |
| --- | ---: | ---: | --- |
| `alpha_walking` | 360 / 360 | 181 | 开始站立，短暂迈动后向后翻倒，随后在地面翻转、位移；未形成稳定步行。 |
| `alpha_stand` | 240 / 240 | 121 | 起初直立，随后明显侧倾并倒地，末帧仍躺倒；不是合格站立。 |
| `alpha_sitstand` | 480 / 480 | 241 | 前半段身体降低、双腿外展，后半段失稳倒地；未见完整坐下再站起。 |
| `alpha_ground_pick` | 240 / 240 | 121 | 向前探低并靠近场中显示箱体，后段身体倾倒；没有实体抓取物交互。 |
| `ball_kick_left` | 180 / 180 | 91 | 腿部摆动后很快侧翻并在地面转动；场中没有实体球。 |
| `ball_kick_right` | 180 / 180 | 91 | 腿部摆动后很快倒地并转动；场中没有实体球。 |
| `roller` | 360 / 360 | 181 | 轮足本体向前位移约 1.2 m，期间曾接近直立，末段翻倒；不是稳定滚行。 |
| `roller_crouch` | 180 / 180 | 91 | 身体明显降低，后段重新接近直立；未验收下蹲高度或接触质量。 |
| `roulade` | 180 / 180 | 91 | 可见连续翻滚和位移，末段尝试抬起；仅凭画面不能判定翻滚技能成功。 |

画面观察依据是每段 0%、25%、50%、75%、100% 的实际 GPU PNG，以及物理姿态序列；本地五点总览为 `review_contact_sheet.png`。源报告 `passed=true` 仅表示所请求的推理与积分步数完成。部分场景道具仅显示，20 个动态道具碰撞体没有导入；箱体、球等目标物的视觉靠近或重叠不构成物理交互证据。

诊断控制为 `τ = clamp(0.55 × (target − q), ±0.6405236)`、`Kd=0`。BAM 上一步外载、延迟队列、动态摩擦、源自碰撞过滤和完整站体接触等价尚未验收。因此本批视频 **不构成 BAM、接触或九项技能验收**。

本地未入 Git 的媒体和原始证据保存在 `.scratch/legacy_nine_native_probe/live_aj_v7_retake/`：

- 合辑 `nine_original_onnx_aj_v7_60_60_p_only_diagnostic.mp4`，SHA-256 `92137ee03a2e206e28d33020cc3275d4538ba34218c2d2e35fe305fdd2c1a514`。
- 总收据 `video_delivery_manifest.json`，SHA-256 `ab6dc653dfb86c95fbddb4d3dd1d2341ad732ba4c89053bb8fb63f2da9729d6c`；记录冻结/live 资产、九个 ONNX、逐段物理报告与视频哈希及帧数。
- `cpu/<name>.json` 为逐段物理、推理、控制量和连续姿态；`gpu/<name>_frames/` 与 `gpu/<name>.mp4` 为对应 GPU 帧和视频。`cpu_batch_receipt.json`、`gpu_batch_receipt.json` 及 `record_cpu.py`、`record_gpu.py`、`deliver_video.py` 记录运行过程。

若 live 科学站任一相关资产再次改变，本批视频随即成为历史场景记录，必须在新身份下重新跑物理、推理和 GPU 捕获。

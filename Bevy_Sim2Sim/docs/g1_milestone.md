# G1 阶段运行与录像

2026-10-04 将当前 G1 开发成果按用户要求合入主干，并在 Bevy 主页展示真实运行录像。完整任务目标继续保留；本页中的诊断成功不授予正式任务能力。

## 已录制的真实运行

| 项目 | 科学站取放 | 源任务近似场景搬箱 |
| --- | --- | --- |
| URI 重录 episode | 20719 | 20720 |
| 原生积分／身体控制更新 | 1,046／1,046 | 2,134／2,134 |
| 新任务策略调用 | N1.7：1 次，原配 40 帧 | N1.6：4 次，原配每块 50 帧 |
| 本片 Qwen 调用 | 0 | 0 |
| 身体与执行器 | 原配 AGILE／native force based | 原配 Homie v2／native force based |
| 明确启用的诊断配置 | 预测关节限制，16 PGS | 4 PGS |
| 严格独立放置检查 | 6.02 秒 | 2.52 秒 |
| 活跃仿真／墙钟时间比 | 0.998876 | 0.999395 |
| 待处理 Tick／控制误期 | 0／5 | 0／0 |
| 环境 | 科学站的 2,553 个真实静态碰撞体，同源渲染 | 原配任务近似仓储场景；科学站持箱行走另见下文 |

两轮都保持 50 Hz、每 Tick 仅一次 20 ms 积分，没有隐藏时间子步、物体附着约束、动画驱动物理或验收真值输入。独立检查使用原始物体全部碰撞顶点、当前支撑接触、手分离、站立和冻结速度限制；不将检查结果作为控制观察。模型和图像边界的显式暂停仍然存在。

搬箱完整运行的箱子水平位移为 1.990227 m（三维位移 2.003680 m），机器人净水平位移为 1.322950 m。初始抓取使用原配 VLA；后续运动由公开地图、实际 RGB 标记定位、自身状态和传统几何控制执行。本片没有再次调用 Qwen。另一次 episode 20618 使用当前新 RGB 验证目标集合，让本地 Qwen 选择搬运，再执行同一路径并观察新图像；两次新 Qwen 请求均及时收到。录制轮与该轮的 2,134 个身体／物体物理步骤完全一致，区别是 episode 和模型边界等待时间。

URI 重录的全部 1,046／2,134 个身体和物体物理步骤分别与原 episode 20619／20620 一致；模型保存的全部 observation 和 action 数组也一致。科学站六张传感器图和搬箱四次 VLA 新图逐字节一致；搬箱部分后续诊断 PNG 有 1–3 个像素、单通道一个 8-bit 级别的差异，严格全 PNG 字节相等检查未通过，保留该结果与差异报告。展示层没有进入模型输入，详见 [URI 渲染与验证](g1_uri_presentation.md)。

## 主页媒体与身份

两段 GIF 均重新录自实际 Bevy 主窗口，采用官方银灰 G1 和 URI 环境着色，连续 **12.00 秒**墙钟原速、150 帧（12.5 fps）。科学站片段取原片 7–19 秒、880×720；搬箱片段取原片 53.5–65.5 秒、880×660。没有时长缩放，裁切和采样仅用于媒体导出。运行窗口保持 1920×1080、8×MSAA，传感器设置保持原样；公开片段排除了首次绘制之前的桌面占位画面。

两次录制基于 `4125e174b57ca808b66dbe915b8599b08fae71df` 上的 URI 展示实现，录制时工作树尚未提交。最终实现先提交为 `067645d`，然后合入主干的科学站抓箱进展；原始执行收据记录每次二进制、配置和模型身份。

| 身份 | 科学站取放 | 源场景搬箱 |
| --- | --- | --- |
| 执行二进制 SHA256 | `363babd66fa9d4619b6b44b62c8e15d8e98a292f2c0a28f97cb8a4714a436662` | `18c2892a9754e5b71901c56b0c0be54e991ca71c0a0d2ba8c0b7ead6ba304960` |
| 完整原片 SHA256 | `7f7ce319239561d550122c56df727dfe11e9422b139e982a0cbbb3cce4872397` | `d295008d8a517e151f28d03daf2298d827fc92b318d3c180d294b97cec330cfb` |
| 主页 GIF SHA256 | `664ad78e83e91aafd633b576c7bca7f32973e04fe0490a87cdc20d90148a53b0` | `b910308ef407c124d5767a1072a8b1d978efffbd207f3f883120b88db19fb58c` |

本体、模型、原配预处理与许可见 [G1 制品清单](g1_artifact_inventory.md)。任务策略分别为 T1 revision `7f78bebf1a90131e7304beacfcd47eb27bad16ab` 和 T2 revision `dfe74af855007f26093f362cd2d7a2f404b64b93`，不混接合同。G1 灰色主要来自中性银灰底色，蓝色只用于环境反光；参考 [官方 G1](https://www.unitree.com/g1/)。

新原片、逐 Tick 日志、模型捕获、配置、诊断失败尝试和差异审计位于 `/home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_uri_presentation_delivery_001/`。先前源版证据仍保留于 `g1_milestone_delivery_001/`。权重、完整原片和临时实验不进入普通 Git；GIF 是用户要求发布的主页媒体。

## 本机准备与运行

在 `Bevy_Sim2Sim` 中运行。需要已准备的 `/home/ethan/models/unitree_g1/` 本体／权重缓存、两套隔离 Python 环境、原生 ORT、可用 X11/Vulkan 桌面和冻结配置引用的校准／模板文件。克隆仓库不会下载这些外部制品。配置拒收 SHA256 不匹配的内容；详细准备入口见 [T1](g1_t1_source_task.md)、[T2](g1_t2_source_task.md)、[本体与控制](g1_agile_native_runner.md)。

```bash
cargo build --locked --bin bevy_sim2sim --features \
  dev_tools,dev_tools_minigame/g1_constraint_diagnostic,dev_tools_minigame/g1_source_lighting
```

科学站录制轮使用如下服务和应用命令；模型服务前台运行，另一终端启动应用。服务监听本机 5567，不应与其他拥有者共享该端口。冻结配置是外部证据中 `recordings/station_final_gray/config.json`。

```bash
/home/ethan/models/unitree_g1/envs/policy_onnx_cu13/bin/python \
  crates/dev_tools/python/scripts/unitree_g1_static_server.py \
  --receipt /home/ethan/Projects/Sai_Lab/.scratch/unitree_g1/policy/static_apple_files.json \
  --device cuda --port 5567 --seed-offset 0

taskset --cpu-list 5-9,15-19 ./target/debug/bevy_sim2sim --scene g1_static_observed_place_diagnostic --robot g1 \
  --g1-config /home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_uri_presentation_delivery_001/recordings/station_final_gray/config.json \
  --g1-ticks 1100 --output .scratch/g1_station_new_run
```

搬箱服务和应用使用另一套 N1.6 环境；两套重模型串行运行。冻结配置为外部证据中 `recordings/mobile_complete/config.json`。

```bash
/home/ethan/models/unitree_g1/envs/policy_gr00t_n16_cu13/bin/python \
  crates/dev_tools/python/scripts/unitree_g1_mobile_server.py \
  --gr00t-source /home/ethan/Projects/Sai_Lab/upstream/unitree_g1/isaac_gr00t_n16 \
  --model-root /home/ethan/models/unitree_g1/mobile_box/dfe74af855007f26093f362cd2d7a2f404b64b93 \
  --receipt /home/ethan/Projects/Sai_Lab/.scratch/unitree_g1/policy/mobile_box_files.json \
  --runtime-env /home/ethan/models/unitree_g1/envs/policy_gr00t_n16_cu13 \
  --port 5568 --seed 42

taskset --cpu-list 5-9,15-19 ./target/debug/bevy_sim2sim --scene g1_mobile_auxiliary_release_diagnostic --robot g1 \
  --g1-config /home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_uri_presentation_delivery_001/recordings/mobile_complete/config.json \
  --g1-ticks 3150 --output .scratch/g1_mobile_new_run
```

重录配置仅调整新 episode、独占服务端口和显示限帧 `diagnostic_render_hz=60`；进程使用上述 CPU 亲和性避开并发主机负载。物理仍为 50 Hz，原控制、超时、相机和 SHA256 绑定资产保持。部分冻结校准文件含迁移前的绝对 PNG 路径，运行前须恢复同一 PNG 的兼容路径；不能通过改校准 JSON 来绕过哈希校验。

应用达到有限预算或反馈阶段会自动退出。手动停止使用应用窗口关闭／对应终端 Ctrl+C；之后对自己前台启动的模型服务 Ctrl+C。录制驱动另保存于外部证据，负责识别自己应用的 `_NET_WM_PID`、只录对应窗口、有限超时及回收自己启动的子进程。不会结束其他实验。

独立检查新运行：

```bash
/home/ethan/models/unitree_g1/envs/policy_gr00t_n16_cu13/bin/python \
  crates/dev_tools/python/scripts/unitree_g1_static_placement_audit.py \
  --definition /home/ethan/models/unitree_g1/task_assets/20261001_frozen/native_task_objects_t1_60_diagnostic_v2.json \
  --trace .scratch/g1_station_new_run/owner_steps.jsonl --output .scratch/g1_station_audit.json

/home/ethan/models/unitree_g1/envs/policy_gr00t_n16_cu13/bin/python \
  crates/dev_tools/python/scripts/unitree_g1_mobile_placement_audit.py \
  --definition /home/ethan/models/unitree_g1/task_assets/20261001_frozen/native_task_objects_t1_60_diagnostic_v2.json \
  --trace .scratch/g1_mobile_new_run/owner_steps.jsonl --output .scratch/g1_mobile_audit.json
```

## 科学站 T2 抓箱迁移

同日原版 episode 20622 在原生科学站完成了单次抓箱诊断：四次真实本体相机图片和原配 N1.6 新推理、200 次单独 20 ms 积分及身体控制更新，末帧箱子中心抬高 0.143950 m。独立日志确认正接触来自左右手、箱子已脱离货架，最低站立余弦为 0.993949。末段有 42 Tick 的抬高／手接触样本；这不等于已验证长期持物或搬运。活跃仿真／墙钟比为 0.995166，控制误期与积压均为零，模型边界暂停继续存在。本轮 Qwen 调用为零，不能当作科学站自然语言闭环或正式 T2 成功。

前置预检执行 100 次单独 50 Hz 积分，机器人站立，箱子／容器位移分别约 0.09／0.47 毫米，并有货架／台车的正支撑冲量。新场景只保留原任务货架和台车的六个碰撞体及对应六个网格，源几何、质量和材质保留；省略原仓库其余几何和电钻。旧视觉导出遗漏的台车桌面已从同一冻结 USD 补回，其源材质明确设置纹理亮度为零和颜色偏移 0.11。其余 122 个原视觉网格、材质和 UV 经逐项比较一致。原版完整仓库入口仍拒绝该科学站场景。

本轮总世界为 58 bodies／2,613 colliders／52 multibody joint handles，含 2,553 个同源科学站碰撞体；原宽平面在零 Tick 时移除。原 Homie v2／N1.6、动作映射、增益、质量、步长和图像有效期均保持，4 PGS 是独立的开发诊断，未增加时间子步或正式执行能力。

原版源码为 `76707ce5266e147e83f4698126193ef2b01c53cf`，原始证据仍保留。主页现使用 URI 重录 episode **20724**：同样四次新图 N1.6 推理、200 次真实积分，全部物理步骤、五张传感器 PNG、四次 observation/action 数组与 episode 20622 一致。因此同一独立接触审计的 14.4 厘米抬升、双手接触和货架分离结果仍成立；没有延长运行来制造稳定持物结果。活跃仿真／墙钟比为 0.997559，控制误期和积压均为零。

URI 抓箱录制基于合并后的 `cc94f6f2ffaec1d0399737b65822b3489f40b26d` 加主窗口色调映射修复；执行二进制 SHA256 为 `f6ecad7cc38547263cf4408d9c5511ec727e0f708e85ca06f8492028e5329a65`，原片 SHA256 为 `3ba6c226fdf751f4bdf610e6da753867a5905266a36bd27dd68d898e03f94de8`。完整原片 13.16 秒，主页 GIF 取 6.12–13.16 秒已绘制的实际窗口，960×540、88 帧、**7.04 秒原速**，SHA256 为 `c1e2d3c1046252d3ee2492909f8d58dbcfe340c6729def82794edff9f3bfc3c2`。其余两段 12 秒 GIF 同样已重录。新收据、源版对照及实际二进制位于 `g1_uri_presentation_delivery_001/recordings/station_box_grasp_uri/` 和 `recordings/runtime/`。

新视觉导出 SHA256：`69129a51ae7fd53ddf9a00cce7170bc780e4d0f4fceff5c720cbe359db7eee9e`，仍绑定原背景物理 `0ecc6d502967d7cca82bd208b2338dedf86505c781bf5d47551b186fa73b86e6`。外部证据位于 `/home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_station_t2_fixture_milestone_001/`，包含冻结配置、独立审计、原速录像与源资产。URI 抓箱重录使用上述 N1.6 服务，将服务端口改为独占的 **5578**；对应冻结配置已经绑定该端口，然后运行：

```bash
taskset --cpu-list 5-9,15-19 ./target/debug/bevy_sim2sim --scene g1_camera_diagnostic --robot g1 \
  --g1-config /home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_uri_presentation_delivery_001/recordings/station_box_grasp_uri/config.json \
  --g1-ticks 200 --output .scratch/g1_station_t2_new_grasp
```

该独立入口仅接受零 Tick 相机初始化或 200 Tick／四块新动作的抓取诊断；不开放科学站搬运、预取或 Qwen 调度。模型／本体的准备和停止沿用上述不同 profile 的命令；新视觉资产的源导出工具是 `crates/dev_tools/python/scripts/unitree_g1_background_visual_export.py` 的 `--restore-station-tabletop` 选项，不能覆盖原仓库视觉制品。

## 科学站 T2 持物行走与停止阶段

同日 URI 重录 episode **20630** 使用四次新本体相机图片和原配 N1.6 推理，在同一科学站世界完成抓箱、100 Tick 持物等待、346 Tick 转向、608 Tick 行走及 100 Tick 零导航停止阶段。实际执行 **1,354 次积分、身体控制和力矩更新**。独立检查确认抓取后的全部 1,154 个样本均有当前求解器的正手接触支撑，无非足部对科学站固定世界的正接触；最低站立余弦 **0.991874**。

箱子自抓取结束处水平移动 **2.293178 米**，机器人自持物等待结束处移动 **1.714570 米**，自身速度积分得到 `[0.072969, 2.049981]` 米。箱子始终持于手中，最后执行两秒零导航停止阶段。末帧机器人水平速度约 0.0405 m/s、箱子线速度约 0.0498 m/s，仍有身体控制的小幅运动；没有通过物体静止放置阈值。本轮使用公开通道的明确转向／两米命令，持物结束后取得新的实际 RGB 帧 6，随后由原有传统控制执行；没有 Qwen 调用、目标容器定位、放置或松手。活跃仿真／墙钟比 **0.999106**，控制误期与待处理 Tick 均为零；模型／图像边界有暂停，不计为全程连续 1× 验收。原 Homie v2、N1.6 合同、身体增益、执行器和 50 Hz 单次积分保持，4 PGS 仍是显式开发诊断。

源代码：`5a2bd5ee62d8d93d180dfc181f2529369e4f2819`；实际应用 SHA256：`9cf544a44da3e8bad854ea544bf6fc88122f57099dee0795a39b52d3f3dd76df`。全部物理记录在 float32 精度下与原渲染 episode 20628 及先行保存动作的机械测试一致。URI 展示层仅用于主窗口；七张传感器 RGB 图像、四次推理的全部 52 个输入／动作数组与原渲染一致。原渲染原片、GIF 和精确二进制仍保留在外部证据中。第一次新窗口运行在 200 Tick 后因命令交接读取了旧抓取回执而退出；先用失败测试复现，再修复异步确认等待。另一次窗口启动遇到显示器休眠下的 Vulkan 错误，零积分、零任务推理；唤醒显示后，零积分相机预检通过，再执行本轮成功运行。失败证据均保留，没有物理调参或训练。

新主页 GIF 为 **900×720、150 帧、12.00 秒原速节选**，取完整原片 23.84–35.84 秒，展示持物行走和停止阶段。仅裁切显示区域、缩小尺寸和降低采样率，没有时长缩放、动作插值或渲染重放。完整窗口仍为 1920×1080、8×MSAA。原片 912 帧／36.48 秒、完整解码通过，SHA256：`3c7c3a42b9daa44aeecca59fecd101415647c695ca253ccf51bfd2c8d9909c0e`；新 GIF SHA256：`85290456bd2b8d3d62263a9688d1d8072bbdb91547a11f941f137b391b206ebb`。

外部证据位于 `/home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_station_t2_carry_milestone_001/`，包含两轮保存动作机械测试、失败运行、窗口预检、成功新推理运行、原片、精确二进制、测试和源码。本轮冻结配置绑定 localhost 5558；复用上述 N1.6 服务命令时将端口改为 **5558**，再运行独立入口：

```bash
./target/debug/bevy_sim2sim --scene g1_station_mobile_carry_diagnostic --robot g1 \
  --g1-config /home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_station_t2_carry_milestone_001/evidence/g1_station_t2_carry_live_20261004_0430/config.json \
  --g1-ticks 2300 --output .scratch/g1_station_t2_new_carry
```

2,300 Tick 是最大预算；完成原控制器的停止阶段后自动退出。停止仍使用自己应用／前台服务终端的 Ctrl+C 或关闭应用窗口。该入口只接受冻结科学站本体／六个原任务支撑几何、四块新原配动作、100 Tick 持物等待和公开通道两米诊断；原有 200 Tick 抓取入口及原仓库入口继续拒绝跨范围配置。中文 UI 的正式执行／导航仍未解锁。

## 仍需完成

T1/T2 正式合同与控制配置、各 5 组初始位置 × 2 个种子的至少 8/10、科学站目标容器放置／松手、从初始指令开始的 Qwen 调度、全程连续 1×、故障恢复、重置与过期隔离、低坡／门槛、共存性能及完整回归仍需验收。中文 UI 目前只开放已验证观察和停止，正式执行／导航按钮不随阶段合入而解锁。主干中的成功样例也不能替代这些要求。

# G1 阶段运行与录像

2026-10-04 将当前 G1 开发成果按用户要求合入主干，并在 Bevy 主页展示真实运行录像。完整任务目标继续保留；本页中的诊断成功不授予正式任务能力。

**最新：双手拇指准备修复后的两例原生开发回归均通过。** 主页新 12 秒原速 GIF 来自原失败位置恢复成功的实际运行；箱子搬运 1.99 米、最终放稳并松手 2.76 秒。原固定五位置 × 两种子的成绩仍为 **1/10，未达到 8/10**，本轮不替换原十轮。详见 [双手准备修复](#双手拇指准备修复与两例原生回归) 与 [十轮结果](#科学站-t2-五位置两种子110未达标)。

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

## 科学站容器放置开发诊断：尚未通过

2026-10-04 新增独立的 `g1_station_mobile_release_diagnostic` 入口，固定最大预算 3,300 Tick。它使用四块新原配 N1.6 抓取动作、100 Tick 自身状态持物等待、公开地图搜索方向、实际辅助相机标记定位、逐段接近和重新观察。原有 200 Tick 抓取、2,300 Tick 持箱行走以及原仓库入口的边界保留；正式 UI 执行／导航仍未解锁。

第一次新视觉运行在 1,493 Tick 拒绝了旧图像：短距离动作刚提交时，主线程沿用了上一段完成回执，在 Tick 1,334 提前请求下一张图。失败测试复现后，交接改为等待新动作自身的观察身份和完成回执；相机时效检查没有放宽。第二次运行跨过此交接，在 1,982 Tick 触发原有“一秒向前不足三厘米”的阻塞保护。独立检查确认没有非足部正固定接触；最后有效图像的箱体目标区边距与真实几何相差约 1.4 mm。

有限机械诊断保存了第二次运行的真实命令前缀：全部 1,982 个物理状态与原轨迹一致，固定正接触只来自科学站地面。随后 100 Tick 零导航持物等待保持站立与手支撑，再进行明确标记的机械松手测试，独立检查通过连续 **2.34 秒**的完全入区、手分离、容器向上支撑与严格速度条件。该轮共 **2,307 次**真实积分，但没有新视觉／任务模型／Qwen 调用，不授予视觉任务能力。

由此加入科学站的有界保护停止：原阻塞阈值不变，原仓库控制器仍直接拒绝；科学站停止向前运动并执行 100 Tick 零导航阶段。回执明确报告 `completed=false`、`blocked_stop_completed=true`，不宣称到达原导航目标。容器诊断必须取得新图像确认位置，才允许继续持物等待／释放；位置不合适时拒绝进一步前进。普通持箱行走发生同类阻塞后报告正确停止，不计为任务成功。

第三次新视觉运行 episode **20635** 在同一科学站完成 **2,082 次**积分／身体控制／力矩更新、四次新 N1.6 推理，并实际通过保护停止。抓取后的全部 1,882 个样本保持当前手接触支撑，机器人站立；与第二次运行的前 1,982 个物理状态、四张原本体 RGB 和全部 52 个模型输入／动作数组一致。活跃仿真／墙钟比 **0.999077**，积压与控制误期均为零。随后新辅助图像中，箱子遮住容器标记的部分编码，定位器拒绝继续；本轮**没有松手或任务成功**。下一步单独验证放置观察视角，保持原配 VLA 相机不变。

本轮源码提交为 `3a22a489c55fccc897612eb1fea94869fdb4aaf6`，实际应用 SHA256 为 `73d7fad394cf090b652387111d60b376a871757d7724b7ef67b2596241eac9af`。本轮五个机械／视觉实验合计 **10,189 次**积分、身体控制和力矩更新，**12 次**新 N1.6 推理；Qwen、N1.7、源 SDK 物理、训练和参数网格均为零。仍是明确的 4 PGS 开发诊断，保持 50 Hz／每 Tick 一次积分。原速窗口录像、失败图像、命令、源码、独立审计和旧回执回归证据存放于 `/home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_station_t2_bin_diagnosis_001/`。

以下保存的是旧版遮挡边界。复用前述 localhost 5558 的 N1.6 服务，并将该配置的 `mobile_scan.vision.script_path` 指向同一证据目录 `source/Bevy_Sim2Sim/crates/dev_tools/python/scripts/unitree_g1_mobile_vision.py` 后，使用该目录内的原始可执行文件；保持原脚本哈希。当前脚本已经增加新相机，不能直接套用旧配置的身份校验。

```bash
/home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_station_t2_bin_diagnosis_001/evidence/g1_station_t2_bin_live_20261004_0435/executed_app --scene g1_station_mobile_release_diagnostic --robot g1 \
  --g1-config .scratch/g1_station_bin_previous_config.json \
  --g1-ticks 3300 --output .scratch/g1_station_bin_new_diagnostic
```

当前主页持箱行走 GIF 继续保留；本次未通过的放置尝试不替代该成功阶段录像。T1/T2 各 8/10、科学站完整视觉放置、初始自然语言 Qwen 调度、全程连续 1×、恢复／地形／性能与完整交付要求均继续保留。

## 科学站放置相机与释放姿态诊断：仍未通过

同日增加独立的 `auxiliary_bin_placement` 被动本体相机：在原 Arena 相机安装点上方 230 mm，原光学安装方向再上倾 7°；相对原有辅助视角抬高 80 mm、向下多倾 8°。原配 N1.6 的 ArenaEgo、原辅助相机和 640×480 针孔参数保留。单次几何诊断先复现旧视角对容器标记下排的遮挡，再推导新安装位；没有相机网格、物理参数搜索或训练。只在科学站容器诊断的已完成保护停止处切换，并在暂停的新边界取得实际图像。

episode **20637** 执行 2,082 次真实积分和四次新 N1.6 推理。新实际图像完整显示两个标记，但开发工具的相机白名单遗漏新名称而拒绝继续。保存图片的独立定位确认两个标记，最短编码边约 30.6 px、最大重投影误差 0.332 px；原配四张 ArenaEgo 图、全部 52 个模型数组及前 2,082 个物理状态与旧版一致。两条失败回归复现了遗漏后，白名单和回执检查统一绑定新固定安装位；仍要求当前双标记、原几何、手部邻近、旧 episode 隔离和原放置条件。新安装位不接受目标记忆替代当前图像。

episode **20638** 完成新相机定位、确认箱体投影入区，再增加 100 Tick 站立持物等待，共 **2,182 次**积分／身体控制／力矩更新、四次新 N1.6 推理。前 2,082 个物理状态与旧版一致，抓取后的全部 1,982 个样本保持正手接触支撑，最低站立余弦 **0.991268**。活跃仿真／墙钟比 **0.998978**，积压和控制误期为零；模型／图像边界仍有暂停。

新鲜释放图像的目标区边距 **44.8 mm**、落差 **321.4 mm**、机器人线速度 **0.00349 m/s** 均满足原条件，但旧手部预测要求掌间距 **491 mm**，超过原 **350 mm** 上限。独立几何复核确认图像箱心与原生真值相差约 **3.43 mm**；包围矩形的预测较保守，但精确凸包距离在 350 mm 张掌处仍只有约 **9.87 mm**（图像）／**8.06 mm**（独立真值），低于现有 **10 mm** 条件，局限位于左手拇指。因此保留拒绝；本轮**没有松手或任务成功**。这些独立真值计算没有进入执行链路，未修改释放合同或阈值。下一步先验证可执行的释放握姿，再进行新视觉任务运行。

本阶段额外执行 **4,264 次**真实积分、身体控制和力矩更新，**8 次**新原配 N1.6 推理；离线相机／凸包诊断零积分、零模型调用。Qwen、N1.7、源 SDK 物理、训练和参数网格均为零。实际运行源码为 `62550d9daa700cfe185e7fb73feb6d6427a5ad1b`，实际应用 SHA256 为 `ee606f099bf2ff691374da1f00b4b562b8f556ec5641e22584d38853f66920f2`。完整失败录像、实际相机图像、精确可执行文件、回归和诊断保存在 `/home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_station_t2_placement_camera_diagnosis_001/`。

在工程目录内，复用 localhost 5558 的原配 N1.6 服务后，下面入口复现本次释放检查拒绝。配置仅将视觉脚本路径移至归档中的同一哈希文件；外部本体、权重、动态库与纹理缓存仍需按制品清单准备。

```bash
python3 - <<'PY'
from pathlib import Path
import json
archive = Path('/home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_station_t2_placement_camera_diagnosis_001')
configuration = json.loads((archive / 'evidence/g1_station_t2_bin_live_20261004_0438/config.json').read_text())
configuration['mobile_scan']['vision']['script_path'] = str(archive / 'source/Bevy_Sim2Sim/crates/dev_tools/python/scripts/unitree_g1_mobile_vision.py')
Path('.scratch').mkdir(exist_ok=True)
Path('.scratch/g1_station_placement_config.json').write_text(json.dumps(configuration, indent=2) + '\n')
PY
/home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_station_t2_placement_camera_diagnosis_001/evidence/g1_station_t2_bin_live_20261004_0438/executed_app \
  --scene g1_station_mobile_release_diagnostic --robot g1 \
  --g1-config .scratch/g1_station_placement_config.json \
  --g1-ticks 3300 --output .scratch/g1_station_placement_new_diagnostic
```

已成功的 12 秒科学站持箱行走 GIF 继续保留，不将本阶段失败片段发布为放置成功。所有正式验收要求继续保留，整体任务仍在推进。
## 科学站视觉搬运与容器释放：单次通过

2026-10-04，episode `20643` 完成一次新的实际科学站视觉运行：四次原配 N1.6 本体相机推理抓箱，当前标记 RGB 定位，传统几何／自身状态搬运，有限原有拇指电机准备，重新观察后松手。箱子水平移动 **1.957984 米**，机器人从持物等待结束处移动 **1.384178 米**；全部 **2,507 Tick** 均为 50 Hz、一次 20 ms 积分、一次真实 Homie 身体推理和一次电机更新，明确保留 4 PGS 开发诊断配置。

独立验收确认箱体所有碰撞顶点进入原配容器底面上方的目标区、没有机器人接触、由容器向上接触冲量支撑、低于既定速度阈值并连续稳定 **2.6 秒**。最终目标区最小边距为 **55.31 毫米**，线速度 `1.27e-6 m/s`、角速度 `1.04e-6 rad/s`；全程站立，最低 upright cosine 为 `0.9912678`。抓取后至松手前的 **2,082 Tick** 全部为当前手部接触支撑，无物体附着约束，且没有脚以外的正固定站体接触冲量。

这次解决了两个具体问题。原矩形包围估算会把旋转箱体的空角算成障碍；新估算以全部原碰撞凸包顶点验证分离平面，仍要求 **10 毫米净空／350 毫米张手上限**，原失败握姿仍被拒绝。一次离线有界几何求解选出左拇指两个原关节目标 `[-0.04068526, -0.18428603] rad`，原电机进行 50 Tick 渐变、50 Tick 收敛，其他手指、手掌／手臂目标和增益保持原控制语义；跟踪误差超过 `0.02 rad` 会暂停。该准备动作仅由当前目标区图像批准，不能授权松手。

初次接入运行 `0442` 真实完成准备，但相机请求在后台命令确认前读取了旧的“等待已完成”状态：第 2182 Tick 图像被交给第 2282 Tick 边界，严格检查拒绝执行。该失败已保留，并先用毫秒级状态回归测试复现 RED，再修正完成状态选择和相机最小原生 Tick 请求。成功运行 `0443` 的准备图像是 frame 14／Tick 2182；**松手依据另一张真实 frame 15／Tick 2282 图像**，目标区边距估计 51.25 毫米，最大张手处最小凸包分离下界 18.53 毫米。没有重写旧图像时间戳。

前 2,182 个物理步骤、四张原配 ArenaEgo RGB 和 52 个原配模型输入／动作数组与先前失败握姿运行逐 float32／逐像素一致；全部 2,507 个身体与物件物理状态与先通过的有限机械测试一致。独立真值只进入验收器，未进入任务定位、指令或动作。功能工作区测试 **341 passed／77 ignored**，默认工作区 **246 passed／35 ignored**；实际视觉 Python 几何回归 5 passed，包括旋转箱体空角、5 毫米不足、真实接触和不可信优化器候选；新状态交接回归先 RED、后 GREEN。其余独立 Python 脚本 61 passed；未改动的原 USD 材质测试在单独 Arena 环境可导入，因本轮未指定原始 USD 路径而 5 skipped。

![科学站持物行走和容器松手，12 秒两段原速节选](media/g1_science_station_box_release_12s.gif)

GIF 是原 1920×1080、25 FPS、8×MSAA 窗口录像的 **36–41 秒行走＋57–64 秒释放**两个原速片段，中间有一次直接切镜；裁切 `[0,180,1080,864]` 后缩到 900×720，共 150 帧、每帧 80 ms。没有加速、补帧、重绘运动或替换物理轨迹。原视频长度 64.48 秒，SHA-256 `5fdb1b0595ba96057a31e99b389835530e8b5f2228902d2abf2bffb12eac63e2`；GIF SHA-256 `c60467e9ccdfdece6e4e9a765010645280138102cfff17cd955782438819361d`。

实际运行源码为 `21bb6e8968813b568b65938dbf9ca5522f06a50b`，执行文件 SHA-256 `458b9c5f8947e1b2ce772f5587443ae6a188ce4f2803e03c78dc96eb06106c39`。完整新／失败录像、逐 Tick 原始记录、当前 RGB／图像身份、模型健康计数、机械对照、测试日志、精确执行文件与源码进入外部只读证据目录：

```text
/home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_station_t2_visual_release_milestone_001
```

本机已有缓存下的复跑入口如下。需要外部原配本体、ORT、Homie、N1.6 权重和站体纹理缓存；这不是从空机器安装的完整手册。服务只绑定本机，启动前先确认 5558 未被占用；停止命令只针对本段自己启动的模型进程。

```bash
set -e
cd /home/ethan/Projects/TempWorktree/Sai_Lab/unitree_g1/Bevy_Sim2Sim
G1_RELEASE_ARCHIVE=/home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_station_t2_visual_release_milestone_001
G1_STAGE_OUTPUT="$PWD/.scratch/g1/station_release_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$G1_STAGE_OUTPUT"
export G1_RELEASE_ARCHIVE G1_STAGE_OUTPUT
/home/ethan/models/unitree_g1/envs/policy_gr00t_n16_cu13/bin/python - <<'PY'
import hashlib, json, os, socket
from pathlib import Path
with socket.socket() as lease_probe:
    lease_probe.bind(('127.0.0.1',5558))
a=Path(os.environ['G1_RELEASE_ARCHIVE']); o=Path(os.environ['G1_STAGE_OUTPUT'])
c=json.loads((a/'evidence/g1_station_t2_bin_live_20261004_0443/config.json').read_text())
p=a/'source/Bevy_Sim2Sim/crates/dev_tools/python/scripts/unitree_g1_mobile_vision.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()==c['mobile_scan']['vision']['script_sha256']
c['mobile_scan']['vision']['script_path']=str(p)
(o/'config.json').write_text(json.dumps(c,indent=2)+'\n')
PY
/home/ethan/models/unitree_g1/envs/policy_gr00t_n16_cu13/bin/python -u \
  "$G1_RELEASE_ARCHIVE/source/Bevy_Sim2Sim/crates/dev_tools/python/scripts/unitree_g1_mobile_server.py" \
  --gr00t-source /home/ethan/Projects/Sai_Lab/upstream/unitree_g1/isaac_gr00t_n16 \
  --model-root /home/ethan/models/unitree_g1/mobile_box/dfe74af855007f26093f362cd2d7a2f404b64b93 \
  --receipt /home/ethan/Projects/Sai_Lab/.scratch/unitree_g1/policy/mobile_box_files.json \
  --runtime-env /home/ethan/models/unitree_g1/envs/policy_gr00t_n16_cu13 \
  --port 5558 --seed 42 --capture-dir "$G1_STAGE_OUTPUT/policy_captures" \
  > "$G1_STAGE_OUTPUT/model.log" 2>&1 &
g1_stage_model_pid=$!
trap 'kill -TERM "$g1_stage_model_pid" 2>/dev/null; wait "$g1_stage_model_pid" || true' EXIT
for g1_stage_attempt in $(seq 1 90); do
  curl --max-time 1 --fail --silent http://127.0.0.1:5558/health && break
  sleep 1
done
curl --max-time 1 --fail http://127.0.0.1:5558/health
kill -0 "$g1_stage_model_pid"
"$G1_RELEASE_ARCHIVE/evidence/g1_station_t2_bin_live_20261004_0443/executed_app" \
  --scene g1_station_mobile_release_diagnostic --robot g1 \
  --g1-config "$G1_STAGE_OUTPUT/config.json" --g1-ticks 3300 \
  --output "$G1_STAGE_OUTPUT/native"
/home/ethan/models/unitree_g1/envs/policy_gr00t_n16_cu13/bin/python \
  "$G1_RELEASE_ARCHIVE/source/Bevy_Sim2Sim/crates/dev_tools/python/scripts/unitree_g1_mobile_placement_audit.py" \
  --definition /home/ethan/models/unitree_g1/task_assets/20261001_frozen/native_task_objects_t1_60_diagnostic_v2.json \
  --trace "$G1_STAGE_OUTPUT/native/owner_steps.jsonl" \
  --output "$G1_STAGE_OUTPUT/independent_placement_audit.json"
kill -TERM "$g1_stage_model_pid"
wait "$g1_stage_model_pid" || true
trap - EXIT
```

**本轮没有调用 Qwen，不是正式 8/10 成绩。** 活跃物理区间 sim/wall 为 `0.99893977`、积压和控制误期均为 0，但模型／图像边界仍有显式暂停，不能计为全程连续 1×。初始自然语言调度、两任务各五组位置×两种子、恢复／重置／故障、模型共存、完整性能和低坡／门槛仍须完成。当前完整目标和全部 28 项验收继续保持 ACTIVE；本阶段合入主干不代表全任务完成。


## 科学站初始自然语言与本地 Qwen：单次闭环

2026-10-04，episode `20648` 从中文指令“请把眼前的棕色箱子搬到蓝色容器里，放稳后松手。”开始，完成一次科学站固定任务开发闭环。初始原配 ArenaEgo RGB 和真实自身状态先交给本地 Qwen；模型返回 `begin_fixed_profile` 前，任务策略推理和物理积分均为零。该决定只准入已声明的原配搬箱流程，不定位看不见的容器、不注册虚构目标，也不授权松手。四块原配 N1.6 新图像抓取后，公开通道搜索和持物行走仍是明确披露的传统控制；新的实际标记 RGB 再让 Qwen 选择 `box_marker_22` → `bin_marker_21`。有限拇指准备、强制重新观察和严格松手检查与上一阶段保持一致。

| 实际本地 Qwen 决定 | 对应原生 Tick | 服务耗时 | 原图像到准入 |
| --- | ---: | ---: | ---: |
| 开始固定 profile | 0 | 7,859 ms | 7,923 ms |
| 当前 RGB 的箱子／容器选择 | 878 | 15,150 ms | 15,685 ms |
| 执行结束后的新图像观察 | 2507 | 10,286 ms | 10,383 ms |

均保留原有 **20 秒图像／请求期限、1 秒仿真年龄上限**，三次 HTTP 请求／结果、零旧结果丢弃、零重试；运行期只调用本机 8002 的已缓存 FP8 Qwen 和 5558 的原配 N1.6。最终反馈仍明确说明动作结束不等于任务通过，收回物理能力后只能观察／停止，独立物理真值不进入 Qwen。初始请求协议不接受目标 ID、坐标、关节值或其他额外动作字段；停用能力、旧 episode、旧帧、超时和错误回复种类均拒绝准入。

独立检查通过：**2,507 次**真实积分、身体推理和电机更新，**四次**新 N1.6 推理；箱子水平位移 **1.957984 米**，机器人持物等待后的位移 **1.384178 米**，最终在容器内与全部机器人接触分离、由容器支撑且连续稳定 **2.6 秒**。所有 2,507 个身体／物件物理状态与先前机械正例逐 float32 完全一致，四张原配 VLA RGB 和 52 个输入／动作数组与原成功前缀一致。抓取后至松手前 2,082 Tick 均为当前手接触支撑，全程站立、无脚以外的正固定站体接触。活跃区间 sim/wall 为 `0.99884216`，积压与控制误期均为 0；模型／图像边界仍暂停，**不是全程连续 1× 成绩**。

另一个真实窗口 episode `20647` 接收“停止任务，不要执行抓取或搬运。”，Qwen 在服务耗时 8,721 ms／图像年龄 8,796 ms 返回停止，原生积分、身体／电机更新和 N1.6 调用均为 **0**。这证明初始停止边界，不代替运行中停止、故障恢复或正式任务成功率。首次接入 `0445` 在零 Tick 因抓取回执尚未创建而退出，没有收到模型决定；现已在初始阶段建立回执，并在抓取交接时合并保留原始事件，新增两项回归覆盖。失败回执及准确执行文件均保留。

实际运行源码 `0440d5d155fc5563bddb376302f348264123e2a2`，实际执行文件 SHA-256 `082df7464c8a858941ef5e6406be0ff6c3b006dcb7c01244b7ba7ecd2fe2a148`。完整原速窗口录像为 1920×1080、25 FPS、8×MSAA、98 秒，SHA-256 `cf5a51ec6387c7e8e07aaa263cf4754379b861a0ba77c9bdd88d9a677ce145ad`。主页继续展示上一轮 12 秒原速 GIF，其物理轨迹已独立核对相同，不把旧片重新标成新 Qwen 录制。证据目录：

```text
/home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_station_initial_qwen_milestone_001
```

复跑先按上一节启动属于自己的原配 localhost 5558 服务，并以原隔离脚本启动 Qwen，确认 8002 的 `/health` 和 `/v1/models`。使用新的输出目录，在加载前准备配置：将本节归档 `evidence/g1_station_initial_qwen_live_20261004_0448/config.json` 的视觉脚本路径改到归档 `source/Bevy_Sim2Sim/crates/dev_tools/python/scripts/unitree_g1_mobile_vision.py`，同时核对原 `script_sha256`；其他本体、策略、相机和动力学配置不改。执行同一归档轮的 `executed_app --scene g1_station_mobile_release_diagnostic --robot g1 --g1-config <新配置> --g1-ticks 3300 --output <新目录>`，再用上一节的独立 placement audit 检查新 `owner_steps.jsonl`。外部权重／动态库／纹理缓存仍必需，所有服务由同一个负责人管理，仅停止自己启动的模型进程与带原 owner 标签的 Qwen 容器。

```bash
/home/ethan/LocalServices/Local_Qwen/qwen38-27b-fp8-dgx-spark/g1-service.sh start
curl --fail --max-time 3 http://127.0.0.1:8002/health
curl --fail --max-time 3 http://127.0.0.1:8002/v1/models
# 完成运行后：
/home/ethan/LocalServices/Local_Qwen/qwen38-27b-fp8-dgx-spark/g1-service.sh stop
```

原服务 profile、权重、镜像、执行器增益、正式 50 Hz 单次积分、N1.6 合同和明确 4 PGS 开发配置均未改变。本阶段合计 2,507 次物理／身体／电机更新、四次新 N1.6 和四次有结果的 Qwen 调用（停止一轮＋完整流程三次），零训练／参数网格／新系统配置／付费资源；全部属于本次运行的服务、窗口和录像进程均已停止。**这只是固定任务的单次闭环样例。** 中文交互窗口仍仅开放观察／停止；任意目标泛化、运行中停止与恢复、正式两任务各 8/10、连续 1×、地形、共存性能及完整从克隆安装的手册仍未通过，全部 28 项验收与持续 PM 继续 ACTIVE。

本阶段功能工作区测试 **348 passed／77 ignored**、默认工作区 **251 passed／35 ignored**、新握持回执开发测试 **68 passed／18 ignored**、任务模块 **62 passed／1 ignored**；结构检查 963 项通过。科学站入口拒绝旧源场景 Qwen scope 与非本机服务 URL，均在创建输出目录、物理或模型运行前返回错误。

## 科学站 T2 五位置两种子：1/10，未达标

此次在任务推理前冻结场景、十个唯一 episode、位置、种子、源码／程序／模型身份、严格放置规则和超时。先进行五组各 100 Tick 的支撑预检，以及零 Tick 的实际相机捕获；全部由原货架／桌面支撑，机器人站立，五张本体图互不相同。预检不进入十轮分子或分母。

以下为相对原配位置的源坐标偏移；高度、朝向、机器人初始状态、物体参数和控制设置均保持原样。A／B 约两米，明确只测试这个匹配任务的小范围初始变化。

| 位置 | 箱子偏移 X/Y（m） | 容器偏移 X/Y（m） | 种子 0 | 种子 42 |
| --- | --- | --- | --- | --- |
| 0 | 0 / 0 | 0 / 0 | 夹持失稳后定位失败 | **成功** |
| 1 | +0.015 / 0 | +0.050 / 0 | 无法确认当前容纳几何 | 净空保护暂停 |
| 2 | −0.015 / 0 | −0.050 / 0 | 有界接近保护暂停 | 录像基础设施失败，零 Tick |
| 3 | 0 / +0.030 | 0 / +0.030 | 净空保护暂停 | 净空保护暂停 |
| 4 | 0 / −0.030 | 0 / −0.030 | 夹持失稳后定位失败 | 握持校正超出原边界 |

结果为 **1/10，未达到 8/10**。九轮实际物理运行均保持站立；两轮失去持续手支撑，一轮在握持目标变化保护处拒绝继续，五轮持物但未取得视觉几何／净空准入。保护暂停不计成功。停止错误不能直接当成物理根因：例如第一轮在 Tick 218 已重新碰到货架，随后箱子落地，最后才表现为标记定位失败。

第八轮在窗口列表遍历时遇到正在关闭的窗口，`xwininfo` 报 `Bad Window/Bad Drawable`。该轮尚未取得初始图像／Qwen／任务策略结果，物理轨迹为空；N1.6 捕获为空，Qwen 完成计数未增加。自有应用和模型进程已回收。这一轮按失败留在十轮分母中，**没有重跑**。原七轮再次独立审计后，仅补完尚未启动的两个 episode；源模型、物理、全部十个配置、阈值和超时保持原冻结身份。之后修复录像工具，使这种竞态只在原 20 秒上限内重读窗口列表，同时保留异常路径的模型计数；回归测试不启动任何物理或模型推理。

每个种子只加载一个原配 N1.6 拥有者，逐轮恢复其**模型构造完成后的 CPU／CUDA 随机状态**，总推理计数不清零。不同 episode 拒收旧请求，正在推理时不能恢复随机状态。原配 `Gr00tPolicy` 没有跨轮任务状态；权重、归一化、动作合同和固定英文任务指令保持原样。新成功轮的四组原始模型输入／输出共 52 个数组与此前成功样例完全一致，真实新执行另计，旧样例没有混入十轮成绩。

成功 episode **20665（位置 0、种子 42）** 完成 2,507 次单独 20 ms 积分／身体／电机更新、四次新原配 N1.6 推理和三次实际本地 Qwen 图片请求。原配策略负责抓箱，公开标记、自身状态和传统几何控制负责后续搜索／搬运／松手；Qwen 分别准入固定任务、选择当前 RGB 已验证目标、读取完成后的新图像。箱子和机器人从第一至最后物理状态的水平位移为 1.941619／1.333278 m；独立真值只用于验收，连续 2.6 秒完全入区、手分离、容器向上支撑和严格速度条件通过。

放置规则在执行前固定：全箱碰撞顶点位于原容器底面凸包上方棱柱中，当前容器向上支撑、无机器人接触，连续至少两秒线速度 <0.02 m/s、角速度 <0.1 rad/s，机器人不跌倒。另要求全程 50 Hz 单次积分、抓取后至松手前真实手支撑、没有非足部固定场景支撑，A／B 距离及箱子位移至少 1.8 m、机器人位移至少 0.8 m，当前初始／目标／结束 Qwen 图像和拇指准备后的新释放图像可追溯。每轮最多 3,300 Tick、四次 VLA、三次 Qwen；原应用超时 120 秒，外层回收上限 150 秒。没有根据结果改门槛。

本阶段总计 **11,102 次**真实积分／身体／电机更新（支撑预检 500＋十轮 10,602）、**36 次**新 N1.6 推理、**16 次**实际 Qwen 结果。三批串行拥有者的 N1.6 计数为 20／8／8；额外加载源于录像工具异常回收，没有重复完成的轮次。Qwen 完成计数增量为 16，错误、超长、取消等计数增量均为零。全部自有应用／模型／录像进程和本地 Qwen 容器已停止。零训练、参数网格、源 SDK 物理、频率／系统配置改动和付费资源；仅唤醒已关闭的显示器。

十轮测试时发布的 GIF 来自成功轮的完整 **1920×1080、25 fps、99.8 秒**原速录像；当前主页已换成下一节的距离修复回归录像，旧素材仍在本节归档中。节选原片 65–70 秒与 81–88 秒，硬切拼接，裁切至 1200×960 后缩小到 900×720，150 帧／80 ms，共 12.00 秒；没有加速、动作插值或物理重放。完整原片 SHA256 为 `fcf800e6c470bc521cbd61721e011e2031a17303dfe0f3bec22703a76bb3ca3f`，新 GIF 为 `f98e2d2c14509b2b36c46a8589ed8e9b9a90cd5f59f8f0b601fc4c9b516ba736`。九段实际完整窗口录像均解码检查通过；中断轮没有伪造录像。

冻结清单 SHA256 为 `cbea9fe6142815f98278d19152a71a2663e9ba4cec34f0fb5b7329d06a3229e9`，原测试工具提交 `bc2e4ce9780ebff24d6f47a863e14d6a261b5419`；实际应用源码为 `0440d5d155fc5563bddb376302f348264123e2a2`，程序 SHA256 为 `082df7464c8a858941ef5e6406be0ff6c3b006dcb7c01244b7ba7ecd2fe2a148`。十轮配置／失败／推理捕获／录像／只读审计、原版工具、后续录像修复、精确程序和 Git bundle 归档于 `/home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_station_t2_frozen_suite_001/`。旧清单严格绑定原提交；新主干修复后的工具不能直接冒用旧代码身份。再次执行必须先冻结新的源身份，并保留本次 1/10 结果。

无需启动模型或物理即可重查成功轮的放置证据；输出使用新的可写目录，归档保持只读：

```bash
export G1_T2_ARCHIVE=/home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_station_t2_frozen_suite_001
mkdir -p .scratch/g1/t2_frozen_review
/home/ethan/models/unitree_g1/envs/policy_gr00t_n16_cu13/bin/python \
  "$G1_T2_ARCHIVE/source/frozen_suite_bc2/Bevy_Sim2Sim/crates/dev_tools/python/scripts/unitree_g1_mobile_placement_audit.py" \
  --definition "$G1_T2_ARCHIVE/asset_receipts/native_task_objects_t1_60_diagnostic_v2.json" \
  --trace "$G1_T2_ARCHIVE/evidence/g1_station_t2_frozen_suite_20261004_0450/position_0_seed_42/native/owner_steps.jsonl" \
  --output .scratch/g1/t2_frozen_review/independent_placement.json
```

新增／修改的移动模型与验收测试 **20 passed**，独立放置的对抗测试 **10 passed**，结构检查 **972 项通过**。本阶段没有修改 Rust 物理或控制实现，上一主干 `b601249` 的 CPU CI 已全部通过。正式成功率仍未达标；下一步先对失败图像估算与独立真实几何做只读对照，区分握持失稳、持物姿态和净空／有界接近条件，再提出最小的视觉校正，不放宽安全边界。T1、中文交互执行／运行中停止／重置／恢复、连续 1×、地形、共存性能、许可清单和完整安装手册仍未完成，全部 28 项与持续 PM 继续 ACTIVE。


## 精细靠近距离修复与两例原生回归

先离线重算冻结失败轮的实际 RGB／自身状态，再独立对照物理日志。位置 1／种子 0 的完整可行距离区间为 `[0.746837, 0.847803]` m，位置 2／种子 0 为 `[0.663163, 0.774959]` m；原规划在求交之前将上界截为 0.70 m，分别抹掉可行区间、将其误截为不足 40 mm 的窄区间。图片与自身状态 Tick 一致，独立实际几何也确认这两例存在可行区间。真值仅用于诊断，没有进入决策或动作。

修复先计算完整几何区间，再要求第一个可行点位于既有五段 × 0.15 m 的预算内。每段仍只执行原 0.10 m 导航请求加原 0.05 m 停止余量，段后必须取得新图片；不改变原配模型合同、身体控制、关节范围、接触、50 Hz 单次积分或放置门槛。新增的两种误判回归在修复前失败、修复后通过；真实失败图像回放转为可继续靠近，原成功轮的五次完整视觉输出不变。

两例开发验证在运行前重新冻结身份，各最多 3,300 Tick／四次 N1.6／三次 Qwen，零重试。**没有把它们加入或替换原十轮，正式成绩仍为 1/10，未达标。**

| 开发回归 | episode | 实际 Tick | N1.6／Qwen | 结果 |
| --- | --- | --- | --- | --- |
| 原失败位置 2／种子 0 | 20670 | 2,202 | 4／2 | 五段真实靠近后箱子投影进入容器；释放姿态检查拒绝，不计成功 |
| 原成功位置 0／种子 42 | 20671 | 2,507 | 4／3 | **成功**，全箱入区、当前容器支撑、手分离及严格速度条件连续 2.6 秒 |

失败回归搬箱 1.963510 m、机器人移动 1.345493 m，抓取后的 2,002 Tick 全部有当前手支撑，机器人站立；末帧独立真实底面边距为 28.8 mm，但箱子仍在手中。视觉估计边距 25.3 mm、落差 205 mm 均通过；原固定拇指准备预测的最小间隙为 **10.53 mm，低于保留的 16 mm 准备门槛**，因此未发出准备或松手命令。该处仍须区分视觉姿态误差与实际持物／手部几何，不放宽门槛。

两例各四组新原配策略 observation/action 共 52 个数组，分别与对应旧轮逐数组相同。失败轮前 1,233 Tick、成功轮全 2,507 Tick 的关节、力矩、根运动及物体物理记录精确一致，只忽略新 episode 身份；确认修复没有改变旧成功轨迹。此次新增 **4,709 次**原生积分／身体／电机更新、**8 次** N1.6、**5 次**实际本地 Qwen 结果；两个 1080p 完整原速视频均解码通过。全部自有应用、录像、模型和 Qwen 服务已退出，零训练／参数网格／新频率／系统配置／付费资源。

当前主页 GIF 来自新成功回归 episode 20671 的 **98.44 秒**实际窗口录像：原片 65–70 秒持箱行走与 79–86 秒释放，硬切拼接，900×720、150 帧／80 ms、共 **12.00 秒原速**。只裁切和降低媒体采样，没有动作插值或物理重放。原片 SHA256 为 `e6e608dda0206f92ffcefc5bfe94c35fe8adc5d26c7d7c34f391f52952a61129`；当前 GIF 为 `2110d58bfffc0969b512a023a57bf46219092def4248b35f9c09098c468d4e02`。

实际源码／应用身份 `ae24fbf8f2de6a5eb843e2eeee60f3ff048d51bc`，核心距离修复提交 `5255633b713760457a8174bf22aedcfbda3e483b`；程序 SHA256 `8fd0a321b7ca4f9bf6fa10923229faf94b5e1c6b3b1d269b4ad2d942a40ec435`，本次有限回归清单 SHA256 `8274b68f7fb8822ea49e8dc2c7bacd4e95719e621854ad68413ba55804f0f068`。配置仅更新唯一 episode 和绑定的新视觉脚本哈希，两个种子均恢复原模型构造后的随机状态。证据、失败、两个原速录像、精确程序、模型捕获、离线根因对照与源身份归档于 `/home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_fine_horizon_fix_milestone_001/`；不包含权重。

本阶段移动 Python 测试 **25 passed**，Rust 开发测试 **68 passed／18 ignored**，结构检查 **975 项通过**，格式和差异检查通过。五个视觉回归必须在已准备的 G1 OpenCV 环境运行；普通轮子 CI 未安装 OpenCV 时明确跳过它们，本机实际运行没有跳过：

```bash
/home/ethan/models/unitree_g1/envs/policy_gr00t_n16_cu13/bin/python -m unittest discover \
  -s crates/dev_tools/python/tests -p test_unitree_g1_mobile_vision.py -v
```

独立重查成功回归的原始放置证据，无需模型或物理：

```bash
export G1_FINE_ARCHIVE=/home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_fine_horizon_fix_milestone_001
mkdir -p .scratch/g1/fine_fix_review
/home/ethan/models/unitree_g1/envs/policy_gr00t_n16_cu13/bin/python \
  crates/dev_tools/python/scripts/unitree_g1_mobile_placement_audit.py \
  --definition /home/ethan/models/unitree_g1/task_assets/20261001_frozen/native_task_objects_t1_60_diagnostic_v2.json \
  --trace "$G1_FINE_ARCHIVE/evidence/physical_regressions/position_0_seed_42/native/owner_steps.jsonl" \
  --output .scratch/g1/fine_fix_review/independent_placement.json
```

这次单因修复解决了距离误判，尚未解决其余握持、姿态与恢复问题。模型／图像暂停、明确 4 PGS 开发配置仍保留；T1/T2 各 8/10、中文交互执行／运行中停止／重置／恢复、连续 1×、地形、共存性能、完整准备手册与全部 28 项验收继续推进。

## 双手拇指准备修复与两例原生回归

2026-10-04 对距离修复后停在容器前的 P2／seed0 建立真实保存 RGB 的快速失败回归。限制松手空间的是右拇指；旧准备只调整左手，对右侧限制碰撞体的导数为零，继续调整左手不能解决该问题。新方案在旧左手姿态已通过时保持原行为，否则由当前 RGB、自身关节状态和公开原始碰撞几何计算一次四关节候选。

候选只改变左右拇指 0／1 四个原配关节，每个目标不超过 ±0.25 rad，并遵守原关节限制。完整 16 个手部凸包分别核对支撑平面，保留 16 mm 准备间隙门槛；求解目标为 18 mm。51 个假设路径采样不得额外向内移动超过 1 mm。原失败图的候选间隙从 10.53 mm 提高到 18.10 mm，但几何预测不授权松手：原电机执行 50 Tick 渐变＋50 Tick 稳定后，必须用新的 RGB／自身状态重新通过原放箱检查。新轮实测四关节最大跟踪误差 0.00326 rad，低于原 0.02 rad 门槛；新图预测最小间隙 18.08 mm，随后才执行原释放控制。

| 实际开发回归 | 原失败位置 P2／seed0 | 原成功位置 P0／seed42 |
| --- | --- | --- |
| episode | 20674 | 20676 |
| 独立任务检查 | 通过 | 通过 |
| 原生积分／身体／电机更新 | 各 2,527 | 各 2,507 |
| 新 N1.6／本地 Qwen 请求 | 4／3 | 4／3 |
| 箱子／机器人水平位移 | 1.991355／1.346917 m | 1.941619／1.333278 m |
| 抓取后至释放前当前手接触支撑 | 2,102 Tick，全程 | 2,082 Tick，全程 |
| 最终完全入区、容器支撑、无机器人接触及稳定窗口 | 2.76 s | 2.60 s |
| 全程站立／非足部固定世界正支撑 | 是／无 | 是／无 |
| 与前轮模型数组逐元素一致 | 52 个 | 52 个 |
| 与前轮物理步骤精确一致 | 准备前全部 2,202 Tick | 全部 2,507 Tick |

两例共 5,034 次真实积分、8 次原配 N1.6 和 6 次本地 Qwen 请求，各轮模型计数、实际图片、动作捕获及 Qwen 服务计数相互核对。保持原 Homie v2、N1.6 权重／预处理、增益、执行器、50 Hz 每 Tick 一次 20 ms 积分，以及明确的 4 PGS／1 solver／1 CCD 开发配置；没有改频率、训练或系统配置。原成功路径不受四关节候选影响。本轮源代码为 `f7d02211b677ac1354a26fb1e28d519e4c457640`，实际应用 SHA256 为 `1db2818636be25afd3ea28138a737b152c1e6fbf6ed410cbc243fa0ba8999bb5`。

首次加载时 N1.6 在自有 Qwen 驻留状态下发生 CUDA 内存分配失败，零物理、零实际推理。改为先加载原 N1.6，再启动同一 Qwen 配置后恢复；第二种子重载期间执行命令收到 SIGTERM，第二个物理案例尚未启动。中断、部分生命周期和退出码均保留，正例随后在独立的有限命令中完成，没有重跑首个成功案例。前后实际积分／推理总数仍在原两例 6,600 Tick／8 N1.6／6 Qwen 预算内。最终自己的模型、应用和录像进程全部关闭，Qwen 正常退出，GPU 仅剩既有桌面进程。

两段完整原片均已完全解码核对；恢复轮原片为 104.16 s，正例原片为 102.64 s。新主页 GIF 取恢复轮原片 65–70 s 和 90–97 s，硬切拼接、裁切显示区域并缩小为 900×720，150 帧、每帧 80 ms，共 12.00 s；没有速度缩放、补造动作或渲染重放。恢复轮原片 SHA256 为 `2703bee920764e427802d8dbb48fa5c4d2bc49fa33b6d1a3c6a072cc512004be`，主页 GIF 为 `0c593437d33e10347f0c4933bbb5bd1564bdf312520518fdef74836d20b7dd6f`。之前的主页 GIF 与失败证据继续保留在旧归档中。

本轮归档为 `/home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_both_thumb_fix_milestone_001/`，包含精确应用、三份冻结清单、原生轨迹、两段原片、真实模型捕获、离线几何对照和加载失败／中断记录。权重继续使用外部缓存。准备、运行和停止命令沿用前文，原 N1.6 服务就绪后再启动自己的 Qwen；每个种子的冷启动和有限案例单独运行。可独立重查归档中的恢复轮放置证据：

```bash
mkdir -p .scratch/g1_both_thumb_review
/home/ethan/models/unitree_g1/envs/policy_gr00t_n16_cu13/bin/python \
  crates/dev_tools/python/scripts/unitree_g1_mobile_placement_audit.py \
  --definition /home/ethan/models/unitree_g1/task_assets/20261001_frozen/native_task_objects_t1_60_diagnostic_v2.json \
  --trace /home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_both_thumb_fix_milestone_001/evidence/t2_both_thumb_serial_load_20261004_0456/position_2_seed_0/native/owner_steps.jsonl \
  --output .scratch/g1_both_thumb_review/placement.json
```

**原正式十轮仍为 1/10，未达到 8/10；本轮两例开发回归不替换该分母。** 夹持失败、平面标记位姿歧义与其余冻结失败尚需逐项处理，之后另行冻结新十轮。T1 成功率、中文交互执行、运行中停止／重置／恢复、变化与故障、低坡／门槛、连续 1× 和完整准备流程仍需完成。

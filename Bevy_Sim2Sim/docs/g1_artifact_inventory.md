# G1 外部制品与许可清单

许可文本核验日期：2026-10-01；2026-10-05 根据实际执行收据补正 T2 源端版本。范围为当前选择的 T1 StaticApple、T2 MobileBox、本地 Qwen、G1 本体与下层控制器。本文是依赖清单，不是任务能力验收或完整发行包 SBOM。

本次直接读取固定 Git checkout、官方 Hugging Face revision API、模型卡及原始许可；现存下载 receipt 仅用于追溯下载来源。源码许可、模型许可、资产许可分项记录，不能互相替代。仓库中的用途声明（例如个人非商业研究）不是上游授予许可的依据。

## 固定的两条任务链

| 项目 | T1：StaticApple | T2：MobileBox |
| --- | --- | --- |
| 任务模型 | [GN1x-Tuned-Arena-G1-Static-PickNPlace][t1_card] | [GN1x-Tuned-Arena-G1-Loco-Manipulation][t2_card] |
| 模型 revision | `7f78bebf1a90131e7304beacfcd47eb27bad16ab` | `dfe74af855007f26093f362cd2d7a2f404b64b93`，来自 `gn1_6` 分支 |
| 实际模型接口 | 已发布的 N1.7 五阶段 ONNX；35 维解码动作，40 步 | 配置为 `Gr00tN1d6`；35 维解码动作，50 步 |
| 实际成功触发的源任务环境 | [Arena release/0.2.1][arena_t1]，`8b4a3a47fc53de23e8205089d71109a2e2348acd` | 同一 release/0.2.1；早期 [development 0.3][arena_t2] 另行保留 |
| 实际成功触发的 Isaac Lab | `e57379c634b42db5a0fe9f754341be6e2a7c7c43` | 同一 `e57379c634b42db5a0fe9f754341be6e2a7c7c43` |
| 原配下层控制器 | AGILE recurrent velocity-height policy，12 个腿部输出 | Homie v2 stand/walk，15 个下层输出 |
| 工程边界 | 单独保存 AGILE 的关节映射、增益、armature、观测和状态 | 单独保存 Homie 的映射、增益、历史观测和状态 |

两条链不因任务动作均为 35 维就可以交换本体控制配置。源配置入口为 [T1 静态评估文档][t1_eval]、[T1 G1 embodiment][g1_t1]、[T2 移动评估文档][t2_eval] 和 [T2 G1 embodiment][g1_t2]。T2 旧 `main` revision `629479fedb1cf97c2f11ddc49eed951c5b750139` 仅为历史缓存，不是这里的执行制品。T2 模型卡含 N1.5 等旧描述，执行型号以固定 `config.json`、processor 与权重接口为依据，不以模型卡段落推断。

T2 实际源端 episode 20112 使用 `release_0_2_1`、SDK `6.0.0-rc.22+release.33481.407f3ea1.gl`、源端 Torch `2.10.0+cu128`；隔离的 N1.6 服务仍为 Torch `2.10.0+cu130`。收据核实 19 次真实推理、942 次控制、3,768 次源端积分及原任务 proximity success。它没有证明松手后支撑两秒：末个可观测箱速约 1.548 m/s，终帧会自动重置。源端 200 Hz／50 Hz 与 Bevy 正式 50 Hz 单次积分分开记录。早期 development 0.3／Lab `ae37b028`／SDK 6.1 的失败仍保留，不当作最终源端版本；具体版本差异的单一因果未隔离。

## 任务模型

### T1：StaticApple

- 官方来源：[固定模型卡][t1_card]；HF API 返回的 SHA 与上述 revision 相同。用途是相机图像、自状态和任务语言到短动作块，不是底层站立控制器。
- 缓存：`/home/ethan/models/unitree_g1/static_apple/7f78bebf1a90131e7304beacfcd47eb27bad16ab/`。实际 ONNX 子目录为 `exports/g1-static-apple-b1-480x640/onnx/leapp-0.5.2/`；[导出说明][t1_export] 和 [graph.yaml][t1_graph] 是五阶段接口及导出元数据来源。
- 原始许可：[固定 revision 的 `LICENCE`][t1_license]，英式拼写；本地文件 Git blob `1385f9fd863b563ef98db57d6f7aced87e218d89` 与官方 API 一致。文本名为 NVIDIA Open Model License Agreement，日期为 2025-10-24。
- **已证实的许可冲突**：该 `LICENCE` 明确允许商业使用；同 revision 的模型卡仍写非商业使用，并将 governing terms 链接到 [2026 NVIDIA Open Model Agreement][noma_2026]。后者也是允许商业使用的不同版本。本文保留三个原始文本之间的差异，不自行决定优先级，也不将此制品标记为已完成商业再发行许可核验。2025 文本的原始网页为 [NVIDIA Open Model License Agreement][noma_2025]。
- 已证实：模型版本、原始许可、ONNX 五阶段及 35×40 接口。未由本清单证实：真实源环境或 Bevy 的抓取闭环成功、发布许可冲突的上游解释。数值 forward 成功本身不等于这些结论。

### T2：MobileBox

- 官方来源：[固定 `gn1_6` 模型卡][t2_card]；HF revision API 返回 `dfe74af855007f26093f362cd2d7a2f404b64b93`。缓存：`/home/ethan/models/unitree_g1/mobile_box/dfe74af855007f26093f362cd2d7a2f404b64b93/`。
- 用途：GR00T N1.6 移动抓取动作块。相关文件为 `config.json`、`processor_config.json`、`statistics.json`、`embodiment_id.json`、`model.safetensors.index.json` 和对应 shards。
- 该 revision 的官方文件列表没有独立 `LICENSE` 或 `LICENCE`；许可入口是模型卡直接链接的 [NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf][nvidia_nc_2022]。本次读到 PDF 正文，第 3.3 节将非商业用途定义为研究或评估，第 3.1 节要求再分发保留完整许可及原有声明。
- 已证实：固定 revision、模型卡许可链接及其原文、N1.6 配置和 35×50 解码接口。未由本清单证实：真实源环境/Bevy 移动抓取闭环成功。接口验证不授予任务 capability。

## G1 本体与下层控制器

### AGILE：T1 原配

- 官方仓库：[nvidia-isaac/WBC-AGILE][agile_repo]；Arena 的 [固定 WBC config][agile_cfg] 指定 `7259792cf10803aab814d101134d493d24c8f22f`。权重在该 revision 的 `agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.onnx`。
- 缓存：`/home/ethan/models/unitree_g1/agile_t1/`，包含 ONNX、`g1_agile.yaml`、Arena policy 源文件和 `provenance.json`。ONNX 的本地 SHA-256 与该 Git revision 的官方 LFS pointer 完全一致：`c8e30ec353bbba464298aeb051963cee17e1f1a484081c726ebb1d7513570da6`，2,050,687 bytes。核验仅读取了 pointer，没有再次下载权重。
- 原始许可：[固定 `LICENCE`][agile_license]。其范围说明把 `agile/algorithms/rsl_rl/**` 列为 BSD-3-Clause，其余仓库内容列为 Apache-2.0；选中的 ONNX 路径属于后者范围。Arena 的 adapter/config 文件另有 Apache-2.0 SPDX 标记和 Arena 许可。
- 已证实：下载权重与固定 upstream LFS 身份相符、明确的仓库许可范围、原配控制接口。未证实：替换为 Homie 或更换物理参数仍保持源任务能力。

### Homie v2：T2 原配

- 版本依据：[Arena 固定 WBC config][homie_cfg] 与 `G1HomiePolicyV2` adapter，而非可推测的另一个 OpenHomie 仓库版本。上游制品 URL 是 [stand.onnx][homie_stand] 和 [walk.onnx][homie_walk]，属于 NVIDIA Isaac 6.1 Arena 资产路径。
- 缓存：`/home/ethan/models/unitree_g1/homie_v2/stand.onnx`、`walk.onnx`；来源记录为同目录 `download_receipt.json`。本次重新计算并匹配 receipt：stand SHA-256 `f645da599d4ca3d29ed273c8f4712620bb680d34977469ca3aeabe5bb9631c18`；walk SHA-256 `7c82255b6905ffcc4468fa7f8ddcf7b70db168cf1042107ccab887cb6a8e5407`；各 1,886,682 bytes。
- 原始许可已核：Arena adapter/config 为 [Apache-2.0][arena_t2_license]。**外部这两个 ONNX 的精确许可文件及与具体二进制的适用关系未核实**：当前下载 receipt 未包含许可，不能从 adapter 的许可或别处同名模型推出。S3 URL 是版本目录而非不可变 Git revision，因此本地文件哈希是制品身份的一部分。
- 已证实：官方调用地址、下载来源、当前本地字节哈希。未证实：外部权重的独立许可适用链、源任务资格、Bevy 物理稳定性或任务成功。

### G1 USD / URDF / meshes 及派生缓存

| 制品 | 官方来源与版本依据 | 本地缓存和 SHA-256 |
| --- | --- | --- |
| G1 USD | [Isaac 6.1 Groot sample USD][g1_usd]；Arena/Isaac Lab 固定配置使用的 `g1_29dof_with_hand_rev_1_0.usd` | `/home/ethan/models/unitree_g1/homie_v2/g1_29dof_with_hand_rev_1_0.usd`；`a7a2bab76981d19a1d76adecdfffec9b52afa34df9ba8e288ccedf410d3ce6bd` |
| WBC URDF | [Arena G1 URDF][g1_urdf] | 同目录 `g1_29dof_with_hand.urdf`；`3dcb9c361753f464fa1f0238cdf800af842909628fd153733075607881c12d62` |
| WBC STL meshes | 同一 Arena `wbc_policy/robot_model/g1/meshes/` 官方路径，各文件 URL/哈希见 `mesh_receipt.json` | 同目录 `meshes/`；不以 URDF hash 代替各 mesh 的 hash |
| 本工程 USD 物理导出 | 上述 USD 的本地转换，不是新发布的上游资产 | 同目录 `g1_physics.json`；`571cb2558c137dccafa2d18adda5021f0885e0f10abf6d61edd62f1c6e8f13bd` |
| 本工程 USD 视觉导出 | 上述 USD 的本地三角面转换；对应 53 bodies、49 visuals | 同目录 `g1_visuals.json`；`98711070da898c75089b66ab35c799900cacb1e2980c537f9fcc1da8c8e9297d` |

USD/URDF 本次重新计算哈希并匹配 receipt；两个派生 JSON 的来源和哈希由 `g1_physics.receipt.json` 记录。共享网格不代表 AGILE 与 Homie 的惯量、增益或控制历史可以合并。

Isaac Lab 固定源码中有 [Unitree BSD-3-Clause 文本][unitree_license]，版权所有人为 Unitree Robotics；也有 [依赖/资产许可索引][lab_licenses]。**尚未建立该 Unitree 文本与本次 Groot sample USD、Arena WBC URDF/STL 的逐项授权对应关系**，因此本文不将所有下载资产统称为 BSD。也不把转换为 JSON 视为取得新的资产许可。Isaac Sim 资产许可页面的浏览工具读取因页面大小超限失败；这里保持未核，而不是用泛化的 Isaac Sim 代码许可补齐。

## Qwen 与推理服务

- 官方模型：[Qwen/Qwen3.8-27B-FP8][qwen_card]；官方 HF revision API 与本机 HF ref 均为 `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`。HF metadata 的 `pipeline_tag` 为 `image-text-to-text`，许可标记为 `apache-2.0`。
- 实际服务权重目录：`/home/ethan/models/Qwen3.8-27B-FP8/`，66 shards，来自既有 ModelScope 下载。HF snapshot 元数据位置：`/home/ethan/models/hf/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/`。**不能把这个 HF revision 直接宣称为平铺 ModelScope 全部权重的已核 revision**；目前只有 config 等有限比对，未建立全 shards 与该 HF revision 的逐文件映射。
- 原始许可：[HF 固定 `LICENSE`][qwen_license] 是 Apache-2.0。本地平铺目录 `LICENSE` 与该官方文件在换行和首尾空白归一化后全文相同；原始字节哈希不同，见下表。模型目录自己的 Apache 许可不来自部署脚本许可。
- 本地部署目录：`/home/ethan/LocalServices/Local_Qwen/qwen38-27b-fp8-dgx-spark/`；Git HEAD `969e52635a6a09b3e0ba8a1088158c001feeab41`，存在本机部署修改，不能把 HEAD 当全部运行配置的不可变快照。本地 `LICENSE` 为 MIT，只覆盖该部署仓库的相应内容。
- 已检查的容器：`vllm/vllm-openai:v0.27.1-aarch64`；镜像 digest `sha256:1c8e60a0841b333c700488cb029d3664807249da0c071e862191b00fe34b228c`，image ID `sha256:2c211a1273b48e8929f893b267aeb1509e6b84654cdbde1bad56d79e3964224d`，build revision `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`。该 revision 的 [vLLM `LICENSE`][vllm_license] 为 Apache-2.0；这不代表镜像内所有 CUDA、库和驱动组件均为 Apache。
- 用途边界：localhost OpenAI-compatible 图片/结构化技能决策，配置服务名 `qwen3.8-27b-fp8`、端口 `8002`。本清单不陈述服务当前在线，也不把已有照片测试当 Bevy 同步图像、自状态和闭环资格。科学站运行使用既有 `g1-service.sh` 与 `local-qwen-g1-validation` owner 标签，生命周期由单个会话负责人管理，见 [运行入口](g1_station_session.md)。

## 源码依赖与精确许可入口

下列五个本地 checkout 的 HEAD 已逐一检查，tracked 文件无修改，所读许可与 `git show HEAD:<file>` 完全一致。

| 源码/用途 | 固定 revision | 本地路径（均在 `/home/ethan/Projects/Sai_Lab/upstream/unitree_g1/`） | 原始许可 |
| --- | --- | --- | --- |
| Arena T1 静态源环境与 adapter | `8b4a3a47fc53de23e8205089d71109a2e2348acd` | `isaaclab_arena_static/` | [`LICENSE.md`][arena_t1_license]：Apache-2.0 |
| Arena T2 移动源环境与 adapter | `7d75c95934c51a0318c957a8831e862ca43c53b5` | `isaaclab_arena/` | [`LICENSE.md`][arena_t2_license]：Apache-2.0 |
| Isaac Lab T1 源实现 | `e57379c634b42db5a0fe9f754341be6e2a7c7c43` | `isaaclab_static/` | [`LICENSE`][lab_t1_license]：BSD-3-Clause；mimic 另有 Apache-2.0 文件 |
| Isaac Lab T2 源实现 | `ae37b028ea415c91ea2bc32609efcd759ed2b974` | `isaaclab/` | [`LICENSE`][lab_t2_license]：BSD-3-Clause；mimic 另有 Apache-2.0 文件 |
| Isaac-GR00T N1.6 T2 loader | `e29d8fc50b0e4745120ae3fb72447986fe638aa6` | `isaac_gr00t_n16/` | [`LICENSE`][gr00t_license]：NVIDIA 自定义许可；3.3 仅非商业研究，且另排除军用、监控、核技术服务、生物特征处理 |

GR00T 源码 `LICENSE` 与 T2 模型卡所链接的 2022 PDF **不是同一文本**：前者的 3.3 更具体，不能只列模型卡许可而漏掉执行 loader 的条款。T1 使用发布的 ONNX 制品，不因为名称相近就继承此 N1.6 源码的许可或接口。

## 本次直接校验的许可与模型卡指纹

| 本地或官方原始文件 | SHA-256 | 校验依据 |
| --- | --- | --- |
| T1 `LICENCE` | `725c51ff83f94cee293f76c8d5a97b7db628c2f0d55db5e65e6de2769190b942` | 本地 Git blob 与固定 HF API 相同 |
| T1 `README.md` | `38489271f0f7a5b6d56e6f964d0348e0b222af0225f50ddfeca5a9a6da85dcb9` | Git blob `9a5d8e55684dfd5a516df8c52841e899b6ef96b5` 与 HF API 相同 |
| T2 `README.md` | `1039ad159289eeff9183ea31319dfc2482d9c1f3ed54e48132839c2da3f5e39f` | Git blob `de4ffac2d1547a7fbad6ff08b173baa666b0dc66` 与 HF API 相同 |
| AGILE 固定 `LICENCE` | `06aff3f5d965a76d6f5fb138fe7f9fe0ffbc87e42eef11b8d09582492f7e88e2` | 直接读取固定 GitHub raw URL |
| Qwen 平铺缓存 `LICENSE` | `50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32` | 直接读取本地原文件 |
| Qwen 固定 HF `LICENSE` | `bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a` | 直接读取固定 HF raw URL；与本地文本归一化相同 |
| Arena 两版本 `LICENSE.md` | `1296f601b312777d3aaaf3c36a9999f489a0e0d1dc5a341fedeccf25918366f1` | 本地文件与各自 Git HEAD 相同 |
| Isaac-GR00T N1.6 `LICENSE` | `564046abbef821cefd5c169d34ee1e96b3dfb72cf0be81de41ef8f4a1323c5a3` | 本地文件与 Git HEAD 相同 |
| Isaac Lab Unitree 许可文本 | `2b794acf1e250f5545ae5a47d81e5685ea5488ce077d1087ed8be5f81e9ba5c0` | 两个固定 Lab checkout 中内容相同；资产适用关系仍待核 |
| vLLM 固定 `LICENSE` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` | 直接读取镜像 build revision 的 GitHub raw URL |

本清单尚未闭合的来源问题是 T1 许可冲突、Homie 外部权重许可、下载的 USD/URDF/STL 与资产许可的对应关系、以及 Qwen 平铺 shards 的完整固定 revision 映射。它们保持独立状态；填写版本或许可清单不改变任务能力开关或物理验收状态。

[t1_card]: https://huggingface.co/nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace/blob/7f78bebf1a90131e7304beacfcd47eb27bad16ab/README.md
[t1_license]: https://huggingface.co/nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace/blob/7f78bebf1a90131e7304beacfcd47eb27bad16ab/LICENCE
[t1_export]: https://huggingface.co/nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace/blob/7f78bebf1a90131e7304beacfcd47eb27bad16ab/exports/g1-static-apple-b1-480x640/onnx/leapp-0.5.2/README.md
[t1_graph]: https://huggingface.co/nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace/blob/7f78bebf1a90131e7304beacfcd47eb27bad16ab/exports/g1-static-apple-b1-480x640/onnx/leapp-0.5.2/graph.yaml
[t2_card]: https://huggingface.co/nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation/blob/dfe74af855007f26093f362cd2d7a2f404b64b93/README.md
[nvidia_nc_2022]: https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf
[noma_2025]: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
[noma_2026]: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/
[arena_t1]: https://github.com/isaac-sim/IsaacLab-Arena/tree/8b4a3a47fc53de23e8205089d71109a2e2348acd
[arena_t2]: https://github.com/isaac-sim/IsaacLab-Arena/tree/7d75c95934c51a0318c957a8831e862ca43c53b5
[arena_t1_license]: https://github.com/isaac-sim/IsaacLab-Arena/blob/8b4a3a47fc53de23e8205089d71109a2e2348acd/LICENSE.md
[arena_t2_license]: https://github.com/isaac-sim/IsaacLab-Arena/blob/7d75c95934c51a0318c957a8831e862ca43c53b5/LICENSE.md
[t1_eval]: https://github.com/isaac-sim/IsaacLab-Arena/blob/8b4a3a47fc53de23e8205089d71109a2e2348acd/docs/pages/example_workflows/static_apple/step_4_evaluation.rst
[t2_eval]: https://github.com/isaac-sim/IsaacLab-Arena/blob/7d75c95934c51a0318c957a8831e862ca43c53b5/docs/pages/example_workflows/locomanipulation/step_5_evaluation.rst
[g1_t1]: https://github.com/isaac-sim/IsaacLab-Arena/blob/8b4a3a47fc53de23e8205089d71109a2e2348acd/isaaclab_arena/embodiments/g1/g1.py
[g1_t2]: https://github.com/isaac-sim/IsaacLab-Arena/blob/7d75c95934c51a0318c957a8831e862ca43c53b5/isaaclab_arena/embodiments/g1/g1.py
[agile_repo]: https://github.com/nvidia-isaac/WBC-AGILE/tree/7259792cf10803aab814d101134d493d24c8f22f
[agile_license]: https://github.com/nvidia-isaac/WBC-AGILE/blob/7259792cf10803aab814d101134d493d24c8f22f/LICENCE
[agile_cfg]: https://github.com/isaac-sim/IsaacLab-Arena/blob/8b4a3a47fc53de23e8205089d71109a2e2348acd/isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/config/configs.py
[homie_cfg]: https://github.com/isaac-sim/IsaacLab-Arena/blob/7d75c95934c51a0318c957a8831e862ca43c53b5/isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/config/configs.py
[homie_stand]: https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/Arena/wbc_policy/models/homie_v2/stand.onnx
[homie_walk]: https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/Arena/wbc_policy/models/homie_v2/walk.onnx
[g1_usd]: https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/Samples/Groot/Robots/g1_29dof_with_hand_rev_1_0.usd
[g1_urdf]: https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/Arena/wbc_policy/robot_model/g1/g1_29dof_with_hand.urdf
[unitree_license]: https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/docs/licenses/assets/unitree-license.txt
[lab_licenses]: https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/docs/source/refs/license.rst
[lab_t1_license]: https://github.com/isaac-sim/IsaacLab/blob/e57379c634b42db5a0fe9f754341be6e2a7c7c43/LICENSE
[lab_t2_license]: https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/LICENSE
[gr00t_license]: https://github.com/NVIDIA/Isaac-GR00T/blob/e29d8fc50b0e4745120ae3fb72447986fe638aa6/LICENSE
[qwen_card]: https://huggingface.co/Qwen/Qwen3.8-27B-FP8/blob/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/README.md
[qwen_license]: https://huggingface.co/Qwen/Qwen3.8-27B-FP8/blob/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/LICENSE
[vllm_license]: https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/LICENSE

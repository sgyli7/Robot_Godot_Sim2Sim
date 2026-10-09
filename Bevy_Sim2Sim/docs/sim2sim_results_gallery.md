# Sim2Sim成果录像来源

2026-10-09。主页展示MicroDuck、Goose和G1各自成果；原速片段只缩小尺寸、降低采样率和量化颜色，不增加模拟动作。以下新GIF按用户明确要求作为主页展示媒体，置于 `assets/game/arts/game_play/textures/`；不用于运行时加载或物理配置。完整源码、权重及原始训练证据遵守备份目录规则。

| 片段 | 来源版本／区间 | 原片SHA256 | 展示GIF SHA256 |
| --- | --- | --- | --- |
| goose_move023_walk_run_stop | 0–33.02秒 | `0d62c3c1d463121ac391c886117e29548ae3794b7eb5aed4de8c69c2f2962cda` | `5ff411f4c9f000ccd8fc05aa400a1d2d2eaf7177f0724a1b742d6ac05a6f9ef8` |
| goose_move041_fast_left_right_stop | 041；完整0–29.02秒，GIF采样29.04秒 | `b95d729a97d2a15a267e12624a51aa5d79f2748c26268901789cff11d3f325c4` | `2c760981f02bec447ccafb91ccac51bf774a2765381464db42b0b577df2e5ebf` |
| goose_historical_turns_stop | 17–37秒 | `1d666f93addee26ef49222221410ea4da0b525bafac0c089370437d423b971ee` | `aef7aee28692f716d1a7ef1387ecd0ed09ef86d46da59df1f687edfc053c2787` |
| microduck_native012_move_cases | 0–20秒 | `e9ce12b1af2710b818416c333d249985b95644823f3109dca30653726de1bc21` | `5c59c6fbe2f9112775354a27af16f49bf4cdcb040aa4f538adcc5ea40863a8b5` |
| microduck_native010_front_back_recovery | 0–10秒 | `bc638928a80f2c2643ed92b1ddc3779679a15aae0040dc57ce5709068441c2e3` | `ee60446230f1eaba809daf9f36970df973a3c5c1771ac4edcac6c1735277ee26` |
| microduck_bevy_game009_roller_preview | 0–19秒 | `21e4c16c946ecefc4ac03d98203b0757c3768187add2b348358fb9a013cc2b1d` | `bba7caf27e2fae4d1a105ce160521ac16838931d256f5cc32ea873d44a2c5588` |
| microduck_legacy_godot_sit_stand | 0–13秒 | `03fe081d12e50e9eaae02f3a8abddf34d3dbac5261b819bdb03bbd2064e12d77` | `6acc400563e2947421b5527685eabf9adee18864431610c3ffee313b6abfa309` |
| microduck_legacy_godot_kick_right | 0–6秒 | `035d19cebdc9679bc81d0f65893525765496eb6151fb3822ecce4d9c84ecb776` | `e7eb11ff674ca9f8d5e047a9f8c09ea2835f3e0f326e18281a94bb367e7044c9` |
| microduck_legacy_godot_roulade | 0–6秒 | `6ff81bb4063056668c908803071954d19b14c4ffe28d64e91d7e7016883ea52b` | `54ed0ba9d455c21a133848f47f6d0afd2aeb70cae229b9491fe9d61f2c3b94c5` |

各段全部来源保留在同日备份 `goose_move_homepage_media_001/media_manifest.json`，含完整源路径、命令、尺寸、帧数、实际播放时长、范围和判分证据。媒体制作新增物理积分、神经网络推理、PPO更新均为0。MicroDuck八案例和前后恢复是多个独立原生轨迹的并排显示；轮滑与历史Godot片段来自实际窗口。Goose为自身保存物理轨迹的连续显示。

G1既有GIF与说明全部保留，当前中文操作、冻结十例和早期成果来自 [G1阶段记录](g1_milestone.md) 与 [操作说明](g1_station_session.md)，未改写其代码、模型或验收结论。

MicroDuck模型与衍生网格仍遵循上游Creative Commons BY-SA-NC来源限制；展示录像不改变模型授权。自研Goose源码／媒体与各第三方组件的许可分别维护。

新增041原片与转换收据保存在同日备份 `goose_move_publication_001/goose041_gallery_update/`。源端视频为真实1450控制Tick、1451帧（含初帧），200Hz积分/50Hz策略；GIF 900×346、363帧，总显示29.04秒，原片29.02秒，差20ms为显示采样取整，没有加速或新增物理。它保留自己的041快跑转弯开发成果，未替换023速度基线，也不授予全向移动或Bevy资格。

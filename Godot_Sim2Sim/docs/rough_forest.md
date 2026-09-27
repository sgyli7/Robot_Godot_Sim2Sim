# Rough Forest Play（Procedural Forest 旁路）

让 MicroDuck 在 **TerrainPatch3D**（来自 [godot-forest-demo](https://github.com/GamesNotDeveloped/godot-forest-demo)，AssetLib #5013）生成的崎岖地面上跑，**不改** `project.godot` 的 `run/main_scene`，默认 `main.tscn` 平地路径保持不变。

## 一次性准备

```bash
cd /path/to/Sai_Lab/Godot_Sim2Sim
./scripts/setup_forest_vendor.sh
# 若 generated/ 机器人场景缺失：
cd /path/to/Sai_Lab/Godot_Sim2Sim && uv run mjcf2godot
```

Vendor 克隆体积较大（约数百 MB），已 gitignore，勿提交。

## 验收命令（推荐）

在 `sim2sim` 目录：

```bash
cd /path/to/Sai_Lab/Godot_Sim2Sim
uv run sim2sim-play --scene res://scenes/rough_forest_play.tscn
```

平地对照（默认，应仍为盒状 Floor）：

```bash
uv run sim2sim-play
# 等价：uv run sim2sim-play --scene res://main.tscn
```

Python / 后端等价：

```python
from sim2sim.backends.godot_backend import GodotBackend
from sim2sim.godot_proc import spawn_godot
# GodotBackend(..., scene="res://scenes/rough_forest_play.tscn", headless=False)
# spawn_godot("res://scenes/rough_forest_play.tscn", headless=False, ...)
```

## 场景行为

- 根节点仍挂 `physics_server.gd`，`RobotHost` 生成路径与 `main.tscn` 相同。
- `World/Floor` **保留**（兼容 sole/jaw/sprung 与 floor checker 的节点路径），但 **collision 关闭、网格隐藏**。
- `World/ForestTerrain`：上游 `TerrainPatch3D.gd`，`generate_collision=true`，`collision_layer/mask=1`，ConcavePolygon 三角网格碰撞。
- `SpawnAlign`：地形 deferred 生成后把原点处地表对齐到 `y=0`，配合默认 `reset_z≈0.125`。
- 高度尺度按 **MicroDuck 体型** 调过（`height_scale=0.18`，约 ±18 cm），不是人类演示里 `18.93` 的山丘；要更陡可在编辑器里加大 `height_scale` / `frequency`。

## 已知限制

| 项 | 说明 |
|----|------|
| Godot 版本 | 上游 demo 标 4.6；本机工程为 **4.7.2 + Jolt**。TerrainPatch3D 为纯 GDScript，一般可直接跑；若遇 API 警告以编辑器输出为准。 |
| 许可 | Forest demo **CC BY 4.0**；模型/草/音乐等见 vendor `README.md` / `LICENSE`。使用需署名。 |
| 完整 test_biomes | 未整场景嵌入（含 Uniplayer、巨树、`res://generated/` 与 MicroDuck `generated/microduck` 冲突、体积与体型尺度问题）。本旁路只用地形脚本+地面材质。完整关卡可在 vendor 工程内单独打开 `levels/test_biomes.tscn`。 |
| 碰撞 | Concave 静态网格；小脚端可能对尖棱敏感。禁止用 SoftBody / HeightMap 蛋盒 spikes 当地形。 |
| physics_server.gd | **未改**；平地逻辑仍走 `World/Floor`（本场景 Floor 无碰撞）。 |
| Vendor | 需先跑 setup；符号链接：`scenery` `textures` `materials` `grass` `levels` `objects` `addons/gnd_biomes`。 |

## 改动文件一览

- `godot/scenes/rough_forest_play.tscn` — 旁路场景
- `godot/scenes/terrain_spawn_align.gd` — 原点地表对齐
- `godot/scenes/forest/TerrainPatch3D.gd` — 上游 TerrainPatch3D 的 MicroDuck fork（`MDTerrainPatch3D`，去掉对 `class_name Biomes` 的硬依赖；碰撞/噪声生成逻辑不变）
- `scripts/setup_forest_vendor.sh` — 统一转发入口；实现仍在 `godot/scripts/setup_forest_vendor.sh`
- `src/sim2sim/play.py` — 增加 `--scene`
- `godot/project.godot` — 默认不启用 `gnd_biomes`（vendor 未克隆时也能开工程）；**未改** `run/main_scene`

## 环境注意（zgx-1c05）

若 shell 里已设置 `SIM2SIM_ROOT` / `PYTHONPATH` 指向 speculative sandbox，
`sim2sim-play` 会用那套 Python + Godot 工程。旁路场景已同步到该 sandbox 的
`godot/scenes/`；以 MicroDuck 树为准时：

```bash
export SIM2SIM_ROOT=/path/to/Sai_Lab/Godot_Sim2Sim
export PYTHONPATH=/path/to/Sai_Lab/Godot_Sim2Sim/src
cd "$SIM2SIM_ROOT"
uv run sim2sim-play --scene res://scenes/rough_forest_play.tscn
```

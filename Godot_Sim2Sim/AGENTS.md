# Godot_Sim2Sim 维护入口

- 新增文件或移动路径前，读 [工程规则](godot_engineering_rules.md) 和 [目录导航](docs/directory_layout.md)，完成后运行 `python3 scripts/check_godot_structure.py`。既有文件原位修改不要求顺带迁移。
- 跨仓库资产、运行或副本更新前，读 [职责约定](docs/repository_ownership.md)，从清单确认维护源，记录兼容修改。
- 本轮目录整理使用隔离 worktree；Tick 调研、运行/控制、共享产物和其他 Agent 的工作保持原状。
- 明确的实现型委派使用 `cursor-grok` skill，一次前台调用；架构、物理/RL 结论和 observation/action contract 由主 Agent 判断。

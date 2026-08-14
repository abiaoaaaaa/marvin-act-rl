# Workspace maintenance instructions

Whenever a change affects a CLI argument, command entry point, default config,
dataset/checkpoint path, environment name, deployment sequence, or real-robot
safety procedure, update `ACT与RL100_DP数据转换训练测试指令汇总.md` in the same
change. Add a dated item under “指令同步记录”. Commands must use explicit Conda
interpreter paths, and long filesystem paths must never be split across shell
lines.

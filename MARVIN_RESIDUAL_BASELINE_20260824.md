# Marvin Residual 基线版本（2026-08-24）

该清单冻结“Human40 稠密 ACT + Residual TD3”在开发人工干预功能之前的可复现版本。

## Git 版本

三个仓库统一使用标签：`marvin-residual-baseline-20260824`。

| 仓库 | 基线提交 | 基线分支 |
|---|---|---|
| `/home/xlb/code/强化微调ACT` | `a77b6bd19478b9530fef608155736e0dddeb53b1`（源码快照；版本清单是其后续提交） | `feat/staged-online-rl-v1` |
| `/home/xlb/code/residual-offpolicy-rl` | `f17d8f783d488c26a2af6988ebd5ba909825eff9` | `snapshot/marvin-residual-baseline-20260824` |
| `/home/xlb/code_marvin/hil-serl` | `87b7c461b9413f8ec47e28fb24e874181676510e` | `snapshot/marvin-residual-baseline-20260824` |

## 模型身份

| 模型 | 路径 | SHA256 / step |
|---|---|---|
| Human40 稠密 ACT latest | `/home/xlb/code/强化微调ACT/outputs/marvin_usb_human40_act_dense_bc/latest/policy/model.safetensors` | `662c2a2e2c090a2db0f63d118aa864634e7eb3b2dd45773e807b0df725f6c9ab` |
| Residual actor latest | `/home/xlb/code/强化微调ACT/outputs/marvin_usb_human40_residual_td3_chunk20_exec5/deployment/residual_actor_latest.pt` | `bb576909f53a5d26a0199812aee3ebc8291aea90f1b7e3568b28e3a8953de2a6`；`global_step=45350` |
| 离线数据统计 | `marvin_usb_rl` | `380e8ba5ed368cd478c325a518a7dbfddd10e52434635006181ac440ecfe5dfa` |

模型、数据集、replay、W&B、缓存和 PDF 不纳入 Git，由上述路径与哈希识别。

## 固定模型真机结果

相同初始位置与相同 ACT base 下，各测试 50 回合：

| 测试 | 成功率 | 成功回合平均 step |
|---|---:|---:|
| ACT-only | 11/50（22%） | 133.0 |
| ACT + Residual step 45350 | 41/50（82%） | 95.49 |

对应目录：

- `/home/xlb/code/强化微调ACT/outputs/eval_human40_act_base_20260821`
- `/home/xlb/code/强化微调ACT/outputs/eval_human40_residual_step45350_20260821`

在线训练 replay 最后 100 回合（episode 335–434）成功 91 回合，在线成功率 91%。该数字来自持续更新策略，不替代固定模型测试结果。

## 验证记录

- 主工作区：`10 passed`。
- Residual 增量缓存：`5 passed`。
- 三个仓库：`git diff --check` 通过。
- HIL-SERL 修改脚本：10个文件完成语法编译。
- 依赖完整 LeRobot 的两个主工作区测试未执行：`lerobot_xlb` 环境没有 pytest；Residual Python 3.10 无法解析当前 LeRobot 使用的 Python 3.12 `type` 语法。

## 后续开发边界

人工干预功能从该统一标签派生新分支，不修改或移动本基线标签。回归时同时检出三个仓库的该标签，并核对两个模型 SHA256。

# ACT直接强化学习微调项目

## 边界

`act_rl`直接更新LeRobot ACT参数，但不修改LeRobot源码；真机仍只通过现有
Marvin HTTP Bridge调用HIL-SERL。核心IQL、Replay和网络不导入LeRobot、RL-100或
Marvin SDK。

```text
LeRobot ACT checkpoint ──► LeRobotACTAdapter ──► IQL/AWAC actor update
                                      ▲
LeRobot RL dataset ──► ChunkTransition├──► Q(s, chunk), V(s)
HIL-SERL Bridge ──► online Replay ─────┘
```

## 当前实现

- `src/act_rl/types.py`：稳定的chunk transition/batch协议。
- `src/act_rl/datasets/lerobot_chunk.py`：LeRobot帧到h-step chunk transition。
- `src/act_rl/policies/lerobot_act.py`：逐样本优势加权ACT L1、CVAE KL、冻结策略锚定。
- `src/act_rl/algorithms/iql.py`：Twin chunk-Q、expectile V和优势加权ACT更新。
- `src/act_rl/replay.py`：离线/在线Replay与按比例混合采样。
- `src/act_rl/environments/bridge.py`：不依赖Gym的Bridge适配。
- `src/act_rl/online_store.py`：崩溃安全的一回合一文件在线Replay。
- `collect_online.py`：ACT提出完整chunk，经Bridge执行前缀并记录真实h-step transition。
- `train_offline.py`：critic warm-up后直接更新ACT，并导出标准`pretrained_model`。

在线采集和学习是两个独立进程/命令：collector只控制环境并写append-only Replay；
learner只读取离线与在线Replay并发布candidate checkpoint，不连接机器人。candidate
必须通过验收后才能手工提升为collector使用的下一版策略。

## 数据要求

现有dense-gripper数据是成功轨迹筛选集，只适合做管线smoke test。正式IQL必须建立
包含失败、安全终止和人工接管轨迹的outcome manifest。禁止把未知轨迹默认标成成功。

### 正式ACT-RL数据转换

新转换器与BC转换器分开。它保留全部完整在线回合（成功和失败），排除无法构造可靠
终点的不完整尾段，写入原始逐步reward、terminated、success和intervention。默认还会
把夹爪转换成与当前ACT一致的持续目标标签。

根据之前BC转换报告，同一批buffer预计有788个完整在线回合，其中成功670、失败118；
再加入20个reference demos后预计共808回合。实际结果以新转换报告为准。

先只分析，不写数据：

```bash
cd /home/xlb/code/强化微调ACT

PYTHONPATH=$PWD:/home/xlb/code/lerobot/src \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python \
  convert_marvin_buffers_to_act_rl.py \
  --checkpoint-dir /home/xlb/code_marvin/hil-serl/examples/experiments/marvin_usb_insertion/checkpoints_scale10_acc300_alpha080_4cam_fresh_20260807 \
  --reference-demo-pkl /home/xlb/code_marvin/hil-serl/examples/demo_data/marvin_usb_insertion_20_demos_2026-08-07_10-16-21.pkl \
  --dry-run
```

正式转换：

```bash
PYTHONPATH=$PWD:/home/xlb/code/lerobot/src \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python \
  convert_marvin_buffers_to_act_rl.py \
  --checkpoint-dir /home/xlb/code_marvin/hil-serl/examples/experiments/marvin_usb_insertion/checkpoints_scale10_acc300_alpha080_4cam_fresh_20260807 \
  --reference-demo-pkl /home/xlb/code_marvin/hil-serl/examples/demo_data/marvin_usb_insertion_20_demos_2026-08-07_10-16-21.pkl \
  --output-root /home/xlb/code/强化微调ACT/datasets/marvin_usb_act_rl_offline \
  --repo-id xlb/marvin_usb_act_rl_offline \
  --image-writer-threads 4
```

输出中的关键文件：

```text
datasets/marvin_usb_act_rl_offline/
├── data/ + videos/                   # ACT输入、动作和RL逐步标签
├── meta/rl_outcomes.json             # 输出episode编号到成功/失败/来源的映射
└── act_rl_conversion_report.json     # 原始回合、筛选和夹爪转换审计
```

源buffer没有保存独立的`safety_abort`位，因此转换器会明确写为未知/false，不会把所有失败
伪装成安全终止。若后续采集需要区分碰撞、人工ESC和普通超时，应在新collector里直接记录。

下面的旧命令只用于现有成功集的临时smoke test；正式数据不需要执行它：

```bash
PYTHONPATH=src /home/xlb/miniconda3/envs/lerobot_xlb/bin/python \
  -m act_rl.scripts.build_replay \
  --dataset-root datasets/marvin_usb_act_success_dense_gripper \
  --output datasets/marvin_usb_act_success_dense_gripper/meta/rl_outcomes_success_only.json \
  --assume-all-success
```

## 离线训练命令（成功集只用于跑通）

```bash
cd /home/xlb/code/强化微调ACT

HF_HOME=$PWD/.cache/huggingface \
PYTHONPATH=$PWD/src:/home/xlb/code/lerobot/src \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python \
  -m act_rl.scripts.train_offline \
  --config configs/act_rl/marvin_offline_iql.json \
  --checkpoint outputs/act_dense_gripper_chunk30_seed1000_500k/checkpoints/115000/pretrained_model \
  --dataset-root datasets/marvin_usb_act_success_dense_gripper \
  --outcomes datasets/marvin_usb_act_success_dense_gripper/meta/rl_outcomes_success_only.json \
  --output-dir outputs/act_rl_offline_iql_smoke \
  --device cuda --steps 100 --critic-warmup-steps 50 --save-freq 100
```

正式转换完成后，将训练命令中的三项替换为：

```text
--dataset-root datasets/marvin_usb_act_rl_offline
--repo-id xlb/marvin_usb_act_rl_offline
--outcomes datasets/marvin_usb_act_rl_offline/meta/rl_outcomes.json
```

`chunk_size=30`是ACT预测和加权重建长度；`executed_steps=10`是Bellman target的真实
环境时间跨度；`critic_horizon=10`确保Q只看到真正执行的动作前缀。部署若改回执行1步，
RL配置、新采集Replay和`critic_horizon`也必须同步改为1。

## 后续里程碑

1. 运行正式转换并审计成功/失败、干预步数、逐步reward及异常超长回合。
2. 验证critic在留出集上能稳定排序成功/失败，不能只看训练loss。
3. 离线回放比较原ACT和微调ACT，检查动作偏移、夹爪状态和越界率。
4. mock Bridge连续采样，将transition写入online Replay。
5. 开启`MixedReplay`，先80%离线/20%在线，再逐步到50/50。
6. candidate checkpoint离线验收后才提升为真机collector的`latest_safe`。

现阶段不需要新增clone。RLinf的IQL、Residual的RLPD混合Replay、Evo-RL的优势标注和
现有Bridge已经提供足够参考；没有复制第三方源文件，因此也没有新增许可证混合问题。

## 在线采集与混合训练

先启动安全mock Bridge，然后采集。collector只写Replay，不更新模型：

```bash
TORCH_HOME=$PWD/.cache/torch \
PYTHONPATH=$PWD/src:/home/xlb/code/lerobot/src \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python -m act_rl.scripts.collect_online \
  --checkpoint outputs/act_rl_offline_iql/checkpoints/step_00010000/pretrained_model \
  --output-dir outputs/act_rl_online_replay \
  --host 127.0.0.1 --port 8765 --device cuda \
  --episodes 1 --max-decisions 20 --executed-steps 10
```

混合训练进程不连接Bridge：

```bash
TORCH_HOME=$PWD/.cache/torch HF_HOME=$PWD/.cache/huggingface \
PYTHONPATH=$PWD/src:/home/xlb/code/lerobot/src \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python -m act_rl.scripts.train_online \
  --config configs/act_rl/marvin_offline_iql.json \
  --model-checkpoint outputs/act_rl_offline_iql/checkpoints/step_00010000/pretrained_model \
  --learner-checkpoint outputs/act_rl_offline_iql/checkpoints/step_00010000.pt \
  --dataset-root datasets/marvin_usb_act_success_dense_gripper \
  --outcomes /ABS/PATH/TO/failure_aware_rl_outcomes.json \
  --online-replay outputs/act_rl_online_replay \
  --output-dir outputs/act_rl_online_mixed \
  --device cuda --steps 10000
```

真机collector还需在命令末尾显式增加`--allow-real-robot`，但只有在失败数据、critic
排序、mock连续回合、动作越界率和checkpoint回退都验收后才能使用。

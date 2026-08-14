# Marvin ACT、ACT-RL 与 RL-100 Diffusion Policy 操作手册

本文汇总当前 Marvin USB 插入项目从 HIL-SERL buffer 到 LeRobot ACT、ACT直接强化
学习微调、RL-100 Diffusion Policy（以下简称 DP）的数据转换、训练、离线检查、Mock
测试和真机测试指令。

维护约定：本文是当前项目的唯一完整指令汇总。后续根据运行反馈修改脚本、参数、目录或
安全流程时，应在同一次代码修改中同步更新本文。最后同步日期：2026-08-12。

Shell换行规则：一个文件路径绝对不能从中间换行；使用多行命令时，反斜杠 `\` 必须是
该行最后一个字符。即使终端提示符显示 `(rl100)`，下面使用绝对路径调用
`.../envs/lerobot_xlb/bin/python` 的命令仍由 `lerobot_xlb` 环境执行。

所有路径和参数均对应当前机器上的代码：

```text
工作目录          /home/xlb/code/强化微调ACT
LeRobot           /home/xlb/code/lerobot
RL-100            /home/xlb/code/RL-100
HIL-SERL          /home/xlb/code_marvin/hil-serl
ACT Conda         /home/xlb/miniconda3/envs/lerobot_xlb
机器人 Conda      /home/xlb/miniconda3/envs/hilserl
DP Conda          /home/xlb/miniconda3/envs/rl100
```

## 1. 统一数据契约

ACT、ACT-RL 和 DP 使用相同的机器人数据契约：

```text
频率          10 Hz
相机          side_policy, wrist_1buchong, wrist_1, wrist_2
图像          uint8 RGB, 128 x 128 x 3
state         float32[24]
action        float32[5] = [dx, dy, dz, drz, gripper]
动作范围      [-1, 1]
```

24维 state 顺序为：

```text
gripper_pose(1)
+ last_action(5)
+ tcp_force(3)
+ tcp_pose(6)
+ tcp_torque(3)
+ tcp_vel(6)
```

当前推荐使用“持续目标夹爪”数据：夹爪关闭期间持续标记 `-1`，打开期间持续
标记 `+1`。原始稀疏版本仍保留，用于对照实验。

当前BC成功集数据统计：

```text
总回合             584
总帧数             49,574
reference demo      20 回合 / 2,497 帧
online success      564 回合 / 47,077 帧
异常排除回合        episode 359
DP/ACT 实际使用      583 回合
```

如果以后确认 episode 359 已修复，必须同时从 ACT 的
`--dataset.exclude_episodes` 和 DP 的 `--exclude-episodes` 中移除，保证对比公平。

## 2. 环境和部署包安装

### 2.1 ACT 环境

```bash
cd /home/xlb/code/强化微调ACT

PYTHONPATH=/home/xlb/code/lerobot/src \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python \
  -m pip install -e . --no-deps
```

### 2.2 HIL-SERL 机器人环境

```bash
cd /home/xlb/code/强化微调ACT

/home/xlb/miniconda3/envs/hilserl/bin/python \
  -m pip install -e . --no-deps
```

### 2.3 RL100 环境

环境创建完成后执行：

```bash
cd /home/xlb/code/强化微调ACT

/home/xlb/miniconda3/envs/rl100/bin/python \
  -m pip install -e . --no-deps
```

数据转换至少需要：

```bash
/home/xlb/miniconda3/envs/rl100/bin/python -c \
  "import av, pyarrow, zarr, numcodecs; print('conversion dependencies OK')"
```

如果 rl100 环境缺少转换依赖：

```bash
/home/xlb/miniconda3/envs/rl100/bin/python -m pip install \
  "zarr==2.12.0" \
  "numcodecs==0.10.2" \
  "pyarrow==14.0.2"
```

## 3. 原始 Buffer 转换为 LeRobot 数据集

原始数据目录：

```text
/home/xlb/code_marvin/hil-serl/examples/experiments/marvin_usb_insertion/
  checkpoints_scale10_acc300_alpha080_4cam_fresh_20260807
```

完整人工 demo：

```text
/home/xlb/code_marvin/hil-serl/examples/demo_data/
  marvin_usb_insertion_20_demos_2026-08-07_10-16-21.pkl
```

筛选规则为：完整回合、成功，并且 online 回合长度严格小于人工 demo 最大长度。

### 3.1 只分析，不写数据

```bash
cd /home/xlb/code/强化微调ACT

/home/xlb/miniconda3/envs/hilserl/bin/python \
  convert_marvin_buffers_to_lerobot.py \
  --checkpoint-dir /home/xlb/code_marvin/hil-serl/examples/experiments/marvin_usb_insertion/checkpoints_scale10_acc300_alpha080_4cam_fresh_20260807 \
  --reference-demo-pkl /home/xlb/code_marvin/hil-serl/examples/demo_data/marvin_usb_insertion_20_demos_2026-08-07_10-16-21.pkl \
  --dry-run
```

### 3.2 正式生成稀疏夹爪 LeRobot 数据集

输出目录必须不存在；脚本不会覆盖已有数据集。

```bash
cd /home/xlb/code/强化微调ACT

PYTHONPATH=/home/xlb/code/lerobot/src \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python \
  convert_marvin_buffers_to_lerobot.py \
  --checkpoint-dir /home/xlb/code_marvin/hil-serl/examples/experiments/marvin_usb_insertion/checkpoints_scale10_acc300_alpha080_4cam_fresh_20260807 \
  --reference-demo-pkl /home/xlb/code_marvin/hil-serl/examples/demo_data/marvin_usb_insertion_20_demos_2026-08-07_10-16-21.pkl \
  --output-root /home/xlb/code/强化微调ACT/datasets/marvin_usb_act_success \
  --repo-id xlb/marvin_usb_act_success
```

转换报告：

```text
/home/xlb/code/强化微调ACT/datasets/marvin_usb_act_success/conversion_report.json
```

### 3.3 生成持续目标夹爪数据集（推荐）

该步骤复用四路视频的硬链接，不重复占用视频空间，只改 Parquet 中的 action 和
state.last_action.gripper。

```bash
cd /home/xlb/code/强化微调ACT

PYTHONPATH=/home/xlb/code/lerobot/src \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python \
  convert_gripper_to_dense_target.py \
  --source-root /home/xlb/code/强化微调ACT/datasets/marvin_usb_act_success \
  --output-root /home/xlb/code/强化微调ACT/datasets/marvin_usb_act_success_dense_gripper \
  --repo-id xlb/marvin_usb_act_success_dense_gripper \
  --video-mode hardlink
```

转换结果应为：

```text
closed_-1      31,256 帧
open_+1        18,318 帧
other          0 帧
```

转换报告：

```text
/home/xlb/code/强化微调ACT/datasets/marvin_usb_act_success_dense_gripper/
  dense_gripper_conversion_report.json
```

## 4. ACT 训练

### 4.1 推荐：持续目标夹爪，从头训练

```bash
cd /home/xlb/code/强化微调ACT

WANDB_PROJECT=marvin-act-gripper-compare \
TORCH_HOME=/home/xlb/code/强化微调ACT/.cache/torch \
PYTHONPATH=/home/xlb/code/lerobot/src \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=xlb/marvin_usb_act_success_dense_gripper \
  --dataset.root=/home/xlb/code/强化微调ACT/datasets/marvin_usb_act_success_dense_gripper \
  '--dataset.exclude_episodes=[359]' \
  --dataset.eval_split=0.1 \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=30 \
  --policy.n_action_steps=1 \
  --batch_size=8 \
  --num_workers=4 \
  --steps=200000 \
  --eval_steps=2000 \
  --env_eval_freq=0 \
  --log_freq=200 \
  --save_freq=5000 \
  --seed=1000 \
  --wandb.enable=true \
  --wandb.project=marvin-act-gripper-compare \
  --wandb.disable_artifact=true \
  --output_dir=/home/xlb/code/强化微调ACT/outputs/act_dense_gripper_chunk30_seed1000_200k
```

说明：

- `chunk_size=30`：训练时预测未来30步动作。
- `n_action_steps=1`：部署默认每次只执行1步，再重新观测。
- `eval_split=0.1`：训练开始时按回合固定划分90%训练和10%验证。
- `eval_steps=2000`：每2000个训练 step 计算一次 hold-out `eval_loss`。
- `env_eval_freq=0`：训练中不连接真机，eval 不是任务成功率测试。
- 若显存不足，先把 `batch_size` 改成4。

### 4.2 稀疏夹爪对照训练

使用相同训练参数，只替换以下三项：

```bash
--dataset.repo_id=xlb/marvin_usb_act_success
--dataset.root=/home/xlb/code/强化微调ACT/datasets/marvin_usb_act_success
--output_dir=/home/xlb/code/强化微调ACT/outputs/act_sparse_gripper_chunk30_seed1000_200k
```

其余参数和 seed 必须保持一致，才能公平对比稀疏标签与持续目标标签。

### 4.3 ACT 断点续训

`--config_path` 指向要恢复的 checkpoint 中的 `pretrained_model` 目录或其中的
`train_config.json`：

```bash
cd /home/xlb/code/强化微调ACT

PYTHONPATH=/home/xlb/code/lerobot/src \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python \
  -m lerobot.scripts.lerobot_train \
  --config_path=/home/xlb/code/强化微调ACT/outputs/act_dense_gripper_chunk30_seed1000_200k/checkpoints/100000/pretrained_model \
  --resume=true \
  --steps=200000
```

恢复时不要换数据集、模型结构或输出目录。`steps` 是最终目标总 step，不是额外训练
step 数。

## 5. ACT 测试

以下示例使用占位变量。把 `ACT_CKPT` 改成实际 checkpoint：

```bash
ACT_CKPT=/home/xlb/code/强化微调ACT/outputs/act_dense_gripper_chunk30_seed1000_200k/checkpoints/200000/pretrained_model
```

### 5.1 只加载 checkpoint，不连接 Bridge

```bash
ACT_CKPT=/ABS/PATH/TO/pretrained_model \
PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/lerobot/src \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python -c \
  "import os; from lerobot_robot_marvin.act_rollout import ACTPolicyRunner; r=ACTPolicyRunner(os.environ['ACT_CKPT'],'cuda','Insert the USB plug',n_action_steps=1); print('ACT checkpoint OK')"
```

### 5.2 启动安全 Mock Bridge

终端一：

```bash
PYTHONPATH=/home/xlb/code/强化微调ACT/src \
/home/xlb/miniconda3/envs/hilserl/bin/python \
  -m lerobot_robot_marvin.bridge_server \
  --backend mock \
  --mock-max-steps 20 \
  --mock-success-at-step 20 \
  --host 127.0.0.1 \
  --port 8765
```

终端二：

```bash
ACT_CKPT=/ABS/PATH/TO/pretrained_model \
TORCH_HOME=/home/xlb/code/强化微调ACT/.cache/torch \
PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/lerobot/src \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python \
  -m lerobot_robot_marvin.act_rollout \
  --checkpoint "${ACT_CKPT}" \
  --host 127.0.0.1 \
  --port 8765 \
  --device cuda \
  --episodes 1 \
  --max-steps 20 \
  --n-action-steps 1 \
  --fps 10
```

稳定后可对比执行10步 chunk 前缀：

```bash
--n-action-steps 10
```

它不是把训练 `chunk_size` 改成10；模型仍预测30步，只连续执行前10步后再推理。

### 5.3 ACT 真机测试

先启动第10节的真机 Bridge，然后在 ACT 终端运行：

```bash
ACT_CKPT=/ABS/PATH/TO/pretrained_model \
TORCH_HOME=/home/xlb/code/强化微调ACT/.cache/torch \
PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/lerobot/src \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python \
  -m lerobot_robot_marvin.act_rollout \
  --checkpoint "${ACT_CKPT}" \
  --host 127.0.0.1 \
  --port 8765 \
  --device cuda \
  --episodes 1 \
  --max-steps 5 \
  --n-action-steps 1 \
  --fps 10 \
  --allow-real-robot \
  --log-level INFO
```

第一次只运行5步和1个 action step，确认方向、夹爪和安全边界后再增加。

## 6. ACT直接强化学习微调（ACT-RL）

ACT-RL不使用第3节的成功轨迹BC筛选集。它需要保留完整的成功和失败回合、真实逐步
reward以及人工干预标记。新转换器默认同步生成持续目标夹爪标签。

### 6.1 ACT-RL数据转换dry-run

只分析，不写数据。下面两个长路径必须各自保持在同一行：

```bash
cd /home/xlb/code/强化微调ACT

PYTHONPATH="$PWD:/home/xlb/code/lerobot/src" \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python \
  convert_marvin_buffers_to_act_rl.py \
  --checkpoint-dir "/home/xlb/code_marvin/hil-serl/examples/experiments/marvin_usb_insertion/checkpoints_scale10_acc300_alpha080_4cam_fresh_20260807" \
  --reference-demo-pkl "/home/xlb/code_marvin/hil-serl/examples/demo_data/marvin_usb_insertion_20_demos_2026-08-07_10-16-21.pkl" \
  --dry-run
```

根据原BC转换报告，预计在线buffer包含788个完整回合，其中成功670、失败118，另有5个
不完整片段被排除。加入20个reference demos后预计输出808回合；以本次转换报告为准。

### 6.2 正式生成ACT-RL离线数据集

输出目录必须不存在。该过程会重新编码四路视频，需要等待完成，不要中途`kill -9`。

```bash
cd /home/xlb/code/强化微调ACT

PYTHONPATH="$PWD:/home/xlb/code/lerobot/src" \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python \
  convert_marvin_buffers_to_act_rl.py \
  --checkpoint-dir "/home/xlb/code_marvin/hil-serl/examples/experiments/marvin_usb_insertion/checkpoints_scale10_acc300_alpha080_4cam_fresh_20260807" \
  --reference-demo-pkl "/home/xlb/code_marvin/hil-serl/examples/demo_data/marvin_usb_insertion_20_demos_2026-08-07_10-16-21.pkl" \
  --output-root "/home/xlb/code/强化微调ACT/datasets/marvin_usb_act_rl_offline" \
  --repo-id xlb/marvin_usb_act_rl_offline \
  --image-writer-threads 4
```

输出：

```text
datasets/marvin_usb_act_rl_offline/
├── data/ 和 videos/                 四路图像、state、action和逐步RL标签
├── meta/rl_outcomes.json            episode成功/失败、来源和干预步数
└── act_rl_conversion_report.json    回合筛选及夹爪转换审计
```

新增字段为`rl.reward`、`rl.terminated`、`rl.success`、`rl.intervention`和
`rl.safety_abort`。旧HIL-SERL buffer没有独立保存安全终止原因，因此转换器不会猜测它；
旧数据中的`rl.safety_abort`统一为false，并在报告中注明限制。

如需保留原始稀疏夹爪命令，增加`--keep-sparse-gripper`。当前ACT checkpoint使用持续
目标夹爪，因此正式主实验不要增加该参数。

### 6.3 ACT-RL离线IQL训练

先训练Twin chunk-Q和expectile V；critic warm-up结束后，通过优势加权L1直接更新ACT
参数，同时保留CVAE KL与冻结ACT prior约束。

```bash
cd /home/xlb/code/强化微调ACT

TORCH_HOME="$PWD/.cache/torch" \
HF_HOME="$PWD/.cache/huggingface" \
PYTHONPATH="$PWD/src:/home/xlb/code/lerobot/src" \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python \
  -m act_rl.scripts.train_offline \
  --config configs/act_rl/marvin_offline_iql.json \
  --checkpoint outputs/act_dense_gripper_chunk30_seed1000_500k/checkpoints/115000/pretrained_model \
  --dataset-root datasets/marvin_usb_act_rl_offline \
  --repo-id xlb/marvin_usb_act_rl_offline \
  --outcomes datasets/marvin_usb_act_rl_offline/meta/rl_outcomes.json \
  --exclude-episodes 556 \
  --output-dir outputs/act_rl_offline_iql \
  --device cuda \
  --steps 50000 \
  --critic-warmup-steps 10000 \
  --save-freq 5000 \
  --num-workers 4 \
  --log-freq 50
```

当前配置：batch size为8；ACT预测并加权重建30步；Bellman target和critic horizon均为
实际执行前10步。critic第一阶段只使用24维state，而ACT仍然使用四路图像和24维state。
正式数据检查发现输出episode 556只有1帧且立即成功，因此训练命令显式排除它。

### 6.4 检查ACT-RL checkpoint

```bash
PYTHONPATH=/home/xlb/code/强化微调ACT/src \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python \
  -m act_rl.scripts.inspect_checkpoint \
  /home/xlb/code/强化微调ACT/outputs/act_rl_offline_iql/checkpoints/step_00010000.pt
```

标准部署模型位于：

```text
outputs/act_rl_offline_iql/checkpoints/step_00010000/pretrained_model
```

它可以直接使用第5节的ACT checkpoint加载、Mock Bridge和真机测试命令。

### 6.5 离线回放比较原ACT和RL-ACT

该命令不连接Bridge，比较微调ACT与原ACT的动作L1、策略偏移、成功/失败Q值和动作越界。
这是训练分布诊断，不是独立测试集成功率：

```bash
cd /home/xlb/code/强化微调ACT

TORCH_HOME="$PWD/.cache/torch" \
HF_HOME="$PWD/.cache/huggingface" \
PYTHONPATH="$PWD/src:/home/xlb/code/lerobot/src" \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python \
  -m act_rl.scripts.evaluate_offline \
  --config configs/act_rl/marvin_offline_iql.json \
  --base-checkpoint outputs/act_dense_gripper_chunk30_seed1000_500k/checkpoints/115000/pretrained_model \
  --model-checkpoint outputs/act_rl_offline_iql/checkpoints/step_00050000/pretrained_model \
  --learner-checkpoint outputs/act_rl_offline_iql/checkpoints/step_00050000.pt \
  --dataset-root datasets/marvin_usb_act_rl_offline \
  --repo-id xlb/marvin_usb_act_rl_offline \
  --outcomes datasets/marvin_usb_act_rl_offline/meta/rl_outcomes.json \
  --exclude-episodes 556 \
  --device cuda \
  --batch-size 8 \
  --max-samples 2048 \
  --num-workers 4 \
  --output outputs/act_rl_offline_iql/eval_step_50000.json
```

将两个checkpoint参数和输出文件中的`50000`同步替换为`15000、20000……45000`，可公平
比较各阶段。`10000`尚未更新ACT，可作为critic warm-up基线。

### 6.6 Mock Bridge在线采集

先使用第5.2节启动Mock Bridge。collector只执行策略并追加Replay，不更新模型：

```bash
cd /home/xlb/code/强化微调ACT

TORCH_HOME="$PWD/.cache/torch" \
PYTHONPATH="$PWD/src:/home/xlb/code/lerobot/src" \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python \
  -m act_rl.scripts.collect_online \
  --checkpoint outputs/act_rl_offline_iql/checkpoints/step_00010000/pretrained_model \
  --output-dir outputs/act_rl_online_replay \
  --host 127.0.0.1 \
  --port 8765 \
  --device cuda \
  --episodes 1 \
  --max-decisions 20 \
  --executed-steps 10
```

在线Replay按“一回合一个NPZ”追加写入，已有episode不会被覆盖。

### 6.7 离线/在线混合更新

learner只读取Replay，不连接Bridge。默认`offline_fraction=0.5`：

```bash
cd /home/xlb/code/强化微调ACT

TORCH_HOME="$PWD/.cache/torch" \
HF_HOME="$PWD/.cache/huggingface" \
PYTHONPATH="$PWD/src:/home/xlb/code/lerobot/src" \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python \
  -m act_rl.scripts.train_online \
  --config configs/act_rl/marvin_offline_iql.json \
  --model-checkpoint outputs/act_rl_offline_iql/checkpoints/step_00010000/pretrained_model \
  --learner-checkpoint outputs/act_rl_offline_iql/checkpoints/step_00010000.pt \
  --dataset-root datasets/marvin_usb_act_rl_offline \
  --repo-id xlb/marvin_usb_act_rl_offline \
  --outcomes datasets/marvin_usb_act_rl_offline/meta/rl_outcomes.json \
  --exclude-episodes 556 \
  --online-replay outputs/act_rl_online_replay \
  --output-dir outputs/act_rl_online_mixed \
  --device cuda \
  --steps 10000 \
  --save-freq 1000 \
  --num-workers 2
```

### 6.8 真机在线采集安全门槛

真机collector必须先启动第10节的hardware Bridge，并在第6.6节命令末尾显式增加
`--allow-real-robot`。它会实际reset并连续执行动作。只有完成失败数据审计、critic排序、
Mock连续回合、动作越界率和checkpoint回退验证后才允许使用。在线learner产生candidate，
不会自动替换collector当前策略。

### 6.9 ACT与RL-ACT执行长度对比

详细指标、last/offline-best/real-best定义和记录模板见
`ACT强化微调详细审计与测试方案.md`。两个模型的网络chunk均为30；以下只改变每次推理后
实际执行的动作前缀，按`1 -> 3 -> 5 -> 10`逐级测试。每一级先用`--max-steps 5`
完成安全筛查，确认没有ESC、危险力峰值和连续动作裁剪后，才增加回合长度。

严格算法归因时，原ACT必须使用RL初始化所对应的115000 checkpoint：

```bash
cd /home/xlb/code/强化微调ACT

TORCH_HOME=/home/xlb/code/强化微调ACT/.cache/torch PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/lerobot/src /home/xlb/miniconda3/envs/lerobot_xlb/bin/python -m lerobot_robot_marvin.act_rollout \
  --checkpoint /home/xlb/code/强化微调ACT/outputs/act_dense_gripper_chunk30_seed1000_500k/checkpoints/115000/pretrained_model \
  --host 127.0.0.1 \
  --port 8765 \
  --device cuda \
  --episodes 3 \
  --max-steps 5 \
  --n-action-steps 1 \
  --fps 10 \
  --allow-real-robot \
  --log-level INFO
```

当前50,000步RL-ACT候选：

```bash
cd /home/xlb/code/强化微调ACT

TORCH_HOME=/home/xlb/code/强化微调ACT/.cache/torch PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/lerobot/src /home/xlb/miniconda3/envs/lerobot_xlb/bin/python -m lerobot_robot_marvin.act_rollout \
  --checkpoint /home/xlb/code/强化微调ACT/outputs/act_rl_offline_iql/checkpoints/step_00050000/pretrained_model \
  --host 127.0.0.1 \
  --port 8765 \
  --device cuda \
  --episodes 3 \
  --max-steps 5 \
  --n-action-steps 1 \
  --fps 10 \
  --allow-real-robot \
  --log-level INFO
```

完成`n_action_steps=1`后，只将两个命令中的`--n-action-steps 1`依次改为`3、5、10`；
模型、初始位姿、回合数、控制频率、成功判定和其他参数保持不变。不要把网络
`chunk_size=30`误认为真机必须执行30步，也不要仅因critic horizon为10就默认部署10步。

原ACT的`200000`是last，`115000`是RL严格基座，二者都不自动等于best。正式部署应先在
固定episode级validation上选offline-best，再用真机成功率、安全终止率、干预率、峰值力和
回合时间选real-best。

## 7. LeRobot 数据转换为 RL-100 Zarr

DP 推荐直接使用持续目标夹爪 LeRobot 数据集，不再读取原始大 buffer。

### 7.1 先转换5个回合进行冒烟测试

输出目录必须不存在。

```bash
cd /home/xlb/code/强化微调ACT

/home/xlb/miniconda3/envs/rl100/bin/python \
  convert_lerobot_to_rl100_zarr.py \
  --source-root /home/xlb/code/强化微调ACT/datasets/marvin_usb_act_success_dense_gripper \
  --output-zarr /home/xlb/code/RL-100/RL-100/data/marvin_usb_bc_smoke.zarr \
  --exclude-episodes 359 \
  --max-episodes 5
```

预期结果：

```text
Episodes: 5
Frames: 629
```

### 7.2 全量转换

```bash
cd /home/xlb/code/强化微调ACT

/home/xlb/miniconda3/envs/rl100/bin/python \
  convert_lerobot_to_rl100_zarr.py \
  --source-root /home/xlb/code/强化微调ACT/datasets/marvin_usb_act_success_dense_gripper \
  --output-zarr /home/xlb/code/RL-100/RL-100/data/marvin_usb_bc.zarr \
  --exclude-episodes 359
```

Zarr 内容：

```text
data/state                       [T, 24]
data/action                      [T, 5]
data/rgb_side_policy             [T, 128, 128, 3]
data/rgb_wrist_1buchong          [T, 128, 128, 3]
data/rgb_wrist_1                 [T, 128, 128, 3]
data/rgb_wrist_2                 [T, 128, 128, 3]
meta/episode_ends
meta/source_episode_index
```

脚本使用临时目录写入，全部成功后才重命名为最终目录。不要在转换过程中强制终止。

## 8. RL-100 DP 训练

当前配置：

```text
四路相机共享 ResNet18 ImageNet 编码器
n_obs_steps          2
n_action_steps       16
horizon              17
DDIM inference       10 steps
state/action         limits 归一化到 [-1,1]
训练/验证            按回合 90% / 10%
top-k checkpoint     val_loss 最低
only_bc              true
WandB                offline
```

### 8.1 5回合、2 epoch 冒烟训练

```bash
conda activate rl100

DATASET_PATH=/home/xlb/code/RL-100/RL-100/data/marvin_usb_bc_smoke.zarr \
OUTPUT_DIR=/home/xlb/code/RL-100/RL-100/data/outputs/marvin_usb_bc_smoke \
EPOCHS=2 \
BATCH_SIZE=2 \
NUM_WORKERS=0 \
bash /home/xlb/code/RL-100/scripts/Diffusion/BC/2D/train_marvin_usb_bc.sh
```

冒烟阶段使用 `NUM_WORKERS=0`，发生异常时能看到原始 traceback。

### 8.2 全量正式训练

```bash
conda activate rl100

DATASET_PATH=/home/xlb/code/RL-100/RL-100/data/marvin_usb_bc.zarr \
OUTPUT_DIR=/home/xlb/code/RL-100/RL-100/data/outputs/marvin_usb_bc_seed1000 \
EPOCHS=200 \
BATCH_SIZE=8 \
NUM_WORKERS=4 \
bash /home/xlb/code/RL-100/scripts/Diffusion/BC/2D/train_marvin_usb_bc.sh
```

显存不足时改为：

```bash
BATCH_SIZE=4
```

checkpoint 默认位于：

```text
/home/xlb/code/RL-100/RL-100/data/outputs/marvin_usb_bc_seed1000/checkpoints/
```

其中：

- `latest.ckpt` 是最近一次 checkpoint；
- `epoch=....-val_loss=....ckpt` 是验证损失最低的 top-k checkpoint；
- checkpoint 内含 `model`、`ema_model`、配置和 normalizer。当前这批
  `marvin_usb_bc_seed1000_retrain_20260813` checkpoint 的检查与真机命令显式使用
  `--checkpoint-source model`；不要省略成默认 `auto`（它会优先选 `ema_model`）。

## 9. RL-100 DP 测试

把 `DP_CKPT` 指向实际 checkpoint，例如：

```bash
DP_CKPT=/home/xlb/code/RL-100/RL-100/data/outputs/marvin_usb_bc_seed1000/checkpoints/latest.ckpt
```

### 9.1 只检查 checkpoint，不连接 Bridge

```bash
DP_CKPT=/ABS/PATH/TO/latest.ckpt \
PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/RL-100/RL-100 \
/home/xlb/miniconda3/envs/rl100/bin/python \
  -m lerobot_robot_marvin.rl100_rollout \
  --checkpoint "${DP_CKPT}" \
  --checkpoint-source model \
  --device cuda:0 \
  --inspect-only
```

该命令检查四路相机、24维 state、5维 action、两帧观测、模型权重和 normalizer。

### 9.2 Zarr 离线回放

以下命令选 episode 0，逐帧运行20次预测，不连接 Bridge：

```bash
DP_CKPT=/ABS/PATH/TO/latest.ckpt \
PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/RL-100/RL-100 \
/home/xlb/miniconda3/envs/rl100/bin/python \
  -m lerobot_robot_marvin.rl100_replay \
  --checkpoint "${DP_CKPT}" \
  --checkpoint-source model \
  --dataset /home/xlb/code/RL-100/RL-100/data/marvin_usb_bc.zarr \
  --episode-index 0 \
  --max-predictions 20 \
  --device cuda:0 \
  --output /home/xlb/code/RL-100/RL-100/data/replay_episode000.npz
```

输出包括预测 chunk、示范 chunk、首动作各维 MAE、动作裁剪次数和推理耗时。

### 9.3 DP Mock Bridge 测试

终端一使用第5.2节的 Mock Bridge。终端二运行：

```bash
DP_CKPT=/ABS/PATH/TO/latest.ckpt \
PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/RL-100/RL-100 \
/home/xlb/miniconda3/envs/rl100/bin/python \
  -m lerobot_robot_marvin.rl100_rollout \
  --checkpoint "${DP_CKPT}" \
  --checkpoint-source model \
  --host 127.0.0.1 \
  --port 8765 \
  --device cuda:0 \
  --episodes 1 \
  --max-steps 20 \
  --actions-per-inference 1 \
  --motion-scale 0.2
```

`actions-per-inference=1` 表示模型虽然输出16步，但只执行第1步就重新推理。

### 9.4 DP 真机 dry-run

先启动第10节的真机 Bridge。dry-run 会执行真实 reset 和策略推理，但不会调用
`env.step()`。reset 本身可能移动机器人，因此仍属于真机操作：

```bash
DP_CKPT=/ABS/PATH/TO/latest.ckpt \
PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/RL-100/RL-100 \
/home/xlb/miniconda3/envs/rl100/bin/python \
  -m lerobot_robot_marvin.rl100_rollout \
  --checkpoint "${DP_CKPT}" \
  --checkpoint-source model \
  --host 127.0.0.1 \
  --port 8765 \
  --device cuda:0 \
  --episodes 1 \
  --max-steps 5 \
  --actions-per-inference 1 \
  --motion-scale 0.2 \
  --dry-run \
  --allow-real-robot
```

### 9.5 DP 第一次实际动作测试

```bash
DP_CKPT=/ABS/PATH/TO/latest.ckpt \
PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/RL-100/RL-100 \
/home/xlb/miniconda3/envs/rl100/bin/python \
  -m lerobot_robot_marvin.rl100_rollout \
  --checkpoint "${DP_CKPT}" \
  --checkpoint-source model \
  --host 127.0.0.1 \
  --port 8765 \
  --device cuda:0 \
  --episodes 1 \
  --max-steps 5 \
  --actions-per-inference 1 \
  --motion-scale 0.2 \
  --allow-real-robot \
  --log-level INFO
```

确认安全后按以下顺序递增：

```text
max_steps:              5 -> 10 -> 30 -> 完整回合
motion_scale:           0.2 -> 0.5 -> 1.0
actions_per_inference:  1 -> 2 -> 4
```

运动缩放只作用于 `dx,dy,dz,drz`，不会缩放夹爪 `-1/+1`。

## 9.6 RL-100 DP 第二阶段离线 RL

第二阶段不是继续训练 BC loss。它复用已经通过真机测试的 DP actor 和 BC 数据归一化器，
依次训练冻结视觉编码器的 IQL double-Q/value、K 步 learned dynamics，再以
UniO4/BPPO 的 clipped policy-gradient 小学习率更新 diffusion actor。部署只需要更新后的
actor、视觉编码器和原 BC normalizer，不加载 Q/V 或 dynamics。

正式 RL 数据使用完整的成功和失败回合：

```text
/home/xlb/code/强化微调ACT/datasets/marvin_usb_act_rl_offline
808 episodes / 90,119 frames
690 success / 118 failure
包含 rl.reward、rl.terminated、rl.success、rl.intervention、rl.safety_abort
```

BC 的 `marvin_usb_bc.zarr` 没有 reward/done，只用于复用 actor 的动作和状态归一化范围，
不能作为第二阶段 critic 数据。

### 9.6.1 安装第二阶段最小依赖

Marvin 是四相机 2D 离线训练，不需要 `mujoco_py`、`pytorch3d` 或 `d4rl`：

```bash
/home/xlb/miniconda3/envs/rl100/bin/python -m pip install --no-deps -e /home/xlb/code/RL-100/third_party/gym-0.21.0
/home/xlb/miniconda3/envs/rl100/bin/python -c "import gym; print(gym.__version__, gym.__file__)"
```

### 9.6.2 全量转换

以下转换已经于 2026-08-14 完成。转换器拒绝覆盖现有目录；不要在正式输出存在时重复运行：

```bash
/home/xlb/miniconda3/envs/rl100/bin/python /home/xlb/code/强化微调ACT/convert_lerobot_to_rl100_zarr.py --source-root /home/xlb/code/强化微调ACT/datasets/marvin_usb_act_rl_offline --output-zarr /home/xlb/code/RL-100/RL-100/data/marvin_usb_offline_rl.zarr --include-rl --return-gamma 0.99
```

正式 Zarr 包含 `state/action`、四路 `rgb_*`、`reward/done/timeout/return` 以及
`success/intervention/safety_abort`。下一观测直接从同一回合的未来帧构造，不复制四路
`next_image`，不会丢失信息。

### 9.6.3 K 步语义

`EXECUTED_STEPS=K` 是 critic、dynamics 和 actor 优化所使用的一次决策动作块长度；默认
`K=5`，与已经验证的真机 `--actions-per-inference 5` 一致。K 可选范围为 1 到 16。
`SEQUENCE_STRIDE` 只控制离线样本密度，正式默认等于 K。

部署时允许先用 `--actions-per-inference 1` 保守验收，但正式性能测试应使用与训练相同的
K。若长期部署改为另一个 N，应该设置 `EXECUTED_STEPS=N` 单独训练一个输出目录，不能把
K=5 的 critic 结果当作 K=16 的结果。

### 9.6.4 正式训练

默认参数：K=5、IQL 20 epochs、dynamics 最多50 epochs、BPPO 4000 updates、batch 8、
actor learning rate `1e-6`、IQL expectile `0.7`、gamma `0.99`、视觉编码器冻结。训练不接触
Bridge 或真机，W&B 默认为本地 offline。

```bash
EXECUTED_STEPS=5 OUTPUT_DIR=/home/xlb/code/RL-100/RL-100/data/outputs/marvin_usb_offline_rl_k5_seed1000 WANDB_MODE=offline /home/xlb/code/强化微调ACT/train_marvin_rl100_offline.sh
```

需要改变正式部署动作块时，K 和输出目录必须一起改变，例如 K=1：

```bash
EXECUTED_STEPS=1 OUTPUT_DIR=/home/xlb/code/RL-100/RL-100/data/outputs/marvin_usb_offline_rl_k1_seed1000 WANDB_MODE=offline /home/xlb/code/强化微调ACT/train_marvin_rl100_offline.sh
```

训练顺序和输出：

```text
BC latest.ckpt 的 model actor
-> IQL Q_bc_20.pt + value_20.pt
-> saved_models_full/dynamics.pth
-> best/model.pt + best/encoder.pt（按固定验证集动作 MSE 选择）
-> last/model.pt + last/encoder.pt
-> checkpoints/offline_best.ckpt + checkpoints/offline_last.ckpt
```

`offline_best.ckpt` 和 `offline_last.ckpt` 已合并 BC 配置与 normalizer，可以由现有 rollout
直接加载。中途停止后不要盲目用同一目录重启 BPPO；当前入口能复用完整保存的 critic 和
dynamics 阶段产物，但不承诺恢复到某个 BPPO update。正式训练应保持终端或 tmux 稳定。

### 9.6.5 部署前检查

```bash
TORCH_HOME=/home/xlb/code/强化微调ACT/.cache/torch PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/RL-100/RL-100 /home/xlb/miniconda3/envs/rl100/bin/python -m lerobot_robot_marvin.rl100_rollout --checkpoint /home/xlb/code/RL-100/RL-100/data/outputs/marvin_usb_offline_rl_k5_seed1000/checkpoints/offline_best.ckpt --checkpoint-source model --device cuda:0 --inspect-only
```

然后先按第10节启动 hardware Bridge。第一次只执行1步动作块、5个控制步并缩放运动：

```bash
PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/RL-100/RL-100 /home/xlb/miniconda3/envs/rl100/bin/python -m lerobot_robot_marvin.rl100_rollout --checkpoint /home/xlb/code/RL-100/RL-100/data/outputs/marvin_usb_offline_rl_k5_seed1000/checkpoints/offline_best.ckpt --checkpoint-source model --host 127.0.0.1 --port 8765 --device cuda:0 --episodes 1 --max-steps 5 --actions-per-inference 1 --motion-scale 0.2 --allow-real-robot --log-level INFO
```

确认方向、夹爪、安全边界和图像都正确后，再测试训练语义对应的 K=5：

```bash
PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/RL-100/RL-100 /home/xlb/miniconda3/envs/rl100/bin/python -m lerobot_robot_marvin.rl100_rollout --checkpoint /home/xlb/code/RL-100/RL-100/data/outputs/marvin_usb_offline_rl_k5_seed1000/checkpoints/offline_best.ckpt --checkpoint-source model --host 127.0.0.1 --port 8765 --device cuda:0 --episodes 1 --max-steps 150 --actions-per-inference 5 --motion-scale 1.0 --allow-real-robot --log-level INFO
```

先比较 `offline_best.ckpt` 和原 BC checkpoint；`offline_last.ckpt` 作为第二候选。离线动作
MSE 只能限制策略漂移，不等于成功率，最终仍以真机成功率、安全终止率、峰值力和干预率
选模。

## 9.7 RL-100 DP 第三阶段在线 PPO

第三阶段从已经通过真机验收的 `offline_best.ckpt` 开始，不再重复 BC、IQL、dynamics 或
离线 BPPO。入口复用 RL-100 原生的 `RL1002D.all_step_action_logprob` 和
`UniPPO.dp_align_update_no_share`，本工作区只补充 Marvin Bridge 环境、四相机 on-policy
buffer、断点与候选 checkpoint 导出。

每条在线 transition 的内容如下：

```text
当前2帧观测（4路uint8 RGB + 24维state）
-> 随机DDIM完整10步去噪轨迹及每步old log-prob
-> 执行前K=5个动作，每个动作均由策略产生
-> Bridge返回actual_action、reward、terminated/truncated和下一观测
-> K步折扣奖励形成一个chunk transition
```

不使用 SpaceMouse 动作覆盖，也不把人工动作写入 buffer。Bridge 返回的 `actual_action`
必须逐元素等于下发的策略动作，否则立即安全停止。RL-100 随机 DDIM 最后一步可能因噪声
略微超出数据集的 `[-1,1]` 支持；在线适配层会投影最终扩散状态并重算最后一步 log-prob，
Bridge 本身仍不静默裁剪动作。日志中的 `projected_values` 用于监控该情况。

网络更新关系：视觉 ResNet 编码器冻结；离线 IQL Q/value 和 learned dynamics 从第二阶段
加载，其中 Q 不在线更新；在线 value critic 由离线 value 初始化后更新；diffusion UNet
actor 以 PPO clipped objective 更新。默认 actor LR `1e-6`、critic LR `1e-4`、buffer 64个
K步 transition、mini-batch 8、每批 PPO 3 epochs、最多10次更新。buffer 严格绑定
`policy_version`，策略更新后清空，禁止把旧策略样本混入下一版策略。

普通真机步骤的动作平滑奖励在所有 `MarvinEnv` 任务中统一为 USB config 公式：首步不施加
动作差分惩罚，后续步骤为 `-0.01 * mean((action[:4] - previous_action[:4]) ** 2)`；夹爪
维度不计入。它替代旧的 `-0.05 * sum(...)`，因此相同动作差下惩罚缩小为旧值的1/20。
Bridge 的人工结果标签仍独立覆盖终止步：成功为 `+1.0`，失败为 `-0.5`。修改后必须重启
hardware Bridge 才会加载新奖励代码，已经保存的历史 buffer 不会被追溯重算。

### 9.7.1 Mock 采集检查

终端一启动 Mock Bridge：

```bash
PYTHONPATH=/home/xlb/code/强化微调ACT/src /home/xlb/miniconda3/envs/hilserl/bin/python -m lerobot_robot_marvin.bridge_server --backend mock --mock-max-steps 10 --mock-success-at-step 10 --host 127.0.0.1 --port 8765
```

终端二只采集、不更新参数。使用独立检查目录，避免把检查 buffer 混入正式训练：

```bash
COLLECT_ONLY=true ONLINE_BATCH_SIZE=2 MINI_BATCH_SIZE=1 MAX_EPISODE_STEPS=10 RESUME_ONLINE=false WANDB_MODE=disabled OUTPUT_DIR=/tmp/marvin-online-collect-check /home/xlb/code/强化微调ACT/train_marvin_rl100_online.sh
```

确认没有出现 `actual_action differs` 安全错误、buffer 达到 `2/2` 和
`Collect-only buffer is full; no optimizer step was executed`。需要在 Mock 上验证一次完整更新时：

```bash
COLLECT_ONLY=false ONLINE_BATCH_SIZE=2 MINI_BATCH_SIZE=1 PPO_EPOCHS=1 MAX_UPDATES=1 MAX_EPISODE_STEPS=10 RESUME_ONLINE=false WANDB_MODE=disabled OUTPUT_DIR=/tmp/marvin-online-update-check /home/xlb/code/强化微调ACT/train_marvin_rl100_online.sh
```

### 9.7.2 真机先采集检查

先按第10节启动 hardware Bridge。这里的“无人工干预”指不覆盖策略动作；奖励仍必须来自
Bridge 已配置的成功/失败键或可靠奖励分类器。第一次仅收集一个64条 on-policy buffer，
不执行梯度更新：

```bash
COLLECT_ONLY=true MAX_UPDATES=1 ALLOW_REAL_ROBOT=true CONFIRM_HARDWARE_RISK=I_UNDERSTAND_THIS_CONTROLS_A_REAL_ROBOT RESUME_ONLINE=false OUTPUT_DIR=/home/xlb/code/RL-100/RL-100/data/outputs/marvin_usb_online_collect_check_k5_seed1000 /home/xlb/code/强化微调ACT/train_marvin_rl100_online.sh
```

检查每回合 reward/终止信号、`executed=5`、动作方向、工作空间保护和
`projected_values` 后停止。该检查目录只用于审计，不继续正式训练。

### 9.7.3 正式在线训练与恢复

正式训练使用新的输出目录；默认每64条 transition 更新一次，共最多10次更新：

```bash
COLLECT_ONLY=false MAX_UPDATES=10 ALLOW_REAL_ROBOT=true CONFIRM_HARDWARE_RISK=I_UNDERSTAND_THIS_CONTROLS_A_REAL_ROBOT RESUME_ONLINE=true OUTPUT_DIR=/home/xlb/code/RL-100/RL-100/data/outputs/marvin_usb_online_rl_k5_seed1000 /home/xlb/code/强化微调ACT/train_marvin_rl100_online.sh
```

正常中断后重复同一条命令即可恢复最近的策略版本、actor/critic optimizer 和尚未填满的
同版本 on-policy buffer。不要改变 K、batch size、第二阶段目录或 checkpoint 后继续复用
旧输出目录。`EXECUTED_STEPS` 理论上可配置，但当前第二阶段是 K=5，脚本会拒绝其他值；
若需 K=1/3/10，必须先按9.6节训练对应 K 的第二阶段模型。

每次更新写入不可覆盖目录：

```text
online_ft/update_0001/policy/       RL-100 actor原生保存
online_ft/update_0001/critic/       在线value critic
online_ft/update_0001/optimizers.pt actor/critic optimizer
online_ft/update_0001/candidate.ckpt 可由现有rollout加载的候选actor
online_ft/latest.json               最新policy_version和候选路径
```

### 9.7.4 候选模型部署

先只检查候选 checkpoint：

```bash
TORCH_HOME=/home/xlb/code/强化微调ACT/.cache/torch PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/RL-100/RL-100 /home/xlb/miniconda3/envs/rl100/bin/python -m lerobot_robot_marvin.rl100_rollout --checkpoint /home/xlb/code/RL-100/RL-100/data/outputs/marvin_usb_online_rl_k5_seed1000/online_ft/update_0001/candidate.ckpt --checkpoint-source model --device cuda:0 --inspect-only
```

再按9.6.5的安全阶梯从 `--max-steps 5 --actions-per-inference 1 --motion-scale 0.2`
开始真机验收，最后才恢复 K=5 和 `motion-scale=1.0`。在线 candidate 不自动成为正式模型；
每个版本都要和第二阶段 `offline_best.ckpt` 比较成功率、安全终止率和动作裁剪情况，失败时
直接回退到上一版不可覆盖的 `candidate.ckpt`。

## 10. ACT、ACT-RL 和 DP 共用的真机 Bridge

ACT 和 DP 都不要直接控制 Marvin SDK；它们统一通过 HIL-SERL Bridge 调用现有
机器人、相机、reset、工作空间限制和10Hz控制。

### 10.1 推荐：不加载奖励分类器，使用人工成功/失败键

终端一：

```bash
cd /home/xlb/code/强化微调ACT

# 注意：下面的 PYTHONPATH 必须整行粘贴；每个续行的 \ 必须是该行最后一个字符。
# /home/xlb/code_marvin/hil-serl/examples 不能拆成两行。
PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code_marvin/hil-serl:/home/xlb/code_marvin/hil-serl/examples:/home/xlb/code_marvin/hil-serl/serl_robot_infra:/home/xlb/code_marvin/hil-serl/serl_launcher \
/home/xlb/miniconda3/envs/hilserl/bin/python \
  -m lerobot_robot_marvin.bridge_server \
  --backend hardware \
  --host 127.0.0.1 \
  --port 8765 \
  --human-success-key \
  --allow-hardware \
  --confirm-hardware-risk=I_UNDERSTAND_THIS_CONTROLS_A_REAL_ROBOT \
  --log-level INFO
```

键盘监听：

```text
SPACE    标记成功并结束当前回合
ESC      标记失败并结束当前回合
Ctrl+C   终止 Bridge；环境关闭并释放机器人和相机
```

当前方案不要添加 `--classifier`。如果将来恢复奖励分类器，只能在
`--classifier` 和 `--human-success-key` 中选择一个，不能同时使用。

### 10.2 两个客户端不能同时连接

ACT 和 DP 测试必须串行进行。一个客户端结束并确认 Bridge/机器人已经释放后，
再启动下一次测试。相机被异常占用时不要直接反复启动；先检查残留进程，必要时正常
结束进程或重新插拔相机。

## 11. 推荐执行顺序

### ACT

```text
buffer dry-run
-> buffer 转 LeRobot
-> 持续目标夹爪转换
-> ACT 训练
-> checkpoint 只加载
-> Mock Bridge
-> 真机5步 / n_action_steps=1
-> 延长回合
-> 对比 n_action_steps=10
```

### DP

```text
LeRobot 持续夹爪数据转5回合 Zarr
-> 2 epoch 冒烟训练
-> 全量 Zarr
-> 正式训练
-> inspect-only
-> Zarr replay
-> Mock Bridge
-> 真机 dry-run
-> 真机5步 / scale=0.2 / actions_per_inference=1
-> 逐步放开参数
-> 在线PPO Mock采集与单次更新
-> 真机collect-only审计
-> 每64条K=5 transition更新一次candidate
-> 逐版本真机验收或回退
```

### ACT-RL

```text
ACT-RL buffer dry-run
-> 转换成功+失败完整回合及逐步reward
-> 离线IQL critic warm-up
-> 优势加权直接更新ACT
-> learner checkpoint检查
-> 标准pretrained_model离线/Mock评估
-> Mock在线采集
-> 离线/在线混合更新
-> candidate验收
-> 最后才启用真机collector
```

## 12. 常见问题

### 输出目录已经存在

转换器和新训练默认拒绝覆盖已有目录。使用新的输出目录，或者在确认数据可删除后由
人工处理旧目录；不要让脚本自动覆盖数据集或 checkpoint。

### ACT 提示 repo_id 缺失

本地训练必须显式加入：

```bash
--policy.push_to_hub=false
```

如果确实要上传 Hugging Face Hub，则设置真实的 `--policy.repo_id`。

### `l1_loss` 下降但 `eval_loss` 上升

这通常是过拟合信号。优先选择验证损失较低的 checkpoint，而不是机械使用最后一个。
ACT 当前需要结合 WandB 曲线人工选择；DP 已按 `val_loss` 自动保存 top-k。

### DP 第一次推理超过2秒

DP rollout 内置0.5秒 heartbeat，在 CUDA 首次推理期间保持 Bridge watchdog 活跃。
每次成功的 `reset/step` 响应本身也记为主控联系；10 Hz 控制流正常时不再向单线程
Bridge 额外排队 `status` 请求。只有策略推理等主控空闲期才发送独立心跳，连续3次
失败才向主线程报错；任意一次成功的 `reset/step/status` 会清除瞬时超时。

收到 `reward=1.0, terminated=True, succeed=True` 后，该回合已成功，rollout 不再用终止后
的 heartbeat 覆盖该结果。Bridge 将显式 `stop/disconnect` 计为正在执行的串行操作，
在夹爪、相机和 SDK 释放较慢时 watchdog 不会重入第二次 stop。更新代码后必须
重启 Bridge 和 rollout 两个进程才会生效。SpaceMouse采集子进程会在其共享
manager关闭前先终止并回收，避免正常退出时出现`ConnectionResetError`。若心跳
持续失败或策略进程退出，Bridge仍会执行终止式 stop。

### 相机关闭提示 `stop() cannot be called before start()`

共享相机别名可能重复执行关闭。这类关闭阶段日志不代表策略推理失败，但必须确认实体
相机、SDK 和机器人最终已释放，下一次 Bridge 能正常初始化全部三台实体 RealSense。

## 13. 指令同步记录

- 2026-08-12：加入ACT-RL failure-aware数据转换、离线IQL、checkpoint检查、Mock在线
  采集、离线/在线混合更新和真机安全门槛；补充长路径不可中途换行的Shell规则。
- 2026-08-12：正式数据检查为808回合/90,119帧；训练显式排除异常1帧episode 556，
  ACT-RL batch size调整为8，正式离线训练改为50,000步和10,000步critic warm-up。
- 2026-08-13：加入ACT-RL只读离线评估命令，用于比较原ACT和各RL checkpoint的动作
  L1、策略偏移、Q/V、成功/失败分组以及反归一化动作越界幅度。
- 2026-08-13：新增`ACT强化微调详细审计与测试方案.md`，同步ACT-115k与RL-50k真机
  对比及非真机评估命令、`n_action_steps=1/3/5/10`逐级安全测试流程，以及
  last/offline-best/real-best的选模规则。
- 2026-08-13：修复DP回合已成功后被心跳超时误报为失败的竞态；主控
  `reset/step`响应现在抑制多余心跳，心跳改为连续3次失败才上报，并防止
  显式慢速`stop`期间watchdog重入设备释放；同步按子进程、manager、机器人
  环境的顺序关闭SpaceMouse，避免退出阶段`ConnectionResetError`；当前重训
  checkpoint的检查、回放和部署示例显式使用`--checkpoint-source model`。
- 2026-08-14：新增RL-100 Marvin第二阶段离线RL流程：全量808回合/90,119帧
  failure-aware Zarr、可配置`EXECUTED_STEPS`、冻结视觉编码器的IQL与learned dynamics、
  UniO4/BPPO actor更新、离线动作MSE选模，以及可被现有真机rollout直接加载的
  `offline_best.ckpt`/`offline_last.ckpt`导出和逐级真机安全命令。
- 2026-08-14：新增RL-100 Marvin第三阶段在线PPO：接入原生扩散轨迹采样与
  `dp_align_update_no_share`，增加四相机on-policy buffer、K=5 Bridge执行、动作一致性
  校验、DDIM最终动作边界投影与log-prob重算、策略版本隔离、断点恢复、不可覆盖candidate
  导出，以及Mock、collect-only、正式真机训练和逐版本回退命令。
- 2026-08-14：将所有任务共用的 `MarvinEnv` 动作平滑奖励统一为 Marvin USB config
  公式：首步不惩罚，后续使用 `-0.01 * mean(delta_action[:4] ** 2)`，不包含夹爪；明确
  修改后需重启hardware Bridge，历史在线buffer不追溯重算。
- 后续更新要求：凡是CLI参数、默认配置、数据目录、checkpoint结构或真机安全流程发生
  变化，都同时修改对应章节并在这里增加一条日期记录。

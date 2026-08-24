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

### 3.4 转换为Residual固定LeRobot v2.1数据集

`residual-offpolicy-rl`固定的LeRobot使用v2.1逐episode布局，不能直接读取当前v3数据集。
转换器必须用`residual`环境执行。默认`--media-storage image`把四相机图像嵌入逐episode
Parquet，绕过当前`residual`环境中TorchCodec与TorchVision VideoReader不兼容的问题；代价是
输出占用空间高于视频格式。输出目录已存在时默认拒绝覆盖，只有明确加入`--overwrite`才删除重建。

BC成功数据转换：

```bash
/home/xlb/miniconda3/envs/residual/bin/python /home/xlb/code/强化微调ACT/convert_marvin_v3_to_residual_v21.py --input-root /home/xlb/code/强化微调ACT/datasets/marvin_usb_act_success_dense_gripper --output-root /home/xlb/code/强化微调ACT/datasets/residual_v21/marvin_usb_bc --repo-id marvin_usb_bc --mode bc --media-storage image --exclude-episodes 359
```

Residual RL成功/失败数据转换；输出额外保留`rl.reward`、`rl.terminated`、`rl.success`、
`rl.intervention`、`rl.safety_abort`，并根据episode边界生成训练代码需要的`next.done`：

```bash
/home/xlb/miniconda3/envs/residual/bin/python /home/xlb/code/强化微调ACT/convert_marvin_v3_to_residual_v21.py --input-root /home/xlb/code/强化微调ACT/datasets/marvin_usb_act_rl_offline --output-root /home/xlb/code/强化微调ACT/datasets/residual_v21/marvin_usb_rl --repo-id marvin_usb_rl --mode rl --media-storage image --exclude-episodes 556
```

先做两回合smoke时，在命令末尾加入`--max-episodes 2`并使用独立输出目录。每次转换完成后检查
输出根目录的`residual_v21_conversion_report.json`；排除异常episode 359后BC应为583回合、
49,573帧，排除异常episode 556后RL应为807回合、90,118帧。

### 3.5 Residual路线1：在LeRobot 0.1.0中重训ACT

转换报告、回合数、帧数和图像轴名全部通过后，使用Residual自带的LeRobot训练新ACT。
BC训练使用`marvin_usb_bc`，Residual learner的归一化统计使用`marvin_usb_rl`，两者不要交换。

```bash
env -u CUDA_HOME -u CUDA_PATH -u LD_PRELOAD LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/home/xlb/miniconda3/envs/residual/lib/python3.10/site-packages/nvidia/cu13/lib:/home/xlb/miniconda3/envs/residual/lib/python3.10/site-packages/torch/lib NUMBA_DISABLE_JIT=1 HF_LEROBOT_HOME=/home/xlb/code/强化微调ACT/datasets/residual_v21 HF_DATASETS_CACHE=/home/xlb/code/强化微调ACT/.cache/huggingface_datasets_residual HUGGINGFACE_HUB_CACHE=/home/xlb/code/强化微调ACT/.cache/huggingface_hub_residual PYTHONPATH=/home/xlb/code/residual-offpolicy-rl /home/xlb/miniconda3/envs/residual/bin/python -m resfit.lerobot.scripts.train_bc_dexmg --dataset marvin_usb_bc --policy act --steps 100000 --batch_size 8 --num_workers 4 --seed 1000 --device cuda --output_dir /home/xlb/code/强化微调ACT/outputs/residual_route1_act --log_freq 100 --save_freq 5000 --eval_fraction 0.1 --eval_freq 2000 --eval_batches 0 --split_seed 1000 --policy_cameras side_policy wrist_1buchong wrist_1 wrist_2
```

使用`split_seed=1000`时，583回合按episode固定分为525个训练回合/44,994帧与58个
验证回合/4,579帧。验证集关闭随机图像增强，每2,000个optimizer step对完整验证集计算
`eval_loss`。ACT的state/action/image归一化统计仅由525个训练回合聚合，验证回合不参与
统计量拟合。划分固定写入`dataset_split.json`，最低验证损失的模型写入`best_eval/policy`。

本地checkpoint保留在`/home/xlb/code/强化微调ACT/outputs/residual_route1_act`；Residual使用
`/home/xlb/code/强化微调ACT/outputs/residual_route1_act/best_eval/policy`。该目录必须同时包含
`config.json`和权重文件。不开启WandB也会保留本地checkpoint。

Residual原始坐标契约为：

```text
RL state       (raw_state - dataset_mean) / max(dataset_std, min_state_std)
RL base action ActionScaler.scale(raw_ACT_action)
RL action      ActionScaler.scale(raw_executed_action)
robot command  ActionScaler.unscale(normalized_base_action + normalized_residual)
```

`agent.actor.action_scale=0.1`控制学习的residual幅度。`marvin.residual_scale=null`为原始
ResFiT路径；若真机初次验证显式设为`0.05`，该0.05也是归一化residual单位，
不是机器人原始命令单位。

#### 路线1 LeRobot 0.1.0 ACT真机5步测试

下面两条命令会实际初始化/reset机器人并下发动作；ACT客户端第5步结束时环境还会执行一次
自动reset。确保急停可用、工作空间无人员和障碍物，并且没有其他ACT、DP或RL客户端连接。

终端一启动hardware Bridge：

```bash
PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code_marvin/hil-serl:/home/xlb/code_marvin/hil-serl/examples:/home/xlb/code_marvin/hil-serl/serl_robot_infra:/home/xlb/code_marvin/hil-serl/serl_launcher /home/xlb/miniconda3/envs/hilserl/bin/python -m lerobot_robot_marvin.bridge_server --backend hardware --host 127.0.0.1 --port 8765 --human-success-key --human-outcome-input footswitch --allow-hardware --confirm-hardware-risk=I_UNDERSTAND_THIS_CONTROLS_A_REAL_ROBOT --log-level INFO
```

Bridge完成机器人和四相机初始化后，终端二加载验证集最优ACT并执行零residual的1回合5步测试。
运行时固定预测20个动作、只执行前5个后重新观测和推理：

```bash
env -u CUDA_HOME -u CUDA_PATH -u LD_PRELOAD LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/home/xlb/miniconda3/envs/residual/lib/python3.10/site-packages/nvidia/cu13/lib:/home/xlb/miniconda3/envs/residual/lib/python3.10/site-packages/torch/lib NUMBA_DISABLE_JIT=1 PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/residual-offpolicy-rl /home/xlb/miniconda3/envs/residual/bin/python -m resfit.rl_finetuning.scripts.collect_marvin --base-policy-checkpoint /home/xlb/code/强化微调ACT/outputs/residual_route1_act/best_eval/policy --base-policy-chunk-size 20 --base-policy-n-action-steps 5 --offline-dataset-root /home/xlb/code/强化微调ACT/datasets/residual_v21/marvin_usb_rl --offline-dataset-name marvin_usb_rl --output-dir /home/xlb/code/强化微调ACT/outputs/marvin_route1_act_chunk20_exec5_real_smoke --device cuda --host 127.0.0.1 --port 8765 --episodes 1 --max-steps 5 --residual-scale 0.0 --actor-action-scale 0.1 --min-action-range 0.1 --min-state-std 0.1 --allow-real-robot
```

不传`--residual-checkpoint`且设置`--residual-scale 0.0`，所以该命令只测试ACT base policy，
不会执行学习得到的residual。Bridge终端可随时按`ESC`标记失败并结束回合，紧急情况优先使用
机器人急停；测试结束后按`Ctrl+C`终止Bridge并确认机器人及相机释放。

上述`marvin_route1_act_chunk20_exec5_real_smoke`仅用于ACT验收，不得复制到正式RL replay目录。

#### Residual路线1：正式真机TD3训练

正式输出统一使用`/home/xlb/code/强化微调ACT/outputs/marvin_residual_td3_chunk20_exec5`：`replay`保存原子写入的
在线回合，`checkpoints`保存完整agent/optimizer断点，`deployment/residual_actor_latest.pt`保存可部署
actor，`cache`保存离线buffer缓存。正式manifest为`schema_version=5`，并固定记录
`base_policy_chunk_size=20`和`base_policy_n_action_steps=5`、
`coordinate_frame=resfit_marvin_bounded_v2`和
`action_contract=marvin_normalized_command_box_v1`；ACT权重、离线数据统计和归一化参数任一不一致都会
拒绝混用。所有ACT原始命令和base+residual合成命令最终都硬裁剪到Bridge要求的`[-1,1]`。

保持上面的hardware Bridge运行，在终端二用零residual采集正式启动回放。`--min-transitions 1000`
指经过`n_step=3`转换后仍可用于学习的transition数，不是NPZ原始行数；命令会按完整回合持续采集，
`--resume`会校验已有回放身份后从下一个episode编号安全追加；`--episodes 1`只要求本次至少新增
一个回合，真正的停止条件仍是1,000条可用样本。6个完整200步回合理论上得到
`6*(200-3+1)=1,188`条可用样本；若回合提前终止，collector会继续补到1,000条：

```bash
env -u CUDA_HOME -u CUDA_PATH -u LD_PRELOAD LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/home/xlb/miniconda3/envs/residual/lib/python3.10/site-packages/nvidia/cu13/lib:/home/xlb/miniconda3/envs/residual/lib/python3.10/site-packages/torch/lib NUMBA_DISABLE_JIT=1 PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/residual-offpolicy-rl /home/xlb/miniconda3/envs/residual/bin/python -m resfit.rl_finetuning.scripts.collect_marvin --base-policy-checkpoint /home/xlb/code/强化微调ACT/outputs/residual_route1_act/best_eval/policy --base-policy-chunk-size 20 --base-policy-n-action-steps 5 --offline-dataset-root /home/xlb/code/强化微调ACT/datasets/residual_v21/marvin_usb_rl --offline-dataset-name marvin_usb_rl --output-dir /home/xlb/code/强化微调ACT/outputs/marvin_residual_td3_chunk20_exec5/replay --device cuda --host 127.0.0.1 --port 8765 --episodes 1 --max-steps 200 --min-transitions 1000 --n-step 3 --resume --residual-scale 0.0 --actor-action-scale 0.1 --min-action-range 0.1 --min-state-std 0.1 --allow-real-robot
```

采集结束后必须先做只读审计，看到`ready: true`且`usable_nstep_transitions >= 1000`才可启动learner：

```bash
PYTHONPATH=/home/xlb/code/residual-offpolicy-rl /home/xlb/miniconda3/envs/residual/bin/python -m resfit.rl_finetuning.scripts.audit_marvin_replay --replay-dir /home/xlb/code/强化微调ACT/outputs/marvin_residual_td3_chunk20_exec5/replay --base-policy-checkpoint /home/xlb/code/强化微调ACT/outputs/residual_route1_act/best_eval/policy --base-policy-chunk-size 20 --base-policy-n-action-steps 5 --offline-dataset-root /home/xlb/code/强化微调ACT/datasets/residual_v21/marvin_usb_rl --offline-dataset-name marvin_usb_rl --actor-action-scale 0.1 --min-action-range 0.1 --min-state-std 0.1 --n-step 3 --minimum-transitions 1000
```

Residual现直接复用vendored LeRobot的gRPC transport实现HIL-SERL/RLPD式长驻
actor/learner：actor异步发送完整episode transition，learner增量加入online replay并持续训练；
learner每50步推送actor+encoder参数，actor只在episode边界原子应用最新参数。NPZ仍先落盘，
作为gRPC断线时的可审计备份。learner限制为80% CUDA显存，actor限制为18%；这不是精确算力
配额，因此learner每个UTD=4更新组后主动让行10ms。online/offline batch保持50:50。critic
warm-up只在首次执行1,000步，完成立即保存；完整断点每1,000步保存并永久保留，
不再自动删除旧的编号checkpoint。

正式learner默认使用W&B online模式，远程project为`marvin-residual-td3-route1`，固定run ID为
`marvinresidualc20e5s1000`。因此`latest.pt`断点续训会继续写入同一个远程run。首次运行前先登录：

```bash
/home/xlb/miniconda3/envs/residual/bin/python -m wandb login
```

每100 learner step远程记录critic/actor loss、Q值、梯度、学习率、Residual L1/L2及饱和率、
online/offline buffer大小、累计在线回合/transition/成功率和训练吞吐；critic warm-up每100次更新
也会记录。在线模式仍在本机`/home/xlb/code/强化微调ACT/wandb`保留运行文件，短暂断网由W&B
后台重试，但启动actor前必须确认learner已完成W&B初始化并监听`127.0.0.1:50051`。

第一轮learner命令如下；此时不需要启动Bridge：

```bash
env -u CUDA_HOME -u CUDA_PATH -u LD_PRELOAD LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/home/xlb/miniconda3/envs/residual/lib/python3.10/site-packages/nvidia/cu13/lib:/home/xlb/miniconda3/envs/residual/lib/python3.10/site-packages/torch/lib NUMBA_DISABLE_JIT=1 CACHE_DIR=/home/xlb/code/强化微调ACT/outputs/marvin_residual_td3_chunk20_exec5/cache MPLCONFIGDIR=/home/xlb/code/强化微调ACT/outputs/marvin_residual_td3_chunk20_exec5/matplotlib PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/residual-offpolicy-rl /home/xlb/miniconda3/envs/residual/bin/python -m resfit.rl_finetuning.scripts.train_residual_td3 --config-path /home/xlb/code/residual-offpolicy-rl/resfit/rl_finetuning/config --config-name marvin_residual
```

程序中断后重新执行同一条learner命令；`resume_checkpoint=auto`会自动用完整
`latest.pt`续训，不会把部署actor误当作训练断点。

看到`Marvin learner gRPC listening on 127.0.0.1:50051`后，保持learner运行，
按上文命令启动hardware Bridge，再在终端二启动长驻actor：

```bash
env -u CUDA_HOME -u CUDA_PATH -u LD_PRELOAD LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/home/xlb/miniconda3/envs/residual/lib/python3.10/site-packages/nvidia/cu13/lib:/home/xlb/miniconda3/envs/residual/lib/python3.10/site-packages/torch/lib NUMBA_DISABLE_JIT=1 PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/residual-offpolicy-rl /home/xlb/miniconda3/envs/residual/bin/python -m resfit.rl_finetuning.scripts.collect_marvin --base-policy-checkpoint /home/xlb/code/强化微调ACT/outputs/residual_route1_act/best_eval/policy --base-policy-chunk-size 20 --base-policy-n-action-steps 5 --offline-dataset-root /home/xlb/code/强化微调ACT/datasets/residual_v21/marvin_usb_rl --offline-dataset-name marvin_usb_rl --output-dir /home/xlb/code/强化微调ACT/outputs/marvin_residual_td3_chunk20_exec5/replay --device cuda --host 127.0.0.1 --port 8765 --max-steps 200 --min-transitions 0 --n-step 3 --resume --residual-checkpoint /home/xlb/code/强化微调ACT/outputs/marvin_residual_td3_chunk20_exec5/deployment/residual_actor_latest.pt --actor-action-scale 0.1 --min-action-range 0.1 --min-state-std 0.1 --async-grpc --learner-host 127.0.0.1 --learner-port 50051 --gpu-memory-fraction 0.18 --continuous --allow-real-robot
```

actor和learner启动后均长期运行，不再按5回合/500步手工切换。actor只在episode边界应用
gRPC收到的最新参数。真机仍要求急停可用，SPACE/ESC结果键由操作者监控。
`marvin.residual_scale=null`遵循原始ResFiT路径；首次并发真机短测建议在actor命令末尾增加
`--residual-scale 0.05`，并监控单步控制时间是否继续约100ms。

若需要对比最新step 11050之前的倒数第二个actor，使用从完整step 11000断点导出的
`/home/xlb/code/强化微调ACT/outputs/marvin_residual_td3_chunk20_exec5/deployment/residual_actor_step_00011000.pt`；评估回放必须写入独立目录：

```bash
env -u CUDA_HOME -u CUDA_PATH -u LD_PRELOAD LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/home/xlb/miniconda3/envs/residual/lib/python3.10/site-packages/nvidia/cu13/lib:/home/xlb/miniconda3/envs/residual/lib/python3.10/site-packages/torch/lib NUMBA_DISABLE_JIT=1 PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/residual-offpolicy-rl /home/xlb/miniconda3/envs/residual/bin/python -m resfit.rl_finetuning.scripts.collect_marvin --base-policy-checkpoint /home/xlb/code/强化微调ACT/outputs/residual_route1_act/best_eval/policy --base-policy-chunk-size 20 --base-policy-n-action-steps 5 --offline-dataset-root /home/xlb/code/强化微调ACT/datasets/residual_v21/marvin_usb_rl --offline-dataset-name marvin_usb_rl --output-dir /home/xlb/code/强化微调ACT/outputs/eval_act_residual_step11000_chunk20_exec5 --device cuda --host 127.0.0.1 --port 8765 --episodes 20 --max-steps 200 --min-transitions 0 --n-step 3 --resume --residual-checkpoint /home/xlb/code/强化微调ACT/outputs/marvin_residual_td3_chunk20_exec5/deployment/residual_actor_step_00011000.pt --residual-scale 0.1 --actor-action-scale 0.1 --min-action-range 0.1 --min-state-std 0.1 --gpu-memory-fraction 0.18 --allow-real-robot
```

Residual collector连接Bridge后会启动独立0.5秒heartbeat。成功的`reset/step`响应会刷新主控联系；
ACT推理或Replay压缩写盘超过2秒时，heartbeat会发送`status`维持watchdog，连续3次失败才向主循环
报错。learner-only不创建heartbeat，也不访问Bridge。终止transition已经在环境内部完成下一回合reset，collector会复用该观测，
不会再额外执行第二次reset。更新心跳代码后必须重启Bridge和collector/learner进程；已经原子保存的
schema 5回合可直接通过`--resume`继续。旧`20/20`的schema 4回放保留在旧目录，
不得与新`20/5`回放混用。

不得用LeRobot 0.6.2的`ACTPolicyRunner`加载这个checkpoint；Marvin Residual现在直接用
Residual仓库内的LeRobot 0.1.0 `ACTPolicy.from_pretrained()`。离线replay生成时，ACT action
queue必须在每个episode边界reset。

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
  --checkpoint outputs/act_rl_offline_iql/checkpoints/step_00050000/pretrained_model \
  --output-dir outputs/act_rl_online_round_001_mock_replay \
  --host 127.0.0.1 \
  --port 8765 \
  --device cuda \
  --episodes 1 \
  --max-decisions 20 \
  --executed-steps 10
```

在线Replay v2按“一回合一个NPZ”追加写入，已有episode不会被覆盖。它同时保存未裁剪的
完整网络输出`policy_action`、经安全裁剪后发给Bridge的`command_action`、真机实际执行前缀
`actual_action`、供critic使用的`critic_action`、`executed_mask`、`intervention_mask`、
`terminated`和`truncated`。ACT loss只使用实际执行前缀；完整原始policy chunk保留用于
策略越界和裁剪率审计。普通时间上限使用`truncated=True`，成功、失败或安全终止使用
`terminated=True`。

### 6.7 在线Replay完整性审计

任何在线更新前必须先审计，`ok`不为`true`时禁止训练：

```bash
cd /home/xlb/code/强化微调ACT

PYTHONPATH="$PWD/src:/home/xlb/code/lerobot/src" \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python \
  -m act_rl.scripts.audit_online_replay \
  --replay outputs/act_rl_online_round_001_mock_replay \
  --config configs/act_rl/marvin_online_round1.json \
  --output outputs/act_rl_online_round_001_mock_replay/audit.json
```

审计检查schema版本、24维state、30×5动作、四路图像、NaN/Inf、安全命令边界、实际执行
步数、padding、提前终止、成功/失败/截断和policy SHA256。同一轮只允许一个已知
policy SHA256；collector和审计器都会拒绝混入其他策略的目录。旧版Replay默认被拒绝，
不能与阶段一正式在线数据混用。`train_online`在加载大模型前会自动重复同一套严格审计，
手工遗漏本节也不会将损坏的Replay用于更新。

### 6.8 第一阶段离线/在线混合更新

learner只读取固定Replay快照，不连接Bridge。默认按在线transition数量自动调整：少于1000
条时offline:online为75:25，1000～4999条时为50:50，达到5000条后为25:75。在线池使用
50%最近窗口和50%全池的带放回采样，并近似平衡成功/失败、提高人工干预样本权重。

```bash
cd /home/xlb/code/强化微调ACT

TORCH_HOME="$PWD/.cache/torch" \
HF_HOME="$PWD/.cache/huggingface" \
PYTHONPATH="$PWD/src:/home/xlb/code/lerobot/src" \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python \
  -m act_rl.scripts.train_online \
  --config configs/act_rl/marvin_online_round1.json \
  --model-checkpoint outputs/act_rl_offline_iql/checkpoints/step_00050000/pretrained_model \
  --learner-checkpoint outputs/act_rl_offline_iql/checkpoints/step_00050000.pt \
  --dataset-root datasets/marvin_usb_act_rl_offline \
  --repo-id xlb/marvin_usb_act_rl_offline \
  --outcomes datasets/marvin_usb_act_rl_offline/meta/rl_outcomes.json \
  --exclude-episodes 556 \
  --online-replay outputs/act_rl_online_round_001_replay \
  --output-dir outputs/act_rl_online_round_001_train \
  --device cuda \
  --steps 300 \
  --save-freq 100 \
  --num-workers 2
```

每个混合batch都更新Q1/Q2、V和Target Q；ACT默认每2个critic step更新1次，ResNet保持冻结。
默认限制每条在线transition最多对应4次learner更新；请求步数超过限制会直接失败。只有在
明确审计过重放风险后才能使用`--allow-high-replay-reuse`。模型checkpoint和learner
checkpoint必须是同一步且actor权重完全一致，否则训练拒绝启动。

### 6.9 真机分轮采集安全门槛

真机collector必须先按第10节启动带`--human-success-key`的hardware Bridge。第一轮固定
使用离线IQL 50k模型采集10～20回合，SPACE标记成功，ESC标记失败：

```bash
cd /home/xlb/code/强化微调ACT

TORCH_HOME="$PWD/.cache/torch" \
PYTHONPATH="$PWD/src:/home/xlb/code/lerobot/src" \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python \
  -m act_rl.scripts.collect_online \
  --checkpoint outputs/act_rl_offline_iql/checkpoints/step_00050000/pretrained_model \
  --output-dir outputs/act_rl_online_round_001_replay \
  --host 127.0.0.1 \
  --port 8765 \
  --device cuda \
  --episodes 10 \
  --max-decisions 20 \
  --executed-steps 10 \
  --gamma 0.99 \
  --allow-real-robot
```

它会实际reset并连续执行动作。先完成Mock连续回合、动作越界率、失败键、超时截断、Replay
审计和checkpoint回退验证。在线learner只产生candidate，不会自动替换collector策略；
candidate通过离线评估、Mock、真机短程安全测试并经人工批准后，下一轮collector才切换，
而且只能在回合reset边界切换。

### 6.9.1 第一轮candidate加载与真机短测

当前没有一个命令同时执行“采集→训练→自动切换Actor”。`collect_online`只采集，
`train_online`只训练candidate，`act_rollout`只加载指定candidate测试。Bridge必须按第10节
独立运行，以保留SPACE、ESC和Ctrl+C安全控制。

第一轮300步更新后，最新candidate为`step_00050300`。重新启动hardware Bridge后，先执行
1回合、5步、每次只执行1个动作的真机短测：

```bash
cd /home/xlb/code/强化微调ACT
TORCH_HOME=/home/xlb/code/强化微调ACT/.cache/torch PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/lerobot/src /home/xlb/miniconda3/envs/lerobot_xlb/bin/python -m lerobot_robot_marvin.act_rollout --checkpoint /home/xlb/code/强化微调ACT/outputs/act_rl_online_round_001_train/checkpoints/step_00050300/pretrained_model --host 127.0.0.1 --port 8765 --device cuda --episodes 1 --max-steps 5 --n-action-steps 1 --fps 10 --allow-real-robot --log-level INFO
```

### 6.9.2 第二轮使用最新Actor采集与继续更新

确认`step_00050300`短测安全后，第二轮collector显式加载该Actor，并写入全新Replay
目录。不要继续向`round_001_replay`追加：

```bash
cd /home/xlb/code/强化微调ACT
TORCH_HOME=/home/xlb/code/强化微调ACT/.cache/torch PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/lerobot/src /home/xlb/miniconda3/envs/lerobot_xlb/bin/python -m act_rl.scripts.collect_online --checkpoint /home/xlb/code/强化微调ACT/outputs/act_rl_online_round_001_train/checkpoints/step_00050300/pretrained_model --output-dir /home/xlb/code/强化微调ACT/outputs/act_rl_online_round_002_replay --host 127.0.0.1 --port 8765 --device cuda --episodes 10 --max-decisions 20 --executed-steps 10 --gamma 0.99 --allow-real-robot
```

采集完成后先审计：

```bash
cd /home/xlb/code/强化微调ACT
PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/lerobot/src /home/xlb/miniconda3/envs/lerobot_xlb/bin/python -m act_rl.scripts.audit_online_replay --replay /home/xlb/code/强化微调ACT/outputs/act_rl_online_round_002_replay --config /home/xlb/code/强化微调ACT/configs/act_rl/marvin_online_round1.json --output /home/xlb/code/强化微调ACT/outputs/act_rl_online_round_002_replay/audit.json
```

第二轮训练必须同时使用来自`step_00050300`的模型目录和learner `.pt`，不能与50k或
其他步数混用：

```bash
cd /home/xlb/code/强化微调ACT
TORCH_HOME=/home/xlb/code/强化微调ACT/.cache/torch HF_HOME=/home/xlb/code/强化微调ACT/.cache/huggingface PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/lerobot/src /home/xlb/miniconda3/envs/lerobot_xlb/bin/python -m act_rl.scripts.train_online --config /home/xlb/code/强化微调ACT/configs/act_rl/marvin_online_round1.json --model-checkpoint /home/xlb/code/强化微调ACT/outputs/act_rl_online_round_001_train/checkpoints/step_00050300/pretrained_model --learner-checkpoint /home/xlb/code/强化微调ACT/outputs/act_rl_online_round_001_train/checkpoints/step_00050300.pt --dataset-root /home/xlb/code/强化微调ACT/datasets/marvin_usb_act_rl_offline --repo-id xlb/marvin_usb_act_rl_offline --outcomes /home/xlb/code/强化微调ACT/datasets/marvin_usb_act_rl_offline/meta/rl_outcomes.json --exclude-episodes 556 --online-replay /home/xlb/code/强化微调ACT/outputs/act_rl_online_round_002_replay --output-dir /home/xlb/code/强化微调ACT/outputs/act_rl_online_round_002_train --device cuda --steps 300 --save-freq 100 --num-workers 2 --log-freq 20
```

若完成300步，第二轮candidate将保存为`step_00050400`、`step_00050500`和
`step_00050600`。若当轮少于75条transition，应降低`--steps`，不要直接使用
`--allow-high-replay-reuse`。

### 6.10 ACT与RL-ACT执行长度对比

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
  --human-outcome-input footswitch \
  --allow-hardware \
  --confirm-hardware-risk=I_UNDERSTAND_THIS_CONTROLS_A_REAL_ROBOT \
  --log-level INFO
```

人工结果输入由`--human-outcome-input`切换，两种模式互斥：

```text
--human-outcome-input keyboard     SPACE=成功，ESC=失败/复位
--human-outcome-input footswitch  Shift+←=成功，Shift+→=失败/复位
Ctrl+C                            终止 Bridge；环境关闭并释放机器人和相机
```

正式脚踏板命令使用`footswitch`；若改用键盘，只需把命令中的`footswitch`改为
`keyboard`并重启Bridge。脚踏板模式下Space/Esc不会标记回合，键盘模式下
Shift+方向键不会标记回合。脚踏失败键不是硬件急停；碰撞、失控或通信异常时
仍必须优先使用机器人急停。

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

- 2026-08-21：Human40 Residual TD3的online ReplayBuffer改为固定
  `marvin_<配置哈希>_incremental` memmap缓存。learner启动时校验已缓存episode的manifest哈希，
  只从权威NPZ replay补导入新增episode；gRPC每收到一个完整episode后立即刷新memmap长度、writer
  状态和原子元数据。正常中断后下次直接加载已提交缓存，不再因完整replay指纹变化而重建全部
  online buffer；episode被删除或修改时会隔离不匹配缓存并安全重建。现有训练、Bridge和collector
  命令均不变。

- 2026-08-21：Human40 Residual TD3正式配置的`checkpoint_interval`从100改为
  1,000 learner step，并将`training_checkpoint_keep_last=0`定义为禁用清理；
  每个1,000-step编号完整checkpoint都永久保留，`latest.pt`和部署actor同步更新。
  每个完整checkpoint约650 MB，训练到300,000 step最多需要约195 GB存储。

- 2026-08-21：`--human-success-key`新增`--human-outcome-input`互斥切换；
  `keyboard`模式只监听`SPACE=success`、`ESC=failure`，`footswitch`模式只监听
  `Shift+←=success`、`Shift+→=failure/reset`。脚踏失败会终止当前回合、传递
  `set_last_episode_result(False)`，并由collector进入下一次失败复位；它不替代硬件急停。
  修改后必须重启hardware Bridge才能生效。

- 2026-08-20：修复`HilserlBackend.step()`在`--human-success-key`模式下未向底层
  Marvin环境传递人工回合结果的问题；Space=success传入`True`、ESC=failure传入`False`，
  使`RESET_BY_EPISODE_RESULT`能够在下一回合正确选择自动USB复位或人工复位。修改后必须重启hardware Bridge。

- 2026-08-20：人工失败复位在机械臂归位并张爪后增加默认`2.0s`等待
  (`MANUAL_RESET_SETTLE_TIME`)，给操作者重新放置USB的时间，避免下一回合在USB尚未放置时立即开始。

- 2026-08-20：修正Hydra命令中包含中文目录的覆盖值；不再把嵌套引号传入路径，统一通过
  `ACT_POLICY`、`ACT_REPLAY`、`ACT_TRAIN`、`ACT_RESIDUAL`环境变量和
  `marvin.key='${oc.env:变量名}'`插值传递，避免`LexerNoViableAltException`及路径中残留引号。
  同时固定learner让行参数为`0.05`，禁止误写为`0.0505`。

- 2026-08-19：新增Residual离线状态分布科学分析脚本与四面板导出，固定对比ACT base、RL step 11000和step 11050的residual幅度、逐维修正、示范方向一致性及真机成功率；结果输出到`outputs/residual_effect_analysis`，不修改训练数据或replay。
- 2026-08-18：新增Residual倒数第二个actor（step 11000）独立部署文件和20回合真机评估命令；评估输出固定到`eval_act_residual_step11000_chunk20_exec5`，不覆盖step 11050评估结果或正式训练replay。
- 2026-08-18：Residual gRPC learner补齐结构化W&B日志并将正式`marvin_residual.yaml`改为
  `mode=online`；固定run ID `marvinresidualc20e5s1000`使`latest.pt`续训写入同一远程run。
  远程每100步记录损失、Q值、梯度、Residual幅值/饱和率、buffer、累计在线成功率及吞吐，
  critic warm-up也同步。修复LeRobot参数队列为空时每秒近千条日志的busy-loop，避免CPU和远程
  console被无效`Push parameters`日志占用。
- 2026-08-18：Residual Marvin由分轮运行升级为gRPC长驻actor/learner。复用vendored
  LeRobot `LearnerService`、protobuf分块传输和重试；actor异步发送episode，learner增量训练并
  每50步发布actor+encoder，actor仅在episode边界换参。learner/actor CUDA显存上限分别为
  80%/18%，learner每个UTD=4更新组后让行10ms。`resume_checkpoint=auto`自动恢复
  `latest.pt`；正式actor改为`--async-grpc --continuous`，不再手工切换采集和训练。
- 2026-08-18：Residual Marvin参考HIL-SERL/RLPD拆分actor与learner。正式`marvin_residual.yaml`
  默认`learner_only=true`，learner不构造环境、不连接Bridge；actor继续使用`collect_marvin`并通过
  schema 5原子NPZ向learner传递完整回合，learner通过原子PyTorch部署checkpoint向下一轮actor发布
  参数。单RTX 4090采用分轮而非同时运行：每轮500 learner step、UTD=4、每100步保存，collector
  固定candidate采集5回合后再续训。critic warm-up完成后在任何真机RPC之前立即保存step 0断点；
  完整编号断点默认仅保留最新两个，以控制磁盘占用。
- 2026-08-18：Residual路线1正式真机TD3的`critic_warmup_steps`由10,000临时调整为1,000，
  用于缩短首次真机联调；正式启动命令不变，仍由`marvin_residual.yaml`读取该值。
- 2026-08-18：Residual Marvin的ACT receding horizon改为`chunk_size=20`、
  `n_action_steps=5`：每次预测20个动作，逐步执行前5个后重新观测和推理；Residual仍每步
  输出一个5维修正。Replay升级为schema 5并绑定这两个参数，新正式路径固定为
  `/home/xlb/code/强化微调ACT/outputs/marvin_residual_td3_chunk20_exec5`；旧`20/20`数据保留但禁止混用。
- 2026-08-18：Residual collector与learner新增0.5秒Bridge heartbeat，覆盖ACT推理、TD3更新和Replay
  写盘的长空闲期，避免2秒watchdog误触发终止式急停；修复hardware stop并发清理时`env=None`竞态、
  非终止步骤重复消耗ACT队列动作及terminal后collector二次reset。正式采集命令恢复为
  `--episodes 1 --min-transitions 1000 --resume`，从已有回放继续补足而不是每次重启再强制新增6回合。
- 2026-08-18：Residual路线1真机在线Replay启动门槛由10,000条调整为1,000条可用n-step样本，
  正式collector至少采集6个回合；6个完整200步回合约产生1,188条可用样本，提前终止时自动续采。
  此次修改时critic-only warm-up仍保留发布代码的10,000次；随后同日为首次真机联调单独降为1,000次。
- 2026-08-18：Residual路线1真机TD3升级为正式`marvin_residual_td3`链路：replay采用schema 4并绑定
  ACT权重、离线数据统计、坐标契约和动作硬边界；`--min-transitions`按n-step转换后的可用样本计数，
  learner强制达到配置的可用在线样本门槛。修复128×128四相机ViT位置嵌入、CPU在线回放/GPU actor
  推理设备边界和terminal final_obs维度；critic warm-up移到首次真机reset之前，在线回合原子持久化。
  新增正式Hydra配置、回放审计、完整agent/optimizer周期检查点、`latest.pt`断点续训和独立部署actor，
  正式路径固定为`/home/xlb/code/强化微调ACT/outputs/marvin_residual_td3`。
- 2026-08-18：新增Residual路线1 LeRobot 0.1.0 `best_eval/policy`真机5步测试流程：hardware
  Bridge与ACT客户端均使用显式Conda解释器；ACT客户端固定正确的系统cuDNN 9.25/CUDA 13
  运行库，使用RL v2.1数据统计量并设置零residual、1回合、5步及`--allow-real-robot`安全门。
  明确客户端结束第5步时会自动reset，测试前必须准备急停且禁止其他客户端同时连接。
- 2026-08-17：Residual LeRobot 0.1.0 BC新增episode级离线hold-out eval；正式命令使用
  `eval_fraction=0.1`、`split_seed=1000`，得到525回合/44,994帧训练集和58回合/
  4,579帧验证集；state/action/image归一化统计严格只使用训练回合。验证集关闭图像增强，
  每2,000 step计算完整`eval_loss`，保存
  `dataset_split.json`、`best_eval_metrics.json`和`best_eval/policy`；Residual RL默认选用
  `best_eval/policy`而不是最后一个checkpoint。RTX 4090正式数据train+eval单步smoke已通过。
- 2026-08-17：Residual的CUDA 13 pip cuDNN 9.20包缺少运行时延迟加载的
  `libcudnn_engines_tensor_ir.so.9`，因而会与系统cuDNN 9.25引擎混用并触发
  `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`。Residual BC CUDA命令改为清除外部CUDA/
  preload设置，并完整使用系统cuDNN 9.25与环境CUDA 13运行库。RTX 4090上的
  convolution forward/backward和正式四相机BC单步训练均已通过。此修复不改动数据或
  checkpoint格式。
- 2026-08-17：Residual Marvin恢复论文/原始代码的坐标链：在离线数据与在线环境中使用
  同一`StateStandardizer`和`ActionScaler`，在归一化空间相加base action与residual后再
  反缩放为机器人命令；修复离线ACT queue跨episode污染。Marvin base policy改为直接
  加载路线1 LeRobot 0.1.0 ACT checkpoint，移除0.6.2 `ACTPolicyRunner`依赖。同步更新
  `collect_marvin`的offline dataset参数、Residual RL默认数据路径和807回合设置；
  BC入口在WandB关闭时不再调用`wandb.log`，并持久保留`--output_dir`中的checkpoint。
  默认`sampling_strategy=uniform`现在使用普通TensorDict replay buffer，不再误创建依赖
  TorchRL C++ segment tree的prioritized buffer；只有显式选择prioritized replay时才要求该扩展。
  Marvin NPZ replay升级为schema 3并显式标记`resfit_normalized_v1`，learner拒绝旧原始坐标
  replay；离线buffer cache加入ACT权重SHA256，避免更换base checkpoint后误用旧cache。
- 2026-08-17：新增`convert_marvin_v3_to_residual_v21.py`，将当前Marvin LeRobot v3的BC与
  failure-aware RL数据转换为ResFiT固定LeRobot可读取的v2.1逐episode布局；RL输出保留原标签并
  从episode边界生成`next.done`，四相机feature显式写入CHW轴名供旧版policy factory识别；
  转换阶段排除BC episode 359和RL episode 556两个异常1帧回合，BC改用持续目标夹爪源数据。
  默认使用嵌入式image parquet，规避`residual`环境当前
  TorchCodec/TorchVision视频解码ABI不兼容，正式BC/RL输出路径分别固定在
  `/home/xlb/code/强化微调ACT/datasets/residual_v21/marvin_usb_bc`和
  `/home/xlb/code/强化微调ACT/datasets/residual_v21/marvin_usb_rl`。
- 2026-08-17：Residual off-policy Marvin 适配改在父仓库`/home/xlb/code/residual-offpolicy-rl`分支
  `feature/marvin-residual-offpolicy`中维护；新增`collect_marvin`单机器人采集入口、
  `marvin_residual_td3_config`配置和本地LeRobot ACT checkpoint参数，使用独立NPZ replay目录，
  禁止Residual learner在Marvin真机上自动随机warm-up。
- 2026-08-17：ResFiT Marvin配置补充本地`marvin_usb_act_success` offline dataset，强制使用冻结ACT
  生成`observation.base_action`；Marvin Bridge reward统一为sparse binary（成功为1，否则为0），
  并修正terminal replay的`final_obs` base action保存逻辑。
- 2026-08-17：撤回自动分类器成功判定及HIL-SERL/Bridge行为改动，保留原始SpaceMouse wrapper；
  本次Residual真机流程由操作者不触碰SpaceMouse，并使用Bridge键盘监听`--human-success-key`。
  Residual层统一按`raw_reward > 0.8`映射为成功1，其余为0；ResFiT learner支持保存
  `marvin.residual_checkpoint` actor bundle，`collect_marvin`支持`--residual-checkpoint`正式在线采集。
- 2026-08-14：补充ACT-RL分轮闭环指令：第一轮`step_00050300`真机5步短测、
  第二轮使用50300采集到独立Replay、审计，并使用严格配对的model/learner继续训练到50600；
  明确当前不自动切换Actor，Bridge仍独立运行。新增命令均使用完整单行，避免续行符丢失。
- 2026-08-14：ACT-RL第一阶段分轮在线训练升级为Replay v2：同时保存未裁剪policy、
  安全command、actual/critic action与执行/干预mask，正确区分`terminated`和`truncated`；
  新增Replay强制审计、训练前自动再审计与单轮policy SHA256限制、
  checkpoint权重配对、按在线transition数量自动75:25/50:50/25:75混合、近期/历史及
  成功/失败/干预分层采样、每条在线数据最多4次更新保护和默认2:1 critic-to-actor频率；
  在线默认改为300步、每100步保存，并将起始模型统一为离线IQL step 50000。
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
- 2026-08-19：新增两份HIL-SERL纯人工demo pickle（各20回合，共40回合、4,707帧）
  的独立转换命令；通过重复`--reference-demo-pkl`合并两个文件，并设置
  `--max-online-episodes 0`，避免把旧checkpoint中的在线回合混入新数据。输出固定为
  `/home/xlb/code/强化微调ACT/datasets/marvin_usb_human40_20260819`，不覆盖已有ACT/Residual数据集。
- 2026-08-19：新增纯人工40回合数据的ACT BC训练命令；训练数据根目录固定为
  `/home/xlb/code/强化微调ACT/datasets`，输出固定为
  `/home/xlb/code/强化微调ACT/outputs/marvin_usb_human40_act_bc`，启用episode级10% hold-out
  验证和W&B在线记录，项目名为`marvin-act-human40-bc`。
- 2026-08-19：修复`convert_marvin_v3_to_residual_v21.py`对LeRobot v3内嵌PNG图像
  (`video_path: null`)的兼容；现在`--media-storage image`可直接从parquet的
  `observation.images.*.bytes`读取四路相机，不再错误调用视频路径格式化。
- 2026-08-19：新增human40 ACT checkpoint的Residual真机rollout命令；冻结
  `marvin_usb_human40_act_bc/best_eval/policy`作为base policy，零residual采集到独立
  `/home/xlb/code/强化微调ACT/outputs/marvin_usb_human40_rl_rollout/replay`，避免覆盖旧的
  `marvin_residual_td3_chunk20_exec5/replay`。
- 2026-08-19：明确Residual LeRobot 0.1.0 BC训练必须读取v2.1数据目录，不能直接把
  v3的`marvin_usb_human40_20260819`作为`--dataset`；human40 BC训练应使用
  `HF_LEROBOT_HOME=/home/xlb/code/强化微调ACT/datasets/residual_v21`和数据集名
  `marvin_usb_human40_bc`，避免因缺少`meta/tasks.jsonl`回退到不存在的Hugging Face repo。
- 2026-08-19：human40 BC中断后，真机rollout默认加载
  `/home/xlb/code/强化微调ACT/outputs/marvin_usb_human40_act_bc/latest/policy`，并将新回放写入
  `/home/xlb/code/强化微调ACT/outputs/marvin_usb_human40_rl_rollout/replay`，不覆盖旧Residual回放。
- 2026-08-19：修正`collect_marvin`的offline数据路径说明；该入口的
  `--offline-dataset-root`必须直接指向具体数据集目录
  `/home/xlb/code/强化微调ACT/datasets/residual_v21/marvin_usb_rl`，不能指向其父目录
  `residual_v21`，否则会因找不到`meta/info.json`而错误回退到Hugging Face查询。
- 2026-08-19：human40数据重新按持续目标夹爪语义转换；先生成
  `/home/xlb/code/强化微调ACT/datasets/marvin_usb_human40_20260819_dense_gripper`，再生成
  `/home/xlb/code/强化微调ACT/datasets/residual_v21/marvin_usb_human40_bc`（40回合/4707帧）。
  原稀疏v3目录`marvin_usb_human40_20260819`已删除；旧稀疏夹爪ACT checkpoint不得继续用于
  rollout，必须重新训练dense BC后再采集Residual数据。
- 2026-08-19：新增`analysis/act_offline_analysis.py`，对dense human40 ACT执行逐帧离线推理，
  输出参考动作/预测动作的均值、方差、MAE、RMSE、episode级积分动作轨迹误差，以及train/eval
  分组图表；默认比较每个样本新建20步ACT队列后的第一个动作。
- 2026-08-19：human40 dense ACT rollout已完成审计：11回合/1109原始transition、
  1087条`n_step=3`可用transition，`ready=true`。Residual TD3必须使用
  `/home/xlb/code/强化微调ACT/outputs/marvin_usb_human40_rl_rollout/replay`和新的
  dense ACT checkpoint，不能混用旧`marvin_residual_td3_chunk20_exec5`断点。
- 2026-08-20：Residual TD3离线Replay缓存命中时改用`LazyMemmapStorage`直接打开磁盘
  memmap，避免每次启动把约35GB图像缓存复制到内存；首次生成缓存仍按原流程执行，后续
  启动应直接复用`outputs/marvin_usb_human40_residual_td3_chunk20_exec5/cache/offline_buffer_cache`。
- 2026-08-20：进一步修复Residual TD3缓存命中启动流程；先读取轻量LeRobot metadata/statistics，
  只有缓存不存在时才构造完整`LeRobotDataset`并扫描14个Arrow分片，避免每次启动重复映射
  `marvin_usb_rl`的全部离线数据。
- 2026-08-20：为Residual TD3启动流程增加`flush`阶段日志，分别标记离线缓存检查、W&B初始化、
  critic warm-up决策、checkpoint保存和learner loop入口，便于定位启动阶段停顿位置。
- 2026-08-20：补充QAgent构建完成、learner checkpoint开始/完成恢复日志，区分视觉网络初始化和
  ReplayBuffer/W&B阶段的耗时。
- 2026-08-20：为Marvin在线NPZ回放增加episode manifest指纹缓存；首次导入147个NPZ后写入
  memmap，后续清单不变时直接加载在线ReplayBuffer，新增episode会自动生成新的缓存指纹。

# LeRobot 对 Marvin USB 插入机器人的部署适配

本目录实现以下隔离架构：

```text
LeRobot ACT（lerobot_xlb）
  -> MarvinRobot / 本机 HTTP RPC
  -> bridge（hilserl）
  -> TrainConfig.get_environment()
  -> 现有 MarvinUSBEnv、wrappers、SDK、相机与安全控制
```

部署包不复制 Marvin SDK、相机或运动控制代码。`RelativeFrame`、动作缩放、工作空间裁剪和实际执行只发生在 HIL-SERL 服务端。默认 bridge 永远是 mock；硬件路径必须双重显式确认。

## 固定数据契约

- 10 Hz。
- 四路 `uint8 HWC 128x128x3`：`side_policy`、`wrist_1buchong`、`wrist_1`、`wrist_2`。
- 24维 state 顺序与 buffer 一致：`gripper_pose(1) + last_action(5) + tcp_force(3) + tcp_pose Euler(6) + tcp_torque(3) + tcp_vel(6)`。
- 5维归一化 action：`dx, dy, dz, drz, gripper`，范围严格为 `[-1, 1]`。
- HIL-SERL 的 `(1, ...)` leading horizon 在 bridge 边界移除。

当前 LeRobot commit `59ab2862` 的 rollout 会过滤掉不以 `.pos/.vel` 结尾的实时标量键。因此插件实时字典使用 `*.pos` 和 `*.vel` 后缀，但向量顺序和值不变。ACT 专用入口直接构造 `observation.state` 和 `action` 向量。

## 安装与 mock 验证

两个环境都从当前工作区 editable 安装；不修改只读上游仓库：

```bash
cd /home/xlb/code/强化微调ACT

/home/xlb/miniconda3/envs/hilserl/bin/python -m pip install -e . --no-deps

PYTHONPATH=/home/xlb/code/lerobot/src \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python -m pip install -e . --no-deps
```

终端一（安全 mock bridge）：

```bash
PYTHONPATH=/home/xlb/code/强化微调ACT/src \
/home/xlb/miniconda3/envs/hilserl/bin/python -m lerobot_robot_marvin.bridge_server \
  --backend mock --mock-success-at-step 20
```

终端二（训练完成后替换 checkpoint）：

```bash
PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/lerobot/src \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python -m lerobot_robot_marvin.act_rollout \
  --checkpoint /ABS/PATH/TO/pretrained_model \
  --device cuda --episodes 1 --max-steps 200 --fps 10 \
  --connect-timeout 30 --reset-timeout 120
```

`MarvinEpisodeController` 在每个 episode 开始显式调用 bridge `reset`，随后调用 policy runner 的 `reset()`，后者同时清除 ACT action queue 和 pre/post processor 状态。`send_action` 不包含 reset 或成功判定。`env.step` 返回的 `next_obs` 会被客户端缓存，下一次 `get_observation` 不会再次采集相机。

## **高风险：真机启动**

只有完成下方验收清单后，才在 `hilserl` 环境显式设置 HIL-SERL 路径并启动。
下面命令可以整段粘贴执行；注意每个续行反斜杠 `\` 必须位于行末，
`/home/xlb/code_marvin/hil-serl/examples` 不可拆成两行：

```bash
cd /home/xlb/code/强化微调ACT

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

ACT 客户端还必须额外添加 `--allow-real-robot`。硬件 bridge 的 `stop` 是终止式急停：调用现有 `env.close()` 下使能并释放资源，恢复需要重启 bridge。服务端 watchdog 默认2秒；reset/step 正在执行时不会误触发，客户端断联或动作 RPC 失败会触发 stop。

硬件 bridge 强制只允许绑定 loopback 地址，拒绝 `0.0.0.0` 或外部网卡。连接机器人/相机和任务 reset 可能明显超过普通 step 的1秒超时，因此分别使用30秒和120秒超时；超时仍会触发终止式 stop。

当前推荐使用 `--human-success-key`：SPACE 标记成功，ESC 标记失败。
如果改用 learned reward，才使用 `--classifier`，二者不能同时传入。
classifier 额外产生的 `side_classifier` 只在服务端用于奖励，ACT 客户端仍只接收固定四路 policy 图像。

控制频率只由 HIL-SERL 的 `MarvinUSBEnv.step()` 维护。ACT 客户端不会再额外 sleep，否则服务端和客户端各等待100ms会使实际频率接近5 Hz。rollout 会打印每回合测得的有效控制频率；模型推理耗时会使它略低于10 Hz。

## 数据集 metadata 注意事项

转换器当前将 state/action names 写成：

```python
"names": {"axes": [...]}
```

训练读取向量通常不受影响，但当前 LeRobot `build_dataset_frame()` 和 `make_robot_action()` 期望 `names` 是列表。实时部署应核对 checkpoint 的 `config.json`、pre/post processor 文件及数据集 `meta/info.json`。若要用通用 `lerobot-rollout/record` 写实时帧，应把 metadata 中两项规范为：

```python
"names": ["gripper_pose", ...]  # state
"names": ["dx", "dy", "dz", "drz", "gripper"]  # action
```

本实现未修改 `convert_marvin_buffers_to_lerobot.py`。ACT 专用入口按固定向量契约运行，不依赖上述字典迭代行为。

## 真机前验收清单

- mock 跑完多 episode，确认每回合 policy reset。
- checkpoint 输入精确为四路相机和24维 state，输出为5维 action。
- shadow mode 检查动作范围、NaN/Inf 和维度，不发送动作。
- 只读观测确认四相机名称、RGB/HWC、状态顺序与 buffer 完全一致。
- 依次验证零动作、单轴极小动作、旋转、夹爪和工作空间限制。
- 验证超时、断线、Ctrl-C 均会停止，且不会重放旧动作。
- 确认 `n_action_steps=1`，控制频率10 Hz；动作缩放没有在 LeRobot 侧重复执行。

ACT 经 checkpoint 反归一化后的有限输出可能轻微超出 `[-1,1]`。部署入口会记录并在模型边界裁剪；NaN/Inf 仍立即拒绝。`MarvinRobot` 和 HIL-SERL 服务端继续执行独立的严格校验/安全限幅，模型侧裁剪不替代机器人侧安全保护。

## 快速测试

```bash
cd /home/xlb/code/强化微调ACT
python -m compileall -q src tests

PYTHONPATH=src /home/xlb/miniconda3/envs/hilserl/bin/python \
  -m unittest discover -s tests -p 'test_protocol.py' -v

PYTHONPATH=src:/home/xlb/code/lerobot/src \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python \
  -m unittest discover -s tests -p 'test_integration.py' -v
```

沙箱若禁止监听 socket，自动测试使用同进程 transport 验证真实 `BridgeService`；协议数组编解码单独测试。实际 localhost HTTP 监听需在正常主机环境做一次 smoke test。

## RL-100 Diffusion Policy 部署

RL-100 仍复用同一个 Bridge。不要把 Marvin SDK、RealSense 或 HIL-SERL
环境复制到 RL-100。RL100 环境只安装本项目的轻量 RPC 包，并通过
`PYTHONPATH` 引入 RL-100：

```bash
cd /home/xlb/code/强化微调ACT

/home/xlb/miniconda3/envs/rl100/bin/python \
  -m pip install -e . --no-deps
```

本包的 Bridge-only 和 RL-100 模块支持 Python 3.8；RL-100 端不需要安装
LeRobot。部署代码只在真正加载 checkpoint 时延迟导入 torch、Hydra 和
`rl_100`。

训练完成后，首先在不接触 Bridge 的情况下检查 checkpoint：

```bash
PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/RL-100/RL-100 \
/home/xlb/miniconda3/envs/rl100/bin/python \
  -m lerobot_robot_marvin.rl100_rollout \
  --checkpoint /ABS/PATH/TO/checkpoints/latest.ckpt \
  --device cuda:0 \
  --inspect-only
```

加载器默认优先使用 `ema_model`，没有 EMA 才使用 `model`，并以 strict
模式恢复 checkpoint 中的 policy 与 normalizer。它拒绝非四相机、非24维
state、非5维 action 或 `n_obs_steps != 2` 的 checkpoint。

然后用转换后的 Zarr 做离线回放。以下命令只预测20次，不连接机器人：

```bash
PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/RL-100/RL-100 \
/home/xlb/miniconda3/envs/rl100/bin/python \
  -m lerobot_robot_marvin.rl100_replay \
  --checkpoint /ABS/PATH/TO/checkpoints/latest.ckpt \
  --dataset /home/xlb/code/RL-100/RL-100/data/marvin_usb_bc.zarr \
  --episode-index 0 \
  --max-predictions 20 \
  --device cuda:0 \
  --output /home/xlb/code/RL-100/RL-100/data/replay_episode000.npz
```

输出包含预测 chunk、示范 chunk、首动作各维 MAE、动作裁剪数和推理耗时。
确认结果后启动 mock bridge：

```bash
PYTHONPATH=/home/xlb/code/强化微调ACT/src \
/home/xlb/miniconda3/envs/hilserl/bin/python \
  -m lerobot_robot_marvin.bridge_server \
  --backend mock \
  --mock-success-at-step 20
```

在 RL100 终端跑完整 RPC 回合：

```bash
PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/RL-100/RL-100 \
/home/xlb/miniconda3/envs/rl100/bin/python \
  -m lerobot_robot_marvin.rl100_rollout \
  --checkpoint /ABS/PATH/TO/checkpoints/latest.ckpt \
  --device cuda:0 \
  --episodes 1 \
  --max-steps 20 \
  --actions-per-inference 1 \
  --motion-scale 0.2
```

DP 推理期间独立 heartbeat 每0.5秒联系 Bridge，防止2秒 watchdog 把正常的
首次 CUDA 推理误判为客户端失联。heartbeat 失败、策略异常、动作 NaN/Inf、
连续5个 chunk 发生越界裁剪时，rollout 都会请求终止式 stop。

真机前先运行 `--dry-run`。该模式会执行任务 reset 并持续推理，但不会调用
`env.step()`，所以仍属于会接触真实硬件的操作，必须添加
`--allow-real-robot`。第一轮实际动作保持：

```text
episodes=1
max_steps=5
actions_per_inference=1
motion_scale=0.2
deterministic DDIM（默认）
```

确认方向和量纲后，再依次增加步数、将 `motion_scale` 提升至0.5/1.0，并将
`actions_per_inference` 从1提升到2或4。夹爪维不参与 motion scale，继续保持
训练标签的 -1/+1 目标状态。

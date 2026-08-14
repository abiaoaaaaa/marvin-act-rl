# LeRobot ACT 训练与 Marvin Buffer 转换说明

## 1. 当前 LeRobot ACT 的训练流程

当前分析对应父目录 `/home/xlb/code/lerobot`，Git commit 为 `59ab2862`。

训练主流程是：

```text
lerobot_train.py
  -> make_dataset()
  -> LeRobotDatasetMetadata
  -> ACTConfig.action_delta_indices = range(chunk_size)
  -> LeRobotDataset 根据 fps 请求未来 chunk_size 个 action
  -> 超出 episode 尾部的位置生成 action_is_pad
  -> ACTPolicy.forward(batch)
  -> L1(action chunk) + kl_weight * CVAE KL
```

因此转换脚本不应预先把单步 action 拼成 chunk。只要 episode 边界、帧顺序和 fps 正确，LeRobot 会按 `chunk_size` 自动构造未来动作序列。

ACT 使用的字段是：

```text
observation.images.<camera>
observation.state
action
```

Marvin 数据转换为：

```text
4 x uint8 RGB image: HWC = 128 x 128 x 3
observation.state: float32[24]
action: float32[5]
fps: 10
```

当前四路策略相机是：

```text
side_policy
wrist_1buchong
wrist_1
wrist_2
```

`side_classifier` 是奖励分类器相机，不默认作为 ACT policy 输入。

## 2. Buffer 与 demo_buffer 的真实语义

`buffer/*.pkl` 是完整 online transition 流，文件内部按 actor 插入顺序排列，跨文件需要按文件名中的数字自然排序。

`demo_buffer/*.pkl` 不是完整 demonstration episode。当前 actor 仅在存在 `intervene_action` 的 step 将 transition 复制到 demo buffer。因此转换脚本把 demo buffer 当作 intervention transition 索引，并通过：

```text
observation.state + action + next_observation.state
```

的稳定哈希，将干预步骤匹配回 online buffer 的完整 episode。

完整轨迹只从 online buffer 恢复。standalone `record_demos.py` pickle 可以通过 `--reference-demo-pkl` 额外加入输出数据集。

## 3. Episode 恢复规则

脚本按以下规则恢复回合：

1. `dones=True` 正常结束 episode。
2. 相邻 transition 必须满足前一条 `next_observations` 等于后一条 `observations`。
3. shard 编号存在缺口时强制断开，避免把缺失数据两侧误拼成一条轨迹。
4. observation 不连续时断开，用于识别未写入 pickle 的 `truncated/reset`。
5. 只有以 `dones=True` 结束的 episode 才被视为完整并可进入 ACT 数据集。

第 5 条是保守策略：当前在线采集代码 reset 条件是 `done or truncated`，但 transition 只保存 `dones=done`，没有保存 truncated。状态不连续可以发现边界，但无法证明中间数据是否缺失，所以推断边界的 episode 不作为优质轨迹。

## 4. 筛选规则

脚本严格执行：

```text
完整 episode
AND 成功（任一步 reward >= success_threshold，默认 0.5）
AND online episode length < max_demo_episode_length
```

`max_demo_episode_length` 按优先级确定：

1. `--max-demo-episode-length` 显式值；
2. `--reference-demo-pkl` 中完整成功 demo 的最大长度；
3. online buffer 中包含 demo_buffer 干预 transition 的完整成功 episode 最大长度。

推荐传入 standalone demo pickle，因为它是真正的完整人工 demonstration；仅从 intervention buffer 推导时，得到的是“含人工干预的 online episode 最大长度”。

严格使用 `<`，长度等于最大 demo 长度的 online episode 不会入选。

## 5. 使用方式

先只分析，不写数据：

```bash
cd /home/xlb/code/强化微调ACT

/home/xlb/miniconda3/envs/hilserl/bin/python \
  convert_marvin_buffers_to_lerobot.py \
  --checkpoint-dir /home/xlb/code_marvin/hil-serl/examples/experiments/marvin_usb_insertion/checkpoints_scale10_acc300_alpha080_4cam_fresh_20260807 \
  --reference-demo-pkl /home/xlb/code_marvin/hil-serl/examples/demo_data/marvin_usb_insertion_20_demos_2026-08-07_10-16-21.pkl \
  --dry-run
```

正式转换需要使用具有 LeRobot 依赖的环境，并让当前父目录 LeRobot 优先进入 `PYTHONPATH`：

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

转换完成后会额外生成：

```text
conversion_report.json
```

其中记录所有识别到的 episode、成功状态、长度、干预步数、边界类型和最终是否入选。

`--max-shards N` 和 `--max-online-episodes N` 仅用于调试。正式转换不要传
`--max-shards`，否则输入只是完整 buffer 的前缀。当前 run 一共有 88 个 online
shard，每个 shard 约 492 MB；dry-run 会顺序扫描一次，正式转换为了避免把大量图像
常驻内存，会先筛选、再顺序扫描一次并写出入选回合。

## 6. ACT 训练示例

数据集生成后，可按当前 LeRobot CLI 使用本地 root：

```bash
cd /home/xlb/code/lerobot

PYTHONPATH=/home/xlb/code/lerobot/src \
/home/xlb/miniconda3/envs/lerobot_xlb/bin/python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=xlb/marvin_usb_act_success \
  --dataset.root=/home/xlb/code/强化微调ACT/datasets/marvin_usb_act_success \
  --policy.type=act \
  --policy.chunk_size=50 \
  --policy.n_action_steps=1 \
  --batch_size=32 \
  --steps=100000 \
  --output_dir=/home/xlb/code/强化微调ACT/outputs/act_marvin_usb
```

对接触任务建议先使用 `n_action_steps=1`，即每步重新观测；`chunk_size` 仍可设为 30～50，让训练目标保留动作块结构。确认部署稳定后，再测试执行 2～5 步前缀。

## 7. 数据条件注意事项

当前目标 run 是 scale10、acc300、4-camera 数据。不要与 scale15/acc200 的 demo 无条件混合。转换脚本只检查 tensor schema，无法从旧 buffer transition 中自动恢复所有物理控制参数。

如果传入 `--reference-demo-pkl`，应确保它与 online buffer 使用相同：

- 4 路 camera key 和裁剪；
- 24 维 state 定义与归一化；
- 5 维 action 定义；
- action scale、EMA 和控制频率；
- 成功奖励定义。

若使用 2026-08-07 run，应优先匹配 2026-08-07 的 standalone demo，而不是 8 月 3/4 日 scale15 数据。

## 8. 已完成的验证

转换脚本已通过以下检查：

- Python 语法检查；
- 人工构造数据的回合恢复、成功筛选和严格长度筛选；
- 当前真实 run 第一个 shard 的解析：图像为 `128x128x3`、state 为 24 维、action 为 5 维；
- 该 shard 中 demo 最大长度为 156，最终筛出一个长度为 143 的完整成功 online 回合；
- 真实四相机数据成功写成默认 video 格式并由 LeRobot 重新读取；
- 使用 `chunk_size=50` 方式读取时得到 `action[50,5]` 和 `action_is_pad[50]`，证明 ACT 的动作块与 episode 尾部 padding 正常。

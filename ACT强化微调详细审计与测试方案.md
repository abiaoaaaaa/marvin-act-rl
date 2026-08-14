# ACT 强化微调详细审计与测试方案

> 审计日期：2026-08-13  
> 适用项目：`/home/xlb/code/强化微调ACT`  
> 当前路线：LeRobot ACT 行为克隆基座 → 离线 Chunk-IQL/优势加权 ACT 直接微调 → 后续离线/在线混合更新

## 1. 结论先行

当前实现已经完成了一条可运行的“直接更新 ACT 参数”的离线强化微调链路：用成功、失败、人工干预轨迹训练 10 步 chunk critic 和 value，再用 IQL 优势权重对 ACT 的 30 步动作块做加权重建。50,000 步模型在训练分布回放上的动作 L1、成功/失败 Q 值分离都优于它继承的 115,000 步 ACT 基座，真机初测也表现良好。

但当前结果还不能严格证明“提升来自 RL 算法”，主要原因是：

1. RL-ACT 比基座额外训练了 40,000 次 actor 更新，并且看到了更大的成功+失败数据集；还缺少同数据、同步数、全权重为 1 的 continued-BC 对照。
2. 当前离线评估是在训练分布上抽样，不是独立 episode hold-out，不能用它判断泛化成功率。
3. critic 只看 24 维状态，不看四路图像；对依赖视觉对齐的 USB 插入，价值函数可能遗漏关键变量。
4. 当前在线 collector 保存的是策略提出的 chunk，没有严格保存人工干预后的实际执行动作；在修复前不能用这些 replay 安全地更新 actor。
5. 当前 RL 会更新除 ResNet18 backbone 外的大部分 ACT，更新范围约 4041 万参数，遗忘风险偏高。

因此，下一阶段最重要的不是立即扩大在线训练，而是先完成：独立验证集、continued-BC 对照、实际执行动作 replay、reward 统一和真机标准化 A/B 测试。当前 50,000 步 RL-ACT 可以冻结为一个候选版本，但不应仅凭 `last` 身份自动成为部署模型。

## 2. 当前系统的数据流和训练流程

```text
HIL-SERL 原始 demo/buffer
        │
        ├─ 成功轨迹筛选 + 持续目标夹爪 ──> ACT BC 数据集
        │                                  │
        │                                  └─> LeRobot ACT（chunk=30）
        │                                           │
        └─ 成功/失败/干预/reward 完整回合 ──> ACT-RL 数据集
                                                    │
                                                    ├─> 10-step Twin Q + V
                                                    └─> 优势加权 ACT 重建
                                                             │
                                                             └─> RL-ACT checkpoint

真机：lerobot_xlb 中的 ACT 推理客户端
  ──HTTP Bridge──> hilserl 中的 MarvinEnv、相机、控制器、reset、人工成功/失败键
```

ACT 和 RL-ACT 的真机执行路径相同，区别只在加载的 `pretrained_model`。HIL-SERL Bridge 不会主动加载 ACT 或旧 RL actor；策略权重由 `lerobot_xlb` 环境中的 rollout 客户端加载。这样可以把模型推理和机器人 SDK 依赖隔离开。

## 3. 当前使用的数据集

### 3.1 原始 ACT 行为克隆数据

数据目录：

```text
/home/xlb/code/强化微调ACT/datasets/marvin_usb_act_success_dense_gripper
```

主要性质：

| 项目 | 当前值 |
|---|---:|
| 回合数 | 584 |
| 帧数 | 49,574 |
| 训练时排除 | episode 359 |
| 训练实际回合 | 583 |
| 数据内容 | 只保留优质成功轨迹 |
| 夹爪标签 | 持续目标状态，关闭期间 `-1`，打开期间 `+1` |
| 图像 | `side_policy`、`wrist_1buchong`、`wrist_1`、`wrist_2`，128×128，10 Hz |
| 状态 | 24 维 |
| 动作 | 5 维：`dx, dy, dz, drz, gripper` |

ACT 训练配置使用 `eval_split=0.1`。这是 LeRobot 在初始数据集上的评估划分，但当前保存目录没有自动标记一个“best checkpoint”；训练总步数是 200,000，最后 checkpoint 是 `200000`，RL 初始化采用的是 `115000`。因此：

- `115000`：当前 RL 实验的严格基座；
- `200000`：原始 ACT 的 last；
- `ACT best`：应通过固定 hold-out 指标重新选择，不能先验认定为上述任意一个。

### 3.2 ACT-RL 离线数据

数据目录：

```text
/home/xlb/code/强化微调ACT/datasets/marvin_usb_act_rl_offline
```

转换后统计：

| 项目 | 转换结果 | 训练实际使用 |
|---|---:|---:|
| 回合数 | 808 | 807 |
| 帧数 | 90,119 | 90,118 |
| 成功回合 | 690 | 689 |
| 失败回合 | 118 | 118 |
| 人工干预帧 | 32,552 | 32,552 |
| 排除项 | — | episode 556（1 帧异常成功回合） |
| 来源 | 788 个 online buffer 回合 + 20 个 reference demo | 同左，减 episode 556 |

数据包含四路相机、24 维状态、5 维动作、逐步 reward、terminated、成功/失败和 intervention 元数据。当前训练实际使用了成功与失败轨迹，但 intervention 标记尚未参与采样或 actor 权重计算。

当前 reward 还需注意版本一致性：旧 buffer 的动作平滑惩罚与当前 MarvinEnv 无分类器在线 reward 的尺度不完全一致。未来混合离线和在线 replay 前，应同时保存 `raw_reward`、统一的 `canonical_reward` 和 `reward_version`，否则 critic 会把“数据来源”误当成价值差异。

### 3.3 当前尚未建立严格 RL hold-out

当前 ACT-RL 数据的 `splits` 是 `train: 0:808`，离线 IQL 训练从除 556 外的所有 episode 采样。`evaluate_offline.py` 也是从同一数据分布抽样，因此报告中的 L1、Q/V 和策略漂移属于训练分布诊断，不是无偏验证指标。

建议按 episode 划分固定 manifest，且按以下字段分层：

- 成功/失败；
- reference demo/online buffer；
- 有干预/无干预；
- 回合长度区间；
- 采集 shard 或时间段。

建议保持 80% train、10% validation、10% test。test 只在最终选模时打开一次，避免反复调参污染测试集。

## 4. ACT 基座网络架构

当前原始 ACT 与 RL 微调后的 ACT 架构完全一致：

| 模块 | 配置 |
|---|---|
| 输入图像 | 四路 128×128 RGB |
| 状态输入 | 24 维 proprioception/force/last action |
| 动作输出 | 30×5 动作块 |
| 视觉 backbone | 共享 ResNet18 |
| Transformer 维度 | 512 |
| 注意力头 | 8 |
| Transformer encoder | 4 层 |
| Transformer decoder | 1 层 |
| VAE | 启用，latent dim=32，VAE encoder=4 层 |
| KL 权重 | 10 |
| 归一化 | STATE/ACTION/VISUAL 均按 checkpoint preprocessor 的 MEAN_STD |
| checkpoint 默认执行长度 | `n_action_steps=1` |

ACT 总参数约 51,578,757。视觉 backbone 约 11,166,912 参数；其余约 40,411,845 参数包括 VAE encoder、Transformer encoder/decoder、状态和 latent 投影、动作头等。

训练时 ACT 使用动作真值经过 CVAE encoder 得到 posterior latent，并优化动作 L1 加 KL；部署时没有动作真值，通常采用 prior 的确定性中心 `z=0` 生成动作。这一区别对 RL prior 和 actor 更新路径很重要。

## 5. 当前离线强化微调方法

### 5.1 更准确的算法名称

当前实现可描述为：

> 固定 10 步前缀的 Chunk-IQL critic + advantage-weighted ACT reconstruction

它借鉴 IQL/AWAC，但不是完整的 Q-Chunking、SAC 或 PPO：

- critic 在离线数据动作上学习，不对任意策略动作做显式 Q 最大化；
- value 使用 expectile regression；
- actor 用指数优势权重加权已有数据动作的 ACT 重建损失；
- 没有 ACT 动作分布的 PPO ratio，也没有在线 SAC actor objective；
- 没有从 ACT 采样多个候选 chunk 再 Best-of-N 选 Q 最大者。

### 5.2 Critic 和 value

当前配置：

| 参数 | 值 |
|---|---:|
| `critic_horizon` | 10 |
| `executed_steps` | 10 |
| `gamma` | 0.99 |
| expectile | 0.7 |
| Q/V hidden dims | 512, 512, 256 |
| critic LR | 3e-4 |
| value LR | 3e-4 |
| target tau | 0.005 |
| critic 输入图像 | 否，`use_images=false` |

critic 输入是 24 维状态与前 10×5 维动作拼接，共 74 维。Twin Q 和 V 都是带 LayerNorm、Mish 的 MLP；critic/value 可训练参数合计约 128 万。

10 步 TD 目标为：

```text
y_t = r_t^(10) + gamma^actual_h × (1-terminal) × V(s_(t+actual_h))
```

V 使用 `min(Q1,Q2)-V` 的 0.7 expectile loss。target Q 和 target encoder 用 `tau=0.005` 软更新。

### 5.3 ACT actor 更新

IQL 优势与权重：

```text
A(s,a_chunk) = min(Q1,Q2) - V(s)
w = clip(exp(3 × A), max=100)
```

ACT actor loss：

```text
L_actor = weighted_L1(action_hat_30, data_action_30)
          + 10 × KL(q(z|state, action) || N(0,I))
          + 0.1 × L1(action_hat, frozen_reference_action)
```

critic 只评价前 10 步，优势权重却施加到完整 30 步 ACT chunk。这是一个可用的工程近似，但意味着后 20 步没有被 critic 逐步验证。

### 5.4 参数更新范围

训练前 10,000 步只更新 Q/V，ACT actor 不更新。10,001～50,000 步每步同时更新 critic、value 和 ACT。

当前参数范围如下：

| 模块 | 是否更新 | 学习率/方式 |
|---|---|---|
| ACT ResNet18 backbone | 冻结 | `backbone_lr=0` |
| ACT VAE encoder | 更新 | actor LR `3e-5` |
| ACT Transformer encoder | 更新 | actor LR `3e-5` |
| ACT Transformer decoder | 更新 | actor LR `3e-5` |
| ACT state/latent/action projections 和 action head | 更新 | actor LR `3e-5` |
| Twin Q + observation encoder | 更新 | `3e-4` |
| V network | 更新 | `3e-4` |
| target Q/encoder | EMA | `tau=0.005` |
| frozen reference ACT | 冻结 | 仅计算 prior |

这意味着 ACT 约 78% 的参数可被 RL actor loss 更新。第一轮严谨消融建议只更新 decoder、action head 和 decoder position embedding，约 540 万参数；如果验证集不足，再逐步解冻最后 1～2 个 Transformer encoder layer。VAE encoder 暂时冻结更容易让 RL 训练路径与部署 `z=0` 路径一致。

### 5.5 当前 prior 的隐患

当前可训练 ACT 处于 train mode，会从 posterior latent 采样；frozen reference 处于 eval mode，通常走 `z=0`。因此 prior L1 可能同时混入“参数变化”和“latent 路径不同”，不是纯粹的策略行为距离。

建议把 prior 改为：同一个观测下，当前策略和 frozen reference 都使用确定性 `z=0`，比较部署动作；标准 CVAE BC loss 可作为单独辅助项保留。这样 prior 才直接约束真实部署策略漂移。

## 6. 当前非真机评估结果

### 6.1 50,000 步完整诊断

在 2,048 个训练分布样本上的结果：

| 指标 | RL-ACT 50k | ACT 基座 115k | 解释 |
|---|---:|---:|---|
| 全部样本动作 L1 | 0.4348 | 0.5010 | RL-ACT 更接近该数据集动作 |
| RL-ACT 与基座 L1 漂移 | 0.2039 | — | 策略已发生明显变化 |
| 成功数据 `Q_data` | 0.5743 | — | critic 对成功轨迹估值较高 |
| 失败数据 `Q_data` | 0.3336 | — | 成功/失败 Q gap≈0.2407 |
| 第一动作越界样本率 | 88.62% | 91.21% | 归一化边界外出现很普遍 |
| 第一动作最大平均超界量 | 0.0304 | 0.0274 | 超界幅度较小，部署会裁剪 |

越界率高但平均超界量小，说明大量预测只是轻微超过数据归一化边界；仍需按动作维分别报告超界率和裁剪率，尤其关注 `dx/dy/dz/drz`，不能只看聚合值。

### 6.2 checkpoint 趋势（固定 256 样本快速诊断）

| RL step | model L1 | base L1 | model-base drift | success-failure Q gap | 平均最大超界量 |
|---:|---:|---:|---:|---:|---:|
| 15k | 0.4626 | 0.4868 | 0.1969 | 0.1987 | 0.0257 |
| 25k | 0.4404 | 0.4868 | 0.1940 | 0.2140 | 0.0271 |
| 35k | 0.4343 | 0.4868 | 0.2034 | 0.2241 | 0.0275 |
| 45k | 0.4264 | 0.4868 | 0.2102 | 0.2164 | 0.0324 |
| 50k | 0.4203 | 0.4868 | 0.2073 | 0.2515 | 0.0287 |

这组结果支持 50k 在训练分布上的拟合和 Q 分离继续改善，但不能代替独立验证或真机成功率。不同评估样本数的结果不能直接横向比较。

### 6.3 还缺少的关键对照

为了把“更多训练/更多数据”的收益与“RL 优势权重”的收益分开，至少需要：

| 实验 | 初始化 | 数据 | actor 权重 | 更新步数/范围 |
|---|---|---|---|---|
| E0 | ACT 115k | — | — | 不更新，严格 RL 基座 |
| E1 continued BC | ACT 115k | 同 ACT-RL 数据 | 全部为 1 | 与 RL 相同步数/范围 |
| E2 success-only BC | ACT 115k | 仅成功子集 | 全部为 1 | 与 RL 相同步数/范围 |
| E3 当前 IQL | ACT 115k | 成功+失败 | `exp(3A)` | 当前方案 |
| E4 gated IQL | ACT 115k | 成功+失败+干预 | top-advantage/gating | 同步数/范围 |
| E5 visual critic | ACT 115k | 同 E4 | gated/weighted | critic 加视觉特征 |

每个实验至少 3 个随机种子。否则单个 seed 的波动可能被误判为算法收益。

## 7. 建议的基于数据集模型输出评估

### 7.1 选模原则：last、offline-best 和 real-best 分开

- `last`：最后保存的 checkpoint，只用于完整性和退化诊断。
- `offline-best`：在固定 validation episodes 上按预先定义的综合指标选出，不能看 test。
- `real-best`：只在 offline-best 候选中进行真机测试后，按真机主指标选出。

对当前实验必须做两类比较：

1. **算法归因比较**：RL-ACT 50k 对 ACT 115k，因为 115k 是它的确切初始化。
2. **最终模型比较**：RL offline-best 对 ACT offline-best；ACT 200k last 只作为附加诊断，不应替代 ACT best。

### 7.2 数据集输出指标

建议在 episode-level hold-out 上计算，并按 success/failure、intervention/non-intervention、接近/接触阶段分组。

| 类别 | 指标 | 用途 |
|---|---|---|
| 动作拟合 | 完整 30 步 normalized L1/L2 | 与现有评估连续 |
| 动作拟合 | 反归一化后每维 MAE/RMSE | 得到真实动作尺度误差 |
| 执行相关 | prefix@1、@3、@5、@10 L1 | 对应真机执行长度 |
| 第一动作 | first-action 每维 MAE、方向一致率 | 闭环 h=1 的核心指标 |
| 夹爪 | open/close accuracy、macro-F1、切换提前/延迟帧数 | 处理离散持续目标标签 |
| 安全 | 每维 OOB rate、max/mean excess、部署裁剪率 | 判断动作是否依赖裁剪 |
| 平滑性 | `||a_t-a_(t-1)||`、jerk、符号抖动次数 | 判断 chunk 是否抖动 |
| 策略漂移 | RL-ACT 与 base 的每维/prefix L1 | 判断遗忘程度 |
| 重规划一致性 | 相邻观测预测重叠部分的 disagreement | 评估 chunk 在 receding horizon 下是否稳定 |
| 价值质量 | success-failure Q gap、AUC、return-Q 相关性 | 判断 critic 排序而非只看均值 |
| 校准 | 按 Q 分桶的真实成功率/回报 | 判断 Q 能否用于选模/选 chunk |
| 性能 | p50/p95 推理耗时、GPU 显存 | 判断 h=1 是否能维持 10 Hz |

离线模型综合排序建议以安全门槛优先：若动作 OOB/裁剪、夹爪时序或重规划不一致明显恶化，不能仅凭 L1 更低入选。通过安全门槛后，再按 validation 的 prefix L1、Q 排序质量和策略漂移选择 2～3 个真机候选。

### 7.3 当前离线命令的正确定位

现有 `evaluate_offline.py` 可用于同一批样本上比较 ACT 115k 与各 RL checkpoint，但报告明确是训练分布诊断。它适合检查 checkpoint 是否学习、是否发生过大漂移、critic 是否把成功估得更高，不适合宣称独立成功率。

后续应扩展 evaluator 支持固定 `--episodes-file` 和按动作维/prefix 输出；在扩展完成前，不要通过改变 `max-samples` 后的结果排序 checkpoint。

### 7.4 当前可复现的非真机评估命令

以下命令不会连接 Bridge 或控制真机。它固定比较 RL-ACT 50k 与其严格基座 ACT 115k，并把 2,048 样本诊断写入 JSON：

```bash
cd /home/xlb/code/强化微调ACT

TORCH_HOME=/home/xlb/code/强化微调ACT/.cache/torch HF_HOME=/home/xlb/code/强化微调ACT/.cache/huggingface PYTHONPATH=/home/xlb/code/强化微调ACT/src:/home/xlb/code/lerobot/src /home/xlb/miniconda3/envs/lerobot_xlb/bin/python -m act_rl.scripts.evaluate_offline \
  --config /home/xlb/code/强化微调ACT/configs/act_rl/marvin_offline_iql.json \
  --base-checkpoint /home/xlb/code/强化微调ACT/outputs/act_dense_gripper_chunk30_seed1000_500k/checkpoints/115000/pretrained_model \
  --model-checkpoint /home/xlb/code/强化微调ACT/outputs/act_rl_offline_iql/checkpoints/step_00050000/pretrained_model \
  --learner-checkpoint /home/xlb/code/强化微调ACT/outputs/act_rl_offline_iql/checkpoints/step_00050000.pt \
  --dataset-root /home/xlb/code/强化微调ACT/datasets/marvin_usb_act_rl_offline \
  --repo-id xlb/marvin_usb_act_rl_offline \
  --outcomes /home/xlb/code/强化微调ACT/datasets/marvin_usb_act_rl_offline/meta/rl_outcomes.json \
  --exclude-episodes 556 \
  --device cuda \
  --batch-size 8 \
  --max-samples 2048 \
  --num-workers 4 \
  --output /home/xlb/code/强化微调ACT/outputs/act_rl_offline_iql/eval_step_50000.json
```

比较其他 RL checkpoint 时，必须同时替换 `--model-checkpoint`、`--learner-checkpoint` 和 `--output` 中的 step；不要把不同 actor 与不同 critic/value checkpoint 混配。正式 hold-out evaluator 实现后，应使用固定 episode manifest 重新生成所有结果。

## 8. 真机 ACT 详细测试设计

### 8.1 chunk size 与执行长度

两个模型的网络 `chunk_size` 都是 30，即每次预测 30 个动作。真机参数 `--n-action-steps` 是每次推理后连续执行其中多少步，再重新观察和规划。

本轮建议比较：`n_action_steps ∈ {1, 3, 5, 10}`。它不是重新训练 chunk=1/3/5/10，而是在同一个 chunk=30 模型上改变 receding-horizon 执行前缀。

USB 接触任务优先从 1 开始。只有前一档无安全问题，才升级到 3、5、10。`10` 与当前 critic horizon 一致，但 open-loop 时间达到 1 秒，可能在接触时放大误差，所以它是研究对照，不是默认推荐值。

### 8.2 模型矩阵

第一阶段最小矩阵：

| 模型 ID | checkpoint | 作用 |
|---|---|---|
| ACT-BASE | ACT `115000` | RL 严格初始化基线 |
| RL-50K | RL `step_00050000` | 当前 RL 候选 |

第二阶段完整矩阵：

| 模型 ID | 选择方法 |
|---|---|
| ACT-BASE | 固定 115000，做算法归因 |
| ACT-LAST | 200000，检查长训末期表现 |
| ACT-BEST | 独立 validation 选出 |
| RL-LAST | 当前训练最后点 50000 |
| RL-BEST | 独立 validation 从 15k～50k 中选出 |
| BC-CONT | continued-BC 最佳点，排除额外训练混淆 |

不要把所有中间 checkpoint 都上真机。先用 hold-out 筛选，再让 ACT-BEST、RL-BEST 和 BC-CONT 进入正式真机对比。

### 8.3 分阶段安全放量

每个模型/执行长度按以下门槛递进：

1. 只加载 checkpoint，不连接真机，确认 config、pre/postprocessor 和输出维度。
2. Mock Bridge 连续 3 回合，确认 reset、终止、断开和动作裁剪。
3. 真机 `max_steps=5`，每个组合 3 回合；出现 ESC、危险力峰值、连续裁剪或异常夹爪动作立即停止该组合。
4. 真机短回合 5～10 次，达到安全门槛后进入完整回合。
5. 正式测试每个组合至少 20 回合；若成功率差异较小，增加到 30～50 回合。

执行长度升级顺序固定为：

```text
1 -> 3 -> 5 -> 10
```

### 8.4 公平性和试验顺序

- 相同插头、插座、机器人初始位姿、相机曝光、控制频率和 workspace 限制；
- 记录每次 reset 后的 TCP 初始误差，超出预设范围的回合标为 invalid 并重做；
- 不连续跑完一个模型再跑另一个，使用随机区组或 ABBA 顺序降低温漂、光照和机械变化影响；
- 操作者在测试开始前看不到当前模型 ID，成功/失败判定规则固定；
- 人工干预回合保留并计为 safety/intervention 事件，不能从成功率分母中静默删除；
- 每次测试记录 checkpoint 绝对路径、文件哈希、代码版本和参数，禁止使用含糊的 `latest` 路径。

一个两模型、四执行长度、每组合 20 回合的正式矩阵共 160 回合。先完成 3 回合安全筛查，再决定是否跑满，避免对明显不安全的 h=10 浪费真机次数。

### 8.5 真机核心指标

主指标：

| 指标 | 定义 |
|---|---|
| 成功率 | 成功回合/所有有效回合，报告 Wilson 95% CI |
| 无干预成功率 | 成功且没有 SpaceMouse/人工接管的比例 |
| 安全终止率 | ESC、workspace/safety stop、异常力终止比例 |

次指标：

| 类别 | 指标 |
|---|---|
| 效率 | 成功回合时间、控制步数、推理次数、重规划次数 |
| 接触质量 | 峰值/均值 TCP force、force impulse、接触持续时间 |
| 稳定性 | 动作裁剪率、OOB 幅度、轨迹 jerk、方向反转次数 |
| 精度 | 最终 TCP 位置/姿态误差、插入深度或任务完成位置 |
| 夹爪 | 错误开闭次数、首次切换时间、重复命令次数 |
| 人机协作 | 干预回合率、干预步数、首次干预时间 |
| 系统性能 | 推理 p50/p95、控制周期 missed deadline、Bridge timeout |

若没有可靠的插入深度传感器，主成功标签仍由固定人工规则产生，但应同步保存视频和末端力/位姿，便于事后盲审。

### 8.6 结果判定建议

部署候选不能只看成功率点估计，建议按以下顺序判断：

1. 安全终止率、峰值力、干预率不劣于 ACT-BEST；
2. 成功率及置信区间优于或至少不劣于 ACT-BEST；
3. 成功回合时间、动作平滑和夹爪时序没有明显退化；
4. 多个随机区组中提升方向一致，而不是集中在某一时段；
5. 若 RL-LAST 与 RL-BEST 不同，默认部署 RL-BEST，保留 RL-LAST 作为诊断结果。

对当前任务，预期 `n_action_steps=1` 最安全，`3` 可能降低推理负担且保持闭环，`5` 和 `10` 更容易在接触阶段积累 open-loop 误差。最终结论必须由上述矩阵给出，而不是用 critic horizon=10 推导部署应执行 10 步。

## 9. 建议的测试记录模板

### 9.1 实验级元数据

```yaml
experiment_id: 2026-08-13_act_vs_rl_horizon
operator: xlb
task: marvin_usb_insertion
bridge_backend: hardware
bridge_classifier: false
human_success_key: true
fps: 10
camera_serials:
  wrist_1: 427622274205
  wrist_2: 427622272953
  side_policy: 036422060870
models:
  ACT_BASE:
    checkpoint: /home/xlb/code/强化微调ACT/outputs/act_dense_gripper_chunk30_seed1000_500k/checkpoints/115000/pretrained_model
    checkpoint_sha256: FILL_ME
  RL_50K:
    checkpoint: /home/xlb/code/强化微调ACT/outputs/act_rl_offline_iql/checkpoints/step_00050000/pretrained_model
    checkpoint_sha256: FILL_ME
n_action_steps: [1, 3, 5, 10]
episodes_per_cell: 20
order: randomized_block_or_ABBA
success_rule: FILL_ME_BEFORE_TEST
safety_force_limit: FILL_ME_BEFORE_TEST
invalid_reset_tolerance: FILL_ME_BEFORE_TEST
```

安全阈值必须沿用当前 MarvinEnv/控制器已经验证的限制；没有经过专项验证时，不要仅为了测试修改 workspace 或力阈值。

### 9.2 每回合 CSV 字段

```text
experiment_id,block_id,episode_id,model_id,checkpoint_sha256,n_action_steps,initial_tcp_error_mm,initial_rot_error_deg,success,intervened,intervention_steps,esc_or_safety_stop,invalid_reset,steps,elapsed_s,inference_p50_ms,inference_p95_ms,max_force_n,mean_contact_force_n,force_impulse_ns,action_clip_rate,max_action_excess,gripper_wrong_transitions,final_position_error_mm,final_rotation_error_deg,notes,video_path,trajectory_path
```

### 9.3 汇总表

| 模型 | n_action_steps | 有效回合 | 成功率 [95% CI] | 无干预成功率 | 干预率 | 安全终止率 | 成功时间中位数 | 峰值力 p95 | 动作裁剪率 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ACT-BASE | 1 |  |  |  |  |  |  |  |  |  |
| ACT-BASE | 3 |  |  |  |  |  |  |  |  |  |
| ACT-BASE | 5 |  |  |  |  |  |  |  |  |  |
| ACT-BASE | 10 |  |  |  |  |  |  |  |  |  |
| RL-50K | 1 |  |  |  |  |  |  |  |  |  |
| RL-50K | 3 |  |  |  |  |  |  |  |  |  |
| RL-50K | 5 |  |  |  |  |  |  |  |  |  |
| RL-50K | 10 |  |  |  |  |  |  |  |  |  |

## 10. 在线训练前必须修复的项目

### P0：Replay 语义与安全

1. 同时保存 `proposed_chunk` 和每一步 Bridge 返回的 `actual_action`；critic/actor 训练使用实际执行动作。
2. 保存 `actual_h`、`exec_mask`、逐步 intervention mask、terminated、truncated；提前终止后不能把未执行动作标成有效。
3. 保存 `raw_reward`、`canonical_reward`、`reward_version`，统一离线和在线 reward 尺度。
4. 固定 checkpoint 哈希和 policy version；collector 不允许模糊加载 `latest`。
5. 修复前只允许 Mock collector 或只读诊断，不用当前在线 replay 更新真机 actor。

### P1：离线可归因评估

1. 建立 episode-level train/validation/test manifest。
2. 增加 continued-BC、success-only BC 和当前 IQL 对照。
3. prior 改为当前与参考策略都走确定性 `z=0`。
4. actor 第一轮仅更新 decoder/action head，再消融大范围解冻。
5. evaluator 增加每维、prefix@1/3/5/10、夹爪和重规划一致性指标。

### P2：价值函数与在线混合

1. critic 使用冻结 ACT 视觉特征或保留相机身份的多视角融合，避免简单平均四路特征。
2. 在线数据比例从 20% 起步：80/20 → 70/30 → 数据充分后 50/50。
3. 新在线 replay 加入后先进行 1,000～2,000 步 critic recalibration，再小学习率更新 decoder/head。
4. 建立 candidate → Mock/离线门槛 → 5 步真机 → 完整 A/B → 手工 promote/rollback 流程。

## 11. 与调研论文和开源项目的对应关系

- **IQL/AWAC**：当前 expectile V、Twin Q 和指数优势加权行为回归来自这一路线；ACT L1 是 log-likelihood policy extraction 的工程替代。
- **Q-Chunking/DQC/AQC**：当前 10 步 Q 已具备 chunk 价值传播雏形，但还没有多候选 Best-of-N、长评价短执行解耦和自适应执行长度。
- **RLPD/HIL-SERL**：可复用 actor-learner、离线/在线 replay、人类干预和真机工程；当前 50/50 混合思想来自该路线，但上线初期不宜立即固定 50/50。
- **AC3/Evo-RL**：可借鉴成功/高优势 gating、top-percentile 筛选、人工干预正样本处理，验证指数权重是否过于温和。
- **PAC-ACT**：适合作为以后显式 Gaussian chunk PPO 的独立研究分支，不建议与当前 IQL 链路同时大改。
- **RL-100**：适合借鉴迭代数据飞轮、candidate 验收和回滚流程，不直接复制 diffusion-specific PPO 到 ACT。
- **Residual off-policy RL**：当前用户选择直接更新 ACT，因此不作为主线；仍可保留为安全基线，判断直接 ACT 微调是否值得额外风险。
- **RLinf**：更适合未来 rollout/分布式训练基础设施，不应成为当前 ACT 离线算法的核心依赖。

## 12. 推荐执行路线

```text
冻结并登记当前 ACT-115k、ACT-200k、RL-50k
  -> 建立 episode-level validation/test manifest
  -> 完成 ACT best 与 RL best 的离线筛选
  -> 训练 continued-BC / success-only BC 对照
  -> 修正 deterministic prior 和 actor 更新范围
  -> 两模型 × h={1,3,5,10} 真机安全筛查
  -> ACT-BEST / BC-CONT / RL-BEST 正式随机区组 A/B
  -> 修复 actual_action replay、reward version 和 exec mask
  -> 80/20 起步的离线/在线 mixed critic recalibration
  -> decoder/head 小步在线更新
  -> candidate 验收后人工 promote，始终保留 rollback checkpoint
```

当前最合理的主线仍是：**ACT 直接参数微调 + IQL/AWAC 式优势加权监督更新 + HIL-SERL 数据飞轮**。等这条路线的归因、验证和 replay 语义都扎实后，再增加视觉 critic、gated advantage、Q 候选选择或单独实现 PAC-ACT。

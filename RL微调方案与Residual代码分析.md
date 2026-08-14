# RL 微调方案与 Residual 代码分析

更新时间：2026-08-11

本文整理当前项目对 ACT、Diffusion Policy、VLA 以及 residual/off-policy、chunk-level RL 的综合判断，并补充分析父目录 `../residual-offpolicy-rl` 的实际代码结构，以及它和 LeRobot 的关系。

## 1. 结论

当前最适合作为起始实现的是：

```text
LeRobot ACT
    + ResFiT / residual-offpolicy-rl
    + HIL-SERL/RLPD 风格的 replay、奖励和真机采集
    + 接触阶段门控的 bounded residual
```

推荐的研究演进顺序是：

1. 冻结 ACT，只训练逐步 residual actor-critic。
2. residual 路线稳定后，增加 ACT candidate proposal + Q-Chunking。
3. 最后再尝试 PAC-ACT/AC3 式 action-chunk policy 更新。
4. Diffusion Policy + RL-100、VLA + RLinf 作为独立模型族，不要第一步强行接入 ACT 的 RL trainer。

核心原因是：ACT 已经提供了强行为先验，但原始 ACT 不是标准的 RL stochastic actor；ResFiT 可以把 ACT 当黑盒 base policy，只学习一个受限的闭环修正策略，避免直接处理 ACT 的 CVAE latent、整段 chunk likelihood 和全参数遗忘问题。

## 2. 模型与 RL 方法的选择

### 2.1 ACT

ACT 一次预测未来动作块，但 LeRobot 的推理接口最终按单步动作执行：动作块被放入 queue，或经过 temporal ensemble 后输出当前动作。因此 ACT 的部署动作单位和训练时的 action chunk 语义并不完全相同。

原始 ACT 不适合直接套普通 PPO/SAC，主要原因是：

- CVAE latent 不是现成的 RL policy distribution；
- chunk 展平后动作维度较高；
- temporal ensembling 后，实际执行动作可能不是某个单独 chunk 的动作；
- 普通单步 critic 与 chunk policy 的时间尺度不一致；
- 全参数 RL 更新容易破坏已有抓取和接近能力。

因此第一阶段应把 ACT 视为冻结的行为先验。

### 2.2 Diffusion Policy

Diffusion Policy 的最终 action likelihood 不天然适合普通 PPO。DPPO/RL-100 的关键是把去噪过程中的 transition 作为 RL policy interface，而不是直接对最终动作黑盒计算 `log pi`。

它更适合专门研究 diffusion policy 的 RL post-training，但需要维护 diffusion-specific 的采样、去噪时间和环境时间对齐逻辑。除非研究目标就是 Diffusion Policy RL，否则不建议把它作为当前 ACT 项目的第一起点。

### 2.3 VLA

VLA 需要区分 action-token head 和连续 flow/action expert：

- action-token VLA 可以对 token 概率求和，较自然地使用 PPO；
- flow/action expert 没有简单的 exact `log pi`，需要 flow-specific policy gradient、advantage conditioning 或 critic-guided policy improvement。

RLinf 更接近 VLA/embodied RL 的训练编排和分布式系统，而不是一个可以无修改替换 ResFiT 的 residual 算法。若目标是语言泛化，应以 RLinf 为主线；若目标是 USB 插入的局部精度提升，VLA 会引入过多额外变量。

### 2.4 方法排序

| 目标 | 首选 | 第二选择 | 暂不作为起点 |
|---|---|---|---|
| 真机快速验证 | ACT + ResFiT | ACT + HIL-SERL/RLPD | 全量 PAC-ACT PPO |
| 稀疏奖励长时程 | residual warm-up 后 Q-Chunking | AC3 | 直接随机探索 |
| ACT 本体研究 | PAC-ACT | AC3/QC | 无 prior 的全参数 PPO |
| Diffusion RL | RL-100 | DPPO | 把 DP 当普通 Gaussian actor |
| VLA/语言泛化 | RLinf | token/flow 专用 RL | 直接套 ACT residual trainer |

## 3. 推荐的三阶段路线

### 阶段一：ACT + bounded residual off-policy RL

策略结构：

```text
o_t
 ├── ACT frozen ----------------> a_base_t
 └── residual actor(o_t,a_base_t) -> delta_t

a_exec_t = SafetyClip(a_base_t + delta_t)
```

建议：

- 冻结 ACT 全部参数；
- 只在 pre-align/contact/insert 阶段启用 residual；
- 接近、抓取和搬运阶段保持纯 ACT；
- residual 初始幅度为正常动作范围的 5%～10%；
- 夹爪动作先沿用 ACT 或状态机；
- residual 优先输出末端笛卡尔 twist，再由现有控制器/IK 转换；
- 使用 demonstrations 初始化 offline buffer，再在线 off-policy 更新；
- 保留人工接管和 safety abort 数据；
- 训练时 critic 输入 observation、base action 和 residual/full action。

这一步的验收问题是：在不降低 ACT 原始抓取/接近能力、不增加安全终止的情况下，RL 是否改善接触成功率和插入耗时。

### 阶段二：ACT candidate proposal + Q-Chunking

```text
ACT(o,z_1...z_N) -> candidate chunks
candidate chunks -> Q(s,A) -> Best-of-N
Best chunk -> execute short prefix -> replan
```

先冻结 ACT，只训练 chunk critic。ACT 需要开放 CVAE latent、增加 stochastic adapter，或使用 Gaussian action head 来生成候选。实际执行窗口应短于 critic 的评价窗口：例如 critic 看 5～10 步，接触阶段只执行 1～3 步。

critic 排序可靠后，再使用 Q-filter 或 advantage-weighted regression 更新 ACT 的小 adapter/action head。不要从第一天就更新视觉 backbone。

### 阶段三：PAC-ACT/AC3 式端到端 chunk RL

PAC-ACT 的合理复现顺序是：

1. 在仿真中实现 chunk return、chunk value、GAE/PPO ratio。
2. 固定 ACT，只训练 critic。
3. 只更新 action head 或 adapter，并加入 frozen ACT behavior prior。
4. 稳定后才考虑 Transformer 的局部解冻。

这条路线研究价值高，但 on-policy 样本效率和真机安全成本都高于 residual off-policy。

## 4. Residual 仓库的实际结构

仓库位置：`../residual-offpolicy-rl`

主要目录：

```text
residual-offpolicy-rl/
└── resfit/
    ├── lerobot/                  # 对 LeRobot 的策略/训练适配层
    │   ├── policies/act/           # ACTConfig、ACTPolicy、ACT 模型实现
    │   ├── policies/diffusion/     # Diffusion policy 适配代码
    │   ├── configs/policies.py     # policy config 基类
    │   ├── dataset/                # 数据转换工具
    │   ├── scripts/train_bc_dexmg.py
    │   └── utils/load_policy.py
    ├── rl_finetuning/
    │   ├── wrappers/residual_env_wrapper.py
    │   ├── scripts/train_residual_td3.py
    │   ├── scripts/train_rlpd_dexmg.py
    │   ├── off_policy/rl/           # Actor、Critic、QAgent
    │   ├── off_policy/networks/     # ViT 等视觉编码器
    │   ├── off_policy/common_utils/
    │   ├── config/                  # Hydra 配置
    │   ├── utils/                   # normalization、replay、checkpoint
    │   └── shell/paper_runs/        # 论文实验脚本
    └── dexmg/
        └── environments/            # 论文用向量化仿真环境
```

这个仓库不是一个新的机器人控制框架。它的核心是：

```text
固定 BC policy
    + Gymnasium vectorized environment wrapper
    + TorchRL replay buffer
    + TD3/RLPD-style critic and residual actor
```

## 5. Residual 与 LeRobot 的关系

### 5.1 不是完全独立的 LeRobot fork

`resfit/lerobot/setup_lerobot.sh` 会从 Hugging Face LeRobot 仓库 clone 固定 commit：

```text
69901b9b6a2300914ca3de0ea14b6fa6e0203bd4
```

然后执行：

```bash
python -m pip install -e deps/lerobot --no-deps
```

也就是说，真正提供 `lerobot.common.*` 的是外部安装到 `deps/lerobot` 的 LeRobot。Residual 仓库中的 `resfit/lerobot` 不是完整 LeRobot 源码，而是一个带 `resfit` 命名空间的补丁/适配层。

### 5.2 两套 import 命名空间

代码同时使用两类 import：

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.normalize import Normalize
```

这些来自外部安装的 LeRobot。

```python
from resfit.lerobot.policies.act.modeling_act import ACTPolicy
from resfit.lerobot.policies.factory import make_policy
```

这些来自 Residual 仓库自己的适配代码。

因此运行时必须同时满足：

1. `deps/lerobot` 已按指定 commit 安装；
2. Residual 仓库根目录在 Python import path 中；
3. `resfit.lerobot` 和 `lerobot.common` 的版本相互兼容。

不能只安装外部最新版 LeRobot 后直接运行 Residual。LeRobot 的公共路径、配置类、数据集字段和 policy base class 变化，都可能造成导入或 checkpoint 兼容问题。

### 5.3 Residual 对 ACT 做了什么

Residual 的 ACT 实现主要做了三类事情：

- 复用 LeRobot 的 Normalize/Unnormalize、Hub 相关能力和数据结构；
- 保留 ACT 的 CVAE/Transformer/action chunk 模型；
- 增加适合逐步真机执行的 `select_action()`，内部管理 action queue 或 temporal ensemble。

`ACTPolicy.select_action()` 的语义是“给当前 observation 返回一个待执行的单步动作”，而不是返回完整 chunk 给 RL critic。因此 ResFiT 当前实际上是：

```text
ACT chunk policy -> queue/ensemble -> current base action
current base action + single-step residual -> environment
```

这正是它能绕开 chunk-level PPO 的原因，也是它最适合作为第一阶段实现的原因。

## 6. 关键调用链

### 6.1 BC 训练

```text
resfit.lerobot.scripts.train_bc_dexmg
    -> LeRobotDataset / LeRobotDatasetMetadata
    -> resfit.lerobot.policies.factory.make_policy
    -> resfit.lerobot.policies.act.ACTPolicy
    -> save_pretrained / W&B artifact
```

Residual 的 BC 训练脚本仍然使用 LeRobot 的数据集、特征和 normalization 体系，但 policy factory 和 ACT 实现来自 `resfit.lerobot`。

### 6.2 Residual RL 训练

```text
train_residual_td3.py
    -> W&B 下载冻结 ACT checkpoint
    -> LeRobotDataset 读取离线数据
    -> ActionScaler / StateStandardizer
    -> create_vectorized_env
    -> BasePolicyVecEnvWrapper
    -> QAgent(residual_actor=True)
    -> TorchRL ReplayBuffer
    -> critic update / actor update
```

### 6.3 环境 wrapper 的动作语义

`BasePolicyVecEnvWrapper` 做了以下工作：

1. 接收原始 observation。
2. 调用冻结的 `base_policy.select_action(raw_obs)`。
3. 使用 `ActionScaler` 将 base action 归一化到大致 `[-1, 1]`。
4. 将 base action 放入 `observation.base_action`。
5. 接收 residual actor 输出的 normalized residual。
6. 计算：

```python
combined_naction = base_naction + residual_naction
env_action = action_scaler.unscale(combined_naction)
```

7. 将实际执行的归一化完整动作写入：

```python
info["scaled_action"] = combined_naction
```

这意味着 replay 中存储的是完整执行动作，而不是单独的 residual。critic 评价的也是完整动作的 Q 值；residual actor 只负责提出修正量。

## 7. QAgent 和 residual actor 的实现

`QAgent(residual_actor=True)` 的结构是：

```text
image(s) -> ViT encoder -> feature
state + base_action -> actor input
state + base_action + full action -> critic input
```

Actor 中 residual 模式会把 `observation.base_action` 拼到 policy input：

```python
all_input = [visual_feature, observation.state, observation.base_action]
residual = actor(all_input)
```

Actor 输出经过 `tanh` 和 `action_scale`，得到受限 residual。

Critic target 中会把下一时刻的 base action 与下一时刻 residual 相加：

```python
next_action = clamp(next_base_action + next_residual_action, -1, 1)
```

Actor loss 则最大化完整动作的 Q 值：

```python
combined_action = clamp(base_action + residual, -1, 1)
actor_loss = -Q(obs, combined_action)
```

代码还包含：

- 多头 critic；
- randomized minimum ensemble；
- target network；
- target action smoothing；
- LayerNorm/ViT encoder；
- n-step replay transform；
- prioritized replay；
- critic warmup；
- offline/online buffer 混合；
- replay buffer 的本地/Hugging Face 缓存。

这些组件说明它不是简单的“给 ACT 后面加一个 MLP”，而是一套完整的 off-policy residual 训练 recipe。

## 8. 离线数据是怎样进入 residual 训练的

`train_residual_td3.py` 中的 offline buffer 有两种语义：

### 模式 A：GT-as-base

```text
base_action = ground-truth action
target_action = ground-truth action
residual target = 0
```

这种模式主要用于让 residual policy 学会从零开始输出接近零的安全动作，但它和在线阶段的 base policy 分布不完全一致。

### 模式 B：base-policy-as-base

```text
base_action = ACT(raw_obs)
target_action = ground-truth action
residual target = ground-truth - ACT action
```

这是更适合你的模式，因为在线训练时环境实际使用的也是 ACT base action 加 residual。配置中对应：

```python
offline_data.use_base_policy_for_base_actions = True
```

使用这个模式时必须确保：

- 离线数据和 ACT observation preprocessing 一致；
- ACT 的 normalization stats 与数据集一致；
- ACT 推理时的 action queue/temporal ensemble 状态正确 reset；
- offline residual 与 online residual 使用同一动作坐标系。

## 9. 与当前项目接入时需要重点修改的地方

### 9.1 ACT 类型检查过于写死

训练入口中包含：

```python
if isinstance(base_cfg, ACTConfig):
    cfg.actor_name = "residual_act"
else:
    raise ValueError(...)
```

这说明当前 residual RL 入口实际上只支持 ACT。即使仓库中有 Diffusion policy 适配文件，`load_policy()` 仍然对 diffusion 分支抛出 `NotImplementedError`。要支持 DP，需要新增统一的 `BasePolicyAdapter`，而不是只改一个类型判断。

### 9.2 wrapper 的 observation_space 没有显式加入 base action

`_augment_obs()` 会添加：

```python
augmented_obs["observation.base_action"] = base_naction
```

但 `_setup_observation_space()` 中没有把 `observation.base_action` 写入新的 `gym.spaces.Dict`，而是保留原 observation space。当前 QAgent 直接按 key 读取，所以仿真可能仍能运行，但这会使 observation space 声明与实际 observation 不一致，影响环境检查、自动推断和未来接入 LeRobot/RLinf。

建议显式添加：

```python
obs_spaces["observation.base_action"] = gym.spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(self.action_dim,),
    dtype=np.float32,
)
```

### 9.3 终止与截断 reset 语义需要统一

wrapper 当前主要根据 `terminated` reset base policy 的 per-environment queue，但训练循环用的是：

```python
done = terminated | truncated
```

如果 episode 因 timeout/truncated 结束而没有清空 ACT queue，下一 episode 可能继续执行上一个 episode 的剩余动作。建议在 `terminated | truncated` 对应的环境 id 上统一 reset，并检查 `final_obs` 的 base action 填充逻辑。

### 9.4 动作坐标系必须先审计

Residual 代码假设：

```text
base action -> dataset action scaler -> normalized action
residual    -> normalized action space
full action -> clamp -> unscale -> environment
```

如果你的 ACT 输出是关节位置、关节增量，而机器人环境使用末端 twist，不能直接做数值相加。需要明确：

- action 的物理含义；
- action 的单位和范围；
- ACT checkpoint 使用的 normalization stats；
- 环境控制周期；
- policy latency；
- residual 是加在归一化空间还是物理空间。

建议第一版只允许一种明确的动作空间，并在 wrapper 中加入离线 action replay/假执行检查。

### 9.5 当前代码是仿真优先，不是即插即用真机代码

主入口依赖：

- `create_vectorized_env`；
- DexMimicGen 环境；
- Gymnasium vectorized API；
- MuJoCo EGL；
- 论文数据集和 W&B artifact。

它没有直接提供你的 USB 机器人 reset、力控、人工接管、安全 stop、实时通信和相机同步。真机接入时应保留它的算法和 replay 结构，但替换环境层和动作安全层。

## 10. 推荐的工程改造边界

不要直接修改 LeRobot 的核心 dataset/robot 代码来实现 residual。建议增加一层独立 adapter：

```text
LeRobotDataset / ACTPolicy
          ↓
BasePolicyAdapter
          ↓
ResidualEnvWrapper
          ↓
SafetyProjector
          ↓
RobotEnv
          ↓
ReplayBuffer / QAgent
```

建议 adapter 至少统一以下接口：

```python
class BasePolicyAdapter:
    def reset(self, env_ids=None): ...
    def predict_action(self, obs): ...
    def predict_chunk(self, obs): ...       # QC/PAC-ACT 使用
    def sample_chunks(self, obs, n): ...    # candidate proposal 使用
    def action_space_spec(self): ...
    def preprocessing_spec(self): ...
```

这样可以让：

- ACT 使用 queue/temporal ensemble；
- Diffusion 使用 denoising sampler；
- VLA 使用 token/flow action interface；
- residual learner 不需要知道 base policy 内部架构。

## 11. 建议的最小落地任务

第一步不要立即做完整 USB RL。建议按下面顺序验证：

1. 用一个固定 ACT checkpoint 做纯推理 rollout。
2. 记录 ACT 原始 action、归一化 action、执行 action 和 latency。
3. 接入 `BasePolicyVecEnvWrapper`，验证 residual=0 时行为与 ACT 完全一致。
4. 用离线数据检查 `base-policy-as-base` 生成的 residual target 分布。
5. 检查 `observation.base_action`、`scaled_action`、terminal/final observation 的形状。
6. 先只训练 critic，确认 Q 值能够区分成功和失败轨迹。
7. 开启小幅 residual actor，并设置 hard safety boundary。
8. 与 ACT-only、普通 BC/DAgger 和无 base policy 的 RL baseline 对比。

## 12. 最终判断

父目录的 Residual 仓库和 LeRobot 的关系可以概括为：

> LeRobot 负责数据格式、policy 基类、normalization 和 ACT 的基础生态；Residual 仓库负责冻结 base policy、逐步 residual 动作、off-policy critic/actor、replay 和训练脚本。

当前最值得复用的不是它的 DexMimicGen 环境，而是以下四个部分：

1. `BasePolicyVecEnvWrapper` 的 base action 注入和动作合成逻辑；
2. `QAgent(residual_actor=True)` 的 residual actor/critic 结构；
3. offline/online replay 混合与 critic warmup；
4. action/state normalization、n-step target、Q ensemble 和安全限幅思路。

当前不应直接照搬的部分是：

1. 固定的 LeRobot commit 和外部依赖安装方式；
2. DexMimicGen 的环境接口；
3. 默认的仿真 reward/终止语义；
4. 仅支持 ACT 的 checkpoint loader；
5. 未显式声明 `observation.base_action` 的 observation space；
6. 没有完全覆盖 truncated episode 的 queue reset 逻辑。

因此建议把 `residual-offpolicy-rl` 当作算法参考和初始训练骨架，把 LeRobot 当作 base policy/dataset 层，再围绕你的真实机器人重新实现环境、动作适配和 safety wrapper。

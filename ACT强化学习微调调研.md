# ACT 策略的强化学习微调：面向 USB 精密插入与长时程操作的调研

> 调研日期：2026-08-10～2026-08-11（2026-08-11 补充统一原理与文献核对）  
> 目标：在已有 ACT（Action Chunking with Transformers）行为克隆策略和 HIL-SERL 实现的基础上，选择一条可在真机上逐步验证、可控风险的强化学习微调路线。

## 1. 结论先行

ACT 可以做强化学习微调，但不建议第一步就用 SAC/PPO 直接更新整个 ACT Transformer。ACT 原本是一个 CVAE 行为克隆策略，输出高维动作序列；它没有现成的、适合普通 actor-critic 的逐步高斯策略接口。直接端到端 RL 会同时遇到高维 chunk 动作、探索破坏已有技能、critic 外推误差和真机安全四个问题。

对当前项目，建议按下面顺序推进：

1. **USB 插入的首选 MVP：冻结 ACT，训练受限的逐步残差策略。** ACT 负责接近、对准等已有动作先验，RL 只输出小幅末端位姿/速度修正；复用 HIL-SERL 的 RLPD/SAC、奖励分类器、示教回放、人类接管和异步采数框架。这条路线改动小、安全边界清楚，最适合验证“RL 是否能补足 ACT 最后几毫米的接触误差”。
2. **第二阶段：ACT 提议 chunk，Q 函数选 chunk。** 借鉴 Q-chunking，从 ACT 的 CVAE latent 或带噪输出采样多个候选动作块，critic 选择价值最高且不偏离示教分布的候选；先冻结 ACT，只训练 critic，稳定后再用 Q-filter/优势加权回归轻量更新 ACT。该路线适合研究离线数据复用和长时程稀疏奖励。
3. **直接微调 ACT 的研究基线：复现 PAC-ACT。** 它明确使用 chunk-level PPO 和 ACT-transferred actor-critic，并加入 behavior-prior 约束，是目前与问题最直接的工作；但尚未找到官方代码，且 on-policy 真机成本更高，建议先仿真复现。
4. **长时程任务：分层，而不是盲目拉长 ACT chunk。** 高层按阶段选择技能/子目标，低层使用短 chunk ACT；RL 主要优化技能选择、重试和恢复，接触阶段再调用残差策略。单一超长 open-loop chunk 会降低遇到扰动时的反应能力。

一句话判断：**精密接触优先残差 RL，长时程稀疏奖励优先 Q-chunking/分层 RL；端到端更新 ACT 放到最后。**

## 2. 为什么 ACT 不能直接照搬普通 RL

### 2.1 ACT 的可利用优势

ACT 用 CVAE + Transformer 一次预测未来一段动作，并通过 temporal ensembling 平滑多个重叠预测。其价值在于：

- 示教给出的动作具有时间一致性，探索不会像逐步独立高斯噪声那样抖动；
- chunk 缩短了有效决策链，适合从人类示教中保留非马尔可夫行为；
- 已有 ACT checkpoint 是很强的行为先验，RL 不必从随机动作重新发现抓取和对准。

原始工作与实现：[ACT/ALOHA 论文](https://arxiv.org/abs/2304.13705)、[官方 ACT 代码](https://github.com/tonyzhaozh/act)、[LeRobot ACT 实现](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/act)。

### 2.2 四个关键困难

1. **动作维度暴涨。** 若单步动作维数为 `d`、chunk 长度为 `K`，critic 需要评价 `K×d` 维动作。普通 SAC 的 Q 学习和随机探索都更难。
2. **open-loop 偏差。** chunk 内后续动作生成时还没看到未来观测；USB 接触、滑脱或卡住后若继续执行旧 chunk，错误会被放大。
3. **策略概率不自然。** ACT 的训练目标是重建损失加 latent KL，不是直接最大化环境回报；把确定性解码结果硬解释成 PPO/SAC 的动作分布并不严谨。
4. **遗忘与离分布动作。** 全参数 RL 更新可能很快破坏 BC 学到的抓取、接近和避碰技能。对真机而言，回报改善之前就可能产生危险动作。

因此，设计时应显式回答三个问题：RL 的动作单位是单步还是 chunk？ACT 是冻结的行为先验还是可更新 actor？执行多少步后重新观测和规划？

## 3. 重要参考工作

下表把“直接相关”和“可迁移的方法”分开看。开源状态按调研日可找到的官方公开实现判断；“未找到”不等于作者未来不会发布。

| 工作 | 核心思想/方法 | 适用场景 | 与 ACT 微调的关系 | 开源状态与链接 |
|---|---|---|---|---|
| **ACT / ALOHA** (2023) | CVAE Transformer 预测动作块，temporal ensembling 融合重叠预测 | 高质量示教、精细双臂/单臂操作 | 基础策略；提供行为先验和候选 chunk | 是：[代码](https://github.com/tonyzhaozh/act)、[论文](https://arxiv.org/abs/2304.13705) |
| **RLPD** (2023) | 在线 off-policy RL 持续等比例混合离线示教与在线回放；Q ensemble、LayerNorm、高 UTD 提升稳定性和样本效率 | 有少量示教、需要真机在线提升 | 可作为残差 actor/critic 或 chunk critic 的训练底座 | 是，MIT：[代码](https://github.com/ikostrikov/rlpd)、[论文](https://arxiv.org/abs/2302.02948) |
| **SERL** (2024) | 面向真机的高样本效率 SAC/DrQ 系统，actor-learner 异步采集，带奖励和 reset 工程 | 图像输入、真实机械臂、接触操作 | 可复用训练基础设施；旧仓库已提示转向 HIL-SERL | 是，Apache-2.0：[代码](https://github.com/rail-berkeley/serl)、[论文](https://arxiv.org/abs/2401.16013) |
| **HIL-SERL** (2024) | RLPD 风格 off-policy RL + 示教 + 人类在线纠正 + 图像奖励分类器 | USB/RAM/SSD 插入、线缆、装配等真机精细任务 | 当前最直接的工程底座；论文包含 USB grasp-insertion | 是，Apache-2.0：[代码](https://github.com/rail-berkeley/hil-serl)、[项目页](https://hil-serl.github.io/)、[论文](https://arxiv.org/abs/2410.21845) |
| **Residual RL from Demonstrations** (2021) | 冻结或保留 BC 基策略，RL 学习叠加的小残差；从图像、proprioception 与稀疏奖励学习 | 基策略已能大致完成、需要纠正接触/动力学误差 | 最适合把 ACT 当安全行为先验，避免初期破坏动作 | 论文公开；未找到作者正式代码：[论文](https://arxiv.org/abs/2106.08050) |
| **Policy Decorator** (ICLR 2025) | 冻结任意大型 BC 策略，SAC 学逐步 residual；用渐进探索和有界残差避免训练初期离开基础策略分布 | 基策略已有非零成功率、稀疏奖励、精细插入及多类操作 | 可直接把 ACT 当黑盒；论文实测 BeT/Diffusion Policy，迁移到 ACT 仍需验证 | 是，MIT：[代码](https://github.com/tongzhoumu/policy_decorator)、[项目页](https://policydecorator.github.io/)、[论文](https://arxiv.org/abs/2412.13630) |
| **ResFiT** (2025/ICLR 2026 Workshop) | 将 action-chunk BC policy 作为黑盒冻结，使用 sample-efficient off-policy RL 学习每一步轻量残差；只需稀疏二值奖励 | 高自由度、真机在线学习、需要保留 BC 动作结构 | 官方实现可用 ACT 训练 base policy，并以 residual TD3/RLPD 风格训练修正器，是当前方案 A 最直接的代码参考 | 是：[代码](https://github.com/amazon-far/residual-offpolicy-rl)、[项目页](https://residual-offpolicy-rl.github.io/)、[论文](https://arxiv.org/abs/2509.19301) |
| **DPPO** (2024) | 将预训练 diffusion policy 作为随机策略，用 PPO 对去噪链做策略梯度微调；强调 on-manifold 探索与 KL/训练技巧 | 仿真并行充分、生成式动作策略、长时程多阶段任务 | 不是 ACT 代码，但证明“先 BC、再策略梯度微调生成式 chunk policy”可行；训练技巧可迁移 | 是：[代码](https://github.com/irom-princeton/dppo)、[论文](https://arxiv.org/abs/2409.00588) |
| **Diffusion Policy** (2023) | 条件扩散模型生成连续 action sequence，擅长表示多模态动作分布并采用 receding-horizon execution | 复杂操作、多种合理轨迹 | DPPO/RL-100 的 BC 基座；最终动作 likelihood 不适合直接当普通 actor likelihood，需要利用去噪链构造 RL 接口 | 是：[论文](https://arxiv.org/abs/2303.04137) |
| **OpenVLA + VLA-RL** (2024/2025) | OpenVLA 将连续机器人动作离散成自回归 action tokens；VLA-RL 直接累加 token log-probability 并使用 PPO/GAE | 大规模 VLA、语言条件操作 | 说明“VLA 可以 PPO”只对具有显式 token 概率的 autoregressive action head 很自然，不代表所有 VLA 都能直接 PPO | [OpenVLA](https://arxiv.org/abs/2406.09246)、[VLA-RL](https://arxiv.org/abs/2505.18719) |
| **π0 / π0.5** (2024/2025) | VLM backbone + flow-matching action expert；π0.5 还加入高层离散语义/子任务预测，但低层机器人动作仍由连续 flow policy 生成 | 通用 VLA、跨任务操作 | 低层 flow action 没有像 token softmax 那样直接可得的 exact log π，因此普通 PPO 并不天然 | [π0](https://arxiv.org/abs/2410.24164)、[π0.5](https://arxiv.org/abs/2504.16054) |
| **π0.6* / Recap** (2025) | 用 value/advantage 把经验标记为更好/更差，并把 advantage indicator 作为 policy condition；继续用 flow-matching supervised objective 训练，推理时条件在“positive advantage” | 大规模机器人离线+在线数据、纠正和自主 rollout | 绕开 flow policy exact likelihood 难算的问题：不强行做 vanilla PPO，而做 advantage-conditioned policy improvement | [论文](https://arxiv.org/abs/2511.14759) |
| **FPO: Flow Matching Policy Gradients** (2025) | 用 conditional flow-matching loss 构造可用于 PPO-style clipping 的 advantage-weighted ratio surrogate | 想直接对 flow policy 做 policy gradient | 另一种“把不可直接算 likelihood 变成可优化 surrogate”的路线 | [论文](https://arxiv.org/abs/2507.21053) |
| **Q-chunking / QC** (NeurIPS 2025) | 在 chunk 动作空间做 TD 学习；先学 expressive behavior policy，再对其采样的 N 个 chunk 用 Q 做 Best-of-N 选择，并采用 n-step backup | 有离线数据、长时程、稀疏奖励、需要在线提升 | 与 ACT 最直接：用 ACT 代替 flow behavior policy产生候选 chunk | 是：[代码](https://github.com/ColinQiyangLi/qc)、[项目页](https://colinqiyangli.github.io/qc/)、[论文](https://arxiv.org/abs/2507.07969) |
| **Decoupled Q-Chunking** (2025) | critic 看长 chunk 加快价值传播，policy 输出/执行短 chunk 保持反应性；用 partial-chunk distilled critic 解耦二者 | 长时程但需要频繁闭环修正的任务 | 对 USB 很有启发：长价值窗口、短执行窗口 | 是：[代码](https://github.com/ColinQiyangLi/dqc)、[项目页](https://colinqiyangli.github.io/dqc/)、[论文](https://arxiv.org/abs/2512.10926) |
| **Adaptive Q-Chunking / AQC** (2026) | 同时考虑多个 chunk 长度；用相对各自 horizon baseline 且经 discount 归一化的 advantage 比较，避免直接比较不同尺度 Q 时总偏向短 chunk | 自由运动与精细接触交替、离线到在线 RL | 很适合 USB：自由空间选长 chunk，接触附近自动切短 chunk | 论文公开；未找到官方代码：[论文](https://arxiv.org/abs/2605.05544) |
| **Adaptive Action CHunking / ACH** (2026) | Transformer 单次前向同时估计多个候选长度的 Q 值，训练和推理时动态决定 chunk 长度 | 多任务、不同状态需要不同时间尺度 | 与 AQC 同期但不是同一工作；可作为动态 chunk selector 的另一实现参考 | 论文公开；未找到官方代码：[论文](https://arxiv.org/abs/2605.10044) |
| **AC3** (AAAI 2026) | DDPG 风格直接生成连续动作块；actor 只从成功轨迹更新，critic 使用 chunk 内 n-step return 与内在奖励 | 少量示教、长时程稀疏奖励、连续 chunk | 可作为比 QC 更简单的 chunk actor-critic 对照；成功轨迹过滤值得借鉴 | 论文公开；未找到官方代码：[论文](https://arxiv.org/abs/2508.11143) |
| **SEAR** (2026) | 纯在线 off-policy action-chunk RL；多时间尺度价值目标、causal critic、随机重规划、receding horizon | 无示教或少先验的在线 chunk RL、MetaWorld | 多时间尺度 critic 与短执行窗口可迁移；真机与 ACT 尚需额外验证 | 论文公开；未找到对应官方代码：[论文](https://arxiv.org/abs/2603.01891) |
| **PAC-ACT** (2026) | 把 PPO 的 actor、critic、回报累计和策略约束统一改到 chunk 层；复用预训练 ACT 的视觉 encoder、Transformer encoder 和动作 decoder，并以混合 behavior-prior 约束抑制遗忘 | 工业精密接触、希望直接优化预训练 ACT | 当前最直接的端到端 ACT RL post-training 工作；Contour 由 60% 到 100%，Square Assembly 由 51.2% 到 98.2%（论文设置） | 论文公开；截至调研日未找到官方代码：[论文](https://arxiv.org/abs/2607.09590) |
| **RL Token / RLT** (2026) | 冻结大策略的特征模型，抽取紧凑 RL token，训练小 actor-critic 并锚定基础动作 | VLA 的精密插入、需要几小时内在线适配 | “冻结大模型 + 小 RL 接口”的原则可直接迁移到 ACT encoder/latent | 论文与 RLinf 实现公开：[论文](https://arxiv.org/abs/2604.23073)、[RLinf 示例](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/rlt.html) |
| **RL-100** (2025/2026) | 以 diffusion visuomotor policy 为初始化，在 denoising process 内使用 clipped PPO-style objective 做真机强化学习；同时支持 single-action 与 action-chunk，并用 consistency distillation 压缩多步 diffusion 控制 | 多类真实操作、BC 到 RL 的持续提升 | 关键借鉴点是“把生成过程本身变成 RL policy interface”，而不是把最终动作黑盒直接套 PPO | 是：[代码](https://github.com/Lei-Kun/RL-100)、[论文](https://arxiv.org/abs/2510.14830) |

### 如何理解这些工作的关系

- **RLPD/SERL/HIL-SERL** 解决“真机 RL 怎样采得快、训得稳、有人能接管”的系统问题，但它们默认的 actor 通常是逐步动作 MLP，不直接解决 ACT chunk 的高维 critic。
- **Residual RL/Policy Decorator/ResFiT/RLT** 解决“大基础策略不要乱动，只学一个小而安全的修正接口”的问题。ResFiT 的公开实现已经把 ACT 用作 action-chunk BC base，是方案 A 最直接的复现起点。
- **Q-chunking/DQC/AQC/ACH/AC3/SEAR** 直接研究“RL 如何处理动作块”。AQC 和 ACH 特别补上了固定 chunk 长度的缺陷，但两者是不同论文，不能混写。
- **PAC-ACT** 是目前与题目最直接的工作：它不把普通 SAC 的单步 critic 接到 ACT 后面，而是将 PPO 与 critic 一并提升到 chunk 时间尺度，并用 behavior prior 保留 ACT 能力。
- **DPPO** 证明生成式 chunk policy 能被在线策略梯度提升，但其梯度定义依赖 diffusion 去噪过程，不能原样套到 CVAE ACT。

## 4. 四种可实现方案

### 方案 A：ACT + 受限逐步残差 RL（推荐先做）

定义：

```text
A_ACT = ACT(o_t)                    # K 步基础动作块
a_base = ensemble_or_first(A_ACT)   # 当前时刻基础动作
delta = pi_RL(z_t, a_base)          # 小型残差 actor
a_exec = safety_clip(a_base + M * delta)
```

其中 `z_t` 可从 ACT 视觉/Transformer encoder 中抽取并 stop-gradient，也可先直接用 HIL-SERL 的图像与 proprio encoder；`M` 是按阶段和动作维度设置的残差上限。

建议实现：

- 第一版完全冻结 ACT，残差网络只输出连续机械臂维度；夹爪维度沿用 ACT 或由状态机控制，避免混合离散/连续 RL 增加不稳定性。
- 若 ACT 输出关节位置目标，优先让残差表示为**小幅末端笛卡尔 twist**，经现有 controller/IK 转换；若当前系统只能叠加关节目标，则每关节单独限幅并做速度、加速度和 workspace clip。
- 只在进入插入阶段后启用残差。远距离接近、抓取和搬运阶段保持纯 ACT，降低训练分布和风险。
- critic 输入 `o_t, a_base, delta`，而不是只输入 `o_t, a_exec`，使它知道基础策略计划和修正量。
- 使用 RLPD 的离线/在线对半采样、Q ensemble、LayerNorm 和较高 UTD；初期 replay 包含 ACT 成功/失败 rollout、示教和人工纠正。
- actor loss 加锚定项：`L_actor = L_RL + λ_delta ||delta||² + λ_smooth ||delta_t-delta_{t-1}||²`。

优点：最低风险、工程改动最小、能利用现有 HIL-SERL。缺点：RL 主要修正局部误差，无法彻底改写 ACT 的高层行为；如果 ACT 从未到达正确插入前状态，残差能力有限。

### 方案 B：ACT 候选 chunk + Q-chunking（推荐核心研究路线）

把一个长度 `K_q` 的动作块当成宏动作：

```text
候选集 C = { ACT(o_t, z_i) } for i=1..N
A* = argmax_A Q(o_t, A) - beta * D(A, A_demo/manifold)
执行 A* 的前 h_exec 步
重新观测并规划
```

TD 目标需与真实执行长度对齐：

```text
y_t = sum(i=0..h_exec-1) gamma^i r_(t+i)
      + gamma^h_exec * Q_target(o_(t+h_exec), A_next)
```

关键设计：

- **候选如何产生：** 从 ACT CVAE prior 采样 latent；若当前 LeRobot ACT 推理固定 `z=0`，需开放 latent 输入。备选是对动作头加入小噪声、保留多个 temporal ensemble 提议，或训练轻量 stochastic adapter。
- **先只训 Q：** ACT 冻结时，Q 只在 ACT 产生的 on-manifold 候选之间选择，远比在整个 `K×d` 空间直接求最大值安全。
- **短执行、长评价：** USB 建议 `h_exec=1~3`，critic 可看 `K_q=5~10`；长时程自由空间动作可逐步增加 `h_exec`。这对应 DQC 的“policy/执行短、critic 窗口长”。
- **再微调 ACT：** critic 已可靠后，对 replay 中高优势 chunk 做 advantage-weighted regression，或仅更新 ACT action head/adapter。始终保留 BC replay 与 KL/L1 锚定，不先全量更新视觉 backbone。
- **训练/执行一致：** replay 必须记录“被选择的完整 chunk、实际执行前缀长度、每步 reward、终止位置和下一观测”，不能只存第一步动作，否则 chunk Q 的 Bellman target 不正确。

优点：最直接利用 ACT 的多模态、时间一致动作先验，适合稀疏回报和长时程。缺点：需要新增 chunk replay、chunk critic、候选采样和 n-step target，复杂度高于方案 A。

### 方案 C：PAC-ACT 式 chunk-level PPO（直接微调 ACT）

PAC-ACT 给出了“直接 RL 后训练 ACT”目前最完整的答案。它把一次 ACT 输出的整个动作块记为 `A_t`，而不是让 critic 只看第一个动作 `a_t`：

```text
o_t -> ACT actor -> A_t = [a_t, ..., a_(t+c-1)]
                   -> execute chunk
                   -> R_t = sum(i=0..c-1) gamma^i r_(t+i)
                   -> V/Q critic and chunk-level PPO update
```

它复用预训练 ACT 的视觉 encoder、Transformer encoder 和动作 decoder 构建 actor，同时构建与 chunk 时间尺度对齐的 critic。更新时除了 PPO 的新旧策略约束，还加入对 pretrained ACT 输出的 behavior-prior 约束，降低 RL 初期把已有动作结构破坏掉的风险。论文还报告：保留 CVAE 随机 latent 会增加策略随机源并与 behavior-prior 约束产生冲突，因此主设置移除 CVAE latent，让 ACT backbone/action decoder 输出 Gaussian chunk policy 的均值，随机性由 RL 阶段的 Gaussian action distribution 提供；这一改动不能在现有 ACT checkpoint 上未经消融直接照搬。

论文设置下，Contour 成功率从 60% 提升到 100%，Square Assembly 从 51.2% 提升到 98.2%；同时报告了接触力安全改善。应把这些数字视为该论文任务与训练条件下的结果，不能直接外推为 USB 真机性能。

对当前项目的复现步骤建议：

1. 先在仿真或数字孪生环境复现 chunk rollout、累计 reward、GAE/return 和 PPO ratio 的维度语义。
2. 用固定 ACT actor 只训练 critic，确认 chunk value 能对成功/失败排序。
3. 开启 action decoder 或小 adapter 更新，保留 frozen-ACT reference 与 behavior-prior penalty。
4. 最后才逐层解冻 Transformer；视觉 backbone 默认冻结。
5. 接触阶段仍设置短 chunk 或提前终止/重规划机制，不因采用 chunk PPO 就强制执行所有剩余动作。

优点：真正优化 ACT 本体，研究问题直接；论文已有精密接触 benchmark 证据。缺点：on-policy PPO 的真机样本复用率低于 off-policy residual，且截至调研日没有找到官方代码，所以工程复现与安全验证成本最高。

### 方案 D：分层 ACT + RL（长时程首选）

将任务拆成可判定阶段，例如：

```text
locate USB -> grasp -> transport -> pre-align -> contact search -> insert -> verify
```

高层策略每隔较长时间或在阶段终止时选择 `skill_id/subgoal`，低层 ACT 在短窗口内执行；插入 skill 内部再使用方案 A。高层奖励包含阶段完成、最终成功、超时和失败恢复成本。

建议先用可解释状态机/阶段分类器作为高层 baseline，再训练高层 Q 策略。这样可以分清性能提升来自低层精度还是高层恢复，而不是把两个难题塞进一个端到端策略。

适合：流程超过数十秒、存在可识别阶段、需要重试/回退/换策略。代价：需要阶段标签或可靠的子目标检测器，系统工程更多。

## 5. 针对 USB 精密插入的推荐实验方案

### 5.1 观测与动作

HIL-SERL 的 USB 设置是很直接的参照：论文使用 wrist/side 视觉、TCP pose/velocity、TCP force/torque 和 gripper position，动作是 6D twist + 夹爪控制，10 Hz 环境频率，初始 20 条示教，图像二分类奖励。对现有装置建议：

- 至少保留腕部近景 + 固定侧视相机；相机时间戳必须与关节/末端状态对齐。
- 接触阶段加入力/力矩，或至少加入电流/跟随误差代理信号。纯 RGB 很难区分“已插入”“顶住端口”“插歪但画面相似”。
- RL 残差动作从 `[dx, dy, dz, dRx, dRy, dRz]` 开始，夹爪锁定；z 方向、旋转和力阈值分别限幅。
- 控制频率先沿用已验证的 HIL-SERL 10 Hz，底层 controller 保持更高频安全控制。

### 5.2 奖励

不要只依赖单一视觉 success classifier。推荐组合：

```text
r = r_success
  + w_progress * delta(progress)
  - w_force * force_violation
  - w_action * ||delta||^2
  - w_jerk * ||delta_t - delta_(t-1)||^2
  - w_time
```

- `r_success`：端口插入成功的视觉分类器，最好再与深度、插入行程、连接握手或接触状态做逻辑 AND。
- `progress`：只使用不会诱导 reward hacking 的信号，如端口相对位姿估计、稳定插入深度；若不可靠，宁可先保留稀疏奖励。
- `force_violation`：超过软阈值立即负奖励并触发撤回；硬阈值由安全控制器终止，不能交给学习策略。
- 人工接管不是“坏数据”：接管前状态、人工动作和恢复轨迹都进入 replay，并标记 intervention mask。

### 5.3 数据与训练循环

1. 用现有 ACT 在扩大后的初始位姿分布上采集成功与失败 rollout，不先在线更新。
2. 将原始示教、ACT rollout、人工纠正统一到同一时间基准和动作语义；离线训练 critic/残差 BC 初始化。
3. 真机上线时使用 HIL 接管，先固定 ACT，只更新残差 actor-critic。
4. 每一轮固定 checkpoint 做独立评测，不用训练 episode 直接报告成功率。
5. 残差方案稳定后，再实现 ACT candidate sampler + chunk critic，与残差方案做同预算对比。

### 5.4 首轮建议超参数（作为起点，不是论文结论）

| 项目 | 建议起点 |
|---|---|
| 环境频率 | 10 Hz |
| ACT 原始 chunk | 保持现有 checkpoint 设置，不立即改 |
| 残差执行窗口 | 1 step，稳定后试 2~3 |
| chunk critic 窗口 | 5 或 10 steps |
| ACT 候选数 `N` | 8；算力允许再试 16 |
| 离线/在线 batch 比例 | 1:1（RLPD 起点） |
| 视觉 encoder | 第一阶段冻结或低学习率 |
| 残差幅度 | 从正常单步控制范围的 5%~10% 起，逐维设置 |
| actor 更新 | critic 预热后开启；actor 学习率低于 critic |
| 终止 | success、超时、workspace、碰撞/力阈值、人类接管 |

## 6. 长时程任务的推荐设计

### 6.1 为什么单纯增加 chunk 长度不够

更长 chunk 能让回报通过 n-step target 更快传播，也能形成连贯探索；但它同时增加动作维度和 open-loop 时间。任务越接触密集、扰动越不可预测，越需要短执行窗口。Q-chunking 的价值加速与 DQC 的短策略/长 critic 解耦，比“ACT 一次输出更长并全部执行”更合理。

### 6.2 推荐的层级

- **任务层（1~数秒）：** 选择子目标/技能、是否重试、是否回退；输入全局视觉、阶段历史和失败计数。
- **技能层（0.2~1 秒）：** ACT 输出 chunk，Q 选择候选；输入局部视觉和 proprioception。
- **接触修正层（单步）：** 残差 RL 或阻抗/力控；输入高频 proprioception 与力觉。
- **安全层（高频确定性）：** workspace、速度、加速度、力矩、碰撞、通信超时约束；独立于学习策略。

长时程的关键不只是最终 sparse reward，还要定义可靠的阶段完成判据和 reset/recovery。若 reset 仍完全依赖人工，训练吞吐会先成为瓶颈。

## 7. 实现落点（结合 ACT 与 HIL-SERL）

建议先建立薄适配层，不复制两套训练系统：

```text
ACT policy server
  -> base action/chunk + optional latent/features
  -> RL wrapper (residual or Best-of-N selector)
  -> safety projection
  -> existing HIL-SERL robot environment
  -> chunk-aware replay
  -> RLPD learner / chunk critic
```

最少需要新增的数据字段：

```text
observation_t
act_chunk_proposals[N, K, d]
selected_chunk[K, d]
base_action[d]
residual_action[d]
executed_prefix_len
per_step_rewards[h]
next_observation
terminated / truncated / intervention / safety_abort
policy_checkpoint_id
timestamps
```

接口注意事项：

- 明确 ACT 动作是绝对关节位置、关节增量还是归一化动作；HIL-SERL 常见 6D twist 不能直接数值相加。
- 将 normalization stats、相机顺序、裁剪、控制频率和 action latency 固化到 checkpoint 元数据。
- learner 更新不能阻塞 robot actor；actor 拉取模型时做原子 checkpoint 切换。
- temporal ensembling 与 chunk-prefix 执行只能选一个明确的数据语义。若融合了多个 chunk，replay 要记录最终实际执行动作，而不能假装执行了某个原始候选。

## 8. 对照实验与验收指标

### 8.1 必做 baseline

1. 原始 ACT，不在线学习。
2. ACT + 更多失败/纠正数据做普通 BC/DAgger 式再训练。
3. HIL-SERL/RLPD 小 MLP 策略，不使用 ACT。
4. ACT + 残差 RL（方案 A）。
5. ACT + Q-chunking（方案 B，第二阶段）。

只有比较 baseline 2，才能证明提升来自 RL 的回报优化，而不是简单增加在线标注数据。

### 8.2 指标

- 任务成功率及 Wilson 置信区间；每个 checkpoint 至少在固定随机化集合上评测。
- 首次成功所需真机 transitions、达到 80%/90% 成功率所需时间。
- 从接触到插入成功的耗时、总任务耗时。
- 人工接管次数/episode、接管持续时间、安全终止次数。
- 峰值/累计接触力、动作 jerk、workspace 越界次数。
- 对初始位置、角度、光照、USB 抓姿和外力扰动的鲁棒性。
- 长时程任务额外报告各阶段成功率、失败转移矩阵和恢复成功率，避免最终成功率掩盖瓶颈。

### 8.3 关键消融

- 冻结 ACT vs. 只训 action head/adapter vs. 全参数更新。
- 单步残差 vs. chunk residual。
- `h_exec = 1/2/4`，`K_q = 5/10/20`。
- 候选数 `N = 1/4/8/16`。
- 是否加入示教/BC anchor、Q ensemble、LayerNorm、HIL intervention。
- 纯视觉奖励 vs. 视觉 + 插入/力觉验证。

## 9. 风险、失败征兆与处理

| 风险 | 早期征兆 | 处理 |
|---|---|---|
| critic 过估计 | Q 持续升高但成功率不升，选出奇怪 chunk | Q ensemble、LayerNorm、behavior constraint、降低 UTD/actor 更新频率 |
| ACT 技能遗忘 | 抓取/接近先变差 | 冻结 backbone，恢复 BC 混合，限制 adapter，保留固定回归集 |
| open-loop 接触失稳 | 接触后仍连续向错误方向运动 | 缩短 `h_exec`，接触时强制逐步重规划，加入力阈值撤回 |
| reward hacking | 分类器高分但物理未插入 | 多信号 success、hard negative、独立验收传感器 |
| 动作语义错误 | 一启用残差就跳变/撞限位 | 在归一化前后明确叠加位置；做离线 replay 和假执行检查 |
| 时间不同步 | 动作与画面错配、训练不收敛 | 记录端到端时间戳和 policy latency，拒绝超阈值样本 |
| 人工数据偏置 | 策略只在被接管分布附近有效 | 扩大 reset 随机化，保留自主失败数据，分层统计 replay |

## 10. 建议里程碑

### M0：数据和接口审计（1 个短周期）

- 核对 ACT 与 HIL-SERL 的 observation/action 语义、归一化、频率、时间戳。
- 建立固定的 USB 评测初始位姿集合和安全阈值。
- 跑出 ACT-only 与 HIL-SERL-only 的可复现实验表。

### M1：残差 RL MVP

- 冻结 ACT；只在 pre-align/contact/insert 阶段启用 bounded residual。
- 复用 HIL-SERL replay、奖励分类器、HIL 和 RLPD learner。
- 验收：相同真机预算下，成功率或接触到成功耗时显著优于 ACT-only，且安全终止不增加。

### M2：Q-chunking 原型

- ACT 暴露随机 latent/候选 chunk；新增 chunk replay 和 Q critic。
- 先 Best-of-N 选择，不更新 ACT；验证 Q 排序与真实 chunk return 的相关性。
- 稳定后用 Q-filter/优势加权回归更新小 adapter。

### M3：长时程分层

- 建立阶段分类器与 recovery 状态机。
- 高层学习技能选择，低层复用 M1/M2。
- 验收按阶段失败率和恢复率，而不仅看最终成功率。

## 11. 最终建议

如果当前目标是尽快在已有 USB 平台上得到可信结果，选择：

> **冻结 ACT + HIL-SERL/RLPD 残差策略 + 接触阶段门控 + 多信号成功判定。**

如果目标是形成更有研究贡献的“RL 微调 action-chunk policy”方法，下一步选择：

> **ACT stochastic proposal + Q-chunking critic + 短执行/长价值窗口 + 后期 advantage-weighted adapter 更新。**

PAC-ACT 已证明 chunk-level PPO 微调 ACT 是值得复现的直接路线，但仍不建议把“无 behavior prior、无分阶段解冻的全参数 PPO”作为第一条真机路线。PAC-ACT 自身也通过 ACT-transferred actor-critic、chunk 时间尺度和混合 behavior-prior 约束处理这些问题；当前缺少官方代码，因此更适合作为仿真研究线与残差 RL 并行对照。

## 12. 2026-08-11 补充调研：ACT、Diffusion Policy、VLA/π 系列为什么能或不能直接做 RL，以及如何把“不能”改造成“可以”

### 12.1 统一判断标准：不是看部署时是否“确定”，而是看训练时有没有 RL 接口

今天讨论中最重要的修正是：**不能因为一个策略部署时输出确定动作，就判断它不能用 PPO；也不能因为一个模型是生成式模型，就认为它天然能用 PPO。**

对于标准 PPO，真正需要的是训练阶段存在一个可采样的随机策略，并能对已采样动作计算新旧策略概率比。最常见形式是：

```text
state/observation s
        ↓
 stochastic policy πθ(a|s)
        ↓ sample
       action a
        ↓
 environment → reward / advantage
        ↓
 log πnew(a|s) - log πold(a|s)
        ↓
 PPO clipped objective
```

核心量为：

```text
ratio = exp(log π_new(a|s) - log π_old(a|s))
```

因此要区分两个阶段：

```text
训练阶段：sample from distribution → 需要探索和概率/替代目标
部署阶段：mean / mode / argmax / deterministic sampler → 可以完全确定
```

例如普通 Gaussian actor 在训练时可以采样 `a ~ N(μθ(s), σ²)`，部署时直接执行 `μθ(s)`；部署确定并不妨碍训练使用 PPO。

由此得到一个统一分类：

| 策略族 | 原生动作形式 | 原生 exact `log π(a|s)` 是否方便 | vanilla PPO 是否自然 | 主要改造方式 |
|---|---|---:|---:|---|
| ACT | Transformer 回归整段连续 action chunk；CVAE latent 主要服务 BC | 否 | 否 | Gaussian chunk policy + chunk-level PPO；或 Q-Chunking / residual RL 绕开 likelihood |
| Diffusion Policy | 多步随机去噪生成 action sequence | 最终 action likelihood 不方便；但每个反向去噪 transition 通常有 tractable density | 最终动作黑盒 PPO 不自然 | DPPO/RL-100：把 denoising chain 重新解释成 inner MDP，在每个去噪 transition 上做 PPO |
| OpenVLA 类 autoregressive VLA | 离散 action tokens | 是，softmax 直接给 token probability | 是 | token log-probability 求和后直接 PPO/GAE |
| π0/π0.5/π0.6 flow VLA | flow matching 生成连续 action chunk | exact likelihood 不直接 | 否 | Recap advantage-conditioning、FPO surrogate、FlowDPG 等 flow-specific RL 接口 |

这张表是后续判断新模型能否接 RL 的通用工具：**先问 action distribution 在哪里，再问概率或 Q/advantage 接口在哪里，而不是只看模型名字。**

### 12.2 ACT：为什么原版不适合“直接套普通 PPO/SAC”

#### 12.2.1 ACT 的原始训练目标

ACT 的核心是 CVAE + Transformer action chunking。训练时 encoder 根据观测和专家动作得到 latent distribution，decoder/Transformer 一次预测未来一段动作：

```text
observation + expert action chunk
              ↓
          CVAE encoder
              ↓
            latent z
              ↓
observation + z
              ↓
     Transformer decoder
              ↓
A_t = [a_t, a_(t+1), ..., a_(t+K-1)]
```

原始目标是行为克隆/重建 + latent KL regularization，而不是环境回报最大化。实际部署常把 latent 固定到 prior 的中心附近，因此执行策略可以表现得近似确定。

这里的关键不是“z=0 所以不能 RL”，而是：**原始 decoder 没有被定义成一个标准 RL stochastic actor，且 ACT 的真正动作单位是高维 chunk。**

#### 12.2.2 为什么不能简单把 ACT 当成普通单步 actor

假设单步机器人动作维度 `d=7`，chunk 长度 `K=50`，那么 ACT 一次决策实际上对应：

```text
50 × 7 = 350 dimensions
```

普通 SAC/PPO 通常处理：

```text
Q(s_t, a_t) 或 π(a_t|s_t)
```

ACT 更自然的是：

```text
Q(s_t, A_t),  A_t=[a_t,...,a_(t+K-1)]
π(A_t|s_t)
```

直接扩展会产生四类困难：

1. **高维联合动作空间。** joint log-probability 是数十/数百维项的和，PPO ratio 可能迅速变得尖锐；Q 网络也更容易 extrapolation/overestimation。
2. **时间相关性。** ACT 的价值之一就是整个 chunk 内动作有连贯结构；如果简单对每一维独立加很大 Gaussian noise，可能把这种结构破坏掉。
3. **credit assignment 与时间尺度不一致。** 如果 actor 每 `K` 步决策一次，却仍按每个环境 step 使用普通 PPO return/critic，会让策略动作、reward、bootstrap 的时间语义错位。
4. **temporal ensembling。** 原 ACT 会融合多个时刻对同一未来动作的预测。如果实际执行动作来自多个 chunk 的加权组合，就必须明确 RL 的“被采样 action”到底是什么；否则 log-probability 和 replay 记录都不严格对应真实执行动作。

因此，“ACT 不能直接 PPO”应理解为：**原版 ACT 没有现成、严格且稳定的 PPO policy interface**，而不是数学上永远不能 PPO。

### 12.3 ACT 如何从“不直接适合 PPO”变成“可以 PPO”：PAC-ACT 路线

最直接的改造是把 ACT 输出解释为一个 Gaussian chunk policy 的均值：

```text
μθ(s) = ACT_backbone(s)
A ~ N(μθ(s), diag(σ²))
```

若 flatten 后的 chunk 有 `K×d` 维，则：

```text
log πθ(A|s) = Σ_i log N(A_i ; μ_i, σ_i²)
```

这样就获得 PPO 所需的 `log π`。但真正重要的不是“加一个 σ”本身，而是把整个 RL 时间尺度也改成 chunk-level：

```text
s_τ
 ↓
ACT/Gaussian actor
 ↓
A_τ = [a_0,...,a_(c-1)]
 ↓ execute c steps
r_0, r_1, ..., r_(c-1)
 ↓
R_τ = Σ_i γ^i r_i
 ↓
s_(τ+1)
```

chunk 间 discount 应与执行长度一致，例如固定 `c` 时使用：

```text
Γ = γ^c
```

然后在 `π(A_τ|s_τ)` 上计算 PPO ratio，并在 chunk 时间尺度计算 value/GAE。

PAC-ACT 的关键稳定化不是 Gaussian 这个简单技巧，而是组合了：

- ACT backbone/action decoder 产生整个 chunk 的均值；
- RL 阶段移除 CVAE latent，把随机性集中到 Gaussian action distribution；
- whole-chunk likelihood 与 chunk-level reward/GAE/PPO 对齐；
- frozen pretrained ACT 作为 behavior prior；
- old/new policy KL 约束，避免高维 chunk 一次更新漂移过大；
- 逐步解冻/保留预训练能力，而不是从一开始全量改写 BC policy。

因此 PAC-ACT 的合理定位是：**“ACT + Gaussian”不是新思想；把预训练 ACT、chunk-level semi-MDP/PPO、behavior prior 和稳定化约束组合成一个可执行 post-training recipe 才是主要贡献。**

#### PAC-ACT 可信度备注

截至 2026-08-11，PAC-ACT 是 2026-07 发布的 arXiv preprint。对它应分层判断：

- **数学可行性：高。** Gaussian policy、chunk return、PPO ratio、behavior prior 都是成熟构件，组合逻辑成立。
- **“这种 recipe 在作者 benchmark 中能工作”：有实验支持。**
- **“已经证明真机 ACT 全参数 PPO 普遍稳定”：证据不足。** 当前不应把论文任务中的成功率/接触力数字直接外推到本项目。
- **工程建议：** 把它作为仿真/数字孪生基线复现，而不是第一条真机探索路线。

### 12.4 ACT 另一条路线：不计算 `log π`，改成 Q-Chunking

如果不想把 ACT 强行变成 likelihood-based actor，可以直接采用 value-based chunk RL。

Q-Chunking 的核心不是“actor 一次输出更多动作”，而是 **critic 本身也把完整动作块当成输入，并在 chunk 时间尺度做 Bellman backup**：

```text
Q(s_t, A_t),  A_t=[a_t,...,a_(t+h-1)]
```

典型 target：

```text
y_t = Σ_(i=0..h-1) γ^i r_(t+i)
      + γ^h Q_target(s_(t+h), A_next)
```

与普通 off-policy n-step return 的区别是，critic 显式 condition on 实际整段 action sequence，因此可以让价值传播一次跨越多个环境 step，同时保留对具体 chunk 的评价。

原始 QC 使用 expressive behavior policy 产生多个 temporally coherent candidate chunks，再用 Q 做 Best-of-N：

```text
behavior policy
   ↓ sample N chunks
A1 A2 ... AN
 \ |      /
   Q(s,A)
     ↓
argmax Q
     ↓
execute
```

#### ACT + QC 是否可行

**可行，但“ACT + QC”是组合方案，不等于原始 QC 论文已经直接微调过 ACT。**

最稳妥的实现顺序：

1. 冻结 pretrained ACT；
2. 让 ACT 能生成多个不同但合理的 candidate chunks；
3. 只训练 chunk critic，在候选集合内做 Best-of-N；
4. critic 可靠后，再做 Q-filter / advantage-weighted regression / adapter update；
5. 最后才考虑更新大部分 ACT 参数。

难点是原版 ACT 推理常固定 latent，所以首先要建立 stochastic proposal：

- 开放 CVAE prior latent `z_i`，得到 `ACT(s,z_i)`；
- 或增加轻量 stochastic adapter；
- 或把 ACT action head Gaussian 化后采样多个 chunk；
- 或利用其他生成式 chunk proposal model。

这条路线的优势是**不要求 exact `log π`**，并且 off-policy replay 可以重复利用真机数据；对 sparse reward 和长时程 recovery 很有吸引力。缺点是要解决 Q overestimation、OOD candidate exploitation 和 chunk replay 的严格数据语义。

### 12.5 为什么 Diffusion Policy 看起来比 ACT 更容易做 RL 微调

#### 12.5.1 Diffusion Policy 原本也不能把最终 action 直接当成普通 PPO actor

Diffusion Policy 生成 action sequence 的过程大致为：

```text
x_K ~ noise
 ↓
denoise K
 ↓
denoise K-1
 ↓
...
 ↓
x_0 = action chunk
```

如果只把最终 `x_0` 当成黑盒输出，`π(x_0|o)` 的 exact likelihood 通常并不方便直接计算。因此早期也不能简单写成：

```text
final_action → log π(final_action|state) → PPO
```

所以 DP 能 PPO **不是因为 diffusion 最终动作天然有一个现成的简单 log-probability。**

#### 12.5.2 DPPO 的关键：把“去噪过程”改造成 inner MDP

DPPO 的核心转换是：把每一个反向去噪 transition 当作 stochastic sub-policy：

```text
x_k
 ↓  πθ(x_(k-1) | x_k, observation)
x_(k-1)
```

如果这个 transition 有 tractable Gaussian density，就能计算：

```text
log πθ(x_(k-1) | x_k, o)
```

然后把整个系统看成两层：

```text
外层 robot MDP：observation → action chunk → environment reward
内层 diffusion MDP：noise → denoise → ... → final action chunk
```

环境 rollout 得到的 advantage 可以用于更新内部去噪 sub-policies。这样原本“最终 action likelihood 不好算”的问题，被转换成“每一步 denoising transition likelihood 可算”。

这就是 DPPO 把“不容易直接 PPO”变成“可以 PPO”的本质。

#### 12.5.3 RL-100 为什么也可以

RL-100 延续这一类思想：不是把 diffusion model 当成完全确定的最终动作黑盒，而是在 denoising process 内应用 clipped PPO-style objective，并同时考虑 single-action 与 action-chunk 控制。其工程意义在于把一个 BC diffusion policy 变成可以通过真机 rollout 回报持续 post-train 的策略。

因此应记住：

```text
错误理解：DP 是随机模型 → 天然就能 PPO
正确理解：DP 最终 likelihood 不方便 → 把 denoising transition 重新定义成 policy → 才能稳定做 PPO-style RL
```

#### 12.5.4 为什么这类工作比 ACT + PPO 多

主要不是“ACT 不行”，而是研究生态和接口成熟度不同：

- Diffusion 本身已经有随机生成链，DPPO 之后形成了较标准的 denoising-RL recipe；
- diffusion 的随机探索容易保持在动作轨迹流形附近，而独立 Gaussian 扰动 ACT chunk 更容易破坏 temporal coherence；
- ACT 的 temporal ensembling、高维 chunk、CVAE latent 和执行前缀语义都让直接 likelihood RL 多一层工程定义；
- 因此很多工作选择 residual RL 或 Q-Chunking，而不是一开始全参数 ACT PPO。

### 12.6 OpenVLA 类 VLA：为什么可以比较自然地直接 PPO

这里必须区分 **autoregressive token VLA** 和 **flow/diffusion VLA**，不能笼统说“VLA 都可以 PPO”。

以 OpenVLA/VLA-RL 为例，连续机器人动作先离散化为 action tokens，Transformer 对每个 token 输出 categorical distribution：

```text
image + language + state
          ↓
       Transformer
          ↓
 logits → softmax
          ↓
P(token_i | previous tokens, observation)
```

因此一串动作 token 的 log-probability 可以自然写成：

```text
log π(a|s) = Σ_i log P(token_i | previous tokens, s)
```

这和 LLM 的 token policy gradient 很像，因此 PPO ratio 是现成的。VLA-RL 就利用这一性质做 PPO/GAE。

所以：

```text
OpenVLA 可以 PPO
≠
所有 VLA 都可以直接 PPO
```

判断依据仍然是 action head 是否给出了可计算概率。

### 12.7 π 系列：为什么不能简单照搬 OpenVLA PPO，又是怎样做 RL post-training 的

#### 12.7.1 π0 / π0.5 的动作接口和 OpenVLA 不同

π0 使用 VLM backbone + continuous action expert，并通过 flow matching 生成连续机器人动作。π0.5 进一步加入高层离散语义/子任务推理，但低层 motor action 仍主要由 continuous flow-matching action expert 产生。

因此它有两种完全不同的输出语义：

```text
高层语义：discrete/autoregressive tokens  → probability natural
低层动作：continuous flow matching       → exact log π not natural
```

如果 RL 优化的是低层连续动作，就不能因为模型里“也有 token”就直接套 OpenVLA 的 token PPO。

#### 12.7.2 Flow matching 为什么不适合 vanilla PPO

flow policy 从噪声通过连续/离散积分过程变成动作：

```text
noise z
 ↓
velocity field vθ(x,t,o)
 ↓ integrate ODE / flow steps
final continuous action
```

最终动作密度不像 categorical token 那样直接从 softmax 读出来。理论上某些 flow 可以通过 change-of-variables/ODE divergence 计算 density，但在大规模机器人 action model 上成本和实现都不够自然，因此 standard PPO 所需的 exact action likelihood 成为障碍。

#### 12.7.3 π0.6* / Recap：不强行求 exact likelihood，而改做 advantage-conditioned policy improvement

Physical Intelligence 的 Recap（RL with Experience and Corrections via Advantage-conditioned Policies）走的是另一条路线。

核心流程：

```text
robot experience / corrections / demos
              ↓
           value model
              ↓
     estimate advantage A(s,a)
              ↓
 positive / negative advantage indicator
              ↓
 train flow policy conditioned on indicator
              ↓
 test-time condition = positive advantage
```

可以把它理解为：

```text
原来：π(a | observation)
改成：π(a | observation, "Advantage: positive")
```

训练仍然可以使用原本擅长的 flow-matching supervised objective，而不必为每个连续 action 精确计算 PPO likelihood。value function/advantage 负责告诉模型“哪些经验比当前基线更好”，policy 通过条件建模把分布往这些更好的行为移动。

这就是 π flow policy 从“vanilla PPO 不自然”变成“能够用 RL 信号 post-train”的关键：**把 RL 信号转化为条件监督/advantage conditioning，而不是强制把 flow policy 变成 token/Gaussian actor。**

这种路线还有一个现实优势：它可以利用 demonstrations、过去 rollout、人工 corrections 等 off-policy 数据，不像标准 on-policy PPO 那样要求绝大多数梯度都来自最新策略采样。

### 12.8 Flow policy 的其他“把不能直接 PPO 变成可以优化”的方法

π/flow matching 不只有 Recap 一条路线。

#### FPO：Flow Matching Policy Gradients

FPO 不直接要求 exact final-action likelihood，而是从 conditional flow-matching objective 构造一个能够与 advantage/PPO clipping 结合的 surrogate ratio。其思想是：

```text
exact log π difficult
        ↓
use flow-matching training quantity
        ↓
construct policy-improvement surrogate
        ↓
PPO-style clipping / advantage weighting
```

这属于“修改 policy-gradient objective 以适配 flow policy”。

#### FlowDPG 等 critic-gradient 路线

另一类工作尝试直接利用 critic 对 action/flow trajectory 的梯度来优化 velocity field，并通过 distillation 等办法避免对整条多步 ODE 反向传播。它们说明 flow policy 并非不能 RL，而是需要专门考虑其生成过程和计算图。

### 12.9 三种生成式动作策略的 RL 接口对比

```text
ACT
│
├─ 原始：continuous chunk regression / CVAE
├─ 问题：没有天然 RL likelihood + chunk 高维
├─ 方案1：Gaussianize → chunk PPO (PAC-ACT)
├─ 方案2：Q(s,chunk) → Q-Chunking
└─ 方案3：freeze base → residual RL

Diffusion Policy
│
├─ 原始：multi-step denoising → continuous chunk
├─ 问题：final action likelihood 不方便
└─ 方案：denoising MDP → DPPO / RL-100

π flow VLA
│
├─ 原始：flow matching → continuous chunk
├─ 问题：exact final action likelihood 不自然
├─ 方案1：advantage conditioning → Recap / π0.6*
├─ 方案2：flow-loss surrogate → FPO
└─ 方案3：critic-gradient / distillation → FlowDPG 类方法

OpenVLA-style autoregressive VLA
│
├─ 原始：action tokens + softmax
└─ 天然可算 token log π → PPO / VLA-RL
```

### 12.10 Q-Chunking、PPO 和 residual RL：它们解决的是不同问题

不要把这三类方法只当成“不同 RL 算法”，它们实际上选择了不同的基础策略接口：

| 路线 | 基础策略是否更新 | 需要 `log π` | RL 数据利用 | 主要优势 | 主要风险 |
|---|---|---|---|---|---|
| ACT + residual SAC/RLPD | 基础 ACT 可冻结 | 否 | off-policy，可大量 replay | 真机安全、样本效率高、实现最稳 | 只能做局部 correction，高层 recovery 受 base policy 覆盖限制 |
| ACT + Gaussian chunk PPO | 更新 ACT | 是 | on-policy 为主 | 直接优化 ACT 本体、结构简单明确 | 高维 chunk ratio、样本消耗、遗忘和探索风险 |
| ACT + Q-Chunking | 可先冻结再轻量更新 | 否 | off-policy，可重复 replay | sparse reward/value propagation、候选动作保持 on-manifold | critic 过估计、candidate/OOD exploitation、工程复杂 |

对真机研究，推荐不是先押注一种，而是建立同一评测预算下的三路线 baseline。

### 12.11 为什么“ACT + Gaussian + PPO”这么简单，却直到最近才成为明确论文主题

这件事并不矛盾。

**简单的是局部技巧：**

```text
network output μ
+ Gaussian σ
→ stochastic actor
→ PPO
```

这是连续控制里很老的做法，本身几乎没有新颖性。

**难的是 action chunk 之后的系统问题：**

- 一次 action 从 7D 变成几十到几百维；
- joint likelihood/ratio 更敏感；
- Gaussian 独立探索可能破坏 chunk 内连贯性；
- chunk 决策频率和环境 reward 频率必须对齐；
- temporal aggregation 会让“实际执行的是哪个 sampled action”变复杂；
- 全参数更新会破坏 BC 已有技能；
- 真机 on-policy 探索昂贵且有安全风险。

因此 2024～2026 年相邻工作逐步选择了不同解决办法：

```text
2023: ACT / Diffusion Policy —— 先证明 action-chunk BC 很强
  ↓
2024: DPPO —— 给 diffusion 建立 denoising policy-gradient 接口
      ResiP —— 不直接动 chunk base，先做 residual RL
  ↓
2025: Q-Chunking —— 直接把 Q-learning 提升到 chunk action space
      VLA-RL —— 对 token VLA 直接使用 PPO
      FPO —— 为 flow matching 构造 policy-gradient surrogate
      RL-100 —— diffusion policy 真机 PPO-style post-training
      π0.6*/Recap —— flow VLA 用 advantage conditioning 做大规模 RL
  ↓
2026: AC3 / SEAR / AQC / ACH —— 继续研究 continuous action-chunk RL
      PAC-ACT —— 明确把 pretrained ACT 改造成 chunk-level PPO actor
```

所以更准确的结论是：**不是以前没人想到 Gaussian，而是直到 action-chunk BC 成为主流后，如何在不毁掉 temporal prior 的情况下做高维 chunk RL 才成为一个独立研究问题。**

### 12.12 对当前项目的统一研究路线

结合今天的讨论，后续不再把问题表述成“ACT 能不能强化学习”，而应表述成：

> **给定一个已经具有较强 BC 能力的 action-chunk policy，哪一种 RL 接口能以最低真机样本和风险提升精细插入、碰撞/滑移恢复、重新抓取和长时程 recovery，同时最少破坏原有技能？**

建议三条主线并行，但按风险排序：

```text
                         Pretrained ACT
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ↓                ↓                ↓
     Residual SAC/RLPD   Gaussian chunk PPO    Q-Chunking
              │                │                │
      low-dim correction   log π(A|s)        Q(s,A)
              │                │                │
        真机优先 MVP        仿真研究基线       核心研究路线
```

推荐实验顺序：

1. **Residual RL**：冻结 ACT，只让 RL 学接触/对齐阶段的小幅 correction，先证明在线 RL 能增加成功率和 recovery；
2. **Q-Chunking**：冻结 ACT proposal，先验证 chunk critic 能对成功/失败候选可靠排序，再做轻量 policy improvement；
3. **PAC-ACT style PPO**：在仿真/数字孪生中验证 Gaussian chunk likelihood、chunk GAE、KL/prior 和逐层解冻；
4. **统一同预算对比**：固定 demonstration、camera、action space、初始状态随机化和真机 transitions，比较成功率、恢复率、安全终止、样本效率和原 BC 技能遗忘。

特别重要的 baseline 是：**ACT + 新收集失败/恢复数据继续做 BC**。只有 RL 方法显著优于“相同新增数据量的普通 BC 再训练”，才能证明收益真正来自 reward-driven policy improvement，而不只是因为看到了更多 recovery 数据。

### 12.13 今天讨论形成的最终原则

1. **部署确定 ≠ 训练不能 PPO。** 训练时能构造 stochastic policy 即可，部署时仍可执行 mean/argmax。
2. **有随机性 ≠ 天然能 PPO。** Diffusion/flow 虽然随机，但最终 action likelihood 可能不方便；需要 denoising MDP、flow surrogate 或 advantage conditioning。
3. **“VLA 可以 PPO”不能一概而论。** OpenVLA token action 很自然；π0/π0.5 的低层 flow action 则不是同一概率接口。
4. **ACT 的关键障碍不是 Transformer，而是 action chunk 的概率、价值和时间尺度。** Gaussian 化只是第一步。
5. **Q-Chunking 是绕过 likelihood 的重要方案。** 它把问题转化为 `Q(s,chunk)`，更适合 off-policy 数据复用和 sparse reward，但需要严格控制 critic/OOD 风险。
6. **真机优先保留行为先验。** behavior prior、KL、residual bound、短执行窗口、receding horizon 和独立安全层都比“全参数直接更新”更重要。
7. **研究价值在接口设计和稳定性，而不在“给 ACT 加 Gaussian”这一行代码。**

## 13. 参考链接汇总

- [ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- [ACT 官方代码](https://github.com/tonyzhaozh/act)
- [RLPD: Efficient Online Reinforcement Learning with Offline Data](https://arxiv.org/abs/2302.02948)
- [RLPD 官方代码](https://github.com/ikostrikov/rlpd)
- [SERL 论文](https://arxiv.org/abs/2401.16013) / [代码](https://github.com/rail-berkeley/serl)
- [HIL-SERL 论文](https://arxiv.org/abs/2410.21845) / [代码](https://github.com/rail-berkeley/hil-serl)
- [Residual Reinforcement Learning from Demonstrations](https://arxiv.org/abs/2106.08050)
- [Policy Decorator 论文](https://arxiv.org/abs/2412.13630) / [代码](https://github.com/tongzhoumu/policy_decorator)
- [ResFiT 论文](https://arxiv.org/abs/2509.19301) / [代码](https://github.com/amazon-far/residual-offpolicy-rl)
- [DPPO 论文](https://arxiv.org/abs/2409.00588) / [代码](https://github.com/irom-princeton/dppo)
- [Q-chunking 论文](https://arxiv.org/abs/2507.07969) / [代码](https://github.com/ColinQiyangLi/qc)
- [Decoupled Q-Chunking 论文](https://arxiv.org/abs/2512.10926) / [代码](https://github.com/ColinQiyangLi/dqc)
- [Adaptive Q-Chunking / AQC](https://arxiv.org/abs/2605.05544)
- [Adaptive Action CHunking / ACH](https://arxiv.org/abs/2605.10044)
- [AC3: Actor-Critic for Continuous Action Chunks](https://arxiv.org/abs/2508.11143)
- [SEAR: Sample Efficient Action Chunking Reinforcement Learning](https://arxiv.org/abs/2603.01891)
- [PAC-ACT: Post-training Actor-Critic for Action Chunking Transformers](https://arxiv.org/abs/2607.09590)
- [RL Token / RLT 论文](https://arxiv.org/abs/2604.23073) / [RLinf 文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/rlt.html)
- [RL-100 论文](https://arxiv.org/abs/2510.14830) / [代码](https://github.com/Lei-Kun/RL-100)
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
- [VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning](https://arxiv.org/abs/2505.18719)
- [π0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)
- [π0.5: a Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054)
- [π0.6* / Recap: RL with Experience and Corrections via Advantage-conditioned Policies](https://arxiv.org/abs/2511.14759)
- [FPO: Flow Matching Policy Gradients](https://arxiv.org/abs/2507.21053)
- [RL for Flow-Matching Policies](https://arxiv.org/abs/2507.15073)
- [FlowDPG: Flow Matching Direct Policy Gradients](https://arxiv.org/abs/2606.22303)


# QChunk 与 HIL-SERL 融合：离线到在线训练实施计划

> 文档日期：2026-08-17  
> 项目阶段：MVP 设计  
> 目标：基于 QChunk 的 action chunk 思想和 HIL-SERL 的 Marvin 真机、人工干预、RLPD replay 及 actor/learner 通信能力，先跑通安全、可恢复、可验证的离线到在线训练闭环。

## 1. 总体结论

第一版定义为 **QC-RLPD-HIL MVP**：以 Marvin HIL-SERL 为主工程，把 QChunk 的固定长度宏动作、K 步 critic target 和行为约束移植到现有 HIL actor/learner 链路中。

主工程：

```text
/home/xlb/code_marvin/hil-serl
```

算法参考工程：

```text
/home/xlb/code/qc
```

建议开发分支：

```text
feature/qchunk-rlpd-hil
```

第一版只解决以下闭环：

```text
现有离线 demo
  → episode-safe action chunk
  → 离线 actor/critic 训练
  → checkpoint 部署到 actor
  → Marvin 真机逐步执行 chunk
  → 人工干预可立即打断
  → 在线 transition 上传 learner
  → offline/online 数据混合更新
  → 新参数回传 actor
  → checkpoint 与 buffer 可恢复
```

第一版不同时引入完整 QC-FQL、flow actor、Best-of-N、动态 K 和 10 Q ensemble，以减少调试变量。

---

## 2. 系统架构

### 2.1 保留 HIL-SERL 的部分

- Marvin 真机环境和安全限幅；
- 四路策略相机和 24 维 proprio state；
- ResNet10 图像编码器；
- actor/learner 分布式结构；
- 现有参数发布和 transition 传输机制；
- memory-efficient replay buffer；
- RLPD 的 online/demo 混合采样；
- SpaceMouse 人工干预；
- checkpoint、日志和断点恢复流程。

### 2.2 从 QChunk 引入的部分

- actor 一次输出长度为 K 的动作序列；
- critic 评估完整动作块 `Q(s, a[t:t+K])`；
- K 步累计折扣奖励；
- `gamma^L` bootstrap，其中 L 是实际执行长度；
- 离线示范行为约束；
- online replay 与 prior data 联合训练。

### 2.3 MVP 网络定义

Marvin 单步动作为：

```text
[dx, dy, dz, drz, gripper]
```

chunk actor 输出：

```text
[B, K, 5]
```

第一版将 5 维动作全部视为连续动作，actor 使用 Tanh Gaussian，critic 输入展平后的 `K×5` 动作。夹爪输出仍在 `[-1, 1]`，由当前 Marvin 环境解释为关闭、不动作或打开。

后续如夹爪学习不稳定，再升级为：

```text
K×4 continuous arm head
+ K×3 categorical gripper head
```

### 2.4 K 的推进顺序

```text
K=1：数据、网络和 Bellman target 对齐
  ↓
K=3：第一版离线到在线闭环和首次真机训练
  ↓
K=5：闭环稳定后的正式对比实验
```

---

## 3. 统一数据契约

磁盘保存和 actor/learner 传输继续使用 primitive transition，而不是只保存聚合后的 chunk：

```python
{
    "episode_id": int,
    "episode_step": int,

    "observations": ...,
    "next_observations": ...,

    "policy_action": np.ndarray,       # [5]，网络输出经安全裁剪
    "executed_action": np.ndarray,     # [5]，EMA/干预后实际执行
    "rewards": float,
    "dones": bool,
    "masks": float,

    "chunk_id": int,
    "chunk_offset": int,
    "proposed_chunk": np.ndarray,      # [K, 5]

    "intervened": bool,
    "source": str,
    "schema_version": int,
}
```

`source` 至少区分：

```text
offline_demo
online_policy
intervention
```

关键语义：

- `policy_action` 是策略真正提出并经过安全裁剪的动作；
- `executed_action` 是 EMA、SpaceMouse 覆盖和环境约束后真正进入机器人控制器的动作；
- `proposed_chunk` 是 chunk 开始时 actor 一次产生的完整序列；
- critic 的训练数据必须与实际执行动作一致；
- chunk 中途被干预后，未执行的剩余动作不得与后续奖励关联；
- `schema_version` 用于拒绝不兼容的旧消息，而不是静默误读。

---

## 4. Chunk Bellman target

对于实际执行长度为 L 的 chunk：

```text
R_L = r_t + gamma r_{t+1} + ... + gamma^(L-1) r_{t+L-1}
```

critic target：

```text
target = R_L
       + gamma^L × bootstrap_mask
       × Q_target(s_{t+L}, pi(s_{t+L}))
```

不能始终写死为 `gamma^K`，因为真机回合可能在 chunk 中途成功、失败、被安全系统截断或被人工干预。

actor loss：

```text
L_actor = L_SAC
        + lambda_offline × L_BC(offline_demo)
        + lambda_intvn × L_BC(intervention)
```

普通在线策略数据不施加 BC loss，避免把在线失败行为重新模仿进策略。

---

## 5. 分阶段实施计划

## P0：冻结基线与实验边界

### 任务

- 固定当前 Marvin 环境、奖励、动作限幅、EMA 和相机配置；
- 保存现有 RLPD checkpoint 和真机评估结果；
- 记录成功率、平均回合步数、干预次数和控制频率；
- 新增独立实验配置 `marvin_usb_qchunk`；
- 新旧输出目录、checkpoint 和 buffer 完全分开；
- 不覆盖现有 `train_rlpd.py`，保留随时回退能力。

### 验收门槛

- 原始 RLPD 入口和 checkpoint 仍可正常使用；
- 新实验不会写入原 RLPD 输出目录；
- 基线指标和代码版本可追溯。

## P1：实现 episode-safe chunk replay

### 任务

在现有 memory-efficient replay buffer 基础上新增：

```python
sample_chunk(
    batch_size,
    chunk_horizon,
    discount,
    source_filter=None,
)
```

输出至少包括：

```python
{
    "observations": ...,
    "actions": ...,             # [B, K, 5]
    "next_observations": ...,
    "rewards": ...,             # [B]，累计折扣奖励
    "duration": ...,            # [B]
    "bootstrap_mask": ...,      # [B]
    "valid_mask": ...,          # [B, K]
    "source": ...,
}
```

### 必须满足

- 不跨 episode；
- 不跨 reset；
- 不跨环形 buffer 的无效区域；
- 不跨 policy/intervention 边界；
- terminal 短 chunk 正确 padding；
- 干预发生后不关联未执行策略动作；
- 支持四相机 frame stack；
- 支持 recent/older 分层采样。

### 单元测试

- K=1 与单步采样一致；
- terminal 位于 offset 0、1、K-1 时结果正确；
- 环形 buffer 回绕后采样正确；
- episode 末尾不会拼接下一回合；
- 干预前后不会产生混合策略 chunk；
-累计奖励、duration 和 bootstrap mask 可手工复算。

## P2：实现 Chunk SAC agent

建议新增：

```text
serl_launcher/agents/continuous/qchunk_sac.py
```

### 网络

```text
Encoder：复用四相机 ResNet10 + 24维 state
Actor：输出 K×5 Tanh Gaussian
Critic：输入 observation embedding + flatten(K×5)
Target critic：soft update
Q ensemble：第一版 2 个 Q
```

### 实现顺序

1. K=1 网络 shape 测试；
2. K=1 单 batch update；
3. K=1 target 与单步 SAC 对齐；
4. checkpoint 保存和恢复；
5. 扩展到 K=3；
6. 最后支持 K=5。

### 验收门槛

- actor 输出 `[B, K, 5]`；
- critic 输入 `[B, K×5]`；
- 无 NaN/Inf；
- encoder 可独立冻结和解冻；
- target critic soft update 正常；
- 相同 checkpoint 恢复后 deterministic 输出一致。

## P3：离线训练

### Offline-A：Chunk BC warm-up

先冻结 critic 和视觉 backbone，仅训练 actor head：

```yaml
chunk_horizon: 3
batch_size: 256
actor_lr: 3e-4
bc_updates: 20_000
encoder_frozen: true
```

记录：

- 整体 action MSE/NLL；
- 每个 chunk offset 的动作误差；
- 每个动作维度的误差；
- gripper 阈值准确率；
- 动作饱和比例；
- train/validation gap。

### Offline-B：Critic warm-up

冻结 actor，训练 K 步 critic：

```yaml
critic_lr: 3e-4
critic_updates: 20_000
num_qs: 2
discount: 0.98
tau: 0.005
```

### Offline-C：联合训练

```yaml
joint_updates: 50_000
bc_alpha: 0.01
cta_ratio: 2
```

### Checkpoint

```text
offline_bc_best
offline_critic_best
offline_joint_best
offline_last
```

### 验收门槛

- validation BC 不持续恶化；
- predicted Q 和 target Q 数量级一致；
- Q 值不持续发散；
- deterministic 离线动作质量不低于 K=1 基线；
- mock 环境能够执行完整 K=3 chunk；
- checkpoint 可以直接由 actor 加载。

## P4：actor 端 chunk 执行和干预

actor 一次推理产生完整 chunk，但逐 primitive 调用 `env.step()`：

```text
生成 [a0, a1, a2]
  → 执行 a0 → 检查干预/终止/安全
  → 执行 a1 → 检查干预/终止/安全
  → 执行 a2 → 检查干预/终止/安全
```

以下任一情况出现时立即清空动作队列：

- `intervene_action`；
- 成功；
- 不可恢复失败；
- truncated；
- watchdog；
- 安全限位；
- 通信异常。

发生人工干预时：

1. 当前 primitive 使用 SpaceMouse 动作；
2. 立即清空剩余策略 chunk；
3. EMA 同步到实际干预动作；
4. 当前 transition 标记 `intervened=True`；
5. 写入 online replay；
6. 同时写入 intervention/demo buffer；
7. 下一步基于最新观测重新推理。

### 验收门槛

- 干预最迟在一个控制周期内接管；
- 未执行动作不会写成 executed action；
- 成功或失败后不会继续执行剩余队列；
- reset 后没有上回合残留动作；
- 日志能够逐步对比 policy action 和 executed action。

## P5：接入现有 actor/learner 通信

### 设计原则

- actor 上传 primitive transition；
- learner 在本地利用 chunk sampler 构造训练 batch；
- 不直接通过网络上传聚合 chunk；
- actor 仅在 chunk 边界切换新参数版本。

这样可以：

- 保留完整原始数据；
- 后续改变 K 时复用已有数据；
- 重新计算不同 discount；
- 检查干预和 terminal 边界；
- 避免在同一 chunk 中混用两个模型版本。

### 通信字段

```text
schema_version
episode_id
episode_step
chunk_id
chunk_offset
policy_action
executed_action
proposed_chunk
intervened
source
```

### 验收门槛

- actor/learner 可以分进程或分机启动；
- 初始离线 checkpoint 能正确发布给 actor；
- 参数版本单调递增；
- actor 只在 chunk 边界替换参数；
- 断线后执行安全停止或进入 collect-only；
- 重连后 transition 不重复、不静默丢失；
- 不兼容 schema 会被明确拒绝。

## P6：在线 RLPD

MVP 保留两个主要数据池：

```text
online replay
demo/intervention replay
```

`demo/intervention replay` 包含启动时加载的离线示范，以及在线新增的人工干预数据。

初始采样比例：

```text
50% online
50% offline demo + intervention
```

每个池内部继续使用：

```text
50% recent
50% older
```

初始参数基线：

```yaml
chunk_horizon: 3
batch_size: 256
training_starts: 500
cta_ratio: 2
discount: 0.98
tau: 0.005
actor_lr: 1e-4
critic_lr: 3e-4
bc_alpha: 0.01
critic_warmup_steps: 2_000
publish_every_updates: 20
random_steps: 0
```

真机必须从离线 checkpoint warm-start，禁止复用 QChunk 仿真示例中的随机动作预填充。

第一版可以保留现有 HIL-SERL 的干预 transition 同时进入 online 和 intervention buffer 的机制，但必须记录由此产生的重复采样权重。第二版再升级为显式三路采样：

```text
50% online policy
25% fixed offline demo
25% intervention
```

## P7：断点恢复

checkpoint 至少保存：

- encoder、actor、critic、target critic；
- temperature；
- optimizer state；
- learner step；
- actor model version；
- RNG state；
- replay/demo buffer manifest；
- episode/chunk ID 计数器；
-数据 schema version；
- chunk horizon 和动作维度。

### 验收门槛

- learner 重启后继续原 step；
- actor 重连后获取最新完整参数；
- replay 插入顺序保持正确；
- recent/older 采样的时间语义不被破坏；
- chunk ID 和 episode ID 不重复；
-不同 K 的 checkpoint 不会被错误加载。

---

## 6. 端到端验收 Gate

## Gate 1：纯数据测试

```text
旧 demo → 数据迁移 → chunk sampler → batch 可视化
```

通过条件：

- 100% 不跨 episode；
- reward、duration、mask 可手工复算；
- 图像、状态和动作时间对齐；
- 干预和 terminal 边界正确。

## Gate 2：K=1 Mock 闭环

```text
offline train
  → checkpoint
  → actor load
  → mock rollout
  → transition upload
  → learner update
  → parameter publish
```

通过条件：

- 离线训练和在线更新使用同一网络定义；
- checkpoint/resume 正常；
-参数回传后 actor 输出发生预期变化；
- 无 NaN、shape mismatch 或 schema mismatch。

## Gate 3：K=3 Mock + 模拟干预

在 chunk 的 offset 0、1、K-1 注入干预。

通过条件：

- 动作队列立即取消；
- EMA 同步正确；
- buffer source 正确；
- 不产生跨干预 chunk；
- online update 正常。

## Gate 4：K=3 真机 collect-only

第一轮只执行和采集，不更新参数：

```text
5 个短回合
限制最大回合步数
SpaceMouse 和急停始终可用
```

通过条件：

- 动作尺度正常；
- 没有明显 chunk 累计漂移；
- 人工干预能够立即接管；
- policy/executed action 日志一致；
-采集 replay 可以重新加载和采样。

## Gate 5：单次在线更新

```text
采集少量数据
  → 暂停真机
  → 执行 1 次 learner update
  → 保存 candidate checkpoint
  → 离线动作审计
  → 恢复真机评估
```

通过后再逐步扩大：

```text
1 update → 10 updates → 100 updates → 连续在线更新
```

---

## 7. 必须记录的指标

### 数据指标

- 各 source 的样本数量和比例；
- 无效 chunk 拒绝数量；
- terminal chunk 比例；
- chunk 平均实际长度；
- intervention chunk 比例；
- recent/older 实际采样比例。

### 策略指标

- actor SAC loss；
- offline BC loss；
- intervention BC loss；
- entropy 和 temperature；
- 每个 offset 的 action MSE/NLL；
- 每个动作维度的饱和率；
- gripper 动作分布。

### critic 指标

- critic loss；
- predicted Q；
- target Q；
- Q ensemble 标准差；
- TD error；
- offline/online/intervention 分来源 TD error。

### 真机指标

- success rate；
- 平均回合步数和时间；
- chunk 完整执行比例；
- chunk 中断比例；
- 人工干预次数和干预步数；
- 干预取消延迟；
- 推理延迟；
- actor 参数版本；
- policy action 与 executed action 差异。

---

## 8. 建议新增代码布局

```text
examples/train_qchunk_rlpd.py

examples/experiments/marvin_usb_qchunk/
    config.py

serl_launcher/agents/continuous/
    qchunk_sac.py

serl_launcher/data/
    chunk_replay_buffer.py
    chunk_sampler.py

serl_launcher/wrappers/
    action_chunk_executor.py

tests/
    test_chunk_sampler.py
    test_chunk_returns.py
    test_chunk_terminal.py
    test_chunk_intervention.py
    test_chunk_ring_buffer.py
    test_qchunk_agent_shapes.py
    test_qchunk_checkpoint.py
    test_qchunk_actor_learner.py
```

现有 `examples/train_rlpd.py` 保留为基线和回退入口。

---

## 9. MVP 暂不实现

- QC-FQL；
- flow matching actor；
- Best-of-N；
- 10 个 Q 网络；
- 动态 chunk horizon；
- receding-horizon chunk；
- categorical hybrid gripper；
- 自动调整 offline/online/intervention 比例；
- actor 侧上传聚合 chunk；
- 新的算法 Bridge 协议；
- 在真实机器人上使用随机策略预填充 buffer。

---

## 10. 实施优先级

最优先的三个开发任务：

1. 统一 primitive transition 数据契约并实现 episode-safe chunk sampler；
2. 实现 K=1 Chunk SAC agent、Bellman target 和单元测试；
3. 实现离线 BC/critic/joint 训练入口和可部署 checkpoint。

在上述三项通过之前，不启动正式真机在线参数更新。之后依次接入：

```text
actor/learner 通信
  → K=3 mock
  → 模拟干预
  → 真机 collect-only
  → 单次在线更新
  → 小规模连续在线训练
```

## 11. MVP 完成定义

当以下链路全部通过，第一版才算完成：

```text
现有 demo
  → episode-safe K=3 chunk
  → 离线 BC/critic/joint 训练
  → 保存 offline checkpoint
  → actor 加载并执行 K=3
  → SpaceMouse 可中断任意 chunk offset
  → primitive transition 经现有通信上传
  → learner 混合 offline/online 数据更新
  → 新参数在 chunk 边界回传 actor
  → checkpoint、optimizer 和 buffer 可完整恢复
```

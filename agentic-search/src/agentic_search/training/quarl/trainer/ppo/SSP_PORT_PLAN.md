# 把 SSP 的 self-play trainer 落到 verl 0.9.0：设计 + 适配清单

对象：`quarl/trainer/ppo/ssp_ray_trainer.py`（1877 行，从 `third_party/SSP` 直接复制）。
结论先说：**算法逻辑（80% 的行数）可以原样保留，但框架粘合层（约 150 行）必须重写**。
直接用现在这份跑，第一步 `import` 就失败。

---

## 0. 一句话诊断

上游 SSP 是 **verl 0.5.x 世代**（README 钉 `cb809d66`，配 `transformers 4.55`/`sglang 0.4.6.post5`）。
那个世代的 verl 是「DataProto 进、DataProto 出，driver 侧算奖励」；
本仓库是 **0.9.0**，「TensorDict 进、agent_loop 异步生成、奖励在 rollout 里流式算」。

两代之间被删掉的 API，恰好是 SSP trainer 的骨架。

实测（本地 venv，`PYTHONPATH=src:src/agentic_search/training/verl`）：

```
IMPORT FAIL: ImportError -> cannot import name 'compute_reward'
             from 'verl.trainer.ppo.reward'
```

---

## 1. SSP 算法骨架（这部分与框架无关，必须保住）

```
第 k 步，策略 π_θ（一个模型演两个角色）
┌────────────────────────────────────────────────────────────────────┐
│ ① Proposer 生成问题                                                  │
│    种子材料 → prompt("请基于材料出一道难题") → 多轮轨迹，可调用 search│
│    采样 n_p 条/prompt（官方 launcher: n_p = 1）                       │
├────────────────────────────────────────────────────────────────────┤
│ ② 问题抽取（不经模型，走 LLM-as-judge 服务）                          │
│    从轨迹里抠出 <question>…</question> + 答案，判有效性               │
│    抽不出来的用 fallback（复用历史题 / 造 dummy）                     │
├────────────────────────────────────────────────────────────────────┤
│ ③ Solver 求解                                                        │
│    抽出的题 → solver prompt → 多轮轨迹，可调用 search                 │
│    每条题采样 n_s 条（= rollout.n = 5）                               │
├────────────────────────────────────────────────────────────────────┤
│ ④ 奖励                                                                │
│    solver:  QA EM vs ground_truth          → 组内 GRPO 标准化        │
│    proposer: r = 1 - success_rate(solver)   ← 跨批次依赖，关键点      │
│              即"出的题越难（solver 通过率越低）分越高"                 │
├────────────────────────────────────────────────────────────────────┤
│ ⑤ 两次更新（同一份 π_θ，可 combine_update 合并成一次 backward）        │
└────────────────────────────────────────────────────────────────────┘
```

**关键结构性约束**：`proposer 的奖励依赖 solver 的结果`。
所以 proposer 奖励**不可能**在 proposer 自己的 rollout 里算出来 —— 这直接决定了下面第 4 节的奖励架构。

其它可选旋钮（都在 `self_play.*`）：`mini_epochs`（同一批 rollout 复用几次更新）、
`reward_dynamic_sampling`（按奖励方差筛样本、攒够再更新）、`dynamic_sampling`（抽取失败就多跑几批）、
`extract_failure.strategy`（抽取失败时 reuse / dummy）。

---

## 2. 原版依赖的 verl API vs 本仓库 0.9.0（逐条实测）

| # | SSP 里的写法（行号） | 0.9.0 现状 | 严重度 |
|---|---|---|---|
| 1 | `from verl.trainer.ppo.reward import compute_reward, compute_reward_async`（L37） | **已删除**，只剩 `extract_reward` / `load_reward_manager` | 崩·import |
| 2 | `super().__init__(..., reward_fn, val_reward_fn, ...)`（L86-99） | `__init__` 签名里**没有** `reward_fn`/`val_reward_fn`（见 §2.1） | 崩·TypeError |
| 3 | `def _validate_config()` 里 `super()._validate_config()`（L114-121） | 基类**没有**这个方法 | 崩·AttributeError |
| 4 | `self.actor_rollout_wg.generate_sequences(...)`（L1154/1274/218） | WG 上**没有** `generate_sequences`；生成只走 `async_rollout_manager` | 崩 |
| 5 | `self.actor_rollout_wg.compute_log_prob(gen_batch)`（L1476） | 签名是 `compute_log_prob(data: TensorDict) -> TensorDict`，且不处理 padding/entropy 拼装 | 崩 |
| 6 | `self.actor_rollout_wg.compute_ref_log_prob(...)`（L1490） | 同上（→ `_compute_ref_log_prob`） | 崩 |
| 7 | `self.actor_rollout_wg.update_actor(gen_batch)`（L1536） | 同上（→ `_update_actor`），且要自己塞 6 个 non-tensor 元数据 | 崩·静默错 |
| 8 | `self.val_reward_fn is not None`（L395/761）、`self.val_reward_fn(...)`（L264） | 0.9.0 无 `val_reward_fn`，验证奖励也来自 `rm_scores` | 崩 |
| 9 | `self.config.reward_model.launch_reward_fn_async`（L1304） | 该字段已废弃 | 崩 |
| 10 | `_validate` 覆盖（L154-337，180 行） | 基类 `_validate(merged=False)` 已重写为 agent_loop 版本 | 删掉即可 |
| 11 | `_balance_batch` 覆盖（L1863） | 基类已有，且多了 `keep_minibatch` 参数 | 建议删覆盖 |
| 12 | 无 `sleep_replicas()` / `update_weights()` | 0.9.0 的 `fit` 里这两步是显式的，且**顺序有讲究**（见 §4.3） | 崩·必须补 |

仍然存在、可以直接用的（已逐项 `hasattr` 验证）：
`apply_kl_penalty`、`compute_advantage`（签名完全一致）、`compute_response_mask`、
`marked_timer`、`reduce_metrics`、`should_save_ckpt_esi`、`Tracking`、
`pad_dataproto_to_divisor`、`agg_loss`、`process_validation_metrics`、
`_balance_batch`/`_dump_generations`/`_load_checkpoint`/`_save_checkpoint`。

### 2.1 0.9.0 的 `__init__`（决定签名怎么改）

```python
RayPPOTrainer(
    config, tokenizer, role_worker_mapping, resource_pool_manager,
    ray_worker_group_cls=RayWorkerGroup, processor=None,
    train_dataset=None, val_dataset=None, collate_fn=None,
    train_sampler=None, device_name=None,
)
```

`build_trainer`（`quarl/algorithm_factory.py:460`）已经是按这个顺序用关键字传的，
所以 **trainer 自己的 `__init__` 必须改成同构**，然后把 `self.sp_config` 等自己那部分放在 `super().__init__` 前后都可以。

---

## 3. 重写方案：只改「粘合层」

### 3.1 `__init__`

```python
def __init__(self, config, tokenizer, role_worker_mapping, resource_pool_manager,
             ray_worker_group_cls=None, processor=None,
             train_dataset=None, val_dataset=None, collate_fn=None,
             train_sampler=None, device_name=None):
    self.sp_config = config.self_play
    super().__init__(
        config=config, tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls, processor=processor,
        train_dataset=train_dataset, val_dataset=val_dataset,
        collate_fn=collate_fn, train_sampler=train_sampler, device_name=device_name,
    )
    self.current_phase = SelfPlayPhase.PROBLEM_GENERATION
    self.sp_data_manager = SSPDataManager.remote()
    self.problem_extractor = ProblemExtractor(...)   # 原样
    self._load_problem_generation_data()             # 原样
```

删掉 `_validate_config` 覆盖（0.9.0 没有这个钩子；`self_play.validate_config` 这个开关随之作废）。

### 3.2 生成：统一成一个 helper

`_generate_proposer_trajectories` 和 `_generate_solver_trajectories` 里 L1112-1160 / L1232-1290 的大段
`pop` 逻辑可以直接换成基类的 `_get_gen_batch`，但**必须自己 repeat**（0.9.0 的 `generate_sequences` 不做 repeat）：

```python
def _rollout(self, batch: DataProto, n: int, phase: str) -> DataProto:
    """一轮生成：batch 原地保留 tensor 字段，返回 union 好的完整轨迹。"""
    gen_batch = self._get_gen_batch(batch)          # 0.9.0 自带：腾出 raw_prompt/tools_kwargs
    gen_batch.meta_info["global_steps"] = self.global_steps
    gen_batch = gen_batch.repeat(repeat_times=n, interleave=True)

    out = self.async_rollout_manager.generate_sequences(gen_batch)   # 唯一生成入口

    self.rollout_n = self.config.actor_rollout_ref.rollout.n        # SSP 自己要用
    batch.non_tensor_batch["uid"] = np.array(
        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
    batch = batch.repeat(repeat_times=n, interleave=True)
    batch = batch.union(out)
    for key in batch.batch:
        if key not in ("old_log_probs", "ref_log_prob", "rm_scores"):
            batch.batch[key] = batch.batch[key].long()
    if "response_mask" not in batch.batch:
        batch.batch["response_mask"] = compute_response_mask(batch)
    return batch
```

注意 `rm_scores` 要排除在 `.long()` 之外（0.9.0 的奖励是 float 张量，转 long 会丢精度/变 0）。

> ⚠️ **`meta_info["temperature"]` 是无效键**（实测）：`AgentLoopWorker.generate_sequences` 的采样参数来自
> `self.rollout_config.temperature`，而 `self.rollout_config = omega_conf_to_dataclass(rollout_config)`
> 是 worker **actor 创建时的快照**（`agent_loop.py:521`），driver 侧改配置不会传播过去。
> 唯一能用的 per-sample 覆盖是 `non_tensor_batch["__do_sample__"]`（只能切成 greedy）。
> ⇒ proposer(0.8)/solver(0.7) 的差异化温度，只有两条路：**（a）接受两者同温度**；
> **（b）在 fork 里加约 4 行**：`generate_sequences` 里读 `batch.meta_info.get("sampling_params")` 覆盖默认值。
> 这是本次移植唯一的框架缺口，需要你拍板。

### 3.3 更新：三处 wg 调用 → 三个基类封装

```python
with marked_timer(f"{role}_old_log_prob", timing_raw):
    old_log_prob, _mfu = self._compute_old_log_prob(gen_batch)   # 内部做 DataProto↔TD、去 padding、entropy
    entropys = old_log_prob.batch.pop("entropys")
    metrics[f"{role}/actor/entropy"] = agg_loss(
        loss_mat=entropys, loss_mask=gen_batch.batch["response_mask"],
        loss_agg_mode=self.config.actor_rollout_ref.actor.loss_agg_mode).detach().item()
    gen_batch = gen_batch.union(old_log_prob)

if self.use_reference_policy:
    with marked_timer(f"{role}_ref", timing_raw):
        gen_batch = gen_batch.union(self._compute_ref_log_prob(gen_batch))

# compute_advantage(...)   ← 原样保留，签名没变

with marked_timer(f"{role}_update_actor", timing_raw):
    with self._override_rollout_n(n_if_this_role):        # ← 见 3.4
        actor_output = self._update_actor(gen_batch)
metrics.update({f"{role}_actor_{k}": v for k, v in reduce_metrics(actor_output.meta_info["metrics"]).items()})
```

`_update_actor` 已经自己塞好 `calculate_entropy / global_batch_size / mini_batch_size / epochs / seed /
dataloader_kwargs / compute_loss`，所以原来那句 `gen_batch.meta_info["multi_turn"] = ...` 可以删。

### 3.4 每个角色自己的 `n`（容易漏，改错了是静默的）

`_update_actor` 里硬编码：

```python
ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n   # ray_trainer.py:1319
```

但 SSP 的 proposer 用 `sp_config.proposer.n`（官方 launcher = **1**），solver 用 `rollout.n`（= 5）。
当 `ppo_mini_batch_size < train_batch_size` 时，这个乘法会让 proposer 的 mini-batch 变成 5 倍大 →
mini-batch 数量、梯度步数全错，而且**不报错**。

最省事的处理（不用 fork `_update_actor`）：

```python
from contextlib import contextmanager

@contextmanager
def _override_rollout_n(self, n: int):
    cfg = self.config.actor_rollout_ref.rollout
    old = cfg.n
    cfg.n = int(n)
    try:
        yield
    finally:
        cfg.n = old
```

### 3.5 验证：直接删掉覆盖

删掉 `_validate`（L154-337）和 `_validate_config`（L114），让基类的 `_validate(merged=False)` 接管。
基类会走 async rollout + `rm_scores`，所以 val 数据要满足：
- `data_source` 能路由到 **solver 奖励函数**（QA EM）
- 带 `reward_model.ground_truth`

同时把 `fit` 里两处 `self.val_reward_fn is not None` 判断删掉（改成无条件调用基类 `_validate`）。

---

## 4. 奖励体系：这是移植里最需要重新设计的一块

### 4.1 0.9.0 的奖励是「rollout 内流式、按样本、异步」

```
AgentLoopWorker → RewardLoopWorker.compute_score(单条) → 标量
                → rm_scores = zeros；rm_scores[最后一个有效 token] = 标量
driver 侧 extract_reward(batch) 就是 batch.batch["rm_scores"]
```

`solver` 的奖励天然适配（就是 QA EM），`proposer` **不适配** —— 它的奖励要等 solver 跑完。

### 4.2 推荐架构：solver 走流式，proposer 走 driver 侧覆盖

```python
# ① solver：已经在 rollout 里算好了，直接取
solver_scores, solver_extra = extract_reward(solver_gen_batch)      # (B, T) 张量
solver_gen_batch.batch["token_level_scores"] = solver_scores

# ② proposer：用 solver 的成功率反推（SSP 原有逻辑，一行不改）
proposer_rewards = self._compute_proposer_rewards(proposer_gen_batch, solver_scores, solver_extra)
proposer_gen_batch.batch["token_level_scores"] = proposer_rewards
```

配套要求：**proposer 侧 rollout 也会被调用 reward manager**，必须让它返回 0（占位），否则会因为
proposer 种子数据没有 `reward_model.ground_truth` 直接 KeyError。做法两种：

- 在 `SSPRewardManager.run_single` 里判断 `extra_info.self_play_role == "proposer"` → `return 0.0`；
- 或者给种子数据塞 `reward_model = {"ground_truth": "", "style": "rule"}` + 在 reward fn 里对空 gt 返回 0。

### 4.3 `SSPRewardManager` 必须重写成新接口

`quarl/reward/manager.py:35` 的 `SSPRewardManager` 是老接口（`__call__(data: DataProto, return_dict=False)`），
0.9.0 要求：

```python
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

class SSPRewardManager(RewardManagerBase):
    def __init__(self, config, tokenizer, compute_score=None,
                 reward_router_address=None, reward_model_tokenizer=None): ...

    async def run_single(self, data: DataProto) -> dict:   # data 只含 1 条
        ...
```

可直接照 `quarl/reward/manager/search_r1_reward_manager.py` 抄（那份已经是新接口了）。

**注册方式也变了**：0.9.0 走 `reward.reward_manager.source=importlib` + `module.path` + `name`，
不是本仓库里那个 `algorithm_factory.load_reward_manager`（**那个函数已经没人调用了，是死代码**）。

### 4.4 ⚠️ `manager.py` 与 `manager/` 同名，随时会踩

```
quarl/reward/manager.py                 ← SSPRewardManager / NaiveRewardManagerComputeWithPrompt
quarl/reward/manager/                   ← 目录，里面是 search_r1_reward_manager.py，且**没有 __init__.py**
```

现在 `from quarl.reward.manager import SSPRewardManager` 能解析到 `manager.py`（模块优先于命名空间包），
但只要有人在 `manager/` 里加一个 `__init__.py`，**所有 import 立刻指向目录，`SSPRewardManager` 消失**。
建议这次一起整理成 `quarl/reward/managers/`（目录 + `__init__.py` 显式导出）。

---

## 5. `fit()` 的执行顺序要重排（0.9.0 才有的 sleep/wake 约束）

0.9.0 的 `fit` 里，`sleep_replicas()` 是在**生成结束后**调用的（把 KV cache 腾给训练），
`update_weights()` 在 actor 更新后调用（唤醒 + 推新权重）。SSP 一次 step 里有**两次生成、两次更新**，
顺序必须是这样：

```
generate(proposer)          ← replicas 醒着
generate(solver)            ← replicas 仍然醒着（同一版权重！）
checkpoint_manager.sleep_replicas()      ← 两次生成都做完才睡
reward: solver(extract) → proposer(driver)
update(proposer)  update(solver)         ← 或者 combine_update 合并成一次
checkpoint_manager.update_weights(step)  ← 醒来 + 推新权重
save_checkpoint（按 save_freq）
```

两个必须知道的副作用：

1. **两次更新是串行 SGD**：proposer 更新完，solver 的 `old_log_prob` 是用**已更新**的权重算的，
   而 solver 的轨迹是**更新前**生成的 → rollout 与 old_log_prob 天然有偏。这是 SSP 原代码就有的行为，
   不是移植 bug；但如果想要严格 on-policy 的 PPO 比值，得把两次 `old_log_prob` 都提前算完再更新。
2. **`combine_update=true` 把两次更新合成一次 backward**（`DataProto.concat` 后一次 `_update_actor`），
   step 时间几乎砍掉 1/3，代价是两个角色的梯度混在一起、不能用不同的 lr/epochs。建议首跑就开。

---

## 6. 除 trainer 之外，还要动的东西

| 项 | 现状 | 要做 |
|---|---|---|
| `quarl/config/self_play_config.yaml` | `hydra.searchpath: file:///root/code/verl/verl/trainer/config`（上游作者机器的绝对路径） | 改成 `${oc.env:WORKSPACE}/verl/verl/trainer/config` 或相对路径 |
| 同上 | `multi_turn.format: quark` | **仓库里没有 `quark` 这个 parser**，要换成 `search`（→`SearchToolParser`，`<search>q</search>`）或 `hermes` |
| 同上 | `reward_model.model.type: default`、`quark.diff_val_reward_fn_config` | 0.9.0 由 reward_loop 接管，这两块大概率是死配置，需实测确认后删 |
| 同上 | 缺 `reward.reward_manager`（importlib 形式） | 补 `source=importlib / module.path=quarl.reward.manager / name=SSPRewardManager` |
| `examples/train_ssp_qwen2.5_7b_instruct.sh` | `rollout.name=sglang_async` | 改 `sglang`（注册表只认 `(vllm|sglang|trtllm, async)`） |
| 同上 | `PPO_MAX_TOKEN_LEN_PER_GPU=1×(prompt+response)`（L101）、显式 `use_dynamic_bsz=False`（L177） | 与 search_r1 同样处理：`use_dynamic_bsz=True`、token 上限 2×。**SSP 一步有 2 次 rollout + 2 次 update，不改的话 step 时间是 search_r1 的 2 倍以上** |
| 同上 | 无 `ASCEND_RT_VISIBLE_DEVICES` export | 补（torch_npu 只认这个；`train_ssp_*` 上一轮已补，确认没回退） |
| 同上 | `OUTPUT_ROOT:-/tmp` | ckpt 落到持久盘（`/data/...`），FSDP ckpt 单步十几 GB |
| `_dump_trajectories`（L1552） | 硬编码 `base_dir = "/primus_oss/output"` | 改成 `self.sp_config.get("dump_dir")` |
| proposer 种子数据 | `self_play.generation_data_path` 指向 jsonl | **必须存在**，否则走 `_create_dummy_generation_data()` 只有 2 条假数据 |
| judge 服务 | `ProblemExtractor` 读 `JUDGE_BASE_URL` / `JUDGE_MODEL`（仓库版已从上游的 `QUARK_*` 改过） | 训练前要起一个 OpenAI 兼容的 judge server，否则问题抽取全失败（全 fallback → proposer 奖励恒为惩罚） |
| `data.clean_question` | 会把 prompt 重写成 `Question: xxx` | **SSP 必须为 false**：proposer 的 prompt 是"基于材料出题"的指令，被清洗掉就完全变味 |
| 检索服务 | — | proposer 和 solver 都要能调 search；proposer 侧只有开 `use_rag_filter` 时才真的检索 |
| `quarl/worker/fsdp_worker.py` | commit `314db00` 已删除（`SSPActorRolloutRefWorker`） | 当前仓库没有引用，确认 ssp 路径不需要它即可（0.9.0 的 `ActorRolloutRefWorker` 已覆盖） |

---

## 7. 逐函数处置表（1877 行怎么分）

| 函数 | 行数 | 处置 |
|---|---|---|
| `_compute_proposer_rewards` | 1317-1451 | **原样保留**（纯张量逻辑，`1-acc` / intermediate / format_only 三档都在） |
| `_apply_reward_dynamic_sampling` | 1010-1081 | 保留（只读 `token_level_scores`） |
| `_generate_proposer_with_dynamic_sampling` | 865-1009 | 保留（只改内部生成调用） |
| `_extract_and_assemble_solver_data` | 1178-1231 | 保留 |
| `_extract_trajectories_from_batch` | 1657-1709 | 保留 |
| `_extract_and_process_problems` | 1710-1755 | 保留（依赖 judge 服务） |
| `_prepare_solving_batch_from_data` | 1591-1656 | 保留（`RLHFDataset`+`collate_fn` 仍可用，且会产出 `raw_prompt`） |
| `_combine/_filter/_align/_create_fallback/_validate_extracted` | 1082-1111, 1756-1862 | 保留 |
| `_load_problem_generation_data` / `_create_dummy_generation_data` | 338-375 | 保留（dummy 数据建议补 `reward_model.ground_truth=""`） |
| `_dump_generations` / `_dump_trajectories` | 123-153, 1552-1590 | 保留，改 dump 路径 |
| `fit` | 376-864 | **重写粘合**：sleep/wake、奖励、更新、验证四处 |
| `_generate_proposer_trajectories` / `_generate_solver_trajectories` | 1112-1177, 1232-1297 | **改写**（合并成 §3.2 的 `_rollout`） |
| `_calculate_self_play_rewards` | 1298-1316 | **改写**（`extract_reward` 取代 `compute_reward`） |
| `_update_on_trajectories` | 1453-1551 | **改写**（3 处 wg 调用 + n 覆盖） |
| `__init__` | 67-113 | **改写**（签名） |
| `_validate_config` / `_validate` / `_balance_batch` | 114-122 / 154-337 / 1863-1877 | **删除**（用基类） |

净效果：**保留约 1300 行，重写约 200 行，删除约 400 行**。

---

## 8. 会遇到的语义偏差（知情，别当 bug 查）

1. **proposer 奖励的粒度是「prompt 级」**，但它被写到**最后一个有效 token**上。
   GRPO 组内标准化用的是 `uid` 分组，所以 proposer 的 `proposer.n` 必须 ≥2 才有组内方差；
   官方 launcher 设 `proposer.n=1` → 走 `len==1 → mean=0,std=1` 分支，优势退化成原始奖励（无 baseline）。
   这是原论文的取舍，不是 bug，但要确认是不是你想要的。
2. **solver 的 `n_s` 与 proposer 的 `n_p` 乘积决定 solver 批大小**（`P × n_p × n_s`）。
   `P=128, n_p=1, n_s=5` → solver 一次生成 640 条 12k 上下文轨迹，**显存/时间是 proposer 的 5 倍**。
   显存不够时先降 `n_s`，不要降 `train_batch_size`（后者会破坏 GRPO 组大小）。
3. **judge 抽不出题时 proposer 奖励恒为 `format_penalty`（-1）**：如果 judge 没起来，
   你会看到 `proposer_reward_mean ≈ -1` 且方差为 0 → 没有梯度，但 loss 正常、不报错。
4. **`meta_info["phase"]` 在 0.9.0 里没人读**（全仓库只有赋值的两处）。
   想按阶段切换 agent/工具，只能走 `extra_info.agent_name` 或 `tools_kwargs`，不能靠这个键。

---

## 9. 建议的落地顺序

| 阶段 | 内容 | 验收 |
|---|---|---|
| P0 | 改 `__init__`、删三个覆盖、import 通过 | `python -c "from ... import SSPRayPPOTrainer"` |
| P1 | 写 `_rollout` + 重写 `_update_on_trajectories` | 用 `proposer.enable=False, solver.enable=True` 跑 1 step，等价于「一次 search_r1」 |
| P2 | 重写奖励（`extract_reward` + proposer 覆盖）+ 新 `SSPRewardManager` | 日志里 `solver_reward_mean` 与 search_r1 同量级；`proposer_reward_mean ≈ 1-solver_acc` |
| P3 | 接 judge + 问题抽取，打开 proposer | `extraction_success_rate > 0.5`，`proposer_reward_std > 0` |
| P4 | 开 `combine_update=true`、dynamic bsz、sleep/wake 时序验证 | step 时间 ≈ 1.3× search_r1（不是 2×） |

**先做的第一个实验**：`P1` 里把 proposer 关掉、只跑 solver，跑通 1 step 就算成功 ——
这一步验证的其实是「你的 `_rollout` + `_update_on_trajectories` 改写对不对」，
把 SSP 的两阶段问题降维成已知能跑的 search_r1 问题。这一步不过，后面全是白搭。

---

## 10. 训练设置对照：search_r1（已重构）vs SSP（现在这份）

**先看范式差异，不是参数差异**：search_r1 走 GRPO、无 critic、无 offload、dynamic bsz；
SSP 现在的 `TASK_TYPE=origin` 分支走 **GAE → `need_critic()` 强制起一个 7B critic**，
所以上游不得不把 actor/ref/critic/reward_model 全部 offload。两条路的显存与吞吐画像完全不同。

| 类别 | 项 | search_r1（`train_searchr1_qwen2.5_7b_instruct.sh`） | SSP（`train_ssp_qwen2.5_7b_instruct.sh`） | 上游 `train_ssp.sh` |
|---|---|---|---|---|
| 算法 | adv_estimator | **grpo** | **gae**（origin）/ grpo（deep_search） | 同 |
| | critic | 不创建（`need_critic`=False） | **必须创建**（7B value model） | 有 `critic.*` 全套 |
| | use_kl_loss | True | False（origin） | 按任务 |
| | clip_ratio_high | 0.285 | 0.2 | 0.285 |
| | lr_warmup | 无 | 5 steps / 0.05 | 5 steps / 0.05 |
| | kl_loss_coef / type | 默认 | 0.01 / low_var_kl（但 KL loss 关闭，无效） | — |
| 采样 | rollout.n | **5** | **1**（origin）/ 5 | 同 |
| | val_kwargs | 默认 | n=1, temp=0.0, **do_sample=True**（矛盾） | n=1, temp=0.0 |
| batch/吞吐 | use_dynamic_bsz | **True** | **False** | False |
| | ppo_max_token_len_per_gpu | **2×**=24576 | **1×**=12288 | 1× |
| | ppo_micro_batch_size_per_gpu | null（dynamic 下忽略） | 4（dynamic 关闭时才生效） | 1 |
| | max_num_batched_tokens | 单独 1×=12288 | = PPO token 上限（耦合） | 同 |
| 显存 | param/optimizer/offload_policy | **全 False（不显式设置）** | **全 True** | 全 True（含 ref/critic/reward_model） |
| | enable_gradient_checkpointing | True | True | True |
| | attention 实现 | `flash_attention_2`（config override_config） | 未设（sdpa） | — |
| 工具 | multi_turn.format | **search**（`SearchToolParser`） | **quark（不存在）** | quark |
| | tool 接线 | `function_tool_path` | `tool_config_path` | `tool_config_path` |
| | max_tool_response_length | **8192**（注释：no eager truncation, SSP-style） | **未设 → 默认 256 字符** | — |
| | max_parallel_calls | 3 | 未设 → 1 | — |
| | chat template | `SEARCH_R1_JINJA` 注入 | **未注入** | — |
| 集群 | n_gpus_per_node | 10 | 8 | 8 |
| | tensor_model_parallel_size | **1** | 2 | 4 |
| | rollout.name | vllm | **sglang_async**（注册表无此值） | sglang_async |
| 奖励 | 接线 | `reward.reward_manager` importlib → `SearchR1RewardManager` + `reward.custom_reward_function` | `reward_model.reward_manager=ssp` | 同 |
| | 额外键 | — | `+custom_reward_functions.quark_score.*`（**死配置**，只有旧版 `load_reward_manager` 读） | 同 |
| 数据 | clean_question | false | 未设（默认 false） | — |
| | max_samples_per_source（val 分层抽样） | 100 | 未设（全量） | — |
| | dataloader_num_workers / filter workers | 4 / 8 | 未设 | — |
| 记录 | logger | console + mlflow（含 DB 预迁移） | 未设 → console | — |
| | trace backend | mlflow（`rollout.trace.backend`） | 未设 | — |
| | save/test freq | 20 / 20 | 5 / 5 | — |
| 评测 | JUDGE_* 环境变量 | **导出**（`search_eval_score` 需要） | **未导出**（但问题抽取 `ProblemExtractor` 依赖它） | — |

### 10.1 按 search_r1 口径重设 SSP 的推荐值

```bash
# 算法：改成 GRPO，砍掉 critic（省一整份 7B）
algorithm.adv_estimator=grpo
critic.enable=False
ROLLOUT_N=5                      # solver 组大小；proposer.n 留在 self_play 里设 1

# 吞吐：与 search_r1 一致
actor_rollout_ref.actor.use_dynamic_bsz=True
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null
PPO_MAX_TOKEN_LEN_PER_GPU=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2))
ROLLOUT_MAX_NUM_BATCHED_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

# 显存：7B / 64G / FSDP 分片，不需要 offload
actor_rollout_ref.actor.fsdp_config.param_offload=False
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
actor_rollout_ref.actor.fsdp_config.offload_policy=False

# 工具：必须显式放大，否则检索结果被截成 256 字符
actor_rollout_ref.rollout.multi_turn.max_tool_response_length=8192
actor_rollout_ref.rollout.multi_turn.max_parallel_calls=3
actor_rollout_ref.rollout.multi_turn.format=search
export SEARCH_R1_JINJA="$(cat "$TRAINING_DIR/quarl/template/search_r1.jinja")"
```

### 10.2 顺带：SSP 单步成本其实不是 2×，是约 1.2×

`n_p=1, n_s=5, batch=256` 时：

```
proposer 轨迹 256×1   =  256
solver   轨迹 256×5   = 1280
合计生成 1536  vs search_r1 的 1280        → 1.2×
更新样本 256+1280=1536 vs 1280             → 1.2×（combine_update=False 时是两次 optimizer step）
```

之前我说"2 倍"是按"两次完整 rollout"估的，实际 proposer 只有 1/5 的体量，应更正为 **1.2~1.3×**。
真正被放大的是**固定开销**（两次 old_log_prob/ref/update、两次权重同步窗口），这部分开
`combine_update=True` 能压掉一大半。

---

## 11. 上游 SSP 的训练参数能不能当"提速参考"？——不能

`third_party/SSP/examples/train_ssp.sh` 的速度相关参数（8 卡、7B actor、TP=4）：

| 参数 | 上游值 | 对吞吐的影响 |
|---|---|---|
| `actor.ppo_micro_batch_size_per_gpu` | **1** | 每次前反向只 1 条 → 严重欠载 |
| `rollout.log_prob_micro_batch_size_per_gpu` | **1** | 同上（纯前向，最该放大） |
| `ref.log_prob_micro_batch_size_per_gpu` | **1** | 同上 |
| `actor.use_dynamic_bsz` | **False** | 按条数切，不按 token 凑批 |
| `actor.ppo_max_token_len_per_gpu` | **12288（1×）** | 官方要求 ≥2× |
| `rollout.max_num_batched_tokens` | 12288（复用同一变量） | — |
| `param_offload` / `optimizer_offload` / `offload_policy` | **全 True** | 每步 CPU↔GPU 搬运 |
| `enable_gradient_checkpointing` | True | +30% 计算 |
| `actor.fsdp_config.fsdp_size` | -1（全分片） | — |
| `ulysses_sequence_parallel_size` | 1（ACTOR_SP=1，但**暴露成变量**） | — |
| `rollout.n` | 5（`TRAIN_METHOD=grpo` 默认） | 生成量与 search_r1 相同 |
| `total_epochs` | **10** | 步数 ×10 |
| `save_freq` / `test_freq` | 5 / 5 | 频繁存 ckpt + 频繁验证 |
| `data.val_batch_size` | **256**（= train batch） | 验证成本 ×4（我们设 64） |

**按我们 4B/40min 的实测标定**（batch 1280 条、8 卡、12k token）：

| | update micro | logprob micro | ref micro | micro-step 合计 |
|---|---|---|---|---|
| 我们（4B，已实测 40 min） | 2 → 80 | 1 → 160 | 1 → 160 | **400**（≈6 s/步） |
| 上游（7B） | 1 → 160 | 1 → 160 | 1 → 160 | **480** |

7B 单步算力约为 4B 的 1.6 倍 ⇒ 上游这套搬到本机约 **75~85 min/step**。

⇒ **上游参数是"能跑通"的默认值，不是"跑得快"的参考。照抄只会更慢。**
真正值得记的是作者**把哪些参数抽成变量**（`ACTOR_SP` / `ACTOR_FSDP_SIZE` / `LOG_PROB_MIRCO_BATCH_SIZE_PER_GPU` / `CRITIC_MIRCO_BATCH_SIZE_PER_GPU` / `REWARD_MIRCO_BATCH_SIZE_PER_GPU` / `ROLLOUT_TP`）——
这等于承认这些值必须按集群重新调，默认值只是占位。

### 11.1 但 proposer 侧这 8 个值值得照抄（与速度无关，影响能不能学到东西）

```bash
self_play.proposer.n=1
self_play.proposer.warm_up_steps=-1        # 上游：-1 ⇒ 第 0 步就更新 proposer（我们默认 5 会延迟）
self_play.proposer.format_penalty=0        # 上游：抽取失败给 0（我们默认 -1，等于额外惩罚）
self_play.proposer.reward_type=1-acc
self_play.proposer.adv_estimator=grpo      # 注意：n=1 时 grpo 退化成无 baseline
self_play.answer_pattern=question          # 上游抽 <question>（我们默认 "answer"）
self_play.extraction_failure.strategy=reuse
self_play.extraction_failure.reuse_success_rate_threshold=1.0
self_play.combine_update=False
self_play.mini_epochs=1
```

---

## 12. 原版 Search-R1（`third_party/Search-R1/train_grpo.sh`）的训练预算 —— 这才是"快"的答案

| 项 | 原版 Search-R1 | 我们现在的 search_r1 launcher |
|---|---|---|
| `data.train_batch_size` | **512** | 256 |
| `data.val_batch_size` | 256 | 64 |
| `actor.ppo_mini_batch_size` | **256** | 128 |
| `actor.ppo_micro_batch_size`（**全局**） | **64 → 每卡 8** | — |
| `rollout.log_prob_micro_batch_size`（全局） | **128 → 每卡 16** | 1（per-gpu） |
| `ref.log_prob_micro_batch_size`（全局） | **128 → 每卡 16** | 1（per-gpu） |
| `data.max_start_length` | **2048** | 4096 |
| `data.max_response_length` | **500** | **8192** |
| `data.max_obs_length` | **500（token）** | 256（字符，默认）|
| `max_turns` | **2** | **10** |
| `trainer.total_training_steps` | 1005 | — |
| `save_freq` / `test_freq` | 100 / 50 | 20 / 20 |
| offload | param + **grad** + optimizer 全开 | 全关 |

**`ppo_micro_batch_size` 是全局量的证据**：0.9.0 的 `workers/config/actor.py:244` 有
`assert ppo_micro_batch_size * sp_size >= n_gpus` —— 只有"全局"量才需要这个约束。
所以原版 `64/8 = 每卡 8`，`128/8 = 每卡 16`。

### 12.1 micro-step 数：他们 80，我们 400

| | update | logprob | ref | 合计 |
|---|---|---|---|---|
| 原版（batch 512×5=2560，mini 256×5=1280，micro 8/16 每卡） | 40 | 20 | 20 | **80** |
| 我们（batch 256×5=1280，mini 128×5=640，micro 2/1/1） | 80 | 160 | 160 | **400** |

每步处理的 token 总量只差 1.7×（他们 27M，我们 46M），但墙钟差 5~8× ⇒
**时间不是花在 token 上，是花在 micro-step 的固定开销上**，与之前的诊断一致。

### 12.2 根因：我们拿 SSP 的预算在跑 Search-R1

原版每条轨迹上限 ≈ `max_start 2048 + response 500×(2-1) + obs 500×2 = 3548` token；
我们现在是 `4096 + 8192 = 12288`，**3.5 倍长**，而且轮数 2 → 10，观测 500 token → 256 字符（单位都变了）。

⇒ Search-R1 baseline 应该回到原版预算（2048 / 500~1024 / obs 500 token / 2~4 turns），
12k 预算留给 SSP（它确实要 10 轮搜索）。两者混用会同时得到"慢"和"和论文对不上"。

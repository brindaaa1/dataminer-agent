# DataMiner Agent

在有明确、可验证 reward 的领域（这里用 OOT AUC）里，让 **LLM 负责发散和决策、确定性代码负责验证和执行** 的通用 agent 范式。主线是从原始多表数据出发、基于领域 know-how 挖掘特征，同时覆盖模型选择和调参；主场景是信贷风控（Lending Club 单表、Home Credit 多表），另用一个非金融场景（酒店预订取消）验证"换场景只换配置文件、不改一行代码"。

## 这是什么系统

- **特征由 know-how 驱动，不是 LLM 现场想象。** `knowhow/templates/` 把特征拆成"实体 × 时间窗口 × 聚合方式 × 派生方式"的模板库，LLM 在这个词表内组合或提出新假设；确定性代码负责把假设实例化成 SQL、做 point-in-time 关联、跑筛选漏斗。模板覆盖不到的假设可以走 `run_code`（LLM 直接写 Python），但产出永远先经过同一套安全检查。
- **模型不是 LLM 直接指定的，是实验出来的。** 候选模型先用低保真度赛马，由确定性的显著性检验排名次；LLM 真正的配置权在于把领域先验注入模型——比如"违约概率随信用分单调下降"这类业务常识变成 GBDT 的单调约束——约束能不能保留仍由对照实验决定。
- **调参分两层。** LLM（外循环）决定"下一步往哪个方向搜、要不要换模型、放大还是收紧搜索空间"；Optuna（内循环，TPE + 剪枝）在给定空间里找具体参数值。LLM 不手动给出学习率这类数值。
- **过拟合当成每轮都会发生的默认状态处理** 每次决策后代码强制跑一遍诊断（gap 过大、单特征信号过强、分数漂移、连续无进展……），诊断码映射到候选动作作为策略先验喂给 LLM；一个实验能不能被接受，只由确定性代码基于配对 bootstrap + 逐步收紧的显著性阈值判定，LLM 的任何自我判断都会被丢弃。

## 架构

```
                              Orchestrator（状态机）
   INTAKE → CLARIFY → PROFILE → PLAN → ┌─ DECIDE → EXECUTE → EVALUATE → RECORD → STOP_CHECK ─┐ → FINAL_GATE → REPORT → CONSOLIDATE
                                       └──────────────────── 未停止 ─────────────────────────┘
                    LLM 只出现在 PLAN / DECIDE；输出经 pydantic 校验，失败回灌错误重试
        ┌───────────────────┬────────────────────┬────────────────────┬────────────────────┐
        ▼                   ▼                    ▼                    ▼                    ▼
  特征工厂 / run_code    Optuna 内循环         Evaluator            Memory              FinalGate
  (模板+PIT+筛选漏斗)   (TPE + 剪枝)      (guardrail诊断码 +     (Lesson生命周期 +      (唯一能读 holdout
                                          配对bootstrap判定)      指纹检索+冷启动)        的模块，只跑一次)
        └───────────────────┴────────────────────┴────────────────────┴────────────────────┘
                                             ▼
                          Data Access Layer（四段切分：train/valid/oot_dev/holdout 物理隔离）
```

## 核心设计

### 1. Holdout 不是靠约束 agent，而是代码层面拿不到

```python
def issue_holdout_token() -> str:
    caller = sys._getframe(1).f_globals.get("__name__")
    if caller != _ALLOWED_ISSUER:          # 只有 evaluation.final_gate
        raise HoldoutAccessError(f"{caller} 无权获取 holdout token")
    return _SECRET
```

`DataAccess.load("holdout")` 没有这个 token 一律抛异常，agent 能调用的所有工具都构造不出它。测试直接静态扫描 `agent/`、`tools/` 目录源码，确认它们不 import `final_gate`。

### 2. 判卷权只属于代码，LLM 的自我判断会被原样记录、绝不采信

```python
if fails:
    out["verdict"] = Verdict.REJECT
elif noninferior and not cmp["significant"]:   # 单调约束的非劣效判定：只有 evaluator 能给 ACCEPT
    out["verdict"] = Verdict.ACCEPT
elif cmp["significant"]:
    out["verdict"] = Verdict.ACCEPT if cand["fidelity"] == "full" else Verdict.PROMISING
```

`evaluate()` 的签名里专门留了 `llm_claimed_verdict` 参数——LLM 想在输出里夹带"我觉得这个应该被接受"，会被记录下来但从不参与判定。判定用的是同一批 OOT-dev 样本上的配对 bootstrap，且显著性阈值随使用次数收紧（对抗"holdout 被用坏"）。

### 3. 特征的默认姿态是"不可信"，要自己挣得信任

```python
for f in [f for f in cand if metrics[f]["auc"] > thr]:
    leak[f] = metrics[f]["auc"]     # 不直接丢弃：交给 guardrail / 人工，不进入 kept
```

筛选漏斗按 **缺失率 → IV → 单特征 AUC 报警 → 相关性去重 → PSI** 的顺序跑。`run_code` 产出的特征无论过没过这条流水线，都会被单独强制标记为可疑（可得时间不可信），必须人工批准才能进最终模型——这是"生成端放开、验证端收紧"这条主线最重的一处体现。

### 4. Memory：LLM 能提出经验，不能批准经验

```python
"""LLM 只能提出候选（key 必须是代码从本任务实验树推导出的枚举键；必须引用真实 exp_id；极性必须与证据判定一致）；
状态流转（CANDIDATE → VALIDATED → ACTIVE / DEPRECATED）全部由本文件的规则决定，LLM 无法晋升。"""
```

指纹距离超过阈值直接冷启动，不会强融不相似任务的经验——单元测试直接验证了酒店任务的经验不会污染信贷任务的记忆。

## 实测发现（如实记录，包括没有得到支持的假设）

| 结论 | 证据 |
|---|---|
| 关掉 guardrail，AUC 从 0.68 虚高到 1.00，上线视角（泄漏字段不可得）塌陷到 0.50 | `examples/a3_leak_demo.png` |
| 多表特征工厂贡献 +0.015~0.02 AUC，是 AutoML 结构性做不到的部分；给 AutoML 同样特征它就追平 agent，恰好证明增量来自特征而非搜索 | `examples/feature_factory_results.json`，`docs/DESIGN_NOTES.md #4` |
| 跨任务 Memory 的数值 warm-start **没有**加速收敛；真实 LLM 会引用历史经验改变决策，但在低复杂度任务上未必带来更好结果 | `examples/memory_demo.md`，`docs/DESIGN_NOTES.md #1` |
| 单调约束在本来就很稳的数据上测不出可保留的稳定性收益，规则如实判定"不保留" | `examples/monotone_ablation.md`，`docs/DESIGN_NOTES.md #2` |
| 训练中途 `kill -9`，同一条命令能从检查点恢复，不重复计算 | `examples/resume_demo.md` |

## 目录

```
agent/       状态机、DECIDE 的 pydantic 校验与重试、动作集合
data/        四段切分、point-in-time 关联、holdout 隔离
features/    模板库展开成 SQL、筛选漏斗
modeling/    model zoo（lgbm / lr_scorecard）、Optuna 内循环、单调约束
evaluation/  guardrail 诊断码、配对 bootstrap、OOT-dev 使用预算收紧、唯一的验收模块
memory/      数据集指纹检索、Lesson 生命周期
tools/       LLM 可调用的工具，含 run_code 沙箱
runtime/     事件日志、检查点 / resume、预算
report/      模型卡 + Mermaid 实验树生成
baselines/   B0（默认参数）、B1（AutoML）对照
eval/        端到端脚本：跑 agent、跑基线、核心实验的复现脚本
knowhow/     数据集字段语义、黑名单、特征模板（格式说明见 knowhow/README.md）
examples/    预先跑好的结果，不需要下载数据也能看
docs/        更详细的设计取舍与实验记录（含负面结果）
```

## 运行

见 `RUN.md`。`pytest` 不需要任何真实数据，约一百个测试全部用合成数据，几十秒内跑完。

## 已知边界

以下部分目前只有设计、没有代码：独立的 critic 审查（用干净上下文复核泄漏/业务合理性，避免主 agent 自我审查的确认偏误）、多任务并发的 Executor 抽象、完整的三级 HITL 门控与异步审批（当前只实现了"泄漏时必须人工确认"和"决策连续校验失败"两个触发点）、分客群建模。`run_code` 的沙箱是子进程 + 超时，不是容器级隔离——如实说明取舍见 `docs/DESIGN_NOTES.md #3`。

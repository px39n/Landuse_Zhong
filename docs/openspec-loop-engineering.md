# OpenSpec Loop v3：运行指南、核心机制与最佳实践

本指南说明仓库当前实际使用的轻量 OpenSpec 工程 Loop。它以
`openspec-loop-engineering` 为唯一 supervisor，用 task registry 的机械指纹约束
开工，用写范围波次组织 Apply，用单一账本和 supervisor Verify 闭合结果。

核心原则是：**合同与执行分离，普查与派出分离，worker 与最终权威分离，普通
验证与审计留痕分离。** Loop 不是 ROSE 或 `aili-delivery-flow` 操作系统，也不创建
第二套 slash、Board、`progress.txt`、worker ledger 或生命周期。

## 1. 适用边界与权威入口

| 场景 | 入口 | 职责 | 不拥有 |
|---|---|---|---|
| 问题仍不清楚 | `openspec-explore` | 比较方案、定位证据与缺口 | Apply、Verify、promotion |
| 新建或补齐合同 | `new / continue / ff / omx-bridge` | 写 proposal、design、specs、tasks | 运行时监督 |
| 边界拷问与语义修订 | `openspec-change-interviewer` | 锁定范围、保留、路径、验收及 active registry | 产品实现、promotion |
| 有限预算执行 | `openspec-loop-engineering` | 唯一 supervisor；plan、dispatch、join、Verify、promotion | 第二生命周期 |
| 单包实现 | `openspec-apply-change --task ... --orchestrated` | 只执行一个 packet 绑定的 ref/attempt | ledger、Verify、PASS |
| 独立验收 | `openspec-verify-change --task ...` | 对 ACCEPT/TEST 给出正式 verdict | Apply、promotion |
| 偏离诊断 | `openspec-unblock-research` | 形成 probe 或 disposition | 自动再派、补预算 |
| false-success 检查 | `silent-failure-hunting` | 一个 ref 或 join 点的一项只读风险检查 | 生命周期、最终 Verify |
| 专项审查 | `review-pipeline` | 用户点名或一个残余风险的一次只读审查 | review swarm、PASS |
| 全量审计 | `monitor-openspec-codex` | 仅在 `retention=full` 或 legacy audit 时写 bundle | 普通 Loop 默认路径 |

`rose` 只是 supervisor-direct 的兼容 Role ID 别名，不是可部署 worker，也不新增
宿主。canonical Role ID 只选择最窄能力；没有合适角色时保持 direct 或明确
`blocked`，不得用 `general` 假装拥有任务。

## 2. Skill tree map

### 2.1 语义源与镜像树

```text
.agents/skills/                         # 唯一 canonical 语义源
├─ openspec-change-interviewer/         # 合同边界、fingerprint readiness
├─ openspec-feature-list/               # tasks -> compact active registry
├─ openspec-loop-engineering/           # 唯一 supervisor
│  └─ references/
│     ├─ canonical-roles.json           # 21 个 Role ID；rose 仅为 supervisor 别名
│     ├─ role-adapter-matrix.md          # Role ID -> 本地入口与写权限
│     ├─ subagent-task-packet.md         # 一个有界 dispatch envelope
│     └─ subagent-result.md              # terminal result；无最终 PASS 权限
├─ openspec-apply-change/                # packet-bound Apply
├─ openspec-verify-change/               # supervisor-owned Verify
├─ openspec-unblock-research/            # ref-local disposition/probe
├─ silent-failure-hunting/               # 只读 subordinate host
└─ review-pipeline/                      # 只读 subordinate host

scripts/sync_openspec_loop_skills.py     # 只做受管语义文件同步
├─ .codex/skills/                        # 机械镜像；保留平台 metadata
└─ .claude/skills/                       # 机械镜像；保留平台 metadata
```

修改 canonical skill 后只走这一条同步链：

```powershell
python scripts/sync_openspec_loop_skills.py
python scripts/sync_openspec_loop_skills.py --check
```

`--check=0` 证明受管文件字节一致，不证明 scheduler、合同测试或产品语义已经通过；
仍须运行对应 tests 和 OpenSpec 验收。

### 2.2 调用与不调用关系

```text
explore -> new|continue|ff -> change-interviewer -> feature-list
tasks + feature_list.json + loop.json
  -> loop-engineering (sole supervisor)
     -> proactive scan
        -> direct: rose (non-deployable alias)
        -> dispatch: apply-change -> result -> actual-dispatch join
                    -> verify-change -> promote
     -> BLOCKED|DEVIATED -> unblock-research
        -> retry|targeted_probe | interviewer/full_auto amendment | stop_budget
     -> advisory only: silent-failure-hunting | review-pipeline
     -> retention=full|explicit audit: monitor-openspec-codex
```

| Caller | 可调用 | 返回给谁 | 明确不得调用/拥有 |
|---|---|---|---|
| interviewer | feature-list、strict validation、check/plan | Loop supervisor | Apply、Verify、promotion |
| Loop supervisor | apply、verify、unblock；必要时两个 review host | 自己完成 join 与处置 | ROSE/delivery-flow supervisor、第二 ledger |
| apply | owner tests；packet 内瞬时重试 | Loop supervisor | verify、promote、ledger、另一个 ref |
| verify | 已 join 证据与完整 `TEST:` | Loop supervisor | Apply、worker 调度、自动修复 |
| unblock | 判别 probe 与 disposition | Loop supervisor/interviewer | 自动再派、自动扩 scope、补充次数 |
| review hosts | 一个 bounded 只读问题 | Loop supervisor | 彼此串联、review fan-out、最终 PASS |

`handoff.json` 是 change 内唯一 worker 协调面。`agents/<agent-id>` 是 packet 中的
逻辑地址，不是仓库目录、skill、slash command 或持久 agent 生命周期。

## 3. 核心设计机制

### 3.1 Start-work latch：机械指纹，不是用户手印

普通 Apply 的开工条件只有三项：

1. active task registry 存在且可解析；
2. `loop.json.contract_fingerprint` 与当前 registry 一致；
3. 没有 `pending_irreversible_policy`。

`seal-preview.md`、`confirmed_at`、legacy `sealed` 字段和
`seal --confirmed` 都不是普通开工门。缺少 `loop.json` 时，`check`/`plan` 可以从
change 已记录的 `thin` retention/path decision 与仓库有限默认自动写入：

- 当前 `contract_fingerprint` 与 `narrative_digest`；
- `autonomy: supervised` 与 `retention: thin`；
- 每个 active ref 的 `max_apply_attempts=2`、`max_unblock_runs=2`；
- 已记录路径及有限测试/执行策略。

自动初始化只写 policy file，不创建 ledger、scratch、cache、product、bundle 或
GUI/Colab 目录，也不授权不可逆操作。第一次 Apply 因此不需要 stamp。

只有以下不可逆政策需要人类明确确认：提高 legacy `hard_ceiling`，改变 retention、
paths 或 scope，破坏性外部写入，凭据，产品 `--commit`，以及 `git push`、PR 或
`main` 操作。`seal-preview.md` 可在这条分支上作为 change-scoped 可选审计诊断，
不能反过来成为第二次开工仪式。

### 3.2 Census → wave → dispatch：普查、拓扑与实际派出分层

`plan` 的字段必须分别阅读：

| 字段 | 含义 | 可被什么缩小 |
|---|---|---|
| `selected_batch` | dependency-ready 的完整普查集 | 依赖、环、局部合同错误、终态 |
| `selected_wave` | 通过四问后的完整写范围独立 Apply 集 | 写范围重叠、包不合格、无稳定 join |
| `apply_remaining` | 当前实际派出的 Apply 余量 | 已记录 Apply 使用量及现有有限预算 |
| `allowed_parallel_applies` | `min(|selected_wave|, apply_remaining)` | 只由上述两项计算 |
| `dispatch_refs` | 本轮实际可派出的确定性前缀 | allowance、iteration/minutes/breakers、写拓扑 |

每个候选包进入 `selected_wave` 前回答四问：

1. 包是否有界且非平凡？
2. 是否选择最窄 canonical Role ID？
3. 写范围是否与同波其他包不相交？
4. 是否有稳定的 supervisor-owned join id？

`selected_wave` 可以宽于 2，也不因 `apply_remaining=0` 被清空。余量为零时只是
`dispatch_refs=[]`，普查和波仍保留供诊断与后续排水。`host_soft_cap`、
`max_subagents`、distinct runtime id、Role ID 数量和只读 worker 数量只可作为
兼容诊断，不得过滤波、消耗 Apply 或制造 stop。

写范围相交时，把后一个 ref 放到下一波、串行 Apply 或 supervisor direct；这叫
拓扑约束，不叫人头闸。未知依赖和环只影响相关 ref，不应清空独立 ready siblings。

### 3.3 Authority 与记账：1 worker = 1 ref = 1 Apply record

每个 Apply dispatch 都创建一份 fresh packet，至少固定：Package ID、Role ID、
local entry、ref、Apply attempt、assignment、acceptance boundary、scope、forbidden
scope、allowed actions、write scope、expected result/evidence、execution、join 和
stop conditions。

一个 Apply worker 只能绑定一个 active ref 和一个 supervisor-owned attempt。
worker 可以在不改变 packet 的前提下多轮使用工具、运行最小检查，并对 429、超时
等瞬时工具/进程故障作包内重试；它不能合并另一 ref、写 ledger、promote、勾选
task 或宣称 PASS。

terminal result 返回后，只有 supervisor 写该 ref/attempt 的一条 canonical Apply
record；重复 `attempt_id` 被拒绝。Goal、stop hook、只读研究/审查、supervisor
Verify、`rose` direct routing 和 Verify 内的证据写入都是 0 Apply、0 调度人头。

写权限按最窄 Role ID 固定：`implementer` 只写 task-owned 产品/合同文件，
`test-engineer` 只写 task-owned 测试，`browser-qa-runner` 与
`e2e-artifact-runner` 只写合同批准的 evidence root；其他 dispatched roles 只读。

### 3.4 Join 与 retry：本波/本 ref 闭合，三类主人分开

同一 worktree 的实际派出成员必须全部返回 terminal result，supervisor 才能对
本轮任一 ref 开始 Verify。barrier 只覆盖 `dispatch_refs`，不是等待尚未派出的
`selected_wave` 尾部，更不是等待全 change。barrier 关闭后，每个 ref 仍按自己的
结果判断：只有 `completed` 且证据可检查者可 Verify；失败 sibling 只阻塞自己。
独立 worktree 可按自己的 join 闭合；无关 explore、只读 review 和不受本波写入
影响的工作不必空等。

三类重试/再派主人不得混写：

- **worker 内部重试（允许）**：terminal result 前的瞬时工具/进程故障；packet
  不变，仍是一轮 Apply；
- **supervisor 新 Apply（允许但过 gate）**：旧结果完成处置后，只有 gate、
  `max_apply_attempts` 和现有预算允许，才创建新 packet/attempt；
- **自动再派（禁止）**：terminal `failed|empty|partial|blocked|unverified` 不得自动
  续跑、换 worker、扩 scope 或恢复旧上下文。

断言失败、语义错误或 packet 的 Role/ref/scope/write scope/permission/acceptance
boundary 改变，都不是瞬时重试。unblock 可以建议 `retry | targeted_probe |
amend_spec | supersede_task | stop_budget`，但不自行 dispatch。

### 3.5 Retention：策略、动态账本与产品证据分离

| retention | 动态尝试 | tracked 决策 | 完整 bundle | 适用场景 |
|---|---|---|---|---|
| `none` | 可丢弃 ledger | 默认无 | 否 | 可快速重建的小任务 |
| `thin`（默认） | recorded disposable root 下的小 ledger/scratch | 仅 promotion receipt 或改变方向的 deviation | 否 | 普通工程与研究 Loop |
| `full` | 明确 retained root | 保留 | 是 | 法规、外部审计或用户明确要求 |

四类相似路径不能混同：

1. `openspec/changes/<id>/unblock/*.json|md`：tracked 决策报告；
2. 已批准 product/bundle root：重型产品或审计证据；
3. recorded scratch 下 `{pytest,cache,tmp}/<ref>/<run-id>/`：可丢弃临时物；
4. ledger `attempts[]`：计数、fingerprint、时长与 disposition。

普通 Loop 不默认写 BUNDLE、EVIDENCE、`progress.txt` 或 `runs.log`。只有明确
`retention=full` 或 legacy audit 才转给 monitor 的不可变 bundle 流程。

## 4. 状态与漂移

```text
pending ──依赖满足──> ready ──Apply──> in_progress
                                      │
                              Verify ─┼──> passed
                                      ├──> blocked
                                      ├──> deviated
                                      └──> ready (supervisor-gated next Apply)

blocked/deviated ──unblock──> retry | targeted_probe |
                              amend_spec | supersede_task | stop_budget
任何非终态 ──ref-local exhaustion/breaker──> maxed
旧任务 ──SUPERSEDES──> superseded
```

只有 `ready` 可进入 Apply。`passed | maxed | superseded` 是终态；
`blocked | deviated | in_progress` 需要显式处置，不能被模型“视为完成”。Apply
自身没有 PASS 权威；supervisor Verify 通过后才可用 `promote` 原子同步 checkbox、
feature state 和新 plan。

漂移分两类：

- **narrative 漂移**：proposal、design 或 specs 改变。运行
  `python scripts/openspec_loop.py reseal <change-id>` 刷新 narrative digest 并继续；
  `narrative_policy: strict` 才升级为暂停。
- **semantic 漂移**：active task registry obligation 改变。保留
  `selected_batch`，清空 wave；`supervised` 暂停问人，`full_auto` 由 supervisor
  记录 `--reason` 后 restamp。若同时改变不可逆政策，仍需人类确认。

勾选 checkbox 不是漂移。promotion 只改变完成状态，不改变 active obligation，
因此也不需要 stamp。

## 5. 有限预算、breaker 与局部耗尽

从 `loop.json` 读取有限预算，不因新 session、run id 或 revision 静默重置。
`revision_attempt_count` / `change_attempt_count` 可以包含 Apply、Verify、Explore、
Unblock 等记录，不能拿来冒充 Apply 使用量；读取 summary 的
`revision_apply_iterations_*` 和 `change_apply_iterations_*` 字段。

每个 ref 的 `max_unblock_runs=2` 在它进入 `blocked|deviated` 前休眠。激活后耗尽只把
该 ref 标成 `maxed|stop_budget`，不得清空 `selected_wave` 或阻止其他独立 ready
Apply。提高已激活的 ref-local unblock allowance 需要适用 authority 和记录 reason；
stamp 不能给已耗尽 ref 续命。

revision/change Apply remainder 只限制实际 `dispatch_refs`。现有 iteration、active
minutes 与 breakers 可以暂停实际派出，但不把 headcount 或 legacy ceiling 变成新
调度器。常见 breaker 包括重复 result fingerprint、连续 no-progress 和重复 semantic
deviation。

legacy `hard_ceiling` 保留为兼容/诊断 policy data：它不是普通 dispatch stop，也不
过滤 `selected_wave` 或消费 `apply_remaining`。Loop 不得自行提高它；提高该字段与
改变 retention、paths、scope 一样，属于不可逆政策人章范围。不要把“不得因
`hard_ceiling` 阻止普通 dispatch”误读成任何门闸都不能拦——fingerprint、未确认的
不可逆政策、现有工作预算、breakers 与写拓扑仍然有效。

`autonomy` 决定语义修订与可逆预算调整的主人，但 autonomy 不扩大仓库权限。
无论 `supervised` 或 `full_auto`，不可逆政策集合始终需要人类确认。

## 6. 从合同到 Apply 的可执行流程

### 6.1 首次或缺 `loop.json`

先确保 change 已写 Artifact Retention Decision，尤其是 `thin` 的 disposable、
pytest basetemp、scratch、product 与 GUI/Colab 选择。推荐普通本地 profile
`A_local_thin`，但必须按当前 `<change-id>` 展开，不能照搬其他 change 的绝对树。

```powershell
python scripts/generate_openspec_feature_list.py --change-id <change-id>
openspec validate <change-id> --strict
python scripts/openspec_loop.py check <change-id>
python scripts/openspec_loop.py plan <change-id> --batch
```

缺文件时 `check`/`plan` 自动初始化 thin policy。确认
`check.ok=true`、`fingerprint_ready=true`、没有
`pending_irreversible_policy`，并分别读取 `selected_batch`、`selected_wave` 与
`dispatch_refs`。普通 Apply 可立即继续，不生成 preview，也不等待 stamp。

不可逆政策分支才使用：

```powershell
# 可选：interviewer 生成 change-scoped seal-preview.md
python scripts/openspec_loop.py seal <change-id> --confirmed [明确的 policy options]
python scripts/openspec_loop.py check <change-id>
```

这条命令只持久化用户明确确认的 policy delta，不是常规启动命令。

### 6.2 每波 drain loop

1. `plan`：保存完整 `selected_batch`，按四问形成 `selected_wave`，只派
   `dispatch_refs`。
2. 为每个实际 Apply 创建 fresh packet；packet 只缩小 authority。
3. worker Apply 并返回结构化 terminal result；worker 不写账本。
4. supervisor 为每个 ref/attempt 写唯一 Apply record。
5. 同 worktree 的实际派出成员全部 join，关闭 wave barrier。
6. 对各自 `completed + inspectable evidence` 的 ref 做 supervisor Verify（0 Apply）。
7. `PASS` 后 `promote`；非 PASS 只处置该 ref。
8. 重新 `plan`，继续剩余独立 ready，直到无可派工作。

Loop 的同回合执行锁坚持“先做后说”：`gate` 对 `dispatch_refs` 返回 `continue`
后，同回合下一项 substantive action 必须是 Apply。可以先发一行进度，但不能以
总结、建议或 handoff 结束回合。

### 6.3 恢复与观察

```powershell
python scripts/openspec_loop.py check <change-id>
python scripts/openspec_loop.py plan <change-id> --batch
python scripts/openspec_loop.py summary <change-id> --run-id <run-id>
```

恢复时复用 recorded paths、fingerprint 与累计预算。出现漂移时按第 4 节处置，
不要顺手修合同或从旧 appendix 重新选择废弃任务。runtime 中仍含 `sealed` 输出键或
“re-run interviewer and seal”等 legacy naming，不得重新解释为 stamp authority。

## 7. 验收与收尾

| 层级 | 时机 | 证明范围 | 正式 bundle |
|---|---|---|---|
| Attempt | 每次局部改动后 | touched module owner tests | 否 |
| Promotion | 首次准备标 PASS | task 完整 `TEST:` + ACCEPT | 默认否 |
| Final | active refs 终态后 | whole-change verify、strict validate、相邻回归 | 默认否 |
| Audit | `retention=full` 或显式 audit | 不可变完整证据链 | 是 |

命令 exit 0 不等于产品语义正确。Verify 必须比较预期与观察的来源、目标、时间、
指标、分辨率、schema 和解释；使用错误表或错误目标，即使测试绿也应为
`DEVIATED`。

无 ready task 也不等于 design 已实现；更严格地说，没有 ready task
**不等于** design 已经实现：

```powershell
python scripts/openspec_loop.py goals <change-id>
python scripts/openspec_loop.py design-verify <change-id> --observation <path>
```

`goals` 检查 `GOAL:` / `COVERED_BY:` 结构覆盖；`design-verify` 用实际 observation
返回 `PASS` 或 `GAP`。缺 observation 是 `unobserved`，不能 PASS。`GAP` 形成 revision
proposal；`supervised` 交回 interviewer，`full_auto` 可在 authority 内记录 reason
后修订并重新排水，但 autonomy 不得制造空合同。只有 design PASS 后才运行
whole-change Verify 和 strict validate。

## 8. 最佳实践

1. **先看五个 plan 字段再派工。** 不把 `selected_batch`、`selected_wave`、
   `apply_remaining`、`allowed_parallel_applies`、`dispatch_refs` 混成一个“并发数”。
2. **任务显式写范围。** 为可写 task 提供 `FILES:` / `WRITE_SCOPE:`；范围不明时
   保守串行或 direct，不能用 ref 名猜测不相交。
3. **packet 小而完整。** 一个 packet 只含一个 ref/attempt，并写清 forbidden scope、
   acceptance boundary、expected evidence 与 join id。
4. **只等待实际派出。** 共享树 join 等 `dispatch_refs`，不是等待
   `selected_wave` 尾部；Verify 仍按 ref 判定，不把 sibling failure 扩大为
   change-wide failure。
5. **让失败保持局部。** blocker、未知依赖、unblock 耗尽和非 PASS 默认只影响当前
   ref；继续排空其他独立 ready。
6. **区分 retry 主人。** worker 只重试瞬时故障；supervisor 只在新 gate 后再派；
   unblock 只给 disposition，不自动执行。
7. **最终权威只在 supervisor。** worker、review host、Goal/stop hook 不写 Apply
   ledger、不 promote、不宣称 PASS，也不因 Role ID 获得人数或预算配额。
8. **默认 thin。** 只保留复现决策所需的 pointers/manifests；不要把完整日志和旧
   spec 堆进 change，也不要每次 attempt 生成 bundle。
9. **兼容字段只做诊断。** 报告 legacy `sealed`、`hard_ceiling`、`max_subagents` 时
   明确其兼容性质，不让它们悄悄回到 dispatch gate。
10. **同步和行为测试分开。** canonical skill 修订后先 sync/check，再跑 skill
    contract tests、runtime tests 和 task `TEST:`；三者不可互相替代。

### 最小命令闭环

```powershell
# Author/current contract -> compact registry -> integrity latch
python scripts/generate_openspec_feature_list.py --change-id <change-id>
openspec validate <change-id> --strict
python scripts/openspec_loop.py check <change-id>
python scripts/openspec_loop.py plan <change-id> --batch

# 只对 dispatch_refs 开 gate；Apply/record 后 join 实际派出成员
python scripts/openspec_loop.py gate <change-id> --run-id <run-id> --ref <ref> --kind apply
# $openspec-apply-change <change-id> --task <ref> --orchestrated
# supervisor records the one Apply result, then actual-dispatch join closes

# Ref-level acceptance and atomic promotion
# $openspec-verify-change <change-id> --task <ref>
python scripts/openspec_loop.py promote <change-id> --ref <ref>

# 仅在 compact index 漂移时修复 index；最终再做严格验证
python scripts/openspec_loop.py sync <change-id>
openspec validate <change-id> --strict
```

`sync <change-id>` 只修 compact index，不修改合同义务，也不代替 Verify 或
promotion。每轮实际派出仍以新 `plan.dispatch_refs` 为准。

## 9. 禁止模式

- 要求先接受 `seal-preview.md` 或运行 `seal --confirmed` 才允许第一次普通 Apply；
- 缺 `confirmed_at` 就清空 `selected_batch` 或 `selected_wave`；
- 用 `host_soft_cap`、`max_subagents`、Role/runtime id 或只读 worker 数量裁切波；
- 一个 packet 合并多个 ref，或让 worker 自报 completed 充当 canonical Apply record；
- terminal semantic failure 后自动再派、换人重跑、扩 scope 或恢复旧上下文；
- 全 change 停工等待一个不相关 ref，或 Verify 前等待未来所有 wave；
- 让 Apply/review host 自行 Verify、promote、写 PASS 或创建第二账本；
- 把 `rose`、delivery-flow、Board、A33 或 provider agent 文件变成运行时 supervisor；
- 用 stamp 重置 ref-local Apply/unblock exhaustion；
- 把 fingerprint latch 放宽成“任何 gate 都不得阻止 dispatch”；
- 新 session、run id 或 revision 静默重置累计预算；
- 把 scratch、product、tracked report 和 ledger 因共有 `attemptNNN` 名称而视为同物；
- ordinary thin Loop 自动升级为 legacy monitor 或 full bundle。

这套机制的稳定性来自三重锁：canonical skill 与镜像字节一致，runtime tests 证明
机械行为，OpenSpec task/whole-change Verify 证明当前 change 的接受条件。任何一项
单独为绿，都不能代替另外两项。

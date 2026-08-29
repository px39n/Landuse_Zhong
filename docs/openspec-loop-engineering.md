# OpenSpec Loop v3 最佳实践

本指南定义一套可以公开审计、长程运行、但不会把 OpenSpec 变成日志仓库的
有限预算工程 Loop。核心原则是：**合同与执行分离，当前合同与历史分离，
尝试状态与产品证据分离，普通验证与审计验证分离。**

## 1. 什么时候由谁接管

| 阶段 | 接管入口 | 主要职责 | 退出条件 |
|---|---|---|---|
| 问题尚不清楚 | `openspec-explore` | 比较方案、定位未知，不写实现 | 问题足够清楚，或形成一个需写入合同的决定 |
| 新建/补齐合同 | `new / continue / ff / omx-bridge` | 写当前 proposal、design、specs、tasks | 形成可执行草案 |
| 边界确认 | `openspec-change-interviewer` | 一次性确认范围、路径、保留级别、测试与预算 | 严格验证通过并生成 `loop.json` |
| 有限预算执行 | `openspec-loop-engineering` | 排空 `ready`；`continue` 后同回合 Apply | 全部通过、合同漂移、偏离、阻塞或预算停止 |
| 单次实现 | `openspec-apply-change --task ... --orchestrated` | 只改选中任务，运行 owner tests | 返回 APPLIED/BLOCKED/DEVIATED |
| 独立验收 | `openspec-verify-change --task ...` | 对照 ACCEPT 与完整 TEST 给出四态 verdict | PASS/FAIL/BLOCKED/DEVIATED |
| 方向修复 | `openspec-unblock-research` | 批判预期与实际差异，设计判别性 probe | retry/probe/amend/supersede/stop |
| 全量留痕审计 | `monitor-openspec-codex` | 不可变 bundle 与 BUNDLE/EVIDENCE 账本 | 仅在 full/audit 明确启用时 |

Loop **只在 interviewer 确认并 seal 后接管**。此后按两类漂移分别处理：

- **narrative 漂移**（proposal、design、specs 改变）：`reseal` 刷新 digest 后
  继续执行，策略字段逐字保留；只有 `narrative_policy: strict` 才升级为暂停。
- **semantic 漂移**（active task registry 的义务改变）：按第 6 节的授权模型
  处理，`supervised` 交回 interviewer，`full_auto` 由 supervisor 记录 reason
  后自行修订。

勾选 checkbox 不是漂移。fingerprint 已中和 checkbox 标记与 `STATE:` 指令，
promotion 不需要 reseal。不得在任何情况下自动沿用一个已失效的边界。

## 2. 单一语义源与跨端同步

- `.agents/skills/` 是 Loop 系列的唯一语义源。
- `.codex/skills/` 与 `.claude/skills/` 是机械镜像；平台元数据（例如
  `.codex/**/agents/openai.yaml`）独立保留。
- Cursor 与 Claude command 只是薄路由，不复制协议正文。

修改 canonical skill 后运行：

```powershell
python scripts/sync_openspec_loop_skills.py
python scripts/sync_openspec_loop_skills.py --check
```

这避免三个端口各自演化出不同的预算、状态或证据规则。

## 3. 三个状态载体

### 当前合同

`proposal.md + design.md + specs/** + active tasks.md` 是人类合同。
fingerprint 只覆盖这些当前内容。旧合同由 Git 保存，不在正文中持续追加完整
“Original Draft”或历史 appendix。

### 紧凑任务状态

`feature_list.json` 仅保存：

```text
id, ref, title, state, passes, depends_on, supersedes,
task_path, accept_hash, test_hash
```

它不复制完整 ACCEPT/TEST，也不保存运行日志。旧版 registry 可读取；只有
change 被实际修改时才写成 v2，不批量迁移历史 change。

### 已确认执行政策

只有明确启用 Loop 的 change 才有 `loop.json`：

```json
{
  "schema_version": "openspec-loop.v3",
  "change_id": "example-change",
  "contract_fingerprint": "<sha256 of the active task registry>",
  "narrative_digest": "<sha256 of proposal/design/specs>",
  "narrative_policy": "advisory",
  "sealed": true,
  "autonomy": "supervised",
  "retention": "thin",
  "paths": {
    "ledger": "test_cache/example-change/loop-ledger.json",
    "scratch": "test_cache/example-change",
    "product": null,
    "bundle": null,
    "gui_colab": null
  },
  "budgets": {},
  "hard_ceiling": {
    "max_iterations": 60,
    "max_active_minutes": 1080,
    "max_self_extensions": 3
  },
  "test_profiles": {},
  "confirmed_at": "<UTC timestamp>"
}
```

`contract_fingerprint` 只覆盖 active task registry 的义务；`narrative_digest`
单独覆盖 proposal、design 与 specs。两者分离，promotion 才不会让 seal 自我
失效。

`loop.json` 保存权威策略，不保存每次 attempt。动态 attempt 计数、结果
fingerprint 与实际活动时长写入已确认的 ledger。

## 4. 首次 seal（开工盖章）

**seal = 开工盖章**：冻住整份 active 合同义务 + 落盘档位 + 预算/autonomy/ceiling，
不是勾选某个 task。一次盖章覆盖指纹内全部 `[#R…]`（一次性全盖）；`promote` 才勾选。

用户感知上，grilling 的最后一问就是盖章：

1. `openspec validate` + `plan --advisory`；
2. 读取本 `openspec/changes/<change-id>/` 合同，推荐 path profile，按 change-id
   展开路径（合同已写的根优先；需要时提议**新**根名，不照搬其他 change）；
3. 写出 `seal-preview.md`：列出**全部** active refs、双树、政策；在对话里作为
   最后一轮 QA（同意盖章 / 改档 / 不盖章）——不要在同意之后再单独念一遍「请 seal」；
4. 用户接受预览包后，才执行 `seal --confirmed` + `check`（机械落盘）。

后续改 design 时的衔接：

| 场景 | 盖章 |
|---|---|
| 仅改 proposal/design/specs 叙事 | `reseal` 刷新 digest，不问路径 |
| `major-revision` 材料性义务变更 | 差量拷问 → 预览再列全量 refs → 语义再盖章 |
| `additive-extension` 追加任务 | 只拷问新任务 → 预览列旧+新全表 → 再盖章 |
| `succession` | 锁定行为后全表再盖章 |
| promote | 永不盖章 |

推荐 thin + `A_local_thin` 示例：

```powershell
python scripts/generate_openspec_feature_list.py --change-id example-change
openspec validate example-change --strict
# interviewer: seal-preview.md = 最后一轮 grilling；用户同意后再：
python scripts/openspec_loop.py seal example-change --confirmed --retention thin --ledger-path auto_test_openspec/example-change/loop/ledger.json --scratch-root test_cache/example-change
python scripts/openspec_loop.py check example-change
```

相同 fingerprint 已有有效 `loop.json` 时，只显示一行复用摘要，不重复询问
D 盘、缓存或 bundle。若用户要改政策，先重生 `seal-preview` 再确认。

## 同回合执行锁

Loop 借 Monitor 的调度原则是“先做后说”，不借它的审计账本，也不强制双代理。
`gate` 返回 `continue` 后，同回合的下一项实质动作必须是 Apply；一行进度说明
可以先出现，但不能用总结、建议或 handoff 结束回合。默认由 main 完成 Apply 与
Verify，不预留 subagent 槽；只有重型或高风险任务且 sealed `max_subagents` 仍有
余量时，才可选派一个无 verify、promote 或 PASS 权限的 Apply worker。

## 5. 状态转换

```text
pending ──依赖满足──> ready ──apply──> in_progress
                                  │
                         verify ──┼──> passed
                                  ├──> blocked
                                  ├──> deviated
                                  └──> ready (bounded FAIL retry)

blocked/deviated ──unblock──> retry | targeted_probe
                            └> amend_spec | supersede_task | stop_budget

任何非终态 ──预算/断路器──> maxed
旧任务 ──SUPERSEDES──> superseded
```

只有 `ready` 可以进入 apply。`passed | maxed | superseded` 是终态。
`blocked | deviated | in_progress` 必须有显式转换，不能由模型自行“视为完成”。

### R37 → R38

当 R37 已 `MAXED` 且新合同写明 `R38 SUPERSEDES R37`：

- R37 变为 `superseded`，不再被选择；
- R38 按自己的依赖判断是否 `ready`；
- R38 不是把 R37 当成“已满足”的伪 PASS；
- 变更 tasks 后重新生成 registry、由 interviewer 确认变化并 reseal。

## 6. 有限预算与断路器

默认预算：

| 层级 | 默认上限 |
|---|---|
| 每任务 | 2 次 apply；每条 attempt chain 1 次 unblock |
| 每 revision | 8 个执行循环；1 次 explore；2 个独立 subagent；120 分钟实际活动 |
| 每 change | 3 个 revision；20 个总循环；360 分钟实际活动 |

活动时间只累计实际工具/运行时长，不累计等待用户的墙钟时间。新 run id、新
session 或 spec revision 都不会清空 change 总预算。零 attempt 的 episode 不
消耗 `change.max_revisions`。

以下任一情况触发 breaker：

- 两次相同结果 fingerprint；
- 两次连续 no-progress；
- 两次重复 semantic deviation；
- 任一任务、revision 或 change 上限耗尽。

### 不可自升的 hard ceiling

上面这些上限都可以扩容，所以它们不能是最终约束。`loop.json` 的
`hard_ceiling` 是唯一任何 Loop 都无权提高的停止条件：

| 字段 | 含义 |
|---|---|
| `max_iterations` | change 级 apply 迭代的绝对上限 |
| `max_active_minutes` | change 级实际活动时间的绝对上限 |
| `max_self_extensions` | 允许的自扩容次数 |

`seal` 设定 ceiling 且必须 `--confirmed`；未显式给出时按已确认的 change 预算
留出余量自动推导。`reseal` 逐字继承 ceiling，拒绝任何超过它的 `--set-max-*`，
并在自扩容次数用尽后拒绝继续扩容。`gate` 会把 `--max-iterations` 之类的覆盖
夹到 ceiling 以内，命中 ceiling 的停止以 `terminal: true` 报告——此时停止并
汇报，不再寻求预算。

### 授权模型

`autonomy` 决定谁能在 ceiling 之下移动边界：

| 动作 | `supervised`（默认） | `full_auto` |
|---|---|---|
| 扩容 change 预算 | 用户明确授权后 `reseal --set-max-* --confirmed --reason` | `reseal --set-max-* --reason`，写入 `budget_extensions` |
| design 级缺口后修订 tasks | 暂停并交回 `openspec-change-interviewer` | 自行修订并 `reseal --allow-semantic-change --reason` |
| 提高 `hard_ceiling`、改动 retention、路径或范围 | 用户确认的 `seal` | 用户确认的 `seal` |

autonomy 不扩大仓库权限。无论哪种模式，破坏性外部写入、凭据、产品 `--commit`
以及 `git push` / PR / `main` 操作都依 `AGENTS.md` 保持人工授权；`full_auto`
的 Loop 需要其中任一项时以 `BLOCKED` 停止。

## 7. 存储矩阵

| retention | ledger/scratch | promotion/deviation 记录 | 每次完整 bundle | 适用场景 |
|---|---|---|---|---|
| `none` | 可丢弃 ledger | 不保留 | 否 | 小型、可快速重建 |
| `thin`（推荐） | 忽略目录下小型 ledger/scratch | 只保留 promotion receipt 或改变方向的 deviation | 否 | 普通工程与研究 Loop |
| `full` | 明确路径 | 保留 | 是 | 法规、外部审计、用户明确要求 |

四种相似路径不能混同：

1. tracked 决策报告：`openspec/changes/<id>/unblock/*.json|md`；
2. 明确产品或审计根下的重型产物；
3. scratch 下 `pytest/cache/tmp/<ref>/<run-id>/`；
4. ledger 的 `attempts[]`，只含计数、fingerprint、时长与 disposition。

普通尝试默认 `return_only`。只有方向改变、长期 blocker 或明确审计要求才
把 unblock 报告写入 change。

## 8. 测试矩阵

| 层级 | 触发时机 | 范围 | 是否生成正式 bundle |
|---|---|---|---|
| Attempt | 每次局部改动后 | touched module 的 owner tests | 否 |
| Promotion | 首次准备把任务标为 PASS | 任务完整 `TEST:` | 否，除非任务明确要求 |
| Final | change 全部任务结束 | whole-change verify、strict validate、相邻回归 | 否 |
| Audit | full retention 或显式 audit | 不可变 bundle、完整证据链 | 是 |

测试命令通过不等于产品正确。verifier 必须比较预期与实际的来源、目标、年份、
指标、分辨率、schema 和研究方向。

## 9. Hard error 与 semantic deviation

- **Hard error**：命令不能完成，例如 import error、权限失败、缺失凭据。
- **Semantic deviation**：命令 exit 0，甚至现有测试通过，但产物使用了错误
  表、年份、数据源、指标、目标或解释。

例如提取器成功输出 28 个单元，但其中 14 个来自邻近表。此时 verdict 是
`DEVIATED`，不能 PASS，也不应只重跑同一命令。unblock v2 必须写清：

- expected vs observed；
- acceptance/evidence refs；
- last-good baseline；
- 带正反证据的 hypotheses；
- 一个能区分主要假设的 probe；
- `retry | targeted_probe | amend_spec | supersede_task | stop_budget`。

Loop-light unblock 上限为 4 次工具调用、4 条证据、180 秒，只给一个主方案
和一个 fallback。

## 10. 恢复、变更与结束

### 恢复执行

```powershell
python scripts/openspec_loop.py check example-change
python scripts/openspec_loop.py plan example-change
python scripts/openspec_loop.py summary example-change --run-id <run-id>
```

恢复时复用已确认路径与累计预算。若 fingerprint 漂移，停止，不“顺便修复”
合同。

### 修改 spec

1. verifier/unblock 返回 `amend_spec` 或 `supersede_task`；
2. Loop 暂停；
3. `supervised` 由 interviewer 只确认变化字段；`full_auto` 由 supervisor 记录
   `--reason` 后直接修订，范围、retention、路径与 `hard_ceiling` 仍需人工确认；
4. 更新当前 proposal/design/specs/tasks，不追加完整旧稿；
5. 重建 feature registry、strict validate、seal 新 revision；
6. 用原 ledger 恢复，change 总预算继续累计。

### 结束

没有 ready task **不等于** design 已经实现。收尾分两层：

```powershell
python scripts/openspec_loop.py goals <change-id>
python scripts/openspec_loop.py design-verify <change-id> --observation <path>
```

- **任务层**：task checkbox、compact state 与 promotion 后的 `check` 三者一致，
  才算该任务完成。晋升用 `promote` 一次原子完成，不手改 checkbox；索引单独漂移
  用 `sync` 修复。
- **design 层**：`GOAL:` 块声明设计义务与 `COVERED_BY:` 覆盖关系。`goals` 报出
  未被覆盖的目标与不服务任何目标的 ref；`design-verify` 对每个目标比较 `ACCEPT:`
  与实际观察，返回 `PASS` 或 `GAP`。没有提供观察即 `unobserved`，属于 `GAP`，
  不得当作 `PASS`——工具只判断结构覆盖，永不推断产品语义。

`GAP` 返回一份 `openspec-loop-revision-proposal.v1`，写明目标、预期与实际、证据
以及建议新增或 supersede 的任务，而不是直接停下。`supervised` 交回 interviewer；
`full_auto` 用 `apply-revision` 自行应用，但提案中的 `add` 必须自带可执行
`TEST:`，否则拒绝——autonomy 不得制造空合同。应用后回到排空循环。

只有 design 层 `PASS` 之后，才运行 whole-change verify 与 strict validation 并
声称完成。

## 11. 禁止模式

- 每轮重复询问 D 盘、缓存、bundle 或 GUI 路径；
- 把日志、trace、完整旧 spec 或每次研究过程堆进 OpenSpec；
- 每次 attempt 都生成完整 bundle；
- 新 revision、run id 或新 session 静默重置总预算；
- 在 `supervised` 下不带 `--confirmed` 扩容，或在任何模式下不带 `--reason`
  扩容；
- 试图提高 `hard_ceiling`，或在命中 ceiling 后继续寻求预算；
- 把 `full_auto` 当成破坏性外部写入、凭据或 `main` 分支操作的授权；
- 用 exit code 0 代替产品语义验收；
- 从历史 appendix 重新选择已废弃任务；
- 让 apply 自己勾选任务或写 PASS；
- 让普通 Loop 自动升级为 legacy monitor；
- 在合同漂移后继续沿用旧 seal；
- 仅因路径都含 `attemptNNN` 就把 scratch、产品、报告和 ledger 当成同一物。

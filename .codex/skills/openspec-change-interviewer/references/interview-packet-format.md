# Interview packet, context, and ADR formats

Use these shapes under `openspec/changes/<change-id>/` unless noted. Omit empty
optional sections. Never expand into a generic coverage form.

## Decision state table

| State | Meaning |
|---|---|
| `proposed` | Candidate under consideration |
| `direction-recorded` | User preference stated; not yet accepted |
| `awaiting-confirmation` | Conditional acceptance pending explicit confirm |
| `accepted` | Decision accepted into the operative contract |
| `rejected` | Explicitly refused |
| `superseded` | Replaced by a later accepted decision |

`accepted` is not implementation authorization. Authorization occurs only at
`seal`.

## `interview.md` (seven sections)

```markdown
## 1. 资料来源与证据
| Source | Inspected | Observed facts | Confidence |

## 2. 当前决策上下文
- Entry mode: new-idea | major-revision | additive-extension | succession | execution-ready
- Change id, sealed fingerprint status, open goals

## 3. 材料性决策与状态
| Decision | Why it changes scope/design/tasks/acceptance | State | Evidence path:line | Related Q | Write-back target |

## 4. 需要你填写的问题
| ID | Question | Why | Impact | Recommended default + evidence | Trade-off | Your answer | Write-back |

## 5. 填写说明
- "同意默认" / "本次不做" / "不确定" are all valid answers
- Permission and destructive questions are never mixed into a batch

## 6. 后续写回映射
- Gate before write-back: only `confirmed` answers become facts

## 7. 答案吸收记录
| ID | Classification | Written fact | Artifact |
```

## `context.md` (terminology only)

```markdown
## Language

**Term Name**: compact definition.
_Avoid_: rejected synonyms.
```

Do not store implementation decisions, drafts, generic programming terms, or
architecture rationale here.

## ADR format

Write an ADR only when all three gates hold:

1. hard to reverse
2. surprising without context
3. the result of a real trade-off

Prefer durable paths `docs/decisions/NNNN-<slug>.md` (sequential). A
change-local `adr.md` may hold `Proposed` drafts until accepted. Lifecycle:
`PROPOSED → ACCEPTED → (SUPERSEDED | DEPRECATED)`. Never delete old ADRs;
supersede by writing a new one that references the old.

Minimal body:

```markdown
# ADR NNNN: Title

Status: Proposed

## Decision

## Why

## Options Considered

## Consequences
```

Required before `Accepted`: decision owner, chosen option, rejected
alternatives, and reversal cost. Missing any of those keeps Status Proposed.

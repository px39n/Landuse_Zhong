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

`accepted` makes a decision part of the operative contract. Ordinary Apply is
eligible when the active registry has a matching `contract_fingerprint` and no
irreversible policy change is awaiting confirmation. Optional preview/stamp
fields are audit diagnostics, not a start-work gate and not broader authority.

## `interview.md` (seven sections)

For Loop R2 or task succession, create or refresh this file before asking the
user a question. Prefill current fingerprint, active ACCEPT, expected/observed,
retention, paths, budgets, and autonomy. Preserve accepted answers and Section
7 history; refresh only stale context. Missing policy defaults to thin +
A_local_thin, Apply 2, Unblock 2, cycle stamps 3, and supervised. Existing
change-local or B_external_heavy paths are inherited and not re-asked.

```markdown
## 1. 资料来源与证据
| Source | Inspected | Observed facts | Confidence |
| Compact evidence anchor (`path:line` or symbol) | What it supports | Unknowns |

## 2. 当前决策上下文
- Entry mode: new-idea | major-revision | additive-extension | succession | execution-ready
- Change id, contract fingerprint status, open goals
- Active task / acceptance boundary
- Target files and write scopes
- Governing rules, paths, and retention decision
- Nearest precedent and verification command
- Unknowns, assumptions, and conflicting sources

## 3. 材料性决策与状态
| Decision | Why it changes scope/design/tasks/acceptance | State | Evidence path:line | Related Q | Write-back target |
| Assumption / loophole category | Measurable outcome | Falsifier or out-of-scope reason | State | ACCEPT/TEST owner |

## 4. 需要你填写的问题
| ID | Question | Why | Impact | Recommended default + evidence | Trade-off | Your answer | Write-back |

## 5. 填写说明
- "同意默认" / "本次不做" / "不确定" are all valid answers
- Permission and destructive questions are never mixed into a batch

## 6. 后续写回与可选协调映射
- Gate before write-back: only `confirmed` answers become facts
- Proactive scan: scan_id, matched_role_id|null, direct|dispatch|blocked, direct_reason, effective_role_id
- Packet/result/join rows: single handoff.json only; see the compact envelopes below

## 7. 答案吸收记录
| ID | Classification | Written fact | Artifact |
```

## Optional `handoff.json` coordination rows

Retain coordination only when it materially helps execution. All rows live in
exactly one `openspec/changes/<change-id>/handoff.json`; do not create
per-worker handoff documents, `handoffs/`, a second ledger, or another process
lifecycle.

```text
scan:
  scan_id, intent, matched_role_id|null, decision=direct|dispatch|blocked,
  direct_reason, effective_role_id, evidence_anchors

dispatches[]:
  package_id, logical_agent=agents/<agent-id>, role_id, local_entry, ref,
  attempt, bounded_assignment, authority, scope, forbidden_scope, write_scope,
  expected_result, expected_evidence, execution=sync|async, join_id,
  continuation, stop_conditions

results[]:
  package_id, status=completed|failed|partial|blocked|unverified,
  inspected_scope, evidence, checks, skipped_checks, blockers, risks,
  unverified, completed_at_utc

joins[]:
  join_id, package_ids, terminal_results_complete, per_ref_verify_eligibility
```

These are portable advisory records. They narrow assignment authority and
cannot give a worker ownership of Verify, promote, PASS, retry dispatch, or
scope expansion. They create no slash command, formal Board, progress log,
retained bundle, or implementation verdict.

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

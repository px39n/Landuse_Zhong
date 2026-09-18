---
name: openspec-unblock-research
description: Diagnose one blocked or DEVIATED OpenSpec ref and return a disposition.
  Do not implement the repair or auto-redispatch.
license: MIT
metadata:
  author: openspec
  version: "3.0"
---

# openspec-unblock-research

In-process diagnosis for one ref. Spawn zero by default.

## When

`next_action=unblock` after `blocked|deviated|failed|empty`.

## Method

1. Compare expected ACCEPT vs observed result/evidence.
2. Return one disposition: `retry | targeted_probe | amend_spec |
   supersede_task | stop_budget`.
3. Do not implement the fix, expand WRITE_SCOPE, reset budgets, or promote.
4. Second unblock on the same fingerprint requires new discriminating evidence;
   otherwise prefer `amend_spec` or `stop_budget`.

## Typed result

```json
{
  "status": "completed",
  "disposition": "retry",
  "reasons": ["transient tool failure"],
  "evidence_refs": [],
  "blockers": [],
  "risks": [],
  "skipped_checks": []
}
```

Supervisor records via `record --kind unblock` when required, then runs `next`.

---
name: openspec-apply-change
description: Implement one receipt-bound OpenSpec task and return a typed result
  file. Leave verification, promotion, and ledger writes to the controller.
license: MIT
metadata:
  author: openspec
  version: "3.0"
---

# openspec-apply-change

Implement exactly one active ref from an authoritative `next` receipt.

## Preflight

1. Require `receipt_id` + packet from `next` (`next_action=apply`).
2. Read bound ACCEPT, TEST, WRITE_SCOPE, packet `goals` (GOAL chain to root),
   `design_anchors`, and relevant source/tests. Goals are authoritative context
   for ACCEPT; do not invent a parallel goal story.
3. Stop on ambiguous scope, missing authority, or needed contract change;
   return to interviewer. Never toggle tasks, write the ledger, Verify, or
   promote.

## Implementation

- Edit only WRITE_SCOPE; preserve unrelated dirty files.
- Run owner checks from the task; full TEST belongs to promotion.
- Wrong direction after exit 0 → `DEVIATED`.
- Product/external writes, credentials, destructive actions, and Git push/PR
  keep human gates.

## Typed result file

```json
{
  "status": "completed",
  "changed_files": ["path"],
  "evidence_refs": ["path"],
  "blockers": [],
  "risks": [],
  "skipped_checks": []
}
```

Statuses: `completed|blocked|deviated|failed|empty`. Empty/missing fields are
`empty`, never PASS. Supervisor submits via
`record --kind apply --receipt <id> --result <file>`.

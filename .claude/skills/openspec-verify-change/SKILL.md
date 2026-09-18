---
name: openspec-verify-change
description: Independently judge one OpenSpec task against ACCEPT after mechanical
  evidence. Never implement, promote, or share the author context.
license: MIT
metadata:
  author: openspec
  version: "3.0"
---

# openspec-verify-change

Independent semantic verification only. Use a fresh context; do not inherit the
author's chat or treat unverified claims as evidence. Inspect the actual changed
source and retained outputs independently.

## When

`next_action=verify_semantic` after mechanical evidence exists for the ref.

## Method

1. Start with ACCEPT/TEST and mechanical evidence pointers; inspect the referenced
   source, diffs and outputs needed to judge every acceptance claim.
2. Compare source, target, metric, schema, and research direction.
3. Exit 0 alone is insufficient; wrong direction → `DEVIATED`.
4. Never edit product code, write the ledger, or promote.

## Typed result file

```json
{
  "verdict": "PASS",
  "reasons": ["matches ACCEPT"],
  "evidence_refs": ["path"],
  "duration_seconds": 1.0
}
```

Verdicts: `PASS|FAIL|BLOCKED|DEVIATED`. Submit through
`record --kind verify --receipt <id> --result <file>` when a receipt is bound,
or return the file to the supervisor for recording.

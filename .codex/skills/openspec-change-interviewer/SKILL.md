---
name: openspec-change-interviewer
description: Author or revise OpenSpec proposal, design, specs, and tasks. Lock
  retention, paths, and acceptance. Never award runtime PASS.
license: MIT
metadata:
  author: openspec
  version: "3.0"
---

# openspec-change-interviewer

Author the operative contract. Ask the minimum questions that change outcome.

## Required locks

- Artifact Retention Decision (none / thin / full) and roots
- WRITE_SCOPE / FILES / ACCEPT / TEST per task
- Dependencies and mandatory goals
- GOAL tree ↔ design anchors ↔ tasks alignment (check affected relations)

Reuse confirmed locks. Ask only for unresolved material fields and wait before
their dependent writes. Silence is not approval. Do not invent `auto_test_openspec/` paths.

## Alignment practice

On every interview / major revision:

1. Check affected GOAL blocks (`GOAL_KIND` / `PARENT_GOAL` / `COVERED_BY` / `ACCEPT`)
   against design anchors and task `GOAL_IDS` / ACCEPT; expand to predecessors
   and mandatory coverage when those relations change.
2. Keep skip metadata explicit: `STATE: superseded|interrupted`, `SUPERSEDES:`,
   `PRODUCTION_HOLD` / ACCEPT phrases (`SHALL NOT execute`, `do not apply`).
3. Record the alignment conclusion in `interview.md`. Controller
   `alignment_warnings` must be cleared or consciously accepted before drain.

## After edits

```powershell
python -B scripts/generate_openspec_feature_list.py --change-id <change>
openspec validate <change> --strict
```

Source authoring never grants ledger, receipt, or PASS authority. Hand off to
`openspec-loop-engineering` / `next` for execution.

---
name: openspec-new-change
description: Start a new OpenSpec change and capture its objective, boundaries and first artifact. Does not execute Loop.
license: MIT
metadata:
  author: openspec
  version: "2.0"
---

# OpenSpec New Change

Create a new change when the user wants a durable contract rather than only
exploration.

1. Resolve a unique kebab-case change id. Ask only if naming or scope has more
   than one materially different interpretation.
2. Inspect `openspec list --json` and the minimum project context needed to
   avoid a duplicate or conflicting change.
3. Create the change with the repository's active OpenSpec schema.
4. Draft the first required artifact from the user's concrete objective,
   boundaries, and known acceptance evidence. Do not manufacture unknown paths,
   data sources, or budgets.
5. Report the created artifact and next unresolved decision.

This skill authors artifacts only. Use `$openspec-change-interviewer <change-id>`
for missing or changed material decisions; reuse existing confirmations.
When execution is requested, hand off to `openspec-loop-engineering`; only its
controller establishes admission and runtime readiness.

Keep only current contract text in OpenSpec. Git history, not an accumulating
appendix, preserves replaced drafts.

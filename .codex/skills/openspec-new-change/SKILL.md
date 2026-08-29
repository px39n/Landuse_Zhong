---
name: openspec-new-change
description: Start a new OpenSpec change and capture the initial problem boundary. Route execution-ready changes through the interviewer before Loop sealing.
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

This skill does not seal or execute Loop. When the package becomes
implementation-ready, invoke `$openspec-change-interviewer <change-id>` in
entry mode `new-idea` (or `succession` when upgrading a historical/main
capability) to grill material boundaries and seal `loop.json`.

Keep only current contract text in OpenSpec. Git history, not an accumulating
appendix, preserves replaced drafts.

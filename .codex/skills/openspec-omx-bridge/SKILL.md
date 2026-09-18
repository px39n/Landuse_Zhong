---
name: openspec-omx-bridge
description: Translate useful OMX-style exploration or planning conclusions into the current OpenSpec contract without duplicating protocols or preserving superseded drafts in appendices.
license: MIT
metadata:
  author: openspec
  version: "2.0"
---

# OpenSpec OMX Bridge

Use this bridge when exploratory or planning work contains decisions that need
to become executable OpenSpec truth.

## Boundary

- OMX/exploration provides questions, options, risks, and execution insights.
- OpenSpec proposal/design/specs/tasks hold the accepted current contract.
- `feature_list.json` is the derived task registry; it grants no runtime authority.
- `loop.json` holds execution configuration; source edits do not grant admission.
- Git preserves superseded contract versions.

The bridge does not copy an entire interview transcript, plan, or historical
draft into OpenSpec.

## Procedure

1. Identify the exact accepted conclusions and unresolved decisions.
2. Index the current change by filenames, headings, sizes, and active task refs.
3. Map each accepted conclusion to the smallest current artifact section.
4. Reuse explicit accepted decisions. Ask only for unresolved material changes
   to scope, acceptance, dependencies, retention, paths, or budgets.
5. Apply only current-state edits. Replace superseded wording instead of adding
   a permanent appendix.
6. Keep `tasks.md` directives explicit and regenerate the compact feature list
   when active tasks change.
7. Run strict OpenSpec validation.

Contract edits can change the fingerprint. Coordinate affected active claims
through the controller before editing; use `openspec-change-interviewer` for
unresolved changed fields. Preserve consumed budgets and historical evidence.
Requested execution returns to `openspec-loop-engineering` for admission.

Use `$monitor-openspec-codex` only when the user explicitly requests legacy
retained audit bookkeeping; it is not the normal bridge destination.

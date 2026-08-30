---
name: openspec-continue-change
description: Continue an existing OpenSpec artifact workflow by creating the next required artifact, without executing implementation or claiming Loop readiness before the registry fingerprint matches.
license: MIT
metadata:
  author: openspec
  version: "2.0"
---

# OpenSpec Continue Change

Continue artifact authoring for one existing change.

1. Resolve the change id without guessing among multiple active changes.
2. Run `openspec status --change "<change-id>" --json`.
3. Identify the next required artifact and load only its declared dependencies
   plus the relevant current sections. Do not read every historical spec.
4. Create or update that artifact according to OpenSpec instructions.
5. Run strict validation when the package has enough artifacts to validate.
6. Report what became ready and the next artifact or decision.

Do not implement product code, run the Loop, or create retained audit evidence.
After the active tasks are complete, regenerate `feature_list.json`, run strict
validation, then use `check`/`plan` to establish a matching
`contract_fingerprint`. Missing `loop.json` may initialize recorded thin
defaults without creating any ledger, scratch, cache, product, or retained
root.

Narrative-only drift is refreshed with `reseal <change-id>`. If this update
changes the active task registry semantically, pause affected dispatch:
`supervised` returns to `$openspec-change-interviewer <change-id>` for the
material decision; `full_auto` lets the sole supervisor restamp with
`--allow-semantic-change --reason`. A new revision or optional stamp never
silently resets change-level or ref-local budgets.

Artifact completion alone does not authorize an irreversible retention, path,
scope, credential, destructive-write, product `--commit`, or Git policy change.
Ordinary execution needs a matching fingerprint and no such policy pending; it
does not need a seal-first ceremony.

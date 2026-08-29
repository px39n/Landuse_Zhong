---
name: openspec-continue-change
description: Continue an existing OpenSpec artifact workflow by creating the next required artifact, without executing implementation or bypassing Loop sealing.
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
If this update changes a previously sealed fingerprint, the active episode is
paused; return to `$openspec-change-interviewer <change-id>` for delta-only
confirmation and a new seal. A new revision never silently resets change-level
budgets.

Once proposal, design/specs, and active tasks are complete, the interviewer is
the required gateway to Loop. Do not treat artifact completion as consent to
choose retention paths or execute.

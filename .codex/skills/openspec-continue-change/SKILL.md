---
name: openspec-continue-change
description: Create the next required artifact for an existing OpenSpec change. Does not execute implementation or grant runtime readiness.
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
After authoring the active task definitions, regenerate `feature_list.json`
and run strict validation. Reuse confirmed retention and scope. Route unresolved
material decisions to `openspec-change-interviewer`. Before changing an active
contract, coordinate affected claims through the controller.

When execution is requested, hand off to `openspec-loop-engineering`; its
controller owns admission, fingerprint checks, budgets and runtime state.
Source revisions never reset consumed usage or grant execution authority.

Artifact completion alone does not authorize an irreversible retention, path,
scope, credential, destructive-write, product `--commit`, or Git policy change.
Report artifact readiness separately from controller-granted execution readiness.

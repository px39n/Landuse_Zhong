---
name: openspec-ff-change
description: Draft the required OpenSpec artifacts in one pass from a concrete objective. Does not execute Loop.
license: MIT
metadata:
  author: openspec
  version: "2.0"
---

# OpenSpec Fast Forward

Create the proposal, necessary design/spec deltas, and active task registry in
one bounded authoring pass when the user's intent is already concrete.

1. Resolve or create the kebab-case change id.
2. Inspect only relevant repository contracts and current OpenSpec indexes.
3. Generate artifacts in schema dependency order.
4. Give every active checkbox a unique `[#R<n>]`, explicit `ACCEPT:`,
   `TEST:`, and `DEPENDS_ON:` or `NO_DEP:`.
5. Generate the compact registry with
   `python scripts/generate_openspec_feature_list.py --change-id <change-id>`.
6. Run `openspec validate <change-id> --strict`.

Reuse confirmed boundaries and ask only for unresolved material choices.
Do not invent retention, external paths, concurrency, or budget choices. Draft ACCEPT/TEST against
`.agents/skills/openspec-change-interviewer/references/acceptance-quality.md`.
Use `$openspec-change-interviewer <change-id>` for missing or changed material
fields. Report artifact readiness; when execution is requested, hand off to
`openspec-loop-engineering` for controller-owned admission.

Do not execute implementation or create audit bundles from this skill.

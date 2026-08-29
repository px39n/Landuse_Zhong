---
name: openspec-ff-change
description: Fast-forward OpenSpec artifact creation to an implementation-ready draft, then require interviewer confirmation before Loop execution.
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

Fast-forward reduces authoring round trips; it does not bypass human boundary
confirmation. Do not invent retention, external paths, concurrency, or budget
choices. Draft ACCEPT/TEST against
`.agents/skills/openspec-change-interviewer/references/acceptance-quality.md`.
Finish by handing off to `$openspec-change-interviewer <change-id>` with the
matching entry mode (`new-idea`, `succession`, or `additive-extension`), which
grills remaining material fields and seals `loop.json`.

Do not execute implementation or create audit bundles from this skill.

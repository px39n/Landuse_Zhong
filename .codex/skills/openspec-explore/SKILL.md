---
name: openspec-explore
description: Explore an idea, code path, or OpenSpec uncertainty without implementing. Supports free discovery and a bounded Loop probe that returns one decision-relevant result.
license: MIT
metadata:
  author: openspec
  version: "2.0"
---

# OpenSpec Explore

Explore is read-only thinking and investigation. It may recommend an artifact
change, but it does not implement product code or silently rewrite a contract.

## Modes

### Free discovery

Use when the problem or design is still open. Follow useful questions, compare
options, inspect the repository, and make uncertainty explicit. No fixed output
or storage is required. When the exploration is literature- or method-facing,
structure findings with
`.agents/skills/openspec-change-interviewer/references/evidence-matrix.md`.
At the end of a useful free discovery, you may leave a light idea capsule
(problem, options, open questions, suggested interviewer entry mode) in chat
or, if the user asks, under the change as a short note — not a seal.

### Loop bounded probe

Use only when a sealed Loop revision has one decision-blocking uncertainty.
The default revision budget permits one Explore episode.

1. State the exact question and which task/ACCEPT clause it affects.
2. Read an index first: relevant filenames, headings, sizes, active task, and
   current fingerprint.
3. Inspect only the evidence needed to answer that question.
4. Stop after one discriminating finding or when the budget cannot answer it.
5. Return `continue | targeted_probe | amend_spec | stop_budget` with evidence.

Do not use Explore as an unbounded prelude to every apply attempt. Do not store
transcripts or large research bundles under the change. In `thin` mode, keep
only a compact decision note when it changes execution direction.

## Contract handoff

- If the finding stays inside the sealed contract, return to Loop.
- If scope, acceptance, design, dependencies, retention, or paths must change,
  pause Loop and hand off to `$openspec-change-interviewer <change-id>` with one
  of `new-idea | major-revision | additive-extension | succession`.
- If a concrete observed result conflicts with ACCEPT, use
  `$openspec-unblock-research` rather than continuing general exploration.

Do not edit product code in either mode. Write OpenSpec artifacts only when the
user explicitly asks to capture the conclusions. Follow
`.agents/skills/openspec-change-interviewer/references/context-discipline.md`
for Search Evidence Packs during broad search.

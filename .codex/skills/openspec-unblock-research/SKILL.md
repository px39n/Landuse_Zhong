---
name: openspec-unblock-research
description: Diagnose a blocked or DEVIATED OpenSpec task from expected-vs-observed evidence and return one bounded disposition without implementing the fix.
---

# OpenSpec Unblock Research

Determine why the task cannot safely continue and which one decision or probe
best separates the remaining hypotheses. This is diagnosis, not another Apply
loop or open-ended literature review.

## Inputs

Normalize only decision-relevant evidence:

- change/ref/attempt and trigger type
- expected outcome from current ACCEPT/TEST
- observed outcome, including successful-but-wrong output
- acceptance/evidence pointers and last good baseline
- actions already tried and their results
- environment or error excerpt only when explanatory
- caller-authorized sink

Ask once only when missing information would change the diagnosis. A stack
trace is optional when the failure is semantic.

## Diagnostic workflow

1. Verify the active contract and actual artifact being judged.
2. Separate facts, inferences, and unknowns; cite paths, hashes, commands,
   metrics, or screenshots.
3. Classify implementation, evidence, direction, contract, or exhausted-
   approach failure.
4. Form the smallest competing hypotheses and grade evidence using
   `.agents/skills/openspec-change-interviewer/references/evidence-matrix.md`.
5. Prefer local evidence; research external authority only for a remaining
   method, API, version, or domain uncertainty.
6. Stop after one discriminating probe, one primary path, and one fallback.

Do not average conflicting evidence into confidence. Target the conflict with
the probe. High confidence needs unambiguous primary evidence or independent
convergence; otherwise preserve the uncertainty.

## Typed output

Return `portable-unblock-report.v2` with facts/unknowns, expected vs observed,
hypotheses and graded evidence, one probe, conclusions, primary/fallback paths,
and exactly one disposition:

`retry|targeted_probe|amend_spec|supersede_task|stop_budget`

The optional repair class is advisory. Keep the short `repair_r1` token in the
report when applicable, but never infer contract-write or dispatch authority
from it. Read `references/portable-unblock-report.v2.md` for schema details and
`references/examples.md` only for hard-error versus semantic-drift examples.

## Loop boundary

When called by Loop, read `references/loop-disposition-gate.md` for ref-local
budgets, R0/R1/R2, second-Unblock rules, sinks, and routing. The supervisor
invokes this skill in-process; do not implement, Verify, promote, resume a
failed task/thread, swap workers, expand WRITE_SCOPE, or start a research swarm.

Product/external writes, credentials, destructive actions, retention/path or
budget changes, and Git push/PR/main remain human-authorized. If tools or data
are unavailable, return an exact collection plan plus `targeted_probe` or
`stop_budget`; do not guess through the gap.

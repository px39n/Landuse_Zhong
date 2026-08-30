---
name: silent-failure-hunting
description: Bounded false-success review for the local OpenSpec Loop. Use when a ref or worker result may look successful while required work, evidence, or failure state was skipped or hidden; read-only and non-delegating.
---

# Silent Failure Hunting

Use this skill for exactly one bounded false-success question inside the local OpenSpec Loop.

## Trigger

- A ref, worker handoff, verifier input, or workflow gate may look successful while required work, evidence, or a failure state was skipped, hidden, stale, partial, blocked, or still unverified.
- The caller needs one read-only false-success review, not broad code review, Apply, or final Verify.

## Caller and role

- Canonical caller: `openspec-loop-engineering`
- Role ID: `silent-failure-reviewer`
- Invocation shape: one fresh bounded review attached to one current ref or one current join point.
- This skill is self-contained and local to the repository Loop.

## The one question

Ask exactly one bounded false-success question and answer only that question:

> What would make this look successful even though required work, evidence, or a negative outcome was skipped, hidden, stale, partial, blocked, or still unverified?

Bound the answer to the current ref, current packet/result, and currently inspected evidence only.

## Required method

1. Read only the minimum evidence the caller already scoped.
2. Do not delegate, redispatch, or reopen planning.
3. Separate observed evidence from inference.
4. Name one concrete false-success mechanism if found; otherwise say none found in the bounded evidence.
5. Return the minimum negative-test need that would distinguish true success from false success.

## Output

Return a compact note with exactly these fields:

- `question`
- `finding`
- `evidence`
- `limitations`
- `negative_test_need`

`evidence` must stay portable and concrete. `limitations` must name what was not checked. `negative_test_need` must name one bounded test or observation still needed, or `none` if the bounded evidence is already discriminating.

## Guardrails

- Read-only.
- Non-delegating.
- Do not Apply.
- Do not run final Verify.
- Do not `seal`, `promote`, or mark PASS.
- Do not retry dispatch, swap workers, or own lifecycle control.
- Do not claim product correctness; report only bounded false-success risk.
- Stop after one terminal review.

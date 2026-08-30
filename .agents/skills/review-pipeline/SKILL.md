---
name: review-pipeline
description: Optional bounded specialist review routing for one unresolved named risk inside the OpenSpec Loop. Use only from openspec-loop-engineering after direct inspection and the smallest relevant check leave one specialist question open.
---

# Review Pipeline

This skill is a bounded helper for `$openspec-loop-engineering <change-id>`.
It may collect one extra specialist review signal for one named risk.
It never becomes a second lifecycle, reviewer swarm, or acceptance owner.

## Trigger

Use only when both hold:

- the caller is `$openspec-loop-engineering <change-id>`; and
- either the user explicitly asks for review of the current loop-owned work, or
  one named specialist risk remains unresolved after direct inspection and the
  smallest relevant check.

If direct inspection already answers the question, stop and report directly.
Do not use this skill for broad confidence seeking, open-ended quality sweeps,
or as a substitute for ordinary owner tests.

## Authority boundary

`openspec-loop-engineering` remains the only owner of apply sequencing, budgets,
Verify, final verdict, promote, reseal, and seal.

This skill must not call:

- `$openspec-apply-change`
- `$openspec-verify-change`
- `$openspec-change-interviewer`
- `$monitor-openspec-codex`

Do not create extra ledgers, review bookkeeping files, or a second handoff
lifecycle. Do not claim PASS, acceptance, promotion, or integration.

## Method

1. Name one review question and the exact risk it would reduce.
2. State the already-inspected files or diff and the smallest relevant check
   that proved insufficient.
3. Choose at most one auxiliary canonical Role ID whose specialization matches
   that risk. If no such role is needed, stop and return direct review notes.
4. Send one compact packet containing:
   - `Role ID`
   - `Ref`
   - `Question`
   - `Scope`
   - `Write scope: none`
   - `Forbidden scope`
   - `Evidence anchors`
   - `Join: immediate`
5. Inspect one result envelope containing:
   - `status: completed | failed | partial | blocked | unverified`
   - `findings`
   - `portable evidence`
   - `remaining Unverified`
6. If the finding is concrete and the main flow has already applied the minimal
   fix, run at most one targeted recheck for the same question.
7. Return one bounded outcome: answered risk, blocker, or remaining
   `Unverified` risk.

## Limits

- Default is no dispatch.
- At most one auxiliary canonical Role ID and one joined result.
- No automatic retry, redispatch, role swap, or review fan-out.
- No async swarm, no parallel review lanes, and no second pass once the
  targeted recheck is consumed.
- A targeted recheck is not full TEST and does not replace task `TEST:` or
  whole-change verification.
- This skill does not issue `PASS`, acceptance, promotion, or integration.

## Result

Report:

- reviewed scope
- named question
- specialist used or skipped
- evidence reviewed
- blocking findings
- whether one targeted recheck ran
- remaining `Unverified` risk

`openspec-loop-engineering` decides the next step.

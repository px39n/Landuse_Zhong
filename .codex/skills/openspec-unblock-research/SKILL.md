---
name: openspec-unblock-research
description: Diagnose a blocked or directionally deviated OpenSpec task using expected-vs-observed evidence, bounded research, and a discriminating probe. Use for hard errors, wrong outputs, metric regression, wrong source or target, evidence gaps, repeated no-progress, or contract mismatch; return a v2 disposition without assuming every failure throws an error.
---

# OpenSpec Unblock Research

This is a supervisor diagnosis skill. It determines why a task cannot safely
continue and what single next decision or experiment will discriminate among
the remaining hypotheses. It does not perform an open-ended implementation
loop.

## Inputs

Normalize the caller's evidence into:

- `trigger_type`: `hard_error|output_deviation|quality_regression|wrong_source_or_target|evidence_gap|no_progress|contract_mismatch`
- task/change id and attempt id when available
- `expected_outcome` from current ACCEPT/TEST
- `observed_outcome`, including successful-but-wrong outputs
- `acceptance_refs` and `evidence_refs`
- `last_good_baseline`, or `null`
- actions already tried and their outcomes
- environment details only when they can explain the deviation
- `error_excerpt`, optional and `null` when no error occurred
- caller-controlled sinks

Ask once only for missing information that would change the diagnosis. Do not
require a stack trace when the observable failure is semantic.

## Output contract

Emit `portable-unblock-report.v2` by default. Read
`references/portable-unblock-report.v2.md` for the canonical shape and
`references/examples.md` when distinguishing a hard error from semantic output
drift. `references/portable-unblock-report.v1.md` remains read-only compatibility
for older reports.

Every report contains:

- facts and unknowns
- expected/observed comparison
- hypotheses with supporting and contradicting evidence graded by
  `.agents/skills/openspec-change-interviewer/references/evidence-matrix.md`
  (`replicated | single-study | conflicting | weak-indirect | not-verified`)
- one bounded discriminating probe
- key conclusions with evidence ids and confidence
- one primary path plus one fallback
- exactly one disposition:
  `retry|targeted_probe|amend_spec|supersede_task|stop_budget`

`error_excerpt` is never a prerequisite for a conclusion. A report cannot
recommend `retry` merely because a command exited zero.

## Workflow

1. Verify the current contract and the actual artifact being judged.
2. Separate facts, inferences, and unknowns. Record source paths, commands,
   hashes, metrics, or screenshots rather than narrative confidence alone.
3. Classify the failure:
   - implementation failure: code did not satisfy a stable contract
   - evidence failure: output may be right but cannot be accepted
   - direction failure: output is valid data for the wrong question
   - contract failure: current ACCEPT/TEST encodes the wrong boundary
   - exhausted approach: repeated attempts add no information
4. Form the smallest competing hypotheses that explain the observation.
5. Use local evidence first. Research external authority only when an API,
   version, method, or domain claim remains uncertain.
6. Stop when one probe can distinguish the leading hypotheses or when the
   budget is exhausted.
7. Return actionable guidance with a verification method for every step.

When evidence conflicts, do not average it into a vague conclusion. State the
conflict and make the next probe target that conflict.

## Loop-light profile

When called by `$openspec-loop-engineering`:

- the sole supervisor invokes this skill in-process; unblock host spawn count is
  zero and this skill is not a third subordinate host or worker lane
- before invocation, align the current ref ACCEPT, `loop.json`, applicable
  requirements, this skill, and the unblock action class; `max_revisions`,
  revision/change minutes, and generic result breakers do not gate diagnosis
- default sink is `return_only`
- at most two unblock runs per ref in one active fingerprint revision
- the first is a repair decision; a second requires a completed second Apply
  and a different failure fingerprint or new discriminating evidence
- with the default two-Apply allowance, the second is a terminal adjudication:
  `amend_spec|supersede_task|stop_budget`, never another retry
- at most 4 tool calls, 4 evidence items, and 180 seconds
- stop after one primary path and one fallback
- do not implement the fix or widen into general literature review
- do not resume the failed Apply task/thread, call `Task.resume` or
  `resume_agent`, spawn explorer/mapper/verifier/review, swap workers, or let a
  worker call Loop gate/record

Persist a report only when it changes task direction, creates a durable
blocker, supersedes a task, or the user explicitly asks for audit evidence.

## Disposition rules

- `retry`: a verified implementation correction exists and the current spec
  remains valid
- `targeted_probe`: one bounded experiment is needed before implementation
- `amend_spec`: acceptance, scope, output authority, or test meaning must change
- `supersede_task`: the current task/approach is terminal but a replacement
  task can preserve the goal
- `stop_budget`: remaining uncertainty cannot be reduced within the recorded
  ref-local budget or authority

Host coupling is fixed:

- `retry` is advisory. Only the supervisor may pass the ordinary gate and open
  a fresh Apply packet when that ref still has positive Apply remainder; the
  failed packet/thread remains terminal and is never resumed.
- `targeted_probe` runs one probe in-process by default. A single read-only
  spawn is allowed only when a concrete wall-clock benefit is recorded on the
  scan row; it joins immediately, consumes zero Apply, and never starts Verify.
- `amend_spec|supersede_task` returns to interviewer or the authorized sole-
  supervisor semantic-restamp path with spawn zero and counters preserved.
- `stop_budget` stops only the affected ref. It does not freeze siblings or buy
  a third research agent.

An explicitly authorized ref-local `max_apply_attempts>=3` may permit the second
unblock to return `retry` or `targeted_probe` for a third Apply. The disposition
is advisory only: unblock never dispatches, swaps a worker, changes a packet,
or replenishes an Apply/unblock allowance. A repeated or missing result
fingerprint without new evidence cannot enter the second unblock.

The second unblock never starts a research swarm. Under the default two-Apply
allowance it remains terminal adjudication only, as defined above.

A first same-ref blocking window or disposition `amend_spec` may make the sole
supervisor eligible for one reasoned
`stamp_source=unblock_self_confirm` semantic reseal in that source fingerprint.
This skill remains diagnosis-only and never edits `tasks.md`. The supervisor's
candidate may only narrow or mechanically correct that ref's ACCEPT/TEST/FILES
inside unchanged WRITE_SCOPE. Widening, framework policy, other refs, DAG,
passed ACCEPT, budget raises, external/product/Git authority, or a repeated
self-confirm returns to the interviewer. A second unblock MUST NOT self-restamp
or default to retry under the two-Apply allowance.

`amend_spec` and `supersede_task` pause Loop and return to
`$openspec-change-interviewer`. Under `full_auto`, only the sole supervisor may
restamp the semantic change with `--allow-semantic-change --reason`. No preview,
stamp, or new revision silently resets budgets.

## Sinks and attempt placement

Use the caller's sinks. When durable output is requested without explicit
paths, use:

`openspec/changes/<change-id>/unblock/task-<id>-attempt-<NNN>.{json,md}`

Choose the next unused number; do not fill historical gaps. Never write a
report into `specs/` or a heavy product root.

Keep four identities separate:

1. tracked decision report under `unblock/`
2. heavy product/run evidence under the recorded approved product or bundle root
3. disposable scratch under the recorded scratch root
4. Loop ledger attempts, containing counters and fingerprints only

If a sink fails, report the sink error and still return the canonical JSON and
guidance in chat. Do not invent a fallback write path.

## Confidence and stopping

High confidence requires either one unambiguous primary source or independent
evidence convergence. Medium confidence requires a discriminating probe. Low
confidence remains a hypothesis and cannot authorize destructive action.

When tools, data, or access are unavailable, return an exact data-to-collect
plan and use `stop_budget` or `targeted_probe`; do not guess through the gap.

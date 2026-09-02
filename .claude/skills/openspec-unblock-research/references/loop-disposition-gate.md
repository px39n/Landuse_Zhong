# Loop Disposition and Repair Gate

Read this reference only when Unblock is invoked by the OpenSpec Loop. The
diagnostic skill still owns facts, hypotheses, evidence grading, and one
discriminating probe; this file owns Loop budgets, repair classes, routing, and
durable sinks.

## Ref-local budget

- The sole supervisor invokes Unblock in-process with spawn zero.
- Default maximum is two Unblock runs per ref and active fingerprint lineage.
- The second run requires a completed second Apply plus new evidence or a
  different failure fingerprint.
- Under default Apply 2, the second run is terminal adjudication:
  `amend_spec|supersede_task|stop_budget`, never retry.
- A repeated/missing failure fingerprint cannot buy another diagnosis.
- One ref's exhaustion never freezes ready siblings.

An explicitly authorized `max_apply_attempts>=3` may let the second Unblock
recommend `retry|targeted_probe`, but Unblock never dispatches or replenishes a
counter. There is no third Unblock and no research swarm.

## Dispositions and repair class

- `retry`: verified implementation correction; stable contract.
- `targeted_probe`: one bounded experiment before implementation.
- `amend_spec`: acceptance, method, scope, authority, or TEST meaning changes.
- `supersede_task`: the approach is terminal and a successor may preserve the
  goal.
- `stop_budget`: remaining uncertainty cannot be reduced within authority.

Missing `repair_class` defaults fail-closed: `retry|targeted_probe` means R0,
`amend_spec` means R2, and `supersede_task|stop_budget` are always R2.

- R0 keeps the contract unchanged and returns to a fresh Apply or one probe.
- R1 is `amend_spec` for one same-ref method-only repair pre-authorized by
  `REPAIR_POLICY: bounded-r1`; the report remains advisory.
- R2 covers GOAL/ACCEPT boundary, refs/DAG, scope, paths, retention, budget,
  external authority, or succession and returns to interviewer.

## `repair_r1`

The runtime independently checks explicit Apply budget exactly 3, one stable
obligation hash, and one `openspec:repair-r1-method` band for the selected ref.
Only that method band, one TEST Run block, and blocked/deviated-to-pending state
may change; WRITE_SCOPE, files, paths, budgets, other refs, hard quantities, and
fail-closed semantics stay fixed. Usage follows the source/target fingerprint
lineage and grants exactly one post-repair Apply. Failure then becomes
`stop_budget`; no second R1 or third Unblock exists.

`unblock_self_confirm` remains a separate first-window mechanical exception for
one non-widening same-ref ACCEPT/TEST/FILES correction inside unchanged
WRITE_SCOPE. It cannot alter framework policy, another ref, DAG, passed ACCEPT,
budget, external/product/Git authority, or run after the second Unblock.

## Routing and sinks

`retry` opens only a fresh supervisor-authorized Apply packet. A targeted probe
runs in-process; one read-only spawn is allowed only for recorded wall-clock
benefit and joins immediately. R2 causes the interviewer to create/refresh the
seven-section `interview.md` first; it never auto-creates a successor change.

Persist only a direction-changing report, durable blocker, succession decision,
or explicitly requested audit at:

`openspec/changes/<change-id>/unblock/task-<id>-attempt-<NNN>.{json,md}`

Use the next unused number. Keep report, heavy product evidence, disposable
scratch, and ledger counters in their separately authorized roots. If a sink
fails, return the canonical report in chat and do not invent another path.

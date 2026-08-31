# Local Dispatch Packet

This packet defines the local logical dispatch envelope carried through the
single change-local `handoff.json.dispatches[]` surface.

```text
Agent: agents/<agent-id>
Package ID:
Role ID:
Local entry:
Ref:
Apply attempt:
Assignment:
Acceptance boundary:
Scope:
Forbidden scope:
Allowed actions:
Write scope:
Expected result:
Expected evidence:
Execution: sync | async
Join: immediate | <join-id> | N/A
Stop when:
```

## Rules

- `Agent` is a logical protocol address, not a repository path, slash command,
  persistent lifecycle, or copied provider projection.
- `Package ID` names one bounded work package.
- `Role ID` is one canonical worker role from `canonical-roles.json` or local
  supervisor-direct `zpy`. Explicit `rose` is legacy/bootstrap input only.
  `general` is not a valid formal owner.
- `Local entry` is the adapter projection from `role-adapter-matrix.md`.
- `Ref` and `Apply attempt` bind one Apply packet to exactly one active ref and
  one supervisor-owned Apply attempt. Non-Apply review or research packets may
  use `Apply attempt: N/A`.
- `Scope`, `Forbidden scope`, `Allowed actions`, and `Write scope` narrow
  runtime authority; they never create new authority.
- `Execution` is `async` only for independent inputs, non-overlapping writes,
  and a stable supervisor-owned `Join` id.
- Every non-supervisor worker remains non-delegating.

## Fresh-packet boundary

Any change to Role ID, local entry, assignment, ref, scope, forbidden scope,
permissions, write scope, acceptance boundary, expected result, or verification
claim requires a new packet.

Every native host packet uses a fresh one-shot task/thread context. A terminal
result closes that context; `Task.resume`, `resume_agent`, old task/thread IDs,
and equivalent continuation are forbidden.

## Native host batch projection

The Loop census may expose direct and dispatchable refs together. A native host
must derive:

```text
host_batch_refs =
  dispatch_refs where routing.decision == dispatch and agent != null
```

Direct `zpy|rose` refs remain supervisor-local even when present in
`dispatch_refs`. One wave produces at most one native host batch and at most one
fresh packet per `host_batch_ref`.

For a static transmission-efficiency test, shared fingerprint, join ID, and
write-disjoint declaration may be measured once as `shared_context_bytes`,
while ref-local projections contribute `packet_delta_bytes`. This normalized
test projection does not replace or redefine the canonical packet schema.

## Retry and redispatch

- A worker may retry only transient tool or process failures inside the same
  unchanged packet before its terminal result.
- A terminal `failed`, empty, `partial`, `blocked`, or `unverified` result does
  not authorize automatic redispatch, resume, role swap, or scope expansion.
- The supervisor decides whether a new Apply attempt is permitted by gate and
  budget.

## Local exclusions

- Do not copy or depend on delivery-flow lifecycle state.
- Do not materialize foreign attached-repository ownership, external worktree
  identity, formal board ownership, per-worker status journals, or a second
  ledger.
- Do not use runtime-private ids as the packet identity.

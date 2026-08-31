# Proactive Delegation Scan

Run this scan at the start of every non-trivial intent, and again only when
changed evidence exposes a materially new work split. This is a local Loop
routing step. It does not create a second lifecycle, consume Apply budget, or claim scheduling headcount.

## Inputs

- exact task or review question
- acceptance boundary
- target files or bounded evidence sources
- governing rules
- one relevant precedent when available
- verification path
- material assumptions, conflicts, and unknowns

Missing material context blocks only the affected package.

## Output

Record all four fields:

```text
matched_role_id: <canonical role id> | null
decision: direct | dispatch | blocked
direct_reason: trivial_work | clarification_or_split | no_matching_specialist | capability_failure | overlap | negative_benefit | N/A
effective_role_id: <canonical role id or local zpy> | N/A
```

`general` is never emitted.

## Decision rules

1. Classify the assignment before choosing a role.
2. Select the narrowest canonical Role ID with the most specific evidence
   contract.
3. Return `dispatch` only when all of these hold:
   - the assignment is one clear bounded non-trivial package;
   - the narrowest matching specialist is available;
   - current effective capabilities and permissions permit the package; and
   - ownership does not overlap another current package.
4. Return `direct` only for one named exception:
   - `trivial_work`
   - `clarification_or_split`
   - `no_matching_specialist`
   - `capability_failure`
   - `overlap`
   - `negative_benefit`
5. A permitted new `direct` package uses local `effective_role_id: zpy` and
   remains supervisor-owned with `agent=null`. Explicit `rose` is accepted only
   as predecessor/history or bounded bootstrap input and is never a new inferred
   output.
6. Missing authority or missing effective capability remains `blocked`; it must
   not be converted into a direct exception.

## Dispatch consequences

- `dispatch` creates exactly one logical `agents/<agent-id>` envelope in
  `handoff.json.dispatches[]`.
- `direct` creates no agent envelope.
- Native hosts derive
  `host_batch_refs = dispatch_refs where routing.decision == dispatch and agent != null`.
  Raw `dispatch_refs` membership never makes a direct `zpy|rose` ref spawnable.
- This scan does not dispatch by itself, does not Verify, does not promote, and does not issue PASS.

## Re-scan boundary

Run a fresh scan only when changed evidence creates a materially new:

- assignment
- scope
- write scope
- acceptance boundary
- verification claim

A terminal `failed`, empty, `partial`, `blocked`, or `unverified` result does
not itself authorize another scan, semantic redispatch, worker substitution, or
scope expansion.

## Local exclusions

- Do not import delivery-flow phases or ownership.
- Do not materialize A33, WT-001, Board owners, progress logs, or review
  arbitration artifacts.
- Do not use worker counts, `max_subagents`, or role quotas to replace the
  write-scope wave decision.

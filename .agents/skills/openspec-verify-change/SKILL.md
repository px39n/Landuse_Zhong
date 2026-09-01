---
name: openspec-verify-change
description: Verify an OpenSpec task or whole change against its active contract. In Loop task mode, return PASS, FAIL, BLOCKED, or DEVIATED without creating a full evidence bundle.
license: MIT
metadata:
  author: openspec
  version: "2.0"
---

# OpenSpec Verify Change

Verify implementation against the current proposal, design, delta specs, and
active task contract. Verification is read-mostly and evidence-first.

## Modes

- `$openspec-verify-change <change-id> --task <ref>` verifies one active task
  for Loop promotion.
- `$openspec-verify-change <change-id>` verifies the whole change before final
  completion or archive. When the registry declares `GOAL:` blocks, this mode
  also closes at the design level and returns `PASS` or `GAP` per goal.

If the change or task is ambiguous, resolve it from the active registry; ask
only when more than one live candidate remains.

## Retention and test tier

Read `loop.json` when present. Do not ask again for cache, D-drive, bundle, or
GUI paths when the sealed fingerprint still matches.

- Attempt verification: run only owner tests relevant to the latest edit.
- Promotion verification: execute the selected task's complete `TEST:` block.
- Final verification: run strict OpenSpec validation plus adjacent regression.
- Audit verification: create a retained bundle only for `retention=full` or an
  explicit audit request.

For `none` or `thin`, never fabricate a full `auto_test_openspec` bundle. Use
the sealed scratch/ledger policy and existing product pointers. If an operation
requires an unapproved external write or destructive action, return `BLOCKED`.

Task Verify is closure inside an already admitted execution chapter. Before
invocation, align the current task ACCEPT, applicable requirements,
`loop.json`, and this skill. A matching fingerprint plus the task's own joined
completed Apply/evidence is sufficient to attempt Verify: `max_revisions`,
revision/change active-minute caps, unrelated-ref results, and generic breakers
MUST NOT block it. Verification still fails or blocks on its own fingerprint,
join, evidence, environment, or semantic preconditions and never edits ACCEPT.
Promotion changes only checkbox/feature state and consumes no cycle stamp.

## Task-mode scope

Resolve the selected ref's checkbox, `ACCEPT:`, `TEST:`, direct dependencies,
and referenced requirements. Do not mark later or unrelated unfinished tasks
as failures. An unchecked selected task is expected before promotion and is not
itself a defect.

Assess:

1. Completeness within the selected task boundary.
2. Correctness against `ACCEPT:` and referenced scenarios.
3. Coherence with active design decisions and repository patterns.
4. Product semantics: source, target, metric, schema, and research direction.

## Verdicts

- `PASS`: required checks and product semantics match ACCEPT.
- `FAIL`: implementation or tests are wrong within the agreed contract.
- `BLOCKED`: verification cannot safely run because authority, dependency,
  environment, credentials, or required evidence is unavailable.
- `DEVIATED`: commands may succeed and tests may even pass, but observed product
  output, data source, metric, target, or research direction differs from
  ACCEPT.

`DEVIATED` is distinct from `FAIL`: it signals that retrying the same patch is
unlikely to repair the direction without diagnosis or contract clarification.

## Design-level closing

A change is not complete because its tasks are checked. When the registry
declares `GOAL:` blocks, compare each goal's `ACCEPT:` to the observed outcome
and report the comparison as an observation file keyed by goal id, with
`status` of `match` or `mismatch`, the observed outcome, and evidence pointers:

```powershell
python scripts/openspec_loop.py goals <change-id>
python scripts/openspec_loop.py design-verify <change-id> --observation <path>
```

A goal with no supplied observation is `unobserved`, which is a `GAP`, not a
`PASS`; the tool decides structural coverage and never infers product semantics.
A `GAP` returns a `openspec-loop-revision-proposal.v1` payload naming the goal,
expected and observed outcomes, evidence, and the tasks it would add or
supersede. Hand that proposal to the supervisor rather than reporting the change
complete, and persist it only when it changes execution direction, under
`openspec/changes/<change-id>/unblock/`.

## Procedure

1. Run `openspec status --change "<change-id>" --json` and obtain the active
   artifact paths.
2. In task mode, load only the selected task block and directly referenced
   contract sections. In whole-change mode, inspect all active tasks and current
   requirements. Do not load superseded appendices as active truth.
3. Inspect implementation and focused tests with exact file/line or artifact
   pointers.
4. Run the applicable test tier. Record the command and result; do not retain
   broad logs during ordinary attempts.
5. Compare expected and observed product outcomes even when the command exits
   zero.
6. Return a concise report and exactly one verdict.

## Output contract

```text
VERDICT: PASS|FAIL|BLOCKED|DEVIATED
REF: R<n>|whole-change
RISK: contract-only|normal|high
ASSURANCE: full|reduced
EXPECTED: <accepted outcome>
OBSERVED: <verified outcome>
EVIDENCE:
- <file, artifact, or command pointer>
NEXT:
- <one bounded action>
```

Every critical finding must be scoped, evidenced, and actionable. Prefer a
lower-severity finding over speculation. If the verdict is `BLOCKED` or
`DEVIATED`, hand the structured comparison to `$openspec-unblock-research`.

---
name: openspec-apply-change
description: Implement an OpenSpec change or one selected task. In Loop orchestration, change only the selected task, run attempt-level owner checks, and leave promotion bookkeeping to the verifier and supervisor.
license: MIT
metadata:
  author: openspec
  version: "2.0"
---

# OpenSpec Apply Change

Implement from the current OpenSpec contract. Keep changes surgical and preserve
unrelated user work.

## Modes

- `$openspec-apply-change <change-id>`: ordinary interactive apply.
- `$openspec-apply-change <change-id> --task <ref> --orchestrated`: one Loop
  attempt for exactly one active task.

In orchestrated mode:

- require `check` to report a matching `contract_fingerprint` with no pending
  irreversible policy decision; a preview, stamp, or `seal --confirmed` is not
  an ordinary start-work gate;
- accept one fresh Apply packet that binds exactly one ref to exactly one
  supervisor-owned Apply attempt, write scope, and join id;
- resolve the task from `feature_list.json` and its source block in `tasks.md`;
- implement only that ref and direct prerequisites already authorized;
- never write the authoritative Apply ledger record, run supervisor Verify,
  promote, toggle the checkbox, set `passes`, or declare final PASS;
- never retry an assertion or semantic failure, automatically redispatch the
  packet, swap workers, or widen its scope;
- never create `BUNDLE`, `EVIDENCE`, `progress.txt`, or `runs.log`.

## Preflight

1. Run `python scripts/openspec_loop.py --repo-root . check <change-id>` in
   orchestrated mode.
2. Read only the selected task's `ACCEPT:`, `TEST:`, dependencies, referenced
   requirements/design sections, and relevant code/tests.
3. Stop if the semantic fingerprint drifted, scope is ambiguous, or
   implementation would require changing the contract. Narrative drift in
   `proposal.md`, `design.md`, or `specs/**` is refreshed with
   `python scripts/openspec_loop.py reseal <change-id>` and does not stop the
   attempt. Hand a semantic pause to `$openspec-change-interviewer`, or to the
   supervisor when `loop.json` records `autonomy: full_auto`.
4. Read the recorded retention/path policy once. Do not invent or repeatedly ask
   for D-drive, cache, product, bundle, GUI, or Colab locations.

## Implementation

1. Inspect the smallest upstream/downstream surface needed for the task.
2. Make the minimum coherent change.
3. A transient tool or process failure may be retried only before the terminal
   result, inside the same unchanged packet and Apply attempt. An assertion or
   semantic failure is terminal and returns to the supervisor.
4. Run attempt-level owner tests only. The complete task `TEST:` belongs to
   supervisor Verify.
5. Compare any produced result to ACCEPT semantics; exit code 0 is not enough.
6. Return changed files, focused checks, observed result, and unresolved risk.

For `retention=none|thin`, do not produce a full validation bundle. Use the
recorded scratch/product paths only when the task needs them. If required
authority or an external dependency is missing, stop with a concrete blocker.

## Attempt result

```text
ATTEMPT: APPLIED|BLOCKED|DEVIATED
REF: R<n>
FILES:
- <path>
CHECKS:
- <command or inspection>
OBSERVED: <product result>
NEXT: $openspec-verify-change <change-id> --task <ref>
```

Loop consumes NEXT: verify in the same turn.
It is not a user handoff or turn-ending speech act.

Use `DEVIATED` when implementation runs but its output/source/metric/direction
does not match ACCEPT. Do not hide that condition behind a passing owner test.

This is one terminal worker result, not an Apply ledger record or a verifier
verdict. The supervisor alone accepts the result and writes exactly one
authoritative Apply record for its packet/ref/attempt; duplicate results do not
create another record or authorize another attempt. In a shared worktree, the
supervisor joins every actually dispatched member of the current wave before
starting Verify for any member.

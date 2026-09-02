# Semantic Verification Contract

Read this reference for `pilot|production|canary`, unspecified legacy tasks, or
whole-change design closing. `smoke` may close mechanically and does not invoke
semantic Verify.

## Admission and retention

Task Verify requires `next_action=verify_semantic`, a current fingerprint, the
task's joined completed Apply/evidence, and CLI `verify-mechanical` PASS. Do not
rerun the command merely to reproduce its receipt. `max_revisions`, generic
breakers, unrelated refs, and revision/change active-minute advisories do not
remove completion right.

Use the recorded retention and paths. Attempt checks are owner-focused;
promotion runs the complete task TEST; final verification adds strict OpenSpec
validation and adjacent regression. Create a full bundle only for
`retention=full` or an explicit audit request.

## Task judgment

Read the selected checkbox, ACCEPT, TEST, direct dependencies, applicable
requirements, implementation, and focused evidence. An unchecked task before
promotion is expected. Judge completeness, correctness, design coherence, and
product source/target/metric/schema/research direction.

- `PASS`: checks and observed semantics match ACCEPT.
- `FAIL`: implementation or tests violate a stable contract.
- `BLOCKED`: authority, dependency, environment, credentials, or required
  evidence prevents safe judgment.
- `DEVIATED`: execution may succeed, but product source, metric, target, or
  research direction differs from ACCEPT.

## Whole-change design closing

Checkbox closure is insufficient when tasks declare GOAL blocks. Compare each
GOAL ACCEPT with an observation keyed by goal id and `status=match|mismatch`,
then run:

```powershell
python scripts/openspec_loop.py goals <change-id>
python scripts/openspec_loop.py design-verify <change-id> --observation <path>
```

An absent observation is `unobserved`/GAP, never PASS. A GAP returns an
`openspec-loop-revision-proposal.v1` naming the expected/observed outcome,
evidence, goal, and affected refs. Return it to the supervisor; persist only a
direction-changing proposal under the approved change-local `unblock/` sink.

## Evidence procedure

Use `openspec status --change "<change-id>" --json` to locate active artifacts.
Inspect only active truth, run the applicable tier, compare outcome semantics
even after exit 0, and return one typed verdict with exact evidence pointers.
Never edit ACCEPT, task state, WRITE_SCOPE, notebook, manifest, or product.

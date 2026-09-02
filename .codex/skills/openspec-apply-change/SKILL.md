---
name: openspec-apply-change
description: Implement an OpenSpec change or one selected task. In Loop mode, edit only the receipt-bound task and leave verification, promotion, and bookkeeping to the supervisor.
license: MIT
metadata:
  author: openspec
  version: "2.0"
---

# OpenSpec Apply Change

Implement the current contract with a surgical diff and preserve unrelated user
work.

## Modes

- `$openspec-apply-change <change-id>`: ordinary interactive Apply.
- `$openspec-apply-change <change-id> --task <ref> --orchestrated`: one Loop
  attempt for exactly one active ref.

## Orchestrated preflight

1. Require the supervisor receipt and run `receipt-check`; never rerun `check`,
   `plan`, or a registry census.
2. Read only the bound ACCEPT, TEST, dependencies, referenced requirements, and
   relevant code/tests. Reuse recorded retention and paths.
3. Stop on fingerprint/harness drift, ambiguous WRITE_SCOPE, missing authority,
   or a needed contract change. Return semantic changes to the supervisor or
   interviewer; a narrative-only warning uses the existing reseal path.
4. Never toggle tasks, set feature state, record the ledger, Verify, promote,
   create audit bundles, or resume an old task/thread.

## Implementation

- Inspect the smallest required code path and implement only the selected ref.
- Run attempt-level owner checks; the complete TEST belongs to promotion.
- Compare source, target, metric, schema, and research direction to ACCEPT; exit
  0 alone is insufficient.
- Use only approved scratch/product roots. Product/external writes, credentials,
  destructive actions, budget changes, and Git push/PR/main retain their human
  authority gates.
- If execution succeeds in the wrong direction, return `DEVIATED` rather than
  hiding it behind a passing check.

For an explicitly adopted task-owned notebook profile, read
`references/execution-notebook-contract.md`. Apply owns execution writes within
that contract; it never rewrites tracked notebook outputs.

## Typed result

```text
RESULT_SCHEMA: openspec-apply-result.v1
RECEIPT_ID: <receipt id>
ATTEMPT: APPLIED|BLOCKED|DEVIATED
REF: R<n>
FILES:
- <path>
CHECKS:
- <command or inspection>
OBSERVED: <product result>
```

The result is terminal for this one-shot context. Only the supervisor may pass
it to `record --kind apply --receipt <token>`; natural-language continuation has
no control authority.

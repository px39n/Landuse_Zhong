# Artifact placement contract

## Free write root

`openspec/changes/<change-id>/` is the only deterministic free write root for
OpenSpec contract artifacts (`proposal.md`, `design.md`, `specs/**`, `tasks.md`,
`interview.md`, `context.md`, `feature_list.json`, `loop.json`, optional
`seal-preview.md`, `unblock/`, and `handoff.json`).

## Ask-once placement

Any write outside the change directory requires one explicit placement choice
before the first write: beside the source, a user-named sibling folder, append
to an existing document, or chat preview only. Silence is not approval.

## Recorded Loop roots

Reuse paths recorded in the change contract or `loop.json`. If `loop.json` is
missing, interviewer may run `check` once to initialize policy and then inspect
one `next --shadow`; when it exists, run only the shadow inspection. Neither
operation creates ledger, scratch, cache, product, bundle, or GUI/Colab roots.

Choose a path profile for this change; never copy another change's absolute
external tree as a default:

- `A_local_thin`: repo-local ledger/scratch and product `null`
- `B_external_heavy`: heavy product root named by this change plus thin pointers
- `C_product_repo`: contract-named in-repo product root
- `D_custom`: user supplies every root once

Record ledger, disposable scratch/pytest, product, bundle, and GUI/Colab roots
once. `seal-preview.md` is optional policy/audit output; it is neither ordinary
start-work authority nor a prerequisite for the first Apply.

## Four path identities

Keep separate: tracked decision reports under `unblock/`, heavy product/audit
evidence, disposable scratch under its approved root, and ledger attempts that
hold counters/fingerprints only. Never invent repository-root `_tmp/`, `tmp/`,
`tmp_*`, `tmp_pytest_*`, `.pytest_tmp`, or unnamed scratch folders.

Before delete or archive, classify paths as
`task-scoped|unrelated pre-existing|generated-ignored|scratch|unknown`. Only
task-scoped or approved scratch paths may be deleted; unknown and unrelated
paths remain untouched.

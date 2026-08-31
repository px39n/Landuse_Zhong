# Artifact placement contract

## Free write root

`openspec/changes/<change-id>/` is the only deterministic free write root for
OpenSpec contract artifacts (`proposal.md`, `design.md`, `specs/**`,
`tasks.md`, `interview.md`, `context.md`, `feature_list.json`, `loop.json`,
`seal-preview.md`, `unblock/`, `handoff.json`).

## Ask-once placement

Any write outside the change directory requires one explicit placement choice
before the first write:

| Option | Meaning |
|---|---|
| A | Beside the source file being edited |
| B | New sibling folder (path named by the user) |
| C | Append to an existing document |
| D | Chat preview only — status stays blocked / unverified for BUILD readiness |

Do not silently pick a path. Silence is not approval.

## Recorded Loop roots

Reuse paths already recorded in the change contract or `loop.json`. For an
ordinary fingerprint-ready change, `check`/`plan` may initialize `loop.json`
from its recorded `thin` retention decision without a preview, stamp, or user
ceremony. Initialization records policy only: it does not create the ledger,
scratch, cache, product, bundle, or GUI/Colab roots.

Paths are chosen by **path profile** and expanded for **this** `<change-id>`
(see `seal-preview-format.md`). Do not copy another change's absolute external
tree as the default.

| Profile | When | Typical expansion |
|---|---|---|
| `A_local_thin` | Ordinary local / tooling | ledger `auto_test_openspec/<change-id>/loop/ledger.json`; scratch `test_cache/<change-id>/`; product `null` |
| `B_external_heavy` | Heavy outputs outside the repo | ledger as above; product = root named by **this** change (propose a new name if missing); pointers under `auto_test_openspec/<change-id>/pointers/` |
| `C_product_repo` | Product inside the repo | ledger/scratch as A; product `outputs/<change-id>/` or a contract-named in-repo root |
| `D_custom` | Rare | user supplies all five roots once |

Recommended ordinary layout when profile A applies:

| Role | Recommended path |
|---|---|
| Ledger | `auto_test_openspec/<change-id>/loop/ledger.json` or under scratch |
| Scratch | `test_cache/<change-id>/` |
| Pytest basetemp | `test_cache/<change-id>/pytest/` (recorded once, not a per-task invention) |
| Product | explicit product root or `null` |
| Bundle | `null` unless `retention=full` |
| GUI/Colab | `outputs/colab_verify/` only when required |

`openspec/changes/<change-id>/seal-preview.md` is optional. Write it only when
the user requests an audit diagnostic or an irreversible policy choice needs
human confirmation, such as raising optional `hard_ceiling` policy data or
changing retention, paths, or scope. The preview narrows that policy review; it
is neither ordinary start-work authority nor a prerequisite for the first Apply.

Forbidden ad-hoc roots: repository-root `_tmp/`, `tmp/`, `tmp_*`,
`tmp_pytest_*`, `.pytest_tmp`, and unnamed scratch folders.

## Four similar path identities

Never conflate paths that share an `attemptNNN` label:

1. Tracked decision report: `openspec/changes/<id>/unblock/*`
2. Heavy product or audit evidence under the confirmed product/bundle root
3. Disposable scratch: `{pytest,cache,tmp}/<ref>/<run-id>/` under recorded scratch
4. Ledger `attempts[]`: counters and fingerprints only

## Hygiene gate

Before delete or archive, classify dirty paths as
`task-scoped | unrelated pre-existing | generated-ignored | scratch | unknown`.
Only task-scoped / scratch candidates may be proposed for deletion via
`$openspec-hygiene` and `scripts/clean_test_residues.py`. Unknown and unrelated
paths never auto-delete.

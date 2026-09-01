# Change: Make semantic admission atomic and preserve Loop completion rights

## Why

The Loop currently applies `change.max_revisions`, active-minute limits, and a
generic run-level breaker to every `gate` kind. A new fingerprint can therefore
admit Apply while still under its cap, become productive when Apply is recorded,
and then lose access to Verify because the same broad gate observes the updated
count. The same mechanism can starve later refs in an already admitted
fingerprint and can turn an ordinary blocked-path diagnosis into a request for a
larger change budget.

The defect is an admission/completion boundary error, not a native-host or VFD
product failure. `vfd-public` values such as `5` and `vfd-gdp` values such as
`14` are change-local emergency history; neither is a repository default.

## What Changes

- Define an execution chapter as one current, already stamped active-registry
  fingerprint, and distinguish its original `ACCEPT` text from later obligation
  changes.
- Move `change.max_revisions` authority to semantic stamping. Ordinary
  `gate --kind apply|explore` never checks it, and all closure actions retain a
  completion right.
- Count at most three charged cycle stamps per execution chapter by default.
  Initial ACCEPT, promotion, narrative-only reseal, and a qualifying
  chapter-outside interviewer stamp remain free.
- Limit minutes and result breakers to new work, scope breakers to the current
  episode/ref/kind, and keep unblock diagnosis ref-local.
- Add one supervised blocking-window exception: the sole supervisor may
  self-restamp one narrowed/corrected ref once per source episode with
  `stamp_source=unblock_self_confirm` and a required reason. The exception
  cannot alter framework policy, widen acceptance, change the DAG, or expand
  write scope.
- Lock new-seal defaults to Apply `2`, Unblock `2`, and cycle stamps `3`, while
  preserving explicitly recorded prior values when reseal omits overrides.
- Require every active Loop call to align the current ACCEPT, `loop.json`, the
  relevant capability spec, and the canonical skill before execution.

## Non-Goals

- No `vfd-public` or `vfd-gdp` live-tree, ledger, budget, or product migration.
- No VFD S1-S6 execution, D-drive write, PNG copy, product `--commit`, or
  provider-specific work.
- No per-ref `max_accept_amends`, cross-fingerprint Apply/Unblock lineage,
  semantic-hash narrowing, ledger CAS/lock, or crash journal.
- No native-host, local `zpy`, wave, join, or one-packet-per-ref redesign.
- No new skill package, Board, `progress.txt`, `runs.log`, BUNDLE, EVIDENCE, or
  second authority ledger.

## Artifact Retention Decision

- Audit retention: `thin`
- Retained evidence root: `auto_test_openspec/openspec-loop-completion-right/`
- Disposable cache root: `test_cache/openspec-loop-completion-right/`
- Pytest basetemp root: `test_cache/openspec-loop-completion-right/pytest/`
- Scratch root: `test_cache/openspec-loop-completion-right/`
- Product/runtime output root: `null`
- GUI/Colab evidence root: `null`

CLI verification retains only compact pointers or the disposable G1
observation; it does not retain per-attempt bundles or use a full-bundle root.

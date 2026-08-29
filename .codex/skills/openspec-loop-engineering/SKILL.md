---
name: openspec-loop-engineering
description: Run a sealed, finite-budget OpenSpec implementation loop for one existing change. Use after scope, retention, task dependencies, and acceptance have been confirmed; pause on contract drift, semantic output deviation, or exhausted budgets. Do not use for open-ended design or full retained-audit monitoring.
---

# OpenSpec Loop Engineering

This is the lightweight execution supervisor for an existing OpenSpec change.
It coordinates one task at a time; it does not author an uncertain contract and
does not replace legacy retained-audit monitoring.

## Entry gate

Run:

```powershell
python scripts/openspec_loop.py check <change-id>
python scripts/openspec_loop.py plan <change-id>
```

Proceed only when `check.ok=true`, `plan.sealed=true`, and `selected_ref` is
non-null. Otherwise:

- missing or unsealed `loop.json` -> `$openspec-change-interviewer <change-id>`
- semantic fingerprint drift, meaning an obligation in the active task registry
  changed -> pause and resolve it through the authority rules below
- narrative drift, meaning `proposal.md`, `design.md`, or `specs/**` changed ->
  refresh with `python scripts/openspec_loop.py reseal <change-id>` and continue;
  under `narrative_policy: strict` treat it as a semantic pause instead
- task/feature drift, unknown dependencies, or cycles -> repair the contract
  and regenerate with `$openspec-feature-list <change-id>`
- no ready task -> summarize terminal and blocked states; do not invent work

Promotion is not drift. The sealed `contract_fingerprint` covers the active task
registry with checkbox marks neutralized and `STATE:` directives removed, so
flipping a checkbox after `PASS` leaves the seal valid and needs no reseal.

`plan --advisory` may expose a candidate during design, but it never authorizes
execution.

## Sources of truth

- Human contract: current proposal, design, specs, and active `tasks.md`.
- Compact task state: `feature_list.json`; it must not duplicate full ACCEPT or
  TEST prose.
- Sealed execution policy: `loop.json`.
- Disposable attempt counters: the ledger path recorded in `loop.json`.
- Prior contract wording: Git history, not an ever-growing appendix.

The ledger is never a second spec. It stores fingerprints, counters, durations,
and dispositions, not logs or product evidence.

## Execution latch

After gate returns continue, the next substantive action in the same turn must be Apply.
Do not end the turn with a summary, suggestion, or handoff while the latch is active.
A one-line progress update naming the ref and action is allowed only when the Apply tool or skill call follows in the same turn.

The default executor is the main agent for both Apply and Verify.
Do not reserve or consume a subagent slot by default.
An Apply worker is optional for a heavy or high-risk task when sealed max_subagents remains.
A worker must not verify, promote, toggle a checkbox, or claim PASS.
If no worker is used, the main agent performs Apply directly; worker availability never weakens the latch.

One invocation uses one run_id; retries append attempts and must not open an empty run per ref.
The latch preserves thin retention and must not create `BUNDLE`, `EVIDENCE`, `progress.txt`, `runs.log`, or per-attempt folders.

## Core loop

Drain the ready queue rather than stopping after one task. `plan --batch`
reports every ready ref, and because `ready` requires each dependency to be
complete, that set is mutually independent by construction. Continue while
`gate` allows, the verifier keeps returning `PASS`, and the batch is non-empty.
Stop on the first non-`PASS`, a breaker, or the ceiling.

For each selected ref:

1. Run `gate` before Explore, Apply, Verify, or Unblock.
2. Use `$openspec-explore <change-id>` only for one concrete unknown; record the
   result immediately.
3. Invoke `$openspec-apply-change <change-id> --task <ref> --orchestrated`.
   While the latch is active, invoke it immediately instead of first emitting
   turn-ending speech. The maker changes only that task's implementation and
   cannot promote state.
4. Record the apply result and actual active duration.
5. Invoke `$openspec-verify-change <change-id> --task <ref>` and record exactly
   one of `PASS|FAIL|BLOCKED|DEVIATED`.
6. After `PASS`, promote with
   `python scripts/openspec_loop.py promote <change-id> --ref <ref>`. It flips
   the checkbox, regenerates the index, refuses any edit that would change the
   sealed fingerprint, and re-runs the plan in one transition. Never hand-edit
   the checkbox; repair index drift alone with `sync <change-id>`.
7. On `FAIL`, allow a bounded implementation retry when the gate permits it.
8. On `BLOCKED` or `DEVIATED`, invoke `$openspec-unblock-research <change-id>`
   once in loop-light mode. Respect its disposition:
   - `retry` or `targeted_probe` -> gate before one more attempt
   - `amend_spec` or `supersede_task` -> amend the contract under the authority
     rules below, regenerate the index, and seal a new revision
   - `stop_budget` -> mark the task `maxed` and stop
9. When no ready task remains, close at the design level before claiming
   completion: run `goals <change-id>` for the coverage matrix, then
   `design-verify <change-id> --observation <path>` for `PASS` or `GAP`. An
   absent observation is `unobserved`, not `PASS`. A `GAP` returns a revision
   proposal; apply it under the authority rules and resume the drain loop. Only
   on `PASS` run whole-change verification and strict OpenSpec validation once.

Every action must be followed by `record` before another action of the same
loop. One invocation keeps the same run id across selected refs and retries; an
intentional resume reuses it, while a genuinely new invocation may create one
new id. Revision/change budgets still span those run ids.

## State semantics

Allowed states:

`pending | ready | in_progress | blocked | deviated | maxed | superseded | passed`

- Only `ready` may enter Apply.
- `passed`, `maxed`, and `superseded` are terminal.
- `blocked`, `deviated`, and `in_progress` require an explicit transition.
- `SUPERSEDES: Rn` makes the replacement authoritative without treating the
  maxed predecessor as a satisfied dependency.
- A command that exits zero can still be `DEVIATED` when outputs use the wrong
  source, target, period, metric, resolution, or acceptance interpretation.

## Finite budgets

Read budgets from `loop.json`; do not silently reset them after a new spec
revision. Recommended defaults are:

- task: 2 apply attempts and 1 unblock run
- revision: 8 apply iterations, 1 explore run, 2 distinct subagents, 120 active
  minutes
- change: 3 revisions, 20 apply iterations or twice the active task count, and
  360 active minutes or ten minutes per task, whichever is larger; a first seal
  derives the change budget from the registry it actually covers

Time means recorded active tool/runtime duration, not time spent waiting for a
user. Two identical result fingerprints, two consecutive `no_progress`
outcomes, or two repeated semantic deviations trigger a breaker.

## Authority

`loop.json` seals two authority fields. Read them once and do not re-litigate
them per attempt.

`hard_ceiling` is the only stop that no Loop may raise. It bounds absolute apply
iterations, absolute active minutes, and the number of self-extensions. `seal`
sets it and requires `--confirmed`; `reseal` inherits it verbatim, refuses a
`--set-max-*` value above it, and refuses any further extension once
`max_self_extensions` is spent. A `gate` stop naming `hard_ceiling_*` is
reported as `terminal: true`: stop and report, do not seek more budget.

`autonomy` decides who may move the boundary below that ceiling.

| action | `supervised` (default) | `full_auto` |
|---|---|---|
| extend a change budget | `reseal --set-max-* --confirmed --reason` after explicit user authorization | `reseal --set-max-* --reason`, recorded in `budget_extensions` |
| amend tasks after a design-level gap | pause and hand off to `$openspec-change-interviewer` | amend the registry, regenerate the index, then `reseal --allow-semantic-change --reason` |
| raise `hard_ceiling`, change retention, paths, or scope | human-confirmed `seal` | human-confirmed `seal` |

Autonomy never widens repository authority. Regardless of mode, destructive
external writes, credentials, product `--commit` runs, and `git push`, pull
request, or `main` operations stay human-authorized under `AGENTS.md`. A
`full_auto` Loop that needs one of those stops as `BLOCKED`.

## Retention and attempt placement

Reuse the one-time policy in `loop.json`; do not ask for D-drive, cache, or
bundle paths on every invocation.

- `none`: disposable ledger only; no retained attempt report.
- `thin`: disposable ledger plus a compact promotion receipt or
  decision-changing deviation report.
- `full`: explicit retained bundles; route execution to
  `$monitor-openspec-codex <change-id>` when legacy BUNDLE/EVIDENCE bookkeeping
  is required.

Classify similarly named attempt paths before reuse or deletion:

1. Tracked decision report: `openspec/changes/<change-id>/unblock/*.json|md`.
2. Heavy product/run evidence under the explicitly confirmed product or bundle
   root.
3. Disposable `{pytest,cache,tmp}/<ref>/<run-id>/` under the confirmed scratch
   root.
4. Ledger `attempts[]`, which is counters and fingerprints only.

Never place heavy logs in specs, never use the change directory as bundle
staging, and never infer identity only because two paths contain the same
`attemptNNN` label.

## Verification profiles

- Attempt: owner tests for touched modules, `-x -q -p no:cacheprovider`; no
  formal bundle.
- Promotion: execute the complete task `TEST:` exactly once before PASS.
- Final: whole-change verification plus `openspec validate <change-id> --strict`.
- Audit: full immutable bundle only when the policy or task explicitly requires
  it.

High-risk work involving credentials, destructive external writes, migrations,
or shared schemas requires an independent verifier. If unavailable, stop as
`BLOCKED`.

## Helper commands

```powershell
python scripts/openspec_loop.py plan <change-id> --batch
python scripts/openspec_loop.py gate <change-id> --run-id <id> --ref <R#> --kind apply
python scripts/openspec_loop.py record <change-id> --run-id <id> --ref <R#> --kind apply --result failure --duration-seconds <n>
python scripts/openspec_loop.py promote <change-id> --ref <R#>
python scripts/openspec_loop.py sync <change-id>
python scripts/openspec_loop.py goals <change-id>
python scripts/openspec_loop.py design-verify <change-id> --observation <path>
python scripts/openspec_loop.py apply-revision <change-id> --proposal <path> --run-id <id>
python scripts/openspec_loop.py summary <change-id> --run-id <id>
```

`promote` requires a recorded verifier pass for that ref in the current episode,
so the verifier stays authoritative. `apply-revision` refuses under
`autonomy: supervised` and refuses whenever `gate` already reports stop.

The sealed ledger path and fingerprint are loaded from `loop.json`; overrides
must match the sealed values.

## Output

During a continuing loop, emit at most one progress line naming the selected
ref and action, then continue with the required tool or skill call. End the turn
only on a stop, a non-`PASS` verdict, an authority gate, or a ceiling.
Consume Apply's NEXT: verify in the same turn; it is not a user handoff.

When the loop stops, report the selected ref, action, verifier verdict, budget
state, remaining distance to `hard_ceiling`, and whether any tracked evidence
was written. Name the autonomy mode whenever a budget was extended or a
contract amended without a human turn. Never claim promotion unless the
checkbox, compact feature state, and post-promotion `check` agree.

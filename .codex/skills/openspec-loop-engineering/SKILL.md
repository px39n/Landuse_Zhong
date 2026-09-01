---
name: openspec-loop-engineering
description: Run the finite-budget OpenSpec implementation supervisor for one existing change. Use when the active registry has a matching fingerprint and no irreversible policy decision is pending; route a write-disjoint Apply wave, join it, and keep Verify, promotion, budgets, and final authority supervisor-owned.
---

# OpenSpec Loop Engineering

This is the sole lightweight execution supervisor for an existing OpenSpec
change. It coordinates the complete ready census and one write-disjoint Apply
wave at a time; it does not author an uncertain contract, create a second
lifecycle, or replace legacy retained-audit monitoring.

## Entry gate

Run:

```powershell
python scripts/openspec_loop.py check <change-id>
python scripts/openspec_loop.py plan <change-id>
```

When `loop.json` is missing, `check`/`plan` initialize recorded `thin` defaults,
the current `contract_fingerprint`, and ref-local Apply/unblock allowances. That
initialization creates no ledger, scratch, cache, product, or retained root and
does not authorize an irreversible operation.

Read these plan fields separately:

- `selected_batch`: the complete dependency-ready census
- `selected_wave`: the complete scan-qualified, write-disjoint ready wave
- `apply_remaining`: the number of `selected_wave` refs with positive per-ref
  Apply remainder
- `allowed_parallel_applies = min(|selected_wave|, apply_remaining)`
- `dispatch_refs`: the deterministic prefix actually eligible for Apply now

Dispatch only when `check.ok=true`, `fingerprint_ready=true`, no
`pending_irreversible_policy` item remains, and `dispatch_refs` is non-empty.
Otherwise:

- semantic fingerprint drift -> keep `selected_batch` visible, empty the wave,
  and resolve the changed active-registry obligation through the authority rules
- pending irreversible policy -> keep the census visible but do not dispatch
- narrative drift, meaning `proposal.md`, `design.md`, or `specs/**` changed ->
  refresh with `python scripts/openspec_loop.py reseal <change-id>` and continue;
  under `narrative_policy: strict` treat it as a semantic pause instead
- task/feature drift, unknown dependencies, or cycles -> repair the contract
  and regenerate with `$openspec-feature-list <change-id>`; only affected refs
  are excluded from readiness
- `apply_remaining=0` -> leave `selected_wave` visible and pause only actual
  Apply dispatch; do not call this a seal or change-wide terminal
- no ready task -> summarize terminal and blocked states; do not invent work

Promotion is not drift. The `contract_fingerprint` covers the active task
registry with checkbox marks neutralized and `STATE:` directives removed, so
flipping a checkbox after `PASS` leaves the fingerprint valid and needs no
semantic restamp.

Legacy `sealed`, `confirmed_at`, `max_subagents`, and `hard_ceiling` fields are
compatibility or policy diagnostics. They are not ordinary dispatch authority.

`plan --advisory` may expose a candidate during design, but it never authorizes
execution.

## Sources of truth

- Human contract: current proposal, design, specs, and active `tasks.md`.
- Compact task state: `feature_list.json`; it must not duplicate full ACCEPT or
  TEST prose.
- Recorded execution policy and fingerprint: `loop.json`.
- Optional coordination metadata: exactly one change-local `handoff.json`.
- Disposable attempt counters: the ledger path recorded in `loop.json`.
- Prior contract wording: Git history, not an ever-growing appendix.

The ledger is never a second spec. It stores fingerprints, counters, durations,
and dispositions, not logs or product evidence.

## Execution chapters and semantic stamps

`ACCEPT` is the exact admitted acceptance text for one execution chapter. One
current `contract_fingerprint` identifies the whole active registry; it is not
one task, Apply, or ACCEPT clause. Before every active Loop CLI or
Apply/Verify/Unblock skill call, align that ACCEPT, applicable main/delta
requirements, `loop.json`, this canonical skill, and whether the action is
semantic admission, new work, closure, or unblock. Same-turn reuse is valid
only while fingerprint, narrative digest, and canonical skill state remain
unchanged. This integrity latch creates no new command or receipt.

New/omitted seal policy defaults to per-ref Apply `2`, per-ref Unblock `2`, and
change cycle stamps `3`. Task count never scales the stamp cap. A prior explicit
value is inherited when reseal omits its override; change-local values such as
`1`, `5`, or `14` are not defaults.

A charged cycle stamp is `apply-revision`, an obligation-changing semantic
reseal after the current chapter has Apply/Unblock work, or the second/later
semantic reseal in an undrained chapter. The first confirmed pre-execution
stamp, or a confirmed interviewer stamp after every prior ref is terminal,
opens a chapter outside the cycle counter. `--confirmed` alone never makes an
in-loop stamp free. Default `max_revisions=3` permits three charged stamps and
rejects the fourth before tasks, feature state, or loop state changes.

Once `loop.json.contract_fingerprint` matches the registry, the chapter is
admitted. Ordinary gate never checks `max_revisions`; a later cap reduction
does not revoke the chapter. Apply/Explore are new work and may be stopped by
active minutes, their kind allowance, and a breaker limited to the current
episode/ref/kind's last two terminal records. Record/join/Verify/promote,
sync/goals/design-verify/summary/stop-hook/review are closure and ignore those
broad stops while retaining their own prerequisites. Unblock checks only the
selected ref's blocking evidence, local allowance, and second-run evidence.

Apply workers, including direct Apply, never edit `tasks.md`. In a current
same-ref blocking or `amend_spec` window, the sole supervisor may use one
reasoned `stamp_source=unblock_self_confirm` semantic reseal per ref/source
episode, without `--confirmed`. It may only make a mechanically non-widening
ACCEPT/TEST/FILES correction inside unchanged WRITE_SCOPE. Other refs, passed
ACCEPT, DAG, framework policy, widened acceptance, budget raises, product/
external/Git authority, and a second-Unblock self-restamp return to
`$openspec-change-interviewer`.

## Routing and execution latch

Before routing every non-trivial intent or evidence-created work split, run the
proactive scan in `references/proactive-delegation-scan.md`. Record the narrowest
`matched_role_id|null`, one `direct|dispatch|blocked` decision, its named reason,
and one `effective_role_id`. A permitted direct package uses non-deployable
local `zpy` (display `ZPY`) and remains supervisor-owned with `agent=null`;
explicit `rose` is predecessor/history/bootstrap compatibility only. Neither
label is authentication authority. The scan itself consumes zero Apply and zero
scheduling headcount.

A ref enters `selected_wave` only when its package is bounded and non-trivial,
uses the narrowest applicable Role ID, has a write scope disjoint from packages
already admitted to the wave, and names a stable supervisor-owned join id.
Write overlap moves a ref to a later wave or direct serial work. It is not a
headcount blocker.

`selected_wave` is never truncated by `host_soft_cap`, `max_subagents`, distinct
worker ids, Role ids, runtime ids, or read-only worker counts. Those values are
diagnostic only. Apply dispatch is limited only through `dispatch_refs`, each
ref's Apply/unblock allowance, optional active-minute and result/semantic
breakers, and write topology.

`dispatch_refs` may contain both supervisor-direct and spawnable refs. Native
hosts must derive
`host_batch_refs = dispatch_refs where routing.decision == dispatch and agent != null`.
A direct `zpy|rose` ref stays in the supervisor process even when visible in
`dispatch_refs`; raw membership never makes it spawnable. One wave creates at
most one native host batch and at most one fresh one-shot packet per
`host_batch_ref`.

The local writer allowlist in `references/role-adapter-matrix.md` is strict:

- only `implementer` writes task-owned implementation or contract files
- only `test-engineer` writes task-owned test files
- only `browser-qa-runner` and `e2e-artifact-runner` write to an already
  approved evidence root
- every other dispatched role is read-only

No Apply role may write `openspec/changes/<change-id>/tasks.md`; semantic task
changes belong to the interviewer, full-auto `apply-revision`, or the bounded
sole-supervisor blocking-window exception above.

After gate returns continue for any `dispatch_refs`, the next substantive
action in the same turn must be Apply. Do not end with a summary, suggestion,
or handoff while that latch is active. A worker may Apply only its packet; it
must not Verify, promote, toggle a checkbox, write the ledger, or claim PASS.

One invocation uses one `run_id`; retries append attempts and must not open an
empty run per ref. The latch preserves thin retention and must not create
`BUNDLE`, `EVIDENCE`, `progress.txt`, `runs.log`, or per-attempt folders.

## Core loop

Drain the ready queue rather than stopping after one task:

1. Run `plan`; preserve the full `selected_batch`, form `selected_wave`, and
   Apply only `dispatch_refs`.
2. For each `host_batch_ref`, create one fresh
   `references/subagent-task-packet.md` envelope. One Apply packet binds exactly
   one ref and one supervisor-owned Apply attempt. Direct refs create no agent
   envelope and remain in-process.
3. Invoke `$openspec-apply-change <change-id> --task <ref> --orchestrated` for
   each `dispatch_ref`. A supervisor-direct `zpy` package follows the same
   ref/attempt/write-scope boundary without creating a worker identity; an
   explicit legacy `rose` packet is allowed only for predecessor/history or the
   bounded bootstrap task.
4. Accept one terminal `references/subagent-result.md` and have the supervisor
   write exactly one authoritative Apply record for that packet/ref/attempt;
   reject duplicates. The worker never writes the authoritative record. Every
   native host task/thread is fresh and closes at its terminal result; never use
   `Task.resume`, `resume_agent`, or a continuation recommendation.
5. Persist and complete the join for every actually dispatched member of the
   shared-worktree wave before starting any member's Verify. A missing, failed,
   partial, blocked, or unverified result blocks Verify only for its own ref
   after the barrier; it does not convert a completed sibling into failure.
6. Supervisor Verify consumes zero Apply and zero scheduling headcount. Invoke
   `$openspec-verify-change <change-id> --task <ref>` only for a ref whose own
   result is completed with inspectable evidence, and record exactly one of
   `PASS|FAIL|BLOCKED|DEVIATED`.
7. After `PASS`, promote with
   `python scripts/openspec_loop.py promote <change-id> --ref <ref>`. It flips
   the checkbox, regenerates the index, refuses any edit that would change the
   fingerprint, and re-runs the plan in one transition. Never hand-edit
   the checkbox; repair index drift alone with `sync <change-id>`.
8. Keep retry ownership distinct:
   - a worker may retry only a transient tool/process failure before its
     terminal result, inside the same unchanged packet and Apply attempt
   - an assertion or semantic failure returns terminally; only the supervisor
     may gate and issue a fresh packet for a new Apply attempt
   - unblock may return a disposition but never dispatch, swap a worker, or
     replenish an exhausted allowance
9. On `BLOCKED` or `DEVIATED`, invoke `$openspec-unblock-research <change-id>`
   in-process with host spawn zero when the ref-local gate permits it. It never
   resumes the failed Apply or spawns explorer/mapper/verifier/review. Exhaustion
   marks only that ref `maxed|stop_budget`; unrelated ready refs and later waves
   remain eligible.
   A first unblock disposition `amend_spec` MAY enter the bounded
   `unblock_self_confirm` latch. A second unblock is terminal adjudication and
   MUST NOT self-restamp or default to a third Apply.
10. Re-plan after each joined wave and continue independent work. When no ready
   task remains, close at the design level before claiming
   completion: run `goals <change-id>` for the coverage matrix, then
   `design-verify <change-id> --observation <path>` for `PASS` or `GAP`. An
   absent observation is `unobserved`, not `PASS`. A `GAP` returns a revision
   proposal; apply it under the authority rules and resume the drain loop. Only
   on `PASS` run whole-change verification and strict OpenSpec validation once.

Each dispatched packet/ref/Apply attempt receives exactly one supervisor Apply
record before a later Apply attempt for that ref; duplicates are rejected.
Zero-Apply actions retain
their own diagnostic records when useful but never manufacture Apply usage. One
invocation keeps the same run id across selected refs and retries; an
intentional resume reuses it, while a genuinely new invocation may create one
new id. Per-ref counters and optional active-minute/breaker state span those run
ids; deleted aggregate Apply-count keys stay removed and cannot return as
runtime authority.

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

## Finite and ref-local budgets

Read recorded budgets from `loop.json`; do not silently reset them after a new
run, session, or spec revision. Missing-loop initialization gives every active
ref `max_apply_attempts=2` and dormant `max_unblock_runs=2`. Unblock allowance
activates only after that ref becomes `blocked|deviated`; exhausting it marks
only that ref `maxed|stop_budget`. A stamp cannot replenish it.

`apply_remaining` is the count of refs in `selected_wave` whose recorded Apply
usage is below that ref's `max_apply_attempts`. Activated `max_unblock_runs`
remains ref-local and does not replenish Apply attempts. Goal evaluation, stop
hooks, read-only research/review, supervisor Verify, and `zpy` direct routing
consume zero Apply and zero scheduling headcount. Explicit legacy `rose` direct
routing remains zero-headcount compatibility only.

`change.max_revisions` is the current chapter's charged cycle-stamp allowance,
not a task/Apply/ACCEPT count and not an ordinary gate reason. Only actual
budget increases consume optional `max_self_extensions`; decreases do not.

Time means recorded active tool/runtime duration, not time spent waiting for a
user. Two identical result fingerprints, two consecutive `no_progress`
outcomes, or two repeated semantic deviations trigger a breaker.

`summary` uses `openspec-loop-summary.v3`. `revision_attempt_count` and
`change_attempt_count` include Apply, Verify, Explore, and Unblock records; they
must never be compared with an Apply-count limit. On the next `check`, `plan`,
or `reseal --migrate`, strip deleted Apply-count fields from legacy loop.json.
Remove only these keys:
`budgets.revision.max_iterations`, `budgets.change.max_iterations`, and
`hard_ceiling.max_iterations`. Treat them as migration residue only: do not copy
them into `apply_remaining`, an Apply gate reason, or newly written loop state.
A zero per-ref Apply remainder does not block a pending Verify; the
kind-specific `gate.decision` remains authoritative.

Legacy `host_soft_cap` and `max_subagents` remain observable compatibility
diagnostics. Optional `hard_ceiling` may retain only active-minute and
self-extension policy data. None of them filters a wave, reduces
`allowed_parallel_applies`, consumes `apply_remaining`, or creates an ordinary
Apply terminal.

## Authority

`loop.json` records the current fingerprint, policy, and autonomy. A matching
fingerprint with no pending irreversible policy is ordinary dispatch authority;
`seal-preview.md`, `confirmed_at`, and `seal --confirmed` are not start-work
gates.

If present, legacy `hard_ceiling` is optional minutes/self-extension policy data
only. It does not gate ordinary dispatch and is not ordinary start-work
authority. Raising that optional policy ceiling still requires explicit human
confirmation; no Loop raises it by itself.

| action | `supervised` (default) | `full_auto` |
|---|---|---|
| raise an activated ref-local unblock ceiling or another recorded budget | explicit user authorization and a reason | supervisor may amend below unchanged irreversible policy with a recorded reason |
| amend tasks after a design-level gap | pause and hand off to `$openspec-change-interviewer`; only a green same-ref blocking window may use one reasoned `unblock_self_confirm` | amend the registry, regenerate the index, then charged `apply-revision` or semantic reseal with a reason |
| raise optional `hard_ceiling`, or change retention, paths, or scope | human-confirmed `seal` | human-confirmed `seal` |

Autonomy never widens repository authority. Regardless of mode, destructive
external writes, credentials, product `--commit` runs, and `git push`, pull
request, or `main` operations stay human-authorized under `AGENTS.md`. A
`full_auto` Loop that needs one of those stops as `BLOCKED`.

## Bounded subordinate hosts

The supervisor may call only these two subordinate hosts:

- `silent-failure-hunting` through Role ID `silent-failure-reviewer`, for one
  bounded false-success question
- `review-pipeline`, only for user-requested review or one named specialist risk
  unresolved by direct inspection plus the smallest relevant check; it may use
  at most one auxiliary canonical Role ID

Both are read-only, non-lifecycle, and zero Apply/headcount. Their results are
advisory and do not replace task `TEST:`, final Verify, verdict, promotion, or
PASS. Neither may auto-retry, redispatch, swap roles, or start another review
swarm.

`handoff.json` is the only change-local coordination surface. Do not create a
delivery-flow operating system, A33/Board ownership, per-worker progress
journals, another slash command, a second ledger, or another supervisor.

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
2. Heavy product/run evidence under the recorded approved product or bundle
   root.
3. Disposable `{pytest,cache,tmp}/<ref>/<run-id>/` under the recorded scratch
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

`promote` requires a recorded supervisor verifier pass for that ref in the
current episode,
so the verifier stays authoritative. `apply-revision` refuses under
`autonomy: supervised` and refuses whenever `gate` already reports stop.

The recorded ledger path and fingerprint are loaded from `loop.json`; overrides
must match those values.

## Output

During a continuing loop, emit at most one progress line naming the current
wave/action, then continue with the required tool or skill call. A local
non-`PASS` blocks only its ref; after the join and disposition, continue other
eligible refs unless a fingerprint, irreversible-policy, per-ref
Apply/unblock, optional active-minute, or breaker gate pauses actual dispatch.
Consume Apply's `NEXT: supervisor_join` in the same turn. Only after the
applicable join closes may the supervisor select Verify; this is not a user
handoff.

When the loop stops, report the full ready census, selected wave, actual
dispatch refs, per-ref action/verifier verdict, applicable budget state, and
whether any tracked evidence was written. Report legacy `hard_ceiling` or
headcount fields only as compatibility context when they are present. Name the
autonomy mode whenever a budget was extended or a contract amended without a
human turn.
Never claim promotion unless the checkbox, compact feature state, and
post-promotion `check` agree.

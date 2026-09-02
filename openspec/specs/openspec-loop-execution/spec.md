# openspec-loop-execution Specification

## Purpose

Define the repository's sole finite OpenSpec execution supervisor: how an
active registry becomes an admitted execution chapter, how ready work is
routed and closed, and how bounded semantic correction remains subordinate to
the active acceptance contract.

The public harness release is OpenSpec Loop v3.2. Its machine control identity
is `openspec-loop-control.v3.2`; the compatible persisted `loop.json` schema
remains `openspec-loop.v3`, receipt remains v1, and ledger v2 remains readable.

## Requirements

### Requirement: Independent ready batch

`plan` SHALL expose the complete dependency-ready census as `selected_batch`,
partition it into a deterministic write-disjoint `selected_wave`, and report
per-ref routing plus `dispatch_refs`. `apply_remaining` SHALL count selected
wave refs with positive per-ref Apply remainder, and
`allowed_parallel_applies` SHALL equal
`min(|selected_wave|, apply_remaining)`. Headcount, Role counts,
`max_subagents`, runtime ids, and read-only workers MUST NOT truncate the wave
or consume Apply allowance.

Native hosts SHALL derive
`host_batch_refs = dispatch_refs where routing.decision == dispatch and agent != null`.
Direct local `zpy` refs remain in the supervisor process with `agent=null`.
Every Apply packet binds one ref and one supervisor-owned attempt, returns one
terminal result to `supervisor_join`, and cannot Verify, promote, resume old
context, or write the authority ledger. The sole supervisor records the result,
closes the actual-wave join, runs task Verify, and promotes a PASS.

Ref-local Apply or Unblock exhaustion SHALL stop only that ref. Failed or
missing wave members SHALL not turn a completed sibling into failure, although
every actual shared-worktree member must return before any member starts
Verify. Automatic post-implementation review/test/Verify lanes and automatic
redispatch are forbidden.

#### Scenario: Direct refs never enter a native host batch

- **WHEN** `dispatch_refs` contains both direct `zpy` and spawnable routed refs
- **THEN** the native host batch SHALL contain only refs whose decision is
  `dispatch` and whose agent is non-null
- **AND** direct refs SHALL remain supervisor-owned

#### Scenario: A shared wave joins before independent task verification

- **WHEN** multiple Apply refs share one write-disjoint worktree wave
- **THEN** every actually dispatched result SHALL reach `supervisor_join`
  before any task Verify starts
- **AND** each task SHALL then Verify and promote independently

### Requirement: Semantic stamp admission and execution completion

The Loop SHALL distinguish the acceptance ruler, an execution chapter, a
charged cycle stamp, and a chapter-outside stamp. `ACCEPT:` SHALL mean the
active task-registry acceptance text admitted at chapter start. A single
`contract_fingerprint` SHALL identify the whole active registry rather than one
task, one ACCEPT line, or one attempt.

New seal and missing-loop initialization SHALL default to per-ref
`max_apply_attempts=2`, per-ref `max_unblock_runs=2`, and
`change.max_revisions=3`. They SHALL also default
`revision.max_explore_runs=1`, `revision.max_active_minutes=120`, and
`change.max_active_minutes=360`. Task count MUST NOT scale
`change.max_revisions`. Reseal without a corresponding override SHALL inherit
an existing recorded value rather than replace it with a default.

The helper SHALL classify every obligation-changing semantic stamp before
writing `tasks.md`, `feature_list.json`, or `loop.json`. `apply-revision`, a
semantic reseal after the current fingerprint episode has Apply or Unblock
work, and the second or later semantic reseal in one undrained execution
chapter SHALL be charged cycle stamps. A supervised confirmed first stamp
before execution, or after all prior chapter refs are terminal, SHALL be a free
chapter-outside stamp. `--confirmed` alone MUST NOT exempt an otherwise charged
stamp.

A charged stamp SHALL be refused before writes when the current chapter's
charged count is greater than or equal to the candidate
`change.max_revisions`. Successful charged stamps SHALL increment the count
once regardless of the number of changed tasks or ACCEPT lines. Initial
ACCEPT, Apply, Explore, Verify, Unblock, narrative reseal, checkbox/STATE, and
promotion SHALL NOT themselves increment it. A matching current fingerprint
SHALL remain admitted when a cap is later lowered.

Ordinary `gate --kind apply|explore` SHALL check fingerprint, active minutes,
its kind-specific allowance, and breakers scoped to the current episode, same
ref, same kind, and last two terminal records. It MUST NOT check
`max_revisions`. `gate` SHALL require `--kind`; Apply, Explore, Verify, and
Unblock SHALL require `--ref`.

`record`, join, Verify, promote, sync, goals, design-verify, summary, stop-hook,
and review SHALL have completion right and MUST NOT be stopped by
`max_revisions`, active-minute caps, or generic result breakers. They SHALL
still enforce their own fingerprint, join, evidence, and verifier preconditions.

Unblock gate SHALL check only the selected ref's blocking evidence,
`max_unblock_runs`, and the second-unblock new-evidence/terminal rule. It MUST
NOT be stopped by revision/change minutes, `max_revisions`, or a generic
breaker. Ref-local exhaustion MUST NOT freeze siblings.

Only a true budget increase SHALL consume optional self-extension allowance.
A reduction SHALL not consume it and MUST NOT revoke an already admitted
fingerprint.

#### Scenario: The fourth cycle stamp is refused before writes

- **GIVEN** one execution chapter has three charged cycle stamps and
  `change.max_revisions=3`
- **WHEN** a fourth charged semantic reseal or `apply-revision` is requested
- **THEN** the command SHALL reject before changing tasks, feature state, or
  loop state
- **AND** an ordinary narrative reseal SHALL remain allowed

#### Scenario: Completion survives a cap crossed by admitted work

- **GIVEN** the current fingerprint is admitted and its Apply result is
  recorded
- **WHEN** historical episode counts or a later lower cap exceed the recorded
  revision ceiling
- **THEN** Verify, promotion, goals, design verification, and other closure
  actions SHALL remain eligible
- **AND** a later ready ref in the same fingerprint SHALL not pay another
  cycle stamp

#### Scenario: Breakers do not leak across refs or kinds

- **WHEN** the latest two terminal results belong to another ref, another kind,
  or another fingerprint episode
- **THEN** they SHALL NOT trigger the current Apply or Explore breaker

#### Scenario: Existing policy is inherited instead of becoming the default

- **GIVEN** an existing change explicitly records non-default values such as
  `max_unblock_runs=1` or `max_revisions=5|14`
- **WHEN** reseal omits the corresponding overrides
- **THEN** those values SHALL be inherited for that change
- **AND** new changes SHALL still receive the repository defaults `2|2|3`

### Requirement: Supervised blocking-window self-restamp

Apply workers MUST NOT edit `tasks.md`. Under supervised autonomy, the sole
supervisor MAY perform one unconfirmed semantic reseal for one blocked ref in
one source fingerprint only when the current episode has same-ref
`blocked|deviated` evidence or a same-ref Unblock disposition `amend_spec`.
This path SHALL require `stamp_source=unblock_self_confirm`, the ref, a
candidate registry, and a non-empty reason. It SHALL be a charged cycle stamp
whenever the cycle rules apply.

The candidate SHALL preserve the active ref set, task order, DAG,
dependencies, `SUPERSEDES`, GOALs, roles, execution/join/worktree metadata, and
WRITE_SCOPE. It MAY only narrow or mechanically correct the selected ref's
ACCEPT, executable TEST, FILES within unchanged WRITE_SCOPE, or one
failure-local sentence. It MUST reject any change to another ref, a passed
ACCEPT, retention, paths, product/bundle/GUI, autonomy, hard ceiling, budget
raise, DAG, WRITE_SCOPE expansion, proposal scope, GOAL, external destructive
authority, Git/PR/main, or an acceptance relaxation.

The same ref MUST NOT self-restamp twice from one source fingerprint. A second
Unblock SHALL be terminal adjudication and MUST NOT open another self-restamp
or default retry under the repository-default two-Apply allowance. A denied
self-restamp SHALL leave `tasks.md`, `feature_list.json`, and `loop.json`
byte-identical and route to the interviewer.

#### Scenario: One narrowed blocked ref self-restamps without confirmation

- **GIVEN** a non-passed ref has current-episode blocking authority and has not
  used the exception from this source fingerprint
- **AND** its candidate changes only an allowlisted, mechanically
  non-widening field
- **WHEN** the sole supervisor reseals with
  `stamp_source=unblock_self_confirm`, the ref, and a reason but no
  `--confirmed`
- **THEN** the semantic fingerprint SHALL update atomically
- **AND** one cycle stamp SHALL be charged when the cycle rules apply
- **AND** no Apply, Unblock, minute, or other budget value SHALL be raised

#### Scenario: Framework and acceptance widening leave all files unchanged

- **WHEN** the candidate widens ACCEPT, edits another ref, changes the DAG or
  WRITE_SCOPE, touches framework policy, or reuses the exception
- **THEN** self-restamp SHALL fail
- **AND** tasks, feature state, and loop policy SHALL remain byte-identical
- **AND** the decision SHALL return to the interviewer

### Requirement: Bounded same-obligation research repair

The Loop SHALL preserve `unblock_self_confirm` and provide a separate charged
`stamp_source=repair_r1` only for a non-passed ref whose active task declares
`REPAIR_POLICY: bounded-r1`, whose explicit ref-local Apply budget is exactly
three,
and whose latest Unblock records `amend_spec` with `repair_class=R1`.

R1 SHALL preserve proposal and specs bytes, GOAL/COVERED_BY, ref set and order,
DAG, roles, FILES, WRITE_SCOPE, paths, retention, autonomy, hard ceiling,
budgets, other refs, and the hard quantities, product path, and fail-closed
semantics of the selected ACCEPT. It MAY change only one selected-ref method
marker or one unambiguous legacy method section, executable TEST Run with all
non-Run TEST text and SCOPE unchanged, and selected state from blocked/deviated
to pending. Marker pairs SHALL be ref-unique, non-nested, non-crossing, and
free of headings or contract directives. An
explicit-marker ref SHALL keep ACCEPT outcome-only and structurally unchanged;
legacy compatibility MAY replace only a mechanically unique method token while
preserving the stable obligation signature. It SHALL reject product `--commit`
or any relaxation.

The supervisor SHALL independently compute a stable obligation hash and SHALL
atomically write candidate design, tasks, generated feature state, and loop
state before strict validation and re-plan. Failure SHALL restore all four
files. Success SHALL record source/target fingerprints, carry Apply/Unblock
usage across that lineage, charge one cycle stamp, and grant exactly one
post-repair Apply. No obligation SHALL receive a second R1 or third Unblock.

#### Scenario: Second Unblock repairs once and executes once

- **GIVEN** two distinct blocking Apply results and two admitted Unblock runs
- **AND** the selected ref is pre-authorized for bounded R1 and Apply three
- **WHEN** the second Unblock classifies a same-obligation method defect as R1
  and the supervisor admits the candidate
- **THEN** prior usage SHALL remain two Apply and two Unblock
- **AND** exactly one post-repair Apply SHALL be dispatchable
- **AND** a failed post-repair Apply SHALL make only that ref `stop_budget`

#### Scenario: Missing or broad repair classification is R2

- **WHEN** `amend_spec` omits `repair_class`, the marker is ambiguous, the
  obligation hash changes, or task succession is required
- **THEN** the Repair Gate SHALL classify the change as R2
- **AND** the supervisor SHALL invoke the interviewer with spawn zero while
  Unblock itself only returns the disposition
- **AND** the interviewer SHALL create or refresh the seven-section
  `interview.md` before asking only the material question
- **AND** no new change SHALL be scaffolded without a separate user decision

### Requirement: Execution notebook ownership remains explicit

An explicitly adopted task-owned execution notebook profile SHALL use one
short protocol Markdown cell and three output-clean code cells tagged
`openspec-params`, `openspec-run`, and `openspec-outputs`. Apply SHALL own heavy
outputs, any run-local executed notebook, and a manifest containing `schema`,
`exec_block`, `run_level`, `source_notebook_sha`, `canonical_params`,
`params_sha`, `started_at`, `ended_at`, `status`, total/completed/resumed/failed
unit counts, output paths and hashes, and `resume_decision`. Verify SHALL be
read-only and SHALL NOT modify source notebook outputs, manifest, or product
data after PASS.

#### Scenario: Resume requires hashes rather than path existence

- **WHEN** a prior unit path exists
- **THEN** the unit SHALL be skipped only if completed state, canonical params
  hash, and every required output hash match
- **AND** stale, partial, failed, or hash-mismatched units SHALL remain pending
- **AND** progress bars SHALL remain observability rather than acceptance

### Requirement: Compiled next action and shadow parity

The Loop SHALL expose `next --intent drain|review|explore|repair` as the compact
control-plane entry. One invocation SHALL observe the registry once and return
one typed `next_action`, selected ref, census summary, deterministic routing,
budget state, and Apply receipts when applicable. `check` SHALL remain the CI
validity surface and `plan` the verbose compatibility/debug surface; a worker
MUST NOT call both before Apply.

`next --shadow` SHALL perform no policy initialization, ledger write, receipt
authorization, product write, or retained evidence write. It SHALL report
legacy parity and ablation fields while any emitted receipt remains
non-authoritative.

#### Scenario: Explicit non-drain intent bypasses the drain latch

- **WHEN** the caller selects review, explore, or repair intent
- **THEN** `next_action` SHALL preserve that intent
- **AND** ready Apply refs SHALL remain observable without forcing Apply

### Requirement: Machine receipts eliminate nested census

An authoritative Apply receipt SHALL bind receipt schema, machine
`harness_contract_hash`, change and contract fingerprint, narrative digest,
run/ref/attempt/packet, Role, target files, WRITE_SCOPE, write policy, join and
wave, worktree mode, and TEST_LEVEL. The harness hash SHALL cover machine
control schemas, budget defaults, repair classes, role/write policies, and test
levels rather than hashing the prose skill.

Active next SHALL register issued receipt claims and compact control/ablation
events in the existing sole ledger; shadow SHALL register neither. Record SHALL
consume exactly one issued claim, reject replay, and create no second ledger.
The model-visible token SHALL be a short opaque receipt id; full target and
WRITE_SCOPE snapshots remain in the ledger claim rather than a base64 prompt.

`receipt-check` SHALL fail closed on malformed data, shadow authority,
contract/narrative/harness drift, ref/run/action mismatch, or irreversible
policy. Apply SHALL validate its receipt instead of invoking check/plan.
`record --receipt` SHALL use the frozen packet snapshot and MUST NOT rebuild the
whole plan. Legacy record without a receipt MAY retain the old census path for
compatibility.

#### Scenario: Apply result is also the typed join event

- **WHEN** the sole supervisor records a terminal Apply result with its receipt
- **THEN** one canonical ledger record SHALL store receipt id, terminal status,
  join id, wave refs, evidence, and routing outcome
- **AND** the response SHALL report typed join state and per-ref Verify
  eligibility
- **AND** natural-language handoff text SHALL grant no control authority

### Requirement: Mechanical and semantic verification are separated

Tasks MAY declare `TEST_LEVEL: smoke|pilot|production|canary`; the compact
feature registry SHALL preserve that optional field without a schema bump.
`verify-mechanical` SHALL execute only mechanically compilable inline CLI Run
commands after a completed joined Apply and SHALL retain only command/output
hashes, exit status, duration, and typed verdict in the ledger.

Smoke mechanical PASS MAY satisfy promotion authority. Pilot, production,
canary, and unspecified legacy tasks SHALL require semantic Verify after
mechanical PASS. Semantic Verify SHALL judge ACCEPT direction, source, metric,
schema, evidence, and `DEVIATED` without rerunning the mechanical command merely
to reproduce its receipt.

A non-CLI GUI/Colab/MIXED contract without a mechanical Run MAY return
`SKIPPED` and route to semantic Verify; a CLI contract without a compilable Run
SHALL be `BLOCKED`. Mechanical Verify MUST NOT automatically execute product
`--commit`, Git push/PR, or recursive deletion commands.

#### Scenario: Ablation measures control reduction per ref

- **WHEN** next, record, Verify, and summary outputs are collected for a run
- **THEN** they SHALL expose census count/routing source, current supervisor
  card bytes, misroute count, false-success count, and semantic deviations
- **AND** evaluation SHALL compare hops, prompt bytes, census calls,
  misrouting, and false-success rather than relying on a subjective speed claim

### Requirement: New Apply records are receipt-only

`record --kind apply` SHALL require an authoritative unconsumed receipt issued
by active `next`. Missing receipt SHALL fail with
`apply_record_requires_receipt` before task lookup, plan construction, loop
initialization, or ledger write. Runtime MAY read and summarize historical
legacy Apply rows but MUST NOT create a new `legacy:*` attempt or reconstruct
packet authority through a full census.

#### Scenario: Missing receipt leaves runtime state untouched

- **WHEN** a caller submits an Apply record without `--receipt`
- **THEN** the command SHALL fail with `apply_record_requires_receipt`
- **AND** `loop.json`, ledger bytes, tasks, and feature state SHALL remain
  unchanged

### Requirement: Promotion is a targeted completion transition

Promotion SHALL read only the current fingerprint, unique selected task and
feature, direct dependency PASS state, and current-episode verifier PASS. It
MUST NOT call `build_plan_payload` before or after the transition. Promotion
SHALL atomically update the checkbox and compact feature state, verify that the
fingerprint, active ref set, and ACCEPT/TEST hashes did not change, and restore
tasks, feature, and ledger bytes on failure.

Successful `openspec-loop-promote.v2` output SHALL contain
`promoted=true`, `ref`, `contract_fingerprint`, and `next_required=true`; it
MUST NOT publish a reconstructed ready/terminal queue. The caller SHALL use a
fresh `next` for subsequent routing.

#### Scenario: Promotion cannot recreate the reducer

- **WHEN** a verified ready ref is promoted while `build_plan_payload` is
  unavailable
- **THEN** the targeted transition SHALL still succeed
- **AND** a fresh `next` SHALL be required to observe downstream readiness

### Requirement: Labeled ablation is the metric authority

Live summary SHALL expose recorded control/work hops and CLI control calls.
Because the CLI cannot observe skill-internal model calls, live
`llm_control_call_count` SHALL default to zero with
`llm_control_visibility=false`. It MUST NOT be presented as a measured absence.

`ablation-evaluate --manifest` SHALL validate an
`openspec-loop-ablation-suite.v1` ordered event manifest with unique case ids,
64-character ACCEPT/input hashes, evidence refs, known hop/executor/event
enums, and contiguous event sequence numbers. Its v1 report SHALL calculate
DEVIATED TP/FN/FP/TN and same-ACCEPT/input consistency. Detection rate SHALL be
null without positive labels; consistency rate SHALL be null without a group
of at least two comparable runs. Event prevalence MUST NOT be relabeled as
detection rate.

#### Scenario: Repeated ACCEPT has conflicting outcomes

- **WHEN** two labeled cases share ACCEPT and input fingerprints but return
  different observed verdicts
- **THEN** the evaluator SHALL count one eligible inconsistent group
- **AND** list the conflicting case ids without rewriting either verdict

### Requirement: Supervisor skill remains an interface card

The canonical Loop SKILL SHALL contain only invocation, typed branch routing,
receipt usage, verification split, and non-negotiable authority invariants.
Detailed lifecycle explanations SHALL remain in runtime, tests, this
specification, AGENTS, and the public guide. Compressing the skill MUST NOT
remove R1, budget, WRITE_SCOPE, evidence isolation, DEVIATED, retention, or
human authority from those machine-tested surfaces.

#### Scenario: Skill invocation loads an interface rather than a second reducer

- **WHEN** the Loop supervisor skill is invoked for an admitted change
- **THEN** it SHALL direct the model to `next --intent` and the returned typed
  branch
- **AND** it SHALL not require the model to recalculate census, routing, budget,
  receipt, or join decisions already produced by the CLI

### Requirement: Research skill discovery remains intent-bound and non-authoritative

A change MAY record repo-relative directories in
`loop.json.control_plane.research_skill_roots`. These roots SHALL be treated as
advisory discovery hints without weights: the supervisor reads a listed skill
only when the selected task's intent requires that capability. Directory order
or existence MUST NOT change ref/Role routing, receipt validity, WRITE_SCOPE,
budget, retention, product authority, or external permissions. The Loop MUST
NOT preload the directories into a second catalog, and an unavailable optional
root MUST NOT block unrelated ready refs.

#### Scenario: A notebook skill is available for one research task

- **WHEN** a change lists a repo-local notebook skill root and the selected ref
  explicitly adopts an execution-notebook profile
- **THEN** the supervisor MAY read that skill for the selected ref
- **AND** `next`, the receipt, TEST_LEVEL, and WRITE_SCOPE SHALL remain the
  execution authority
- **AND** refs that do not require notebooks SHALL not load or depend on it

### Requirement: Per-call authority alignment

Before each work node, the compiled control plane SHALL align current contract
fingerprint, narrative digest, `loop.json`, machine harness hash, receipt, and
action class. Apply SHALL read only its receipt-bound ACCEPT and applicable
requirements; semantic Verify, Unblock, and interviewer SHALL additionally read
the semantic evidence relevant to their node. A mismatch SHALL pause the call
and route to interviewer, narrative reseal, skill sync, or Unblock as
applicable. The model MUST NOT reread and reinterpret the full supervisor skill
to duplicate a successful machine alignment.

#### Scenario: An unchanged same-turn contract reuses the alignment result

- **WHEN** fingerprint, narrative digest, harness hash, and receipt remain
  unchanged
- **THEN** the supervisor MAY reuse its alignment result
- **AND** any change to those inputs SHALL require a fresh `next` or
  `receipt-check`

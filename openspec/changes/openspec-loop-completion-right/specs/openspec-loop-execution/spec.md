## MODIFIED Requirements

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

### Requirement: Per-call authority alignment

Before every active Loop command or Apply/Verify/Unblock skill invocation, the
sole supervisor SHALL align the current ACCEPT text, applicable main and delta
requirements, `loop.json`, the canonical skill, and whether the action is
semantic admission, new work, closure, or unblock. A mismatch SHALL pause the
call and route to the applicable interviewer, narrative reseal, skill sync, or
unblock path. This alignment MUST NOT create a new command, receipt, ledger,
worker, or lifecycle.

#### Scenario: An unchanged same-turn contract reuses the alignment result

- **WHEN** fingerprint, narrative digest, and canonical skill state remain
  unchanged in the same supervisor turn
- **THEN** the supervisor MAY reuse its alignment result
- **AND** any change to those inputs SHALL require a fresh check

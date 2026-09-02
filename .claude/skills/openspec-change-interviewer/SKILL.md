---
name: openspec-change-interviewer
description: Clarify and write back an OpenSpec change with bounded grilling, measurable acceptance, and a current task-registry fingerprint. Use for new ideas, major revisions, additive extensions, succession, irreversible retention/path/scope decisions, or semantic drift. Preview and stamp fields are optional audit diagnostics, not ordinary start-work gates.
---

# OpenSpec Change Interviewer

Own clarification and current-contract write-back under
`openspec/changes/<change-id>/`. Do not implement product code or promote task
state. This skill updates the one OpenSpec contract; it does not create another
spec lifecycle, ledger, supervisor, or handoff surface.

Ordinary start-work readiness is a matching `contract_fingerprint` with no
pending irreversible policy decision. If `loop.json` is missing, one `check`
records thin defaults and the current fingerprint without creating any
ledger, scratch, cache, product, or retained root. Legacy `sealed`,
`confirmed_at`, `seal-preview.md`, and stamp fields are optional audit
diagnostics, not start-work authority.

Load on demand from `references/`: `seal-preview-format.md`,
`artifact-placement.md`, `acceptance-quality.md`, `interview-packet-format.md`,
`context-discipline.md`, `evidence-matrix.md`.

## Invocation

`$openspec-change-interviewer <change-id>`

If the change does not exist, ask before scaffolding. If the user explicitly
waives interviewing, write only confirmed facts and retain unresolved items as
`Open Question:`; unresolved scope or irreversible policy keeps the affected
boundary not ready.

## Phase A: index before content

Start with a cheap inventory:

- paths, byte sizes, headings, task refs, checkbox/state directives
- current `feature_list.json` schema and ref set
- `loop.json` fingerprint, retention profile, paths, and budgets when present
- for a semantic amendment, any ref-local allowance or optional runtime advisory
  from the compact shadow action or reseal
- strict OpenSpec validation status when available

Read `tasks.md` first. Load only the proposal/design/spec sections needed to
resolve the active gap. Use targeted search and heading slices.

When invoked from Loop for `repair_class=R2` or any `supersede_task`, the first
write is to create or refresh `interview.md` using the seven sections in
`references/interview-packet-format.md`. Populate evidence, current decision
context, accepted decisions, recorded fingerprint/retention/paths/budgets/
autonomy, active ACCEPT, and Unblock expected/observed before asking a question.
Preserve accepted answers and Section 7 history. `seal-preview.md` never
substitutes for this packet.

Classify into exactly one entry mode:

| Mode | When | Readiness path |
|---|---|---|
| `new-idea` | S1 fuzzy / first contract | full grill → strict validation → fingerprint readiness |
| `major-revision` | S2 design/obligation material-delta | delta grill → rebuild registry and fingerprint |
| `additive-extension` | S3 append tasks / pilot→batch | grill new tasks only → rebuild the full registry |
| `succession` | S4 upgrade from main/archive specs | lock behavior → rebuild the full registry |
| `execution-ready` | fingerprint still matches | one-line policy reuse; no path re-ask |

Also route output deviation to `$openspec-unblock-research` and residue cleanup
to `$openspec-hygiene`.

## Phase B: boundary grilling

Ask material questions only. Prefer path **profile** letters over five free-form
roots (`references/seal-preview-format.md`). Recommend `thin` + `A_local_thin`
for ordinary local work; never default a new change to another change's
external absolute tree.

Record retention, path profile, evidence depth, test profiles, budgets, and
`autonomy` in the existing change contract. Use the optional `seal-preview.md`
only when an audit diagnostic helps or an irreversible policy choice needs
human confirmation; do not make it a second ceremony.

For a semantic amendment, preserve each ref's `max_apply_attempts` and dormant
`max_unblock_runs` unless the amendment explicitly changes that ref's policy.
`apply_remaining` is the number of selected-wave refs with positive per-ref
Apply remainder. Deleted Apply-count fields are migration residue only: ignore
them on read, drop them on the next `check`, `plan`, or `reseal --migrate`,
and never reintroduce them into written loop state. Optional active-minute or
breaker advisories may still pause dispatch; raising an activated ref-local
ceiling remains a separate authority decision.

If a valid `loop.json` exists and its fingerprint still matches, show a
one-line policy summary and reuse it. Do not ask again about D-drive or cache
paths. If only some fields drifted, ask only those fields.

## Phase C: current-contract write-back

Write only the operative contract: proposal, design, specs, tasks (and optional
`context.md` / ADR per references). Use Git history for superseded wording.

Loop-ready directives: `DEPENDS_ON`, `INDEPENDENT`, `NO_DEP`, `STATE`,
`SUPERSEDES`, optional `FILES:`.

Design obligations as optional `GOAL:` blocks:

```text
- GOAL G1: every prefecture-year surface closes without --commit
  - COVERED_BY: R7-R16
  - ACCEPT: 22 city-year pairs present in the retained manifest
```

Every active task needs `[#R...]`, `ACCEPT:`, and executable `TEST:`. After task
changes:

```powershell
python scripts/generate_openspec_feature_list.py <change-id>
```

### Mode-specific design-change bridges

**major-revision.** Classify deltas:
`covered | material-question | material-delta | ordinary-steering | unverified`.
Only `material-delta` changes the semantic fingerprint. Ordinary steering or
narrative-only design edits → `reseal` (digest only). Material-delta → update
tasks (`SUPERSEDES` as needed) → regenerate the full active registry.

**additive-extension.** Append-only. Grill new ACCEPT/TEST/placement. Regenerate
the registry with **old and new** refs together. Reuse the path profile unless
the extension needs a new product root.

**succession.** Read main `openspec/specs/<capability>/` + archive; express
`## MODIFIED Requirements`; lock current behavior with tests; then full-registry
fingerprint readiness.

## Validate and establish readiness

Natural flow after grill write-back:

1. `openspec validate <change-id> --strict`
2. `python scripts/generate_openspec_feature_list.py --change-id <change-id>`
3. When `loop.json` is missing, run `check` once for thin initialization; do not
   call verbose `plan`.
4. When `loop.json` exists, run only `next --shadow` for compact readiness
   inspection. Confirm a matching fingerprint, no issues, and no
   `pending_irreversible_policy`. Ordinary active next may issue Apply receipts
   at this latch without a preview, stamp, or user ceremony.
5. If the user requests audit diagnostics, or the change raises
   `hard_ceiling`, changes retention, paths, or scope, prepare the optional
   change-scoped `seal-preview.md` per `references/seal-preview-format.md`.
   Obtain explicit human confirmation only for that irreversible policy, then
   persist it with the narrow applicable options.

```powershell
# Missing-loop initialization only:
python scripts/openspec_loop.py check <change-id>
# Ordinary readiness inspection:
python scripts/openspec_loop.py next <change-id> --intent drain --run-id <id> --shadow
# Optional irreversible-policy/audit persistence only:
python scripts/openspec_loop.py seal <change-id> --confirmed [explicit policy options]
```

Recommend `supervised` unless the user asked for Loop self-amend without a new
interview; recommend `full_auto` only for repository-source deliverables.

A ref-local or optional-runtime advisory does not mutate `loop.json`, does not
fail the readiness latch, and is not a `check` warning.

When `execution-ready` and the fingerprint still matches, one-line reuse — do
not regenerate path questions. Do not ask again about D-drive, cache, or bundle
paths. An irreversible policy change alone requires explicit confirmation.

Neither thin initialization nor optional preview/stamp diagnostics create
ledger, scratch, product, bundle, or GUI/Colab directories.

## Execution chapters and stamp classification

`ACCEPT` is the task ruler admitted at the start of one whole-registry
execution chapter. The initial ACCEPT, Apply/Verify/Unblock, promotion,
checkbox/`STATE:`, and narrative reseal do not consume a cycle stamp. New and
omitted policy defaults are Apply `2`, Unblock `2`, and chapter cycle stamps
`3`; task count never scales the stamp cap, while omitted overrides inherit an
existing change's recorded values.

Before writing a semantic fingerprint, classify the stamp. `apply-revision`, a
reseal after current-chapter Apply/Unblock work, and the second/later semantic
reseal in an undrained chapter are charged. The first supervised confirmed
semantic stamp before execution, or after all prior chapter refs are terminal,
is chapter-outside and free. `--confirmed` by itself never makes a charged
stamp free. The fourth charged stamp at the default cap is refused before
authority files change; narrative reseal remains available.

Once a fingerprint is current, ordinary gate does not recheck the stamp cap and
a later reduction does not revoke it. Historical emergency values such as
`max_unblock_runs=1` or `max_revisions=5|14` remain change-local inheritance,
not repository defaults.

## Amendment during execution

Narrative drift in `proposal.md`, `design.md`, or `specs/**` is not a semantic
amendment. Direct the caller to
`python scripts/openspec_loop.py reseal <change-id>` (digest refresh; keep
retention, paths, budgets, test profiles). Only a semantic change to the active
task registry needs the authority branch below.

When Loop reports semantic fingerprint drift, `DEVIATED`, `amend_spec`, or
`supersede_task` under `autonomy: supervised`:

1. pause the affected dispatch and preserve recorded counters
2. compare observed vs ACCEPT / last good baseline
3. ask only the decision that changes scope, output authority, tasks, or budget
4. update the contract and feature index
5. reseal the fingerprint through the authorized path, then run one
   `next --shadow`

The Loop Repair Gate classifies a pre-authorized same-obligation method repair
as R1 and handles it without this skill. Missing/ambiguous classification,
GOAL or ACCEPT-boundary change, DAG/ref/path/retention/budget/external authority
change, and every task succession are R2. For R2, ask only the material gap;
reuse recorded local or B_external_heavy paths without re-asking. If no policy
exists, prefill thin + A_local_thin, Apply 2, Unblock 2, cycle stamps 3, and
supervised. The user may answer `同意默认`. Succession may be recommended but
must not scaffold a new change without a separate user decision.

Under `autonomy: full_auto` the sole supervisor may amend the active registry
and run `reseal --allow-semantic-change --reason <reason>`. Under `supervised`,
return here for the material decision except for the one bounded blocking-window
exception: a non-passed same-ref `blocked|deviated` or `amend_spec` disposition
may let the sole supervisor use one reasoned
`stamp_source=unblock_self_confirm` candidate per source episode without
`--confirmed`. It may only make a mechanically non-widening correction to that
ref's ACCEPT/TEST/FILES inside unchanged WRITE_SCOPE. Any other ref, DAG,
passed ACCEPT, acceptance widening, framework policy, budget raise, external/
product/Git authority, or a second-Unblock self-restamp remains here. Stay
involved only for scope, retention, paths, output authority, or another
decision reserved for the user.

Ref-local Apply/unblock exhaustion is local and cannot be reset by a stamp.
Raising an activated ref-local ceiling requires the applicable authority and a
recorded reason. If present, legacy `hard_ceiling` is optional
minutes/self-extension policy data only; it is neither an Apply gate nor an
ordinary start-work authority surface. Raising that optional policy ceiling
still requires human confirmation.

## Handoff

- design open → stay here or `$openspec-explore`
- fingerprint-ready ordinary execution → `$openspec-loop-engineering <change-id>`
- one task without Loop → `$openspec-apply-change <change-id>`
- residue → `$openspec-hygiene`
- full audit → `$monitor-openspec-codex <change-id>`

Report changed artifacts, fingerprint/readiness result, ref coverage count,
unresolved questions, any optional policy diagnostic, and the exact next
command. Use exactly one change-local `handoff.json` if coordination metadata is
needed; do not create another lifecycle or supervisor.

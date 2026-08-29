---
name: openspec-change-interviewer
description: Clarify and write back an OpenSpec change with grilling, then stamp (seal) start-work authority for the Loop via a natural seal-preview ceremony covering the whole active task registry. Use for new ideas, major revisions, additive extensions, succession, retention/path decisions, or semantic drift. Do not repeatedly ask unchanged storage questions.
---

# OpenSpec Change Interviewer

Own clarification and current-contract write-back under
`openspec/changes/<change-id>/`. Do not implement product code or promote task
state.

**Seal = 开工盖章**（start-work stamp）: freeze the whole active registry's
obligations plus path profile, retention, budgets, `autonomy`, and
`hard_ceiling` into `loop.json`. It does **not** check off tasks; `promote`
does. One stamp covers **every** active `[#R…]` at once.

Grilling's last turn **is** the stamp: accept `seal-preview.md`, then run
`seal --confirmed` only to freeze that acceptance. Do not invent a second
"please seal" conversation after the user already agreed to the preview.

Load on demand from `references/`: `seal-preview-format.md`,
`artifact-placement.md`, `acceptance-quality.md`, `interview-packet-format.md`,
`context-discipline.md`, `evidence-matrix.md`.

## Invocation

`$openspec-change-interviewer <change-id>`

If the change does not exist, ask before scaffolding. If the user explicitly
waives interviewing, write only confirmed facts and retain unresolved items as
`Open Question:`; do not stamp an unresolved execution boundary.

## Phase A: index before content

Start with a cheap inventory:

- paths, byte sizes, headings, task refs, checkbox/state directives
- current `feature_list.json` schema and ref set
- `loop.json` fingerprint, retention profile, paths, and budgets when present
- strict OpenSpec validation status when available

Read `tasks.md` first. Load only the proposal/design/spec sections needed to
resolve the active gap. Use targeted search and heading slices.

Classify into exactly one entry mode:

| Mode | When | Stamp path |
|---|---|---|
| `new-idea` | S1 fuzzy / first contract | full grill → first-seal preview (all refs) |
| `major-revision` | S2 design/obligation material-delta | delta grill → semantic-restamp (all refs again) |
| `additive-extension` | S3 append tasks / pilot→batch | grill new tasks only → restamp full registry |
| `succession` | S4 upgrade from main/archive specs | lock behavior → restamp full registry |
| `execution-ready` | fingerprint still matches | one-line reuse; no path re-ask |

Also route output deviation to `$openspec-unblock-research` and residue cleanup
to `$openspec-hygiene`.

## Phase B: boundary grilling

Ask material questions only. Prefer path **profile** letters over five free-form
roots (`references/seal-preview-format.md`). Recommend `thin` + `A_local_thin`
for ordinary local work; never default a new change to another change's
external absolute tree.

Compress retention, path profile, evidence depth, test profiles, budgets, and
`autonomy` + `hard_ceiling` into the eventual seal-preview packet rather than a
disconnected second ceremony.

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
tasks (`SUPERSEDES` as needed) → stamp preview listing **all** active refs.

**additive-extension.** Append-only. Grill new ACCEPT/TEST/placement. Restamp
preview must list **old and new** refs together (full-registry stamp, not
per-task seals). Reuse path profile unless the extension needs a new product
root.

**succession.** Read main `openspec/specs/<capability>/` + archive; express
`## MODIFIED Requirements`; lock current behavior with tests; then full-registry
stamp.

## Validate and stamp (seal)

Natural flow after grill write-back:

1. `openspec validate <change-id> --strict`
2. `python scripts/openspec_loop.py plan <change-id> --advisory`
3. Build `openspec/changes/<change-id>/seal-preview.md` per
   `references/seal-preview-format.md`: list **every** active ref under
   "一次性全盖", path profile expansion for this change-id, trees, policy.
   Show it in chat as the final grilling question — not a separate chore.
4. User accepts the packet (agree stamp / switch profile / adjust policy /
   do not stamp). Permission and destructive ops stay alone.
5. Only after acceptance, freeze and verify:

```powershell
python scripts/openspec_loop.py seal <change-id> --confirmed --retention <none|thin|full> --ledger-path <path> [--autonomy supervised|full_auto] [--hard-ceiling-max-iterations N] [confirmed path and budget options]
python scripts/openspec_loop.py check <change-id>
```

Omitted options inherit the prior seal; omitted `hard_ceiling` derives headroom.
Recommend `supervised` unless the user asked for Loop self-amend without a new
interview; recommend `full_auto` only for repository-source deliverables.

When `execution-ready` and the fingerprint still matches, one-line reuse — do
not regenerate path questions. Policy change → rewrite preview → re-confirm.

Do not create ledger/scratch/product/bundle directories merely by sealing.

## Amendment during execution

Narrative drift in `proposal.md`, `design.md`, or `specs/**` is not a semantic
amendment. Direct the caller to
`python scripts/openspec_loop.py reseal <change-id>` (digest refresh; keep
retention, paths, budgets, test profiles). Only a semantic change to the active
task registry needs a new stamp ceremony.

When Loop reports semantic fingerprint drift, `DEVIATED`, `amend_spec`, or
`supersede_task` under `autonomy: supervised`:

1. pause and preserve the disposable ledger
2. compare observed vs ACCEPT / last good baseline
3. ask only the decision that changes scope, output authority, tasks, or budget
4. update the contract and feature index
5. rewrite `seal-preview.md` with the **full** active ref list → user accepts →
   `seal --confirmed` (or `reseal --allow-semantic-change --confirmed`)

Under `autonomy: full_auto` the supervisor owns steps 2, 4, and 5 with a
recorded `--reason`. Stay involved only for scope, retention, paths, output
authority, or `hard_ceiling`.

Extending budgets requires `--confirmed` under `supervised` and a recorded
reason under `full_auto`; `hard_ceiling` caps either and rises only via
human-confirmed `seal`.

## Handoff

- design open → stay here or `$openspec-explore`
- stamped ordinary execution → `$openspec-loop-engineering <change-id>`
- one task without Loop → `$openspec-apply-change <change-id>`
- residue → `$openspec-hygiene`
- full audit → `$monitor-openspec-codex <change-id>`

Report changed artifacts, stamp kind, ref coverage count, unresolved questions,
`check` result, and the exact next command.

# OpenSpec Loop operator guide (live)

This file under `.agents/skills/openspec-loop-engineering/` is the **live**
operator contract for Cursor and Codex. Optional human briefs:
`docs/openspec-loop-overview.md` (and `-cursor`). Old
`docs/openspec-loop-engineering*.md` names are redirects or archive only.

Canonical facade: `scripts/openspec_loop.py`. Check `--help` for unfamiliar flags
or a changed CLI; reuse current verified help. Host adapter: `host-adapter.md`.

Source delivery and passing unit tests are **not** Root PASS. Only the
deterministic controller may grant promotion/closure under current evidence.

---

## 1. Roles

| Actor | Owns | Must not |
| --- | --- | --- |
| Main session `role_id=zpy` | Business supervision, skill routing, user QA | Self-grant PASS; write ledger |
| Deterministic controller | Admission, ledger, budget, promotion, closure | Be impersonated by workers |
| Native platform tools | Real Task / Codex spawn | Fabricate tool responses |
| Semantic workers (skills) | Implement / diagnose / propose within WRITE_SCOPE | Dispatch workers; write ledger; sign authority |
| Independent verifier | Semantic verdict evidence | Share author context |
| User | Retention decisions, irreversible policy, business choices | Be replaced by silence |

There is **no** host-stdio LLM pipe and **no** separate supervisor actor.

---

## 2. Public interface

```powershell
python -B scripts/openspec_loop.py next <change> [--intent drain]
python -B scripts/openspec_loop.py record <change> --ref <R#> --kind apply|verify|unblock --result <file> [--receipt <id>]
python -B scripts/openspec_loop.py verify-mechanical <change> --ref <R#>
python -B scripts/openspec_loop.py promote <change> --ref <R#>
python -B scripts/openspec_loop.py status <change>
python -B scripts/openspec_loop.py cancel <change> [--ref <R#>]
python -B scripts/openspec_loop.py answer <change> --token <t> --choice <id>
```

### `next_action` values

`apply | wait_join | verify_mechanical | verify_semantic | promote | unblock |
interview | needs_user | endpoint | stop`

Apply requires its issued receipt; do not reuse it for Verify or Unblock.

Default `next` output is compact: action, selected ref, census counts, short
receipt id, and blockers. Follow exactly one action.

### Worker result shape

Workers return a JSON file with `status` (or `verdict`) plus
`changed_files`, `evidence_refs`, `blockers`, `risks`, `skipped_checks` for
apply/unblock. Empty/missing status → terminal `empty`, never PASS.
Verification uses `verdict`, `reasons`, `evidence_refs`, `duration_seconds`.

---

## 3. Eight-skill toolchain

Canonical bodies live under `.agents/skills/`. Synchronize mirrors:

```powershell
python -B scripts/sync_openspec_loop_skills.py
python -B scripts/sync_openspec_loop_skills.py --check
```

| Skill | Call when |
| --- | --- |
| `openspec-loop-engineering` | Drive facade / triage `next_action` |
| `openspec-change-interviewer` | Author / revise contracts |
| `openspec-feature-list` | Rebuild `feature_list.json` |
| `openspec-explore` | Read-only investigation |
| `openspec-apply-change` | Implement one receipt-bound task |
| `openspec-verify-change` | Independent semantic verification |
| `openspec-unblock-research` | Diagnose one FAIL/DEVIATED/blocked cause |
| `review-pipeline` | One specialist read-only risk answer |

---

## 4. Q/A boundary (minimum user input)

When `next` returns `needs_user`, ask **immediately** using the returned
`question` + enumerated `options` (Cursor AskQuestion / Codex direct QA). Then:

```powershell
python -B scripts/openspec_loop.py answer <change> --token <answer_token> --choice <option-id>
python -B scripts/openspec_loop.py next <change> --intent drain
```

Answers persist in `openspec/changes/<change>/loop-decisions.json` and are
reused. Do not invent retention roots into `auto_test_openspec/`.

### Unified interrupts (same boundary)

These no longer silent-`stop` / auto-`unblock` without a stance:

| `interrupt_kind` | When | Stance options |
| --- | --- | --- |
| `retention` | missing Artifact Retention Decision | thin / full / none (existing) |
| `exception` | apply/verify `failed`/`empty`/`blocked`/`deviated` | conservative stop · moderate unblock (+2/+2/+1) · aggressive expand (+5/+5/+5) · interview |
| `authorization` | task/result requires resume or production auth | deny · resume+modest budgets · production+aggressive budgets |
| `budget` | apply/unblock capacity exhausted | stop · moderate expand · aggressive expand |
| `legacy` | historical ledger read-only | stop · fresh v5 · fresh v5 + aggressive grants |

Aggressive authorization writes `resume-authorization.json` and
`production-authorization.json` under the change dir and raises revision /
unblock / apply grants on the ledger. Conservative never expands capacity.

After a stance answer of moderate/aggressive, `next` returns `unblock` once.
`record --kind unblock` is **receipt-less** (spawn 0, same as mechanical/
semantic joins). A `disposition=retry` (or `status=completed`) unblock join
clears the failure and the next `next` re-issues apply — it must not loop on
`unblock` forever.

---

## 4b. Skip tier + goal-fed packet + alignment lint

### Skip tier (auto)

Before exception/auth/dispatch, `next` auto-signs refs that must not execute:

- `STATE: superseded` / `STATE: interrupted`
- `PRODUCTION_HOLD` or ACCEPT with `SHALL NOT execute`, `superseded by R#…do not apply`, or `do not resume`
- Reverse edge: another task's `SUPERSEDES: R#`

Ledger records `skips[]` + `task_state=superseded` (not a promotion; never Root
PASS). Dependencies treat skipped refs as satisfied so successor leaves can
dispatch. `status` / `next` report `skipped_refs`.

### Goal-fed apply packet

Apply receipts include compact `goals` (GOAL_IDS → parent chain to root:
`goal_id` / `kind` / `mandatory` / `accept`) and `design_anchors` extracted from
ACCEPT Evidence-anchor phrases. Workers must use this GOAL context; do not
inline design.md.

### Alignment lint

`status` (and `interview` payloads) emit `alignment_warnings` for:

- `GOAL_IDS` → missing goal; goal `COVERED_BY` → missing ref
- task lists a goal but that goal's COVERED_BY omits the ref
- mandatory goal whose covers are all superseded with no successor claimant

Interview checks affected GOAL ↔ design ↔ task relations, including predecessor
and mandatory coverage impacts. Reuse confirmed decisions and keep
STATE/SUPERSEDES/COVERED_BY honest.

---

## 5. Async dispatch (Cursor ↔ Codex handoff)

1. `next` → `apply` + receipt packet.
2. Spawn one native worker with actual platform tools.
3. Worker writes result file; supervisor `record`s it.
4. `verify-mechanical` → optional semantic Verify → `promote`.
5. `status` handoff summary is the cross-App transfer surface.
6. Reconcile actual native worker state before waiting or recovering. After the
   prior writer is confirmed stopped, `cancel --ref` then reissue on either App.

Provider is metadata only; claims are not locked to one App.

---

## 6. Budgets and closure

Defaults: Apply=2, Unblock=2, revisions=3 per ref/fingerprint lineage.
Unlimited grant dimensions still count usage. Local exhaustion does not stop
ready siblings. Completion rights survive exhaustion only under current
evidence/authority. No fabricated PASS or budget reset.

Legacy ledgers (v2/v3/v4) are **read-only**; continue under a fresh v5 ledger
path without rewriting history.

---

## 7. Anti-patterns

| Do not | Do instead |
| --- | --- |
| Treat pytest green as Root PASS | Require controller promotion/closure |
| Blind-continue after `needs_user` | Ask once; `answer`; rerun `next` |
| host-stdio LLM pipe | Async spawn + result file + `record` |
| Worker-write ledger / self-PASS | Controller `record` / `promote` only |
| Squeeze all skills into loop-engineering | Route to the named skill |
| Invent retention roots | Ask Artifact Retention Decision once |
| Treat docs briefs as live workflow | Use this skill; docs overview is optional |

---

## 8. Related paths

- Root contract: `AGENTS.md`
- Host triage: `host-adapter.md` (this folder)
- Cursor thin command: `.cursor/commands/openspec-loop-engineering.md`
- Sync tool: `scripts/sync_openspec_loop_skills.py`
- Optional brief: `docs/openspec-loop-overview.md`
- Landuse v3.2 text (archive only): `docs/archive/openspec-loop-engineering-v3.2.md`

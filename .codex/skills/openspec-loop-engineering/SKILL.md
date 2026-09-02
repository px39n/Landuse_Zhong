---
name: openspec-loop-engineering
description: Run one existing OpenSpec change through the compiled next/receipt control plane while keeping Apply, Verify, Unblock, interviewer, promotion, and final authority separate.
---

# OpenSpec Loop Engineering

Use this as a thin interface to `scripts/openspec_loop.py`; compiled
`next --intent` owns census, routing, budgets, receipts, joins, and repair
admission.

## Entry

```powershell
python scripts/openspec_loop.py next <change-id> --intent drain --run-id <id>
```

Use `--shadow` only for read-only migration/ablation. `check` is CI and `plan`
is verbose compatibility/debug. Follow exactly one returned `next_action`:

- `apply`: dispatch each authoritative receipt once
- `verify_mechanical`: run the CLI mechanical verifier
- `verify_semantic`: invoke `$openspec-verify-change`
- `promote`: promote only an admitted verifier PASS
- `unblock`: invoke `$openspec-unblock-research` in-process, spawn zero
- `repair|interview`: use the Repair Gate or interviewer
- `review|explore|wait_join|design_verify|stop`: perform only that branch

## Apply and join

Before work, validate the packet without another census:

```powershell
python scripts/openspec_loop.py receipt-check <change-id> --ref <R#> --receipt <token>
```

Invoke `$openspec-apply-change <change-id> --task <R#> --orchestrated` with the
receipt-bound ref, attempt, role, WRITE_SCOPE, join, and TEST_LEVEL. Apply
returns one terminal typed result; only the supervisor submits it through
`record --kind apply --receipt <token>`, which is the join event. Prose handoff
and shadow receipts have no authority. New Apply records without an issued
receipt fail before registry lookup or ledger write; legacy rows remain
read-only history.

## Verification and repair

Run `verify-mechanical` after a joined Apply. `smoke` may promote mechanically;
`pilot|production|canary` and legacy tasks require semantic Verify, which may
return `DEVIATED` even after exit 0.

`promote` performs one targeted fingerprint/task/dependency/verifier check,
atomically updates task/feature state, returns `next_required=true`, and never
rebuilds the full plan. Run `next` afterward.

`repair_r1` is the single bounded same-obligation method repair. Read
`.agents/skills/openspec-unblock-research/references/loop-disposition-gate.md`
when the returned action requires repair; broader or ambiguous change is R2
interviewer.

## Authority invariants

- Current fingerprint, narrative digest, harness hash, action class, and an
  authoritative receipt must align before work.
- Only the supervisor records, promotes, reseals, syncs, or changes task state.
- Apply never edits `tasks.md`; Verify and Unblock never implement fixes.
- Per-ref budgets and completion rights remain runtime-enforced; one exhausted
  ref does not stop ready siblings.
- Retention, product/external writes, credentials, destructive actions,
  budget expansion, and Git push/PR/main keep their human gates.
- Optional `control_plane.research_skill_roots` are unweighted intent hints,
  never routing or write authority.

Read packet/result schemas under `references/` only for the selected branch;
read `docs/openspec-loop-engineering.md` for the full lifecycle. On stop, report
the action, ref, verifier state, budget, and whether tracked evidence was written.
Use `ablation-evaluate --manifest` for labeled DEVIATED detection or cross-run
consistency; live CLI telemetry cannot observe LLM control calls.

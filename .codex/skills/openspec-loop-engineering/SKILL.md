---
name: openspec-loop-engineering
description: Drive one OpenSpec change through the compact next/record controller
  while keeping Apply, Verify, Unblock, and promotion authority separate.
---

# openspec-loop-engineering

**Live workflow entry** for OpenSpec Loop. Canonical under `.agents/skills/`.
Thin interface to `scripts/openspec_loop.py`. The deterministic controller owns
admission, ledger, budget, promotion, and closure. Main session zpy supervises
business QA only. Full operator contract: `references/operator-guide.md`.

## Entry

```powershell
python -B scripts/openspec_loop.py next <change-id> --intent drain
python -B scripts/openspec_loop.py status <change-id>
```

Follow exactly one returned `next_action`:

- `needs_user`: present `question` + `options` immediately (Cursor AskQuestion /
  Codex direct QA), then `answer --token <t> --choice <id>` and rerun `next`.
  Same boundary covers retention, exceptions, authorization, budget exhaustion,
  and legacy ledgers (conservative / moderate / aggressive stances).
- `apply`: spawn one native worker with the receipt packet (includes GOAL chain);
  do not inline work. Superseded / SHALL NOT execute refs are auto-skipped.
- `wait_join`: reconcile the receipt with actual native worker state. Wait for
  a live worker; record a terminal worker's actual result. If dispatch failed
  or no result can arrive, report that condition and use controller-managed
  recovery. A receipt alone does not prove liveness; a timeout does not prove
  termination. Do not reissue work while an earlier writer may still be active.
- `verify_mechanical`: run the CLI verifier
- `verify_semantic`: invoke `openspec-verify-change` in an independent context
- `promote`: promote only after admitted verifier PASS
- `unblock`: invoke `openspec-unblock-research` in-process (spawn 0)
- `interview`: hand to `openspec-change-interviewer`
- `endpoint` / `stop`: report and stop

## Record and authority

```powershell
python -B scripts/openspec_loop.py record <change> --ref <R#> --receipt <id> --kind apply --result <file>
python -B scripts/openspec_loop.py verify-mechanical <change> --ref <R#>
python -B scripts/openspec_loop.py promote <change> --ref <R#>
python -B scripts/openspec_loop.py cancel <change> [--ref <R#>]
python -B scripts/openspec_loop.py answer <change> --token <t> --choice <id>
```

Workers return a result file only. Empty result → `empty` terminal, never PASS.
Only the controller records, promotes, or closes. Provider is metadata; unfinished
claims cancel-and-reissue across Cursor/Codex. See `references/host-adapter.md`
and `references/operator-guide.md`. Do not treat `docs/` as the live workflow.

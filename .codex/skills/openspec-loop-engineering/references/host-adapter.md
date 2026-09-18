# Host and toolchain adapter

Canonical skill bodies stay provider-neutral. This note is the thin host
disambiguation layer for Cursor and Codex. Operator contract:
`operator-guide.md` in this folder (live). `docs/` is not the workflow root.

## Live work versus source production

Public facade:

```powershell
python -B scripts/openspec_loop.py next <change> --intent drain
python -B scripts/openspec_loop.py status <change>
python -B scripts/openspec_loop.py cancel <change> [--ref <R#>]
```

Native spawn is **asynchronous** and owned by the agent platform — not by a
Python host-stdio pipe.

| Provider | Native lifecycle capabilities |
| --- | --- |
| Codex | Actual exposed spawn, status/wait and interrupt tools; namespaces vary |
| Cursor | Actual exposed Task or equivalent native lifecycle tools |

Resolve capabilities from the current tool descriptions, not an exact-name
allowlist. Never invent a missing close operation or spoof another host's tools.
Missing required native capabilities → degraded apply under user authority
(implement WRITE_SCOPE + owner TEST, no ledger, no Loop PASS).

## Async dispatch / recover

1. `next` returns `apply` + authoritative `receipts[]` with packet.
2. Main session spawns one worker with the packet (one ref, one attempt).
3. Worker writes a result JSON file; does not write the ledger.
4. Supervisor `record --receipt --result <file>`; empty → terminal `empty`.
5. Follow the next `next_action` (verify / promote / unblock).

For `wait_join`, inspect actual native worker state. Wait only while it is live;
record its terminal result when available. Dispatch failure or a terminal
worker without output requires honest failure reporting and controller-managed
recovery. A receipt does not prove liveness; a timeout does not prove termination.

Provider is metadata only. After confirming the prior native writer has stopped,
unfinished claims may `cancel --ref` and reissue on either App. `status` handoff
summary is the cross-platform transfer surface. Uncertain ownership blocks
replacement writers, not independent read-only work.

## Q/A boundary

When `next_action=needs_user`, present `question` + enumerated `options`
immediately:

- Cursor: AskQuestion (or equivalent structured prompt)
- Codex: direct user QA in the main session

Then:

```powershell
python -B scripts/openspec_loop.py answer <change> --token <answer_token> --choice <option-id>
python -B scripts/openspec_loop.py next <change> --intent drain
```

Answers persist under `openspec/changes/<change>/loop-decisions.json`.

## Symptom triage

| Symptom | Next step |
| --- | --- |
| `needs_user` | Ask once; `answer`; rerun `next` |
| `legacy_ledger_readonly` | Open a fresh v5 ledger path; do not mutate history |
| `empty` / empty result | Record as empty; unblock or stop that ref |
| missing native tools | Degraded apply; no ledger events |

## Toolchain

loop-engineering coordinates the facade. interviewer/feature-list author spec.
apply implements one task. verify independently assesses. unblock diagnoses.
explore and review-pipeline stay read-only.

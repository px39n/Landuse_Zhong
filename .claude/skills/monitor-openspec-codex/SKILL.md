---
name: monitor-openspec-codex
description: Legacy audit-only OpenSpec supervision with retained per-attempt bundles and append-only evidence bookkeeping. Use only when the user or task explicitly requires full audit retention.
license: MIT
metadata:
  version: "2.0"
---

# Monitor OpenSpec Codex - Legacy Audit Only

This skill is intentionally outside the ordinary OpenSpec Loop.

Use it only when at least one condition is explicit:

- the user requests retained audit supervision;
- the active task requires a full validation bundle;
- sealed `loop.json` has `retention=full`.

Otherwise route execution to `$openspec-loop-engineering <change-id>`. Do not
activate monitor because an OPSX command, Loop attempt, or verifier ran.

Before execution:

1. Require an implementation-ready, sealed contract with no fingerprint drift.
2. Confirm the full-retention bundle, product, scratch, and GUI/Colab paths.
3. Read `references/legacy-audit-contract.md`.
4. Run strict OpenSpec validation.

Monitor owns legacy `BUNDLE` / `EVIDENCE` / `progress.txt` / `runs.log`
bookkeeping and immutable run folders. Workers implement one approved task and
must not claim PASS; the supervisor independently verifies and records PASS.

Audit artifacts do not become active spec text. Contract changes pause monitor,
return to `$openspec-change-interviewer`, and require resealing. A new revision
does not reset change-level budgets.

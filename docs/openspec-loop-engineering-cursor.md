# OpenSpec Loop v3.2 on Cursor

This file is a platform entry point, not a second copy of the protocol.
The persisted schema remains v3 while the machine control identity is
`openspec-loop-control.v3.2`. Apply records require issued receipts, promotion
uses a targeted completion transition, and labeled ablation remains an
explicit evaluation command rather than an inferred LLM metric.

- Start with `/openspec-loop-engineering <change-id>`.
- Cursor commands under `.cursor/commands/` are thin routers to the canonical
  `.agents/skills/openspec-*` skills.
- The lifecycle, seal, budgets, states, retention levels, test tiers, semantic
  deviation handling, and amendment rules are maintained only in
  [`openspec-loop-engineering.md`](openspec-loop-engineering.md).
- `scripts/openspec_loop.py` owns compiled `next`, `receipt-check`, typed
  `record`/join, `verify-mechanical`, promotion, seal/reseal, and summary;
  `check` is CI while `plan`/`gate` remain compatibility/debug. Apply, semantic
  Verify, Explore, Unblock, and interviewer remain skill-driven actions.
- The sealed `autonomy` mode and the unraisable `hard_ceiling` are defined in the
  canonical guide; this note does not restate them.
- `monitor-openspec-codex` is available only for explicit full-retention or
  legacy audit work.

Do not expand this platform note into another protocol copy. Update the
canonical skills and public guide, then run
`python scripts/sync_openspec_loop_skills.py`.

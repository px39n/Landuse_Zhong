# OpenSpec Project Policy

## OpenSpec Loop v3

- Read `docs/openspec-loop-engineering.md` for lifecycle, state, retention,
  test tiers, amendment flow, the authority model, and finite budgets.
- Ordinary Loop execution requires interviewer confirmation and a matching
  sealed `loop.json`. Narrative drift in `proposal.md`/`design.md`/`specs/**`
  is refreshed by `reseal` and does not pause work; a semantic change to the
  active task registry pauses it, and promotion is not drift.
- A semantic pause follows the sealed `autonomy` field: `supervised` needs
  confirmation and resealing, `full_auto` needs a recorded `--reason`, and the
  sealed `hard_ceiling` can be raised only by a human-confirmed `seal`.
- Default `thin` execution runs owner tests per attempt, the complete task
  `TEST:` at promotion, and strict OpenSpec validation at final completion.
- Retained full bundles are required only when the user or task requests them,
  `retention=full`, or legacy audit mode is active.
- Current OpenSpec files contain current contract truth; Git retains replaced
  versions. Do not append complete old drafts or attempt logs to active specs.

Landuse-specific GEE, GCS, D-drive, billing, and live-execution gates remain in
`AGENTS.md` and the selected change contract. Loop v3 does not relax them.

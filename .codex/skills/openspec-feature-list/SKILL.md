---
name: openspec-feature-list
description: Generate the compact OpenSpec Loop v2 feature registry from the active tasks.md contract while preserving compatible runtime state.
metadata:
  short-description: Build compact feature_list.json deterministically
---

# OpenSpec Feature List

Generate or update `openspec/changes/<change-id>/feature_list.json` with the
repository script:

```powershell
python scripts/generate_openspec_feature_list.py --change-id <change-id>
```

If the change id is missing and cannot be inferred uniquely, ask once.

## Source and boundary

- `tasks.md` is the active task contract.
- `feature_list.json` is derived dynamic state, not a second copy of ACCEPT or
  TEST prose.
- Git history preserves superseded contract wording. Historical/appendix task
  blocks are not executable and are not copied into the active registry.
- Do not modify `tasks.md` from this skill.

## Compact v2 entry

Each active entry contains only:

- `id`
- `ref`
- `title`
- `state`
- `passes`
- `depends_on`
- `supersedes`
- `task_path`
- `accept_hash`
- `test_hash`

Optional compact execution fields are copied only when declared:

- `repair_policy` from `REPAIR_POLICY: bounded-r1`
- `test_level` from `TEST_LEVEL: smoke|pilot|production|canary`

States are `pending | ready | in_progress | blocked | deviated | maxed |
superseded | passed`.

The generator validates unique `[#R<n>]` refs plus required `ACCEPT:` and
`TEST:` directives. It preserves compatible v1/v2 pass and terminal state when
the active task remains, computes readiness from dependencies, and marks a
superseded target non-runnable. It writes deterministic UTF-8 JSON only after
validation succeeds.

## After generation

Run shadow next to inspect the compact action without claiming execution authority:

```powershell
python scripts/openspec_loop.py --repo-root . next <change-id> --intent drain --run-id <id> --shadow
```

Generating the registry alone does not authorize implementation. Ordinary Loop
execution requires active `next` to see a matching `contract_fingerprint` and
no `pending_irreversible_policy`; if `loop.json` is missing it may initialize
recorded thin defaults without creating product/evidence paths. Preview, stamp, legacy
`sealed`/`confirmed_at`, headcount, and `hard_ceiling` fields are not ordinary
dispatch authority.

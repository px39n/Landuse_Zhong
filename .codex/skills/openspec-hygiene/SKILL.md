---
name: openspec-hygiene
description: Inventory and safely clean OpenSpec and repository residues (cache, scratch, dead modules, drifted docs). Classify dirty paths before any delete; never auto-delete unknown or unrelated files. Use after Loop design-verify PASS, before archive, or when residue/cache explosion is reported.
---

# OpenSpec Hygiene

Read-only inventory first. Deletion is always explicit and user-authorized.
This skill does not implement product features, seal Loop policy, or promote
tasks.

## Invocation

`$openspec-hygiene <change-id>` or `$openspec-hygiene repo`

## Inventory

Run a cheap classification over the requested scope (one change, sealed scratch
roots, or whole repository). Cover folders, caches, source, and docs:

- `test_cache/`, `auto_test_openspec/`, root `tmp_pytest_*`, `.pytest_tmp`,
  `_tmp/`, `tmp/`, `tmp_*`
- orphaned `__pycache__` / disposable basetemp trees under sealed scratch
- unused one-off `.py` utilities created outside a task `FILES:` budget
- docs that conflict with code/tests (mark stale; do not silently rewrite)

Classify every dirty path as exactly one of:

| Class | Action |
|---|---|
| `task-scoped` | Candidate for cleanup if the owning task/change is done |
| `unrelated pre-existing` | Never auto-delete |
| `generated-ignored` | Note only; respect `.gitignore` |
| `scratch` | Candidate via clean_test_residues when under confirmed roots |
| `unknown` | Never auto-delete; ask before any action |

Reuse `references/artifact-placement.md` under
`$openspec-change-interviewer` for path identity rules (unblock vs product vs
scratch vs ledger).

## Delete procedure

1. Prefer dry-run:

```powershell
python scripts/clean_test_residues.py --dry-run
```

2. Present the classified list. Retained evidence under `auto_test_openspec/**`
   that the task contract keeps must not be listed for deletion.
3. Delete only after the user types `YES` to the cleanup tool's confirmation
   prompt (or an equivalent explicit approval for a named path list).
4. Do not clean unrelated dirty worktrees or user WIP.

## Integration

- After `design-verify` PASS and before archive, recommend one hygiene pass.
- Loop may suggest hygiene; it must not auto-run deletes.
- No background timers or hooks create cleanup jobs.

## Output

```text
HYGIENE: INVENTORY|CLEANED|BLOCKED
SCOPE: <change-id|repo>
CLASSES:
- task-scoped: <paths>
- scratch: <paths>
- unrelated: <paths>
- generated-ignored: <paths>
- unknown: <paths>
ACTIONS:
- <dry-run or approved delete summary>
NEXT:
- <one bounded action>
```

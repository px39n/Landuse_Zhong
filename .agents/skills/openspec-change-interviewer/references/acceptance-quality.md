# Acceptance quality contract

Active tasks must be production-output oriented. Presence of `ACCEPT:` / `TEST:`
lines is not enough.

## ACCEPT

Each active task ACCEPT block must include:

1. Concrete expected behavior or product outcome.
2. At least one path-like product anchor (`outputs/`, `src/`, `scripts/`,
   `docs/`, `openspec/`, a drive letter path, or a `*.py|*.csv|*.json|*.md`
   pattern), **or** an explicit `NO_ARTIFACT: <reason>` when the deliverable is
   purely in-repo contract text with no product file.
3. A testable acceptance signal (metric threshold, schema field, checkbox
   evidence pointer, or exact command observation).
4. Every implementation-affecting assumption expressed as a measurable
   outcome, its falsifier, an evidence anchor, and either `confirmed` or
   `Open Question` state.

Slogan text ("safer", "idempotent", "follow existing logic") is incomplete.

### Success-criteria restatement

Before making fuzzy goals executable, restate them as measurable outcomes and
confirm them or retain them as open questions:

> "提高分类精度" → "OA ≥ X% recorded in `outputs/<change>/metrics.json`,
> figure under `figure/...`, manifest schema unchanged — correct?"

### Assumption and loophole pass

An assumption must use this compact shape:

```text
Assumption: <claim>
Measurable outcome: <observable threshold or behavior>
Falsifier: <observation that disproves it>
Evidence anchor: <path:line, symbol, fixture, or command>
State: confirmed | Open Question
```

Before a task is execution-ready, cover safety, migration, compatibility,
rollback, and empty/error/permission states in ACCEPT/TEST, or mark each omitted
category `out of scope because <reason>`. A generic "handled elsewhere" is not
a loophole decision.

## TEST

Each active task TEST block must include at least one executable `Run:` line
(pytest, script, openspec CLI, or other shell-invocable command). Inspections
may follow, but cannot replace `Run:` unless `NO_ARTIFACT:` applies and the
TEST names a concrete read-only verification command.

When `loop.json` or the task contract records a scratch root, pytest basetemp
belongs under that root. A conventional `test_cache/<change-id>/pytest/` path
is only a proposal until selected; do not invent repo-root `tmp_pytest_*` or
`.pytest_tmp` paths in TEST blocks.

## Module placement declaration

Before making a task that adds Python modules execution-ready, classify each
new `.py` as exactly one of:

| Kind | Home |
|---|---|
| Runner / CLI | `scripts/` (argparse entry, orchestration) |
| Parameter / config | `config.yaml`, package config modules, or change-declared config path |
| Model / library code | `src/<package>/` |

A task that would create unclassified new modules is not execution-ready.

## `FILES:` budget

Optional per-task directive listing paths the task may create or edit. During
apply, creating a planned-out path is material discovery: pause and return to
`$openspec-change-interviewer` (or the full_auto supervisor) instead of
silently expanding scope.

## Feature-list mechanical check

`python scripts/generate_openspec_feature_list.py` warns when ACCEPT lacks a
path anchor / `NO_ARTIFACT:` or TEST lacks `Run:`. Pass `--strict-quality` to
fail generation. Existing changes remain writable under the default warn mode.

# AGENTS.md - Landuse_Zhong

This file is the project contract for `C:\Phd_art\Landuse_Zhong` and all child
paths. System/developer instructions and a nearer child `AGENTS.md` remain
authoritative for their scopes.

## Repository safety

- Modify files only inside this repository unless the user explicitly names a
  different target.
- Preserve the existing dirty working tree. Do not revert, clean, regenerate,
  or reformat unrelated files.
- Read-only Git commands are allowed. Check `git status --short`, the current
  branch, and worktrees before branch or worktree operations.
- Commit or push only on `wip/*` branches after scope-appropriate verification.
  Changes intended for the default `master` branch go through a PR; Codex does
  not commit directly to `master`, merge, enable auto-merge, or close the PR as
  completed.
- Preserve Chinese and mixed Chinese-English files as UTF-8. On Windows, read
  them with an explicit UTF-8 encoding and verify edited text for mojibake or
  the Unicode replacement character.

## Project map

`Landuse_Zhong` is a Python/Jupyter geospatial research repository for U.S.
cropland abandonment, photovoltaic suitability, model training, multiscale
policy analysis, and Earth Engine/GCS execution.

- `function/`: reusable S0/S1/S2 modeling, review, GEE, billing, queue, and
  retained-output contracts.
- `tools/`: deterministic CLIs, notebook generators, preflight checks, and
  artifact builders.
- Root notebooks: user-facing orchestration and publication analysis. Keep
  generated notebooks synchronized with their generator modules and tests.
- `tests/`: contract, regression, notebook, runtime, budget, and security tests.
- `openspec/`: retained change contracts. `openspec/changes/ee-california` is
  the active S2 California contract unless the user selects another change.
- `data/contracts/`: compact versioned source, pricing, schema, and execution
  contracts. Large/raw data remains in ignored or external roots.
- `outputs/`: only compact retained manifests, summaries, and reports belong in
  Git; large runtime products stay in their declared external roots.
- `requirements/`, `geo.yml`, `renviro_geo.yml`: environment contracts.

## Before editing

1. Identify the owning stage and read the nearest contract first: relevant
   README section, OpenSpec artifact, `data/contracts` manifest, or retained
   output README.
2. Trace the narrow path: contract/config -> `function/` module -> `tools/`
   runner or notebook generator -> tests -> retained outputs.
3. Treat source, tests, manifests, and generated artifacts as executable truth.
   If prose disagrees, report the conflict before broadening scope.
4. Inspect CLI parsers and defaults before running a tool. Do not infer that a
   stage is dry-run or read-only from its name alone.

## Earth Engine, GCS, paid APIs, and external state

- Default to read-only probes, dry-runs, local fixtures, and synthetic/minimal
  canaries. Never submit GEE exports, create Assets, write or overwrite GCS
  objects, start paid model/API work, or launch statewide/CONUS runs merely to
  validate code.
- Live GEE work requires an explicit user-authorized execution request plus all
  repository gates: matching project/account/bucket identity, fresh acceptance,
  immutable task-spec fingerprints, budget evidence, and required execution
  inputs. `--execute-exports` is never implied.
- For S2 California, do not launch full California work until the required
  minimal GEE unit/canary succeeds and the complete GEE -> GCS -> D-drive chain
  is checksum-verified. A canary acceptance is not statewide acceptance.
- Fail closed on missing or stale billing balance, EECU evidence, task identity,
  checksums, generation numbers, or external-output evidence. Do not retry the
  same failed compute graph without following the contract's fallback order.
- Do not perform OAuth or change user credentials automatically. Never print,
  serialize, commit, or copy tokens, API keys, service-account material, `.env`
  contents, or private cloud identifiers into reports.
- Treat recursive deletes, overwrite syncs, Asset creation/deletion, bucket
  writes, and paid submissions as destructive or externally consequential.

## Implementation style

- Keep diffs surgical and reuse existing helpers before adding abstractions or
  dependencies.
- Preserve manifest schemas, task IDs, run IDs, fingerprints, CRS/grid metadata,
  nodata/encoding rules, output roots, and notebook/generator parity.
- Use structured parsers for JSON, YAML, notebooks, GeoJSON, NetCDF, rasters,
  and tabular data; avoid ad hoc string rewrites when an API is practical.
- Keep notebooks thin: parameters, inspection, and orchestration belong there;
  reusable computation belongs in `function/` and deterministic generation in
  `tools/`.
- For cleanup or refactoring, lock existing behavior with focused tests first.

## OpenSpec Loop v3

- `.agents/skills/` is the semantic source for the Loop skill series; use
  `python scripts/sync_openspec_loop_skills.py` to refresh `.codex` and
  `.claude` mirrors without replacing platform metadata.
- Before every active Loop call (`check`, `plan`, `seal`, `reseal`, `gate`,
  `record`, `promote`, `sync`, `apply-revision`, `goals`, `design-verify`,
  `summary`, or Apply/Verify/Unblock skill invocation), the sole supervisor
  must align the current task `ACCEPT` text, applicable main/delta requirements,
  `loop.json`, the canonical `.agents` skill, and the action class. Same-turn
  reuse is allowed only while fingerprint, narrative digest, and canonical
  skill state remain unchanged. A mismatch routes to interviewer, narrative
  reseal, skill sync, or unblock; this latch creates no command or receipt.
- `ACCEPT` is the admitted ruler for one whole-registry execution chapter.
  Apply/Verify/Unblock, checkbox/`STATE:`, promotion, and narrative reseal do
  not consume a cycle stamp. `apply-revision`, an obligation-changing reseal
  after Apply/Unblock work, and a second or later semantic reseal in an
  undrained chapter do. A qualifying first pre-execution or post-drain
  confirmed interviewer stamp is chapter-outside and free; `--confirmed` alone
  never makes an in-loop stamp free.
- New/omitted policy defaults are per-ref Apply `2`, per-ref Unblock `2`, and
  change cycle stamps `3`; task count never scales the stamp cap, and omitted
  reseal overrides inherit recorded policy. `max_revisions` gates only a
  charged semantic stamp before authority files change. Ordinary Apply/Explore
  never rechecks it; record/join/Verify/promote/sync/goals/design-verify/
  summary/stop-hook/review retain completion right. Unblock is ref-local, and a
  matching fingerprint remains admitted if the cap is later lowered.
- Apply workers never edit `tasks.md`. In a current same-ref blocking or
  `amend_spec` window, the sole supervisor may use one reasoned
  `stamp_source=unblock_self_confirm` semantic reseal per source episode,
  without `--confirmed`, only for mechanically non-widening ACCEPT/TEST/FILES
  correction inside unchanged WRITE_SCOPE. Framework policy, other refs, DAG,
  passed ACCEPT, widened acceptance, budget raises, external/product/Git
  authority, and a second-Unblock self-restamp return to the interviewer.
- Loop start-work authority is a matching `loop.json.contract_fingerprint` with
  no pending irreversible policy decision. `seal-preview.md`, `sealed`, and
  `confirmed_at` are optional audit/compatibility data, not ordinary Apply
  gates. Narrative drift in proposal/design/specs uses `reseal`; active-registry
  drift uses the semantic authority above. Promotion is fingerprint-neutral.
  Do not copy another change's external path tree as a default.
- Raising an optional hard ceiling, changing retention/paths/scope, destructive
  external writes, credentials, product `--commit`, and Git push/PR/`master`
  remain human-authorized and subject to the repository gates above regardless
  of autonomy. Ordinary thin/default policy needs no extra stamp ceremony.
- Default to `thin` retention, finite budgets, compact feature state, and
  attempt/promotion/final test tiers. Reuse confirmed GCS, D-drive, cache, and
  product paths instead of asking for them again on every attempt.
- Full retained bundles are required only when explicitly requested,
  `retention=full`, or legacy audit mode is active.
- A successful command with scientifically wrong data, metrics, provenance, or
  outputs is `DEVIATED`, not `PASS`; route it through bounded unblock research.
- `monitor-openspec-codex` remains legacy/audit-only. The public lifecycle and
  amendment rules are in `docs/openspec-loop-engineering.md`.
- Loop autonomy never authorizes GEE exports, GCS writes, paid API work,
  credentials, product `--commit` runs, `git push`, PR operations, or writes to
  `master`; those actions remain governed by the Landuse-specific gates above.

## Verification

Use the smallest proof that covers the change, then run adjacent tests.

```powershell
conda activate geo
python -m pytest tests/test_gee_runtime.py tests/test_gee_spectral_notebook_contract.py -q
python -m pytest -q
openspec validate ee-california --strict
```

- Full pytest is required only when the changed surface is shared broadly or
  the active contract requires it; do not run it as a ritual for documentation
  or isolated skill changes.
- Live GEE/API tests remain explicit opt-in and must never be enabled by an
  ordinary local test command.
- For OpenSpec work, keep proposal/design/specs/tasks mutually consistent. Keep
  `feature_list.json` synchronized when that artifact exists or the task system
  requires it.
- Contract-test PASS is not evidence that external exports or retained outputs
  exist. Report local verification separately from live execution evidence.

## GitNexus cost gate

Use `rg`, direct reads, Git diffs, and focused tests by default. GitNexus is
optional and should be used only when explicitly requested or when a complex,
high-risk cross-module flow materially benefits from graph evidence. A stale or
missing index is not a blocker and does not justify automatic re-indexing.

## Completion report

Report changed files, verification commands and outcomes, any external action
that was intentionally not run, retained artifact locations, and remaining
risks. Never claim a live task, export, upload, download, or D-drive handoff
completed without file- or API-backed evidence.

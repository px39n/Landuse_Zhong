## Active task registry

- GOAL G1: Loop charges only in-chapter obligation rewrites while admitted work retains completion authority
  - COVERED_BY: R1-R3
  - ACCEPT: new seal defaults are Apply 2 / Unblock 2 / cycle stamps 3; the fourth charged cycle stamp is refused before registry writes; same-fingerprint Apply and all closure actions are not starved by revision, minute, or cross-ref breaker state; one blocked ref may use the bounded supervised self-restamp once; canonical skills, mirrors, public guide, main spec, and strict validation agree

- Entry mode: `succession` from the merged `e2e-harness` / `e2e-host` Loop contract.
- Confirmed retention: `thin`; retained `auto_test_openspec/openspec-loop-completion-right/`; disposable cache, pytest, scratch, and ledger under `test_cache/openspec-loop-completion-right/`; product, GUI/Colab, and bundle `null`.
- Repository defaults remain `max_apply_attempts=2`, `max_unblock_runs=2`, `max_revisions=3`, `max_explore_runs=1`, revision active minutes `120`, and change active minutes `360`; prior explicit seals inherit unchanged when no override is supplied.
- All refs use local `zpy` direct execution under the sole supervisor. No host spawn, VFD live tree, product run, D-drive write, PNG, new skill package, or R4 is in scope.

## 1. Runtime authority

- [x] 1.1 Implement execution-chapter stamp accounting, completion right, and blocked-window self-restamp [#R1]
  - DEPENDS_ON: none
  - INDEPENDENT: yes
  - ROLE_ID: zpy
  - EXECUTION: sync
  - JOIN: immediate
  - WRITE_SCOPE: `scripts/openspec_loop.py`, `tests/test_openspec_loop.py`
  - FILES: `scripts/openspec_loop.py`, `tests/test_openspec_loop.py`
  - ACCEPT: `scripts/openspec_loop.py` writes and inherits default Apply 2 / Unblock 2 / cycle-stamp 3 policy without scaling `max_revisions` by task count; records minimal chapter stamp state; classifies semantic reseal and `apply-revision` before writes; permits three charged stamps and rejects the fourth with tasks/feature/loop bytes unchanged; removes `max_revisions` from ordinary gate; applies minutes and generic breakers only to `apply|explore`; scopes breakers to the current episode/ref/kind and last two terminal records; gives closure and ref-local unblock their specified rights; requires gate kind/ref arguments; counts only real budget increases as self-extensions; rejects any Apply record that writes `tasks.md`; and implements one reasoned `unblock_self_confirm` candidate transaction whose structural allowlist, framework deny-list, non-widening rule, one-ref/episode limit, and second-unblock terminal rule are mechanically enforced. Synthetic R62/R54 shapes prove same-fingerprint work is not starved without reading live VFD trees or ledgers.
  - TEST: SCOPE: CLI
    - Run: `python -m pytest tests/test_openspec_loop.py -x -q -p no:cacheprovider --basetemp test_cache/openspec-loop-completion-right/pytest/R1`
    - Verify: all runtime tests pass; rejected semantic transactions leave the three authority files byte-identical; no live VFD path or product sink is touched.

## 2. Governance and public workflow

- [x] 2.1 Align AGENTS, canonical skills, and public guide to chapter/stamp semantics [#R2]
  - DEPENDS_ON: none
  - INDEPENDENT: yes
  - ROLE_ID: zpy
  - EXECUTION: sync
  - JOIN: immediate
  - WRITE_SCOPE: `AGENTS.md`, `.agents/skills/openspec-loop-engineering/SKILL.md`, `.agents/skills/openspec-change-interviewer/SKILL.md`, `.agents/skills/openspec-verify-change/SKILL.md`, `.agents/skills/openspec-unblock-research/SKILL.md`, `docs/openspec-loop-engineering.md`, `tests/test_openspec_loop_skills.py`
  - FILES: `AGENTS.md`, `.agents/skills/openspec-loop-engineering/SKILL.md`, `.agents/skills/openspec-change-interviewer/SKILL.md`, `.agents/skills/openspec-verify-change/SKILL.md`, `.agents/skills/openspec-unblock-research/SKILL.md`, `docs/openspec-loop-engineering.md`, `tests/test_openspec_loop_skills.py`
  - ACCEPT: Root governance, the four canonical skills, and public guide define ACCEPT, execution chapter, cycle stamp, and chapter-outside stamp identically; publish defaults 2/2/3 without treating VFD 5/14/1 as defaults; require each active Loop call to align current ACCEPT, loop policy, capability requirements, canonical skill, and action class; keep completion/unblock rights and sole-writer authority explicit; describe the one reasoned blocking-window self-restamp and its allow/deny lists without creating a new command, skill, receipt, agent, or ledger; state that implementers may make scoped narrative edits but Apply never edits tasks; update the existing §2.3 harness graph rather than creating a second lifecycle; `NO_ARTIFACT: contract-and-test-only` means this ref writes no product, GUI, or runtime evidence artifact.
  - TEST: SCOPE: CLI
    - Run: `python -m pytest tests/test_openspec_loop_skills.py::test_completion_right_stamp_and_alignment_contract_is_explicit tests/test_openspec_loop_skills.py::test_unblock_host_policy_is_in_process_and_ref_local tests/test_openspec_loop_skills.py::test_public_guide_has_atomic_harness_graph_and_tool_flow tests/test_openspec_e2e_references.py::test_one_shot_host_batch_returns_supervisor_join -x -q -p no:cacheprovider --basetemp test_cache/openspec-loop-completion-right/pytest/R2`
    - Verify: canonical wording and predecessor host/join rules pass together; no VFD action, Cursor projection, or excluded lifecycle artifact appears.

## 3. Canonical mirrors and main capability

- [x] 3.1 Sync canonical mirrors and intelligently update openspec-loop-execution [#R3]
  - DEPENDS_ON: R1, R2
  - ROLE_ID: zpy
  - EXECUTION: sync
  - JOIN: immediate
  - WRITE_SCOPE: `.codex/skills/openspec-loop-engineering/**`, `.codex/skills/openspec-change-interviewer/**`, `.codex/skills/openspec-verify-change/**`, `.codex/skills/openspec-unblock-research/**`, `.claude/skills/openspec-loop-engineering/**`, `.claude/skills/openspec-change-interviewer/**`, `.claude/skills/openspec-verify-change/**`, `.claude/skills/openspec-unblock-research/**`, `openspec/specs/openspec-loop-execution/spec.md`
  - FILES: `.codex/skills/openspec-loop-engineering/**`, `.codex/skills/openspec-change-interviewer/**`, `.codex/skills/openspec-verify-change/**`, `.codex/skills/openspec-unblock-research/**`, `.claude/skills/openspec-loop-engineering/**`, `.claude/skills/openspec-change-interviewer/**`, `.claude/skills/openspec-verify-change/**`, `.claude/skills/openspec-unblock-research/**`, `openspec/specs/openspec-loop-execution/spec.md`
  - ACCEPT: `python scripts/sync_openspec_loop_skills.py` updates only configured `.codex` and `.claude` semantic mirrors from `.agents`, its `--check` is byte-exact, and no `.cursor` projection is created. `$openspec-sync-specs openspec-loop-completion-right` intelligently merges only this change's `## MODIFIED Requirements` into `openspec/specs/openspec-loop-execution/spec.md`, creates the capability when absent, preserves unrelated requirements/scenarios, remains idempotent on a second merge, and leaves the active change unarchived. The full focused pytest command, strict change validation, `check`, `goals`, observed `design-verify`, and whole-change Verify all pass.
  - TEST: SCOPE: CLI
    - Run: `python scripts/sync_openspec_loop_skills.py --check`
    - Run: `python -m pytest tests/test_openspec_loop.py tests/test_openspec_loop_skills.py tests/test_openspec_e2e_references.py -x -q -p no:cacheprovider --basetemp test_cache/openspec-loop-completion-right/pytest/all`
    - Run: `openspec validate openspec-loop-completion-right --strict`
    - Verify: mirrors and main capability match canonical semantics, all commands exit `0`, and the change remains active with no live VFD execution or migration.

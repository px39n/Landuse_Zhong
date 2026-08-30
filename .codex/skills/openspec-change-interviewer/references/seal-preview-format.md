# Optional seal preview / audit diagnostic format (可选盖章预览)

`seal-preview.md` is an **optional audit diagnostic** for reviewing retention,
paths, budgets, `autonomy`, `hard_ceiling`, and whole-registry fingerprint
coverage. It is not a start-work gate. Ordinary Apply/dispatch does not require
this file, `confirmed_at`, or `seal --confirmed` when the recorded
`contract_fingerprint` matches and no irreversible policy change awaits human
confirmation.

Use the preview when a human-readable policy review is useful, or before
raising `hard_ceiling`, changing retention/paths/scope, destructive external
writes, credentials, product `--commit`, or `git push`/PR/`main` operations.
The preview records a decision; it does not itself authorize the later risky
operation, check off tasks, or create retained/cache roots.

## Required sections

```markdown
# 盖章预览 / Seal preview: <change-id>

Entry mode: new-idea | major-revision | additive-extension | succession | execution-ready
Audit diagnostic fields (optional):
- previewed_at:
- confirmed_at:
- stamp_kind: first-policy-record | policy-restamp | semantic-diagnostic
- reviewer:

## 1. Contract fingerprint coverage（完整义务面）
- change_id:
- semantic fingerprint covers: **entire** active tasks.md obligations
  (checkbox / STATE neutralized) — not a single task
- Active refs observed by this diagnostic:
  - R1 …
  - R2 …
  (list every [#R…] in the active registry)
- ready now / still blocked by deps: (informational only)
- narrative digest covers: proposal.md, design.md, specs/**

## 2. Path profile (mode → expanded for this change-id)

Selected profile: A_local_thin | B_external_heavy | C_product_repo | D_custom

| Role | Expanded path |
|---|---|
| ledger | |
| scratch | |
| pytest basetemp | |
| product | |
| bundle | |
| gui_colab | |

## 3. Trees

### 授权流
tasks/feature registry → matching fingerprint → check/plan → Apply

Optional irreversible-policy branch:
policy delta → seal-preview（本页）→ 用户确认该 policy delta → loop.json

### 落盘流
change-id → ledger / scratch / product|pointers（本 change 展开）

## 4. Execution policy
- retention:
- test profiles:
- budgets:
- remaining-work budget advisory (only if the CLI payload has one):
  - configured / recommended / shortfall
- autonomy:
- hard_ceiling:

## 5. Authority boundary
- Ordinary fingerprint-matched work: check, plan, Apply, Verify, promote, sync
- Human-confirmed policy: raise hard_ceiling; change retention, paths, or scope
- Risky operation still requires its own authority: product --commit,
  credentials, destructive external writes, git push / PR / main

## 6. 可选政策确认 / audit acknowledgement
- [ ] 记录当前档位与政策（不是普通 Apply/dispatch 的开工门）
- [ ] 改档位为: A | B | C | D
- [ ] 采用新建根名: <path or null>
- [ ] 调整 budgets / autonomy / hard_ceiling: <delta>
- [ ] 仅保留诊断，不变更任何 policy
```

## Path profiles (modes; expand with `<change-id>`)

Recommend a profile only after reading this change's proposal/design/tasks and
any existing `loop.json`. Prefer paths already named in the contract.

| Profile id | Meaning | Expansion for this change-id |
|---|---|---|
| `A_local_thin` | Disposable + ledger in-repo; no external heavy product | ledger=`auto_test_openspec/<change-id>/loop/ledger.json`; scratch=`test_cache/<change-id>/`; pytest=`test_cache/<change-id>/pytest/`; product/bundle/gui=`null` |
| `B_external_heavy` | Heavy outputs outside the repo; repo keeps ledger + thin pointers | ledger as in A; scratch=`test_cache/<change-id>/` or `null` if tests are also external; **product** = root declared by **this** change (propose a **new** name if missing — do not copy another change's tree); pointers=`auto_test_openspec/<change-id>/pointers/` (or a change-chosen subname) |
| `C_product_repo` | Product under the repo | ledger/scratch as in A; product=`outputs/<change-id>/` or a contract-named in-repo root (may create e.g. `outputs/<change-id>/surfaces/`) |
| `D_custom` | Rare | User supplies all five roots once; record them in the preview and do not re-ask |

### Expansion discipline

- The first key of every recommended path is **`<change-id>`**.
- Contract-named roots win; profile is only a label.
- Unnamed external heavy: propose a new root in the preview; never auto-fill
  another project's absolute path.
- Fingerprint and policy match: reuse recorded paths; do not re-ask.
- Forbidden: repo-root `tmp_pytest_*`, `.pytest_tmp`, unnamed `_tmp/`.

Naming a path profile creates no directory and no retained bundle. Bundle stays
`null` unless `retention=full`, a task `TEST:` block, or explicit legacy audit
mode requires one.

### Anti-pattern (do not copy)

Do not treat any other change's external tree as the default template for a new
change.

## When design / contract changes later

| Situation | Fingerprint / optional diagnostic behavior |
|---|---|
| Narrative-only design/proposal/specs edit | `reseal` digest refresh; **no** new path QA if policy unchanged |
| `major-revision` / material-delta (ACCEPT/TEST/deps/SUPERSEDES) | Rebuild the active fingerprint; `supervised` pauses for the semantic decision, `full_auto` may restamp with a reason |
| `additive-extension` (append tasks) | Grill new ACCEPT/TEST/paths, then rebuild the fingerprint over **old+new** refs |
| `succession` | Lock behavior and MODIFIED specs, then rebuild the full-registry fingerprint |
| Task `promote` `[x]` | Never a policy stamp; fingerprint ignores checkboxes |

When this optional diagnostic is used, put `configured / recommended /
shortfall` in it only when remaining work exceeds the recorded change cap.
Never silently raise an inherited value, turn the advisory into a start-work
gate, or make it a later `check` warning.

## QA rule

When a preview is needed, ask one frontier question for the whole policy delta.
Acceptance applies only to the listed diagnostic/policy fields; it is not a
general implementation ceremony and does not authorize destructive or external
operations. Run `seal --confirmed` only when the accepted change belongs to the
human-confirmed irreversible-policy set above.

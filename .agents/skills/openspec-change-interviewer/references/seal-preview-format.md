# Seal preview format (盖章预览)

`seal` = **开工盖章**：把当前整份 active 合同义务 + 落盘档位 + 预算/autonomy/ceiling
冻进 `loop.json`。它不勾选任何任务；`promote` 才勾选。一次盖章覆盖指纹内
**全部** active `[#R…]`（一次性全盖义务面）。

Write `openspec/changes/<change-id>/seal-preview.md` as the **last grilling
turn**. Accepting this packet **is** the stamp ceremony; then run
`seal --confirmed` only to freeze the accepted packet. Writing the preview
alone is not authorization.

## Required sections

```markdown
# 盖章预览 / Seal preview: <change-id>

Entry mode: new-idea | major-revision | additive-extension | succession | execution-ready
Stamp kind: first-seal | policy-restamp | semantic-restamp

## 1. Contract coverage（一次性全盖）
- change_id:
- semantic fingerprint covers: **entire** active tasks.md obligations
  (checkbox / STATE neutralized) — not a single task
- All active refs stamped this turn:
  - R1 …
  - R2 …
  (list every [#R…] in the active registry)
- ready now / still blocked by deps: (informational only; still stamped)
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
grill → seal-preview（本页）→ 用户同意盖章 → loop.json → Loop 排空

### 落盘流
change-id → ledger / scratch / product|pointers（本 change 展开）

## 4. Execution policy
- retention:
- test profiles:
- budgets:
- autonomy:
- hard_ceiling:

## 5. Authority boundary
- Safe after stamp: gate, record, narrative reseal, promote, sync
- Risky always human: product --commit, credentials, git push / PR / main

## 6. 盖章确认
- [ ] 同意推荐档位与政策，盖章开工（一次性全盖上列全部 refs）
- [ ] 改档位为: A | B | C | D
- [ ] 采用新建根名: <path or null>
- [ ] 调整 budgets / autonomy / hard_ceiling: <delta>
- [ ] 本轮不盖章
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
- Fingerprint match: reuse stamp; do not re-ask paths.
- Forbidden: repo-root `tmp_pytest_*`, `.pytest_tmp`, unnamed `_tmp/`.

### Anti-pattern (do not copy)

Do not treat any other change's external tree as the default template for a new
change.

## When design / contract changes later

| Situation | Stamp behavior |
|---|---|
| Narrative-only design/proposal/specs edit | `reseal` digest refresh; **no** new path QA if policy unchanged |
| `major-revision` / material-delta (ACCEPT/TEST/deps/SUPERSEDES) | Delta grill → rewrite preview listing **all** active refs again → semantic restamp |
| `additive-extension` (append tasks) | Grill only new ACCEPT/TEST/paths → preview lists **old+new** refs (full registry) → semantic restamp |
| `succession` | Lock behavior, MODIFIED specs → full-registry preview → semantic restamp |
| Task `promote` `[x]` | Never a stamp; fingerprint ignores checkboxes |

## QA rule

One frontier question for the whole packet. Accepting defaults **is** agreeing
to stamp. Then run `seal --confirmed` as the mechanical freeze. Do not re-ask
the same boundary as a separate "please seal now" ritual.

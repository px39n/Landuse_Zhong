---
name: openspec-feature-list
description: Rebuild the derived OpenSpec feature_list.json from tasks.md. Never
  mutate the ledger or award PASS.
license: MIT
metadata:
  author: openspec
  version: "3.0"
---

# openspec-feature-list

```powershell
python -B scripts/generate_openspec_feature_list.py --change-id <change>
```

Rebuilds the compact active registry only. Does not seal, admit, or promote.
Run after interviewer task edits and before `next` when the index may be stale.

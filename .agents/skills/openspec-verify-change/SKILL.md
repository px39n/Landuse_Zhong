---
name: openspec-verify-change
description: Verify one OpenSpec task or a whole change against active acceptance and observed evidence, returning a typed semantic verdict without implementing fixes.
license: MIT
metadata:
  author: openspec
  version: "2.0"
---

# OpenSpec Verify Change

Verify active contract truth against implementation and product evidence.
Verification is evidence-first and read-only.

## Modes and admission

- `$openspec-verify-change <change-id> --task <ref>`: semantic promotion verdict
  for one selected task.
- `$openspec-verify-change <change-id>`: whole-change and GOAL closing.

Task mode requires `next_action=verify_semantic`, a current joined Apply, and
CLI `verify-mechanical` PASS. Do not rerun its command or census. `smoke` closes
mechanically; `pilot|production|canary` and legacy tasks use this skill.

Read only the selected ACCEPT, TEST, dependencies, applicable requirements,
implementation, and focused evidence. An unchecked task before promotion is
not a defect. Use recorded retention/paths; never create a full bundle unless
`retention=full` or audit was explicitly requested.

For detailed task judgment and GOAL/design closing, read
`references/semantic-verification-contract.md`. For an adopted execution
notebook, also read
`.agents/skills/openspec-apply-change/references/execution-notebook-contract.md`.
Verify remains read-only and never expands WRITE_SCOPE or human authority.

## Verdict

- `PASS`: observed checks and semantics match ACCEPT.
- `FAIL`: implementation or tests violate a stable contract.
- `BLOCKED`: required authority, environment, dependency, or evidence is absent.
- `DEVIATED`: execution may succeed, but source, target, metric, product, or
  research direction differs from ACCEPT.

Return exactly one `PASS|FAIL|BLOCKED|DEVIATED` verdict. Never edit ACCEPT,
tasks, feature state, notebook, manifest, or product; only the supervisor may
promote.

```text
VERDICT: PASS|FAIL|BLOCKED|DEVIATED
REF: R<n>|whole-change
RISK: contract-only|normal|high
ASSURANCE: full|reduced
EXPECTED: <accepted outcome>
OBSERVED: <verified outcome>
EVIDENCE:
- <file, artifact, or command pointer>
NEXT:
- <one bounded action>
```

For `BLOCKED` or `DEVIATED`, return the structured comparison to
`$openspec-unblock-research`; do not implement the repair.

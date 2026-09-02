# Canonical Role Adapter Matrix

Source registry: `canonical-roles.json` (`aili-canonical-roles/v1`, 21 records).
This table is the local OpenSpec Loop projection. It is authoritative for local
runtime routing. `general` is never a formal owner and is intentionally absent.
Upstream canonical `rose` remains a non-deployable provenance/legacy-input row.
Lowercase `zpy` (display `ZPY`) is the local non-deployable supervisor-direct
and prompt-validation role ID; it is outside the upstream 21-role namespace.

| Role ID | Local entry | Deployable | Writable scope | Local boundary |
|---|---|---:|---|---|
| `zpy` | `openspec-loop-engineering` | no | supervisor-only | New local direct output; always `agent=null`, zero scheduling headcount, and never authentication authority. |
| `rose` | `openspec-loop-engineering` | no | supervisor-only | Upstream provenance and explicit legacy/bootstrap input only; never a new inferred local output. |
| `solution-architect` | `openspec-explore` | conditional | none | Bounded technical options and impact analysis only. |
| `implementer` | `openspec-apply-change` | yes | task-owned implementation or contract files | One packet = one ref = one supervisor-owned Apply attempt. |
| `code-scout` | `openspec-explore` | yes | none | Read-only code locality and pattern scouting. |
| `doc-researcher` | `openspec-explore` | yes | none | Read-only local-doc evidence only. |
| `web-researcher` | `openspec-explore` | conditional | none | Public web evidence only; no local-file reads through this role. |
| `plan-auditor` | `openspec-explore` | yes | none | Plan/spec/task audit only; findings are advisory. |
| `code-reviewer` | `review-pipeline` | conditional | none | Specialist review only; never final Verify or PASS. |
| `security-auditor` | `review-pipeline` | conditional | none | Scoped trust-boundary review only. |
| `test-engineer` | `openspec-apply-change` | yes | task-owned test files only | Must stop rather than edit production files. |
| `test-coverage-reviewer` | `review-pipeline` | conditional | none | Coverage sufficiency review only. |
| `pr-test-analyzer` | `review-pipeline` | conditional | none | Diff/CI test impact analysis only. |
| `ai-regression-scout` | `review-pipeline` | conditional | none | Prompt/skill/routing regression scenarios only. |
| `silent-failure-reviewer` | `silent-failure-hunting` | conditional | none | One false-success question at a time. |
| `browser-qa-runner` | `openspec-verify-change` | conditional | approved evidence root only | No product, contract, task, or test edits. |
| `e2e-artifact-runner` | `openspec-verify-change` | conditional | approved evidence root only | Evidence package only; no product edits. |
| `convergence-reviewer` | `review-pipeline` | conditional | none | Consistency and completeness review only. |
| `spec-miner` | `openspec-explore` | yes | none | Candidate requirements from current code/tests/docs only. |
| `agent-evaluator` | `review-pipeline` | conditional | none | Evaluate bounded worker output only. |
| `opensource-sanitizer` | `review-pipeline` | conditional | none | Public/package exposure review only. |
| `web-performance-auditor` | `review-pipeline` | conditional | none | Measured performance review only. |

## Local rules

- Adapter mappings may narrow tools, actions, commands, and syntax but must not
  widen repository authority, lifecycle ownership, verification selection, or
  final-verdict authority.
- New permitted direct routes and records emit `zpy`. Explicit `rose` remains
  readable only for predecessor/history and the bounded bootstrap task.
- A `zpy` or `rose` label never authenticates a caller, creates authority, or
  permits native host dispatch; both remain supervisor-direct with `agent=null`.
- Ordinary Apply routing comes only from compiled `next`; do not repeat a
  proactive scan. For a genuine `review|explore` residual question, the
  specialist scan may return a named `direct` reason or `blocked` when the local
  entry is unavailable, incapable, or overlaps another package; it must not
  substitute `general`.
- Only `implementer` may edit task-owned implementation or contract files.
- Only `test-engineer` may edit task-owned test files.
- Only `browser-qa-runner` and `e2e-artifact-runner` may write, and only to the
  already approved evidence root.
- All other dispatched roles are read-only.

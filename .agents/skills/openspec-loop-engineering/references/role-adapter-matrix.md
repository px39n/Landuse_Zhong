# Canonical Role Adapter Matrix

Source registry: `canonical-roles.json` (`aili-canonical-roles/v1`, 21 records).
This table is the local OpenSpec Loop projection. It is authoritative for local
runtime routing. `general` is never a formal owner and is intentionally absent.
`rose` is a non-deployable compatibility alias for supervisor-direct work only.

| Role ID | Local entry | Deployable | Writable scope | Local boundary |
|---|---|---:|---|---|
| `rose` | `openspec-loop-engineering` | no | supervisor-only | Direct-only alias; never dispatched or installed as a worker. |
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
- If the selected local entry is unavailable, lacks the needed capability, or
  would overlap another current package, the proactive scan must return a named
  `direct` reason or `blocked`; it must not substitute `general`.
- Only `implementer` may edit task-owned implementation or contract files.
- Only `test-engineer` may edit task-owned test files.
- Only `browser-qa-runner` and `e2e-artifact-runner` may write, and only to the
  already approved evidence root.
- All other dispatched roles are read-only.

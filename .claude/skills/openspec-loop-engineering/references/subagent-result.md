# Local Dispatch Result

This result is the local portable terminal envelope written under the same
single-handoff contract. It feeds supervisor join and Verify; it does not carry
final PASS authority.

```text
CANONICAL RESULT:
agent: agents/<agent-id>
result_id:
trace_id:
package_id:
role_id:
local_entry:
ref:
apply_attempt:
status: completed | failed | partial | blocked | unverified
confidence: HIGH | MED | LOW | VERY LOW | UNKNOWN
declared_repository:
artifact_destination:
inspected_scope:
write_scope:
summary:
evidence:
changed_files:
verification:
checks:
freshness:
skipped_checks:
blockers:
risks:
unverified:
continuation_recommendation: same-package | new-package | none
findings:
```

Each finding has exactly:

```text
finding_id:
source:
claim:
severity:
evidence_anchors:
affected_requirement:
proposed_disposition: fix | refute-with-counter-evidence | accept-named-risk | Unverified-block
required_action:
verification:
```

## Rules

- Evidence must support the reported status.
- `agent`, `package_id`, `role_id`, `ref`, and `apply_attempt` must match the
  dispatch packet.
- Portable evidence is required. An opaque runtime-private id cannot be the
  only completion proof.
- `same-package` is valid only when all package-defining fields remain
  unchanged.
- Keep raw logs, broad dumps, and full files out of the result.
- A no-finding result still reports inspected scope, checks, freshness, skipped
  checks, blockers, risks, and `unverified` items.
- The result does not issue final PASS, acceptance, promotion, or integration.
- A terminal `failed`, empty, `partial`, `blocked`, or `unverified` result does not authorize automatic redispatch, old-context resume, role swap, or scope expansion.
- Semantic check failure is not a transient retry.
- No result grants permission for another operation, nested delegation, or
  final-verdict ownership.

## Local exclusions

- Do not materialize delivery-flow lifecycle ownership.
- Do not materialize foreign attached-repository ownership, external worktree
  identity, formal board ownership, per-worker status journals, dispute
  sidecars, or a second ledger.
- Do not duplicate or rebind repository identity, approvals, or command
  authority outside the packet.

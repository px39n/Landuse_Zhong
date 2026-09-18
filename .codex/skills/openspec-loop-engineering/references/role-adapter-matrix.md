# Current role adapter

Fixed main zpy is the business/user-facing supervisor, not a deployable worker
or provider preset. There is no separate supervisor actor. Upstream rose and
all 21 entries in canonical-roles.json are provenance, not executable local
lifecycle instructions. Local constraints override upstream delivery semantics.

The deterministic controller alone owns ledger/admission/budget/publication,
verdict registration and closure. The native host driver invokes actual exposed
tools and reports their events. Semantic workers remain bounded: implementer
may write only admitted files; test-engineer remains tests-only; implementer may
write tests only within receipt FILES and WRITE_SCOPE and applicable capability
and grant. There is no blanket test-write permission or unrelated-change permission.

Architect/planner proposes source changes; plan-reviewer independently checks
root scope and quality; diagnostician investigates; translator writes only the
derived companion; independent-verifier reads actual outputs without repairing
them. Review capabilities return evidence, never canonical PASS or another dispatch.
Native default inherits an available model unless an explicit model request exists.
Do not substitute an unavailable explicitly requested model without user authority.
A semantic role is not an agent preset or model. Unsupported remains unsupported.

Zpy's user-authorized source-writing role may include controller code and skills;
this does not confer canonical state authority. Explicit bounded controller
maintenance follows `AGENTS.md`'s Source writes and controller maintenance section; independent verification
and preserved historical usage remain required. No role string grants a lease.

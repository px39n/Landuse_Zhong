# Admitted host work

The current `next` facade returns `receipts[]` with `receipt_id`, `ref`,
`action` and a packet for Apply. Use the controller's supplied identifiers,
ACCEPT, TEST, WRITE_SCOPE, goals and design anchors; do not manufacture
authority, hashes or a frozen source snapshot.

The native host driver uses actual exposed spawn and lifecycle tools.
Python cannot create native execution evidence by claiming a tool label.
Use available native default with an inherited model unless explicitly requested.
Unsupported capability stays unsupported. Never launch a separate supervisor.

Each worker receives only its admitted scope, relevant source and action schema.
Workers do not dispatch nested agents, mutate tasks/ledger or enlarge authority.
Use an independent verifier context. Shared-worktree verification waits for all
actually dispatched writers to join. Main supervises; the controller owns
admission, budget, verdict registration and closure.

Reconcile receipts with actual host state before waiting or recovering.
Cancellation and a timeout do not by themselves prove native termination.
Confirm that an earlier writer has stopped before controller cancellation and
reissue; report uncertain ownership instead of launching another writer.
Return the action's typed result file and relinquish write authority.

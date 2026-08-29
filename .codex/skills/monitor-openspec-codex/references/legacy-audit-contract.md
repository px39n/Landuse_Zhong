# Legacy retained-audit contract

Load this reference only after audit mode has been explicitly selected. It is
not part of ordinary Loop context.

## Roles

- Supervisor selects one eligible active task, dispatches one worker attempt,
  independently validates it, and owns final bookkeeping.
- Worker implements only the selected task, produces the requested bundle, and
  reports observations without PASS/FAIL.
- Only the supervisor may toggle a checkbox, mark `passes=true`, write
  `EVIDENCE`, append `progress.txt`, or append `runs.log`.

## Required paths

Use only paths confirmed in the sealed full-retention policy:

- contract: `openspec/changes/<change-id>/`
- immutable run bundle: the configured bundle root
- product output: the configured product root
- disposable pytest/scratch: the configured scratch root
- optional GUI/Colab evidence: the configured GUI/Colab root
- PASS ledger: the configured audit ledger

Do not invent a third staging root. Never place machine-produced logs, inputs,
screenshots, or outputs inside proposal/design/spec/task files.

## Per-attempt sequence

1. Run `check` and `plan`; reject fingerprint drift or a non-ready task.
2. Allocate a monotonic run id and immutable run folder.
3. Record startup provenance: timestamp, worker command/runtime, Git base, task
   ref, and exact task text.
4. Worker implements one task and writes a runnable bundle with the task's
   accepted inputs, outputs, commands, and assertions.
5. Worker adds at most one `BUNDLE (RUN #n)` pointer under that task.
6. Supervisor inspects bundle completeness and runs the declared checks.
7. Supervisor compares product semantics to ACCEPT, including source, target,
   metric, schema, and research direction.
8. Supervisor records exactly one verdict:
   `PASS | FAIL | BLOCKED | DEVIATED`.
9. On PASS only, toggle the selected checkbox, update the matching feature
   state, append `EVIDENCE (RUN #n)`, and append audit ledgers.
10. Record the event in the Loop ledger so audit attempts still consume the
    sealed change/revision budgets.

Every run folder is append-only after verdict. A retry uses a new run id and
folder.

## Minimum bundle

- task identity and exact ACCEPT/TEST references
- startup provenance
- executable CLI command or GUI/Colab runbook
- reproducible input pointers when inputs are required
- produced output or immutable output pointer
- machine-decidable assertions
- command logs and exit codes
- screenshot/checkpoint index when GUI or Colab applies

A command exit code of zero is insufficient. If output is validly produced but
semantically wrong, record `DEVIATED` and invoke the v2 unblock protocol.

## Failure policy

- Missing authority, credentials, environment, or required retained path:
  `BLOCKED`.
- Implementation/test defect within an unchanged contract: `FAIL`.
- Successful execution with wrong source/target/metric/direction: `DEVIATED`.
- Repeated result fingerprint or exhausted budget: stop; do not create another
  bundle.
- Contract ambiguity or required scope/design change: return to interviewer and
  reseal before more execution.

Use `portable-unblock-report.v2` for direction-changing diagnosis. Ordinary
failed attempts do not require a permanent unblock report in addition to their
immutable audit bundle.

## Completion report

Report:

- last run id and task ref;
- verdict and exact verification commands;
- retained bundle/product/evidence pointers;
- current revision/change budget counters;
- whether the next action is another ready task, interviewer amendment, or
  budget stop.

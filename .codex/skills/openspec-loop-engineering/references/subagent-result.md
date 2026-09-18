# Semantic result transport

Return the action's typed JSON result in the approved result file. The supervisor
submits it through `record --ref <ref> --kind <kind> --result <file>`.
Apply requires its current `--receipt <id>`; semantic Verify and in-process
Unblock use the current controller's receipt requirements. Do not borrow an
Apply receipt for another action or fabricate native evidence.

Apply fields: status, changed_files, evidence_refs, blockers, risks, skipped_checks.
Unblock adds disposition and reasons; use its skill's disposition values.
Semantic verification fields: verdict, reasons, evidence_refs, duration_seconds.
Use the current action schema. Source authoring and advisory Explore/Review
are not separate runtime verdicts or permission to write the ledger.

Evidence refs point to approved current files. Report failed, blocked, deviated
or empty work honestly; a missing result must not become PASS.
Independent verification supplies the semantic judgment; only the controller
registers it. Worker return relinquishes its write authority; a message alone
cannot authorize more implementation.

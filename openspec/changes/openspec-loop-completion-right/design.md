# Design: execution chapters, charged cycle stamps, and completion right

## Context

The active registry fingerprint is a whole-registry identity. A task, one
`ACCEPT:` line, Apply, Verify, Unblock, and a checkbox are not independent
revisions. The current helper nevertheless compares the number of productive
fingerprint episodes with `change.max_revisions` inside every ordinary gate.
This is non-atomic: a permitted Apply can change the count that a later Verify
sees.

The successor keeps finite governance while moving it to the only place it can
be sound: before an obligation-changing semantic stamp. Existing host and join
rules remain unchanged.

## Decision 1: Vocabulary is operative

- **ACCEPT** is the exact acceptance text present when an execution chapter
  starts. Work against that text consumes the ref's Apply allowance. Meeting it
  leads only to Verify and promotion. Editing the text after work starts means
  changing the ruler and may consume a cycle stamp.
- **Execution chapter** is one active-registry lineage rooted in an admitted
  `contract_fingerprint`. It can contain many refs and Apply attempts.
- **Cycle stamp** is an obligation-changing semantic reseal within that active
  chapter. A stamp changes the whole-registry fingerprint once even when it
  changes several clauses.
- **Chapter-outside stamp** is the first confirmed interviewer/human semantic
  stamp before execution, or a confirmed stamp after the previous chapter has
  fully drained with no ready, blocked, deviated, or in-progress ref. It opens
  the next chapter and is not charged.

Initial seal/init, narrative reseal, Apply, Verify, Unblock, Explore, promotion,
checkbox, and `STATE:` do not themselves consume a cycle stamp.

## Decision 2: Minimal stamp state in loop.json

`loop.json` gains one policy-state object, not a second receipt or ledger:

```json
{
  "stamp_state": {
    "chapter_id": "<root fingerprint>",
    "semantic_stamps_in_chapter": 0,
    "charged_cycle_stamps": 0,
    "self_confirmed_refs": {},
    "last_stamp": null
  }
}
```

`last_stamp`, when present, records only source/target fingerprints,
`stamp_source`, optional ref, whether it was charged, the required reason, and
time. `self_confirmed_refs` maps a source fingerprint to refs already using the
one supervised exception. No attempt evidence or product log is copied into
this object.

New initialization writes zero counters. A legacy config without this object
adopts its already-current fingerprint as an admitted chapter with zero new
cycle stamps; historical VFD episodes do not become a new global stop.

## Decision 3: Stamp classification and admission

Classification happens before changing `tasks.md`, `feature_list.json`, or
`loop.json`:

1. `apply-revision` is always charged.
2. `stamp_source=unblock_self_confirm` in a valid blocking window is charged.
3. A semantic reseal after the current fingerprint episode has an Apply or
   Unblock record is charged unless the prior chapter is fully terminal and a
   confirmed interviewer stamp is opening a new chapter.
4. The second and later semantic stamp in a still-undrained chapter is charged.
5. The first confirmed pre-execution stamp is free. A confirmed stamp after a
   fully terminal drain opens a new zero-count chapter.

`--confirmed` alone never changes a charged classification. A charged stamp is
rejected when `charged_cycle_stamps >= candidate max_revisions`; otherwise it
increments the count exactly once. The default `3` therefore permits three
cycle stamps and rejects the fourth before writes. It is neither a lifetime
contract cap, a task-count formula, nor three stamps per task or ACCEPT line.

Once `loop.json.contract_fingerprint` matches the registry, that fingerprint is
admitted. Ordinary gate never rechecks `max_revisions`, and lowering the cap
does not revoke it.

## Decision 4: New work, closure, and unblock are separate

- `gate --kind apply|explore` checks fingerprint, active minutes, its
  kind-specific allowance, and only the last two terminal records from the
  current episode with the same ref and kind.
- `record`, join, Verify, promote, sync, goals, design-verify, summary,
  stop-hook, and review are closure. They cannot be stopped by
  `max_revisions`, active-minute caps, or the generic breaker, but retain their
  own fingerprint/join/evidence/verdict preconditions.
- `gate --kind unblock` checks only the selected ref's blocking evidence,
  `max_unblock_runs`, and the second-unblock new-evidence/terminal rule.
- `gate` requires `--kind`; Apply, Explore, Verify, and Unblock require `--ref`.

Only actual budget increases count against `max_self_extensions`. Decreasing a
budget is not a self-extension and cannot revoke the already admitted
fingerprint.

## Decision 5: Blocking-window self-restamp

No new top-level command is created. `reseal` accepts the narrow optional
surface:

```text
--stamp-source unblock_self_confirm
--ref <R#>
--candidate-tasks <path under configured scratch>
--reason <required>
```

The active registry must still match the sealed fingerprint. Blocking
authority requires same-ref `blocked|deviated` evidence in the current episode
or the latest same-ref Unblock disposition `amend_spec`. The ref must not be
passed. The same ref/source fingerprint can use this exception once. A second
Unblock is terminal and cannot self-restamp.

The sole supervisor creates the candidate. Apply workers may never edit active
or candidate `tasks.md`. The validator requires identical ref set/order,
task ids, DAG, dependencies, `SUPERSEDES`, GOALs, role/execution/join/worktree,
and WRITE_SCOPE. Other refs must be unchanged. The selected ref may only:

- clarify, add a missing gate to, or mechanically correct ACCEPT without
  widening it;
- repair an executable TEST without expanding scope;
- correct FILES while leaving derived WRITE_SCOPE unchanged;
- change one failure-local sentence.

Retention/paths/product/autonomy/hard ceiling, budget raises, other refs, DAG,
WRITE_SCOPE expansion, new external roots, passed ACCEPT, proposal scope,
GOALs, credentials, destructive writes, Git/PR/main, and acceptance weakening
are denied. Any natural-language replacement the helper cannot mechanically
prove non-widening returns to the interviewer.

The command snapshots the three active files, validates and generates the
candidate fingerprint/feature payload before writes, then uses atomic
single-file replacements. A caught failure restores original bytes. This is
command-level failure atomicity under the sole supervisor; process/power-loss
crash atomicity remains out of scope with CAS/journal.

## Decision 6: Defaults and inheritance

Every new seal or missing-loop thin initialization defaults to:

```text
task.max_apply_attempts      = 2
task.max_unblock_runs        = 2
change.max_revisions         = 3
revision.max_explore_runs    = 1
revision.max_active_minutes  = 120
change.max_active_minutes    = 360
```

Task count may increase active minutes but never `max_revisions`. A prior
explicit value is inherited when reseal omits `--set-*`. Historical values 1,
5, or 14 are fixtures/change-local policy, not defaults, and are not migrated
by this successor.

## Decision 7: Writers and alignment

- Apply workers never edit `tasks.md`.
- An implementer may edit proposal/design/specs only when the task WRITE_SCOPE
  permits it; the supervisor then performs narrative reseal with no cycle
  stamp and no interviewer.
- Normal supervised task obligations require interviewer/user confirmation.
- The blocking-window exception is the sole unconfirmed supervised semantic
  stamp. `full_auto` retains `apply-revision`, which is always charged.

Before every active Loop call the supervisor aligns current ACCEPT, applicable
main/delta requirements, `loop.json`, the canonical skill, and the action
classification. Same-turn reuse is valid only while fingerprint, narrative
digest, and canonical skill state remain unchanged. The latch creates no new
command, receipt, worker, or ledger.

## VFD fixture boundary

Synthetic fixtures may reproduce high historical episode counts and locally
sealed values. They MUST NOT read or modify live VFD change trees or ledgers,
encode 14/5 as defaults, run product tasks, or assert unrelated recycle
requirements.

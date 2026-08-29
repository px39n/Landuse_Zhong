# Unblock v2 examples

## Exit code 0, but the result is wrong

This is a semantic deviation, not a successful attempt and not necessarily a
hard error.

```json
{
  "schema": "portable-unblock-report.v2",
  "meta": {
    "created_at": "2026-08-28T12:00:00Z",
    "created_by": "codex",
    "change_id": "extend-sectoral-data",
    "task_id": "1.38",
    "attempt_id": "task-1.38-attempt-002"
  },
  "trigger": {
    "trigger_type": "output_deviation",
    "summary": "The extractor exited 0 but selected a neighboring table.",
    "error_excerpt": null
  },
  "comparison": {
    "expected_outcome": "Rows come from the accepted table family and year.",
    "observed_outcome": "Fourteen target cells came from a neighboring table.",
    "deviation_magnitude": "14 of 28 target cells",
    "acceptance_refs": ["tasks.md#R38", "specs/table-selection/spec.md"],
    "evidence_refs": ["ev:cmd:1", "ev:artifact:1"],
    "last_good_baseline": "run-20260827"
  },
  "diagnosis": {
    "facts": [
      "The command exited 0.",
      "The selected page id differs from the accepted baseline."
    ],
    "unknowns": ["Whether ranking or contract ambiguity caused the choice."],
    "hypotheses": [
      {
        "id": "H1",
        "claim": "A generic keyword outweighs table-family evidence.",
        "supporting_evidence": ["ev:artifact:1"],
        "contradicting_evidence": [],
        "confidence": "medium"
      },
      {
        "id": "H2",
        "claim": "The current acceptance text allows both pages.",
        "supporting_evidence": ["ev:cmd:1"],
        "contradicting_evidence": ["ev:artifact:1"],
        "confidence": "low"
      }
    ]
  },
  "research": {
    "providers_used": ["repo", "command"],
    "evidence": [
      {
        "id": "ev:cmd:1",
        "kind": "command",
        "pointer": "python scripts/check_selection.py --case disputed",
        "summary": "The run exited 0 and emitted the wrong page id.",
        "captured_at": "2026-08-28T12:00:00Z"
      },
      {
        "id": "ev:artifact:1",
        "kind": "artifact",
        "pointer": "test_cache/extend-sectoral-data/selection.json",
        "summary": "The selected table family differs from ACCEPT.",
        "captured_at": "2026-08-28T12:01:00Z"
      }
    ]
  },
  "discriminating_probe": {
    "action": "Emit component scores for the two disputed pages.",
    "distinguishes": ["H1", "H2"],
    "expected_signal": "Only H1 predicts a table-family score inversion.",
    "budget": {"max_calls": 1, "max_seconds": 60},
    "verify": "Compare component scores and selected page id."
  },
  "key_conclusions": [
    {
      "claim": "The current output must not be promoted despite exit code 0.",
      "confidence": "high",
      "evidence_ids": ["ev:cmd:1", "ev:artifact:1"]
    }
  ],
  "guidance": [
    {
      "step": 1,
      "action": "Run the one-case score probe.",
      "why": "It distinguishes implementation drift from contract ambiguity.",
      "verify": "The score components explain the selected page.",
      "fallback": "Return to interviewer if both pages satisfy the written contract."
    }
  ],
  "disposition": {
    "action": "targeted_probe",
    "rationale": "One bounded probe separates the two live hypotheses."
  },
  "sinks": {
    "requested": [{"type": "return_only", "content": "both"}],
    "applied": [],
    "errors": []
  }
}
```

## Hard-error variant

Use the same shape with `trigger_type=hard_error`; place the concise exact
error in `error_excerpt`. The expected/observed comparison remains required so
the investigation stays tied to ACCEPT rather than stopping at stack-trace
repair.

## v1 compatibility

Readers may consume `portable-unblock-report.v1`. Writers emit v2 and do not
rewrite historical v1 reports solely for migration.

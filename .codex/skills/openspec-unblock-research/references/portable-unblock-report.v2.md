# Portable Unblock Report — v2

The v2 report treats a hard error as one possible trigger rather than the
definition of a blocker. JSON is canonical; Markdown guidance is derived.

## Shape

```json
{
  "schema": "portable-unblock-report.v2",
  "meta": {
    "created_at": "2026-08-28T12:00:00Z",
    "created_by": "codex",
    "change_id": "example-change",
    "task_id": "1.38",
    "attempt_id": "task-1.38-attempt-001"
  },
  "trigger": {
    "trigger_type": "output_deviation",
    "summary": "The command succeeded but selected the wrong yearbook table.",
    "error_excerpt": null
  },
  "comparison": {
    "expected_outcome": "Accepted rows come from the named table and year.",
    "observed_outcome": "Rows came from a neighboring table with exit code 0.",
    "deviation_magnitude": "14 of 28 target cells",
    "acceptance_refs": ["tasks.md:44", "specs/table-selection/spec.md"],
    "evidence_refs": ["ev:cmd:1", "ev:artifact:1"],
    "last_good_baseline": "run-20260827 or null"
  },
  "diagnosis": {
    "facts": ["..."],
    "unknowns": ["..."],
    "hypotheses": [
      {
        "id": "H1",
        "claim": "The page ranking overweights a generic keyword.",
        "supporting_evidence": ["ev:artifact:1"],
        "contradicting_evidence": [],
        "confidence": "medium"
      }
    ]
  },
  "research": {
    "providers_used": [],
    "evidence": [
      {
        "id": "ev:cmd:1",
        "kind": "repo|command|artifact|docs|web",
        "pointer": "path, command, or URL",
        "summary": "short verified observation",
        "captured_at": "2026-08-28T12:00:00Z"
      }
    ]
  },
  "discriminating_probe": {
    "action": "Run ranking on the two disputed pages with component scores emitted.",
    "distinguishes": ["H1", "H2"],
    "expected_signal": "The accepted page must win on table-family and year evidence.",
    "budget": {"max_calls": 1, "max_seconds": 60},
    "verify": "Compare the emitted score components and selected page id."
  },
  "key_conclusions": [
    {
      "claim": "Current output cannot be promoted despite exit code 0.",
      "confidence": "high",
      "evidence_ids": ["ev:cmd:1", "ev:artifact:1"]
    }
  ],
  "guidance": [
    {
      "step": 1,
      "action": "Run the discriminating probe.",
      "why": "It separates ranking failure from contract ambiguity.",
      "verify": "Named score comparison is retained.",
      "fallback": "Pause for spec amendment if both pages satisfy the current contract."
    }
  ],
  "disposition": {
    "action": "targeted_probe",
    "rationale": "One bounded experiment can choose between the leading hypotheses."
  },
  "sinks": {"requested": [], "applied": [], "errors": []}
}
```

Validate machine output against
`references/portable-unblock-report.v2.schema.json`.

## Invariants

- `expected_outcome` and `observed_outcome` are required.
- `error_excerpt` is nullable.
- Every conclusion has at least one evidence id.
- The probe states what competing hypotheses it distinguishes.
- The disposition is exactly one of `retry`, `targeted_probe`, `amend_spec`,
  `supersede_task`, or `stop_budget`.
- Sink paths are caller-controlled; `return_only` is valid.

## Compatibility

Readers may accept v1 reports. Writers emit v2 unless a caller explicitly
requires v1. Do not rewrite historical v1 reports solely for migration.

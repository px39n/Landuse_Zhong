# Unblock v2 templates

## Minimal caller context

Use this return-only input for an ordinary Loop attempt. `error_excerpt` is
optional; expected and observed outcomes are not.

```json
{
  "change_id": "",
  "task_id": "",
  "attempt_id": "",
  "trigger_type": "hard_error|output_deviation|quality_regression|wrong_source_or_target|evidence_gap|no_progress|contract_mismatch",
  "summary": "",
  "error_excerpt": null,
  "expected_outcome": "",
  "observed_outcome": "",
  "acceptance_refs": [],
  "evidence_refs": [],
  "last_good_baseline": null,
  "already_tried": [],
  "sinks": [{"type": "return_only", "content": "both"}]
}
```

## Hypothesis template

```json
{
  "id": "H1",
  "claim": "",
  "supporting_evidence": [],
  "contradicting_evidence": [],
  "confidence": "low"
}
```

## Discriminating probe template

```json
{
  "action": "",
  "distinguishes": ["H1", "H2"],
  "expected_signal": "",
  "budget": {"max_calls": 1, "max_seconds": 60},
  "verify": ""
}
```

## Disposition template

Choose exactly one action.

```json
{
  "action": "retry|targeted_probe|amend_spec|supersede_task|stop_budget",
  "rationale": ""
}
```

## Durable sink template

Ordinary attempts remain `return_only`. Add a durable sink only when the result
changes task direction, establishes a long-lived blocker, or the user requests
audit retention.

```json
[
  {"type": "json_file", "content": "report_json", "path": "openspec/changes/<change-id>/unblock/<attempt-id>.json"},
  {"type": "markdown_file", "content": "guidance_md", "path": "openspec/changes/<change-id>/unblock/<attempt-id>.md"}
]
```

Sink paths are caller-controlled. Never invent a D-drive, cache, bundle, or
repository evidence path.

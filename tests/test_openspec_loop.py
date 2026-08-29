from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "openspec_loop.py"


def load_loop_module():
    """Import the helper in-process, for guards that need a fault injected."""
    if str(SCRIPT_PATH.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT_PATH.parent))
    import openspec_loop

    return openspec_loop


def write_contract(repo_root: Path, change_id: str, tasks_text: str, feature_payload: dict) -> None:
    change_dir = repo_root / "openspec" / "changes" / change_id
    change_dir.mkdir(parents=True, exist_ok=True)
    (change_dir / "tasks.md").write_text(tasks_text, encoding="utf-8")
    (change_dir / "feature_list.json").write_text(
        json.dumps(feature_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_loop(repo_root: Path, *args: str, expected_exit: int = 0) -> dict:
    normalized = list(args)
    if normalized and normalized[0] == "plan" and "--advisory" not in normalized:
        normalized.append("--advisory")
    if normalized and normalized[0] in {"record", "gate", "summary"}:
        ledger_index = normalized.index("--ledger-path") + 1 if "--ledger-path" in normalized else None
        if ledger_index is not None:
            fingerprint = ensure_sealed_runtime(repo_root, Path(normalized[ledger_index]))
            if "--contract-fingerprint" in normalized:
                fp_index = normalized.index("--contract-fingerprint") + 1
                normalized[fp_index] = fingerprint
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--repo-root", str(repo_root), *normalized],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == expected_exit, result.stderr or result.stdout
    return json.loads(result.stdout)


def run_loop_bytes(repo_root: Path, *args: str, expected_exit: int = 0, env: dict[str, str] | None = None) -> bytes:
    normalized = list(args)
    if normalized and normalized[0] == "plan" and "--advisory" not in normalized:
        normalized.append("--advisory")
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--repo-root", str(repo_root), *normalized],
        capture_output=True,
        text=False,
        check=False,
        env=env,
    )
    assert result.returncode == expected_exit, result.stderr or result.stdout
    return result.stdout


def ensure_sealed_runtime(repo_root: Path, ledger: Path) -> str:
    change_dir = repo_root / "openspec" / "changes" / "demo"
    if not (change_dir / "tasks.md").exists():
        write_contract(
            repo_root,
            "demo",
            "## Active Task Registry\n\n- [ ] 0.1 Runtime task [#R0]\n",
            base_features([("R0", "0.1", False, False)]),
        )
    loop_path = change_dir / "loop.json"
    if not loop_path.exists():
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--repo-root",
                str(repo_root),
                "seal",
                "demo",
                "--confirmed",
                "--retention",
                "thin",
                "--ledger-path",
                str(ledger),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(loop_path.read_text(encoding="utf-8"))["contract_fingerprint"]


def test_stateful_commands_require_a_sealed_policy(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--repo-root",
            str(tmp_path),
            "summary",
            "demo",
            "--contract-fingerprint",
            "abc",
            "--run-id",
            "run-a",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert "not sealed" in payload["error"]
    assert not (tmp_path / "test_cache").exists()


def base_features(entries: list[tuple[str, str, bool, bool]]) -> dict:
    return {
        "change_id": "demo",
        "features": {
            ref: {
                "task_id": task_id,
                "task_checked": checked,
                "passes": passes,
            }
            for ref, task_id, checked, passes in entries
        },
    }


def test_plan_unblocks_when_a_completed_dependency_appears_later_in_tasks(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [x] 1.5 Shared locator [#R5]
  - DEPENDS_ON: none
- [ ] 1.7 Close city surface [#R7]
  - DEPENDS_ON: 1.5, 1.42
- [x] 1.42 Remaining shine years [#R42]
  - DEPENDS_ON: 1.5
"""
    features = base_features(
        [
            ("R5", "1.5", True, True),
            ("R7", "1.7", False, False),
            ("R42", "1.42", True, True),
        ]
    )
    write_contract(tmp_path, "demo", tasks, features)

    payload = run_loop(tmp_path, "plan", "demo")

    r7 = next(task for task in payload["tasks"] if task["ref"] == "R7")
    assert r7["ready"] is True
    assert r7["blocked_reasons"] == []
    assert payload["selected_ref"] == "R7"


def test_plan_uses_document_order_for_ready_tasks_without_explicit_dependencies(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [x] 1.1 Finish setup [#R1]
- [ ] 1.2 Implement parser [#R2]
- [ ] 1.3 Implement verifier [#R3]
"""
    features = base_features(
        [
            ("R1", "1.1", True, True),
            ("R2", "1.2", False, False),
            ("R3", "1.3", False, False),
        ]
    )
    write_contract(tmp_path, "demo", tasks, features)

    payload = run_loop(tmp_path, "plan", "demo")

    assert payload["ready_refs"] == ["R2"]
    assert payload["selected_ref"] == "R2"


def test_plan_selects_first_of_multiple_ready_tasks_in_tasks_order(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [x] 1.1 Baseline [#R1]
- [ ] 1.2 Branch A [#R2]
  - INDEPENDENT: yes
- [ ] 1.3 Branch B [#R3]
  - INDEPENDENT: yes
"""
    features = base_features(
        [
            ("R1", "1.1", True, True),
            ("R2", "1.2", False, False),
            ("R3", "1.3", False, False),
        ]
    )
    write_contract(tmp_path, "demo", tasks, features)

    payload = run_loop(tmp_path, "plan", "demo")

    assert payload["ready_refs"] == ["R2", "R3"]
    assert payload["selected_ref"] == "R2"


def test_plan_supports_explicit_none_without_falling_back_to_document_order(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [x] 1.1 Bootstrap [#R1]
- [ ] 1.2 Independent follow-up [#R2]
  - DEPENDS_ON: none.
- [ ] 1.3 Downstream [#R3]
"""
    features = base_features(
        [
            ("R1", "1.1", True, True),
            ("R2", "1.2", False, False),
            ("R3", "1.3", False, False),
        ]
    )
    write_contract(tmp_path, "demo", tasks, features)

    payload = run_loop(tmp_path, "plan", "demo")
    by_ref = {item["ref"]: item for item in payload["tasks"]}

    assert by_ref["R2"]["dependencies"] == []
    assert by_ref["R2"]["dependency_sources"] == ["DEPENDS_ON:none"]
    assert payload["ready_refs"] == ["R2"]


def test_plan_supports_explicit_none_with_explanatory_semicolon_note(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [ ] 1.1 Root task [#R1]
  - DEPENDS_ON: none; root task has no predecessor.
"""
    features = base_features([("R1", "1.1", False, False)])
    write_contract(tmp_path, "demo", tasks, features)

    payload = run_loop(tmp_path, "plan", "demo")

    assert not [issue for issue in payload["issues"] if "not sealed" not in issue]
    assert payload["selected_ref"] == "R1"
    assert payload["tasks"][0]["dependency_sources"] == ["DEPENDS_ON:none"]


def test_plan_supports_range_task_id_and_semicolon_note_dependencies(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [x] 1.1 R1 done [#R1]
- [x] 1.2 R2 done [#R2]
- [ ] 1.3 R3 waits on range [#R3]
  - DEPENDS_ON: R1-R2; note: R9 is historical context only.
- [ ] 1.4 R4 waits on task id [#R4]
  - DEPENDS_ON: 1.3.
"""
    features = base_features(
        [
            ("R1", "1.1", True, True),
            ("R2", "1.2", True, True),
            ("R3", "1.3", False, False),
            ("R4", "1.4", False, False),
        ]
    )
    write_contract(tmp_path, "demo", tasks, features)

    payload = run_loop(tmp_path, "plan", "demo")
    by_ref = {item["ref"]: item for item in payload["tasks"]}

    assert payload["ready_refs"] == ["R3"]
    assert by_ref["R3"]["dependencies"] == ["R1", "R2"]
    assert by_ref["R3"]["dependency_sources"] == ["R1-R2"]
    assert by_ref["R4"]["dependencies"] == ["R3"]


def test_plan_supports_semicolon_separated_dependency_segments(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [x] 1.1 R1 done [#R1]
- [x] 1.2 R2 done [#R2]
- [ ] 1.3 R3 waits on two semicolon items [#R3]
  - DEPENDS_ON: R1; R2.
"""
    features = base_features(
        [
            ("R1", "1.1", True, True),
            ("R2", "1.2", True, True),
            ("R3", "1.3", False, False),
        ]
    )
    write_contract(tmp_path, "demo", tasks, features)

    payload = run_loop(tmp_path, "plan", "demo")
    by_ref = {item["ref"]: item for item in payload["tasks"]}

    assert by_ref["R3"]["dependencies"] == ["R1", "R2"]
    assert by_ref["R3"]["dependency_sources"] == ["R1", "R2"]


def test_plan_ignores_explanatory_segments_but_still_reports_later_unknown_dependencies(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [x] 1.1 R1 done [#R1]
- [x] 1.2 R2 done [#R2]
- [ ] 1.3 R3 mixed dep segments [#R3]
  - DEPENDS_ON: R1-R2; note: `R9` is historical context only.; R404.
"""
    features = base_features(
        [
            ("R1", "1.1", True, True),
            ("R2", "1.2", True, True),
            ("R3", "1.3", False, False),
        ]
    )
    write_contract(tmp_path, "demo", tasks, features)

    payload = run_loop(tmp_path, "plan", "demo")
    by_ref = {item["ref"]: item for item in payload["tasks"]}

    assert by_ref["R3"]["dependencies"] == ["R1", "R2"]
    assert "unknown dependency `R404`" in "\n".join(by_ref["R3"]["issues"])


def test_plan_ignores_historical_sections_and_fenced_tasks(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [x] 1.1 Current done [#R1]
- [ ] 1.2 Current target [#R2]

```md
- [ ] 9.9 Fenced fake task [#R999]
```

## Historical Task Ledger (Rollback-Only; Non-Operative)

- [ ] 2.1 Historical task [#R3]
"""
    features = base_features(
        [
            ("R1", "1.1", True, True),
            ("R2", "1.2", False, False),
        ]
    )
    write_contract(tmp_path, "demo", tasks, features)

    payload = run_loop(tmp_path, "plan", "demo")

    refs = [item["ref"] for item in payload["tasks"]]
    assert refs == ["R1", "R2"]
    assert payload["selected_ref"] == "R2"


def test_plan_reports_unknown_dependencies_cycles_and_drift(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [ ] 1.1 First [#R1]
  - DEPENDS_ON: R2.
- [ ] 1.2 Second [#R2]
  - DEPENDS_ON: R1.
- [x] 1.3 Drifted [#R3]
  - DEPENDS_ON: R404.
"""
    features = base_features(
        [
            ("R1", "1.1", False, False),
            ("R2", "1.2", False, False),
            ("R3", "1.3", True, False),
        ]
    )
    write_contract(tmp_path, "demo", tasks, features)

    payload = run_loop(tmp_path, "plan", "demo")

    assert payload["selected_ref"] is None
    issues = "\n".join(payload["issues"])
    assert "dependency cycle" in issues
    assert "unknown dependency `R404`" in issues
    assert "checkbox/passes drift" in issues


def test_plan_reports_missing_feature_entries_as_state_drift(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [ ] 1.1 First [#R1]
- [ ] 1.2 Missing feature entry [#R2]
"""
    features = base_features([("R1", "1.1", False, False)])
    write_contract(tmp_path, "demo", tasks, features)

    payload = run_loop(tmp_path, "plan", "demo")
    by_ref = {item["ref"]: item for item in payload["tasks"]}

    assert payload["ready_refs"] == ["R1"]
    assert "missing feature_list entry" in by_ref["R2"]["issues"]
    assert by_ref["R2"]["ready"] is False


def test_plan_supports_legacy_no_dep_exclusions_without_clearing_all_dependencies(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [x] 1.1 Done root [#R1]
- [ ] 1.2 Incomplete branch [#R2]
- [x] 1.3 Done branch [#R3]
- [ ] 1.4 Legacy no-dep exclusions [#R4]
  - NO_DEP: R2/R3.
"""
    features = base_features(
        [
            ("R1", "1.1", True, True),
            ("R2", "1.2", False, False),
            ("R3", "1.3", True, True),
            ("R4", "1.4", False, False),
        ]
    )
    write_contract(tmp_path, "demo", tasks, features)

    payload = run_loop(tmp_path, "plan", "demo")
    by_ref = {item["ref"]: item for item in payload["tasks"]}

    assert by_ref["R4"]["dependencies"] == ["R1"]
    assert "R4" in payload["ready_refs"]


def test_plan_keeps_no_dep_yes_as_full_dependency_clear(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [x] 1.1 Done root [#R1]
- [ ] 1.2 Explicit clear [#R2]
  - NO_DEP: yes
"""
    features = base_features(
        [
            ("R1", "1.1", True, True),
            ("R2", "1.2", False, False),
        ]
    )
    write_contract(tmp_path, "demo", tasks, features)

    payload = run_loop(tmp_path, "plan", "demo")
    by_ref = {item["ref"]: item for item in payload["tasks"]}

    assert by_ref["R2"]["dependencies"] == []
    assert by_ref["R2"]["dependency_sources"] == ["independent"]


def test_plan_reports_missing_tasks_file_without_traceback(tmp_path: Path) -> None:
    feature_dir = tmp_path / "openspec" / "changes" / "demo"
    feature_dir.mkdir(parents=True, exist_ok=True)
    (feature_dir / "feature_list.json").write_text(
        json.dumps(base_features([("R1", "1.1", False, False)]), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    payload = run_loop(tmp_path, "plan", "demo")

    assert payload["selected_ref"] is None
    assert payload["tasks"] == []
    assert any("missing tasks file:" in issue for issue in payload["issues"])


def test_plan_reports_missing_feature_list_file_without_traceback(tmp_path: Path) -> None:
    change_dir = tmp_path / "openspec" / "changes" / "demo"
    change_dir.mkdir(parents=True, exist_ok=True)
    (change_dir / "tasks.md").write_text(
        "## Active Task Registry\n\n- [ ] 1.1 First [#R1]\n",
        encoding="utf-8",
    )

    payload = run_loop(tmp_path, "plan", "demo")

    assert payload["selected_ref"] is None
    assert payload["tasks"] == []
    assert any("missing feature_list file:" in issue for issue in payload["issues"])


def test_plan_emits_utf8_json_under_legacy_pythonioencoding(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [ ] 1.1 处理中文 ↔ 输出 [#R1]
"""
    features = base_features([("R1", "1.1", False, False)])
    write_contract(tmp_path, "demo", tasks, features)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252"
    stdout = run_loop_bytes(tmp_path, "plan", "demo", env=env)
    payload = json.loads(stdout.decode("utf-8"))

    assert payload["tasks"][0]["title"] == "处理中文 ↔ 输出"


def test_plan_contract_fingerprint_changes_when_an_obligation_changes(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [ ] 1.1 First [#R1]
"""
    features = base_features([("R1", "1.1", False, False)])
    write_contract(tmp_path, "demo", tasks, features)
    first = run_loop(tmp_path, "plan", "demo")

    promoted_tasks = """## Active Task Registry

- [x] 1.1 First [#R1]
"""
    write_contract(tmp_path, "demo", promoted_tasks, base_features([("R1", "1.1", True, True)]))
    promoted = run_loop(tmp_path, "plan", "demo")
    assert promoted["contract_fingerprint"] == first["contract_fingerprint"]

    changed_tasks = """## Active Task Registry

- [x] 1.1 First [#R1]
  - ACCEPT: the parser rejects an unknown ref.
"""
    write_contract(tmp_path, "demo", changed_tasks, base_features([("R1", "1.1", True, True)]))
    second = run_loop(tmp_path, "plan", "demo")

    assert first["contract_fingerprint"] != second["contract_fingerprint"]


def test_record_gate_and_summary_enforce_budgets_and_breakers(tmp_path: Path) -> None:
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    fingerprint = "abc123"

    run_loop(
        tmp_path,
        "record",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-a",
        "--ref",
        "R2",
        "--kind",
        "apply",
        "--result",
        "failure",
        "--error-text",
        "Boom failed",
        "--ledger-path",
        str(ledger),
    )
    run_loop(
        tmp_path,
        "record",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-a",
        "--ref",
        "R2",
        "--kind",
        "apply",
        "--result",
        "failure",
        "--error-text",
        "Boom failed",
        "--ledger-path",
        str(ledger),
    )

    gate = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-a",
        "--ref",
        "R2",
        "--kind",
        "apply",
        "--ledger-path",
        str(ledger),
        expected_exit=2,
    )
    assert gate["decision"] == "stop"
    assert "repeated_result_breaker" in gate["reasons"]
    assert any(reason.startswith("task_max_apply_attempts_reached:R2") for reason in gate["reasons"])

    summary = run_loop(
        tmp_path,
        "summary",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-a",
        "--ledger-path",
        str(ledger),
    )
    assert summary["attempt_count"] == 2
    assert summary["per_ref"]["R2"]["apply:failure"] == 2
    ledger_payload = json.loads(ledger.read_text(encoding="utf-8"))
    attempt = ledger_payload["episodes"][0]["runs"][0]["attempts"][0]
    assert "error_excerpt" not in attempt
    assert attempt["error_fingerprint"]
    assert "Boom failed" not in ledger.read_text(encoding="utf-8")


def test_gate_stops_after_two_no_progress_attempts_and_one_explore_budget(tmp_path: Path) -> None:
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    fingerprint = "xyz789"

    run_loop(
        tmp_path,
        "record",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-b",
        "--ref",
        "R5",
        "--kind",
        "explore",
        "--result",
        "success",
        "--ledger-path",
        str(ledger),
    )
    explore_gate = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-b",
        "--ref",
        "R5",
        "--kind",
        "explore",
        "--ledger-path",
        str(ledger),
        expected_exit=2,
    )
    assert "revision_explore_budget_exhausted:1" in explore_gate["reasons"]

    run_loop(
        tmp_path,
        "record",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-b",
        "--ref",
        "R6",
        "--kind",
        "apply",
        "--result",
        "no_progress",
        "--ledger-path",
        str(ledger),
    )
    run_loop(
        tmp_path,
        "record",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-b",
        "--ref",
        "R6",
        "--kind",
        "verify",
        "--result",
        "no_progress",
        "--ledger-path",
        str(ledger),
    )
    no_progress_gate = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-b",
        "--ref",
        "R6",
        "--kind",
        "apply",
        "--ledger-path",
        str(ledger),
        expected_exit=2,
    )
    assert "no_progress_breaker" in no_progress_gate["reasons"]


def test_gate_treats_max_subagents_as_distinct_allocated_subagents_per_run(tmp_path: Path) -> None:
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    fingerprint = "active-subagents"

    run_loop(
        tmp_path,
        "record",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-c",
        "--ref",
        "R8",
        "--kind",
        "explore",
        "--result",
        "success",
        "--subagent-id",
        "agent-a",
        "--subagent-id",
        "agent-b",
        "--ledger-path",
        str(ledger),
    )
    run_loop(
        tmp_path,
        "record",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-c",
        "--ref",
        "R8",
        "--kind",
        "verify",
        "--result",
        "success",
        "--subagent-id",
        "agent-a",
        "--ledger-path",
        str(ledger),
    )

    gate = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-c",
        "--ref",
        "R8",
        "--kind",
        "apply",
        "--max-subagents",
        "2",
        "--subagent-id",
        "agent-c",
        "--ledger-path",
        str(ledger),
        expected_exit=2,
    )
    assert "revision_max_subagents_exceeded:2" in gate["reasons"]


def test_gate_budgets_span_run_ids_within_one_revision(tmp_path: Path) -> None:
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    fingerprint = "run-scope"

    run_loop(
        tmp_path,
        "record",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-old",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "failure",
        "--ledger-path",
        str(ledger),
    )
    run_loop(
        tmp_path,
        "record",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-old",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "failure",
        "--ledger-path",
        str(ledger),
    )

    old_gate = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-old",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--max-iterations",
        "2",
        "--ledger-path",
        str(ledger),
        expected_exit=2,
    )
    assert "revision_max_iterations_reached:2" in old_gate["reasons"]

    verify_after_last_apply = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-old",
        "--ref",
        "R1",
        "--kind",
        "verify",
        "--max-iterations",
        "2",
        "--ledger-path",
        str(ledger),
    )
    assert verify_after_last_apply["decision"] == "continue"

    fresh_gate = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-fresh",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--max-iterations",
        "2",
        "--ledger-path",
        str(ledger),
        expected_exit=2,
    )
    assert "revision_max_iterations_reached:2" in fresh_gate["reasons"]

    run_loop(
        tmp_path,
        "record",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-helpers",
        "--ref",
        "R2",
        "--kind",
        "explore",
        "--result",
        "success",
        "--ledger-path",
        str(ledger),
    )
    helper_gate = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-helpers",
        "--ref",
        "R2",
        "--kind",
        "apply",
        "--max-iterations",
        "1",
        "--ledger-path",
        str(ledger),
        expected_exit=2,
    )
    assert "revision_max_iterations_reached:1" in helper_gate["reasons"]


def test_gate_counts_active_duration_not_wall_clock_waiting(tmp_path: Path) -> None:
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    fingerprint = "time-budget"

    first_gate = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-waiting",
        "--max-minutes",
        "120",
        "--ledger-path",
        str(ledger),
    )
    assert first_gate["decision"] == "continue"
    assert ledger.exists()

    payload = json.loads(ledger.read_text(encoding="utf-8"))
    run = payload["episodes"][0]["runs"][0]
    run["started_at_utc"] = "2000-01-01T00:00:00Z"
    ledger.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    expired_gate = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-waiting",
        "--max-minutes",
        "1",
        "--ledger-path",
        str(ledger),
    )
    assert expired_gate["decision"] == "continue"

    run_loop(
        tmp_path,
        "record",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-waiting",
        "--ref",
        "R1",
        "--kind",
        "verify",
        "--result",
        "failure",
        "--duration-seconds",
        "61",
        "--ledger-path",
        str(ledger),
    )
    active_budget_gate = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-waiting",
        "--max-minutes",
        "1",
        "--ledger-path",
        str(ledger),
        expected_exit=2,
    )
    assert "revision_active_minutes_reached:1" in active_budget_gate["reasons"]


def test_seal_is_required_and_contract_drift_pauses_execution(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [ ] 1.1 Implement parser [#R1]
  - DEPENDS_ON: none
"""
    write_contract(tmp_path, "demo", tasks, base_features([("R1", "1.1", False, False)]))

    unsealed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--repo-root", str(tmp_path), "plan", "demo"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert unsealed.returncode == 0
    unsealed_payload = json.loads(unsealed.stdout)
    assert unsealed_payload["candidate_ref"] == "R1"
    assert unsealed_payload["selected_ref"] is None

    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    seal = run_loop(
        tmp_path,
        "seal",
        "demo",
        "--confirmed",
        "--retention",
        "thin",
        "--ledger-path",
        str(ledger),
    )
    assert seal["written"] is True
    assert run_loop(tmp_path, "check", "demo")["ok"] is True

    (tmp_path / "openspec" / "changes" / "demo" / "tasks.md").write_text(
        tasks.replace("Implement parser", "Implement parser safely"),
        encoding="utf-8",
    )
    drift = run_loop(tmp_path, "check", "demo", expected_exit=2)
    assert any("fingerprint drift" in issue for issue in drift["issues"])


def test_check_rejects_incomplete_or_invalid_sealed_policy(tmp_path: Path) -> None:
    write_contract(
        tmp_path,
        "demo",
        "## Active Task Registry\n\n- [ ] 1.1 Implement policy [#R1]\n",
        base_features([("R1", "1.1", False, False)]),
    )
    ledger = tmp_path / "test_cache" / "demo" / "loop-ledger.json"
    ensure_sealed_runtime(tmp_path, ledger)
    loop_path = tmp_path / "openspec" / "changes" / "demo" / "loop.json"
    policy = json.loads(loop_path.read_text(encoding="utf-8"))
    del policy["paths"]["product"]
    policy["budgets"]["revision"]["max_iterations"] = 0
    del policy["test_profiles"]["promotion"]
    policy["confirmed_at"] = ""
    loop_path.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    checked = run_loop(tmp_path, "check", "demo", expected_exit=2)

    assert "loop.json paths missing: product" in checked["issues"]
    assert (
        "loop.json budgets.revision must be positive integers: max_iterations"
        in checked["issues"]
    )
    assert "loop.json test_profiles missing: promotion" in checked["issues"]
    assert "loop.json confirmed_at is required" in checked["issues"]


def test_plan_skips_maxed_task_and_selects_its_replacement(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [x] 1.36 Source-backed baseline [#R36]
  - DEPENDS_ON: none
- [ ] 1.37 Exhausted recovery [#R37]
  - DEPENDS_ON: R36
  - MAXED (RUN #51): repeated output deviation
- [ ] 1.38 Replacement recovery [#R38]
  - DEPENDS_ON: R36
  - SUPERSEDES: R37
"""
    features = base_features(
        [
            ("R36", "1.36", True, True),
            ("R37", "1.37", False, False),
            ("R38", "1.38", False, False),
        ]
    )
    write_contract(tmp_path, "demo", tasks, features)

    payload = run_loop(tmp_path, "plan", "demo")
    by_ref = {task["ref"]: task for task in payload["tasks"]}
    assert payload["candidate_ref"] == "R38"
    assert by_ref["R37"]["effective_state"] == "superseded"
    assert by_ref["R38"]["supersedes"] == ["R37"]


def test_repeated_semantic_deviation_trips_result_breaker(tmp_path: Path) -> None:
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    for _ in range(2):
        run_loop(
            tmp_path,
            "record",
            "demo",
            "--run-id",
            "run-deviation",
            "--ref",
            "R1",
            "--kind",
            "verify",
            "--result",
            "deviated",
            "--observation-text",
            "command exits zero but output targets the wrong year",
            "--disposition",
            "targeted_probe",
            "--ledger-path",
            str(ledger),
        )
    gate = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--run-id",
        "run-deviation",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--ledger-path",
        str(ledger),
        expected_exit=2,
    )
    assert "repeated_result_breaker" in gate["reasons"]
    assert "repeated_deviation_breaker" in gate["reasons"]


def test_new_revision_does_not_reset_change_iteration_budget(tmp_path: Path) -> None:
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    run_loop(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "revision-one",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "failure",
        "--duration-seconds",
        "5",
        "--ledger-path",
        str(ledger),
    )
    tasks_path = tmp_path / "openspec" / "changes" / "demo" / "tasks.md"
    tasks_path.write_text(
        tasks_path.read_text(encoding="utf-8").replace("Runtime task", "Runtime task v2"),
        encoding="utf-8",
    )
    run_loop(
        tmp_path,
        "seal",
        "demo",
        "--confirmed",
        "--retention",
        "thin",
        "--ledger-path",
        str(ledger),
        "--max-total-iterations",
        "1",
    )
    gate = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--run-id",
        "revision-two",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--ledger-path",
        str(ledger),
        expected_exit=2,
    )
    assert "change_max_iterations_reached:1" in gate["reasons"]
    assert gate["budgets"]["change"]["max_iterations"] == 1


def write_narrative(repo_root: Path, change_id: str, marker: str) -> None:
    change_dir = repo_root / "openspec" / "changes" / change_id
    (change_dir / "proposal.md").write_text(f"# proposal {marker}\n", encoding="utf-8")
    (change_dir / "design.md").write_text(f"# design {marker}\n", encoding="utf-8")
    specs_dir = change_dir / "specs" / "demo-capability"
    specs_dir.mkdir(parents=True, exist_ok=True)
    (specs_dir / "spec.md").write_text(f"# spec {marker}\n", encoding="utf-8")


def seal_demo(repo_root: Path, ledger: Path, *extra: str) -> dict:
    return run_loop(
        repo_root,
        "seal",
        "demo",
        "--confirmed",
        "--retention",
        "thin",
        "--ledger-path",
        str(ledger),
        *extra,
    )


def read_config(repo_root: Path, change_id: str = "demo") -> dict:
    path = repo_root / "openspec" / "changes" / change_id / "loop.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_semantic_fingerprint_survives_promotion_and_narrative_edits(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [ ] 1.1 Implement parser [#R1]
  - DEPENDS_ON: none
  - STATE: ready
"""
    write_contract(tmp_path, "demo", tasks, base_features([("R1", "1.1", False, False)]))
    write_narrative(tmp_path, "demo", "first")
    before = run_loop(tmp_path, "plan", "demo")

    promoted = tasks.replace("- [ ] 1.1", "- [x] 1.1").replace(
        "  - STATE: ready\n", "  - STATE: passed\n"
    )
    write_contract(tmp_path, "demo", promoted, base_features([("R1", "1.1", True, True)]))
    write_narrative(tmp_path, "demo", "second")
    after = run_loop(tmp_path, "plan", "demo")

    assert after["contract_fingerprint"] == before["contract_fingerprint"]
    assert after["narrative_digest"] != before["narrative_digest"]

    renamed = promoted.replace("Implement parser", "Implement parser safely")
    (tmp_path / "openspec" / "changes" / "demo" / "tasks.md").write_text(
        renamed, encoding="utf-8"
    )
    reworded = run_loop(tmp_path, "plan", "demo")
    assert reworded["contract_fingerprint"] != before["contract_fingerprint"]


def test_advisory_narrative_drift_warns_while_strict_blocks(tmp_path: Path) -> None:
    write_contract(
        tmp_path,
        "demo",
        "## Active Task Registry\n\n- [ ] 1.1 Implement parser [#R1]\n",
        base_features([("R1", "1.1", False, False)]),
    )
    write_narrative(tmp_path, "demo", "first")
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    seal_demo(tmp_path, ledger)
    assert run_loop(tmp_path, "check", "demo")["ok"] is True

    write_narrative(tmp_path, "demo", "second")
    advisory = run_loop(tmp_path, "check", "demo")
    assert advisory["ok"] is True
    assert advisory["sealed"] is True
    assert any("narrative drift" in warning for warning in advisory["warnings"])
    assert not any("narrative drift" in issue for issue in advisory["issues"])

    seal_demo(tmp_path, ledger, "--narrative-policy", "strict")
    write_narrative(tmp_path, "demo", "third")
    strict = run_loop(tmp_path, "check", "demo", expected_exit=2)
    assert strict["ok"] is False
    assert any("narrative drift" in issue for issue in strict["issues"])


def test_reseal_refreshes_narrative_and_refuses_a_semantic_change(tmp_path: Path) -> None:
    tasks = "## Active Task Registry\n\n- [ ] 1.1 Implement parser [#R1]\n"
    write_contract(tmp_path, "demo", tasks, base_features([("R1", "1.1", False, False)]))
    write_narrative(tmp_path, "demo", "first")
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    seal_demo(
        tmp_path,
        ledger,
        "--scratch-root",
        "test_cache/demo",
        "--max-revisions",
        "7",
        "--max-apply-attempts",
        "4",
    )
    before = read_config(tmp_path)

    write_narrative(tmp_path, "demo", "second")
    refreshed = run_loop(tmp_path, "reseal", "demo")
    assert refreshed["written"] is True
    assert refreshed["semantic_change"] is False
    assert refreshed["revision_charged"] is False

    after = read_config(tmp_path)
    assert after["contract_fingerprint"] == before["contract_fingerprint"]
    assert after["narrative_digest"] != before["narrative_digest"]
    for field in ("retention", "paths", "budgets", "test_profiles", "confirmed_at"):
        assert after[field] == before[field]
    assert run_loop(tmp_path, "check", "demo")["ok"] is True

    (tmp_path / "openspec" / "changes" / "demo" / "tasks.md").write_text(
        tasks.replace("Implement parser", "Implement parser safely"), encoding="utf-8"
    )
    refused = run_loop(tmp_path, "reseal", "demo", expected_exit=2)
    assert refused["written"] is False
    assert refused["semantic_change"] is True
    assert any("$openspec-change-interviewer" in issue for issue in refused["issues"])
    assert read_config(tmp_path)["contract_fingerprint"] == before["contract_fingerprint"]


def test_reseal_migrate_upgrades_a_v2_config_without_losing_policy(tmp_path: Path) -> None:
    write_contract(
        tmp_path,
        "demo",
        "## Active Task Registry\n\n- [ ] 1.1 Implement parser [#R1]\n",
        base_features([("R1", "1.1", False, False)]),
    )
    write_narrative(tmp_path, "demo", "first")
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    seal_demo(tmp_path, ledger, "--max-revisions", "7", "--max-total-iterations", "44")

    loop_path = tmp_path / "openspec" / "changes" / "demo" / "loop.json"
    legacy = read_config(tmp_path)
    legacy["schema_version"] = "openspec-loop.v2"
    legacy["contract_fingerprint"] = "legacy-byte-hash"
    legacy.pop("narrative_digest", None)
    legacy.pop("narrative_policy", None)
    loop_path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    blocked = run_loop(tmp_path, "check", "demo", expected_exit=2)
    migrate_issues = [issue for issue in blocked["issues"] if "--migrate" in issue]
    assert len(migrate_issues) == 1
    assert "openspec-loop.v2" in migrate_issues[0]

    refused = run_loop(tmp_path, "reseal", "demo", expected_exit=2)
    assert any("--migrate" in issue for issue in refused["issues"])

    migrated = run_loop(tmp_path, "reseal", "demo", "--migrate")
    assert migrated["migrated"] is True
    config = read_config(tmp_path)
    assert config["schema_version"] == "openspec-loop.v3"
    assert config["narrative_policy"] == "advisory"
    assert config["budgets"]["change"]["max_revisions"] == 7
    assert config["budgets"]["change"]["max_iterations"] == 44
    assert config["paths"] == legacy["paths"]
    assert config["test_profiles"] == legacy["test_profiles"]
    assert run_loop(tmp_path, "check", "demo")["ok"] is True


def test_seal_inherits_prior_policy_and_names_changed_fields(tmp_path: Path) -> None:
    write_contract(
        tmp_path,
        "demo",
        "## Active Task Registry\n\n- [ ] 1.1 Implement parser [#R1]\n",
        base_features([("R1", "1.1", False, False)]),
    )
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    seal_demo(
        tmp_path,
        ledger,
        "--scratch-root",
        "test_cache/demo",
        "--max-revisions",
        "7",
        "--max-apply-attempts",
        "4",
    )

    inherited = run_loop(tmp_path, "seal", "demo", "--confirmed")
    assert inherited["written"] is True
    assert inherited["inherited_from_prior_seal"] is True
    assert inherited["changed_fields"] == []

    config = read_config(tmp_path)
    assert config["retention"] == "thin"
    assert config["paths"]["ledger"] == str(ledger)
    assert config["paths"]["scratch"] == "test_cache/demo"
    assert config["budgets"]["change"]["max_revisions"] == 7
    assert config["budgets"]["task"]["max_apply_attempts"] == 4

    changed = run_loop(
        tmp_path, "seal", "demo", "--confirmed", "--max-revisions", "9"
    )
    assert changed["changed_fields"] == ["budgets.change.max_revisions"]
    assert read_config(tmp_path)["budgets"]["change"]["max_revisions"] == 9


def test_first_seal_still_requires_retention_and_ledger_path(tmp_path: Path) -> None:
    write_contract(
        tmp_path,
        "demo",
        "## Active Task Registry\n\n- [ ] 1.1 Implement parser [#R1]\n",
        base_features([("R1", "1.1", False, False)]),
    )
    refused = run_loop(tmp_path, "seal", "demo", "--confirmed", expected_exit=2)
    assert refused["written"] is False
    assert "--retention is required for a first seal" in refused["issues"]
    assert "--ledger-path is required for a first seal" in refused["issues"]


def episode(fingerprint: str, attempts: int) -> dict:
    return {
        "contract_fingerprint": fingerprint,
        "runs": [
            {
                "run_id": f"run-{fingerprint}",
                "started_at_utc": "2026-01-01T00:00:00Z",
                "attempts": [
                    {
                        "recorded_at_utc": "2026-01-01T00:00:00Z",
                        "ref": "R1",
                        "kind": "apply",
                        "result": "failure",
                        "duration_seconds": 0,
                    }
                    for _ in range(attempts)
                ],
                "allocated_subagent_ids": [],
            }
        ],
    }


def write_ledger(path: Path, episodes: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "openspec-loop-ledger.v2",
                "change_id": "demo",
                "episodes": episodes,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_zero_attempt_episodes_do_not_consume_the_change_revision_budget(
    tmp_path: Path,
) -> None:
    write_contract(
        tmp_path,
        "demo",
        "## Active Task Registry\n\n- [ ] 1.1 Implement parser [#R1]\n",
        base_features([("R1", "1.1", False, False)]),
    )
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    seal_demo(tmp_path, ledger, "--max-revisions", "1")
    write_ledger(
        ledger,
        [episode("stale-a", 1), episode("stale-b", 0), episode("stale-c", 0)],
    )

    allowed = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--run-id",
        "run-now",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--ledger-path",
        str(ledger),
    )
    assert allowed["decision"] == "continue"
    assert not any(
        reason.startswith("change_max_revisions_exceeded") for reason in allowed["reasons"]
    )

    summary = run_loop(
        tmp_path,
        "summary",
        "demo",
        "--run-id",
        "run-now",
        "--ledger-path",
        str(ledger),
    )
    assert summary["revision_count"] == 1
    assert summary["revision_count_total"] == 4

    payload = json.loads(ledger.read_text(encoding="utf-8"))
    for entry in payload["episodes"]:
        if entry["contract_fingerprint"] == "stale-b":
            entry["runs"][0]["attempts"] = episode("stale-b", 1)["runs"][0]["attempts"]
    ledger.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    exhausted = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--run-id",
        "run-now",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--ledger-path",
        str(ledger),
        expected_exit=2,
    )
    assert "change_max_revisions_exceeded:1" in exhausted["reasons"]


def seal_ceiling_demo(tmp_path: Path, *extra: str) -> Path:
    write_contract(
        tmp_path,
        "demo",
        "## Active Task Registry\n\n- [ ] 1.1 Implement parser [#R1]\n",
        base_features([("R1", "1.1", False, False)]),
    )
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    seal_demo(tmp_path, ledger, *extra)
    return ledger


def test_hard_ceiling_is_derived_at_seal_and_bounds_gate_overrides(tmp_path: Path) -> None:
    ledger = seal_ceiling_demo(tmp_path)
    config = read_config(tmp_path)
    assert config["hard_ceiling"] == {
        "max_iterations": 60,
        "max_active_minutes": 1080,
        "max_self_extensions": 3,
    }
    assert config["autonomy"] == "supervised"

    gate = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--run-id",
        "run-clamp",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--max-iterations",
        "999",
        "--max-active-minutes",
        "99999",
        "--ledger-path",
        str(ledger),
    )
    assert gate["decision"] == "continue"
    assert gate["terminal"] is False
    assert gate["budgets"]["revision"]["max_iterations"] == 60
    assert gate["budgets"]["revision"]["max_active_minutes"] == 1080
    assert gate["self_extensions_used"] == 0


def test_hard_ceiling_stops_the_gate_as_a_terminal_decision(tmp_path: Path) -> None:
    ledger = seal_ceiling_demo(
        tmp_path,
        "--max-total-iterations",
        "2",
        "--hard-ceiling-max-iterations",
        "5",
    )
    fingerprint = read_config(tmp_path)["contract_fingerprint"]
    write_ledger(ledger, [episode(fingerprint, 2)])

    extendable = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--run-id",
        "run-ceiling",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--ledger-path",
        str(ledger),
        expected_exit=2,
    )
    assert "change_max_iterations_reached:2" in extendable["reasons"]
    assert extendable["terminal"] is False

    run_loop(
        tmp_path,
        "reseal",
        "demo",
        "--set-max-total-iterations",
        "5",
        "--confirmed",
        "--reason",
        "authorized extension",
    )
    write_ledger(ledger, [episode(fingerprint, 5)])

    terminal = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--run-id",
        "run-ceiling",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--ledger-path",
        str(ledger),
        expected_exit=2,
    )
    assert "hard_ceiling_iterations_reached:5" in terminal["reasons"]
    assert terminal["terminal"] is True


def test_reseal_never_raises_hard_ceiling_and_refuses_a_budget_above_it(
    tmp_path: Path,
) -> None:
    seal_ceiling_demo(
        tmp_path,
        "--max-total-iterations",
        "20",
        "--hard-ceiling-max-iterations",
        "30",
    )

    refused = run_loop(
        tmp_path,
        "reseal",
        "demo",
        "--set-max-total-iterations",
        "40",
        "--confirmed",
        "--reason",
        "too much",
        expected_exit=2,
    )
    assert refused["written"] is False
    assert any("exceeds hard_ceiling.max_iterations 30" in issue for issue in refused["issues"])
    assert read_config(tmp_path)["budgets"]["change"]["max_iterations"] == 20

    allowed = run_loop(
        tmp_path,
        "reseal",
        "demo",
        "--set-max-total-iterations",
        "30",
        "--confirmed",
        "--reason",
        "authorized extension",
    )
    assert allowed["changed_fields"] == ["change.max_iterations"]
    config = read_config(tmp_path)
    assert config["budgets"]["change"]["max_iterations"] == 30
    assert config["hard_ceiling"]["max_iterations"] == 30


def test_reseal_refuses_a_further_extension_once_self_extensions_are_used(
    tmp_path: Path,
) -> None:
    seal_ceiling_demo(tmp_path, "--hard-ceiling-max-self-extensions", "1")

    first = run_loop(
        tmp_path,
        "reseal",
        "demo",
        "--set-max-revisions",
        "4",
        "--confirmed",
        "--reason",
        "first extension",
    )
    assert first["self_extensions_used"] == 1
    entry = read_config(tmp_path)["budget_extensions"][0]
    assert entry["fields"] == ["max_revisions"]
    assert entry["reason"] == "first extension"

    exhausted = run_loop(
        tmp_path,
        "reseal",
        "demo",
        "--set-max-revisions",
        "5",
        "--confirmed",
        "--reason",
        "second extension",
        expected_exit=2,
    )
    assert any("max_self_extensions reached: 1" in issue for issue in exhausted["issues"])
    assert read_config(tmp_path)["budgets"]["change"]["max_revisions"] == 4


def test_autonomy_supervised_requires_confirmation_but_full_auto_records_a_reason(
    tmp_path: Path,
) -> None:
    seal_ceiling_demo(tmp_path)

    unconfirmed = run_loop(
        tmp_path,
        "reseal",
        "demo",
        "--set-max-revisions",
        "4",
        "--reason",
        "self extension",
        expected_exit=2,
    )
    assert unconfirmed["autonomy"] == "supervised"
    assert any("requires --confirmed" in issue for issue in unconfirmed["issues"])

    unreasoned = run_loop(
        tmp_path,
        "reseal",
        "demo",
        "--set-max-revisions",
        "4",
        "--confirmed",
        expected_exit=2,
    )
    assert any("requires --reason" in issue for issue in unreasoned["issues"])

    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    seal_demo(tmp_path, ledger, "--autonomy", "full_auto")
    self_extended = run_loop(
        tmp_path,
        "reseal",
        "demo",
        "--set-max-revisions",
        "4",
        "--reason",
        "self extension after design gap",
    )
    assert self_extended["autonomy"] == "full_auto"
    config = read_config(tmp_path)
    assert config["budgets"]["change"]["max_revisions"] == 4
    assert config["budget_extensions"][0]["confirmed"] is False
    assert config["budget_extensions"][0]["autonomy"] == "full_auto"


def test_autonomy_full_auto_may_amend_the_contract_with_a_recorded_reason(
    tmp_path: Path,
) -> None:
    tasks = "## Active Task Registry\n\n- [ ] 1.1 Implement parser [#R1]\n"
    write_contract(tmp_path, "demo", tasks, base_features([("R1", "1.1", False, False)]))
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    seal_demo(tmp_path, ledger, "--autonomy", "full_auto")

    amended = tasks + "- [ ] 1.2 Close the design gap [#R2]\n  - DEPENDS_ON: 1.1\n"
    (tmp_path / "openspec" / "changes" / "demo" / "tasks.md").write_text(
        amended, encoding="utf-8"
    )
    write_contract(
        tmp_path,
        "demo",
        amended,
        base_features([("R1", "1.1", False, False), ("R2", "1.2", False, False)]),
    )

    unreasoned = run_loop(
        tmp_path, "reseal", "demo", "--allow-semantic-change", expected_exit=2
    )
    assert any("requires --reason" in issue for issue in unreasoned["issues"])

    accepted = run_loop(
        tmp_path,
        "reseal",
        "demo",
        "--allow-semantic-change",
        "--reason",
        "design-level gap R2",
    )
    assert accepted["semantic_change"] is True
    assert accepted["autonomy"] == "full_auto"
    assert run_loop(tmp_path, "check", "demo")["ok"] is True


def test_seal_refuses_a_change_budget_above_an_explicit_hard_ceiling(tmp_path: Path) -> None:
    write_contract(
        tmp_path,
        "demo",
        "## Active Task Registry\n\n- [ ] 1.1 Implement parser [#R1]\n",
        base_features([("R1", "1.1", False, False)]),
    )
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    refused = run_loop(
        tmp_path,
        "seal",
        "demo",
        "--confirmed",
        "--retention",
        "thin",
        "--ledger-path",
        str(ledger),
        "--max-total-iterations",
        "40",
        "--hard-ceiling-max-iterations",
        "10",
        expected_exit=2,
    )
    assert refused["written"] is False
    assert "budgets.change.max_iterations exceeds hard_ceiling.max_iterations" in refused["issues"]
    assert not (tmp_path / "openspec" / "changes" / "demo" / "loop.json").exists()


PROMOTABLE_TASKS = """## Active Task Registry

- [ ] 1.1 Implement parser [#R1]
  - DEPENDS_ON: none
  - ACCEPT: the parser rejects an unknown ref.
  - TEST: SCOPE: CLI
    - Run: `python -c "pass"`
    - Verify: exit 0
- [ ] 1.2 Implement verifier [#R2]
  - DEPENDS_ON: 1.1
  - ACCEPT: the verifier returns exactly one verdict.
  - TEST: SCOPE: CLI
    - Run: `python -c "pass"`
    - Verify: exit 0
"""


def seal_promotable_demo(tmp_path: Path) -> Path:
    write_contract(
        tmp_path,
        "demo",
        PROMOTABLE_TASKS,
        base_features([("R1", "1.1", False, False), ("R2", "1.2", False, False)]),
    )
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    seal_demo(tmp_path, ledger)
    return ledger


def record_verify_pass(tmp_path: Path, ledger: Path, ref: str) -> None:
    run_loop(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "run-promote",
        "--ref",
        ref,
        "--kind",
        "verify",
        "--result",
        "pass",
        "--duration-seconds",
        "30",
        "--ledger-path",
        str(ledger),
    )


def test_promote_cannot_leave_index_drift(tmp_path: Path) -> None:
    ledger = seal_promotable_demo(tmp_path)
    record_verify_pass(tmp_path, ledger, "R1")

    promoted = run_loop(tmp_path, "promote", "demo", "--ref", "R1")
    assert promoted["promoted"] is True
    assert promoted["ready_refs"] == ["R2"]

    tasks_text = (tmp_path / "openspec" / "changes" / "demo" / "tasks.md").read_text(
        encoding="utf-8"
    )
    assert "- [x] 1.1 Implement parser [#R1]" in tasks_text
    assert "- [ ] 1.2 Implement verifier [#R2]" in tasks_text

    features = json.loads(
        (tmp_path / "openspec" / "changes" / "demo" / "feature_list.json").read_text(
            encoding="utf-8"
        )
    )["features"]
    assert features["R1"]["passes"] is True

    checked = run_loop(tmp_path, "check", "demo")
    assert checked["ok"] is True
    assert not any("drift" in issue for issue in checked["issues"])
    assert checked["contract_fingerprint"] == promoted["contract_fingerprint"]


def test_promote_refuses_an_unverified_ref(tmp_path: Path) -> None:
    seal_promotable_demo(tmp_path)
    tasks_path = tmp_path / "openspec" / "changes" / "demo" / "tasks.md"
    before = tasks_path.read_bytes()

    refused = run_loop(tmp_path, "promote", "demo", "--ref", "R1", expected_exit=2)
    assert refused["promoted"] is False
    assert any("no recorded verifier pass" in issue for issue in refused["issues"])
    assert tasks_path.read_bytes() == before


def test_promote_refuses_a_ref_that_is_not_ready(tmp_path: Path) -> None:
    ledger = seal_promotable_demo(tmp_path)
    record_verify_pass(tmp_path, ledger, "R2")
    tasks_path = tmp_path / "openspec" / "changes" / "demo" / "tasks.md"
    before = tasks_path.read_bytes()

    refused = run_loop(tmp_path, "promote", "demo", "--ref", "R2", expected_exit=2)
    assert refused["promoted"] is False
    assert any("not `ready`" in issue for issue in refused["issues"])
    assert tasks_path.read_bytes() == before


def test_promote_reverts_the_contract_when_the_transition_cannot_complete(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    loop = load_loop_module()
    ledger = seal_promotable_demo(tmp_path)
    record_verify_pass(tmp_path, ledger, "R1")
    change_dir = tmp_path / "openspec" / "changes" / "demo"
    tasks_before = (change_dir / "tasks.md").read_bytes()
    features_before = (change_dir / "feature_list.json").read_bytes()

    def refuse(*_args, **_kwargs):
        raise ValueError("promotion would change the sealed obligations")

    monkeypatch.setattr(loop, "regenerate_feature_index", refuse)
    args = loop.build_parser().parse_args(
        ["--repo-root", str(tmp_path), "promote", "demo", "--ref", "R1"]
    )

    assert args.func(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["promoted"] is False
    assert "sealed obligations" in payload["issues"][0]
    assert (change_dir / "tasks.md").read_bytes() == tasks_before
    assert (change_dir / "feature_list.json").read_bytes() == features_before


def test_sync_repairs_index_drift_without_touching_tasks(tmp_path: Path) -> None:
    ledger = seal_promotable_demo(tmp_path)
    record_verify_pass(tmp_path, ledger, "R1")
    run_loop(tmp_path, "promote", "demo", "--ref", "R1")

    change_dir = tmp_path / "openspec" / "changes" / "demo"
    stale = json.loads((change_dir / "feature_list.json").read_text(encoding="utf-8"))
    stale["features"]["R1"]["passes"] = False
    (change_dir / "feature_list.json").write_text(
        json.dumps(stale, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    drifted = run_loop(tmp_path, "check", "demo", expected_exit=2)
    assert any("checkbox/passes drift" in issue for issue in drifted["issues"])

    tasks_before = (change_dir / "tasks.md").read_bytes()
    synced = run_loop(tmp_path, "sync", "demo")
    assert synced["ok"] is True
    assert synced["issues"] == []
    assert (change_dir / "tasks.md").read_bytes() == tasks_before
    assert synced["contract_fingerprint"] == drifted["contract_fingerprint"]
    assert run_loop(tmp_path, "check", "demo")["ok"] is True


def test_plan_batch_offers_every_independent_ready_ref(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [x] 1.1 Baseline [#R1]
- [ ] 1.2 Branch A [#R2]
  - INDEPENDENT: yes
- [ ] 1.3 Branch B [#R3]
  - INDEPENDENT: yes
"""
    features = base_features(
        [
            ("R1", "1.1", True, True),
            ("R2", "1.2", False, False),
            ("R3", "1.3", False, False),
        ]
    )
    write_contract(tmp_path, "demo", tasks, features)

    single = run_loop(tmp_path, "plan", "demo")
    assert single["selected_ref"] == "R2"
    assert single["selected_batch"] == ["R2"]

    batched = run_loop(tmp_path, "plan", "demo", "--batch")
    assert batched["selected_ref"] == "R2"
    assert batched["selected_batch"] == ["R2", "R3"]


def test_plan_batch_never_shadows_a_dependency_of_a_batched_ref(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [x] 1.1 Root [#R1]
- [x] 1.2 Middle [#R2]
  - DEPENDS_ON: 1.1
- [ ] 1.3 Leaf [#R3]
  - DEPENDS_ON: 1.2
- [ ] 1.4 Sibling [#R4]
  - INDEPENDENT: yes
"""
    features = base_features(
        [
            ("R1", "1.1", True, True),
            ("R2", "1.2", True, True),
            ("R3", "1.3", False, False),
            ("R4", "1.4", False, False),
        ]
    )
    write_contract(tmp_path, "demo", tasks, features)

    batched = run_loop(tmp_path, "plan", "demo", "--batch")
    by_ref = {task["ref"]: task for task in batched["tasks"]}

    assert batched["selected_batch"] == ["R3", "R4"]
    assert by_ref["R3"]["dependency_closure"] == ["R1", "R2"]
    for ref in batched["selected_batch"]:
        closure = set(by_ref[ref]["dependency_closure"])
        assert closure.isdisjoint(set(batched["selected_batch"]))


def test_plan_batch_is_empty_without_a_selected_ref(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [ ] 1.1 Cyclic A [#R1]
  - DEPENDS_ON: R2
- [ ] 1.2 Cyclic B [#R2]
  - DEPENDS_ON: R1
"""
    features = base_features([("R1", "1.1", False, False), ("R2", "1.2", False, False)])
    write_contract(tmp_path, "demo", tasks, features)

    batched = run_loop(tmp_path, "plan", "demo", "--batch")
    assert batched["selected_ref"] is None
    assert batched["selected_batch"] == []


GOAL_TASKS = """## Active Task Registry

- [x] 1.1 Build the surface [#R1]
- [ ] 1.2 Close the surface [#R2]
  - INDEPENDENT: yes
- [ ] 1.3 Unrelated chore [#R3]
  - INDEPENDENT: yes

## Design goals

- GOAL G1: every surface closes without --commit
  - COVERED_BY: R1-R2
  - ACCEPT: both refs report a closed surface in the retained manifest
"""
GOAL_FEATURES = base_features(
    [
        ("R1", "1.1", True, True),
        ("R2", "1.2", False, False),
        ("R3", "1.3", False, False),
    ]
)


def test_goals_reports_the_coverage_matrix_and_orphan_refs(tmp_path: Path) -> None:
    write_contract(tmp_path, "demo", GOAL_TASKS, GOAL_FEATURES)

    payload = run_loop(tmp_path, "goals", "demo")

    assert payload["ok"] is True
    assert payload["goal_count"] == 1
    goal = payload["goals"][0]
    assert goal["goal_id"] == "G1"
    assert goal["covered_by"] == ["R1", "R2"]
    assert goal["states"] == {"R1": "passed", "R2": "ready"}
    assert goal["covered"] is True
    assert goal["accept"].startswith("both refs report")
    assert payload["orphan_refs"] == ["R3"]


def test_goals_reports_an_uncovered_goal_as_a_failure(tmp_path: Path) -> None:
    tasks = GOAL_TASKS.replace("  - COVERED_BY: R1-R2\n", "  - COVERED_BY: R404\n")
    write_contract(tmp_path, "demo", tasks, GOAL_FEATURES)

    payload = run_loop(tmp_path, "goals", "demo", expected_exit=2)

    assert payload["ok"] is False
    assert payload["uncovered_goals"] == ["G1"]
    assert any("unknown COVERED_BY `R404`" in issue for issue in payload["issues"])


def test_goals_treats_a_fully_superseded_goal_as_uncovered(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [ ] 1.1 Exhausted attempt [#R1]
  - STATE: maxed
- [ ] 1.2 Replacement [#R2]
  - SUPERSEDES: R1
  - INDEPENDENT: yes

## Design goals

- GOAL G1: the exhausted approach still closes
  - COVERED_BY: R1
  - ACCEPT: the retained manifest is closed
"""
    features = base_features([("R1", "1.1", False, False), ("R2", "1.2", False, False)])
    write_contract(tmp_path, "demo", tasks, features)

    payload = run_loop(tmp_path, "goals", "demo", expected_exit=2)

    assert payload["uncovered_goals"] == ["G1"]
    assert any("every covering ref is superseded" in issue for issue in payload["issues"])


def test_goals_are_optional_but_operative(tmp_path: Path) -> None:
    write_contract(
        tmp_path,
        "demo",
        "## Active Task Registry\n\n- [ ] 1.1 Only a task [#R1]\n",
        base_features([("R1", "1.1", False, False)]),
    )
    empty = run_loop(tmp_path, "goals", "demo")
    assert empty["goal_count"] == 0
    assert empty["goals"] == []
    assert empty["orphan_refs"] == []
    assert empty["ok"] is True

    write_contract(tmp_path, "demo", GOAL_TASKS, GOAL_FEATURES)
    before = run_loop(tmp_path, "plan", "demo")["contract_fingerprint"]
    amended = GOAL_TASKS.replace(
        "  - ACCEPT: both refs report a closed surface in the retained manifest\n",
        "  - ACCEPT: both refs report a closed surface for all five years\n",
    )
    write_contract(tmp_path, "demo", amended, GOAL_FEATURES)

    assert run_loop(tmp_path, "plan", "demo")["contract_fingerprint"] != before


CLOSED_GOAL_TASKS = """## Active Task Registry

- [x] 1.1 Build the surface [#R1]
  - ACCEPT: the surface exists for every pair.
  - TEST: SCOPE: CLI
    - Run: `python -c "pass"`
    - Verify: exit 0
- [x] 1.2 Close the surface [#R2]
  - ACCEPT: the surface closes without --commit.
  - TEST: SCOPE: CLI
    - Run: `python -c "pass"`
    - Verify: exit 0

## Design goals

- GOAL G1: every surface closes without --commit
  - COVERED_BY: R1-R2
  - ACCEPT: both refs report a closed surface in the retained manifest
"""
CLOSED_GOAL_FEATURES = base_features(
    [("R1", "1.1", True, True), ("R2", "1.2", True, True)]
)


def write_observation(tmp_path: Path, status: str, observed: str) -> Path:
    path = tmp_path / "observation.json"
    path.write_text(
        json.dumps(
            {"G1": {"status": status, "observed": observed, "evidence": ["manifest.csv"]}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def seal_closed_goal_demo(tmp_path: Path) -> None:
    write_contract(tmp_path, "demo", CLOSED_GOAL_TASKS, CLOSED_GOAL_FEATURES)
    seal_demo(tmp_path, tmp_path / "test_cache" / "demo" / "loop" / "ledger.json")


def test_design_verify_passes_only_with_an_observed_match(tmp_path: Path) -> None:
    seal_closed_goal_demo(tmp_path)

    unobserved = run_loop(tmp_path, "design-verify", "demo", expected_exit=2)
    assert unobserved["verdict"] == "GAP"
    assert unobserved["goals"][0]["status"] == "unobserved"

    observation = write_observation(tmp_path, "match", "22 of 22 pairs closed")
    passed = run_loop(
        tmp_path, "design-verify", "demo", "--observation", str(observation)
    )
    assert passed["verdict"] == "PASS"
    assert passed["revision_proposals"] == []
    assert passed["blockers"] == []


def test_design_verify_gap_emits_a_revision_proposal_for_a_mismatch(
    tmp_path: Path,
) -> None:
    seal_closed_goal_demo(tmp_path)
    observation = write_observation(tmp_path, "mismatch", "14 of 22 pairs used a neighbour table")

    gap = run_loop(
        tmp_path,
        "design-verify",
        "demo",
        "--observation",
        str(observation),
        expected_exit=2,
    )

    assert gap["verdict"] == "GAP"
    assert gap["written"] == []
    proposal = gap["revision_proposals"][0]
    assert proposal["schema"] == "openspec-loop-revision-proposal.v1"
    assert proposal["goal"] == "G1"
    assert proposal["observed"] == "14 of 22 pairs used a neighbour table"
    assert proposal["evidence"] == ["manifest.csv"]
    actions = [(entry["action"], entry.get("ref")) for entry in proposal["proposed"]]
    assert ("supersede", "R1") in actions
    assert ("supersede", "R2") in actions
    assert ("add", None) in actions


def test_design_verify_revision_proposal_persists_only_under_unblock(
    tmp_path: Path,
) -> None:
    seal_closed_goal_demo(tmp_path)
    observation = write_observation(tmp_path, "mismatch", "wrong source table")

    gap = run_loop(
        tmp_path,
        "design-verify",
        "demo",
        "--observation",
        str(observation),
        "--write-proposal",
        expected_exit=2,
    )

    assert len(gap["written"]) == 1
    written = tmp_path / gap["written"][0]
    assert written.parent == tmp_path / "openspec" / "changes" / "demo" / "unblock"
    assert json.loads(written.read_text(encoding="utf-8"))["goal"] == "G1"
    for residue in ("BUNDLE", "EVIDENCE", "progress.txt", "runs.log"):
        assert not (tmp_path / "openspec" / "changes" / "demo" / residue).exists()
    assert not (tmp_path / "auto_test_openspec").exists()


def test_design_verify_refuses_while_ready_work_remains(tmp_path: Path) -> None:
    write_contract(tmp_path, "demo", GOAL_TASKS, GOAL_FEATURES)
    seal_demo(tmp_path, tmp_path / "test_cache" / "demo" / "loop" / "ledger.json")
    observation = write_observation(tmp_path, "match", "closed")

    premature = run_loop(
        tmp_path,
        "design-verify",
        "demo",
        "--observation",
        str(observation),
        expected_exit=2,
    )
    assert premature["verdict"] == "GAP"
    assert any("ready work remains" in blocker for blocker in premature["blockers"])


def write_proposal(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "proposal.json"
    path.write_text(
        json.dumps(
            {
                "schema": "openspec-loop-revision-proposal.v1",
                "change_id": "demo",
                "goal": "G1",
                "expected": "both refs report a closed surface",
                "observed": "14 of 22 pairs used a neighbour table",
                "evidence": ["manifest.csv"],
                "proposed": entries,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def seal_revisable_demo(tmp_path: Path, *extra: str) -> Path:
    write_contract(tmp_path, "demo", CLOSED_GOAL_TASKS, CLOSED_GOAL_FEATURES)
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    seal_demo(tmp_path, ledger, *extra)
    return ledger


def test_full_auto_can_apply_revision_without_an_interview(tmp_path: Path) -> None:
    seal_revisable_demo(tmp_path, "--autonomy", "full_auto")
    before = read_config(tmp_path)["contract_fingerprint"]
    proposal = write_proposal(
        tmp_path,
        [
            {"action": "supersede", "ref": "R2", "reason": "wrong source table"},
            {
                "action": "add",
                "title": "re-cover design goal G1",
                "depends_on": ["R1"],
                "accept": "all 22 pairs come from the target table",
                "test": ["python -c \"pass\""],
            },
        ],
    )

    applied = run_loop(
        tmp_path, "apply-revision", "demo", "--proposal", str(proposal)
    )

    assert applied["applied"] is True
    assert applied["autonomy"] == "full_auto"
    assert applied["added_refs"] == ["R3"]
    assert applied["superseded_refs"] == ["R2"]
    assert applied["contract_fingerprint"] != before
    assert applied["ready_refs"] == ["R3"]
    # The sandbox holds no OpenSpec project, so the validator is absent rather
    # than failing; the real-repo path is exercised by the change's final task.
    assert applied["strict_validation"] == "unavailable"

    tasks_text = (tmp_path / "openspec" / "changes" / "demo" / "tasks.md").read_text(
        encoding="utf-8"
    )
    assert "- [ ] 1.3 re-cover design goal G1 [#R3]" in tasks_text
    assert "  - SUPERSEDES: R2" in tasks_text
    assert "## Design goals" in tasks_text
    assert tasks_text.index("[#R3]") < tasks_text.index("## Design goals")
    assert run_loop(tmp_path, "check", "demo")["ok"] is True


def test_supervised_autonomy_refuses_to_apply_revision(tmp_path: Path) -> None:
    seal_revisable_demo(tmp_path)
    change_dir = tmp_path / "openspec" / "changes" / "demo"
    before = (change_dir / "tasks.md").read_bytes()
    proposal = write_proposal(
        tmp_path,
        [
            {
                "action": "add",
                "title": "cover design goal G1",
                "depends_on": [],
                "accept": "the manifest is closed",
                "test": ["python -c \"pass\""],
            }
        ],
    )

    refused = run_loop(
        tmp_path,
        "apply-revision",
        "demo",
        "--proposal",
        str(proposal),
        expected_exit=2,
    )

    assert refused["applied"] is False
    assert refused["autonomy"] == "supervised"
    assert any(
        "$openspec-change-interviewer" in issue for issue in refused["issues"]
    )
    assert (change_dir / "tasks.md").read_bytes() == before


def test_apply_revision_refuses_a_proposal_without_an_executable_test(
    tmp_path: Path,
) -> None:
    seal_revisable_demo(tmp_path, "--autonomy", "full_auto")
    change_dir = tmp_path / "openspec" / "changes" / "demo"
    before = (change_dir / "tasks.md").read_bytes()
    proposal = write_proposal(
        tmp_path,
        [
            {
                "action": "add",
                "title": "cover design goal G1",
                "depends_on": [],
                "accept": "the manifest is closed",
                "test": None,
            }
        ],
    )

    refused = run_loop(
        tmp_path,
        "apply-revision",
        "demo",
        "--proposal",
        str(proposal),
        expected_exit=2,
    )

    assert any("executable `test`" in issue for issue in refused["issues"])
    assert (change_dir / "tasks.md").read_bytes() == before


def test_apply_revision_stops_at_the_hard_ceiling(tmp_path: Path) -> None:
    ledger = seal_revisable_demo(
        tmp_path,
        "--autonomy",
        "full_auto",
        "--max-total-iterations",
        "2",
        "--hard-ceiling-max-iterations",
        "2",
    )
    fingerprint = read_config(tmp_path)["contract_fingerprint"]
    write_ledger(ledger, [episode(fingerprint, 2)])
    change_dir = tmp_path / "openspec" / "changes" / "demo"
    before = (change_dir / "tasks.md").read_bytes()
    proposal = write_proposal(
        tmp_path,
        [
            {
                "action": "add",
                "title": "cover design goal G1",
                "depends_on": [],
                "accept": "the manifest is closed",
                "test": ["python -c \"pass\""],
            }
        ],
    )

    refused = run_loop(
        tmp_path,
        "apply-revision",
        "demo",
        "--proposal",
        str(proposal),
        expected_exit=2,
    )

    assert refused["applied"] is False
    assert refused["terminal"] is True
    assert any("hard_ceiling_iterations_reached:2" in issue for issue in refused["issues"])
    assert (change_dir / "tasks.md").read_bytes() == before


def write_large_registry(tmp_path: Path, task_count: int) -> None:
    lines = ["## Active Task Registry", ""]
    entries = []
    for index in range(1, task_count + 1):
        lines.append(f"- [ ] 1.{index} Task {index} [#R{index}]")
        lines.append("  - INDEPENDENT: yes")
        entries.append((f"R{index}", f"1.{index}", False, False))
    write_contract(tmp_path, "demo", "\n".join(lines) + "\n", base_features(entries))


def test_first_seal_scales_the_change_budget_to_task_count(tmp_path: Path) -> None:
    write_large_registry(tmp_path, 50)
    seal_demo(tmp_path, tmp_path / "test_cache" / "demo" / "loop" / "ledger.json")

    config = read_config(tmp_path)
    change = config["budgets"]["change"]
    assert change["max_iterations"] == 100
    assert change["max_active_minutes"] == 500
    assert change["max_revisions"] == 3
    assert config["hard_ceiling"]["max_iterations"] > change["max_iterations"]
    assert config["hard_ceiling"]["max_active_minutes"] > change["max_active_minutes"]
    assert run_loop(tmp_path, "check", "demo")["ok"] is True


def test_first_seal_keeps_a_small_registry_on_the_flat_budget(tmp_path: Path) -> None:
    write_large_registry(tmp_path, 3)
    seal_demo(tmp_path, tmp_path / "test_cache" / "demo" / "loop" / "ledger.json")

    change = read_config(tmp_path)["budgets"]["change"]
    assert change["max_iterations"] == 20
    assert change["max_active_minutes"] == 360


def test_a_confirmed_change_budget_is_never_rescaled(tmp_path: Path) -> None:
    write_large_registry(tmp_path, 50)
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    seal_demo(tmp_path, ledger, "--max-total-iterations", "5")
    assert read_config(tmp_path)["budgets"]["change"]["max_iterations"] == 5

    run_loop(tmp_path, "seal", "demo", "--confirmed")
    assert read_config(tmp_path)["budgets"]["change"]["max_iterations"] == 5


def test_ledger_lineage_names_the_episode_an_amendment_continues(
    tmp_path: Path,
) -> None:
    tasks = PROMOTABLE_TASKS
    write_contract(
        tmp_path,
        "demo",
        tasks,
        base_features([("R1", "1.1", False, False), ("R2", "1.2", False, False)]),
    )
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    seal_demo(tmp_path, ledger, "--autonomy", "full_auto")
    first_fingerprint = read_config(tmp_path)["contract_fingerprint"]

    for _ in range(2):
        run_loop(
            tmp_path,
            "record",
            "demo",
            "--run-id",
            "run-before",
            "--ref",
            "R1",
            "--kind",
            "apply",
            "--result",
            "failure",
            "--duration-seconds",
            "90",
            "--ledger-path",
            str(ledger),
        )

    (tmp_path / "openspec" / "changes" / "demo" / "tasks.md").write_text(
        tasks.replace("the parser rejects an unknown ref.", "the parser rejects any unknown ref."),
        encoding="utf-8",
    )
    run_loop(
        tmp_path,
        "reseal",
        "demo",
        "--allow-semantic-change",
        "--reason",
        "design-level gap",
    )
    second_fingerprint = read_config(tmp_path)["contract_fingerprint"]
    assert second_fingerprint != first_fingerprint

    run_loop(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "run-after",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "success",
        "--duration-seconds",
        "30",
        "--ledger-path",
        str(ledger),
    )

    summary = run_loop(
        tmp_path, "summary", "demo", "--run-id", "run-after", "--ledger-path", str(ledger)
    )
    chain = summary["amendment_chain"]
    assert [entry["contract_fingerprint"] for entry in chain] == [
        first_fingerprint,
        second_fingerprint,
    ]
    assert chain[0]["supersedes_fingerprint"] is None
    assert chain[1]["supersedes_fingerprint"] == first_fingerprint
    assert summary["change_attempt_count"] == 3
    assert summary["change_active_seconds"] == 210
    assert summary["revision_attempt_count"] == 1
    assert summary["revision_count"] == 2


def test_ledger_lineage_tolerates_a_ledger_without_lineage(tmp_path: Path) -> None:
    write_contract(
        tmp_path,
        "demo",
        "## Active Task Registry\n\n- [ ] 1.1 Implement parser [#R1]\n",
        base_features([("R1", "1.1", False, False)]),
    )
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    seal_demo(tmp_path, ledger)
    fingerprint = read_config(tmp_path)["contract_fingerprint"]
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        json.dumps(
            {
                "schema_version": "openspec-loop-ledger.v1",
                "change_id": "demo",
                "episodes": [
                    {
                        "contract_fingerprint": "legacy-episode",
                        "started_at_utc": "2026-01-01T00:00:00Z",
                        "attempts": [
                            {
                                "ref": "R1",
                                "kind": "apply",
                                "result": "failure",
                                "duration_seconds": 45,
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_loop(
        tmp_path, "summary", "demo", "--run-id", "run-new", "--ledger-path", str(ledger)
    )
    chain = summary["amendment_chain"]
    assert chain[0]["contract_fingerprint"] == "legacy-episode"
    assert chain[0]["attempt_count"] == 1
    assert chain[1]["contract_fingerprint"] == fingerprint
    assert chain[1]["supersedes_fingerprint"] == "legacy-episode"
    assert summary["change_active_seconds"] == 45


def test_thin_policy_creates_only_the_configured_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    run_loop(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "thin-run",
        "--ref",
        "R1",
        "--kind",
        "verify",
        "--result",
        "pass",
        "--ledger-path",
        str(ledger),
    )
    config = json.loads(
        (tmp_path / "openspec" / "changes" / "demo" / "loop.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["retention"] == "thin"
    assert config["paths"]["bundle"] is None
    assert ledger.exists()
    assert not (tmp_path / "auto_test_openspec").exists()

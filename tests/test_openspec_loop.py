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


def write_thin_retention_decision(repo_root: Path, change_id: str = "demo") -> None:
    change_dir = repo_root / "openspec" / "changes" / change_id
    change_dir.mkdir(parents=True, exist_ok=True)
    (change_dir / "proposal.md").write_text(
        f"""## Artifact Retention Decision

- Audit retention: `thin`
- Retained evidence root: `auto_test_openspec/{change_id}/`
- Disposable cache root: `test_cache/{change_id}/`
- Pytest basetemp root: `test_cache/{change_id}/pytest/`
- Scratch root: `test_cache/{change_id}/tmp/`
- Product/runtime output root: `null`
- GUI/Colab evidence root: `null`

## What Changes
""",
        encoding="utf-8",
    )


def run_loop_direct(repo_root: Path, *args: str, expected_exit: int = 0) -> dict:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--repo-root", str(repo_root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == expected_exit, result.stderr or result.stdout
    return json.loads(result.stdout)


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


def test_stateful_commands_require_an_active_registry(tmp_path: Path) -> None:
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
    assert "missing contract artifacts" in payload["error"]
    assert not (tmp_path / "test_cache").exists()


def test_missing_loop_auto_initializes_thin_defaults(tmp_path: Path) -> None:
    write_contract(
        tmp_path,
        "demo",
        """## Active Task Registry

- [ ] 1.1 First ready ref [#R1]
  - INDEPENDENT: yes
- [ ] 1.2 Second ready ref [#R2]
  - INDEPENDENT: yes
""",
        base_features([("R1", "1.1", False, False), ("R2", "1.2", False, False)]),
    )
    write_thin_retention_decision(tmp_path)
    loop_path = tmp_path / "openspec" / "changes" / "demo" / "loop.json"

    planned = run_loop_direct(tmp_path, "plan", "demo", "--batch")

    assert planned["selected_batch"] == ["R1", "R2"]
    config = json.loads(loop_path.read_text(encoding="utf-8"))
    assert config["retention"] == "thin"
    assert config["contract_fingerprint"] == planned["contract_fingerprint"]
    assert config["per_ref_budgets"] == {
        "R1": {"max_apply_attempts": 2, "max_unblock_runs": 2},
        "R2": {"max_apply_attempts": 2, "max_unblock_runs": 2},
    }
    assert config["paths"] == {
        "ledger": "test_cache/demo/loop/ledger.json",
        "scratch": "test_cache/demo/tmp/",
        "product": None,
        "bundle": "auto_test_openspec/demo/",
        "gui_colab": None,
    }
    assert "sealed" not in config
    assert "confirmed_at" not in config
    assert "hard_ceiling" not in config
    assert not (tmp_path / "test_cache").exists()
    assert not (tmp_path / "auto_test_openspec").exists()


def test_first_apply_without_stamp(tmp_path: Path) -> None:
    write_contract(
        tmp_path,
        "demo",
        "## Active Task Registry\n\n- [ ] 1.1 Runtime task [#R1]\n",
        base_features([("R1", "1.1", False, False)]),
    )
    write_thin_retention_decision(tmp_path)

    gate = run_loop_direct(
        tmp_path,
        "gate",
        "demo",
        "--run-id",
        "first-apply",
        "--ref",
        "R1",
        "--kind",
        "apply",
    )

    assert gate["decision"] == "continue"
    assert gate["hard_ceiling"] is None
    assert not any(reason.startswith("hard_ceiling_") for reason in gate["reasons"])
    run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "first-apply",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "success",
    )
    config = json.loads(
        (tmp_path / "openspec" / "changes" / "demo" / "loop.json").read_text(
            encoding="utf-8"
        )
    )
    assert "sealed" not in config
    assert "confirmed_at" not in config
    assert "hard_ceiling" not in config
    assert not (tmp_path / "auto_test_openspec").exists()


def write_two_ref_runtime(tmp_path: Path) -> None:
    write_contract(
        tmp_path,
        "demo",
        """## Active Task Registry

- [ ] 1.1 Recover blocked ref [#R1]
  - INDEPENDENT: yes
- [ ] 1.2 Independent ready ref [#R2]
  - INDEPENDENT: yes
""",
        base_features([("R1", "1.1", False, False), ("R2", "1.2", False, False)]),
    )
    write_thin_retention_decision(tmp_path)


def record_direct_attempt(
    tmp_path: Path,
    kind: str,
    result: str,
    observation: str,
    *,
    disposition: str | None = None,
) -> dict:
    args = [
        "record",
        "demo",
        "--run-id",
        "ref-local",
        "--ref",
        "R1",
        "--kind",
        kind,
        "--result",
        result,
        "--observation-text",
        observation,
    ]
    if disposition is not None:
        args.extend(["--disposition", disposition])
    return run_loop_direct(tmp_path, *args)


def test_dormant_unblock_requires_blocking_evidence(tmp_path: Path) -> None:
    write_two_ref_runtime(tmp_path)

    gate = run_loop_direct(
        tmp_path,
        "gate",
        "demo",
        "--run-id",
        "ref-local",
        "--ref",
        "R1",
        "--kind",
        "unblock",
        expected_exit=2,
    )

    assert gate["decision"] == "stop"
    assert gate["reasons"] == ["task_unblock_dormant:R1"]
    planned = run_loop_direct(tmp_path, "plan", "demo", "--batch")
    assert planned["selected_batch"] == ["R1", "R2"]


def test_ref_local_unblock_keeps_independent_ready_work(tmp_path: Path) -> None:
    write_two_ref_runtime(tmp_path)
    record_direct_attempt(tmp_path, "apply", "failure", "apply one")
    record_direct_attempt(tmp_path, "verify", "blocked", "blocking evidence one")
    record_direct_attempt(
        tmp_path,
        "unblock",
        "success",
        "repair direction one",
        disposition="retry",
    )

    planned = run_loop_direct(tmp_path, "plan", "demo", "--batch")
    by_ref = {task["ref"]: task for task in planned["tasks"]}

    assert planned["selected_batch"] == ["R1", "R2"]
    assert by_ref["R1"]["unblock_active"] is True
    assert by_ref["R1"]["unblock_runs_used"] == 1
    assert by_ref["R2"]["ready"] is True


def test_ref_local_stop_budget_survives_legacy_stamp(tmp_path: Path) -> None:
    write_two_ref_runtime(tmp_path)
    record_direct_attempt(tmp_path, "apply", "failure", "apply one")
    record_direct_attempt(tmp_path, "verify", "blocked", "blocking evidence one")
    record_direct_attempt(
        tmp_path,
        "unblock",
        "success",
        "repair direction one",
        disposition="retry",
    )
    record_direct_attempt(tmp_path, "apply", "failure", "apply two")
    record_direct_attempt(tmp_path, "verify", "blocked", "blocking evidence two")
    record_direct_attempt(
        tmp_path,
        "unblock",
        "success",
        "terminal budget decision",
        disposition="stop_budget",
    )

    before_stamp = run_loop_direct(tmp_path, "plan", "demo", "--batch")
    before_by_ref = {task["ref"]: task for task in before_stamp["tasks"]}
    assert before_stamp["selected_batch"] == ["R2"]
    assert before_by_ref["R1"]["effective_state"] == "maxed"
    assert before_by_ref["R1"]["budget_disposition"] == "stop_budget"
    assert before_by_ref["R2"]["ready"] is True

    run_loop_direct(tmp_path, "seal", "demo", "--confirmed")
    after_stamp = run_loop_direct(tmp_path, "plan", "demo", "--batch")
    after_by_ref = {task["ref"]: task for task in after_stamp["tasks"]}
    assert after_stamp["selected_batch"] == ["R2"]
    assert after_by_ref["R1"]["effective_state"] == "maxed"
    assert after_by_ref["R1"]["max_unblock_runs"] == 2


def test_fingerprint_latch_preserves_census_during_registry_drift(
    tmp_path: Path,
) -> None:
    write_two_ref_runtime(tmp_path)
    initial = run_loop_direct(tmp_path, "plan", "demo", "--batch")
    assert initial["selected_wave"] == ["R1", "R2"]

    tasks_path = tmp_path / "openspec" / "changes" / "demo" / "tasks.md"
    tasks_path.write_text(
        tasks_path.read_text(encoding="utf-8").replace(
            "Independent ready ref", "Semantically changed ready ref"
        ),
        encoding="utf-8",
    )

    drifted = run_loop_direct(tmp_path, "plan", "demo", "--batch")
    checked = run_loop_direct(tmp_path, "check", "demo", expected_exit=2)
    assert drifted["fingerprint_ready"] is False
    assert drifted["selected_batch"] == ["R1", "R2"]
    assert drifted["selected_wave"] == []
    assert checked["selected_batch"] == ["R1", "R2"]
    assert checked["selected_wave"] == []
    assert any("fingerprint drift" in issue for issue in checked["issues"])


def test_irreversible_policy_gate_keeps_ready_census_visible(tmp_path: Path) -> None:
    write_two_ref_runtime(tmp_path)
    run_loop_direct(tmp_path, "plan", "demo", "--batch")
    loop_path = tmp_path / "openspec" / "changes" / "demo" / "loop.json"
    config = json.loads(loop_path.read_text(encoding="utf-8"))
    config["pending_irreversible_policy"] = ["retention"]
    loop_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    planned = run_loop_direct(tmp_path, "plan", "demo", "--batch")
    checked = run_loop_direct(tmp_path, "check", "demo", expected_exit=2)
    blocked_gate = run_loop_direct(
        tmp_path,
        "gate",
        "demo",
        "--ref",
        "R1",
        "--kind",
        "apply",
        expected_exit=2,
    )

    assert planned["fingerprint_ready"] is True
    assert planned["selected_batch"] == ["R1", "R2"]
    assert planned["selected_wave"] == []
    assert planned["irreversible_policy_pending"] == ["retention"]
    assert checked["selected_batch"] == ["R1", "R2"]
    assert "unconfirmed irreversible policy" in blocked_gate["error"]


def test_legacy_stamp_compatibility_fields_are_diagnostic(tmp_path: Path) -> None:
    write_two_ref_runtime(tmp_path)
    run_loop_direct(
        tmp_path,
        "seal",
        "demo",
        "--confirmed",
        "--retention",
        "thin",
        "--ledger-path",
        "test_cache/demo/loop/ledger.json",
    )
    loop_path = tmp_path / "openspec" / "changes" / "demo" / "loop.json"
    legacy = json.loads(loop_path.read_text(encoding="utf-8"))
    assert legacy["sealed"] is True
    assert legacy["confirmed_at"]
    assert run_loop_direct(tmp_path, "check", "demo")["ok"] is True

    legacy.pop("sealed")
    legacy.pop("confirmed_at")
    loop_path.write_text(
        json.dumps(legacy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    unstamped = run_loop_direct(tmp_path, "plan", "demo", "--batch")
    assert unstamped["fingerprint_ready"] is True
    assert unstamped["selected_batch"] == ["R1", "R2"]
    assert unstamped["selected_wave"] == ["R1", "R2"]
    assert run_loop_direct(tmp_path, "check", "demo")["ok"] is True


def test_registry_drift_latch_allows_narrative_reseal_path(tmp_path: Path) -> None:
    write_two_ref_runtime(tmp_path)
    run_loop_direct(tmp_path, "plan", "demo", "--batch")
    change_dir = tmp_path / "openspec" / "changes" / "demo"
    (change_dir / "design.md").write_text("## Updated narrative only\n", encoding="utf-8")

    planned = run_loop_direct(tmp_path, "plan", "demo", "--batch")
    gate = run_loop_direct(
        tmp_path,
        "gate",
        "demo",
        "--ref",
        "R1",
        "--kind",
        "apply",
    )

    assert planned["fingerprint_ready"] is True
    assert planned["selected_wave"] == ["R1", "R2"]
    assert any("narrative drift" in warning for warning in planned["warnings"])
    assert gate["decision"] == "continue"
    assert any("narrative drift" in warning for warning in gate["warnings"])


def test_registry_drift_keeps_dependency_and_diagnostics_ref_local(
    tmp_path: Path,
) -> None:
    write_contract(
        tmp_path,
        "demo",
        """## Active Task Registry

- [ ] 1.1 Broken local dependency [#R1]
  - DEPENDS_ON: R404
- [ ] 1.2 Independent ready ref [#R2]
  - INDEPENDENT: yes
""",
        base_features([("R1", "1.1", False, False), ("R2", "1.2", False, False)]),
    )
    write_thin_retention_decision(tmp_path)
    initial = run_loop_direct(tmp_path, "plan", "demo", "--batch")
    assert initial["selected_batch"] == ["R2"]
    assert initial["selected_wave"] == ["R2"]
    assert any("unknown dependency `R404`" in issue for issue in initial["issues"])

    loop_path = tmp_path / "openspec" / "changes" / "demo" / "loop.json"
    config = json.loads(loop_path.read_text(encoding="utf-8"))
    del config["paths"]["product"]
    loop_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    diagnostic = run_loop_direct(tmp_path, "plan", "demo", "--batch")
    assert diagnostic["sealed"] is False
    assert diagnostic["fingerprint_ready"] is True
    assert diagnostic["wave_ready"] is True
    assert diagnostic["selected_batch"] == ["R2"]
    assert diagnostic["selected_wave"] == ["R2"]


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


def test_summary_v3_separates_apply_iterations_from_all_attempts(tmp_path: Path) -> None:
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    write_contract(
        tmp_path,
        "demo",
        "## Active Task Registry\n\n- [ ] 0.1 Runtime task [#R0]\n",
        base_features([("R0", "0.1", False, False)]),
    )
    fingerprint = seal_demo(
        tmp_path,
        ledger,
        "--max-total-iterations",
        "24",
    )["contract_fingerprint"]

    for index in range(4):
        for kind, result in (("apply", "success"), ("verify", "pass")):
            run_loop(
                tmp_path,
                "record",
                "demo",
                "--contract-fingerprint",
                fingerprint,
                "--run-id",
                "run-budget-usage",
                "--ref",
                f"R{index}",
                "--kind",
                kind,
                "--result",
                result,
                "--ledger-path",
                str(ledger),
            )

    summary = run_loop(
        tmp_path,
        "summary",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-budget-usage",
        "--ledger-path",
        str(ledger),
    )

    assert summary["schema_version"] == "openspec-loop-summary.v3"
    assert summary["revision_attempt_count"] == 8
    assert summary["revision_apply_iterations_used"] == 4
    assert summary["revision_apply_iterations_remaining"] == 4
    assert summary["change_apply_iterations_used"] == 4
    assert summary["change_apply_iterations_remaining"] == 20


def test_two_unblocks_require_new_evidence_and_make_the_default_second_terminal(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    fingerprint = ensure_sealed_runtime(tmp_path, ledger)
    assert read_config(tmp_path)["budgets"]["task"] == {
        "max_apply_attempts": 2,
        "max_unblock_runs": 2,
    }

    def record(
        kind: str,
        result: str,
        observation: str,
        *,
        disposition: str | None = None,
        expected_exit: int = 0,
    ) -> dict:
        args = [
            "record",
            "demo",
            "--contract-fingerprint",
            fingerprint,
            "--run-id",
            "run-two-unblocks",
            "--ref",
            "R0",
            "--kind",
            kind,
            "--result",
            result,
            "--observation-text",
            observation,
            "--ledger-path",
            str(ledger),
        ]
        if disposition is not None:
            args.extend(["--disposition", disposition])
        return run_loop(tmp_path, *args, expected_exit=expected_exit)

    record("apply", "failure", "apply attempt one")
    record("verify", "blocked", "missing evidence one")
    first_gate = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-two-unblocks",
        "--ref",
        "R0",
        "--kind",
        "unblock",
        "--ledger-path",
        str(ledger),
    )
    assert first_gate["decision"] == "continue"
    record("unblock", "success", "repair direction one", disposition="retry")

    record("apply", "failure", "apply attempt two")
    record("verify", "blocked", "missing evidence two")
    second_gate = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-two-unblocks",
        "--ref",
        "R0",
        "--kind",
        "unblock",
        "--ledger-path",
        str(ledger),
    )
    assert second_gate["decision"] == "continue"

    refused_retry = record(
        "unblock",
        "success",
        "terminal diagnosis",
        disposition="retry",
        expected_exit=2,
    )
    assert "second unblock is terminal" in refused_retry["error"]
    record(
        "unblock",
        "success",
        "terminal diagnosis",
        disposition="amend_spec",
    )

    third_gate = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "run-two-unblocks",
        "--ref",
        "R0",
        "--kind",
        "unblock",
        "--ledger-path",
        str(ledger),
        expected_exit=2,
    )
    assert "task_unblock_budget_exhausted:R0:2" in third_gate["reasons"]


def test_second_unblock_rejects_repeated_blocking_evidence(tmp_path: Path) -> None:
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    fingerprint = ensure_sealed_runtime(tmp_path, ledger)
    for index in range(2):
        run_loop(
            tmp_path,
            "record",
            "demo",
            "--contract-fingerprint",
            fingerprint,
            "--run-id",
            "run-repeated-unblock",
            "--ref",
            "R0",
            "--kind",
            "apply",
            "--result",
            "failure",
            "--observation-text",
            f"apply {index}",
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
            "run-repeated-unblock",
            "--ref",
            "R0",
            "--kind",
            "verify",
            "--result",
            "blocked",
            "--observation-text",
            "same blocking evidence",
            "--ledger-path",
            str(ledger),
        )
        if index == 0:
            run_loop(
                tmp_path,
                "record",
                "demo",
                "--contract-fingerprint",
                fingerprint,
                "--run-id",
                "run-repeated-unblock",
                "--ref",
                "R0",
                "--kind",
                "unblock",
                "--result",
                "success",
                "--observation-text",
                "first repair",
                "--disposition",
                "retry",
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
        "run-repeated-unblock",
        "--ref",
        "R0",
        "--kind",
        "unblock",
        "--ledger-path",
        str(ledger),
        expected_exit=2,
    )
    assert "second_unblock_requires_new_evidence:R0" in gate["reasons"]


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


def test_headcount_fields_are_diagnostic_and_do_not_reduce_wave_or_allowance(
    tmp_path: Path,
) -> None:
    tasks = """## Active Task Registry

- [ ] 1.1 Ref one [#R1]
  - INDEPENDENT: yes
  - FILES: `src/a.py`
  - WRITE_SCOPE: `src/a.py`
- [ ] 1.2 Ref two [#R2]
  - INDEPENDENT: yes
  - FILES: `tests/test_a.py`
  - WRITE_SCOPE: `tests/test_a.py`
- [ ] 1.3 Ref three [#R3]
  - INDEPENDENT: yes
  - FILES: `docs/a.md`
  - WRITE_SCOPE: `docs/a.md`
"""
    write_contract(
        tmp_path,
        "demo",
        tasks,
        base_features([("R1", "1.1", False, False), ("R2", "1.2", False, False), ("R3", "1.3", False, False)]),
    )
    write_thin_retention_decision(tmp_path)
    plan = run_loop_direct(tmp_path, "plan", "demo")
    assert plan["selected_wave"] == ["R1", "R2", "R3"]
    assert plan["allowed_parallel_applies"] == 3

    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    gate = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--run-id",
        "run-c",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--max-subagents",
        "1",
        "--subagent-id",
        "agent-a",
        "--subagent-id",
        "agent-b",
        "--ledger-path",
        str(ledger),
    )
    assert gate["decision"] == "continue"
    assert not any("max_subagents" in reason for reason in gate["reasons"])


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


def test_fingerprint_initialization_and_contract_drift_pauses_execution(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [ ] 1.1 Implement parser [#R1]
  - DEPENDS_ON: none
    """
    write_contract(tmp_path, "demo", tasks, base_features([("R1", "1.1", False, False)]))
    write_thin_retention_decision(tmp_path)

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
    assert unsealed_payload["selected_ref"] == "R1"
    assert unsealed_payload["selected_batch"] == ["R1"]
    assert unsealed_payload["selected_wave"] == ["R1"]
    assert run_loop_direct(tmp_path, "check", "demo")["ok"] is True

    (tmp_path / "openspec" / "changes" / "demo" / "tasks.md").write_text(
        tasks.replace("Implement parser", "Implement parser safely"),
        encoding="utf-8",
    )
    drift = run_loop(tmp_path, "check", "demo", expected_exit=2)
    assert drift["selected_batch"] == ["R1"]
    assert drift["selected_wave"] == []
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
    assert not any("confirmed_at" in issue for issue in checked["issues"])


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


def test_narrative_drift_warns_under_advisory_and_legacy_strict(tmp_path: Path) -> None:
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
    strict = run_loop(tmp_path, "check", "demo")
    assert strict["ok"] is True
    assert strict["sealed"] is True
    assert strict["selected_wave"] == ["R1"]
    assert any("narrative drift" in warning for warning in strict["warnings"])
    assert not any("narrative drift" in issue for issue in strict["issues"])


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


def test_hard_ceiling_is_derived_at_seal_but_does_not_bound_gate_overrides(tmp_path: Path) -> None:
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
    assert gate["budgets"]["revision"]["max_iterations"] == 999
    assert gate["budgets"]["revision"]["max_active_minutes"] == 99999
    assert gate["hard_ceiling"] == config["hard_ceiling"]
    assert gate["self_extensions_used"] == 0


def test_hard_ceiling_remains_diagnostic_after_change_budget_stop(tmp_path: Path) -> None:
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
    assert "change_max_iterations_reached:5" in terminal["reasons"]
    assert not any(reason.startswith("hard_ceiling_") for reason in terminal["reasons"])
    assert terminal["terminal"] is False


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
    assert single["selected_batch"] == ["R2", "R3"]

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


def test_selected_batch_census_keeps_all_ready_refs_visible_without_batch_flag(
    tmp_path: Path,
) -> None:
    tasks = """## Active Task Registry

- [ ] 1.1 First ref [#R1]
  - INDEPENDENT: yes
- [ ] 1.2 Second ref [#R2]
  - INDEPENDENT: yes
- [ ] 1.3 Broken dep [#R3]
  - DEPENDS_ON: R404
"""
    write_contract(
        tmp_path,
        "demo",
        tasks,
        base_features(
            [("R1", "1.1", False, False), ("R2", "1.2", False, False), ("R3", "1.3", False, False)]
        ),
    )
    write_thin_retention_decision(tmp_path)

    payload = run_loop_direct(tmp_path, "plan", "demo")

    assert payload["selected_ref"] == "R1"
    assert payload["selected_batch"] == ["R1", "R2"]
    assert payload["selected_wave"] == ["R1", "R2"]


def test_selected_wave_write_scope_partitions_overlaps_without_truncating_three_way_wave(
    tmp_path: Path,
) -> None:
    tasks = """## Active Task Registry

- [ ] 1.1 Implement core [#R1]
  - INDEPENDENT: yes
  - FILES: `src/a.py`
  - WRITE_SCOPE: `src/a.py`
- [ ] 1.2 Add tests [#R2]
  - INDEPENDENT: yes
  - FILES: `tests/test_a.py`
  - WRITE_SCOPE: `tests/test_a.py`
- [ ] 1.3 Update docs [#R3]
  - INDEPENDENT: yes
  - FILES: `docs/a.md`
  - WRITE_SCOPE: `docs/a.md`
- [ ] 1.4 Overlap docs [#R4]
  - INDEPENDENT: yes
  - FILES: `docs/a.md`
  - WRITE_SCOPE: `docs/a.md`
"""
    write_contract(
        tmp_path,
        "demo",
        tasks,
        base_features(
            [
                ("R1", "1.1", False, False),
                ("R2", "1.2", False, False),
                ("R3", "1.3", False, False),
                ("R4", "1.4", False, False),
            ]
        ),
    )
    write_thin_retention_decision(tmp_path)

    payload = run_loop_direct(tmp_path, "plan", "demo")
    by_ref = {item["ref"]: item for item in payload["tasks"]}

    assert payload["selected_batch"] == ["R1", "R2", "R3", "R4"]
    assert payload["selected_wave"] == ["R1", "R2", "R3"]
    assert payload["allowed_parallel_applies"] == 3
    assert by_ref["R4"]["wave_eligible"] is False
    assert "write_scope_overlap" in by_ref["R4"]["wave_blockers"]


def test_dependency_contract_errors_stay_local_to_bad_refs(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [ ] 1.1 Good ref [#R1]
  - INDEPENDENT: yes
- [ ] 1.2 Unknown dep [#R2]
  - DEPENDS_ON: R404
- [ ] 1.3 Good sibling [#R3]
  - INDEPENDENT: yes
"""
    write_contract(
        tmp_path,
        "demo",
        tasks,
        base_features(
            [("R1", "1.1", False, False), ("R2", "1.2", False, False), ("R3", "1.3", False, False)]
        ),
    )
    write_thin_retention_decision(tmp_path)

    payload = run_loop_direct(tmp_path, "plan", "demo")
    by_ref = {item["ref"]: item for item in payload["tasks"]}

    assert payload["selected_batch"] == ["R1", "R3"]
    assert payload["selected_wave"] == ["R1", "R3"]
    assert by_ref["R2"]["ready"] is False
    assert "unknown dependency `R404`" in "\n".join(by_ref["R2"]["issues"])


def write_apply_wave_contract(tmp_path: Path, count: int = 3) -> None:
    lines = ["## Active Task Registry", ""]
    features = []
    for index in range(1, count + 1):
        lines.extend(
            [
                f"- [ ] 1.{index} Apply ref {index} [#R{index}]",
                "  - INDEPENDENT: yes",
                f"  - FILES: `src/ref{index}.py`",
                f"  - WRITE_SCOPE: `src/ref{index}.py`",
            ]
        )
        features.append((f"R{index}", f"1.{index}", False, False))
    write_contract(tmp_path, "demo", "\n".join(lines) + "\n", base_features(features))
    write_thin_retention_decision(tmp_path)


def test_allowed_parallel_applies_limits_dispatch_without_truncating_wave(
    tmp_path: Path,
) -> None:
    write_apply_wave_contract(tmp_path)
    initial = run_loop_direct(tmp_path, "plan", "demo")
    assert initial["selected_wave"] == ["R1", "R2", "R3"]

    loop_path = tmp_path / "openspec" / "changes" / "demo" / "loop.json"
    config = json.loads(loop_path.read_text(encoding="utf-8"))
    config["budgets"]["revision"]["max_iterations"] = 1
    loop_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    limited = run_loop_direct(tmp_path, "plan", "demo")
    assert limited["selected_wave"] == ["R1", "R2", "R3"]
    assert limited["allowed_parallel_applies"] == 1
    assert limited["dispatch_refs"] == ["R1"]

    run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "completed",
        "--attempt-id",
        "attempt-r1",
        "--role-id",
        "implementer",
        "--changed-file",
        "src/ref1.py",
        "--evidence",
        "inspect:R1",
    )
    exhausted = run_loop_direct(tmp_path, "plan", "demo")
    assert exhausted["apply_remaining"] == 0
    assert exhausted["selected_wave"] == ["R1", "R2", "R3"]
    assert exhausted["allowed_parallel_applies"] == 0
    assert exhausted["dispatch_refs"] == []


def test_apply_identity_is_one_supervisor_record_per_ref_attempt(
    tmp_path: Path,
) -> None:
    write_apply_wave_contract(tmp_path, count=1)
    run_loop_direct(tmp_path, "plan", "demo")
    common = (
        "record",
        "demo",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "completed",
        "--attempt-id",
        "attempt-one",
        "--role-id",
        "implementer",
        "--changed-file",
        "src/ref1.py",
        "--evidence",
        "inspect:R1",
    )
    recorded = run_loop_direct(tmp_path, *common)
    assert recorded["attempt_id"] == "attempt-one"
    assert recorded["apply_record_owner"] == "supervisor"
    duplicate = run_loop_direct(tmp_path, *common, expected_exit=2)
    assert "duplicate canonical Apply attempt_id" in duplicate["error"]

    worker = run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "failed",
        "--attempt-id",
        "attempt-two",
        "--record-owner",
        "worker",
        expected_exit=2,
    )
    assert "only the supervisor" in worker["error"]
    combined = run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--ref",
        "R1,R2",
        "--kind",
        "apply",
        "--result",
        "failed",
        "--attempt-id",
        "attempt-three",
        expected_exit=2,
    )
    assert "exactly one ref" in combined["error"]
    role_drift = run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "failed",
        "--attempt-id",
        "attempt-four",
        "--role-id",
        "test-engineer",
        expected_exit=2,
    )
    assert "Role ID does not match" in role_drift["error"]


def test_role_write_allowlist_stays_within_packet_scope_and_evidence_root(
    tmp_path: Path,
) -> None:
    tasks = """## Active Task Registry

- [ ] 1.1 Implementation [#R1]
  - INDEPENDENT: yes
  - ROLE_ID: implementer
  - FILES: `src/pkg/**`
  - WRITE_SCOPE: `src/pkg/**`
- [ ] 1.2 Tests [#R2]
  - INDEPENDENT: yes
  - ROLE_ID: test-engineer
  - FILES: `tests/**`
  - WRITE_SCOPE: `tests/**`, `src/test_support/**`
- [ ] 1.3 Browser evidence [#R3]
  - INDEPENDENT: yes
  - ROLE_ID: browser-qa-runner
  - FILES: `auto_test_openspec/demo/**`
  - WRITE_SCOPE: `auto_test_openspec/demo/**`
- [ ] 1.4 Review [#R4]
  - INDEPENDENT: yes
  - ROLE_ID: code-reviewer
  - FILES: `docs/**`
  - WRITE_SCOPE: `docs/**`
"""
    write_contract(
        tmp_path,
        "demo",
        tasks,
        base_features([(f"R{i}", f"1.{i}", False, False) for i in range(1, 5)]),
    )
    write_thin_retention_decision(tmp_path)
    run_loop_direct(tmp_path, "plan", "demo")

    def record(ref: str, attempt: str, role: str, changed: str, exit_code: int = 0) -> dict:
        return run_loop_direct(
            tmp_path,
            "record",
            "demo",
            "--ref",
            ref,
            "--kind",
            "apply",
            "--result",
            "completed",
            "--attempt-id",
            attempt,
            "--role-id",
            role,
            "--changed-file",
            changed,
            "--evidence",
            f"inspect:{ref}",
            expected_exit=exit_code,
        )

    assert record("R1", "impl-ok", "implementer", "src/pkg/core.py")["attempt_id"] == "impl-ok"
    assert "write_scope_violation" in record(
        "R1", "impl-outside", "implementer", "src/other.py", 2
    )["error"]
    assert record("R2", "tests-ok", "test-engineer", "tests/test_core.py")["attempt_id"] == "tests-ok"
    assert "tests_only" in record(
        "R2", "tests-source", "test-engineer", "src/test_support/data.txt", 2
    )["error"]
    assert record(
        "R3", "evidence-ok", "browser-qa-runner", "auto_test_openspec/demo/browser.json"
    )["attempt_id"] == "evidence-ok"
    assert "evidence_root" in record(
        "R3", "evidence-outside", "browser-qa-runner", "outputs/browser.json", 2
    )["error"]
    assert "read-only" in record(
        "R4", "review-write", "code-reviewer", "docs/review.md", 2
    )["error"]


def test_shared_wave_join_blocks_early_verify_but_failed_sibling_is_local(
    tmp_path: Path,
) -> None:
    write_apply_wave_contract(tmp_path, count=2)
    plan = run_loop_direct(tmp_path, "plan", "demo")
    join_id = plan["routing"][0]["join_id"]

    narrowed = run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "wave-run",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "completed",
        "--attempt-id",
        "narrowed-attempt",
        "--role-id",
        "implementer",
        "--evidence",
        "inspect:R1",
        "--join-id",
        join_id,
        "--wave-ref",
        "R1",
        expected_exit=2,
    )
    assert "wave_refs must equal" in narrowed["error"]

    def apply(ref: str, result: str) -> dict:
        return run_loop_direct(
            tmp_path,
            "record",
            "demo",
            "--run-id",
            "wave-run",
            "--ref",
            ref,
            "--kind",
            "apply",
            "--result",
            result,
            "--attempt-id",
            f"attempt-{ref.lower()}",
            "--role-id",
            "implementer",
            "--changed-file",
            f"src/ref{ref[1:]}.py",
            "--evidence",
            f"inspect:{ref}",
            "--join-id",
            join_id,
            "--wave-ref",
            "R1",
            "--wave-ref",
            "R2",
        )

    apply("R1", "completed")
    early = run_loop_direct(
        tmp_path,
        "gate",
        "demo",
        "--run-id",
        "wave-run",
        "--ref",
        "R1",
        "--kind",
        "verify",
        expected_exit=2,
    )
    assert any("wave_join_pending" in reason for reason in early["reasons"])

    apply("R2", "failed")
    own_ok = run_loop_direct(
        tmp_path,
        "gate",
        "demo",
        "--run-id",
        "wave-run",
        "--ref",
        "R1",
        "--kind",
        "verify",
    )
    assert own_ok["decision"] == "continue"
    sibling_failed = run_loop_direct(
        tmp_path,
        "gate",
        "demo",
        "--run-id",
        "wave-run",
        "--ref",
        "R2",
        "--kind",
        "verify",
        expected_exit=2,
    )
    assert any("verify_requires_completed_apply:R2:failed" in reason for reason in sibling_failed["reasons"])


def test_retry_ownership_keeps_transient_retries_inside_one_apply_attempt(
    tmp_path: Path,
) -> None:
    write_apply_wave_contract(tmp_path, count=1)
    run_loop_direct(tmp_path, "plan", "demo")
    run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "retry-run",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "failed",
        "--attempt-id",
        "attempt-retry",
        "--role-id",
        "implementer",
        "--transient-retries",
        "2",
    )
    summary = run_loop_direct(tmp_path, "summary", "demo", "--run-id", "retry-run")
    assert summary["revision_apply_iterations_used"] == 1
    latest = summary["latest_attempt"]
    assert latest["transient_retries"] == 2
    assert latest["auto_redispatch"] is False
    assert latest["next_apply_owner"] == "supervisor_gate"
    next_gate = run_loop_direct(
        tmp_path,
        "gate",
        "demo",
        "--run-id",
        "retry-run",
        "--ref",
        "R1",
        "--kind",
        "apply",
    )
    assert next_gate["decision"] == "continue"


def test_zero_apply_actions_use_no_apply_budget_or_scheduling_headcount(
    tmp_path: Path,
) -> None:
    tasks = """## Active Task Registry

- [ ] 1.1 Worker Apply [#R1]
  - INDEPENDENT: yes
  - FILES: `src/ref1.py`
  - WRITE_SCOPE: `src/ref1.py`
- [ ] 1.2 Supervisor-direct Apply [#R2]
  - INDEPENDENT: yes
"""
    write_contract(
        tmp_path,
        "demo",
        tasks,
        base_features([("R1", "1.1", False, False), ("R2", "1.2", False, False)]),
    )
    write_thin_retention_decision(tmp_path)
    run_loop_direct(tmp_path, "plan", "demo")
    for kind, result in (
        ("goal", "success"),
        ("stop_hook", "success"),
        ("review", "success"),
        ("verify", "pass"),
    ):
        run_loop_direct(
            tmp_path,
            "record",
            "demo",
            "--run-id",
            "zero-run",
            "--ref",
            "R1",
            "--kind",
            kind,
            "--result",
            result,
            "--subagent-id",
            f"trace-{kind}",
        )
    summary = run_loop_direct(tmp_path, "summary", "demo", "--run-id", "zero-run")
    assert summary["revision_apply_iterations_used"] == 0
    assert all(
        item["consumes_apply_attempt"] is False
        and item["consumes_scheduling_headcount"] is False
        and item["allocated_subagent_ids"] == []
        for item in summary["attempts"]
    )
    gate = run_loop_direct(
        tmp_path,
        "gate",
        "demo",
        "--run-id",
        "zero-run",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--max-subagents",
        "1",
        "--current-allocated-subagents",
        "99",
    )
    assert gate["decision"] == "continue"

    run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "direct-run",
        "--ref",
        "R2",
        "--kind",
        "apply",
        "--result",
        "completed",
        "--attempt-id",
        "direct-attempt",
        "--evidence",
        "inspect:R2",
    )
    direct = run_loop_direct(tmp_path, "summary", "demo", "--run-id", "direct-run")
    assert direct["revision_apply_iterations_used"] == 1
    assert direct["latest_attempt"]["role_id"] == "rose"
    assert direct["latest_attempt"]["consumes_scheduling_headcount"] is False


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


def test_apply_revision_reports_change_budget_without_global_hard_ceiling_terminal(tmp_path: Path) -> None:
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
    assert refused["terminal"] is False
    assert any("change_max_iterations_reached:2" in issue for issue in refused["issues"])
    assert not any("hard_ceiling_iterations_reached" in issue for issue in refused["issues"])
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


def test_semantic_seal_reports_task_derived_budget_shortfall_without_rescaling(
    tmp_path: Path,
) -> None:
    write_large_registry(tmp_path, 12)
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    seal_demo(tmp_path, ledger, "--max-total-iterations", "24")

    write_large_registry(tmp_path, 19)
    restamped = run_loop(tmp_path, "seal", "demo", "--confirmed")

    assert restamped["semantic_change"] is True
    assert restamped["budget_advisories"] == [
        {
            "field": "budgets.change.max_iterations",
            "configured": 24,
            "recommended": 38,
            "shortfall": 14,
        }
    ]
    assert read_config(tmp_path)["budgets"]["change"]["max_iterations"] == 24


def test_semantic_reseal_reports_the_same_non_mutating_budget_advisory(
    tmp_path: Path,
) -> None:
    write_large_registry(tmp_path, 12)
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    seal_demo(tmp_path, ledger, "--max-total-iterations", "24")

    write_large_registry(tmp_path, 19)
    restamped = run_loop(
        tmp_path,
        "reseal",
        "demo",
        "--allow-semantic-change",
        "--confirmed",
    )

    assert restamped["semantic_change"] is True
    advisory = restamped["budget_advisories"][0]
    assert advisory == {
        "field": "budgets.change.max_iterations",
        "configured": 24,
        "recommended": 38,
        "shortfall": 14,
    }
    assert "basis" not in advisory
    assert read_config(tmp_path)["budgets"]["change"]["max_iterations"] == 24


def write_mixed_registry(tmp_path: Path, passed: int, pending: int) -> None:
    lines = ["## Active Task Registry", ""]
    entries = []
    for index in range(1, passed + pending + 1):
        done = index <= passed
        mark = "x" if done else " "
        lines.append(f"- [{mark}] 1.{index} Task {index} [#R{index}]")
        lines.append("  - INDEPENDENT: yes")
        entries.append((f"R{index}", f"1.{index}", done, done))
    write_contract(tmp_path, "demo", "\n".join(lines) + "\n", base_features(entries))


def test_restamp_does_not_warn_when_remaining_work_fits(
    tmp_path: Path,
) -> None:
    write_large_registry(tmp_path, 12)
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    seal_demo(tmp_path, ledger, "--max-total-iterations", "24")

    write_mixed_registry(tmp_path, passed=13, pending=6)
    restamped = run_loop(tmp_path, "seal", "demo", "--confirmed")

    assert restamped["semantic_change"] is True
    assert restamped["budget_advisories"] == []
    assert read_config(tmp_path)["budgets"]["change"]["max_iterations"] == 24


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

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


def is_legacy_apply_fixture(args: tuple[str, ...] | list[str]) -> bool:
    values = list(args)
    if not values or values[0] != "record" or "--kind" not in values:
        return False
    if values[values.index("--kind") + 1] != "apply" or "--receipt" in values:
        return False
    packet_fields = {
        "--attempt-id",
        "--packet-id",
        "--role-id",
        "--changed-file",
        "--join-id",
        "--wave-ref",
        "--transient-retries",
        "--record-owner",
    }
    return not any(field in values for field in packet_fields)


def seed_legacy_apply_fixture(repo_root: Path, args: tuple[str, ...] | list[str]) -> dict:
    """Write historical v2-style evidence without reopening the CLI side door."""

    values = list(args)

    def option(name: str, default: str | None = None) -> str | None:
        return values[values.index(name) + 1] if name in values else default

    change_id = values[1]
    run_id = str(option("--run-id", "default"))
    ref = str(option("--ref", "R0"))
    result = str(option("--result", "failure"))
    explicit_ledger = option("--ledger-path")
    if explicit_ledger:
        ledger_path = Path(explicit_ledger)
    else:
        default_ledger = repo_root / "test_cache" / change_id / "loop" / "ledger.json"
        ensure_sealed_runtime(repo_root, default_ledger)
        config = json.loads(
            (repo_root / "openspec" / "changes" / change_id / "loop.json").read_text(
                encoding="utf-8"
            )
        )
        ledger_path = repo_root / config["paths"]["ledger"]
    fingerprint = ensure_sealed_runtime(repo_root, ledger_path)
    module = load_loop_module()
    ledger = module.load_or_init_ledger(ledger_path, change_id)
    _, run = module.load_run(ledger, change_id, fingerprint, run_id)
    prior = len(run.setdefault("attempts", []))
    observation = option("--observation-text")
    error_text = option("--error-text")
    duration = int(str(option("--duration-seconds", "0")))
    attempt = {
        "recorded_at_utc": module.utc_now(),
        "ref": ref,
        "kind": "apply",
        "result": result,
        "action": option("--action", ""),
        "error_fingerprint": module.error_fingerprint(error_text),
        "result_fingerprint": module.error_fingerprint(observation or error_text),
        "duration_seconds": max(duration, 0),
        "apply_record_owner": "supervisor",
        "consumes_apply_attempt": True,
        "consumes_scheduling_headcount": False,
        "canonical_apply_record": True,
        "attempt_id": f"legacy-fixture:{run_id}:{ref}:{prior + 1}",
        "packet_id": f"legacy-fixture:{change_id}:{ref}:{prior + 1}",
        "role_id": "zpy",
        "changed_files": [],
        "evidence": [],
        "join_id": None,
        "wave_refs": [ref],
        "worktree_mode": "legacy",
        "terminal_status": module.canonical_apply_status(result),
        "transient_retries": 0,
        "auto_redispatch": False,
        "next_apply_owner": "supervisor_gate",
        "receipt_id": None,
        "routing_outcome": option("--routing-outcome"),
    }
    run["attempts"].append(attempt)
    ledger["updated_at_utc"] = module.utc_now()
    module.write_json_atomic(ledger_path, ledger)
    return {
        "schema_version": module.SCHEMA_LEDGER,
        "ledger_path": str(ledger_path),
        "run_id": run_id,
        "contract_fingerprint": fingerprint,
        "attempt_id": attempt["attempt_id"],
        "terminal_status": attempt["terminal_status"],
        "legacy_fixture": True,
    }


def test_loop_release_v32_keeps_the_v3_persisted_schema() -> None:
    module = load_loop_module()

    assert module.HARNESS_CONTRACT_VERSION == "openspec-loop-control.v3.2"
    assert module.SCHEMA_LOOP == "openspec-loop.v3"
    assert module.SCHEMA_PROMOTE == "openspec-loop-promote.v2"
    assert module.SCHEMA_SUMMARY == "openspec-loop-summary.v4"


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
    if expected_exit == 0 and is_legacy_apply_fixture(args):
        return seed_legacy_apply_fixture(repo_root, args)
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
    if expected_exit == 0 and is_legacy_apply_fixture(normalized):
        return seed_legacy_apply_fixture(repo_root, normalized)
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--repo-root", str(repo_root), *normalized],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == expected_exit, result.stderr or result.stdout
    return json.loads(result.stdout)


def issue_apply_receipts(repo_root: Path, run_id: str) -> dict[str, dict]:
    payload = run_loop_direct(
        repo_root,
        "next",
        "demo",
        "--intent",
        "drain",
        "--run-id",
        run_id,
    )
    assert payload["next_action"] == "apply", payload
    return {receipt["ref"]: receipt for receipt in payload["receipts"]}


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
    assert "max_iterations" not in config["budgets"]["revision"]
    assert "max_iterations" not in config["budgets"]["change"]
    assert config["per_ref_budgets"] == {
        "R1": {"max_apply_attempts": 2, "max_unblock_runs": 2},
        "R2": {"max_apply_attempts": 2, "max_unblock_runs": 2},
    }
    assert config["budgets"]["task"] == {
        "max_apply_attempts": 2,
        "max_unblock_runs": 2,
    }
    assert config["budgets"]["change"]["max_revisions"] == 3
    assert config["stamp_state"]["charged_cycle_stamps"] == 0
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
    checked = run_loop_direct(tmp_path, "check", "demo")
    assert checked["ok"] is True
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
    receipt = issue_apply_receipts(tmp_path, "first-apply")["R1"]
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
        "--receipt",
        receipt["token"],
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


def test_summary_v3_keeps_attempt_totals_without_apply_iteration_fields(tmp_path: Path) -> None:
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
        "--max-total-active-minutes",
        "240",
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

    assert summary["schema_version"] == "openspec-loop-summary.v4"
    assert summary["revision_attempt_count"] == 8
    assert summary["change_attempt_count"] == 8
    assert "revision_apply_iterations_used" not in summary
    assert "revision_apply_iterations_remaining" not in summary
    assert "change_apply_iterations_used" not in summary
    assert "change_apply_iterations_remaining" not in summary


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


def test_gate_scopes_no_progress_breaker_by_episode_ref_and_kind(tmp_path: Path) -> None:
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
    mixed_kind_gate = run_loop(
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
    )
    assert mixed_kind_gate["decision"] == "continue"
    assert "no_progress_breaker" not in mixed_kind_gate["reasons"]

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
    no_progress_gate = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "another-run",
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


def test_gate_omits_revision_and_change_iteration_stops_but_keeps_task_max_apply_attempts(
    tmp_path: Path,
) -> None:
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
        "--ledger-path",
        str(ledger),
        expected_exit=2,
    )
    assert "task_max_apply_attempts_reached:R1:2" in old_gate["reasons"]
    assert not any("revision_max_iterations_reached" in reason for reason in old_gate["reasons"])
    assert not any("change_max_iterations_reached" in reason for reason in old_gate["reasons"])

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
        "R2",
        "--kind",
        "apply",
        "--ledger-path",
        str(ledger),
    )
    assert fresh_gate["decision"] == "continue"
    assert not any("revision_max_iterations_reached" in reason for reason in fresh_gate["reasons"])
    assert not any("change_max_iterations_reached" in reason for reason in fresh_gate["reasons"])

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
        "--ledger-path",
        str(ledger),
    )
    assert helper_gate["decision"] == "continue"
    assert not any("revision_max_iterations_reached" in reason for reason in helper_gate["reasons"])


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
        "--ref",
        "R1",
        "--kind",
        "apply",
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
        "--ref",
        "R1",
        "--kind",
        "apply",
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
        "--ref",
        "R1",
        "--kind",
        "apply",
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
    policy["budgets"]["revision"]["max_explore_runs"] = 0
    del policy["test_profiles"]["promotion"]
    policy["confirmed_at"] = ""
    loop_path.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    checked = run_loop(tmp_path, "check", "demo", expected_exit=2)

    assert "loop.json paths missing: product" in checked["issues"]
    assert (
        "loop.json budgets.revision must be positive integers: max_explore_runs"
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


def test_verify_deviations_do_not_trip_an_apply_breaker(tmp_path: Path) -> None:
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
    )
    assert gate["decision"] == "continue"
    assert "repeated_result_breaker" not in gate["reasons"]
    assert "repeated_deviation_breaker" not in gate["reasons"]


def test_change_iterations_do_not_stop_a_new_revision_apply(tmp_path: Path) -> None:
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
        "reseal",
        "demo",
        "--allow-semantic-change",
        "--confirmed",
        "--set-max-total-active-minutes",
        "1",
        "--reason",
        "open the next admitted fingerprint with a narrower minute budget",
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
    )
    assert gate["decision"] == "continue"
    assert gate["budgets"]["change"]["max_active_minutes"] == 1


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
    seal_demo(tmp_path, ledger, "--max-revisions", "7", "--max-total-active-minutes", "440")

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
    assert "max_iterations" not in config["budgets"]["change"]
    assert "max_iterations" not in config.get("budgets", {}).get("revision", {})
    assert "max_iterations" not in config.get("hard_ceiling", {})
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

    refused = run_loop(
        tmp_path,
        "seal",
        "demo",
        "--confirmed",
        "--max-revisions",
        "9",
        expected_exit=2,
    )
    assert refused["issues"] == ["a budget change requires --reason"]
    assert read_config(tmp_path)["budgets"]["change"]["max_revisions"] == 7

    changed = run_loop(
        tmp_path,
        "seal",
        "demo",
        "--confirmed",
        "--max-revisions",
        "9",
        "--reason",
        "user confirmed seal budget increase",
    )
    assert changed["changed_fields"] == ["budgets.change.max_revisions"]
    assert changed["self_extensions_used"] == 1
    config = read_config(tmp_path)
    assert config["budgets"]["change"]["max_revisions"] == 9
    extension = config["budget_extensions"][-1]
    assert extension["from"] == {"max_revisions": 7}
    assert extension["to"] == {"max_revisions": 9}
    assert extension["reason"] == "user confirmed seal budget increase"


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


def test_historical_productive_episodes_do_not_gate_same_fingerprint_apply(
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

    still_allowed = run_loop(
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
    assert still_allowed["decision"] == "continue"
    assert not any(
        reason.startswith("change_max_revisions")
        for reason in still_allowed["reasons"]
    )


def stamp_task_text(accept: str, *, second_ref: bool = False) -> str:
    tail = """
- [ ] 1.2 Independent sibling [#R2]
  - INDEPENDENT: yes
  - FILES: `src/b.py`
  - WRITE_SCOPE: `src/b.py`
  - ACCEPT: sibling remains exact.
  - TEST: SCOPE: CLI
    - Run: `python -c "pass"`
""" if second_ref else ""
    return f"""## Active Task Registry

- [ ] 1.1 Runtime ref [#R1]
  - INDEPENDENT: yes
  - FILES: `src/a.py`
  - WRITE_SCOPE: `src/a.py`
  - ACCEPT: {accept}
  - TEST: SCOPE: CLI
    - Run: `python -c "pass"`
{tail}"""


def write_stamp_demo(tmp_path: Path, accept: str, *, second_ref: bool = False) -> Path:
    entries = [("R1", "1.1", False, False)]
    if second_ref:
        entries.append(("R2", "1.2", False, False))
    write_contract(tmp_path, "demo", stamp_task_text(accept, second_ref=second_ref), base_features(entries))
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    seal_demo(
        tmp_path,
        ledger,
        "--scratch-root",
        "test_cache/demo/tmp",
    )
    return ledger


def repair_task_text(
    method: str,
    *,
    state: str = "blocked",
    repair_policy: bool = True,
    test_module: str = "src.old_exec",
) -> str:
    policy = "  - REPAIR_POLICY: bounded-r1\n" if repair_policy else ""
    return f"""## Active Task Registry

- GOAL G1: retain the two-row research product
  - COVERED_BY: R1
  - ACCEPT: exactly 2 rows remain under `outputs/result.json` with fail-closed semantics

- [ ] 1.1 Repairable research method [#R1]
  - INDEPENDENT: yes
  - STATE: {state}
{policy}  - FILES: `src/old_exec.py`
  - WRITE_SCOPE: `src`
  - ACCEPT: exactly 2 rows MUST remain at `outputs/result.json`; failures remain fail-closed.
  - TEST: SCOPE: CLI
    - Run: `python -m {test_module}`
    - Verify: exact row count and output hash remain required
"""


def repair_design_text(method: str, *, outside: str = "stable") -> str:
    return f"""# Design

## Context

Confirmed scope is {outside}.

## Method for R1

<!-- openspec:repair-r1-method ref=R1 start -->
Use `{method}` for the bounded execution block.
<!-- openspec:repair-r1-method ref=R1 end -->

## Risks

Product paths and fail-closed behavior remain frozen.
"""


def write_repair_demo(
    tmp_path: Path,
    *,
    repair_policy: bool = True,
    max_apply_attempts: int = 3,
) -> Path:
    tasks = repair_task_text("locator_v19", repair_policy=repair_policy)
    features = base_features([("R1", "1.1", False, False)])
    features["features"]["R1"].update(
        {"state": "blocked", "repair_policy": "bounded-r1" if repair_policy else None}
    )
    if not repair_policy:
        features["features"]["R1"].pop("repair_policy")
    write_contract(tmp_path, "demo", tasks, features)
    write_thin_retention_decision(tmp_path)
    change_dir = tmp_path / "openspec" / "changes" / "demo"
    (change_dir / "design.md").write_text(
        repair_design_text("locator_v19"), encoding="utf-8"
    )
    specs_dir = change_dir / "specs" / "demo-capability"
    specs_dir.mkdir(parents=True, exist_ok=True)
    (specs_dir / "spec.md").write_text(
        "# Demo specification\n\nTwo rows and fail-closed behavior are required.\n",
        encoding="utf-8",
    )
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    seal_demo(
        tmp_path,
        ledger,
        "--scratch-root",
        "test_cache/demo/tmp",
        "--max-apply-attempts",
        str(max_apply_attempts),
    )
    return ledger


def record_two_blocked_repair_attempts(tmp_path: Path) -> None:
    record_direct_attempt(tmp_path, "apply", "blocked", "old method failure one")
    record_direct_attempt(
        tmp_path,
        "unblock",
        "success",
        "first repair decision",
        disposition="retry",
    )
    record_direct_attempt(tmp_path, "apply", "blocked", "old method failure two")
    run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "ref-local",
        "--ref",
        "R1",
        "--kind",
        "unblock",
        "--result",
        "success",
        "--observation-text",
        "method path is over-specified",
        "--disposition",
        "amend_spec",
        "--repair-class",
        "R1",
    )


def write_repair_candidates(tmp_path: Path, *, outside: str = "stable") -> tuple[Path, Path]:
    scratch = tmp_path / "test_cache" / "demo" / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    tasks = scratch / "candidate-tasks.md"
    design = scratch / "candidate-design.md"
    tasks.write_text(
        repair_task_text(
            "named_exec_block",
            state="pending",
            test_module="src.new_exec",
        ),
        encoding="utf-8",
    )
    design.write_text(
        repair_design_text("named_exec_block", outside=outside), encoding="utf-8"
    )
    return tasks, design


def test_repair_policy_round_trips_to_feature_registry(tmp_path: Path) -> None:
    write_repair_demo(tmp_path)
    module = load_loop_module()
    tasks = (
        tmp_path / "openspec" / "changes" / "demo" / "tasks.md"
    ).read_text(encoding="utf-8")

    payload = module.build_feature_payload_from_tasks_text(tmp_path, "demo", tasks)

    assert payload["features"]["R1"]["repair_policy"] == "bounded-r1"
    planned = run_loop_direct(tmp_path, "plan", "demo")
    item = next(task for task in planned["tasks"] if task["ref"] == "R1")
    assert item["repair_policy"] == "bounded-r1"


def test_unblock_repair_class_defaults_are_fail_closed(tmp_path: Path) -> None:
    retry_root = tmp_path / "retry"
    write_two_ref_runtime(retry_root)
    record_direct_attempt(retry_root, "apply", "blocked", "retry blocker")
    record_direct_attempt(
        retry_root,
        "unblock",
        "success",
        "verified implementation correction",
        disposition="retry",
    )
    retry_config = read_config(retry_root)
    retry_ledger = json.loads(
        (retry_root / retry_config["paths"]["ledger"]).read_text(encoding="utf-8")
    )
    retry_attempt = retry_ledger["episodes"][0]["runs"][0]["attempts"][-1]
    assert retry_attempt["repair_class"] == "R0"

    amend_root = tmp_path / "amend"
    write_two_ref_runtime(amend_root)
    record_direct_attempt(amend_root, "apply", "blocked", "contract blocker")
    record_direct_attempt(
        amend_root,
        "unblock",
        "success",
        "contract boundary requires a human decision",
        disposition="amend_spec",
    )
    amend_config = read_config(amend_root)
    amend_ledger = json.loads(
        (amend_root / amend_config["paths"]["ledger"]).read_text(encoding="utf-8")
    )
    amend_attempt = amend_ledger["episodes"][0]["runs"][0]["attempts"][-1]
    assert amend_attempt["repair_class"] == "R2"

    invalid_root = tmp_path / "invalid"
    write_two_ref_runtime(invalid_root)
    record_direct_attempt(invalid_root, "apply", "blocked", "supersede blocker")
    refused = run_loop_direct(
        invalid_root,
        "record",
        "demo",
        "--run-id",
        "ref-local",
        "--ref",
        "R1",
        "--kind",
        "unblock",
        "--result",
        "success",
        "--observation-text",
        "replacement required",
        "--disposition",
        "supersede_task",
        "--repair-class",
        "R1",
        expected_exit=2,
    )
    assert "supersede_task always requires repair_class R2" in refused["error"]


def test_repair_r1_changes_only_marked_method_and_carries_one_apply(
    tmp_path: Path,
) -> None:
    write_repair_demo(tmp_path)
    record_two_blocked_repair_attempts(tmp_path)
    candidate_tasks, candidate_design = write_repair_candidates(tmp_path)
    before = read_config(tmp_path)
    source_fingerprint = before["contract_fingerprint"]

    applied = run_loop_direct(
        tmp_path,
        "reseal",
        "demo",
        "--allow-semantic-change",
        "--stamp-source",
        "repair_r1",
        "--ref",
        "R1",
        "--candidate-tasks",
        str(candidate_tasks),
        "--candidate-design",
        str(candidate_design),
        "--reason",
        "replace the over-specified method without changing the obligation",
    )

    assert applied["revision_charged"] is True
    after = read_config(tmp_path)
    target_fingerprint = after["contract_fingerprint"]
    assert target_fingerprint != source_fingerprint
    history = after["stamp_state"]["repair_r1_history"]
    assert history == [
        {
            "ref": "R1",
            "source_fingerprint": source_fingerprint,
            "target_fingerprint": target_fingerprint,
            "obligation_hash": history[0]["obligation_hash"],
            "post_repair_apply_limit": 1,
            "recorded_at_utc": history[0]["recorded_at_utc"],
        }
    ]
    planned = run_loop_direct(tmp_path, "plan", "demo")
    item = next(task for task in planned["tasks"] if task["ref"] == "R1")
    assert item["effective_state"] == "ready"
    assert item["apply_attempts_used"] == 2
    assert item["unblock_runs_used"] == 2
    assert planned["dispatch_refs"] == ["R1"]

    third = run_loop_direct(
        tmp_path,
        "gate",
        "demo",
        "--run-id",
        "repair-third",
        "--ref",
        "R1",
        "--kind",
        "apply",
    )
    assert third["decision"] == "continue"
    run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "repair-third",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "completed",
    )
    exhausted = run_loop_direct(
        tmp_path,
        "gate",
        "demo",
        "--run-id",
        "repair-third",
        "--ref",
        "R1",
        "--kind",
        "apply",
        expected_exit=2,
    )
    assert "repair_r1_post_apply_exhausted:R1" in exhausted["reasons"]
    no_third_unblock = run_loop_direct(
        tmp_path,
        "gate",
        "demo",
        "--run-id",
        "repair-third",
        "--ref",
        "R1",
        "--kind",
        "unblock",
        expected_exit=2,
    )
    assert "repair_r1_unblock_closed:R1" in no_third_unblock["reasons"]


def test_repair_r1_after_first_apply_still_grants_only_one_post_repair_apply(
    tmp_path: Path,
) -> None:
    write_repair_demo(tmp_path)
    record_direct_attempt(tmp_path, "apply", "blocked", "first method failure")
    run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "ref-local",
        "--ref",
        "R1",
        "--kind",
        "unblock",
        "--result",
        "success",
        "--observation-text",
        "first diagnosis proves a method-only defect",
        "--disposition",
        "amend_spec",
        "--repair-class",
        "R1",
    )
    candidate_tasks, candidate_design = write_repair_candidates(tmp_path)
    run_loop_direct(
        tmp_path,
        "reseal",
        "demo",
        "--allow-semantic-change",
        "--stamp-source",
        "repair_r1",
        "--ref",
        "R1",
        "--candidate-tasks",
        str(candidate_tasks),
        "--candidate-design",
        str(candidate_design),
        "--reason",
        "repair after the first blocked Apply",
    )
    initial = run_loop_direct(tmp_path, "plan", "demo")
    assert initial["dispatch_refs"] == ["R1"]
    run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "repair-final",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "completed",
    )

    after = run_loop_direct(tmp_path, "plan", "demo")
    item = next(task for task in after["tasks"] if task["ref"] == "R1")
    assert item["apply_attempts_used"] == 2
    assert item["max_apply_attempts"] == 3
    assert item["repair_apply_exhausted"] is True
    assert after["apply_remaining"] == 0
    assert after["dispatch_refs"] == []


def test_repair_r1_requires_opt_in_and_rejects_unmarked_design_change(
    tmp_path: Path,
) -> None:
    write_repair_demo(tmp_path, repair_policy=False)
    record_two_blocked_repair_attempts(tmp_path)
    candidate_tasks, candidate_design = write_repair_candidates(tmp_path)
    change_dir = tmp_path / "openspec" / "changes" / "demo"
    before = {
        name: (change_dir / name).read_bytes()
        for name in ("design.md", "tasks.md", "feature_list.json", "loop.json")
    }

    refused = run_loop_direct(
        tmp_path,
        "reseal",
        "demo",
        "--allow-semantic-change",
        "--stamp-source",
        "repair_r1",
        "--ref",
        "R1",
        "--candidate-tasks",
        str(candidate_tasks),
        "--candidate-design",
        str(candidate_design),
        "--reason",
        "not opted in",
        expected_exit=2,
    )
    assert any("repair_r1_policy_required" in issue for issue in refused["issues"])
    assert refused["repair_class"] == "R2"
    for name, payload in before.items():
        assert (change_dir / name).read_bytes() == payload

    # A separately sealed opted-in contract still rejects protected Context edits.
    protected_root = tmp_path / "protected"
    write_repair_demo(protected_root)
    record_two_blocked_repair_attempts(protected_root)
    change_dir = protected_root / "openspec" / "changes" / "demo"
    candidate_tasks, candidate_design = write_repair_candidates(
        protected_root, outside="changed protected scope"
    )
    protected_before = {
        name: (change_dir / name).read_bytes()
        for name in ("design.md", "tasks.md", "feature_list.json", "loop.json")
    }
    rejected = run_loop_direct(
        protected_root,
        "reseal",
        "demo",
        "--allow-semantic-change",
        "--stamp-source",
        "repair_r1",
        "--ref",
        "R1",
        "--candidate-tasks",
        str(candidate_tasks),
        "--candidate-design",
        str(candidate_design),
        "--reason",
        "attempt protected edit",
        expected_exit=2,
    )
    assert any("design_outside_method" in issue for issue in rejected["issues"])
    assert rejected["repair_class"] == "R2"
    for name, payload in protected_before.items():
        assert (change_dir / name).read_bytes() == payload


def test_repair_r1_legacy_design_fallback_rejects_multiple_changed_sections() -> None:
    module = load_loop_module()
    before = """# Design

## Method
Use old method.

## Runtime
Use old runtime.
"""
    after = """# Design

## Method
Use new method.

## Runtime
Use new runtime.
"""

    projection, issues = module.design_method_change_projection(
        before,
        after,
        ref="R1",
        active_ref_count=1,
    )

    assert projection is None
    assert issues == ["repair_r1_design_ambiguous_or_outside_method:R1"]


def test_repair_r1_marker_layout_rejects_missing_and_duplicate_pairs() -> None:
    module = load_loop_module()
    before = repair_design_text("legacy_parser")
    missing = before.replace(
        "<!-- openspec:repair-r1-method ref=R1 end -->", ""
    )
    projection, issues = module.design_method_change_projection(
        before,
        missing,
        ref="R1",
        active_ref_count=1,
    )
    assert projection is None
    assert any("marker_pair_invalid" in issue for issue in issues)

    duplicate = before + """
## Runtime for R1
<!-- openspec:repair-r1-method ref=R1 start -->
Use another method.
<!-- openspec:repair-r1-method ref=R1 end -->
"""
    layout = module.repair_r1_marker_layout_issues(
        duplicate,
        active_refs=["R1"],
        source="candidate",
    )
    assert any("pair_count:R1" in issue for issue in layout)


def test_legacy_accept_signature_allows_only_mechanical_method_tokens() -> None:
    module = load_loop_module()
    before = (
        "exactly 2 rows MUST pass via `legacy_parser` at "
        "`outputs/result.json`; failures remain fail-closed."
    )
    after = before.replace("`legacy_parser`", "`named_executor`")

    assert module.stable_accept_obligation_signature(before) == (
        module.stable_accept_obligation_signature(after)
    )
    assert module.stable_accept_obligation_signature(before) != (
        module.stable_accept_obligation_signature(before.replace("2 rows", "3 rows"))
    )
    assert module.stable_accept_obligation_signature(before) != (
        module.stable_accept_obligation_signature(
            before.replace("outputs/result.json", "outputs/other.json")
        )
    )


def test_repair_r1_test_signature_treats_multiline_run_as_one_exec_block() -> None:
    module = load_loop_module()
    before = """SCOPE: CLI
Run:
```powershell
python -m src.old_exec
```
Verify: exact hashes remain required
"""
    after = before.replace("src.old_exec", "src.new_exec --pilot")

    assert module.stable_test_non_run_signature(before) == (
        module.stable_test_non_run_signature(after)
    )
    assert module.stable_test_non_run_signature(before) != (
        module.stable_test_non_run_signature(
            after.replace("exact hashes remain required", "best effort")
        )
    )


def test_repair_r1_rejects_new_test_write_path_outside_scope(tmp_path: Path) -> None:
    write_repair_demo(tmp_path)
    record_direct_attempt(tmp_path, "apply", "blocked", "method failure")
    run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "ref-local",
        "--ref",
        "R1",
        "--kind",
        "unblock",
        "--result",
        "success",
        "--observation-text",
        "method-only repair",
        "--disposition",
        "amend_spec",
        "--repair-class",
        "R1",
    )
    candidate_tasks, candidate_design = write_repair_candidates(tmp_path)
    candidate_tasks.write_text(
        candidate_tasks.read_text(encoding="utf-8").replace(
            "`python -m src.new_exec`",
            "`python -m src.new_exec --output data/out.csv`",
        ),
        encoding="utf-8",
    )

    refused = run_loop_direct(
        tmp_path,
        "reseal",
        "demo",
        "--allow-semantic-change",
        "--stamp-source",
        "repair_r1",
        "--ref",
        "R1",
        "--candidate-tasks",
        str(candidate_tasks),
        "--candidate-design",
        str(candidate_design),
        "--reason",
        "attempt an undeclared output root",
        expected_exit=2,
    )
    assert any("test_path_outside_scope" in issue for issue in refused["issues"])
    assert refused["repair_class"] == "R2"


def test_repair_r1_rejects_files_changes_and_cross_ref_markers(tmp_path: Path) -> None:
    write_repair_demo(tmp_path)
    record_two_blocked_repair_attempts(tmp_path)
    candidate_tasks, candidate_design = write_repair_candidates(tmp_path)
    candidate_tasks.write_text(
        candidate_tasks.read_text(encoding="utf-8").replace(
            "`src/old_exec.py`", "`src/new_exec.py`"
        ),
        encoding="utf-8",
    )

    files_refused = run_loop_direct(
        tmp_path,
        "reseal",
        "demo",
        "--allow-semantic-change",
        "--stamp-source",
        "repair_r1",
        "--ref",
        "R1",
        "--candidate-tasks",
        str(candidate_tasks),
        "--candidate-design",
        str(candidate_design),
        "--reason",
        "attempt to replace the task-owned file",
        expected_exit=2,
    )
    assert any("files_changed" in issue for issue in files_refused["issues"])
    assert files_refused["repair_class"] == "R2"

    candidate_tasks, candidate_design = write_repair_candidates(tmp_path)
    candidate_tasks.write_text(
        candidate_tasks.read_text(encoding="utf-8").replace(
            "exactly 2 rows MUST remain", "exactly 3 rows MUST remain"
        ),
        encoding="utf-8",
    )
    accept_refused = run_loop_direct(
        tmp_path,
        "reseal",
        "demo",
        "--allow-semantic-change",
        "--stamp-source",
        "repair_r1",
        "--ref",
        "R1",
        "--candidate-tasks",
        str(candidate_tasks),
        "--candidate-design",
        str(candidate_design),
        "--reason",
        "attempt to widen the outcome",
        expected_exit=2,
    )
    assert any("accept_outcome_changed" in issue for issue in accept_refused["issues"])
    assert accept_refused["repair_class"] == "R2"

    candidate_tasks, candidate_design = write_repair_candidates(tmp_path)
    candidate_tasks.write_text(
        candidate_tasks.read_text(encoding="utf-8").replace(
            "exact row count and output hash remain required",
            "best effort output inspection",
        ),
        encoding="utf-8",
    )
    test_refused = run_loop_direct(
        tmp_path,
        "reseal",
        "demo",
        "--allow-semantic-change",
        "--stamp-source",
        "repair_r1",
        "--ref",
        "R1",
        "--candidate-tasks",
        str(candidate_tasks),
        "--candidate-design",
        str(candidate_design),
        "--reason",
        "attempt to relax a non-Run test obligation",
        expected_exit=2,
    )
    assert any("test_non_run_changed" in issue for issue in test_refused["issues"])
    assert test_refused["repair_class"] == "R2"

    candidate_tasks, candidate_design = write_repair_candidates(tmp_path)
    candidate_design.write_text(
        candidate_design.read_text(encoding="utf-8").replace(
            "Use `named_exec_block` for the bounded execution block.",
            """<!-- openspec:repair-r1-method ref=R2 start -->
Use `named_exec_block` for the bounded execution block.
<!-- openspec:repair-r1-method ref=R2 end -->""",
        ),
        encoding="utf-8",
    )
    marker_refused = run_loop_direct(
        tmp_path,
        "reseal",
        "demo",
        "--allow-semantic-change",
        "--stamp-source",
        "repair_r1",
        "--ref",
        "R1",
        "--candidate-tasks",
        str(candidate_tasks),
        "--candidate-design",
        str(candidate_design),
        "--reason",
        "attempt a nested marker for another ref",
        expected_exit=2,
    )
    assert any("marker_layout" in issue for issue in marker_refused["issues"])
    assert marker_refused["repair_class"] == "R2"


def test_repair_r1_requires_exact_explicit_apply_three_budget(tmp_path: Path) -> None:
    write_repair_demo(tmp_path, max_apply_attempts=4)
    record_two_blocked_repair_attempts(tmp_path)
    candidate_tasks, candidate_design = write_repair_candidates(tmp_path)

    refused = run_loop_direct(
        tmp_path,
        "reseal",
        "demo",
        "--allow-semantic-change",
        "--stamp-source",
        "repair_r1",
        "--ref",
        "R1",
        "--candidate-tasks",
        str(candidate_tasks),
        "--candidate-design",
        str(candidate_design),
        "--reason",
        "budget must be exactly the explicit research allowance",
        expected_exit=2,
    )

    assert any("requires_max_apply_attempts_3" in issue for issue in refused["issues"])
    assert refused["repair_class"] == "R2"


def test_failed_post_repair_apply_stops_the_repaired_ref(tmp_path: Path) -> None:
    write_repair_demo(tmp_path)
    record_two_blocked_repair_attempts(tmp_path)
    candidate_tasks, candidate_design = write_repair_candidates(tmp_path)
    run_loop_direct(
        tmp_path,
        "reseal",
        "demo",
        "--allow-semantic-change",
        "--stamp-source",
        "repair_r1",
        "--ref",
        "R1",
        "--candidate-tasks",
        str(candidate_tasks),
        "--candidate-design",
        str(candidate_design),
        "--reason",
        "admit one bounded method repair",
    )
    run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "repair-final-failure",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "blocked",
        "--observation-text",
        "the repaired method still cannot satisfy the outcome",
    )

    planned = run_loop_direct(tmp_path, "plan", "demo")
    item = next(task for task in planned["tasks"] if task["ref"] == "R1")
    assert item["effective_state"] == "maxed"
    assert item["budget_disposition"] == "stop_budget"
    assert planned["dispatch_refs"] == []


def test_repair_r1_feature_generation_failure_keeps_all_four_files(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    write_repair_demo(tmp_path)
    record_two_blocked_repair_attempts(tmp_path)
    candidate_tasks, candidate_design = write_repair_candidates(tmp_path)
    change_dir = tmp_path / "openspec" / "changes" / "demo"
    before = {
        name: (change_dir / name).read_bytes()
        for name in ("design.md", "tasks.md", "feature_list.json", "loop.json")
    }
    loop = load_loop_module()

    def refuse(*_args, **_kwargs):
        raise ValueError("injected feature generation failure")

    monkeypatch.setattr(loop, "build_feature_payload_from_tasks_text", refuse)
    args = loop.build_parser().parse_args(
        [
            "--repo-root",
            str(tmp_path),
            "reseal",
            "demo",
            "--allow-semantic-change",
            "--stamp-source",
            "repair_r1",
            "--ref",
            "R1",
            "--candidate-tasks",
            str(candidate_tasks),
            "--candidate-design",
            str(candidate_design),
            "--reason",
            "exercise feature generation rollback",
        ]
    )

    assert args.func(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert "injected feature generation failure" in payload["issues"][0]
    for name, content in before.items():
        assert (change_dir / name).read_bytes() == content


def test_repair_r1_strict_validation_failure_restores_all_four_files(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    write_repair_demo(tmp_path)
    record_two_blocked_repair_attempts(tmp_path)
    candidate_tasks, candidate_design = write_repair_candidates(tmp_path)
    change_dir = tmp_path / "openspec" / "changes" / "demo"
    before = {
        name: (change_dir / name).read_bytes()
        for name in ("design.md", "tasks.md", "feature_list.json", "loop.json")
    }
    loop = load_loop_module()
    monkeypatch.setattr(
        loop,
        "strict_validation_state",
        lambda *_args, **_kwargs: ("failed", "injected strict validation failure"),
    )
    args = loop.build_parser().parse_args(
        [
            "--repo-root",
            str(tmp_path),
            "reseal",
            "demo",
            "--allow-semantic-change",
            "--stamp-source",
            "repair_r1",
            "--ref",
            "R1",
            "--candidate-tasks",
            str(candidate_tasks),
            "--candidate-design",
            str(candidate_design),
            "--reason",
            "exercise strict validation rollback",
        ]
    )

    assert args.func(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert "injected strict validation failure" in payload["issues"][0]
    for name, content in before.items():
        assert (change_dir / name).read_bytes() == content


def test_cycle_stamp_cap_charges_in_chapter_and_rejects_the_fourth_before_writes(
    tmp_path: Path,
) -> None:
    write_stamp_demo(tmp_path, "exactly 2 rows MUST pass.")
    change_dir = tmp_path / "openspec" / "changes" / "demo"
    tasks_path = change_dir / "tasks.md"
    feature_path = change_dir / "feature_list.json"
    loop_path = change_dir / "loop.json"

    # The first confirmed pre-execution semantic stamp is chapter-outside/free.
    for index in range(1, 5):
        tasks_path.write_text(
            stamp_task_text(
                "exactly 2 rows MUST pass. "
                + " ".join(f"Gate {item} MUST remain." for item in range(1, index + 1))
            ),
            encoding="utf-8",
        )
        result = run_loop(
            tmp_path,
            "reseal",
            "demo",
            "--allow-semantic-change",
            "--confirmed",
            "--reason",
            f"confirmed stamp {index}",
        )
        assert result["written"] is True

    config = read_config(tmp_path)
    assert config["stamp_state"]["semantic_stamps_in_chapter"] == 4
    assert config["stamp_state"]["charged_cycle_stamps"] == 3

    tasks_path.write_text(
        stamp_task_text(
            "exactly 2 rows MUST pass. "
            + " ".join(f"Gate {item} MUST remain." for item in range(1, 6))
        ),
        encoding="utf-8",
    )
    before = {
        "tasks": tasks_path.read_bytes(),
        "feature": feature_path.read_bytes(),
        "loop": loop_path.read_bytes(),
    }
    refused = run_loop(
        tmp_path,
        "reseal",
        "demo",
        "--allow-semantic-change",
        "--confirmed",
        "--reason",
        "fourth charged cycle stamp",
        expected_exit=2,
    )
    assert "change_cycle_stamp_budget_exhausted:3" in refused["issues"]
    assert tasks_path.read_bytes() == before["tasks"]
    assert feature_path.read_bytes() == before["feature"]
    assert loop_path.read_bytes() == before["loop"]


def test_completion_and_unblock_rights_ignore_minutes_and_generic_breakers(
    tmp_path: Path,
) -> None:
    ledger = write_stamp_demo(tmp_path, "exactly 2 rows MUST pass.")
    fingerprint = read_config(tmp_path)["contract_fingerprint"]
    run_loop(
        tmp_path,
        "record",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "closure",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "completed",
        "--duration-seconds",
        "61",
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
        "closure",
        "--ref",
        "R1",
        "--kind",
        "review",
        "--result",
        "no_progress",
        "--duration-seconds",
        "61",
        "--ledger-path",
        str(ledger),
    )

    for kind in ("verify", "goal", "review", "stop_hook"):
        args = [
            "gate",
            "demo",
            "--run-id",
            "closure",
            "--kind",
            kind,
            "--max-minutes",
            "1",
            "--ledger-path",
            str(ledger),
        ]
        if kind == "verify":
            args.extend(["--ref", "R1"])
        gate = run_loop(tmp_path, *args)
        assert gate["decision"] == "continue"
        assert not any("active_minutes" in reason for reason in gate["reasons"])

    apply_gate = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--run-id",
        "closure",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--max-minutes",
        "1",
        "--ledger-path",
        str(ledger),
        expected_exit=2,
    )
    assert "revision_active_minutes_reached:1" in apply_gate["reasons"]

    run_loop(
        tmp_path,
        "record",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "closure",
        "--ref",
        "R1",
        "--kind",
        "verify",
        "--result",
        "blocked",
        "--observation-text",
        "new blocking evidence",
        "--ledger-path",
        str(ledger),
    )
    unblock = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--run-id",
        "closure",
        "--ref",
        "R1",
        "--kind",
        "unblock",
        "--max-minutes",
        "1",
        "--ledger-path",
        str(ledger),
    )
    assert unblock["decision"] == "continue"


def test_gate_requires_kind_and_ref_for_ref_scoped_actions(tmp_path: Path) -> None:
    ledger = write_stamp_demo(tmp_path, "exactly 2 rows MUST pass.")
    missing_kind = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--repo-root",
            str(tmp_path),
            "gate",
            "demo",
            "--ledger-path",
            str(ledger),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert missing_kind.returncode == 2
    assert "--kind" in missing_kind.stderr

    missing_ref = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--kind",
        "verify",
        "--ledger-path",
        str(ledger),
        expected_exit=2,
    )
    assert "requires --ref" in missing_ref["error"]


def test_unblock_self_confirm_allows_one_narrow_same_ref_candidate(tmp_path: Path) -> None:
    ledger = write_stamp_demo(tmp_path, "exactly 2 rows MUST pass.", second_ref=True)
    config_before = read_config(tmp_path)
    fingerprint = config_before["contract_fingerprint"]
    run_loop(
        tmp_path,
        "record",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "blocked",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "blocked",
        "--observation-text",
        "missing exact gate",
        "--ledger-path",
        str(ledger),
    )
    candidate = tmp_path / "test_cache" / "demo" / "tmp" / "candidate-tasks.md"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(
        stamp_task_text(
            "exactly 2 rows MUST pass. The source hash MUST also match.",
            second_ref=True,
        ),
        encoding="utf-8",
    )
    applied = run_loop(
        tmp_path,
        "reseal",
        "demo",
        "--allow-semantic-change",
        "--stamp-source",
        "unblock_self_confirm",
        "--ref",
        "R1",
        "--candidate-tasks",
        str(candidate),
        "--reason",
        "narrow R1 after blocked Apply",
    )
    assert applied["revision_charged"] is True
    assert applied["charged_cycle_stamps"] == 1
    config_after = read_config(tmp_path)
    assert config_after["stamp_state"]["self_confirmed_refs"][fingerprint] == ["R1"]
    assert config_after["budgets"] == config_before["budgets"]
    assert "source hash MUST also match" in (
        tmp_path / "openspec" / "changes" / "demo" / "tasks.md"
    ).read_text(encoding="utf-8")


def test_unblock_self_confirm_rejects_widening_and_keeps_authority_files(
    tmp_path: Path,
) -> None:
    ledger = write_stamp_demo(tmp_path, "exactly 2 rows MUST pass.")
    fingerprint = read_config(tmp_path)["contract_fingerprint"]
    run_loop(
        tmp_path,
        "record",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "blocked",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "blocked",
        "--observation-text",
        "missing exact gate",
        "--ledger-path",
        str(ledger),
    )
    change_dir = tmp_path / "openspec" / "changes" / "demo"
    candidate = tmp_path / "test_cache" / "demo" / "tmp" / "widened.md"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(stamp_task_text("rows may pass."), encoding="utf-8")
    before = {
        name: (change_dir / name).read_bytes()
        for name in ("tasks.md", "feature_list.json", "loop.json")
    }
    refused = run_loop(
        tmp_path,
        "reseal",
        "demo",
        "--allow-semantic-change",
        "--stamp-source",
        "unblock_self_confirm",
        "--ref",
        "R1",
        "--candidate-tasks",
        str(candidate),
        "--reason",
        "attempt to relax acceptance",
        expected_exit=2,
    )
    assert any("accept_widening" in issue for issue in refused["issues"])
    for name, payload in before.items():
        assert (change_dir / name).read_bytes() == payload


def test_unblock_self_confirm_rejects_a_second_same_ref_source_stamp(
    tmp_path: Path,
) -> None:
    ledger = write_stamp_demo(tmp_path, "exactly 2 rows MUST pass.")
    fingerprint = read_config(tmp_path)["contract_fingerprint"]
    run_loop(
        tmp_path,
        "record",
        "demo",
        "--contract-fingerprint",
        fingerprint,
        "--run-id",
        "blocked",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "blocked",
        "--observation-text",
        "missing exact gate",
        "--ledger-path",
        str(ledger),
    )
    loop_path = tmp_path / "openspec" / "changes" / "demo" / "loop.json"
    config = read_config(tmp_path)
    config["stamp_state"]["self_confirmed_refs"] = {fingerprint: ["R1"]}
    loop_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    candidate = tmp_path / "test_cache" / "demo" / "tmp" / "repeat.md"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(
        stamp_task_text("exactly 2 rows MUST pass. One more gate MUST hold."),
        encoding="utf-8",
    )
    refused = run_loop(
        tmp_path,
        "reseal",
        "demo",
        "--allow-semantic-change",
        "--stamp-source",
        "unblock_self_confirm",
        "--ref",
        "R1",
        "--candidate-tasks",
        str(candidate),
        "--reason",
        "repeat",
        expected_exit=2,
    )
    assert any("self_restamp_already_used" in issue for issue in refused["issues"])


def test_apply_record_cannot_claim_a_tasks_md_write(tmp_path: Path) -> None:
    ledger = write_stamp_demo(tmp_path, "exactly 2 rows MUST pass.")
    receipt = issue_apply_receipts(tmp_path, "worker")["R1"]
    refused = run_loop(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "worker",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "completed",
        "--receipt",
        receipt["token"],
        "--changed-file",
        "openspec/changes/demo/tasks.md",
        "--ledger-path",
        str(ledger),
        expected_exit=2,
    )
    assert "may not edit tasks.md" in refused["error"]


def seal_ceiling_demo(tmp_path: Path, *extra: str) -> Path:
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
        "--hard-ceiling-max-active-minutes",
        "1080",
        "--hard-ceiling-max-self-extensions",
        "3",
        *extra,
    )
    return ledger


def test_hard_ceiling_keeps_minutes_only_and_does_not_bound_gate_overrides(tmp_path: Path) -> None:
    ledger = seal_ceiling_demo(tmp_path)
    config = read_config(tmp_path)
    assert config["hard_ceiling"] == {
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
        "--max-active-minutes",
        "99999",
        "--ledger-path",
        str(ledger),
    )
    assert gate["decision"] == "continue"
    assert gate["terminal"] is False
    assert gate["budgets"]["revision"]["max_active_minutes"] == 99999
    assert gate["hard_ceiling"] == config["hard_ceiling"]
    assert gate["self_extensions_used"] == 0


def test_deleted_iteration_keys_are_ignored_and_migrated(tmp_path: Path) -> None:
    ledger = seal_ceiling_demo(tmp_path)
    fingerprint = read_config(tmp_path)["contract_fingerprint"]
    loop_path = tmp_path / "openspec" / "changes" / "demo" / "loop.json"
    config = read_config(tmp_path)
    config["budgets"]["revision"]["max_iterations"] = 2
    config["budgets"]["change"]["max_iterations"] = 2
    config.setdefault("hard_ceiling", {})["max_iterations"] = 5
    loop_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
        "--max-apply-attempts",
        "3",
        "--ledger-path",
        str(ledger),
    )
    assert extendable["decision"] == "continue"
    assert not any("max_iterations_reached" in reason for reason in extendable["reasons"])
    assert extendable["terminal"] is False
    migrated = run_loop_direct(tmp_path, "plan", "demo")
    assert migrated["selected_wave"] == ["R1"]
    assert migrated["apply_remaining"] == 0
    cleaned = read_config(tmp_path)
    assert "max_iterations" not in cleaned["budgets"]["revision"]
    assert "max_iterations" not in cleaned["budgets"]["change"]
    assert "max_iterations" not in cleaned["hard_ceiling"]


def test_reseal_never_raises_hard_ceiling_and_refuses_a_budget_above_it(
    tmp_path: Path,
) -> None:
    seal_ceiling_demo(
        tmp_path,
        "--max-total-active-minutes",
        "20",
        "--hard-ceiling-max-active-minutes",
        "30",
    )

    refused = run_loop(
        tmp_path,
        "reseal",
        "demo",
        "--set-max-total-active-minutes",
        "40",
        "--confirmed",
        "--reason",
        "too much",
        expected_exit=2,
    )
    assert refused["written"] is False
    assert any("exceeds hard_ceiling.max_active_minutes 30" in issue for issue in refused["issues"])
    assert read_config(tmp_path)["budgets"]["change"]["max_active_minutes"] == 20

    allowed = run_loop(
        tmp_path,
        "reseal",
        "demo",
        "--set-max-total-active-minutes",
        "30",
        "--confirmed",
        "--reason",
        "authorized extension",
    )
    assert allowed["changed_fields"] == ["change.max_active_minutes"]
    config = read_config(tmp_path)
    assert config["budgets"]["change"]["max_active_minutes"] == 30
    assert config["hard_ceiling"]["max_active_minutes"] == 30


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


def test_seal_budget_increase_consumes_self_extension_ceiling(
    tmp_path: Path,
) -> None:
    seal_ceiling_demo(tmp_path, "--hard-ceiling-max-self-extensions", "1")

    first = run_loop(
        tmp_path,
        "seal",
        "demo",
        "--confirmed",
        "--max-revisions",
        "4",
        "--reason",
        "confirmed seal extension",
    )
    assert first["self_extensions_used"] == 1
    assert read_config(tmp_path)["budget_extensions"][-1]["to"] == {
        "max_revisions": 4
    }

    refused = run_loop(
        tmp_path,
        "seal",
        "demo",
        "--confirmed",
        "--max-revisions",
        "5",
        "--reason",
        "second seal extension",
        expected_exit=2,
    )
    assert any("max_self_extensions reached: 1" in issue for issue in refused["issues"])
    assert read_config(tmp_path)["budgets"]["change"]["max_revisions"] == 4


def test_budget_reduction_is_not_a_self_extension_or_admission_revocation(
    tmp_path: Path,
) -> None:
    ledger = seal_ceiling_demo(
        tmp_path,
        "--max-revisions",
        "5",
        "--hard-ceiling-max-self-extensions",
        "1",
    )
    reduced = run_loop(
        tmp_path,
        "reseal",
        "demo",
        "--set-max-revisions",
        "2",
        "--confirmed",
        "--reason",
        "reduce an emergency umbrella",
    )
    assert reduced["self_extensions_used"] == 0
    assert read_config(tmp_path)["budgets"]["change"]["max_revisions"] == 2
    gate = run_loop(
        tmp_path,
        "gate",
        "demo",
        "--run-id",
        "after-reduction",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--ledger-path",
        str(ledger),
    )
    assert gate["decision"] == "continue"


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
        "--max-total-active-minutes",
        "40",
        "--hard-ceiling-max-active-minutes",
        "10",
        expected_exit=2,
    )
    assert refused["written"] is False
    assert "budgets.change.max_active_minutes exceeds hard_ceiling.max_active_minutes" in refused["issues"]
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
    assert promoted["schema_version"] == "openspec-loop-promote.v2"
    assert promoted["next_required"] is True
    assert "ready_refs" not in promoted
    assert "selected_ref" not in promoted
    assert "terminal_refs" not in promoted

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


def test_promote_does_not_rebuild_the_plan_payload(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    loop = load_loop_module()
    ledger = seal_promotable_demo(tmp_path)
    record_verify_pass(tmp_path, ledger, "R1")

    def refuse(*_args, **_kwargs):
        raise AssertionError("promote must not call build_plan_payload")

    monkeypatch.setattr(loop, "build_plan_payload", refuse)
    args = loop.build_parser().parse_args(
        ["--repo-root", str(tmp_path), "promote", "demo", "--ref", "R1"]
    )

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "openspec-loop-promote.v2"
    assert payload["promoted"] is True
    assert payload["next_required"] is True


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


def test_promote_rolls_back_task_feature_and_ledger_when_telemetry_write_fails(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    loop = load_loop_module()
    ledger = seal_promotable_demo(tmp_path)
    record_verify_pass(tmp_path, ledger, "R1")
    change_dir = tmp_path / "openspec" / "changes" / "demo"
    before = {
        "tasks": (change_dir / "tasks.md").read_bytes(),
        "feature": (change_dir / "feature_list.json").read_bytes(),
        "ledger": ledger.read_bytes(),
    }
    original_write = loop.write_json_atomic

    def refuse(path: Path, payload: dict) -> None:
        if Path(path) == ledger:
            raise OSError("telemetry write failed")
        original_write(path, payload)

    monkeypatch.setattr(loop, "write_json_atomic", refuse)
    args = loop.build_parser().parse_args(
        ["--repo-root", str(tmp_path), "promote", "demo", "--ref", "R1"]
    )

    assert args.func(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["promoted"] is False
    assert "telemetry write failed" in payload["issues"][0]
    assert (change_dir / "tasks.md").read_bytes() == before["tasks"]
    assert (change_dir / "feature_list.json").read_bytes() == before["feature"]
    assert ledger.read_bytes() == before["ledger"]


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


def test_local_zpy_direct_role_preserves_legacy_rose_bootstrap(
    tmp_path: Path,
) -> None:
    tasks = """## Active Task Registry

- [ ] 1.1 Legacy bootstrap [#R1]
  - INDEPENDENT: yes
  - ROLE_ID: rose
  - FILES: `docs/bootstrap.md`
  - WRITE_SCOPE: `docs/bootstrap.md`
  - JOIN: role-bootstrap
- [ ] 1.2 Local supervisor [#R2]
  - INDEPENDENT: yes
  - ROLE_ID: zpy
  - FILES: `docs/zpy.md`
  - WRITE_SCOPE: `docs/zpy.md`
  - JOIN: role-bootstrap
- [ ] 1.3 Inferred direct [#R3]
  - INDEPENDENT: yes
  - JOIN: role-bootstrap
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
    routes = {route["ref"]: route for route in payload["routing"]}

    assert routes["R1"]["matched_role_id"] == "rose"
    assert routes["R1"]["effective_role_id"] == "rose"
    assert routes["R1"]["decision"] == "direct"
    assert routes["R1"]["agent"] is None
    assert routes["R2"]["matched_role_id"] == "zpy"
    assert routes["R2"]["effective_role_id"] == "zpy"
    assert routes["R2"]["decision"] == "direct"
    assert routes["R2"]["agent"] is None
    assert routes["R3"]["matched_role_id"] is None
    assert routes["R3"]["effective_role_id"] == "zpy"
    assert routes["R3"]["decision"] == "direct"
    assert routes["R3"]["agent"] is None
    assert all(route["write_policy"] == "supervisor_direct" for route in routes.values())
    receipts = issue_apply_receipts(tmp_path, "role-run")

    run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "role-run",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "completed",
        "--receipt",
        receipts["R1"]["token"],
        "--changed-file",
        "docs/bootstrap.md",
        "--evidence",
        "inspect:R1",
    )
    run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "role-run",
        "--ref",
        "R2",
        "--kind",
        "apply",
        "--result",
        "completed",
        "--receipt",
        receipts["R2"]["token"],
        "--changed-file",
        "docs/zpy.md",
        "--evidence",
        "inspect:R2",
    )
    summary = run_loop_direct(tmp_path, "summary", "demo", "--run-id", "role-run")
    by_ref = {attempt["ref"]: attempt for attempt in summary["attempts"]}
    assert by_ref["R1"]["role_id"] == "rose"
    assert by_ref["R2"]["role_id"] == "zpy"
    assert by_ref["R1"]["consumes_scheduling_headcount"] is False
    assert by_ref["R2"]["consumes_scheduling_headcount"] is False


def test_dispatch_refs_keep_direct_zpy_refs_out_of_host_batch(tmp_path: Path) -> None:
    tasks = """## Active Task Registry

- [ ] 1.1 Direct supervisor ref [#Rd]
  - INDEPENDENT: yes
  - ROLE_ID: zpy
  - JOIN: mixed-host-batch
- [ ] 1.2 Implement lane one [#Ri1]
  - INDEPENDENT: yes
  - ROLE_ID: implementer
  - FILES: `src/lane_one.py`
  - WRITE_SCOPE: `src/lane_one.py`
  - JOIN: mixed-host-batch
- [ ] 1.3 Implement lane two [#Ri2]
  - INDEPENDENT: yes
  - ROLE_ID: implementer
  - FILES: `src/lane_two.py`
  - WRITE_SCOPE: `src/lane_two.py`
  - JOIN: mixed-host-batch
"""
    write_contract(
        tmp_path,
        "demo",
        tasks,
        base_features(
            [("Rd", "1.1", False, False), ("Ri1", "1.2", False, False), ("Ri2", "1.3", False, False)]
        ),
    )
    write_thin_retention_decision(tmp_path)

    payload = run_loop_direct(tmp_path, "plan", "demo")
    routes = {route["ref"]: route for route in payload["routing"]}

    assert payload["selected_wave"] == ["Rd", "Ri1", "Ri2"]
    assert payload["dispatch_refs"] == ["Rd", "Ri1", "Ri2"]
    assert routes["Rd"]["decision"] == "direct"
    assert routes["Rd"]["effective_role_id"] == "zpy"
    assert routes["Rd"]["agent"] is None
    assert routes["Ri1"]["decision"] == "dispatch"
    assert routes["Ri1"]["agent"] is not None
    assert routes["Ri2"]["decision"] == "dispatch"
    assert routes["Ri2"]["agent"] is not None

    host_batch_refs = [
        ref
        for ref in payload["dispatch_refs"]
        if routes[ref]["decision"] == "dispatch" and routes[ref]["agent"] is not None
    ]
    assert host_batch_refs == ["Ri1", "Ri2"]


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


def test_apply_remaining_is_per_ref_and_limits_dispatch_without_truncating_wave(
    tmp_path: Path,
) -> None:
    write_apply_wave_contract(tmp_path)
    initial = run_loop_direct(tmp_path, "plan", "demo")
    assert initial["selected_wave"] == ["R1", "R2", "R3"]

    loop_path = tmp_path / "openspec" / "changes" / "demo" / "loop.json"
    config = json.loads(loop_path.read_text(encoding="utf-8"))
    config["budgets"]["revision"]["max_iterations"] = 1
    config["budgets"]["change"]["max_iterations"] = 1
    loop_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    limited = run_loop_direct(tmp_path, "plan", "demo")
    assert limited["selected_wave"] == ["R1", "R2", "R3"]
    assert limited["apply_remaining"] == 3
    assert limited["allowed_parallel_applies"] == 3
    assert limited["dispatch_refs"] == ["R1", "R2", "R3"]
    assert "revision_apply_iterations_remaining" not in limited
    assert "change_apply_iterations_remaining" not in limited
    cleaned = read_config(tmp_path)
    assert "max_iterations" not in cleaned["budgets"]["revision"]
    assert "max_iterations" not in cleaned["budgets"]["change"]

    for _ in range(2):
        seed_legacy_apply_fixture(
            tmp_path,
            ("record", "demo", "--ref", "R1", "--kind", "apply", "--result", "completed"),
        )
    exhausted = run_loop_direct(tmp_path, "plan", "demo")
    assert exhausted["selected_wave"] == ["R1", "R2", "R3"]
    assert exhausted["apply_remaining"] == len(exhausted["dispatch_refs"])
    assert exhausted["allowed_parallel_applies"] == 2
    assert exhausted["dispatch_refs"] == ["R2", "R3"]
    assert "revision_apply_iterations_remaining" not in exhausted
    assert "change_apply_iterations_remaining" not in exhausted


def test_apply_identity_is_one_supervisor_record_per_ref_attempt(
    tmp_path: Path,
) -> None:
    write_apply_wave_contract(tmp_path, count=2)
    run_loop_direct(tmp_path, "plan", "demo")
    receipts = issue_apply_receipts(tmp_path, "identity-run")
    first = receipts["R1"]
    common = (
        "record",
        "demo",
        "--run-id",
        "identity-run",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "completed",
        "--receipt",
        first["token"],
        "--attempt-id",
        first["attempt_id"],
        "--changed-file",
        "src/ref1.py",
        "--evidence",
        "inspect:R1",
    )
    recorded = run_loop_direct(tmp_path, *common)
    assert recorded["attempt_id"] == first["attempt_id"]
    assert recorded["apply_record_owner"] == "supervisor"
    duplicate = run_loop_direct(tmp_path, *common, expected_exit=2)
    assert "receipt claim is not available" in duplicate["error"]

    second = receipts["R2"]
    worker = run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "identity-run",
        "--ref",
        "R2",
        "--kind",
        "apply",
        "--result",
        "failed",
        "--receipt",
        second["token"],
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
    assert "apply_record_requires_receipt" in combined["error"]
    role_drift = run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "identity-run",
        "--ref",
        "R2",
        "--kind",
        "apply",
        "--result",
        "failed",
        "--receipt",
        second["token"],
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
    receipts = issue_apply_receipts(tmp_path, "role-write")

    def record(ref: str, role: str, changed: str, exit_code: int = 0) -> dict:
        receipt = receipts[ref]
        return run_loop_direct(
            tmp_path,
            "record",
            "demo",
            "--run-id",
            "role-write",
            "--ref",
            ref,
            "--kind",
            "apply",
            "--result",
            "completed",
            "--receipt",
            receipt["token"],
            "--role-id",
            role,
            "--changed-file",
            changed,
            "--evidence",
            f"inspect:{ref}",
            expected_exit=exit_code,
        )

    assert "write_scope_violation" in record(
        "R1", "implementer", "src/other.py", 2
    )["error"]
    assert record("R1", "implementer", "src/pkg/core.py")["attempt_id"] == receipts["R1"]["attempt_id"]
    assert "tests_only" in record(
        "R2", "test-engineer", "src/test_support/data.txt", 2
    )["error"]
    assert record("R2", "test-engineer", "tests/test_core.py")["attempt_id"] == receipts["R2"]["attempt_id"]
    assert "evidence_root" in record(
        "R3", "browser-qa-runner", "outputs/browser.json", 2
    )["error"]
    assert record(
        "R3", "browser-qa-runner", "auto_test_openspec/demo/browser.json"
    )["attempt_id"] == receipts["R3"]["attempt_id"]
    assert "R4" not in receipts


def test_shared_wave_join_blocks_early_verify_but_failed_sibling_is_local(
    tmp_path: Path,
) -> None:
    write_apply_wave_contract(tmp_path, count=2)
    plan = run_loop_direct(tmp_path, "plan", "demo")
    join_id = plan["routing"][0]["join_id"]
    receipts = issue_apply_receipts(tmp_path, "wave-run")

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
        "--receipt",
        receipts["R1"]["token"],
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
            "--receipt",
            receipts[ref]["token"],
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
    receipt = issue_apply_receipts(tmp_path, "retry-run")["R1"]
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
        "--receipt",
        receipt["token"],
        "--transient-retries",
        "2",
    )
    summary = run_loop_direct(tmp_path, "summary", "demo", "--run-id", "retry-run")
    assert summary["revision_attempt_count"] == 1
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
            "R0",
            "--kind",
            kind,
            "--result",
            result,
            "--subagent-id",
            f"trace-{kind}",
    )
    summary = run_loop_direct(tmp_path, "summary", "demo", "--run-id", "zero-run")
    assert summary["revision_attempt_count"] == 4
    assert summary["change_attempt_count"] == 4
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

    receipt = issue_apply_receipts(tmp_path, "direct-run")["R2"]
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
        "--receipt",
        receipt["token"],
        "--evidence",
        "inspect:R2",
    )
    direct = run_loop_direct(tmp_path, "summary", "demo", "--run-id", "direct-run")
    assert direct["latest_attempt"]["role_id"] == "zpy"
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
    assert read_config(tmp_path)["stamp_state"]["charged_cycle_stamps"] == 1

    tasks_text = (tmp_path / "openspec" / "changes" / "demo" / "tasks.md").read_text(
        encoding="utf-8"
    )
    assert "- [ ] 1.3 re-cover design goal G1 [#R3]" in tasks_text
    assert "  - SUPERSEDES: R2" in tasks_text
    assert "## Design goals" in tasks_text
    assert tasks_text.index("[#R3]") < tasks_text.index("## Design goals")
    assert run_loop(tmp_path, "check", "demo")["ok"] is True


def test_apply_revision_rejects_an_exhausted_cycle_stamp_before_any_write(
    tmp_path: Path,
) -> None:
    seal_revisable_demo(tmp_path, "--autonomy", "full_auto")
    change_dir = tmp_path / "openspec" / "changes" / "demo"
    loop_path = change_dir / "loop.json"
    config = read_config(tmp_path)
    config["stamp_state"]["charged_cycle_stamps"] = 3
    config["stamp_state"]["semantic_stamps_in_chapter"] = 3
    loop_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    proposal = write_proposal(
        tmp_path,
        [
            {
                "action": "add",
                "title": "blocked fourth cycle",
                "depends_on": [],
                "accept": "the manifest remains exact",
                "test": ["python -c \"pass\""],
            }
        ],
    )
    before = {
        name: (change_dir / name).read_bytes()
        for name in ("tasks.md", "feature_list.json", "loop.json")
    }
    refused = run_loop(
        tmp_path,
        "apply-revision",
        "demo",
        "--proposal",
        str(proposal),
        expected_exit=2,
    )
    assert "change_cycle_stamp_budget_exhausted:3" in refused["issues"]
    for name, payload in before.items():
        assert (change_dir / name).read_bytes() == payload


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


def test_apply_revision_ignores_deleted_change_apply_total_keys(tmp_path: Path) -> None:
    ledger = seal_revisable_demo(
        tmp_path,
        "--autonomy",
        "full_auto",
    )
    fingerprint = read_config(tmp_path)["contract_fingerprint"]
    loop_path = tmp_path / "openspec" / "changes" / "demo" / "loop.json"
    config = read_config(tmp_path)
    config["budgets"]["change"]["max_iterations"] = 2
    config.setdefault("hard_ceiling", {})["max_iterations"] = 2
    loop_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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

    applied = run_loop(
        tmp_path,
        "apply-revision",
        "demo",
        "--proposal",
        str(proposal),
    )

    assert applied["applied"] is True
    assert not any("max_iterations_reached" in issue for issue in applied["issues"])
    assert (change_dir / "tasks.md").read_bytes() != before
    cleaned = read_config(tmp_path)
    assert "max_iterations" not in cleaned["budgets"]["change"]
    assert "max_iterations" not in cleaned.get("hard_ceiling", {})


def write_large_registry(tmp_path: Path, task_count: int) -> None:
    lines = ["## Active Task Registry", ""]
    entries = []
    for index in range(1, task_count + 1):
        lines.append(f"- [ ] 1.{index} Task {index} [#R{index}]")
        lines.append("  - INDEPENDENT: yes")
        entries.append((f"R{index}", f"1.{index}", False, False))
    write_contract(tmp_path, "demo", "\n".join(lines) + "\n", base_features(entries))


def test_first_seal_scales_change_active_minutes_to_task_count(tmp_path: Path) -> None:
    write_large_registry(tmp_path, 50)
    seal_demo(tmp_path, tmp_path / "test_cache" / "demo" / "loop" / "ledger.json")

    config = read_config(tmp_path)
    change = config["budgets"]["change"]
    assert change["max_active_minutes"] == 500
    assert change["max_revisions"] == 3
    assert "max_iterations" not in change
    assert "hard_ceiling" not in config
    assert run_loop(tmp_path, "check", "demo")["ok"] is True


def test_first_seal_keeps_a_small_registry_on_the_flat_budget(tmp_path: Path) -> None:
    write_large_registry(tmp_path, 3)
    seal_demo(tmp_path, tmp_path / "test_cache" / "demo" / "loop" / "ledger.json")

    change = read_config(tmp_path)["budgets"]["change"]
    assert change["max_active_minutes"] == 360


def test_a_confirmed_change_active_minutes_budget_is_never_rescaled(tmp_path: Path) -> None:
    write_large_registry(tmp_path, 50)
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    seal_demo(tmp_path, ledger, "--max-total-active-minutes", "5")
    assert read_config(tmp_path)["budgets"]["change"]["max_active_minutes"] == 5

    run_loop(tmp_path, "seal", "demo", "--confirmed")
    assert read_config(tmp_path)["budgets"]["change"]["max_active_minutes"] == 5


def test_semantic_seal_no_longer_reports_task_derived_apply_shortfall(
    tmp_path: Path,
) -> None:
    write_large_registry(tmp_path, 12)
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    seal_demo(tmp_path, ledger, "--max-total-active-minutes", "24")

    write_large_registry(tmp_path, 19)
    restamped = run_loop(
        tmp_path,
        "reseal",
        "demo",
        "--allow-semantic-change",
        "--confirmed",
        "--reason",
        "chapter-outside registry growth keeps the prior runtime policy",
    )

    assert restamped["semantic_change"] is True
    assert restamped["budget_advisories"] == []
    assert read_config(tmp_path)["budgets"]["change"]["max_active_minutes"] == 24


def test_semantic_reseal_no_longer_reports_apply_shortfall_advisories(
    tmp_path: Path,
) -> None:
    write_large_registry(tmp_path, 12)
    ledger = tmp_path / "test_cache" / "demo" / "loop" / "ledger.json"
    seal_demo(tmp_path, ledger, "--max-total-active-minutes", "24")

    write_large_registry(tmp_path, 19)
    restamped = run_loop(
        tmp_path,
        "reseal",
        "demo",
        "--allow-semantic-change",
        "--confirmed",
    )

    assert restamped["semantic_change"] is True
    assert restamped["budget_advisories"] == []
    assert read_config(tmp_path)["budgets"]["change"]["max_active_minutes"] == 24


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
    seal_demo(tmp_path, ledger, "--max-total-active-minutes", "24")

    write_mixed_registry(tmp_path, passed=13, pending=6)
    restamped = run_loop(
        tmp_path,
        "reseal",
        "demo",
        "--allow-semantic-change",
        "--confirmed",
        "--reason",
        "chapter-outside mixed registry restamp",
    )

    assert restamped["semantic_change"] is True
    assert restamped["budget_advisories"] == []
    assert read_config(tmp_path)["budgets"]["change"]["max_active_minutes"] == 24


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


def test_apply_record_without_receipt_fails_before_runtime_or_ledger_write(
    tmp_path: Path,
) -> None:
    write_contract(
        tmp_path,
        "demo",
        "## Active Task Registry\n\n- [ ] 1.1 Runtime task [#R1]\n",
        base_features([("R1", "1.1", False, False)]),
    )
    write_thin_retention_decision(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--repo-root",
            str(tmp_path),
            "record",
            "demo",
            "--ref",
            "R1",
            "--kind",
            "apply",
            "--result",
            "completed",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 2
    assert json.loads(result.stdout)["error"] == "apply_record_requires_receipt"
    assert not (tmp_path / "openspec" / "changes" / "demo" / "loop.json").exists()
    assert not (tmp_path / "test_cache").exists()


def test_next_shadow_matches_legacy_plan_and_emits_receipts(tmp_path: Path) -> None:
    write_apply_wave_contract(tmp_path, count=2)
    legacy = run_loop_direct(tmp_path, "plan", "demo")

    shadow = run_loop_direct(
        tmp_path,
        "next",
        "demo",
        "--intent",
        "drain",
        "--run-id",
        "shadow-run",
        "--shadow",
    )

    assert shadow["schema_version"] == "openspec-loop-next.v1"
    assert shadow["shadow"] is True
    assert shadow["next_action"] == "apply"
    assert shadow["selected_batch"] == legacy["selected_batch"]
    assert shadow["selected_wave"] == legacy["selected_wave"]
    assert shadow["dispatch_refs"] == legacy["dispatch_refs"]
    assert shadow["shadow_parity"]["matches_legacy"] is True
    assert shadow["ablation"]["census_calls"] == 1
    assert shadow["ablation"]["routing_source"] == "cli"
    assert len(shadow["receipts"]) == 2

    assert shadow["receipts"][0]["token"] is None
    assert shadow["receipts"][0]["authoritative"] is False
    assert "src/ref1.py" not in json.dumps(shadow)


def test_next_shadow_is_read_only_before_loop_initialization(tmp_path: Path) -> None:
    write_apply_wave_contract(tmp_path, count=1)
    loop_path = tmp_path / "openspec" / "changes" / "demo" / "loop.json"

    shadow = run_loop_direct(
        tmp_path,
        "next",
        "demo",
        "--intent",
        "drain",
        "--run-id",
        "read-only-shadow",
        "--shadow",
    )

    assert shadow["next_action"] == "interview"
    assert shadow["receipts"] == []
    assert not loop_path.exists()


def test_next_shadow_does_not_persist_legacy_sanitization(tmp_path: Path) -> None:
    write_apply_wave_contract(tmp_path, count=1)
    run_loop_direct(tmp_path, "plan", "demo")
    loop_path = tmp_path / "openspec" / "changes" / "demo" / "loop.json"
    config = json.loads(loop_path.read_text(encoding="utf-8"))
    config["budgets"]["revision"]["max_iterations"] = 99
    loop_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    before = loop_path.read_bytes()

    run_loop_direct(
        tmp_path,
        "next",
        "demo",
        "--intent",
        "drain",
        "--run-id",
        "sanitization-shadow",
        "--shadow",
    )

    assert loop_path.read_bytes() == before


def test_next_preserves_explicit_review_intent_without_forcing_apply(tmp_path: Path) -> None:
    write_apply_wave_contract(tmp_path, count=1)
    run_loop_direct(tmp_path, "plan", "demo")

    reviewed = run_loop_direct(
        tmp_path,
        "next",
        "demo",
        "--intent",
        "review",
        "--run-id",
        "review-run",
    )

    assert reviewed["next_action"] == "review"
    assert reviewed["dispatch_refs"] == ["R1"]
    assert reviewed["receipts"] == []


def test_receipt_record_skips_nested_census_and_materializes_join(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    write_apply_wave_contract(tmp_path, count=1)
    run_loop_direct(tmp_path, "plan", "demo")
    next_payload = run_loop_direct(
        tmp_path,
        "next",
        "demo",
        "--intent",
        "drain",
        "--run-id",
        "receipt-run",
    )
    receipt = next_payload["receipts"][0]
    assert receipt["token"] == receipt["receipt_id"]
    assert len(receipt["token"]) == 20
    assert "src/ref1.py" not in json.dumps(next_payload)
    issued = run_loop_direct(
        tmp_path,
        "receipt-check",
        "demo",
        "--ref",
        "R1",
        "--receipt",
        receipt["token"],
    )
    assert issued["authoritative"] is True
    assert issued["claim_state"] == "issued"
    outside = run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "receipt-run",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "completed",
        "--receipt",
        receipt["token"],
        "--changed-file",
        "src/outside.py",
        "--evidence",
        "inspect:R1",
        expected_exit=2,
    )
    assert "write_scope_violation" in outside["error"]
    assert run_loop_direct(
        tmp_path,
        "receipt-check",
        "demo",
        "--receipt",
        receipt["token"],
    )["claim_state"] == "issued"
    loop = load_loop_module()

    def refuse(*_args, **_kwargs):
        raise AssertionError("receipt record must not rebuild the plan")

    monkeypatch.setattr(loop, "build_plan_payload", refuse)
    args = loop.build_parser().parse_args(
        [
            "--repo-root",
            str(tmp_path),
            "record",
            "demo",
            "--run-id",
            "receipt-run",
            "--ref",
            "R1",
            "--kind",
            "apply",
            "--result",
            "completed",
            "--receipt",
            receipt["token"],
            "--changed-file",
            "src/ref1.py",
            "--evidence",
            "inspect:R1",
        ]
    )

    assert args.func(args) == 0
    recorded = json.loads(capsys.readouterr().out)
    assert recorded["receipt_id"] == receipt["receipt_id"]
    assert recorded["join"]["state"] == "closed"
    assert recorded["join"]["verify_eligible"] is True
    consumed = run_loop_direct(
        tmp_path,
        "receipt-check",
        "demo",
        "--ref",
        "R1",
        "--receipt",
        receipt["token"],
    )
    assert consumed["authoritative"] is False
    assert consumed["claim_state"] == "consumed"
    replay = run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "receipt-run",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "completed",
        "--receipt",
        receipt["token"],
        expected_exit=2,
    )
    assert "receipt claim is not available" in replay["error"]
    summary = run_loop_direct(
        tmp_path, "summary", "demo", "--run-id", "receipt-run"
    )
    assert summary["ablation"]["control_hop_count"] == 4
    assert summary["ablation"]["cli_control_call_count"] == 4
    assert summary["ablation"]["llm_control_call_count"] == 0
    assert summary["ablation"]["llm_control_visibility"] is False


def test_receipt_join_binds_only_the_actual_dispatch_wave(tmp_path: Path) -> None:
    write_apply_wave_contract(tmp_path, count=3)
    run_loop_direct(tmp_path, "plan", "demo")
    for _ in (1, 2):
        seed_legacy_apply_fixture(
            tmp_path,
            (
                "record",
                "demo",
                "--run-id",
                "legacy-exhaustion",
                "--ref",
                "R1",
                "--kind",
                "apply",
                "--result",
                "completed",
            ),
        )
    limited = run_loop_direct(tmp_path, "plan", "demo")
    assert limited["selected_wave"] == ["R1", "R2", "R3"]
    assert limited["dispatch_refs"] == ["R2", "R3"]
    module = load_loop_module()
    receipts = module.build_apply_receipts(
        limited, run_id="receipt-wave", shadow=True
    )
    decoded = [item["snapshot"] for item in receipts]
    assert [item["wave_refs"] for item in decoded] == [["R2", "R3"], ["R2", "R3"]]


def test_receipt_fails_closed_after_contract_or_harness_drift(tmp_path: Path) -> None:
    write_apply_wave_contract(tmp_path, count=1)
    run_loop_direct(tmp_path, "plan", "demo")
    payload = run_loop_direct(
        tmp_path,
        "next",
        "demo",
        "--intent",
        "drain",
        "--run-id",
        "drift-run",
    )
    token = payload["receipts"][0]["token"]
    tasks_path = tmp_path / "openspec" / "changes" / "demo" / "tasks.md"
    tasks_path.write_text(
        tasks_path.read_text(encoding="utf-8").replace("Apply ref 1", "Changed ref 1"),
        encoding="utf-8",
    )

    refused = run_loop_direct(
        tmp_path,
        "receipt-check",
        "demo",
        "--ref",
        "R1",
        "--receipt",
        token,
        expected_exit=2,
    )
    assert refused["valid"] is False
    assert "contract fingerprint drift" in refused["issues"]


def test_shadow_receipt_cannot_authorize_an_apply_record(tmp_path: Path) -> None:
    write_apply_wave_contract(tmp_path, count=1)
    run_loop_direct(tmp_path, "plan", "demo")
    shadow = run_loop_direct(
        tmp_path,
        "next",
        "demo",
        "--intent",
        "drain",
        "--run-id",
        "shadow-authority",
        "--shadow",
    )

    refused = run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "shadow-authority",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "completed",
        "--receipt",
        shadow["receipts"][0]["receipt_id"],
        expected_exit=2,
    )
    assert "receipt was not issued by active next" in refused["error"]


def write_mechanical_verify_contract(
    tmp_path: Path,
    level: str = "smoke",
    *,
    scope: str = "CLI",
    include_run: bool = True,
    command: str = "python -c \"print('mechanical-ok')\"",
) -> None:
    run_line = f"    - Run: `{command}`\n" if include_run else ""
    tasks = f'''## Active Task Registry

- [ ] 1.1 Verify command [#R1]
  - INDEPENDENT: yes
  - FILES: `src/ref1.py`
  - WRITE_SCOPE: `src/ref1.py`
  - TEST_LEVEL: {level}
  - ACCEPT: command exits successfully without semantic deviation
  - TEST: SCOPE: {scope}
{run_line}'''
    write_contract(
        tmp_path,
        "demo",
        tasks,
        base_features([("R1", "1.1", False, False)]),
    )
    feature_payload = load_loop_module().build_feature_payload_from_tasks_text(
        tmp_path, "demo", tasks
    )
    (tmp_path / "openspec" / "changes" / "demo" / "feature_list.json").write_text(
        json.dumps(feature_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_thin_retention_decision(tmp_path)


def test_test_level_round_trips_and_mechanical_verify_is_typed(tmp_path: Path) -> None:
    write_mechanical_verify_contract(tmp_path, level="smoke")
    planned = run_loop_direct(tmp_path, "plan", "demo")
    assert planned["tasks"][0]["test_level"] == "smoke"
    generated = json.loads(
        (tmp_path / "openspec" / "changes" / "demo" / "feature_list.json").read_text(
            encoding="utf-8"
        )
    )
    assert generated["features"]["R1"]["test_level"] == "smoke"

    next_payload = run_loop_direct(
        tmp_path,
        "next",
        "demo",
        "--intent",
        "drain",
        "--run-id",
        "mechanical-run",
    )
    receipt = next_payload["receipts"][0]["token"]
    run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "mechanical-run",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "completed",
        "--receipt",
        receipt,
        "--changed-file",
        "src/ref1.py",
        "--evidence",
        "inspect:R1",
    )

    before_verify = run_loop_direct(
        tmp_path,
        "next",
        "demo",
        "--intent",
        "drain",
        "--run-id",
        "mechanical-run",
        "--shadow",
    )
    assert before_verify["next_action"] == "verify_mechanical"

    verified = run_loop_direct(
        tmp_path,
        "verify-mechanical",
        "demo",
        "--run-id",
        "mechanical-run",
        "--ref",
        "R1",
    )
    assert verified["schema_version"] == "openspec-loop-mechanical-verify.v1"
    assert verified["verdict"] == "PASS"
    assert verified["semantic_verify_required"] is False
    assert verified["commands"][0]["exit_code"] == 0
    assert "mechanical-ok" not in json.dumps(verified)

    next_after = run_loop_direct(
        tmp_path,
        "next",
        "demo",
        "--intent",
        "drain",
        "--run-id",
        "mechanical-run",
        "--shadow",
    )
    assert next_after["next_action"] == "promote"
    summary = run_loop_direct(
        tmp_path,
        "summary",
        "demo",
        "--run-id",
        "mechanical-run",
    )
    assert summary["ablation"]["next_call_count"] == 1
    assert summary["ablation"]["census_count"] == 1
    assert summary["ablation"]["recorded_hop_count"] == 5
    assert summary["ablation"]["control_hop_count"] == 3
    assert summary["ablation"]["work_hop_count"] == 2
    assert summary["ablation"]["cli_control_call_count"] == 3
    assert summary["ablation"]["llm_control_call_count"] == 0
    assert summary["ablation"]["llm_control_visibility"] is False
    assert summary["per_ref"]["R1"]["next_call_count"] == 1
    assert summary["per_ref"]["R1"]["recorded_hop_count"] == 3


def test_pilot_mechanical_pass_routes_to_semantic_verify(tmp_path: Path) -> None:
    write_mechanical_verify_contract(tmp_path, level="pilot")
    run_loop_direct(tmp_path, "plan", "demo")
    next_payload = run_loop_direct(
        tmp_path,
        "next",
        "demo",
        "--intent",
        "drain",
        "--run-id",
        "pilot-run",
    )
    run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "pilot-run",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "completed",
        "--receipt",
        next_payload["receipts"][0]["token"],
        "--changed-file",
        "src/ref1.py",
        "--evidence",
        "inspect:R1",
    )
    verified = run_loop_direct(
        tmp_path,
        "verify-mechanical",
        "demo",
        "--run-id",
        "pilot-run",
        "--ref",
        "R1",
    )
    assert verified["verdict"] == "PASS"
    assert verified["semantic_verify_required"] is True

    next_after = run_loop_direct(
        tmp_path,
        "next",
        "demo",
        "--intent",
        "drain",
        "--run-id",
        "pilot-run",
        "--shadow",
    )
    assert next_after["next_action"] == "verify_semantic"


def test_non_cli_mechanical_stage_skips_to_semantic_verify(tmp_path: Path) -> None:
    write_mechanical_verify_contract(
        tmp_path, level="production", scope="GUI", include_run=False
    )
    run_loop_direct(tmp_path, "plan", "demo")
    next_payload = run_loop_direct(
        tmp_path,
        "next",
        "demo",
        "--intent",
        "drain",
        "--run-id",
        "gui-run",
    )
    run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "gui-run",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "completed",
        "--receipt",
        next_payload["receipts"][0]["token"],
        "--changed-file",
        "src/ref1.py",
        "--evidence",
        "inspect:R1",
    )
    verified = run_loop_direct(
        tmp_path,
        "verify-mechanical",
        "demo",
        "--run-id",
        "gui-run",
        "--ref",
        "R1",
    )
    assert verified["verdict"] == "SKIPPED"
    assert verified["test_scope"] == "GUI"
    assert verified["semantic_verify_required"] is True


def test_mechanical_verify_blocks_separately_authorized_mutation(tmp_path: Path) -> None:
    write_mechanical_verify_contract(
        tmp_path,
        command="python -c \"print('x')\" --commit",
    )
    run_loop_direct(tmp_path, "plan", "demo")
    next_payload = run_loop_direct(
        tmp_path,
        "next",
        "demo",
        "--intent",
        "drain",
        "--run-id",
        "forbidden-run",
    )
    run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "forbidden-run",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "completed",
        "--receipt",
        next_payload["receipts"][0]["token"],
        "--changed-file",
        "src/ref1.py",
        "--evidence",
        "inspect:R1",
    )

    blocked = run_loop_direct(
        tmp_path,
        "verify-mechanical",
        "demo",
        "--run-id",
        "forbidden-run",
        "--ref",
        "R1",
        expected_exit=2,
    )
    assert blocked["verdict"] == "BLOCKED"
    assert "separately human-authorized" in blocked["issues"][0]


def test_summary_reports_ablation_false_success_and_misroute(tmp_path: Path) -> None:
    write_apply_wave_contract(tmp_path, count=1)
    run_loop_direct(tmp_path, "plan", "demo")
    run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "ablation-run",
        "--ref",
        "R1",
        "--kind",
        "apply",
        "--result",
        "completed",
        "--routing-outcome",
        "misrouted",
    )
    run_loop_direct(
        tmp_path,
        "record",
        "demo",
        "--run-id",
        "ablation-run",
        "--ref",
        "R1",
        "--kind",
        "verify",
        "--result",
        "deviated",
    )

    summary = run_loop_direct(
        tmp_path,
        "summary",
        "demo",
        "--run-id",
        "ablation-run",
    )
    assert summary["ablation"]["misroute_count"] == 1
    assert summary["ablation"]["false_success_count"] == 1
    assert summary["ablation"]["semantic_deviation_count"] == 1


def test_ablation_evaluate_reports_labeled_detection_and_consistency() -> None:
    report = run_loop_direct(
        Path.cwd(),
        "ablation-evaluate",
        "--manifest",
        str(
            Path(__file__).parent
            / "fixtures"
            / "openspec_loop"
            / "ablation-suite.v1.json"
        ),
    )

    assert report["schema_version"] == "openspec-loop-ablation-report.v1"
    assert report["case_count"] == 4
    assert report["control"]["control_hop_count"] == 5
    assert report["control"]["work_hop_count"] == 8
    assert report["control"]["cli_control_call_count"] == 4
    assert report["control"]["llm_control_call_count"] == 1
    assert report["control"]["llm_control_visibility"] is True
    assert report["deviation_detection"] == {
        "true_positive": 1,
        "false_negative": 1,
        "false_positive": 0,
        "true_negative": 2,
        "detection_rate": 0.5,
        "false_positive_rate": 0.0,
    }
    consistency = report["same_accept_consistency"]
    assert consistency["eligible_group_count"] == 2
    assert consistency["consistent_group_count"] == 1
    assert consistency["consistency_rate"] == 0.5
    assert consistency["conflicts"][0]["case_ids"] == [
        "deviation-detected",
        "deviation-missed",
    ]


def test_ablation_evaluate_uses_null_without_positive_or_repeat_group(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "suite.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "openspec-loop-ablation-suite.v1",
                "suite_id": "null-boundaries",
                "cases": [
                    {
                        "case_id": "only-pass",
                        "ref": "R1",
                        "accept_hash": "a" * 64,
                        "input_fingerprint": "b" * 64,
                        "expected_verdict": "PASS",
                        "observed_verdict": "PASS",
                        "evidence_refs": ["fixture:only-pass"],
                        "events": [
                            {
                                "seq": 1,
                                "hop_class": "control",
                                "executor": "cli",
                                "kind": "next",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_loop_direct(
        tmp_path, "ablation-evaluate", "--manifest", str(manifest)
    )
    assert report["deviation_detection"]["detection_rate"] is None
    assert report["same_accept_consistency"]["consistency_rate"] is None


def test_ablation_evaluate_rejects_invalid_or_unverifiable_cases(
    tmp_path: Path,
) -> None:
    base_case = {
        "case_id": "case-a",
        "ref": "R1",
        "accept_hash": "a" * 64,
        "input_fingerprint": "b" * 64,
        "expected_verdict": "PASS",
        "observed_verdict": "PASS",
        "evidence_refs": ["fixture:case-a"],
        "events": [
            {
                "seq": 1,
                "hop_class": "control",
                "executor": "cli",
                "kind": "next",
            }
        ],
    }
    invalid_cases = [
        [{**base_case, "accept_hash": "bad"}],
        [{**base_case, "evidence_refs": []}],
        [{**base_case, "events": [{**base_case["events"][0], "seq": 2}]}],
        [{**base_case, "events": [{**base_case["events"][0], "executor": "human"}]}],
        [base_case, dict(base_case)],
    ]
    for index, cases in enumerate(invalid_cases):
        manifest = tmp_path / f"invalid-{index}.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "openspec-loop-ablation-suite.v1",
                    "suite_id": f"invalid-{index}",
                    "cases": cases,
                }
            ),
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--repo-root",
                str(tmp_path),
                "ablation-evaluate",
                "--manifest",
                str(manifest),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert result.returncode == 2
        assert json.loads(result.stdout)["error"].startswith("ablation")

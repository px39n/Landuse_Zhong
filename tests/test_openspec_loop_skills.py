from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover - minimal environments use structural checks
    Draft202012Validator = None


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = REPO_ROOT / ".agents" / "skills"
MIRROR_ROOTS = (
    REPO_ROOT / ".codex" / "skills",
    REPO_ROOT / ".claude" / "skills",
)
SKILL_NAMES = (
    "openspec-loop-engineering",
    "openspec-change-interviewer",
    "openspec-explore",
    "openspec-apply-change",
    "openspec-verify-change",
    "openspec-unblock-research",
    "openspec-new-change",
    "openspec-continue-change",
    "openspec-ff-change",
    "openspec-omx-bridge",
    "openspec-feature-list",
    "openspec-hygiene",
    "silent-failure-hunting",
    "review-pipeline",
    "monitor-openspec-codex",
)
CURSOR_WRAPPERS = {
    "openspec-loop-engineering.md": "openspec-loop-engineering",
    "openspec-change-interviewer.md": "openspec-change-interviewer",
    "openspec-explore.md": "openspec-explore",
    "openspec-apply-change.md": "openspec-apply-change",
    "openspec-verify-change.md": "openspec-verify-change",
    "openspec-unblock-research.md": "openspec-unblock-research",
    "openspec-feature-list.md": "openspec-feature-list",
    "openspec-new-change.md": "openspec-new-change",
    "openspec-continue-change.md": "openspec-continue-change",
    "openspec-ff-change.md": "openspec-ff-change",
    "openspec-omx-bridge.md": "openspec-omx-bridge",
    "monitor-openspec-codex.md": "monitor-openspec-codex",
}
CLAUDE_WRAPPERS = {
    "opsx/loop.md": "openspec-loop-engineering",
    "opsx/interview.md": "openspec-change-interviewer",
    "opsx/explore.md": "openspec-explore",
    "opsx/apply.md": "openspec-apply-change",
    "opsx/verify.md": "openspec-verify-change",
    "opsx/unblock.md": "openspec-unblock-research",
    "opsx/feature-list.md": "openspec-feature-list",
    "opsx/new.md": "openspec-new-change",
    "opsx/continue.md": "openspec-continue-change",
    "opsx/ff.md": "openspec-ff-change",
    "opsx/omx-bridge.md": "openspec-omx-bridge",
    "monitor-openspec-codex.md": "monitor-openspec-codex",
}
MOJIBAKE_TOKENS = ("涓", "鍩", "鈥", "\ufffd")


def canonical_files(name: str) -> dict[Path, bytes]:
    root = CANONICAL_ROOT / name
    files = {Path("SKILL.md"): (root / "SKILL.md").read_bytes()}
    references = root / "references"
    if references.is_dir():
        for path in sorted(references.rglob("*")):
            if path.is_file():
                files[Path("references") / path.relative_to(references)] = (
                    path.read_bytes()
                )
    return files


def read_skill(name: str) -> str:
    return (CANONICAL_ROOT / name / "SKILL.md").read_text(encoding="utf-8")


def test_agents_is_the_byte_exact_semantic_source() -> None:
    for name in SKILL_NAMES:
        expected = canonical_files(name)
        for mirror_root in MIRROR_ROOTS:
            for relative, content in expected.items():
                assert (mirror_root / name / relative).read_bytes() == content


def test_sync_check_reports_no_drift() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "sync_openspec_loop_skills.py"),
            "--check",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_platform_metadata_is_not_mirrored() -> None:
    metadata = (
        REPO_ROOT
        / ".codex"
        / "skills"
        / "openspec-loop-engineering"
        / "agents"
        / "openai.yaml"
    )
    assert metadata.is_file()
    assert not (
        CANONICAL_ROOT / "openspec-loop-engineering" / "agents" / "openai.yaml"
    ).exists()
    assert not (
        REPO_ROOT
        / ".claude"
        / "skills"
        / "openspec-loop-engineering"
        / "agents"
        / "openai.yaml"
    ).exists()


def assert_thin_wrapper(path: Path, skill_name: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert f".agents/skills/{skill_name}/SKILL.md" in text
    assert len(text.splitlines()) <= 12
    assert "thin router" in text.lower() or skill_name == "monitor-openspec-codex"


def test_cursor_and_claude_commands_are_thin_routers() -> None:
    for relative, skill_name in CURSOR_WRAPPERS.items():
        assert_thin_wrapper(REPO_ROOT / ".cursor" / "commands" / relative, skill_name)
    claude_root = REPO_ROOT / ".claude" / "commands"
    for relative, skill_name in CLAUDE_WRAPPERS.items():
        assert_thin_wrapper(claude_root / relative, skill_name)


def test_skills_have_valid_minimal_frontmatter() -> None:
    for name in SKILL_NAMES:
        text = read_skill(name)
        assert text.startswith("---\n")
        frontmatter = text.split("---", 2)[1]
        assert f"name: {name}" in frontmatter
        assert "description:" in frontmatter
        assert "compatibility:" not in frontmatter


def test_loop_lifecycle_and_budget_contract_is_explicit() -> None:
    loop = read_skill("openspec-loop-engineering")
    verify = read_skill("openspec-verify-change")
    unblock = read_skill("openspec-unblock-research")
    apply = read_skill("openspec-apply-change")
    operator = (
        CANONICAL_ROOT
        / "openspec-loop-engineering"
        / "references"
        / "operator-guide.md"
    ).read_text(encoding="utf-8")
    nl = chr(10)
    combined = " ".join(nl.join((loop, verify, unblock, apply, operator)).split())

    for phrase in (
        "--intent",
        "needs_user",
        "answer --token",
        "record --kind apply",
        "verify-mechanical",
        "PASS|FAIL|BLOCKED|DEVIATED",
        "retry | targeted_probe | amend_spec",
        "Apply=2",
        "Unblock=2",
        "revisions=3",
    ):
        assert phrase.lower() in combined.lower()
    assert "DEVIATED" in verify and "DEVIATED" in unblock and "DEVIATED" in apply
    assert "operator-guide.md" in loop
    assert "host-adapter.md" in loop
    assert "receipt-check" not in loop



def test_unblock_host_policy_is_in_process_and_ref_local() -> None:
    unblock = read_skill("openspec-unblock-research")
    disposition = (
        CANONICAL_ROOT
        / "openspec-unblock-research"
        / "references"
        / "loop-disposition-gate.md"
    ).read_text(encoding="utf-8")
    operator = (
        CANONICAL_ROOT
        / "openspec-loop-engineering"
        / "references"
        / "operator-guide.md"
    ).read_text(encoding="utf-8")
    nl = chr(10)
    normalized = " ".join((unblock + nl + disposition + nl + operator).split())

    assert "Spawn zero by default" in unblock
    assert "Do not implement the fix" in unblock
    assert "retry | targeted_probe | amend_spec" in unblock
    assert "Second unblock on the same fingerprint" in unblock
    assert "Diagnosis grants no implementation" in disposition
    assert "record --kind unblock" in operator



def test_completion_right_stamp_and_alignment_contract_is_explicit() -> None:
    loop = read_skill("openspec-loop-engineering")
    interview = read_skill("openspec-change-interviewer")
    operator = (
        CANONICAL_ROOT
        / "openspec-loop-engineering"
        / "references"
        / "operator-guide.md"
    ).read_text(encoding="utf-8")
    nl = chr(10)
    joined = nl.join((loop, interview, operator))
    normalized = " ".join(joined.split())

    assert "Only the controller records, promotes, or closes" in loop
    assert "needs_user" in loop
    assert "answer --token" in loop
    assert "Source authoring never grants ledger, receipt, or PASS authority" in interview
    assert "Apply=2" in operator
    assert "Unblock=2" in operator
    assert "revisions=3" in operator
    assert "Root PASS" in operator
    assert "max_revisions=14" not in normalized



def test_seal_preview_is_change_scoped_and_not_yearbook_template() -> None:
    interview = read_skill("openspec-change-interviewer")
    preview = (
        CANONICAL_ROOT
        / "openspec-change-interviewer"
        / "references"
        / "seal-preview-format.md"
    ).read_text(encoding="utf-8")
    placement = (
        CANONICAL_ROOT
        / "openspec-change-interviewer"
        / "references"
        / "artifact-placement.md"
    ).read_text(encoding="utf-8")

    assert "Artifact Retention Decision" in interview
    assert "seal-preview.md" in preview
    assert "advisory" in preview.lower()
    assert "runtime PASS" in preview
    assert "second ledger" in preview
    assert "city_yearbooks" not in preview
    assert "retention" in placement.lower() or "path" in placement.lower()



def test_two_drift_classes_are_documented_with_distinct_recoveries() -> None:
    loop = read_skill("openspec-loop-engineering")
    apply = read_skill("openspec-apply-change")
    interview = read_skill("openspec-change-interviewer")
    operator = (
        CANONICAL_ROOT
        / "openspec-loop-engineering"
        / "references"
        / "operator-guide.md"
    ).read_text(encoding="utf-8")
    nl = chr(10)
    combined = " ".join(nl.join((loop, apply, interview, operator)).split())

    assert "DEVIATED" in apply
    assert "alignment_warnings" in interview
    assert "wrong direction" in apply.lower() or "DEVIATED" in apply
    assert "controller" in combined.lower()
    assert "promote" in loop



def test_one_authority_model_keeps_apply_count_ref_local() -> None:
    loop = read_skill("openspec-loop-engineering")
    interview = read_skill("openspec-change-interviewer")
    apply = read_skill("openspec-apply-change")
    operator = (
        CANONICAL_ROOT
        / "openspec-loop-engineering"
        / "references"
        / "operator-guide.md"
    ).read_text(encoding="utf-8")

    assert "Only the controller records, promotes, or closes" in loop
    assert "human gates" in apply.lower()
    assert "Apply=2" in operator
    assert "Unblock=2" in operator
    assert (
        "per ref" in operator.lower()
        or "per-ref" in operator.lower()
        or "fingerprint" in operator.lower()
    )
    assert "User" in operator
    assert "Source authoring never grants" in interview



def test_the_loop_drains_the_ready_queue_and_promotes_atomically() -> None:
    loop = read_skill("openspec-loop-engineering")
    guide = (REPO_ROOT / "docs" / "openspec-loop-engineering.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join((loop + "\n" + guide).split())

    assert "selected_wave" in normalized
    assert "record --kind apply --receipt" in normalized
    assert "promote" in normalized and "重新 `next`" in normalized
    assert "For the first ready task only" not in normalized


def test_loop_same_turn_execution_latch() -> None:
    loop = read_skill("openspec-loop-engineering")
    apply = read_skill("openspec-apply-change")
    operator = (
        CANONICAL_ROOT
        / "openspec-loop-engineering"
        / "references"
        / "operator-guide.md"
    ).read_text(encoding="utf-8")

    assert "authoritative" in apply.lower()
    assert "receipt" in loop and "receipt" in apply
    assert "Do not reissue work while an earlier writer may still be active" in loop
    assert "Apply requires its issued receipt" in operator
    assert "shadow receipts" not in loop



def test_design_level_closing_is_documented_across_skills_and_guide() -> None:
    loop = read_skill("openspec-loop-engineering")
    verify = read_skill("openspec-verify-change")
    interview = read_skill("openspec-change-interviewer")
    operator = (
        CANONICAL_ROOT
        / "openspec-loop-engineering"
        / "references"
        / "operator-guide.md"
    ).read_text(encoding="utf-8")

    assert "GOAL" in interview
    assert "COVERED_BY" in interview
    assert "endpoint" in loop
    assert "Independent semantic verification" in verify
    assert "Root PASS" in operator



def test_feature_registry_is_compact_and_monitor_is_audit_only() -> None:
    feature = read_skill("openspec-feature-list")
    apply = read_skill("openspec-apply-change")
    monitor = read_skill("monitor-openspec-codex")

    assert "generate_openspec_feature_list.py" in feature
    assert "Does not seal, admit, or promote" in feature
    assert "receipt_id" in apply
    assert "record --kind apply --receipt" in apply
    assert "Legacy Audit Only" in monitor or "legacy" in monitor.lower()
    assert "outside the ordinary OpenSpec Loop" in monitor or "audit" in monitor.lower()



def test_unblock_v2_example_validates_and_v1_remains_readable() -> None:
    reference_root = CANONICAL_ROOT / "openspec-unblock-research" / "references"
    schema = json.loads(
        (reference_root / "portable-unblock-report.v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    examples = (reference_root / "examples.md").read_text(encoding="utf-8")
    unblock = read_skill("openspec-unblock-research")

    assert schema["required"] == ["receipt_ref", "result"]
    assert "status" in schema["properties"]["result"]["required"]
    assert "repair_class" not in schema.get("properties", {})
    assert "disposition" in unblock
    assert "DEVIATED" in examples
    assert (reference_root / "portable-unblock-report.v1.md").is_file()
    assert (reference_root / "portable-unblock-report.v2.md").is_file()



def test_python_310_compatibility_and_no_heavy_contract_appendices() -> None:
    runner = (REPO_ROOT / "scripts" / "openspec_loop.py").read_text(
        encoding="utf-8"
    )
    assert "datetime.UTC" not in runner
    assert "from datetime import UTC" not in runner

    combined = "\n".join(read_skill(name) for name in SKILL_NAMES)
    assert "Move superseded drafts into an appendix" not in combined
    assert "read all available artifacts" not in combined.lower()
    assert "every attempt must create" not in combined.lower()


def test_docs_and_governance_point_to_the_public_guide() -> None:
    guide = (REPO_ROOT / "docs" / "openspec-loop-engineering.md").read_text(
        encoding="utf-8"
    )
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    project = (REPO_ROOT / "openspec" / "project.md").read_text(encoding="utf-8")
    assert "OpenSpec Loop v3" in guide
    assert "docs/openspec-loop-engineering.md" in agents
    assert "docs/openspec-loop-engineering.md" in project
    assert "retention=full" in agents
    assert "retention=full" in project


def test_public_guide_has_atomic_harness_graph_and_tool_flow() -> None:
    guide = (REPO_ROOT / "docs" / "openspec-loop-engineering.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(guide.split())

    assert "### 1.1 `e2e-host` 原子对齐索引" in guide
    assert "### 2.3 Harness graph：workflow、skill 与 host tool 流" in guide
    assert "### 2.4 Skill → tool → artifact 原子调用表" in guide
    assert "```mermaid" in guide
    assert "flowchart TD" in guide
    assert "纯文本等价流" in guide

    for phrase in (
        "loop.json.paths.ledger",
        "test_cache/<change-id>/loop/ledger.json",
        "interview.md",
        "host_batch_refs = dispatch_refs where routing.decision == dispatch and agent != null",
        "record --receipt",
        "verify-mechanical",
        "Task.resume",
        "resume_agent",
        "shared_context_bytes + Σpacket_delta_bytes + Σresult_bytes",
        "return_only",
        "the registry declares no design goal",
        "uncovered_goals=[]",
        "orphan_refs=[]",
    ):
        assert phrase in normalized

    assert "helper 不创建第二个 host-batch runtime object" in normalized
    assert "worker/host 不写它" in guide
    assert "python scripts/openspec_loop.py record <change-id>" in guide
    assert "python scripts/openspec_loop.py design-verify <change-id> --observation" in guide


def test_touched_text_is_utf8_clean() -> None:
    paths = [REPO_ROOT / "docs" / "openspec-loop-engineering.md"]
    for name in SKILL_NAMES:
        paths.extend((CANONICAL_ROOT / name).rglob("*.md"))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert all(token not in text for token in MOJIBAKE_TOKENS), path


def test_repair_r1_details_live_in_reference_spec_and_runtime() -> None:
    loop = read_skill("openspec-loop-engineering")
    unblock = read_skill("openspec-unblock-research")
    disposition = (
        CANONICAL_ROOT
        / "openspec-unblock-research"
        / "references"
        / "loop-disposition-gate.md"
    ).read_text(encoding="utf-8")
    operator = (
        CANONICAL_ROOT
        / "openspec-loop-engineering"
        / "references"
        / "operator-guide.md"
    ).read_text(encoding="utf-8")

    assert "unblock" in loop
    assert "amend_spec" in unblock or "stop_budget" in unblock
    assert "Diagnosis grants no implementation" in disposition
    assert "record --kind unblock" in operator
    assert "repair_r1" not in loop



def test_unblock_v2_repair_class_is_optional_and_fail_closed() -> None:
    root = CANONICAL_ROOT / "openspec-unblock-research" / "references"
    schema = json.loads(
        (root / "portable-unblock-report.v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    disposition = (root / "loop-disposition-gate.md").read_text(encoding="utf-8")
    unblock = read_skill("openspec-unblock-research")

    assert "repair_class" not in schema.get("properties", {})
    assert set(schema["required"]) == {"receipt_ref", "result"}
    assert "current controller authority decides any repair" in disposition
    assert "Do not implement the fix" in unblock



def test_interviewer_creates_the_seven_section_r2_packet_first() -> None:
    interviewer = read_skill("openspec-change-interviewer")
    packet = (
        CANONICAL_ROOT
        / "openspec-change-interviewer"
        / "references"
        / "interview-packet-format.md"
    ).read_text(encoding="utf-8")

    assert "interview.md" in interviewer
    assert "Artifact Retention Decision" in interviewer
    assert "next_action=interview" in packet
    assert "Source authoring does not establish installed/runtime PASS" in packet
    assert "budget_lineage_id" in packet



def test_execution_notebook_contract_keeps_apply_writer_and_verify_reader() -> None:
    apply_skill = read_skill("openspec-apply-change")
    verify_skill = read_skill("openspec-verify-change")
    notebook = (
        CANONICAL_ROOT
        / "openspec-apply-change"
        / "references"
        / "execution-notebook-contract.md"
    ).read_text(encoding="utf-8")

    for tag in ("openspec-params", "openspec-run", "openspec-outputs"):
        assert tag in notebook
    assert "Apply owns execution writes" in notebook
    assert "WRITE_SCOPE" in apply_skill
    assert (
        "Never edit product code" in verify_skill
        or "read-only" in verify_skill.lower()
        or "Independently" in verify_skill
    )
    for field in (
        "schema",
        "exec_block",
        "run_level",
        "source_notebook_sha",
        "canonical_params",
        "params_sha",
        "started_at",
        "ended_at",
        "status",
        "resume_decision",
    ):
        assert f"`{field}`" in notebook or field in notebook



def test_high_frequency_skills_are_thin_compiled_interface_cards() -> None:
    cards = {
        "loop": read_skill("openspec-loop-engineering"),
        "apply": read_skill("openspec-apply-change"),
        "verify": read_skill("openspec-verify-change"),
        "unblock": read_skill("openspec-unblock-research"),
    }

    for name, card in cards.items():
        assert len(card.splitlines()) <= 80, (name, len(card.splitlines()))
    assert "DEVIATED" in cards["apply"]
    assert "DEVIATED" in cards["verify"]
    assert "DEVIATED" in cards["unblock"]
    assert "--intent" in cards["loop"] and "next" in cards["loop"]
    assert "needs_user" in cards["loop"]
    assert "answer --token" in cards["loop"]
    assert "receipt-check" not in cards["loop"]
    assert "WRITE_SCOPE" in cards["apply"]
    assert "Spawn zero" in cards["unblock"]
    assert "NEXT: supervisor_join" not in chr(10).join(cards.values())



def test_interviewer_and_adapter_do_not_duplicate_compiled_routing() -> None:
    interviewer = read_skill("openspec-change-interviewer")
    matrix = (
        CANONICAL_ROOT
        / "openspec-loop-engineering"
        / "references"
        / "role-adapter-matrix.md"
    ).read_text(encoding="utf-8")
    loop = read_skill("openspec-loop-engineering")

    compact = " ".join(interviewer.split())
    assert "Hand off to `openspec-loop-engineering` / `next`" in compact
    assert "Never award runtime PASS" in interviewer or "never grants" in compact.lower()
    assert (
        "deterministic controller alone owns" in matrix.lower()
        or "Deterministic controller alone owns" in matrix
    )
    assert "needs_user" in loop
    assert "next --shadow" not in interviewer



def test_public_guide_describes_machine_v32_and_current_path() -> None:
    guide = (REPO_ROOT / "docs" / "openspec-loop-engineering.md").read_text(
        encoding="utf-8"
    )
    cursor = (
        REPO_ROOT / "docs" / "openspec-loop-engineering-cursor.md"
    ).read_text(encoding="utf-8")

    assert guide.startswith("# OpenSpec Loop v3.2")
    assert cursor.startswith("# OpenSpec Loop v3.2")
    assert "openspec-loop-control.v3.2" in guide
    for reference in (
        "execution-notebook-contract.md",
        "semantic-verification-contract.md",
        "loop-disposition-gate.md",
    ):
        assert reference in guide
    for stale in (
        "NEXT: supervisor_join",
        "fresh gate",
        "先看五个 plan 字段",
    ):
        assert stale not in guide

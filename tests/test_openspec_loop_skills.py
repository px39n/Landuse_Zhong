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
    interview = read_skill("openspec-change-interviewer")
    verify = read_skill("openspec-verify-change")
    unblock = read_skill("openspec-unblock-research")
    disposition = (
        CANONICAL_ROOT
        / "openspec-unblock-research"
        / "references"
        / "loop-disposition-gate.md"
    ).read_text(encoding="utf-8")
    guide = (REPO_ROOT / "docs" / "openspec-loop-engineering.md").read_text(
        encoding="utf-8"
    )
    combined = " ".join(
        "\n".join((loop, interview, verify, unblock, disposition, guide)).split()
    )

    for phrase in (
        "next --intent",
        "receipt-check",
        "PASS|FAIL|BLOCKED|DEVIATED",
        "retry|targeted_probe|amend_spec|supersede_task|stop_budget",
        "max_apply_attempts",
        "max_unblock_runs",
        "cycle stamp",
    ):
        assert phrase.lower() in combined.lower()
    assert "DEVIATED" in loop and "DEVIATED" in verify and "DEVIATED" in unblock
    assert "revision_apply_iterations_used|remaining" not in combined
    assert "change_apply_iterations_used|remaining" not in combined


def test_unblock_host_policy_is_in_process_and_ref_local() -> None:
    unblock = read_skill("openspec-unblock-research")
    disposition = (
        CANONICAL_ROOT
        / "openspec-unblock-research"
        / "references"
        / "loop-disposition-gate.md"
    ).read_text(encoding="utf-8")
    normalized = " ".join((unblock + "\n" + disposition).split())

    assert "supervisor invokes this skill in-process" in normalized
    assert "do not implement" in normalized
    assert "resume a failed task/thread" in normalized
    assert "research swarm" in normalized
    assert "two Unblock runs per ref" in normalized
    assert "no third Unblock" in normalized
    assert "fresh supervisor-authorized Apply packet" in normalized
    assert "One ref's exhaustion never freezes ready siblings" in normalized


def test_completion_right_stamp_and_alignment_contract_is_explicit() -> None:
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    loop = read_skill("openspec-loop-engineering")
    interview = read_skill("openspec-change-interviewer")
    verify = read_skill("openspec-verify-change")
    unblock = read_skill("openspec-unblock-research")
    guide = (REPO_ROOT / "docs" / "openspec-loop-engineering.md").read_text(
        encoding="utf-8"
    )
    joined = "\n".join((agents, loop, interview, verify, unblock, guide))
    normalized = " ".join(joined.split())

    for phrase in (
        "execution chapter",
        "cycle stamp",
        "chapter-outside",
        "stamp_source=unblock_self_confirm",
        "max_apply_attempts=2",
        "max_unblock_runs=2",
        "max_revisions=3",
        "completion right",
    ):
        assert phrase.lower() in normalized.lower()

    assert "next --intent" in normalized
    assert "receipt" in loop
    assert "Only the supervisor records, promotes, reseals" in loop
    assert "Apply never edits `tasks.md`" in loop
    assert "The fourth charged stamp" in interview
    assert "completion right" in normalized.lower()
    assert "second R1" in normalized
    assert "harness_contract_hash" in guide
    assert "charged_cycle_stamps gate" in guide
    assert "max_revisions=14" not in agents
    assert "max_revisions=14" not in loop
    assert "max_revisions = task" not in normalized


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
    guide = (REPO_ROOT / "docs" / "openspec-loop-engineering.md").read_text(
        encoding="utf-8"
    )

    assert "seal-preview.md" in interview
    assert "optional audit" in interview.lower()
    assert "Only after acceptance" not in interview
    assert "neither ordinary start-work authority" in " ".join(placement.split())
    assert "path profile" in preview.lower()
    assert "A_local_thin" in preview
    assert "B_external_heavy" in preview
    assert "<change-id>" in preview
    assert "Anti-pattern" in preview
    assert "city_yearbooks" not in preview.split("Anti-pattern")[0]
    assert "additive-extension" in preview
    assert "major-revision" in preview
    assert "configured / recommended / shortfall" in preview
    assert "active_ref_count * max_apply_attempts" not in preview
    assert "is not a `check` warning" in " ".join(interview.split())
    assert "path profile" in placement.lower()
    assert "seal-preview" in guide
    assert "change-id" in guide
    assert "A_local_thin" in guide
    assert "change-scoped 可选审计诊断" in guide


def test_two_drift_classes_are_documented_with_distinct_recoveries() -> None:
    loop = read_skill("openspec-loop-engineering")
    apply = read_skill("openspec-apply-change")
    interview = read_skill("openspec-change-interviewer")
    guide = (REPO_ROOT / "docs" / "openspec-loop-engineering.md").read_text(
        encoding="utf-8"
    )

    combined = " ".join("\n".join((loop, apply, interview, guide)).split())
    assert "semantic fingerprint drift" in combined or "semantic drift" in combined
    assert "narrative 漂移" in combined
    assert "reseal" in combined
    assert "narrative-only warning uses the existing reseal path" in combined
    assert "narrative 漂移" in guide
    assert "semantic 漂移" in guide
    assert "勾选 checkbox 不是漂移" in guide


def test_one_authority_model_keeps_apply_count_ref_local() -> None:
    loop = read_skill("openspec-loop-engineering")
    interview = read_skill("openspec-change-interviewer")
    guide = (REPO_ROOT / "docs" / "openspec-loop-engineering.md").read_text(
        encoding="utf-8"
    )

    normalized = " ".join("\n".join((loop, interview, guide)).split())
    assert "hard_ceiling" in normalized
    assert "per-ref" in normalized
    assert "human" in normalized.lower()
    assert "Revision/change Apply limits contribute only to `apply_remaining`" not in normalized
    assert "hard_ceiling" in interview
    assert "Ref-local Apply/unblock exhaustion is local" in interview
    assert "summary 的 revision/change Apply" in guide
    assert "used/remaining 输出都已删除" in guide
    assert "Apply 次数权威只有各 ref" in guide
    assert "autonomy 不扩大仓库权限" in guide
    assert "唯一任何 Loop 都无权提高的停止条件" not in guide


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
    guide = (REPO_ROOT / "docs" / "openspec-loop-engineering.md").read_text(
        encoding="utf-8"
    )

    normalized = " ".join("\n".join((loop, apply, guide)).split())
    assert "authoritative receipt" in normalized
    assert "shadow receipts" in normalized
    assert "same turn" in normalized or "同回合" in normalized
    assert "natural-language continuation has no control authority" in normalized
    assert "NEXT: supervisor_join" not in normalized


def test_design_level_closing_is_documented_across_skills_and_guide() -> None:
    loop = read_skill("openspec-loop-engineering")
    verify = read_skill("openspec-verify-change")
    interview = read_skill("openspec-change-interviewer")
    guide = (REPO_ROOT / "docs" / "openspec-loop-engineering.md").read_text(
        encoding="utf-8"
    )

    semantic = (
        CANONICAL_ROOT
        / "openspec-verify-change"
        / "references"
        / "semantic-verification-contract.md"
    ).read_text(encoding="utf-8")
    combined = " ".join("\n".join((loop, verify, semantic, interview, guide)).split())
    assert "design closing" in combined.lower()
    assert "openspec-loop-revision-proposal.v1" in combined
    assert "unobserved" in combined and "GAP" in combined
    assert "GOAL G1:" in interview
    assert "COVERED_BY:" in interview
    assert "没有 ready task **不等于** design 已经实现" in " ".join(guide.split())
    assert "autonomy 不得制造空合同" in guide


def test_feature_registry_is_compact_and_monitor_is_audit_only() -> None:
    feature = read_skill("openspec-feature-list")
    apply = read_skill("openspec-apply-change")
    monitor = read_skill("monitor-openspec-codex")
    for field in (
        "`id`",
        "`ref`",
        "`state`",
        "`accept_hash`",
        "`test_hash`",
    ):
        assert field in feature
    assert "not a second copy of ACCEPT" in feature
    assert "generate_openspec_feature_list.py" in feature
    assert "receipt-check" in apply
    assert "Do not run `check`, `plan`" in apply or "never rerun `check`" in apply
    assert "Legacy Audit Only" in monitor
    assert "outside the ordinary OpenSpec Loop" in monitor
    assert "retention=full" in monitor


def test_unblock_v2_example_validates_and_v1_remains_readable() -> None:
    reference_root = CANONICAL_ROOT / "openspec-unblock-research" / "references"
    schema = json.loads(
        (reference_root / "portable-unblock-report.v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    examples = (reference_root / "examples.md").read_text(encoding="utf-8")
    match = re.search(r"```json\s*(\{.*?\})\s*```", examples, re.DOTALL)
    assert match is not None
    payload = json.loads(match.group(1))
    if Draft202012Validator is not None:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(payload)
    else:
        assert set(schema["required"]) <= set(payload)
        assert payload["schema"] == schema["properties"]["schema"]["const"]
        assert payload["trigger"]["trigger_type"] in schema["properties"]["trigger"][
            "properties"
        ]["trigger_type"]["enum"]
    assert payload["trigger"]["error_excerpt"] is None
    assert payload["trigger"]["trigger_type"] == "output_deviation"
    assert (reference_root / "portable-unblock-report.v1.md").is_file()


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
    guide = (REPO_ROOT / "docs" / "openspec-loop-engineering.md").read_text(
        encoding="utf-8"
    )
    spec = (
        REPO_ROOT / "openspec" / "specs" / "openspec-loop-execution" / "spec.md"
    ).read_text(encoding="utf-8")
    runtime = (REPO_ROOT / "scripts" / "openspec_loop.py").read_text(
        encoding="utf-8"
    )

    assert "repair_r1" in loop and "repair_r1" in unblock
    details = " ".join("\n".join((disposition, guide, spec, runtime)).split())
    for token in (
        "repair_class",
        "REPAIR_POLICY: bounded-r1",
        "openspec:repair-r1-method",
        "obligation hash",
        "exactly one post-repair Apply",
        "no third Unblock",
    ):
        assert token in details


def test_unblock_v2_repair_class_is_optional_and_fail_closed() -> None:
    root = CANONICAL_ROOT / "openspec-unblock-research" / "references"
    schema = json.loads(
        (root / "portable-unblock-report.v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    disposition = (root / "loop-disposition-gate.md").read_text(encoding="utf-8")

    assert schema["properties"]["repair_class"]["enum"] == ["R0", "R1", "R2"]
    assert "repair_class" not in schema["required"]
    assert "Missing `repair_class` defaults fail-closed" in disposition
    assert "`supersede_task|stop_budget` are always R2" in disposition


def test_interviewer_creates_the_seven_section_r2_packet_first() -> None:
    interviewer = read_skill("openspec-change-interviewer")
    packet = (
        CANONICAL_ROOT
        / "openspec-change-interviewer"
        / "references"
        / "interview-packet-format.md"
    ).read_text(encoding="utf-8")

    assert "the first write is to create or refresh `interview.md`" in " ".join(
        interviewer.split()
    )
    for section in range(1, 8):
        assert f"## {section}." in packet
    assert "thin + A_local_thin" in " ".join(packet.split())


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
    assert "references/execution-notebook-contract.md" in apply_skill
    assert "execution-notebook-contract.md" in verify_skill
    assert "Apply owns execution writes" in apply_skill
    assert "Verify remains read-only" in verify_skill
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
        assert f"`{field}`" in notebook


def test_high_frequency_skills_are_thin_compiled_interface_cards() -> None:
    cards = {
        "loop": read_skill("openspec-loop-engineering"),
        "apply": read_skill("openspec-apply-change"),
        "verify": read_skill("openspec-verify-change"),
        "unblock": read_skill("openspec-unblock-research"),
    }

    for name, card in cards.items():
        assert len(card.splitlines()) <= 80, (name, len(card.splitlines()))
        assert "DEVIATED" in card
    assert "next --intent" in cards["loop"]
    assert "receipt-check" in cards["loop"] and "receipt-check" in cards["apply"]
    assert "WRITE_SCOPE" in cards["loop"]
    assert "WRITE_SCOPE" in cards["apply"]
    assert "WRITE_SCOPE" in cards["verify"]
    assert "repair_r1" in cards["loop"] and "repair_r1" in cards["unblock"]
    assert "NEXT: supervisor_join" not in "\n".join(cards.values())


def test_interviewer_and_adapter_do_not_duplicate_compiled_routing() -> None:
    interviewer = read_skill("openspec-change-interviewer")
    matrix = (
        CANONICAL_ROOT
        / "openspec-loop-engineering"
        / "references"
        / "role-adapter-matrix.md"
    ).read_text(encoding="utf-8")

    compact = " ".join(interviewer.split())
    assert "When `loop.json` exists, run only `next --shadow`" in compact
    assert "When `loop.json` is missing, run `check` once" in compact
    assert "re-run `check`/`plan`" not in interviewer
    assert "Ordinary Apply routing comes only from compiled `next`" in matrix
    assert "review|explore" in matrix


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

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

    normalized = " ".join(loop.split())
    assert "matching fingerprint with no pending irreversible policy" in normalized
    assert "Missing-loop initialization" in loop
    assert "pending | ready | in_progress | blocked | deviated | maxed |" in loop
    assert "max_apply_attempts=2" in loop
    assert "max_unblock_runs=2" in loop
    assert "positive per-ref Apply remainder" in normalized
    assert "Revision/change Apply limits contribute only to `apply_remaining`" not in loop
    assert "8 apply iterations" not in loop
    assert "20 apply iterations" not in loop
    assert "fingerprint drift" in loop
    assert "index" in interview.lower()
    assert "one-line" in interview
    assert "Do not ask again about D-drive" in interview
    assert "seal-preview" in interview
    assert "path profile" in interview
    assert "A_local_thin" in interview or "change-id" in interview
    assert "PASS|FAIL|BLOCKED|DEVIATED" in verify
    assert "even when the command exits zero" in " ".join(verify.split())
    assert "at most 4 tool calls" in unblock
    assert "180 seconds" in unblock
    assert "retry|targeted_probe|amend_spec|supersede_task|stop_budget" in unblock
    assert "the second is a terminal adjudication" in unblock
    assert "different failure fingerprint or new discriminating evidence" in unblock
    assert "revision_apply_iterations_used|remaining" not in loop
    assert "change_apply_iterations_used|remaining" not in loop
    assert "budgets.revision.max_iterations" in loop
    assert "budgets.change.max_iterations" in loop
    assert "hard_ceiling.max_iterations" in loop
    assert "migration residue only" in normalized
    assert "historical diagnostics" not in normalized


def test_unblock_host_policy_is_in_process_and_ref_local() -> None:
    unblock = read_skill("openspec-unblock-research")
    normalized = " ".join(unblock.split())

    assert "sole supervisor invokes this skill in-process" in normalized
    assert "unblock host spawn count is zero" in normalized
    assert "not a third subordinate host or worker lane" in normalized
    assert "do not resume the failed Apply task/thread" in normalized
    assert "Task.resume" in unblock
    assert "resume_agent" in unblock
    assert "spawn explorer/mapper/verifier/review" in normalized
    assert "worker call Loop gate/record" in normalized

    assert "Only the supervisor may pass the ordinary gate" in normalized
    assert "open a fresh Apply packet" in normalized
    assert "targeted_probe` runs one probe in-process by default" in normalized
    assert "single read-only spawn" in normalized
    assert "wall-clock benefit is recorded on the scan row" in normalized
    assert "joins immediately" in normalized
    assert "never starts Verify" in normalized
    assert "semantic-restamp path with spawn zero" in normalized
    assert "stops only the affected ref" in normalized
    assert "does not freeze siblings or buy a third research agent" in normalized
    assert "second unblock never starts a research swarm" in normalized

    assert "at most 4 tool calls, 4 evidence items, and 180 seconds" in normalized
    assert "at most two unblock runs per ref" in normalized
    assert "amend_spec|supersede_task|stop_budget" in unblock
    assert "Persist a report only when it changes task direction" in unblock


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

    for text in (loop, apply, interview):
        assert "semantic fingerprint drift" in text or "semantic" in text
        assert "reseal" in text
    assert "narrative drift" in loop
    assert "Promotion is not drift." in loop
    assert "narrative_policy: strict" in loop
    assert "Narrative drift" in interview
    assert "does not stop the attempt" in " ".join(apply.split())
    assert "narrative 漂移" in guide
    assert "semantic 漂移" in guide
    assert "勾选 checkbox 不是漂移" in guide


def test_one_authority_model_keeps_apply_count_ref_local() -> None:
    loop = read_skill("openspec-loop-engineering")
    interview = read_skill("openspec-change-interviewer")
    guide = (REPO_ROOT / "docs" / "openspec-loop-engineering.md").read_text(
        encoding="utf-8"
    )

    normalized = " ".join(loop.split())
    assert "Optional `hard_ceiling` may retain only active-minute" in normalized
    assert "does not gate ordinary dispatch" in normalized
    assert "Raising that optional policy ceiling still requires explicit human" in normalized
    assert "max_apply_attempts=2" in loop
    assert "max_unblock_runs=2" in loop
    assert "Revision/change Apply limits contribute only to `apply_remaining`" not in loop
    assert "supervised" in loop and "full_auto" in loop
    assert "stay human-authorized under `AGENTS.md`" in loop
    assert "hard_ceiling" in interview
    assert "Ref-local Apply/unblock exhaustion is local" in interview
    assert "summary 的 revision/change Apply" in guide
    assert "used/remaining 输出都已删除" in guide
    assert "Apply 次数权威只有各 ref" in guide
    assert "autonomy 不扩大仓库权限" in guide
    assert "唯一任何 Loop 都无权提高的停止条件" not in guide


def test_the_loop_drains_the_ready_queue_and_promotes_atomically() -> None:
    loop = read_skill("openspec-loop-engineering")
    normalized = " ".join(loop.split())

    assert "Drain the ready queue rather than stopping after one task:" in normalized
    assert "For the first ready task only" not in loop
    assert "write scope disjoint from packages already admitted" in normalized
    for command in ("promote", "sync", "goals", "design-verify", "apply-revision"):
        assert f"python scripts/openspec_loop.py {command}" in loop
    flowed = " ".join(loop.split())
    assert "Never hand-edit the checkbox" in flowed
    assert "requires a recorded supervisor verifier pass for that ref" in flowed


def test_loop_same_turn_execution_latch() -> None:
    loop = read_skill("openspec-loop-engineering")
    apply = read_skill("openspec-apply-change")
    guide = (REPO_ROOT / "docs" / "openspec-loop-engineering.md").read_text(
        encoding="utf-8"
    )

    normalized = " ".join(loop.split())
    for phrase in (
        "next substantive action in the same turn must be Apply",
        "Do not end with a summary, suggestion, or handoff",
        "A permitted direct package uses non-deployable local `zpy` (display `ZPY`) and remains supervisor-owned",
        "A worker may Apply only its packet; it must not Verify, promote",
        "One invocation uses one `run_id`",
        "must not create `BUNDLE`, `EVIDENCE`, `progress.txt`, `runs.log`",
    ):
        assert phrase in normalized
    assert "exact next command" not in loop
    normalized_apply = " ".join(apply.split())
    assert "NEXT: supervisor_join" in apply
    assert "NEXT: $openspec-verify-change" not in apply
    assert "not a user handoff" in normalized_apply
    assert "同回合" in guide
    assert "先做后说" in guide
    normalized_guide = " ".join(guide.split())
    assert (
        "`gate` 对 `dispatch_refs` 返回 `continue` 后，同回合下一项 "
        "substantive action 必须是 Apply"
    ) in normalized_guide


def test_design_level_closing_is_documented_across_skills_and_guide() -> None:
    loop = read_skill("openspec-loop-engineering")
    verify = read_skill("openspec-verify-change")
    interview = read_skill("openspec-change-interviewer")
    guide = (REPO_ROOT / "docs" / "openspec-loop-engineering.md").read_text(
        encoding="utf-8"
    )

    assert "close at the design level before claiming" in loop
    assert "## Design-level closing" in verify
    assert "openspec-loop-revision-proposal.v1" in verify
    assert "`unobserved`, which is a `GAP`, not a" in verify
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
    assert (
        "python scripts/openspec_loop.py --repo-root . plan <change-id> --advisory"
        in feature
    )
    assert (
        "python scripts/openspec_loop.py --repo-root . check <change-id>" in apply
    )
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
        "NEXT: supervisor_join",
        "Task.resume",
        "resume_agent",
        "shared_context_bytes + Σpacket_delta_bytes + Σresult_bytes",
        "record(apply|unblock)",
        "return_only",
        "最多 4 tool calls、4 evidence items、180s",
        "the registry declares no design goal",
        "uncovered_goals=[]",
        "orphan_refs=[]",
        "Autopilot 最后接管",
    ):
        assert phrase in normalized

    assert "helper 并没有额外的 host-batch runtime object" in normalized
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

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


def normalized(text: str) -> str:
    """Make prose assertions insensitive to wrapping and repeated whitespace."""
    return " ".join(text.split())


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


def test_fingerprint_latch_and_ref_local_budget_contract_is_explicit() -> None:
    loop = read_skill("openspec-loop-engineering")
    interview = read_skill("openspec-change-interviewer")
    verify = read_skill("openspec-verify-change")
    unblock = read_skill("openspec-unblock-research")
    flowed = normalized(loop)

    assert "missing or unsealed `loop.json`" not in loop
    assert "When `loop.json` is missing, `check`/`plan` initialize" in flowed
    assert "recorded `thin` defaults" in flowed
    assert "current `contract_fingerprint`" in flowed
    assert "pending_irreversible_policy" in loop
    assert "`selected_batch`" in loop
    assert "`selected_wave`" in loop
    assert "`dispatch_refs`" in loop
    assert "pending | ready | in_progress | blocked | deviated | maxed |" in loop
    assert "max_apply_attempts=2" in loop
    assert "max_unblock_runs=2" in loop
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
    assert "revision_apply_iterations_used|remaining" in loop
    assert "change_apply_iterations_used|remaining" in loop


def test_fingerprint_ready_seal_preview_is_optional_and_change_scoped() -> None:
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
    interview_flowed = normalized(interview)
    preview_flowed = normalized(preview)
    placement_flowed = normalized(placement)

    assert "seal-preview.md" in interview
    assert "Only after acceptance" not in interview
    assert "matching `contract_fingerprint`" in interview_flowed
    assert "optional audit diagnostic" in interview_flowed
    assert "not ordinary start-work gates" in interview_flowed
    assert "一次性全盖" in interview or "every active" in interview.lower()
    assert "optional audit diagnostic" in preview_flowed.lower()
    assert "not a start-work gate" in preview_flowed
    assert "Ordinary Apply/dispatch does not require" in preview_flowed
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
    assert "before first seal" not in placement_flowed.lower()
    assert "optional" in placement_flowed.lower()
    assert "seal-preview" in guide
    assert "fingerprint" in guide
    assert "可选" in guide
    assert "change-id" in guide
    assert "A_local_thin" in guide


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


def test_supervisor_authority_bounds_irreversible_policy_without_dispatch_ceiling() -> None:
    loop = read_skill("openspec-loop-engineering")
    apply = read_skill("openspec-apply-change")
    verify = read_skill("openspec-verify-change")
    interview = read_skill("openspec-change-interviewer")
    guide = (REPO_ROOT / "docs" / "openspec-loop-engineering.md").read_text(
        encoding="utf-8"
    )
    flowed = normalized(loop)
    apply_flowed = normalized(apply)
    verify_flowed = normalized(verify)

    assert "`hard_ceiling` is the only stop that no Loop may raise" not in loop
    assert "matching sealed `loop.json`" not in apply
    assert "sealed fingerprint" not in verify
    assert "matching fingerprint with no pending irreversible policy" in flowed
    assert "Legacy `hard_ceiling` remains policy compatibility data" in flowed
    assert "requires explicit human confirmation" in flowed
    assert "supervised" in loop and "full_auto" in loop
    assert "stay human-authorized under `AGENTS.md`" in loop
    assert "contract_fingerprint" in apply
    assert "packet" in apply_flowed
    assert "supervisor" in apply_flowed
    assert "contract_fingerprint" in verify
    assert "supervisor" in verify_flowed
    assert "hard_ceiling" in interview
    assert "autonomy: supervised" in interview
    assert "autonomy: full_auto" in interview
    assert "唯一任何 Loop 都无权提高的停止条件" not in guide
    assert "不可逆" in guide
    assert "autonomy 不扩大仓库权限" in guide


def test_selected_wave_apply_identity_join_and_atomic_promotion() -> None:
    loop = read_skill("openspec-loop-engineering")
    apply = read_skill("openspec-apply-change")
    verify = read_skill("openspec-verify-change")
    flowed = normalized(loop)
    apply_flowed = normalized(apply)
    verify_flowed = normalized(verify)

    assert "Drain the ready queue rather than stopping after one task" in flowed
    assert "For the first ready task only" not in loop
    assert "preserve the full `selected_batch`" in flowed
    assert "form `selected_wave`" in flowed
    assert "Apply only `dispatch_refs`" in flowed
    assert "One Apply packet binds exactly one ref" in flowed
    assert "one supervisor-owned Apply attempt" in flowed
    assert "exactly one authoritative Apply record" in flowed
    assert "complete the join for every actually dispatched member" in flowed
    assert "before starting any member's Verify" in flowed
    assert "matching sealed `loop.json`" not in apply
    assert "sealed fingerprint" not in verify
    assert "contract_fingerprint" in apply_flowed
    assert "exactly one ref" in apply_flowed
    assert "supervisor-owned Apply attempt" in apply_flowed
    assert "authoritative Apply record" in apply_flowed
    assert "join" in apply_flowed
    assert "contract_fingerprint" in verify_flowed
    assert "zero Apply" in verify_flowed
    assert "zero scheduling headcount" in verify_flowed
    assert "join" in verify_flowed
    for command in ("promote", "sync", "goals", "design-verify", "apply-revision"):
        assert f"python scripts/openspec_loop.py {command}" in loop
    assert "Never hand-edit the checkbox" in flowed
    assert "requires a recorded supervisor verifier pass for that ref" in flowed


def test_subordinate_host_and_authority_same_turn_execution_latch() -> None:
    loop = read_skill("openspec-loop-engineering")
    apply = read_skill("openspec-apply-change")
    guide = (REPO_ROOT / "docs" / "openspec-loop-engineering.md").read_text(
        encoding="utf-8"
    )
    flowed = normalized(loop)
    apply_flowed = normalized(apply)

    for phrase in (
        "next substantive action in the same turn must be Apply",
        "Do not end with a summary, suggestion, or handoff",
        "One invocation uses one `run_id`",
        "must not create `BUNDLE`, `EVIDENCE`, `progress.txt`, `runs.log`",
    ):
        assert phrase in flowed
    assert "A worker may Apply only its packet" in flowed
    assert "must not Verify, promote, toggle a checkbox, write the ledger, or claim PASS" in flowed
    assert "The supervisor may call only these two subordinate hosts" in flowed
    assert "`silent-failure-hunting`" in loop
    assert "`review-pipeline`" in loop
    assert "another supervisor" in loop
    assert "exact next command" not in loop
    assert "NEXT: $openspec-verify-change" in apply
    assert "not a user handoff" in apply_flowed
    assert "同回合" in guide
    assert "先做后说" in guide


def test_design_level_closing_is_documented_across_skills_and_guide() -> None:
    loop = read_skill("openspec-loop-engineering")
    verify = read_skill("openspec-verify-change")
    interview = read_skill("openspec-change-interviewer")
    guide = (REPO_ROOT / "docs" / "openspec-loop-engineering.md").read_text(
        encoding="utf-8"
    )
    guide_flowed = normalized(guide)

    assert "close at the design level before claiming" in loop
    assert "## Design-level closing" in verify
    assert "openspec-loop-revision-proposal.v1" in verify
    assert "`unobserved`, which is a `GAP`, not a" in verify
    assert "GOAL G1:" in interview
    assert "COVERED_BY:" in interview
    assert "无 ready task 也不等于 design 已实现" in guide_flowed
    assert "缺 observation 是 `unobserved`，不能 PASS" in guide_flowed
    assert "只有 design PASS 后才运行 whole-change Verify" in guide_flowed


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


def test_touched_text_is_utf8_clean() -> None:
    paths = [REPO_ROOT / "docs" / "openspec-loop-engineering.md"]
    for name in SKILL_NAMES:
        paths.extend((CANONICAL_ROOT / name).rglob("*.md"))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert all(token not in text for token in MOJIBAKE_TOKENS), path

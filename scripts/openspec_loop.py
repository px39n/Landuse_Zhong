from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


SCHEMA_LOOP = "openspec-loop.v3"
LEGACY_SCHEMA_LOOP = "openspec-loop.v2"
NARRATIVE_POLICIES = ("advisory", "strict")
AUTONOMY_MODES = ("supervised", "full_auto")
DEFAULT_HARD_CEILING = {
    "max_active_minutes": 1080,
    "max_self_extensions": 3,
}
# Ceiling key -> the change budget it is allowed to bound.
HARD_CEILING_BUDGET_LINKS = (
    ("max_active_minutes", "max_active_minutes"),
)
SCHEMA_PLAN = "openspec-loop-plan.v2"
SCHEMA_CHECK = "openspec-loop-check.v2"
SCHEMA_LEDGER = "openspec-loop-ledger.v2"
SCHEMA_GATE = "openspec-loop-gate.v2"
SCHEMA_SUMMARY = "openspec-loop-summary.v3"
SCHEMA_PROMOTE = "openspec-loop-promote.v1"
SCHEMA_SYNC = "openspec-loop-sync.v1"
SCHEMA_GOALS = "openspec-loop-goals.v1"
SCHEMA_DESIGN_VERIFY = "openspec-loop-design-verify.v1"
SCHEMA_REVISION_PROPOSAL = "openspec-loop-revision-proposal.v1"
SCHEMA_APPLY_REVISION = "openspec-loop-apply-revision.v1"
OBSERVATION_STATES = ("match", "mismatch")
TERMINAL_APPLY_RESULTS = {
    "blocked",
    "completed",
    "deviated",
    "empty",
    "failed",
    "failure",
    "no_progress",
    "partial",
    "success",
    "unverified",
}
COMPLETED_APPLY_RESULTS = {"completed", "success"}
CANONICAL_APPLY_STATUSES = {
    "completed",
    "failed",
    "empty",
    "partial",
    "blocked",
    "unverified",
}
ZERO_APPLY_KINDS = {"explore", "verify", "goal", "stop_hook", "review"}

TASK_STATES = {
    "pending",
    "ready",
    "in_progress",
    "blocked",
    "deviated",
    "maxed",
    "superseded",
    "passed",
}
TERMINAL_STATES = {"maxed", "superseded", "passed"}
PAUSED_STATES = {"blocked", "deviated", "in_progress"}
DEFAULT_BUDGETS = {
    "task": {"max_apply_attempts": 2, "max_unblock_runs": 2},
    "revision": {
        "max_explore_runs": 1,
        "max_subagents": 2,
        "max_active_minutes": 120,
    },
    "change": {
        "max_revisions": 3,
        "max_active_minutes": 360,
    },
}
DEFAULT_TEST_PROFILES = {
    "attempt": "targeted owner tests; no retained bundle",
    "promotion": "complete task TEST exactly once before PASS",
    "final": "whole-change verification plus strict OpenSpec validation",
    "audit": "legacy monitor only when retention=full or explicitly required",
}

NON_OPERATIVE_HEADING_KEYWORDS = (
    "historical",
    "rollback",
    "rollback-only",
    "superseded",
    "appendix",
)

CHECKBOX_RE = re.compile(
    r"^\s*-\s*\[(?P<mark>[ xX])\]\s+"
    r"(?P<task_id>\d+(?:\.\d+)+)\s+"
    r"(?P<title>.*?)\s+\[#(?P<ref>R[^\]\s]+)\]\s*$"
)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
GOAL_RE = re.compile(
    r"^\s*-\s*GOAL\s+(?P<goal_id>[A-Za-z0-9_.-]+)\s*:\s*(?P<title>.+?)\s*$"
)
GOAL_DIRECTIVE_RE = re.compile(
    r"^\s*-\s*(?P<key>COVERED_BY|ACCEPT)\s*:\s*(?P<value>.+?)\s*$",
    re.IGNORECASE,
)
DIRECTIVE_RE = re.compile(
    r"^\s*-\s*(?P<key>DEPENDS_ON|INDEPENDENT|NO_DEP|STATE|SUPERSEDES|FILES|WRITE_SCOPE|ROLE_ID|EXECUTION|JOIN|WORKTREE)\s*:\s*(?P<value>.+?)\s*$",
    re.IGNORECASE,
)
LEGACY_STATE_RE = re.compile(
    r"^\s*-\s*(?P<state>MAXED|BLOCKED|DEVIATED|SUPERSEDED)\b",
    re.IGNORECASE,
)
CHECKBOX_MARK_RE = re.compile(r"^(\s*-\s*\[)[ xX](\]\s+\d+(?:\.\d+)+\s+)")
STATE_DIRECTIVE_RE = re.compile(r"^\s*-\s*STATE\s*:", re.IGNORECASE)
REF_RANGE_RE = re.compile(
    r"^(?P<start>R[A-Za-z0-9_.-]+)\s*-\s*(?P<end>R[A-Za-z0-9_.-]+)$"
)
TASK_ID_RE = re.compile(r"^\d+(?:\.\d+)+$")
PURE_DEPENDENCY_TOKEN_RE = re.compile(r"^(R[A-Za-z0-9_.-]+|\d+(?:\.\d+)+|R[A-Za-z0-9_.-]+\s*-\s*R[A-Za-z0-9_.-]+)$")

DependencyMode = Literal["implicit", "explicit", "explicit-none"]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "done"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def sha256_texts(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
        digest.update(b"\0")
    return digest.hexdigest()


def read_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def configure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


@dataclass(slots=True)
class TaskEntry:
    order: int
    line_number: int
    task_id: str
    ref: str
    title: str
    checked: bool
    depends_raw: str | None = None
    independent_raw: str | None = None
    no_dep_raw: str | None = None
    state_raw: str | None = None
    supersedes_raw: str | None = None
    files_raw: str | None = None
    write_scope_raw: str | None = None
    role_id_raw: str | None = None
    execution_raw: str | None = None
    join_raw: str | None = None
    worktree_raw: str | None = None
    dependencies: list[str] = field(default_factory=list)
    dependency_closure: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    dependency_sources: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    feature_passes: bool | None = None
    feature_task_checked: bool | None = None
    feature_task_id: str | None = None
    feature_state: str | None = None
    feature_state_explicit: bool = False
    effective_state: str = "pending"
    completed: bool = False
    ready: bool = False
    blocked_reasons: list[str] = field(default_factory=list)
    max_apply_attempts: int | None = None
    apply_attempts_used: int = 0
    max_unblock_runs: int | None = None
    unblock_runs_used: int = 0
    unblock_active: bool = False
    budget_disposition: str | None = None
    target_files: list[str] = field(default_factory=list)
    write_scope: list[str] = field(default_factory=list)
    routing: dict[str, Any] | None = None
    execution_mode: str = "async"
    join_id: str | None = None
    worktree_mode: str = "shared"
    write_policy: str = "read_only"
    allowed_write_roots: list[str] = field(default_factory=list)
    wave_eligible: bool = False
    wave_blockers: list[str] = field(default_factory=list)

    def to_plan_item(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "line_number": self.line_number,
            "task_id": self.task_id,
            "ref": self.ref,
            "title": self.title,
            "checked": self.checked,
            "feature_passes": self.feature_passes,
            "feature_task_checked": self.feature_task_checked,
            "feature_state": self.feature_state,
            "effective_state": self.effective_state,
            "dependencies": self.dependencies,
            "dependency_closure": self.dependency_closure,
            "supersedes": self.supersedes,
            "dependency_sources": self.dependency_sources,
            "completed": self.completed,
            "ready": self.ready,
            "issues": self.issues,
            "blocked_reasons": self.blocked_reasons,
            "max_apply_attempts": self.max_apply_attempts,
            "apply_attempts_used": self.apply_attempts_used,
            "max_unblock_runs": self.max_unblock_runs,
            "unblock_runs_used": self.unblock_runs_used,
            "unblock_active": self.unblock_active,
            "budget_disposition": self.budget_disposition,
            "target_files": self.target_files,
            "write_scope": self.write_scope,
            "routing": self.routing,
            "execution_mode": self.execution_mode,
            "join_id": self.join_id,
            "worktree_mode": self.worktree_mode,
            "write_policy": self.write_policy,
            "allowed_write_roots": self.allowed_write_roots,
            "wave_eligible": self.wave_eligible,
            "wave_blockers": self.wave_blockers,
        }


def strip_trailing_period(value: str) -> str:
    cleaned = value.strip()
    while cleaned.endswith("."):
        cleaned = cleaned[:-1].rstrip()
    return cleaned


def is_non_operative_heading(text: str) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in NON_OPERATIVE_HEADING_KEYWORDS)


def parse_task_file(tasks_text: str) -> list[TaskEntry]:
    lines = tasks_text.splitlines()
    has_active_heading = any("active task registry" in line.lower() for line in lines)
    in_fence = False
    active_scope = not has_active_heading
    current_task: TaskEntry | None = None
    tasks: list[TaskEntry] = []

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        heading_match = HEADING_RE.match(line)
        if heading_match:
            heading_text = heading_match.group(2).strip().lower()
            if "active task registry" in heading_text:
                active_scope = True
            elif is_non_operative_heading(heading_text):
                active_scope = False
            continue

        if not active_scope:
            continue

        checkbox_match = CHECKBOX_RE.match(line)
        if checkbox_match:
            current_task = TaskEntry(
                order=len(tasks) + 1,
                line_number=line_number,
                task_id=checkbox_match.group("task_id"),
                ref=checkbox_match.group("ref"),
                title=checkbox_match.group("title").strip(),
                checked=checkbox_match.group("mark").strip().lower() == "x",
            )
            tasks.append(current_task)
            continue

        if current_task is None:
            continue

        directive_match = DIRECTIVE_RE.match(line)
        if not directive_match:
            legacy_state = LEGACY_STATE_RE.match(line)
            if legacy_state and current_task.state_raw is None:
                current_task.state_raw = legacy_state.group("state").lower()
            continue

        key = directive_match.group("key").upper()
        value = strip_trailing_period(directive_match.group("value"))
        if key == "DEPENDS_ON":
            current_task.depends_raw = value
        elif key == "INDEPENDENT":
            current_task.independent_raw = value
        elif key == "NO_DEP":
            current_task.no_dep_raw = value
        elif key == "STATE":
            current_task.state_raw = value.lower().replace("-", "_")
        elif key == "SUPERSEDES":
            current_task.supersedes_raw = value
        elif key == "FILES":
            current_task.files_raw = value
        elif key == "WRITE_SCOPE":
            current_task.write_scope_raw = value
        elif key == "ROLE_ID":
            current_task.role_id_raw = value.strip("`")
        elif key == "EXECUTION":
            current_task.execution_raw = value
        elif key == "JOIN":
            current_task.join_raw = value
        elif key == "WORKTREE":
            current_task.worktree_raw = value

    return tasks


@dataclass(slots=True)
class GoalEntry:
    line_number: int
    goal_id: str
    title: str
    covered_by_raw: str | None = None
    accept_raw: str | None = None
    covered_by: list[str] = field(default_factory=list)
    live_refs: list[str] = field(default_factory=list)
    states: dict[str, str] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def to_item(self) -> dict[str, Any]:
        return {
            "line_number": self.line_number,
            "goal_id": self.goal_id,
            "title": self.title,
            "covered_by": self.covered_by,
            "live_refs": self.live_refs,
            "states": self.states,
            "accept": self.accept_raw,
            "covered": bool(self.live_refs) and not self.issues,
            "issues": self.issues,
        }


def parse_goal_blocks(tasks_text: str) -> list[GoalEntry]:
    """Read optional `GOAL:` blocks from the operative scope.

    Goals are parsed separately from tasks so that a registry without any goal
    stays valid and so a task's own `ACCEPT:` can never be read as a goal's.
    """
    lines = tasks_text.splitlines()
    has_active_heading = any("active task registry" in line.lower() for line in lines)
    in_fence = False
    active_scope = not has_active_heading
    current_goal: GoalEntry | None = None
    goals: list[GoalEntry] = []

    for line_number, line in enumerate(lines, start=1):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        heading_match = HEADING_RE.match(line)
        if heading_match:
            heading_text = heading_match.group(2).strip().lower()
            if "active task registry" in heading_text:
                active_scope = True
            elif is_non_operative_heading(heading_text):
                active_scope = False
            continue

        if not active_scope:
            continue

        goal_match = GOAL_RE.match(line)
        if goal_match:
            current_goal = GoalEntry(
                line_number=line_number,
                goal_id=goal_match.group("goal_id"),
                title=goal_match.group("title").strip(),
            )
            goals.append(current_goal)
            continue

        if CHECKBOX_RE.match(line):
            current_goal = None
            continue

        if current_goal is None:
            continue

        directive_match = GOAL_DIRECTIVE_RE.match(line)
        if not directive_match:
            continue
        key = directive_match.group("key").upper()
        value = directive_match.group("value").strip()
        if key == "COVERED_BY":
            current_goal.covered_by_raw = value
        else:
            current_goal.accept_raw = value

    return goals


def resolve_goal_coverage(goals: list[GoalEntry], tasks: list[TaskEntry]) -> None:
    ref_map = {task.ref: task for task in tasks}
    ordered_refs = [task.ref for task in tasks]
    task_id_map = {task.task_id: task.ref for task in tasks}
    seen_ids: set[str] = set()

    for goal in goals:
        if goal.goal_id in seen_ids:
            goal.issues.append(f"duplicate goal id `{goal.goal_id}`")
        seen_ids.add(goal.goal_id)
        if not goal.accept_raw:
            goal.issues.append("goal has no ACCEPT")
        if goal.covered_by_raw is None:
            goal.issues.append("goal has no COVERED_BY")
            continue

        for token in parse_ref_tokens(goal.covered_by_raw):
            resolved, issue = resolve_dependency_token(
                token,
                ordered_refs=ordered_refs,
                known_refs=set(ref_map),
                task_id_map=task_id_map,
            )
            if issue:
                goal.issues.append(issue.replace("dependency", "COVERED_BY"))
                continue
            for candidate in resolved:
                if candidate not in goal.covered_by:
                    goal.covered_by.append(candidate)

        for ref in goal.covered_by:
            state = ref_map[ref].effective_state
            goal.states[ref] = state
            if state != "superseded":
                goal.live_refs.append(ref)
        if goal.covered_by and not goal.live_refs:
            goal.issues.append("every covering ref is superseded")


def parse_dependency_spec(raw: str | None) -> tuple[DependencyMode, list[str], list[str]]:
    if raw is None:
        return "implicit", [], []

    segments = [strip_trailing_period(segment) for segment in re.split(r"[;；]", raw) if segment.strip()]
    if not segments:
        return "explicit-none", [], ["empty DEPENDS_ON directive"]
    if segments[0].lower() == "none":
        conflicting_tokens: list[str] = []
        for segment in segments[1:]:
            parts = [
                strip_trailing_period(part)
                for part in re.split(r"[,，]", segment)
                if part.strip()
            ]
            if parts and all(PURE_DEPENDENCY_TOKEN_RE.fullmatch(part) for part in parts):
                conflicting_tokens.extend(parts)
        if conflicting_tokens:
            joined = ",".join(conflicting_tokens)
            return "explicit-none", [], [f"DEPENDS_ON none conflicts with `{joined}`"]
        return "explicit-none", [], []

    tokens: list[str] = []
    for segment in segments:
        parts = [strip_trailing_period(part) for part in re.split(r"[,，]", segment) if part.strip()]
        if parts and all(PURE_DEPENDENCY_TOKEN_RE.fullmatch(part) for part in parts):
            tokens.extend(parts)
            continue

    if not tokens:
        return "explicit-none", [], ["empty DEPENDS_ON directive"]
    return "explicit", tokens, []


def parse_no_dep_tokens(raw: str | None) -> tuple[list[str], list[str]]:
    if raw is None or normalize_bool(raw):
        return [], []

    segments = [strip_trailing_period(segment) for segment in re.split(r"[;；]", raw) if segment.strip()]
    tokens: list[str] = []
    issues: list[str] = []

    for segment in segments:
        parts = [strip_trailing_period(part) for part in re.split(r"[/,，]", segment) if part.strip()]
        if parts and all(PURE_DEPENDENCY_TOKEN_RE.fullmatch(part) for part in parts):
            tokens.extend(parts)
            continue

    for token in tokens:
        if token.lower() == "none":
            issues.append("unsupported NO_DEP token `none`")

    return [token for token in tokens if token.lower() != "none"], issues


def load_feature_map(feature_text: str) -> dict[str, Any]:
    payload = json.loads(feature_text)
    if not isinstance(payload, dict):
        raise ValueError("feature_list.json must be a JSON object")
    features = payload.get("features")
    if not isinstance(features, dict):
        raise ValueError("feature_list.json missing top-level features map")
    return features


def normalize_state(value: Any, *, checked: bool, passes: bool) -> str:
    if checked and passes:
        return "passed"
    if isinstance(value, str):
        candidate = value.strip().lower().replace("-", "_")
        legacy = {
            "done": "passed",
            "complete": "passed",
            "historical": "superseded",
            "in-progress": "in_progress",
        }.get(candidate, candidate)
        if legacy in TASK_STATES:
            return legacy
    return "pending"


def parse_ref_tokens(raw: str | None) -> list[str]:
    if raw is None:
        return []
    tokens: list[str] = []
    for segment in re.split(r"[,，;；]", raw):
        candidate = strip_trailing_period(segment)
        if candidate and candidate not in tokens:
            tokens.append(candidate)
    return tokens


def expand_ref_range(
    token: str,
    ordered_refs: list[str],
    known_refs: set[str],
) -> tuple[list[str], str | None]:
    match = REF_RANGE_RE.fullmatch(token)
    if not match:
        return [], f"unsupported dependency token `{token}`"

    start_token = match.group("start")
    end_token = match.group("end")
    if start_token not in known_refs or end_token not in known_refs:
        return [], f"unknown dependency range `{token}`"

    start_idx = ordered_refs.index(start_token)
    end_idx = ordered_refs.index(end_token)
    if start_idx > end_idx:
        return [], f"reversed dependency range `{token}`"
    return ordered_refs[start_idx : end_idx + 1], None


def resolve_dependency_token(
    token: str,
    *,
    ordered_refs: list[str],
    known_refs: set[str],
    task_id_map: dict[str, str],
) -> tuple[list[str], str | None]:
    candidate = strip_trailing_period(token)
    if candidate in known_refs:
        return [candidate], None
    if candidate in task_id_map:
        return [task_id_map[candidate]], None
    if REF_RANGE_RE.fullmatch(candidate):
        return expand_ref_range(candidate, ordered_refs, known_refs)
    if TASK_ID_RE.fullmatch(candidate):
        return [], f"unknown dependency `{candidate}`"
    if candidate.startswith("R") and "-" in candidate:
        return [], f"unknown dependency range `{candidate}`"
    return [], f"unknown dependency `{candidate}`"


def build_dependency_graph(tasks: list[TaskEntry], features: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    ref_map = {task.ref: task for task in tasks}
    task_id_map = {task.task_id: task.ref for task in tasks}
    ordered_refs = [task.ref for task in tasks]
    known_refs = set(ref_map)

    for task in tasks:
        feature = features.get(task.ref)
        if not isinstance(feature, dict):
            task.issues.append("missing feature_list entry")
            continue

        task.feature_passes = normalize_bool(feature.get("passes"))
        if "task_checked" in feature:
            task.feature_task_checked = normalize_bool(feature.get("task_checked"))
        task.feature_task_id = feature.get("id", feature.get("task_id"))
        task.feature_state_explicit = "state" in feature
        task.feature_state = normalize_state(
            feature.get("state", feature.get("status")),
            checked=task.checked,
            passes=task.feature_passes,
        )

        if task.feature_task_checked is not None and task.feature_task_checked != task.checked:
            task.issues.append("checkbox/feature_list task_checked drift")
        if task.feature_passes != task.checked:
            task.issues.append("checkbox/passes drift")
        if task.feature_task_id and task.feature_task_id != task.task_id:
            task.issues.append("task_id drift")
        if task.state_raw and task.state_raw not in TASK_STATES:
            task.issues.append(f"unsupported STATE `{task.state_raw}`")

    for task in tasks:
        for token in parse_ref_tokens(task.supersedes_raw):
            resolved, issue = resolve_dependency_token(
                token,
                ordered_refs=ordered_refs,
                known_refs=known_refs,
                task_id_map=task_id_map,
            )
            if issue:
                task.issues.append(issue.replace("dependency", "supersedes"))
                continue
            for candidate in resolved:
                if candidate == task.ref:
                    task.issues.append("task cannot supersede itself")
                elif candidate not in task.supersedes:
                    task.supersedes.append(candidate)

    superseded_targets = {
        candidate
        for task in tasks
        for candidate in task.supersedes
    }

    for task in tasks:
        if not task.state_raw or not task.feature_state_explicit:
            continue
        expected_state = (
            "superseded" if task.ref in superseded_targets else task.state_raw
        )
        if (
            {expected_state, task.feature_state} != {"pending", "ready"}
            and expected_state != task.feature_state
        ):
            task.issues.append("task/feature state drift")

    for idx, task in enumerate(tasks):
        if normalize_bool(task.independent_raw) or normalize_bool(task.no_dep_raw):
            task.dependencies = []
            task.dependency_sources = ["independent"]
            continue

        mode, explicit_tokens, token_issues = parse_dependency_spec(task.depends_raw)
        task.issues.extend(token_issues)

        if mode == "explicit-none":
            task.dependencies = []
            task.dependency_sources = ["DEPENDS_ON:none"]
            continue

        if mode == "explicit":
            resolved: list[str] = []
            for token in explicit_tokens:
                expanded, dependency_issue = resolve_dependency_token(
                    token,
                    ordered_refs=ordered_refs,
                    known_refs=known_refs,
                    task_id_map=task_id_map,
                )
                if dependency_issue:
                    task.issues.append(dependency_issue)
                    continue
                for candidate in expanded:
                    if candidate not in resolved:
                        resolved.append(candidate)
            task.dependencies = resolved
            task.dependency_sources = explicit_tokens
            continue

        resolved_exclusions: list[str] = []
        exclusion_tokens, exclusion_issues = parse_no_dep_tokens(task.no_dep_raw)
        task.issues.extend(exclusion_issues)
        for token in exclusion_tokens:
            expanded, dependency_issue = resolve_dependency_token(
                token,
                ordered_refs=ordered_refs,
                known_refs=known_refs,
                task_id_map=task_id_map,
            )
            if dependency_issue:
                task.issues.append(dependency_issue)
                continue
            for candidate in expanded:
                if candidate not in resolved_exclusions:
                    resolved_exclusions.append(candidate)

        task.dependencies = [
            prior.ref
            for prior in tasks[:idx]
            if prior.ref not in resolved_exclusions and prior.ref not in task.supersedes
        ]
        task.dependency_sources = ["document-order"] + [f"NO_DEP:{token}" for token in exclusion_tokens]

    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(ref: str, trail: list[str]) -> None:
        if ref in visited:
            return
        if ref in visiting:
            cycle = trail[trail.index(ref) :] + [ref]
            cycle_msg = "dependency cycle: " + " -> ".join(cycle)
            for cycle_ref in set(cycle):
                if cycle_msg not in ref_map[cycle_ref].issues:
                    ref_map[cycle_ref].issues.append(cycle_msg)
            return

        visiting.add(ref)
        trail.append(ref)
        for dep in ref_map[ref].dependencies:
            if dep in ref_map:
                dfs(dep, trail)
        trail.pop()
        visiting.remove(ref)
        visited.add(ref)

    for task in tasks:
        dfs(task.ref, [])

    # A batch caller needs the closure, not just direct edges, to see that two
    # ready refs cannot shadow each other.
    closure: dict[str, set[str]] = {}

    def resolve_closure(ref: str, trail: set[str]) -> set[str]:
        if ref in closure:
            return closure[ref]
        if ref in trail:
            return set()
        trail.add(ref)
        reachable: set[str] = set()
        for dep in ref_map[ref].dependencies:
            if dep in ref_map:
                reachable.add(dep)
                reachable |= resolve_closure(dep, trail)
        trail.discard(ref)
        closure[ref] = reachable
        return reachable

    for task in tasks:
        task.dependency_closure = sorted(resolve_closure(task.ref, set()))

    # Completion is independent of document order. A later-line dependency
    # (for example R42 after city-close R7) must already be marked completed
    # before unmet edges are evaluated.
    for task in tasks:
        task.completed = (
            task.checked
            and task.feature_passes is True
            and not any("drift" in issue for issue in task.issues)
        )
        if task.completed:
            task.effective_state = "passed"
        elif task.ref in superseded_targets:
            task.effective_state = "superseded"
        elif task.state_raw in TASK_STATES:
            task.effective_state = task.state_raw
        elif task.feature_state in TASK_STATES:
            task.effective_state = task.feature_state

    for task in tasks:
        unmet = [dep for dep in task.dependencies if dep in ref_map and not ref_map[dep].completed]
        if unmet:
            task.blocked_reasons.extend([f"waiting on {dep}" for dep in unmet])
        task.blocked_reasons.extend(task.issues)
        if (
            not task.completed
            and task.effective_state not in TERMINAL_STATES | PAUSED_STATES
            and not task.blocked_reasons
        ):
            task.effective_state = "ready"
            task.ready = True

    for task in tasks:
        issues.extend([f"{task.ref}: {issue}" for issue in task.issues])

    return issues


def contract_paths(repo_root: Path, change_id: str) -> tuple[Path, Path]:
    change_dir = repo_root / "openspec" / "changes" / change_id
    return change_dir / "tasks.md", change_dir / "feature_list.json"


def loop_config_path(repo_root: Path, change_id: str) -> Path:
    return repo_root / "openspec" / "changes" / change_id / "loop.json"


def narrative_artifact_paths(repo_root: Path, change_id: str) -> list[Path]:
    change_dir = repo_root / "openspec" / "changes" / change_id
    paths = [
        change_dir / name
        for name in ("proposal.md", "design.md")
        if (change_dir / name).is_file()
    ]
    specs_dir = change_dir / "specs"
    if specs_dir.is_dir():
        paths.extend(
            path
            for path in specs_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".md", ".txt"}
        )
    return sorted(paths, key=lambda path: path.relative_to(change_dir).as_posix())


def normalize_active_tasks_text(tasks_text: str) -> str:
    """Reduce tasks.md to its operative registry, free of execution state.

    Checkbox marks and STATE directives are the Loop's own bookkeeping, so
    promoting a task must not read as a contract amendment.
    """
    lines = tasks_text.splitlines()
    has_active_heading = any("active task registry" in line.lower() for line in lines)
    in_fence = False
    active_scope = not has_active_heading
    kept: list[str] = []

    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            if active_scope:
                kept.append(line.rstrip())
            continue

        if not in_fence:
            heading_match = HEADING_RE.match(line)
            if heading_match:
                heading_text = heading_match.group(2).strip().lower()
                if "active task registry" in heading_text:
                    active_scope = True
                elif is_non_operative_heading(heading_text):
                    active_scope = False
                if active_scope:
                    kept.append(line.rstrip())
                continue

        if not active_scope:
            continue
        if not in_fence and STATE_DIRECTIVE_RE.match(line):
            continue
        if not in_fence:
            line = CHECKBOX_MARK_RE.sub(r"\1 \2", line)
        kept.append(line.rstrip())

    return "\n".join(kept) + "\n"


def compute_semantic_fingerprint(repo_root: Path, change_id: str) -> str:
    tasks_path = repo_root / "openspec" / "changes" / change_id / "tasks.md"
    if not tasks_path.is_file():
        raise ValueError(f"missing contract artifacts for change `{change_id}`")
    normalized = normalize_active_tasks_text(read_utf8(tasks_path))
    return sha256_texts(b"openspec-loop.semantic.v3", normalized.encode("utf-8"))


def compute_narrative_digest(repo_root: Path, change_id: str) -> str:
    change_dir = repo_root / "openspec" / "changes" / change_id
    digest = hashlib.sha256()
    for path in narrative_artifact_paths(repo_root, change_id):
        digest.update(path.relative_to(change_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def autonomy_of(config: dict[str, Any] | None) -> str:
    if isinstance(config, dict):
        candidate = config.get("autonomy")
        if isinstance(candidate, str) and candidate.strip().lower() in AUTONOMY_MODES:
            return candidate.strip().lower()
    return "supervised"


def hard_ceiling_of(config: dict[str, Any] | None) -> dict[str, int] | None:
    raw = config.get("hard_ceiling") if isinstance(config, dict) else None
    if not isinstance(raw, dict):
        return None
    ceiling: dict[str, int] = {}
    for key in DEFAULT_HARD_CEILING:
        value = raw.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            ceiling[key] = value
    return ceiling or None


def self_extensions_used(config: dict[str, Any] | None) -> int:
    if not isinstance(config, dict):
        return 0
    recorded = config.get("budget_extensions")
    return len(recorded) if isinstance(recorded, list) else 0


def narrative_policy_of(config: dict[str, Any] | None) -> str:
    if not isinstance(config, dict):
        return "advisory"
    candidate = config.get("narrative_policy")
    if isinstance(candidate, str) and candidate.strip().lower() in NARRATIVE_POLICIES:
        return candidate.strip().lower()
    return "advisory"


def load_loop_config(repo_root: Path, change_id: str) -> dict[str, Any] | None:
    path = loop_config_path(repo_root, change_id)
    if not path.exists():
        return None
    payload = json.loads(read_utf8(path))
    if not isinstance(payload, dict):
        raise ValueError("loop.json must be a JSON object")
    return sanitize_loop_config(payload)


def sanitize_loop_config(config: dict[str, Any]) -> dict[str, Any]:
    sanitized = json.loads(json.dumps(config, ensure_ascii=False))
    budgets = sanitized.get("budgets")
    if isinstance(budgets, dict):
        revision = budgets.get("revision")
        if isinstance(revision, dict):
            revision.pop("max_iterations", None)
        change = budgets.get("change")
        if isinstance(change, dict):
            change.pop("max_iterations", None)
    hard_ceiling = sanitized.get("hard_ceiling")
    if isinstance(hard_ceiling, dict):
        hard_ceiling.pop("max_iterations", None)
        if not hard_ceiling:
            sanitized.pop("hard_ceiling", None)
    return sanitized


def maybe_persist_sanitized_loop_config(
    repo_root: Path,
    change_id: str,
    original: dict[str, Any] | None,
    sanitized: dict[str, Any] | None,
) -> None:
    if original is None or sanitized is None:
        return
    if original == sanitized:
        return
    write_json_atomic(loop_config_path(repo_root, change_id), sanitized)


def hard_ceiling_issues(config: dict[str, Any]) -> list[str]:
    ceiling = config.get("hard_ceiling")
    if not isinstance(ceiling, dict):
        return []

    issues: list[str] = []
    invalid = sorted(
        key
        for key in set(DEFAULT_HARD_CEILING) & set(ceiling)
        if isinstance(ceiling[key], bool)
        or not isinstance(ceiling[key], int)
        or ceiling[key] <= 0
    )
    if invalid:
        issues.append(
            "loop.json hard_ceiling must be positive integers: " + ", ".join(invalid)
        )
    budgets = config.get("budgets")
    change_budget = budgets.get("change") if isinstance(budgets, dict) else None
    if isinstance(change_budget, dict):
        for ceiling_key, budget_key in HARD_CEILING_BUDGET_LINKS:
            limit = ceiling.get(ceiling_key)
            value = change_budget.get(budget_key)
            if (
                isinstance(limit, int)
                and not isinstance(limit, bool)
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value > limit
            ):
                issues.append(
                    f"loop.json budgets.change.{budget_key} exceeds "
                    f"hard_ceiling.{ceiling_key}"
                )
    return issues


def loop_config_issues(config: dict[str, Any] | None, change_id: str) -> list[str]:
    if config is None:
        return ["contract is not sealed: missing loop.json"]
    issues: list[str] = []
    schema_version = config.get("schema_version")
    if schema_version == LEGACY_SCHEMA_LOOP:
        issues.append(
            f"loop.json uses {LEGACY_SCHEMA_LOOP}; run "
            f"`python scripts/openspec_loop.py reseal {change_id} --migrate`"
        )
    elif schema_version != SCHEMA_LOOP:
        issues.append(f"unsupported loop schema `{schema_version}`")
    else:
        narrative_digest = config.get("narrative_digest")
        if not isinstance(narrative_digest, str) or not narrative_digest.strip():
            issues.append("loop.json narrative_digest is required")
        policy = config.get("narrative_policy")
        if not isinstance(policy, str) or policy.strip().lower() not in NARRATIVE_POLICIES:
            issues.append("loop.json narrative_policy must be advisory or strict")
        autonomy = config.get("autonomy")
        if (
            not isinstance(autonomy, str)
            or autonomy.strip().lower() not in AUTONOMY_MODES
        ):
            issues.append("loop.json autonomy must be supervised or full_auto")
        if "hard_ceiling" in config:
            issues.extend(hard_ceiling_issues(config))
    if config.get("change_id") != change_id:
        issues.append("loop.json change_id mismatch")
    if config.get("retention") not in {"none", "thin", "full"}:
        issues.append("loop.json retention must be none, thin, or full")
    paths = config.get("paths")
    required_paths = {"ledger", "scratch", "product", "bundle", "gui_colab"}
    if not isinstance(paths, dict):
        issues.append("loop.json paths must be an object")
    else:
        missing_paths = sorted(required_paths - set(paths))
        if missing_paths:
            issues.append("loop.json paths missing: " + ", ".join(missing_paths))
        invalid_paths = sorted(
            key
            for key in required_paths & set(paths)
            if paths[key] is not None and not isinstance(paths[key], str)
        )
        if invalid_paths:
            issues.append(
                "loop.json path values must be string or null: "
                + ", ".join(invalid_paths)
            )
    if not isinstance(paths, dict) or not paths.get("ledger"):
        issues.append("loop.json paths.ledger is required")
    if config.get("retention") == "full" and (
        not isinstance(paths, dict) or not paths.get("bundle")
    ):
        issues.append("full retention requires paths.bundle")
    budgets = config.get("budgets")
    if not isinstance(budgets, dict):
        issues.append("loop.json budgets is required")
    else:
        for scope, defaults in DEFAULT_BUDGETS.items():
            values = budgets.get(scope)
            if not isinstance(values, dict):
                issues.append(f"loop.json budgets.{scope} is required")
                continue
            missing = sorted(set(defaults) - set(values))
            if missing:
                issues.append(
                    f"loop.json budgets.{scope} missing: " + ", ".join(missing)
                )
            invalid = sorted(
                key
                for key in set(defaults) & set(values)
                if isinstance(values[key], bool)
                or not isinstance(values[key], int)
                or values[key] <= 0
            )
            if invalid:
                issues.append(
                    f"loop.json budgets.{scope} must be positive integers: "
                    + ", ".join(invalid)
                )
    profiles = config.get("test_profiles")
    required_profiles = {"attempt", "promotion", "final", "audit"}
    if not isinstance(profiles, dict):
        issues.append("loop.json test_profiles is required")
    else:
        missing_profiles = sorted(required_profiles - set(profiles))
        if missing_profiles:
            issues.append(
                "loop.json test_profiles missing: " + ", ".join(missing_profiles)
            )
    return issues


def merge_budget_defaults(raw: Any) -> dict[str, dict[str, int]]:
    merged = {scope: values.copy() for scope, values in DEFAULT_BUDGETS.items()}
    if not isinstance(raw, dict):
        return merged
    for scope, defaults in merged.items():
        supplied = raw.get(scope)
        if not isinstance(supplied, dict):
            continue
        for key in defaults:
            value = supplied.get(key)
            if isinstance(value, int) and value > 0:
                defaults[key] = value
    return merged


RETENTION_PROFILE_LABELS = {
    "Audit retention": "retention",
    "Retained evidence root": "retained",
    "Disposable cache root": "disposable",
    "Pytest basetemp root": "pytest_basetemp",
    "Scratch root": "scratch",
    "Product/runtime output root": "product",
    "GUI/Colab evidence root": "gui_colab",
}


def recorded_retention_profile(repo_root: Path, change_id: str) -> dict[str, Any]:
    proposal_path = repo_root / "openspec" / "changes" / change_id / "proposal.md"
    if not proposal_path.is_file():
        raise ValueError("missing recorded Artifact Retention Decision in proposal.md")

    values: dict[str, str | None] = {}
    in_decision = False
    for line in read_utf8(proposal_path).splitlines():
        if line.strip().lower() == "## artifact retention decision":
            in_decision = True
            continue
        if in_decision and line.startswith("## "):
            break
        if not in_decision or not line.startswith("- ") or ":" not in line:
            continue
        label, raw = line[2:].split(":", 1)
        key = RETENTION_PROFILE_LABELS.get(label.strip())
        if key is None:
            continue
        token = re.split(r"[;；]", raw, maxsplit=1)[0].strip().rstrip("。. ")
        token = token.strip("`")
        values[key] = None if token.lower() == "null" else token

    missing = sorted(set(RETENTION_PROFILE_LABELS.values()) - set(values))
    if missing:
        raise ValueError(
            "Artifact Retention Decision missing fields: " + ", ".join(missing)
        )
    if values["retention"] != "thin":
        raise ValueError("automatic Loop initialization requires recorded thin retention")
    disposable = values["disposable"]
    if not disposable:
        raise ValueError("automatic Loop initialization requires a disposable cache root")

    ledger = (Path(disposable) / "loop" / "ledger.json").as_posix()
    return {
        **values,
        "paths": {
            "ledger": ledger,
            "scratch": values["scratch"],
            "product": values["product"],
            "bundle": values["retained"],
            "gui_colab": values["gui_colab"],
        },
    }


def initialize_thin_loop_config(
    repo_root: Path,
    change_id: str,
    tasks: list[TaskEntry],
    *,
    fingerprint: str,
    narrative_digest: str,
) -> dict[str, Any]:
    retention_profile = recorded_retention_profile(repo_root, change_id)
    budgets = merge_budget_defaults(None)
    change_defaults = budgets["change"].copy()
    change_defaults["max_active_minutes"] = max(
        change_defaults["max_active_minutes"], 10 * len(tasks)
    )
    budgets["change"] = change_defaults
    config = {
        "schema_version": SCHEMA_LOOP,
        "change_id": change_id,
        "contract_fingerprint": fingerprint,
        "narrative_digest": narrative_digest,
        "narrative_policy": "advisory",
        "autonomy": "supervised",
        "retention": retention_profile["retention"],
        "paths": retention_profile["paths"],
        "retention_profile": {
            key: retention_profile[key]
            for key in ("retained", "disposable", "pytest_basetemp", "scratch", "product", "gui_colab")
        },
        "budgets": budgets,
        "per_ref_budgets": {
            task.ref: DEFAULT_BUDGETS["task"].copy() for task in tasks
        },
        "test_profiles": DEFAULT_TEST_PROFILES.copy(),
        "initialized_from": "thin-default",
    }
    write_json_atomic(loop_config_path(repo_root, change_id), config)
    return config


def initialize_loop_from_registry(repo_root: Path, change_id: str) -> dict[str, Any]:
    tasks_path, feature_path = contract_paths(repo_root, change_id)
    if not tasks_path.is_file() or not feature_path.is_file():
        raise ValueError(f"missing contract artifacts for change `{change_id}`")
    tasks = parse_task_file(read_utf8(tasks_path))
    if not tasks:
        raise ValueError(f"change `{change_id}` has no active task registry")
    features = load_feature_map(read_utf8(feature_path))
    build_dependency_graph(tasks, features)
    return initialize_thin_loop_config(
        repo_root,
        change_id,
        tasks,
        fingerprint=compute_semantic_fingerprint(repo_root, change_id),
        narrative_digest=compute_narrative_digest(repo_root, change_id),
    )


def configured_ledger_path(
    repo_root: Path,
    config: dict[str, Any],
    override: Path | None,
) -> Path:
    configured = Path(str(config["paths"]["ledger"]))
    if not configured.is_absolute():
        configured = repo_root / configured
    configured = configured.resolve()
    if override is None:
        return configured
    candidate = override if override.is_absolute() else repo_root / override
    candidate = candidate.resolve()
    if candidate != configured:
        raise ValueError(
            f"ledger path override does not match sealed policy: {candidate} != {configured}"
        )
    return candidate


def ensure_episode_runs(episode: dict[str, Any]) -> None:
    if "runs" in episode:
        return
    legacy_attempts = episode.pop("attempts", [])
    legacy_started_at = episode.pop("started_at_utc", utc_now())
    legacy_allocated = episode.pop("allocated_subagent_ids", [])
    episode["runs"] = [
        {
            "run_id": "default",
            "started_at_utc": legacy_started_at,
            "attempts": legacy_attempts,
            "allocated_subagent_ids": legacy_allocated,
        }
    ]


def load_run(
    ledger: dict[str, Any],
    change_id: str,
    fingerprint: str,
    run_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    episodes = ledger.setdefault("episodes", [])
    for episode in episodes:
        if episode.get("contract_fingerprint") == fingerprint:
            ensure_episode_runs(episode)
            for run in episode["runs"]:
                if run.get("run_id") == run_id:
                    return episode, run
            run = {
                "run_id": run_id,
                "started_at_utc": utc_now(),
                "attempts": [],
                "allocated_subagent_ids": [],
            }
            episode["runs"].append(run)
            return episode, run

    episode: dict[str, Any] = {"contract_fingerprint": fingerprint, "runs": []}
    # An authorized amendment opens a new episode. Naming the episode it
    # continues keeps a revision distinguishable from an accidental reseal.
    predecessor = next(
        (
            candidate.get("contract_fingerprint")
            for candidate in reversed(episodes)
            if attempts_for_episode(candidate)
        ),
        None,
    )
    if predecessor:
        episode["supersedes_fingerprint"] = predecessor
    run = {
        "run_id": run_id,
        "started_at_utc": utc_now(),
        "attempts": [],
        "allocated_subagent_ids": [],
    }
    episode["runs"].append(run)
    episodes.append(episode)
    ledger["change_id"] = change_id
    ledger["schema_version"] = SCHEMA_LEDGER
    return episode, run


def load_or_init_ledger(path: Path, change_id: str) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": SCHEMA_LEDGER, "change_id": change_id, "episodes": []}
    payload = json.loads(read_utf8(path))
    if not isinstance(payload, dict):
        raise ValueError("ledger must be a JSON object")
    payload.setdefault("episodes", [])
    prior_schema = payload.get("schema_version")
    if prior_schema and prior_schema != SCHEMA_LEDGER:
        payload.setdefault("migrated_from", prior_schema)
    payload["schema_version"] = SCHEMA_LEDGER
    payload.setdefault("change_id", change_id)
    return payload


def error_fingerprint(error_text: str | None) -> str | None:
    if not error_text:
        return None
    normalized = re.sub(r"\s+", " ", error_text.strip().lower())
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def narrative_drift_message(change_id: str) -> str:
    return (
        "narrative drift in proposal/design/specs; run "
        f"`python scripts/openspec_loop.py reseal {change_id}` to refresh"
    )


def resolve_runtime_policy(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], str, Path, dict[str, dict[str, int]], list[str]]:
    repo_root = args.repo_root.resolve()
    config = load_loop_config(repo_root, args.change_id)
    if config is None:
        config = initialize_loop_from_registry(repo_root, args.change_id)
    issues = loop_config_issues(config, args.change_id)
    if issues or config is None:
        raise ValueError("; ".join(issues))
    current_fingerprint = compute_semantic_fingerprint(repo_root, args.change_id)
    if config.get("contract_fingerprint") != current_fingerprint:
        raise ValueError("contract fingerprint drift; re-run interviewer and seal")
    supplied = getattr(args, "contract_fingerprint", None)
    if supplied and supplied != current_fingerprint:
        raise ValueError("supplied contract fingerprint does not match sealed contract")
    irreversible_pending = pending_irreversible_policy(config)
    if irreversible_pending:
        raise ValueError(
            "unconfirmed irreversible policy: " + ", ".join(irreversible_pending)
        )
    warnings: list[str] = []
    if config.get("narrative_digest") != compute_narrative_digest(
        repo_root, args.change_id
    ):
        message = narrative_drift_message(args.change_id)
        warnings.append(message)
    ledger_path = configured_ledger_path(
        repo_root,
        config,
        getattr(args, "ledger_path", None),
    )
    budgets = merge_budget_defaults(config.get("budgets"))
    return repo_root, config, current_fingerprint, ledger_path, budgets, warnings


def normalize_subagent_ids(raw_ids: list[str] | None) -> list[str]:
    if not raw_ids:
        return []
    deduped: list[str] = []
    for raw_id in raw_ids:
        candidate = raw_id.strip()
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def ref_budget_of(
    config: dict[str, Any],
    budgets: dict[str, dict[str, int]],
    ref: str | None,
) -> dict[str, int]:
    budget = budgets["task"].copy()
    per_ref = config.get("per_ref_budgets")
    raw = per_ref.get(ref) if ref and isinstance(per_ref, dict) else None
    if isinstance(raw, dict):
        for key in budget:
            value = raw.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                budget[key] = value
    return budget


def pending_irreversible_policy(config: dict[str, Any] | None) -> list[str]:
    if not isinstance(config, dict):
        return []
    raw = config.get("pending_irreversible_policy")
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, dict) and raw.get("confirmed") is not True:
        changes = raw.get("changes")
        if isinstance(changes, list):
            return [str(item).strip() for item in changes if str(item).strip()]
        return ["unspecified"] if raw.get("pending") else []
    return []


def has_blocking_evidence(attempts: list[dict[str, Any]], ref: str) -> bool:
    return any(
        attempt.get("ref") == ref
        and attempt.get("kind") in {"apply", "verify"}
        and attempt.get("result") in {"blocked", "deviated"}
        for attempt in attempts
    )


DEPLOYABLE_ROLE_IDS = {
    "implementer",
    "test-engineer",
    "browser-qa-runner",
    "e2e-artifact-runner",
    "code-scout",
    "doc-researcher",
    "spec-miner",
}
LOCAL_SUPERVISOR_ROLE_ID = "zpy"
LEGACY_SUPERVISOR_ROLE_ID = "rose"
SUPERVISOR_DIRECT_ROLE_IDS = {
    LOCAL_SUPERVISOR_ROLE_ID,
    LEGACY_SUPERVISOR_ROLE_ID,
}
EVIDENCE_WRITER_ROLE_IDS = {"browser-qa-runner", "e2e-artifact-runner"}
READ_ONLY_ROLE_IDS = {
    "solution-architect",
    "code-scout",
    "doc-researcher",
    "web-researcher",
    "plan-auditor",
    "code-reviewer",
    "security-auditor",
    "test-coverage-reviewer",
    "pr-test-analyzer",
    "ai-regression-scout",
    "silent-failure-reviewer",
    "convergence-reviewer",
    "spec-miner",
    "agent-evaluator",
    "opensource-sanitizer",
    "web-performance-auditor",
}
ROLE_WRITE_POLICIES = {
    "implementer": {"mode": "repo", "roots": ["task-owned implementation/contract files"]},
    "test-engineer": {"mode": "tests", "roots": ["task-owned test files"]},
    "browser-qa-runner": {"mode": "evidence", "roots": ["approved evidence root"]},
    "e2e-artifact-runner": {"mode": "evidence", "roots": ["approved evidence root"]},
}


def parse_scope_tokens(raw: str | None) -> list[str]:
    if not raw:
        return []
    quoted = re.findall(r"`([^`]+)`", raw)
    candidates = quoted or re.split(r"[,，;；]", raw)
    scopes: list[str] = []
    for candidate in candidates:
        value = candidate.strip().replace("\\", "/").rstrip("/")
        if value.endswith("/**"):
            value = value[:-3].rstrip("/")
        elif value.endswith("/*"):
            value = value[:-2].rstrip("/")
        elif "*" in value.rsplit("/", 1)[-1] and "/" in value:
            value = value.rsplit("/", 1)[0]
        if value and value not in scopes:
            scopes.append(value)
    return scopes


def is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    name = normalized.rsplit("/", 1)[-1]
    return (
        normalized.startswith("tests/")
        or normalized.startswith("scripts/tests/")
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def infer_candidate_role(task: TaskEntry) -> tuple[str | None, str]:
    if task.role_id_raw:
        return task.role_id_raw, "explicit_role_id"
    if not task.target_files:
        return None, "no_matching_specialist"
    test_flags = [is_test_path(path) for path in task.target_files]
    if all(test_flags):
        return "test-engineer", "task_owned_tests"
    if any(test_flags):
        return None, "clarification_or_split"
    return "implementer", "task_owned_implementation"


def normalize_execution_mode(raw: str | None) -> str:
    if isinstance(raw, str) and raw.strip().lower() == "sync":
        return "sync"
    return "async"


def normalize_worktree_mode(raw: str | None) -> str:
    if isinstance(raw, str) and raw.strip().lower() in {"independent", "isolated"}:
        return "independent"
    return "shared"


def normalize_join_id(task: TaskEntry, *, change_id: str, fingerprint: str) -> str | None:
    if isinstance(task.join_raw, str):
        candidate = task.join_raw.strip()
        if not candidate or candidate.lower() in {"none", "missing", "null"}:
            return None
        return candidate
    if normalize_worktree_mode(task.worktree_raw) == "shared":
        return f"join:{change_id}:{fingerprint[:12]}:wave-1"
    return f"join:{change_id}:{task.ref.lower()}"


def write_policy_for_role(role_id: str | None) -> dict[str, Any]:
    if not role_id:
        return {"mode": "read_only", "roots": []}
    if role_id in SUPERVISOR_DIRECT_ROLE_IDS:
        return {"mode": "supervisor_direct", "roots": []}
    return ROLE_WRITE_POLICIES.get(role_id, {"mode": "read_only", "roots": []})


def scopes_overlap(left: list[str], right: list[str]) -> bool:
    for left_scope in left:
        for right_scope in right:
            if (
                left_scope == right_scope
                or left_scope.startswith(right_scope + "/")
                or right_scope.startswith(left_scope + "/")
            ):
                return True
    return False


def route_ready_wave(
    change_id: str,
    fingerprint: str,
    tasks: list[TaskEntry],
    selected_batch: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    task_by_ref = {task.ref: task for task in tasks}
    admitted_scopes: list[list[str]] = []
    wave_refs: list[str] = []
    routes: list[dict[str, Any]] = []

    for ref in selected_batch:
        task = task_by_ref[ref]
        task.target_files = parse_scope_tokens(task.files_raw)
        task.write_scope = parse_scope_tokens(task.write_scope_raw) or list(
            task.target_files
        )
        if not task.write_scope:
            task.write_scope = [f"ref:{ref}"]
        task.execution_mode = normalize_execution_mode(task.execution_raw)
        task.worktree_mode = normalize_worktree_mode(task.worktree_raw)
        task.join_id = normalize_join_id(task, change_id=change_id, fingerprint=fingerprint)
        matched_role, role_fit_reason = infer_candidate_role(task)
        if matched_role == "general":
            decision = "blocked"
            direct_reason = "capability_failure"
            effective_role = None
        elif matched_role is None:
            decision = "direct"
            direct_reason = role_fit_reason
            effective_role = LOCAL_SUPERVISOR_ROLE_ID
        elif matched_role in SUPERVISOR_DIRECT_ROLE_IDS:
            decision = "direct"
            direct_reason = "no_matching_specialist"
            # Preserve an explicit legacy `rose` packet long enough to record
            # the bootstrap attempt that introduces local `zpy` support. New
            # inferred direct work uses `zpy` above.
            effective_role = matched_role
        elif matched_role in DEPLOYABLE_ROLE_IDS | READ_ONLY_ROLE_IDS:
            decision = "dispatch"
            direct_reason = "N/A"
            effective_role = matched_role
        else:
            decision = "blocked"
            direct_reason = "capability_failure"
            effective_role = None

        blockers: list[str] = []
        disjoint = not any(
            scopes_overlap(task.write_scope, admitted) for admitted in admitted_scopes
        )
        if not task.join_id:
            blockers.append("missing_join")
        if decision != "blocked" and not disjoint:
            decision = "direct"
            direct_reason = "overlap"
            effective_role = LOCAL_SUPERVISOR_ROLE_ID
            blockers.append("write_scope_overlap")
        elif not disjoint:
            blockers.append("write_scope_overlap")
        if decision == "blocked":
            blockers.append("role_or_capability_blocked")
        zero_apply_role = effective_role in READ_ONLY_ROLE_IDS
        if zero_apply_role:
            blockers.append("zero_apply_role")
        qualified = (
            decision != "blocked"
            and disjoint
            and bool(task.join_id)
            and not zero_apply_role
        )
        if qualified:
            wave_refs.append(ref)
            admitted_scopes.append(task.write_scope)
        write_policy = write_policy_for_role(effective_role)
        task.write_policy = write_policy["mode"]
        task.allowed_write_roots = (
            list(task.write_scope)
            if task.write_policy == "supervisor_direct"
            else list(write_policy["roots"])
        )
        task.wave_eligible = qualified
        task.wave_blockers = blockers
        route = {
            "ref": ref,
            "package_id": f"{change_id}:{ref}:apply",
            "agent": (
                f"agents/{change_id}-{ref.lower()}" if decision == "dispatch" else None
            ),
            "bounded_non_trivial": True,
            "matched_role_id": matched_role,
            "decision": decision,
            "direct_reason": direct_reason,
            "effective_role_id": effective_role,
            "role_fit_reason": role_fit_reason,
            "write_scope": list(task.write_scope),
            "execution_mode": task.execution_mode,
            "join_id": task.join_id,
            "worktree_mode": task.worktree_mode,
            "write_policy": task.write_policy,
            "allowed_write_roots": task.allowed_write_roots,
            "execution_lane": "zero_apply" if zero_apply_role else "apply",
            "questions": {
                "bounded_non_trivial": True,
                "narrowest_role": decision != "blocked",
                "write_scope_disjoint": disjoint,
                "supervisor_join": bool(task.join_id),
            },
            "wave_status": "selected" if qualified else (
                "zero_apply_lane"
                if zero_apply_role
                else "deferred_write_overlap"
                if "write_scope_overlap" in blockers
                else "blocked"
            ),
            "later_wave_eligible": (
                decision != "blocked" and not disjoint and not zero_apply_role
            ),
            "wave_eligible": qualified,
            "wave_blockers": blockers,
        }
        task.routing = route
        routes.append(route)
    return wave_refs, routes


def apply_budget_snapshot(
    repo_root: Path,
    change_id: str,
    config: dict[str, Any],
    fingerprint: str,
    tasks: list[TaskEntry],
    selected_wave: list[str],
) -> dict[str, int]:
    tasks_by_ref = {task.ref: task for task in tasks}
    apply_remaining = sum(
        1
        for ref in selected_wave
        if (task := tasks_by_ref.get(ref)) is not None
        and task.max_apply_attempts is not None
        and task.apply_attempts_used < task.max_apply_attempts
    )
    return {"apply_remaining": apply_remaining}


def apply_runtime_budget_state(
    repo_root: Path,
    change_id: str,
    config: dict[str, Any],
    fingerprint: str,
    tasks: list[TaskEntry],
) -> None:
    budgets = merge_budget_defaults(config.get("budgets"))
    attempts: list[dict[str, Any]] = []
    ledger_path = configured_ledger_path(repo_root, config, None)
    if ledger_path.is_file():
        ledger = load_or_init_ledger(ledger_path, change_id)
        episode = next(
            (
                item
                for item in ledger.get("episodes", [])
                if item.get("contract_fingerprint") == fingerprint
            ),
            None,
        )
        if episode is not None:
            attempts = attempts_for_episode(episode)

    for task in tasks:
        task_budget = ref_budget_of(config, budgets, task.ref)
        task.max_apply_attempts = task_budget["max_apply_attempts"]
        task.max_unblock_runs = task_budget["max_unblock_runs"]
        task.apply_attempts_used = sum(
            1
            for attempt in attempts
            if attempt.get("ref") == task.ref and attempt.get("kind") == "apply"
        )
        task.unblock_runs_used = sum(
            1
            for attempt in attempts
            if attempt.get("ref") == task.ref and attempt.get("kind") == "unblock"
        )
        task.unblock_active = has_blocking_evidence(attempts, task.ref)
        if (
            not task.completed
            and task.unblock_runs_used >= task_budget["max_unblock_runs"]
        ):
            task.effective_state = "maxed"
            task.ready = False
            task.budget_disposition = "stop_budget"
            if "stop_budget" not in task.blocked_reasons:
                task.blocked_reasons.append("stop_budget")


def build_plan_payload(
    repo_root: Path,
    change_id: str,
    *,
    advisory: bool,
    batch: bool = False,
) -> dict[str, Any]:
    tasks_path, feature_path = contract_paths(repo_root, change_id)
    missing: list[str] = []
    if not tasks_path.exists():
        missing.append(f"missing tasks file: {tasks_path}")
    if not feature_path.exists():
        missing.append(f"missing feature_list file: {feature_path}")
    if missing:
        return {
            "schema_version": SCHEMA_PLAN,
            "change_id": change_id,
            "sealed": False,
            "contract_fingerprint": None,
            "sealed_fingerprint": None,
            "narrative_digest": None,
            "sealed_narrative_digest": None,
            "candidate_ref": None,
            "selected_ref": None,
            "selected_batch": [],
            "selected_wave": [],
            "routing": [],
            "apply_remaining": 0,
            "allowed_parallel_applies": 0,
            "dispatch_refs": [],
            "fingerprint_ready": False,
            "wave_ready": False,
            "irreversible_policy_pending": [],
            "ready_refs": [],
            "blocked_refs": [],
            "terminal_refs": [],
            "issues": missing,
            "warnings": [],
            "tasks": [],
        }

    try:
        tasks = parse_task_file(read_utf8(tasks_path))
        features = load_feature_map(read_utf8(feature_path))
        contract_issues = build_dependency_graph(tasks, features)
        fingerprint = compute_semantic_fingerprint(repo_root, change_id)
        narrative_digest = compute_narrative_digest(repo_root, change_id)
        loop_path = loop_config_path(repo_root, change_id)
        raw_config = None
        if loop_path.exists():
            raw_config = json.loads(read_utf8(loop_path))
            if not isinstance(raw_config, dict):
                raise ValueError("loop.json must be a JSON object")
        config = sanitize_loop_config(raw_config) if raw_config is not None else None
        maybe_persist_sanitized_loop_config(repo_root, change_id, raw_config, config)
        if config is None and not advisory and tasks:
            config = initialize_thin_loop_config(
                repo_root,
                change_id,
                tasks,
                fingerprint=fingerprint,
                narrative_digest=narrative_digest,
            )
        policy_issues = loop_config_issues(config, change_id)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return {
            "schema_version": SCHEMA_PLAN,
            "change_id": change_id,
            "sealed": False,
            "contract_fingerprint": None,
            "sealed_fingerprint": None,
            "narrative_digest": None,
            "sealed_narrative_digest": None,
            "candidate_ref": None,
            "selected_ref": None,
            "selected_batch": [],
            "selected_wave": [],
            "routing": [],
            "apply_remaining": 0,
            "allowed_parallel_applies": 0,
            "dispatch_refs": [],
            "fingerprint_ready": False,
            "wave_ready": False,
            "irreversible_policy_pending": [],
            "ready_refs": [],
            "blocked_refs": [],
            "terminal_refs": [],
            "issues": [str(exc)],
            "warnings": [],
            "tasks": [],
        }

    warnings: list[str] = []
    sealed_fingerprint = config.get("contract_fingerprint") if config else None
    sealed_narrative_digest = config.get("narrative_digest") if config else None
    fingerprint_ready = bool(config) and sealed_fingerprint == fingerprint
    if config and not fingerprint_ready:
        policy_issues.append("contract fingerprint drift; re-run interviewer and seal")
    if config and sealed_narrative_digest != narrative_digest:
        message = narrative_drift_message(change_id)
        warnings.append(message)
    irreversible_pending = pending_irreversible_policy(config)
    if irreversible_pending:
        policy_issues.append(
            "unconfirmed irreversible policy: " + ", ".join(irreversible_pending)
        )
    if config is not None:
        apply_runtime_budget_state(repo_root, change_id, config, fingerprint, tasks)
    sealed = not policy_issues
    ready_refs = [task.ref for task in tasks if task.ready]
    terminal_refs = [
        task.ref for task in tasks if task.effective_state in TERMINAL_STATES
    ]
    blocked_refs = [
        task.ref
        for task in tasks
        if not task.completed and not task.ready and task.ref not in terminal_refs
    ]
    candidate_ref = ready_refs[0] if ready_refs else None
    selected_ref = candidate_ref
    # The batch is a census, not an allowance.  Keeping every dependency-ready
    # ref visible lets the supervisor make routing and topology decisions
    # without losing work merely because this invocation dispatches fewer refs.
    selected_batch = list(ready_refs)
    routed_wave, routing = route_ready_wave(
        change_id,
        fingerprint,
        tasks,
        selected_batch,
    )
    wave_ready = fingerprint_ready and not irreversible_pending
    selected_wave = routed_wave if wave_ready else []
    if not wave_ready:
        for route in routing:
            if route["wave_status"] == "selected":
                route["wave_status"] = "latch_blocked"
    budget_snapshot = (
        apply_budget_snapshot(
            repo_root,
            change_id,
            config,
            fingerprint,
            tasks,
            selected_wave,
        )
        if config is not None
        else {"apply_remaining": 0}
    )
    apply_remaining = budget_snapshot["apply_remaining"]
    allowed_parallel_applies = min(len(selected_wave), apply_remaining)
    tasks_by_ref = {task.ref: task for task in tasks}
    dispatch_refs = [
        ref
        for ref in selected_wave
        if (task := tasks_by_ref.get(ref)) is not None
        and task.max_apply_attempts is not None
        and task.apply_attempts_used < task.max_apply_attempts
    ][:allowed_parallel_applies]
    return {
        "schema_version": SCHEMA_PLAN,
        "change_id": change_id,
        "sealed": sealed,
        "contract_fingerprint": fingerprint,
        "sealed_fingerprint": sealed_fingerprint,
        "narrative_digest": narrative_digest,
        "sealed_narrative_digest": sealed_narrative_digest,
        "candidate_ref": candidate_ref,
        "selected_ref": selected_ref,
        "selected_batch": selected_batch,
        "selected_wave": selected_wave,
        "routing": routing,
        "batch_requested": batch,
        **budget_snapshot,
        "allowed_parallel_applies": allowed_parallel_applies,
        "dispatch_refs": dispatch_refs,
        "fingerprint_ready": fingerprint_ready,
        "wave_ready": wave_ready,
        "irreversible_policy_pending": irreversible_pending,
        "ready_refs": ready_refs,
        "blocked_refs": blocked_refs,
        "terminal_refs": terminal_refs,
        "issues": contract_issues + policy_issues,
        "warnings": warnings,
        "tasks": [task.to_plan_item() for task in tasks],
    }


def cmd_plan(args: argparse.Namespace) -> int:
    payload = build_plan_payload(
        args.repo_root.resolve(),
        args.change_id,
        advisory=args.advisory,
        batch=args.batch,
    )
    emit_json(payload)
    return 0


def cmd_goals(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    tasks_path, feature_path = contract_paths(repo_root, args.change_id)
    if not tasks_path.exists() or not feature_path.exists():
        emit_json(
            {
                "schema_version": SCHEMA_GOALS,
                "change_id": args.change_id,
                "ok": False,
                "goals": [],
                "orphan_refs": [],
                "issues": [f"missing contract artifacts for change `{args.change_id}`"],
            }
        )
        return 2

    tasks_text = read_utf8(tasks_path)
    tasks = parse_task_file(tasks_text)
    build_dependency_graph(tasks, load_feature_map(read_utf8(feature_path)))
    goals = parse_goal_blocks(tasks_text)
    resolve_goal_coverage(goals, tasks)

    covered_refs = {ref for goal in goals for ref in goal.covered_by}
    # With no goal declared every ref would be an orphan, which is noise rather
    # than a finding, so orphans are only meaningful once goals exist.
    orphan_refs = (
        [
            task.ref
            for task in tasks
            if task.ref not in covered_refs
            and task.effective_state != "superseded"
        ]
        if goals
        else []
    )
    issues = [
        f"{goal.goal_id}: {issue}" for goal in goals for issue in goal.issues
    ]
    uncovered = [goal.goal_id for goal in goals if not goal.to_item()["covered"]]
    emit_json(
        {
            "schema_version": SCHEMA_GOALS,
            "change_id": args.change_id,
            "ok": not issues and not uncovered,
            "goal_count": len(goals),
            "uncovered_goals": uncovered,
            "goals": [goal.to_item() for goal in goals],
            "orphan_refs": orphan_refs,
            "issues": issues,
        }
    )
    return 0 if not issues and not uncovered else 2


def load_goal_observations(path: Path | None) -> dict[str, dict[str, Any]]:
    """Read the verifier's per-goal observations.

    The tool decides structural completeness; it cannot judge whether a product
    outcome matches a goal, so that judgment is supplied rather than inferred.
    """
    if path is None:
        return {}
    payload = json.loads(read_utf8(path))
    if not isinstance(payload, dict):
        raise ValueError("observation file must be a JSON object keyed by goal id")
    observations: dict[str, dict[str, Any]] = {}
    for goal_id, entry in payload.items():
        if not isinstance(entry, dict):
            raise ValueError(f"observation for `{goal_id}` must be an object")
        status = str(entry.get("status", "")).strip().lower()
        if status not in OBSERVATION_STATES:
            raise ValueError(
                f"observation for `{goal_id}` must set status to match or mismatch"
            )
        observations[goal_id] = {
            "status": status,
            "observed": entry.get("observed"),
            "evidence": entry.get("evidence") or [],
        }
    return observations


def assess_goal(
    goal: GoalEntry,
    observation: dict[str, Any] | None,
) -> dict[str, Any]:
    item = goal.to_item()
    reasons: list[str] = list(goal.issues)
    blocking_refs = [
        ref for ref in goal.live_refs if goal.states.get(ref) != "passed"
    ]
    if not item["covered"]:
        reasons.append("goal is not covered by a live ref")
    if blocking_refs:
        reasons.append("covering refs are not passed: " + ", ".join(blocking_refs))
    if observation is None:
        status = "unobserved"
        reasons.append("no verifier observation was supplied")
    elif observation["status"] == "mismatch":
        status = "mismatch"
        reasons.append("observed outcome does not match ACCEPT")
    else:
        status = "match"
    return {
        "goal_id": goal.goal_id,
        "status": "match" if status == "match" and not reasons else status,
        "expected": goal.accept_raw,
        "observed": (observation or {}).get("observed"),
        "evidence": (observation or {}).get("evidence", []),
        "covered_by": goal.covered_by,
        "live_refs": goal.live_refs,
        "blocking_refs": blocking_refs,
        "reasons": reasons,
    }


def build_revision_proposal(
    change_id: str,
    assessment: dict[str, Any],
) -> dict[str, Any]:
    proposed: list[dict[str, Any]] = []
    # `test` is deliberately left null: a proposal may not manufacture an
    # executable obligation, so applying it requires the supervisor to supply one.
    if not assessment["live_refs"]:
        proposed.append(
            {
                "action": "add",
                "title": f"cover design goal {assessment['goal_id']}",
                "depends_on": [],
                "accept": assessment["expected"],
                "test": None,
            }
        )
    elif assessment["status"] == "mismatch":
        for ref in assessment["live_refs"]:
            proposed.append(
                {
                    "action": "supersede",
                    "ref": ref,
                    "reason": "observed outcome does not match the goal ACCEPT",
                }
            )
        proposed.append(
            {
                "action": "add",
                "title": f"re-cover design goal {assessment['goal_id']}",
                "depends_on": [],
                "accept": assessment["expected"],
                "test": None,
            }
        )
    return {
        "schema": SCHEMA_REVISION_PROPOSAL,
        "change_id": change_id,
        "goal": assessment["goal_id"],
        "expected": assessment["expected"],
        "observed": assessment["observed"],
        "evidence": assessment["evidence"],
        "reasons": assessment["reasons"],
        "proposed": proposed,
    }


def cmd_design_verify(args: argparse.Namespace) -> int:
    repo_root, config, fingerprint, _, _, warnings = resolve_runtime_policy(args)
    payload = build_plan_payload(repo_root, args.change_id, advisory=False)
    tasks_path, feature_path = contract_paths(repo_root, args.change_id)
    tasks_text = read_utf8(tasks_path)
    tasks = parse_task_file(tasks_text)
    build_dependency_graph(tasks, load_feature_map(read_utf8(feature_path)))
    goals = parse_goal_blocks(tasks_text)
    resolve_goal_coverage(goals, tasks)
    observations = load_goal_observations(args.observation)

    assessments = [assess_goal(goal, observations.get(goal.goal_id)) for goal in goals]
    blockers: list[str] = []
    if not goals:
        blockers.append("the registry declares no design goal")
    if payload["ready_refs"]:
        blockers.append("ready work remains: " + ", ".join(payload["ready_refs"]))
    gapped = [item for item in assessments if item["status"] != "match"]
    verdict = "PASS" if not gapped and not blockers else "GAP"

    proposals = [build_revision_proposal(args.change_id, item) for item in gapped]
    written: list[str] = []
    if args.write_proposal and proposals:
        target_dir = repo_root / "openspec" / "changes" / args.change_id / "unblock"
        stamp = utc_now().replace(":", "").replace("-", "")
        for proposal in proposals:
            path = target_dir / f"revision-{proposal['goal']}-{stamp}.json"
            write_json_atomic(path, proposal)
            written.append(str(path.relative_to(repo_root).as_posix()))

    emit_json(
        {
            "schema_version": SCHEMA_DESIGN_VERIFY,
            "change_id": args.change_id,
            "verdict": verdict,
            "contract_fingerprint": fingerprint,
            "retention": config["retention"],
            "autonomy": autonomy_of(config),
            "ready_refs": payload["ready_refs"],
            "blockers": blockers,
            "goals": assessments,
            "revision_proposals": proposals,
            "written": written,
            "warnings": warnings,
        }
    )
    return 0 if verdict == "PASS" else 2


def next_task_slots(tasks: list[TaskEntry]) -> tuple[str, str]:
    minors: list[int] = []
    ref_numbers: list[int] = []
    for task in tasks:
        parts = task.task_id.split(".")
        if parts[-1].isdigit():
            minors.append(int(parts[-1]))
        digits = re.sub(r"\D", "", task.ref)
        if digits:
            ref_numbers.append(int(digits))
    section = tasks[0].task_id.split(".")[0] if tasks else "1"
    return (
        f"{section}.{max(minors, default=0) + 1}",
        f"R{max(ref_numbers, default=0) + 1}",
    )


def render_task_block(
    *,
    task_id: str,
    ref: str,
    title: str,
    depends_on: list[str],
    supersedes: list[str],
    accept: str,
    test_commands: list[str],
) -> str:
    lines = [f"- [ ] {task_id} {title} [#{ref}]"]
    lines.append(
        "  - DEPENDS_ON: " + (", ".join(depends_on) if depends_on else "none")
    )
    if supersedes:
        lines.append("  - SUPERSEDES: " + ", ".join(supersedes))
    lines.append(f"  - ACCEPT: {accept}")
    lines.append("  - TEST: SCOPE: CLI")
    for command in test_commands:
        lines.append(f"    - Run: `{command}`")
    lines.append("    - Verify: exit 0")
    return "\n".join(lines) + "\n"


def registry_insertion_point(lines: list[str], tasks: list[TaskEntry]) -> int:
    """Insert after the last task, before whatever section follows it."""
    last_task_line = max(task.line_number for task in tasks)
    insert_at = len(lines)
    for index in range(last_task_line, len(lines)):
        if HEADING_RE.match(lines[index]):
            insert_at = index
            break
    while insert_at > 0 and not lines[insert_at - 1].strip():
        insert_at -= 1
    return insert_at


def normalize_proposal_entries(
    proposal: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    additions: list[dict[str, Any]] = []
    supersedes: list[str] = []
    issues: list[str] = []
    entries = proposal.get("proposed")
    if not isinstance(entries, list) or not entries:
        return [], [], ["proposal has no `proposed` entries"]

    for entry in entries:
        if not isinstance(entry, dict):
            issues.append("each proposed entry must be an object")
            continue
        action = str(entry.get("action", "")).strip().lower()
        if action == "supersede":
            ref = entry.get("ref")
            if not isinstance(ref, str) or not ref:
                issues.append("a supersede entry requires `ref`")
            else:
                supersedes.append(ref)
            continue
        if action != "add":
            issues.append(f"unsupported proposed action `{action}`")
            continue
        title = entry.get("title")
        accept = entry.get("accept")
        test_commands = entry.get("test")
        if not isinstance(title, str) or not title.strip():
            issues.append("an add entry requires `title`")
        if not isinstance(accept, str) or not accept.strip():
            issues.append("an add entry requires `accept`")
        if not isinstance(test_commands, list) or not test_commands:
            issues.append(
                "an add entry requires an executable `test`; a proposal may not "
                "manufacture an empty obligation"
            )
        if issues:
            continue
        additions.append(
            {
                "title": title.strip(),
                "accept": accept.strip(),
                "test": [str(command) for command in test_commands],
                "depends_on": [
                    str(ref) for ref in (entry.get("depends_on") or [])
                ],
            }
        )

    if supersedes and not additions:
        issues.append("a supersede entry requires a replacement add entry")
    return additions, supersedes, issues


def strict_validation_state(repo_root: Path, change_id: str) -> tuple[str, str]:
    # Outside an OpenSpec project the validator cannot resolve the change at
    # all, which is a missing tool rather than an invalid amendment.
    if not (repo_root / "openspec" / "project.md").is_file():
        return "unavailable", "no OpenSpec project at the repository root"
    try:
        completed = subprocess.run(
            ["openspec", "validate", change_id, "--strict"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            shell=sys.platform == "win32",
        )
    except OSError as exc:
        return "unavailable", str(exc)
    if completed.returncode != 0:
        return "failed", (completed.stdout or completed.stderr).strip()
    return "passed", ""


def apply_revision_failure(change_id: str, issues: list[str], **extra: Any) -> int:
    emit_json(
        {
            "schema_version": SCHEMA_APPLY_REVISION,
            "change_id": change_id,
            "applied": False,
            "issues": issues,
            **extra,
        }
    )
    return 2


def cmd_apply_revision(args: argparse.Namespace) -> int:
    repo_root, config, fingerprint, ledger_path, budgets, warnings = (
        resolve_runtime_policy(args)
    )
    autonomy = autonomy_of(config)
    if autonomy != "full_auto":
        return apply_revision_failure(
            args.change_id,
            [
                f"autonomy `{autonomy}` may not self-amend; hand the proposal to "
                f"`$openspec-change-interviewer {args.change_id}`"
            ],
            autonomy=autonomy,
        )

    ceiling = hard_ceiling_of(config)
    ledger = load_or_init_ledger(ledger_path, args.change_id)
    episode, run = load_run(ledger, args.change_id, fingerprint, args.run_id)
    stop_reasons = gate_reasons(
        ledger,
        episode,
        run,
        None,
        None,
        budgets=budgets,
        hard_ceiling=ceiling,
        task_budget=None,
        current_allocated_subagents=None,
        prospective_subagent_ids=[],
    )
    if stop_reasons:
        return apply_revision_failure(
            args.change_id,
            ["gate reports stop: " + ", ".join(stop_reasons)],
            autonomy=autonomy,
            hard_ceiling=ceiling,
            terminal=any(
                reason.startswith("hard_ceiling_") for reason in stop_reasons
            ),
        )

    proposal = json.loads(read_utf8(args.proposal))
    if not isinstance(proposal, dict) or proposal.get("schema") != SCHEMA_REVISION_PROPOSAL:
        return apply_revision_failure(
            args.change_id,
            [f"proposal must declare schema `{SCHEMA_REVISION_PROPOSAL}`"],
            autonomy=autonomy,
        )
    additions, supersedes, issues = normalize_proposal_entries(proposal)
    if issues:
        return apply_revision_failure(args.change_id, issues, autonomy=autonomy)

    tasks_path, feature_path = contract_paths(repo_root, args.change_id)
    original_tasks = tasks_path.read_bytes()
    original_features = feature_path.read_bytes() if feature_path.exists() else None
    tasks = parse_task_file(original_tasks.decode("utf-8"))
    if not tasks:
        return apply_revision_failure(
            args.change_id, ["no active task registry to amend"], autonomy=autonomy
        )

    lines = original_tasks.decode("utf-8").splitlines(keepends=True)
    insert_at = registry_insertion_point(lines, tasks)
    known_refs = {task.ref for task in tasks}
    unknown_supersedes = sorted(set(supersedes) - known_refs)
    if unknown_supersedes:
        return apply_revision_failure(
            args.change_id,
            ["unknown supersede ref: " + ", ".join(unknown_supersedes)],
            autonomy=autonomy,
        )

    rendered: list[str] = []
    added_refs: list[str] = []
    synthetic = list(tasks)
    for index, addition in enumerate(additions):
        task_id, ref = next_task_slots(synthetic)
        rendered.append(
            render_task_block(
                task_id=task_id,
                ref=ref,
                title=addition["title"],
                depends_on=addition["depends_on"],
                supersedes=supersedes if index == 0 else [],
                accept=addition["accept"],
                test_commands=addition["test"],
            )
        )
        added_refs.append(ref)
        synthetic.append(
            TaskEntry(
                order=len(synthetic) + 1,
                line_number=0,
                task_id=task_id,
                ref=ref,
                title=addition["title"],
                checked=False,
            )
        )

    chunk = "\n" + "\n".join(rendered)
    if insert_at < len(lines):
        chunk += "\n"
    lines.insert(insert_at, chunk)
    tasks_path.write_bytes("".join(lines).encode("utf-8"))

    try:
        regenerate_feature_index(repo_root, args.change_id)
        state, detail = strict_validation_state(repo_root, args.change_id)
        if state == "failed":
            raise ValueError(f"strict validation failed: {detail}")
        reseal = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--repo-root",
                str(repo_root),
                "reseal",
                args.change_id,
                "--allow-semantic-change",
                "--reason",
                args.reason or f"revision proposal for goal {proposal.get('goal')}",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if reseal.returncode != 0:
            raise ValueError(f"reseal refused the amendment: {reseal.stdout.strip()}")
        resealed = json.loads(reseal.stdout)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        tasks_path.write_bytes(original_tasks)
        if original_features is None:
            feature_path.unlink()
        else:
            feature_path.write_bytes(original_features)
        return apply_revision_failure(args.change_id, [str(exc)], autonomy=autonomy)

    after = build_plan_payload(repo_root, args.change_id, advisory=False, batch=True)
    emit_json(
        {
            "schema_version": SCHEMA_APPLY_REVISION,
            "change_id": args.change_id,
            "applied": True,
            "autonomy": autonomy,
            "goal": proposal.get("goal"),
            "added_refs": added_refs,
            "superseded_refs": supersedes,
            "strict_validation": state,
            "prior_contract_fingerprint": fingerprint,
            "contract_fingerprint": resealed["contract_fingerprint"],
            "hard_ceiling": resealed["hard_ceiling"],
            "self_extensions_used": resealed["self_extensions_used"],
            "ready_refs": after["ready_refs"],
            "selected_batch": after["selected_batch"],
            "issues": after["issues"],
            "warnings": warnings,
        }
    )
    return 0 if not after["issues"] else 2


def cmd_check(args: argparse.Namespace) -> int:
    payload = build_plan_payload(args.repo_root.resolve(), args.change_id, advisory=False)
    result = {
        "schema_version": SCHEMA_CHECK,
        **{key: value for key, value in payload.items() if key != "schema_version"},
        "ok": payload["sealed"] and not payload["issues"],
    }
    emit_json(result)
    return 0 if result["ok"] else 2


SEAL_PATH_ARGS = (
    ("ledger", "ledger_path"),
    ("scratch", "scratch_root"),
    ("product", "product_root"),
    ("bundle", "bundle_root"),
    ("gui_colab", "gui_colab_root"),
)
SEAL_CEILING_ARGS = (
    ("max_active_minutes", "hard_ceiling_max_active_minutes"),
    ("max_self_extensions", "hard_ceiling_max_self_extensions"),
)
SEAL_BUDGET_ARGS = (
    ("task", "max_apply_attempts", "max_apply_attempts"),
    ("task", "max_unblock_runs", "max_unblock_runs"),
    ("revision", "max_explore_runs", "max_explore_runs"),
    ("revision", "max_subagents", "max_subagents"),
    ("revision", "max_active_minutes", "max_active_minutes"),
    ("change", "max_revisions", "max_revisions"),
    ("change", "max_active_minutes", "max_total_active_minutes"),
)


def inherit_value(supplied: Any, prior: Any, default: Any) -> Any:
    """Resolve a sealed field as supplied, then prior seal, then built-in default.

    An omitted flag must never silently revert a confirmed policy value.
    """
    if supplied is not None:
        return supplied
    if prior is not None:
        return prior
    return default


def seal_changed_fields(prior: dict[str, Any], config: dict[str, Any]) -> list[str]:
    if not prior:
        return []
    changed: list[str] = []
    for field_name in ("retention", "narrative_policy", "autonomy"):
        if prior.get(field_name) != config[field_name]:
            changed.append(field_name)
    prior_ceiling = (
        prior.get("hard_ceiling") if isinstance(prior.get("hard_ceiling"), dict) else {}
    )
    config_ceiling = (
        config.get("hard_ceiling") if isinstance(config.get("hard_ceiling"), dict) else {}
    )
    for key, _ in SEAL_CEILING_ARGS:
        if prior_ceiling.get(key) != config_ceiling.get(key):
            changed.append(f"hard_ceiling.{key}")
    prior_paths = prior.get("paths") if isinstance(prior.get("paths"), dict) else {}
    for key, _ in SEAL_PATH_ARGS:
        if prior_paths.get(key) != config["paths"][key]:
            changed.append(f"paths.{key}")
    prior_budgets = (
        prior.get("budgets") if isinstance(prior.get("budgets"), dict) else {}
    )
    for scope, field_name, _ in SEAL_BUDGET_ARGS:
        prior_scope = (
            prior_budgets.get(scope)
            if isinstance(prior_budgets.get(scope), dict)
            else {}
        )
        if prior_scope.get(field_name) != config["budgets"][scope][field_name]:
            changed.append(f"budgets.{scope}.{field_name}")
    return changed


def cmd_seal(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    if not args.confirmed:
        emit_json(
            {
                "schema_version": SCHEMA_LOOP,
                "change_id": args.change_id,
                "written": False,
                "issues": ["--confirmed is required after user boundary confirmation"],
            }
        )
        return 2

    preflight = build_plan_payload(repo_root, args.change_id, advisory=True)
    contract_issues = [
        issue
        for issue in preflight["issues"]
        if "loop.json" not in issue
        and "not sealed" not in issue
        and "fingerprint drift" not in issue
        and "narrative drift" not in issue
    ]
    if contract_issues or preflight["contract_fingerprint"] is None:
        emit_json(
            {
                "schema_version": SCHEMA_LOOP,
                "change_id": args.change_id,
                "written": False,
                "issues": contract_issues or preflight["issues"],
            }
        )
        return 2

    prior = load_loop_config(repo_root, args.change_id) or {}
    prior_paths = prior.get("paths") if isinstance(prior.get("paths"), dict) else {}
    prior_budgets = (
        prior.get("budgets") if isinstance(prior.get("budgets"), dict) else {}
    )

    paths = {
        key: inherit_value(getattr(args, attribute), prior_paths.get(key), None)
        for key, attribute in SEAL_PATH_ARGS
    }
    retention = inherit_value(args.retention, prior.get("retention"), None)
    missing_first_seal = [
        message
        for flag, value, message in (
            ("--retention", retention, "--retention is required for a first seal"),
            (
                "--ledger-path",
                paths["ledger"],
                "--ledger-path is required for a first seal",
            ),
        )
        if not value
    ]
    if missing_first_seal:
        emit_json(
            {
                "schema_version": SCHEMA_LOOP,
                "change_id": args.change_id,
                "written": False,
                "issues": missing_first_seal,
            }
        )
        return 2
    if retention == "full" and not paths["bundle"]:
        emit_json(
            {
                "schema_version": SCHEMA_LOOP,
                "change_id": args.change_id,
                "written": False,
                "issues": ["full retention requires --bundle-root"],
            }
        )
        return 2

    budgets: dict[str, dict[str, int]] = {"task": {}, "revision": {}, "change": {}}
    change_defaults = DEFAULT_BUDGETS["change"].copy()
    change_defaults["max_active_minutes"] = max(
        change_defaults["max_active_minutes"], 10 * len(preflight["tasks"])
    )
    for scope, field_name, attribute in SEAL_BUDGET_ARGS:
        prior_scope = (
            prior_budgets.get(scope) if isinstance(prior_budgets.get(scope), dict) else {}
        )
        fallback = (
            change_defaults[field_name]
            if scope == "change"
            else DEFAULT_BUDGETS[scope][field_name]
        )
        budgets[scope][field_name] = inherit_value(
            getattr(args, attribute),
            prior_scope.get(field_name),
            fallback,
        )
    invalid_budget_paths = [
        f"{scope}.{key}"
        for scope, values in budgets.items()
        for key, value in values.items()
        if value <= 0
    ]
    if invalid_budget_paths:
        emit_json(
            {
                "schema_version": SCHEMA_LOOP,
                "change_id": args.change_id,
                "written": False,
                "issues": [
                    "budgets must be positive: " + ", ".join(invalid_budget_paths)
                ],
            }
        )
        return 2
    narrative_policy = inherit_value(
        args.narrative_policy,
        prior.get("narrative_policy"),
        "advisory",
    )
    autonomy = inherit_value(args.autonomy, prior.get("autonomy"), "supervised")
    prior_ceiling = (
        prior.get("hard_ceiling") if isinstance(prior.get("hard_ceiling"), dict) else {}
    )
    ceiling_requested = bool(prior_ceiling) or any(
        getattr(args, attribute_name) is not None
        for _, attribute_name in SEAL_CEILING_ARGS
    )
    hard_ceiling = (
        {
            key: inherit_value(
                getattr(args, attribute),
                prior_ceiling.get(key),
                None,
            )
            for key, attribute in SEAL_CEILING_ARGS
        }
        if ceiling_requested
        else {}
    )
    hard_ceiling = {key: value for key, value in hard_ceiling.items() if value is not None}
    if hard_ceiling:
        invalid_ceiling_paths = [
            key
            for key, value in hard_ceiling.items()
            if not isinstance(value, int) or value <= 0
        ]
        if invalid_ceiling_paths:
            emit_json(
                {
                    "schema_version": SCHEMA_LOOP,
                    "change_id": args.change_id,
                    "written": False,
                    "issues": [
                        "hard_ceiling must be positive: "
                        + ", ".join(sorted(invalid_ceiling_paths))
                    ],
                }
            )
            return 2
        contradictions = [
            f"budgets.change.{budget_key} exceeds hard_ceiling.{ceiling_key}"
            for ceiling_key, budget_key in HARD_CEILING_BUDGET_LINKS
            if ceiling_key in hard_ceiling
            and budgets["change"][budget_key] > hard_ceiling[ceiling_key]
        ]
        if contradictions:
            emit_json(
                {
                    "schema_version": SCHEMA_LOOP,
                    "change_id": args.change_id,
                    "written": False,
                    "issues": contradictions,
                }
            )
            return 2
    prior_per_ref = (
        prior.get("per_ref_budgets")
        if isinstance(prior.get("per_ref_budgets"), dict)
        else {}
    )
    per_ref_budgets: dict[str, dict[str, int]] = {}
    for task in preflight["tasks"]:
        ref = task["ref"]
        task_budget = budgets["task"].copy()
        prior_task_budget = prior_per_ref.get(ref)
        if isinstance(prior_task_budget, dict):
            for key in task_budget:
                value = prior_task_budget.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    task_budget[key] = value
        if args.max_apply_attempts is not None:
            task_budget["max_apply_attempts"] = args.max_apply_attempts
        if args.max_unblock_runs is not None:
            task_budget["max_unblock_runs"] = args.max_unblock_runs
        per_ref_budgets[ref] = task_budget
    config = {
        "schema_version": SCHEMA_LOOP,
        "change_id": args.change_id,
        "contract_fingerprint": preflight["contract_fingerprint"],
        "narrative_digest": preflight["narrative_digest"],
        "narrative_policy": narrative_policy,
        "sealed": True,
        "autonomy": autonomy,
        "retention": retention,
        "paths": paths,
        "budgets": budgets,
        "per_ref_budgets": per_ref_budgets,
        "test_profiles": DEFAULT_TEST_PROFILES.copy(),
        "confirmed_at": args.confirmed_at or utc_now(),
    }
    if hard_ceiling:
        config["hard_ceiling"] = hard_ceiling
    changed_fields = seal_changed_fields(prior, config)
    semantic_change = bool(prior) and (
        prior.get("contract_fingerprint") != config["contract_fingerprint"]
    )
    path = loop_config_path(repo_root, args.change_id)
    write_json_atomic(path, config)
    emit_json(
        {
            "schema_version": SCHEMA_LOOP,
            "change_id": args.change_id,
            "written": True,
            "path": str(path),
            "inherited_from_prior_seal": bool(prior),
            "semantic_change": semantic_change,
            "changed_fields": changed_fields,
            "budget_advisories": [],
            "contract_fingerprint": config["contract_fingerprint"],
            "narrative_digest": config["narrative_digest"],
        }
    )
    return 0


CHANGE_BUDGET_OVERRIDES = (
    ("set_max_revisions", "max_revisions"),
    ("set_max_total_active_minutes", "max_active_minutes"),
)


def reseal_failure(change_id: str, issues: list[str], **extra: Any) -> int:
    emit_json(
        {
            "schema_version": SCHEMA_LOOP,
            "change_id": change_id,
            "written": False,
            "issues": issues,
            **extra,
        }
    )
    return 2


def cmd_reseal(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    change_id = args.change_id
    path = loop_config_path(repo_root, change_id)
    raw_config = None
    if path.exists():
        raw_config = json.loads(read_utf8(path))
        if not isinstance(raw_config, dict):
            return reseal_failure(change_id, ["loop.json must be a JSON object"])
    config = sanitize_loop_config(raw_config) if raw_config is not None else None
    if config is None:
        return reseal_failure(
            change_id,
            [
                "no loop.json to reseal; run "
                f"`python scripts/openspec_loop.py seal {change_id} --confirmed ...`"
            ],
        )

    schema_version = config.get("schema_version")
    legacy = schema_version == LEGACY_SCHEMA_LOOP
    if legacy and not args.migrate:
        return reseal_failure(
            change_id,
            [f"loop.json uses {LEGACY_SCHEMA_LOOP}; re-run reseal with --migrate"],
        )
    if not legacy and schema_version != SCHEMA_LOOP:
        return reseal_failure(change_id, [f"unsupported loop schema `{schema_version}`"])

    semantic = compute_semantic_fingerprint(repo_root, change_id)
    narrative = compute_narrative_digest(repo_root, change_id)

    # A v2 fingerprint hashed four whole files, so it can never equal a v3
    # semantic hash. Only a v3 config can report a real semantic change.
    autonomy = autonomy_of(config)
    semantic_change = not legacy and config.get("contract_fingerprint") != semantic
    if semantic_change and not (
        args.allow_semantic_change and (args.confirmed or autonomy == "full_auto")
    ):
        return reseal_failure(
            change_id,
            [
                "semantic contract change detected; run "
                f"`$openspec-change-interviewer {change_id}` or pass "
                "--allow-semantic-change --confirmed"
            ],
            semantic_change=True,
            autonomy=autonomy,
        )
    if semantic_change and not args.confirmed and not args.reason:
        return reseal_failure(
            change_id,
            ["autonomy `full_auto` requires --reason for a semantic amendment"],
            semantic_change=True,
            autonomy=autonomy,
        )

    updated = json.loads(json.dumps(config, ensure_ascii=False))
    updated["schema_version"] = SCHEMA_LOOP
    updated["contract_fingerprint"] = semantic
    updated["narrative_digest"] = narrative
    updated["narrative_policy"] = args.narrative_policy or narrative_policy_of(config)

    changed_fields: list[str] = []
    change_budgets = updated.setdefault("budgets", {}).setdefault("change", {})
    requested: dict[str, int] = {}
    for attribute, field_name in CHANGE_BUDGET_OVERRIDES:
        value = getattr(args, attribute, None)
        if value is None:
            continue
        if value <= 0:
            return reseal_failure(
                change_id, [f"budgets must be positive: change.{field_name}"]
            )
        requested[field_name] = value

    extension = {
        field: value
        for field, value in requested.items()
        if change_budgets.get(field) != value
    }
    prior_ceiling = (
        config.get("hard_ceiling")
        if isinstance(config.get("hard_ceiling"), dict)
        else None
    )
    if extension:
        if not args.reason:
            return reseal_failure(
                change_id, ["a budget extension requires --reason"], autonomy=autonomy
            )
        if autonomy != "full_auto" and not args.confirmed:
            return reseal_failure(
                change_id,
                ["autonomy `supervised` requires --confirmed for a budget extension"],
                autonomy=autonomy,
            )
        if prior_ceiling is not None:
            ceiling = hard_ceiling_of(config)
            over_ceiling = [
                f"change.{field} {extension[field]} exceeds "
                f"hard_ceiling.{ceiling_key} {ceiling[ceiling_key]}"
                for ceiling_key, field in HARD_CEILING_BUDGET_LINKS
                if field in extension and extension[field] > ceiling[ceiling_key]
            ]
            if over_ceiling:
                return reseal_failure(
                    change_id,
                    over_ceiling,
                    autonomy=autonomy,
                    hard_ceiling=ceiling,
                )
            used = self_extensions_used(config)
            if used >= ceiling["max_self_extensions"]:
                return reseal_failure(
                    change_id,
                    [
                        "hard_ceiling.max_self_extensions reached: "
                        f"{ceiling['max_self_extensions']}"
                    ],
                    autonomy=autonomy,
                    hard_ceiling=ceiling,
                    self_extensions_used=used,
                )
        previous = {field: change_budgets.get(field) for field in sorted(extension)}
        for field in sorted(extension):
            change_budgets[field] = extension[field]
            changed_fields.append(f"change.{field}")
        updated.setdefault("budget_extensions", []).append(
            {
                "recorded_at_utc": utc_now(),
                "scope": "change",
                "fields": sorted(extension),
                "from": previous,
                "to": {field: extension[field] for field in sorted(extension)},
                "autonomy": autonomy,
                "confirmed": bool(args.confirmed),
                "reason": args.reason,
            }
        )
    if args.narrative_policy and args.narrative_policy != narrative_policy_of(config):
        changed_fields.append("narrative_policy")

    updated["autonomy"] = autonomy
    if prior_ceiling is None:
        updated.pop("hard_ceiling", None)
    else:
        updated["hard_ceiling"] = {
            key: prior_ceiling[key]
            for key, _ in SEAL_CEILING_ARGS
            if isinstance(prior_ceiling.get(key), int)
            and not isinstance(prior_ceiling.get(key), bool)
            and prior_ceiling[key] > 0
        }
        if not updated["hard_ceiling"]:
            updated.pop("hard_ceiling", None)

    post_issues = loop_config_issues(updated, change_id)
    if post_issues:
        return reseal_failure(change_id, post_issues)

    restamp_plan = build_plan_payload(repo_root, change_id, advisory=True)
    updated["resealed_at"] = utc_now()
    write_json_atomic(path, updated)
    emit_json(
        {
            "schema_version": SCHEMA_LOOP,
            "change_id": change_id,
            "written": True,
            "path": str(path),
            "migrated": legacy,
            "semantic_change": semantic_change,
            "revision_charged": False,
            "changed_fields": changed_fields,
            "budget_advisories": [],
            "autonomy": updated["autonomy"],
            "hard_ceiling": updated.get("hard_ceiling"),
            "self_extensions_used": self_extensions_used(updated),
            "contract_fingerprint": semantic,
            "narrative_digest": narrative,
        }
    )
    return 0


def regenerate_feature_index(repo_root: Path, change_id: str) -> None:
    """Rebuild the compact index with the canonical generator.

    Importing it keeps one `tasks.md` parser instead of a second copy inside
    this helper that could drift from the registry schema.
    """
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import generate_openspec_feature_list as generator

    generator.generate(repo_root, change_id)


def recorded_verify_pass(ledger: dict[str, Any], fingerprint: str, ref: str) -> bool:
    for episode in ledger.get("episodes", []):
        if episode.get("contract_fingerprint") != fingerprint:
            continue
        for attempt in attempts_for_episode(episode):
            if (
                attempt.get("ref") == ref
                and attempt.get("kind") == "verify"
                and attempt.get("result") in {"pass", "success"}
            ):
                return True
    return False


def promote_failure(change_id: str, issues: list[str], **extra: Any) -> int:
    emit_json(
        {
            "schema_version": SCHEMA_PROMOTE,
            "change_id": change_id,
            "promoted": False,
            "issues": issues,
            **extra,
        }
    )
    return 2


def cmd_promote(args: argparse.Namespace) -> int:
    repo_root, _, fingerprint, ledger_path, _, warnings = resolve_runtime_policy(args)
    before = build_plan_payload(repo_root, args.change_id, advisory=False)
    if not before["sealed"] or before["issues"]:
        return promote_failure(
            args.change_id,
            before["issues"] or ["contract is not sealed"],
            warnings=warnings,
        )

    selected = next(
        (task for task in before["tasks"] if task["ref"] == args.ref), None
    )
    if selected is None:
        return promote_failure(args.change_id, [f"unknown ref `{args.ref}`"])
    if selected["effective_state"] != "ready":
        return promote_failure(
            args.change_id,
            [f"{args.ref} is `{selected['effective_state']}`, not `ready`"],
        )
    if not recorded_verify_pass(
        load_or_init_ledger(ledger_path, args.change_id), fingerprint, args.ref
    ):
        return promote_failure(
            args.change_id,
            [f"no recorded verifier pass for {args.ref} in the current episode"],
        )

    tasks_path, feature_path = contract_paths(repo_root, args.change_id)
    original_tasks = tasks_path.read_bytes()
    original_features = feature_path.read_bytes() if feature_path.exists() else None
    # Decode without universal-newline translation so an untouched line stays
    # byte-identical, whatever the checkout's line endings are.
    lines = original_tasks.decode("utf-8").splitlines(keepends=True)
    index = selected["line_number"] - 1
    promoted_line = CHECKBOX_MARK_RE.sub(r"\1x\2", lines[index], count=1)
    if promoted_line == lines[index]:
        return promote_failure(
            args.change_id,
            [f"line {selected['line_number']} is not a promotable checkbox"],
        )
    lines[index] = promoted_line
    tasks_path.write_bytes("".join(lines).encode("utf-8"))

    try:
        if compute_semantic_fingerprint(repo_root, args.change_id) != fingerprint:
            raise ValueError("promotion would change the sealed obligations")
        regenerate_feature_index(repo_root, args.change_id)
        after = build_plan_payload(repo_root, args.change_id, advisory=False)
        if after["issues"]:
            raise ValueError("; ".join(after["issues"]))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        tasks_path.write_bytes(original_tasks)
        if original_features is None:
            feature_path.unlink()
        else:
            feature_path.write_bytes(original_features)
        return promote_failure(args.change_id, [str(exc)])

    emit_json(
        {
            "schema_version": SCHEMA_PROMOTE,
            "change_id": args.change_id,
            "promoted": True,
            "ref": args.ref,
            "contract_fingerprint": fingerprint,
            "issues": [],
            "warnings": warnings,
            "ready_refs": after["ready_refs"],
            "selected_ref": after["selected_ref"],
            "terminal_refs": after["terminal_refs"],
        }
    )
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    repo_root, _, fingerprint, _, _, warnings = resolve_runtime_policy(args)
    tasks_path, _ = contract_paths(repo_root, args.change_id)
    before_tasks = tasks_path.read_bytes()
    regenerate_feature_index(repo_root, args.change_id)
    if tasks_path.read_bytes() != before_tasks:
        raise ValueError("sync must not modify tasks.md")
    payload = build_plan_payload(repo_root, args.change_id, advisory=False)
    emit_json(
        {
            "schema_version": SCHEMA_SYNC,
            "change_id": args.change_id,
            "ok": payload["sealed"] and not payload["issues"],
            "contract_fingerprint": fingerprint,
            "issues": payload["issues"],
            "warnings": warnings + payload["warnings"],
            "ready_refs": payload["ready_refs"],
            "selected_ref": payload["selected_ref"],
        }
    )
    return 0 if payload["sealed"] and not payload["issues"] else 2


def canonical_apply_status(result: str) -> str:
    normalized = result.strip().lower()
    if normalized in {"completed", "success"}:
        return "completed"
    if normalized in {"failed", "failure", "fail"}:
        return "failed"
    if normalized in {"no_progress", "partial"}:
        return "partial"
    if normalized in {"deviated", "unverified"}:
        return "unverified"
    if normalized in {"empty", "blocked"}:
        return normalized
    raise ValueError(f"unsupported canonical Apply result: {result}")


def normalize_record_path(raw: str) -> str:
    normalized = raw.strip().replace("\\", "/").rstrip("/")
    if ".." in normalized.split("/"):
        raise ValueError(f"record path may not traverse its declared scope: {raw}")
    return normalized


def path_within_declared_scope(path: str, scopes: list[str]) -> bool:
    normalized = normalize_record_path(path)
    return any(
        normalized == scope or normalized.startswith(scope + "/")
        for scope in scopes
    )


def path_within_approved_root(repo_root: Path, path: str, root: str) -> bool:
    candidate = Path(path)
    approved = Path(root)
    candidate = candidate if candidate.is_absolute() else repo_root / candidate
    approved = approved if approved.is_absolute() else repo_root / approved
    try:
        candidate.resolve().relative_to(approved.resolve())
    except ValueError:
        return False
    return True


def approved_evidence_root(config: dict[str, Any]) -> str | None:
    profile = config.get("retention_profile")
    if isinstance(profile, dict):
        retained = profile.get("retained")
        if isinstance(retained, str) and retained.strip():
            return retained.strip()
    paths = config.get("paths")
    if isinstance(paths, dict):
        bundle = paths.get("bundle")
        if isinstance(bundle, str) and bundle.strip():
            return bundle.strip()
    return None


def record_task_context(
    repo_root: Path,
    change_id: str,
    ref: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if "," in ref or "，" in ref or ";" in ref:
        raise ValueError("one Apply packet may bind exactly one ref")
    payload = build_plan_payload(repo_root, change_id, advisory=False, batch=True)
    task = next((item for item in payload.get("tasks", []) if item["ref"] == ref), None)
    if task is None:
        raise ValueError(f"unknown active ref `{ref}`")
    route = next(
        (item for item in payload.get("routing", []) if item["ref"] == ref),
        task.get("routing") or {},
    )
    return payload, task, route


def validate_record_writes(
    repo_root: Path,
    config: dict[str, Any],
    task: dict[str, Any],
    role_id: str,
    changed_files: list[str],
) -> None:
    if not changed_files:
        return
    scopes = [normalize_record_path(item) for item in task.get("write_scope", [])]
    if role_id in {"implementer", "test-engineer"} | SUPERVISOR_DIRECT_ROLE_IDS:
        if not scopes:
            raise ValueError(f"task `{task['ref']}` has no declared write_scope")
        outside = [
            item for item in changed_files if not path_within_declared_scope(item, scopes)
        ]
        if outside:
            raise ValueError(
                f"role_write_scope_violation:{role_id}:" + ",".join(outside)
            )
        if role_id == "implementer" and any(is_test_path(item) for item in changed_files):
            raise ValueError("role_write_allowlist:implementer_cannot_write_tests")
        if role_id == "test-engineer" and any(
            not is_test_path(item) for item in changed_files
        ):
            raise ValueError("role_write_allowlist:test_engineer_tests_only")
        return
    if role_id in EVIDENCE_WRITER_ROLE_IDS:
        root = approved_evidence_root(config)
        if root is None:
            raise ValueError("role_write_allowlist:no_approved_evidence_root")
        outside = [
            item
            for item in changed_files
            if not path_within_approved_root(repo_root, item, root)
        ]
        if outside:
            raise ValueError(
                f"role_write_allowlist:evidence_root:" + ",".join(outside)
            )
        return
    raise ValueError(f"role_write_allowlist:{role_id}:read_only")


def cmd_record(args: argparse.Namespace) -> int:
    repo_root, config, fingerprint, ledger_path, budgets, warnings = resolve_runtime_policy(args)
    ledger = load_or_init_ledger(ledger_path, args.change_id)
    episode, run = load_run(ledger, args.change_id, fingerprint, args.run_id)
    revision_attempts = attempts_for_episode(episode)
    task_budget = ref_budget_of(config, budgets, args.ref)
    if args.record_owner != "supervisor":
        raise ValueError("only the supervisor may write canonical Loop records")
    if args.kind == "apply" and any(
        delimiter in args.ref for delimiter in (",", "，", ";")
    ):
        raise ValueError("one Apply packet may bind exactly one ref")
    if args.kind == "apply" and args.result == "pass":
        raise ValueError("an Apply worker may not claim PASS")
    strict_apply = args.kind == "apply" and any(
        (
            args.attempt_id,
            args.packet_id,
            args.role_id,
            args.changed_file,
            args.evidence,
            args.join_id,
            args.wave_ref,
            args.transient_retries,
        )
    )
    payload: dict[str, Any] = {}
    task: dict[str, Any] = {}
    route: dict[str, Any] = {}
    if strict_apply:
        payload, task, route = record_task_context(
            repo_root,
            args.change_id,
            args.ref,
        )
    if args.kind == "unblock":
        if not has_blocking_evidence(revision_attempts, args.ref):
            raise ValueError(f"task_unblock_dormant:{args.ref}")
        if args.disposition is None:
            raise ValueError("unblock records require --disposition")
        prior_unblocks = [
            attempt
            for attempt in revision_attempts
            if attempt.get("ref") == args.ref and attempt.get("kind") == "unblock"
        ]
        max_unblock_runs = task_budget["max_unblock_runs"]
        if len(prior_unblocks) >= max_unblock_runs:
            raise ValueError(
                f"task_unblock_budget_exhausted:{args.ref}:{max_unblock_runs}"
            )
        if len(prior_unblocks) == 1:
            second_unblock_issues = second_unblock_reasons(
                revision_attempts,
                args.ref,
            )
            if second_unblock_issues:
                raise ValueError("; ".join(second_unblock_issues))
            prior_apply_count = apply_iteration_count(
                [attempt for attempt in revision_attempts if attempt.get("ref") == args.ref]
            )
            retry_dispositions = {"retry", "targeted_probe"}
            if (
                prior_apply_count >= task_budget["max_apply_attempts"]
                and args.disposition in retry_dispositions
            ):
                raise ValueError(
                    "second unblock is terminal after the final apply attempt; "
                    "use amend_spec, supersede_task, or stop_budget"
                )
    allocated_subagent_ids = normalize_subagent_ids(args.subagent_id)
    if not allocated_subagent_ids and args.subagents_used:
        existing = len(run.setdefault("allocated_subagent_ids", []))
        allocated_subagent_ids = [
            f"legacy-anon-{args.run_id}-{existing + index + 1}" for index in range(args.subagents_used)
        ]
    runtime_trace_ids = list(allocated_subagent_ids)
    if args.kind in ZERO_APPLY_KINDS:
        allocated_subagent_ids = []

    apply_record_owner = "supervisor" if args.kind == "apply" else None
    attempt_id: str | None = None
    packet_id: str | None = None
    role_id: str | None = None
    join_id: str | None = None
    wave_refs: list[str] = []
    worktree_mode: str | None = None
    terminal_status: str | None = None
    changed_files = [
        normalize_record_path(item)
        for item in args.changed_file
        if item and item.strip()
    ]
    evidence = [item.strip() for item in args.evidence if item and item.strip()]
    if args.kind == "apply":
        prior_apply_count = sum(
            1
            for item in revision_attempts
            if item.get("ref") == args.ref and item.get("kind") == "apply"
        )
        attempt_id = args.attempt_id or (
            f"legacy:{args.run_id}:{args.ref}:{prior_apply_count + 1}"
        )
        if any(
            item.get("kind") == "apply" and item.get("attempt_id") == attempt_id
            for item in revision_attempts
        ):
            raise ValueError(f"duplicate canonical Apply attempt_id `{attempt_id}`")
        expected_packet_id = (
            f"{route.get('package_id')}:{attempt_id}"
            if strict_apply and route.get("package_id")
            else f"{args.change_id}:{args.ref}:{attempt_id}"
        )
        if args.packet_id and args.packet_id != expected_packet_id:
            raise ValueError("Apply packet_id does not match the routed packet snapshot")
        packet_id = expected_packet_id
        if any(
            item.get("kind") == "apply" and item.get("packet_id") == packet_id
            for item in revision_attempts
        ):
            raise ValueError(f"duplicate canonical Apply packet_id `{packet_id}`")
        expected_role_id = route.get("effective_role_id") or LOCAL_SUPERVISOR_ROLE_ID
        if strict_apply and args.role_id and args.role_id != expected_role_id:
            raise ValueError("Apply Role ID does not match the routed packet snapshot")
        role_id = expected_role_id
        if role_id == "general":
            raise ValueError("general is not a deployable Apply Role ID")
        if strict_apply and role_id in READ_ONLY_ROLE_IDS:
            raise ValueError(f"zero-Apply read-only role cannot own Apply: {role_id}")
        if strict_apply:
            validate_record_writes(repo_root, config, task, role_id, changed_files)
        elif changed_files:
            raise ValueError("changed files require an active packet write_scope")
        expected_join_id = route.get("join_id")
        if strict_apply and args.join_id and args.join_id != expected_join_id:
            raise ValueError("Apply join_id does not match the routed packet snapshot")
        join_id = expected_join_id if strict_apply else args.join_id
        worktree_mode = route.get("worktree_mode") or "legacy"
        expected_wave_refs = (
            [args.ref]
            if worktree_mode == "independent"
            else [
                item["ref"]
                for item in payload.get("routing", [])
                if item.get("join_id") == join_id
                and item["ref"] in payload.get("selected_wave", [])
            ]
        )
        if args.wave_ref:
            for wave_ref in args.wave_ref:
                candidate = wave_ref.strip()
                if "," in candidate or "，" in candidate or ";" in candidate:
                    raise ValueError("each --wave-ref must name exactly one ref")
                if candidate and candidate not in wave_refs:
                    wave_refs.append(candidate)
        elif strict_apply and join_id:
            wave_refs = list(expected_wave_refs)
        if not wave_refs:
            wave_refs = [args.ref]
        if args.ref not in wave_refs:
            raise ValueError("Apply wave snapshot must contain its packet ref")
        if strict_apply:
            if wave_refs != expected_wave_refs:
                raise ValueError(
                    "Apply wave_refs must equal the routed join group in ready order"
                )
            known_refs = {item["ref"] for item in payload.get("tasks", [])}
            unknown_wave_refs = [item for item in wave_refs if item not in known_refs]
            if unknown_wave_refs:
                raise ValueError(
                    "unknown Apply wave refs: " + ",".join(unknown_wave_refs)
                )
        terminal_status = canonical_apply_status(args.result)
        if terminal_status not in CANONICAL_APPLY_STATUSES:
            raise ValueError(f"non-terminal Apply result: {terminal_status}")
        if args.transient_retries < 0:
            raise ValueError("--transient-retries must be non-negative")
    elif args.kind == "verify" and args.join_id:
        join_reasons = historical_join_reasons(
            attempts_for_episode(episode),
            args.ref,
            args.join_id,
        )
        if join_reasons:
            raise ValueError("; ".join(join_reasons))

    attempt = {
        "recorded_at_utc": utc_now(),
        "ref": args.ref,
        "kind": args.kind,
        "result": args.result,
        "action": args.action,
        "error_fingerprint": error_fingerprint(args.error_text),
        "result_fingerprint": error_fingerprint(args.observation_text or args.error_text),
        "disposition": args.disposition,
        "allocated_subagent_ids": allocated_subagent_ids,
        "runtime_trace_ids": runtime_trace_ids,
        "duration_seconds": max(args.duration_seconds, 0),
        "apply_record_owner": apply_record_owner,
        "consumes_apply_attempt": args.kind == "apply",
        "consumes_scheduling_headcount": (
            args.kind == "apply"
            and role_id not in SUPERVISOR_DIRECT_ROLE_IDS
            and bool(route.get("agent"))
        ),
        "canonical_apply_record": args.kind == "apply",
        "attempt_id": attempt_id,
        "packet_id": packet_id,
        "role_id": role_id,
        "changed_files": changed_files,
        "evidence": evidence,
        "join_id": join_id,
        "wave_refs": wave_refs,
        "worktree_mode": worktree_mode,
        "terminal_status": terminal_status,
        "transient_retries": args.transient_retries if args.kind == "apply" else 0,
        "auto_redispatch": False if args.kind == "apply" else None,
        "next_apply_owner": "supervisor_gate" if args.kind == "apply" else None,
    }
    run.setdefault("attempts", []).append(attempt)
    run_allocated = run.setdefault("allocated_subagent_ids", [])
    for subagent_id in allocated_subagent_ids:
        if subagent_id not in run_allocated:
            run_allocated.append(subagent_id)
    ledger["updated_at_utc"] = utc_now()
    write_json_atomic(ledger_path, ledger)
    emit_json(
        {
            "schema_version": SCHEMA_LEDGER,
            "ledger_path": str(ledger_path),
            "run_id": args.run_id,
            "contract_fingerprint": fingerprint,
            "apply_record_owner": apply_record_owner,
            "attempt_id": attempt_id,
            "packet_id": packet_id,
            "terminal_status": terminal_status,
            "warnings": warnings,
        }
    )
    return 0


def attempts_for_episode(episode: dict[str, Any]) -> list[dict[str, Any]]:
    ensure_episode_runs(episode)
    return [attempt for run in episode.get("runs", []) for attempt in run.get("attempts", [])]


def attempts_for_change(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        attempt
        for episode in ledger.get("episodes", [])
        for attempt in attempts_for_episode(episode)
    ]


def amendment_chain(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "contract_fingerprint": episode.get("contract_fingerprint"),
            "supersedes_fingerprint": episode.get("supersedes_fingerprint"),
            "attempt_count": len(attempts_for_episode(episode)),
        }
        for episode in ledger.get("episodes", [])
    ]


def productive_revision_count(ledger: dict[str, Any]) -> int:
    """Count revisions that actually executed work.

    A seal that drifted before its first attempt consumed no budget, so it must
    not shorten the change's remaining revisions.
    """
    return sum(
        1 for episode in ledger.get("episodes", []) if attempts_for_episode(episode)
    )


def active_seconds(attempts: list[dict[str, Any]]) -> int:
    return sum(max(int(attempt.get("duration_seconds", 0) or 0), 0) for attempt in attempts)


def apply_iteration_count(attempts: list[dict[str, Any]]) -> int:
    """Count only Apply actions; verification and diagnosis remain observable."""
    return sum(1 for attempt in attempts if attempt.get("kind") == "apply")


def latest_attempt_for_ref(
    attempts: list[dict[str, Any]],
    ref: str,
    *,
    kind: str,
) -> dict[str, Any] | None:
    for attempt in reversed(attempts):
        if attempt.get("ref") == ref and attempt.get("kind") == kind:
            return attempt
    return None


def status_of_apply_record(attempt: dict[str, Any]) -> str | None:
    recorded = attempt.get("terminal_status")
    if isinstance(recorded, str) and recorded in CANONICAL_APPLY_STATUSES:
        return recorded
    result = attempt.get("result")
    if not isinstance(result, str):
        return None
    try:
        return canonical_apply_status(result)
    except ValueError:
        return None


def historical_join_reasons(
    episode_attempts: list[dict[str, Any]],
    ref: str,
    join_id: str,
) -> list[str]:
    own = next(
        (
            attempt
            for attempt in reversed(episode_attempts)
            if attempt.get("kind") == "apply"
            and attempt.get("ref") == ref
            and attempt.get("join_id") == join_id
        ),
        None,
    )
    if own is None:
        return [f"verify_requires_apply_result:{ref}:{join_id}"]
    wave_refs = [
        str(item)
        for item in own.get("wave_refs", [])
        if isinstance(item, str) and item
    ] or [ref]
    if own.get("worktree_mode") != "independent":
        pending: list[str] = []
        for candidate in wave_refs:
            latest = next(
                (
                    attempt
                    for attempt in reversed(episode_attempts)
                    if attempt.get("kind") == "apply"
                    and attempt.get("ref") == candidate
                    and attempt.get("join_id") == join_id
                ),
                None,
            )
            if latest is None or status_of_apply_record(latest) is None:
                pending.append(candidate)
        if pending:
            return [f"wave_join_pending:{join_id}:" + ",".join(pending)]
    own_status = status_of_apply_record(own)
    if own_status != "completed":
        return [f"verify_requires_completed_apply:{ref}:{own_status or 'missing'}"]
    evidence = own.get("evidence")
    if not isinstance(evidence, list) or not any(str(item).strip() for item in evidence):
        return [f"verify_requires_inspectable_evidence:{ref}"]
    return []


def verify_wave_reasons(
    repo_root: Path,
    change_id: str,
    fingerprint: str,
    episode: dict[str, Any],
    ref: str | None,
) -> list[str]:
    if not ref:
        return []
    episode_attempts = attempts_for_episode(episode)
    latest_own = latest_attempt_for_ref(episode_attempts, ref, kind="apply")
    if latest_own and latest_own.get("join_id"):
        return historical_join_reasons(
            episode_attempts,
            ref,
            str(latest_own["join_id"]),
        )
    payload = build_plan_payload(repo_root, change_id, advisory=False, batch=True)
    if ref not in payload.get("selected_wave", []):
        return []
    tasks = {item["ref"]: item for item in payload.get("tasks", [])}
    task = tasks.get(ref)
    if not task:
        return []
    if task.get("worktree_mode") == "independent":
        return []
    wave_refs = payload.get("selected_wave", [])
    pending = [
        candidate
        for candidate in wave_refs
        if latest_attempt_for_ref(episode_attempts, candidate, kind="apply") is None
    ]
    if pending:
        return ["wave_join_pending:" + ",".join(pending)]
    latest = latest_attempt_for_ref(episode_attempts, ref, kind="apply")
    if latest is None:
        return [f"verify_requires_apply_result:{ref}"]
    result = str(latest.get("result") or "").strip().lower()
    if result not in TERMINAL_APPLY_RESULTS:
        return [f"verify_requires_terminal_apply:{ref}:{result or 'missing'}"]
    if result not in COMPLETED_APPLY_RESULTS:
        return [f"verify_requires_completed_apply:{ref}:{result}"]
    return []


def second_unblock_reasons(
    revision_attempts: list[dict[str, Any]],
    ref: str,
) -> list[str]:
    """Require a completed second attempt and decision-changing evidence."""
    ref_attempts = [
        attempt for attempt in revision_attempts if attempt.get("ref") == ref
    ]
    reasons: list[str] = []
    if apply_iteration_count(ref_attempts) < 2:
        reasons.append(f"second_unblock_requires_second_apply:{ref}")

    blocking = [
        attempt
        for attempt in ref_attempts
        if attempt.get("kind") in {"apply", "verify"}
        and attempt.get("result") in {"blocked", "deviated"}
    ]
    if len(blocking) < 2:
        reasons.append(f"second_unblock_requires_two_blocking_results:{ref}")
        return reasons

    prior_fingerprint = blocking[-2].get("result_fingerprint") or blocking[-2].get(
        "error_fingerprint"
    )
    latest_fingerprint = blocking[-1].get("result_fingerprint") or blocking[-1].get(
        "error_fingerprint"
    )
    if (
        not prior_fingerprint
        or not latest_fingerprint
        or prior_fingerprint == latest_fingerprint
    ):
        reasons.append(f"second_unblock_requires_new_evidence:{ref}")
    return reasons


def gate_reasons(
    ledger: dict[str, Any],
    episode: dict[str, Any],
    run: dict[str, Any],
    next_ref: str | None,
    next_kind: str | None,
    *,
    budgets: dict[str, dict[str, int]],
    hard_ceiling: dict[str, int] | None,
    task_budget: dict[str, int] | None,
    current_allocated_subagents: int | None,
    prospective_subagent_ids: list[str],
) -> list[str]:
    reasons: list[str] = []
    run_attempts = run.get("attempts", [])
    revision_attempts = attempts_for_episode(episode)
    change_attempts = attempts_for_change(ledger)
    task_budget = task_budget or budgets["task"]
    revision_budget = budgets["revision"]
    change_budget = budgets["change"]

    # Revision/change Apply totals are compatibility diagnostics only.  The
    # selected ref's task budget (plus ref-local unblock authority) owns Apply
    # count gating, so aggregate totals cannot starve an independent ready ref.

    revision_minutes = active_seconds(revision_attempts) / 60
    change_minutes = active_seconds(change_attempts) / 60
    if revision_minutes >= revision_budget["max_active_minutes"]:
        reasons.append(
            f"revision_active_minutes_reached:{revision_budget['max_active_minutes']}"
        )
    if change_minutes >= change_budget["max_active_minutes"]:
        reasons.append(
            f"change_active_minutes_reached:{change_budget['max_active_minutes']}"
        )

    # Legacy hard-ceiling policy is retained for diagnostics and migration.
    # It is not a change-wide Apply stop; an exhaustible stop budget belongs to
    # the affected ref's unblock latch and cannot stop independent ready refs.
    _ = hard_ceiling

    revision_count = productive_revision_count(ledger)
    if revision_count > change_budget["max_revisions"]:
        reasons.append(f"change_max_revisions_exceeded:{change_budget['max_revisions']}")

    # Runtime IDs and headcount are observable capacity signals only.  Apply
    # authority is bounded by iteration/minute/breaker and write topology, not
    # by host_soft_cap/max_subagents/distinct-worker counts.
    _ = current_allocated_subagents, prospective_subagent_ids

    if next_ref and next_kind == "apply":
        prior = [
            attempt
            for attempt in revision_attempts
            if attempt.get("ref") == next_ref and attempt.get("kind") == "apply"
        ]
        if len(prior) >= task_budget["max_apply_attempts"]:
            reasons.append(
                f"task_max_apply_attempts_reached:{next_ref}:{task_budget['max_apply_attempts']}"
            )

    if next_kind == "explore":
        prior = [
            attempt for attempt in revision_attempts if attempt.get("kind") == "explore"
        ]
        if len(prior) >= revision_budget["max_explore_runs"]:
            reasons.append(
                f"revision_explore_budget_exhausted:{revision_budget['max_explore_runs']}"
            )

    if next_ref and next_kind == "unblock":
        if not has_blocking_evidence(revision_attempts, next_ref):
            reasons.append(f"task_unblock_dormant:{next_ref}")
        prior = [
            attempt
            for attempt in revision_attempts
            if attempt.get("ref") == next_ref and attempt.get("kind") == "unblock"
        ]
        if len(prior) >= task_budget["max_unblock_runs"]:
            reasons.append(
                f"task_unblock_budget_exhausted:{next_ref}:{task_budget['max_unblock_runs']}"
            )
        elif len(prior) == 1:
            reasons.extend(second_unblock_reasons(revision_attempts, next_ref))

    if len(run_attempts) >= 2:
        last_two = run_attempts[-2:]
        fingerprints = [
            attempt.get("result_fingerprint") or attempt.get("error_fingerprint")
            for attempt in last_two
        ]
        if fingerprints[0] and fingerprints[0] == fingerprints[1]:
            reasons.append("repeated_result_breaker")
        results = [attempt.get("result") for attempt in last_two]
        if results == ["no_progress", "no_progress"]:
            reasons.append("no_progress_breaker")
        if results == ["deviated", "deviated"]:
            reasons.append("repeated_deviation_breaker")

    return reasons


def cmd_gate(args: argparse.Namespace) -> int:
    _, config, fingerprint, ledger_path, budgets, warnings = resolve_runtime_policy(args)
    ceiling = hard_ceiling_of(config)
    # Per-run overrides retain their ordinary minute semantics. A legacy
    # hard ceiling is diagnostic and does not cap or terminalize Apply.
    if args.max_active_minutes is not None:
        budgets["revision"]["max_active_minutes"] = args.max_active_minutes
    if args.max_subagents is not None:
        budgets["revision"]["max_subagents"] = args.max_subagents
    if args.max_apply_attempts is not None:
        budgets["task"]["max_apply_attempts"] = args.max_apply_attempts
    task_budget = ref_budget_of(config, budgets, args.ref)
    if args.max_apply_attempts is not None:
        task_budget["max_apply_attempts"] = args.max_apply_attempts
    ledger = load_or_init_ledger(ledger_path, args.change_id)
    episode, run = load_run(ledger, args.change_id, fingerprint, args.run_id)
    reasons = gate_reasons(
        ledger,
        episode,
        run,
        args.ref,
        args.kind,
        budgets=budgets,
        hard_ceiling=ceiling,
        task_budget=task_budget,
        current_allocated_subagents=args.current_allocated_subagents,
        prospective_subagent_ids=normalize_subagent_ids(args.subagent_id),
    )
    if args.kind == "verify":
        reasons.extend(
            verify_wave_reasons(
                args.repo_root.resolve(),
                args.change_id,
                fingerprint,
                episode,
                args.ref,
            )
        )
    # Persist a newly created run before any maker work begins. Without this,
    # repeated pre-attempt gate calls would reset started_at_utc and silently
    # bypass the per-run time budget until the first record command.
    ledger["updated_at_utc"] = utc_now()
    write_json_atomic(ledger_path, ledger)
    attempts = run.get("attempts", [])
    emit_json(
        {
            "schema_version": SCHEMA_GATE,
            "change_id": args.change_id,
            "contract_fingerprint": fingerprint,
            "run_id": args.run_id,
            "decision": "stop" if reasons else "continue",
            "terminal": False,
            "reasons": reasons,
            "warnings": warnings,
            "attempt_count": len(attempts),
            "active_seconds": active_seconds(attempts_for_episode(episode)),
            "budgets": budgets,
            "autonomy": autonomy_of(config),
            "hard_ceiling": ceiling,
            "self_extensions_used": self_extensions_used(config),
            "ledger_path": str(ledger_path),
        }
    )
    return 2 if reasons else 0


def cmd_summary(args: argparse.Namespace) -> int:
    _, config, fingerprint, ledger_path, budgets, warnings = resolve_runtime_policy(args)
    ledger = load_or_init_ledger(ledger_path, args.change_id)
    episode, run = load_run(ledger, args.change_id, fingerprint, args.run_id)
    attempts = run.get("attempts", [])
    revision_attempts = attempts_for_episode(episode)
    change_attempts = attempts_for_change(ledger)
    per_ref: dict[str, dict[str, int]] = {}
    for attempt in attempts:
        ref = attempt.get("ref") or "unknown"
        bucket = per_ref.setdefault(ref, {"total": 0})
        bucket["total"] += 1
        key = f"{attempt.get('kind')}:{attempt.get('result')}"
        bucket[key] = bucket.get(key, 0) + 1
    emit_json(
        {
            "schema_version": SCHEMA_SUMMARY,
            "change_id": args.change_id,
            "contract_fingerprint": fingerprint,
            "run_id": args.run_id,
            "ledger_path": str(ledger_path),
            "warnings": warnings,
            "attempt_count": len(attempts),
            "revision_attempt_count": len(revision_attempts),
            "change_attempt_count": len(change_attempts),
            "revision_active_seconds": active_seconds(revision_attempts),
            "change_active_seconds": active_seconds(change_attempts),
            "revision_count": productive_revision_count(ledger),
            "revision_count_total": len(ledger.get("episodes", [])),
            "amendment_chain": amendment_chain(ledger),
            "budgets": budgets,
            "autonomy": autonomy_of(config),
            "hard_ceiling": hard_ceiling_of(config),
            "self_extensions_used": self_extensions_used(config),
            "attempts": attempts,
            "latest_attempt": attempts[-1] if attempts else None,
            "per_ref": per_ref,
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight OpenSpec loop helper.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("change_id")
    plan.add_argument(
        "--advisory",
        action="store_true",
        help="show the candidate task before seal without authorizing execution",
    )
    plan.add_argument(
        "--batch",
        action="store_true",
        help="legacy compatibility flag; selected_batch is always the complete "
        "dependency-ready census",
    )
    plan.set_defaults(func=cmd_plan)

    check = subparsers.add_parser("check")
    check.add_argument("change_id")
    check.set_defaults(func=cmd_check)

    goals = subparsers.add_parser("goals")
    goals.add_argument("change_id")
    goals.set_defaults(func=cmd_goals)

    design_verify = subparsers.add_parser("design-verify")
    design_verify.add_argument("change_id")
    design_verify.add_argument("--contract-fingerprint")
    design_verify.add_argument("--ledger-path", type=Path)
    design_verify.add_argument(
        "--observation",
        type=Path,
        help="JSON keyed by goal id with status match|mismatch, observed, evidence",
    )
    design_verify.add_argument(
        "--write-proposal",
        action="store_true",
        help="persist the proposal under the change's unblock/ decision path, "
        "for a gap that changes execution direction",
    )
    design_verify.set_defaults(func=cmd_design_verify)

    apply_revision = subparsers.add_parser("apply-revision")
    apply_revision.add_argument("change_id")
    apply_revision.add_argument("--proposal", type=Path, required=True)
    apply_revision.add_argument("--run-id", default="default")
    apply_revision.add_argument("--reason")
    apply_revision.add_argument("--contract-fingerprint")
    apply_revision.add_argument("--ledger-path", type=Path)
    apply_revision.set_defaults(func=cmd_apply_revision)

    seal = subparsers.add_parser("seal")
    seal.add_argument("change_id")
    seal.add_argument("--confirmed", action="store_true")
    # Options default to None so an omitted flag inherits the prior seal instead
    # of silently reverting confirmed policy to a built-in default.
    seal.add_argument("--retention", choices=("none", "thin", "full"))
    seal.add_argument(
        "--narrative-policy",
        choices=NARRATIVE_POLICIES,
        help="advisory: proposal/design/specs drift warns; strict: it blocks",
    )
    seal.add_argument(
        "--autonomy",
        choices=AUTONOMY_MODES,
        help="supervised: extension and amendment need --confirmed; "
        "full_auto: a recorded --reason suffices, bounded by hard_ceiling",
    )
    seal.add_argument("--hard-ceiling-max-active-minutes", type=int)
    seal.add_argument("--hard-ceiling-max-self-extensions", type=int)
    seal.add_argument("--ledger-path")
    seal.add_argument("--scratch-root")
    seal.add_argument("--product-root")
    seal.add_argument("--bundle-root")
    seal.add_argument("--gui-colab-root")
    seal.add_argument("--confirmed-at")
    seal.add_argument("--max-apply-attempts", type=int)
    seal.add_argument("--max-unblock-runs", type=int)
    seal.add_argument("--max-explore-runs", type=int)
    seal.add_argument("--max-subagents", type=int)
    seal.add_argument("--max-active-minutes", type=int)
    seal.add_argument("--max-revisions", type=int)
    seal.add_argument("--max-total-active-minutes", type=int)
    seal.set_defaults(func=cmd_seal)

    reseal = subparsers.add_parser("reseal")
    reseal.add_argument("change_id")
    reseal.add_argument(
        "--migrate",
        action="store_true",
        help=f"upgrade a {LEGACY_SCHEMA_LOOP} config, preserving every policy field",
    )
    reseal.add_argument("--allow-semantic-change", action="store_true")
    reseal.add_argument("--confirmed", action="store_true")
    reseal.add_argument("--narrative-policy", choices=NARRATIVE_POLICIES)
    reseal.add_argument("--reason")
    reseal.add_argument("--set-max-revisions", type=int)
    reseal.add_argument("--set-max-total-active-minutes", type=int)
    reseal.set_defaults(func=cmd_reseal)

    promote = subparsers.add_parser("promote")
    promote.add_argument("change_id")
    promote.add_argument("--ref", required=True)
    promote.add_argument(
        "--verdict",
        choices=("PASS",),
        default="PASS",
        help="only a verifier PASS may promote; the record itself must exist",
    )
    promote.add_argument("--contract-fingerprint")
    promote.add_argument("--ledger-path", type=Path)
    promote.set_defaults(func=cmd_promote)

    sync = subparsers.add_parser("sync")
    sync.add_argument("change_id")
    sync.add_argument("--contract-fingerprint")
    sync.add_argument("--ledger-path", type=Path)
    sync.set_defaults(func=cmd_sync)

    record = subparsers.add_parser("record")
    record.add_argument("change_id")
    record.add_argument("--contract-fingerprint")
    record.add_argument("--run-id", default="default")
    record.add_argument("--ref", required=True)
    record.add_argument(
        "--kind",
        choices=("apply", "verify", "explore", "unblock", "goal", "stop_hook", "review"),
        required=True,
    )
    record.add_argument(
        "--result",
        choices=(
            "success",
            "failure",
            "completed",
            "failed",
            "empty",
            "partial",
            "blocked",
            "deviated",
            "unverified",
            "no_progress",
            "pass",
            "fail",
        ),
        required=True,
    )
    record.add_argument("--action", default="")
    record.add_argument("--error-text")
    record.add_argument("--observation-text")
    record.add_argument(
        "--disposition",
        choices=("retry", "targeted_probe", "amend_spec", "supersede_task", "stop_budget"),
    )
    record.add_argument("--subagents-used", type=int, default=0)
    record.add_argument("--subagent-id", action="append", default=[])
    record.add_argument(
        "--record-owner",
        choices=("supervisor", "worker"),
        default="supervisor",
    )
    record.add_argument("--attempt-id")
    record.add_argument("--packet-id")
    record.add_argument("--role-id")
    record.add_argument("--changed-file", action="append", default=[])
    record.add_argument("--evidence", action="append", default=[])
    record.add_argument("--join-id")
    record.add_argument("--wave-ref", action="append", default=[])
    record.add_argument("--transient-retries", type=int, default=0)
    record.add_argument("--duration-seconds", type=int, default=0)
    record.add_argument("--ledger-path", type=Path)
    record.set_defaults(func=cmd_record)

    gate = subparsers.add_parser("gate")
    gate.add_argument("change_id")
    gate.add_argument("--contract-fingerprint")
    gate.add_argument("--run-id", default="default")
    gate.add_argument("--ref")
    gate.add_argument("--kind")
    gate.add_argument("--ledger-path", type=Path)
    gate.add_argument("--subagent-id", action="append", default=[])
    gate.add_argument("--current-allocated-subagents", type=int)
    gate.add_argument("--max-active-minutes", "--max-minutes", type=int)
    gate.add_argument("--max-subagents", type=int)
    gate.add_argument("--max-apply-attempts", type=int)
    gate.set_defaults(func=cmd_gate)

    summary = subparsers.add_parser("summary")
    summary.add_argument("change_id")
    summary.add_argument("--contract-fingerprint")
    summary.add_argument("--run-id", default="default")
    summary.add_argument("--ledger-path", type=Path)
    summary.set_defaults(func=cmd_summary)

    return parser


def main() -> int:
    configure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        emit_json(
            {
                "schema_version": "openspec-loop-error.v2",
                "error": str(exc),
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

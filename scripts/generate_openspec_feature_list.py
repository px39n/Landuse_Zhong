"""Generate a compact OpenSpec feature index from the active tasks registry."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "openspec-feature-list.v2"
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
CHECKBOX = re.compile(r"^\s*-\s*\[(?P<checked>[ xX])\]\s+(?P<body>.+)$")
REF = re.compile(r"\[#(?P<ref>R[A-Za-z0-9_.-]+)\]")
TASK_ID = re.compile(r"^(?P<id>\d+(?:\.\d+)+)\s+")
HEADING = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*$")
DIRECTIVE = re.compile(
    r"^\s*-\s*(?P<key>DEPENDS_ON|INDEPENDENT|NO_DEP|STATE|SUPERSEDES|REPAIR_POLICY|TEST_LEVEL)\s*:\s*(?P<value>.*?)\s*$",
    re.IGNORECASE,
)
ACCEPT = re.compile(r"^\s*-\s*ACCEPT\s*:\s*(?P<value>.*)$", re.IGNORECASE)
TEST = re.compile(r"^\s*-\s*TEST\s*:\s*(?P<value>.*)$", re.IGNORECASE)
LEGACY_STATE = re.compile(
    r"^\s*-\s*(?P<state>MAXED|BLOCKED|DEVIATED|SUPERSEDED)\b",
    re.IGNORECASE,
)
NON_OPERATIVE = ("historical", "rollback", "superseded", "appendix")
PATH_ANCHOR = re.compile(
    r"(?:"
    r"\b(?:outputs|src|scripts|docs|openspec|tests|figure|colab|cloud|data)/"
    r"|[A-Za-z]:\\|"
    r"NO_ARTIFACT\s*:"
    r"|\*\.(?:py|csv|json|md|yaml|yml|geojson|tif|tiff|zarr)"
    r")",
    re.IGNORECASE,
)
RUN_LINE = re.compile(r"\bRun\s*:", re.IGNORECASE)


@dataclass(slots=True)
class ParsedTask:
    line: int
    ref: str
    task_id: str
    title: str
    checked: bool
    accept_lines: list[str] = field(default_factory=list)
    test_lines: list[str] = field(default_factory=list)
    depends_raw: str | None = None
    independent_raw: str | None = None
    no_dep_raw: str | None = None
    state_raw: str | None = None
    supersedes_raw: str | None = None
    repair_policy_raw: str | None = None
    test_level_raw: str | None = None


def digest_text(lines: list[str]) -> str:
    normalized = "\n".join(line.strip() for line in lines if line.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


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
        temporary = Path(handle.name)
    temporary.replace(path)


def truthy(value: str | None) -> bool:
    return bool(value and value.strip().lower() in {"true", "yes", "1"})


def split_refs(raw: str | None) -> list[str]:
    if not raw:
        return []
    refs: list[str] = []
    for segment in re.split(r"[;；]", raw):
        parts = [token.strip().rstrip(".") for token in re.split(r"[,，]", segment)]
        parts = [token for token in parts if token and token.lower() != "none"]
        if not parts:
            continue
        if not all(
            re.fullmatch(r"R[A-Za-z0-9_.-]+|\d+(?:\.\d+)+", token)
            for token in parts
        ):
            continue
        for candidate in parts:
            if candidate not in refs:
                refs.append(candidate)
    return refs


def resolve_refs(
    raw: str | None,
    ordered_refs: list[str],
    task_id_map: dict[str, str],
) -> list[str]:
    resolved: list[str] = []
    for token in split_refs(raw):
        candidates: list[str]
        if token in ordered_refs:
            candidates = [token]
        elif token in task_id_map:
            candidates = [task_id_map[token]]
        else:
            range_match = re.fullmatch(
                r"(?P<start>R[A-Za-z0-9_.]+)-(?P<end>R[A-Za-z0-9_.]+)",
                token,
            )
            if not range_match:
                raise ValueError(f"unknown ref token `{token}`")
            start = range_match.group("start")
            end = range_match.group("end")
            if start not in ordered_refs or end not in ordered_refs:
                raise ValueError(f"unknown ref range `{token}`")
            first = ordered_refs.index(start)
            last = ordered_refs.index(end)
            if first > last:
                raise ValueError(f"reversed ref range `{token}`")
            candidates = ordered_refs[first : last + 1]
        for candidate in candidates:
            if candidate not in resolved:
                resolved.append(candidate)
    return resolved


def parse_tasks(text: str) -> list[ParsedTask]:
    lines = text.splitlines()
    has_active_heading = any(
        (match := HEADING.match(line))
        and "active task registry" in match.group("title").lower()
        for line in lines
    )
    active = not has_active_heading
    in_fence = False
    blocks: list[tuple[int, list[str]]] = []
    current: tuple[int, list[str]] | None = None

    for line_number, line in enumerate(lines, start=1):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        heading = HEADING.match(line)
        if heading:
            title = heading.group("title").lower()
            if "active task registry" in title:
                active = True
            elif any(marker in title for marker in NON_OPERATIVE):
                active = False
            if current:
                blocks.append(current)
                current = None
            continue
        if not active:
            continue
        if CHECKBOX.match(line):
            if current:
                blocks.append(current)
            current = (line_number, [line])
        elif current:
            current[1].append(line)
    if current:
        blocks.append(current)

    tasks: list[ParsedTask] = []
    seen: set[str] = set()
    for line_number, block in blocks:
        match = CHECKBOX.match(block[0])
        assert match is not None
        body = match.group("body")
        ref_match = REF.search(body)
        if not ref_match:
            raise ValueError(f"line {line_number}: active task missing [#R...] ref")
        ref = ref_match.group("ref")
        if ref in seen:
            raise ValueError(f"duplicate active ref `{ref}`")
        seen.add(ref)
        id_match = TASK_ID.match(body)
        task_id = id_match.group("id") if id_match else "?"
        title = body[id_match.end() :] if id_match else body
        title = REF.sub("", title).strip()
        task = ParsedTask(
            line=line_number,
            ref=ref,
            task_id=task_id,
            title=title,
            checked=match.group("checked").lower() == "x",
        )
        mode: str | None = None
        for raw_line in block[1:]:
            directive = DIRECTIVE.match(raw_line)
            if directive:
                mode = None
                key = directive.group("key").upper()
                value = directive.group("value").strip().rstrip(".")
                if key == "DEPENDS_ON":
                    task.depends_raw = value
                elif key == "INDEPENDENT":
                    task.independent_raw = value
                elif key == "NO_DEP":
                    task.no_dep_raw = value
                elif key == "STATE":
                    task.state_raw = value.lower().replace("-", "_")
                elif key == "SUPERSEDES":
                    task.supersedes_raw = value
                elif key == "REPAIR_POLICY":
                    task.repair_policy_raw = value.lower()
                elif key == "TEST_LEVEL":
                    task.test_level_raw = value.lower().replace("-", "_")
                continue
            accept = ACCEPT.match(raw_line)
            if accept:
                mode = "accept"
                if accept.group("value").strip():
                    task.accept_lines.append(accept.group("value").strip())
                continue
            test = TEST.match(raw_line)
            if test:
                mode = "test"
                if test.group("value").strip():
                    task.test_lines.append(test.group("value").strip())
                continue
            legacy = LEGACY_STATE.match(raw_line)
            if legacy and task.state_raw is None:
                task.state_raw = legacy.group("state").lower()
                mode = None
                continue
            stripped = raw_line.strip()
            if stripped.upper().startswith(
                ("- BUNDLE", "- EVIDENCE", "- UNBLOCK", "- NEEDS", "- ORDERING")
            ):
                mode = None
                continue
            if mode and stripped.startswith("- "):
                getattr(task, f"{mode}_lines").append(stripped[2:].strip())
            elif mode and stripped:
                getattr(task, f"{mode}_lines").append(stripped)
        if not task.accept_lines or not task.test_lines:
            raise ValueError(f"task {ref} must contain ACCEPT and TEST")
        if task.state_raw and task.state_raw not in TASK_STATES:
            raise ValueError(f"task {ref} has unsupported STATE `{task.state_raw}`")
        if task.repair_policy_raw not in {None, "bounded-r1"}:
            raise ValueError(
                f"task {ref} has unsupported REPAIR_POLICY `{task.repair_policy_raw}`"
            )
        if task.test_level_raw not in {None, "smoke", "pilot", "production", "canary"}:
            raise ValueError(
                f"task {ref} has unsupported TEST_LEVEL `{task.test_level_raw}`"
            )
        tasks.append(task)
    return tasks


def build_features(tasks: list[ParsedTask], existing: dict[str, Any]) -> dict[str, Any]:
    refs = [task.ref for task in tasks]
    known = set(refs)
    task_id_map = {task.task_id: task.ref for task in tasks}
    features: dict[str, Any] = {}
    completed: set[str] = set()

    for index, task in enumerate(tasks):
        supersedes = resolve_refs(task.supersedes_raw, refs, task_id_map)
        unknown_supersedes = [ref for ref in supersedes if ref not in known]
        if unknown_supersedes:
            raise ValueError(f"task {task.ref} supersedes unknown refs {unknown_supersedes}")

        if truthy(task.independent_raw) or truthy(task.no_dep_raw):
            dependencies: list[str] = []
        elif task.depends_raw is not None:
            dependencies = resolve_refs(task.depends_raw, refs, task_id_map)
        elif task.no_dep_raw:
            exclusions = set(resolve_refs(task.no_dep_raw, refs, task_id_map))
            dependencies = [ref for ref in refs[:index] if ref not in exclusions]
        else:
            dependencies = [ref for ref in refs[:index] if ref not in supersedes]
        unknown_dependencies = [ref for ref in dependencies if ref not in known]
        if unknown_dependencies:
            raise ValueError(f"task {task.ref} depends on unknown refs {unknown_dependencies}")

        previous = existing.get(task.ref, {}) if isinstance(existing, dict) else {}
        previous_state = previous.get("state", previous.get("status"))
        if task.checked:
            state = "passed"
        elif task.state_raw:
            state = task.state_raw
        elif isinstance(previous_state, str) and previous_state.lower() in {
            "in_progress",
            "blocked",
            "deviated",
            "maxed",
            "superseded",
        }:
            state = previous_state.lower()
        elif all(ref in completed for ref in dependencies):
            state = "ready"
        else:
            state = "pending"
        if state == "passed":
            completed.add(task.ref)

        features[task.ref] = {
            "id": task.task_id,
            "ref": task.ref,
            "title": task.title,
            "state": state,
            "passes": state == "passed",
            "depends_on": dependencies,
            "supersedes": supersedes,
            "task_path": f"tasks.md:{task.line}",
            "accept_hash": digest_text(task.accept_lines),
            "test_hash": digest_text(task.test_lines),
        }
        if task.repair_policy_raw:
            features[task.ref]["repair_policy"] = task.repair_policy_raw
        if task.test_level_raw:
            features[task.ref]["test_level"] = task.test_level_raw

    superseded = {ref for feature in features.values() for ref in feature["supersedes"]}
    for ref in superseded:
        if ref in features and features[ref]["state"] != "passed":
            features[ref]["state"] = "superseded"
    return features


def accept_quality_issues(ref: str, accept_lines: list[str]) -> list[str]:
    joined = "\n".join(accept_lines)
    if PATH_ANCHOR.search(joined):
        return []
    return [
        f"task {ref} ACCEPT lacks a path-like product anchor or NO_ARTIFACT: "
        "(warn; pass --strict-quality to fail)"
    ]


def test_quality_issues(ref: str, test_lines: list[str]) -> list[str]:
    joined = "\n".join(test_lines)
    if RUN_LINE.search(joined):
        return []
    return [
        f"task {ref} TEST lacks an executable Run: line "
        "(warn; pass --strict-quality to fail)"
    ]


def quality_warnings(tasks: list[ParsedTask]) -> list[str]:
    warnings: list[str] = []
    for task in tasks:
        warnings.extend(accept_quality_issues(task.ref, task.accept_lines))
        warnings.extend(test_quality_issues(task.ref, task.test_lines))
    return warnings


def generate(
    repo_root: Path,
    change_id: str,
    *,
    strict_quality: bool = False,
) -> dict[str, Any]:
    change_dir = repo_root / "openspec" / "changes" / change_id
    tasks_path = change_dir / "tasks.md"
    out_path = change_dir / "feature_list.json"
    if not tasks_path.exists():
        raise ValueError(f"missing tasks file: {tasks_path}")
    existing: dict[str, Any] = {}
    if out_path.exists():
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("features"), dict):
            existing = payload["features"]
    tasks = parse_tasks(tasks_path.read_text(encoding="utf-8"))
    warnings = quality_warnings(tasks)
    if warnings and strict_quality:
        raise ValueError("; ".join(warnings))
    features = build_features(tasks, existing)
    payload = {
        "schema_version": SCHEMA,
        "change_id": change_id,
        "registry": {
            "active_source": "tasks.md active registry",
            "active_ref_count": len(features),
            "history_source": "git",
        },
        "features": features,
    }
    if warnings:
        payload["quality_warnings"] = warnings
    write_json_atomic(out_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("change_id", nargs="?")
    parser.add_argument("--change-id", dest="change_id_flag")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--strict-quality",
        action="store_true",
        help="fail when ACCEPT lacks a path anchor / NO_ARTIFACT or TEST lacks Run:",
    )
    args = parser.parse_args()
    change_id = args.change_id_flag or args.change_id
    if not change_id:
        parser.error("change_id is required")
    payload = generate(
        args.repo_root.resolve(),
        change_id,
        strict_quality=args.strict_quality,
    )
    for warning in payload.get("quality_warnings", []):
        print(f"warning: {warning}", file=sys.stderr)
    print(
        f"Wrote openspec/changes/{change_id}/feature_list.json "
        f"with {len(payload['features'])} active features"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Synchronize canonical OpenSpec Loop skills into platform mirrors.

Only SKILL.md and references/** are shared. Platform metadata such as
.codex/skills/<name>/agents/openai.yaml is deliberately untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable


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
    "monitor-openspec-codex",
)
MIRROR_ROOTS = (Path(".codex/skills"), Path(".claude/skills"))


def canonical_files(skill_dir: Path) -> dict[Path, bytes]:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        raise FileNotFoundError(f"missing canonical skill: {skill_md}")

    files = {Path("SKILL.md"): skill_md.read_bytes()}
    references = skill_dir / "references"
    if references.is_dir():
        for path in sorted(references.rglob("*")):
            if path.is_file():
                files[Path("references") / path.relative_to(references)] = (
                    path.read_bytes()
                )
    return files


def stale_reference_files(skill_dir: Path, expected: Iterable[Path]) -> list[Path]:
    references = skill_dir / "references"
    if not references.is_dir():
        return []
    expected_set = {path for path in expected if path.parts[0] == "references"}
    return [
        path
        for path in sorted(references.rglob("*"))
        if path.is_file() and path.relative_to(skill_dir) not in expected_set
    ]


def inspect(repo_root: Path) -> list[str]:
    issues: list[str] = []
    source_root = repo_root / ".agents" / "skills"
    for name in SKILL_NAMES:
        expected = canonical_files(source_root / name)
        for mirror_root in MIRROR_ROOTS:
            destination = repo_root / mirror_root / name
            for relative, content in expected.items():
                target = destination / relative
                if not target.is_file():
                    issues.append(f"missing: {target.relative_to(repo_root)}")
                elif target.read_bytes() != content:
                    issues.append(f"different: {target.relative_to(repo_root)}")
            for stale in stale_reference_files(destination, expected):
                issues.append(f"stale: {stale.relative_to(repo_root)}")
    return issues


def remove_empty_reference_dirs(root: Path) -> None:
    if not root.is_dir():
        return
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


def synchronize(repo_root: Path) -> list[str]:
    changed: list[str] = []
    source_root = repo_root / ".agents" / "skills"
    for name in SKILL_NAMES:
        expected = canonical_files(source_root / name)
        for mirror_root in MIRROR_ROOTS:
            destination = repo_root / mirror_root / name
            for relative, content in expected.items():
                target = destination / relative
                if not target.is_file() or target.read_bytes() != content:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                    changed.append(f"updated: {target.relative_to(repo_root)}")
            for stale in stale_reference_files(destination, expected):
                stale.unlink()
                changed.append(f"removed: {stale.relative_to(repo_root)}")
            remove_empty_reference_dirs(destination / "references")
            if any(path.parts[0] == "references" for path in expected):
                (destination / "references").mkdir(parents=True, exist_ok=True)
    return changed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to the script's parent repository)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report drift without changing mirrors",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    try:
        if args.check:
            issues = inspect(repo_root)
            if issues:
                print("\n".join(issues))
                return 1
            print("OpenSpec Loop skill mirrors are synchronized.")
            return 0

        changed = synchronize(repo_root)
        issues = inspect(repo_root)
        if issues:
            print("\n".join(issues), file=sys.stderr)
            return 1
        if changed:
            print("\n".join(changed))
        else:
            print("OpenSpec Loop skill mirrors were already synchronized.")
        return 0
    except (FileNotFoundError, OSError) as exc:
        print(f"sync failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

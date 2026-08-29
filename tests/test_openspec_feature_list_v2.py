from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_openspec_feature_list.py"
REPO_ROOT = Path(__file__).resolve().parents[1]
LOOP_SCRIPT = REPO_ROOT / "scripts" / "openspec_loop.py"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "openspec_loop"
SPEC = importlib.util.spec_from_file_location("openspec_feature_list_v2", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_tasks(repo: Path, text: str, existing: dict | None = None) -> Path:
    change_dir = repo / "openspec" / "changes" / "demo"
    change_dir.mkdir(parents=True)
    (change_dir / "tasks.md").write_text(text, encoding="utf-8")
    if existing is not None:
        (change_dir / "feature_list.json").write_text(
            json.dumps(existing, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return change_dir


def test_generator_emits_compact_hash_index_without_contract_duplication(tmp_path: Path) -> None:
    change_dir = write_tasks(
        tmp_path,
        """## Active Task Registry

- [ ] 1.1 Build parser [#R1]
  - DEPENDS_ON: none
  - ACCEPT: The parser preserves the named output contract.
  - TEST: Run: python -m pytest tests/test_parser.py -q
""",
    )

    payload = MODULE.generate(tmp_path, "demo")
    feature = payload["features"]["R1"]

    assert payload["schema_version"] == "openspec-feature-list.v2"
    assert set(feature) == {
        "id",
        "ref",
        "title",
        "state",
        "passes",
        "depends_on",
        "supersedes",
        "task_path",
        "accept_hash",
        "test_hash",
    }
    assert feature["state"] == "ready"
    rendered = (change_dir / "feature_list.json").read_text(encoding="utf-8")
    assert "preserves the named output contract" not in rendered
    assert "pytest tests/test_parser.py" not in rendered
    first_bytes = (change_dir / "feature_list.json").read_bytes()
    MODULE.generate(tmp_path, "demo")
    assert (change_dir / "feature_list.json").read_bytes() == first_bytes


def test_generator_marks_maxed_task_superseded_and_replacement_ready(tmp_path: Path) -> None:
    write_tasks(
        tmp_path,
        """## Active Task Registry

- [x] 1.36 Baseline [#R36]
  - DEPENDS_ON: none
  - ACCEPT: Baseline exists.
  - TEST: Run: verify baseline
- [ ] 1.37 Exhausted attempt [#R37]
  - DEPENDS_ON: R36
  - STATE: maxed
  - ACCEPT: Old attempt is retained only as diagnosis.
  - TEST: Run: verify old attempt
- [ ] 1.38 Replacement [#R38]
  - DEPENDS_ON: R36
  - SUPERSEDES: R37
  - ACCEPT: Replacement answers the current question.
  - TEST: Run: verify replacement
""",
    )

    features = MODULE.generate(tmp_path, "demo")["features"]

    assert features["R36"]["state"] == "passed"
    assert features["R37"]["state"] == "superseded"
    assert features["R38"]["state"] == "ready"
    assert features["R38"]["supersedes"] == ["R37"]


def test_generator_preserves_legacy_blocked_state_when_task_is_untouched(tmp_path: Path) -> None:
    existing = {
        "features": {
            "R1": {
                "task_id": "1.1",
                "status": "blocked",
                "passes": False,
            }
        }
    }
    write_tasks(
        tmp_path,
        """- [ ] 1.1 Await external source [#R1]
  - DEPENDS_ON: none
  - ACCEPT: Source provenance is confirmed.
  - TEST: Run: verify provenance
""",
        existing,
    )

    feature = MODULE.generate(tmp_path, "demo")["features"]["R1"]

    assert feature["state"] == "blocked"
    assert feature["passes"] is False


def test_generator_fails_closed_when_acceptance_is_missing(tmp_path: Path) -> None:
    write_tasks(
        tmp_path,
        """- [ ] 1.1 Missing acceptance [#R1]
  - DEPENDS_ON: none
  - TEST: Run: verify something
""",
    )

    with pytest.raises(ValueError, match="ACCEPT and TEST"):
        MODULE.generate(tmp_path, "demo")


def copy_change_fixture(tmp_path: Path, change_id: str) -> Path:
    target = tmp_path / "openspec" / "changes" / change_id
    target.parent.mkdir(parents=True)
    shutil.copytree(FIXTURES / change_id, target)
    return target


def test_extend_sectoral_fragment_selects_r38_not_maxed_r37(
    tmp_path: Path,
) -> None:
    copy_change_fixture(tmp_path, "extend-sectoral-data")
    features = MODULE.generate(tmp_path, "extend-sectoral-data")["features"]

    assert features["R37"]["state"] == "superseded"
    assert features["R38"]["state"] == "ready"
    assert features["R38"]["supersedes"] == ["R37"]

    completed = subprocess.run(
        [
            sys.executable,
            str(LOOP_SCRIPT),
            "--repo-root",
            str(tmp_path),
            "plan",
            "extend-sectoral-data",
            "--advisory",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["candidate_ref"] == "R38"
    assert plan["selected_ref"] == "R38"
    assert not [
        issue for issue in plan["issues"] if "task/feature state drift" in issue
    ]

    ledger = tmp_path / "test_cache" / "extend-sectoral-data" / "loop-ledger.json"
    sealed = subprocess.run(
        [
            sys.executable,
            str(LOOP_SCRIPT),
            "--repo-root",
            str(tmp_path),
            "seal",
            "extend-sectoral-data",
            "--confirmed",
            "--retention",
            "thin",
            "--ledger-path",
            str(ledger),
            "--scratch-root",
            str(ledger.parent),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert sealed.returncode == 0, sealed.stdout + sealed.stderr

    checked = subprocess.run(
        [
            sys.executable,
            str(LOOP_SCRIPT),
            "--repo-root",
            str(tmp_path),
            "check",
            "extend-sectoral-data",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    check_payload = json.loads(checked.stdout)
    assert check_payload["ok"] is True
    assert check_payload["selected_ref"] == "R38"


def test_r50_fragment_excludes_rollback_only_history(tmp_path: Path) -> None:
    copy_change_fixture(tmp_path, "r50-page-multitarget-vision-pipeline")
    features = MODULE.generate(
        tmp_path, "r50-page-multitarget-vision-pipeline"
    )["features"]

    assert set(features) == {"R64", "R65", "R66"}
    assert features["R64"]["state"] == "passed"
    assert features["R65"]["state"] == "ready"
    assert features["R66"]["state"] == "pending"
    assert "R36" not in features


def test_generator_warns_on_weak_accept_but_still_writes(tmp_path: Path) -> None:
    change_dir = write_tasks(
        tmp_path,
        """## Active Task Registry

- [ ] 1.1 Vague accept [#R1]
  - DEPENDS_ON: none
  - ACCEPT: Make it safer and idempotent.
  - TEST: Run: python -m pytest tests/test_parser.py -q
""",
    )

    payload = MODULE.generate(tmp_path, "demo")
    assert "R1" in payload["features"]
    assert payload["quality_warnings"]
    assert any("ACCEPT lacks" in item for item in payload["quality_warnings"])
    assert (change_dir / "feature_list.json").is_file()


def test_generator_strict_quality_fails_without_run_line(tmp_path: Path) -> None:
    write_tasks(
        tmp_path,
        """## Active Task Registry

- [ ] 1.1 No run line [#R1]
  - DEPENDS_ON: none
  - ACCEPT: Write outputs/demo/result.json with schema v1.
  - TEST: Inspect the result visually.
""",
    )

    with pytest.raises(ValueError, match="Run:"):
        MODULE.generate(tmp_path, "demo", strict_quality=True)


def test_generator_accepts_no_artifact_and_run_line(tmp_path: Path) -> None:
    write_tasks(
        tmp_path,
        """## Active Task Registry

- [ ] 1.1 Contract text only [#R1]
  - DEPENDS_ON: none
  - ACCEPT: NO_ARTIFACT: proposal wording only under openspec/changes/demo/
  - TEST: Run: openspec validate demo --strict
""",
    )

    payload = MODULE.generate(tmp_path, "demo", strict_quality=True)
    assert "quality_warnings" not in payload
    assert payload["features"]["R1"]["state"] == "ready"

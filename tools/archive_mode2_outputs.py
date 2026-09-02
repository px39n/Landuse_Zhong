from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


XARRAY_ROOT = Path(r"D:\xarray")
HISTORY_ROOT = XARRAY_ROOT / "history_data"
CANCELLED_ROOT = HISTORY_ROOT / "cancelled_mode2_20260819"

PRODUCTION_PATHS = (
    XARRAY_ROOT / "final_mask_1km_new.nc",
    XARRAY_ROOT / "abandon_2",
    XARRAY_ROOT / "merged_chunk_2",
)

CANCELLED_PATHS = (
    XARRAY_ROOT / "final_mask_1km_mode_2.nc",
    XARRAY_ROOT / "final_mask_1km_mode_2.tmp.nc",
    XARRAY_ROOT / "abandon_mode_2",
    XARRAY_ROOT / "merged_chunk_mode_2",
    XARRAY_ROOT / "mode2_pipeline_manifest.json",
    XARRAY_ROOT / "mode2_pipeline.log",
)


def _assert_within(path: Path, parent: Path) -> None:
    resolved = path.resolve(strict=False)
    resolved.relative_to(parent.resolve(strict=True))


def _fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    total_bytes = 0
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        size = item.stat().st_size
        total_bytes += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return {
        "path": str(path),
        "kind": "directory" if path.is_dir() else "file",
        "file_count": len(files),
        "total_bytes": total_bytes,
        "aggregate_sha256": digest.hexdigest().upper(),
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def archive() -> dict[str, object]:
    root = XARRAY_ROOT.resolve(strict=True)
    history = HISTORY_ROOT.resolve(strict=True)
    cancelled = CANCELLED_ROOT.resolve(strict=False)
    _assert_within(history, root)
    _assert_within(cancelled, history)
    manifest_path = HISTORY_ROOT / "archive_manifest_20260819.json"
    journal_path = HISTORY_ROOT / "archive_manifest_20260819.in_progress.json"
    if manifest_path.exists():
        raise FileExistsError(f"Archive manifest already exists: {manifest_path}")
    if journal_path.exists():
        raise FileExistsError(f"Unresolved archive journal already exists: {journal_path}")

    moves: list[tuple[Path, Path, str]] = []
    for source in PRODUCTION_PATHS:
        if not source.exists():
            raise FileNotFoundError(f"Required legacy production path is missing: {source}")
        destination = HISTORY_ROOT / source.name
        moves.append((source, destination, "legacy_production"))
    for source in CANCELLED_PATHS:
        if source.exists():
            destination = CANCELLED_ROOT / source.name
            moves.append((source, destination, "cancelled_mode2"))

    for source, destination, _ in moves:
        _assert_within(source, root)
        _assert_within(destination, history)
        if destination.exists():
            raise FileExistsError(f"Archive destination already exists: {destination}")

    records = []
    for source, destination, category in moves:
        record = _fingerprint(source)
        record.update({"destination": str(destination), "category": category, "moved": False, "verified": False})
        records.append(record)

    manifest = {
        "status": "in_progress",
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        "xarray_root": str(XARRAY_ROOT),
        "history_root": str(HISTORY_ROOT),
        "correct_mode_reclass_retained": str(XARRAY_ROOT / "reclass_lccs_1km.nc"),
        "old_reclass_already_archived": str(HISTORY_ROOT / "reclass_lccs_1km.nc"),
        "records": records,
    }
    _atomic_json(journal_path, manifest)

    CANCELLED_ROOT.mkdir(parents=True, exist_ok=True)
    for index, (source, destination, _) in enumerate(moves):
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
        records[index]["moved"] = True
        _atomic_json(journal_path, manifest)

    for record in records:
        destination = Path(str(record["destination"]))
        observed = _fingerprint(destination)
        if (
            observed["file_count"] != record["file_count"]
            or observed["total_bytes"] != record["total_bytes"]
            or observed["aggregate_sha256"] != record["aggregate_sha256"]
        ):
            raise RuntimeError(f"Post-move fingerprint mismatch: {destination}")
        record["verified"] = True
        _atomic_json(journal_path, manifest)

    manifest["status"] = "complete"
    _atomic_json(manifest_path, manifest)
    journal_path.unlink()
    print(f"[ARCHIVE] moved={len(records)} manifest={manifest_path}")
    for record in records:
        print(
            "[ARCHIVE] "
            f"{record['category']} {record['path']} -> {record['destination']} "
            f"files={record['file_count']} bytes={record['total_bytes']} "
            f"sha256={record['aggregate_sha256']}"
        )
    return manifest


if __name__ == "__main__":
    archive()

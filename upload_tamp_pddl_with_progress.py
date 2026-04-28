#!/usr/bin/env python3
"""Upload a local TAMP-PDDL tree to a remote server with tqdm progress bars."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, List

from tqdm import tqdm


DEFAULT_SOURCE = Path(__file__).resolve().parent
DEFAULT_HOST = "gvlab2.iiit.ac.in"
DEFAULT_REMOTE_ROOT = "/home/projects/long-horizon/TAMP-PDDL"
CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class UploadEntry:
    kind: str
    local_path: Path
    relative_path: PurePosixPath
    size: int = 0
    link_target: str = ""


def _collect_entries(source_root: Path) -> List[UploadEntry]:
    entries: List[UploadEntry] = []

    def visit(path: Path) -> None:
        for child in sorted(path.iterdir(), key=lambda item: item.name):
            rel = PurePosixPath(child.relative_to(source_root).as_posix())
            if child.is_symlink():
                entries.append(
                    UploadEntry(
                        kind="symlink",
                        local_path=child,
                        relative_path=rel,
                        link_target=os.readlink(child),
                    )
                )
                continue
            if child.is_dir():
                entries.append(UploadEntry(kind="dir", local_path=child, relative_path=rel))
                visit(child)
                continue
            if child.is_file():
                entries.append(
                    UploadEntry(
                        kind="file",
                        local_path=child,
                        relative_path=rel,
                        size=child.stat().st_size,
                    )
                )

    visit(source_root)
    return entries


def _ssh_run(host: str, remote_command: str, capture_output: bool = False) -> subprocess.CompletedProcess:
    wrapped_command = f"/bin/sh -lc {shlex.quote(remote_command)}"
    return subprocess.run(
        ["ssh", host, wrapped_command],
        check=True,
        text=True,
        capture_output=capture_output,
    )


def _remote_path(remote_root: str, rel_path: PurePosixPath) -> str:
    base = PurePosixPath(remote_root)
    return str(base / rel_path)


def _ensure_remote_dir(host: str, remote_dir: str) -> None:
    _ssh_run(host, f"mkdir -p {shlex.quote(remote_dir)}")


def _reset_remote_root(host: str, remote_root: str) -> None:
    _ssh_run(
        host,
        f"rm -rf {shlex.quote(remote_root)} && mkdir -p {shlex.quote(remote_root)}",
    )


def _upload_file(
    host: str,
    remote_root: str,
    entry: UploadEntry,
    overall_bar: tqdm,
    show_per_file: bool,
) -> None:
    remote_file = _remote_path(remote_root, entry.relative_path)
    remote_dir = str(PurePosixPath(remote_file).parent)
    remote_tmp = f"{remote_file}.uploading"

    _ensure_remote_dir(host, remote_dir)

    command = (
        f"cat > {shlex.quote(remote_tmp)} && "
        f"mv {shlex.quote(remote_tmp)} {shlex.quote(remote_file)}"
    )
    process = subprocess.Popen(
        ["ssh", host, "/bin/sh", "-lc", command],
        stdin=subprocess.PIPE,
    )

    file_bar = None
    if show_per_file:
        file_bar = tqdm(
            total=entry.size,
            desc=entry.relative_path.as_posix(),
            unit="B",
            unit_scale=True,
            dynamic_ncols=True,
            leave=False,
        )

    try:
        assert process.stdin is not None
        with entry.local_path.open("rb") as handle:
            while True:
                chunk = handle.read(CHUNK_SIZE)
                if not chunk:
                    break
                process.stdin.write(chunk)
                overall_bar.update(len(chunk))
                if file_bar is not None:
                    file_bar.update(len(chunk))
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"upload failed for {entry.relative_path} (ssh exit {return_code})")
    finally:
        if file_bar is not None:
            file_bar.close()


def _upload_symlink(host: str, remote_root: str, entry: UploadEntry) -> None:
    remote_path = _remote_path(remote_root, entry.relative_path)
    remote_dir = str(PurePosixPath(remote_path).parent)
    _ensure_remote_dir(host, remote_dir)
    _ssh_run(
        host,
        f"ln -sfn {shlex.quote(entry.link_target)} {shlex.quote(remote_path)}",
    )


def _create_remote_dirs(host: str, remote_root: str, entries: Iterable[UploadEntry]) -> None:
    for entry in entries:
        if entry.kind != "dir":
            continue
        _ensure_remote_dir(host, _remote_path(remote_root, entry.relative_path))


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload TAMP-PDDL to the remote server with tqdm progress")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Local source directory to upload")
    parser.add_argument("--host", default=DEFAULT_HOST, help="SSH host or alias")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT, help="Remote destination directory")
    parser.add_argument(
        "--no-fresh",
        action="store_true",
        help="Do not delete the remote target before uploading",
    )
    parser.add_argument(
        "--no-per-file",
        action="store_true",
        help="Disable the per-file tqdm bar and keep only the overall bar",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be uploaded")
    args = parser.parse_args()

    source_root = Path(args.source).resolve()
    if not source_root.is_dir():
        print(f"Source directory not found: {source_root}", file=sys.stderr)
        return 1

    entries = _collect_entries(source_root)
    file_entries = [entry for entry in entries if entry.kind == "file"]
    dir_entries = [entry for entry in entries if entry.kind == "dir"]
    symlink_entries = [entry for entry in entries if entry.kind == "symlink"]
    total_bytes = sum(entry.size for entry in file_entries)

    print(f"Source      : {source_root}")
    print(f"Remote host : {args.host}")
    print(f"Remote root : {args.remote_root}")
    print(f"Files       : {len(file_entries)}")
    print(f"Dirs        : {len(dir_entries)}")
    print(f"Symlinks    : {len(symlink_entries)}")
    print(f"Total bytes : {total_bytes} ({_format_bytes(total_bytes)})")

    if args.dry_run:
        return 0

    if not args.no_fresh:
        print("Resetting remote target...")
        _reset_remote_root(args.host, args.remote_root)
    else:
        _ensure_remote_dir(args.host, args.remote_root)

    if dir_entries:
        print("Creating remote directories...")
        _create_remote_dirs(args.host, args.remote_root, dir_entries)

    if symlink_entries:
        print("Creating remote symlinks...")
        for entry in symlink_entries:
            _upload_symlink(args.host, args.remote_root, entry)

    print("Uploading files...")
    overall_bar = tqdm(
        total=total_bytes,
        desc="overall",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
    )
    try:
        for index, entry in enumerate(file_entries, start=1):
            overall_bar.set_postfix(file=f"{index}/{len(file_entries)}")
            _upload_file(
                host=args.host,
                remote_root=args.remote_root,
                entry=entry,
                overall_bar=overall_bar,
                show_per_file=not args.no_per_file,
            )
    finally:
        overall_bar.close()

    print("Upload complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Prepare private SQLite databases before handing their paths to SQLite."""

import os
from pathlib import Path

from telegram_mcp.private_files import _trusted_parent, create_private_file, open_existing_regular


def prepare_private_sqlite_file(path: Path) -> Path:
    if os.name != "posix":
        raise OSError("Cannot verify private SQLite file permissions on this platform.")

    parent = _trusted_parent(path.parent)
    if parent.stat().st_mode & 0o022:
        raise PermissionError(f"SQLite directory must not be writable by other users: {parent}")
    path = parent / path.name
    for suffix in ("-journal", "-wal", "-shm"):
        sibling = path.with_name(path.name + suffix)
        try:
            fd = open_existing_regular(sibling, os.O_RDONLY, private=True)
        except FileNotFoundError:
            continue
        os.close(fd)

    try:
        fd = open_existing_regular(path, os.O_RDWR, private=True)
    except FileNotFoundError:
        try:
            fd = create_private_file(path)
        except FileExistsError:
            fd = open_existing_regular(path, os.O_RDWR, private=True)
    os.close(fd)
    return path

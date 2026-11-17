"""Local storage for session credentials and their advisory locks."""

import os
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


def _require_private_directories() -> None:
    if os.name == "nt":
        # These are the first releases where mkdir(0o700) supplies a private ACL.
        minimum_patch = {10: 15, 11: 10, 12: 4}.get(sys.version_info.minor, 0)
        if sys.version_info.micro < minimum_patch:
            raise OSError("Update Python to create private session directories on Windows.")


def _is_reparse_point(info: os.stat_result) -> bool:
    return bool(getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _trusted_parent(path: Path) -> Path:
    """Resolve parent aliases and reject ancestors that other users can replace."""
    path = path.resolve(strict=True)
    for directory in (*reversed(path.parents), path):
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or _is_reparse_point(info):
            raise OSError(f"Unsafe session directory: {directory}")
        if os.name == "posix":
            other_writable = info.st_mode & 0o022
            if info.st_uid not in (0, os.geteuid()) or (
                other_writable and not info.st_mode & stat.S_ISVTX
            ):
                raise PermissionError(f"Unsafe session directory: {directory}")
    return path


def _check_private(info: os.stat_result, path: Path) -> None:
    if os.name == "posix" and (info.st_uid != os.geteuid() or info.st_mode & 0o077):
        raise PermissionError(f"Session path must be accessible only to its owner: {path}")


def prepare_lock_directory(path: Path) -> Path:
    _require_private_directories()
    path = _trusted_parent(path.parent) / path.name
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or _is_reparse_point(info):
        raise OSError(f"Unsafe session lock directory: {path}")
    _check_private(info, path)
    return path


@contextmanager
def private_staging_directory(parent: Path):
    _require_private_directories()
    with TemporaryDirectory(prefix=".telegram-mcp-", dir=_trusted_parent(parent)) as name:
        path = Path(name)
        _check_private(path.lstat(), path)
        yield path


def _check_regular(info: os.stat_result, path: Path) -> None:
    if not stat.S_ISREG(info.st_mode) or _is_reparse_point(info) or info.st_nlink != 1:
        raise OSError(f"Expected a regular session file with a single link: {path}")


_OPEN_FLAGS = (
    getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
)


def create_private_file(path: Path) -> int:
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | _OPEN_FLAGS, 0o600)
    try:
        opened = os.fstat(fd)
        _check_regular(opened, path)
        _check_private(opened, path)
        return fd
    except BaseException:
        os.close(fd)
        raise


def open_existing_regular(path: Path, flags: int, *, private: bool = False) -> int:
    before = path.lstat()
    _check_regular(before, path)
    fd = os.open(path, flags | _OPEN_FLAGS)
    try:
        opened = os.fstat(fd)
        current = path.lstat()
        _check_regular(opened, path)
        _check_regular(current, path)
        if (
            not opened.st_ino
            or not os.path.samestat(before, opened)
            or not os.path.samestat(opened, current)
        ):
            raise OSError(f"Session file changed while opening: {path}")
        if private:
            _check_private(opened, path)
        return fd
    except BaseException:
        os.close(fd)
        raise

"""Local storage for session credentials and their advisory locks."""

import os
import stat
import sys
from collections import deque
from contextlib import contextmanager
from ctypes import CDLL, POINTER, byref, c_char_p, c_int, c_void_p, get_errno
from errno import EINVAL, ENOENT
from functools import cache
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


@cache
def _darwin_acl_api():
    lib = CDLL(None, use_errno=True)
    lib.acl_get_link_np.argtypes = (c_char_p, c_int)
    lib.acl_get_link_np.restype = c_void_p
    lib.acl_get_fd_np.argtypes = (c_int, c_int)
    lib.acl_get_fd_np.restype = c_void_p
    lib.acl_get_entry.argtypes = (c_void_p, c_int, POINTER(c_void_p))
    lib.acl_get_tag_type.argtypes = (c_void_p, POINTER(c_int))
    lib.acl_free.argtypes = (c_void_p,)
    return lib


def _check_darwin_acl(path: Path, fd: int | None = None) -> None:
    if sys.platform != "darwin":
        return
    lib = _darwin_acl_api()
    ACL_TYPE_EXTENDED = 0x100
    ACL_FIRST_ENTRY = 0
    ACL_NEXT_ENTRY = -1
    ACL_EXTENDED_DENY = 2
    if fd is None:
        acl = lib.acl_get_link_np(os.fsencode(path), ACL_TYPE_EXTENDED)
    else:
        acl = lib.acl_get_fd_np(fd, ACL_TYPE_EXTENDED)
    if not acl:
        if get_errno() == ENOENT:
            return
        raise OSError(f"Cannot verify extended ACL on {path}")

    try:
        entry = c_void_p()
        tag = c_int()
        index = ACL_FIRST_ENTRY
        while lib.acl_get_entry(acl, index, byref(entry)) == 0:
            if lib.acl_get_tag_type(entry, byref(tag)) != 0:
                raise OSError(f"Cannot verify extended ACL entry on {path}")
            # A macOS ACL can grant access independently of the POSIX mode.
            # Deny entries, including the home directory's deny-delete, are safe.
            if tag.value != ACL_EXTENDED_DENY:
                raise PermissionError(f"Extended ACL allows access to {path}")
            index = ACL_NEXT_ENTRY
        if get_errno() != EINVAL:
            raise OSError(f"Cannot finish verifying extended ACL on {path}")
    finally:
        lib.acl_free(acl)


def _trusted_parent(path: Path, *, resolve_aliases: bool = True) -> Path:
    """Resolve parent aliases and reject ancestors that other users can replace."""
    if os.name == "posix":
        return _trusted_posix_parent(path.absolute(), resolve_aliases=resolve_aliases)

    path = path.resolve(strict=True) if resolve_aliases else Path(os.path.abspath(path))
    for directory in (*reversed(path.parents), path):
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or _is_reparse_point(info):
            raise OSError(f"Unsafe session directory: {directory}")
    return path


def _trusted_posix_parent(path: Path, *, resolve_aliases: bool) -> Path:
    resolved = Path(path.anchor)
    _check_darwin_acl(resolved)
    pending = deque(path.parts[1:])
    links_left = 40
    owners = (0, os.geteuid())
    while pending:
        part = pending.popleft()
        if part == "..":
            resolved = resolved.parent
            continue
        entry = resolved / part
        info = entry.lstat()
        if info.st_uid not in owners:
            raise PermissionError(f"Unsafe session directory owner: {entry}")
        _check_darwin_acl(entry)
        if stat.S_ISLNK(info.st_mode) and resolve_aliases and links_left:
            links_left -= 1
            target = entry.readlink()
            if target.is_absolute():
                resolved = Path(target.anchor)
                target = target.relative_to(resolved)
            pending.extendleft(reversed(target.parts))
            continue
        if not stat.S_ISDIR(info.st_mode):
            raise OSError(f"Unsafe session directory: {entry}")
        if info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX:
            raise PermissionError(f"Unsafe session directory permissions: {entry}")
        resolved = entry
    return resolved


def _check_private(info: os.stat_result, path: Path, fd: int | None = None) -> None:
    if os.name == "posix" and (info.st_uid != os.geteuid() or info.st_mode & 0o077):
        raise PermissionError(f"Session path must be accessible only to its owner: {path}")
    _check_darwin_acl(path, fd)


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


def prepare_private_directory(
    path: Path,
    *,
    create: bool = True,
    exist_ok: bool = True,
    resolve_parent_aliases: bool = True,
) -> Path:
    _require_private_directories()
    path = _trusted_parent(path.parent, resolve_aliases=resolve_parent_aliases) / path.name
    if create:
        path.mkdir(mode=0o700, exist_ok=exist_ok)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or _is_reparse_point(info):
        raise OSError(f"Expected a private, regular directory: {path}")
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
        _check_private(opened, path, fd)
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
            _check_private(opened, path, fd)
        return fd
    except BaseException:
        os.close(fd)
        raise

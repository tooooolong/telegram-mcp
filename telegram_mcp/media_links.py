"""Signed, expiring URLs that move files between the server and a remote client.

File-path tools read and write the server's disk, which a client on another
machine cannot reach. With TELEGRAM_MEDIA_BASE_URL set, download_media also
returns a URL that serves the downloaded file, and prepare_upload returns a URL
that accepts one file, which send_file and the other sending tools can then
read by path. Both are served by this server's own HTTP transport, carry an
HMAC over the path and an expiry, and stop working when they expire or the
server restarts. Both stay inside the first server allowed root.
"""

import hashlib
import hmac
import logging
import os
import re
import secrets
import shutil
import time
from pathlib import Path
from threading import Lock
from typing import Optional
from urllib.parse import quote, urlencode

from telegram_mcp.private_files import (
    create_private_file,
    open_existing_regular,
    prepare_private_directory,
)

MEDIA_ROUTE_PREFIX = "/media/"
UPLOAD_ROUTE_PREFIX = "/upload/"
TRANSFER_SUBDIR = ".telegram-mcp-transfers"
DOWNLOAD_SUBDIR = "downloads"
UPLOAD_SUBDIR = "uploads"
URL_TTL_SECONDS = 3600
MAX_TRANSFER_SLOTS = 64
_slot_lock = Lock()

_SLOT_ID = re.compile(r"[0-9a-f]{32}")
_SLOT_MARKER = ".telegram-mcp-slot"
UPLOAD_PARTIAL = ".telegram-mcp-part"

# Per process: a restart invalidates every URL handed out before it.
_SECRET = secrets.token_bytes(32)

logger = logging.getLogger(__name__)


def base_url() -> Optional[str]:
    """Public URL under which clients reach this server, or None if unset."""
    return os.getenv("TELEGRAM_MEDIA_BASE_URL", "").strip().rstrip("/") or None


def _now(now: Optional[float]) -> float:
    return now if now is not None else time.time()


def _signature(kind: str, rel_path: str, expires: int) -> str:
    message = f"{kind}\n{rel_path}\n{expires}".encode()
    return hmac.new(_SECRET, message, hashlib.sha256).hexdigest()


def _signed(kind: str, prefix: str, rel: str, now: Optional[float]) -> str:
    expires = int(_now(now)) + URL_TTL_SECONDS
    query = urlencode({"expires": expires, "sig": _signature(kind, rel, expires)})
    return f"{base_url()}{prefix}{quote(rel)}?{query}"


def _verified(kind: str, rel_path: str, expires: str, sig: str, now: Optional[float]) -> bool:
    try:
        expires_at = int(expires)
    except ValueError:
        return False
    if expires_at < _now(now) or not sig.isascii():
        return False
    return hmac.compare_digest(sig, _signature(kind, rel_path, expires_at))


def _transfer_directory(root: Path, kind: str, *, create: bool) -> Path:
    if os.name != "posix":
        raise OSError("Signed media transfers require POSIX private-directory checks.")
    base = prepare_private_directory(
        root / TRANSFER_SUBDIR, create=create, resolve_parent_aliases=False
    )
    return prepare_private_directory(base / kind, create=create, resolve_parent_aliases=False)


def _valid_filename(name: str) -> bool:
    return (
        name.isprintable()
        and name not in {".", "..", _SLOT_MARKER, UPLOAD_PARTIAL}
        and "/" not in name
        and "\\" not in name
        and len(os.fsencode(name)) <= 255
    )


def _slot_directory(path: Path, *, create: bool = False) -> Path:
    if not _SLOT_ID.fullmatch(path.name):
        raise ValueError("Invalid transfer slot identifier.")
    return prepare_private_directory(
        path, create=create, exist_ok=False, resolve_parent_aliases=False
    )


def _slot_filename(directory: Path) -> str:
    marker = directory / _SLOT_MARKER
    with os.fdopen(open_existing_regular(marker, os.O_RDONLY, private=True), "rb") as info:
        name = os.fsdecode(info.read(256))
    if not _valid_filename(name):
        raise ValueError("Invalid transfer slot metadata.")
    return name


def _clean_slot(slot: Path, name: str) -> None:
    try:
        (slot / name).unlink(missing_ok=True)
        (slot / UPLOAD_PARTIAL).unlink(missing_ok=True)
        (slot / _SLOT_MARKER).unlink(missing_ok=True)
        slot.rmdir()
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("Could not remove transfer slot %s: %s", slot.name, e)


def _count_active_slots(directory: Path, deadline: float) -> int:
    count = 0
    for path in directory.iterdir():
        if not _SLOT_ID.fullmatch(path.name):
            continue
        count += 1
        try:
            slot = _slot_directory(path)
            if slot.stat().st_mtime >= deadline:
                continue
            name = _slot_filename(slot)
        except (OSError, ValueError):
            continue
        _clean_slot(slot, name)
        if not slot.exists():
            count -= 1
    return count


def _new_slot(root: Path, kind: str, filename: str, now: Optional[float]) -> Path:
    with _slot_lock:
        directory = _transfer_directory(root, kind, create=True)
        if _count_active_slots(directory, _now(now) - 2 * URL_TTL_SECONDS) >= MAX_TRANSFER_SLOTS:
            raise OSError("Too many active signed transfers. Try again after older slots expire.")
        slot = _slot_directory(directory / secrets.token_hex(16), create=True)
        try:
            marker = slot / _SLOT_MARKER
            with os.fdopen(create_private_file(marker), "wb") as info:
                info.write(os.fsencode(filename))
        except BaseException:
            _clean_slot(slot, filename)
            raise
    return slot / filename


def _resolve_target(rel_path: str, root: Path, kind: str) -> Optional[Path]:
    slot_id, _, filename = rel_path.partition("/")
    if not _SLOT_ID.fullmatch(slot_id) or not _valid_filename(filename):
        return None
    try:
        directory = _transfer_directory(root, kind, create=False)
        slot = _slot_directory(directory / slot_id)
        if filename == _slot_filename(slot):
            return slot / filename
    except (OSError, ValueError):
        return None
    return None


def download_url(
    file_path: Path,
    root: Optional[Path],
    now: Optional[float] = None,
    *,
    source_path: Optional[Path] = None,
) -> Optional[str]:
    """Copy a completed download into a private slot and sign that transfer."""
    if base_url() is None or root is None:
        return None
    file_path = file_path.resolve()
    if not file_path.is_relative_to(root) or not _valid_filename(file_path.name):
        return None
    try:
        if source_path is not None:
            prepare_private_directory(
                source_path.parent, create=False, resolve_parent_aliases=False
            )
        with os.fdopen(open_existing_regular(source_path or file_path, os.O_RDONLY), "rb") as src:
            snapshot = _new_slot(root, DOWNLOAD_SUBDIR, file_path.name, now)
            try:
                with os.fdopen(create_private_file(snapshot), "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
            except BaseException:
                _clean_slot(snapshot.parent, snapshot.name)
                raise
    except OSError as exc:
        logger.warning("Could not reserve signed download storage: %s", exc)
        return None
    rel = f"{snapshot.parent.name}/{snapshot.name}"
    return _signed("media", MEDIA_ROUTE_PREFIX, rel, now)


def resolve_download(
    rel_path: str, expires: str, sig: str, root: Optional[Path], now: Optional[float] = None
) -> Optional[Path]:
    if root is None or not _verified("media", rel_path, expires, sig, now):
        return None
    target = _resolve_target(rel_path, root, DOWNLOAD_SUBDIR)
    if target is None:
        return None
    try:
        os.close(open_existing_regular(target, os.O_RDONLY, private=True))
    except OSError:
        return None
    return target


def upload_slot(
    filename: str, root: Optional[Path], now: Optional[float] = None
) -> Optional[dict]:
    """{url, file_path} for one upload under root, or None without a base URL, root or name."""
    name = Path(filename.replace("\\", "/")).name.strip()
    if base_url() is None or root is None or not _valid_filename(name):
        return None
    target = _new_slot(root, UPLOAD_SUBDIR, name, now)
    rel = f"{target.parent.name}/{name}"
    return {
        "url": _signed("upload", UPLOAD_ROUTE_PREFIX, rel, now),
        "file_path": str(target),
    }


def resolve_upload(
    rel_path: str, expires: str, sig: str, root: Optional[Path], now: Optional[float] = None
) -> Optional[Path]:
    """Where an upload URL writes, or None if it is invalid, expired or already used."""
    if root is None or not _verified("upload", rel_path, expires, sig, now):
        return None
    target = _resolve_target(rel_path, root, UPLOAD_SUBDIR)
    if target is None or target.exists():
        return None
    return target

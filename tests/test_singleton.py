from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from telegram_mcp import private_files, singleton
from telegram_mcp.singleton import SessionLock, SessionLockError, _PID_OFFSET, session_identity


def test_holder_pid_nonexistent(tmp_path: Path):
    lock = SessionLock("default", "test-absent", lock_dir=tmp_path)
    assert lock.holder_pid() is None


def test_holder_pid_empty_or_short(tmp_path: Path):
    lock = SessionLock("default", "test-empty", lock_dir=tmp_path)
    lock.path.touch(mode=0o600)
    lock.path.write_bytes(b"")
    assert lock.holder_pid() is None

    lock.path.write_bytes(b"123")  # less than _PID_OFFSET on Windows
    if _PID_OFFSET > 0:
        assert lock.holder_pid() is None
    else:
        assert lock.holder_pid() == 123


def test_holder_pid_corrupt_content(tmp_path: Path):
    lock = SessionLock("default", "test-corrupt", lock_dir=tmp_path)
    lock.path.touch(mode=0o600)
    lock.path.write_bytes(b" " * _PID_OFFSET + b"not_a_number")
    assert lock.holder_pid() is None

    lock.path.write_bytes(b" " * _PID_OFFSET + b"-42")
    assert lock.holder_pid() is None


def test_exclusive_lock_records_and_reports_pid(tmp_path: Path):
    lock_dir = tmp_path / "locks"
    lock = SessionLock("default", "test-exclusive", lock_dir=lock_dir)
    lock.acquire()
    try:
        assert lock.holder_pid() == os.getpid()
        if os.name != "nt":
            assert lock_dir.stat().st_mode & 0o777 == 0o700
            assert lock.path.stat().st_mode & 0o777 == 0o600
    finally:
        lock.release()
    assert lock.holder_pid() is None


def test_default_lock_directory_is_stable_across_temporary_directories(tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    profile.mkdir(mode=0o700)
    if os.name == "nt":
        monkeypatch.setenv("USERPROFILE", str(profile))
    else:
        monkeypatch.setattr(
            singleton.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=str(profile))
        )
    lock_dirs = []
    for temp_dir in (tmp_path / "first", tmp_path / "second"):
        temp_dir.mkdir()
        monkeypatch.setenv("TMPDIR", str(temp_dir))
        monkeypatch.setattr(tempfile, "tempdir", None)
        lock_dirs.append(singleton.session_lock_directory())

    assert lock_dirs == [profile.resolve() / ".telegram-mcp-locks"] * 2


def test_shared_lock_clears_holder_pid(tmp_path: Path):
    lock = SessionLock("default", "test-shared", lock_dir=tmp_path)
    lock.path.touch(mode=0o600)
    lock.path.write_bytes(b" " * _PID_OFFSET + b"99999\n")

    lock.acquire(shared=True)
    try:
        assert lock.holder_pid() is None
    finally:
        lock.release()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions only")
def test_lock_refuses_directory_accessible_to_other_users(tmp_path):
    lock_dir = tmp_path / "public"
    lock_dir.mkdir()
    lock_dir.chmod(0o755)

    with pytest.raises(SessionLockError):
        SessionLock("default", "test-untrusted", lock_dir=lock_dir).acquire(grace_seconds=0)

    assert list(lock_dir.iterdir()) == []


def test_lock_refuses_hard_link_before_changing_file(tmp_path):
    lock = SessionLock("default", "test-hard-link", lock_dir=tmp_path)
    original = tmp_path / "unrelated"
    original.touch(mode=0o600)
    original.write_bytes(b"unrelated data")
    os.link(original, lock.path)

    with pytest.raises(SessionLockError):
        lock.acquire(grace_seconds=0)

    assert original.read_bytes() == b"unrelated data"


def test_lock_checks_opened_file_before_changing_it(tmp_path, monkeypatch):
    lock = SessionLock("default", "test-lock-identity", lock_dir=tmp_path)
    lock.path.touch(mode=0o600)
    unrelated = tmp_path / "unrelated"
    unrelated.touch(mode=0o600)
    unrelated.write_bytes(b"unrelated data")

    real_open = os.open

    def open_different_file(path, flags, *args, **kwargs):
        if path == lock.path and not flags & os.O_CREAT:
            return real_open(unrelated, flags, *args, **kwargs)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(private_files.os, "open", open_different_file)
    with pytest.raises(SessionLockError, match="changed while opening"):
        lock.acquire(grace_seconds=0)

    assert unrelated.read_bytes() == b"unrelated data"


def test_multiprocess_collision_reports_child_pid(tmp_path: Path):
    script = f"""
import os, sys, time
from telegram_mcp.singleton import SessionLock
from pathlib import Path

lock = SessionLock("default", "mp-session", lock_dir=Path(r"{tmp_path}"))
lock.acquire()
print(f"ACQUIRED:{{os.getpid()}}", flush=True)
time.sleep(3)
lock.release()
"""
    proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        line = proc.stdout.readline().strip()
        assert line.startswith("ACQUIRED:")
        child_pid = int(line.split(":")[1])

        # Contender fails to acquire and correctly names the child's PID
        contender = SessionLock("default", "mp-session", lock_dir=tmp_path)
        with pytest.raises(SessionLockError, match=f"held by PID {child_pid}"):
            contender.acquire(grace_seconds=0.2, poll_interval=0.05)
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_session_identity_variants():
    class DummySession:
        def __init__(self, filename=None, save_val=None):
            self.filename = filename
            self.save_val = save_val

        def save(self):
            return self.save_val

    class DummyClient:
        def __init__(self, session=None):
            self.session = session

    c1 = DummyClient(DummySession(filename="foo.session"))
    assert session_identity(c1).startswith("file:")

    c2 = DummyClient(DummySession(save_val="auth_token_xyz"))
    assert session_identity(c2) == "string:auth_token_xyz"

    c3 = DummyClient(None)
    assert session_identity(c3).startswith("anon:")

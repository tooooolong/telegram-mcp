import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from telethon.crypto import AuthKey
from telethon.sessions import SQLiteSession, StringSession

import migrate_session


def test_env_names_support_default_and_labeled_accounts():
    assert migrate_session._env_names(None) == (
        "TELEGRAM_SESSION_STRING",
        "TELEGRAM_SESSION_NAME",
        "telegram_mcp_session",
    )
    assert migrate_session._env_names("Work-2") == (
        "TELEGRAM_SESSION_STRING_WORK_2",
        "TELEGRAM_SESSION_NAME_WORK_2",
        "telegram_mcp_session_work_2",
    )


def test_env_names_reject_invalid_account_label():
    with pytest.raises(ValueError, match="account label"):
        migrate_session._env_names("work account")


def test_target_paths_accepts_optional_session_suffix(tmp_path):
    without_suffix = tmp_path / "telegram"
    with_suffix = tmp_path / "telegram.session"

    assert migrate_session._target_paths(str(without_suffix)) == (
        str(without_suffix),
        with_suffix.resolve(),
    )
    assert migrate_session._target_paths(str(with_suffix)) == (
        str(without_suffix),
        with_suffix.resolve(),
    )


@pytest.mark.asyncio
async def test_migrate_copies_auth_key_verifies_and_hardens_file(tmp_path, monkeypatch):
    target = tmp_path / "telegram"
    session_file = tmp_path / "telegram.session"
    source = StringSession()
    source.set_dc(2, "149.154.167.40", 443)
    source.auth_key = AuthKey(bytes(range(256)))
    saved = {}

    def open_database(name):
        path = Path(name)
        assert path.is_file()
        if os.name != "nt":
            assert path.stat().st_mode & 0o777 == 0o600
            assert path.parent.stat().st_mode & 0o777 == 0o700
        return SQLiteSession(name)

    class FakeClient:
        def __init__(self, session, api_id, api_hash, **kwargs):
            saved["client"] = (session, api_id, api_hash, kwargs)

        async def connect(self):
            saved["connected"] = True

        async def is_user_authorized(self):
            return True

        async def get_me(self):
            return SimpleNamespace(id=42, first_name="Ada")

        async def disconnect(self):
            saved["disconnected"] = True

    monkeypatch.setattr(migrate_session, "SQLiteSession", open_database)
    monkeypatch.setattr(migrate_session, "TelegramClient", FakeClient)
    monkeypatch.setattr(
        migrate_session, "client_identity_kwargs", lambda: {"device_model": "test"}
    )

    created, user = await migrate_session.migrate(source.save(), 123, "hash", str(target))

    assert created == session_file.resolve()
    assert user.id == 42
    assert saved["client"][1:] == (123, "hash", {"device_model": "test"})
    assert saved["connected"] is True
    assert saved["disconnected"] is True
    if os.name != "nt":  # POSIX permissions only
        assert session_file.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.iterdir()) == [session_file]
    persistent = SQLiteSession(str(session_file))
    try:
        assert (persistent.dc_id, persistent.server_address, persistent.port) == (
            2,
            "149.154.167.40",
            443,
        )
        assert persistent.auth_key.key == source.auth_key.key
    finally:
        persistent.close()


@pytest.mark.asyncio
async def test_migrate_refuses_to_overwrite_existing_session(tmp_path):
    session_file = tmp_path / "telegram.session"
    session_file.touch()

    with pytest.raises(FileExistsError, match="already exists"):
        await migrate_session.migrate("secret", 123, "hash", str(tmp_path / "telegram"))


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["unauthorized", "failed_with_existing", "publish_collision"])
async def test_migrate_cleans_failed_database_and_keeps_other_files(
    tmp_path, monkeypatch, outcome
):
    target = tmp_path / "telegram"
    session_file = tmp_path / "telegram.session"
    source = SimpleNamespace(dc_id=2, server_address="server", port=443, auth_key=object())

    class FakeSQLiteSession:
        def __init__(self, name):
            path = Path(name if name.endswith(".session") else f"{name}.session")
            path.write_bytes(b"dummy SQLite content")

        def set_dc(self, *args):
            return None

        def save(self):
            return None

        def close(self):
            return None

        auth_key = None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            return None

        async def connect(self):
            return None

        async def is_user_authorized(self):
            if outcome != "unauthorized":
                session_file.write_bytes(b"unrelated existing file")
            return outcome == "publish_collision"

        async def get_me(self):
            return SimpleNamespace(id=42)

        async def disconnect(self):
            return None

    monkeypatch.setattr(migrate_session, "StringSession", lambda value: source)
    monkeypatch.setattr(migrate_session, "SQLiteSession", FakeSQLiteSession)
    monkeypatch.setattr(migrate_session, "TelegramClient", FakeClient)

    error = FileExistsError if outcome == "publish_collision" else RuntimeError
    with pytest.raises(error):
        await migrate_session.migrate("secret", 123, "hash", str(target))

    if outcome != "unauthorized":
        assert list(tmp_path.iterdir()) == [session_file]
        assert session_file.read_bytes() == b"unrelated existing file"
    else:
        assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_run_reports_missing_source_session(monkeypatch, capsys):
    monkeypatch.setenv("TELEGRAM_API_ID", "123")
    monkeypatch.setenv("TELEGRAM_API_HASH", "hash")
    monkeypatch.delenv("TELEGRAM_SESSION_STRING", raising=False)

    result = await migrate_session._run(SimpleNamespace(account=None, target=None))

    assert result == 1
    assert "TELEGRAM_SESSION_STRING must be set" in capsys.readouterr().err

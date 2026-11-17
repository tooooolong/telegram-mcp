import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

from telegram_mcp import runner
import telegram_mcp.runtime as runtime_module


class _FakeSession:
    def __init__(self, identity: str):
        self._identity = identity

    def save(self):
        return self._identity


class _FakeClient:
    def __init__(
        self,
        *,
        authorized: bool,
        identity: str = "test-identity",
        username: str | None = "work_account",
        usernames: list | None = None,
    ):
        self.authorized = authorized
        self.connected = False
        self.disconnected = False
        self.started = False
        self.get_me_calls = 0
        self.username = username
        self.usernames = usernames
        self.session = _FakeSession(identity)

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.disconnected = True

    async def is_user_authorized(self):
        return self.authorized

    async def get_me(self):
        self.get_me_calls += 1
        return SimpleNamespace(username=self.username, usernames=self.usernames)

    async def start(self):
        self.started = True


@pytest.fixture(autouse=True)
def _isolate_session_locks(tmp_path, monkeypatch):
    # Give each test its own lock directory (so locks don't leak across tests
    # or collide with a real telegram-mcp instance running on the machine)
    # and a near-zero grace period (so a deliberately-contested lock in a
    # test fails fast instead of sleeping through the real default).
    import telegram_mcp.singleton as singleton_module

    original_init = singleton_module.SessionLock.__init__

    def _init_with_tmp_dir(self, label, session_identity, *, lock_dir=tmp_path):
        original_init(self, label, session_identity, lock_dir=lock_dir)

    monkeypatch.setattr(singleton_module.SessionLock, "__init__", _init_with_tmp_dir)
    monkeypatch.setattr(runner, "_lock_grace_seconds", lambda: 0.01)
    # load_dotenv() may have pulled TELEGRAM_SESSION_LOCK from the developer's
    # .env; tests assume the exclusive default unless they set it themselves.
    monkeypatch.delenv("TELEGRAM_SESSION_LOCK", raising=False)
    monkeypatch.setattr(runner, "_service_configured", False)
    monkeypatch.setattr(
        runtime_module, "SERVER_ALLOWED_ROOTS", runtime_module.SERVER_ALLOWED_ROOTS
    )
    monkeypatch.setattr(
        runtime_module, "EXTENSION_ALLOWLISTS", runtime_module.EXTENSION_ALLOWLISTS
    )
    yield
    runner._session_locks.clear()


@pytest.mark.asyncio
async def test_connect_authorized_client_uses_existing_session_without_interactive_start():
    client = _FakeClient(authorized=True)

    await runner._connect_authorized_client("default", client)

    assert client.connected is True
    assert client.started is False


@pytest.mark.asyncio
async def test_connect_authorized_client_rejects_unauthorized_session():
    client = _FakeClient(authorized=False)

    with pytest.raises(RuntimeError, match="Interactive phone login is disabled"):
        await runner._connect_authorized_client("default", client)

    assert client.connected is True
    assert client.started is False


@pytest.mark.asyncio
async def test_connect_authorized_client_skips_identity_check_when_unset():
    client = _FakeClient(authorized=True, username="anyone")

    await runner._connect_authorized_client("default", client)

    assert client.get_me_calls == 0
    assert client.disconnected is False


@pytest.mark.asyncio
async def test_connect_authorized_client_accepts_expected_username(monkeypatch):
    monkeypatch.setenv("TELEGRAM_EXPECTED_USERNAME", " @Work_Account ")
    client = _FakeClient(authorized=True, username="work_account")

    await runner._connect_authorized_client("default", client)

    assert client.get_me_calls == 1
    assert client.disconnected is False


@pytest.mark.asyncio
async def test_connect_authorized_client_rejects_wrong_account(monkeypatch):
    monkeypatch.setenv("TELEGRAM_EXPECTED_USERNAME", "work_account")
    client = _FakeClient(authorized=True, username="personal_account")

    with pytest.raises(RuntimeError, match="different account") as excinfo:
        await runner._connect_authorized_client("default", client)

    assert "TELEGRAM_EXPECTED_USERNAME" in str(excinfo.value)
    # The other account's username is not echoed into logs.
    assert "personal_account" not in str(excinfo.value)
    assert client.disconnected is True


@pytest.mark.asyncio
async def test_connect_authorized_client_rejects_account_without_username(monkeypatch):
    monkeypatch.setenv("TELEGRAM_EXPECTED_USERNAME", "work_account")
    client = _FakeClient(authorized=True, username=None)

    with pytest.raises(RuntimeError, match="different account"):
        await runner._connect_authorized_client("default", client)

    assert client.disconnected is True


@pytest.mark.asyncio
async def test_connect_authorized_client_accepts_active_collectible_username(monkeypatch):
    monkeypatch.setenv("TELEGRAM_EXPECTED_USERNAME", "collectible")
    client = _FakeClient(
        authorized=True,
        username=None,
        usernames=[
            SimpleNamespace(username="inactive", active=False),
            SimpleNamespace(username="Collectible", active=True),
        ],
    )

    await runner._connect_authorized_client("default", client)

    assert client.disconnected is False


@pytest.mark.asyncio
async def test_connect_authorized_client_ignores_inactive_collectible_username(monkeypatch):
    monkeypatch.setenv("TELEGRAM_EXPECTED_USERNAME", "inactive")
    client = _FakeClient(
        authorized=True,
        username="primary",
        usernames=[SimpleNamespace(username="inactive", active=False)],
    )

    with pytest.raises(RuntimeError, match="different account"):
        await runner._connect_authorized_client("default", client)


@pytest.mark.asyncio
async def test_connect_authorized_client_per_label_expected_username(monkeypatch):
    monkeypatch.setenv("TELEGRAM_EXPECTED_USERNAME", "personal_account")
    monkeypatch.setenv("TELEGRAM_EXPECTED_USERNAME_WORK", "work_account")
    work = _FakeClient(authorized=True, identity="session-work", username="work_account")
    personal = _FakeClient(
        authorized=True, identity="session-personal", username="personal_account"
    )

    await runner._connect_authorized_client("work", work)
    await runner._connect_authorized_client("personal", personal)

    assert work.disconnected is False
    assert personal.disconnected is False


@pytest.mark.asyncio
async def test_connect_authorized_client_per_label_mismatch_names_label(monkeypatch):
    monkeypatch.setenv("TELEGRAM_EXPECTED_USERNAME_WORK", "work_account")
    client = _FakeClient(authorized=True, username="personal_account")

    with pytest.raises(RuntimeError, match="TELEGRAM_EXPECTED_USERNAME_WORK") as excinfo:
        await runner._connect_authorized_client("work", client)

    assert "'work'" in str(excinfo.value)
    assert client.disconnected is True


@pytest.mark.asyncio
async def test_connect_authorized_client_refuses_concurrent_duplicate_session():
    first = _FakeClient(authorized=True, identity="shared-session")
    second = _FakeClient(authorized=True, identity="shared-session")

    await runner._connect_authorized_client("default", first)

    with pytest.raises(runner.SessionLockError, match="already connected"):
        await runner._connect_authorized_client("default", second)

    assert second.connected is False

    runner._session_locks["default"].release()
    runner._session_locks.clear()


@pytest.mark.asyncio
async def test_connect_authorized_client_allows_different_sessions_concurrently():
    first = _FakeClient(authorized=True, identity="session-a")
    second = _FakeClient(authorized=True, identity="session-b")

    await runner._connect_authorized_client("default", first)
    await runner._connect_authorized_client("work", second)

    assert first.connected is True
    assert second.connected is True


@pytest.mark.asyncio
async def test_connect_authorized_client_refuses_same_session_under_another_label():
    first = _FakeClient(authorized=True, identity="shared-session")
    second = _FakeClient(authorized=True, identity="shared-session")

    await runner._connect_authorized_client("default", first)

    with pytest.raises(runner.SessionLockError, match="already connected"):
        await runner._connect_authorized_client("work", second)

    assert second.connected is False

    runner._session_locks["default"].release()
    runner._session_locks.clear()


@pytest.mark.asyncio
async def test_shared_lock_mode_lets_instances_share_a_session(monkeypatch):
    monkeypatch.setenv("TELEGRAM_SESSION_LOCK", "shared")
    first = _FakeClient(authorized=True, identity="shared-session")
    second = _FakeClient(authorized=True, identity="shared-session")

    await runner._connect_authorized_client("default", first)
    first_lock = runner._session_locks["default"]
    await runner._connect_authorized_client("default", second)

    assert first.connected is True
    assert second.connected is True

    first_lock.release()
    runner._session_locks["default"].release()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first_mode, second_mode", [("exclusive", "shared"), ("shared", "exclusive")]
)
async def test_shared_and_exclusive_instances_never_overlap(monkeypatch, first_mode, second_mode):
    first = _FakeClient(authorized=True, identity="shared-session")
    second = _FakeClient(authorized=True, identity="shared-session")

    monkeypatch.setenv("TELEGRAM_SESSION_LOCK", first_mode)
    await runner._connect_authorized_client("default", first)
    first_lock = runner._session_locks["default"]

    monkeypatch.setenv("TELEGRAM_SESSION_LOCK", second_mode)
    with pytest.raises(runner.SessionLockError, match="already connected"):
        await runner._connect_authorized_client("default", second)

    assert second.connected is False
    first_lock.release()


@pytest.mark.asyncio
async def test_invalid_transport_exits_with_error(monkeypatch, capsys):
    monkeypatch.setenv("MCP_TRANSPORT", "ftp")

    with pytest.raises(SystemExit) as excinfo:
        await runner._main()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "ftp" in captured.err
    assert "stdio" in captured.err


@pytest.mark.asyncio
async def test_lock_error_names_the_exclusive_holder():
    first = _FakeClient(authorized=True, identity="shared-session")
    second = _FakeClient(authorized=True, identity="shared-session")

    await runner._connect_authorized_client("default", first)
    lock = runner._session_locks["default"]

    with pytest.raises(runner.SessionLockError, match=f"held by PID {os.getpid()}"):
        await runner._connect_authorized_client("default", second)

    lock.release()
    assert lock.path.read_text() == ""  # a released lock names nobody


@pytest.mark.parametrize(
    "value, shared",
    [(None, False), ("", False), ("exclusive", False), ("shared", True), (" Shared ", True)],
)
def test_session_lock_mode_parsing(monkeypatch, value, shared):
    if value is None:
        monkeypatch.delenv("TELEGRAM_SESSION_LOCK", raising=False)
    else:
        monkeypatch.setenv("TELEGRAM_SESSION_LOCK", value)

    assert runner._session_lock_shared() is shared


def test_session_lock_mode_rejects_unknown_values(monkeypatch):
    monkeypatch.setenv("TELEGRAM_SESSION_LOCK", "sometimes")

    with pytest.raises(SystemExit, match="Invalid TELEGRAM_SESSION_LOCK 'sometimes'"):
        runner._session_lock_shared()


class _FakeSettings:
    def __init__(self):
        self.host = None
        self.port = None
        self.transport_security = None


class _FakeMcp:
    def __init__(self):
        self.settings = _FakeSettings()
        self.ran = None

    async def run_stdio_async(self):
        self.ran = "stdio"

    async def run_sse_async(self):
        self.ran = "sse"

    async def run_streamable_http_async(self):
        self.ran = "http"


@pytest.mark.asyncio
async def test_serve_stdio(monkeypatch):
    fake = _FakeMcp()
    monkeypatch.setattr(runner, "mcp", fake)

    await runner._serve("stdio")

    assert fake.ran == "stdio"


@pytest.mark.asyncio
async def test_embedded_serve_rejects_unknown_transport(monkeypatch):
    from telegram_mcp import server

    fake = _FakeMcp()
    monkeypatch.setattr(runner, "mcp", fake)

    with pytest.raises(ValueError, match="Invalid MCP_TRANSPORT"):
        await server.serve("unknown")

    assert fake.ran is None


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["http", "sse"])
async def test_serve_requires_configured_token_for_network_transports(monkeypatch, transport):
    fake = _FakeMcp()
    monkeypatch.setattr(runner, "mcp", fake)
    monkeypatch.delenv("MCP_AUTH_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="MCP_AUTH_TOKEN"):
        await runner._serve(transport)

    assert fake.ran is None


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["http", "sse"])
async def test_serve_http_transports_bind_host_and_port(monkeypatch, transport):
    fake = _FakeMcp()
    monkeypatch.setattr(runner, "mcp", fake)
    monkeypatch.setenv("MCP_AUTH_TOKEN", "test-operator-token")
    monkeypatch.setenv("MCP_HOST", "0.0.0.0")
    monkeypatch.setenv("MCP_PORT", "9000")

    await runner._serve(transport)

    assert fake.ran == transport
    assert fake.settings.host == "0.0.0.0"
    assert fake.settings.port == 9000


@pytest.mark.asyncio
async def test_serve_http_uses_default_host_and_port(monkeypatch):
    fake = _FakeMcp()
    monkeypatch.setattr(runner, "mcp", fake)
    monkeypatch.setenv("MCP_AUTH_TOKEN", "test-operator-token")
    monkeypatch.delenv("MCP_HOST", raising=False)
    monkeypatch.delenv("MCP_PORT", raising=False)

    await runner._serve("http")

    assert fake.ran == "http"
    assert fake.settings.host == "127.0.0.1"
    assert fake.settings.port == 8765


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
async def test_serve_http_enables_localhost_only_protection_by_default(monkeypatch, host):
    """Host/Origin validation must be on even without MCP_ALLOWED_HOSTS.

    Otherwise any web page in the user's browser could POST tool calls to the
    local server (cross-origin request forgery / DNS rebinding).
    """
    fake = _FakeMcp()
    monkeypatch.setattr(runner, "mcp", fake)
    monkeypatch.setenv("MCP_AUTH_TOKEN", "test-operator-token")
    monkeypatch.setenv("MCP_HOST", host)
    monkeypatch.delenv("MCP_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("MCP_ALLOWED_ORIGINS", raising=False)

    await runner._serve("http")

    security = fake.settings.transport_security
    assert security.enable_dns_rebinding_protection is True
    assert security.allowed_hosts == ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    assert security.allowed_origins == [
        "http://127.0.0.1:*",
        "http://localhost:*",
        "http://[::1]:*",
    ]


@pytest.mark.asyncio
async def test_serve_http_allowed_origins_without_hosts_replace_local_origins(monkeypatch):
    fake = _FakeMcp()
    monkeypatch.setattr(runner, "mcp", fake)
    monkeypatch.setenv("MCP_AUTH_TOKEN", "test-operator-token")
    monkeypatch.delenv("MCP_ALLOWED_HOSTS", raising=False)
    monkeypatch.setenv("MCP_ALLOWED_ORIGINS", "http://localhost:3000")

    await runner._serve("sse")

    security = fake.settings.transport_security
    assert security.enable_dns_rebinding_protection is True
    assert security.allowed_hosts == ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    assert security.allowed_origins == ["http://localhost:3000"]


@pytest.mark.asyncio
async def test_serve_http_configures_allowed_hosts(monkeypatch):
    fake = _FakeMcp()
    monkeypatch.setattr(runner, "mcp", fake)
    monkeypatch.setenv("MCP_AUTH_TOKEN", "test-operator-token")
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", "mcp.example.com, localhost:8765")
    monkeypatch.setenv("MCP_ALLOWED_ORIGINS", "https://mcp.example.com")

    await runner._serve("http")

    security = fake.settings.transport_security
    assert security.enable_dns_rebinding_protection is True
    assert security.allowed_hosts == ["mcp.example.com", "localhost:8765"]
    assert security.allowed_origins == ["https://mcp.example.com"]


def test_file_extension_overrides_are_validated_before_tools_are_pruned(monkeypatch):
    """TELEGRAM_FILE_EXTENSIONS must not be rejected for a tool that exposure hid.

    ``_apply_exposed_tools_mode`` removes non-exposed tools from the tool
    manager, and ``_apply_file_extension_overrides`` validates tool names
    against that same manager. Running them in the wrong order aborts startup
    with "unknown tool send_file" on a configuration that is perfectly valid:
    narrowing send_file's extensions while send_file is not exposed at all.
    """
    calls: list[str] = []

    monkeypatch.setattr(
        runner._runtime,
        "_apply_exposed_tools_mode",
        lambda *a, **k: calls.append("exposed") or ([], []),
    )
    monkeypatch.setattr(
        runner._runtime,
        "_apply_file_extension_overrides",
        lambda *a, **k: calls.append("extensions") or {},
    )
    monkeypatch.setattr(runner, "_configure_allowed_roots_from_cli", lambda *a, **k: None)
    monkeypatch.setattr(runner._transcription, "validate_transcription_config", lambda: None)
    monkeypatch.setattr(runner, "_session_lock_shared", lambda: None)
    monkeypatch.setattr(runner.asyncio, "run", lambda coro: coro.close())

    runner.main()

    assert calls.index("extensions") < calls.index("exposed"), (
        "TELEGRAM_FILE_EXTENSIONS must be validated against the full tool set, "
        "before TELEGRAM_EXPOSED_TOOLS prunes it"
    )


def test_cli_transport_flag_sets_env_var(monkeypatch):
    """--transport http writes _CLI_TRANSPORT; main() propagates it to MCP_TRANSPORT."""
    from telegram_mcp import runtime

    monkeypatch.delenv("MCP_TRANSPORT", raising=False)
    runtime._configure_allowed_roots_from_cli(["--transport", "http"])
    assert runtime._CLI_TRANSPORT == "http"

    monkeypatch.setattr(runner, "_configure_allowed_roots_from_cli", lambda *a, **k: None)
    monkeypatch.setattr(runner._runtime, "_CLI_TRANSPORT", "http")
    monkeypatch.setattr(runner._runtime, "_CLI_HOST", None)
    monkeypatch.setattr(runner._runtime, "_CLI_PORT", None)
    monkeypatch.setattr(runner._runtime, "_apply_file_extension_overrides", lambda: None)
    monkeypatch.setattr(runner._runtime, "_apply_exposed_tools_mode", lambda: ([], []))
    monkeypatch.setattr(runner._transcription, "validate_transcription_config", lambda: None)
    monkeypatch.setattr(runner, "_session_lock_shared", lambda: None)
    monkeypatch.setattr(runner.asyncio, "run", lambda coro: coro.close())

    runner.main()

    assert os.environ.get("MCP_TRANSPORT") == "http"


def test_cli_transport_flag_overrides_env_var(monkeypatch):
    """CLI --transport stdio overrides MCP_TRANSPORT=http already in environment."""
    from telegram_mcp import runtime

    monkeypatch.setenv("MCP_TRANSPORT", "http")
    runtime._configure_allowed_roots_from_cli(["--transport", "stdio"])
    assert runtime._CLI_TRANSPORT == "stdio"


def test_cli_host_and_port_flags(monkeypatch):
    """--host and --port populate _CLI_HOST and _CLI_PORT in runtime."""
    from telegram_mcp import runtime

    runtime._configure_allowed_roots_from_cli(["--host", "0.0.0.0", "--port", "9000"])
    assert runtime._CLI_HOST == "0.0.0.0"
    assert runtime._CLI_PORT == 9000


def test_main_prints_the_tools_exposure_hides(monkeypatch, capsys):
    """The hidden names are what a user copies into read-only+<tool>."""
    monkeypatch.setattr(runner, "_configure_allowed_roots_from_cli", lambda *a, **k: None)
    monkeypatch.setattr(runner._runtime, "_apply_file_extension_overrides", lambda: None)
    monkeypatch.setattr(
        runner._runtime,
        "_apply_exposed_tools_mode",
        lambda: (["send_message", "export_chat_invite"], []),
    )
    monkeypatch.setattr(runner._transcription, "validate_transcription_config", lambda: None)
    monkeypatch.setattr(runner, "_session_lock_shared", lambda: None)
    monkeypatch.setattr(runner.asyncio, "run", lambda coro: coro.close())

    runner.main()

    err = capsys.readouterr().err
    assert "hides 2 tool(s)" in err
    assert "export_chat_invite, send_message" in err


def _run_main_with_exposure_result(monkeypatch, result):
    monkeypatch.setattr(runner, "_configure_allowed_roots_from_cli", lambda *a, **k: None)
    monkeypatch.setattr(runner._runtime, "_apply_file_extension_overrides", lambda: None)
    monkeypatch.setattr(runner._runtime, "_apply_exposed_tools_mode", lambda: result)
    monkeypatch.setattr(runner._transcription, "validate_transcription_config", lambda: None)
    monkeypatch.setattr(runner, "_session_lock_shared", lambda: None)
    monkeypatch.setattr(runner.asyncio, "run", lambda coro: coro.close())

    runner.main()


def test_main_reports_excluded_tools_separately(monkeypatch, capsys):
    """Excluded names are not '+' candidates, so they get their own line."""
    _run_main_with_exposure_result(
        monkeypatch, (["delete_message", "ban_user"], ["list_chats", "get_participants"])
    )

    err = capsys.readouterr().err
    assert (
        "hides 2 tool(s); list any of them after '+' to expose it: ban_user, delete_message" in err
    )
    assert "excludes 2 tool(s) listed after '-': get_participants, list_chats" in err


def test_main_reports_only_exclusions_for_all_base(monkeypatch, capsys):
    _run_main_with_exposure_result(monkeypatch, ([], ["delete_message", "delete_messages_bulk"]))

    err = capsys.readouterr().err
    assert "after '+'" not in err
    assert "excludes 2 tool(s) listed after '-': delete_message, delete_messages_bulk" in err


# --- an implicit default account that never logged in is skipped, not fatal -------


def _only_session_env(monkeypatch, **env):
    for key in list(os.environ):
        if key.startswith(("TELEGRAM_SESSION_STRING", "TELEGRAM_SESSION_NAME")):
            monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(runtime_module, "StringSession", lambda value=None: f"str::{value}")
    monkeypatch.setattr(runtime_module, "_build_client", lambda session, label: object())
    monkeypatch.setattr(runtime_module, "OPTIONAL_ACCOUNTS", set())


def test_preset_session_name_next_to_labelled_accounts_is_optional(monkeypatch):
    _only_session_env(
        monkeypatch,
        TELEGRAM_SESSION_STRING_WORK="work-string",
        TELEGRAM_SESSION_NAME="telegram_session",
    )

    accounts = runtime_module._discover_accounts()

    assert set(accounts) == {"work", "default"}
    assert runtime_module.OPTIONAL_ACCOUNTS == {"default"}


def test_lone_session_name_stays_required(monkeypatch):
    _only_session_env(monkeypatch, TELEGRAM_SESSION_NAME="telegram_session")

    accounts = runtime_module._discover_accounts()

    assert set(accounts) == {"default"}
    assert runtime_module.OPTIONAL_ACCOUNTS == set()


class _DialogsClient(_FakeClient):
    async def get_dialogs(self):
        return []


@pytest.mark.asyncio
async def test_unauthorized_optional_default_is_skipped(monkeypatch, capsys):
    work = _DialogsClient(authorized=True, identity="work-session")
    default = _DialogsClient(authorized=False, identity="default-session")
    accounts = {"work": work, "default": default}
    monkeypatch.setattr(runner, "clients", accounts)
    monkeypatch.setattr(runtime_module, "OPTIONAL_ACCOUNTS", {"default"})

    await asyncio.gather(
        *(runner._connect_or_skip(label, client) for label, client in list(accounts.items()))
    )

    assert list(accounts) == ["work"]
    assert default.disconnected is True
    assert "default" not in runner._session_locks
    assert "Skipping account 'default'" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_authorized_optional_default_is_kept(monkeypatch):
    accounts = {
        "work": _DialogsClient(authorized=True, identity="work-session"),
        "default": _DialogsClient(authorized=True, identity="default-session"),
    }
    monkeypatch.setattr(runner, "clients", accounts)
    monkeypatch.setattr(runtime_module, "OPTIONAL_ACCOUNTS", {"default"})

    for label, client in list(accounts.items()):
        await runner._connect_or_skip(label, client)

    assert list(accounts) == ["work", "default"]


@pytest.mark.asyncio
async def test_unauthorized_labelled_account_still_refuses_to_start(monkeypatch):
    accounts = {"work": _DialogsClient(authorized=False, identity="work-session")}
    monkeypatch.setattr(runner, "clients", accounts)
    monkeypatch.setattr(runtime_module, "OPTIONAL_ACCOUNTS", {"default"})

    with pytest.raises(RuntimeError, match="'work' is not authorized"):
        await runner._connect_or_skip("work", accounts["work"])

    assert list(accounts) == ["work"]


@pytest.mark.asyncio
async def test_main_serves_labelled_accounts_when_preset_default_is_not_logged_in(
    monkeypatch, capsys
):
    accounts = {
        "work": _DialogsClient(authorized=True, identity="work-session"),
        "default": _DialogsClient(authorized=False, identity="default-session"),
    }
    served = []

    async def fake_serve(transport):
        async with runtime_module.mcp.service_lifespan():
            served.append(transport)

    monkeypatch.setattr(runner, "clients", accounts)
    monkeypatch.setattr(runtime_module, "OPTIONAL_ACCOUNTS", {"default"})
    monkeypatch.setattr(runner, "_serve", fake_serve)
    monkeypatch.setenv("MCP_TRANSPORT", "stdio")

    await runner._main()

    err = capsys.readouterr().err
    assert served == ["stdio"]
    assert list(accounts) == ["work"]
    assert "Skipping account 'default'" in err
    assert "Telegram client(s) started (work)." in err


@pytest.mark.asyncio
async def test_embedded_stdio_starts_and_releases_the_same_sessions(monkeypatch):
    from mcp.server.fastmcp import FastMCP
    from telegram_mcp import server
    from telegram_mcp.tools import events

    account = _DialogsClient(authorized=True, identity="work-session")
    monkeypatch.setattr(runner, "clients", {"work": account})
    monkeypatch.setenv("TELEGRAM_EXPECTED_USERNAME_WORK", "work_account")
    monkeypatch.setattr(events, "_feed_task", None)
    feed_task = None

    async def fake_stdio(self):
        nonlocal feed_task
        assert self is runtime_module.mcp
        assert account.get_me_calls == 1
        assert "work" in runner._session_locks
        assert not account.disconnected
        feed_task = asyncio.create_task(asyncio.sleep(3600))
        events._feed_task = feed_task

    monkeypatch.setattr(FastMCP, "run_stdio_async", fake_stdio)
    await server.serve("stdio")

    assert account.disconnected
    assert not runner._session_locks
    assert feed_task.cancelled()
    assert not events.feed_enabled()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 0.0),
        ("", 0.0),
        ("0", 0.0),
        ("120", 120.0),
        ("2.5", 30.0),
        ("-5", 0.0),
        ("soon", 0.0),
    ],
)
def test_catch_up_interval(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("TELEGRAM_CATCH_UP_INTERVAL", raising=False)
    else:
        monkeypatch.setenv("TELEGRAM_CATCH_UP_INTERVAL", raw)
    assert runner._catch_up_interval() == expected


class _CatchUpClient:
    def __init__(self, connected=True, error=None):
        self.connected = connected
        self.error = error
        self.catch_ups = 0

    def is_connected(self):
        return self.connected

    async def catch_up(self):
        self.catch_ups += 1
        if self.error:
            raise self.error


@pytest.mark.asyncio
async def test_periodic_catch_up_skips_disconnected_clients_and_survives_errors(monkeypatch):
    up = _CatchUpClient()
    down = _CatchUpClient(connected=False)
    failing = _CatchUpClient(error=ConnectionError("gone"))
    monkeypatch.setattr(runner, "clients", {"up": up, "down": down, "failing": failing})

    task = asyncio.create_task(runner._periodic_catch_up(0.01))

    async def _two_rounds():
        while up.catch_ups < 2:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_two_rounds(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert down.catch_ups == 0
    assert failing.catch_ups >= 2  # an error does not stop the loop

"""Tests for MCP 2026-07-28 protocol negotiation, backward compatibility, and discovery."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.server.fastmcp.exceptions import ToolError
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Mount
from starlette.testclient import TestClient

from telegram_mcp.transport import (
    DEFAULT_PROTOCOL_VERSION,
    LATEST_PROTOCOL_VERSION,
    TelegramFastMCP,
    SUPPORTED_PROTOCOL_VERSIONS,
    UnsupportedProtocolVersionError,
    create_streamable_http_app,
    install_protocol_compatibility,
    is_supported_protocol_version,
    negotiate_protocol_version,
)
from telegram_mcp import runner, runtime
from telegram_mcp.runtime import mcp
import telegram_mcp.server as server

_TEST_TOKEN = "test-operator-token"
_AUTH_HEADERS = {"Authorization": f"Bearer {_TEST_TOKEN}"}


@pytest.fixture(autouse=True)
def _configure_auth(monkeypatch):
    @asynccontextmanager
    async def test_lifespan():
        yield

    monkeypatch.setattr(mcp, "_service_lifespan", test_lifespan)
    monkeypatch.setenv("MCP_AUTH_TOKEN", _TEST_TOKEN)


@pytest.mark.parametrize(
    "app_factory",
    [server.get_http_app, mcp.streamable_http_app, mcp.sse_app],
)
@pytest.mark.parametrize("token", [None, "", "   ", "invalid token"])
def test_mcp_network_apps_require_valid_token_configuration(monkeypatch, app_factory, token):
    if token is None:
        monkeypatch.delenv("MCP_AUTH_TOKEN")
    else:
        monkeypatch.setenv("MCP_AUTH_TOKEN", token)

    with pytest.raises(RuntimeError, match="MCP_AUTH_TOKEN"):
        app_factory()


@pytest.mark.parametrize(
    "authorization",
    [
        {},
        {"Authorization": "Basic test-operator-token"},
        {"Authorization": "Bearer wrong-token"},
        [("Authorization", "Bearer test-operator-token"), ("Authorization", "Bearer wrong-token")],
    ],
)
def test_http_rejects_unauthorized_discovery_and_mcp_requests(authorization):
    with TestClient(server.get_http_app(), base_url="http://localhost:8765") as client:
        get = client.get("/mcp", headers=authorization)
        post = client.post("/mcp", headers=authorization, json=_TOOLS_LIST)
        assert get.status_code == post.status_code == 401
        assert get.headers["www-authenticate"] == "Bearer"


def test_http_custom_routes_require_auth_even_with_media_prefix():
    instance = TelegramFastMCP("test")

    @instance.custom_route("/media/status", methods=["GET"])
    async def status(request):
        return PlainTextResponse("ready")

    with TestClient(instance.streamable_http_app(), base_url="http://localhost:8765") as client:
        assert client.get("/media/status").status_code == 401
        assert client.get("/media/status", headers=_AUTH_HEADERS).text == "ready"


def test_sse_requires_bearer_auth_for_both_endpoints():
    with TestClient(mcp.sse_app(), base_url="http://localhost:8765") as client:
        assert client.get("/sse").status_code == 401
        assert client.post("/messages/", json=_TOOLS_LIST).status_code == 401
        response = client.post("/messages/", headers=_AUTH_HEADERS, json=_TOOLS_LIST)
        assert response.status_code == 400
        assert "session_id" in response.text


@pytest.mark.parametrize("factory", ["streamable_http_app", "sse_app"])
def test_authorized_browser_can_read_from_allowed_origin(monkeypatch, factory):
    monkeypatch.setenv("MCP_ALLOWED_ORIGINS", "https://client.example.com")
    instance = TelegramFastMCP("test")

    @instance.custom_route("/status", methods=["GET"])
    async def status(request):
        return PlainTextResponse("ready", headers={"Vary": "Accept"})

    app = getattr(instance, factory)()
    headers = {**_AUTH_HEADERS, "Origin": "https://client.example.com"}
    with TestClient(app, base_url="http://localhost:8765") as client:
        response = client.get("/status", headers=headers)

    assert response.text == "ready"
    assert response.headers["access-control-allow-origin"] == "https://client.example.com"
    assert response.headers["vary"] == "Accept, Origin"


@pytest.mark.asyncio
async def test_sse_authorized_client_receives_message_endpoint():
    app = mcp.sse_app()
    endpoint_sent = asyncio.Event()
    response = []

    async def receive():
        await endpoint_sent.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        response.append(message)
        if b"event: endpoint" in message.get("body", b""):
            endpoint_sent.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/sse",
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"host", b"localhost:8765"),
            (b"authorization", f"Bearer {_TEST_TOKEN}".encode()),
        ],
    }
    async with app.lifespan():
        await asyncio.wait_for(app(scope, receive, send), timeout=2)

    assert response[0]["status"] == 200
    assert any(b"data: /messages/?session_id=" in m.get("body", b"") for m in response)


@pytest.mark.parametrize(
    "app_factory, path",
    [(server.get_http_app, "/mcp"), (mcp.sse_app, "/sse")],
)
def test_embedded_app_requires_lifespan(app_factory, path):
    client = TestClient(app_factory(), base_url="http://localhost:8765")

    response = client.get(path, headers=_AUTH_HEADERS)

    assert response.status_code == 503
    assert "has not completed startup" in response.text


@pytest.mark.asyncio
async def test_direct_tool_calls_require_service_lifespan():
    with pytest.raises(ToolError, match="has not completed startup"):
        await server.get_server().call_tool("list_accounts", {})


@pytest.fixture
def embedded_client(tmp_path, monkeypatch):
    account = SimpleNamespace(
        connect=AsyncMock(),
        disconnect=AsyncMock(),
        is_user_authorized=AsyncMock(return_value=True),
        get_me=AsyncMock(return_value=SimpleNamespace(username="work_account")),
        get_dialogs=AsyncMock(return_value=[]),
    )
    session_lock = runner.SessionLock
    monkeypatch.setattr(
        runner,
        "SessionLock",
        lambda label, identity: session_lock(label, identity, lock_dir=tmp_path),
    )
    monkeypatch.setattr(runner, "clients", {"work": account})
    monkeypatch.setattr(runner, "_service_configured", False)
    monkeypatch.setattr(runtime, "SERVER_ALLOWED_ROOTS", runtime.SERVER_ALLOWED_ROOTS)
    monkeypatch.setattr(runtime, "EXTENSION_ALLOWLISTS", runtime.EXTENSION_ALLOWLISTS)
    monkeypatch.setattr(mcp._tool_manager, "_tools", dict(mcp._tool_manager._tools))
    monkeypatch.setattr(mcp, "_service_lifespan", runtime._telegram_service_lifespan)
    monkeypatch.setenv("TELEGRAM_EXPECTED_USERNAME_WORK", "work_account")
    return account


@pytest.mark.parametrize("app_factory", [server.get_http_app, mcp.sse_app])
def test_embedded_lifespan_applies_security_configuration(
    app_factory, embedded_client, tmp_path, monkeypatch
):
    root = tmp_path / "allowed"
    monkeypatch.setenv("TELEGRAM_ALLOWED_ROOTS", str(root))
    monkeypatch.setenv("TELEGRAM_FILE_EXTENSIONS", "send_file:.pdf")
    monkeypatch.setenv("TELEGRAM_EXPOSED_TOOLS", "read-only")
    app = app_factory()
    embedded_client.connect.assert_not_called()

    with TestClient(app, base_url="http://localhost:8765"):
        embedded_client.get_me.assert_awaited_once()
        assert runner._session_locks.keys() == {"work"}
        assert runtime.SERVER_ALLOWED_ROOTS == [root.resolve()]
        assert runtime.EXTENSION_ALLOWLISTS["send_file"] == {".pdf"}
        tools = {tool.name for tool in mcp._tool_manager.list_tools()}
        assert "list_accounts" in tools
        assert "send_file" not in tools

    embedded_client.disconnect.assert_awaited()
    assert not runner._session_locks


def test_embedded_lifespan_rejects_wrong_account_and_releases_lock(embedded_client, monkeypatch):
    monkeypatch.setenv("TELEGRAM_EXPECTED_USERNAME_WORK", "different_account")

    with pytest.raises(RuntimeError, match="logged in to a different account"):
        with TestClient(server.get_http_app(), base_url="http://localhost:8765"):
            pytest.fail("The app should not start with an unexpected Telegram account")

    embedded_client.disconnect.assert_awaited()
    assert not runner._session_locks
    assert mcp._service_loop is None


def test_mounted_embedded_app_uses_parent_lifespan(embedded_client):
    app = server.get_http_app()

    @asynccontextmanager
    async def lifespan(_application):
        async with app.lifespan():
            yield

    parent = Starlette(routes=[Mount("/telegram", app)], lifespan=lifespan)
    with TestClient(parent, base_url="http://localhost:8765", headers=_AUTH_HEADERS) as client:
        response = client.post("/telegram/mcp", headers=_POST_HEADERS, json=_TOOLS_LIST)
        assert response.status_code == 200
        assert '"tools":' in response.text
        embedded_client.get_me.assert_awaited_once()

    embedded_client.disconnect.assert_awaited()


def test_embedded_app_builds_separate_streamable_http_session_managers(embedded_client):
    apps = [server.get_http_app(), server.get_http_app()]
    for app in apps:
        with TestClient(app, base_url="http://localhost:8765", headers=_AUTH_HEADERS) as client:
            response = client.post("/mcp", headers=_POST_HEADERS, json=_TOOLS_LIST)
            assert response.status_code == 200
            assert '"tools":' in response.text

    assert embedded_client.get_me.await_count == 2


def test_supported_protocol_versions():
    """Verify supported protocol revisions and constants."""
    assert "2026-07-28" in SUPPORTED_PROTOCOL_VERSIONS
    assert "2025-11-25" in SUPPORTED_PROTOCOL_VERSIONS
    assert "2025-03-26" in SUPPORTED_PROTOCOL_VERSIONS
    assert "2024-11-05" in SUPPORTED_PROTOCOL_VERSIONS
    assert LATEST_PROTOCOL_VERSION == "2026-07-28"
    assert DEFAULT_PROTOCOL_VERSION == "2025-11-25"


def test_is_supported_protocol_version():
    """Verify version membership checks."""
    assert is_supported_protocol_version("2026-07-28") is True
    assert is_supported_protocol_version("2025-11-25") is True
    assert is_supported_protocol_version("2024-11-05") is True
    assert is_supported_protocol_version("invalid-version") is False
    assert is_supported_protocol_version("2099-01-01") is False


def test_negotiate_protocol_version():
    """Verify version negotiation logic."""
    assert negotiate_protocol_version(None) == DEFAULT_PROTOCOL_VERSION
    assert negotiate_protocol_version("") == DEFAULT_PROTOCOL_VERSION
    assert negotiate_protocol_version("2026-07-28") == "2026-07-28"
    assert negotiate_protocol_version("2025-11-25") == "2025-11-25"
    assert negotiate_protocol_version("2025-03-26") == "2025-03-26"
    assert negotiate_protocol_version("2024-11-05") == "2024-11-05"

    with pytest.raises(UnsupportedProtocolVersionError) as exc_info:
        negotiate_protocol_version("2099-01-01")
    assert "Unsupported protocol version: 2099-01-01" in str(exc_info.value)


def test_sdk_runtime_patch_applied():
    """Verify that install_protocol_compatibility patched SDK globals."""
    install_protocol_compatibility()

    import mcp.shared.version
    import mcp.types
    import mcp.server.streamable_http
    import mcp.server.session

    assert "2026-07-28" in mcp.shared.version.SUPPORTED_PROTOCOL_VERSIONS
    assert mcp.types.LATEST_PROTOCOL_VERSION == "2026-07-28"
    assert "2026-07-28" in mcp.server.streamable_http.SUPPORTED_PROTOCOL_VERSIONS
    assert "2026-07-28" in mcp.server.session.SUPPORTED_PROTOCOL_VERSIONS


def test_server_module_exports():
    """Verify telegram_mcp.server public module exports."""
    assert server.get_server() is mcp
    assert server.LATEST_PROTOCOL_VERSION == "2026-07-28"
    assert callable(server.get_http_app)
    assert callable(server.serve)
    assert callable(server.main)


def test_http_transport_negotiation_2026_07_28():
    """Verify clients requesting 2026-07-28 negotiate 2026-07-28 successfully over HTTP."""
    app = create_streamable_http_app(mcp)
    with TestClient(app, base_url="http://localhost:8765", headers=_AUTH_HEADERS) as client:
        response = client.post(
            "/mcp",
            headers={
                "MCP-Protocol-Version": "2026-07-28",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2026-07-28",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0"},
                },
            },
        )
        assert response.status_code == 200
        assert response.headers.get("mcp-protocol-version") == "2026-07-28"
        assert '"protocolVersion":"2026-07-28"' in response.text


def test_http_transport_backwards_compatibility_2025_11_25():
    """Verify older clients requesting 2025-11-25 negotiate 2025-11-25 over HTTP."""
    app = create_streamable_http_app(mcp)
    with TestClient(app, base_url="http://localhost:8765", headers=_AUTH_HEADERS) as client:
        response = client.post(
            "/mcp",
            headers={
                "MCP-Protocol-Version": "2025-11-25",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0"},
                },
            },
        )
        assert response.status_code == 200
        assert response.headers.get("mcp-protocol-version") == "2025-11-25"
        assert '"protocolVersion":"2025-11-25"' in response.text


def test_http_transport_backwards_compatibility_2024_11_05():
    """Verify legacy clients requesting 2024-11-05 negotiate 2024-11-05 over HTTP."""
    app = create_streamable_http_app(mcp)
    with TestClient(app, base_url="http://localhost:8765", headers=_AUTH_HEADERS) as client:
        response = client.post(
            "/mcp",
            headers={
                "MCP-Protocol-Version": "2024-11-05",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "legacy-client", "version": "0.1"},
                },
            },
        )
        assert response.status_code == 200
        assert response.headers.get("mcp-protocol-version") == "2024-11-05"
        assert '"protocolVersion":"2024-11-05"' in response.text


def test_http_transport_unsupported_protocol_rejected():
    """Verify unsupported protocol versions receive HTTP 400 with supported versions list."""
    app = create_streamable_http_app(mcp)
    with TestClient(app, base_url="http://localhost:8765", headers=_AUTH_HEADERS) as client:
        response = client.post(
            "/mcp",
            headers={
                "MCP-Protocol-Version": "2099-01-01",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            json={
                "jsonrpc": "2.0",
                "id": 4,
                "method": "initialize",
                "params": {"protocolVersion": "2099-01-01"},
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "Unsupported protocol version: 2099-01-01" in data["error"]["message"]
        assert "2026-07-28" in data["error"]["message"]


def test_http_transport_options_preflight():
    """Verify OPTIONS preflight returns 204 with protocol and CORS headers."""
    app = create_streamable_http_app(mcp)
    with TestClient(app, base_url="http://localhost:8765", headers=_AUTH_HEADERS) as client:
        response = client.options("/mcp", headers={"MCP-Protocol-Version": "2026-07-28"})
        assert response.status_code == 204
        assert response.headers.get("mcp-protocol-version") == "2026-07-28"
        assert "OPTIONS" in response.headers.get("allow", "")
        # Never a wildcard: no Origin, no CORS grant (#378).
        assert "access-control-allow-origin" not in response.headers


_TOOLS_LIST = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
_POST_HEADERS = {
    "MCP-Protocol-Version": "2026-07-28",
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


@pytest.mark.parametrize(
    "method,path",
    [
        ("OPTIONS", "/mcp"),
        ("GET", "/mcp"),
        ("POST", "/mcp"),
        ("PUT", "/upload/x?exp=1&sig=x"),
        ("GET", "/media/x?exp=1&sig=x"),
    ],
)
def test_http_transport_rejects_cross_origin_requests(method, path):
    """CWE-942 (#378): a web page on another origin must not reach the server at all."""
    app = create_streamable_http_app(mcp)
    with TestClient(app, base_url="http://localhost:8765", headers=_AUTH_HEADERS) as client:
        headers = dict(_POST_HEADERS, Origin="https://evil.example")
        kwargs = {"json": _TOOLS_LIST} if method == "POST" else {}
        response = client.request(method, path, headers=headers, **kwargs)
        assert response.status_code == 403
        assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize(
    "origin", ["null", "http://localhost.evil.example", "http://localhost:80.evil"]
)
def test_http_transport_rejects_lookalike_origins(origin):
    app = create_streamable_http_app(mcp)
    with TestClient(app, base_url="http://localhost:8765", headers=_AUTH_HEADERS) as client:
        response = client.options("/mcp", headers={"Origin": origin})
        assert response.status_code == 403


def test_http_transport_preflight_echoes_only_allowed_origin():
    app = create_streamable_http_app(mcp)
    with TestClient(app, base_url="http://localhost:8765", headers=_AUTH_HEADERS) as client:
        response = client.options("/mcp", headers={"Origin": "http://localhost:6274"})
        assert response.status_code == 204
        assert response.headers.get("access-control-allow-origin") == "http://localhost:6274"
        assert "origin" in response.headers.get("vary", "").lower()


def test_http_transport_allows_requests_without_origin():
    """Non-browser MCP clients send no Origin header and keep working."""
    app = create_streamable_http_app(mcp)
    with TestClient(app, base_url="http://localhost:8765", headers=_AUTH_HEADERS) as client:
        response = client.post("/mcp", headers=_POST_HEADERS, json=_TOOLS_LIST)
        assert response.status_code == 200


def test_http_transport_honours_configured_allowed_origins(monkeypatch):
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", "mcp.example.com")
    monkeypatch.setenv("MCP_ALLOWED_ORIGINS", "https://app.example.com")
    app = create_streamable_http_app(mcp)
    with TestClient(app, base_url="http://mcp.example.com") as client:
        ok = client.options("/mcp", headers={"Origin": "https://app.example.com"})
        assert ok.status_code == 204
        assert ok.headers.get("access-control-allow-origin") == "https://app.example.com"
        # Local origins are no longer implied once the operator names their own.
        denied = client.options("/mcp", headers={"Origin": "http://localhost:3000"})
        assert denied.status_code == 403


def test_http_transport_discovery_get_probe():
    """Verify plain GET discovery probe returns JSON server metadata."""
    app = create_streamable_http_app(mcp)
    with TestClient(app, base_url="http://localhost:8765", headers=_AUTH_HEADERS) as client:
        response = client.get(
            "/mcp",
            headers={
                "MCP-Protocol-Version": "2026-07-28",
                "Accept": "application/json",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "telegram"
        assert "2026-07-28" in data["protocolVersions"]
        assert data["latestProtocolVersion"] == "2026-07-28"
        assert response.headers.get("mcp-protocol-version") == "2026-07-28"


def test_http_transport_modern_stateless_tools_list():
    """Verify modern 2026-07-28 stateless call to tools/list executes cleanly."""
    app = create_streamable_http_app(mcp)
    with TestClient(app, base_url="http://localhost:8765", headers=_AUTH_HEADERS) as client:
        response = client.post(
            "/mcp",
            headers={
                "MCP-Protocol-Version": "2026-07-28",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            json={
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/list",
                "params": {},
            },
        )
        assert response.status_code == 200
        assert response.headers.get("mcp-protocol-version") == "2026-07-28"
        assert '"tools":' in response.text


@pytest.mark.asyncio
async def test_session_handshake_protocol_negotiation():
    """Verify ServerSession handshake negotiates 2026-07-28, 2025-11-25, 2024-11-05 cleanly."""
    from unittest.mock import AsyncMock, MagicMock
    from mcp.server.session import ServerSession
    from mcp.server.models import InitializationOptions
    from mcp.shared.session import RequestResponder
    from mcp.types import (
        ClientCapabilities,
        ClientRequest,
        Implementation,
        InitializeRequest,
        InitializeRequestParams,
        ServerCapabilities,
    )

    install_protocol_compatibility()

    init_options = InitializationOptions(
        server_name="telegram",
        server_version="2.0.1",
        capabilities=ServerCapabilities(),
    )

    for version in ("2026-07-28", "2025-11-25", "2024-11-05"):
        mock_read = MagicMock()
        mock_write = AsyncMock()

        session = ServerSession(
            read_stream=mock_read,
            write_stream=mock_write,
            init_options=init_options,
        )

        req = InitializeRequest(
            method="initialize",
            params=InitializeRequestParams(
                protocolVersion=version,
                capabilities=ClientCapabilities(),
                clientInfo=Implementation(name="test", version="1.0"),
            ),
        )

        responder = RequestResponder(
            request_id="1",
            request_meta=None,
            request=ClientRequest(req),
            session=session,
            on_complete=lambda r: None,
        )

        await session._received_request(responder)
        assert mock_write.send.called
        sent_session_msg = mock_write.send.call_args[0][0]
        result_dict = sent_session_msg.message.root.result
        assert result_dict["protocolVersion"] == version

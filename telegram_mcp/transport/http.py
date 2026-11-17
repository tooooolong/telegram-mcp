"""Authenticated MCP network transports and streamable HTTP protocol negotiation."""

from __future__ import annotations

import hmac
import logging
import os
import re
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse, Response
from starlette.routing import Match
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from telegram_mcp.transport.protocol import (
    DEFAULT_PROTOCOL_VERSION,
    LATEST_PROTOCOL_VERSION,
    MCP_PROTOCOL_VERSION_HEADER,
    SUPPORTED_PROTOCOL_VERSIONS,
    install_protocol_compatibility,
)

logger = logging.getLogger(__name__)

_DEFAULT_ALLOWED_HOSTS = ("127.0.0.1:*", "localhost:*", "[::1]:*")
_DEFAULT_ALLOWED_ORIGINS = ("http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*")


def require_http_auth_token() -> bytes:
    token = os.getenv("MCP_AUTH_TOKEN", "")
    if not re.fullmatch(r"[A-Za-z0-9._~+/-]+=*", token):
        raise RuntimeError("Set MCP_AUTH_TOKEN to a valid bearer token to enable HTTP or SSE.")
    return token.encode("ascii")


def configure_transport_security(mcp_instance: FastMCP) -> None:
    allowed_hosts = [
        value.strip() for value in os.getenv("MCP_ALLOWED_HOSTS", "").split(",") if value.strip()
    ]
    allowed_origins = [
        value.strip() for value in os.getenv("MCP_ALLOWED_ORIGINS", "").split(",") if value.strip()
    ]
    if not allowed_hosts:
        allowed_hosts = list(_DEFAULT_ALLOWED_HOSTS)
        if not allowed_origins:
            allowed_origins = list(_DEFAULT_ALLOWED_ORIGINS)

    mcp_instance.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )


def _has_bearer_token(headers: list[tuple[bytes, bytes]], expected_token: bytes) -> bool:
    values = [value for key, value in headers if key == b"authorization"]
    if len(values) != 1:
        return False
    scheme, _, credentials = values[0].partition(b" ")
    return scheme.lower() == b"bearer" and hmac.compare_digest(
        credentials.lstrip(b" "), expected_token
    )


def _origin_matches(origin: str, allowed_origins: Any) -> bool:
    """Match an Origin header like FastMCP does: exact, or ``scheme://host:*``."""
    for allowed in allowed_origins or ():
        if origin == allowed:
            return True
        if allowed.endswith(":*") and origin.startswith(allowed[:-2] + ":"):
            port = origin[len(allowed) - 1 :]
            if port.isdigit():
                return True
    return False


class ProtocolNegotiationMiddleware:
    """Protect network routes and negotiate the streamable HTTP protocol."""

    def __init__(self, app: Starlette, mcp_instance: FastMCP, token: bytes, *, streamable: bool):
        self.app = app
        self.allowed_origins = tuple(mcp_instance.settings.transport_security.allowed_origins)
        self.streamable_path = mcp_instance.settings.streamable_http_path if streamable else None
        self.signed_endpoints = frozenset(getattr(mcp_instance, "_signed_endpoints", ()))
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw_headers = scope.get("headers", [])
        headers = dict(raw_headers)
        raw_origin = headers.get(b"origin")
        origin = raw_origin.decode("latin-1").strip() if raw_origin else None
        if origin and not _origin_matches(origin, self.allowed_origins):
            logger.warning("Rejected request from disallowed Origin: %s", origin)
            response = Response("Invalid Origin header", status_code=403)
            await response(scope, receive, send)
            return
        cors_origin_headers = (
            {"Access-Control-Allow-Origin": origin, "Vary": "Origin"} if origin else {}
        )

        route = next(
            (route for route in self.app.routes if route.matches(scope)[0] is Match.FULL), None
        )
        if getattr(route, "endpoint", None) in self.signed_endpoints:
            await self.app(scope, receive, send)
            return

        # Preflight is answered here without executing any app route.
        if scope["method"] == "OPTIONS":
            response = Response(
                status_code=204,
                headers={
                    "Allow": "GET, POST, DELETE, OPTIONS, HEAD",
                    **cors_origin_headers,
                    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS, HEAD",
                    "Access-Control-Allow-Headers": (
                        "Content-Type, Accept, MCP-Protocol-Version, MCP-Session-Id, Authorization"
                    ),
                    "Access-Control-Expose-Headers": "MCP-Protocol-Version, MCP-Session-Id",
                },
            )
            if self.streamable_path is not None:
                response.headers[MCP_PROTOCOL_VERSION_HEADER] = (
                    headers.get(MCP_PROTOCOL_VERSION_HEADER.encode("latin-1"), b"")
                    .decode("latin-1")
                    .strip()
                    or LATEST_PROTOCOL_VERSION
                )
            await response(scope, receive, send)
            return

        if not _has_bearer_token(raw_headers, self.token):
            response = Response(
                "Bearer authentication required",
                status_code=401,
                headers={"WWW-Authenticate": "Bearer", **cors_origin_headers},
            )
            await response(scope, receive, send)
            return

        async def send_with_origin(message: Message) -> None:
            if origin and message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                response_headers["Access-Control-Allow-Origin"] = origin
                vary = response_headers.get("Vary", "")
                if "origin" not in {value.strip().lower() for value in vary.split(",")}:
                    response_headers.add_vary_header("Origin")
            await send(message)

        if self.streamable_path is None:
            await self.app(scope, receive, send_with_origin)
            return

        is_mcp_endpoint = getattr(route, "path", None) == self.streamable_path
        await self._negotiate(
            scope, receive, send_with_origin, headers, cors_origin_headers, is_mcp_endpoint
        )

    async def _negotiate(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        headers: dict,
        cors_origin_headers: dict,
        is_mcp_endpoint: bool,
    ) -> None:
        raw_version_bytes = headers.get(MCP_PROTOCOL_VERSION_HEADER.encode("latin-1"))
        requested_version = (
            raw_version_bytes.decode("latin-1").strip() if raw_version_bytes else None
        )

        if requested_version:
            if requested_version not in SUPPORTED_PROTOCOL_VERSIONS:
                supported_str = ", ".join(SUPPORTED_PROTOCOL_VERSIONS)
                error_body = {
                    "jsonrpc": "2.0",
                    "id": "server-error",
                    "error": {
                        "code": -32600,
                        "message": (
                            f"Bad Request: Unsupported protocol version: {requested_version}. "
                            f"Supported versions: {supported_str}"
                        ),
                    },
                }
                response = JSONResponse(
                    status_code=400,
                    content=error_body,
                    headers={
                        MCP_PROTOCOL_VERSION_HEADER: DEFAULT_PROTOCOL_VERSION,
                    },
                )
                await response(scope, receive, send)
                return
            negotiated_version = requested_version
        else:
            negotiated_version = DEFAULT_PROTOCOL_VERSION

        # Plain GET probes receive discovery metadata on the MCP route.
        accept_header = headers.get(b"accept", b"").decode("latin-1").lower()
        if (
            is_mcp_endpoint
            and scope["method"] == "GET"
            and "text/event-stream" not in accept_header
        ):
            discovery_data = {
                "name": "telegram",
                "version": "2.0.1",
                "protocolVersions": list(SUPPORTED_PROTOCOL_VERSIONS),
                "latestProtocolVersion": LATEST_PROTOCOL_VERSION,
                "negotiatedProtocolVersion": negotiated_version,
            }
            response = JSONResponse(
                status_code=200,
                content=discovery_data,
                headers={
                    MCP_PROTOCOL_VERSION_HEADER: negotiated_version,
                    **cors_origin_headers,
                    "Access-Control-Expose-Headers": "MCP-Protocol-Version, MCP-Session-Id",
                },
            )
            await response(scope, receive, send)
            return

        async def send_with_protocol_header(message: Message) -> None:
            if message["type"] == "http.response.start":
                resp_headers = list(message.get("headers", []))
                header_names = {h[0].lower() for h in resp_headers}

                if MCP_PROTOCOL_VERSION_HEADER.encode("latin-1") not in header_names:
                    resp_headers.append(
                        (
                            MCP_PROTOCOL_VERSION_HEADER.encode("latin-1"),
                            negotiated_version.encode("latin-1"),
                        )
                    )

                if b"access-control-expose-headers" not in header_names:
                    expose_val = b"*, mcp-protocol-version, mcp-session-id"
                    resp_headers.append((b"access-control-expose-headers", expose_val))

                message["headers"] = resp_headers
            await send(message)

        await self.app(scope, receive, send_with_protocol_header)


class TelegramFastMCP(FastMCP):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._signed_endpoints: set = set()

    def signed_route(self, path: str, *, methods: list[str]):
        """Register a route whose handler verifies its own signed URL."""

        def register(endpoint):
            self._signed_endpoints.add(endpoint)
            return self.custom_route(path, methods=methods)(endpoint)

        return register

    def streamable_http_app(self) -> ASGIApp:
        return _build_streamable_http_app(self, super().streamable_http_app)

    def sse_app(self, mount_path: Optional[str] = None) -> ASGIApp:
        token = require_http_auth_token()
        configure_transport_security(self)
        return ProtocolNegotiationMiddleware(
            super().sse_app(mount_path), self, token, streamable=False
        )

    async def run_streamable_http_async(self) -> None:
        await run_streamable_http_async(self)

    async def run_sse_async(self, mount_path: Optional[str] = None) -> None:
        await _run_network_app(self, self.sse_app(mount_path))


def _build_streamable_http_app(mcp_instance: FastMCP, app_factory) -> ASGIApp:
    token = require_http_auth_token()
    configure_transport_security(mcp_instance)
    install_protocol_compatibility()

    # A session manager that has shut down cannot serve a new app lifespan.
    session_mgr = mcp_instance._session_manager
    if session_mgr is not None and getattr(session_mgr, "_has_started", False):
        mcp_instance._session_manager = None

    return ProtocolNegotiationMiddleware(app_factory(), mcp_instance, token, streamable=True)


def create_streamable_http_app(mcp_instance: FastMCP) -> ASGIApp:
    """Build the authenticated app with MCP protocol negotiation."""
    if isinstance(mcp_instance, TelegramFastMCP):
        return mcp_instance.streamable_http_app()
    return _build_streamable_http_app(mcp_instance, mcp_instance.streamable_http_app)


async def _run_network_app(mcp_instance: FastMCP, app: ASGIApp) -> None:
    import uvicorn

    config = uvicorn.Config(
        app,
        host=mcp_instance.settings.host,
        port=mcp_instance.settings.port,
        log_level=mcp_instance.settings.log_level.lower(),
        # Access logs include query strings, which contain signed media credentials.
        access_log=False,
    )
    await uvicorn.Server(config).serve()


async def run_streamable_http_async(mcp_instance: FastMCP) -> None:
    await _run_network_app(mcp_instance, create_streamable_http_app(mcp_instance))

"""Application entrypoints for the Telegram MCP server."""

from telegram_mcp.install_guard import UnsafeInstallationError, assert_safe_distribution

try:
    assert_safe_distribution()
except UnsafeInstallationError as exc:
    raise SystemExit(str(exc)) from None

import sys
from contextlib import asynccontextmanager
from telethon.errors import AuthKeyDuplicatedError, BotMethodInvalidError

# Ensure sys.stderr is reconfigured for UTF-8 on Windows where possible
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

from telegram_mcp import runtime as _runtime
from telegram_mcp import transcription as _transcription
from telegram_mcp.runtime import *
from telegram_mcp.singleton import (
    DEFAULT_GRACE_SECONDS,
    SessionLock,
    SessionLockError,
    session_identity,
)
from telegram_mcp.transport.http import configure_transport_security, require_http_auth_token

import telegram_mcp.tools  # noqa: F401 - registers MCP tools via decorators
from telegram_mcp.tools.events import stop_incoming_feed

_session_locks: dict[str, SessionLock] = {}
_service_configured = False


def _lock_grace_seconds() -> float:
    raw = os.getenv("TELEGRAM_LOCK_GRACE_SECONDS")
    if not raw:
        return DEFAULT_GRACE_SECONDS
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_GRACE_SECONDS


_MIN_CATCH_UP_INTERVAL = 30.0


def _catch_up_interval() -> float:
    """Seconds between forced update catch-ups (TELEGRAM_CATCH_UP_INTERVAL); 0 = off.

    Positive values below 30 s are raised to 30 s: each catch-up is one getDifference.
    """
    raw = os.getenv("TELEGRAM_CATCH_UP_INTERVAL")
    if not raw:
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    return max(value, _MIN_CATCH_UP_INTERVAL) if value > 0 else 0.0


async def _periodic_catch_up(interval: float) -> None:
    """Ask Telegram for missed updates every `interval` seconds.

    Telethon only fetches a difference by itself after 15 minutes without updates, and
    its automatic reconnect does not catch up. On a flaky network this bounds how late
    an update can reach the event handlers. When nothing was missed, the request returns
    an empty difference.
    """
    while True:
        await asyncio.sleep(interval)
        for label, cl in list(clients.items()):
            if not cl.is_connected():
                continue
            try:
                await cl.catch_up()
            except Exception as exc:
                print(f"Update catch-up failed for '{label}': {exc}", file=sys.stderr)


_SESSION_LOCK_MODES = ("exclusive", "shared")


def _session_lock_shared() -> bool:
    """``TELEGRAM_SESSION_LOCK``: exclusive (default) | shared.

    ``exclusive`` refuses to start while another instance on this host holds
    the same session. ``shared`` takes the lock in shared mode instead, so any
    number of instances on this host can use one session -- safe only when
    they all reach Telegram from the same IP, since Telegram invalidates a
    session it sees from two IPs at once. Shared and exclusive instances never
    overlap, so the default keeps its guarantee.
    """
    raw = (os.getenv("TELEGRAM_SESSION_LOCK") or "exclusive").strip().lower()
    if raw not in _SESSION_LOCK_MODES:
        accepted = ", ".join(_SESSION_LOCK_MODES)
        raise SystemExit(f"Invalid TELEGRAM_SESSION_LOCK '{raw}'. Expected one of: {accepted}.")
    return raw == "shared"


def _normalize_username(value) -> str:
    return (value or "").strip().lstrip("@").casefold()


def _expected_username(label: str) -> str:
    """TELEGRAM_EXPECTED_USERNAME_<LABEL>, else TELEGRAM_EXPECTED_USERNAME; "" if unset."""
    raw = os.getenv(f"TELEGRAM_EXPECTED_USERNAME_{label.upper()}") or os.getenv(
        "TELEGRAM_EXPECTED_USERNAME"
    )
    return _normalize_username(raw)


async def _verify_expected_username(label: str, client) -> None:
    """Refuse to serve a session logged in to a different account."""
    expected = _expected_username(label)
    if not expected:
        return

    me = await client.get_me()
    usernames = {_normalize_username(getattr(me, "username", None))}
    for entry in getattr(me, "usernames", None) or []:
        if getattr(entry, "active", False):
            usernames.add(_normalize_username(getattr(entry, "username", None)))
    usernames.discard("")

    if expected in usernames:
        return

    try:
        await client.disconnect()
    except Exception:
        pass
    raise RuntimeError(
        f"Telegram client '{label}' is logged in to a different account than "
        f"TELEGRAM_EXPECTED_USERNAME_{label.upper()} / TELEGRAM_EXPECTED_USERNAME expects."
    )


class _NotAuthorized(RuntimeError):
    """The session connects but was never logged in."""


async def _connect_authorized_client(label, client) -> None:
    # First, prevent our own duplicate-spawn case outright: a per-session lock
    # means a second instance of this server never even attempts to connect
    # while another instance already holds the same session (see
    # telegram_mcp/singleton.py for why and how). The lock only sees processes
    # on this host; TELEGRAM_SESSION_LOCK=shared lets those share one session
    # where they all reach Telegram from a single IP.
    lock = SessionLock(label, session_identity(client))
    acquire = asyncio.create_task(
        asyncio.to_thread(
            lock.acquire, grace_seconds=_lock_grace_seconds(), shared=_session_lock_shared()
        )
    )
    try:
        await asyncio.shield(acquire)
    except asyncio.CancelledError:
        # The thread can acquire the lock after startup has been cancelled.
        await asyncio.gather(acquire, return_exceptions=True)
        lock.release()
        raise
    _session_locks[label] = lock

    # Once we hold the lock, still tolerate a transient AuthKeyDuplicatedError
    # from Telegram itself (e.g. the same session briefly seen from two IPs
    # during a VPN reconnect) with a bounded retry, since that's not caused by
    # a second instance of this server and a blip shouldn't take the server
    # down. Give each concurrent client its own session (TELEGRAM_SESSION_STRINGS
    # pool or TELEGRAM_SESSION_STRING_<LABEL>) to avoid the collision entirely.
    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        try:
            await client.connect()
            break
        except AuthKeyDuplicatedError:
            if attempt >= max_attempts:
                raise
            delay = min(2**attempt, 15)
            print(
                f"AuthKeyDuplicatedError connecting '{label}' (attempt "
                f"{attempt}/{max_attempts}): session in use from another IP. "
                f"Retrying in {delay}s. If this persists, give each concurrent "
                "client its own session via TELEGRAM_SESSION_STRINGS or "
                "TELEGRAM_SESSION_STRING_<LABEL>.",
                file=sys.stderr,
            )
            try:
                await client.disconnect()
            except Exception:
                pass
            await asyncio.sleep(delay)

    if await client.is_user_authorized():
        await _verify_expected_username(label, client)
        return

    raise _NotAuthorized(
        f"Telegram client '{label}' is not authorized. Interactive phone login "
        "is disabled for the MCP server because it runs over stdio. Generate a "
        "session string with `uv run session_string_generator.py`, then set "
        "TELEGRAM_SESSION_STRING or TELEGRAM_SESSION_STRING_<LABEL> in .env. "
        "For existing file sessions, run the login outside the MCP server first."
    )


async def _connect_or_skip(label, client) -> None:
    """Connect one account; drop an optional one that was never logged in."""
    try:
        await _connect_authorized_client(label, client)
    except _NotAuthorized:
        if label not in _runtime.OPTIONAL_ACCOUNTS:
            raise
        lock = _session_locks.pop(label, None)
        try:
            await client.disconnect()
        except Exception:
            pass
        finally:
            if lock is not None:
                lock.release()
        clients.pop(label, None)
        print(
            f"Skipping account '{label}': TELEGRAM_SESSION_NAME names a session that is not "
            "logged in, and labelled accounts are configured. Unset TELEGRAM_SESSION_NAME "
            "or log that session in to use it as the default account.",
            file=sys.stderr,
        )


async def _serve(transport: str) -> None:
    """Run the MCP server on the selected transport.

    HTTP transports let one long-lived process hold a single shared Telegram
    connection while multiple local MCP clients connect over HTTP, instead of
    each client spawning its own Telethon session (which Telegram
    throttles/flags). "http" is streamable HTTP — the current MCP transport
    that Claude Code (`--transport http`) and Codex (`--url`) speak natively;
    "sse" is kept for clients that only support the legacy SSE transport.
    """
    valid_transports = ("stdio", "http", "sse")
    if transport not in valid_transports:
        accepted = ", ".join(valid_transports)
        raise ValueError(f"Invalid MCP_TRANSPORT '{transport}'. Expected one of: {accepted}.")
    if transport in ("http", "sse"):
        require_http_auth_token()
        mcp.settings.host = os.getenv("MCP_HOST", "127.0.0.1")
        mcp.settings.port = int(os.getenv("MCP_PORT", "8765"))
        configure_transport_security(mcp)
        if transport == "http":
            await mcp.run_streamable_http_async()
        else:
            await mcp.run_sse_async()
    else:
        # Use the asynchronous entrypoint instead of mcp.run()
        await mcp.run_stdio_async()


async def _warm_caches() -> None:
    async def warm_client(label: str, client) -> None:
        try:
            await client.get_dialogs()
        except BotMethodInvalidError:
            print(
                f"Skipping entity cache pre-warm for bot client '{label}' "
                "(dialogs restricted for bots).",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"Entity cache warm failed for '{label}': {exc}", file=sys.stderr)

    await asyncio.gather(*(warm_client(label, cl) for label, cl in list(clients.items())))
    print("Entity caches warmed.", file=sys.stderr)


@asynccontextmanager
async def _service_lifespan():
    _prepare_service_configuration()
    accounts = list(clients.items())
    labels = ", ".join(label for label, _ in accounts)
    print(f"Starting {len(accounts)} Telegram client(s) ({labels})...", file=sys.stderr)
    tasks = [asyncio.create_task(_connect_or_skip(label, cl)) for label, cl in accounts]
    try:
        await asyncio.gather(*tasks)
        labels = ", ".join(clients.keys())
        print(f"Telegram client(s) started ({labels}).", file=sys.stderr)
        # Cache warming may need more time than the MCP client's startup timeout.
        tasks.append(asyncio.create_task(_warm_caches()))
        catch_up_interval = _catch_up_interval()
        if catch_up_interval:
            tasks.append(asyncio.create_task(_periodic_catch_up(catch_up_interval)))
        yield
    finally:
        for task in tasks:
            task.cancel()
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.gather(*(cl.disconnect() for _, cl in accounts), return_exceptions=True)
            await stop_incoming_feed()
        finally:
            for lock in _session_locks.values():
                lock.release()
            _session_locks.clear()


async def _main() -> None:
    try:
        await _serve(os.getenv("MCP_TRANSPORT", "stdio").lower())
    except Exception as e:
        print(f"Error starting client: {e}", file=sys.stderr)
        if isinstance(e, sqlite3.OperationalError) and "database is locked" in str(e):
            print(
                "Database lock detected. Please ensure no other instances are running.",
                file=sys.stderr,
            )
        elif isinstance(e, SessionLockError):
            print(
                "Another instance of this MCP server already holds this Telegram "
                "session (e.g. the client restarted the connector without the old "
                "process exiting yet). This instance is exiting instead of "
                "connecting a second time, which would risk Telegram invalidating "
                "the session for both. Retry once the other instance is gone, or "
                "set TELEGRAM_SESSION_LOCK=shared if several instances on this "
                "host are meant to share one session (safe when they all reach "
                "Telegram from the same IP).",
                file=sys.stderr,
            )
        sys.exit(1)


def _prepare_service_configuration(argv=None) -> None:
    global _service_configured
    if _service_configured:
        return
    _configure_allowed_roots_from_cli(argv)
    _transcription.validate_transcription_config()
    _session_lock_shared()

    # Before _apply_exposed_tools_mode(): that prunes non-exposed tools from
    # the tool manager, and the extension overrides validate tool names
    # against that same manager. Narrowing send_file's extensions while
    # send_file is not exposed is a valid configuration, so the name check
    # has to see the full tool set.
    _runtime._apply_file_extension_overrides()
    hidden, excluded = _runtime._apply_exposed_tools_mode()
    if hidden:
        # These names are the menu for the "+" allowlist; without this line the
        # only way to find them is reading the tool annotations in the source.
        print(
            f"TELEGRAM_EXPOSED_TOOLS hides {len(hidden)} tool(s); list any of them "
            f"after '+' to expose it: {', '.join(sorted(hidden))}",
            file=sys.stderr,
        )
    if excluded:
        print(
            f"TELEGRAM_EXPOSED_TOOLS excludes {len(excluded)} tool(s) listed "
            f"after '-': {', '.join(sorted(excluded))}",
            file=sys.stderr,
        )
    _service_configured = True


def main() -> None:
    _prepare_service_configuration(sys.argv[1:])
    # Apply CLI transport/host/port overrides to environment (runtime sets globals).
    transport = _runtime._CLI_TRANSPORT or "stdio"
    host = _runtime._CLI_HOST
    port = _runtime._CLI_PORT

    if transport != "stdio":
        os.environ["MCP_TRANSPORT"] = transport
    if host:
        os.environ["MCP_HOST"] = host
    if port is not None:
        os.environ["MCP_PORT"] = str(port)
    asyncio.run(_main())


if __name__ == "__main__":
    main()

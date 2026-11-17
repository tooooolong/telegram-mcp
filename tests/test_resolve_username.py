"""Unit tests for resolve_username tool and normalize_username helper."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from telethon import functions, types
from telethon.errors import UsernameNotOccupiedError

from telegram_mcp.tools import chats
from telegram_mcp.tools.chats import normalize_username, resolve_username


def test_normalize_username_variants():
    assert normalize_username("durov") == "durov"
    assert normalize_username("@durov") == "durov"
    assert normalize_username("  @durov  ") == "durov"
    assert normalize_username("https://t.me/durov") == "durov"
    assert normalize_username("http://t.me/durov/") == "durov"
    assert normalize_username("t.me/durov") == "durov"
    assert normalize_username("https://t.me/durov?start=ref") == "durov"
    assert normalize_username("") == ""
    assert normalize_username("   ") == ""


@pytest.mark.asyncio
async def test_resolve_username_user_success(monkeypatch):
    mock_user = types.User(
        id=1234567,
        first_name="Pavel",
        last_name="Durov",
        phone="+1234567890",
        bot=False,
        verified=True,
    )
    mock_response = types.contacts.ResolvedPeer(
        peer=types.PeerUser(user_id=1234567),
        users=[types.User(id=1), mock_user],
        chats=[],
    )

    cl = AsyncMock()
    cl.return_value = mock_response

    monkeypatch.setattr(chats, "get_client", lambda account=None: cl)
    monkeypatch.setattr(chats, "ensure_connected", AsyncMock())

    raw = await resolve_username("@durov")
    data = json.loads(raw)

    assert data["query"] == "@durov"
    assert data["username"] == "durov"
    assert data["peer_id"] == 1234567
    assert data["peer_type"] == "user"
    assert data["first_name"] == "Pavel"
    assert data["last_name"] == "Durov"
    assert data["verified"] is True
    assert data["is_bot"] is False

    # Verify request sent normalized username without '@'
    call_req = cl.call_args[0][0]
    assert isinstance(call_req, functions.contacts.ResolveUsernameRequest)
    assert call_req.username == "durov"


@pytest.mark.asyncio
async def test_resolve_username_channel_success(monkeypatch):
    mock_channel = types.Channel(
        id=987654321,
        title="Telegram News",
        broadcast=True,
        megagroup=False,
        participants_count=50000,
        photo=types.ChatPhotoEmpty(),
        date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    mock_response = types.contacts.ResolvedPeer(
        peer=types.PeerChannel(channel_id=987654321),
        users=[types.User(id=987654321)],
        chats=[mock_channel],
    )

    cl = AsyncMock()
    cl.return_value = mock_response

    monkeypatch.setattr(chats, "get_client", lambda account=None: cl)
    monkeypatch.setattr(chats, "ensure_connected", AsyncMock())

    raw = await resolve_username("https://t.me/telegram")
    data = json.loads(raw)

    assert data["username"] == "telegram"
    assert data["peer_id"] == -1000987654321
    assert data["peer_type"] == "channel"
    assert data["title"] == "Telegram News"
    assert data["broadcast"] is True
    assert data["participants_count"] == 50000


@pytest.mark.asyncio
async def test_resolve_username_invalid_empty():
    res = await resolve_username("   @@@   ")
    assert "Invalid username" in res


@pytest.mark.asyncio
async def test_resolve_username_error_handling(monkeypatch):
    cl = AsyncMock()
    cl.side_effect = ValueError("Username not found")

    monkeypatch.setattr(chats, "get_client", lambda account=None: cl)
    monkeypatch.setattr(chats, "ensure_connected", AsyncMock())

    res = await resolve_username("unknown_user_404")
    assert "error" in res.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "value"),
    [
        ("TELEGRAM_ALLOWED_CHAT_IDS", "14"),
        ("TELEGRAM_BLOCKED_CHAT_IDS", "21"),
        ("TELEGRAM_BLOCKED_CHAT_IDS", "@example_user"),
    ],
)
async def test_resolve_username_returns_same_response_for_restricted_and_missing(
    monkeypatch, policy, value
):
    monkeypatch.setenv(policy, value)
    cl = AsyncMock(
        side_effect=[
            types.contacts.ResolvedPeer(
                peer=types.PeerUser(user_id=21),
                users=[types.User(id=21, username="example_user")],
                chats=[],
            ),
            UsernameNotOccupiedError(request=None),
        ]
    )
    monkeypatch.setattr(chats, "get_client", lambda account=None: cl)
    monkeypatch.setattr(chats, "ensure_connected", AsyncMock())

    assert (
        await resolve_username("@example_user")
        == await resolve_username("@unused_name")
        == "No Telegram chat or user found for this username."
    )
    assert cl.await_count == 2

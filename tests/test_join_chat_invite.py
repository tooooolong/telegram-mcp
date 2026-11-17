"""Chat policy is checked on Telegram's verified invite target before joining."""

from types import SimpleNamespace

import pytest
from telethon.tl import functions, types

from telegram_mcp import runtime
from telegram_mcp.tools import groups


@pytest.fixture(params=["join_chat_by_link", "import_chat_invite"])
def invite_tool(request, monkeypatch):
    monkeypatch.delenv("TELEGRAM_ALLOWED_CHAT_IDS", raising=False)
    monkeypatch.delenv("TELEGRAM_BLOCKED_CHAT_IDS", raising=False)
    monkeypatch.setattr(runtime, "ALLOWED_CHAT_IDS", None)

    async def connected(client):
        pass

    monkeypatch.setattr(groups, "ensure_connected", connected)

    async def call():
        if request.param == "join_chat_by_link":
            return await groups.join_chat_by_link("https://t.me/+sample")
        return await groups.import_chat_invite("+sample")

    return call


@pytest.fixture
def invite_client(monkeypatch):
    class InviteClient:
        preview = None

        def __init__(self):
            self.joins = []

        async def __call__(self, request):
            assert request.hash == "sample"
            if isinstance(request, functions.messages.CheckChatInviteRequest):
                return self.preview
            assert isinstance(request, functions.messages.ImportChatInviteRequest)
            self.joins.append(request.hash)
            return SimpleNamespace(chats=[])

    client = InviteClient()
    monkeypatch.setattr(groups, "get_client", lambda account=None: client)
    return client


def _preview():
    return types.ChatInvite(
        title="Preview", photo=types.PhotoEmpty(1), participants_count=1, color=0
    )


@pytest.mark.asyncio
async def test_invite_without_identity_respects_configured_policy(
    monkeypatch, invite_tool, invite_client
):
    invite_client.preview = _preview()
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "@approved")

    result = await invite_tool()

    assert "TELEGRAM_ALLOWED_CHAT_IDS" in result
    assert "Preview" not in result
    assert invite_client.joins == []


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["-1000000000042", "-1000000000042:10", "43"])
async def test_invite_authorizes_the_identified_chat(
    monkeypatch, invite_tool, invite_client, policy
):
    chat = types.Channel(id=42, title="Team", photo=types.ChatPhotoEmpty(), date=None)
    invite_client.preview = types.ChatInvitePeek(chat=chat, expires=None)
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", policy)

    result = await invite_tool()

    if policy == "-1000000000042":
        assert result == "Joined chat via invite hash."
        assert invite_client.joins == ["sample"]
    else:
        assert "TELEGRAM_ALLOWED_CHAT_IDS" in result
        assert "Team" not in result
        assert invite_client.joins == []


@pytest.mark.asyncio
async def test_invite_to_already_joined_hidden_chat_returns_no_title(
    monkeypatch, invite_tool, invite_client
):
    chat = types.Channel(id=42, title="Private Team", photo=types.ChatPhotoEmpty(), date=None)
    invite_client.preview = types.ChatInviteAlready(chat=chat)
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "43")

    assert "TELEGRAM_ALLOWED_CHAT_IDS" in await invite_tool()
    assert invite_client.joins == []


@pytest.mark.asyncio
async def test_invite_without_chat_policy_remains_usable(invite_tool, invite_client):
    invite_client.preview = _preview()

    assert await invite_tool() == "Joined chat via invite hash."
    assert invite_client.joins == ["sample"]

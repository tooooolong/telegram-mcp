"""import_contacts must build real TL objects and send contacts.importContacts."""

from unittest.mock import AsyncMock

import pytest
from telethon import TelegramClient, functions, types
from telethon.sessions import StringSession

from telegram_mcp.tools import contacts


class ImportServer(TelegramClient):
    def __init__(self):
        super().__init__(StringSession(), 12345, "dummy_hash")
        self.requests = []

    async def __call__(self, request, ordered=False, flood_sleep_threshold=None):
        self.requests.append(request)
        assert isinstance(request, functions.contacts.ImportContactsRequest)
        imported = [
            types.ImportedContact(user_id=100 + c.client_id, client_id=c.client_id)
            for c in request.contacts
        ]
        return types.contacts.ImportedContacts(
            imported=imported, popular_invites=[], retry_contacts=[], users=[]
        )


@pytest.fixture
def server(monkeypatch):
    client = ImportServer()

    async def connected(cl=None):
        return None

    monkeypatch.setattr(contacts, "get_client", lambda account=None: client)
    monkeypatch.setattr(contacts, "ensure_connected", connected)
    return client


@pytest.mark.asyncio
async def test_import_contacts_sends_the_contacts(server):
    result = await contacts.import_contacts(
        [
            {"phone": "+79990001122", "first_name": "Иван", "last_name": "Петров"},
            {"phone": "+79990003344", "first_name": "Мария"},
        ]
    )

    assert result == "Imported 2 contacts."
    (request,) = server.requests
    assert [(c.phone, c.first_name, c.last_name) for c in request.contacts] == [
        ("+79990001122", "Иван", "Петров"),
        ("+79990003344", "Мария", ""),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [{"first_name": "Иван"}, {"phone": "+79990001122"}, "+79990001122"],
    ids=["no-phone", "no-first-name", "not-a-dict"],
)
async def test_import_contacts_explains_a_malformed_contact(server, bad):
    result = await contacts.import_contacts([{"phone": "+7999", "first_name": "A"}, bad])

    assert result.startswith("Error: every contact needs")
    assert server.requests == []


@pytest.mark.asyncio
async def test_add_contact_can_still_import_a_phone_without_policy(server):
    result = await contacts.add_contact(phone="+79990001122", first_name="Example")

    assert "added successfully." in result
    (request,) = server.requests
    assert request.contacts[0].phone == "+79990001122"


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["TELEGRAM_ALLOWED_CHAT_IDS", "TELEGRAM_BLOCKED_CHAT_IDS"])
async def test_unverified_phone_import_is_disabled_with_chat_policy(server, monkeypatch, policy):
    monkeypatch.setenv(policy, "100")

    add = await contacts.add_contact(phone="+79990001122", first_name="Example")
    batch = await contacts.import_contacts([{"phone": "+79990001122", "first_name": "Example"}])

    assert add == batch
    assert "Use add_contact with an allowed Telegram username." in batch
    assert server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("allowed", "blocked", "should_send"),
    [("21", "", True), ("14", "", False), ("21", "21", False)],
)
async def test_username_contact_addition_uses_authorized_resolved_user(
    monkeypatch, allowed, blocked, should_send
):
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", allowed)
    monkeypatch.setenv("TELEGRAM_BLOCKED_CHAT_IDS", blocked)
    client = AsyncMock(
        side_effect=[
            types.contacts.ResolvedPeer(
                peer=types.PeerUser(user_id=21),
                users=[types.User(id=19), types.User(id=21, access_hash=27)],
                chats=[],
            ),
            types.UpdatesTooLong(),
        ]
    )
    monkeypatch.setattr(contacts, "get_client", lambda account=None: client)
    monkeypatch.setattr(contacts, "ensure_connected", AsyncMock())

    result = await contacts.add_contact(username="@example_user", first_name="Example")

    requests = [call.args[0] for call in client.await_args_list]
    assert isinstance(requests[0], functions.contacts.ResolveUsernameRequest)
    if should_send:
        assert "added successfully" in result
        assert isinstance(requests[1], functions.contacts.AddContactRequest)
        assert requests[1].id == types.InputUser(user_id=21, access_hash=27)
    else:
        assert "Could not find the requested Telegram user" in result
        assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "request_type"),
    [
        (contacts.delete_contact, functions.contacts.DeleteContactsRequest),
        (contacts.block_user, functions.contacts.BlockRequest),
        (contacts.unblock_user, functions.contacts.UnblockRequest),
    ],
)
@pytest.mark.parametrize("blocked", ["", "@example_user"])
async def test_standalone_contact_actions_require_allowed_resolved_user(
    monkeypatch, tool, request_type, blocked
):
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "21")
    monkeypatch.setenv("TELEGRAM_BLOCKED_CHAT_IDS", blocked)
    user = types.User(id=21, username="example_user", access_hash=27)
    client = AsyncMock()
    monkeypatch.setattr(contacts, "get_client", lambda account=None: client)
    monkeypatch.setattr(contacts, "resolve_entity", AsyncMock(return_value=user))

    result = await tool(user_id=21)

    if blocked:
        assert "Could not find the requested Telegram user" in result
        client.assert_not_awaited()
    else:
        assert result.startswith(("Contact with user ID 21", "User 21"))
        client.assert_awaited_once()
        assert isinstance(client.await_args.args[0], request_type)

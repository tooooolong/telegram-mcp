"""search_global retains Telegram offsets behind opaque continuation tokens."""

import json
from collections import OrderedDict
from datetime import datetime, timezone

import pytest
from telethon import TelegramClient, functions, types
from telethon.sessions import StringSession

from telegram_mcp.tools import messages

NOW = datetime(2026, 10, 16, 9, 0, tzinfo=timezone.utc)
CHANNEL = types.Channel(
    id=1555000000, title="Новости", photo=types.ChatPhotoEmpty(), date=NOW, access_hash=5
)
FRIEND = types.User(id=777, first_name="Friend", access_hash=77)


def _results():
    """Newest first. Message ids are per chat: a small private chat has low ids."""
    posts = [
        types.Message(
            id=5000 - n, peer_id=types.PeerChannel(CHANNEL.id), date=NOW, message=f"отчёт {n}"
        )
        for n in range(300)
    ]
    # The 101st hit is message 3 of a private chat: Telethon's iterator stops here,
    # because a first id <= the chunk size "means" nothing older is left.
    dm = types.Message(id=3, peer_id=types.PeerUser(FRIEND.id), date=NOW, message="отчёт DM")
    return posts[:100] + [dm] + posts[100:]


class GlobalSearchServer(TelegramClient):
    """Answers messages.searchGlobal like Telegram: from the cursor, `limit` at a time."""

    def __init__(self, results=None, response_type=types.messages.MessagesSlice, account_id=51):
        super().__init__(StringSession(), 12345, "dummy_hash")
        self.results = _results() if results is None else results
        self.response_type = response_type
        self.account_id = account_id
        self.requests = []

    async def get_me(self, *, input_peer):
        assert input_peer
        return types.InputPeerUser(self.account_id, 0)

    def _position(self, request):
        if not request.offset_id:
            return 0
        for index, msg in enumerate(self.results):
            same_peer = isinstance(request.offset_peer, types.InputPeerEmpty) or (
                msg.peer_id is not None
                and messages.utils.get_peer_id(msg.peer_id)
                == messages.utils.get_peer_id(request.offset_peer)
            )
            if msg.id == request.offset_id and same_peer:
                # The server's next_rate, carried back (Telethon pairs it with the last
                # non-empty message, so it may belong to a later MessageEmpty).
                assert request.offset_rate >= 1000 + index
                return index + 1
        raise AssertionError("cursor points at no result")

    async def __call__(self, request, ordered=False, flood_sleep_threshold=None):
        assert isinstance(request, functions.messages.SearchGlobalRequest)
        self.requests.append(request)
        start = self._position(request)
        chunk = self.results[start : start + request.limit]
        kwargs = dict(messages=chunk, chats=[CHANNEL], users=[FRIEND], topics=[])
        if self.response_type is types.messages.Messages:
            return types.messages.Messages(**kwargs)
        return types.messages.MessagesSlice(
            count=len(self.results),
            next_rate=1000 + start + len(chunk) - 1 if chunk else None,
            inexact=False,
            **kwargs,
        )


@pytest.fixture
def server(monkeypatch):
    client = GlobalSearchServer()

    async def connected(cl=None):
        return None

    monkeypatch.delenv("TELEGRAM_ALLOWED_CHAT_IDS", raising=False)
    monkeypatch.delenv("TELEGRAM_BLOCKED_CHAT_IDS", raising=False)
    monkeypatch.setattr(messages, "_search_cursors", OrderedDict())
    monkeypatch.setattr(messages, "get_client", lambda account=None: client)
    monkeypatch.setattr(messages, "ensure_connected", connected)
    return client


def ids(result):
    return [(r["chat_id"], r["id"]) for r in json.loads(result)["results"]]


@pytest.mark.asyncio
async def test_telethon_iterator_stops_at_a_low_id_from_another_chat(server):
    # The model reproduces the bug: the chunk that starts with the low id is the last.
    found = await server.get_messages(None, limit=400, search="отчёт")
    assert len(found) == 200 < len(server.results)


@pytest.mark.asyncio
async def test_deep_page_reads_past_a_low_id_from_another_chat(server):
    page6 = json.loads(await messages.search_global("отчёт", page=6, page_size=20))
    assert ids(json.dumps(page6))[0] == (FRIEND.id, 3)
    assert len(page6["results"]) == 20
    assert page6["exhausted"] is False
    assert [r.limit for r in server.requests] == [100, 20]


@pytest.mark.asyncio
async def test_cursor_walks_every_result_once_one_request_per_call(server):
    seen, cursor, calls = [], None, 0
    while True:
        result = json.loads(await messages.search_global("отчёт", page_size=40, cursor=cursor))
        calls += 1
        seen += [(r["chat_id"], r["id"]) for r in result["results"]]
        if result["exhausted"]:
            assert result["next_cursor"] is None
            break
        cursor = result["next_cursor"]
    assert seen == [(messages.utils.get_peer_id(m.peer_id), m.id) for m in server.results]
    assert len(set(seen)) == len(server.results) == 301
    assert len(server.requests) == calls + 1  # the last call learns the end from an empty slice


@pytest.mark.asyncio
async def test_pages_and_cursor_agree(server):
    pages = [ids(await messages.search_global("отчёт", page=n, page_size=50)) for n in (1, 2, 3)]
    first = json.loads(await messages.search_global("отчёт", page_size=50))
    second = await messages.search_global("отчёт", page_size=50, cursor=first["next_cursor"])
    assert pages[0] == ids(json.dumps(first))
    assert pages[1] == ids(second)
    assert len({k for page in pages for k in page}) == 150


@pytest.mark.asyncio
async def test_cursor_needs_no_entity_cache(server, monkeypatch):
    first = json.loads(await messages.search_global("отчёт", page=6, page_size=20, account="work"))
    fresh = GlobalSearchServer()  # A replacement client for the same Telegram account.
    monkeypatch.setattr(messages, "get_client", lambda account=None: fresh)
    second = await messages.search_global(
        "отчёт", page_size=20, cursor=first["next_cursor"], account="work-alias"
    )
    after = server.results[120]  # page 6 of 20 ends with result 119
    assert ids(second)[0] == (messages.utils.get_peer_id(after.peer_id), after.id)
    peer = fresh.requests[0].offset_peer
    assert isinstance(peer, types.InputPeerChannel) and peer.access_hash == CHANNEL.access_hash


@pytest.mark.asyncio
async def test_page_past_the_end_keeps_the_plain_message(server):
    assert await messages.search_global("отчёт", page=20, page_size=20) == (
        "No messages found for this page."
    )


@pytest.mark.asyncio
async def test_a_non_slice_response_is_the_whole_result(server):
    server.response_type = types.messages.Messages
    server.results = server.results[:5]
    result = json.loads(await messages.search_global("отчёт", page_size=20))
    assert len(result["results"]) == 5
    assert result["exhausted"] is True and result["next_cursor"] is None
    assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_cursor_steps_over_empty_messages(server):
    real = server.results[:6]
    server.results = real[:3] + [types.MessageEmpty(id=99, peer_id=None)] + real[3:]
    first = json.loads(await messages.search_global("отчёт", page_size=4))
    assert [r["id"] for r in first["results"]] == [m.id for m in real[:4]]
    rest = json.loads(
        await messages.search_global("отчёт", page_size=20, cursor=first["next_cursor"])
    )
    assert [r["id"] for r in rest["results"]] == [m.id for m in real[4:]]
    assert rest["exhausted"] is True


@pytest.mark.asyncio
async def test_endless_empty_messages_end_the_search(server):
    server.results = server.results[:2] + [
        types.MessageEmpty(id=900 - n, peer_id=None) for n in range(50)
    ]
    result = json.loads(await messages.search_global("отчёт", page_size=10))
    assert len(result["results"]) == 2 and result["exhausted"] is True
    assert len(server.requests) == 1 + messages.SEARCH_GLOBAL_MAX_EMPTY_CHUNKS


@pytest.mark.asyncio
async def test_scoped_search_pages_by_visible_matches(server, monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", str(FRIEND.id))
    second_message = types.Message(
        id=2, peer_id=types.PeerUser(FRIEND.id), date=NOW, message="next"
    )
    hidden_posts = [m for m in server.results if isinstance(m.peer_id, types.PeerChannel)]
    server.results = (
        hidden_posts[:2] + [server.results[100]] + hidden_posts[2:5] + [second_message]
    )

    first = json.loads(await messages.search_global("отчёт", page_size=1))
    assert [(r["chat_id"], r["id"]) for r in first["results"]] == [(FRIEND.id, 3)]
    assert first["exhausted"] is False
    second = json.loads(
        await messages.search_global("отчёт", page_size=1, cursor=first["next_cursor"])
    )
    assert [(r["chat_id"], r["id"]) for r in second["results"]] == [(FRIEND.id, 2)]
    assert second["next_cursor"] is None and second["exhausted"] is True
    assert ids(await messages.search_global("отчёт", page=2, page_size=1)) == [(FRIEND.id, 2)]


@pytest.mark.asyncio
async def test_scoped_no_match_has_same_response_with_or_without_hidden_hits(server, monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", str(FRIEND.id))
    server.results = server.results[:100]
    hidden = await messages.search_global("отчёт")
    server.results = []
    absent = await messages.search_global("отчёт")

    assert hidden == absent == "No messages found for this page."


@pytest.mark.asyncio
async def test_scoped_small_page_searches_full_telegram_batches(server, monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", str(FRIEND.id))

    result = await messages.search_global("отчёт", page_size=1)

    assert ids(result) == [(FRIEND.id, 3)]
    assert all(request.limit == messages.SEARCH_GLOBAL_CHUNK for request in server.requests)


@pytest.mark.asyncio
async def test_cursor_is_tied_to_its_query_and_account(server, monkeypatch):
    first = json.loads(await messages.search_global("отчёт", page_size=10, account="work"))
    result = await messages.search_global("другое", cursor=first["next_cursor"], account="work")
    assert "error" in result.lower()

    replacement = GlobalSearchServer(account_id=52)
    monkeypatch.setattr(messages, "get_client", lambda account=None: replacement)
    result = await messages.search_global("отчёт", cursor=first["next_cursor"], account="work")
    assert "error" in result.lower()
    assert replacement.requests == []
    assert len(server.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("env", "previous", "current"),
    [
        ("TELEGRAM_ALLOWED_CHAT_IDS", None, str(FRIEND.id)),
        (
            "TELEGRAM_ALLOWED_CHAT_IDS",
            f"{messages.utils.get_peer_id(CHANNEL)}:2",
            f"{messages.utils.get_peer_id(CHANNEL)}:3",
        ),
        ("TELEGRAM_BLOCKED_CHAT_IDS", None, str(FRIEND.id)),
    ],
)
async def test_cursor_is_invalidated_by_privacy_policy_changes(
    server, monkeypatch, env, previous, current
):
    if previous is not None:
        monkeypatch.setenv(env, previous)
        for result in server.results[:2]:
            result.reply_to = types.MessageReplyHeader(forum_topic=True, reply_to_msg_id=2)
    first = json.loads(await messages.search_global("отчёт", page_size=1))
    assert first["next_cursor"] is not None
    request_count = len(server.requests)
    monkeypatch.setenv(env, current)

    result = await messages.search_global("отчёт", cursor=first["next_cursor"])

    assert "restart the search" in result.lower()
    assert len(server.requests) == request_count


@pytest.mark.asyncio
async def test_cursor_tokens_are_random_and_expire(server, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(messages, "_search_cursor_clock", lambda: clock[0])
    first = json.loads(await messages.search_global("отчёт", page_size=1))
    repeat = json.loads(await messages.search_global("отчёт", page_size=1))
    assert len(first["next_cursor"]) == len(repeat["next_cursor"]) == 43
    assert first["next_cursor"] != repeat["next_cursor"]

    clock[0] += messages.SEARCH_CURSOR_TTL
    result = await messages.search_global("отчёт", cursor=first["next_cursor"])

    assert "restart the search" in result.lower()
    assert len(server.requests) == 2
    assert not messages._search_cursors


@pytest.mark.asyncio
async def test_cursor_eviction_and_process_restart(server, monkeypatch):
    monkeypatch.setattr(messages, "SEARCH_CURSOR_LIMIT", 1)
    first = json.loads(await messages.search_global("отчёт", page_size=1))
    second = json.loads(await messages.search_global("отчёт", page_size=1))

    result = await messages.search_global("отчёт", cursor=first["next_cursor"])
    assert "restart the search" in result.lower()
    assert len(server.requests) == 2

    messages._search_cursors.clear()
    result = await messages.search_global("отчёт", cursor=second["next_cursor"])
    assert "restart the search" in result.lower()
    assert len(server.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"page": 0},
        {"page_size": 0},
        {"page_size": 101},
        {"cursor": "not-a-cursor"},
        {"cursor": "e30=", "page": 2},
    ],
)
async def test_rejects_bad_paging_before_any_request(server, kwargs):
    result = await messages.search_global("отчёт", **kwargs)
    assert "error" in result.lower()
    assert server.requests == []


@pytest.mark.asyncio
async def test_hidden_tail_does_not_produce_a_cursor(server, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BLOCKED_CHAT_IDS", str(messages.utils.get_peer_id(CHANNEL)))
    server.results = [server.results[100]] + server.results[:5]
    result_with_hidden_tail = json.loads(await messages.search_global("отчёт", page_size=1))
    server.results = server.results[:1]
    result_without_hidden_tail = json.loads(await messages.search_global("отчёт", page_size=1))

    assert result_with_hidden_tail == result_without_hidden_tail
    assert result_with_hidden_tail["next_cursor"] is None
    assert result_with_hidden_tail["exhausted"] is True

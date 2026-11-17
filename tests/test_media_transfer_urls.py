"""Signed download and upload URLs for a client on another machine."""

import inspect
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

import pytest
from starlette.testclient import TestClient

from telegram_mcp import media_links, runtime
from telegram_mcp.tools import media
from telegram_mcp.transport.http import create_streamable_http_app

BASE = "https://tg.example/acct"
_prepare_upload = inspect.unwrap(media.prepare_upload)
_download_media = inspect.unwrap(media.download_media)
pytestmark = pytest.mark.skipif(os.name != "posix", reason="Signed transfers require POSIX")


@pytest.fixture
def root(tmp_path, monkeypatch):
    root = (tmp_path / "files").resolve()
    root.mkdir()
    monkeypatch.setattr(runtime, "SERVER_ALLOWED_ROOTS", [root, tmp_path / "other"])
    monkeypatch.setenv("TELEGRAM_MEDIA_BASE_URL", BASE + "/")
    return root


@pytest.fixture
def client(monkeypatch):
    @asynccontextmanager
    async def test_lifespan():
        yield

    monkeypatch.setattr(runtime.mcp, "_service_lifespan", test_lifespan)
    monkeypatch.setenv("MCP_AUTH_TOKEN", "test-operator-token")
    with TestClient(
        create_streamable_http_app(runtime.mcp), base_url="http://localhost:8765"
    ) as c:
        yield c


def _local(url):
    parts = urlsplit(url)
    return parts.path.removeprefix("/acct") + "?" + parts.query


@pytest.mark.asyncio
async def test_upload_round_trip_lands_under_first_root(root, client):
    slot = json.loads(await _prepare_upload("../../etc/report 1.pdf"))
    target = Path(slot["file_path"])
    assert target.is_relative_to(root / media_links.TRANSFER_SUBDIR / media_links.UPLOAD_SUBDIR)
    assert slot["file_path"].endswith(("/report 1.pdf", "\\report 1.pdf"))

    response = client.put(_local(slot["url"]), content=iter([b"%PDF-", b"bytes"]))

    assert response.status_code == 201
    assert response.json() == {"file_path": slot["file_path"], "size": 10}
    assert target.read_bytes() == b"%PDF-bytes"
    assert target.parent.stat().st_mode & 0o777 == 0o700
    assert target.stat().st_mode & 0o777 == 0o600
    assert not target.with_name(media_links.UPLOAD_PARTIAL).exists()
    assert await runtime._resolve_readable_file_path(
        raw_path=str(target), ctx=None, tool_name="send_file"
    ) == (target, None)


@pytest.mark.asyncio
async def test_upload_url_takes_one_upload(root, client):
    url = _local(json.loads(await _prepare_upload("a.txt"))["url"])
    assert client.put(url, content=b"one").status_code == 201
    assert client.put(url, content=b"two").status_code == 404


@pytest.mark.asyncio
async def test_bad_upload_urls_are_refused(root, client):
    slot = json.loads(await _prepare_upload("a.txt"))
    parts = urlsplit(slot["url"])
    path = parts.path.removeprefix("/acct")
    rel = path.removeprefix(media_links.UPLOAD_ROUTE_PREFIX)
    assert (
        client.put(path + "?" + parts.query.replace("sig=", "sig=0"), content=b"x").status_code
        == 404
    )
    assert (
        client.put(path.replace("a.txt", "b.txt") + "?" + parts.query, content=b"x").status_code
        == 404
    )
    past = int(time.time()) - 1
    expired = media_links._signature("upload", rel, past)
    assert client.put(f"{path}?expires={past}&sig={expired}", content=b"x").status_code == 404

    future = int(time.time()) + 60
    escaping = "../escape.txt"
    sig = media_links._signature("upload", escaping, future)
    assert media_links.resolve_upload(escaping, str(future), sig, root) is None
    # A download signature never authorises an upload.
    sig = media_links._signature("media", rel, future)
    assert media_links.resolve_upload(rel, str(future), sig, root) is None


@pytest.mark.asyncio
async def test_oversize_upload_is_refused(root, client, monkeypatch):
    monkeypatch.setitem(media.MAX_FILE_BYTES, "send_file", 4)
    slot = json.loads(await _prepare_upload("big.bin"))
    assert client.put(_local(slot["url"]), content=b"12345").status_code == 413
    target = Path(slot["file_path"])
    assert not target.exists()
    assert not target.with_name(media_links.UPLOAD_PARTIAL).exists()
    assert client.put(_local(slot["url"]), content=b"1234").status_code == 201


@pytest.mark.asyncio
async def test_no_upload_without_base_root_or_name(root, monkeypatch):
    assert not (await _prepare_upload("..")).startswith("{")
    monkeypatch.setattr(runtime, "SERVER_ALLOWED_ROOTS", [])
    assert not (await _prepare_upload("a.txt")).startswith("{")
    monkeypatch.setattr(runtime, "SERVER_ALLOWED_ROOTS", [root])
    monkeypatch.delenv("TELEGRAM_MEDIA_BASE_URL")
    assert not (await _prepare_upload("a.txt")).startswith("{")


def test_download_url_serves_file_and_refuses_bad_links(root, client, tmp_path):
    (root / "downloads").mkdir()
    file = root / "downloads" / "photo 1.jpg"
    file.write_bytes(b"jpeg")
    url = media_links.download_url(file, root)
    assert url.startswith(BASE + "/media/")
    assert "/photo%201.jpg?" in url

    response = client.get(_local(url))
    assert response.status_code == 200 and response.content == b"jpeg"
    partial = client.get(_local(url), headers={"Range": "bytes=0-1"})
    assert partial.status_code == 206 and partial.content == b"jp"

    assert client.get(_local(url).replace("sig=", "sig=0")).status_code == 404
    outside = tmp_path / "secret.txt"
    outside.write_text("x")
    assert media_links.download_url(outside, root) is None
    future = int(time.time()) + 60
    sig = media_links._signature("media", "../secret.txt", future)
    assert media_links.resolve_download("../secret.txt", str(future), sig, root) is None


@pytest.mark.asyncio
async def test_download_links_keep_distinct_versions_of_one_output(root, client):
    file = root / "report.txt"
    file.write_bytes(b"first report")
    first = media_links.download_url(file, root)
    file.write_bytes(b"updated report")
    second = media_links.download_url(file, root)

    assert client.get(_local(first)).content == b"first report"
    assert client.get(_local(second)).content == b"updated report"
    relative = unquote(urlsplit(first).path.removeprefix("/acct/media/"))
    snapshot = root / media_links.TRANSFER_SUBDIR / media_links.DOWNLOAD_SUBDIR / relative
    readable, read_error = await runtime._resolve_readable_file_path(
        raw_path=str(snapshot), ctx=None, tool_name="send_file"
    )
    writable, write_error = await runtime._resolve_writable_file_path(
        raw_path=str(snapshot), default_filename="report.txt", ctx=None, tool_name="download_media"
    )
    assert readable is None and "signed download URL" in read_error
    assert writable is None and "reserved" in write_error


@pytest.mark.asyncio
async def test_download_media_reports_the_url(root, monkeypatch):
    target = root / "downloads" / "doc.pdf"
    target.parent.mkdir()
    message = SimpleNamespace(id=7, media=object(), file=SimpleNamespace(size=3))

    class Client:
        async def get_messages(self, entity, ids):
            return message

        async def download_media(self, target_media, file, progress_callback):
            path = Path(file).with_suffix(".pdf")
            path.write_bytes(b"pdf")
            return str(path)

    async def resolve_entity(chat_id, cl):
        return "entity"

    async def resolve_path(**kwargs):
        return Path(kwargs["raw_path"] or target).resolve(), None

    monkeypatch.setattr(media, "get_client", lambda account=None: Client())
    monkeypatch.setattr(media, "resolve_entity", resolve_entity)
    monkeypatch.setattr(media, "_resolve_writable_file_path", resolve_path)

    result = await _download_media(123, 7, file_path=str(target))

    assert result.startswith(f"Media downloaded to {target}. From another machine, fetch it from ")
    assert f"{BASE}/media/" in result
    assert "/doc.pdf?" in result


def _age_directory(path):
    stamp = time.time() - 3 * media_links.URL_TTL_SECONDS
    os.utime(path, (stamp, stamp))


def test_sweep_removes_only_stale_managed_slots(root):
    old = Path(media_links.upload_slot("old.txt", root)["file_path"])
    new = Path(media_links.upload_slot("new.txt", root)["file_path"])
    unrelated = old.parent.parent / ("0" * 32)
    unrelated.mkdir(mode=0o700)
    local = root / "uploads" / "my-project"
    local.mkdir(parents=True)
    for path in (old.parent, unrelated, local):
        _age_directory(path)

    media_links.upload_slot("next.txt", root)

    assert not old.parent.exists()
    assert new.parent.is_dir()
    assert unrelated.is_dir()
    assert local.is_dir()


def test_transfer_slots_have_a_capacity_limit(root, monkeypatch):
    monkeypatch.setattr(media_links, "MAX_TRANSFER_SLOTS", 1)
    first = media_links.upload_slot("first.txt", root)
    with pytest.raises(OSError, match="Too many active signed transfers"):
        media_links.upload_slot("second.txt", root)

    _age_directory(Path(first["file_path"]).parent)

    assert media_links.upload_slot("second.txt", root) is not None


def test_sweep_handles_os_errors_gracefully(root, monkeypatch, caplog):
    old = Path(media_links.upload_slot("old.txt", root)["file_path"])
    _age_directory(old.parent)

    def fail_unlink(*args, **kwargs):
        raise PermissionError("Locked by another process")

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with caplog.at_level(logging.WARNING, logger="telegram_mcp.media_links"):
        media_links.upload_slot("next.txt", root)

    assert "Could not remove transfer slot" in caplog.text


def test_cleanup_does_not_walk_untracked_subdirectories(root):
    old = Path(media_links.upload_slot("old.txt", root)["file_path"])
    notes = old.parent / "notes"
    notes.mkdir()
    note = notes / "example.txt"
    note.write_text("operator note", encoding="utf-8")
    _age_directory(old.parent)

    media_links.upload_slot("next.txt", root)

    assert note.read_text(encoding="utf-8") == "operator note"


@pytest.mark.asyncio
async def test_transfer_storage_requires_an_owned_private_directory(root):
    staging = root / media_links.TRANSFER_SUBDIR
    staging.mkdir(mode=0o755)
    staging.chmod(0o755)

    response = await _prepare_upload("report.txt")

    assert "Could not prepare a private upload slot" in response
    assert list(staging.iterdir()) == []


@pytest.mark.asyncio
async def test_transfer_storage_refuses_an_alias_to_a_service_directory(root):
    destination = root / "operator-files"
    destination.mkdir(mode=0o700)
    alias = root / media_links.TRANSFER_SUBDIR
    alias.symlink_to(destination, target_is_directory=True)

    response = await _prepare_upload("report.txt")

    assert "Could not prepare a private upload slot" in response
    assert list(destination.iterdir()) == []


@pytest.mark.asyncio
async def test_transfer_storage_disables_when_private_checks_are_unavailable(root, monkeypatch):
    # Patch only the platform check rather than changing the host's pathlib implementation.
    monkeypatch.setattr(
        media_links, "os", SimpleNamespace(name="nt", fsencode=os.fsencode, getenv=os.getenv)
    )

    assert "Use a POSIX host" in await _prepare_upload("report.txt")

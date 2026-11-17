import asyncio
import io
import inspect
import sys
from threading import BoundedSemaphore
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from mcp.server.fastmcp import Image

from telegram_mcp import document_preview
from telegram_mcp.tools import media

_inspect_document = inspect.unwrap(media.inspect_document)


def _make_dummy_pdf(text: str = "", pages: int = 1) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        page = writer.add_blank_page(width=72, height=72)
        if text:
            resources = DictionaryObject(
                {
                    NameObject("/Font"): DictionaryObject(
                        {
                            NameObject("/F1"): DictionaryObject(
                                {
                                    NameObject("/Type"): NameObject("/Font"),
                                    NameObject("/Subtype"): NameObject("/Type1"),
                                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                                }
                            )
                        }
                    )
                }
            )
            page[NameObject("/Resources")] = resources
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 12 Tf 5 50 Td ({text}) Tj ET".encode("ascii"))
            page[NameObject("/Contents")] = writer._add_object(stream)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class MockClient:
    def __init__(self, message=None, media_bytes=b""):
        self.message = message
        self.media_bytes = media_bytes

    async def get_messages(self, entity, ids):
        return self.message

    async def download_media(self, msg, file):
        file.write(self.media_bytes)
        return file


@pytest.mark.asyncio
async def test_inspect_document_refuses_reported_size_before_download(monkeypatch):
    msg = SimpleNamespace(
        id=34,
        media=object(),
        file=SimpleNamespace(name="large.txt", mime_type="text/plain", size=51),
    )
    client = MockClient(msg)
    client.download_media = AsyncMock()
    monkeypatch.setattr(media, "get_client", lambda account=None: client)
    monkeypatch.setattr(media, "resolve_entity", AsyncMock(return_value="entity"))
    monkeypatch.setattr(media, "DOCUMENT_PREVIEW_MAX_BYTES", 50)

    result = await _inspect_document(123, 34)

    assert "too large" in result
    client.download_media.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("declared_size", [None, 2])
async def test_inspect_document_stops_at_actual_byte_limit(monkeypatch, declared_size):
    msg = SimpleNamespace(
        id=35,
        media=object(),
        file=SimpleNamespace(name="large.txt", mime_type="text/plain", size=declared_size),
    )
    client = MockClient(msg)
    monkeypatch.setattr(media, "get_client", lambda account=None: client)
    monkeypatch.setattr(media, "resolve_entity", AsyncMock(return_value="entity"))
    monkeypatch.setattr(media, "DOCUMENT_PREVIEW_MAX_BYTES", 10)

    async def write_chunks(msg, file):
        file.write(b"first")
        file.write(b"second")
        pytest.fail("The download should stop as soon as it exceeds the limit")

    client.download_media = write_chunks

    result = await _inspect_document(123, 35)

    assert "too large" in result


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_mode", ["cancelled", "download_failed"])
async def test_document_preview_bounds_active_downloads_and_releases_slot(monkeypatch, exit_mode):
    monkeypatch.setattr(media, "_DOCUMENT_PREVIEW_SLOTS", BoundedSemaphore(1))
    started = asyncio.Event()
    finish = asyncio.Event()
    message = SimpleNamespace(
        id=37, media=object(), file=SimpleNamespace(name="doc.txt", mime_type="text/plain")
    )
    client = MockClient(message, b"ready")
    monkeypatch.setattr(media, "get_client", lambda account=None: client)
    monkeypatch.setattr(media, "resolve_entity", AsyncMock(return_value="entity"))

    async def failing_download(msg, file):
        started.set()
        await finish.wait()
        raise OSError("download failed")

    client.download_media = failing_download
    preview = asyncio.create_task(_inspect_document(123, 37))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        assert "preview is busy" in await _inspect_document(123, 37)
        if exit_mode == "cancelled":
            preview.cancel()
            with pytest.raises(asyncio.CancelledError):
                await preview
        else:
            finish.set()
            assert "An error occurred" in await preview

        client.download_media = MockClient.download_media.__get__(client)
        assert await _inspect_document(123, 37) == "ready"
    finally:
        if not preview.done():
            preview.cancel()
            await asyncio.gather(preview, return_exceptions=True)


@pytest.mark.asyncio
async def test_inspect_document_no_media(monkeypatch):
    msg = SimpleNamespace(id=1, media=None)
    client = MockClient(message=msg)
    monkeypatch.setattr(media, "get_client", lambda account=None: client)
    monkeypatch.setattr(media, "resolve_entity", AsyncMock(return_value="entity"))

    result = await _inspect_document(123, 1)
    assert "no attached document" in result


@pytest.mark.asyncio
async def test_inspect_document_image(monkeypatch):
    img_data = b"\x89PNG\r\n\x1a\nfakeimage"
    msg = SimpleNamespace(
        id=2,
        media=object(),
        file=SimpleNamespace(name="photo.png", mime_type="image/png"),
    )
    client = MockClient(message=msg, media_bytes=img_data)
    monkeypatch.setattr(media, "get_client", lambda account=None: client)
    monkeypatch.setattr(media, "resolve_entity", AsyncMock(return_value="entity"))

    result = await _inspect_document(123, 2)
    assert isinstance(result, Image)
    assert result.data == img_data
    assert result._format == "png"


@pytest.mark.asyncio
async def test_inspect_document_text(monkeypatch):
    txt_data = "Hello, world! Multi-language text.".encode("utf-8")
    msg = SimpleNamespace(
        id=3,
        media=object(),
        file=SimpleNamespace(name="notes.txt", mime_type="text/plain"),
    )
    client = MockClient(message=msg, media_bytes=txt_data)
    monkeypatch.setattr(media, "get_client", lambda account=None: client)
    monkeypatch.setattr(media, "resolve_entity", AsyncMock(return_value="entity"))

    result = await _inspect_document(123, 3)
    assert result == "Hello, world! Multi-language text."


@pytest.mark.skipif(sys.platform != "linux", reason="PDF worker requires Linux resource limits")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data, expected",
    [
        (_make_dummy_pdf(), "does not contain a text layer"),
        (_make_dummy_pdf("Sample Contract Text Line 1"), "Sample Contract Text Line 1"),
        (b"invalid PDF", "Error reading PDF"),
    ],
)
async def test_inspect_document_pdf_worker(monkeypatch, data, expected):
    msg = SimpleNamespace(
        id=4,
        media=object(),
        file=SimpleNamespace(name="contract.pdf", mime_type="application/pdf"),
    )
    client = MockClient(message=msg, media_bytes=data)
    monkeypatch.setattr(media, "get_client", lambda account=None: client)
    monkeypatch.setattr(media, "resolve_entity", AsyncMock(return_value="entity"))

    result = await _inspect_document(123, 4)

    assert expected in result


@pytest.mark.skipif(sys.platform == "linux", reason="PDF worker available on Linux")
@pytest.mark.asyncio
async def test_pdf_preview_declines_when_no_hard_memory_limit_is_available():
    result = await document_preview.preview_pdf(_make_dummy_pdf(), "contract.pdf")

    assert "cannot enforce the parser's memory limit" in result


@pytest.mark.asyncio
async def test_pdf_preview_kills_and_reaps_worker_at_deadline(monkeypatch):
    worker = SimpleNamespace(killed=False, reaped=False)
    worker.kill = lambda: setattr(worker, "killed", True)

    async def communicate(data=None):
        if data is not None:
            await asyncio.Event().wait()
        worker.reaped = True
        return b"", b""

    worker.communicate = communicate
    monkeypatch.setattr(
        document_preview, "sys", SimpleNamespace(platform="linux", executable="python")
    )
    monkeypatch.setattr(
        document_preview.asyncio, "create_subprocess_exec", AsyncMock(return_value=worker)
    )
    monkeypatch.setattr(document_preview, "PDF_PREVIEW_TIMEOUT_SECONDS", 0.01)

    result = await document_preview.preview_pdf(_make_dummy_pdf(), "contract.pdf")

    assert "exceeded its processing time limit" in result
    assert worker.killed and worker.reaped


def test_pdf_preview_caps_pages_and_text(monkeypatch):
    monkeypatch.setattr(document_preview, "PDF_PREVIEW_MAX_PAGES", 2)
    monkeypatch.setattr(document_preview, "DOCUMENT_PREVIEW_MAX_TEXT", 50)
    data = _make_dummy_pdf("Sample Contract Text Line 1", pages=4)

    result = document_preview._extract_pdf(data)

    assert result["page_count"] == 4
    assert len(result["text"]) <= 50
    assert "--- Page 3 ---" not in result["text"]
    assert result["truncated"] is True


@pytest.mark.asyncio
async def test_document_text_preview_is_truncated(monkeypatch):
    message = SimpleNamespace(
        id=36, media=object(), file=SimpleNamespace(name="doc.txt", mime_type="text/plain")
    )
    text = "é" * 8_000 + "水" * 8_000 + "text" * 2_000
    client = MockClient(message, text.encode("utf-8"))
    monkeypatch.setattr(media, "get_client", lambda account=None: client)
    monkeypatch.setattr(media, "resolve_entity", AsyncMock(return_value="entity"))

    result = await _inspect_document(123, 36)

    preview, marker = result.split("\n\n")
    assert len(preview) == 20_000
    assert marker == "[Preview truncated at 20,000 characters.]"


@pytest.mark.asyncio
async def test_inspect_document_unsupported(monkeypatch):
    msg = SimpleNamespace(
        id=5,
        media=object(),
        file=SimpleNamespace(name="archive.zip", mime_type="application/zip"),
    )
    client = MockClient(message=msg, media_bytes=b"zipdata")
    monkeypatch.setattr(media, "get_client", lambda account=None: client)
    monkeypatch.setattr(media, "resolve_entity", AsyncMock(return_value="entity"))

    result = await _inspect_document(123, 5)
    assert "is not currently supported for direct text analysis" in result


def test_inspect_document_is_registered_and_read_only():
    from telegram_mcp import runtime

    registered = {tool.name: tool for tool in runtime.mcp._tool_manager.list_tools()}
    assert "inspect_document" in registered
    assert registered["inspect_document"].annotations.readOnlyHint is True


@pytest.mark.asyncio
async def test_inspect_document_zero_disk_footprint(monkeypatch, tmp_path):
    download_targets = []

    async def mock_download_media(msg, file):
        download_targets.append(file)
        file.write(b"Line 1\nLine 2")
        return file

    client = SimpleNamespace(
        get_messages=AsyncMock(
            return_value=SimpleNamespace(
                id=10,
                media=object(),
                file=SimpleNamespace(name="doc.txt", mime_type="text/plain"),
            )
        ),
        download_media=mock_download_media,
    )
    monkeypatch.setattr(media, "get_client", lambda account=None: client)
    monkeypatch.setattr(media, "resolve_entity", AsyncMock(return_value="entity"))

    before_files = list(tmp_path.iterdir())
    res = await _inspect_document(123, 10)
    after_files = list(tmp_path.iterdir())

    assert res == "Line 1\nLine 2"
    assert len(download_targets) == 1 and isinstance(download_targets[0], io.BytesIO)
    assert before_files == after_files  # Zero disk pollution


GIF = b"GIF89a\x01\x00\x01\x00\x00\x00\x00;"
JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF"
WEBP = b"RIFF\x04\x00\x00\x00WEBPVP8 "


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name, mime, data, expected",
    [
        ("smile.gif", "image/gif", GIF, "gif"),
        ("renamed.jpg", "image/jpeg", GIF, "gif"),  # declared type is wrong, bytes decide
        ("photo.jpg", "image/jpeg", JPEG, "jpeg"),
        ("sticker.webp", "image/webp", WEBP, "webp"),
    ],
)
async def test_inspect_document_names_the_image_format_from_its_bytes(
    monkeypatch, name, mime, data, expected
):
    msg = SimpleNamespace(id=31, media=object(), file=SimpleNamespace(name=name, mime_type=mime))
    monkeypatch.setattr(media, "get_client", lambda account=None: MockClient(msg, data))
    monkeypatch.setattr(media, "resolve_entity", AsyncMock(return_value="entity"))

    result = await _inspect_document(123, 31)

    assert isinstance(result, Image)
    assert result._format == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name, mime, data",
    [
        ("scan.bmp", "image/bmp", b"BM6\x00\x00\x00"),
        ("IMG.heic", "image/heic", b"\x00\x00\x00\x18ftypheic"),
    ],
)
async def test_inspect_document_refuses_images_the_claude_api_cannot_take(
    monkeypatch, name, mime, data
):
    msg = SimpleNamespace(id=32, media=object(), file=SimpleNamespace(name=name, mime_type=mime))
    monkeypatch.setattr(media, "get_client", lambda account=None: MockClient(msg, data))
    monkeypatch.setattr(media, "resolve_entity", AsyncMock(return_value="entity"))

    result = await _inspect_document(123, 32)

    assert isinstance(result, str)
    assert "supported formats are JPEG, PNG, GIF and WebP" in result


@pytest.mark.asyncio
async def test_inspect_document_does_not_download_a_file_it_cannot_read(monkeypatch):
    msg = SimpleNamespace(
        id=33,
        media=object(),
        file=SimpleNamespace(name="backup.zip", mime_type="application/zip", size=3_145_728),
    )
    client = MockClient(msg, b"zip")
    client.download_media = AsyncMock()
    monkeypatch.setattr(media, "get_client", lambda account=None: client)
    monkeypatch.setattr(media, "resolve_entity", AsyncMock(return_value="entity"))

    result = await _inspect_document(123, 33)

    assert "is not currently supported" in result
    client.download_media.assert_not_called()

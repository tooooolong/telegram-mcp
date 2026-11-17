"""Parse PDF previews in a child process with hard resource limits."""

from __future__ import annotations

import asyncio
import io
import json
import sys
from pathlib import Path

DOCUMENT_PREVIEW_MAX_BYTES = 5 * 1024 * 1024
DOCUMENT_PREVIEW_MAX_TEXT = 20_000
PDF_PREVIEW_MAX_PAGES = 20
PDF_PREVIEW_MAX_MEMORY = 256 * 1024 * 1024
PDF_PREVIEW_MAX_CPU_SECONDS = 4
PDF_PREVIEW_TIMEOUT_SECONDS = 5.0

_PDF_PREVIEW_UNAVAILABLE = (
    "PDF preview is unavailable because this host cannot enforce the parser's memory limit."
)
_PDF_WORKER_UNAVAILABLE = 3


def _apply_pdf_worker_limits() -> bool:
    if sys.platform != "linux":
        return False

    import resource

    try:
        for resource_id, bound in (
            (resource.RLIMIT_AS, PDF_PREVIEW_MAX_MEMORY),
            (resource.RLIMIT_CPU, PDF_PREVIEW_MAX_CPU_SECONDS),
            (resource.RLIMIT_CORE, 0),
        ):
            soft, hard = resource.getrlimit(resource_id)
            limit = min(value for value in (bound, soft, hard) if value != resource.RLIM_INFINITY)
            resource.setrlimit(resource_id, (limit, limit))
    except (OSError, ValueError):
        return False
    return True


def _extract_pdf(data: bytes) -> dict:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    page_count = len(reader.pages)
    parts = []
    remaining = DOCUMENT_PREVIEW_MAX_TEXT
    has_text = False
    text_truncated = False
    for index in range(min(page_count, PDF_PREVIEW_MAX_PAGES)):
        text = (reader.pages[index].extract_text() or "").strip()
        has_text |= bool(text)
        prefix = f"--- Page {index + 1} ---\n"
        if parts:
            prefix = "\n\n" + prefix
        parts.append(prefix[:remaining])
        remaining = max(0, remaining - len(prefix))
        parts.append(text[:remaining])
        if len(text) > remaining or not remaining:
            text_truncated = True
            break
        remaining -= len(text)

    return {
        "page_count": page_count,
        "text": "".join(parts).rstrip() if has_text else "",
        "truncated": text_truncated or page_count > PDF_PREVIEW_MAX_PAGES,
    }


def _pdf_worker() -> int:
    if not _apply_pdf_worker_limits():
        return _PDF_WORKER_UNAVAILABLE

    data = sys.stdin.buffer.read(DOCUMENT_PREVIEW_MAX_BYTES + 1)
    if len(data) > DOCUMENT_PREVIEW_MAX_BYTES:
        return 1

    try:
        result = _extract_pdf(data)
    except Exception:
        return 1
    sys.stdout.buffer.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
    return 0


async def preview_pdf(data: bytes, filename: str) -> str:
    if sys.platform != "linux":
        return _PDF_PREVIEW_UNAVAILABLE

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-B",
        str(Path(__file__).resolve()),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env={},
    )
    try:
        output, _ = await asyncio.wait_for(
            process.communicate(data), timeout=PDF_PREVIEW_TIMEOUT_SECONDS
        )
    except (asyncio.TimeoutError, asyncio.CancelledError) as error:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        await process.communicate()
        if isinstance(error, asyncio.CancelledError):
            raise
        return "Error reading PDF. The preview exceeded its processing time limit."

    if process.returncode == _PDF_WORKER_UNAVAILABLE:
        return _PDF_PREVIEW_UNAVAILABLE
    if process.returncode != 0:
        return "Error reading PDF. The file may be damaged or exceed a processing limit."

    return _format_pdf_preview(json.loads(output), filename)


def _format_pdf_preview(result: dict, filename: str) -> str:
    page_count = result["page_count"]
    if result["text"]:
        preview = f"Contents of document '{filename}' ({page_count} pages):\n\n{result['text']}"
    elif result["truncated"]:
        preview = f"No text layer found in the preview of '{filename}' ({page_count} pages)."
    else:
        preview = (
            f"Document '{filename}' ({page_count} pages) does not contain a text layer "
            "(possibly a scanned image without OCR)."
        )
    if result["truncated"]:
        preview += (
            f"\n\n[Preview limited to {PDF_PREVIEW_MAX_PAGES} pages "
            f"and {DOCUMENT_PREVIEW_MAX_TEXT:,} characters.]"
        )
    return preview


if __name__ == "__main__":
    raise SystemExit(_pdf_worker())

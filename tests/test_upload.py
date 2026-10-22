import asyncio
from io import BytesIO
import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

import pdf_worker
import upload_processing as processing


@pytest.fixture
def client(monkeypatch):
    # Mock at the provider boundary before importing main: no provider setup,
    # API key, database connection, or AI request is needed for these tests.
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused/unused")
    provider = ModuleType("litellm")
    completion = AsyncMock(return_value=SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content="สรุปทดสอบ"))
    ]))
    provider.Router = Mock(return_value=SimpleNamespace(acompletion=completion))
    monkeypatch.setitem(sys.modules, "litellm", provider)
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    monkeypatch.setattr(main.asyncpg, "create_pool", Mock(side_effect=AssertionError("database accessed")))
    main.app.dependency_overrides[main.current_user] = lambda: {"userId": 1}
    monkeypatch.setattr(processing, "upload_slots", asyncio.Semaphore(2))
    # Intentionally do not enter TestClient's lifespan (which starts PostgreSQL).
    test_client = TestClient(main.app)
    try:
        yield test_client, completion
    finally:
        test_client.close()
        main.app.dependency_overrides.clear()
        sys.modules.pop("main", None)


def make_pdf(*texts):
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
        })
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.mark.parametrize("filename,payload", [
    ("notes.txt", "เนื้อหาทดสอบ".encode()),
    ("report.PDF", make_pdf("Hello PDF")),
])
def test_normal_upload_preserves_response(client, filename, payload):
    http, ai = client
    response = http.post("/api/upload", files={"file": (filename, payload)})
    assert response.status_code == 200
    assert response.json() == {
        "fileName": filename, "summary": "สรุปทดสอบ", "model_used": "document-model"
    }
    ai.assert_awaited_once()
    assert ai.call_args.kwargs["model"] == "document-model"
    expected = "Hello PDF" if filename.endswith("PDF") else payload.decode()
    assert expected in ai.call_args.kwargs["messages"][0]["content"]


def test_exact_limit_and_oversize(client, monkeypatch):
    http, ai = client
    monkeypatch.setattr(processing, "MAX_UPLOAD_BYTES", 100)
    assert http.post("/api/upload", files={"file": ("a.txt", b"x" * 100)}).status_code == 200
    ai.reset_mock()
    result = http.post("/api/upload", files={"file": ("a.txt", b"x" * 101)})
    assert result.status_code == 413
    assert set(result.json()) == {"detail"}
    ai.assert_not_awaited()


@pytest.mark.parametrize("declared_size", [None, 1])
def test_actual_size_checked_in_chunks_and_file_closed(monkeypatch, declared_size):
    monkeypatch.setattr(processing, "MAX_UPLOAD_BYTES", 100)
    monkeypatch.setattr(processing, "READ_CHUNK_BYTES", 16)
    file = UploadFile(BytesIO(b"x" * 1000), filename="a.txt", size=declared_size)
    original_read = file.read
    file.read = AsyncMock(side_effect=original_read)
    with pytest.raises(HTTPException) as error:
        asyncio.run(processing.extract_upload(file))
    assert error.value.status_code == 413
    assert file.file.closed
    sizes = [call.args[0] for call in file.read.call_args_list]
    assert all(0 < size <= 16 for size in sizes)
    assert sum(sizes) == 101


@pytest.mark.parametrize("payload", [b"not a PDF", b"%PDF-1.7\nbroken"])
def test_invalid_pdf_is_400_without_ai(client, payload):
    http, ai = client
    result = http.post("/api/upload", files={"file": ("bad.pdf", payload)})
    assert result.status_code == 400
    assert set(result.json()) == {"detail"}
    ai.assert_not_awaited()


@pytest.mark.parametrize("filename,payload", [("a.txt", b" \n"), ("a.pdf", make_pdf(""))])
def test_empty_text_is_400(client, filename, payload):
    http, ai = client
    assert http.post("/api/upload", files={"file": (filename, payload)}).status_code == 400
    ai.assert_not_awaited()


def test_pdf_page_limit_in_real_worker(client, monkeypatch):
    http, ai = client
    monkeypatch.setattr(processing, "PDF_MAX_PAGES", 1)
    result = http.post("/api/upload", files={"file": ("pages.pdf", make_pdf("First", "Second"))})
    assert result.status_code == 200
    content = ai.call_args.kwargs["messages"][0]["content"]
    assert "First" in content and "Second" not in content


def test_pdf_stops_before_extracting_more_pages(monkeypatch):
    first = SimpleNamespace(extract_text=Mock(return_value="x" * 12001))
    second = SimpleNamespace(extract_text=Mock(side_effect=AssertionError("extra extraction")))
    monkeypatch.setattr("pypdf.PdfReader", lambda _: SimpleNamespace(pages=iter([first, second])))
    assert pdf_worker.extract_text(b"", 50, 12000) == "x" * 12000
    second.extract_text.assert_not_called()


def test_pdf_never_visits_pages_past_limit(monkeypatch):
    def pages():
        yield SimpleNamespace(extract_text=lambda: "first")
        raise AssertionError("visited past page limit")
    monkeypatch.setattr("pypdf.PdfReader", lambda _: SimpleNamespace(pages=pages()))
    assert pdf_worker.extract_text(b"", 1, 12000) == "first"


def test_text_prompt_is_bounded(client):
    http, ai = client
    assert http.post("/api/upload", files={"file": ("a.txt", b"x" * 13000)}).status_code == 200
    assert ai.call_args.kwargs["messages"][0]["content"].split("\n\n", 1)[1] == "x" * 12000


@pytest.mark.parametrize("cancel", [False, True])
def test_parser_timeout_or_cancellation_kills_and_reaps(monkeypatch, cancel):
    async def scenario():
        started = asyncio.Event()
        async def communicate(_):
            started.set()
            await asyncio.Event().wait()
        process = SimpleNamespace(returncode=None, communicate=communicate,
                                  kill=Mock(), wait=AsyncMock())
        monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
        monkeypatch.setattr(processing, "PDF_TIMEOUT_SECONDS", 0.01)
        task = asyncio.create_task(processing.extract_pdf(b"%PDF"))
        await started.wait()
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(HTTPException) as error:
                await task
            assert error.value.status_code == 400
        process.kill.assert_called_once()
        process.wait.assert_awaited_once()
    asyncio.run(scenario())


def test_resource_failure_is_400(client, monkeypatch):
    http, ai = client
    process = SimpleNamespace(returncode=-9, communicate=AsyncMock(return_value=(b"", b"")),
                              wait=AsyncMock())
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    result = http.post("/api/upload", files={"file": ("a.pdf", b"%PDF")})
    assert result.status_code == 400
    ai.assert_not_awaited()


def test_busy_uploads_rejected_without_reading(monkeypatch):
    async def scenario():
        slots = asyncio.Semaphore(1)
        monkeypatch.setattr(processing, "upload_slots", slots)
        file = UploadFile(BytesIO(b"hello"), filename="a.txt")
        file.read = AsyncMock()
        async with slots:
            with pytest.raises(HTTPException) as error:
                await processing.extract_upload(file)
        assert error.value.status_code == 503
        file.read.assert_not_awaited()
        assert file.file.closed
    asyncio.run(scenario())


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_invalid_environment_limit_fails_fast(monkeypatch, value):
    monkeypatch.setenv("MAX_UPLOAD_BYTES", value)
    with pytest.raises(ValueError):
        processing.positive_int("MAX_UPLOAD_BYTES", 10)


def test_environment_limit_can_be_configured(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1234")
    assert processing.positive_int("MAX_UPLOAD_BYTES", 10) == 1234


def test_real_subprocess_timeout_is_reaped(monkeypatch):
    async def scenario():
        create = asyncio.create_subprocess_exec
        children = []
        async def slow_worker(*args, **kwargs):
            process = await create(sys.executable, "-c", "import time; time.sleep(60)", **kwargs)
            children.append(process)
            return process
        monkeypatch.setattr(asyncio, "create_subprocess_exec", slow_worker)
        monkeypatch.setattr(processing, "PDF_TIMEOUT_SECONDS", 0.05)
        with pytest.raises(HTTPException) as error:
            await processing.extract_pdf(b"%PDF")
        assert error.value.status_code == 400
        assert len(children) == 1 and children[0].returncode is not None
    asyncio.run(scenario())


def test_real_worker_memory_ceiling_rejects_pdf(client, monkeypatch):
    http, ai = client
    monkeypatch.setattr(processing, "PDF_MEMORY_MB", 1)
    result = http.post("/api/upload", files={"file": ("a.pdf", make_pdf("Hello"))})
    assert result.status_code == 400
    ai.assert_not_awaited()

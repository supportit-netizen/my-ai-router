"""Bound upload reads and isolate untrusted PDF parsing from the web process."""

import asyncio
import json
import os
from pathlib import Path
import sys

from fastapi import HTTPException, UploadFile


def positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


MAX_UPLOAD_BYTES = positive_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
PDF_MAX_PAGES = positive_int("PDF_MAX_PAGES", 50)
PDF_TIMEOUT_SECONDS = positive_int("PDF_TIMEOUT_SECONDS", 10)
PDF_MEMORY_MB = positive_int("PDF_MEMORY_MB", 512)
MAX_CONCURRENT_UPLOADS = positive_int("MAX_CONCURRENT_UPLOADS", 2)
MAX_TEXT_CHARS = 12000
READ_CHUNK_BYTES = 64 * 1024
upload_slots = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)


async def read_upload(file: UploadFile) -> bytes:
    # The metadata is only an early optimization; count actual bytes as well.
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="ไฟล์มีขนาดเกินกำหนด")
    payload = bytearray()
    while True:
        # Read at most one byte past the limit, including at the exact boundary.
        chunk = await file.read(min(READ_CHUNK_BYTES, MAX_UPLOAD_BYTES - len(payload) + 1))
        if not chunk:
            return bytes(payload)
        if len(payload) + len(chunk) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="ไฟล์มีขนาดเกินกำหนด")
        payload.extend(chunk)


async def extract_pdf(payload: bytes) -> str:
    if not payload.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="ไฟล์นี้ไม่ใช่ PDF ที่ถูกต้อง")
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(Path(__file__).with_name("pdf_worker.py")),
        str(PDF_MAX_PAGES), str(MAX_TEXT_CHARS), str(PDF_MEMORY_MB),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        output, _ = await asyncio.wait_for(
            process.communicate(payload), timeout=PDF_TIMEOUT_SECONDS
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=400, detail="อ่าน PDF ไม่สำเร็จ: ใช้เวลาเกินกำหนด") from exc
    finally:
        # Includes timeout, client cancellation and unexpected communication errors.
        # Never release the admission slot while a parser is still running.
        if process.returncode is None:
            process.kill()
        await process.wait()
    if process.returncode != 0:
        raise HTTPException(status_code=400, detail="อ่าน PDF ไม่สำเร็จ: ไฟล์ไม่ถูกต้องหรือใช้ทรัพยากรเกินกำหนด")
    return json.loads(output)


async def extract_upload(file: UploadFile) -> str:
    try:
        # Do not accumulate an unbounded queue of in-memory upload payloads.
        if upload_slots.locked():
            raise HTTPException(status_code=503, detail="ระบบกำลังประมวลผลไฟล์ กรุณาลองใหม่อีกครั้ง")
        async with upload_slots:
            payload = await read_upload(file)
            if file.filename and file.filename.lower().endswith(".pdf"):
                return await extract_pdf(payload)
            return payload.decode("utf-8", errors="replace")[:MAX_TEXT_CHARS]
    finally:
        await file.close()

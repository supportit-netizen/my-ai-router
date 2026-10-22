# my-ai-router

## Upload limits

Set these environment variables before starting the application (restart after
changes). All values must be positive integers; invalid values fail startup.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAX_UPLOAD_BYTES` | `10485760` | Maximum file size, 10 MiB; larger files return HTTP 413 |
| `PDF_MAX_PAGES` | `50` | Maximum pages considered for the summary |
| `PDF_TIMEOUT_SECONDS` | `10` | PDF worker wall-clock deadline, including startup |
| `PDF_MEMORY_MB` | `512` | PDF worker virtual address-space ceiling in MiB |
| `MAX_CONCURRENT_UPLOADS` | `2` | Concurrent file reads/parsers per web process; excess requests return HTTP 503 |

Files are read in chunks of at most 64 KiB, counting actual bytes even when
upload size metadata is missing or incorrect. Files are closed on success and
failure. Successful responses retain `fileName`, `summary`, and `model_used`.
Errors retain FastAPI's `{"detail": "..."}` envelope. Invalid, empty, timed-out,
or resource-exhausting PDFs return HTTP 400 without contacting the AI provider.

Only the first 12,000 characters are passed to the document model. PDF extraction
stops at that character budget or the page budget, whichever comes first; later
pages are intentionally omitted. PDF parsing runs in a disposable subprocess,
so a single compressed page or malformed page tree is subject to both the memory
ceiling and timeout. A timed-out or cancelled parser is killed and reaped before
its admission slot is released. The memory ceiling requires Linux/POSIX
`resource.RLIMIT_AS` (supported by the Docker image); unsupported platforms fail
PDF processing closed. Size the worker count and ceiling for the container's
memory budget; the ceiling covers virtual memory, not just resident memory.

FastAPI/Starlette parses multipart bodies and spools uploads before entering the
endpoint. These application limits bound subsequent reads and PDF processing;
configure a request-body size limit and connection/rate limits at the ingress
proxy as well to bound multipart transfer and temporary-disk use. Allow multipart
overhead above `MAX_UPLOAD_BYTES`. Each Uvicorn worker has its own upload slots.

## Regression tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use generated PDFs and mocked AI completions. They require neither a real
AI API key nor a running PostgreSQL server. Linux CI also exercises the real
PDF subprocess and its resource-limit setup.

"""One disposable PDF parser. Only bounded text is written to stdout."""

from io import BytesIO
from itertools import islice
import json
import sys


def extract_text(payload: bytes, max_pages: int, max_chars: int) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(payload))
    parts = []
    remaining = max_chars
    # Do not compute len(reader.pages): that may traverse an enormous page tree.
    for page in islice(reader.pages, max_pages):
        if parts:
            parts.append("\n")
            remaining -= 1
            if remaining <= 0:
                break
        text = (page.extract_text() or "")[:remaining]
        parts.append(text)
        remaining -= len(text)
        if remaining <= 0:
            break
    return "".join(parts)


if __name__ == "__main__":
    try:
        # Production runs in the Linux Python Docker image. Fail closed on
        # platforms without resource limits rather than silently losing them.
        import resource

        max_pages, max_chars, memory_mb = map(int, sys.argv[1:])
        memory_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        # Imports, decompression, page-tree traversal and extraction all run
        # inside this limit and the parent's wall-clock deadline.
        payload = sys.stdin.buffer.read()
        print(json.dumps(extract_text(payload, max_pages, max_chars)))
    except Exception:
        sys.exit(1)

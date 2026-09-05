import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any

import asyncpg
import bcrypt
import jwt
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from litellm import Router
from pydantic import BaseModel, EmailStr
from pypdf import PdfReader


DATABASE_URL = os.environ["DATABASE_URL"]
JWT_SECRET = os.getenv("JWT_SECRET", os.getenv("LITELLM_MASTER_KEY", "change-this-secret"))
JWT_ALGORITHM = "HS256"


def normalize_model(value: str, provider: str) -> str:
    """Accept legacy model names already stored in Railway variables."""
    if provider == "gemini" and value.startswith("google/"):
        return f"gemini/{value.removeprefix('google/')}"
    if provider == "openrouter" and not value.startswith("openrouter/"):
        return f"openrouter/{value}"
    return value


MODEL_LIST = [
    {
        "model_name": "general-model",
        "litellm_params": {
            "model": normalize_model(
                os.getenv("GENERAL_MODEL", "gemini/gemini-2.5-flash"), "gemini"
            ),
            "api_key": os.getenv("GEMINI_API_KEY"),
        },
    },
    {
        "model_name": "code-model",
        "litellm_params": {
            "model": normalize_model(
                os.getenv("CODE_MODEL", "openai/gpt-4o-mini"), "openrouter"
            ),
            "api_key": os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY")),
        },
    },
    {
        "model_name": "document-model",
        "litellm_params": {
            "model": normalize_model(
                os.getenv("DOCUMENT_MODEL", "openai/gpt-4o-mini"), "openrouter"
            ),
            "api_key": os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY")),
        },
    },
    {
        "model_name": "image-model",
        "litellm_params": {
            "model": normalize_model(
                os.getenv("IMAGE_MODEL", "openai/gpt-image-1"), "openrouter"
            ),
            "api_key": os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY")),
        },
    },
]

router = Router(model_list=MODEL_LIST, num_retries=2, timeout=90)
db_pool: asyncpg.Pool | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global db_pool
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
              id SERIAL PRIMARY KEY,
              email TEXT UNIQUE NOT NULL,
              password_hash TEXT NOT NULL,
              created_at TIMESTAMP DEFAULT NOW()
            )
            """
        )
    yield
    await db_pool.close()


app = FastAPI(title="modest-harmony on LiteLLM", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Credentials(BaseModel):
    email: EmailStr
    password: str


class ChatRequest(BaseModel):
    prompt: str


class OpenAIChatRequest(BaseModel):
    model: str = "general-model"
    messages: list[dict[str, Any]]
    stream: bool = False


def issue_token(user_id: int, email: str) -> str:
    return jwt.encode(
        {
            "userId": user_id,
            "email": email,
            "exp": datetime.now(timezone.utc) + timedelta(days=7),
        },
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )


async def current_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="กรุณาเข้าสู่ระบบ")
    try:
        return jwt.decode(authorization[7:], JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="เซสชันหมดอายุ กรุณาเข้าสู่ระบบใหม่") from exc


def lite_response_text(response: Any) -> str:
    content = response.choices[0].message.content
    return content or ""


@app.get("/")
async def root():
    return {"status": "ready", "engine": "LiteLLM", "service": "modest-harmony"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/register")
async def register(body: Credentials):
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="รหัสผ่านต้องมีอย่างน้อย 6 ตัวอักษร")
    assert db_pool is not None
    password_hash = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()
    try:
        row = await db_pool.fetchrow(
            "INSERT INTO users (email, password_hash) VALUES ($1, $2) RETURNING id, email",
            body.email.lower(),
            password_hash,
        )
    except asyncpg.UniqueViolationError as exc:
        raise HTTPException(status_code=409, detail="อีเมลนี้ถูกใช้งานแล้ว") from exc
    return {"token": issue_token(row["id"], row["email"]), "email": row["email"]}


@app.post("/api/login")
async def login(body: Credentials):
    assert db_pool is not None
    row = await db_pool.fetchrow(
        "SELECT id, email, password_hash FROM users WHERE email = $1", body.email.lower()
    )
    if not row or not bcrypt.checkpw(body.password.encode(), row["password_hash"].encode()):
        raise HTTPException(status_code=401, detail="อีเมลหรือรหัสผ่านไม่ถูกต้อง")
    return {"token": issue_token(row["id"], row["email"]), "email": row["email"]}


@app.post("/api/chat")
async def chat(body: ChatRequest, _: dict[str, Any] = Depends(current_user)):
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Please provide a prompt")

    lower = prompt.lower()
    if any(word in lower for word in ("image", "draw", "รูปภาพ", "วาด")):
        model = "image-model"
        response = await router.aimage_generation(model=model, prompt=prompt)
        item = response.data[0]
        image_url = item.get("url") if isinstance(item, dict) else getattr(item, "url", None)
        image_b64 = item.get("b64_json") if isinstance(item, dict) else getattr(item, "b64_json", None)
        if image_b64:
            image_url = f"data:image/png;base64,{image_b64}"
        return {
            "routedTo": "IMAGE",
            "model_used": model,
            "result": "สร้างภาพเรียบร้อยแล้วครับ",
            "imageUrl": image_url,
        }

    is_code = any(word in lower for word in ("code", "python", "โค้ด", "bug", "เขียนโปรแกรม"))
    alias = "code-model" if is_code else "general-model"
    response = await router.acompletion(
        model=alias,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "routedTo": "CODE" if is_code else "GENERAL",
        "model_used": alias,
        "result": lite_response_text(response),
        "imageUrl": None,
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), _: dict[str, Any] = Depends(current_user)):
    payload = await file.read()
    if file.filename and file.filename.lower().endswith(".pdf"):
        if not payload.startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail="ไฟล์นี้ไม่ใช่ PDF ที่ถูกต้อง")
        try:
            extracted = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(payload)).pages)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"อ่าน PDF ไม่สำเร็จ: {exc}") from exc
    else:
        extracted = payload.decode("utf-8", errors="replace")
    if not extracted.strip():
        raise HTTPException(status_code=400, detail="ไม่สามารถอ่านเนื้อหาจากไฟล์นี้ได้")
    response = await router.acompletion(
        model="document-model",
        messages=[
            {
                "role": "user",
                "content": f"สรุปเนื้อหาไฟล์ต่อไปนี้เป็นภาษาไทย กระชับ ได้ใจความ:\n\n{extracted[:12000]}",
            }
        ],
    )
    return {
        "fileName": file.filename,
        "summary": lite_response_text(response),
        "model_used": "document-model",
    }


@app.post("/v1/chat/completions")
async def openai_compatible(
    body: OpenAIChatRequest,
    authorization: str | None = Header(default=None),
):
    expected = os.getenv("LITELLM_MASTER_KEY")
    if expected and authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid API key")
    if body.stream:
        raise HTTPException(status_code=400, detail="Streaming is not enabled on this compatibility route")
    response = await router.acompletion(model=body.model, messages=body.messages)
    return response.model_dump()


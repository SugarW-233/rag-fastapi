from fastapi import FastAPI

from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router
from app.core.config import settings
from app.db.database import Base, engine

settings.upload_dir.mkdir(
    parents=True,
    exist_ok=True,
)

settings.chroma_dir.mkdir(
    parents=True,
    exist_ok=True,
)

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="基于 FastAPI、LangChain 和 Chroma 的 RAG 后端",
)

app.include_router(documents_router)
app.include_router(chat_router)


@app.get(
    "/health",
    tags=["System"],
)
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": settings.app_name,
    }

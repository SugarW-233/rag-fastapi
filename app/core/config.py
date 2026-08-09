from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """项目配置，自动从环境变量和.env文件读取"""

    app_name: str = "RAG FastAPI Backend"

    openai_api_key: str
    openai_chat_model: str = "gpt-5.4-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    chroma_collection: str = "rag_knowledge_base"
    chroma_dir: Path = Path("storage/chroma")
    upload_dir: Path = Path("storage/uploads")

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4

    max_upload_size_mb: int = 10

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """只创建一次 Settings 对象。"""
    return Settings()


settings = get_settings()

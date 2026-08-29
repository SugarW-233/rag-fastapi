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

    # 最终交给相关性判断和 LLM 的文档数量。
    top_k: int = 4

    # 混合检索候选数量。
    dense_candidate_k: int = 20
    bm25_candidate_k: int = 20
    fusion_candidate_k: int = 20
    rrf_k: int = 60

    # Reranker 配置。
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_batch_size: int = 8
    reranker_device: str = "cpu"
    reranker_max_length: int = 512

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

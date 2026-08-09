from functools import lru_cache

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import settings


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    """创建 Embeddings 模型"""

    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )


@lru_cache
def get_vector_store() -> Chroma:
    """创建或连接本地 Chroma 向量数据库"""

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )


@lru_cache
def get_llm() -> ChatOpenAI:
    """创建聊天模型"""

    return ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,
    )


@lru_cache
def delete_document_vectors(
    document_id: str,
    chunk_count: int | None,
) -> None:
    if not chunk_count:
        return

    chunk_ids = [f"{document_id}:{index}" for index in range(chunk_count)]

    get_vector_store().delete(ids=chunk_ids)

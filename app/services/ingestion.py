from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.services.bm25_store import get_bm25_store
from app.services.vector_store import get_vector_store


def load_file(file_path: Path) -> list[Document]:
    """根据文件扩展名选择不同 Loader"""

    suffix = file_path.suffix.lower()  # pathlib.Path 对象  .suffix后缀(带.) .lower小写

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))

    elif suffix == ".txt":
        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
        )

    else:
        raise ValueError(f"暂不支持该文件类型：{suffix}")

    return loader.load()


def ingest_file(
    file_path: Path,
    original_name: str,
    document_id: str,
) -> dict:
    """
    完整入库流程：

    文件
    -> Loader
    -> Document
    -> Splitter
    -> Chunk
    -> Embedding
    -> Chroma
    -> BM25 缓存失效
    """

    documents = load_file(file_path)

    if not documents:
        raise ValueError("文件没有解析出任何内容")

    # 给每个原始 Document 补充 metadata。
    for document in documents:
        document.metadata.update(
            {
                "document_id": document_id,
                "file_name": original_name,
                "source": original_name,
            }
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    # 删除可能出现的空文本块。
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

    if not chunks:
        raise ValueError("文件切分后没有可用文本")

    # 每个 Chunk 使用唯一 ID。
    chunk_ids = [f"{document_id}:{index}" for index in range(len(chunks))]

    # 同时把 chunk_id 写入 metadata。
    # 后面的混合检索将使用它判断两个结果是否是同一个 chunk。
    for chunk, chunk_id in zip(
        chunks,
        chunk_ids,
    ):
        chunk.metadata["chunk_id"] = chunk_id

    vector_store = get_vector_store()

    try:
        vector_store.add_documents(
            documents=chunks,
            ids=chunk_ids,
        )

    except Exception:
        # add_documents 可能只写入了部分 chunk，
        # 尝试删除本次文档对应的所有 chunk。
        vector_store.delete(
            ids=chunk_ids,
        )

        raise

    finally:
        # 无论写入成功还是发生部分写入后回滚，
        # 下次 BM25 查询都应该重新读取 Chroma。
        get_bm25_store().invalidate()

    return {
        "document_count": len(documents),
        "chunk_count": len(chunks),
    }

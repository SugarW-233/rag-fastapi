from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
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
    """

    documents = load_file(file_path)

    if not documents:
        raise ValueError("文件没有解析出任何内容")

    # 给每个原始 Document 补充 metadata
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

    # 删除可能出现的空文本块
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

    if not chunks:
        raise ValueError("文件切分后没有可用文本")

    # 每个 Chunk 使用唯一ID
    chunk_ids = [f"{document_id}:{index}" for index in range(len(chunks))]

    vector_store = get_vector_store()

    # Chroma会调用Embedding模型，然后保存文本、向量和metadata
    try:
        vector_store.add_documents(
            documents=chunks,
            ids=chunk_ids,
        )
    except Exception:
        vector_store.delete(ids=chunk_ids)
        raise

    return {
        "document_count": len(documents),
        "chunk_count": len(chunks),
    }

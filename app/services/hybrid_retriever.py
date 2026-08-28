import asyncio
import hashlib
from collections.abc import Sequence

from langchain_core.documents import Document

from app.services.bm25_store import get_bm25_store
from app.services.vector_store import get_vector_store


def document_key(
    document: Document,
) -> str:
    """
    为文本块生成稳定身份。

    新数据优先使用 document_id、page 和 start_index，
    这样能够兼容 metadata 中没有 chunk_id 的旧数据。
    """

    metadata = document.metadata

    document_id = metadata.get("document_id")
    page = metadata.get("page")
    start_index = metadata.get("start_index")

    if document_id is not None and start_index is not None:
        return f"location:{document_id}:{page}:{start_index}"

    chunk_id = metadata.get("chunk_id")

    if chunk_id:
        return f"chunk:{chunk_id}"

    # 最后的兼容方案：
    # 根据来源、页码和正文生成稳定哈希。
    source = metadata.get(
        "source",
        metadata.get("file_name", ""),
    )

    identity_text = "\x1f".join(
        [
            str(document_id or ""),
            str(source or ""),
            str(page or ""),
            document.page_content,
        ]
    )

    digest = hashlib.sha256(identity_text.encode("utf-8")).hexdigest()

    return f"hash:{digest}"


async def dense_search(
    question: str,
    top_k: int,
) -> list[Document]:
    """使用 Chroma 进行向量检索。"""

    if top_k < 1:
        return []

    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": top_k,
        },
    )

    documents = await retriever.ainvoke(question)

    return list(documents)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Document]],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[Document]:
    """
    使用 Reciprocal Rank Fusion 融合多组排序结果。

    每个文本块在一组结果中的贡献为：

        1 / (rrf_k + rank)

    rank 从 1 开始。
    """

    if top_k < 1:
        return []

    if rrf_k < 1:
        raise ValueError("rrf_k 必须大于 0")

    scores: dict[str, float] = {}
    documents_by_key: dict[str, Document] = {}

    for ranking in rankings:
        # 防止同一个检索器意外返回重复 chunk，
        # 导致同一路结果被重复计分。
        seen_in_ranking: set[str] = set()

        for rank, document in enumerate(
            ranking,
            start=1,
        ):
            key = document_key(document)

            if key in seen_in_ranking:
                continue

            seen_in_ranking.add(key)

            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

            existing_document = documents_by_key.get(key)

            if existing_document is None:
                documents_by_key[key] = Document(
                    page_content=(document.page_content),
                    metadata=dict(document.metadata),
                )
            else:
                # 合并 Dense 和 BM25 两边的 metadata。
                # 例如保留 BM25Store 添加的 bm25_score。
                merged_metadata = {
                    **existing_document.metadata,
                    **document.metadata,
                }

                documents_by_key[key] = Document(
                    page_content=(existing_document.page_content),
                    metadata=merged_metadata,
                )

    sorted_keys = sorted(
        scores,
        key=lambda key: (
            -scores[key],
            key,
        ),
    )

    results: list[Document] = []

    for key in sorted_keys[:top_k]:
        document = documents_by_key[key]

        results.append(
            Document(
                page_content=document.page_content,
                metadata={
                    **document.metadata,
                    "rrf_score": scores[key],
                },
            )
        )

    return results


async def hybrid_retrieve(
    question: str,
    *,
    dense_top_k: int,
    bm25_top_k: int,
    fusion_top_k: int,
    rrf_k: int = 60,
) -> list[Document]:
    """
    并行执行 Dense 和 BM25 检索，再使用 RRF 融合。

    这里返回的是 reranker 的候选集合，
    还不是最终交给 LLM 的 top_k。
    """

    if not question.strip():
        return []

    bm25_store = get_bm25_store()

    dense_documents, bm25_documents = await asyncio.gather(
        dense_search(
            question=question,
            top_k=dense_top_k,
        ),
        asyncio.to_thread(
            bm25_store.search,
            question,
            bm25_top_k,
        ),
    )

    return reciprocal_rank_fusion(
        [
            dense_documents,
            bm25_documents,
        ],
        top_k=fusion_top_k,
        rrf_k=rrf_k,
    )

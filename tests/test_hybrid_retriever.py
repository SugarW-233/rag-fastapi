import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.documents import Document

from app.services import hybrid_retriever


def make_document(
    document_id: str,
    content: str,
    *,
    start_index: int = 0,
    chunk_id: str | None = None,
    bm25_score: float | None = None,
) -> Document:
    metadata = {
        "document_id": document_id,
        "file_name": f"{document_id}.txt",
        "start_index": start_index,
    }

    if chunk_id is not None:
        metadata["chunk_id"] = chunk_id

    if bm25_score is not None:
        metadata["bm25_score"] = bm25_score

    return Document(
        page_content=content,
        metadata=metadata,
    )


def test_document_key_works_with_old_metadata() -> None:
    dense_document = make_document(
        document_id="doc-1",
        content="相同文本",
        start_index=100,
    )

    bm25_document = make_document(
        document_id="doc-1",
        content="相同文本",
        start_index=100,
        chunk_id="doc-1:2",
    )

    assert hybrid_retriever.document_key(
        dense_document
    ) == hybrid_retriever.document_key(bm25_document)


def test_rrf_merges_and_deduplicates_documents() -> None:
    document_a = make_document(
        "doc-a",
        "文档 A",
    )

    dense_document_b = make_document(
        "doc-b",
        "文档 B",
    )

    bm25_document_b = make_document(
        "doc-b",
        "文档 B",
        chunk_id="doc-b:0",
        bm25_score=4.5,
    )

    document_c = make_document(
        "doc-c",
        "文档 C",
    )

    results = hybrid_retriever.reciprocal_rank_fusion(
        [
            [
                document_a,
                dense_document_b,
            ],
            [
                bm25_document_b,
                document_c,
            ],
        ],
        top_k=3,
        rrf_k=60,
    )

    result_ids = [document.metadata["document_id"] for document in results]

    # B 同时出现在两路检索中，因此融合后排名最高。
    assert result_ids == [
        "doc-b",
        "doc-a",
        "doc-c",
    ]

    # B 只能出现一次。
    assert result_ids.count("doc-b") == 1

    # Dense 与 BM25 的 metadata 应当合并。
    assert results[0].metadata["bm25_score"] == 4.5

    assert isinstance(
        results[0].metadata["rrf_score"],
        float,
    )


def test_rrf_rejects_invalid_rrf_k() -> None:
    with pytest.raises(
        ValueError,
        match="rrf_k 必须大于 0",
    ):
        hybrid_retriever.reciprocal_rank_fusion(
            [],
            top_k=4,
            rrf_k=0,
        )


def test_hybrid_retrieve_calls_both_retrievers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dense_document = make_document(
        "doc-dense",
        "向量检索结果",
    )

    bm25_document = make_document(
        "doc-bm25",
        "关键词检索结果",
        bm25_score=3.2,
    )

    dense_search_mock = AsyncMock(
        return_value=[
            dense_document,
        ]
    )

    fake_bm25_store = Mock()
    fake_bm25_store.search.return_value = [
        bm25_document,
    ]

    monkeypatch.setattr(
        hybrid_retriever,
        "dense_search",
        dense_search_mock,
    )

    monkeypatch.setattr(
        hybrid_retriever,
        "get_bm25_store",
        lambda: fake_bm25_store,
    )

    results = asyncio.run(
        hybrid_retriever.hybrid_retrieve(
            question="测试问题",
            dense_top_k=10,
            bm25_top_k=8,
            fusion_top_k=5,
            rrf_k=60,
        )
    )

    dense_search_mock.assert_awaited_once_with(
        question="测试问题",
        top_k=10,
    )

    fake_bm25_store.search.assert_called_once_with(
        "测试问题",
        8,
    )

    assert len(results) == 2

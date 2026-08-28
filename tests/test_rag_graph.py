import asyncio
from unittest.mock import AsyncMock

import pytest
from langchain_core.documents import Document

from app.services import rag


def make_document(
    document_id: str,
    content: str,
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "document_id": document_id,
            "file_name": f"{document_id}.txt",
            "start_index": 0,
        },
    )


def test_retrieve_documents_uses_hybrid_and_reranker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_a = make_document(
        "doc-a",
        "混合检索候选 A",
    )

    candidate_b = make_document(
        "doc-b",
        "混合检索候选 B",
    )

    reranked_document = Document(
        page_content=candidate_b.page_content,
        metadata={
            **candidate_b.metadata,
            "reranker_score": 0.9,
        },
    )

    hybrid_mock = AsyncMock(
        return_value=[
            candidate_a,
            candidate_b,
        ]
    )

    reranker_mock = AsyncMock(
        return_value=[
            reranked_document,
        ]
    )

    monkeypatch.setattr(
        rag,
        "hybrid_retrieve",
        hybrid_mock,
    )

    monkeypatch.setattr(
        rag,
        "arerank_documents",
        reranker_mock,
    )

    results = asyncio.run(
        rag.retrieve_documents(
            question="测试混合检索",
            top_k=1,
        )
    )

    assert results == [
        reranked_document,
    ]

    hybrid_mock.assert_awaited_once_with(
        question="测试混合检索",
        dense_top_k=max(
            rag.settings.dense_candidate_k,
            1,
        ),
        bm25_top_k=max(
            rag.settings.bm25_candidate_k,
            1,
        ),
        fusion_top_k=max(
            rag.settings.fusion_candidate_k,
            1,
        ),
        rrf_k=rag.settings.rrf_k,
    )

    reranker_mock.assert_awaited_once_with(
        question="测试混合检索",
        documents=[
            candidate_a,
            candidate_b,
        ],
        top_k=1,
    )


def test_retrieve_documents_skips_reranker_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hybrid_mock = AsyncMock(return_value=[])

    reranker_mock = AsyncMock()

    monkeypatch.setattr(
        rag,
        "hybrid_retrieve",
        hybrid_mock,
    )

    monkeypatch.setattr(
        rag,
        "arerank_documents",
        reranker_mock,
    )

    results = asyncio.run(
        rag.retrieve_documents(
            question="不存在的问题",
            top_k=4,
        )
    )

    assert results == []

    reranker_mock.assert_not_awaited()


def test_no_documents_routes_to_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieve_mock = AsyncMock(return_value=[])

    monkeypatch.setattr(
        rag,
        "retrieve_documents",
        retrieve_mock,
    )

    response = asyncio.run(
        rag.answer_question(
            question="知识库里不存在的问题",
            top_k=4,
        )
    )

    assert response.answer == ("根据当前知识库无法确定。")

    assert response.sources == []

    retrieve_mock.assert_awaited_once_with(
        question="知识库里不存在的问题",
        top_k=4,
    )


def test_generated_refusal_has_no_sources(monkeypatch):
    document = Document(
        page_content="资料只介绍了相关主题，但没有问题要求的具体答案。",
        metadata={
            "document_id": "doc-1",
            "file_name": "test.txt",
            "page": 1,
        },
    )

    generate_mock = AsyncMock(
        return_value=("根据当前知识库无法确定具体答案。现有资料只说明了相关主题。[1]")
    )

    monkeypatch.setattr(
        rag,
        "generate_grounded_answer",
        generate_mock,
    )

    result = asyncio.run(
        rag.generate_answer_node(
            {
                "question": "资料中没有答案的问题",
                "documents": [document],
            }
        )
    )

    assert result["answer"] == rag.REFUSAL_ANSWER
    assert result["sources"] == []


def test_generated_supported_answer_keeps_sources(monkeypatch):
    document = Document(
        page_content="平台明确不接受任何加密货币付款。",
        metadata={
            "document_id": "doc-2",
            "file_name": "billing.txt",
            "page": 1,
        },
    )

    generate_mock = AsyncMock(return_value="平台不接受比特币付款。[1]")

    monkeypatch.setattr(
        rag,
        "generate_grounded_answer",
        generate_mock,
    )

    result = asyncio.run(
        rag.generate_answer_node(
            {
                "question": "平台是否接受比特币付款？",
                "documents": [document],
            }
        )
    )

    assert result["answer"] == "平台不接受比特币付款。[1]"
    assert len(result["sources"]) == 1
    assert result["sources"][0].file_name == "billing.txt"

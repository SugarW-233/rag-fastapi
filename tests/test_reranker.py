import asyncio
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from app.services import reranker


def make_document(
    document_id: str,
    content: str,
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "document_id": document_id,
            "file_name": f"{document_id}.txt",
        },
    )


def test_reranker_sorts_by_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_a = make_document(
        "doc-a",
        "不太相关的内容",
    )

    document_b = make_document(
        "doc-b",
        "最相关的内容",
    )

    document_c = make_document(
        "doc-c",
        "部分相关的内容",
    )

    fake_model = Mock()

    # 对应 A、B、C 的模型分数。
    fake_model.predict.return_value = [
        0.1,
        0.9,
        0.5,
    ]

    monkeypatch.setattr(
        reranker,
        "get_reranker",
        lambda: fake_model,
    )

    results = reranker.rerank_documents(
        question="哪个内容最相关",
        documents=[
            document_a,
            document_b,
            document_c,
        ],
        top_k=2,
    )

    result_ids = [document.metadata["document_id"] for document in results]

    assert result_ids == [
        "doc-b",
        "doc-c",
    ]

    assert results[0].metadata["reranker_score"] == 0.9

    assert results[1].metadata["reranker_score"] == 0.5

    # 不应修改原始 Document。
    assert "reranker_score" not in (document_b.metadata)

    predict_arguments = fake_model.predict.call_args

    pairs = predict_arguments.args[0]

    assert pairs == [
        [
            "哪个内容最相关",
            "不太相关的内容",
        ],
        [
            "哪个内容最相关",
            "最相关的内容",
        ],
        [
            "哪个内容最相关",
            "部分相关的内容",
        ],
    ]

    assert (
        predict_arguments.kwargs["batch_size"] == reranker.settings.reranker_batch_size
    )


def test_reranker_preserves_order_for_equal_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_a = make_document(
        "doc-a",
        "文档 A",
    )

    document_b = make_document(
        "doc-b",
        "文档 B",
    )

    fake_model = Mock()

    fake_model.predict.return_value = [
        0.5,
        0.5,
    ]

    monkeypatch.setattr(
        reranker,
        "get_reranker",
        lambda: fake_model,
    )

    results = reranker.rerank_documents(
        question="测试问题",
        documents=[
            document_a,
            document_b,
        ],
        top_k=2,
    )

    assert [document.metadata["document_id"] for document in results] == [
        "doc-a",
        "doc-b",
    ]


def test_reranker_handles_empty_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_reranker_mock = Mock()

    monkeypatch.setattr(
        reranker,
        "get_reranker",
        get_reranker_mock,
    )

    assert (
        reranker.rerank_documents(
            question="测试问题",
            documents=[],
            top_k=4,
        )
        == []
    )

    assert (
        reranker.rerank_documents(
            question="   ",
            documents=[
                make_document(
                    "doc-a",
                    "文档 A",
                )
            ],
            top_k=4,
        )
        == []
    )

    assert (
        reranker.rerank_documents(
            question="测试问题",
            documents=[
                make_document(
                    "doc-a",
                    "文档 A",
                )
            ],
            top_k=0,
        )
        == []
    )

    # 无有效输入时不应加载模型。
    get_reranker_mock.assert_not_called()


def test_reranker_rejects_wrong_score_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = Mock()

    fake_model.predict.return_value = [
        0.8,
    ]

    monkeypatch.setattr(
        reranker,
        "get_reranker",
        lambda: fake_model,
    )

    with pytest.raises(
        RuntimeError,
        match="分数数量与候选文档数量不一致",
    ):
        reranker.rerank_documents(
            question="测试问题",
            documents=[
                make_document(
                    "doc-a",
                    "文档 A",
                ),
                make_document(
                    "doc-b",
                    "文档 B",
                ),
            ],
            top_k=2,
        )


def test_async_reranker_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = make_document(
        "doc-a",
        "文档 A",
    )

    rerank_mock = Mock(
        return_value=[
            document,
        ]
    )

    monkeypatch.setattr(
        reranker,
        "rerank_documents",
        rerank_mock,
    )

    results = asyncio.run(
        reranker.arerank_documents(
            question="测试问题",
            documents=[
                document,
            ],
            top_k=1,
        )
    )

    assert results == [
        document,
    ]

    rerank_mock.assert_called_once_with(
        question="测试问题",
        documents=[
            document,
        ],
        top_k=1,
    )

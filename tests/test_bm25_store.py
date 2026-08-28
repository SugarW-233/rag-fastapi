from unittest.mock import Mock

import pytest

from app.services import bm25_store


def build_chroma_result() -> dict:
    return {
        "ids": [
            "doc-fastapi:0",
            "doc-chroma:0",
            "doc-hash:0",
        ],
        "documents": [
            "FastAPI 用于构建 Python Web API。",
            "系统使用 Chroma 保存文本向量和文档元数据。",
            "上传文件使用 SHA-256 哈希值判断是否重复。",
        ],
        "metadatas": [
            {
                "document_id": "doc-fastapi",
                "file_name": "fastapi.txt",
                "start_index": 0,
            },
            {
                "document_id": "doc-chroma",
                "file_name": "chroma.txt",
                "start_index": 0,
            },
            {
                "document_id": "doc-hash",
                "file_name": "upload.txt",
                "start_index": 0,
            },
        ],
    }


@pytest.fixture
def fake_vector_store(
    monkeypatch: pytest.MonkeyPatch,
) -> Mock:
    vector_store = Mock()

    vector_store.get.return_value = build_chroma_result()

    monkeypatch.setattr(
        bm25_store,
        "get_vector_store",
        lambda: vector_store,
    )

    return vector_store


def test_bm25_search_returns_relevant_document(
    fake_vector_store: Mock,
) -> None:
    store = bm25_store.BM25Store()

    results = store.search(
        query="文本向量保存在 Chroma 的什么位置",
        top_k=2,
    )

    assert results

    first_result = results[0]

    assert first_result.metadata["document_id"] == ("doc-chroma")

    assert first_result.metadata["file_name"] == ("chroma.txt")

    # 原 metadata 没有 chunk_id，应使用 Chroma 返回的 ID。
    assert first_result.metadata["chunk_id"] == ("doc-chroma:0")

    assert isinstance(
        first_result.metadata["bm25_score"],
        float,
    )

    fake_vector_store.get.assert_called_once_with(
        include=[
            "documents",
            "metadatas",
        ],
    )


def test_bm25_search_returns_empty_for_unknown_terms(
    fake_vector_store: Mock,
) -> None:
    store = bm25_store.BM25Store()

    results = store.search(
        query="董事长的手机号码",
        top_k=4,
    )

    assert results == []


def test_bm25_search_returns_empty_for_invalid_top_k(
    fake_vector_store: Mock,
) -> None:
    store = bm25_store.BM25Store()

    results = store.search(
        query="Chroma",
        top_k=0,
    )

    assert results == []

    # top_k 无效时不需要读取 Chroma。
    fake_vector_store.get.assert_not_called()


def test_bm25_reuses_index_until_invalidated(
    fake_vector_store: Mock,
) -> None:
    store = bm25_store.BM25Store()

    first_results = store.search(
        query="Chroma 文本向量",
        top_k=2,
    )

    second_results = store.search(
        query="SHA-256 重复文件",
        top_k=2,
    )

    assert first_results
    assert second_results

    # 两次检索共用第一次构建的索引。
    assert fake_vector_store.get.call_count == 1

    store.invalidate()

    third_results = store.search(
        query="FastAPI Web API",
        top_k=2,
    )

    assert third_results

    # invalidate 后，下一次检索会重新读取 Chroma。
    assert fake_vector_store.get.call_count == 2


def test_rebuild_immediately_reloads_chroma(
    fake_vector_store: Mock,
) -> None:
    store = bm25_store.BM25Store()

    store.rebuild()

    assert fake_vector_store.get.call_count == 1

    results = store.search(
        query="Chroma",
        top_k=2,
    )

    assert results

    # rebuild 已经完成，search 不会再次读取 Chroma。
    assert fake_vector_store.get.call_count == 1

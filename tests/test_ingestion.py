from pathlib import Path
from unittest.mock import Mock

import pytest

from app.services import ingestion


def test_ingestion_adds_document_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = tmp_path / "knowledge.txt"

    file_path.write_text(
        ("FastAPI 是一个 Python Web 框架。LangChain 可以帮助开发大模型应用。"),
        encoding="utf-8",
    )

    fake_vector_store = Mock()
    fake_bm25_store = Mock()

    monkeypatch.setattr(
        ingestion,
        "get_vector_store",
        lambda: fake_vector_store,
    )

    monkeypatch.setattr(
        ingestion,
        "get_bm25_store",
        lambda: fake_bm25_store,
    )

    result = ingestion.ingest_file(
        file_path=file_path,
        original_name="knowledge.txt",
        document_id="document-123",
    )

    assert result["document_count"] == 1
    assert result["chunk_count"] >= 1

    fake_vector_store.add_documents.assert_called_once()

    call_arguments = fake_vector_store.add_documents.call_args.kwargs

    chunks = call_arguments["documents"]
    chunk_ids = call_arguments["ids"]

    assert len(chunks) == result["chunk_count"]
    assert len(chunk_ids) == result["chunk_count"]

    for index, chunk in enumerate(chunks):
        expected_chunk_id = f"document-123:{index}"

        assert chunk.metadata["document_id"] == ("document-123")

        assert chunk.metadata["file_name"] == ("knowledge.txt")

        assert chunk.metadata["source"] == ("knowledge.txt")

        assert chunk.metadata["chunk_id"] == (expected_chunk_id)

        assert chunk_ids[index] == (expected_chunk_id)

    fake_bm25_store.invalidate.assert_called_once_with()


def test_ingestion_invalidates_bm25_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = tmp_path / "knowledge.txt"

    file_path.write_text(
        "用于测试失败回滚的文本内容。",
        encoding="utf-8",
    )

    fake_vector_store = Mock()
    fake_bm25_store = Mock()

    fake_vector_store.add_documents.side_effect = RuntimeError("模拟 Chroma 写入失败")

    monkeypatch.setattr(
        ingestion,
        "get_vector_store",
        lambda: fake_vector_store,
    )

    monkeypatch.setattr(
        ingestion,
        "get_bm25_store",
        lambda: fake_bm25_store,
    )

    with pytest.raises(
        RuntimeError,
        match="模拟 Chroma 写入失败",
    ):
        ingestion.ingest_file(
            file_path=file_path,
            original_name="knowledge.txt",
            document_id="document-123",
        )

    add_arguments = fake_vector_store.add_documents.call_args.kwargs

    chunk_ids = add_arguments["ids"]

    fake_vector_store.delete.assert_called_once_with(
        ids=chunk_ids,
    )

    fake_bm25_store.invalidate.assert_called_once_with()

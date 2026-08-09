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
        "FastAPI 是一个 Python Web 框架。LangChain 可以帮助开发大模型应用。",
        encoding="utf-8",
    )

    fake_vector_store = Mock()

    monkeypatch.setattr(
        ingestion,
        "get_vector_store",
        lambda: fake_vector_store,
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
        assert chunk.metadata["document_id"] == ("document-123")

        assert chunk.metadata["file_name"] == ("knowledge.txt")

        assert chunk.metadata["source"] == ("knowledge.txt")

        assert chunk_ids[index] == (f"document-123:{index}")

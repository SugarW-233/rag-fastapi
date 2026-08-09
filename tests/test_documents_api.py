from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.routes import documents as documents_route
from app.core.config import settings
from app.db.models import DocumentRecord


@pytest.fixture
def fake_ingestion(
    monkeypatch: pytest.MonkeyPatch,
) -> Mock:
    mock = Mock(
        return_value={
            "document_count": 2,
            "chunk_count": 3,
        }
    )

    monkeypatch.setattr(
        documents_route,
        "ingest_file",
        mock,
    )

    return mock


def upload_test_pdf(
    client: TestClient,
):
    return client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "manual.pdf",
                b"%PDF-1.4\nfake-pdf-for-tests",
                "application/pdf",
            )
        },
    )


def test_duplicate_file_is_rejected(
    client: TestClient,
    fake_ingestion: Mock,
) -> None:
    first_response = upload_test_pdf(client)
    second_response = upload_test_pdf(client)

    assert first_response.status_code == 201
    assert second_response.status_code == 409

    assert second_response.json()["detail"]["message"] == "该文件已经上传"

    # 第二次上传在真正入库前就被哈希去重拦截。
    assert fake_ingestion.call_count == 1


def test_document_list_returns_uploaded_record(
    client: TestClient,
    fake_ingestion: Mock,
) -> None:
    upload_response = upload_test_pdf(client)

    assert upload_response.status_code == 201

    document_id = upload_response.json()["document_id"]

    list_response = client.get("/api/v1/documents")

    assert list_response.status_code == 200

    documents = list_response.json()

    assert len(documents) == 1

    document = documents[0]

    assert document["id"] == document_id
    assert document["file_name"] == "manual.pdf"
    assert document["status"] == "completed"
    assert document["page_count"] == 2
    assert document["chunk_count"] == 3


def test_delete_removes_database_file_and_vectors(
    client: TestClient,
    fake_ingestion: Mock,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delete_vectors_mock = Mock()

    monkeypatch.setattr(
        documents_route,
        "delete_document_vectors",
        delete_vectors_mock,
    )

    upload_response = upload_test_pdf(client)

    assert upload_response.status_code == 201

    document_id = upload_response.json()["document_id"]

    record = db_session.scalar(
        select(DocumentRecord).where(DocumentRecord.id == document_id)
    )

    assert record is not None

    stored_path = settings.upload_dir / record.stored_name

    assert stored_path.exists()

    delete_response = client.delete(f"/api/v1/documents/{document_id}")

    assert delete_response.status_code == 204

    delete_vectors_mock.assert_called_once_with(
        document_id=document_id,
        chunk_count=3,
    )

    assert not stored_path.exists()

    db_session.expire_all()

    deleted_record = db_session.get(
        DocumentRecord,
        document_id,
    )

    assert deleted_record is None


def test_delete_document_vectors_builds_chunk_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import vector_store

    fake_store = Mock()

    monkeypatch.setattr(
        vector_store,
        "get_vector_store",
        lambda: fake_store,
    )

    vector_store.delete_document_vectors(
        document_id="doc-123",
        chunk_count=3,
    )

    fake_store.delete.assert_called_once_with(
        ids=[
            "doc-123:0",
            "doc-123:1",
            "doc-123:2",
        ]
    )

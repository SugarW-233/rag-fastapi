from fastapi.testclient import TestClient


def test_health_check(
    client: TestClient,
) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "RAG FastAPI Backend",
    }


def test_empty_question_returns_422(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/chat",
        json={
            "question": "",
        },
    )

    assert response.status_code == 422

    body = response.json()

    assert "detail" in body
    assert body["detail"][0]["loc"] == [
        "body",
        "question",
    ]


def test_unsupported_file_returns_415(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "virus.exe",
                b"not-a-real-program",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 415
    assert response.json()["detail"] == ("目前只支持 PDF 和 TXT 文件")

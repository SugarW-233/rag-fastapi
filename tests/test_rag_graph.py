import asyncio
from unittest.mock import AsyncMock

import pytest

from app.services import rag


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

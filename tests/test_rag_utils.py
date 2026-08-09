from langchain_core.documents import Document

from app.services.rag import (
    format_context,
    get_page_number,
)


def test_page_number_is_converted_to_one_based() -> None:
    document = Document(
        page_content="测试内容",
        metadata={
            "file_name": "manual.pdf",
            "page": 2,
        },
    )

    assert get_page_number(document) == 3

    context = format_context([document])

    assert "第 3 页" in context
    assert "第 manual.pdf 页" not in context

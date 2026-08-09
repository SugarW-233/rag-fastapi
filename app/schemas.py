from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UploadResponse(BaseModel):
    document_id: str
    file_name: str
    document_count: int
    chunk_count: int
    message: str


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=4000,
        description="用户提出的问题",
    )

    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="最多检索多少个文本块",
    )


class SourceItem(BaseModel):
    index: int
    document_id: str | None = None
    file_name: str
    page: int | None = None
    content_preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class DocumentItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    file_name: str
    file_size: int
    page_count: int | None
    chunk_count: int | None
    status: str
    error_message: str | None
    created_at: datetime
    updated_at: datetime

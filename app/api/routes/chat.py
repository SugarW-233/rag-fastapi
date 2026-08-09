from fastapi import APIRouter, HTTPException

from app.schemas import ChatRequest, ChatResponse
from app.services.rag import answer_question

router = APIRouter(
    prefix="/api/v1/chat",
    tags=["Chat"],
)


@router.post(
    "",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
) -> ChatResponse:
    """单轮RAG问答接口。"""

    try:
        return await answer_question(
            question=request.question,
            top_k=request.top_k,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="RAG问答处理失败",
        ) from exc

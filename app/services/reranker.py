import asyncio
from collections.abc import Sequence
from functools import lru_cache
from threading import Lock

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.core.config import settings

# 避免多个请求同时占用 GPU 进行推理。
_inference_lock = Lock()


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """
    创建并缓存 Reranker 模型。

    第一次调用时加载模型，之后的请求复用同一个模型。
    """

    return CrossEncoder(
        settings.reranker_model,
        device=settings.reranker_device,
        max_length=settings.reranker_max_length,
    )


def rerank_documents(
    question: str,
    documents: Sequence[Document],
    top_k: int,
) -> list[Document]:
    """
    使用 CrossEncoder 对候选文本块重新排序。

    输入应当是混合检索产生的候选集合；
    输出是最终交给相关性判断和 LLM 的 top_k 个文本块。
    """

    if top_k < 1:
        return []

    if not question.strip():
        return []

    if not documents:
        return []

    pairs = [
        [
            question,
            document.page_content,
        ]
        for document in documents
    ]

    # CrossEncoder/PyTorch 推理是同步操作。
    # 使用锁避免多个并发请求同时挤占 GPU。
    with _inference_lock:
        model = get_reranker()

        raw_scores = model.predict(
            pairs,
            batch_size=settings.reranker_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    scores = [float(score) for score in raw_scores]

    if len(scores) != len(documents):
        raise RuntimeError("Reranker 返回的分数数量与候选文档数量不一致")

    # 分数越高越相关。
    # 分数相同时保留混合检索原有顺序。
    ranked_indices = sorted(
        range(len(documents)),
        key=lambda index: (
            -scores[index],
            index,
        ),
    )

    results: list[Document] = []

    for index in ranked_indices[:top_k]:
        document = documents[index]

        # 返回新的 Document，避免修改原候选对象。
        results.append(
            Document(
                page_content=document.page_content,
                metadata={
                    **document.metadata,
                    "reranker_score": scores[index],
                },
            )
        )

    return results


async def arerank_documents(
    question: str,
    documents: Sequence[Document],
    top_k: int,
) -> list[Document]:
    """
    rerank_documents 的异步包装。

    把同步的模型推理放在线程中执行，避免直接阻塞
    FastAPI 和 LangGraph 所在的异步事件循环。
    """

    return await asyncio.to_thread(
        rerank_documents,
        question=question,
        documents=documents,
        top_k=top_k,
    )

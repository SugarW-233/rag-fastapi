import unicodedata
from functools import lru_cache
from threading import RLock

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.services.vector_store import get_vector_store


class BM25Store:
    """
    基于 Chroma 中已有文本块构建的内存 BM25 索引。

    Chroma 是持久化数据源，BM25 只保存在当前进程内存中。

    首次搜索时从 Chroma 构建索引；上传或删除文档后调用
    invalidate()，下一次搜索时自动重新构建。
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._tokenizer = jieba.Tokenizer()

        self._documents: list[Document] = []
        self._document_token_sets: list[set[str]] = []
        self._index: BM25Okapi | None = None

        # 新实例默认没有构建索引。
        self._dirty = True

    def _tokenize(
        self,
        text: str,
    ) -> list[str]:
        """
        对中文、英文和数字进行统一处理。

        NFKC 会把部分全角字符转换成标准形式；
        casefold 用来统一英文字母大小写；
        jieba.lcut_for_search 适合搜索场景，会提供更细粒度的中文词语。
        """

        normalized_text = unicodedata.normalize(
            "NFKC",
            text,
        ).casefold()

        raw_tokens = self._tokenizer.lcut_for_search(
            normalized_text,
        )

        tokens: list[str] = []

        for raw_token in raw_tokens:
            token = raw_token.strip()

            if not token:
                continue

            # 排除空格和纯标点，但保留中文、英文与数字。
            if not any(character.isalnum() for character in token):
                continue

            tokens.append(token)

        return tokens

    def _rebuild_locked(self) -> None:
        """
        从 Chroma 重新读取所有文本块并构建 BM25。

        调用该方法前，必须已经持有 self._lock。
        """

        vector_store = get_vector_store()

        result = vector_store.get(
            include=[
                "documents",
                "metadatas",
            ],
        )

        chunk_ids = result.get("ids") or []
        contents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        if not (len(chunk_ids) == len(contents) == len(metadatas)):
            raise RuntimeError("Chroma 返回的文本块、正文和 metadata 数量不一致")

        new_documents: list[Document] = []
        tokenized_corpus: list[list[str]] = []
        new_token_sets: list[set[str]] = []

        for chunk_id, content, metadata in zip(
            chunk_ids,
            contents,
            metadatas,
        ):
            if not isinstance(content, str):
                continue

            if not content.strip():
                continue

            tokens = self._tokenize(content)

            if not tokens:
                continue

            chunk_metadata = dict(metadata or {})

            # 兼容目前已经存在于 Chroma 中、但 metadata
            # 里还没有 chunk_id 的旧文本块。
            chunk_metadata.setdefault(
                "chunk_id",
                str(chunk_id),
            )

            document = Document(
                page_content=content,
                metadata=chunk_metadata,
            )

            new_documents.append(document)
            tokenized_corpus.append(tokens)
            new_token_sets.append(set(tokens))

        # 先完成所有构建工作，再整体替换旧状态。
        # 如果上面的过程抛出异常，旧状态不会被部分覆盖。
        if tokenized_corpus:
            new_index: BM25Okapi | None = BM25Okapi(tokenized_corpus)
        else:
            new_index = None

        self._documents = new_documents
        self._document_token_sets = new_token_sets
        self._index = new_index
        self._dirty = False

    def rebuild(self) -> None:
        """立即从 Chroma 重新构建 BM25 索引。"""

        with self._lock:
            self._rebuild_locked()

    def invalidate(self) -> None:
        """
        标记当前 BM25 索引已经过期。

        这里只修改状态，不立即读取 Chroma。
        下一次调用 search() 时才会重建。
        """

        with self._lock:
            self._dirty = True

    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[Document]:
        """
        使用 BM25 检索相关文本块。

        返回结果按照 BM25 分数从高到低排序。
        每个返回文档的 metadata 中会增加 bm25_score。
        """

        if top_k < 1:
            return []

        with self._lock:
            if self._dirty:
                self._rebuild_locked()

            if self._index is None:
                return []

            if not self._documents:
                return []

            query_tokens = self._tokenize(query)

            if not query_tokens:
                return []

            query_token_set = set(query_tokens)

            # 先排除完全没有词语重叠的文档。
            # 否则在所有分数都为 0 时，BM25 仍可能按照
            # 原始顺序返回几个实际上不相关的文档。
            matching_indices = [
                index
                for index, document_tokens in enumerate(self._document_token_sets)
                if query_token_set.intersection(document_tokens)
            ]

            if not matching_indices:
                return []

            scores = self._index.get_scores(query_tokens)

            # 所有候选分数都等于 0 时，不返回任意文档。
            has_nonzero_score = any(
                abs(float(scores[index])) > 1e-12 for index in matching_indices
            )

            if not has_nonzero_score:
                return []

            ranked_indices = sorted(
                matching_indices,
                key=lambda index: float(scores[index]),
                reverse=True,
            )

            results: list[Document] = []

            for index in ranked_indices[:top_k]:
                original_document = self._documents[index]
                score = float(scores[index])

                # 返回新 Document，避免修改内存缓存中的原对象。
                result_document = Document(
                    page_content=original_document.page_content,
                    metadata={
                        **original_document.metadata,
                        "bm25_score": score,
                    },
                )

                results.append(result_document)

            return results


@lru_cache(maxsize=1)
def get_bm25_store() -> BM25Store:
    """
    每个应用进程只创建一个 BM25Store。

    这里缓存的是 BM25Store 对象，不是每次搜索的结果。
    """

    return BM25Store()

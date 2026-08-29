import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
import pandas as pd

DEFAULT_DATASET = Path("eval/data/questions.csv")
DEFAULT_RESULTS_DIR = Path("eval/results")

REFUSAL_MARKERS = [
    "根据当前知识库无法确定",
    "知识库中没有检索到相关资料",
    "无法根据提供的资料回答",
]


def parse_should_answer(value: Any) -> bool:
    """把 CSV 中的 yes/no 转换成布尔值。"""

    normalized = str(value).strip().lower()

    if normalized in {"yes", "true", "1"}:
        return True

    if normalized in {"no", "false", "0"}:
        return False

    raise ValueError(f"should_answer 只能是 yes/no, 当前值为：{value}")


def split_keywords(value: Any) -> list[str]:
    """把 PDF|TXT 转换成 ['PDF', 'TXT']。"""

    if pd.isna(value):
        return []

    return [keyword.strip() for keyword in str(value).split("|") if keyword.strip()]


def normalize_optional_text(value: Any) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def is_refusal(answer: str) -> bool:
    return any(marker in answer for marker in REFUSAL_MARKERS)


def validate_dataset(dataset: pd.DataFrame) -> None:
    required_columns = {
        "id",
        "question",
        "should_answer",
        "expected_keywords",
        "expected_source",
    }

    missing_columns = required_columns - set(dataset.columns)

    if missing_columns:
        raise ValueError("评测 CSV 缺少字段：" + ", ".join(sorted(missing_columns)))

    if dataset.empty:
        raise ValueError("评测数据不能为空")

    if dataset["id"].duplicated().any():
        duplicated = dataset.loc[
            dataset["id"].duplicated(),
            "id",
        ].tolist()

        raise ValueError(f"题目 ID 重复: {duplicated}")

    for _, row in dataset.iterrows():
        should_answer = parse_should_answer(row["should_answer"])

        expected_keywords = split_keywords(row["expected_keywords"])

        if should_answer and not expected_keywords:
            raise ValueError(f"{row['id']} 应当回答，但没有设置关键词")


async def evaluate_case(
    client: httpx.AsyncClient,
    row: pd.Series,
    top_k: int | None,
) -> dict[str, Any]:
    question = str(row["question"]).strip()

    should_answer = parse_should_answer(row["should_answer"])

    expected_keywords = split_keywords(row["expected_keywords"])

    expected_source = normalize_optional_text(row["expected_source"])

    request_body: dict[str, Any] = {
        "question": question,
    }

    if top_k is not None:
        request_body["top_k"] = top_k

    started_at = perf_counter()

    try:
        response = await client.post(
            "/api/v1/chat",
            json=request_body,
        )

        response.raise_for_status()

        response_body = response.json()

        answer = str(response_body.get("answer", "")).strip()

        sources = response_body.get("sources", [])

        source_files = sorted(
            {
                str(source.get("file_name", ""))
                for source in sources
                if source.get("file_name")
            }
        )

        refused = is_refusal(answer)

        matched_keywords = [
            keyword
            for keyword in expected_keywords
            if keyword.casefold() in answer.casefold()
        ]

        keyword_ok = len(matched_keywords) == len(expected_keywords)

        if expected_source:
            source_ok = expected_source in source_files
        else:
            source_ok = True

        if should_answer:
            passed = not refused and keyword_ok and source_ok
        else:
            # 不该回答的问题必须拒答，并且不返回引用。
            passed = refused and not sources

        error = ""

    except Exception as exc:
        answer = ""
        source_files = []
        refused = False
        matched_keywords = []
        keyword_ok = False
        source_ok = False
        passed = False
        error = f"{type(exc).__name__}: {exc}"

    latency_ms = round(
        (perf_counter() - started_at) * 1000,
        2,
    )

    return {
        "id": row["id"],
        "question": question,
        "should_answer": should_answer,
        "expected_keywords": "|".join(expected_keywords),
        "matched_keywords": "|".join(matched_keywords),
        "keyword_ok": keyword_ok,
        "expected_source": expected_source,
        "source_files": "|".join(source_files),
        "source_ok": source_ok,
        "refused": refused,
        "passed": passed,
        "latency_ms": latency_ms,
        "answer": answer,
        "error": error,
    }


def percentage(
    values: pd.Series,
) -> float:
    if values.empty:
        return 0.0

    return round(
        float(values.mean() * 100),
        2,
    )


def build_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    answerable = results[results["should_answer"]]

    unanswerable = results[~results["should_answer"]]

    return pd.DataFrame(
        [
            {
                "metric": "total_questions",
                "value": len(results),
            },
            {
                "metric": "overall_pass_rate",
                "value": percentage(results["passed"]),
            },
            {
                "metric": "answerable_pass_rate",
                "value": percentage(answerable["passed"]),
            },
            {
                "metric": "refusal_accuracy",
                "value": percentage(unanswerable["passed"]),
            },
            {
                "metric": "average_latency_ms",
                "value": round(
                    float(results["latency_ms"].mean()),
                    2,
                ),
            },
        ]
    )


async def run_evaluation(
    dataset_path: Path,
    output_path: Path,
    base_url: str,
    top_k: int | None,
) -> None:
    dataset = pd.read_csv(
        dataset_path,
        encoding="utf-8",
    )

    validate_dataset(dataset)

    evaluation_results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=120,
    ) as client:
        for _, row in dataset.iterrows():
            print(f"正在评测 {row['id']}：{row['question']}")

            result = await evaluate_case(
                client=client,
                row=row,
                top_k=top_k,
            )

            evaluation_results.append(result)

    results = pd.DataFrame(evaluation_results)

    summary = build_summary(results)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")

    # utf-8-sig 方便使用 Excel 打开中文 CSV。
    results.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n逐题结果：")
    print(
        results[
            [
                "id",
                "should_answer",
                "keyword_ok",
                "source_ok",
                "refused",
                "passed",
                "latency_ms",
            ]
        ].to_string(index=False)
    )

    print("\n汇总结果：")
    print(summary.to_string(index=False))

    print(f"\n明细已保存：{output_path}")
    print(f"汇总已保存：{summary_path}")


def parse_arguments() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    default_output = DEFAULT_RESULTS_DIR / f"evaluation_{timestamp}.csv"

    parser = argparse.ArgumentParser(description="运行 RAG 小型评测集")

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="评测 CSV 路径",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="评测结果输出路径",
    )

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="RAG FastAPI 服务地址",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="覆盖默认接口的 top_k",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    asyncio.run(
        run_evaluation(
            dataset_path=arguments.dataset,
            output_path=arguments.output,
            base_url=arguments.base_url,
            top_k=arguments.top_k,
        )
    )


if __name__ == "__main__":
    main()

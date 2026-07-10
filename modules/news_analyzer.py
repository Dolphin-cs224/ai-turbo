from __future__ import annotations

import json
import os
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


DEFAULT_ANALYSIS_RESULT = {
    "theme_news_score": 50,
    "news_sentiment": "중립",
    "summary": "AI 뉴스 분석을 실행할 수 없어 기본값으로 표시합니다.",
    "article_based_count": 0,
    "title_only_count": 0,
    "analysis_basis": "분석 불가",
    "ai_enabled": False,
    "error": None,
}


def _normalize_news_items(news_data: Any, max_items: int = 30) -> list[dict]:
    """
    Streamlit에서 전달되는 뉴스 데이터를 AI 분석용 리스트로 정리한다.
    pandas DataFrame, list[dict] 둘 다 처리할 수 있게 만든다.

    article_text가 있으면 본문 기반 분석 자료로 사용하고,
    article_text가 없으면 제목 기반 보조 자료로 사용한다.
    """

    if news_data is None:
        return []

    if hasattr(news_data, "to_dict"):
        rows = news_data.to_dict("records")
    elif isinstance(news_data, list):
        rows = news_data
    else:
        return []

    normalized = []

    for row in rows[:max_items]:
        if not isinstance(row, dict):
            continue

        title = (
            row.get("title")
            or row.get("headline")
            or row.get("news_title")
            or ""
        )

        if not title:
            continue

        article_text = str(row.get("article_text", "") or "").strip()
        article_readable = bool(row.get("article_readable", False)) and bool(article_text)

        normalized.append(
            {
                "title": str(title),
                "region": str(row.get("region", "미분류")),
                "source": str(row.get("source", "출처 미상")),
                "published": str(
                    row.get("published")
                    or row.get("published_at")
                    or row.get("pub_date")
                    or row.get("date")
                    or ""
                ),
                "link": str(row.get("link", "")),
                "article_url": str(row.get("article_url", row.get("link", ""))),
                "article_domain": str(row.get("article_domain", "")),
                "article_readable": article_readable,
                "article_text": article_text[:4000],
                "article_error": str(row.get("article_error", "") or ""),
            }
        )

    return normalized


def _safe_json_loads(text: str) -> dict:
    """
    AI 응답에서 JSON 부분만 안전하게 추출한다.
    """

    if not text:
        return {}

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and start < end:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}

    return {}


def _build_news_text(news_items: list[dict]) -> tuple[str, int, int]:
    """
    AI에게 전달할 뉴스 텍스트를 만든다.
    본문이 있는 기사는 본문까지 포함하고,
    본문이 없는 기사는 제목만 보조 근거로 표시한다.
    """

    blocks = []
    article_based_count = 0
    title_only_count = 0

    for idx, item in enumerate(news_items, start=1):
        if item["article_readable"]:
            article_based_count += 1
            body = item["article_text"][:4000]

            blocks.append(
                f"""
[뉴스 {idx} / 본문 기반]
권역: {item['region']}
출처: {item['source']}
발행일: {item['published']}
제목: {item['title']}
본문:
{body}
"""
            )
        else:
            title_only_count += 1

            blocks.append(
                f"""
[뉴스 {idx} / 제목 기반 보조 자료]
권역: {item['region']}
출처: {item['source']}
발행일: {item['published']}
제목: {item['title']}
본문 추출 실패 사유: {item['article_error'] or '본문 없음'}
"""
            )

    return "\n".join(blocks), article_based_count, title_only_count


def analyze_theme_news(
    theme_name: str,
    news_data: Any,
    model: str | None = None,
) -> dict:
    """
    수집된 테마 뉴스를 AI로 분석한다.

    반환값:
    - theme_news_score: 0~100
    - news_sentiment: 긍정 / 중립 / 부정
    - summary
    - positive_points
    - negative_points
    - key_regions
    - article_based_count
    - title_only_count
    - analysis_basis
    """

    if load_dotenv is not None:
        load_dotenv(".env", override=True)

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = model or os.getenv("OPENAI_MODEL", "claude-sonnet-4-6")

    if not api_key:
        result = DEFAULT_ANALYSIS_RESULT.copy()
        result["error"] = "OPENAI_API_KEY가 설정되어 있지 않습니다."
        return result

    if OpenAI is None:
        result = DEFAULT_ANALYSIS_RESULT.copy()
        result["error"] = "openai 패키지가 설치되어 있지 않습니다."
        return result

    news_items = _normalize_news_items(news_data)

    if not news_items:
        result = DEFAULT_ANALYSIS_RESULT.copy()
        result["summary"] = "분석할 뉴스가 없습니다."
        result["error"] = "뉴스 데이터가 비어 있습니다."
        return result

    news_text, article_based_count, title_only_count = _build_news_text(news_items)

    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)

    system_prompt = """
너는 투자대회용 AI 투자 분석 보조자다.
너의 역할은 특정 테마와 관련된 뉴스의 분위기를 보수적으로 분석하는 것이다.

중요 원칙:
- 실제 투자 추천을 하지 않는다.
- 기사 본문이 있는 경우 본문 내용을 주요 근거로 사용한다.
- 기사 본문이 없는 경우 제목만 보조 근거로 사용한다.
- 제목만 있는 뉴스는 점수에 강하게 반영하지 않는다.
- 단순 기대감, 전망, 의견성 기사는 높은 점수를 주지 않는다.
- 실제 수주, 실적 개선, 정부 정책, 대규모 투자, 공급 계약, 생산 확대, 수요 증가가 본문에서 확인될 때만 높은 점수를 줄 수 있다.
- 부정 뉴스, 규제, 공급 과잉, 실적 악화, 경영진 이탈, 기술 실패, 보안 이슈는 감점한다.
- summary는 반드시 긍정 요인과 부정 요인을 모두 포함한 중립적 시각으로 작성한다.
- summary는 과도하게 낙관적이거나 비관적으로 쓰지 않는다.
- summary에는 본문 기반으로 확인된 내용과 아직 확인하기 어려운 한계를 함께 적는다.
- 결과는 반드시 JSON 형식으로만 출력한다.
"""

    user_prompt = f"""
분석 대상 테마: {theme_name}

아래 뉴스들을 바탕으로 테마 뉴스 분위기를 분석해줘.

뉴스 자료:
{news_text}

본문 기반 기사 수: {article_based_count}
제목 기반 보조 기사 수: {title_only_count}

점수 산정 기준:
- 80~100: 본문에서 강한 수요 증가, 대규모 투자, 수주, 실적 개선, 정책 지원이 명확히 확인됨
- 60~79: 긍정 흐름이 우세하지만 일부는 기대감 또는 제목 기반 자료에 의존함
- 40~59: 긍정과 부정이 혼재하거나 근거가 제한적임
- 20~39: 부정 이슈가 우세하거나 성장 기대가 약함
- 0~19: 강한 부정 뉴스가 다수 확인됨

주의:
- 본문 기반 기사 수가 적으면 점수를 보수적으로 산정해라.
- 제목 기반 기사만으로는 75점 이상을 주지 마라.
- 기사 내용이 테마와 직접 관련이 약하면 점수를 낮춰라.
- 요약에는 뉴스 본문을 읽은 근거와 한계를 함께 써라.
- summary에는 긍정적 해석과 부정적 또는 제한적 해석을 모두 포함해라.
- summary는 '긍정적으로 볼 점은 ~이나, 제한점은 ~이다'와 같은 균형 잡힌 문장으로 작성해라.

아래 JSON 형식으로만 답해줘.

{{
  "theme_news_score": 0부터 100 사이의 정수,
  "news_sentiment": "긍정" 또는 "중립" 또는 "부정",
  "summary": ""긍정 요인과 부정 요인을 모두 포함한 중립적 뉴스 분위기 요약 4~5문장",
  "article_based_count": {article_based_count},
  "title_only_count": {title_only_count},
  "analysis_basis": "본문 기반 중심" 또는 "본문+제목 혼합" 또는 "제목 기반 제한적 분석"
}}
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = response.choices[0].message.content
        parsed = _safe_json_loads(content)

        result = DEFAULT_ANALYSIS_RESULT.copy()

        result["theme_news_score"] = int(parsed.get("theme_news_score", 50))
        result["theme_news_score"] = max(0, min(100, result["theme_news_score"]))

        result["news_sentiment"] = parsed.get("news_sentiment", "중립")
        result["summary"] = parsed.get("summary", "")
        result["article_based_count"] = int(parsed.get("article_based_count", article_based_count))
        result["title_only_count"] = int(parsed.get("title_only_count", title_only_count))
        result["analysis_basis"] = parsed.get("analysis_basis", "본문+제목 혼합")
        result["ai_enabled"] = True
        result["error"] = None

        return result

    except Exception as e:
        result = DEFAULT_ANALYSIS_RESULT.copy()
        result["article_based_count"] = article_based_count
        result["title_only_count"] = title_only_count
        result["analysis_basis"] = "분석 실패"
        result["error"] = str(e)
        return result
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
    "positive_points": [],
    "negative_points": [],
    "key_regions": [],
    "ai_enabled": False,
    "error": None,
}


def _normalize_news_items(news_data: Any, max_items: int = 30) -> list[dict]:
    """
    Streamlit에서 전달되는 뉴스 데이터를 AI 분석용 리스트로 정리한다.
    pandas DataFrame, list[dict] 둘 다 처리할 수 있게 만든다.
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

        normalized.append(
            {
                "title": str(title),
                "region": str(row.get("region", "미분류")),
                "source": str(row.get("source", "출처 미상")),
                "published": str(
                    row.get("published")
                    or row.get("published_at")
                    or row.get("date")
                    or ""
                ),
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
    """

    if load_dotenv is not None:
        load_dotenv()

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
    
    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)

    news_text = "\n".join(
        [
            f"- [{item['region']}] {item['title']} / 출처: {item['source']} / 발행일: {item['published']}"
            for item in news_items
        ]
    )

    system_prompt = """
너는 투자대회용 AI 투자 분석 보조자다.
뉴스 제목만 보고 특정 테마의 분위기를 분석한다.

주의사항:
- 실제 투자 추천을 하지 않는다.
- 과장하지 않는다.
- 뉴스 제목 기반의 제한적인 분석임을 전제로 판단한다.
- 결과는 반드시 JSON 형식으로만 출력한다.
"""

    user_prompt = f"""
분석 대상 테마: {theme_name}

아래 뉴스 제목들을 바탕으로 테마 뉴스 분위기를 분석해줘.

뉴스 목록:
{news_text}

아래 JSON 형식으로만 답해줘.

{{
  "theme_news_score": 0부터 100 사이의 정수,
  "news_sentiment": "긍정" 또는 "중립" 또는 "부정",
  "summary": "뉴스 분위기 요약 2~3문장",
  "positive_points": ["긍정 요인 1", "긍정 요인 2"],
  "negative_points": ["부정 요인 1", "부정 요인 2"],
  "key_regions": ["국내", "미국", "일본", "중동"]
}}

점수 기준:
- 80~100: 강한 긍정
- 60~79: 긍정 우위
- 40~59: 중립
- 20~39: 부정 우위
- 0~19: 강한 부정
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
        result["positive_points"] = parsed.get("positive_points", [])
        result["negative_points"] = parsed.get("negative_points", [])
        result["key_regions"] = parsed.get("key_regions", [])
        result["ai_enabled"] = True
        result["error"] = None

        return result

    except Exception as e:
        result = DEFAULT_ANALYSIS_RESULT.copy()
        result["error"] = str(e)
        return result
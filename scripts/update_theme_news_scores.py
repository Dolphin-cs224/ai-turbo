from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
import time

import pandas as pd

from modules.news_collector import collect_theme_news
from modules.news_reader import enrich_news_with_article_text
from modules.news_analyzer import analyze_theme_news


WATCHLIST_PATH = "data/watchlist.csv"
SCORE_OUTPUT_PATH = "data/theme_news_scores.csv"
NEWS_OUTPUT_PATH = "data/theme_news_items.csv"

REGIONS = ["국내", "미국", "일본", "중동"]

MAX_NEWS_PER_KEYWORD = 5
MAX_ARTICLES_PER_REGION = 4
MAX_CHARS_PER_ARTICLE = 2500

SLEEP_SECONDS = 2


def load_themes() -> list[str]:
    df = pd.read_csv(WATCHLIST_PATH, dtype={"code": str})
    themes = sorted(df["theme"].dropna().unique())
    return themes


def prepare_news_items_for_reading(news_items: list[dict]) -> pd.DataFrame:
    news_df = pd.DataFrame(news_items)

    if news_df.empty:
        return news_df

    # 같은 링크 또는 같은 제목의 중복 뉴스 제거
    if "link" in news_df.columns:
        news_df = news_df.drop_duplicates(subset=["link"], keep="first")

    if "title" in news_df.columns:
        news_df = news_df.drop_duplicates(subset=["title"], keep="first")

    # 발행일 기준 최신순 정렬
    if "pub_date" in news_df.columns:
        news_df["_published_at"] = pd.to_datetime(
            news_df["pub_date"],
            errors="coerce",
            utc=True,
        )

        if "region" in news_df.columns:
            news_df = news_df.sort_values(
                by=["region", "_published_at"],
                ascending=[True, False],
                na_position="last",
            )
        else:
            news_df = news_df.sort_values(
                by=["_published_at"],
                ascending=False,
                na_position="last",
            )

        news_df = news_df.drop(columns=["_published_at"])

    return news_df.reset_index(drop=True)


def update_one_theme(theme: str) -> dict:
    print("=" * 80)
    print(f"테마 분석 시작: {theme}")

    news_items = collect_theme_news(
        theme,
        max_items_per_keyword=MAX_NEWS_PER_KEYWORD,
        regions=REGIONS,
    )

    enriched_news = []

    news_df = prepare_news_items_for_reading(news_items)

    if not news_df.empty and "region" in news_df.columns:
        for region in REGIONS:
            region_news = news_df[news_df["region"].eq(region)].to_dict("records")

            if not region_news:
                continue

            region_enriched_news = enrich_news_with_article_text(
                region_news,
                max_articles=MAX_ARTICLES_PER_REGION,
                max_chars_per_article=MAX_CHARS_PER_ARTICLE,
            )

            enriched_news.extend(region_enriched_news)
    else:
        enriched_news = enrich_news_with_article_text(
            news_items,
            max_articles=MAX_ARTICLES_PER_REGION,
            max_chars_per_article=MAX_CHARS_PER_ARTICLE,
        )

    analysis = analyze_theme_news(
        theme,
        enriched_news,
    )

    article_based_count = analysis.get("article_based_count", 0)
    title_only_count = analysis.get("title_only_count", 0)

    print(f"뉴스 수집 수: {len(news_items)}")
    print(f"본문 기반 기사 수: {article_based_count}")
    print(f"제목 기반 기사 수: {title_only_count}")
    print(f"뉴스 점수: {analysis.get('theme_news_score', 50)}")
    print(f"분위기: {analysis.get('news_sentiment', '중립')}")

    analysis_result = {
        "updated_at": datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S"),
        "theme": theme,
        "theme_news_score": analysis.get("theme_news_score", 50),
        "news_sentiment": analysis.get("news_sentiment", "중립"),
        "summary": analysis.get("summary", ""),
        "positive_points": " | ".join(analysis.get("positive_points", [])),
        "negative_points": " | ".join(analysis.get("negative_points", [])),
        "key_regions": " | ".join(analysis.get("key_regions", [])),
        "article_based_count": article_based_count,
        "title_only_count": title_only_count,
        "analysis_basis": analysis.get("analysis_basis", "분석 기준 없음"),
        "ai_enabled": analysis.get("ai_enabled", False),
        "error": analysis.get("error", None),
        "news_count": len(news_items),
    }

    return analysis_result, enriched_news


def main() -> None:
    themes = load_themes()

    print(f"전체 테마 수: {len(themes)}")

    results = []
    all_news_items = []

    for idx, theme in enumerate(themes, start=1):
        print(f"[{idx}/{len(themes)}] {theme}")

        try:
            result = update_one_theme(theme)
            results.append(result)

            for item in enriched_news:
                item["updated_at"] = result["updated_at"]
                item["theme"] = theme
                all_news_items.append(item)

        except Exception as e:
            print(f"테마 분석 실패: {theme} / {e}")

            results.append(
                {
                    "updated_at": datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S"),
                    "theme": theme,
                    "theme_news_score": 50,
                    "news_sentiment": "중립",
                    "summary": "분석 실패로 기본값을 사용했습니다.",
                    "positive_points": "",
                    "negative_points": "",
                    "key_regions": "",
                    "article_based_count": 0,
                    "title_only_count": 0,
                    "analysis_basis": "분석 실패",
                    "ai_enabled": False,
                    "error": str(e),
                    "news_count": 0,
                }
            )

        time.sleep(SLEEP_SECONDS)

    result_df = pd.DataFrame(results)
    result_df.to_csv(SCORE_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    news_df = pd.DataFrame(all_news_items)
    news_df.to_csv(NEWS_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(f"뉴스 점수 저장 완료: {SCORE_OUTPUT_PATH}")
    print(f"뉴스 목록 저장 완료: {NEWS_OUTPUT_PATH}")
    print(result_df[["theme", "theme_news_score", "news_sentiment", "article_based_count", "title_only_count"]])
if __name__ == "__main__":
    main()
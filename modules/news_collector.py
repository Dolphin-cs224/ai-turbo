# modules/news_collector.py

import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

from modules.news_keywords import get_theme_keywords


def fetch_news_by_keyword(keyword: str, max_items: int = 5) -> list[dict]:
    """
    Google News RSS에서 키워드 기반 뉴스를 수집한다.
    API 키 없이 뉴스 제목, 링크, 발행일을 가져온다.
    """

    encoded_keyword = quote_plus(keyword)

    url = (
        "https://news.google.com/rss/search?"
        f"q={encoded_keyword}"
        "&hl=ko"
        "&gl=KR"
        "&ceid=KR:ko"
    )

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    root = ET.fromstring(response.content)

    news_items = []

    for item in root.findall(".//item")[:max_items]:
        title = item.findtext("title", default="")
        link = item.findtext("link", default="")
        pub_date = item.findtext("pubDate", default="")
        source = item.findtext("source", default="")

        news_items.append(
            {
                "keyword": keyword,
                "title": title,
                "link": link,
                "pub_date": pub_date,
                "source": source,
            }
        )

    return news_items


def collect_theme_news(theme: str, max_items_per_keyword: int = 3) -> list[dict]:
    """
    테마명을 입력하면 해당 테마의 키워드 목록으로 뉴스를 수집한다.
    중복 링크는 제거한다.
    """

    keywords = get_theme_keywords(theme)
    collected_news = []
    seen_links = set()

    for keyword in keywords:
        try:
            news_items = fetch_news_by_keyword(
                keyword=keyword,
                max_items=max_items_per_keyword
            )

            for item in news_items:
                if item["link"] in seen_links:
                    continue

                seen_links.add(item["link"])
                item["theme"] = theme
                collected_news.append(item)

        except Exception as e:
            collected_news.append(
                {
                    "theme": theme,
                    "keyword": keyword,
                    "title": f"뉴스 수집 실패: {e}",
                    "link": "",
                    "pub_date": "",
                    "source": "",
                }
            )

    return collected_news
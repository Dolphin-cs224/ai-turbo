# modules/news_collector.py

import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

from modules.news_keywords import get_theme_keywords


REGION_SETTINGS = {
    "국내": {
        "hl": "ko",
        "gl": "KR",
        "ceid": "KR:ko",
        "suffix": "",
    },
    "미국": {
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
        "suffix": "stock market earnings investment",
    },
    "일본": {
        "hl": "ja",
        "gl": "JP",
        "ceid": "JP:ja",
        "suffix": "日本 株式市場 企業",
    },
    "중동": {
        "hl": "en",
        "gl": "US",
        "ceid": "US:en",
        "suffix": "Middle East Gulf Saudi UAE Qatar oil LNG defense energy investment",
    },
}


REGIONAL_KEYWORD_TRANSLATIONS = {
    "미국": {
        "HBM": "HBM memory",
        "AI 반도체": "AI semiconductor",
        "메모리 반도체": "memory chip semiconductor",
        "반도체 수출": "semiconductor exports",
        "엔비디아": "Nvidia",
        "전기차": "electric vehicle EV",
        "하이브리드": "hybrid vehicle",
        "리튬": "lithium battery",
        "양극재": "cathode material battery",
        "국제유가": "oil prices",
        "LNG선": "LNG carrier",
        "선박 수주": "shipbuilding orders",
        "방산 수출": "defense exports",
        "무기 수주": "defense contract arms deal",
        "전력기기": "power equipment",
        "변압기": "transformer power grid",
        "AI 전력": "AI power demand data center",
    },
    "일본": {
        "HBM": "HBM",
        "AI 반도체": "AI半導体",
        "메모리 반도체": "メモリ半導体",
        "반도체 수출": "半導体 輸出",
        "엔비디아": "NVIDIA",
        "전기차": "電気自動車 EV",
        "하이브리드": "ハイブリッド車",
        "리튬": "リチウム 電池",
        "양극재": "正極材 電池",
        "국제유가": "原油価格",
        "LNG선": "LNG船",
        "선박 수주": "造船 受注",
        "방산 수출": "防衛 輸出",
        "무기 수주": "防衛 受注",
        "전력기기": "電力機器",
        "변압기": "変圧器 電力網",
        "AI 전력": "AI 電力需要 データセンター",
    },
    "중동": {
        "HBM": "HBM AI data center",
        "AI 반도체": "AI semiconductor data center",
        "메모리 반도체": "memory chip AI server",
        "반도체 수출": "semiconductor exports Gulf",
        "엔비디아": "Nvidia Middle East AI",
        "국제유가": "oil prices Middle East",
        "정유": "refinery Middle East",
        "석유화학": "petrochemical Middle East",
        "LNG선": "LNG carrier Qatar UAE",
        "선박 수주": "shipbuilding Middle East LNG",
        "방산 수출": "defense exports Middle East",
        "무기 수주": "arms deal Middle East Saudi UAE",
        "전력기기": "power equipment Middle East",
        "변압기": "transformer power grid Middle East",
        "AI 전력": "AI data center power Middle East",
        "원전": "nuclear power Saudi UAE",
        "SMR": "SMR nuclear Middle East",
    },
}


def localize_keyword(keyword: str, region: str) -> str:
    """
    권역에 맞게 검색 키워드를 변환한다.
    등록되지 않은 키워드는 원래 키워드를 그대로 사용한다.
    """

    return REGIONAL_KEYWORD_TRANSLATIONS.get(region, {}).get(keyword, keyword)


def fetch_news_by_keyword(
    keyword: str,
    max_items: int = 5,
    region: str = "국내",
) -> list[dict]:
    """
    Google News RSS에서 키워드와 권역을 기준으로 뉴스를 수집한다.
    """

    region_config = REGION_SETTINGS.get(region, REGION_SETTINGS["국내"])
    localized_keyword = localize_keyword(keyword, region)

    search_query = f"{localized_keyword} {region_config['suffix']}".strip()
    encoded_query = quote_plus(search_query)

    url = (
        "https://news.google.com/rss/search?"
        f"q={encoded_query}"
        f"&hl={region_config['hl']}"
        f"&gl={region_config['gl']}"
        f"&ceid={region_config['ceid']}"
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
                "region": region,
                "keyword": keyword,
                "localized_keyword": localized_keyword,
                "search_query": search_query,
                "title": title,
                "link": link,
                "pub_date": pub_date,
                "source": source,
            }
        )

    return news_items


def collect_theme_news(
    theme: str,
    max_items_per_keyword: int = 3,
    regions: list[str] | None = None,
) -> list[dict]:
    """
    테마명을 입력하면 해당 테마의 키워드 목록으로 국내/해외 뉴스를 수집한다.
    중복 링크는 제거한다.
    """

    if regions is None:
        regions = ["국내"]

    keywords = get_theme_keywords(theme)
    collected_news = []
    seen_links = set()

    for region in regions:
        for keyword in keywords:
            try:
                news_items = fetch_news_by_keyword(
                    keyword=keyword,
                    max_items=max_items_per_keyword,
                    region=region,
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
                        "region": region,
                        "keyword": keyword,
                        "localized_keyword": localize_keyword(keyword, region),
                        "search_query": keyword,
                        "title": f"뉴스 수집 실패: {e}",
                        "link": "",
                        "pub_date": "",
                        "source": "",
                    }
                )

    return collected_news
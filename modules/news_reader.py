from __future__ import annotations

import html as html_lib
import json
import re
from typing import Any
from urllib.parse import unquote, urlparse

import requests
from bs4 import BeautifulSoup

try:
    import trafilatura
except ImportError:
    trafilatura = None

try:
    from googlenewsdecoder import gnewsdecoder
except ImportError:
    gnewsdecoder = None


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _clean_text(text: str) -> str:
    """
    기사 본문에서 불필요한 공백을 정리한다.
    """

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u200b", "")
    return text.strip()


def _is_probably_noise(text: str) -> bool:
    """
    기사 본문이 아닌 메뉴, 저작권, 광고성 문구를 대략적으로 제외한다.
    """

    if not text:
        return True

    noise_keywords = [
        "무단전재",
        "재배포 금지",
        "Copyright",
        "All rights reserved",
        "구독",
        "로그인",
        "회원가입",
        "광고",
        "기사제보",
        "영상",
        "댓글",
        "공유하기",
    ]

    if len(text) < 30:
        return True

    return any(keyword in text for keyword in noise_keywords)


def _is_google_news_url(url: str) -> bool:
    """
    Google News 중간 링크인지 확인한다.
    """

    if not url:
        return False

    domain = urlparse(url).netloc.lower()
    return "news.google.com" in domain


def _is_valid_article_candidate(url: str) -> bool:
    """
    Google News HTML 안에서 찾은 URL 중 실제 기사 후보만 남긴다.
    """

    if not url.startswith("http"):
        return False

    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    blocked_domains = [
         "google.com",
        "news.google.com",
        "accounts.google.com",
        "support.google.com",
        "gstatic.com",
        "fonts.gstatic.com",
        "fonts.googleapis.com",
        "googleapis.com",
        "googleusercontent.com",
        "schema.org",
        "youtube.com",
        "youtu.be",
    ]

    if any(blocked in domain for blocked in blocked_domains):
        return False

    blocked_extensions = [
        ".css",
        ".js",
        ".mjs",
        ".json",
        ".xml",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".webp",
        ".ico",
        ".woff",
        ".woff2",
        ".ttf",
    ]

    path = parsed.path.lower()

    if any(path.endswith(ext) for ext in blocked_extensions):
        return False

    return True


def resolve_google_news_url(url: str, timeout: int = 10) -> dict:
    """
    Google News RSS 중간 링크에서 실제 언론사 기사 URL을 추출한다.

    성공 시:
    {
        "resolved_url": 실제 기사 URL,
        "resolved": True,
        "resolve_error": None
    }
    """

    result = {
        "resolved_url": url or "",
        "resolved": False,
        "resolve_error": None,
    }

    if not url:
        result["resolve_error"] = "URL이 비어 있습니다."
        return result

    if not _is_google_news_url(url):
        result["resolved_url"] = url
        result["resolved"] = True
        return result

    if gnewsdecoder is None:
        result["resolve_error"] = "googlenewsdecoder 패키지가 설치되어 있지 않습니다."
        return result

    try:
        decoded = gnewsdecoder(url, interval=1)

        if isinstance(decoded, dict) and decoded.get("status"):
            decoded_url = decoded.get("decoded_url", "")

            if decoded_url and _is_valid_article_candidate(decoded_url):
                result["resolved_url"] = decoded_url
                result["resolved"] = True
                return result

            result["resolve_error"] = "디코딩된 URL이 기사 URL 후보로 적절하지 않습니다."
            return result

        result["resolve_error"] = str(decoded.get("message", "Google News URL 디코딩에 실패했습니다."))
        return result

    except Exception as e:
        result["resolve_error"] = str(e)
        return result

def _collect_paragraph_texts(container) -> list[str]:
    """
    특정 HTML 영역 안에서 문단 텍스트를 수집한다.
    """

    paragraph_texts = []

    if container is None:
        return paragraph_texts

    for paragraph in container.find_all(["p", "div", "span"]):
        text = _clean_text(paragraph.get_text(" ", strip=True))

        if _is_probably_noise(text):
            continue

        if text in paragraph_texts:
            continue

        paragraph_texts.append(text)

    return paragraph_texts


def _extract_article_text_from_soup(soup: BeautifulSoup) -> str:
    """
    여러 방식으로 기사 본문을 추출한다.
    """

    # 1. JSON-LD 안의 articleBody 우선 탐색
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            raw = script.string or script.get_text()
            if not raw:
                continue

            data = json.loads(raw)

            candidates = data if isinstance(data, list) else [data]

            for item in candidates:
                if not isinstance(item, dict):
                    continue

                article_body = item.get("articleBody")
                description = item.get("description")

                if article_body and len(article_body) > 100:
                    return _clean_text(article_body)

                if description and len(description) > 100:
                    return _clean_text(description)

        except Exception:
            continue

    # 2. 대표적인 기사 본문 선택자 탐색
    selectors = [
        "article",
        "[itemprop='articleBody']",
        "[class*='article-body']",
        "[class*='articleBody']",
        "[class*='article_body']",
        "[class*='news_body']",
        "[class*='news-body']",
        "[class*='content-body']",
        "[class*='story-body']",
        "[class*='storyBody']",
        "[class*='view_content']",
        "[class*='article-content']",
        "[class*='article_content']",
        "[id*='article']",
        "[id*='content']",
    ]

    for selector in selectors:
        container = soup.select_one(selector)
        texts = _collect_paragraph_texts(container)
        article_text = _clean_text(" ".join(texts))

        if len(article_text) > 300:
            return article_text

    # 3. 마지막 fallback: 전체 p 태그
    paragraph_texts = []

    for paragraph in soup.find_all("p"):
        text = _clean_text(paragraph.get_text(" ", strip=True))

        if _is_probably_noise(text):
            continue

        if text in paragraph_texts:
            continue

        paragraph_texts.append(text)

    return _clean_text(" ".join(paragraph_texts))


def _extract_article_text_with_trafilatura(html_text: str, url: str = "") -> str:
    """
    trafilatura를 사용해 기사 본문을 추출한다.
    BeautifulSoup 방식이 실패했을 때 fallback으로 사용한다.
    """

    if trafilatura is None or not html_text:
        return ""

    try:
        extracted = trafilatura.extract(
            html_text,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )

        return _clean_text(extracted or "")

    except Exception:
        return ""


def fetch_article_text(url: str, max_chars: int = 6000, timeout: int = 10) -> dict:
    """
    뉴스 링크에서 기사 본문을 추출한다.

    반환값:
    - article_text: 추출된 본문
    - article_url: 최종 접속 URL
    - article_domain: 도메인
    - article_readable: 본문 추출 성공 여부
    - article_error: 오류 메시지
    """

    result = {
        "article_text": "",
        "article_url": url or "",
        "article_domain": "",
        "article_readable": False,
        "article_error": None,
        "google_news_resolved": False,
        "google_news_resolve_error": None,
    }

    if not url:
        result["article_error"] = "URL이 비어 있습니다."
        return result

    try:
        resolve_result = resolve_google_news_url(url, timeout=timeout)

        target_url = resolve_result.get("resolved_url", url)
        result["google_news_resolved"] = bool(resolve_result.get("resolved"))
        result["google_news_resolve_error"] = resolve_result.get("resolve_error")

        response = requests.get(
            target_url,
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            allow_redirects=True,
        )
        response.raise_for_status()

        final_url = response.url
        result["article_url"] = final_url
        result["article_domain"] = urlparse(final_url).netloc

        html_text = response.text

        soup = BeautifulSoup(html_text, "html.parser")

        article_text = _extract_article_text_from_soup(soup)

        if not article_text:
            article_text = _extract_article_text_with_trafilatura(
                html_text,
                url=final_url,
            )

        for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
            tag.decompose()

        if not article_text:
            article_text = _extract_article_text_from_soup(soup)

        if not article_text:
            article_text = _extract_article_text_with_trafilatura(
                str(soup),
                url=final_url,
            )

        if not article_text:
            if result["google_news_resolve_error"]:
                result["article_error"] = result["google_news_resolve_error"]
            else:
                result["article_error"] = "본문을 추출하지 못했습니다."
            return result

        result["article_text"] = article_text[:max_chars]
        result["article_readable"] = True

        return result

    except Exception as e:
        result["article_error"] = str(e)
        return result


def enrich_news_with_article_text(
    news_data: Any,
    max_articles: int = 10,
    max_chars_per_article: int = 6000,
) -> list[dict]:
    """
    수집된 뉴스 목록에 기사 본문을 추가한다.

    pandas DataFrame 또는 list[dict] 둘 다 입력 가능.
    """

    if news_data is None:
        return []

    if hasattr(news_data, "to_dict"):
        rows = news_data.to_dict("records")
    elif isinstance(news_data, list):
        rows = news_data
    else:
        return []

    enriched_news = []

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue

        item = row.copy()

        if idx < max_articles:
            article_result = fetch_article_text(
                item.get("link", ""),
                max_chars=max_chars_per_article,
            )

            item.update(article_result)
        else:
            item.update(
                {
                    "article_text": "",
                    "article_url": item.get("link", ""),
                    "article_domain": "",
                    "article_readable": False,
                    "article_error": "본문 수집 제한 개수를 초과했습니다.",
                    "google_news_resolved": False,
                    "google_news_resolve_error": None,
                }
            )

        enriched_news.append(item)

    return enriched_news
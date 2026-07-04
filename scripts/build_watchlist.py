import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


# -----------------------------
# 설정
# -----------------------------
KOSPI_TOP_PERCENT = 0.20
KOSDAQ_TOP_PERCENT = 0.01

OUTPUT_PATH = Path("data/watchlist.csv")

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def get_naver_market_sum(sosok: int, market_name: str, max_pages: int = 60):
    """
    네이버 금융 시가총액 순위에서 종목명, 종목코드, 시가총액을 가져온다.

    sosok=0 : KOSPI
    sosok=1 : KOSDAQ
    """
    result = []

    for page in range(1, max_pages + 1):
        url = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}"

        response = requests.get(url, headers=HEADERS)
        response.encoding = "euc-kr"

        # 1. BeautifulSoup으로 종목명 + 종목코드 추출
        soup = BeautifulSoup(response.text, "html.parser")

        code_rows = []

        for link in soup.select("a.tltle"):
            name = link.get_text(strip=True)
            href = link.get("href", "")

            match = re.search(r"code=(\d+)", href)

            if match:
                code_rows.append({
                    "name": name,
                    "code": match.group(1)
                })

        if not code_rows:
            break

        code_df = pd.DataFrame(code_rows)

        # 2. pandas.read_html로 시가총액 표 추출
        tables = pd.read_html(response.text)

        market_tables = [
            table for table in tables
            if "종목명" in table.columns and "시가총액" in table.columns
        ]

        if not market_tables:
            break

        table_df = market_tables[0]
        table_df = table_df[["종목명", "시가총액"]].dropna()
        table_df = table_df.rename(columns={
            "종목명": "name",
            "시가총액": "market_cap"
        })

        # 3. 종목명 기준으로 코드와 시가총액 합치기
        page_df = pd.merge(code_df, table_df, on="name", how="inner")
        page_df["market"] = market_name

        if page_df.empty:
            break

        result.append(page_df)

    if not result:
        raise ValueError(f"{market_name} 데이터를 가져오지 못했습니다.")

    df = pd.concat(result, ignore_index=True)
    df = df.drop_duplicates(subset=["code"]).reset_index(drop=True)

    return df

def remove_preferred_stocks(df: pd.DataFrame):
    """
    우선주 제외
    예: 삼성전자우, 현대차2우B 등
    """
    def is_preferred(name):
        return (
            name.endswith("우")
            or "우B" in name
            or "2우" in name
            or "3우" in name
        )

    return df[~df["name"].apply(is_preferred)].copy()


def classify_theme(name: str):
    """
    기초 테마 분류
    나중에 AI 분류로 고도화 가능
    """
    semiconductor_keywords = [
        "삼성전자", "SK하이닉스", "한미반도체", "리노공업",
        "DB하이텍", "HPSP", "솔브레인", "동진쎄미켐"
    ]

    power_keywords = [
        "HD현대일렉트릭", "LS ELECTRIC", "효성중공업",
        "LS", "대한전선", "가온전선"
    ]

    ship_keywords = [
        "HD현대중공업", "삼성중공업", "한화오션",
        "HD한국조선해양", "현대미포조선"
    ]

    defense_keywords = [
        "한화에어로스페이스", "LIG넥스원", "현대로템",
        "한국항공우주", "풍산"
    ]

    bio_keywords = [
        "삼성바이오로직스", "셀트리온", "유한양행",
        "알테오젠", "HLB"
    ]

    for keyword in semiconductor_keywords:
        if keyword in name:
            return "AI반도체"

    for keyword in power_keywords:
        if keyword in name:
            return "전력인프라"

    for keyword in ship_keywords:
        if keyword in name:
            return "조선"

    for keyword in defense_keywords:
        if keyword in name:
            return "방산"

    for keyword in bio_keywords:
        if keyword in name:
            return "바이오"

    return "미분류"


def build_watchlist():
    print("네이버 금융에서 KOSPI 시가총액 데이터를 가져오는 중...")
    kospi = get_naver_market_sum(sosok=0, market_name="KOSPI")

    print("네이버 금융에서 KOSDAQ 시가총액 데이터를 가져오는 중...")
    kosdaq = get_naver_market_sum(sosok=1, market_name="KOSDAQ")

    # 우선주 제외
    kospi = remove_preferred_stocks(kospi)
    kosdaq = remove_preferred_stocks(kosdaq)

    # 상위 비율 계산
    kospi_top_count = max(1, int(len(kospi) * KOSPI_TOP_PERCENT))
    kosdaq_top_count = max(1, int(len(kosdaq) * KOSDAQ_TOP_PERCENT))

    kospi_top = kospi.head(kospi_top_count).copy()
    kosdaq_top = kosdaq.head(kosdaq_top_count).copy()

    kospi_top["rank"] = range(1, len(kospi_top) + 1)
    kosdaq_top["rank"] = range(1, len(kosdaq_top) + 1)

    kospi_top["universe_rule"] = "KOSPI_TOP_20_PERCENT"
    kosdaq_top["universe_rule"] = "KOSDAQ_TOP_1_PERCENT"

    universe = pd.concat([kospi_top, kosdaq_top], ignore_index=True)

    universe["code"] = universe["code"].astype(str).str.zfill(6)
    universe["theme"] = universe["name"].apply(classify_theme)
    universe["sub_theme"] = "미분류"
    universe["max_weight"] = 0.10
    universe["active"] = "Y"
    universe["memo"] = ""

    watchlist = universe[
        [
            "code",
            "name",
            "theme",
            "sub_theme",
            "market",
            "rank",
            "market_cap",
            "max_weight",
            "active",
            "universe_rule",
            "memo"
        ]
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    watchlist.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("watchlist.csv 자동 생성 완료")
    print(f"KOSPI 전체 종목 수: {len(kospi)}")
    print(f"KOSPI 상위 20% 종목 수: {len(kospi_top)}")
    print(f"KOSDAQ 전체 종목 수: {len(kosdaq)}")
    print(f"KOSDAQ 상위 1% 종목 수: {len(kosdaq_top)}")
    print(f"최종 관심종목 수: {len(watchlist)}")
    print(f"저장 위치: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_watchlist()
import pandas as pd
from pykrx import stock
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path


# -----------------------------
# 설정
# -----------------------------
KOSPI_TOP_PERCENT = 0.20
KOSDAQ_TOP_PERCENT = 0.01

OUTPUT_PATH = Path("data/watchlist.csv")


def get_recent_market_date(max_lookback_days=14):
    """
    오늘이 휴장일이거나 데이터가 없을 수 있으므로
    최근 거래 가능한 날짜를 찾는다.
    """
    today = datetime.now(ZoneInfo("Asia/Seoul"))

    for i in range(max_lookback_days):
        date = today - timedelta(days=i)
        date_str = date.strftime("%Y%m%d")

        try:
            df = stock.get_market_cap_by_ticker(date_str, market="KOSPI")
            if df is not None and not df.empty:
                return date_str
        except Exception:
            continue

    raise ValueError("최근 거래일 데이터를 찾지 못했습니다.")


def get_top_market_cap_stocks(date, market, top_percent):
    """
    특정 시장에서 시가총액 상위 n% 종목을 가져온다.
    """
    df = stock.get_market_cap_by_ticker(date, market=market)

    if df is None or df.empty:
        raise ValueError(f"{market} 시가총액 데이터를 가져오지 못했습니다.")

    df = df.reset_index()
    df = df.rename(columns={"티커": "code"})

    # 혹시 컬럼명이 다르게 나올 경우 대비
    if "code" not in df.columns:
        df = df.rename(columns={df.columns[0]: "code"})

    df["code"] = df["code"].astype(str).str.zfill(6)
    df["name"] = df["code"].apply(stock.get_market_ticker_name)
    df["market"] = market

    df = df.sort_values("시가총액", ascending=False).reset_index(drop=True)

    total_count = len(df)
    top_count = max(1, int(total_count * top_percent))

    top_df = df.head(top_count).copy()
    top_df["rank"] = range(1, len(top_df) + 1)
    top_df["universe_rule"] = f"{market}_TOP_{int(top_percent * 100)}%"

    return top_df


def build_watchlist():
    date = get_recent_market_date()

    kospi_df = get_top_market_cap_stocks(
        date=date,
        market="KOSPI",
        top_percent=KOSPI_TOP_PERCENT
    )

    kosdaq_df = get_top_market_cap_stocks(
        date=date,
        market="KOSDAQ",
        top_percent=KOSDAQ_TOP_PERCENT
    )

    universe = pd.concat([kospi_df, kosdaq_df], ignore_index=True)

    # 대시보드에서 사용할 컬럼 추가
    universe["theme"] = "미분류"
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
            "시가총액",
            "max_weight",
            "active",
            "universe_rule",
            "memo"
        ]
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    watchlist.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"기준일: {date}")
    print(f"KOSPI 상위 20% 종목 수: {len(kospi_df)}")
    print(f"KOSDAQ 상위 1% 종목 수: {len(kosdaq_df)}")
    print(f"전체 관심종목 수: {len(watchlist)}")
    print(f"저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_watchlist()

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from pykrx import stock
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from modules.news_collector import collect_theme_news
from modules.news_analyzer import analyze_theme_news
from modules.news_reader import enrich_news_with_article_text

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY and OPENAI_API_KEY != "your_api_key_here":
    ai_status = "AI API 연결 준비 완료"
else:
    ai_status = "AI API 키 미설정"

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="AI Investment Copilot",
    layout="wide"
)

st.title("AI Investment Copilot")
st.write("AI 기반 투자 의사결정 및 자동매매 확장형 시스템입니다.")

# -----------------------------
# 사이드바 설정
# -----------------------------
st.sidebar.header("분석 설정")

analysis_days = st.sidebar.slider(
    "분석 기간",
    min_value=120,
    max_value=450,
    value=400,
    step=30
)

st.sidebar.markdown("### AI 설정 상태")
st.sidebar.write(ai_status)

show_debug = st.sidebar.checkbox(
    "디버그 정보 보기",
    value=True
)

# 한국 시간 기준
today_kst = datetime.now(ZoneInfo("Asia/Seoul"))

# 오늘 데이터가 아직 완전히 반영되지 않을 수 있으므로 하루 전까지 조회
end = today_kst - timedelta(days=1)
start = end - timedelta(days=analysis_days)

start_date = start.strftime("%Y%m%d")
end_date = end.strftime("%Y%m%d")

st.sidebar.write(f"조회 시작일: `{start_date}`")
st.sidebar.write(f"조회 종료일: `{end_date}`")

# -----------------------------
# 관심종목 불러오기
# -----------------------------
st.subheader("관심 종목 목록")

try:
    watchlist = pd.read_csv("data/watchlist.csv", dtype={"code": str})

    required_columns = {"code", "name", "theme"}
    if not required_columns.issubset(watchlist.columns):
        st.error("watchlist.csv에는 code, name, theme 컬럼이 모두 있어야 합니다.")
        st.stop()

    # 종목코드 정리: 6자리 문자열로 변환
    watchlist["code"] = (
        watchlist["code"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
        .str.zfill(6)
    )

    st.dataframe(watchlist, width="stretch")

except FileNotFoundError:
    st.error("data/watchlist.csv 파일이 없습니다.")
    st.stop()

except Exception as e:
    st.error("watchlist.csv를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


# -----------------------------
# pykrx 데이터 불러오기 함수
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_price_data(code: str, start_date: str, end_date: str):
    """
    pykrx에서 개별 종목의 OHLCV 데이터를 불러오는 함수
    """
    return stock.get_market_ohlcv_by_date(start_date, end_date, code)


def analyze_stock(code, name, theme, foreign_ratio, per, roe):
    """
    종목별 주가 데이터 분석 함수
    """
    try:
        code = str(code).strip().zfill(6)

        df = load_price_data(code, start_date, end_date)

        if df is None:
            return None, {
                "code": code,
                "name": name,
                "theme": theme,
                "reason": "pykrx가 None을 반환했습니다."
            }

        if df.empty:
            return None, {
                "code": code,
                "name": name,
                "theme": theme,
                "reason": "pykrx가 빈 데이터프레임을 반환했습니다."
            }

        required_price_columns = {"종가", "거래량"}
        if not required_price_columns.issubset(df.columns):
            return None, {
                "code": code,
                "name": name,
                "theme": theme,
                "reason": f"필수 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}"
            }

        if len(df) < 30:
            return None, {
                "code": code,
                "name": name,
                "theme": theme,
                "reason": f"데이터가 30거래일 미만입니다. 현재 {len(df)}개"
            }

        close = df["종가"].dropna()
        volume = df["거래량"].fillna(0)

        if len(close) < 30:
            return None, {
                "code": code,
                "name": name,
                "theme": theme,
                "reason": "종가 데이터가 부족합니다."
            }

        current_price = int(close.iloc[-1])

        def calc_return(price_series, days):
            """
            지정한 거래일 기준 수익률 계산
            데이터가 부족하면 0으로 처리
            """
            if len(price_series) <= days:
                return 0

            past_price = price_series.iloc[-days - 1]
            current_price_for_return = price_series.iloc[-1]

            if past_price == 0:
                return 0

            return ((current_price_for_return / past_price) - 1) * 100

        ret_5d = calc_return(close, 5)
        ret_20d = calc_return(close, 20)
        ret_60d = calc_return(close, 60)
        ret_120d = calc_return(close, 120)
        ret_240d = calc_return(close, 240)

        volume_5d = volume.tail(5).mean()
        volume_20d = volume.tail(20).mean()

        if volume_20d == 0:
            volume_ratio = 0
        else:
            volume_ratio = volume_5d / volume_20d

        ma20 = close.rolling(20).mean().iloc[-1]
        trend = ((close.iloc[-1] / ma20) - 1) * 100

        volatility = close.pct_change().tail(20).std() * 100

        return {
            "code": code,
            "name": name,
            "theme": theme,
            "price": current_price,
            "ret_5d": round(ret_5d, 2),
            "ret_20d": round(ret_20d, 2),
            "ret_60d": round(ret_60d, 2),
            "ret_120d": round(ret_120d, 2),
            "ret_240d": round(ret_240d, 2),
            "foreign_ratio": foreign_ratio,
            "per": per,
            "roe": roe,
            "volume_ratio": round(volume_ratio, 2),
            "trend": round(trend, 2),
            "volatility": round(volatility, 2),
            "data_count": len(df)
        }, None

    except Exception as e:
        return None, {
            "code": code,
            "name": name,
            "theme": theme,
            "reason": str(e)
        }


# -----------------------------
# 전체 종목 분석 실행
# -----------------------------
st.subheader("주가 데이터 분석")

results = []
errors = []

with st.spinner("주가 데이터를 불러오고 분석하는 중입니다..."):
    for _, row in watchlist.iterrows():
        result, error = analyze_stock(
            row["code"],
            row["name"],
            row["theme"],
            row["foreign_ratio"],
            row["per"],
            row["roe"]
        )

        if result is not None:
            results.append(result)

        if error is not None:
            errors.append(error)

# -----------------------------
# 오류 종목 표시
# -----------------------------
if errors:
    st.warning("일부 종목의 주가 데이터를 불러오지 못했습니다.")
    error_df = pd.DataFrame(errors)
    st.dataframe(error_df, width="stretch")

    st.info(
        "위 표의 reason을 확인하세요. "
        "종목코드 문제인지, pykrx 데이터 수집 문제인지, 날짜 문제인지 확인할 수 있습니다."
    )

# -----------------------------
# 분석 결과 표시
# -----------------------------
if not results:
    st.error("분석 가능한 주가 데이터가 없습니다.")

    st.markdown("### 확인할 것")
    st.write("1. watchlist.csv의 종목코드가 6자리인지 확인")
    st.write("2. pykrx가 Streamlit Cloud에서 데이터를 정상적으로 가져오는지 확인")
    st.write("3. 조회 기간에 거래 데이터가 존재하는지 확인")
    st.write("4. 위 오류 표의 reason 내용을 확인")

    st.stop()

df = pd.DataFrame(results)

# 숫자형 변환
numeric_cols = [
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "ret_120d",
    "ret_240d",
    "foreign_ratio",
    "per",
    "roe",
    "volume_ratio",
    "trend",
    "volatility"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

required_price_cols = [
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "ret_120d",
    "ret_240d",
    "volume_ratio",
    "trend",
    "volatility"
]

df = df.dropna(subset=required_price_cols)

df["foreign_ratio"] = df["foreign_ratio"].fillna(df["foreign_ratio"].median())
df["per"] = df["per"].fillna(df["per"].median())
df["roe"] = df["roe"].fillna(df["roe"].median())

valid_per = df["per"].where(df["per"] > 0)
valid_per = valid_per.fillna(valid_per.median())

per_score = (1 - valid_per.rank(pct=True)) * 100
roe_score = df["roe"].rank(pct=True) * 100
foreign_score = df["foreign_ratio"].rank(pct=True) * 100

df["fundamental_score"] = (
    roe_score * 0.5
    + per_score * 0.3
    + foreign_score * 0.2
).round(1)

if df.empty:
    st.error("계산 가능한 숫자 데이터가 없습니다.")
    st.stop()

# -----------------------------
# 점수화
# -----------------------------
df["momentum_score"] = (
    df["ret_240d"].rank(pct=True) * 10
    + df["ret_120d"].rank(pct=True) * 15
    + df["ret_60d"].rank(pct=True) * 25
    + df["ret_20d"].rank(pct=True) * 25
    + df["ret_5d"].rank(pct=True) * 10
    + df["volume_ratio"].rank(pct=True) * 15
    - df["volatility"].rank(pct=True) * 10
)

df["momentum_score"] = df["momentum_score"].round(1)

df["theme_score"] = (
    df.groupby("theme")["momentum_score"]
    .transform("mean")
    .round(1)
)

df["risk_score"] = (
    100
    - df["volatility"].rank(pct=True) * 40
    - df["ret_5d"].rank(pct=True) * 20
    - df["ret_20d"].rank(pct=True) * 20
    - df["volume_ratio"].rank(pct=True) * 20
).clip(0, 100).round(1)

vol_threshold = df["volatility"].quantile(0.8)
volume_threshold = df["volume_ratio"].quantile(0.8)

def classify_risk_type(row):
    risk_types = []

    if row["ret_5d"] > 10:
        risk_types.append("단기 급등")

    if row["ret_20d"] > 20:
        risk_types.append("20일 과열")

    if row["volatility"] >= vol_threshold:
        risk_types.append("고변동성")

    if row["volume_ratio"] >= volume_threshold:
        risk_types.append("거래량 과열")

    if row["trend"] < 0:
        risk_types.append("추세 약화")

    if not risk_types:
        return "특이 리스크 없음"

    return ", ".join(risk_types)

df["risk_type"] = df.apply(classify_risk_type, axis=1)

df["total_score"] = (
    df["momentum_score"] * 0.55
    + df["theme_score"] * 0.15
    + df["fundamental_score"] * 0.20
    + df["risk_score"] * 0.1
).round(1)

# -----------------------------
# 핵심 지표
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("분석 성공 종목 수", len(df))

with col2:
    top_theme = df.groupby("theme")["total_score"].mean().sort_values(ascending=False).index[0]
    st.metric("가장 강한 테마", top_theme)

with col3:
    top_stock = df.sort_values("total_score", ascending=False).iloc[0]["name"]
    st.metric("최상위 후보", top_stock)

with col4:
    avg_total_score = round(df["total_score"].mean(), 1)
    st.metric("평균 점수", avg_total_score)

# -----------------------------
# 매수 후보 TOP 10
# -----------------------------
st.subheader("매수 후보 TOP 10")

top10 = df.sort_values("total_score", ascending=False).head(10)

display_cols = [
    "code",
    "name",
    "theme",
    "price",
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "ret_120d",
    "ret_240d",
    "volume_ratio",
    "trend",
    "volatility",
    "momentum_score",
    "theme_score",
    "fundamental_score",
    "risk_score",
    "risk_type",
    "total_score"
]

st.dataframe(top10[display_cols], width="stretch")

# -----------------------------
# 테마별 평균 점수
# -----------------------------
st.subheader("테마별 평균 점수")

theme_total_score = (
    df.groupby("theme")["total_score"]
    .mean()
    .sort_values(ascending=False)
    .round(1)
)

st.bar_chart(theme_total_score)

st.dataframe(
    theme_total_score.reset_index().rename(columns={"total_score": "theme_avg_total_score"}),
    width="stretch"
)

# -----------------------------
# 리스크 경고
# -----------------------------
st.subheader("리스크 경고")

vol_threshold = df["volatility"].quantile(0.8)

risk_df = df[
    (df["ret_5d"] > 10) &
    (df["volatility"] >= vol_threshold)
].sort_values("volatility", ascending=False)

if not risk_df.empty:
    st.warning("단기 급등과 고변동성이 동시에 나타난 종목이 있습니다.")
    st.dataframe(
        risk_df[
            [
                "code",
                "name",
                "theme",
                "ret_5d",
                "ret_20d",
                "ret_60d",
                "ret_120d",
                "volatility",
                "momentum_score",
                "theme_score",
                "fundamental_score",
                "risk_score",
                "risk_type",
                "total_score"
            ]
        ],
        width="stretch"
    )
else:
    st.success("현재 기준으로 단기 급등 + 고변동성 위험 종목은 많지 않습니다.")


# -----------------------------
# 테마 뉴스 수집 테스트
# -----------------------------
st.subheader("테마 뉴스 수집 테스트")

selected_news_theme = st.selectbox(
    "뉴스를 수집할 테마를 선택하세요",
    sorted(df["theme"].unique())
)

selected_news_regions = st.multiselect(
    "뉴스 권역을 선택하세요",
    ["국내", "미국", "일본", "중동"],
    default=["국내", "미국", "일본", "중동"]
)

max_news_per_keyword = st.slider(
    "키워드당 뉴스 수",
    min_value=1,
    max_value=5,
    value=2
)

if "theme_news_items" not in st.session_state:
    st.session_state["theme_news_items"] = []

if "theme_news_df" not in st.session_state:
    st.session_state["theme_news_df"] = None

if "theme_news_analysis" not in st.session_state:
    st.session_state["theme_news_analysis"] = None

if st.button("테마 뉴스 수집"):
    news_items = collect_theme_news(
        selected_news_theme,
        max_items_per_keyword=max_news_per_keyword,
        regions=selected_news_regions
    )

    st.session_state["theme_news_items"] = news_items
    st.session_state["theme_news_analysis"] = None

    if news_items:
        st.session_state["theme_news_df"] = pd.DataFrame(news_items)
    else:
        st.session_state["theme_news_df"] = None

news_df = st.session_state.get("theme_news_df")

if news_df is not None and not news_df.empty:
    st.write("권역별 뉴스 수")
    st.dataframe(
        news_df.groupby("region").size().reset_index(name="news_count"),
        width="stretch"
    )

    st.dataframe(news_df, width="stretch")

    if st.button("AI 뉴스 분석 실행"):
        with st.spinner("기사 본문을 읽고 AI가 뉴스 분위기를 분석하는 중입니다..."):
            enriched_news = enrich_news_with_article_text(
                news_df,
                max_articles=10,
                max_chars_per_article=2500
            )

            enriched_news_df = pd.DataFrame(enriched_news)

            analysis_result = analyze_theme_news(
                selected_news_theme,
                enriched_news_df
            )

            st.session_state["theme_news_df"] = enriched_news_df
            st.session_state["theme_news_analysis"] = analysis_result

    analysis_result = st.session_state.get("theme_news_analysis")

    if analysis_result:
        st.write("AI 뉴스 분석 결과")

        if analysis_result.get("ai_enabled"):
            st.success("AI 뉴스 분석 완료")
        else:
            st.warning("AI 뉴스 분석이 기본값으로 처리되었습니다.")

        st.metric(
            "테마 뉴스 점수",
            analysis_result.get("theme_news_score", 50)
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "본문 기반 기사",
            analysis_result.get("article_based_count", 0)
        )

        col2.metric(
            "제목 기반 기사",
            analysis_result.get("title_only_count", 0)
        )

        col3.metric(
            "분석 기준",
            analysis_result.get("analysis_basis", "분석 기준 없음")
        )

        st.write("뉴스 분위기")
        st.write(analysis_result.get("news_sentiment", "중립"))

        st.write("요약")
        st.write(analysis_result.get("summary", ""))

        st.write("긍정 요인")
        st.write(analysis_result.get("positive_points", []))

        st.write("부정 요인")
        st.write(analysis_result.get("negative_points", []))

        st.write("주요 권역")
        st.write(analysis_result.get("key_regions", []))

        if analysis_result.get("error"):
            st.error(analysis_result.get("error"))

else:
    st.info("수집된 뉴스가 없습니다. 먼저 테마 뉴스를 수집하세요.")


# -----------------------------
# 전체 분석 결과
# -----------------------------
with st.expander("전체 분석 결과 보기"):
    st.dataframe(
        df.sort_values("total_score", ascending=False),
        width="stretch"
    )

# -----------------------------
# 디버그 정보
# -----------------------------
if show_debug:
    with st.expander("디버그 정보"):
        st.write("조회 시작일:", start_date)
        st.write("조회 종료일:", end_date)
        st.write("watchlist.csv 원본")
        st.dataframe(watchlist, width="stretch")

        st.write("분석 성공 데이터")
        st.dataframe(df, width="stretch")

        if errors:
            st.write("분석 실패 데이터")
            st.dataframe(pd.DataFrame(errors), width="stretch")

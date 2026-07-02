import streamlit as st
import pandas as pd
from modules.data_loader import load_watchlist, load_price_data
from modules.indicator import calc_indicators
from modules.scorer import calc_score

st.set_page_config(page_title="AI Investment Copilot", layout="wide")
st.title("AI Investment Copilot")
st.write("AI 기반 투자 의사결정 및 자동매매 확장형 시스템입니다.")

watchlist = load_watchlist()
if watchlist is None:
    st.warning("data/watchlist.csv 파일이 아직 없습니다.")
    st.stop()

st.subheader("관심 종목 목록")
st.dataframe(watchlist, use_container_width=True)

with st.spinner("주가 데이터를 불러오는 중..."):
    rows = []
    for _, r in watchlist.iterrows():
        df = load_price_data(r["code"])
        if df is None or len(df) < 25:
            continue
        ind = calc_indicators(df)
        ind["code"] = r["code"]
        ind["name"] = r["name"]
        ind["theme"] = r["theme"]
        rows.append(ind)

if not rows:
    st.error("주가 데이터를 불러오지 못했습니다. 종목코드를 확인하세요.")
    st.stop()

result = pd.DataFrame(rows)
result = calc_score(result)
result = result.sort_values("score", ascending=False)

st.subheader("매수 후보 TOP 10")
top10 = result.head(10)[["code", "name", "theme", "score", "ret_5d", "ret_20d", "vol_ratio", "ma20_gap", "volatility"]]
st.dataframe(top10, use_container_width=True)

st.subheader("테마별 평균 점수")
theme_score = result.groupby("theme")["score"].mean().sort_values(ascending=False).reset_index()
st.bar_chart(theme_score.set_index("theme"))

st.subheader("리스크 경고")
risk = result[(result["volatility"] > result["volatility"].quantile(0.8)) | (result["vol_ratio"] < 0.7)]
if len(risk) == 0:
    st.info("현재 특별한 리스크 경고 종목이 없습니다.")
else:
    for _, r in risk.iterrows():
        reasons = []
        if r["volatility"] > result["volatility"].quantile(0.8):
            reasons.append("변동성 과열")
        if r["vol_ratio"] < 0.7:
            reasons.append("거래량 급감")
        st.warning(f"{r['name']} ({r['code']}): " + ", ".join(reasons))

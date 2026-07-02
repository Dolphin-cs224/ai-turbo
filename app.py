import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI Investment Copilot", layout="wide")

st.title("AI Investment Copilot")
st.write("키움증권 대학생 투자대회를 위한 AI 기반 투자 의사결정 보조 대시보드입니다.")

st.subheader("관심 종목 목록")

try:
    watchlist = pd.read_csv("data/watchlist.csv", dtype={"code": str})
    st.dataframe(watchlist, use_container_width=True)
except FileNotFoundError:
    st.warning("data/watchlist.csv 파일이 아직 없습니다.")

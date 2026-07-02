"""매수/매도 신호 생성 모듈 (추후 확장용 스텁).

score, indicator 결과를 받아 규칙 기반으로 매수/매도 후보를 판정한다.
현재는 app.py에서 직접 TOP10을 뽑고 있으나, 규칙이 복잡해지면
이 모듈로 로직을 옮겨서 관리한다.
"""


def generate_signals(df, buy_top_n=10, stop_loss_pct=-5, trail_stop_pct=-5):
    buy_candidates = df.sort_values("score", ascending=False).head(buy_top_n)
    return {"buy": buy_candidates}

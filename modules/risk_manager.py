"""주문 전 리스크 체크 모듈 (추후 확장용 스텁).

자동매매 단계에서 주문 직전 반드시 이 모듈을 통과시킨다.
"""

MAX_POSITION_RATIO = 0.1
MAX_THEME_RATIO = 0.3
DAILY_LOSS_LIMIT_RATIO = -0.03


def check_order(order, account_state):
    reasons = []
    if account_state.get("today_pnl_ratio", 0) <= DAILY_LOSS_LIMIT_RATIO:
        reasons.append("일일 손실 한도 초과")
    if order.get("position_ratio", 0) > MAX_POSITION_RATIO:
        reasons.append("종목당 최대 비중 초과")
    if order.get("theme_ratio", 0) > MAX_THEME_RATIO:
        reasons.append("테마 비중 초과")
    return {"approved": len(reasons) == 0, "reasons": reasons}

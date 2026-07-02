import pandas as pd


def calc_indicators(df: pd.DataFrame) -> dict:
    close = df["종가"]
    volume = df["거래량"]

    ret_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) > 6 else 0
    ret_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 21 else 0

    vol_recent = volume.iloc[-5:].mean()
    vol_prev = volume.iloc[-20:-5].mean()
    vol_ratio = vol_recent / vol_prev if vol_prev > 0 else 1

    ma20 = close.rolling(20).mean().iloc[-1]
    ma20_gap = (close.iloc[-1] / ma20 - 1) * 100 if ma20 > 0 else 0

    volatility = close.pct_change().rolling(20).std().iloc[-1] * 100

    return {
        "ret_5d": round(ret_5d, 2),
        "ret_20d": round(ret_20d, 2),
        "vol_ratio": round(vol_ratio, 2),
        "ma20_gap": round(ma20_gap, 2),
        "volatility": round(volatility, 2),
    }

import pandas as pd


def _minmax(s: pd.Series) -> pd.Series:
    if s.max() == s.min():
        return s * 0
    return (s - s.min()) / (s.max() - s.min()) * 100


def calc_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ret5_score = _minmax(df["ret_5d"])
    ret20_score = _minmax(df["ret_20d"])
    vol_score = _minmax(df["vol_ratio"])
    trend_score = _minmax(df["ma20_gap"])
    risk_score = _minmax(df["volatility"])

    df["score"] = (
        ret5_score * 0.25
        + ret20_score * 0.25
        + vol_score * 0.2
        + trend_score * 0.2
        - risk_score * 0.1
    ).round(1)
    return df

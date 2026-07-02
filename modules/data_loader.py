import pandas as pd
from datetime import datetime, timedelta

try:
    from pykrx import stock
except ImportError:
    stock = None


def load_watchlist(path="data/watchlist.csv"):
    try:
        return pd.read_csv(path, dtype={"code": str})
    except FileNotFoundError:
        return None


def load_price_data(code, days=60):
    if stock is None:
        return None
    end = datetime.today()
    start = end - timedelta(days=days)
    try:
        df = stock.get_market_ohlcv(
            start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), code
        )
        return df
    except Exception:
        return None

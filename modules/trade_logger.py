"""매매일지 기록 모듈 (추후 확장용 스텁)."""
import csv
import os
from datetime import datetime


LOG_PATH = "data/trade_log.csv"
FIELDS = ["date", "code", "name", "side", "price", "qty", "reason", "ai_score", "note"]


def log_trade(record: dict):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    write_header = not os.path.exists(LOG_PATH)
    record.setdefault("date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in FIELDS})

"""Dashboard numbers."""
import sqlite3
from datetime import datetime, timedelta

import pandas as pd

from src import dashboard_data as D


def _df(pnls, qty=2):
    t0 = datetime(2026, 9, 28, 12)
    return pd.DataFrame({
        "market_id": ["KXHIGHNY-26SEP28-B72.5"] * len(pnls),
        "pnl": pnls, "quantity": [qty] * len(pnls),
        "exit_time": [t0 + timedelta(hours=10 * i) for i in range(len(pnls))],
        "settled": [True] * len(pnls), "niche": ["weather"] * len(pnls),
    })


def test_kpis():
    k = D.kpis(_df([1.0, -3.0, 0.5, 2.0]))
    assert k["trades"] == 4 and k["win_rate"] == 0.75
    assert k["total_pnl"] == 0.5
    assert k["edge_per_contract_c"] == 100 * 0.5 / 8
    assert k["max_drawdown"] == -3.0        # peak 1.0 -> trough -2.0
    assert k["days_traded"] == 2


def test_empty_is_safe():
    k = D.kpis(pd.DataFrame())
    assert k["trades"] == 0 and k["total_pnl"] == 0.0
    assert D.pnl_by_day(pd.DataFrame()).empty and D.equity_curve(pd.DataFrame()).empty


def test_trades_reads_db(tmp_path):
    db = tmp_path / "t.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE trade_logs (market_id, side, entry_price, exit_price, quantity, pnl, "
              "entry_timestamp, exit_timestamp, rationale, strategy)")
    c.execute("INSERT INTO trade_logs VALUES ('KXRT-DIG-60','YES',0.4,1.0,5,3.0,"
              "'2026-09-29T10:00:00','2026-09-30T10:00:00','x | EXIT: market_resolution',NULL)")
    c.commit()
    c.close()
    df = D.trades(str(db))
    assert df.loc[0, "niche"] == "rotten_tomatoes" and bool(df.loc[0, "settled"])
    assert df.loc[0, "pnl_per_contract_c"] == 60.0

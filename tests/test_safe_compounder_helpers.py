"""Regression test: Kalshi v2 returns numeric fields as STRINGS ("0.00").

A live run placed real orders but then crashed on `if fill_count > 0` because
fill_count was the string "0.00" (str > int -> TypeError), mis-reporting placed
orders as errors. _to_int_count coerces these safely.
"""
from src.strategies.safe_compounder import _to_int_count


def test_to_int_count_handles_v2_strings_and_ints():
    assert _to_int_count("0.00") == 0
    assert _to_int_count("16.00") == 16
    assert _to_int_count(3) == 3
    assert _to_int_count(None) == 0
    assert _to_int_count("") == 0
    assert _to_int_count("garbage") == 0


def test_run_never_cancels_other_strategies_orders(tmp_path):
    # Regression (2026-09-26): the legacy "Step 0: cancel YES orders" ran even in
    # dry mode and its side filter matched the agent's NO sells — the daily job
    # would have cancelled every resting order on the live account once
    # cancel_order worked. run() must never cancel orders it didn't place.
    import asyncio
    from src.strategies.safe_compounder import SafeCompounder

    class _Client:
        cancelled = []
        async def get_balance(self):
            return {"balance": 1000, "portfolio_value": 0}
        async def get_orders(self, **kw):
            return {"orders": [{"order_id": "agent-no-sell", "side": "yes", "ticker": "T"}]}
        async def cancel_order(self, order_id):
            self.cancelled.append(order_id)

    sc = SafeCompounder(_Client(), db_path=str(tmp_path / "t.db"), dry_run=True)
    async def _no_markets():
        return []
    sc._fetch_all_markets = _no_markets
    for dry in (True, False):
        try:
            asyncio.run(sc.run(dry_run=dry))
        except Exception:
            pass  # later steps may need more fakes; only the cancel matters
    assert _Client.cancelled == []

"""
Kalshi bot control panel (Streamlit).

    python cli.py dashboard        # serves on http://localhost:8501 (server only)

View it from your computer through an SSH tunnel:
    ssh -L 8501:localhost:8501 root@<server-ip>

Top: results (paper or live) in the style of a trading dashboard.
Bottom: strategy controls that the running bot picks up within ~1 minute.
Live trading can't be switched on from here.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import dashboard_data as D  # noqa: E402
from src import runtime_config as RC  # noqa: E402

ACCENT = "#8b5cf6"
GREEN = "#34d399"
RED = "#f87171"
NICHE_LABELS = {"rotten_tomatoes": "Rotten Tomatoes", "trump_mentions": "Trump mentions",
                "weather": "Weather", "sports": "Sports", "stocks": "Stocks", "other": "Other"}

st.set_page_config(page_title="Kalshi Bot Control", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 3rem; max-width: 1400px;}
      div[data-testid="stMetric"] {background: rgba(139,92,246,0.06); border: 1px solid rgba(139,92,246,0.25);
          border-radius: 12px; padding: 12px 16px;}
      div[data-testid="stMetricLabel"] p {text-transform: uppercase; letter-spacing: .06em; font-size: .72rem; opacity: .75;}
      h3 {letter-spacing: .04em;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=20, max_entries=1)
def load_all():
    return {
        "trades": D.trades(),
        "positions": D.open_positions(),
        "learning": D.learning_summary(),
        "predictions": D.recent_predictions(),
        "wht": D.would_have_traded(float(RC.get("MIN_EDGE"))),
        "ai": D.ai_spend_today(),
        "status": D.bot_status(),
        "maker": D.maker_data("v1"),
        "maker_v2": D.maker_data("v2"),
    }


def show_search(res, what: str, key: str) -> None:
    """The 'search every split' result: best-looking groups, judged honestly for how many were tried."""
    from src import edge_scan
    res = edge_scan.from_jsonable(res) if res and not isinstance(res.get("best"), pd.DataFrame) else (res or {})
    tests = res.get("tests") or 0
    if not tests:
        st.info("Not enough settled data yet to search the splits (each group needs 30+ separate events).")
        return
    passing = res.get("passing", 0)
    lucky = res.get("expected_by_luck", round(0.025 * tests, 1))
    if passing:
        st.success(f"{passing} group(s) still look positive after allowing for the {tests} groups tried. "
                   f"Worth watching: if one stays positive as more {what} settle, it's a real lead.")
    else:
        st.warning(f"Tried {tests} groups (every split, and every pair of splits). {res.get('positive_95', 0)} looked "
                   f"positive at the usual 95% bar, about what luck alone produces ({lucky}). None survive once "
                   f"you allow for how many were tried: no hidden pocket of profit so far.")
    best = res["best"]
    if isinstance(best, pd.DataFrame) and not best.empty:
        st.dataframe(best[["split", "group", "bets", "events", "c_per_contract", "margin_95", "95% sure",
                           "survives the search"]],
                     hide_index=True, use_container_width=True,
                     column_config={"c_per_contract": st.column_config.NumberColumn("¢ / contract", format="%+.2f"),
                                    "margin_95": st.column_config.NumberColumn("± (¢)", format="%.1f")})


@st.cache_data(ttl=600, max_entries=2)
def _maker_search(variant: str):
    """Cached: the search reads every settled bet, so don't redo it on every click."""
    from src import maker_analysis as MA
    return MA.search_every_split(D.maker_data(variant)["features"])


def money(x: float) -> str:
    return f"{'-' if x < 0 else '+'}${abs(x):,.2f}"


def chart_layout(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(showgrid=False), yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
                      showlegend=False)
    return fig


data = load_all()
status = data["status"]
cfg = RC.current()

# ----------------------------------------------------------------------------
# Header / status bar
# ----------------------------------------------------------------------------
h1, h2 = st.columns([5, 1], vertical_alignment="center")
with h1:
    st.markdown("## Kalshi Bot Control")
with h2:
    if st.button("↻ Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

s1, s2, s3, s4 = st.columns(4)
mins = status["minutes_ago"]
if mins is None:
    s1.metric("Bot", "no log found")
elif mins < 30:  # a full scan of every niche can take ~20 minutes
    s1.metric("Bot", "● running", f"active {mins:.0f} min ago", delta_color="off")
elif status.get("session_running") is False:
    s1.metric("Bot", "○ off", f"stopped; last ran {mins / 60:.0f}h ago", delta_color="off")
else:
    s1.metric("Bot", "⚠ stalled?", f"last activity {mins:.0f} min ago", delta_color="off")
s2.metric("Mode", {"paper": "PAPER", "LIVE": "LIVE $"}.get(status["mode"], "unknown"))
s3.metric("New trades", "PAUSED" if cfg["PAUSED"] else "ON")
ai = data["ai"]
cap = cfg["DAILY_AI_COST_LIMIT"]
s4.metric("AI spend today", f"\\${ai['cost']:.2f} of \\${cap:.2f}", f"{ai['requests']} requests", delta_color="off")

# Live tests first; finished ones live in the Archive tab (their numbers stay, frozen).
tab_fav, tab_poly, tab_maker, tab_whale, tab_ctrl, tab_arch = st.tabs(
    ["Favorites test", "Polymarket check", "Maker test", "Whale follow", "Strategy controls", "Archive"])
with tab_arch:
    st.caption("Finished tests, kept for the record. None of these found an edge; their processes are stopped "
               "and their numbers are frozen at the end.")
    tab_perf = st.expander("Original AI bot: performance (no edge vs. the market; bot off)")
    tab_pos = st.expander("Original AI bot: open positions")
    tab_learn = st.expander("Original AI bot: learning")
    tab_pred = st.expander("Original AI bot: predictions")
    tab_gap = st.expander("Speed gaps, crypto 15-minute markets (stopped Oct 24: nothing a 1-3 s bot can catch)")
    tab_arb = st.expander("Price inconsistencies (stopped Oct 24: tiny edges, almost nothing buyable)")

# ----------------------------------------------------------------------------
# Performance
# ----------------------------------------------------------------------------
with tab_perf:
    df = data["trades"]
    niches_present = sorted(df["niche"].unique()) if not df.empty else []
    pick = st.multiselect("Niches", niches_present, default=niches_present,
                          format_func=lambda n: NICHE_LABELS.get(n, n), label_visibility="collapsed")
    view = df[df["niche"].isin(pick)] if not df.empty else df
    settled_only = st.toggle("Settled trades only (held to resolution)", value=False)
    if settled_only and not view.empty:
        view = view[view["settled"]]

    k = D.kpis(view)
    c = st.columns(6)
    c[0].metric("Trades", k["trades"], f"{k['settled']} settled", delta_color="off")
    c[1].metric("Win rate", f"{k['win_rate']:.1%}" if k["win_rate"] is not None else "—")
    c[2].metric("Edge / contract", f"{k['edge_per_contract_c']:.2f}¢" if k["edge_per_contract_c"] is not None else "—")
    c[3].metric("Total P&L", money(k["total_pnl"]))
    c[4].metric("Max drawdown", money(k["max_drawdown"]) if k["max_drawdown"] else "$0.00")
    c[5].metric("Days traded", k["days_traded"])

    if view.empty:
        st.info("No closed trades yet. Positions now hold until their markets settle, so results "
                "appear as markets resolve (weather daily; Trump weekly; films on their dates).")
    else:
        left, right = st.columns(2)
        with left:
            st.markdown("### Equity over time")
            st.caption("Cumulative P&L in trade order. Healthy = steady climb; one big step = one event carried it.")
            eq = D.equity_curve(view)
            fig = go.Figure(go.Scatter(x=eq["exit_time"], y=eq["equity"], mode="lines",
                                       line=dict(color=ACCENT, width=2), fill="tozeroy",
                                       fillcolor="rgba(139,92,246,0.18)"))
            st.plotly_chart(chart_layout(fig), use_container_width=True)
        with right:
            st.markdown("### P&L by day")
            st.caption("Consistency across days matters more than the total.")
            byd = D.pnl_by_day(view)
            fig = go.Figure(go.Bar(x=byd["day"].astype(str), y=byd["pnl"],
                                   marker_color=[GREEN if v >= 0 else RED for v in byd["pnl"]],
                                   customdata=byd["trades"],
                                   hovertemplate="%{x}<br>P&L $%{y:.2f}<br>%{customdata} trades<extra></extra>"))
            st.plotly_chart(chart_layout(fig), use_container_width=True)

        st.markdown("### By niche")
        bn = D.by_niche(view)
        st.dataframe(
            bn.assign(niche=bn["niche"].map(lambda n: NICHE_LABELS.get(n, n))),
            hide_index=True, use_container_width=True,
            column_config={
                "win_rate": st.column_config.NumberColumn("win rate", format="percent"),
                "total_pnl": st.column_config.NumberColumn("total P&L", format="dollar"),
                "pnl_per_contract_c": st.column_config.NumberColumn("¢ / contract", format="%.2f"),
            },
        )
        st.markdown("### Recent closed trades")
        st.dataframe(
            view.sort_values("exit_time", ascending=False).head(100)[
                ["exit_time", "niche", "market_id", "side", "entry_price", "exit_price", "quantity", "pnl", "exit_reason"]],
            hide_index=True, use_container_width=True,
            column_config={"pnl": st.column_config.NumberColumn("P&L", format="dollar"),
                           "entry_price": st.column_config.NumberColumn("entry", format="%.3f"),
                           "exit_price": st.column_config.NumberColumn("exit", format="%.3f")},
        )

# ----------------------------------------------------------------------------
# Strategy controls
# ----------------------------------------------------------------------------
with tab_ctrl:
    st.caption("Changes are saved to the server and picked up by the running bot within about a minute "
               "(new niches within ~5 minutes). No restart needed. Live trading can't be enabled here.")
    overridden = set(RC.overrides())
    groups: dict = {}
    for spec in RC.SPEC:
        groups.setdefault(spec["group"], []).append(spec)

    with st.form("controls"):
        new_vals = {}
        for group, specs in groups.items():
            st.markdown(f"### {group}")
            cols = st.columns(2)
            for i, spec in enumerate(specs):
                key, cur = spec["key"], cfg[spec["key"]]
                label = spec["label"] + ("  •" if key in overridden else "")
                col = cols[i % 2]
                if spec["type"] == "bool":
                    new_vals[key] = col.toggle(label, value=bool(cur), help=spec["help"])
                elif spec["type"] in ("niches", "multi"):
                    opts = spec.get("options", RC.ALL_NICHES)
                    chosen = [n for n in str(cur).split(",") if n]
                    new_vals[key] = col.multiselect(label, opts, default=[n for n in chosen if n in opts],
                                                    format_func=lambda n: NICHE_LABELS.get(n, n.upper()),
                                                    help=spec["help"])
                else:
                    scale = spec.get("scale", 1)
                    is_int = spec["type"] == "int"
                    fmt = "%d" if is_int else ("%.0f" if float(spec["step"] * scale).is_integer() else "%.1f")
                    shown = col.number_input(
                        f"{label} ({spec['unit']})",
                        format=fmt,
                        min_value=float(spec["min"] * scale) if not is_int else int(spec["min"]),
                        max_value=float(spec["max"] * scale) if not is_int else int(spec["max"]),
                        value=float(cur * scale) if not is_int else int(cur),
                        step=float(spec["step"] * scale) if not is_int else int(spec["step"]),
                        help=spec["help"],
                    )
                    new_vals[key] = shown / scale if not is_int else int(shown)
        b1, b2 = st.columns([1, 1])
        save = b1.form_submit_button("Save changes", type="primary", use_container_width=True)
        reset = b2.form_submit_button("Reset all to defaults", use_container_width=True)

    if save:
        def _differs(k, v):
            spec = next(sp for sp in RC.SPEC if sp["key"] == k)
            nv = RC._coerce(spec, v)
            return abs(nv - cfg[k]) > 1e-9 if isinstance(nv, (int, float)) and not isinstance(nv, bool) else nv != cfg[k]
        changed = {k: v for k, v in new_vals.items() if k in overridden or _differs(k, v)}
        if not new_vals.get("NICHES"):
            st.error("Pick at least one niche (to stop trading entirely, use 'Pause new trades').")
        else:
            try:
                RC.save(changed)
                st.success("Saved. The bot will use these settings from its next cycle.")
                st.cache_data.clear()
            except (ValueError, KeyError) as e:
                st.error(str(e))
    if reset:
        RC.reset()
        st.success("All dashboard overrides removed; the bot is back on its .env / built-in defaults.")
        st.cache_data.clear()
        st.rerun()
    st.caption("• = set from this dashboard (overrides .env).")

# ----------------------------------------------------------------------------
# Open positions
# ----------------------------------------------------------------------------
with tab_pos:
    pos = data["positions"]
    if pos.empty:
        st.info("No open positions.")
    else:
        a, b, cc = st.columns(3)
        a.metric("Open positions", len(pos))
        b.metric("Capital at risk", f"${pos['cost'].sum():,.2f}")
        cc.metric("Max payout", f"${pos['max_payout'].sum():,.2f}")
        st.dataframe(pos.assign(niche=pos["niche"].map(lambda n: NICHE_LABELS.get(n, n))),
                     hide_index=True, use_container_width=True,
                     column_config={"entry_price": st.column_config.NumberColumn("entry", format="%.3f"),
                                    "cost": st.column_config.NumberColumn(format="dollar"),
                                    "max_payout": st.column_config.NumberColumn("pays if right", format="dollar")})

# ----------------------------------------------------------------------------
# Learning
# ----------------------------------------------------------------------------
with tab_learn:
    lr = data["learning"]
    st.caption("Every prediction is graded when its market settles. Lower Brier = more accurate. "
               "The bot earns trust (and keeps trading a niche) only by beating the market's own price.")
    if lr.empty:
        st.info("No predictions recorded yet.")
    else:
        st.dataframe(
            lr.assign(niche=lr["niche"].map(lambda n: NICHE_LABELS.get(n, n))),
            hide_index=True, use_container_width=True,
            column_config={
                "settled_events": st.column_config.NumberColumn("settled events"),
                "waiting_markets": st.column_config.NumberColumn("waiting on results"),
                "paused_by_learning": st.column_config.CheckboxColumn("auto-paused"),
                "bot_brier": st.column_config.NumberColumn("bot Brier", format="%.3f"),
                "market_brier": st.column_config.NumberColumn("market Brier", format="%.3f"),
                "trust_in_bot": st.column_config.NumberColumn("trust in bot", format="percent"),
                "rt_drift_pts": st.column_config.NumberColumn("RT drift (pts)", format="%+.1f"),
            },
        )

    st.markdown("#### If you had followed every bot bet")
    st.caption(f"Each settled market where the bot disagreed with the market by at least "
               f"{100 * float(cfg['MIN_EDGE']):.0f} pts (your Min edge): buy the side the bot liked at the "
               "market price, pay Kalshi's fee, hold to settlement. No other gates. Positive ¢/contract = "
               "the bot's disagreements were right more often than the price implied; negative = the market "
               "was right. 'Underdog' = bets at 25¢ or less, where the losses so far have come from.")
    wht = data["wht"]
    if wht.empty:
        st.info("No settled disagreements yet.")
    else:
        st.dataframe(
            wht.assign(niche=wht["niche"].map(lambda n: NICHE_LABELS.get(n, n))),
            hide_index=True, use_container_width=True,
            column_config={
                "bets": st.column_config.NumberColumn("bets"),
                "events": st.column_config.NumberColumn("events"),
                "win_rate": st.column_config.NumberColumn("win rate", format="percent"),
                "avg_price": st.column_config.NumberColumn("avg price paid", format="%.2f"),
                "cents_per_contract": st.column_config.NumberColumn("¢ / contract", format="%+.1f"),
                "dollars_per_contract_each": st.column_config.NumberColumn("$ if 1 contract each",
                                                                           format="%+.2f"),
                "underdog_bets": st.column_config.NumberColumn("underdog bets"),
                "underdog_cents": st.column_config.NumberColumn("underdog ¢ / contract", format="%+.1f"),
            },
        )

# ----------------------------------------------------------------------------
# Predictions
# ----------------------------------------------------------------------------
with tab_pred:
    pr = data["predictions"]
    if pr.empty:
        st.info("No predictions recorded yet.")
    else:
        pr = pr.assign(gap_pts=100 * (pr["our_prob"] - pr["market_prob"]))
        st.dataframe(
            pr[["time", "niche", "market_id", "our_prob", "market_prob", "gap_pts", "confidence", "result"]],
            hide_index=True, use_container_width=True,
            column_config={
                "our_prob": st.column_config.NumberColumn("bot", format="percent"),
                "market_prob": st.column_config.NumberColumn("market", format="percent"),
                "gap_pts": st.column_config.NumberColumn("gap (pts)", format="%+.0f"),
                "confidence": st.column_config.NumberColumn(format="percent"),
            },
        )

# ----------------------------------------------------------------------------
# Maker test
# ----------------------------------------------------------------------------
with tab_maker:
    st.caption("Patient favorite buyer (paper only). Posts buy orders at the bid on the favorite side (55-95c) "
               "of markets settling within 2 days and waits for a seller. An order only counts as filled if "
               "Kalshi's public trade tape shows someone actually sold at or below our price after we posted, "
               "past the queue ahead of us. Held to settlement; maker fee charged on every fill. "
               "Runs as its own process: tmux session 'maker'.")
    ver = st.radio("Version", ["v1: original", "v2: before the event, first in line, one per match"],
                   horizontal=True, key="maker_ver")
    v2 = ver.startswith("v2")
    if v2:
        st.caption("v2 changes, from v1's record: at most one bet per match; only before the event (expected end "
                   "4h+ away, orders pulled under 3h); only where the bid/ask gap is 3c+ so it's first in line; "
                   "orders pulled when our side's price falls 2c or the bid drops below ours; checks every 15s. "
                   "tmux session 'maker2'.")
    mk = data["maker_v2" if v2 else "maker"]
    rep = mk["report"]
    if not rep or not rep["posted"]:
        st.info("No orders yet for this version. On the server: tmux session "
                + ("'maker2' running `python cli.py maker --variant v2`." if v2 else "'maker' running `python cli.py maker`."))
    else:
        a, b, cc, d, e, f = st.columns(6)
        a.metric("Orders posted", rep["posted"], f"{rep['open']} resting now", delta_color="off")
        b.metric("Filled", rep["filled"], f"{rep['fill_rate']:.0%} of finished orders", delta_color="off")
        cc.metric("Settled", rep["settled"], f"{rep['awaiting']} awaiting result", delta_color="off")
        d.metric("Win rate", f"{rep['win_rate']:.0%}" if rep["win_rate"] is not None else "–",
                 f"avg price {rep['avg_price']:.2f}" if rep["avg_price"] else None, delta_color="off")
        e.metric("¢ / contract", f"{rep['c_per_contract']:+.2f}¢" if rep["c_per_contract"] is not None else "–",
                 f"{rep['return_on_cost']:+.1%} on money spent" if rep["return_on_cost"] is not None else None,
                 delta_color="off")
        f.metric("Total P&L", money(rep["pnl"]), f"{rep['events']} events", delta_color="off")
        st.caption("Judge it on ¢/contract after ~200 settled fills across many events. Positive and steady = "
                   "an edge worth considering; around zero or negative = no edge. Win rate alone means little: "
                   "buying 80c favorites should win ~80% just to break even.")
        bk = pd.DataFrame(rep["buckets"])
        bk = bk[bk["settled"] > 0]
        if not bk.empty:
            st.markdown("#### By price paid")
            st.dataframe(bk, hide_index=True, use_container_width=True,
                         column_config={"win rate": st.column_config.NumberColumn(format="percent"),
                                        "avg price": st.column_config.NumberColumn(format="%.2f"),
                                        "c_per_contract": st.column_config.NumberColumn("¢ / contract",
                                                                                        format="%+.2f"),
                                        "pnl": st.column_config.NumberColumn("P&L", format="dollar")})
        od = mk["orders"]
        settled = od[od["status"] == "settled"].sort_values("settled", ascending=False)
        if not settled.empty:
            sp = settled.assign(cum=settled.sort_values("settled")["pnl"].cumsum())
            fig = go.Figure(go.Scatter(x=sp.sort_values("settled")["settled"], y=sp.sort_values("settled")["cum"],
                                       mode="lines", line=dict(width=2)))
            st.markdown("#### Equity (settled maker fills)")
            st.plotly_chart(chart_layout(fig, 260), use_container_width=True)
        # ---- slice and dice ----------------------------------------------------
        from src import maker_analysis as MA
        feats = mk["features"]
        st.markdown("#### Slice the results")
        st.caption("Pick a way to split the settled bets. Each row says whether that group is clearly making "
                   "money, clearly losing, or still too noisy to tell (the ± margin shrinks as more events "
                   "settle). Look for a group that stays 'clearly positive' as data piles up.")
        dim_label = st.selectbox("Split by", list(MA.DIMENSIONS), index=0, key=f"maker_dim_{'v2' if v2 else 'v1'}")
        tbl = MA.slice_table(feats, MA.DIMENSIONS[dim_label])
        if tbl.empty:
            st.info("No settled bets yet.")
        else:
            st.dataframe(tbl, hide_index=True, use_container_width=True,
                         column_config={"win rate": st.column_config.NumberColumn(format="percent"),
                                        "avg price": st.column_config.NumberColumn(format="%.2f"),
                                        "c_per_contract": st.column_config.NumberColumn("¢ / contract",
                                                                                        format="%+.2f"),
                                        "margin": st.column_config.NumberColumn("± margin (¢)", format="%.1f"),
                                        "pnl": st.column_config.NumberColumn("P&L", format="dollar")})
            top = tbl.head(15)
            fig = go.Figure(go.Bar(x=top["group"].astype(str), y=top["c_per_contract"],
                                   error_y=dict(type="data", array=top["margin"], visible=True),
                                   marker_color=["#2ecc71" if v > 0 else "#e74c3c" for v in top["c_per_contract"]]))
            st.plotly_chart(chart_layout(fig, 280), use_container_width=True)
        st.markdown("#### Search every split for an edge")
        st.caption("Tries every split above that is known when the order is posted (not hindsight like 'how it "
                   "filled'), and every PAIR of them, e.g. 'sports × evening'. Trying hundreds of groups always "
                   "turns up a few that look good by luck, so a group only counts if it survives that.")
        show_search(_maker_search("v2" if v2 else "v1"), "bets", key=f"search_{'v2' if v2 else 'v1'}")
        if not feats.empty:
            st.download_button("Download every order (CSV)", feats.to_csv(index=False).encode(),
                               file_name=f"maker_{'v2' if v2 else 'v1'}_orders.csv", mime="text/csv",
                               key=f"dl_{'v2' if v2 else 'v1'}")

        st.markdown("#### Orders")
        show = od.sort_values("posted_ts", ascending=False)[
            ["posted", "status", "ticker", "title", "side", "price", "qty", "queue_ahead", "fill", "result", "pnl",
             "note"]].head(300)
        st.dataframe(show, hide_index=True, use_container_width=True,
                     column_config={"price": st.column_config.NumberColumn(format="%.2f"),
                                    "queue_ahead": st.column_config.NumberColumn("queue ahead", format="%.0f"),
                                    "pnl": st.column_config.NumberColumn("P&L", format="dollar")})

# ----------------------------------------------------------------------------
# Favorites paper test (separate study)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=120, max_entries=1)
def _fav():
    from src import favorites as F
    return F.report()


with tab_fav:
    from src import favorites as F
    st.caption("Paper only, separate from every other test. Buys (on paper) the favorite side of markets closing "
               "in 6-24 hours when it costs 90-98c, one bet per market, at the real ask plus Kalshi's fee. "
               "A second arm only buys after a $1,000+ trade on that favorite, one minute later. The idea came "
               "from the whale study's split search; this checks it on fresh data only, under rules fixed in "
               "advance. Runs as tmux session 'fav'.")
    fr = _fav()
    if not fr.get("started"):
        st.info("Not started yet. On the server: tmux session 'fav' running `python cli.py favorites`.")
    else:
        st.markdown(f"**Day {fr['days']:.1f} of {F.VERDICT['min_days']}** (original two arms) · about "
                    f"{fr['qualifying_per_scan']:.0f} markets qualify per scan. Verdict rule (fixed in advance): "
                    f"after {F.VERDICT['min_days']} days and {F.VERDICT['min_events']} separate settled events, an arm "
                    "passes only if its ¢/contract is above zero by more than its ± margin.")
        for arm, label in F.ARMS.items():
            sa = fr["arms"][arm]
            st.markdown(f"#### {label}")
            if arm in F.NEW_ARMS and (fr.get("arm_started") or {}).get(arm):
                st.caption(f"Added {datetime.fromtimestamp(fr['arm_started'][arm]).strftime('%b %d')} from the "
                           f"one-week check-in; judged only on bets from that day on (day "
                           f"{fr['arm_days'][arm]:.1f} of {F.VERDICT['min_days']}).")
            a, b, cc, d, e = st.columns(5)
            a.metric("Paper bets", f"{sa['bets']:,}", f"{sa['settled']:,} settled", delta_color="off")
            b.metric("Separate events", f"{sa['events']:,}", f"of {F.VERDICT['min_events']} needed", delta_color="off")
            if sa.get("win_rate") is not None:
                cc.metric("Win rate", f"{sa['win_rate']:.1%}", f"needs {sa['breakeven']:.1%} to break even",
                          delta_color="off")
                d.metric("¢ / contract", f"{sa['c_per_contract']:+.2f}¢",
                         f"± {sa['margin']:.2f}¢" if sa.get("margin") is not None else None, delta_color="off")
                e.metric("Losses", f"{sa['losses']:,}", f"P&L at 10/bet: {money(sa['pnl_10'])}", delta_color="off")
            if sa["verdict"].startswith("PASSED"):
                st.success(f"Verdict: {sa['verdict']}")
            elif sa["verdict"].startswith("losing"):
                st.error(f"Verdict: {sa['verdict']}")
            else:
                st.info(f"Verdict: {sa['verdict']}")
        st.caption("Win rate vs. break-even is the whole story here: at ~95c you win ~5c most of the time and lose "
                   "~95c when the favorite fails, so one loss erases ~19 wins. Wait for the event count before "
                   "trusting any number.")
        df = fr["df"]
        done = df[df["result"].isin(["yes", "no"])].sort_values("settled_ts") if not df.empty else df
        if not done.empty:
            st.markdown("#### Equity (per contract, settled bets)")
            fig = go.Figure()
            for arm, color in (("favorites", ACCENT), ("whale", "#2ecc71"), ("fav_93_95", "#f39c12"), ("fav_1_6h", "#3498db")):
                g = done[done["arm"] == arm]
                if not g.empty:
                    fig.add_trace(go.Scatter(x=g["settled_et"], y=g["c"].cumsum() / 100, mode="lines", name=F.ARMS[arm],
                                             line=dict(width=2, color=color)))
            st.plotly_chart(chart_layout(fig, 260).update_layout(showlegend=True), use_container_width=True)

            # ---- what real money would be now (paper replay) ----------------------
            st.markdown("#### If you'd started with real money")
            st.caption(f"A replay of every paper bet, in order, starting with each amount. Each bet uses up to "
                       f"{F.BET_SHARE:.0%} of the starting amount, capped by the contracts actually for sale at our "
                       "price and by the cash on hand (money comes back when a market settles; a bet with no cash "
                       "free is skipped). Kalshi's fee is charged for each order's size. Open bets count at cost. "
                       "Real orders usually do a bit worse than paper.")
            bk = fr.get("bankroll") or {}
            rows_bk = []
            for arm, label in F.ARMS.items():
                for b, r in (bk.get(arm) or {}).items():
                    rows_bk.append({"arm": label, "started with": r["start"], "now": r["value"], "P&L": r["pnl"],
                                    "return": r["ret"], "worst drop": -r["max_drop"],
                                    "bets taken": r["taken"], "skipped (cash tied up)": r["skipped"],
                                    "avg contracts per bet": r["contracts_per_bet"], "still open": r["open"]})
            if rows_bk:
                st.dataframe(pd.DataFrame(rows_bk), hide_index=True, use_container_width=True,
                             column_config={"started with": st.column_config.NumberColumn(format="dollar"),
                                            "now": st.column_config.NumberColumn(format="dollar"),
                                            "P&L": st.column_config.NumberColumn(format="dollar"),
                                            "return": st.column_config.NumberColumn(format="percent"),
                                            "worst drop": st.column_config.NumberColumn(format="dollar"),
                                            "avg contracts per bet": st.column_config.NumberColumn(format="%.1f")})
                pick_b = st.radio("Show value over time for", list(F.BANKROLLS), horizontal=True,
                                  format_func=lambda b: f"${b:,}", key="fav_bankroll")
                fig = go.Figure()
                for arm, color in (("favorites", ACCENT), ("whale", "#2ecc71"), ("fav_93_95", "#f39c12"), ("fav_1_6h", "#3498db")):
                    cv = (bk.get(arm) or {}).get(pick_b, {}).get("curve") or []
                    if cv:
                        xs = pd.to_datetime([t for t, _ in cv], unit="s", utc=True).tz_convert("America/New_York").tz_localize(None)
                        fig.add_trace(go.Scatter(x=xs, y=[v for _, v in cv], mode="lines", name=F.ARMS[arm],
                                                 line=dict(width=2, color=color, shape="hv")))
                fig.add_hline(y=pick_b, line_dash="dot", line_color="rgba(128,128,128,0.6)")
                st.plotly_chart(chart_layout(fig, 280).update_layout(showlegend=True, yaxis_tickprefix="$"),
                                use_container_width=True)
            st.markdown("#### Split the results")
            arm_pick = st.radio("Arm", list(F.ARMS), format_func=lambda a: F.ARMS[a], horizontal=True, key="fav_arm")
            sp = fr["splits"].get(arm_pick) or {}
            if sp:
                lab = st.selectbox("Split by", list(sp), key="fav_split")
                t = sp[lab]
                if not t.empty:
                    st.dataframe(t, hide_index=True, use_container_width=True,
                                 column_config={"win_rate": st.column_config.NumberColumn("win rate", format="percent"),
                                                "avg_price": st.column_config.NumberColumn("avg price", format="%.3f"),
                                                "c_per_contract": st.column_config.NumberColumn("¢ / contract", format="%+.2f"),
                                                "margin": st.column_config.NumberColumn("± (¢)", format="%.2f")})
        if not df.empty:
            st.markdown("#### Latest paper bets")
            show = df.sort_values("entered_ts", ascending=False)[
                ["entered_et", "arm", "ticker", "title", "side", "price", "hours_left", "depth", "result", "c"]].head(200)
            st.dataframe(show, hide_index=True, use_container_width=True,
                         column_config={"price": st.column_config.NumberColumn(format="%.2f"),
                                        "hours_left": st.column_config.NumberColumn("hours to close", format="%.1f"),
                                        "depth": st.column_config.NumberColumn("buyable at price", format="%.0f"),
                                        "c": st.column_config.NumberColumn("result (¢)", format="%+.1f")})

# ----------------------------------------------------------------------------
# Polymarket price check (separate study)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=120, max_entries=1)
def _poly():
    return D.poly_check()


with tab_poly:
    st.caption("Measurement only, no trading on either site, separate from every other test. Matches Kalshi "
               "game-winner markets (NFL, NBA, NHL, MLB, college football) with the same game on Polymarket and "
               "records both prices: every 5 seconds from an hour before a game until it ends. Asks: is Polymarket's "
               "price a better guide, how fast does Kalshi follow when Polymarket moves, and would buying on KALSHI "
               "when its price sits below Polymarket's have paid after fees? Runs as tmux session 'poly'; numbers "
               "refresh every 10 minutes.")
    pr = _poly()
    if not pr:
        st.info("No report yet. On the server: tmux session 'poly' running `python cli.py poly-check` "
                "(first try `python cli.py poly-check --probe` to check the matching).")
    else:
        a, b, cc, d = st.columns(4)
        a.metric("Games matched", f"{pr.get('games_matched', 0):,}")
        b.metric("Games settled", f"{pr.get('games_settled', 0):,}")
        cc.metric("Samples", f"{pr.get('samples', 0):,}")
        d.metric("Hours recorded", f"{pr.get('hours', 0):.1f}")
        disc = pr.get("discovery") or {}
        if disc:
            st.caption("Latest matching: " + " · ".join(
                f"{lg.upper()} {s_.get('matched', 0)} of {s_.get('poly_games', 0)}" if "error" not in s_
                else f"{lg.upper()} error" for lg, s_ in disc.items()))
        pv = pr.get("preview") or {}
        if pv.get("samples_with_both_prices"):
            st.markdown("#### So far (before any game has finished)")
            st.caption("From every moment both sites had a price. Results (sections 1-3) need finished games: each "
                       "is scored about 2 hours after it starts, once Kalshi posts the result.")
            a, b, cc, d = st.columns(4)
            a.metric("Games with prices", f"{pv['games_sampled']:,}", f"{pv['games_sampled_live']} recorded live",
                     delta_color="off")
            b.metric("Typical difference", f"{pv['median_diff_c']:.1f}¢", "Kalshi vs Polymarket middle price",
                     delta_color="off")
            cc.metric("Kalshi 2¢+ cheaper", f"{pv['share_gap_2c']:.1%}", "of moments, after Kalshi's fee",
                      delta_color="off")
            d.metric("Kalshi 5¢+ cheaper", f"{pv['share_gap_5c']:.1%}", "of moments, after Kalshi's fee",
                     delta_color="off")
        st.markdown("#### 1. Which price predicts the winner better?")
        st.caption("Lower Brier = more accurate. If Polymarket is clearly lower, it's a better 'fair price' than "
                   "Kalshi's own, and the gaps below may be real.")
        if pr.get("calibration"):
            st.dataframe(pd.DataFrame(pr["calibration"]), hide_index=True, use_container_width=True,
                         column_config={"Polymarket Brier": st.column_config.NumberColumn(format="%.4f"),
                                        "Kalshi Brier": st.column_config.NumberColumn(format="%.4f")})
        else:
            st.info("Waiting for the first matched games to finish.")
        st.markdown("#### 2. When Polymarket jumps 3c+, how fast does Kalshi follow?")
        if pr.get("lag") and pr.get("jumps"):
            st.dataframe(pd.DataFrame(pr["lag"]), hide_index=True, use_container_width=True,
                         column_config={"share Kalshi has followed": st.column_config.NumberColumn(format="percent"),
                                        "share Polymarket kept": st.column_config.NumberColumn(format="percent")})
        st.markdown("#### 3. Buying on Kalshi when it's cheaper than Polymarket")
        st.caption("For each gap: buy on Kalshi at the ask seen 0, 5, 15 or 30 seconds later, pay the fee, hold to "
                   "the result. ± counts each game once. Positive at 5-15s late = something a home-built bot could catch.")
        if pr.get("gaps"):
            cfg_ = {c: st.column_config.NumberColumn(format="%+.2f") for c in
                    ("0s late: c/contract", "5s late: c/contract", "15s late: c/contract", "30s late: c/contract")}
            cfg_.update({c: st.column_config.NumberColumn(format="%.2f") for c in
                         ("0s late: ± (c)", "5s late: ± (c)", "15s late: ± (c)", "30s late: ± (c)")})
            st.dataframe(pd.DataFrame(pr["gaps"]), hide_index=True, use_container_width=True, column_config=cfg_)

# ----------------------------------------------------------------------------
# Speed gaps (15-minute crypto study)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=600, max_entries=1)
def _gap_report():
    return D.gap_study()


with tab_gap:
    st.caption("Measurement only, no trading. Every second: bitcoin/ether/solana from Coinbase, Kraken and "
               "Bitstamp, and Kalshi's 15-minute up/down markets. A 'gap' is Kalshi's ask sitting below our fair "
               "value by more than the taker fee. The question: do gaps last long enough, and pay enough, for a "
               "bot that reacts in 1-3 seconds? Runs as tmux session 'gap', which updates these numbers every 10 minutes.")
    g = _gap_report()
    if not g or not g["samples"]:
        st.info("No report yet. The gap study (tmux session 'gap', `python cli.py gap-study`) writes one "
                "10 minutes after it starts.")
    else:
        a, b, cc = st.columns(3)
        a.metric("Hours recorded", f"{g['hours']:.1f}")
        b.metric("Samples", f"{g['samples']:,}")
        cc.metric("Markets settled", g["markets_settled"])
        st.markdown("#### 1. Is our fair value as accurate as Kalshi's own price?")
        st.caption("Lower Brier = more accurate. If ours is worse than Kalshi's, the 'gaps' below are mostly our "
                   "own error, not real opportunities.")
        if g["calibration"]:
            st.dataframe(pd.DataFrame(g["calibration"]), hide_index=True, use_container_width=True,
                         column_config={"our model Brier": st.column_config.NumberColumn(format="%.4f"),
                                        "Kalshi price Brier": st.column_config.NumberColumn(format="%.4f")})
        st.markdown("#### 2. Gaps, and what buying them would have earned")
        st.caption("For each gap: buy the cheap side at the ask seen 0, 1, 3 or 10 seconds after the gap "
                   "appeared, pay the fee, hold to settlement. Positive at 1-3s late = an edge a fast home-built "
                   "bot could chase. Positive only at 0s = only the fastest pros can get it.")
        gap_cfg = {c: st.column_config.NumberColumn(format="%+.2f") for c in
                   ("0s late: c/contract", "1s late: c/contract", "3s late: c/contract", "10s late: c/contract")}
        gap_cfg.update({c: st.column_config.NumberColumn(format="%.2f") for c in
                        ("0s late: ± (c)", "1s late: ± (c)", "3s late: ± (c)", "10s late: ± (c)")})
        gap_cfg.update({"per day": st.column_config.NumberColumn(format="%.0f"),
                        "median seconds": st.column_config.NumberColumn(format="%.1f"),
                        "share lasting 3s+": st.column_config.NumberColumn(format="percent")})
        if g["gaps"]:
            st.caption("± counts every gap on the same 15-minute market together (they share one outcome), so it's "
                       "wider than a naive count would give, and honest.")
            st.dataframe(pd.DataFrame(g["gaps"]), hide_index=True, use_container_width=True, column_config=gap_cfg)
        if g.get("lag"):
            st.markdown("#### 3. How fast does Kalshi catch up when the exchanges jump?")
            st.caption(f"{g.get('jumps', 0):,} times our fair value jumped 4c+ in one second. For each: how much of "
                       "that jump Kalshi's price had absorbed N seconds later, and how much of it our own fair "
                       "value kept (if ours falls back, the jump was exchange noise, not news). A speed edge needs "
                       "Kalshi to stay well behind for 1-3 seconds while ours holds.")
            st.dataframe(pd.DataFrame(g["lag"]), hide_index=True, use_container_width=True,
                         column_config={"share Kalshi has priced in": st.column_config.NumberColumn(format="percent"),
                                        "share our fair value kept": st.column_config.NumberColumn(format="percent")})
        if g.get("fast_gaps"):
            st.markdown("#### 4. Pure speed gaps (Kalshi's own price + the exchanges' last 5 seconds)")
            st.caption("Here the 'fair value' is Kalshi's own price 5 seconds ago, moved by however much the "
                       "exchanges moved since. That removes our model's general disagreement with Kalshi: a gap only "
                       "shows up when the exchanges just moved and Kalshi hasn't. This is the Dexter's Lab trade "
                       "in its purest form.")
            st.dataframe(pd.DataFrame(g["fast_gaps"]), hide_index=True, use_container_width=True, column_config=gap_cfg)
        if g.get("gaps_split"):
            st.markdown("#### 5. The 5c gaps, split by coin, time left and side")
            st.dataframe(pd.DataFrame(g["gaps_split"]), hide_index=True, use_container_width=True,
                         column_config=gap_cfg)

# ----------------------------------------------------------------------------
# Whale follow (separate study)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=120, max_entries=1)
def _whale():
    return D.whale_study()


with tab_whale:
    st.caption("Measurement only, no trading, separate from every other test. Watches Kalshi's whole public "
               "trade tape for unusually large trades ($1,000+), then asks: if you had copied each one ONE MINUTE "
               "LATE, at the real ask plus the taker fee, and held to settlement, would you have made money? "
               "Runs as tmux session 'whale'.")
    wd = _whale()
    sm, wl = wd["summary"], wd["latest"]
    if not sm.get("events"):
        st.info("No whale report yet. The whale study (tmux session 'whale', `python cli.py whale-study`) writes "
                "one when it starts and every 10 minutes after.")
    else:
        if wd.get("computed_ts"):
            st.caption(f"Numbers as of {datetime.fromtimestamp(wd['computed_ts']).strftime('%a %H:%M')} "
                       "(the study refreshes them every 10 minutes).")
        a, b, cc, d, e = st.columns(5)
        a.metric("Whale trades seen", f"{sm['events']:,}")
        b.metric("Settled", f"{sm.get('settled', 0):,}", f"{sm.get('markets', 0)} markets", delta_color="off")
        cc.metric("Copy ¢ / contract", f"{sm['copy_c']:+.2f}¢" if sm.get("copy_c") is not None else "–",
                  f"± {sm['margin']:.1f}¢" if sm.get("margin") else None, delta_color="off")
        d.metric("Win rate", f"{sm['win_rate']:.0%}" if sm.get("win_rate") is not None else "–",
                 f"avg copy price {sm['avg_entry']:.2f}" if sm.get("avg_entry") else None, delta_color="off")
        e.metric("Price 30 min later", f"{sm['drift_30m']:+.2f}¢" if sm.get("drift_30m") is not None else "–",
                 "vs our copy price", delta_color="off")
        st.caption("'Price 30 min later' above zero means prices keep drifting the whale's way after you could "
                   "have copied: the signal survives the delay. Copy ¢/contract is the bottom line after fees.")
        st.markdown("#### Slice the results")
        slices = wd["slices"]
        if slices:
            wdim = st.selectbox("Split by", list(slices), index=0, key="whale_dim")
            wt = slices[wdim]
            if wt.empty:
                st.info("No settled whale trades yet. Most markets take hours to days to settle.")
            else:
                st.dataframe(wt, hide_index=True, use_container_width=True,
                             column_config={"win rate": st.column_config.NumberColumn(format="percent"),
                                            "avg copy price": st.column_config.NumberColumn(format="%.2f"),
                                            "c_per_contract": st.column_config.NumberColumn("¢ / contract", format="%+.2f"),
                                            "margin": st.column_config.NumberColumn("± margin (¢)", format="%.1f"),
                                            "drift 30m (c)": st.column_config.NumberColumn(format="%+.2f")})
        st.markdown("#### Search every split for an edge")
        st.caption("Every split above, and every pair of them (e.g. '$25k+ × sports'), each group needing 50+ "
                   "separate markets. Judged for how many groups were tried.")
        show_search(wd.get("search"), "whale trades", key="search_whale")
        if wd.get("bankroll"):
            st.markdown("#### If you'd copied the surviving groups with real money")
            st.caption("Every whale trade in each surviving group, copied one minute late at the real ask, replayed in "
                       "order from the start of the study: 5% of the starting amount per copy, cash returning at "
                       "settlement, Kalshi's fee per order size. Settled copies only. CAUTION: these groups were "
                       "found by searching this same data, so this flatters them; the Favorites test tab is the "
                       "honest check on fresh data.")
            rows_g = []
            for gb in wd["bankroll"]:
                for b, r in gb["sims"].items():
                    rows_g.append({"group": f"{gb['split']}: {gb['group']}", "started with": r["start"],
                                   "now": r["value"], "P&L": r["pnl"], "return": r["ret"], "worst drop": -r["max_drop"],
                                   "copies taken": r["taken"], "skipped (cash tied up)": r["skipped"],
                                   "avg contracts": r["contracts_per_bet"]})
            st.dataframe(pd.DataFrame(rows_g), hide_index=True, use_container_width=True,
                         column_config={"started with": st.column_config.NumberColumn(format="dollar"),
                                        "now": st.column_config.NumberColumn(format="dollar"),
                                        "P&L": st.column_config.NumberColumn(format="dollar"),
                                        "return": st.column_config.NumberColumn(format="percent"),
                                        "worst drop": st.column_config.NumberColumn(format="dollar"),
                                        "avg contracts": st.column_config.NumberColumn(format="%.1f")})
        if not wl.empty:
            st.markdown("#### Latest whale trades")
            st.dataframe(wl, hide_index=True, use_container_width=True,
                         column_config={"notional": st.column_config.NumberColumn("size ($)", format="dollar"),
                                        "vwap": st.column_config.NumberColumn("whale price", format="%.2f"),
                                        "entry_ask": st.column_config.NumberColumn("copy price (+1 min)", format="%.2f"),
                                        "drift_30m_c": st.column_config.NumberColumn("30 min later (c)", format="%+.1f"),
                                        "copy_c": st.column_config.NumberColumn("copy result (c)", format="%+.1f")})

# ----------------------------------------------------------------------------
# Price inconsistencies (separate study)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=300, max_entries=1)
def _arb():
    return D.arb_scan()


with tab_arb:
    st.caption("Measurement only, no trading, separate from every other test. Every 2 minutes it checks every "
               "open Kalshi event for prices that can't all be right: buying every NO (or every YES) in a "
               "one-winner event, or two rungs of an 'above/below X' ladder, for less than the guaranteed payout "
               "after fees. 'Sets available' = how many it could actually have bought at those prices. "
               "Runs as tmux session 'arb'.")
    ar = _arb()
    if not ar.get("scans"):
        st.info("No scans yet. On the server: tmux session 'arb' running `python cli.py arb-scan`.")
    else:
        a, b, cc = st.columns(3)
        a.metric("Scans", f"{ar['scans']:,}", f"{ar['hours']:.1f} hours", delta_color="off")
        b.metric("Events per scan", f"{ar['events_per_scan']:,.0f}", f"{ar['scan_seconds']:.0f}s per scan",
                 delta_color="off")
        cc.metric("Inconsistencies found", f"{ar.get('total', 0):,}")
        st.caption("'all YES ... list may be incomplete' is NOT riskless (an unlisted outcome could win), so "
                   "judge only the other kinds. A real opportunity = a decent edge, 10+ sets available, lasting "
                   "more than a scan or two, AND paying out soon: 10c locked up for a year is a 10%/year return, "
                   "not free money.")
        if not ar["by_kind"].empty:
            st.dataframe(ar["by_kind"], hide_index=True, use_container_width=True,
                         column_config={"per_day": st.column_config.NumberColumn("per day", format="%.1f"),
                                        "median_edge_c": st.column_config.NumberColumn("median edge (c/set)", format="%.1f"),
                                        "median_minutes": st.column_config.NumberColumn("median minutes open", format="%.1f"),
                                        "median_days_to_payout": st.column_config.NumberColumn("median days to payout", format="%.0f"),
                                        "with_10plus_sets": st.column_config.NumberColumn("with 10+ sets"),
                                        "profit_if_all_taken": st.column_config.NumberColumn("profit if all taken", format="dollar")})
        st.markdown("#### Worth a look")
        wk = ar.get("worth")
        if wk is None or wk.empty:
            st.info("Nothing yet that clears every bar: 2c+ edge per set, 10+ sets actually buyable, still there on "
                    "a second scan, paying out within a month, and not the 'list may be incomplete' kind.")
        else:
            st.dataframe(wk, hide_index=True, use_container_width=True,
                         column_config={"edge_c": st.column_config.NumberColumn("edge (c/set)", format="%.1f"),
                                        "contracts": st.column_config.NumberColumn("sets available", format="%.0f"),
                                        "days_to_payout": st.column_config.NumberColumn("pays out in (days)", format="%.0f"),
                                        "lasted_min": st.column_config.NumberColumn("lasted (min)", format="%.1f"),
                                        "profit_if_taken": st.column_config.NumberColumn("profit if taken", format="dollar")})
        if not ar["recent"].empty:
            st.markdown("#### Latest")
            st.dataframe(ar["recent"], hide_index=True, use_container_width=True,
                         column_config={"edge_c": st.column_config.NumberColumn("edge (c/set)", format="%.1f"),
                                        "contracts": st.column_config.NumberColumn("sets available", format="%.0f"),
                                        "days_to_payout": st.column_config.NumberColumn("pays out in (days)", format="%.0f"),
                                        "lasted_min": st.column_config.NumberColumn("lasted (min)", format="%.1f")})

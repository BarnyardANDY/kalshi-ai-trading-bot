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
else:
    s1.metric("Bot", "⚠ stalled?", f"last activity {mins:.0f} min ago", delta_color="off")
s2.metric("Mode", {"paper": "PAPER", "LIVE": "LIVE $"}.get(status["mode"], "unknown"))
s3.metric("New trades", "PAUSED" if cfg["PAUSED"] else "ON")
ai = data["ai"]
cap = cfg["DAILY_AI_COST_LIMIT"]
s4.metric("AI spend today", f"\\${ai['cost']:.2f} of \\${cap:.2f}", f"{ai['requests']} requests", delta_color="off")

tab_perf, tab_maker, tab_gap, tab_whale, tab_arb, tab_ctrl, tab_pos, tab_learn, tab_pred = st.tabs(
    ["Performance", "Maker test", "Speed gaps", "Whale follow", "Price inconsistencies", "Strategy controls",
     "Open positions", "Learning", "Predictions"])

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
        if g["gaps"]:
            st.dataframe(pd.DataFrame(g["gaps"]), hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------
# Whale follow (separate study)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=600, max_entries=1)
def _whale():
    return D.whale_study()


with tab_whale:
    st.caption("Measurement only, no trading, separate from every other test. Watches Kalshi's whole public "
               "trade tape for unusually large trades ($1,000+), then asks: if you had copied each one ONE MINUTE "
               "LATE, at the real ask plus the taker fee, and held to settlement, would you have made money? "
               "Runs as tmux session 'whale'.")
    wd = _whale()
    sm, wf = wd["summary"], wd["features"]
    if not sm.get("events"):
        st.info("No whale trades recorded yet. On the server: tmux session 'whale' running "
                "`python cli.py whale-study`.")
    else:
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
        from src import whale as W
        st.markdown("#### Slice the results")
        wdim = st.selectbox("Split by", list(W.DIMENSIONS), index=0, key="whale_dim")
        wt = W.slice_table(wf, W.DIMENSIONS[wdim])
        if wt.empty:
            st.info("No settled whale trades yet. Most markets take hours to days to settle.")
        else:
            st.dataframe(wt, hide_index=True, use_container_width=True,
                         column_config={"win rate": st.column_config.NumberColumn(format="percent"),
                                        "avg copy price": st.column_config.NumberColumn(format="%.2f"),
                                        "c_per_contract": st.column_config.NumberColumn("¢ / contract", format="%+.2f"),
                                        "margin": st.column_config.NumberColumn("± margin (¢)", format="%.1f"),
                                        "drift 30m (c)": st.column_config.NumberColumn(format="%+.2f")})
        st.markdown("#### Latest whale trades")
        show = wf.sort_values("ts", ascending=False)[["time_et", "ticker", "title", "side", "contracts", "notional",
                                                        "vwap", "entry_ask", "drift_30m_c", "result", "copy_c"]].head(200)
        st.dataframe(show, hide_index=True, use_container_width=True,
                     column_config={"notional": st.column_config.NumberColumn("size ($)", format="dollar"),
                                    "vwap": st.column_config.NumberColumn("whale price", format="%.2f"),
                                    "entry_ask": st.column_config.NumberColumn("copy price (+1 min)", format="%.2f"),
                                    "drift_30m_c": st.column_config.NumberColumn("30 min later (c)", format="%+.1f"),
                                    "copy_c": st.column_config.NumberColumn("copy result (c)", format="%+.1f")})
        st.download_button("Download whale trades (CSV)", wf.to_csv(index=False).encode(),
                           file_name="whale_trades.csv", mime="text/csv", key="dl_whale")

# ----------------------------------------------------------------------------
# Price inconsistencies (separate study)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=300, max_entries=1)
def _arb():
    return D.arb_scan()


with tab_arb:
    st.caption("Measurement only, no trading, separate from every other test. Every 30 seconds it checks every "
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
                                        "with_10plus_sets": st.column_config.NumberColumn("with 10+ sets"),
                                        "profit_if_all_taken": st.column_config.NumberColumn("profit if all taken", format="dollar")})
        if not ar["recent"].empty:
            st.markdown("#### Latest")
            st.dataframe(ar["recent"], hide_index=True, use_container_width=True,
                         column_config={"edge_c": st.column_config.NumberColumn("edge (c/set)", format="%.1f"),
                                        "contracts": st.column_config.NumberColumn("sets available", format="%.0f"),
                                        "days_to_payout": st.column_config.NumberColumn("pays out in (days)", format="%.0f"),
                                        "lasted_min": st.column_config.NumberColumn("lasted (min)", format="%.1f")})

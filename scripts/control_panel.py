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


@st.cache_data(ttl=20)
def load_all():
    return {
        "trades": D.trades(),
        "positions": D.open_positions(),
        "learning": D.learning_summary(),
        "predictions": D.recent_predictions(),
        "ai": D.ai_spend_today(),
        "status": D.bot_status(),
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
elif mins < 5:
    s1.metric("Bot", "● running", f"active {mins:.0f} min ago", delta_color="off")
else:
    s1.metric("Bot", "⚠ stalled?", f"last activity {mins:.0f} min ago", delta_color="off")
s2.metric("Mode", {"paper": "PAPER", "LIVE": "LIVE $"}.get(status["mode"], "unknown"))
s3.metric("New trades", "PAUSED" if cfg["PAUSED"] else "ON")
ai = data["ai"]
cap = cfg["DAILY_AI_COST_LIMIT"]
s4.metric("AI spend today", f"\\${ai['cost']:.2f} of \\${cap:.2f}", f"{ai['requests']} requests", delta_color="off")

tab_perf, tab_ctrl, tab_pos, tab_learn, tab_pred = st.tabs(
    ["Performance", "Strategy controls", "Open positions", "Learning", "Predictions"])

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

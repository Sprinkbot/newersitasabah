import pandas as pd
import streamlit as st
import yfinance as yf

from engine import PAIRS, TECH_INFO, analyze, currency_strength, fmt, pick_non_overlapping, pip_size

st.set_page_config(page_title="Forex Scanner", page_icon="📈", layout="wide")

st.title("📈 Forex Scanner")
st.caption("Runs 6 different trading techniques on 28 forex pairs (1H chart) and ranks the best setups. "
           "Signals come from closed candles only, so they don't flicker or repaint.")

with st.sidebar:
    st.header("Settings")
    min_conf = st.selectbox("Show setups that are", ["All", "Good or better", "Strong only"], index=1)
    if st.button("Refresh now"):
        st.cache_data.clear()
    st.markdown("---")
    st.markdown("**Techniques it runs**")
    for name, desc in TECH_INFO.items():
        st.markdown(f"**{name}**  \n{desc}")
    st.markdown("---")
    st.markdown("**Extra factors that adjust the score**")
    st.markdown("- **Currency strength:** is the base currency actually strong and the quote weak (last 24h)?\n"
                "- **Daily trend:** does the bigger daily picture agree?\n"
                "- **No doubling up:** the top picks never bet on the same currency twice")
    st.caption("Ranking = how many techniques agree, weighted by how well each fits the market right now "
               "(trend tools in strong trends, range tools in quiet markets). Techniques that disagree = WAIT.")


@st.cache_data(ttl=60, show_spinner=False)
def load_data():
    tickers = [f"{p}=X" for p in PAIRS]
    raw = yf.download(tickers, period="90d", interval="1h", group_by="ticker",
                      auto_adjust=False, progress=False, threads=True)
    out = {}
    for p in PAIRS:
        try:
            df = raw[f"{p}=X"][["Open", "High", "Low", "Close"]].dropna()
        except KeyError:
            continue
        if len(df) > 400:
            out[p] = df
    return out


def pips(pair, a, b):
    return abs(a - b) / pip_size(pair)


def age_text(a):
    return "this candle" if a == 0 else f"{a} candle{'s' if a > 1 else ''} ago"


def trade_card(r, top=False):
    pair, entry = r["pair"], r["entry"]
    badge = {"Strong": "🔥", "Good": "✅", "Weak": "⚠️"}[r["confidence"]]
    with st.container(border=True):
        st.markdown(f"{'🏆 ' if top else ''}**{pair}** · live `{fmt(pair, r['live'])}` · {badge} {r['confidence']} ({r['score']:.1f})")
        st.caption(f"Market: {r['regime']} · ADX {r['adx']:.0f} · best fit: {r['best_fit']}")
        st.markdown("**Techniques agreeing:** " + ", ".join(f"{nm} ({age_text(a)})" for nm, a in r["techs"]))
        for kind, txt in r.get("factors", []):
            st.caption({"good": "✅ ", "ok": "➖ ", "bad": "⛔ "}[kind] + txt)
        st.markdown(f"Entry `{fmt(pair, entry)}`")
        for i, p in enumerate(r["tps"], 1):
            st.markdown(f":green[TP{i}] `{fmt(pair, p)}` ({pips(pair, p, entry):.1f} pips)")
        for i, p in enumerate(r["sls"], 1):
            st.markdown(f":red[SL{i}] `{fmt(pair, p)}` ({pips(pair, p, entry):.1f} pips)")
        st.caption("Pick ONE stop (SL2 is the main one). Price hits TP1: move the stop to entry. Hits TP2: move it to TP1.")


def wait_card(r):
    arrow = {1: "↗ leaning up", -1: "↘ leaning down", 0: "• no direction"}[r["bias"]]
    with st.container(border=True):
        st.markdown(f"**{r['pair']}** · `{fmt(r['pair'], r['live'])}` · {arrow}")
        st.caption(f"{r['reason']} · ADX {r['adx']:.0f}")


def keep(r):
    if min_conf == "Strong only":
        return r["confidence"] == "Strong"
    if min_conf == "Good or better":
        return r["confidence"] in ("Strong", "Good")
    return True


@st.fragment(run_every=60)
def scanner():
    data = load_data()
    if not data:
        st.error("Couldn't load price data. Try Refresh now in a minute.")
        return

    strength = currency_strength(data)
    results = []
    for p, df in data.items():
        try:
            results.append(analyze(p, df, strength))
        except Exception as e:  # one bad pair should not break the page
            st.warning(f"{p}: skipped ({e})")

    active = sorted([r for r in results if r["status"] != "WAIT"], key=lambda r: (-r["score"], r["age"]))
    shown = [r for r in active if keep(r)]
    buys = [r for r in shown if r["status"] == "BUY"]
    sells = [r for r in shown if r["status"] == "SELL"]
    waits = sorted([r for r in results if r["status"] == "WAIT"], key=lambda r: (r["bias"] == 0, -r["adx"]))

    last_ts = max(df.index[-1] for df in data.values())
    st.caption(f"Latest candle: {last_ts:%Y-%m-%d %H:%M} UTC · auto-refreshes every 60s · {len(results)} pairs scanned")

    if strength:
        order = sorted(strength, key=strength.get, reverse=True)
        st.markdown("**Currency strength (24h), strongest to weakest:** " + " › ".join(f"{c} {strength[c]:+.2f}%" for c in order))

    st.subheader("🏆 Best setups right now (no doubling up)")
    picks = pick_non_overlapping(shown, 3)
    if not shown:
        st.info("No setups meet your filter right now. Waiting is a position too. (Lone single-technique signals are hidden by default, change the filter in the sidebar to see them.)")
    else:
        cols = st.columns(len(picks))
        for col, r in zip(cols, picks):
            with col:
                with st.container(border=True):
                    st.markdown(f"### {'🟢 BUY' if r['status'] == 'BUY' else '🔴 SELL'} {r['pair']}")
                    st.markdown(f"{r['confidence']} · score {r['score']:.1f} · {len(r['techs'])} technique(s) agree")
                    st.caption("Best fit: " + r["best_fit"])
        skipped = len(shown) - len(picks)
        if skipped > 0:
            st.caption(f"{skipped} other setup(s) are listed below but skipped here because they'd bet on a currency you're already using.")
    st.markdown("---")

    top_pairs = {r["pair"] for r in picks if r["confidence"] != "Weak"}
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader(f"🟢 BUY ({len(buys)})")
        if not buys:
            st.info("No buy setups.")
        for r in buys:
            trade_card(r, r["pair"] in top_pairs)
    with c2:
        st.subheader(f"🔴 SELL ({len(sells)})")
        if not sells:
            st.info("No sell setups.")
        for r in sells:
            trade_card(r, r["pair"] in top_pairs)
    with c3:
        st.subheader(f"⚪ WAIT ({len(waits)})")
        for r in waits:
            wait_card(r)


scanner()

st.markdown("---")
st.caption("Educational tool, not financial advice. Free Yahoo Finance data can be delayed by a minute or more and has no real forex volume, "
           "so check the setup on your broker's own price before trading. More techniques agreeing is a better sign, not a guarantee.")

"""Multi-technique forex scanner engine.
Six different trading techniques run on every pair. Signals use CLOSED candles only
(the still-forming candle is only used for the live price and to check if a signal already played out)."""
import numpy as np
import pandas as pd

PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    "AUDJPY", "AUDCAD", "AUDCHF", "AUDNZD",
    "CADJPY", "CADCHF", "CHFJPY", "NZDJPY", "NZDCAD", "NZDCHF",
]

SL_MULT = (0.7, 1.0, 1.4)   # SL1 tight, SL2 main, SL3 wide (in R)
TP_MULT = (1.0, 2.0, 3.0)   # TP1, TP2, TP3 (in R)
LOOKBACK_BARS = 3           # a signal stays "live" for this many closed candles
STALE_R = 0.25              # if live price is more than 0.25R from the entry, the entry is gone

TREND_TECHS = {"Trend pullback", "EMA momentum cross", "MACD + 4H trend"}
RANGE_TECHS = {"Mean reversion", "S/R rejection"}
TECH_INFO = {
    "Trend pullback": "Dip to the EMA21 inside a clean 1H + 4H trend, then it resumes",
    "EMA momentum cross": "EMA 9 crosses EMA 21 with the bigger trend and MACD agreeing",
    "MACD + 4H trend": "MACD crosses its signal line in the direction of the 4H trend",
    "Breakout": "Close beyond the 20-candle high/low with real trend strength",
    "Mean reversion": "Quiet market: price snaps back from a Bollinger Band + RSI extreme",
    "S/R rejection": "Price rejects a 50-candle high/low with a long wick",
}


def pip_size(pair):
    return 0.01 if "JPY" in pair else 0.0001


def fmt(pair, x):
    return f"{x:.3f}" if "JPY" in pair else f"{x:.5f}"


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rma(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def indicators(df):
    d = df.copy()
    c, h, l = d["Close"], d["High"], d["Low"]
    d["ema9"], d["ema21"], d["ema50"], d["ema200"] = ema(c, 9), ema(c, 21), ema(c, 50), ema(c, 200)

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = rma(tr, 14)

    delta = c.diff()
    gain = rma(delta.clip(lower=0), 14)
    loss = rma((-delta).clip(lower=0), 14)
    d["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    up, dn = h.diff(), -l.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=d.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=d.index)
    d["diP"] = 100 * rma(plus, 14) / d["atr"]
    d["diM"] = 100 * rma(minus, 14) / d["atr"]
    dx = 100 * (d["diP"] - d["diM"]).abs() / (d["diP"] + d["diM"]).replace(0, np.nan)
    d["adx"] = rma(dx.fillna(0), 14)

    d["macd"] = ema(c, 12) - ema(c, 26)
    d["macdSig"] = ema(d["macd"], 9)
    d["hist"] = d["macd"] - d["macdSig"]

    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    d["bbUp"], d["bbLo"] = mid + 2 * sd, mid - 2 * sd
    d["bbwRank"] = ((4 * sd) / mid).rolling(100).rank(pct=True) * 100

    d["atrAvg"] = d["atr"].rolling(50).mean()
    d["rng3"] = (h - l).rolling(3).max()
    d["swLow"], d["swHigh"] = l.rolling(5).min(), h.rolling(5).max()
    d["donHi"], d["donLo"] = h.rolling(20).max().shift(1), l.rolling(20).min().shift(1)
    d["hh50"], d["ll50"] = h.rolling(50).max().shift(1), l.rolling(50).min().shift(1)
    return d


def htf_trend(df):
    """4H trend from the last CLOSED 4H candle (no look-ahead)."""
    h4 = df.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    e50, e200 = ema(h4["Close"], 50), ema(h4["Close"], 200)
    t = pd.Series(0, index=h4.index)
    t[(h4["Close"] > e200) & (e50 > e200)] = 1
    t[(h4["Close"] < e200) & (e50 < e200)] = -1
    t = t.shift(1)
    return t.reindex(df.index, method="ffill").fillna(0)


def signals(d, htf):
    c, h, l, o = d["Close"], d["High"], d["Low"], d["Open"]
    atr, adx = d["atr"], d["adx"]

    ldn = d.index.tz_convert("Europe/London")
    hr = pd.Series(ldn.hour, index=d.index)
    dow = pd.Series(ldn.dayofweek, index=d.index)
    time_ok = (hr >= 8) & (hr < 21) & (hr != 13) & ~((dow == 4) & (hr >= 17))
    calm = d["rng3"] < 2.5 * atr
    base = time_ok & calm

    stack_up = (d["ema21"] > d["ema50"]) & (d["ema50"] > d["ema200"]) & ((d["ema50"] - d["ema200"]) > 0.5 * atr) & (d["ema200"] > d["ema200"].shift(10))
    stack_dn = (d["ema21"] < d["ema50"]) & (d["ema50"] < d["ema200"]) & ((d["ema200"] - d["ema50"]) > 0.5 * atr) & (d["ema200"] < d["ema200"].shift(10))
    trend_up = (c > d["ema21"]) & stack_up & (htf == 1)
    trend_dn = (c < d["ema21"]) & stack_dn & (htf == -1)
    regime = (adx > 22) & (atr > d["atrAvg"]) & (d["bbwRank"] > 20)

    # 1) Trend pullback
    pull_up = (l.rolling(3).min() <= d["ema21"]) & (c > d["ema21"]) & (c > h.shift(1)) & (c > o)
    pull_dn = (h.rolling(3).max() >= d["ema21"]) & (c < d["ema21"]) & (c < l.shift(1)) & (c < o)
    strong_up = (c > l + 0.6 * (h - l)).astype(int)
    strong_dn = (c < h - 0.6 * (h - l)).astype(int)
    adx_rise = (adx > adx.shift(3)).astype(int)
    sc_l = 1 + ((d["rsi"] > 50) & (d["rsi"] < 65)).astype(int) + (d["hist"] > 0).astype(int) + adx_rise + strong_up
    sc_s = 1 + ((d["rsi"] < 50) & (d["rsi"] > 35)).astype(int) + (d["hist"] < 0).astype(int) + adx_rise + strong_dn
    pb_buy = base & trend_up & (d["diP"] > d["diM"]) & regime & ((c - d["ema21"]) < 1.5 * atr) & pull_up & (sc_l >= 4)
    pb_sell = base & trend_dn & (d["diM"] > d["diP"]) & regime & ((d["ema21"] - c) < 1.5 * atr) & pull_dn & (sc_s >= 4)

    # 2) EMA momentum cross
    x_up = ((d["ema9"] > d["ema21"]) & (d["ema9"].shift() <= d["ema21"].shift())).rolling(3).max().fillna(0).astype(bool)
    x_dn = ((d["ema9"] < d["ema21"]) & (d["ema9"].shift() >= d["ema21"].shift())).rolling(3).max().fillna(0).astype(bool)
    ec_buy = base & x_up & (c > d["ema50"]) & (d["ema50"] > d["ema200"]) & (d["hist"] > 0) & (adx > 18) & ((c - d["ema21"]) < 1.5 * atr)
    ec_sell = base & x_dn & (c < d["ema50"]) & (d["ema50"] < d["ema200"]) & (d["hist"] < 0) & (adx > 18) & ((d["ema21"] - c) < 1.5 * atr)

    # 3) MACD cross with the 4H trend
    m_up = ((d["macd"] > d["macdSig"]) & (d["macd"].shift() <= d["macdSig"].shift())).rolling(2).max().fillna(0).astype(bool)
    m_dn = ((d["macd"] < d["macdSig"]) & (d["macd"].shift() >= d["macdSig"].shift())).rolling(2).max().fillna(0).astype(bool)
    mh_buy = base & m_up & (htf == 1) & (c > d["ema50"]) & (d["macd"] > 0) & (adx > 22)
    mh_sell = base & m_dn & (htf == -1) & (c < d["ema50"]) & (d["macd"] < 0) & (adx > 22)

    # 4) Breakout (spikes allowed, but not absurd ones)
    bo_base = time_ok & (d["rng3"] < 3.5 * atr)
    decisive = (h - l) > 1.0 * atr
    bo_buy = bo_base & (c > d["donHi"]) & (c > d["ema50"]) & (adx > 20) & (adx > adx.shift(3)) & decisive & (c > l + 0.7 * (h - l)) & ((c - d["ema21"]) < 2.5 * atr)
    bo_sell = bo_base & (c < d["donLo"]) & (c < d["ema50"]) & (adx > 20) & (adx > adx.shift(3)) & decisive & (c < h - 0.7 * (h - l)) & ((d["ema21"] - c) < 2.5 * atr)

    # 5) Mean reversion (quiet / ranging market only)
    mr_buy = base & (adx < 20) & (d["rsi"] < 30) & (l <= d["bbLo"]) & (c > o)
    mr_sell = base & (adx < 20) & (d["rsi"] > 70) & (h >= d["bbUp"]) & (c < o)

    # 6) Support / resistance rejection (long wick at a 50-candle extreme)
    body = (c - o).abs().clip(lower=0.1 * atr)
    low_wick = np.minimum(o, c) - l
    up_wick = h - np.maximum(o, c)
    sr_buy = base & (l <= d["ll50"] + 0.3 * atr) & (c > o) & (low_wick >= 1.5 * body) & (adx < 25)
    sr_sell = base & (h >= d["hh50"] - 0.3 * atr) & (c < o) & (up_wick >= 1.5 * body) & (adx < 25)

    risk_l = (c - d["swLow"] + 0.2 * atr).clip(lower=atr, upper=2.2 * atr)
    risk_s = (d["swHigh"] - c + 0.2 * atr).clip(lower=atr, upper=2.2 * atr)

    strats = {
        "Trend pullback": (pb_buy, pb_sell),
        "EMA momentum cross": (ec_buy, ec_sell),
        "MACD + 4H trend": (mh_buy, mh_sell),
        "Breakout": (bo_buy, bo_sell),
        "Mean reversion": (mr_buy, mr_sell),
        "S/R rejection": (sr_buy, sr_sell),
    }
    return dict(strats=strats, trend_up=trend_up, trend_dn=trend_dn, risk_l=risk_l, risk_s=risk_s)


CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]


def currency_strength(data, bars=24):
    """Average % move of each currency vs the others over the last `bars` closed hourly candles."""
    acc = {c: [] for c in CURRENCIES}
    for pair, df in data.items():
        c = df["Close"].iloc[:-1]
        if len(c) <= bars:
            continue
        chg = (c.iloc[-1] / c.iloc[-1 - bars] - 1) * 100
        acc[pair[:3]].append(chg)
        acc[pair[3:]].append(-chg)
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def daily_dir(closed):
    """Daily trend from the last COMPLETED day: price vs EMA20 and the EMA slope."""
    d1 = closed["Close"].resample("1D").last().dropna()
    if len(d1) < 25:
        return 0
    e = ema(d1, 20)
    c, ee = d1.iloc[-2], e.iloc[-2]
    if c > ee and ee > e.iloc[-5]:
        return 1
    if c < ee and ee < e.iloc[-5]:
        return -1
    return 0


def pick_non_overlapping(setups, max_n=3):
    """Best setups that don't bet on the same currency twice (EURUSD buy + USDCHF sell both short USD, etc)."""
    chosen, used = [], set()
    for r in setups:
        cur = {r["pair"][:3], r["pair"][3:]}
        if cur & used:
            continue
        chosen.append(r)
        used |= cur
        if len(chosen) == max_n:
            break
    return chosen


def weight(name, adx):
    """How well a technique fits the current market. Trend tools like strong trends, range tools like quiet markets."""
    if name in TREND_TECHS:
        return 1.0 if adx >= 22 else 0.6 if adx >= 18 else 0.3
    if name in RANGE_TECHS:
        return 1.0 if adx < 18 else 0.6 if adx < 22 else 0.3
    return 1.0


def regime_label(adx):
    return "Trending" if adx >= 22 else "Ranging" if adx < 18 else "Mixed"


def confidence(score):
    return "Strong" if score >= 2.5 else "Good" if score >= 1.5 else "Weak"


def _levels(d, closed, df, sig, i, side, live):
    """Levels for a signal on candle i. Returns None if it already hit SL2 or TP1."""
    entry = float(d["Close"].iloc[i])
    R = float((sig["risk_l"] if side == 1 else sig["risk_s"]).iloc[i])
    if not np.isfinite(R) or R <= 0:
        return None
    sls = [entry - side * m * R for m in SL_MULT]
    tps = [entry + side * m * R for m in TP_MULT]
    later = pd.concat([closed.iloc[i + 1:], df.iloc[-1:]])
    if side == 1:
        stopped, tp_hit = bool((later["Low"] <= sls[1]).any()), bool((later["High"] >= tps[0]).any())
    else:
        stopped, tp_hit = bool((later["High"] >= sls[1]).any()), bool((later["Low"] <= tps[0]).any())
    if stopped or tp_hit:
        return None
    drift = side * (live - entry) / R          # + = price moved in the trade's favor, - = against it
    if abs(drift) > STALE_R:
        return "stale"                          # entry price is gone, don't chase
    return dict(entry=entry, sls=sls, tps=tps, r_pips=R * SL_MULT[1] / pip_size("EURUSD"))


def analyze(pair, df, strength=None):
    """df includes the still-forming last candle. `strength` = output of currency_strength()."""
    live = float(df["Close"].iloc[-1])
    closed = df.iloc[:-1]
    d = indicators(closed)
    sig = signals(d, htf_trend(closed))
    n = len(d)
    adx = float(d["adx"].iloc[-1])
    bias = 1 if bool(sig["trend_up"].iloc[-1]) else (-1 if bool(sig["trend_dn"].iloc[-1]) else 0)
    regime = regime_label(adx)

    found = {1: [], -1: []}   # side -> list of (tech, age, levels)
    stale_seen = 0
    for name, (b, s) in sig["strats"].items():
        for side, ser in ((1, b), (-1, s)):
            for k in range(1, LOOKBACK_BARS + 1):
                i = n - k
                if i < 0:
                    break
                if bool(ser.iloc[i]):
                    lv = _levels(d, closed, df, sig, i, side, live)
                    if lv == "stale":
                        stale_seen += 1
                    elif lv:
                        found[side].append((name, k - 1, lv))
                    break

    out = dict(pair=pair, live=live, status="WAIT", bias=bias, adx=adx, regime=regime, score=0.0,
               confidence="", techs=[], reason="")
    nb, ns = len(found[1]), len(found[-1])
    if nb and ns:
        out["reason"] = f"Techniques disagree ({nb} say buy, {ns} say sell)"
        return out
    if not nb and not ns:
        if stale_seen:
            out["reason"] = "A signal fired but price already moved away from the entry, don't chase"
        elif bias == 0:
            out["reason"] = f"No technique has a setup. Market is {regime.lower()}, no clean trend"
        else:
            out["reason"] = f"No technique has a setup yet. Market is {regime.lower()}, trend {'up' if bias == 1 else 'down'}"
        return out

    side = 1 if nb else -1
    hits = found[side]
    score = sum(weight(nm, adx) for nm, _, _ in hits) + (0.3 if any(a == 0 for _, a, _ in hits) else 0.0)
    factors = []
    if strength:
        ranks = {c: i + 1 for i, c in enumerate(sorted(strength, key=strength.get, reverse=True))}
        b, q = pair[:3], pair[3:]
        if b in ranks and q in ranks:
            eff = (ranks[q] - ranks[b]) * side
            txt = f"{b} #{ranks[b]} vs {q} #{ranks[q]} of 8 (1 = strongest)"
            if eff >= 4:
                score += 0.7
                factors.append(("good", f"Currency strength strongly supports it: {txt}"))
            elif eff >= 1:
                score += 0.3
                factors.append(("ok", f"Currency strength leans your way: {txt}"))
            else:
                score -= 1.0
                factors.append(("bad", f"Currency strength is AGAINST it: {txt}"))
    dd = daily_dir(closed)
    if dd == side:
        score += 0.4
        factors.append(("good", "Daily trend agrees"))
    elif dd == -side:
        score -= 0.6
        factors.append(("bad", "Daily trend is AGAINST it"))
    else:
        factors.append(("ok", "Daily trend is flat / unclear"))
    score = max(score, 0.0)
    best = sorted(hits, key=lambda x: (x[1], -weight(x[0], adx)))[0]     # freshest signal sets the levels
    fit = max(hits, key=lambda x: weight(x[0], adx))                      # best fit for this market type
    lv = best[2]
    lv["r_pips"] = abs(lv["entry"] - lv["sls"][1]) / pip_size(pair)
    out.update(status="BUY" if side == 1 else "SELL", score=round(score, 2), confidence=confidence(score),
               techs=sorted([(nm, a) for nm, a, _ in hits], key=lambda x: x[1]), best_fit=fit[0], factors=factors,
               entry=lv["entry"], tps=lv["tps"], sls=lv["sls"], age=best[1], r_pips=lv["r_pips"])
    return out

#!/usr/bin/env python3
"""Backtest MVRV-weighted BTC accumulation against plain DCA.

A fixed monthly contribution lands in a cash buffer on the 1st. Every Monday
each strategy buys `base_weekly * multiplier`, capped at the cash it holds,
so every strategy invests the same money and leftover cash counts at face
value. The multiplier reads the previous day's MVRV, so a buy never sees its
own day's close.

Strategies:
  dca        multiplier 1 (the benchmark)
  bands      fixed MVRV bands: <1:3x, <1.5:2x, <2:1x, <2.5:0.5x, <3:0.25x, else 0x
             (thresholds picked with hindsight of past cycle tops)
  relative   (trailing 4y median MVRV / MVRV) ** k, clipped to [0, 4]
             (no hindsight: only uses MVRV history up to the previous day)
  reserve    buy 0.75x while MVRV >= 1, keeping the rest as dry powder,
             then 4x while MVRV < 1 (price below realized price)

Usage:
    python3 scripts/btc_accumulation_backtest.py
    python3 scripts/btc_accumulation_backtest.py --data btc.csv --plots out/
"""

import argparse
import io
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

API_BASE = "https://bitview.space/api/series"
HEADERS = {"User-Agent": "btc-accumulation-backtest"}
BANDS = [(1.0, 3.0), (1.5, 2.0), (2.0, 1.0), (2.5, 0.5), (3.0, 0.25)]
MEDIAN_WINDOW_DAYS = 4 * 365
MAX_MULTIPLIER = 4.0
WINDOW_YEARS = 4
START_YEARS = [2014, 2016, 2018, 2020, 2022, 2024]
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
ORDINAL = ["#86b6ef", "#2a78d6", "#104281"]


def load_data(path):
    """Daily `date,price_close,mvrv` from a local CSV or the Bitview API."""
    if path:
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        end = (date.today() + timedelta(days=1)).isoformat()
        url = f"{API_BASE}/bulk?series=date,price_close,mvrv&index=day1&start=2010-07-18&end={end}&format=csv"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp:
            df = pd.read_csv(io.StringIO(resp.read().decode()), parse_dates=["date"])
        if df.date.max() < pd.Timestamp(date.today()) - pd.Timedelta(days=7):
            raise RuntimeError(f"Bitview data ends at {df.date.max().date()}; the response looks truncated")
    df = df.dropna(subset=["price_close", "mvrv"])
    df = df[df.price_close > 0].set_index("date").sort_index()
    return df.rename(columns={"price_close": "price"})


def band_multiplier(mvrv):
    out = np.zeros_like(mvrv)
    for upper, mult in reversed(BANDS):
        out[mvrv < upper] = mult
    return out


def build_multipliers(df, ks):
    """Multiplier per strategy, aligned to the trading day, lagged one day."""
    mvrv = df.mvrv
    median = mvrv.rolling(MEDIAN_WINDOW_DAYS, min_periods=365).median()
    ratio = (median / mvrv).shift(1)
    mults = {
        "dca": pd.Series(1.0, index=df.index),
        "bands": pd.Series(band_multiplier(mvrv.to_numpy()), index=df.index).shift(1),
    }
    for k in ks:
        mults[f"relative_k{k:g}"] = (ratio**k).clip(0, MAX_MULTIPLIER)
    mults["reserve"] = pd.Series(np.where(mvrv < 1.0, 4.0, 0.75), index=df.index).shift(1)
    return {name: m.fillna(1.0) for name, m in mults.items()}, median


def xirr(days, flows):
    """Annualized money-weighted return; days are offsets from the first flow."""
    years = np.asarray(days) / 365.25
    flows = np.asarray(flows)
    lo, hi = -0.99, 100.0
    npv = lambda r: np.sum(flows / (1 + r) ** years)
    if npv(lo) * npv(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def simulate(df, mult, start, end, monthly, fee, cash_yield):
    window = df.loc[start:end]
    price = window.price.to_numpy()
    m = mult.loc[start:end].to_numpy()
    day = window.index.day.to_numpy()
    monday = window.index.weekday.to_numpy() == 0
    base = monthly * 12 / 52
    daily_yield = (1 + cash_yield) ** (1 / 365) - 1

    cash = btc = contributed = spent = 0.0
    values = np.empty(len(window))
    contribs = np.empty(len(window))
    flow_days, flows = [], []
    for i in range(len(window)):
        cash *= 1 + daily_yield
        if i == 0 or day[i] == 1:
            cash += monthly
            contributed += monthly
            flow_days.append(i)
            flows.append(-monthly)
        if monday[i]:
            usd = min(cash, base * m[i])
            if usd > 0:
                btc += usd * (1 - fee) / price[i]
                cash -= usd
                spent += usd
        values[i] = cash + btc * price[i]
        contribs[i] = contributed

    final = values[-1]
    peak = np.maximum.accumulate(values)
    return {
        "series": pd.DataFrame({"value": values, "contributed": contribs}, index=window.index),
        "contributed": contributed,
        "final": final,
        "multiple": final / contributed,
        "irr": xirr(flow_days + [len(window) - 1], flows + [final]),
        "btc": btc,
        "avg_cost": spent / btc if btc else float("nan"),
        "cash_share": cash / final,
        "max_dd": float(np.max(1 - values / peak)),
        "worst_vs_invested": float(np.min(values / contribs - 1)),
    }


def rolling_windows(df, mults, names, args):
    last = df.index[-1]
    starts = pd.date_range(args.start, last - pd.DateOffset(years=WINDOW_YEARS), freq="MS")
    rows = []
    for start in starts:
        end = start + pd.DateOffset(years=WINDOW_YEARS) - pd.Timedelta(days=1)
        base = simulate(df, mults["dca"], start, end, args.monthly, args.fee, args.cash_yield)
        row = {"start": start, "dca_irr": base["irr"]}
        for name in names:
            res = simulate(df, mults[name], start, end, args.monthly, args.fee, args.cash_yield)
            row[name] = res["final"] / base["final"] - 1
        rows.append(row)
    return pd.DataFrame(rows).set_index("start")


def pct(x, digits=1):
    return f"{x * 100:+.{digits}f}%"


def print_period_table(df, mults, names, args):
    print(f"\n== Start year -> {df.index[-1].date()} (US${args.monthly:,.0f}/month) ==")
    header = f"{'start':<6} {'strategy':<14} {'invested':>11} {'final':>12} {'x':>6} {'IRR/yr':>8} {'avg cost':>9} {'cash':>6} {'max dd':>7} {'worst':>7} {'vs dca':>7}"
    print(header)
    print("-" * len(header))
    for year in START_YEARS:
        start = pd.Timestamp(f"{year}-01-01")
        if start < df.index[0]:
            continue
        base = None
        for name in ["dca"] + names:
            r = simulate(df, mults[name], start, None, args.monthly, args.fee, args.cash_yield)
            base = base or r
            print(
                f"{year:<6} {name:<14} {r['contributed']:>11,.0f} {r['final']:>12,.0f} "
                f"{r['multiple']:>6.2f} {pct(r['irr']):>8} {r['avg_cost']:>9,.0f} "
                f"{r['cash_share'] * 100:>5.0f}% {r['max_dd'] * 100:>6.0f}% "
                f"{pct(r['worst_vs_invested'], 0):>7} {pct(r['final'] / base['final'] - 1):>7}"
            )
        print()


def print_rolling_summary(rolling, names):
    print(f"== Rolling {WINDOW_YEARS}y windows, monthly starts {rolling.index[0].date()} .. {rolling.index[-1].date()} (n={len(rolling)}) ==")
    print(f"{'strategy':<14} {'beats dca':>10} {'median':>8} {'worst':>8} {'best':>8}")
    for name in names:
        col = rolling[name]
        print(f"{name:<14} {(col > 0).mean() * 100:>9.0f}% {pct(col.median()):>8} {pct(col.min()):>8} {pct(col.max()):>8}")
    print(f"dca alone: median IRR {pct(rolling.dca_irr.median())}/yr, worst {pct(rolling.dca_irr.min())}/yr")


def print_signal(df, mults, median, names):
    today = df.index[-1]
    print(f"\n== Signal for the next buy (after {today.date()}) ==")
    print(f"MVRV {df.mvrv.iloc[-1]:.3f} | 4y median {median.iloc[-1]:.3f} | price US${df.price.iloc[-1]:,.0f}")
    # Today's close is known, so the next buy uses it (the backtest lag is one day).
    mvrv = df.mvrv.iloc[-1]
    current = {
        "bands": band_multiplier(np.array([mvrv]))[0],
        "reserve": 4.0 if mvrv < 1.0 else 0.75,
        **{n: min((median.iloc[-1] / mvrv) ** float(n.split("_k")[1]), MAX_MULTIPLIER) for n in names if n.startswith("relative")},
    }
    for name in names:
        print(f"{name:<14} {current[name]:.2f}x base")


def style_axes(ax, title):
    ax.set_facecolor(INK["surface"])
    ax.set_title(title, loc="left", color=INK["primary"], fontsize=11)
    ax.grid(axis="y", color=INK["grid"], lw=0.6)
    ax.tick_params(colors=INK["muted"], labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["axis"])


def label_ends(ax, items, min_gap=0.07):
    """Direct labels right of the plot: colored dot at the line end, ink text."""
    from matplotlib.dates import date2num

    to_axes = ax.transData + ax.transAxes.inverted()
    placed = sorted(((to_axes.transform((date2num(x), y))[1], x, y, text, color) for x, y, text, color in items), reverse=True)
    last = None
    for ay, x, y, text, color in placed:
        ay = ay if last is None else min(ay, last - min_gap)
        last = ay
        ax.plot([x], [y], "o", color=color, ms=5, mec=INK["surface"], mew=1.5, zorder=5)
        ax.annotate(text, xy=(1.01, ay), xycoords="axes fraction", va="center", fontsize=9, color=INK["secondary"], annotation_clip=False)


def plot(df, mults, median, names, rolling, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    out = Path(args.plots)
    out.mkdir(parents=True, exist_ok=True)
    rel = [n for n in names if n.startswith("relative")]
    headline = rel[len(rel) // 2]
    shown = {headline: SERIES[0], "bands": SERIES[1], "reserve": SERIES[2]}
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": INK["surface"]})

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for name, color in shown.items():
        ax.plot(rolling.index, rolling[name] * 100, color=color, lw=1.4, label=name)
    ax.axhline(0, color=INK["primary"], lw=0.9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    style_axes(ax, f"Valor final vs DCA simples, janelas de {WINDOW_YEARS} anos (eixo x = mês de início)")
    label_ends(ax, [(rolling.index[-1], rolling[n].iloc[-1] * 100, n, c) for n, c in shown.items()])
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK["secondary"])
    fig.subplots_adjust(left=0.07, right=0.85, top=0.9, bottom=0.08)
    fig.savefig(out / "rolling.png", dpi=150)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    items = []
    for year, color in zip([2014, 2018, 2022], ORDINAL):
        r = simulate(df, mults["dca"], pd.Timestamp(f"{year}-01-01"), None, args.monthly, args.fee, args.cash_yield)
        multiple = r["series"].value / r["series"].contributed
        ax.plot(multiple.index, multiple, color=color, lw=1.4, label=f"início {year}")
        items.append((multiple.index[-1], multiple.iloc[-1], f"{year}: {multiple.iloc[-1]:.1f}x", color))
    ax.axhline(1, color=INK["primary"], lw=0.9)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    style_axes(ax, "DCA simples: patrimônio ÷ total aportado (abaixo de 1x = no prejuízo)")
    label_ends(ax, items)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK["secondary"])
    fig.subplots_adjust(left=0.07, right=0.85, top=0.9, bottom=0.08)
    fig.savefig(out / "multiple.png", dpi=150)

    view = df.loc[pd.Timestamp(args.start) :]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, height_ratios=[1.3, 1])
    ax1.plot(view.index, view.mvrv, color=INK["secondary"], lw=1, label="MVRV")
    ax1.plot(view.index, median.loc[view.index], color=INK["primary"], lw=1.2, ls="--", label="mediana 4 anos")
    for upper, _ in BANDS:
        ax1.axhline(upper, color=SERIES[1], lw=0.6, alpha=0.5)
    style_axes(ax1, "MVRV (preço ÷ preço realizado), mediana móvel e faixas fixas (laranja)")
    ax1.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK["secondary"])
    for name in (headline, "bands"):
        ax2.plot(view.index, mults[name].loc[view.index], color=shown[name], lw=1.1, label=name)
    style_axes(ax2, "Multiplicador de compra (x valor base semanal)")
    ax2.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK["secondary"])
    fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.06, hspace=0.25)
    fig.savefig(out / "signal.png", dpi=150)
    print(f"\nPlots written to {out}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="Local CSV with date,price_close,mvrv (default: fetch from Bitview)")
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--monthly", type=float, default=1000.0, help="Monthly contribution in USD")
    parser.add_argument("--fee", type=float, default=0.005, help="Fee per buy (0.005 = 0.5%%)")
    parser.add_argument("--cash-yield", type=float, default=0.0, help="Annual yield on idle cash")
    parser.add_argument("--ks", default="1,2,3", help="Exponents for the relative strategy")
    parser.add_argument("--plots", help="Directory to write PNG charts")
    args = parser.parse_args()

    df = load_data(args.data)
    ks = [float(k) for k in args.ks.split(",")]
    mults, median = build_multipliers(df, ks)
    names = [n for n in mults if n != "dca"]

    print(f"Data {df.index[0].date()} .. {df.index[-1].date()} | fee {args.fee * 100:.2f}% | cash yield {args.cash_yield * 100:.1f}%/yr")
    print_period_table(df, mults, names, args)
    rolling = rolling_windows(df, mults, names, args)
    print_rolling_summary(rolling, names)
    print_signal(df, mults, median, names)
    if args.plots:
        plot(df, mults, median, names, rolling, args)


if __name__ == "__main__":
    main()

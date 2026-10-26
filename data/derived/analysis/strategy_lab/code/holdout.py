"""Untouched-holdout test of the FROZEN rules on TIME holdings, 2023-05-22 .. 2025-12-31.

The rules were chosen on 2026-01-02..09-23 only; this period was never looked at.
Parameters are copied verbatim from the 2026 study. Only the calendar, the weekly
PDF source (holdout_fetch.py) and the price window change. Costs follow the user's
zero-commission broker (0%) with a 0.3% one-way stress for spreads.
"""

import json

import numpy as np
import pandas as pd

import common as C
import data as D
import engine as E
from gbrc.etf import market

HOLD = D.CACHE / "holdout"
PDF_DIR = HOLD / "time_weekly"
PRICE_START, PRICE_END = "2022-09-01", "2026-01-16"
BT_START, BT_END = "2023-05-22", "2025-12-31"
ETF_END = "2026-01-02"  # KRX close on 2026-01-02 reflects the US 2025-12-31 close

# frozen from the 2026 study: (source, score, N, rebalance index, buffer, weighting 0=EQ 1=FW)
RULES = {
    "TIME 미국 보유 복제·주1회": ("T", "W", 500, 1, 0, 1),
    "TIME 비중 Top5·주1회": ("T", "W", 5, 1, 5, 1),
    "H6e 모멘텀120 Top5·주1회·동일": ("T", "MOM120", 5, 1, 5, 0),
    "H6b 모멘텀120 Top5·주1회·펀드비중": ("T", "MOM120", 5, 1, 5, 1),
    "H6f 모멘텀120 Top10·주1회·동일": ("T", "MOM120", 10, 1, 10, 0),
}
WINDOWS = {"2023(5/22~)": ("2023-05-22", "2023-12-29"), "2024": ("2023-12-29", "2024-12-31"),
           "2025": ("2024-12-31", "2025-12-31"),
           "23년 가을 조정 7/31~10/27": ("2023-07-31", "2023-10-27"),
           "24년 7~8월 급락 7/10~8/7": ("2024-07-10", "2024-08-07"),
           "25년 관세 폭락 2/19~4/8": ("2025-02-19", "2025-04-08"),
           "25년 반등 4/8~6/30": ("2025-04-08", "2025-06-30")}


def load_pdfs():
    snaps = {}
    for p in sorted(PDF_DIR.glob("*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        snaps[s["actual_date"]] = s
    return snaps


def us_weights(snap):
    weights, codes = {}, {}
    for h in snap["holdings"]:
        code = h["code"]
        if not market.is_us_listed(code):
            continue
        company = market.company_key(market.holding_key(code))
        company = D.COMPANY_ALIASES.get(company, company)
        weights[company] = weights.get(company, 0.0) + float(h["weight"] or 0.0)
        codes.setdefault(company, set()).add(code)
    return weights, codes


def load_prices(tickers):
    paths = {f: HOLD / f"prices_{f.replace(' ', '_').lower()}.csv" for f in D.PRICE_FIELDS}
    if all(p.exists() for p in paths.values()):
        return {f: pd.read_csv(p, index_col=0, parse_dates=True) for f, p in paths.items()}
    import yfinance as yf

    frame = yf.download(sorted(set(tickers)), start=PRICE_START, end=PRICE_END,
                        auto_adjust=False, progress=False, threads=True)
    out = {f: frame[f].dropna(axis=1, how="all") for f in D.PRICE_FIELDS}
    for f, p in paths.items():
        out[f].to_csv(p)
    return out


def build_d(snaps, prices, shift=0):
    """Fund panel dict in the study's format; `shift` delays each PDF by k US sessions."""
    us_days = [x.strftime("%Y-%m-%d") for x in prices["Adj Close"]["QQQ"].dropna().index]
    raw = sorted(snaps)
    keyed = {}
    for p in raw:
        i = next(k for k, x in enumerate(us_days) if x >= p) + shift
        keyed[us_days[min(i, len(us_days) - 1)]] = p
    dates = sorted(keyed)
    weights, codes = {}, {}
    for key, p in keyed.items():
        w, c = us_weights(snaps[p])
        weights[key] = w
        for comp, cs in c.items():
            codes.setdefault(comp, set()).update(cs)
    empty_flow = {str(L): {x: {} for x in dates} for L in D.FLOW_WINDOWS}
    empty_new = {str(L): {x: [] for x in dates} for L in D.FLOW_WINDOWS}
    panel_t = {"dates": dates, "weights": weights, "flow_pp": empty_flow, "new_in": empty_new,
               "codes": {k: sorted(v) for k, v in codes.items()}, "metrics": {}}
    panel_k = {"dates": dates, "weights": {x: {} for x in dates}, "flow_pp": empty_flow,
               "new_in": empty_new, "codes": {}, "metrics": {}}
    companies = sorted(codes)
    return {"panels": {"T": panel_t, "K": panel_k}, "companies": companies,
            "ticker": {c: D.price_ticker(codes[c]) for c in companies}, "pdf_dates": dates,
            "prices": prices}


def etf_return(prices, tk, start, end):
    s = prices["Adj Close"][tk].dropna()
    a = s[s.index > start]
    b = s[s.index <= end]
    return b.iloc[-1] / a.iloc[0] - 1


def window_return(m, krw, start, end):
    idx = [i for i, x in enumerate(m.days) if start < x <= end]
    return np.prod(1 + krw[idx]) - 1


def fmt_vs(x, bench):
    return f"{x / bench:4.1f}배" if bench > 0 else f"{(x - bench) * 100:+5.1f}%p"


def main():
    snaps = load_pdfs()
    all_codes = {}
    for s in snaps.values():
        for comp, cs in us_weights(s)[1].items():
            all_codes.setdefault(comp, set()).update(cs)
    tickers = [D.price_ticker(cs) for cs in all_codes.values()]
    prices = load_prices([t for t in tickers if t] + D.REF_TICKERS)
    E.BT_START, E.BT_END = BT_START, BT_END

    d0 = build_d(snaps, prices)
    adj = prices["Adj Close"]
    missing = sorted(c for c, t in d0["ticker"].items() if t not in adj.columns)
    max_w = {c: max(us_weights(s)[0].get(c, 0.0) for s in snaps.values()) for c in missing}
    print(f"weekly PDFs {len(snaps)} ({min(snaps)} .. {max(snaps)}), US companies {len(all_codes)}, "
          f"unpriced {len(missing)}: " + ", ".join(f"{c.split()[0]}(max {w:.1f}%)" for c, w in sorted(max_w.items(), key=lambda x: -x[1])))

    etf_total = etf_return(prices, "456600.KS", BT_START, ETF_END)
    s = adj["456600.KS"].dropna()
    s = s[(s.index > BT_START) & (s.index <= ETF_END)]
    print(f"TIME ETF max drawdown {((s / s.cummax()) - 1).min() * 100:.1f}%")
    q = adj["QQQ"].dropna()
    fx = adj["KRW=X"].ffill()
    q0, q1 = q[q.index >= BT_START].iloc[0], q[q.index <= BT_END].iloc[-1]
    f0, f1 = fx[fx.index >= BT_START].iloc[0], fx[fx.index <= BT_END].iloc[-1]
    print(f"TIME ETF (KRX {BT_START}+1 -> {ETF_END}) {etf_total * 100:+.1f}% | QQQ KRW {(q1 / q0 * f1 / f0 - 1) * 100:+.1f}%")

    for cost in (0.0, 0.003):
        for ex, shift in (("close", 0), ("open", 0), ("open", 1), ("open", 2)):
            d = build_d(snaps, prices, shift)
            E.BT_START = d["pdf_dates"][0]  # a delayed run starts when its first PDF is usable
            m = E.build_market(d, exec_mode=ex)
            names, stack = E.build_scores(m)
            years = len(m.days) / 252
            print(f"\n== cost {cost * 100:.1f}% · {ex} +{shift}d · {m.days[0]} -> {m.days[-1]} ({years:.1f}y) ==")
            print("rule                              | total    xETF  | CAGR  | MDD  | Sharpe | hold")
            for k, (s, sc, N, rb, buf, wt) in RULES.items():
                krw, nh, _ = C.simulate(m, stack, [names.index((s, "all", sc)), N, wt, buf, rb, 0,
                                                   E.SOURCES.index(s)], cost=cost)
                st = C.stats(krw)
                cagr = (1 + st["total"]) ** (1 / years) - 1
                print(f"{k:33s} | {st['total'] * 100:+7.1f}% {fmt_vs(st['total'], etf_total)} | {cagr * 100:+5.0f}% |"
                      f" {st['mdd'] * 100:4.0f}% | {st['sharpe']:5.2f} | {nh[1:].mean():4.1f}")
            if cost == 0.0 and shift in (0, 1) and (ex == "open") == (shift == 1):
                print(f"\n  windows ({ex} +{shift}d, 0%): rule / TIME ETF")
                heads = list(RULES)
                for wname, (a, b) in WINDOWS.items():
                    eb = etf_return(prices, "456600.KS", a, b)
                    cells = []
                    for k in heads:
                        s, sc, N, rb, buf, wt = RULES[k]
                        krw = C.simulate(m, stack, [names.index((s, "all", sc)), N, wt, buf, rb, 0,
                                                    E.SOURCES.index(s)], cost=cost)[0]
                        r = window_return(m, krw, a, b)
                        cells.append(f"{r * 100:+6.1f}%")
                    print(f"  {wname:24s} ETF {eb * 100:+6.1f}% | " + " | ".join(cells))
                print("  columns: " + " | ".join(heads))


if __name__ == "__main__":
    main()

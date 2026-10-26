"""POST-HOC (after seeing the holdout): is 'TIME conviction Top-N' a plateau or a spike?

Same rule family in both periods, cost 0 (user's broker), open execution one session
after the PDF. Everything printed here was chosen after the holdout result was known,
so it is exploration, not validation.
"""

import numpy as np

import common as C
import data as D
import engine as E
import holdout as H

NS = (3, 5, 7, 10)


def run_period(m, names, stack, etf_total, label):
    print(f"\n== {label}: TIME weight Top-N, weekly, buffer N, cost 0, open +1 session ==")
    print("N   | FW total  xETF  MDD  | EQ total  xETF  MDD")
    for N in NS:
        cells = []
        for wt in (1, 0):
            krw = C.simulate(m, stack, [names.index(("T", "all", "W")), N, wt, N, 1, 0, 0], cost=0.0)[0]
            st = C.stats(krw)
            cells.append(f"{st['total'] * 100:+7.1f}% {st['total'] / etf_total:4.1f}x {st['mdd'] * 100:4.0f}%")
        print(f"{N:3d} | " + " | ".join(cells))
    repl = C.stats(C.simulate(m, stack, [names.index(("T", "all", "W")), 500, 1, 0, 1, 0, 0], cost=0.0)[0])
    print(f"TIME US book weekly {repl['total'] * 100:+.1f}% ({repl['total'] / etf_total:.1f}x, MDD {repl['mdd'] * 100:.0f}%)")


# holdout 2023-05 .. 2025-12 (weekly PDFs shifted by one US session)
snaps = H.load_pdfs()
all_codes = {}
for s in snaps.values():
    for comp, cs in H.us_weights(s)[1].items():
        all_codes.setdefault(comp, set()).update(cs)
prices = H.load_prices([D.price_ticker(cs) for cs in all_codes.values()] + D.REF_TICKERS)
E.BT_END = H.BT_END
d = H.build_d(snaps, prices, shift=1)
E.BT_START = d["pdf_dates"][0]
m = E.build_market(d, exec_mode="open")
names, stack = E.build_scores(m)
run_period(m, names, stack, H.etf_return(prices, "456600.KS", H.BT_START, H.ETF_END),
           "HOLDOUT 2023-05-23 .. 2025-12-31 (TIME ETF +228.2%)")

# 2026 study period with the same execution (daily PDFs, one-row delay)
E.BT_START, E.BT_END = "2026-01-02", "2026-09-22"
d26 = D.load()
m26 = E.build_market(d26, exec_mode="open")
names26, stack26 = E.build_scores(m26)
stack26 = E.apply_delay(m26, stack26, 1)
run_period(m26, names26, stack26, 0.684, "STUDY 2026-01-02 .. 2026-09-22 (TIME ETF +68.4%)")

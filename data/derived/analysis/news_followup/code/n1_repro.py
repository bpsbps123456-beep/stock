"""N1 정의 재현: Codex 2024~2026 SOXX 숫자(M 33/48·+1.32%, 대조 59.8%·+0.59%, 주가만 37/59·+0.79%).
1) Codex 가격 스냅샷(읽기만) + 내 코드  2) 내 yfinance 자료 + 같은 창  3) 내 자료 + 1994~ 연속 사슬.
-> results/n1_repro.csv"""
import json
import pathlib
import sys

import exchange_calendars as xcals
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "code"))
from n1_lib import attach_controls, conditions, daily_table, onsets, summarize  # noqa: E402

CODEX = HERE.parents[3] / "research" / "news_turning_points_20261025"
cm = json.loads((CODEX / "metrics.json").read_text(encoding="utf-8"))["macro"]

cal = xcals.get_calendar("XNYS", start="1994-01-01", end="2026-12-31")
ses_all = cal.sessions_in_range("1994-01-03", "2026-10-23").tz_localize(None)
ses_codex = ses_all[ses_all >= "2023-12-01"]


def frames_codex():
    p = pd.read_csv(CODEX / "prices.csv", header=[0, 1], index_col=0, parse_dates=True)
    return p["open"].reindex(ses_codex), p["close"].reindex(ses_codex)


def frames_mine(kind="raw", ses=ses_all):
    raw = pd.read_parquet(HERE / "cache" / f"n1_{kind}.parquet")
    op = raw.pivot(index="date", columns="sym", values="O").reindex(ses)
    cl = raw.pivot(index="date", columns="sym", values="C").reindex(ses)
    return op, cl


def run(op, cl, sym, sig_year_min, base_years=(2024, 2026)):
    first = cl[sym].first_valid_index()
    o, c, t = op[sym].loc[first:], cl[sym].loc[first:], cl["^TNX"].loc[first:]
    d = daily_table(o, c, t)
    mask = (d.year >= base_years[0]) & (d.year <= base_years[1])
    out = {}
    for fam, cond in conditions(c, t).items():
        ent = onsets(cond, sig_year_min)
        ev = attach_controls(d, ent, base_mask=mask)
        ev = ev[(ev.year >= base_years[0]) & (ev.year <= base_years[1])]
        out[fam] = (summarize(ev, 5), ev)
    return out


rows = []
for label, (op, cl), sym_min in [
    ("1_codex_snapshot", frames_codex(), 2024),
    ("2_yf_raw_same_window", frames_mine("raw", ses_codex), 2024),
    ("3_yf_raw_chain_from_2001", frames_mine("raw"), None),
    ("4_yf_divadj_same_window", frames_mine("adj", ses_codex), 2024),
]:
    res = run(op, cl, "SOXX", sym_min)
    for fam in ["M", "C", "D"]:
        s, ev = res[fam]
        r = dict(version=label, family=fam, **s)
        if fam in cm:
            a = cm[fam]["all"]
            r.update(codex_n=a["n5"], codex_wins=a["wins5"], codex_mean=a["mean5"],
                     codex_b_hit=a["matched_hit5"], codex_b_mean=a["matched_mean5"],
                     codex_a_hit=a["year_all_hit5"], codex_a_mean=a["year_all_mean5"])
        r["entries"] = ",".join(str(x.date()) for x in ev.index)
        rows.append(r)
        print(label, fam, s["n"], s["wins"], f"{s['mean']*100:+.3f}%", f"b {s['b_hit']*100:.2f}% {s['b_mean']*100:+.3f}%",
              f"a {s['a_hit']*100:.2f}% {s['a_mean']*100:+.3f}%")
df = pd.DataFrame(rows)
# Codex 진입일 목록과 대조
mo = pd.read_csv(CODEX / "macro_outcomes.csv")
codex_entries = {f: set(g.entry) for f, g in mo.groupby("family")}
for _, r in df.iterrows():
    if r.family in codex_entries:
        mine = set(r.entries.split(",")) if r.entries else set()
        print(r.version, r.family, "Codex에만", sorted(codex_entries[r.family] - mine), "내 쪽에만", sorted(mine - codex_entries[r.family]))
df.drop(columns=["entries"]).to_csv(HERE / "results" / "n1_repro.csv", index=False, encoding="utf-8-sig")

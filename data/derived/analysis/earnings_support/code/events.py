"""사건 표 준비: 종목마다 1시간봉을 한 번 읽어 160분봉 구역(판단 = 기준일 정규장 마감)과 체결·매도 창 봉을 만든다.

실행: py data/derived/analysis/earnings_support/code/events.py
출력: cache/ev_meta.parquet(사건 한 줄), cache/ev_bars.parquet(창 봉), cache/ev_zones.parquet(구역), results/events_index.csv

봉 순서(seg): 0 = 기준일 애프터(16:00–20:00), 1 = 반응일 프리장(04:00–09:30), 2 = 반응일 정규장, 3~6 = 2~5번째 거래일 정규장.
체결 창 A = seg 0(장 마감 뒤 발표만)·1·2, B = 1·2, C = 2. 익절 확인은 체결 봉 뒤의 정규장 봉(seg ≥ 2).
오류 체결 거름용으로 각 봉의 '다음 봉 저가'(1시간봉 전체 순서 기준)를 함께 둔다.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from lib import (CACHE, RES, abs_move_history, bars160, calendar, load_events, load_h1, mod, period_of,  # noqa: E402
                 pivots, prior_scale, zones)


def ai_set() -> set[str]:
    from fetch import ai_us
    return ai_us()


def prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ev = load_events()
    cal = calendar()
    sess = cal.sessions
    pos = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(sess)}
    d1 = {f: pd.read_parquet(CACHE / f"d1_{f}.parquet") for f in ("open", "close")}
    d1day = {f: {d.strftime("%Y-%m-%d"): i for i, d in enumerate(x.index)} for f, x in d1.items()}
    hist = abs_move_history()
    ai = ai_set()
    meta, bars, zrows, skipped = [], [], [], {}
    t0 = time.time()
    tickers = sorted(set(ev.ticker))
    eid = 0
    for n, t in enumerate(tickers):
        h = load_h1(t)
        if h is None:
            skipped["no_h1"] = skipped.get("no_h1", 0) + int((ev.ticker == t).sum())
            continue
        b = bars160(h)
        piv = pivots(b)
        last_bar_of = {d: i for i, d in enumerate(b["day"].dt.strftime("%Y-%m-%d"))}
        hday = h.index.tz_localize(None).normalize().strftime("%Y-%m-%d").to_numpy()
        hm = mod(h.index)
        O, Hh, L, C = (h[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
        NL = np.append(L[1:], np.nan)

        def px(field, day, fallback):
            f = d1[field]
            if t in f.columns and day in d1day[field]:
                v = f[t].iat[d1day[field][day]]
                if np.isfinite(v):
                    return float(v)
            return fallback

        for r in ev[ev.ticker == t].itertuples():
            rd = r.reaction_date
            if rd not in pos or pos[rd] + 4 >= len(sess):
                skipped["calendar"] = skipped.get("calendar", 0) + 1
                continue
            i0 = pos[rd]
            S = [sess[i0 + k].strftime("%Y-%m-%d") for k in range(5)]
            base = sess[i0 - 1].strftime("%Y-%m-%d")
            if base not in last_bar_of:
                skipped["no_base_bar"] = skipped.get("no_base_bar", 0) + 1
                continue
            D = last_bar_of[base]
            ref_bar = float(b["close"].iloc[D])
            sup, res, total = zones(piv, D, ref_bar)
            segs = [(0, (hday == base) & (hm >= 960) & (hm < 1200)), (1, (hday == S[0]) & (hm >= 240) & (hm < 570))]
            segs += [(2 + k, (hday == S[k]) & (hm >= 570) & (hm < 960)) for k in range(5)]
            idx, seg = [], []
            for code, m in segs:
                ii = np.flatnonzero(m)
                idx.extend(ii.tolist())
                seg.extend([code] * len(ii))
            idx, seg = np.array(idx, int), np.array(seg, int)
            if not (seg == 2).any():
                skipped["no_reaction_bars"] = skipped.get("no_reaction_bars", 0) + 1
                continue
            reg1, s2, s5 = seg == 2, seg == 3, seg == 6
            base_close = px("close", base, ref_bar)
            meta.append({
                "eid": eid, "ticker": t, "date": r.date, "reaction_date": rd, "base_date": base, "amc": bool(r.amc),
                "mcap_b": float(r.mcap_b), "sector": r.sector, "ai": t in ai, "period": period_of(rd),
                "r0": float(r.r0), "s": prior_scale(hist, t, rd), "base_close": base_close, "ref_bar": ref_bar,
                "x1": px("close", S[0], float(C[idx][reg1][-1])),
                "x2": px("open", S[1], float(O[idx][s2][0]) if s2.any() else np.nan),
                "x3": px("close", S[1], float(C[idx][s2][-1]) if s2.any() else np.nan),
                "x4": px("close", S[4], float(C[idx][s5][-1]) if s5.any() else np.nan),
                "swings": total, "n_sup": len(sup)})
            bars.append(pd.DataFrame({"eid": eid, "seg": seg, "o": O[idx], "h": Hh[idx], "l": L[idx], "c": C[idx],
                                      "nl": NL[idx]}))
            for side, zs in (("S", sup), ("R", res[:3])):
                for k, z in enumerate(zs, 1):
                    zrows.append((eid, side, k, z[0], z[1], z[2], z[3]))
            eid += 1
        if n % 100 == 0:
            print(f"{n}/{len(tickers)} {t} 사건 {len(meta)} ({time.time() - t0:.0f}s)", flush=True)
    print("빠진 사건:", skipped, flush=True)
    return (pd.DataFrame(meta), pd.concat(bars, ignore_index=True),
            pd.DataFrame(zrows, columns=["eid", "side", "k", "lo", "hi", "cnt", "last"]))


def load() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    meta = pd.read_parquet(CACHE / "ev_meta.parquet")
    bars = pd.read_parquet(CACHE / "ev_bars.parquet")
    zs = pd.read_parquet(CACHE / "ev_zones.parquet")
    by = {e: g for e, g in bars.groupby("eid", sort=False)}
    return meta, by, zs


def main() -> None:
    meta, bars, zs = prepare()
    meta.to_parquet(CACHE / "ev_meta.parquet")
    bars.to_parquet(CACHE / "ev_bars.parquet")
    zs.to_parquet(CACHE / "ev_zones.parquet")
    RES.mkdir(parents=True, exist_ok=True)
    meta.to_csv(RES / "events_index.csv", index=False)
    print(len(meta), "사건 준비", meta.groupby("period").size().to_dict(), flush=True)


if __name__ == "__main__":
    main()

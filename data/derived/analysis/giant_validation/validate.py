"""거인의 어깨 검증 (2026-10-05, Path-Q 연구). 결과는 같은 폴더의 validation.md / validation.json.

A. 나스닥 쏠림: 2026년 매일 QQQ 고점 괴리와 구성 종목 중 신고가권 비율
B. ◆ 검증: 2026-02~09 매주, 그 시점까지의 데이터로만 ◆ 네 조건을 다시 계산하고 다음 거래일 종가에
   진입해 10·20거래일 수익을 AI 유니버스 균등 평균과 비교한다(규칙은 탭과 같고 과거에 맞춰 고치지 않음)
C. 두 펀드 실매매: 직전 20개 PDF에서 산 종목·판 종목의 이후 수익(PDF D는 D-1 바스켓, 그날 아침 공개)
D. 개미 가설: 국내 83종목, 네이버 일별 개인·외국인·기관 순매수(60거래일)로 앞 구간 수급 → 뒤 구간 수익
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import ai_ath_gap  # noqa: E402
from desk import giant  # noqa: E402
from gbrc.etf.flows import flows_between  # noqa: E402
from gbrc.etf.trades import load_fund_snapshots  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT = Path(__file__).resolve().parent
HORIZONS = (10, 20)
EVAL_START, STEP = "2026-02-02", 5
PERIODS = [("상승기", "2026-02-02", "2026-06-19"), ("급락기", "2026-06-22", "2026-07-29"),
           ("반등기", "2026-07-30", "2026-12-31")]
BUY, SELL = {"매수", "신규"}, {"매도", "편출"}


def load():
    rows = ai_ath_gap.universe_rows()
    uni = [s for _, _, s in rows]
    qqq = [t for t in (OUT / "qqq_members_2026-10-05.txt").read_text().split() if t != "GOOG"]
    syms = sorted(set(uni) | set(qqq) | {"SOXX", "QQQ", "QQQE"})
    with ThreadPoolExecutor(max_workers=8) as pool:
        bars = {s: b for s, b in pool.map(lambda s: ai_ath_gap.fetch(s, None), syms) if b is not None}
    return rows, uni, qqq, bars


def features(bars):
    """종목마다 시점별 종가·전고점 괴리(2010년 이후 고가 누적 최대)·50일 U/D·최근 21일 신고가 여부."""
    out = {}
    for s, b in bars.items():
        b = b[b.index >= "2010-01-01"]
        if len(b) < 2:
            continue
        c, hi, v = b["Close"], b["High"], b["Volume"].astype(float)
        ath = hi.cummax()
        chg = c.diff()
        up = v.where(chg > 0, 0.0).rolling(50).sum()
        dn = v.where(chg < 0, 0.0).rolling(50).sum()
        out[s] = pd.DataFrame({"c": c, "gap": c / ath - 1, "ud": up / dn.replace(0, np.nan),
                               "nh21": hi.rolling(21, min_periods=1).max() >= ath * (1 - 1e-9),
                               "n": np.arange(1, len(b) + 1)})
    return out


def asof(df, t):
    i = df.index.searchsorted(pd.Timestamp(t), side="right") - 1
    return df.iloc[i] if i >= 0 else None


def fwd(df, entry, exit_):
    a, b = asof(df, entry), asof(df, exit_)
    if a is None or b is None or a.name == b.name and entry != exit_:
        return None
    return float(b["c"] / a["c"] - 1)


def part_a(F, qqq, uni, cal):
    days = [d for d in cal if d >= pd.Timestamp("2026-01-02")]
    series = []
    for d in days:
        row = {"date": d.date().isoformat(), "qqq_gap": float(asof(F["QQQ"], d)["gap"])}
        for label, names in (("qqq", qqq), ("ai", uni)):
            vals = [asof(F[s], d) for s in names if s in F]
            vals = [v for v in vals if v is not None and v["n"] > 60]
            row[f"{label}_n"] = len(vals)
            row[f"{label}_near10"] = float(np.mean([v["gap"] >= -0.10 for v in vals]))
            row[f"{label}_nh21"] = float(np.mean([bool(v["nh21"]) and v["gap"] >= -0.10 for v in vals]))
        series.append(row)
    rets = {}
    for k in ("QQQ", "QQQE", "SOXX"):
        if k in F:
            rets[k] = {"ytd": fwd(F[k], "2025-12-31", days[-1]), "since_0729": fwd(F[k], "2026-07-29", days[-1])}
    return series, rets


def snapshot(F, uni, sub_of, t, anchor):
    """t 시점에 탭이 보였을 ◆ 네 조건(그 시점까지의 데이터만)."""
    feats = {}
    for s in uni:
        if s not in F:
            continue
        now = asof(F[s], t)
        if now is None or now["n"] < 60 or pd.isna(now["ud"]):
            continue
        low = asof(F[s], anchor["low_date"])
        feats[s] = {"since": float(now["c"] / low["c"] - 1) if low is not None else None,
                    "gap": float(now["gap"]), "ud": float(now["ud"]), "nh21": bool(now["nh21"]),
                    "r63": None}
        back = F[s].index.searchsorted(pd.Timestamp(t), side="right") - 1 - 63
        if back >= 0:
            feats[s]["r63"] = float(now["c"] / F[s]["c"].iloc[back] - 1)
    valid = {s: f for s, f in feats.items() if f["since"] is not None}
    uni_med = median(f["since"] for f in valid.values())
    by_sub = {}
    for s, f in valid.items():
        by_sub.setdefault(sub_of[s], []).append(s)
    for sub, members in by_sub.items():
        med = median(valid[s]["since"] for s in members)
        for rank, s in enumerate(sorted(members, key=lambda x: -valid[x]["since"]), 1):
            f = valid[s]
            f["c"] = [med > uni_med, rank <= giant.LEADERS_PER_SUB, f["gap"] >= ai_ath_gap.NEAR_PCT / 100,
                      f["ud"] >= 1.0]
    return valid


def baskets(valid):
    top_mom = sorted((s for s in valid if valid[s]["r63"] is not None), key=lambda s: -valid[s]["r63"])[:20]
    return {
        "◆ 네 조건": [s for s, f in valid.items() if all(f["c"])],
        "◆ 3개 이상": [s for s, f in valid.items() if sum(f["c"]) >= 3],
        "① 강한 섹터": [s for s, f in valid.items() if f["c"][0]],
        "② 섹터 1~2위": [s for s, f in valid.items() if f["c"][1]],
        "③ 신고가권(-10%)": [s for s, f in valid.items() if f["c"][2]],
        "④ 매집(U/D≥1)": [s for s, f in valid.items() if f["c"][3]],
        "최근 21일 신고가": [s for s, f in valid.items() if f["nh21"] and f["gap"] >= -0.10],
        "참고: 63일 수익 상위 20": top_mom,
        "AI 유니버스 전체": list(valid),
    }


def fund_sets(snaps, t, uni_set):
    """t 아침까지 공개된 PDF 기준 직전 20개 PDF의 실매매(두 펀드)."""
    out = {"buy": set(), "sell": set(), "TIME": set(), "KoAct": set()}
    for fund, s in snaps.items():
        dates = [d for d in sorted(s) if d <= t]
        if len(dates) < 21:
            continue
        res = flows_between(s, dates[-21], dates[-1])
        for key, fl in res.flows.items():
            sym = giant.universe_symbol(key, uni_set)
            if not sym:
                continue
            if fl.kind in BUY:
                out["buy"].add(sym)
                out[fund].add(sym)
            elif fl.kind in SELL:
                out["sell"].add(sym)
    out["both_buy"] = out["TIME"] & out["KoAct"]
    return out


def part_bc(F, uni, sub_of, cal):
    uni_set = set(uni)
    snaps = {f: load_fund_snapshots(fid, "2026-10-05") for f, fid in giant.FUNDS.items()}
    soxx = F["SOXX"]["c"]
    cal = list(cal)
    first = next(i for i, d in enumerate(cal) if d >= pd.Timestamp(EVAL_START))
    records = []
    for ti in range(first, len(cal) - 1, STEP):
        t = cal[ti]
        anchor = giant.find_anchor(soxx[soxx.index <= t])
        valid = snapshot(F, uni, sub_of, t, anchor)
        groups = baskets(valid)
        fs = fund_sets(snaps, t.date().isoformat(), uni_set)
        groups.update({"펀드 매수(20 PDF)": sorted(fs["buy"]), "펀드 매도(20 PDF)": sorted(fs["sell"]),
                       "두 펀드 모두 매수": sorted(fs["both_buy"])})
        for h in HORIZONS:
            if ti + 1 + h >= len(cal):
                continue
            entry, exit_ = cal[ti + 1], cal[ti + 1 + h]
            rets = {s: fwd(F[s], entry, exit_) for s in valid}
            for name, members in groups.items():
                vals = [rets[s] for s in members if rets.get(s) is not None]
                if vals:
                    records.append({"t": t.date().isoformat(), "h": h, "basket": name, "n": len(vals),
                                    "ret": float(np.mean(vals)), "anchor_low": anchor["low_date"],
                                    "members": members if name in ("◆ 네 조건", "두 펀드 모두 매수") else None})
    df = pd.DataFrame(records)
    base = df[df.basket == "AI 유니버스 전체"][["t", "h", "ret"]].rename(columns={"ret": "uni"})
    df = df.merge(base, on=["t", "h"])
    df["excess"] = df["ret"] - df["uni"]
    return df


def summarize_bc(df):
    out = []
    for (name, h), g in df.groupby(["basket", "h"], sort=False):
        for pname, a, b in [("전체", "2026-01-01", "2026-12-31")] + PERIODS:
            sub = g[(g.t >= a) & (g.t <= b)]
            if sub.empty:
                continue
            out.append({"basket": name, "h": h, "period": pname, "dates": len(sub),
                        "avg_n": round(sub.n.mean(), 1), "ret": round(sub.ret.mean() * 100, 2),
                        "excess": round(sub.excess.mean() * 100, 2), "median_excess": round(sub.excess.median() * 100, 2),
                        "hit": round((sub.excess > 0).mean() * 100)})
    return pd.DataFrame(out)


def compound(df, name, h=20):
    """겹치지 않는 20거래일 구간을 같은 날짜 격자로 이어 붙인다. 그날 바구니가 비면 현금(0%)."""
    grid = sorted(df[(df.basket == "AI 유니버스 전체") & (df.h == h)].t.unique())[::h // STEP]
    uni = df[(df.basket == "AI 유니버스 전체") & (df.h == h)].set_index("t").uni
    mine = df[(df.basket == name) & (df.h == h)].set_index("t").ret
    rets = [float(mine.get(t, 0.0)) for t in grid]
    return {"periods": len(grid), "basket": round((np.prod([1 + r for r in rets]) - 1) * 100, 1),
            "universe": round((np.prod([1 + float(uni[t]) for t in grid]) - 1) * 100, 1),
            "empty": sum(t not in mine.index for t in grid), "from": grid[0], "to": grid[-1]}


def naver_rows(code):
    r = requests.get(giant.NAVER_TREND.format(code=code), timeout=15,
                     headers={"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)"})
    return sorted(r.json(), key=lambda x: x["bizdate"])


def part_d(F, uni):
    cache = json.loads((ROOT / "data" / "derived" / "desk" / "giant.json").read_text(encoding="utf-8"))
    kr = [s for s in uni if s.endswith((".KS", ".KQ")) and cache["stocks"].get(s, {}).get("mcap")]
    with ThreadPoolExecutor(max_workers=8) as pool:
        flows = dict(zip(kr, pool.map(lambda s: naver_rows(s.split(".")[0]), kr)))
    tests = [("급락기 수급 → 반등기 수익", "20260707", "20260729", "2026-07-29", "2026-10-02"),
             ("반등 초반 수급 → 이후 수익", "20260730", "20260828", "2026-08-28", "2026-10-02"),
             ("같은 기간(참고)", "20260730", "20261002", "2026-07-29", "2026-10-02")]
    out = []
    per_stock = {}
    for name, fa, fb, ra, rb in tests:
        rows = []
        for s in kr:
            mcap = cache["stocks"][s]["mcap"]
            win = [r for r in flows[s] if fa <= r["bizdate"] <= fb]
            if not win or s not in F:
                continue
            val = lambda k: sum(giant._num(r[k]) * giant._num(r["closePrice"]) for r in win) / mcap * 100
            ret = fwd(F[s], ra, rb)
            if ret is None:
                continue
            rows.append({"s": s, "indiv": val("individualPureBuyQuant"), "foreign": val("foreignerPureBuyQuant"),
                         "inst": val("organPureBuyQuant"), "ret": ret * 100})
        d = pd.DataFrame(rows)
        d["whale"] = d.foreign + d.inst
        res = {"test": name, "flow_window": f"{fa}~{fb}", "return_window": f"{ra}~{rb}", "n": len(d)}
        for k in ("indiv", "foreign", "inst", "whale"):
            rho, p = spearmanr(d[k], d.ret)
            q = d[k].quantile([0.2, 0.8])
            res[k] = {"rho": round(float(rho), 2), "p": round(float(p), 3),
                      "top20_ret": round(float(d[d[k] >= q[0.8]].ret.median()), 1),
                      "bottom20_ret": round(float(d[d[k] <= q[0.2]].ret.median()), 1)}
        split = d[(d.indiv > 0) & (d.whale < 0)], d[(d.indiv < 0) & (d.whale > 0)]
        res["개인 매수·고래 매도"] = {"n": len(split[0]), "median_ret": round(float(split[0].ret.median()), 1) if len(split[0]) else None}
        res["개인 매도·고래 매수"] = {"n": len(split[1]), "median_ret": round(float(split[1].ret.median()), 1) if len(split[1]) else None}
        res["all_median_ret"] = round(float(d.ret.median()), 1)
        out.append(res)
        per_stock[name] = d.assign(name=d.s.map(ai_ath_gap.NAMES.get)).round(2).to_dict("records")
    return out, per_stock


def main():
    rows, uni, qqq, bars = load()
    sub_of = {s: (g, sub) for g, sub, s in rows}
    F = features(bars)
    cal = F["SOXX"].index[F["SOXX"].index >= "2025-06-01"]
    breadth, idx_rets = part_a(F, qqq, uni, cal)
    df = part_bc(F, uni, sub_of, cal)
    table = summarize_bc(df)
    comp = {name: compound(df, name) for name in df.basket.unique()}
    diamonds = df[(df.basket == "◆ 네 조건") & (df.h == 20)][["t", "n", "ret", "uni", "members"]]
    kr_tests, kr_stocks = part_d(F, uni)
    result = {"breadth": breadth, "index_returns": idx_rets, "basket_table": table.to_dict("records"),
              "compound_20d": comp, "diamond_history": diamonds.to_dict("records"),
              "kr_flow_tests": kr_tests, "kr_flow_stocks": kr_stocks}
    (OUT / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"index_returns": idx_rets, "last_breadth": breadth[-1],
                      "compound_20d": comp, "kr_flow_tests": kr_tests}, ensure_ascii=False, indent=1, default=str))
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(table[table.h == 20].to_string(index=False))
        print(table[(table.h == 10) & (table.period == "전체")].to_string(index=False))


if __name__ == "__main__":
    main()

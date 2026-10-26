"""회복 등급(S~D)은 그 뒤에도 유지되는가 (2026-10-10, Path-Q 연구).

사용자 질문: 거인의 어깨 탭의 회복 등급이 지금도 잘 유지되는지 장중 회복탄력성·고점 대비 하락률 등
여러 면에서 다시 보고, 유지되거나 깨지는 이유를 생각한다.

등급 계산은 desk/giant.py를 그대로 쓴다(find_anchor·own_recovery·grade_of·recovery_grade).
등급 시점 G에는 G까지의 종가만 쓴다. 평가 지표는 G 다음 날부터 END까지다(연도 내 전방 검증).

결과를 보기 전에 고정한 설정
  G1 = 기준 저점 + GRADE_LAG 거래일(저점 구간이 닫힌 첫날), G2 = 2026-09-08, END = 2026-10-08
  장중 급락일: 저가가 전일 종가보다 3% 이상 낮은 날. 회복 비율 = (종가-저가)/(전일 종가-저가)
  종가 위치 = (종가-저가)/(고가-저가), 모든 날 평균
  SOXX 하락일·상승일: SOXX 일간 수익률 -1% 이하 / +1% 이상(미국 상장만, 날짜가 같아야 하므로)
  지표 7개 × 등급 시점 2개 = 14개 비교. 다중 비교이므로 p 0.05 하나로 결론을 내지 않는다.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import ai_ath_gap  # noqa: E402
from desk import giant  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT = Path(__file__).resolve().parent
CACHE = OUT / "bars.csv.gz"
END = "2026-10-08"
G2 = "2026-09-08"
DIP = -0.03
SOXX_MOVE = 0.01
ORDER = ["S", "A", "B", "C", "D"]
COLS = ["Open", "High", "Low", "Close", "Volume"]


def load_bars() -> dict[str, pd.DataFrame]:
    """유니버스·벤치마크 일봉. 첫 실행에 받아 CSV로 두고 다시 쓴다(같은 날 재실행 재현용)."""
    if CACHE.exists():
        long = pd.read_csv(CACHE, parse_dates=["Date"])
        return {s: g.set_index("Date")[COLS] for s, g in long.groupby("symbol")}
    syms = [s for _, _, s in ai_ath_gap.universe_rows()] + giant.BENCHMARKS
    with ThreadPoolExecutor(max_workers=8) as pool:
        bars = {s: b[COLS] for s, b in pool.map(lambda s: ai_ath_gap.fetch(s, None), syms) if b is not None}
    # 등급 계산은 2010년 이후 고가만 쓴다(giant.own_recovery)
    bars = {s: b[b.index >= "2010-01-01"] for s, b in bars.items()}
    long = pd.concat([b.rename_axis("Date").reset_index().assign(symbol=s) for s, b in bars.items()])
    long.to_csv(CACHE, index=False)
    return bars


def upto(b: pd.DataFrame, day: str) -> pd.DataFrame:
    return b[b.index <= pd.Timestamp(day)]


def grades_at(bars: dict, day: str) -> tuple[dict, dict, dict]:
    """day까지의 자료만으로 종목 회복률·등급과 소분류 등급을 다시 계산한다."""
    anchor = giant.find_anchor(upto(bars[giant.ANCHOR], day)["Close"])
    stock, sub_members = {}, {}
    for group, sub, sym in ai_ath_gap.universe_rows():
        b = bars.get(sym)
        if b is None:
            continue
        b = upto(b, day)
        if len(b) < 60:
            continue
        rec = giant.own_recovery(b, anchor)
        stock[sym] = {"recovery": rec, "grade": giant.grade_of(rec), "sub": sub, "group": group,
                      "market": ai_ath_gap.market_of(sym)}
        sub_members.setdefault(sub, []).append(stock[sym])
    subs = {sub: giant.recovery_grade(ms) for sub, ms in sub_members.items()}
    return anchor, stock, subs


def window_metrics(b: pd.DataFrame, start: str, end: str, soxx_ret: pd.Series) -> dict | None:
    """start 종가에서 출발해 end까지의 성과·낙폭·장중 회복 지표."""
    b = b[(b.index >= pd.Timestamp(start)) & (b.index <= pd.Timestamp(end))]
    if len(b) < 10:
        return None
    c, h, lo = b["Close"], b["High"], b["Low"]
    prev = c.shift(1)
    path = c / c.iloc[0]
    dd = path / path.cummax() - 1
    dip_day = (lo / prev - 1) <= DIP
    rec_share = ((c - lo) / (prev - lo))[dip_day & ((prev - lo) > 0)]
    rng = (h - lo).where((h - lo) > 0)
    clv = ((c - lo) / rng).iloc[1:]
    out = {
        "ret": round(float(path.iloc[-1] - 1) * 100, 2),
        "mdd": round(float(dd.min()) * 100, 2),
        "off_high": round(float(dd.iloc[-1]) * 100, 2),
        "dip_days": int(dip_day.sum()),
        "dip_recover": round(float(rec_share.mean()) * 100, 1) if len(rec_share) else None,
        "clv": round(float(clv.mean()) * 100, 1) if clv.notna().any() else None,
        "sessions": len(b) - 1,
    }
    r = c.pct_change().iloc[1:]
    s = soxx_ret.reindex(r.index)
    down, up = s <= -SOXX_MOVE, s >= SOXX_MOVE
    out["soxx_down"] = round(float(r[down].mean()) * 100, 2) if down.sum() >= 3 else None
    out["soxx_up"] = round(float(r[up].mean()) * 100, 2) if up.sum() >= 3 else None
    return out


METRICS = [
    ("ret", "평가 구간 수익률"),
    ("mdd", "구간 내 최대 낙폭"),
    ("off_high", "구간 고점 대비 지금"),
    ("dip_recover", "장중 -3% 밀린 날 종가까지 되찾은 비율"),
    ("clv", "종가 위치(그날 고저 범위 중)"),
    ("soxx_down", "SOXX -1% 하락일 평균(미국)"),
    ("soxx_up", "SOXX +1% 상승일 평균(미국)"),
]


def compare(stock_g: dict, metrics: dict) -> dict:
    """등급별 중앙값, 회복률과의 Spearman, S·A 대 C·D 비교."""
    rows = [(s, g) for s, g in stock_g.items() if g["recovery"] is not None and s in metrics]
    out = {}
    for key, label in METRICS:
        pairs = [(g["recovery"], metrics[s][key], g["grade"]) for s, g in rows if metrics[s].get(key) is not None]
        if len(pairs) < 10:
            continue
        rec, val, grd = zip(*pairs)
        rho, p = spearmanr(rec, val)
        by = {k: [v for v, gg in zip(val, grd) if gg == k] for k in ORDER}
        hi, lo_ = by["S"] + by["A"], by["C"] + by["D"]
        mw = mannwhitneyu(hi, lo_).pvalue if len(hi) >= 5 and len(lo_) >= 5 else None
        out[key] = {
            "label": label, "n": len(pairs), "rho": round(float(rho), 2), "p": round(float(p), 3),
            "median": {k: (round(median(v), 2) if v else None) for k, v in by.items()},
            "count": {k: len(v) for k, v in by.items()},
            "SA": round(median(hi), 2) if hi else None, "CD": round(median(lo_), 2) if lo_ else None,
            "p_SA_CD": round(float(mw), 3) if mw is not None else None,
        }
    return out


def sub_compare(subs: dict, stock_g: dict, metrics: dict) -> dict:
    """소분류 등급(3종목 이상)과 소분류 지표 중앙값의 순위 상관."""
    out = {}
    for key, label in METRICS:
        xs, ys = [], []
        for sub, sg in subs.items():
            if sg["recovery"] is None:
                continue
            vals = [metrics[s][key] for s, g in stock_g.items()
                    if g["sub"] == sub and s in metrics and metrics[s].get(key) is not None]
            if len(vals) >= 3:
                xs.append(sg["recovery"])
                ys.append(median(vals))
        if len(xs) >= 8:
            rho, p = spearmanr(xs, ys)
            out[key] = {"label": label, "n": len(xs), "rho": round(float(rho), 2), "p": round(float(p), 3)}
    return out


def transitions(a: dict, b: dict) -> dict:
    """등급 a → 등급 b 이동표(둘 다 등급이 있는 종목)."""
    table = {k: {kk: 0 for kk in ORDER} for k in ORDER}
    for s, ga in a.items():
        gb = b.get(s, {})
        if ga.get("grade") and gb.get("grade"):
            table[ga["grade"]][gb["grade"]] += 1
    same = sum(table[k][k] for k in ORDER)
    total = sum(sum(r.values()) for r in table.values())
    return {"table": table, "same": same, "total": total}


def pre_beta(bars: dict, sym: str, day: str, n: int = 60) -> float:
    """day까지 n거래일 SOXX 대비 베타(미국 상장만, 사전 값)."""
    ret = upto(bars[sym], day)["Close"].iloc[-(n + 1):].pct_change().dropna()
    s = bars["SOXX"]["Close"].pct_change().reindex(ret.index)
    ok = s.notna()
    return float(np.cov(ret[ok], s[ok])[0, 1] / s[ok].var())


def partial_rho(x, y, ctrl) -> tuple[float, float]:
    """순위로 바꾼 뒤 ctrl을 걷어 낸 x·y의 Spearman."""
    c = np.column_stack([np.ones(len(x)), rankdata(ctrl)])
    res = [r - c @ np.linalg.lstsq(c, r, rcond=None)[0] for r in (rankdata(x), rankdata(y))]
    rho, p = spearmanr(*res)
    return round(float(rho), 2), round(float(p), 3)


def beta_control(bars: dict, stock_g: dict, day: str, ys: dict) -> dict:
    """등급 효과가 사전 베타(등락 폭)를 맞춘 뒤에도 남는지. ys = {이름: {종목: 값}}."""
    us = [s for s, g in stock_g.items() if g["recovery"] is not None and g["market"] == "미국"]
    beta = {s: pre_beta(bars, s, day) for s in us}
    out = {"n": len(us), "beta_median": {k: round(median([beta[s] for s in us if stock_g[s]["grade"] == k]), 2)
                                         for k in ORDER if any(stock_g[s]["grade"] == k for s in us)}}
    for name, vals in ys.items():
        ss = [s for s in us if vals.get(s) is not None]
        x, y, b = [stock_g[s]["recovery"] for s in ss], [vals[s] for s in ss], [beta[s] for s in ss]
        rho, p = spearmanr(x, y)
        out[name] = {"raw": [round(float(rho), 2), round(float(p), 3)], "beta_adj": list(partial_rho(x, y, b)),
                     "beta_only": round(float(spearmanr(b, y)[0]), 2)}
    return out


def main() -> None:
    bars = load_bars()
    full_anchor = giant.find_anchor(upto(bars[giant.ANCHOR], END)["Close"])
    g1 = full_anchor["grade_to"]
    soxx = upto(bars["SOXX"], END)["Close"]
    soxx_ret = soxx.pct_change()

    result = {"end": END, "anchor": full_anchor, "dates": {}}
    snaps = {}
    for label, day in (("G1", g1), ("G2", G2), ("D1006", "2026-10-06"), ("END", END)):
        anchor, stock_g, subs = grades_at(bars, day)
        snaps[label] = (day, stock_g, subs)
        result["dates"][label] = {"day": day, "anchor": [anchor["peak_date"], anchor["low_date"]],
                                  "graded": sum(1 for g in stock_g.values() if g["grade"])}

    for label in ("G1", "G2"):
        day, stock_g, subs = snaps[label]
        metrics = {}
        for s in stock_g:
            m = window_metrics(upto(bars[s], END), day, END, soxx_ret)
            if m is not None:
                if ai_ath_gap.market_of(s) != "미국":
                    m["soxx_down"] = m["soxx_up"] = None
                metrics[s] = m
        result["dates"][label].update(
            soxx=window_metrics(upto(bars["SOXX"], END), day, END, soxx_ret),
            stocks=compare(stock_g, metrics),
            stocks_overseas=compare({s: g for s, g in stock_g.items() if g["market"] != "한국"}, metrics),
            subs=sub_compare(subs, stock_g, metrics),
            grade_counts={k: sum(1 for g in stock_g.values() if g["grade"] == k) for k in ORDER},
            per_stock={s: {**stock_g[s], **metrics.get(s, {})} for s in stock_g if stock_g[s]["grade"]},
        )

    result["transitions"] = {
        "G1->END": transitions(snaps["G1"][1], snaps["END"][1]),
        "G2->END": transitions(snaps["G2"][1], snaps["END"][1]),
        "D1006->END": transitions(snaps["D1006"][1], snaps["END"][1]),
    }
    result["sub_grades"] = {
        label: {sub: [sg["grade"], sg["recovery"], sg["recovery_n"]] for sub, sg in snaps[label][2].items()
                if sg["grade"]}
        for label in ("G1", "G2", "D1006", "END")
    }
    # 최근 하락(10/6 종가→END)을 10/6 등급으로: 미국 상장만(아시아 10/8 종가는 미국 10/7까지만 반영)
    _, g06, _ = snaps["D1006"]
    latest = {}
    for s, g in g06.items():
        if g["grade"] and g["market"] == "미국":
            b = upto(bars[s], END)
            c06, c_end = giant.on_or_before(b["Close"], "2026-10-06"), float(b["Close"].iloc[-1])
            last = b.iloc[-1]
            latest[s] = {"grade": g["grade"], "recovery": g["recovery"], "ret": (c_end / c06 - 1) * 100,
                         "clv_1008": float((last["Close"] - last["Low"]) / (last["High"] - last["Low"]) * 100)
                         if last["High"] > last["Low"] else None}
    rho, p = spearmanr([v["recovery"] for v in latest.values()], [v["ret"] for v in latest.values()])
    result["latest_dip_us"] = {
        "n": len(latest), "rho": round(float(rho), 2), "p": round(float(p), 3),
        "ret_median": {k: round(median([v["ret"] for v in latest.values() if v["grade"] == k]), 2)
                       for k in ORDER if any(v["grade"] == k for v in latest.values())},
        "clv_1008_median": {k: round(median([v["clv_1008"] for v in latest.values()
                                             if v["grade"] == k and v["clv_1008"] is not None]), 1)
                            for k in ORDER if any(v["grade"] == k for v in latest.values())},
        "count": {k: sum(1 for v in latest.values() if v["grade"] == k) for k in ORDER},
    }
    # 등락 폭(사전 60거래일 SOXX 베타)을 맞춘 뒤에도 등급 효과가 남나(미국 상장)
    result["beta_control"] = {
        label: beta_control(bars, snaps[label][1], snaps[label][0],
                            {k: {s: v.get(k) for s, v in result["dates"][label]["per_stock"].items()}
                             for k in ("off_high", "mdd", "ret")})
        for label in ("G1", "G2")
    }
    result["beta_control"]["latest_dip"] = beta_control(bars, g06, "2026-10-06",
                                                        {"ret": {s: v["ret"] for s, v in latest.items()}})
    # 회복률은 급락 깊이를 얼마나 따라가나(G1 시점, 종목 자신의 고점→저점)
    anchor_g1 = giant.find_anchor(upto(bars[giant.ANCHOR], g1)["Close"])
    depth, recs = [], []
    for s, g in snaps["G1"][1].items():
        if g["recovery"] is None:
            continue
        c = upto(bars[s], g1)["Close"]
        win = c[(c.index >= pd.Timestamp(anchor_g1["grade_from"])) & (c.index <= pd.Timestamp(anchor_g1["low_date"]))]
        pk_at = win.idxmax()
        low = float(c[(c.index >= pk_at) & (c.index <= pd.Timestamp(anchor_g1["grade_to"]))].min())
        depth.append((low / float(win.max()) - 1) * 100)
        recs.append(g["recovery"])
    rho, p = spearmanr(depth, recs)
    by_grade = {k: [d for d, r in zip(depth, recs) if giant.grade_of(r) == k] for k in ORDER}
    result["depth_vs_recovery_G1"] = {"n": len(depth), "rho": round(float(rho), 2), "p": round(float(p), 4),
                                      "depth_median": {k: round(median(v), 1) for k, v in by_grade.items() if v}}
    after = soxx[soxx.index > pd.Timestamp(full_anchor["low_date"])]
    hi_at = after.idxmax()
    result["soxx_path"] = {"post_low_high": [hi_at.date().isoformat(), round(float(after.max()), 2)],
                           "end": round(float(soxx.iloc[-1]), 2),
                           "off_high": round((float(soxx.iloc[-1]) / float(after.max()) - 1) * 100, 2),
                           "last10": [[d.date().isoformat(), round(float(v), 2)] for d, v in soxx.iloc[-10:].items()]}
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str),
                                     encoding="utf-8")

    print("anchor", full_anchor["peak_date"], "->", full_anchor["low_date"], "G1", g1)
    print("soxx", result["soxx_path"])
    for label in ("G1", "G2"):
        d = result["dates"][label]
        print(f"\n=== {label} {d['day']} → {END}  등급 수 {d['grade_counts']}  SOXX {d['soxx']}")
        for scope in ("stocks", "stocks_overseas"):
            print(f"  [{scope}]")
            for r in d[scope].values():
                med = " ".join(f"{k}:{r['median'][k]}({r['count'][k]})" for k in ORDER)
                print(f"   {r['label']:<28} n{r['n']:>3} ρ{r['rho']:+.2f} p{r['p']:.3f} | SA {r['SA']} CD {r['CD']} "
                      f"p{r['p_SA_CD']} | {med}")
        print("  [subs]")
        for r in d["subs"].values():
            print(f"   {r['label']:<28} n{r['n']:>3} ρ{r['rho']:+.2f} p{r['p']:.3f}")
    for k, t in result["transitions"].items():
        print(f"\n{k}: 같은 등급 {t['same']}/{t['total']}")
        for g in ORDER:
            print("  ", g, t["table"][g])
    print("\nlatest dip (10/6→END, 미국)", result["latest_dip_us"])
    print("depth vs recovery G1", result["depth_vs_recovery_G1"])
    for k, v in result["beta_control"].items():
        print("beta control", k, v)


if __name__ == "__main__":
    main()

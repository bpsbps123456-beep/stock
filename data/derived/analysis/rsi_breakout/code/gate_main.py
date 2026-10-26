"""수익 우선 + 현금 전환 시기(2026-10-10, 연구용). desk/·gbrc/·docs/·watchlist.py와 기존 연구 파일은 고치지 않는다.

질문(사용자, 2026-10-10): 9·10월에도 그랬나? 반도체 쏠림은 괜찮고, 승률보다 수익률이 중요하며,
                       현금 비중을 높이는 시기를 두는 건 좋은 전략이다.
섹터 이름: TIME 미분류 56개를 분류한 뒤(커밋 d5ce794)의 watchlist.py. 그래서 '반도체 예산'은 이제 진짜 반도체(+DXYZ)다.

사전 등록 격자(결과를 본 뒤 바꾸지 않음) — 바탕 B 4 × 현금 전환 G 4 = 16개
  B  live50 : 데스크 규칙(TIME 미국, 120일 상위 7, 버퍼 5, TIME 비중, 종목 25%·섹터 60%·반도체 50%)
     c774   : 같은데 반도체 60%(#774)
     nocap  : 반도체 예산 없음(종목 25%·섹터 60%만)
     sub5eq : 버킷 T-sub-N5-eq(bucket_main.py, 세부 섹터 버킷당 1개, 5종목 같은 비중, 반도체 예산 없음)
  G  none   : 현금 전환 없음
     soxx_w : 재조정일 결정 종가에 SOXX < SMA20이면 그 주 반도체 예산 종목을 현금으로(bucket_main 사후 b와 같음)
     soxx_d : 매 거래일 전날 종가로 SOXX < SMA20을 보고, 아래로 가면 그날 시가에 반도체 종목을 팔고,
              위로 돌아오면 그날 시가에 이번 주 목표 비중(재조정 때 값, 남은 현금 안에서)으로 되산다
     qqq_w  : 재조정일 결정 종가에 QQQ < SMA20이면 그 주 전부 현금
선택: 앞 1/2~5/29 수익 1위. 확인: 뒤 6/1~10/9. 함께 보고: 9월·10월·9/1~10/9·전체·최대 낙폭·현금 평균.
강건성(선택 근거 아님, 바탕 4 × {none, 선택된 G}):
  결정 요일(그 ISO 주에서 그 요일 이후 첫 거래일) 월~금, SMA 10·20·50(G가 SMA를 쓸 때),
  2023-05-22~2025-12-31 보조(verify_bucket_holdout.py 독립 엔진, ISIN 3종목 복원, 공시 +1세션 'lag1'와 앞선 PDF 'strict').
시점: k일 시가 결정은 k일보다 앞선 PDF와 k-1일 종가만 쓴다. 수익: USD 시가→시가, 수수료 0(§1), 스프레드 미반영.
재현 확인: live50·none = desk run_model(현재 RULE), c774·none = run_model(#774 상한) 최대 차 < 1e-9.
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/rsi_breakout/code/gate_main.py
결과: data/derived/analysis/rsi_breakout/results/gate/main.json
"""
import json, os, sys
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bucket_main import BucketEngine, BParams, SOXX_SRC  # noqa: E402
from deskx import START  # noqa: E402
from desk import config  # noqa: E402
from desk.strategies import run_model, iso_week, in_semis_budget, UNKNOWN  # noqa: E402
from desk.validate import codex_caps  # noqa: E402

OUT = os.path.join(HERE, "../results/gate/main.json")
FRONT, BACK, SEP, FULL = "앞 1/2~5/29", "뒤 6/1~10/9", "9/1~10/9", "전체"
SHOW_MONTHS = ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]
BASES = {"live50": BParams(gran="name", wmode="fund", n=7, semis_cap=0.50),
         "c774": BParams(gran="name", wmode="fund", n=7, semis_cap=0.60),
         "nocap": BParams(gran="name", wmode="fund", n=7, semis_cap=1.0),
         "sub5eq": BParams(gran="sub", wmode="eq", n=5)}
GATES = ("none", "soxx_w", "soxx_d", "qqq_w")


def index_series(sym):
    o = pd.read_parquet(SOXX_SRC, columns=["date", "sym", "Close"])
    o = o[o.sym == sym].copy()
    o["date"] = pd.to_datetime(o["date"]).dt.strftime("%Y-%m-%d")
    s = o.set_index("date")["Close"].astype(float).sort_index()
    return s[~s.index.duplicated()]


class Signals:
    def __init__(self):
        self.px = {s: index_series(s) for s in ("SOXX", "QQQ")}
        self.sma = {}

    def below(self, sym, day, n):
        """day 종가(없으면 그 전 마지막 값)가 SMA n 아래인가."""
        if (sym, n) not in self.sma:
            self.sma[sym, n] = self.px[sym].rolling(n, min_periods=n).mean()
        s, m = self.px[sym], self.sma[sym, n]
        i = s.index.searchsorted(day, side="right") - 1
        if i < 0 or not np.isfinite(m.iloc[i]):
            return False
        return bool(s.iloc[i] < m.iloc[i])


def gate_sym(g):
    return {"soxx_w": "SOXX", "soxx_d": "SOXX", "qqq_w": "QQQ"}.get(g)


def apply_gate(tgt, g, on, semis):
    if not on or g == "none":
        return dict(tgt)
    if g == "qqq_w":
        return {}
    return {x: v for x, v in tgt.items() if not semis(x)}


def weekday(d):
    return date.fromisoformat(d).weekday()


def run_gated(E, sig, p, g, sma=20, wd=0):
    """Engine.run과 같은 시가 경로(손절 없음)에 현금 전환과 결정 요일을 얹는다."""
    semis = lambda x: in_semis_budget(E.sector.get(x, UNKNOWN))
    sym = gate_sym(g)
    days = [d for d in E.days if d >= START]
    k0 = E.days.index(days[0])
    value, rows = [1.0], []
    w, base, selected = {}, {}, set()
    started, on, last_wk = False, False, None
    switches = 0
    for k in range(k0, len(E.days)):
        day = E.days[k]
        if k > k0:
            rets = {}
            for key in w:
                o = E.col("opn", key)
                rets[key] = (o[k] / o[k - 1] - 1) if o is not None and np.isfinite(o[k]) and np.isfinite(o[k - 1]) \
                    and o[k - 1] > 0 else 0.0
            gross = sum(wt * rets[key] for key, wt in w.items())
            total = sum(wt * (1 + rets[key]) for key, wt in w.items()) + (1 - sum(w.values()))
            w = {key: wt * (1 + rets[key]) / total for key, wt in w.items()}
            value.append(value[-1] * (1 + gross))
        signal = sym is not None and sig.below(sym, E.days[k - 1], sma)
        wk = iso_week(day)
        reb = (not started) or (wk != last_wk and weekday(day) >= wd)
        if reb:
            keep, tgt, ranked = E.target(p, k, day, selected)
            if ranked:
                started, last_wk = True, wk
                base, selected = dict(tgt), set(keep)
                if g != "none" and signal != on:
                    switches += 1
                on = signal and g != "none"
                w = apply_gate(base, g, on, semis)
        elif g == "soxx_d" and started and signal != on:
            switches += 1
            if signal:
                w = {x: v for x, v in w.items() if not semis(x)}
            else:
                add = {x: v for x, v in base.items() if semis(x)}
                room, need = 1 - sum(w.values()), sum(add.values())
                scale = min(1.0, room / need) if need > 0 else 0.0
                for x, v in add.items():
                    w[x] = w.get(x, 0.0) + v * scale
            on = signal
        rows.append(dict(day=day, w=dict(w), on=on))
    return pd.Series(value, index=days), rows, switches


def summarize(E, name, v, rows, switches):
    r = E.evaluate(name, v, rows)
    out = {"name": name}
    for lab in (FRONT, BACK, SEP, FULL):
        out[lab] = r[lab]["수익"]
    out["낙폭"] = r[FULL]["낙폭"]
    out["월별"] = {m: r["월별"][m] for m in SHOW_MONTHS}
    out["반도체평균"] = r["반도체평균"]
    out["현금평균"] = float(np.mean([1 - sum(row["w"].values()) for row in rows]))
    out["전환일비율"] = float(np.mean([row["on"] for row in rows]))
    out["전환횟수"] = switches
    out["마지막보유"] = r["마지막보유"]
    return out


def line(r):
    m = " ".join(f"{r['월별'][k] * 100:+6.1f}" for k in SHOW_MONTHS)
    return (f"{r['name']:<16} 앞 {r[FRONT] * 100:+7.1f}  뒤 {r[BACK] * 100:+6.1f}  9/1~ {r[SEP] * 100:+6.1f}  "
            f"전체 {r[FULL] * 100:+7.1f}  낙폭 {r['낙폭'] * 100:6.1f}  | 6~10월 {m} | 현금 {r['현금평균'] * 100:4.1f}% "
            f"전환 {r['전환횟수']}")


# ---------------------------------------------------------------- 2023~2025 보조
def holdout(bases, gates, sig):
    import verify_bucket_holdout as V
    D = V.Data(restore=True)
    rules = {"live50": V.desk_rule(semis=0.50), "c774": V.desk_rule(semis=0.60),
             "nocap": V.desk_rule(semis=10.0), "sub5eq": V.bucket_rule("sub", 5, "eq")}
    out = {"unclassified": sorted({D.tk[k] for k, s in D.sector.items() if s == UNKNOWN}), "runs": {}}

    def sim(rule, mode, g):
        semis = lambda x: in_semis_budget(D.sector.get(x, UNKNOWN))
        sym = gate_sym(g)
        k0 = next(i for i, d in enumerate(D.days) if d >= V.START)
        k1 = max(i for i, d in enumerate(D.days) if d <= V.END)
        sched = V.schedule(D, mode, k0, k1)
        val = np.ones(k1 - k0 + 1)
        w, base, cur, on, n_on = {}, {}, [], False, 0
        for k in range(k0, k1 + 1):
            t = k - k0
            if t > 0:
                g_, new = 0.0, {}
                for key, wt in w.items():
                    j = D.j[key]
                    a, b = D.O[k - 1, j], D.O[k, j]
                    r = b / a - 1 if np.isfinite(a) and np.isfinite(b) and a > 0 else 0.0
                    g_ += wt * r
                    new[key] = wt * (1 + r)
                tot = sum(new.values()) + 1 - sum(w.values())
                w = {key: x / tot for key, x in new.items()}
                val[t] = val[t - 1] * (1 + g_)
            signal = sym is not None and sig.below(sym, D.days[k - 1], 20)
            if k in sched:
                res = rule(D, k, sched[k], cur, ())
                if res is not None:
                    cur, base = list(res[0]), dict(res[1])
                    on = signal and g != "none"
                    w = apply_gate(base, g, on, semis)
            elif g == "soxx_d" and base and signal != on:
                if signal:
                    w = {x: v for x, v in w.items() if not semis(x)}
                else:
                    add = {x: v for x, v in base.items() if semis(x)}
                    room, need = 1 - sum(w.values()), sum(add.values())
                    scale = min(1.0, room / need) if need > 0 else 0.0
                    for x, v in add.items():
                        w[x] = w.get(x, 0.0) + v * scale
                on = signal
            n_on += on
        v = pd.Series(val, index=D.days[k0:k1 + 1])
        st = V.stats(D, v, [{"day": D.days[k0]}])
        return {"전체": st["full"], "낙폭": st["mdd"], "연도": st["years"], "급락": st["stress"],
                "전환일비율": n_on / len(v)}

    for b in bases:
        for g in gates:
            for mode in ("lag1", "strict"):
                out["runs"][f"{b}·{g}·{mode}"] = sim(rules[b], mode, g)
    return out


def main():
    E = BucketEngine()
    sig = Signals()
    # 재현 확인
    v, rows, _ = run_gated(E, sig, BASES["live50"], "none")
    ref = run_model(E.book, start=START)
    d_live = float(np.max(np.abs(v.to_numpy() - np.asarray(ref.value))))
    v6, _, _ = run_gated(E, sig, BASES["c774"], "none")
    with codex_caps():
        ref6 = run_model(E.book, start=START)
    d_774 = float(np.max(np.abs(v6.to_numpy() - np.asarray(ref6.value))))
    print(f"재현: live50 vs run_model(반도체 {config.RULE['semis_cap']:.0%}) 최대 차 {d_live:.2e}, c774 vs #774 {d_774:.2e}")
    assert d_live < 1e-9 and d_774 < 1e-9

    grid = {}
    for b, p in BASES.items():
        for g in GATES:
            v, rows, sw = run_gated(E, sig, p, g)
            grid[f"{b}·{g}"] = summarize(E, f"{b}·{g}", v, rows, sw)
    print("\n== 격자 16개 (USD 시가 경로, 2026-01-02~10-09)")
    for r in sorted(grid.values(), key=lambda r: -r[FRONT]):
        print(line(r))
    best = max(grid.values(), key=lambda r: r[FRONT])
    bb, bg = best["name"].split("·")
    print(f"\n앞 구간 1위: {best['name']}  (뒤 {best[BACK] * 100:+.1f}%, 9/1~ {best[SEP] * 100:+.1f}%)")

    # 강건성: 결정 요일, SMA 길이
    gates_r = ("none",) + ((bg,) if bg != "none" else ("soxx_w",))
    rob = {"요일": {}, "SMA": {}}
    for b, p in BASES.items():
        for g in gates_r:
            for wd in range(5):
                v, rows, sw = run_gated(E, sig, p, g, wd=wd)
                r = summarize(E, f"{b}·{g}·요일{wd}", v, rows, sw)
                rob["요일"][r["name"]] = {k: r[k] for k in (FRONT, BACK, SEP, FULL, "낙폭")} | {"월별": r["월별"]}
            if g != "none":
                for n in (10, 20, 50):
                    v, rows, sw = run_gated(E, sig, p, g, sma=n)
                    r = summarize(E, f"{b}·{g}·SMA{n}", v, rows, sw)
                    rob["SMA"][r["name"]] = {k: r[k] for k in (FRONT, BACK, SEP, FULL, "낙폭")} | {"전환횟수": r["전환횟수"]}
    print("\n== 결정 요일(월=0) 강건성: 전체 / 뒤 / 낙폭")
    for b in BASES:
        for g in gates_r:
            xs = [rob["요일"][f"{b}·{g}·요일{wd}"] for wd in range(5)]
            print(f"{b}·{g:<7} 전체 " + " ".join(f"{x[FULL] * 100:+6.1f}" for x in xs)
                  + "  | 뒤 " + " ".join(f"{x[BACK] * 100:+5.1f}" for x in xs)
                  + "  | 낙폭 " + " ".join(f"{x['낙폭'] * 100:5.1f}" for x in xs))
    print("\n== SMA 길이 강건성: 앞 / 뒤 / 전체 / 낙폭 / 전환")
    for k, x in rob["SMA"].items():
        print(f"{k:<24} 앞 {x[FRONT] * 100:+6.1f} 뒤 {x[BACK] * 100:+5.1f} 전체 {x[FULL] * 100:+6.1f} "
              f"낙폭 {x['낙폭'] * 100:5.1f} 전환 {x['전환횟수']}")

    ho = holdout(BASES, GATES, sig)
    print(f"\n== 2023-05-22~2025-12-31 보조(ISIN 복원). 남은 미분류: {ho['unclassified']}")
    for k, x in ho["runs"].items():
        y = " ".join(f"{a} {b * 100:+6.1f}" for a, b in x["연도"].items())
        print(f"{k:<24} 전체 {x['전체'] * 100:+7.1f} 낙폭 {x['낙폭'] * 100:6.1f} | {y} | 전환 {x['전환일비율'] * 100:4.1f}%")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"reproduce": {"live50": d_live, "c774": d_774}, "grid": grid, "best_front": best["name"],
                   "robust": rob, "holdout": ho}, f, ensure_ascii=False, indent=1, default=str)
    print("저장:", OUT)


if __name__ == "__main__":
    main()

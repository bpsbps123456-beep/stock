"""gate_main.py 적대 검증 #1 — 2026 독립 재계산(2026-10-10, 연구용). 기존 파일은 고치지 않는다.

gate_main.run_gated/apply_gate를 가져오지 않고 경로를 새로 짠다.
  - 가격·TIME 비중·섹터: BucketEngine(얼린 가격 복사본, deskx._use_frozen_prices)에서 읽기만.
  - 목표 비중: live50·c774·nocap은 순위·버퍼 선택을 여기서 다시 짜고 desk.strategies.project_caps로 상한만 적용
    (E.target과 매 재조정 비교), sub5eq는 BucketEngine.target(버킷 선택)을 그대로 쓴다.
  - 경로: 비중이 아니라 '달러 보유액 + 현금'으로 굴린다. 결측 시가는 기본 0% 수익(데스크와 같은 관례),
    대안으로 '마지막 유효 시가'로 평가한 경로도 함께 잰다.
  - 신호: stop_research ohlc.parquet SOXX/QQQ 종가를 E.days[k-1] 날짜로 정확히 찾고(없으면 오류),
    SMA는 그날까지 n행 평균을 직접 계산(rolling 미사용).
  - 지표: 시가 가치 V에서 구간 [a,b] = V[b 다음 세션 시가]/V[a 첫 세션 시가]-1, 낙폭은 시작 1.0 포함.
결정 요일 wd: 각 ISO 주에서 weekday ≥ wd인 첫 거래일(그런 날이 없으면 그 주 건너뜀 = gate_main 정의),
  대안 'roll': 없으면 그 주 마지막 거래일.
실행: $env:PYTHONIOENCODING='utf-8'; py data/derived/analysis/rsi_breakout/code/verify_gate_2026.py
결과: data/derived/analysis/rsi_breakout/results/gate/verify_gate_2026.json
"""
import json, os, sys
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bucket_main import BucketEngine, BParams, SOXX_SRC  # noqa: E402  (가격·순위·E.target만)
from desk import config  # noqa: E402
from desk.strategies import project_caps, in_semis_budget, iso_week, UNKNOWN  # noqa: E402

OUT = os.path.join(HERE, "../results/gate/verify_gate_2026.json")
MAIN = os.path.join(HERE, "../results/gate/main.json")
START, END = "2026-01-02", "2026-10-09"
PER = {"앞": ("2026-01-02", "2026-05-29"), "뒤": ("2026-06-01", "2026-10-09"), "9/1~": ("2026-09-01", "2026-10-09"),
       "9월": ("2026-09-01", "2026-09-30"), "10월": ("2026-10-01", "2026-10-31"), "전체": (START, END),
       "6월": ("2026-06-01", "2026-06-30"), "7월": ("2026-07-01", "2026-07-31"), "8월": ("2026-08-01", "2026-08-31")}
CAPS = {"live50": 0.50, "c774": 0.60, "nocap": 1.0}
SUB5 = BParams(gran="sub", wmode="eq", n=5)


class Data:
    def __init__(self):
        self.E = E = BucketEngine()
        self.days = list(E.days)
        self.K = len(self.days)
        self.k0 = next(i for i, d in enumerate(self.days) if d >= START)
        self.sector = dict(E.sector)
        self.pdfs = sorted(E.book.pdfs)
        self._o, self._a = {}, {}
        o = pd.read_parquet(SOXX_SRC, columns=["date", "sym", "Close"])
        o["d"] = pd.to_datetime(o["date"]).dt.strftime("%Y-%m-%d")
        self.close = {}
        for sym in ("SOXX", "QQQ"):
            s = o[o.sym == sym].drop_duplicates("d").set_index("d")["Close"].astype(float).sort_index()
            self.close[sym] = s
        self._sig = {}

    def opn(self, key):
        if key not in self._o:
            self._o[key] = self.E.col("opn", key)
        return self._o[key]

    def adj(self, key):
        if key not in self._a:
            self._a[key] = self.E.col("adj", key)
        return self._a[key]

    def semis(self, key):
        return in_semis_budget(self.sector.get(key, UNKNOWN))

    def below(self, sym, k, n):
        """k일 시가 결정용: E.days[k-1] 종가 < 그날까지 n행 평균. 날짜가 정확히 없으면 오류."""
        ck = (sym, k, n)
        if ck not in self._sig:
            s = self.close[sym]
            d = self.days[k - 1]
            pos = s.index.get_loc(d)              # KeyError면 날짜 불일치
            assert pos + 1 >= n
            sma = float(s.iloc[pos - n + 1:pos + 1].mean())
            self._sig[ck] = bool(s.iloc[pos] < sma)
        return self._sig[ck]

    # ------------------------------------------------ 목표 비중
    def my_target(self, cap, k, session, current):
        prior = [p for p in self.pdfs if p < session]
        if not prior:
            return None
        tw = self.E.tw[prior[-1]]
        rows = []
        for key in sorted(tw):
            if tw[key] <= 0:
                continue
            px = self.adj(key)
            if px is None or k - 1 - 120 < 0:
                continue
            p1, p0 = px[k - 1], px[k - 1 - 120]
            if not (np.isfinite(p1) and np.isfinite(p0) and p0 > 0):
                continue
            rows.append((key, p1 / p0 - 1))
        if not rows:
            return None
        rows.sort(key=lambda r: -r[1])
        order = [r[0] for r in rows]
        keep = [x for x in order[:12] if x in current][:7]
        for x in order:
            if len(keep) >= 7:
                break
            if x not in keep:
                keep.append(x)
        tot = sum(tw[x] for x in keep)
        raw = {x: tw[x] / tot for x in keep}
        saved = dict(config.RULE)
        try:
            config.RULE.update(stock_cap=0.25, sector_cap=0.60, semis_cap=cap)
            tgt = project_caps(raw, self.sector)
        finally:
            config.RULE.clear()
            config.RULE.update(saved)
        return keep, tgt

    def target(self, base, k, session, current, check=None):
        if base == "sub5eq":
            keep, tgt, ranked = self.E.target(SUB5, k, session, current)
            return (keep, tgt) if ranked else None
        res = self.my_target(CAPS[base], k, session, current)
        if check is not None:
            p = BParams(gran="name", wmode="fund", n=7, semis_cap=CAPS[base])
            keep2, tgt2, ranked2 = self.E.target(p, k, session, current)
            if res is None:
                check.append(0.0 if not ranked2 else 1.0)
            else:
                d = max([abs(res[1].get(x, 0) - tgt2.get(x, 0)) for x in set(res[1]) | set(tgt2)] + [0.0])
                d += 0.0 if keep2 == res[0] else 1.0
                check.append(d)
        return res


def schedule_ok(D, k, wd, mode, done_weeks):
    day = D.days[k]
    wk = iso_week(day)
    if wk in done_weeks:
        return False
    if date.fromisoformat(day).weekday() >= wd:
        return True
    if mode == "roll":       # 그 주에 wd 이후 거래일이 없고 오늘이 그 주 마지막 거래일이면 오늘
        nxt = D.days[k + 1] if k + 1 < D.K else None
        if nxt is None or iso_week(nxt) != wk:
            later = [d for d in D.days[k + 1:] if iso_week(d) == wk]
            return not later
    return False


def simulate(D, base, gate, wd=0, sma=20, missing="zero", sched="skip", check=None, reentry="scale"):
    sym = {"soxx_w": "SOXX", "soxx_d": "SOXX", "qqq_w": "QQQ"}.get(gate)
    hold, cash = {}, 1.0
    last_px = {}
    started, done, selected, base_w, state = False, set(), set(), {}, False
    V, W, ST, switches, rebs = [], [], [], 0, []
    nan_hits = 0
    for k in range(D.k0, D.K):
        day = D.days[k]
        if k > D.k0:
            for key in list(hold):
                o = D.opn(key)
                a, b = (o[k - 1], o[k]) if o is not None else (np.nan, np.nan)
                if missing == "zero":
                    r = b / a - 1 if np.isfinite(a) and np.isfinite(b) and a > 0 else 0.0
                    if not (np.isfinite(a) and np.isfinite(b)):
                        nan_hits += 1
                else:                       # 마지막 유효 시가로 평가
                    if np.isfinite(b) and b > 0:
                        r = b / last_px[key] - 1 if key in last_px else 0.0
                        last_px[key] = b
                    else:
                        r = 0.0
                        nan_hits += 1
                hold[key] *= 1 + r
        val = cash + sum(hold.values())
        V.append(val)
        sig = sym is not None and D.below(sym, k, sma)
        is_reb = (not started) or schedule_ok(D, k, wd, sched, done)
        acted = False
        if is_reb:
            res = D.target(base, k, day, selected, check)
            if res is not None:
                keep, tgt = res
                started = True
                done.add(iso_week(day))
                selected, base_w = set(keep), dict(tgt)
                on = gate != "none" and sig
                if on != state and gate != "none":
                    switches += 1
                if on and gate == "qqq_w":
                    w = {}
                elif on:
                    w = {x: v for x, v in tgt.items() if not D.semis(x)}
                else:
                    w = dict(tgt)
                hold = {x: val * v for x, v in w.items()}
                cash = val - sum(hold.values())
                state = on
                rebs.append(day)
                acted = True
        if not acted and gate == "soxx_d" and started and sig != state:
            switches += 1
            if sig:
                for x in [x for x in hold if D.semis(x)]:
                    cash += hold.pop(x)
            else:
                add = {x: v for x, v in base_w.items() if D.semis(x)}
                need = sum(add.values()) * val
                if reentry == "scale":
                    scale = min(1.0, cash / need) if need > 0 else 0.0
                    for x, v in add.items():
                        amt = v * val * scale
                        hold[x] = hold.get(x, 0.0) + amt
                        cash -= amt
                else:                          # 대안: 되살 때 이번 주 목표 전체로 다시 맞춤
                    hold = {x: val * v for x, v in base_w.items()}
                    cash = val - sum(hold.values())
            state = sig
        for x in hold:
            o = D.opn(x)
            if o is not None and np.isfinite(o[k]) and o[k] > 0:
                last_px[x] = o[k]
        W.append({x: h / val for x, h in hold.items()})
        ST.append(state)
    return pd.Series(V, index=D.days[D.k0:]), W, ST, switches, rebs, nan_hits


def metrics(D, v, W, ST):
    days = list(v.index)
    x = v.to_numpy()
    out = {}
    for lab, (a, b) in PER.items():
        ia = next(i for i, d in enumerate(days) if d >= a)
        ib = max(i for i, d in enumerate(days) if d <= b)
        ie = min(ib + 1, len(days) - 1)
        out[lab] = float(x[ie] / x[ia] - 1)
    out["MDD"] = float(np.min(x / np.maximum.accumulate(x) - 1))
    out["현금평균"] = float(np.mean([1 - sum(w.values()) for w in W]))
    out["전환일비율"] = float(np.mean(ST))
    sem = {}
    for m in ("2026-07", "2026-08", "2026-09", "2026-10"):
        sem[m] = float(np.mean([sum(wt for k, wt in w.items() if D.semis(k)) for d, w in zip(days, W)
                                if d.startswith(m)]))
    out["반도체비중"] = sem
    return out


def main():
    D = Data()
    main_json = json.load(open(MAIN, encoding="utf-8"))
    grid, rob = main_json["grid"], main_json["robust"]
    keymap = {"앞": "앞 1/2~5/29", "뒤": "뒤 6/1~10/9", "9/1~": "9/1~10/9", "전체": "전체", "MDD": "낙폭"}
    month = {"6월": "2026-06", "7월": "2026-07", "8월": "2026-08", "9월": "2026-09", "10월": "2026-10"}

    def gap(mine, ref):
        g = {}
        for a, b in keymap.items():
            g[a] = mine[a] - ref[b]
        for a, b in month.items():
            g[a] = mine[a] - ref["월별"][b]
        return g

    res = {"grid": {}, "gaps": {}, "target_check": {}, "nan_hits": {}, "missing_alt": {}}
    print("== 2026 격자 독립 재계산 (gate_main 값과의 차, %p)")
    worst = 0.0
    for base in ("live50", "c774", "nocap", "sub5eq"):
        for gate in ("none", "soxx_w", "soxx_d", "qqq_w"):
            chk = [] if base != "sub5eq" else None
            v, W, ST, sw, rebs, nh = simulate(D, base, gate, check=chk)
            m = metrics(D, v, W, ST)
            m["전환횟수"] = sw
            m["재조정수"] = len(rebs)
            name = f"{base}·{gate}"
            res["grid"][name] = m
            g = gap(m, grid[name])
            g["현금평균"] = m["현금평균"] - grid[name]["현금평균"]
            g["전환횟수"] = sw - grid[name]["전환횟수"]
            res["gaps"][name] = g
            res["nan_hits"][name] = nh
            if chk is not None:
                res["target_check"][name] = max(chk) if chk else None
            worst = max(worst, max(abs(val) for kk, val in g.items() if kk != "전환횟수"))
            v2, W2, ST2, *_ = simulate(D, base, gate, missing="ffill")
            m2 = metrics(D, v2, W2, ST2)
            res["missing_alt"][name] = {k: m2[k] - m[k] for k in ("앞", "뒤", "전체", "MDD")}
            print(f"{name:<14} 앞 {m['앞']*100:+7.2f} 뒤 {m['뒤']*100:+6.2f} 9월 {m['9월']*100:+6.2f} 10월 {m['10월']*100:+5.2f} "
                  f"전체 {m['전체']*100:+7.2f} MDD {m['MDD']*100:6.2f} 현금 {m['현금평균']*100:4.1f} 전환 {sw:2d} | "
                  f"최대차 {max(abs(val) for kk, val in g.items() if kk != '전환횟수')*100:.2e}%p 전환차 {g['전환횟수']} "
                  f"목표차 {res['target_check'].get(name)} 결측 {nh}")
    print(f"최대 절대차(%p): {worst*100:.3e}")
    res["max_gap_pp"] = worst * 100

    # 순위
    rank_front = sorted(res["grid"], key=lambda n: -res["grid"][n]["앞"])
    rank_full = sorted(res["grid"], key=lambda n: -res["grid"][n]["전체"])
    print("앞 상위3:", [(n, round(res["grid"][n]["앞"] * 100, 2)) for n in rank_front[:3]])
    print("전체 상위3:", [(n, round(res["grid"][n]["전체"] * 100, 2)) for n in rank_full[:3]])
    res["rank_front"], res["rank_full"] = rank_front, rank_full

    # C4: soxx_d·qqq_w vs none
    c4 = {}
    for base in ("live50", "c774", "nocap", "sub5eq"):
        n0 = res["grid"][f"{base}·none"]
        for gate in ("soxx_d", "qqq_w", "soxx_w"):
            x = res["grid"][f"{base}·{gate}"]
            c4[f"{base}·{gate}"] = {k: round((x[k] - n0[k]) * 100, 2) for k in ("앞", "뒤", "전체", "MDD")}
    res["vs_none"] = c4
    print("\n== 게이트 - none (%p):")
    for k, x in c4.items():
        print(f"  {k:<16} {x}")

    # C5 반도체 비중(9월)
    print("\n== 9월 수익 / 9월 평균 반도체 예산 비중(none)")
    for base in ("live50", "c774", "nocap", "sub5eq"):
        x = res["grid"][f"{base}·none"]
        print(f"  {base:<7} 9월 {x['9월']*100:+6.2f}  반도체 {x['반도체비중']['2026-09']*100:5.1f}%  10월 {x['10월']*100:+5.2f}")

    # 요일 강건성
    print("\n== 결정 요일(월=0) 강건성 — 전체 / 앞 / 뒤 / MDD")
    res["weekday"] = {}
    for base in ("live50", "c774", "nocap", "sub5eq"):
        for gate in ("none", "soxx_w"):
            for sched in ("skip", "roll"):
                row = []
                for wd in range(5):
                    v, W, ST, sw, rebs, _ = simulate(D, base, gate, wd=wd, sched=sched)
                    m = metrics(D, v, W, ST)
                    m["재조정수"] = len(rebs)
                    m["전환횟수"] = sw
                    row.append(m)
                    if sched == "skip":
                        ref = rob["요일"][f"{base}·{gate}·요일{wd}"]
                        m["gap_전체"] = m["전체"] - ref["전체"]
                        m["gap_max"] = max(abs(gap(m, ref)[kk]) for kk in keymap)
                res["weekday"][f"{base}·{gate}·{sched}"] = row
                extra = ("  최대차 " + " ".join(f"{r['gap_max']*100:.1e}" for r in row)) if sched == "skip" else ""
                print(f"{base}·{gate:<7}[{sched}] 전체 " + " ".join(f"{r['전체']*100:+6.1f}" for r in row)
                      + " | 앞 " + " ".join(f"{r['앞']*100:+6.1f}" for r in row)
                      + " | 뒤 " + " ".join(f"{r['뒤']*100:+5.1f}" for r in row)
                      + " | MDD " + " ".join(f"{r['MDD']*100:5.1f}" for r in row)
                      + " | 재조정 " + " ".join(str(r['재조정수']) for r in row) + extra)
    for base in ("live50", "c774", "nocap", "sub5eq"):
        for sched in ("skip", "roll"):
            a = res["weekday"][f"{base}·none·{sched}"]
            b = res["weekday"][f"{base}·soxx_w·{sched}"]
            print(f"  {base} soxx_w-none [{sched}] 전체 " + " ".join(f"{(y['전체']-x['전체'])*100:+6.1f}" for x, y in zip(a, b))
                  + " | 앞 " + " ".join(f"{(y['앞']-x['앞'])*100:+6.1f}" for x, y in zip(a, b))
                  + " | 뒤 " + " ".join(f"{(y['뒤']-x['뒤'])*100:+5.1f}" for x, y in zip(a, b))
                  + " | MDD " + " ".join(f"{(y['MDD']-x['MDD'])*100:+5.1f}" for x, y in zip(a, b)))

    # soxx_d 되사기 방식 대안
    print("\n== soxx_d 되사기 대안(이번 주 목표 전체로 다시 맞춤) - 원래")
    res["soxx_d_reentry_full"] = {}
    for base in ("live50", "c774", "nocap", "sub5eq"):
        v, W, ST, sw, *_ = simulate(D, base, "soxx_d", reentry="full")
        m = metrics(D, v, W, ST)
        res["soxx_d_reentry_full"][base] = m
        n0 = res["grid"][f"{base}·none"]
        print(f"  {base:<7} 앞 {m['앞']*100:+7.2f} 뒤 {m['뒤']*100:+6.2f} 전체 {m['전체']*100:+7.2f} MDD {m['MDD']*100:6.2f}"
              f"  (none 대비 전체 {(m['전체']-n0['전체'])*100:+.1f}%p)")

    # SMA 강건성(live50)
    print("\n== SMA 길이(live50·soxx_w, 월요일)")
    res["sma"] = {}
    for n in (10, 15, 20, 30, 50):
        v, W, ST, sw, *_ = simulate(D, "live50", "soxx_w", sma=n)
        m = metrics(D, v, W, ST)
        res["sma"][n] = m
        print(f"  SMA{n:<3} 앞 {m['앞']*100:+7.2f} 뒤 {m['뒤']*100:+6.2f} 전체 {m['전체']*100:+7.2f} MDD {m['MDD']*100:6.2f} 전환 {sw}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print("저장:", os.path.abspath(OUT))


if __name__ == "__main__":
    main()

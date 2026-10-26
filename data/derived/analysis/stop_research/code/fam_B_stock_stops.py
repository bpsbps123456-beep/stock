"""갈래 B: 종목별 손절 주문(장중) + 재진입 (2026-10-10).

데스크 7종목 규칙(주간 교체, 반도체 50%)은 그대로 두고, 보유 종목마다 손절 주문을 얹는다(H.StockEngine).
이전 연구(종가 고점 대비 10·15·20% 추적, 판 자리 현금/다음 순위)는 기각됐다(판 종목 다음 재조정까지 평균 +3%, 73% 재매수).
이번에는 (1) 장중 스탑 주문(고정·추적·ATR 샹들리에)과 (2) 재진입 방식(현금 대기·1/2주 금지·주중 재진입)을 잰다.

시점
  - 손절선: k일 장중 주문 가격 L은 k-1일 종가까지 아는 값(산 가격, 산 뒤 최고 종가·고가, k-1까지 ATR14)으로 정한다.
    체결: 시가 ≤ L이면 시가, 저가 ≤ L이면 L (× (1-slip)). 정규장 고가·저가(ohlc.parquet) 기준.
  - 종가 판정(비교용): k-1일 종가가 선 아래면 k일 시가에 판다(exit_open).
  - 주중 재진입(reup): k-1일 종가 > 손절 체결가×1.05 또는 > 직전 10일 최고 종가면 k일 시가에 그 주 목표 비중까지 다시 산다.
  - 금지(ban1/ban2): 손절 뒤 처음 1·2번의 주간 재조정에서 그 종목 몫을 현금으로 둔다.
격자는 아래 상수로 사전 고정한다(2026 뒤 구간·보유 기간을 보고 바꾸지 않는다).
실행: PYTHONIOENCODING=utf-8 py fam_B_stock_stops.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402

FAMILY = "B_stock_stops"

# ----------------------------------------------------------------------------- 격자(사전 고정)
INTRADAY_STOPS = ([("fixed", x) for x in (0.08, 0.12, 0.15, 0.20)]       # 산 가격 대비 고정 %
                  + [("trail", x) for x in (0.15, 0.20, 0.25, 0.30)]     # 산 뒤 최고 종가 대비 추적 %
                  + [("chand", m) for m in (2, 3, 4, 5)])                # 산 뒤 최고 고가 - m×ATR14
REENTRY = ("cash", "ban1", "ban2", "reup")
CLOSE_STOPS = [("fixed", 0.12), ("fixed", 0.20), ("trail", 0.20), ("trail", 0.30), ("chand", 3), ("chand", 5)]
CLOSE_REENTRY = ("cash", "reup")
ATR_N = 14
REUP_MULT = 1.05          # 종가 > 손절 체결가 × 1.05
REUP_HIGH_N = 10          # 또는 종가 > 직전 10일 최고 종가
SLIP_CHECK = 0.005        # 앞 고르기 변형을 다시 잴 미끄러짐
REBUY_WINDOW = 15         # '다시 샀나' 판정 창(거래일)

KIND_KO = {"fixed": "고정", "trail": "추적", "chand": "ATR샹들리에"}
RE_KO = {"cash": "재조정까지 현금", "ban1": "1주 금지", "ban2": "2주 금지", "reup": "주중 재진입"}

WIN_FEBMAR = [("2/3~2/5", "2026-02-03", "2026-02-05"), ("2/25~3/6", "2026-02-25", "2026-03-06"),
              ("3/19~3/30", "2026-03-19", "2026-03-30")]
WIN_JULY = ("AI쇼크 6/22~7/29", "2026-06-22", "2026-07-29")
REBOUND_N = 10            # 하락 구간 끝 다음 10거래일을 '반등 구간'으로 본다


def spec_name(mode, kind, lvl, re):
    lv = f"{lvl:.0%}" if kind != "chand" else f"{lvl}ATR"
    return f"{'장중' if mode == 'intraday' else '종가'} {KIND_KO[kind]} {lv} / {RE_KO[re]}"


def grid():
    out = [("intraday", k, l, r) for (k, l) in INTRADAY_STOPS for r in REENTRY]
    out += [("close", k, l, r) for (k, l) in CLOSE_STOPS for r in CLOSE_REENTRY]
    return out


# ----------------------------------------------------------------------------- 정책
class StopPolicy(H.NoPolicy):
    """엔진 콜백. 기록용으로 엔진 지역변수 w(그날 시가 거래 뒤 비중)를 호출 프레임에서 읽는다(공유 파일 수정 없이 노출 기록)."""

    def __init__(self, eng, mode, kind, lvl, reentry):
        self.eng, self.mode, self.kind, self.lvl, self.reentry = eng, mode, kind, lvl, reentry
        self.reb_k = np.array(sorted(eng.k_of[d] for d in eng.reb))
        self.last_stop: dict = {}
        self.exp: dict = {}
        self.stop_w: dict = {}
        self.entries: dict = {}            # (key, entry_k) -> entry px
        self._atr: dict = {}

    def atr(self, key, k):
        if (key, k) not in self._atr:
            self._atr[(key, k)] = self.eng.atr(key, k, ATR_N)
        return self._atr[(key, k)]

    def level(self, key, k, st):
        if self.kind == "fixed":
            return st["entry"] * (1 - self.lvl)
        if self.kind == "trail":
            return st["peak"] * (1 - self.lvl)
        a = self.atr(key, k)
        return st["high_peak"] - self.lvl * a if np.isfinite(a) else float("nan")

    @staticmethod
    def _w():
        f = sys._getframe(1)
        while f is not None and not (f.f_code.co_name == "run" and "w" in f.f_locals):
            f = f.f_back
        return f.f_locals["w"]

    # 재조정 때 금지
    def allow(self, key, k, st, eng):
        if self.reentry not in ("ban1", "ban2") or key not in self.last_stop:
            return True
        n = 1 if self.reentry == "ban1" else 2
        s = self.last_stop[key]
        passed = int(((self.reb_k > s) & (self.reb_k <= k)).sum())
        return passed > n

    # 종가 판정(비교용)
    def exit_open(self, key, k, st, eng):
        if self.mode != "close":
            return False
        c = eng.C[key][k - 1]
        L = self.level(key, k, st)
        if np.isfinite(c) and np.isfinite(L) and c <= L:
            self.last_stop[key] = k
            self.stop_w[(key, k)] = float(self._w().get(key, np.nan))
            return True
        return False

    def reenter(self, key, k, st, eng):
        if self.reentry != "reup" or k <= st.get("stopped_k", k):
            return False
        c = eng.C[key][k - 1]
        sp = st.get("stop_px", eng.O[key][st["stopped_k"]])
        prev = eng.C[key][max(0, k - 1 - REUP_HIGH_N):k - 1]
        prev = prev[np.isfinite(prev)]
        if not np.isfinite(c):
            return False
        return bool(c > sp * REUP_MULT or (len(prev) == REUP_HIGH_N and c > prev.max()))

    # 장중 주문(그리고 매일 보유 기록)
    def stop(self, key, k, st, eng):
        w = self._w()
        self.exp[k] = float(sum(w.values()))
        self.entries.setdefault((key, st["entry_k"]), float(st["entry"]))
        if self.mode != "intraday":
            return None
        L = self.level(key, k, st)
        if not np.isfinite(L):
            return None
        o, lo = eng.O[key][k], eng.L[key][k]
        if (np.isfinite(o) and o <= L) or (np.isfinite(lo) and lo <= L):
            self.last_stop[key] = k
            self.stop_w[(key, k)] = float(w[key])
        return L


class NeverPolicy(StopPolicy):
    def stop(self, key, k, st, eng):
        super().stop(key, k, st, eng)
        return None


# ----------------------------------------------------------------------------- 실행·통계
ENG: dict = {}


def engine(bn):
    if bn not in ENG:
        ENG[bn] = H.StockEngine(bn)
    return ENG[bn]


def exposure(eng, pol):
    b = eng.b
    vals = []
    for d, wt in zip(b.days, b.weights):
        bs = sum(wt.values())
        vals.append(1.0 if bs < 1e-9 else min(1.0, pol.exp.get(eng.k_of[d], 0.0) / bs))
    return pd.Series(vals, index=b.days)


def stop_stats(eng, pol, events, lo=None, hi=None):
    """손절 사건: 체결 → 다음 재조정 시가까지 그 종목 수익, 재매수 여부·가격, 비중 가중 효과(+ = 놓친 상승)."""
    b = eng.b
    inv = {v: k for k, v in eng.tk.items()}
    day_of = {v: d for d, v in eng.k_of.items()}
    reb_idx = sorted(b.days.index(d) for d in eng.reb if d in b.days)
    rows = []
    for ev in events:
        if ev["type"] not in ("stop", "exit_open"):
            continue
        d = ev["day"]
        if lo and not (lo <= d <= hi):
            continue
        key, i = inv[ev["key"]], b.days.index(ev["day"])
        k = eng.k_of[d]
        nxt = next((j for j in reb_idx if j > i), len(b.days) - 1)
        kn = eng.k_of[b.days[nxt]]
        r_next = eng.O[key][kn] / ev["px"] - 1
        reb = sorted((ek, px) for (kk, ek), px in pol.entries.items() if kk == key and k < ek <= k + REBUY_WINDOW)
        if ev["type"] == "exit_open":
            reb = [(ek, px) for ek, px in reb if ek > k]
        w = pol.stop_w.get((key, k), np.nan)
        rows.append({"day": d, "tk": ev["key"], "type": ev["type"], "px": ev["px"], "w": w, "r_next": r_next,
                     "rebuy": bool(reb), "rebuy_above": bool(reb and reb[0][1] > ev["px"]),
                     "rebuy_day": day_of[reb[0][0]] if reb else None,
                     "rebuy_rel": (reb[0][1] / ev["px"] - 1) if reb else None})
    if not rows:
        return {"n": 0}, rows
    r = np.array([x["r_next"] for x in rows])
    wr = np.array([x["w"] * x["r_next"] for x in rows])
    nb = sum(x["rebuy"] for x in rows)
    return {"n": len(rows), "next_mean": float(r.mean()), "next_med": float(np.median(r)), "next_up": float((r > 0).mean()),
            "w_effect_sum": float(np.nansum(wr)), "rebuy_share": nb / len(rows),
            "rebuy_above_share": (sum(x["rebuy_above"] for x in rows) / nb) if nb else None}, rows


def run_variant(spec, slip=0.0, keep_detail=False):
    mode, kind, lvl, re = spec
    out_r, out_e, det = {}, {}, {}
    for bn in ("2026", "hold"):
        eng = engine(bn)
        pol = StopPolicy(eng, mode, kind, lvl, re)
        r, ev = eng.run(pol, slip=slip)
        out_r[bn], out_e[bn] = r, exposure(eng, pol)
        st_all, rows = stop_stats(eng, pol, ev)
        det[bn] = {"stats": st_all}
        if bn == "2026":
            det[bn]["ai"] = stop_stats(eng, pol, ev, WIN_JULY[1], WIN_JULY[2])[0]
        if keep_detail:
            det[bn].update(events=ev, rows=rows, pol=pol)
    name = spec_name(*spec) + (f" (slip {slip:.1%})" if slip else "")
    res = H.evaluate_returns(name, out_r, {"mode": mode, "kind": kind, "level": lvl, "reentry": re, "slip": slip},
                             out_e)
    for bn in ("2026", "hold"):
        res[bn]["손절"] = det[bn]["stats"]
    res["2026"]["손절_AI쇼크"] = det["2026"]["ai"]
    return res, out_r, out_e, det


def win_ret(r, lo, hi):
    return H.stats(r, lo, hi)["수익"]


def after(days, d, n):
    i = days.index(d) if d in days else next(j for j, x in enumerate(days) if x > d)
    return days[min(len(days) - 1, i + 1)], days[min(len(days) - 1, i + n)]


def behavior(spec, label):
    """앞 고르기(또는 대체) 변형의 7월·2~3월 행동."""
    res, rr, ee, det = run_variant(spec, keep_detail=True)
    b, eng = H.base("2026"), engine("2026")
    r, e, base_r = rr["2026"], ee["2026"], b.ret
    rows = det["2026"]["rows"]
    out = {"label": label, "name": res["name"], "trace": H.trace(e)}
    wins = [WIN_JULY] + WIN_FEBMAR
    out["windows"] = []
    for nm, lo, hi in wins:
        r0, r1 = after(b.days, hi, REBOUND_N)
        ev = [x for x in rows if lo <= x["day"] <= hi]
        out["windows"].append({
            "구간": nm, "기본": win_ret(base_r, lo, hi), "변형": win_ret(r, lo, hi),
            "반등구간": f"{r0}~{r1}", "기본_반등": win_ret(base_r, r0, r1), "변형_반등": win_ret(r, r0, r1),
            "손절수": len(ev), "다시삼": sum(x["rebuy"] for x in ev),
            "손절가위재매수": sum(x["rebuy_above"] for x in ev),
            "평균노출_구간": float(e[(e.index >= lo) & (e.index <= hi)].mean()),
            "사건": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()} for x in ev]})
    out["july_events_6_15_9_30"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}
                                    for x in rows if "2026-06-15" <= x["day"] <= "2026-09-30"]
    out["exp_daily_6_15_8_14"] = {d: round(v, 3) for d, v in e["2026-06-15":"2026-08-14"].items()}
    out["events_reenter"] = [x for x in det["2026"]["events"] if x["type"] == "reenter"
                             and "2026-06-15" <= x["day"] <= "2026-09-30"]
    return out


def fmt(r):
    a, h = r["2026"], r["hold"]
    s = a["손절"]
    sm = f"손절 {s['n']:>3} 다음재조정 {s.get('next_mean', float('nan')):+.1%}/{s.get('next_med', float('nan')):+.1%} 재매수 {s.get('rebuy_share') or 0:.0%}" if s["n"] else "손절   0"
    return (f"{r['name']:<34} 전체 {a['전체']['수익']:+.0%}/{a['전체']['낙폭']:.1%} 앞 {a['앞']['수익']:+.0%}/{a['앞']['낙폭']:.1%} "
            f"뒤 {a['뒤']['수익']:+.1%}/{a['뒤']['낙폭']:.1%} AI {a['AI쇼크']['수익']:+.1%} | 25 {h['2025']['수익']:+.0%}/{h['2025']['낙폭']:.0%} "
            f"24 {h['2024']['수익']:+.0%}/{h['2024']['낙폭']:.0%} | 노출 {a['평균노출']:.2f} | {sm}")


def main():
    # 0) 엔진 재현 확인: 아무 손절도 안 거는 정책이 기본 경로와 같아야 한다
    for bn in ("2026", "hold"):
        eng = engine(bn)
        pol = NeverPolicy(eng, "intraday", "fixed", 0.5, "cash")
        r, _ = eng.run(pol)
        d = float(np.max(np.abs(r.to_numpy() - H.base(bn).ret.to_numpy())))
        e = exposure(eng, pol)
        print(f"[재현] {bn} 최대 차 {d:.1e}, 노출 최소 {e.min():.4f}")
        assert d < 1e-6 and e.min() > 0.999

    specs = grid()
    print("격자", len(specs))
    rows, by_name = [], {}
    for sp in specs:
        res = run_variant(sp)[0]
        rows.append(res)
        by_name[res["name"]] = sp
        print(fmt(res))

    br = H.base_result()
    picks = {k: H.select_front(rows, br, kp) for k, kp in (("keep90", 0.90), ("keep80", 0.80), ("keep70", 0.70))}
    par = sorted(H.pareto_2026(rows, br, 0.90), key=lambda r: -r["2026"]["전체"]["수익"])

    # 1) 앞 고르기 변형을 미끄러짐 0.5%로 다시
    slip_rows = {}
    for k, v in picks.items():
        if v is not None:
            rs = run_variant(by_name[v["name"]], slip=SLIP_CHECK)[0]
            slip_rows[k] = H.pick_row(f"{k} slip0.5%", rs)
            print(f"[{k} slip]", fmt(rs))
    par_slip = []
    for v in par[:3]:
        rs = run_variant(by_name[v["name"]], slip=SLIP_CHECK)[0]
        par_slip.append(H.pick_row("사후 파레토 slip0.5%", rs))
        print("[파레토 slip]", fmt(rs))

    # 2) 행동 분석 대상: keep90 → keep80 → keep70 → 사후 최선
    tgt_label, tgt = next(((k, v) for k, v in picks.items() if v is not None), (None, None))
    if tgt is None and par:
        tgt_label, tgt = "사후 파레토 최선", par[0]
    beh = [behavior(by_name[tgt["name"]], tgt_label)] if tgt is not None else []
    # 비교용으로 손절 수가 많은 대표(장중 추적 20%·현금, ATR 3·주중 재진입)도 같은 표로
    for sp in (("intraday", "trail", 0.20, "cash"), ("intraday", "chand", 3, "reup")):
        if tgt is None or by_name[tgt["name"]] != sp:
            beh.append(behavior(sp, "비교"))
    for bh in beh:
        print(f"\n== 행동: {bh['label']} {bh['name']}")
        for w in bh["windows"]:
            print(f"  {w['구간']}: 기본 {w['기본']:+.1%} 변형 {w['변형']:+.1%} | 반등 {w['반등구간']} 기본 {w['기본_반등']:+.1%} "
                  f"변형 {w['변형_반등']:+.1%} | 손절 {w['손절수']} 다시삼 {w['다시삼']} (손절가 위 {w['손절가위재매수']}) 노출 {w['평균노출_구간']:.2f}")
        for x in bh["july_events_6_15_9_30"]:
            print("   ", x)

    stats_tbl = {r["name"]: {"2026": r["2026"]["손절"], "hold": r["hold"]["손절"], "AI쇼크": r["2026"]["손절_AI쇼크"]}
                 for r in rows}
    grid_desc = (f"장중 손절 {len(INTRADAY_STOPS)}종(고정 8·12·15·20%, 추적 15·20·25·30%, ATR14 샹들리에 m=2·3·4·5) × "
                 f"재진입 {len(REENTRY)}종(재조정까지 현금·1주 금지·2주 금지·주중 재진입[종가>손절가×1.05 또는 10일 최고 종가]) = "
                 f"{len(INTRADAY_STOPS) * len(REENTRY)}, + 종가 판정 비교 {len(CLOSE_STOPS)}종(고정 12·20%, 추적 20·30%, ATR 3·5) × "
                 f"{len(CLOSE_REENTRY)}(현금·주중 재진입) = {len(CLOSE_STOPS) * len(CLOSE_REENTRY)}; 합계 {len(specs)}. "
                 f"미끄러짐 0(앞 고르기·파레토 상위 3개만 0.5% 재측정).")
    extra = {"slip_picks": slip_rows, "slip_pareto_top3": par_slip, "stop_stats": stats_tbl, "behavior": beh,
             "params": {"ATR_N": ATR_N, "REUP_MULT": REUP_MULT, "REUP_HIGH_N": REUP_HIGH_N, "SLIP_CHECK": SLIP_CHECK,
                        "REBUY_WINDOW": REBUY_WINDOW, "REBOUND_N": REBOUND_N}}
    print()
    H.summarize_family(FAMILY, rows, grid_desc, extra)


if __name__ == "__main__":
    main()

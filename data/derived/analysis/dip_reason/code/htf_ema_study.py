"""160분봉 RSI 과열 + EMA 10·25·60 구조로 3분봉 이격 매수를 조심할지(2026-10-07 사용자 가설).
가설: 160분 RSI14 ≥ 70이고 동시에 160분 EMA 10·25·60이 정배열이 아니거나(역배열·엉킴) 서로 지나치게 가까우면 조심한다.
정배열이면 반등이 강하다고 본다.
값은 모두 매수 순간에 안다: dev = 지금 가격(매수선)을 진행 중 160분봉 종가로 넣은 RSI·EMA, conf = 마지막 완성 160분봉 값.
160분 EMA60은 5분봉 시작(7/10)에서 60봉이 지나야 생긴다(7/27 이후). 그 전에는 조건이 꺼진다.
조합:
  RSI 값 dev/conf(EMA도 같은 쪽) × RSI 기준 60·65·70·75 또는 없음
  구조: 정배열 아님 / 역배열 / 엉킴 / 간격 < 1·2·3·5% / 정배열 아님 또는 간격 < X(사용자 안) / 대조: 정배열이고 간격 ≥ X
  조심: 안 삼 / -4% · 범위: 24시간 / 정규장만 · 기간 앞(5분봉 근사)·뒤(3분봉) · 종목 사용자 78·참고 127
사용: py htf_ema_study.py <raw 폴더> <출력 폴더>"""
import os, sys, time
import numpy as np, pandas as pd
from htf_study import PERIODS, base_arrays, run_cfg, metrics, NOBUY
from htf import intraday_bars, rma_state, _rsi_from, ema_state
from fetch import universe, user_universe

TF = 160
XS = [0.01, 0.02, 0.03, 0.05]


def feats(raw, s, A):
    """기준봉마다 160분 RSI14·EMA10/25/60(dev, conf)와 구조."""
    d = A["d"]
    t, p = d.index, A["p2"]
    m5 = pd.read_parquet(f"{raw}/5m_{s}.parquet")
    hs, hc = intraday_bars(m5, TF)
    hc = hc.astype(float)
    anchor = t.normalize() + pd.Timedelta(hours=4)
    off = ((t - anchor) // pd.Timedelta(minutes=TF)).astype(np.int64)
    cur = anchor + pd.to_timedelta(off * TF, unit="min")
    j = np.searchsorted(hs.as_unit("ns").asi8, cur.as_unit("ns").asi8, side="left") - 1
    ok, jj = j >= 0, np.maximum(j, 0)
    g, l, r = rma_state(hc, 14)
    out = {"rsi_dev": np.where(ok, _rsi_from(g[jj], l[jj], hc[jj], p, 14), np.nan),
           "rsi_conf": np.where(ok, r[jj], np.nan)}
    for kind in ("dev", "conf"):
        es = []
        for n in (10, 25, 60):
            e = np.where(ok, ema_state(hc, n)[jj], np.nan)
            if kind == "dev":
                a = 2 / (n + 1)
                e = a * p + (1 - a) * e
            es.append(e)
        e10, e25, e60 = es
        valid = ~np.isnan(e60)
        out[f"up_{kind}"] = valid & (e10 > e25) & (e25 > e60)
        out[f"down_{kind}"] = valid & (e10 < e25) & (e25 < e60)
        out[f"mix_{kind}"] = valid & ~out[f"up_{kind}"] & ~out[f"down_{kind}"]
        out[f"spread_{kind}"] = np.where(valid, (np.maximum(np.maximum(e10, e25), e60) - np.minimum(np.minimum(e10, e25), e60)) / p, np.nan)
        out[f"valid_{kind}"] = valid
    return out


def structures(F, kind):
    """{이름: 봉마다 조건(bool)} — RSI와 AND로 묶을 구조."""
    up, down, mix, sp, v = F[f"up_{kind}"], F[f"down_{kind}"], F[f"mix_{kind}"], F[f"spread_{kind}"], F[f"valid_{kind}"]
    sp0 = np.nan_to_num(sp, nan=9.0)
    S = {"정배열 아님": v & ~up, "역배열": down, "엉킴": mix}
    for x in XS:
        S[f"간격<{x:.0%}"] = v & (sp0 < x)
        S[f"정배열 아님 또는 간격<{x:.0%}"] = v & (~up | (sp0 < x))
        S[f"(대조) 정배열·간격≥{x:.0%}"] = up & (sp0 >= x)
    S["(대조) 정배열"] = up
    S["구조 조건 없음"] = np.ones_like(up, dtype=bool)
    return S


if __name__ == "__main__":
    raw, outdir = sys.argv[1], sys.argv[2]
    U = user_universe()
    UNIV = {"사용자 78": U["합집합"], "참고 127": list(universe())}
    syms = sorted(set(UNIV["사용자 78"]) | set(UNIV["참고 127"]))
    rows, evs = [], []
    for pname, (mins, el, rl, lo, hi) in PERIODS.items():
        t0 = time.time()
        S = {}
        for s in syms:
            try:
                S[s] = base_arrays(raw, s, mins, el, rl)
            except FileNotFoundError:
                pass
        F = {s: feats(raw, s, A) for s, A in S.items()}
        days = sorted({d for A in S.values() for d in set(A["day"]) if lo <= d <= hi})
        for scope in ("24시간", "정규장만"):
            reg_only = scope == "정규장만"
            bg = lambda A, s: np.where(A["reg"] | (not reg_only), 0.02, NOBUY)
            base = run_cfg(S, bg, lo, hi)
            for uname, us in UNIV.items():
                rows.append(dict(기간=pname, 범위=scope, 종목=uname, 규칙="지금 지표", **metrics(base, base, us, days)))
            if scope == "24시간":                             # 사건 분석용: 지금 지표 매수마다 값 붙이기
                pos = {s: pd.Series(np.arange(len(A["d"])), index=A["d"].index) for s, A in S.items()}
                ii = [pos[s].loc[tt] for s, tt in zip(base.sym, base.t)]
                ev = base.copy()
                for k in F[next(iter(F))]:
                    ev[k] = [F[s][k][i] for s, i in zip(base.sym, ii)]
                ev["기간"] = pname
                evs.append(ev)
            for kind in ("dev", "conf"):
                for th in (None, 60, 65, 70, 75):
                    for sname in structures(F[next(iter(F))], kind):
                        if th is None and not (sname in ("정배열 아님", "역배열", "엉킴", "(대조) 정배열") or sname.startswith("간격<")):
                            continue
                        if th is not None and th != 70 and not (sname == "정배열 아님" or sname.startswith("정배열 아님 또는") or sname == "구조 조건 없음"):
                            continue
                        for aname, av in (("안 삼", NOBUY), ("-4%", 0.04)):
                            def fn(A, s, kind=kind, th=th, sname=sname, av=av):
                                st = structures(F[s], kind)[sname]
                                hot = st if th is None else st & (np.nan_to_num(F[s][f"rsi_{kind}"], nan=-1) >= th)
                                return np.where(bg(A, s) >= 1, NOBUY, np.where(hot, av, 0.02))
                            tr = run_cfg(S, fn, lo, hi)
                            lab = f"[{kind}] " + (f"RSI≥{th} & " if th is not None else "") + f"{sname} → {aname}"
                            for uname, us in UNIV.items():
                                rows.append(dict(기간=pname, 범위=scope, 종목=uname, 규칙=lab, 값=kind, 기준=th, 구조=sname, 조심=aname,
                                                 **metrics(tr, base, us, days)))
        print(f"{pname} 끝 {time.time() - t0:.0f}초", file=sys.stderr)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "htf_ema_rules.csv"), index=False, encoding="utf-8-sig")
    pd.concat(evs).to_parquet(os.path.join(outdir, "htf_ema_events.parquet"))
    print("저장", len(rows), "행")

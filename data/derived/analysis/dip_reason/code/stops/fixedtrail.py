"""칼손절 연구 — 고정·추적·본전 손절 기본군(fixedtrail, 2026-10-07). 다른 계열의 비교 기준.

규칙(모두 봉 i 시가에 판다, 값은 i-1 봉까지만):
  고정종가  c[i-1] <= ent*(1-p)
  고정저가  l[i-1] <= ent*(1-p)
  추적      c[i-1] <= max(ent, max c[ei..i-1]) * (1-p)
  본전      max(c 또는 h)[ei..i-1] >= ent*(1+X) 를 한 번 찍은 뒤 c[i-1] <= ent
  시간 창   k = i-ei(봉), nb = N분 // mins. '처음N'은 k <= nb 일 때만, 'N후'는 k > nb 일 때만 작동
  세션      'ent정규'는 매수 봉이 정규장인 거래만, 'ent프리애프터'는 그 밖, '깨어'는 봉 i가 KST 08~24시일 때만 작동

선택: early 사용자78 24시간 차이 최대(보조 조건: early 사용자78 정규장만 차이 >= 0). late는 전방 확인으로만 본다.
사용: py stops/fixedtrail.py   (결과: ../results/stops/fixedtrail.txt)
"""
import os, sys, time
from functools import partial
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stop_lab import compare, stop_audit, load, base_trades, run, UNIV  # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "stops", "fixedtrail.txt"))


def make_rule(A, kind, p=0.0, basis="c", arm="c", win=None, ses=None):
    c, l, h, mins = A["c"], A["l"], A["h"], A["mins"]
    nb = None if win is None else win[1] // mins
    if ses == "깨어":                                   # 봉 i 시각이 KST 08:00~23:59
        kh = A["idx"].tz_convert("Asia/Seoul").hour
        awake = np.asarray(kh >= 8)
    sesA = A["ses"]
    st = [-1, -1, 0.0, 0.0]                             # ei, 마지막 i, 최고 종가, 최고 고가 (i-1까지)

    def rule(i, ei, ent):
        if st[0] == ei and st[1] == i - 1:
            pk, hk = max(st[2], c[i - 1]), max(st[3], h[i - 1])
        else:
            pk, hk = c[ei:i].max(), h[ei:i].max()
        st[0], st[1], st[2], st[3] = ei, i, pk, hk
        k = i - ei
        if win is not None:
            if win[0] == "first" and k > nb:
                return False
            if win[0] == "after" and k <= nb:
                return False
        if ses == "ent정규" and sesA[ei] != "정규":
            return False
        if ses == "ent프리애프터" and sesA[ei] == "정규":
            return False
        if ses == "깨어" and not awake[i]:
            return False
        if kind == "fixed":
            return (c[i - 1] if basis == "c" else l[i - 1]) <= ent * (1 - p)
        if kind == "trail":
            return c[i - 1] <= max(ent, pk) * (1 - p)
        if kind == "be":
            top = pk if arm == "c" else hk
            return top >= ent * (1 + p) and c[i - 1] <= ent
        raise ValueError(kind)
    return rule


def name_of(sp):
    kind, p = sp["kind"], sp.get("p", 0)
    if kind == "fixed":
        s = f"고정{'종가' if sp.get('basis', 'c') == 'c' else '저가'}-{p:.0%}"
    elif kind == "trail":
        s = f"추적-{p:.0%}"
    else:
        s = f"본전+{p:.1%}({'종가' if sp.get('arm', 'c') == 'c' else '고가'}무장)"
    if sp.get("win"):
        w, n = sp["win"]
        s += f"|{'처음' if w == 'first' else ''}{n}분{'후' if w == 'after' else ''}"
    if sp.get("ses"):
        s += f"|{sp['ses']}"
    return s


def grid():
    G = []
    G += [dict(kind="fixed", p=p, basis="c") for p in (.02, .03, .04, .05, .06, .07, .08, .10)]          # 8
    G += [dict(kind="fixed", p=p, basis="l") for p in (.02, .03, .04, .05, .06, .08, .10)]               # 7
    G += [dict(kind="trail", p=p) for p in (.02, .03, .04, .05, .06)]                                     # 5
    G += [dict(kind="be", p=x, arm="c") for x in (.005, .01, .015, .02)]                                  # 4
    G += [dict(kind="be", p=x, arm="h") for x in (.01, .02)]                                              # 2
    G += [dict(kind="fixed", p=p, basis="c", win=("first", n)) for p in (.03, .05) for n in (15, 30, 60, 120)]   # 8
    G += [dict(kind="fixed", p=p, basis="l", win=("first", n)) for p in (.03, .05) for n in (15, 30, 60)]        # 6
    G += [dict(kind="fixed", p=p, basis="c", win=("after", n)) for p in (.02, .03, .05) for n in (15, 30, 60, 120)]  # 12
    G += [dict(kind="trail", p=p, win=(w, n)) for p in (.03, .05) for w in ("first", "after") for n in (30, 60)]  # 8
    G += [dict(kind="be", p=.01, arm="c", win=("first", n)) for n in (30, 60)]                            # 2
    return G                                                                                              # 62


def cells(df):
    """사용자78·전체137 × 기간 × 범위 → 차이·t·손절·최악·10월6일."""
    out = {}
    for r in df.itertuples():
        key = f"{r.기간}_{'24h' if r.범위 == '24시간' else 'reg'}_{'U78' if r.종목 == '사용자78' else 'A137'}"
        out[key] = dict(diff=r.차이, t=r.t, stops=r.손절, worst=r.최악, oct6=r.차이_10월6일, sum=r.합)
    return out


def job(sp):
    df = compare(partial(make_rule, **sp), name_of(sp), show=False)
    return sp, df


def summary_row(nm, cl):
    g = lambda k, f="diff": cl[k][f]
    return dict(규칙=nm,
                e24=g("early_24h_U78"), e24t=g("early_24h_U78", "t"), e24손절=g("early_24h_U78", "stops"),
                eReg=g("early_reg_U78"), eRegt=g("early_reg_U78", "t"),
                l24=g("late_24h_U78"), l24t=g("late_24h_U78", "t"), lReg=g("late_reg_U78"),
                e24_137=g("early_24h_A137"), l24_137=g("late_24h_A137"),
                eReg_137=g("early_reg_A137"), lReg_137=g("late_reg_A137"),
                oct6_l24=g("late_24h_U78", "oct6"))


def lookahead_check(sp, period="late", n=300, seed=0):
    """봉 i 이후를 잘라낸 배열(A[k][:i])로 만든 규칙과 전체 배열 규칙이 같은 판단을 하는지 표본으로 확인.
    i 이후 값을 읽으면 IndexError 또는 불일치가 난다."""
    S = load(period)
    b = base_trades(period)
    rng = np.random.default_rng(seed)
    rows = b.sample(min(n, len(b)), random_state=seed)
    bad = 0
    arr_keys = ("o", "h", "l", "c", "v", "r", "ema", "tk", "dead", "reg", "mod", "hot7", "ses", "day")
    for x in rows.itertuples():
        A = S[x.sym]
        if x.xi <= x.ei + 1:
            continue
        i = int(rng.integers(x.ei + 1, x.xi + 1))
        full = make_rule(A, **sp)
        for j in range(x.ei + 1, i):                    # 시험대처럼 봉을 차례로 넘겨 누적 상태 경로도 같이 확인
            full(j, x.ei, x.ent)
        T = dict(A)
        for k in arr_keys:
            T[k] = A[k][:i]
        T["idx"] = A["idx"][:i + 1] if sp.get("ses") == "깨어" else A["idx"][:i]   # '깨어'는 봉 i의 시각(미리 아는 값)만 쓴다
        try:
            same = make_rule(T, **sp)(i, x.ei, x.ent) == full(i, x.ei, x.ent)
        except IndexError:
            same = False
        bad += (not same)
    return bad, len(rows)


if __name__ == "__main__":
    t0 = time.time()
    G = grid()
    with Pool(6) as pool:
        res = pool.map(job, G)
    tables, summ = [], []
    for sp, df in res:
        nm = name_of(sp)
        tables.append(df)
        summ.append(summary_row(nm, cells(df)))
    sm = pd.DataFrame(summ)
    spec_of = {name_of(sp): sp for sp in G}

    # 2단계: early로 고른 1단계 최선안에 세션 제한(사전에 정한 3변형)을 얹어 본다(역시 early로만 판단)
    elig = sm[sm.eReg >= 0].sort_values("e24", ascending=False)
    top1 = elig.iloc[0].규칙 if len(elig) else sm.sort_values("e24", ascending=False).iloc[0].규칙
    G2 = [dict(spec_of[top1], ses=s) for s in ("ent정규", "ent프리애프터", "깨어")]
    with Pool(3) as pool:
        res2 = pool.map(job, G2)
    for sp, df in res2:
        nm = name_of(sp)
        tables.append(df)
        summ.append(summary_row(nm, cells(df)))
        spec_of[nm] = sp
    sm = pd.DataFrame(summ)
    n_rules = len(sm)
    allpos = sm[(sm.e24 > 0) & (sm.eReg > 0) & (sm.l24 > 0) & (sm.lReg > 0)]

    elig = sm[sm.eReg >= 0].sort_values("e24", ascending=False)
    best = elig.iloc[0].규칙 if len(elig) and elig.iloc[0].e24 > 0 else None
    pick = best or sm.sort_values("e24", ascending=False).iloc[0].규칙
    bsp = spec_of[pick]
    mk = partial(make_rule, **bsp)
    aud = {}
    for p in ("early", "late"):
        for reg in (False, True):
            aud[(p, reg)] = stop_audit(mk, p, reg_only=reg)
    la_bad, la_n = lookahead_check(bsp, "late")
    la_bad2, la_n2 = lookahead_check(bsp, "early")

    # 차순위(early 순위, 보조 조건 통과, 선택 규칙의 세션 변형 제외) 5개와 고정종가-5% 참고의 손절 감사
    base_nm = pick.split("|")[0] if bsp.get("ses") else None
    others = [r for r in elig.규칙 if r != pick and not (base_nm and r.startswith(base_nm + "|") and spec_of[r].get("ses"))][:5]
    others += [r for r in ["고정종가-5%"] if r not in others]
    aud2 = {}
    for nm in others:
        m2 = partial(make_rule, **spec_of[nm])
        for p in ("early", "late"):
            a = stop_audit(m2, p)
            t = run(load(p), m2, p, False, syms=UNIV["사용자78"])
            s = t[t.why == "손절"]
            a["체결가>매수가"] = int((s.px > s.ent).sum())             # 프리·애프터 튀는 틱으로 손절됐다가 정상가에 팔린 흔적
            aud2[(nm, p)] = a
    full = pd.concat(tables, ignore_index=True)
    fmt = lambda v: f"{v:+.2f}"
    with open(OUT, "w", encoding="utf-8") as f:
        w = lambda s="": f.write(s + "\n")
        w("칼손절 연구 — 고정·추적·본전 손절 기본군(fixedtrail) 2026-10-07")
        w("기준: 지금 지표(전날 +7% 규칙 포함). 사용자78 24시간 early +27.12 · late +17.04, 정규장만 early +1.47 · late +2.28")
        w(f"시험한 규칙 수: {n_rules} (1단계 격자 {len(G)} + 2단계 세션 변형 {len(G2)})")
        w(f"사용자78 네 칸(early/late × 24시간/정규장만) 모두 플러스: {len(allpos)}개 -> {', '.join(allpos.규칙) if len(allpos) else '없음'}")
        w("선택 규칙: early 사용자78 24시간 차이 최대, 보조 조건 early 사용자78 정규장만 차이 >= 0. late는 고르지 않고 확인만.")
        w(f"선택: {pick}" + ("" if best else "  (early 24시간 차이가 플러스인 규칙이 없어 '최선'은 이름뿐)"))
        w()
        w("== 요약(차이 = 규칙 합 - 기준 합, 수익률 합 단위). e=early(선택용) l=late(확인용) 24=24시간 Reg=정규장만")
        show = sm.sort_values("e24", ascending=False).copy()
        for k in ("e24", "e24t", "eReg", "eRegt", "l24", "l24t", "lReg", "e24_137", "l24_137", "eReg_137", "lReg_137", "oct6_l24"):
            show[k] = show[k].map(fmt)
        w(show.to_string(index=False))
        w()
        w(f"== 선택 규칙 '{pick}' 전체 표")
        bt = full[full.규칙 == pick].copy()
        w(bt.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        w()
        w("== stop_audit (사용자78)")
        for (p, reg), a in aud.items():
            w(f"  {p} {'정규장만' if reg else '24시간'}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in a.items()))
        w()
        w("== 차순위·참고 규칙 stop_audit (사용자78 24시간, 체결가>매수가 = 손절인데 매수가보다 비싸게 팔린 건수)")
        for (nm, p), a in aud2.items():
            w(f"  {nm} {p}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in a.items()))
        w()
        w(f"== 미래 참조 확인: A[k][:i]로 자른 배열 규칙 vs 전체 배열 규칙, 불일치/IndexError late {la_bad}/{la_n}, early {la_bad2}/{la_n2}")
        w()
        w("== 모든 규칙 전체 표")
        w(full.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(open(OUT, encoding="utf-8").read().split("== 모든 규칙 전체 표")[0])
    print(f"{time.time() - t0:.0f}s")

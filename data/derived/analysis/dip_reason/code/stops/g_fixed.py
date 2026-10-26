"""칼손절 연구 — 고정·추적 손절 + 40분봉 RSI 조건(g_fixed, 2026-10-07).

사용자 가설(10/7): "40분봉 RSI가 30 이하(이미 과매도)면 손절하면 안 좋다." -> 손절 신호가 나도 40분봉 RSI가 낮으면 보류,
40분봉 RSI가 아직 높은데 빠질 때만 자른다.

손절 신호(봉 i 시가에 판다, 값은 i-1 봉까지만):
  고정종가-p   c[i-1] <= ent*(1-p)                                   p = 2·3·4·5·7%
  추적-p       c[i-1] <= max(ent, max c[ei..i-1]) * (1-p)            p = 3·5%
40분봉 RSI 조건(stop_lab.htf_rsi(A, 40, 14), 인덱스 i 값 그대로 = 봉 i 시가 판단 때 아는 값):
  dev[i]  진행 중 40분봉 RSI(그 전 완성 40분봉 RMA 상태 + c[i-1]),  conf[i] 봉 i 시작까지 완성된 마지막 40분봉 RSI
  켬 = src[i] > g (반대 조건은 src[i] <= g), NaN이면 꺼짐. g = 25·30·35·40·45
방식:
  재판단  봉마다 '신호 and 켬'이면 판다(조건이 꺼져 있으면 다음 봉에 신호·조건 둘 다 다시 본다)
  첫신호  그 거래에서 신호가 처음 난 봉에 켬이면 판다. 꺼져 있으면 그 거래는 끝까지 손절 안 함
  래치    신호가 한 번 나면 무장, 그 뒤 처음 켬인 봉에 판다(그때 가격이 손절선 위로 돌아왔어도 판다) — 2단계 변형
단계(모두 early로만 고름, late는 확인만):
  1단계  조건 없는 같은 손절 7 + dev 조건 7손절 × g5 × (재판단·첫신호) 70
  2단계  1단계 early 최선 손절·방식에 conf 조건 g5 × (재판단·첫신호) 10 + dev 래치 g5 5
  대조   최종 선택의 (src·g·방식)으로 7손절 모두 '반대 조건'(src <= g일 때만 자름) 7
선택: 조건 붙은 규칙 중 early 사용자78 24시간 차이 최대(보조 조건 early 사용자78 정규장만 차이 >= 0).
사용: py stops/g_fixed.py   (결과: ../results/stops/g_fixed.txt)
"""
import os, sys, time
from datetime import date
from functools import partial
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stop_lab import compare, stop_audit, load, base_trades, run, htf_rsi, UNIV, RAW  # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "stops", "g_fixed.txt"))
STOPS = [("fixed", .02), ("fixed", .03), ("fixed", .04), ("fixed", .05), ("fixed", .07), ("trail", .03), ("trail", .05)]
GS = (25, 30, 35, 40, 45)


def make_rule(A, kind, p, src=None, g=None, mode="재판단", rev=False, _S=None):
    c = A["c"]
    if src is not None:
        S = _S if _S is not None else htf_rsi(A, 40, 14)[0 if src == "dev" else 1]
    st = dict(ei=-1, last=-1, pk=-np.inf, first=None, armed=None)

    def gate(i):
        if src is None:
            return True
        v = S[i]
        if v != v:                                       # NaN
            return False
        return v <= g if rev else v > g

    def step(i, ent):
        st["pk"] = max(st["pk"], c[i - 1])               # 최고 종가 c[ei..i-1]
        if kind == "fixed":
            s = c[i - 1] <= ent * (1 - p)
        else:
            s = c[i - 1] <= max(ent, st["pk"]) * (1 - p)
        if mode == "재판단":
            return bool(s and gate(i))
        if mode == "첫신호":
            if st["first"] is None:
                if s:
                    st["first"] = i
                    return gate(i)
                return False
            return False                                 # 첫 신호 때 꺼져 있었으면 이 거래는 손절 안 함
        if mode == "래치":
            if st["armed"] is None and s:
                st["armed"] = i
            return st["armed"] is not None and gate(i)
        raise ValueError(mode)

    def rule(i, ei, ent):
        if not (st["ei"] == ei and st["last"] == i - 1):  # 새 거래 또는 순서가 끊긴 호출 -> 처음부터 다시 쌓는다
            st.update(ei=ei, last=ei, pk=-np.inf, first=None, armed=None)
            for j in range(ei + 1, i):
                step(j, ent)
        st["last"] = i
        return step(i, ent)
    return rule


def name_of(sp):
    kind, p = sp["kind"], sp["p"]
    s = f"{'고정종가' if kind == 'fixed' else '추적'}-{p:.0%}"
    if sp.get("src"):
        s += f"|{sp['src']}{'<=' if sp.get('rev') else '>'}{sp['g']}|{sp.get('mode', '재판단')}"
    return s


def cells(df):
    out = {}
    for r in df.itertuples():
        key = f"{r.기간}_{'24h' if r.범위 == '24시간' else 'reg'}_{'U78' if r.종목 == '사용자78' else 'A137'}"
        out[key] = dict(diff=r.차이, t=r.t, stops=r.손절, worst=r.최악, oct6=r.차이_10월6일, sum=r.합)
    return out


CDIR = os.path.join(os.path.dirname(RAW), "g_fixed_cache")       # 규칙별 compare 표 캐시(메모리 부족으로 끊겨도 이어 돌림)


def job(sp):
    os.makedirs(CDIR, exist_ok=True)
    fn = os.path.join(CDIR, name_of(sp).replace("|", "_").replace("<=", "le").replace(">", "gt").replace("%", "") + ".csv")
    if os.path.exists(fn):
        return sp, pd.read_csv(fn, encoding="utf-8")
    for k in range(3):
        try:
            df = compare(partial(make_rule, **sp), name_of(sp), show=False)
            break
        except MemoryError:
            time.sleep(20)
    else:
        df = compare(partial(make_rule, **sp), name_of(sp), show=False)
    df.to_csv(fn, index=False, encoding="utf-8")
    return sp, df


def summary_row(nm, cl, stage):
    g = lambda k, f="diff": cl[k][f]
    return dict(규칙=nm, 단계=stage,
                e24=g("early_24h_U78"), e24t=g("early_24h_U78", "t"), e24손절=g("early_24h_U78", "stops"),
                eReg=g("early_reg_U78"), eRegt=g("early_reg_U78", "t"),
                l24=g("late_24h_U78"), l24t=g("late_24h_U78", "t"), l24손절=g("late_24h_U78", "stops"), lReg=g("late_reg_U78"),
                e24_137=g("early_24h_A137"), l24_137=g("late_24h_A137"),
                eReg_137=g("early_reg_A137"), lReg_137=g("late_reg_A137"),
                oct6_l24=g("late_24h_U78", "oct6"))


def run_jobs(G, n=3):
    with Pool(n) as pool:
        return pool.map(job, G)


def lookahead_check(sp, period="late", n=300, seed=0):
    """(1) 규칙: 가격 배열을 A[k][:i]로, RSI 배열을 [:i+1]로 자른 규칙과 전체 배열 규칙이 같은 판단인지(누적 상태 경로 포함).
    (2) 40분봉 RSI: 5분봉을 '봉 i-1이 속한 40분봉 시작' 이전으로 잘라 다시 만든 dev[i], '봉 i가 속한 40분봉 시작' 이전으로 자른 conf[i]가
        htf_rsi 값과 같은지. 둘 다 i 이후 값을 읽으면 불일치/IndexError가 난다."""
    from htf import intraday_bars, rma_state, _rsi_from
    S = load(period)
    b = base_trades(period)
    rng = np.random.default_rng(seed)
    rows = b.sample(min(n, len(b)), random_state=seed)
    bad, bad_rsi, m = 0, 0, 0
    keys = ("o", "h", "l", "c", "v", "r", "ema", "tk", "dead", "reg", "mod", "hot7", "ses", "day")
    m5c = {}
    for x in rows.itertuples():
        A = S[x.sym]
        if x.xi <= x.ei + 1:
            continue
        m += 1
        i = int(rng.integers(x.ei + 1, x.xi + 1))
        full = make_rule(A, **sp)
        for j in range(x.ei + 1, i):
            full(j, x.ei, x.ent)
        T = dict(A)
        for k in keys:
            T[k] = A[k][:i]
        T["idx"] = A["idx"][:i]
        Sfull = htf_rsi(A, 40, 14)[0 if sp.get("src", "dev") == "dev" else 1]
        try:
            same = make_rule(T, **sp, _S=Sfull[:i + 1])(i, x.ei, x.ent) == full(i, x.ei, x.ent)
        except IndexError:
            same = False
        bad += (not same)
        # (2) 40분봉 RSI 독립 재계산
        if x.sym not in m5c:
            m5c[x.sym] = pd.read_parquet(os.path.join(RAW, f"5m_{x.sym}.parquet"))
        m5 = m5c[x.sym]
        dev, conf = htf_rsi(A, 40, 14)
        idx = A["idx"]
        step = pd.Timedelta(minutes=40)
        anc_prev = idx[i - 1].normalize() + pd.Timedelta(hours=4)
        cur_prev = anc_prev + ((idx[i - 1] - anc_prev) // step) * step
        anc = idx[i].normalize() + pd.Timedelta(hours=4)
        cur_i = anc + ((idx[i] - anc) // step) * step
        hs, hc = intraday_bars(m5[m5.index < cur_prev], 40)
        g_, l_, r_ = rma_state(hc.astype(float), 14)
        d2 = _rsi_from(g_[-1:], l_[-1:], hc[-1:].astype(float), np.array([A["c"][i - 1]]), 14)[0] if len(hc) else np.nan
        hs2, hc2 = intraday_bars(m5[m5.index < cur_i], 40)
        c2 = rma_state(hc2.astype(float), 14)[2][-1] if len(hc2) else np.nan
        ok_d = (np.isnan(d2) and np.isnan(dev[i])) or abs(d2 - dev[i]) < 1e-6
        ok_c = (np.isnan(c2) and np.isnan(conf[i])) or abs(c2 - conf[i]) < 1e-6
        bad_rsi += (not (ok_d and ok_c))
    return bad, bad_rsi, m


def day_split(mk, period, syms, reg_only=False):
    """날짜별 (규칙 합 - 기준 합). 가장 큰 기여일과 그날을 뺀 차이."""
    t = run(load(period), mk, period, reg_only, syms=syms)
    b = base_trades(period, reg_only)
    b = b[b.sym.isin(syms)]
    days = sorted(set(b.day) | set(t.day))
    d = t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0)
    top = d.idxmax()
    return d, top, d[top], d.sum() - d[top]


def pair_decomp(mk_g, mk_u, period, syms):
    """같은 매수(sym, ei)로 짝지은 거래에서 조건 붙은 규칙 - 조건 없는 규칙. 무조건만 손절 / 둘 다 손절 / 조건부만 손절로 나눈다."""
    S = load(period)
    tg = run(S, mk_g, period, False, syms=syms).set_index(["sym", "ei"])
    tu = run(S, mk_u, period, False, syms=syms).set_index(["sym", "ei"])
    m = tg[["ret", "why"]].join(tu[["ret", "why"]], how="inner", lsuffix="_g", rsuffix="_u")
    out = {}
    for nm, msk in (("무조건만손절(조건이 막음)", (m.why_u == "손절") & (m.why_g != "손절")),
                    ("둘다손절", (m.why_u == "손절") & (m.why_g == "손절")),
                    ("조건부만손절", (m.why_u != "손절") & (m.why_g == "손절"))):
        x = m[msk]
        out[nm] = dict(n=len(x), 득실합=(x.ret_g - x.ret_u).sum(), 헛손절=(x.ret_g > 0).mean() if nm.startswith("무조건만") and len(x) else np.nan)
    out["짝없는거래(재매수 차이)"] = dict(조건부만있음=len(tg.index.difference(tu.index)), 무조건만있음=len(tu.index.difference(tg.index)),
                                  조건부합=tg.loc[tg.index.difference(tu.index)].ret.sum(), 무조건합=tu.loc[tu.index.difference(tg.index)].ret.sum())
    return out


def stop_detail(mk, period, syms):
    """선택 규칙의 손절 거래 목록: 매수·매도 세션, 손절 수익, 기준(손절 없음) 수익, 매도 봉 conf·dev, 매수 뒤 분.
    체결가>매수가 = 프리·애프터 얇은 체결로 종가가 -X%를 찍었다가 정상 가격에 팔린 흔적."""
    S = load(period)
    t = run(S, mk, period, False, syms=syms)
    b = base_trades(period)
    b = b[b.sym.isin(syms)].set_index(["sym", "ei"]).ret
    s = t[t.why == "손절"].copy()
    s["기준"] = [b.get((r.sym, r.ei), np.nan) for r in s.itertuples()]
    s["매도세션"] = [S[r.sym]["ses"][r.xi] for r in s.itertuples()]
    s["conf"] = [htf_rsi(S[r.sym], 40, 14)[1][r.xi] for r in s.itertuples()]
    s["dev"] = [htf_rsi(S[r.sym], 40, 14)[0][r.xi] for r in s.itertuples()]
    s["분"] = [(r.xi - r.ei) * S[r.sym]["mins"] for r in s.itertuples()]
    return s


def fmt_audit(a):
    return ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in a.items())


if __name__ == "__main__":
    t0 = time.time()
    # 1단계
    G0 = [dict(kind=k, p=p) for k, p in STOPS]
    G1 = [dict(kind=k, p=p, src="dev", g=g, mode=md) for k, p in STOPS for g in GS for md in ("재판단", "첫신호")]
    res = run_jobs(G0 + G1)
    tables, summ, spec_of = [], [], {}
    for sp, df in res:
        nm = name_of(sp)
        tables.append(df)
        summ.append(summary_row(nm, cells(df), "1-무조건" if "src" not in sp else "1-dev"))
        spec_of[nm] = sp
    sm = pd.DataFrame(summ)
    print(f"1단계 {len(res)}개 {time.time() - t0:.0f}s", flush=True)

    def pick_best(df):
        el = df[df.eReg >= 0].sort_values("e24", ascending=False)
        return (el if len(el) else df.sort_values("e24", ascending=False)).iloc[0].규칙

    b1 = spec_of[pick_best(sm[sm.단계 == "1-dev"])]
    # 2단계: 1단계 최선의 손절·방식에 conf 조건, 같은 손절에 래치 방식
    G2 = [dict(kind=b1["kind"], p=b1["p"], src="conf", g=g, mode=md) for g in GS for md in ("재판단", "첫신호")]
    G2 += [dict(kind=b1["kind"], p=b1["p"], src="dev", g=g, mode="래치") for g in GS]
    for sp, df in run_jobs(G2):
        nm = name_of(sp)
        tables.append(df)
        summ.append(summary_row(nm, cells(df), "2-" + sp["src"] + ("-래치" if sp["mode"] == "래치" else "")))
        spec_of[nm] = sp
    sm = pd.DataFrame(summ)
    print(f"2단계 {len(G2)}개 {time.time() - t0:.0f}s", flush=True)

    gated = sm[sm.단계 != "1-무조건"]
    pick = pick_best(gated)
    bsp = spec_of[pick]
    # 대조: 같은 (src·g·방식)으로 7손절 모두 반대 조건
    G3 = [dict(kind=k, p=p, src=bsp["src"], g=bsp["g"], mode=bsp["mode"], rev=True) for k, p in STOPS]
    for sp, df in run_jobs(G3):
        nm = name_of(sp)
        tables.append(df)
        summ.append(summary_row(nm, cells(df), "대조-반대"))
        spec_of[nm] = sp
    sm = pd.DataFrame(summ)
    print(f"대조 {len(G3)}개 {time.time() - t0:.0f}s", flush=True)

    n_rules = len(sm)
    allpos = sm[(sm.e24 > 0) & (sm.eReg > 0) & (sm.l24 > 0) & (sm.lReg > 0)]
    full = pd.concat(tables, ignore_index=True)

    ung_nm = name_of(dict(kind=bsp["kind"], p=bsp["p"]))
    rev_nm = name_of(dict(bsp, rev=True))
    mk = partial(make_rule, **bsp)
    mk_u = partial(make_rule, **spec_of[ung_nm])
    mk_r = partial(make_rule, **spec_of[rev_nm])
    U = UNIV["사용자78"]
    aud = {}
    for lab, m_ in (("선택", mk), ("무조건", mk_u), ("반대", mk_r)):
        for p in ("early", "late"):
            for reg in (False, True):
                aud[(lab, p, reg)] = stop_audit(m_, p, reg_only=reg)
    dec = {p: pair_decomp(mk, mk_u, p, U) for p in ("early", "late")}
    dsp = {p: day_split(mk, p, U) for p in ("early", "late")}
    det = {p: stop_detail(mk, p, U) for p in ("early", "late")}
    la = lookahead_check(bsp, "late")
    la2 = lookahead_check(bsp, "early")
    # 차순위: 같은 손절·방식의 g 이웃이 아닌, 서로 다른 손절 중 early 상위
    el = gated[gated.eReg >= 0].sort_values("e24", ascending=False)
    runners = [r for r in el.규칙 if r != pick][:5]
    # 첫 거래일 RSI NaN(5분봉이 7/14 04:00부터라 40분봉 14개가 쌓이기 전) 비율
    nan_note = {}
    for p in ("early", "late"):
        S = load(p)
        b = base_trades(p)
        b = b[b.sym.isin(U)]
        nn = sum(np.isnan(htf_rsi(S[x.sym], 40, 14)[0][x.ei + 1]) for x in b.itertuples())
        nan_note[p] = (nn, len(b))

    fmt = lambda v: f"{v:+.2f}"
    with open(OUT, "w", encoding="utf-8") as f:
        w = lambda s="": f.write(s + "\n")
        w("칼손절 연구 — 고정·추적 손절 + 40분봉 RSI 조건(g_fixed) 2026-10-07")
        w("가설: 40분봉 RSI 30 이하(이미 과매도)면 손절 보류, 40분봉 RSI가 아직 높은데 빠질 때만 자른다.")
        w("기준: 지금 지표(전날 +7% 규칙 포함). 사용자78 24시간 early +27.12 · late +17.04, 정규장만 early +1.47 · late +2.28")
        w(f"시험한 규칙 수: {n_rules} (1단계 무조건 {len(G0)} + dev 조건 {len(G1)}, 2단계 conf·래치 {len(G2)}, 대조 반대 조건 {len(G3)})")
        w(f"사용자78 네 칸 모두 플러스: {len(allpos)}개 -> {', '.join(allpos.규칙) if len(allpos) else '없음'}")
        w("선택 규칙: 조건 붙은 규칙 중 early 사용자78 24시간 차이 최대, 보조 조건 early 사용자78 정규장만 차이 >= 0. late는 확인만.")
        w(f"1단계 최선(2단계 손절·방식 결정): {name_of(b1)}")
        w(f"선택: {pick}   | 같은 손절 조건 없음: {ung_nm}   | 반대 조건: {rev_nm}")
        w(f"40분봉 RSI NaN(첫 거래일, 조건 꺼짐 처리) 매수 다음 봉 기준: early {nan_note['early'][0]}/{nan_note['early'][1]}, late {nan_note['late'][0]}/{nan_note['late'][1]}")
        w()
        w("== 요약(차이 = 규칙 합 - 기준 합). e=early(선택용) l=late(확인용) 24=24시간 Reg=정규장만, _137=전체137")
        show = sm.sort_values("e24", ascending=False).copy()
        for k in ("e24", "e24t", "eReg", "eRegt", "l24", "l24t", "lReg", "e24_137", "l24_137", "eReg_137", "lReg_137", "oct6_l24"):
            show[k] = show[k].map(fmt)
        w(show.to_string(index=False))
        w()
        w("== 손절별 조건 효과 표(사용자78 e24 / eReg / l24 / lReg): 무조건 vs dev 재판단·첫신호 g별")
        for k, p in STOPS:
            base_nm = name_of(dict(kind=k, p=p))
            r0 = sm[sm.규칙 == base_nm].iloc[0]
            w(f"  {base_nm:10s} 무조건          {r0.e24:+.2f} {r0.eReg:+.2f} {r0.l24:+.2f} {r0.lReg:+.2f}  (손절 e{int(r0.e24손절)}/l{int(r0.l24손절)})")
            for md in ("재판단", "첫신호"):
                for g in GS:
                    nm = name_of(dict(kind=k, p=p, src="dev", g=g, mode=md))
                    r = sm[sm.규칙 == nm].iloc[0]
                    w(f"  {'':10s} dev>{g} {md:4s}   {r.e24:+.2f} {r.eReg:+.2f} {r.l24:+.2f} {r.lReg:+.2f}  (손절 e{int(r.e24손절)}/l{int(r.l24손절)})")
            rn = name_of(dict(kind=k, p=p, src=bsp["src"], g=bsp["g"], mode=bsp["mode"], rev=True))
            r = sm[sm.규칙 == rn].iloc[0]
            w(f"  {'':10s} 반대 {bsp['src']}<={bsp['g']} {bsp['mode']} {r.e24:+.2f} {r.eReg:+.2f} {r.l24:+.2f} {r.lReg:+.2f}  (손절 e{int(r.e24손절)}/l{int(r.l24손절)})")
        w()
        w(f"== 선택 규칙 '{pick}' 전체 표")
        w(full[full.규칙 == pick].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        w()
        w(f"== 같은 손절 조건 없음 '{ung_nm}' 전체 표")
        w(full[full.규칙 == ung_nm].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        w()
        w(f"== 반대 조건 '{rev_nm}' 전체 표")
        w(full[full.규칙 == rev_nm].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        w()
        w("== stop_audit (사용자78)")
        for (lab, p, reg), a in aud.items():
            w(f"  {lab} {p} {'정규장만' if reg else '24시간'}: {fmt_audit(a)}")
        w()
        w("== 같은 매수 짝 비교: 선택(조건부) - 무조건, 사용자78 24시간 (득실합 + = 조건부가 나음)")
        for p, dd in dec.items():
            for k2, v in dd.items():
                w(f"  {p} {k2}: " + ", ".join(f"{kk}={vv:+.3f}" if isinstance(vv, float) else f"{kk}={vv}" for kk, vv in v.items()))
        w()
        w("== 날짜 집중(선택 규칙, 사용자78 24시간): 가장 큰 기여일과 그날 뺀 차이")
        for p, (d, top, tv, rest) in dsp.items():
            w(f"  {p}: 차이 {d.sum():+.3f}, 최대 기여일 {top} {tv:+.3f}, 그날 빼면 {rest:+.3f}, 나은날 {(d > 1e-12).sum()} 나쁜날 {(d < -1e-12).sum()}")
        w()
        w("== 선택 규칙 손절 거래 상세(사용자78 24시간)")
        for p, s_ in det.items():
            w(f"  {p}: 손절 {len(s_)}건, 매수 세션 {s_.ses.value_counts().to_dict()}, 매도 세션 {s_.매도세션.value_counts().to_dict()}, "
              f"체결가>매수가 {int((s_.px > s_.ent).sum())}건, 손절-기준 합 {(s_.ret - s_.기준).sum():+.3f} "
              f"(매도 세션별 {(s_.ret - s_.기준).groupby(s_.매도세션).sum().round(3).to_dict()}), 매수 뒤 분 중앙값 {s_.분.median():.0f}, "
              f"매도 봉 conf 중앙값 {s_.conf.median():.1f} dev 중앙값 {s_.dev.median():.1f}")
        w(det["late"][["sym", "day", "ses", "매도세션", "ret", "기준", "conf", "dev", "분"]].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
        w()
        w("== 차순위(조건 붙은 규칙, early 순위, 보조 조건 통과)")
        for r in runners:
            x = sm[sm.규칙 == r].iloc[0]
            w(f"  {r}: e24 {x.e24:+.2f} (t {x.e24t:+.2f}) eReg {x.eReg:+.2f} l24 {x.l24:+.2f} (t {x.l24t:+.2f}) lReg {x.lReg:+.2f} | 137 e24 {x.e24_137:+.2f} l24 {x.l24_137:+.2f}")
        w()
        w(f"== 미래 참조 확인(표본 거래·무작위 봉): 규칙 불일치/IndexError late {la[0]}/{la[2]}, early {la2[0]}/{la2[2]}; "
          f"40분봉 RSI 독립 재계산(5분봉을 판단 시점 앞에서 자름) 불일치 late {la[1]}/{la[2]}, early {la2[1]}/{la2[2]}")
        w()
        w("== 모든 규칙 전체 표")
        w(full.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(open(OUT, encoding="utf-8").read().split("== 모든 규칙 전체 표")[0])
    print(f"{time.time() - t0:.0f}s")

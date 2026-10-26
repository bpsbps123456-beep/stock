"""흘러내림 시간손절 + 40분봉 RSI 조건(2026-10-07). 사용자 가설: 40분봉 RSI가 30 이하(이미 과매도)면 손절하면 안 좋다.
=> 흘러내림 손절 신호가 나와도 40분봉 RSI(dev 또는 conf, 인덱스 i 값)가 g 이하면 자르지 않는다.

손절 바탕(stops/grind.py와 같은 정의, 봉 i 시가에 팔지 정하고 봉 i-1까지만 씀; 그 봉이 정리 시각(dead[ei]) 이후면 안 봄):
  F1 T Y     매수 봉 ei + k봉(k = ceil(T / 봉 길이분), early 5분 / late 3분) 봉 i 한 번: 종가 c[ei..i-1] 최고 < 매수가 x (1+Y)
  F5 M max   프리·애프터 매수(ses[ei] != 정규)만, ei 뒤 첫 정규장 봉 중 ET 분 >= 570+M 인 봉 i 한 번: c[ei..i-1] 최고 < 매수가 x 1.005
  F5 M close 위와 같은 봉 i 한 번: c[i-1] < 매수가
조건(gate, 봉 i의 값 = 미래 참조 없음, htf_rsi(A, 40, 14)):
  dev>g   dev[i] > g 일 때만 자른다(dev[i] = 직전 완성 40분봉까지 RMA 상태에 c[i-1]을 넣은 진행 중 RSI, 트레이딩뷰 실시간 값)
  conf>g  conf[i] > g (봉 i 시작까지 완성된 마지막 40분봉 RSI)
  rev dev<=g  반대 조건(40분봉 RSI가 낮을 때만 자름) — 조건 효과 분리용 비교
  band lo<dev<=hi  그 구간에서만 자름(흘러내림 30~45 구간 확인용)
  NaN이면 어느 조건도 참이 아니다(자르지 않음).
보류형(defer): 검사 시각에 조건이 막으면 버리지 않고, 검사 시각 이후 매 봉 '아직 반등 없음(c[ei..i-1] 최고 < 매수가 x (1+Y))이고 조건 참'인
  첫 봉 i에 판다(가설의 '보류'를 문자 그대로). 반등 없음은 한 번 깨지면 다시 참이 안 되므로 상태 없이 같은 뜻.
  조건 없는 보류형 = 조건 없는 1회형(같은 손절).

고르기(early만): 조건 붙은 규칙(단 rev 제외) 중 사용자78 정규장만 차이 >= 0 안에서 사용자78 24시간 차이 최대. late는 확인용 계산.
1단계: 바탕 11개 x {조건 없음, dev>30/35/40/45, 보류형 dev>30/40(F5 close 제외)}
2단계(1단계 early 상위 3개로만 정함): 같은 매개변수 conf 판, 반대 조건, 최선안 band 30~45, 경계면 g 한 칸 바깥.
3단계(비교 전용): 모든 바탕의 rev dev<=30 1회형 — 조건이 손절을 '좋은 손절/나쁜 손절'로 가르는지 보려고(고르기 대상 아님).
사용: py stops/g_grind.py   (code 폴더에서)   출력: ../results/stops/g_grind.txt"""
import os, sys, time, json
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
from stop_lab import load, compare, stop_audit, run, base_trades, htf_rsi, UNIV, RAW

OUT = os.path.normpath(os.path.join(CODE, "..", "results", "stops", "g_grind.txt"))
JS = os.path.join(os.path.dirname(RAW), "g_grind_result.json")
U78 = UNIV["사용자78"]
WORKERS = 4


class Tee:
    def __init__(self, path):
        self.f, self.o = open(path, "w", encoding="utf-8"), sys.stdout

    def write(self, s):
        self.o.write(s); self.f.write(s); self.f.flush()

    def flush(self):
        self.o.flush(); self.f.flush()


def _k(T, mins):
    return max(1, int(np.ceil(T / mins)))


# spec = (base, gate, defer); base = ("F1", T, Y) | ("F5", M, mode); gate = None | (kind, mode, a, b)
def gate_arr(A, gate):
    n = len(A["c"])
    if gate is None:
        return np.ones(n, bool)
    kind, mode, a, b = gate
    dev, conf = htf_rsi(A, 40, 14)
    x = dev if kind == "dev" else conf
    with np.errstate(invalid="ignore"):
        if mode == "gt":
            return x > a
        if mode == "le":
            return x <= a
        return (x > a) & (x <= b)


def make(spec):
    base, gate, defer = spec

    def make_rule(A):
        c, tk, dead = A["c"], A["tk"], A["dead"]
        n = len(c)
        g = gate_arr(A, gate)
        if base[0] == "F1":
            _, T, Y = base
            kT = _k(T, A["mins"])
            if not defer:
                def rule(i, ei, ent):
                    return i - ei == kT and tk[i] < dead[ei] and g[i] and c[ei:i].max() < ent * (1 + Y)
            else:
                def rule(i, ei, ent):
                    return i - ei >= kT and g[i] and tk[i] < dead[ei] and c[ei:i].max() < ent * (1 + Y)
            return rule
        _, M, mode = base
        ok = A["reg"] & (A["mod"] >= 570 + M)
        pos = np.flatnonzero(ok)
        k = np.searchsorted(pos, np.arange(n + 1))
        first_ok = np.where(k < len(pos), pos[np.minimum(k, len(pos) - 1)], -1)
        chk = first_ok[1:]                                  # chk[ei] = ei+1 이상에서 첫 검사 봉
        off = A["ses"] != "정규"

        def cond(i, ei, ent):
            return c[ei:i].max() < ent * 1.005 if mode == "max" else c[i - 1] < ent
        if not defer:
            def rule(i, ei, ent):
                return off[ei] and i == chk[ei] and tk[i] < dead[ei] and g[i] and cond(i, ei, ent)
        else:
            def rule(i, ei, ent):
                return off[ei] and chk[ei] >= 0 and i >= chk[ei] and g[i] and tk[i] < dead[ei] and cond(i, ei, ent)
        return rule
    return make_rule


def name(spec):
    base, gate, defer = spec
    b = f"F1 T{base[1]} Y{base[2]:+.1%}" if base[0] == "F1" else f"F5 M{base[1]} {base[2]}"
    if gate is None:
        return b + " | 조건없음"
    kind, mode, a, bb = gate
    gs = {"gt": f"{kind}>{a:g}", "le": f"rev {kind}<={a:g}"}[mode] if mode != "band" else f"band {a:g}<{kind}<={bb:g}"
    return f"{b} | {gs}" + (" 보류" if defer else "")


def job(spec):
    return name(spec), compare(make(spec), name(spec), show=False)


def cells(df):
    g = df.set_index(["기간", "범위", "종목"])
    return {k: g.loc[k] for k in g.index}


def summarize(nm, spec, df):
    c = cells(df)
    row = dict(규칙=nm, spec=spec)
    for p in ("early", "late"):
        for r, rk in (("24시간", "24"), ("정규장만", "R")):
            for un, uk in (("사용자78", ""), ("전체137", "_137")):
                x = c[(p, r, un)]
                row[f"{p[0]}{rk}{uk}"] = x.차이
                row[f"{p[0]}{rk}{uk}_t"] = x.t
                row[f"{p[0]}{rk}{uk}_손절"] = x.손절
    return row


def line(nm, df):
    c = cells(df)
    u = lambda p, r: c[(p, r, "사용자78")]
    a = lambda p, r: c[(p, r, "전체137")]
    e24, er, l24, lr = u("early", "24시간"), u("early", "정규장만"), u("late", "24시간"), u("late", "정규장만")
    return (f"{nm:<40} | e24 {e24.차이:+6.2f} (t {e24.t:+5.2f}, 손절 {e24.손절:4d}) eR {er.차이:+6.2f} | "
            f"l24 {l24.차이:+6.2f} (t {l24.t:+5.2f}, 손절 {l24.손절:4d}) lR {lr.차이:+6.2f} | "
            f"137 e24 {a('early', '24시간').차이:+6.2f} l24 {a('late', '24시간').차이:+6.2f}")


BUCKETS = [(-1, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50), (50, 101)]


def bucket_diag(spec, period):
    """조건 없는 손절의 손절 거래를 같은 매수 봉 기준 거래와 짝지어, 손절 봉 i의 dev[i] 구간별 '손절 - 기준' (재매수 금지 효과 빠짐)."""
    S = load(period)
    t = run(S, make(spec), period, False, syms=U78)
    b = base_trades(period, False)
    b = b[b.sym.isin(U78)].set_index(["sym", "ei"]).ret
    st = t[t.why == "손절"].copy()
    st["dev"] = [htf_rsi(S[x.sym], 40, 14)[0][x.xi] for x in st.itertuples()]
    st["conf"] = [htf_rsi(S[x.sym], 40, 14)[1][x.xi] for x in st.itertuples()]
    m = st.set_index(["sym", "ei"]).join(b.rename("기준"), how="inner")
    m["득실"] = m.ret - m.기준
    out = []
    for lo, hi in BUCKETS:
        x = m[(m.dev > lo) & (m.dev <= hi)]
        out.append(f"{'≤25' if lo < 0 else f'{lo}~{hi}' if hi < 101 else '>50'}: {len(x):3d}건 평균 {x.득실.mean() if len(x) else 0:+.2%} 합 {x.득실.sum():+.2f}")
    na = m.dev.isna().sum()
    return len(m), m.득실.sum(), out, na, m


def day_dep(spec, period):
    t = run(load(period), make(spec), period, False, syms=U78)
    b = base_trades(period, False)
    b = b[b.sym.isin(U78)]
    days = sorted(set(b.day) | set(t.day))
    d = t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0)
    return d.sort_values(ascending=False)


def lookahead_price(spec, period, n_max=400):
    """손절된 거래마다 봉 i 이후 h l c v r(봉 i+1 이후 o)를 NaN으로 지워도 같은 결정인지(가격 쪽)."""
    S = load(period)
    mk = make(spec)
    t = run(S, mk, period, False, syms=U78)
    st = t[t.why == "손절"].head(n_max)
    bad = 0
    for x in st.itertuples():
        A = dict(S[x.sym])
        for k in ("h", "l", "c", "v", "r"):
            a = A[k].copy(); a[x.xi:] = np.nan; A[k] = a
        o = A["o"].copy(); o[x.xi + 1:] = np.nan; A["o"] = o
        if not mk(A)(x.xi, x.ei, x.ent):
            bad += 1
    return len(st), bad


def lookahead_htf(spec, period, n_max=60, tf=40, n=14):
    """손절 봉 i의 dev[i]·conf[i]를 5분봉을 '봉 i 시작 시각까지 끝난 봉'만 남기고 다시 계산해 같은지."""
    from htf import intraday_bars, rma_state, _rsi_from
    S = load(period)
    t = run(S, make(spec), period, False, syms=U78)
    st = t[t.why == "손절"].head(n_max)
    worst, cnt = 0.0, 0
    m5c = {}
    step = pd.Timedelta(minutes=tf)
    for x in st.itertuples():
        A = S[x.sym]; i = x.xi
        if x.sym not in m5c:
            m5c[x.sym] = pd.read_parquet(os.path.join(RAW, f"5m_{x.sym}.parquet"))
        m5 = m5c[x.sym]
        ti = A["idx"][i]
        m5t = m5[m5.index + pd.Timedelta(minutes=5) <= ti]
        hs, hc = intraday_bars(m5t, tf); hc = hc.astype(float)
        g, l, r = rma_state(hc, n)
        tp = A["idx"][i - 1]
        ap = tp.normalize() + pd.Timedelta(hours=4)
        cur_prev = ap + ((tp - ap) // step) * step
        j = np.searchsorted(hs.as_unit("ns").asi8, cur_prev.as_unit("ns").value, side="left") - 1
        dv = float(_rsi_from(np.array([g[j]]), np.array([l[j]]), np.array([hc[j]]), np.array([A["c"][i - 1]]), n)[0])
        ai = ti.normalize() + pd.Timedelta(hours=4)
        cur_i = ai + ((ti - ai) // step) * step
        jc = np.searchsorted(hs.as_unit("ns").asi8, cur_i.as_unit("ns").value, side="left") - 1
        cf = float(r[jc])
        dev, conf = htf_rsi(A, tf, n)
        worst = max(worst, abs(dv - dev[i]), abs(cf - conf[i]))
        cnt += 1
    return cnt, worst


def fmt_df(df):
    f = df.copy()
    for k, fm in (("평균", "{:+.2%}"), ("승률", "{:.0%}"), ("최악", "{:+.1%}"), ("하위5", "{:+.1%}"), ("합", "{:+.2f}"),
                  ("차이", "{:+.2f}"), ("t", "{:+.2f}"), ("손절평균", "{:+.2%}"), ("차이_10월6일", "{:+.2f}")):
        f[k] = f[k].map(lambda v: "" if pd.isna(v) else fm.format(v))
    return f.to_string(index=False)


def main():
    sys.stdout = Tee(OUT)
    t0 = time.time()
    print(__doc__)
    BASES = [("F1", T, Y) for T in (30, 60, 120, 240) for Y in (0.0, 0.005)] + [("F1", 180, 0.0), ("F5", 30, "max"), ("F5", 30, "close")]

    # 0. early 진단(선택 전, early만): 조건 없는 손절의 손절 시점 dev 구간별 득실
    print("\n=== 0. early 진단: 조건 없는 손절, 손절 봉 dev[i] 구간별 '손절 - 기준'(사용자78 24시간, 같은 매수 봉 짝; 재매수 금지 효과는 빠짐)")
    for bs in BASES:
        nn, tot, out, na, _ = bucket_diag((bs, None, False), "early")
        print(f"{name((bs, None, False))}: 짝 {nn}건 합 {tot:+.2f} (dev NaN {na}건)\n    " + " | ".join(out))
    print(f"[{time.time() - t0:.0f}s]")

    stage1 = []
    for bs in BASES:
        stage1.append((bs, None, False))
        for g in (30, 35, 40, 45):
            stage1.append((bs, ("dev", "gt", g, None), False))
        if not (bs[0] == "F5" and bs[2] == "close"):
            for g in (30, 40):
                stage1.append((bs, ("dev", "gt", g, None), True))

    rows, tables = [], {}

    def run_specs(specs):
        with ProcessPoolExecutor(WORKERS) as ex:
            for spec, (nm, df) in zip(specs, ex.map(job, specs)):
                tables[nm] = (spec, df)
                rows.append(summarize(nm, spec, df))
                print(line(nm, df), f"[{time.time() - t0:5.0f}s]")

    print(f"\n=== 1단계: 바탕 {len(BASES)}개 x 조건 ({len(stage1)}개 규칙)")
    run_specs(stage1)

    R = pd.DataFrame(rows).set_index("규칙")
    gated = R[R.spec.map(lambda s: s[1] is not None and s[1][1] != "le")]
    cand = gated[gated.eR >= 0] if (gated.eR >= 0).any() else gated
    top = list(cand.sort_values("e24", ascending=False).index[:3])
    print("\n1단계 early 상위 3(조건 붙은 규칙, eR >= 0 안에서 e24 최대):", top)

    stage2, seen = [], set(tables)
    for nm in top:
        bs, (kind, mode, g, _), defer = tables[nm][0]
        for sp in ((bs, ("conf", "gt", g, None), defer), (bs, ("dev", "le", g, None), defer)):
            if name(sp) not in seen:
                stage2.append(sp); seen.add(name(sp))
    bs, (kind, mode, g, _), defer = tables[top[0]][0]
    extra = [(bs, ("dev", "band", 30, 45), defer)]
    if g == 45:
        extra.append((bs, ("dev", "gt", 50, None), defer))
    if g == 30:
        extra.append((bs, ("dev", "gt", 25, None), defer))
    for sp in extra:
        if name(sp) not in seen:
            stage2.append(sp); seen.add(name(sp))
    print(f"\n=== 2단계: early 상위 3의 conf 판·반대 조건, 최선안 band·경계 ({len(stage2)}개 규칙)")
    run_specs(stage2)

    stage3 = [sp for sp in ((bs, ("dev", "le", 30, None), False) for bs in BASES) if name(sp) not in seen]
    print(f"\n=== 3단계(비교 전용, 고르기 대상 아님): 모든 바탕의 반대 조건 1회형 rev dev<=30 ({len(stage3)}개 규칙)")
    run_specs(stage3)
    print("\n=== 조건 분리: 같은 바탕의 손절을 dev>30(가설: 자름)과 dev<=30(가설: 보류)으로 나눈 결과(사용자78 차이, 손절 수)")
    for bs in BASES:
        u, gtd, rv = (tables[name((bs, g, False))][1] for g in (None, ("dev", "gt", 30, None), ("dev", "le", 30, None)))
        s = []
        for lab, d in (("조건없음", u), ("dev>30", gtd), ("rev<=30", rv)):
            c = cells(d)
            e, l = c[("early", "24시간", "사용자78")], c[("late", "24시간", "사용자78")]
            s.append(f"{lab} e24 {e.차이:+.2f}({e.손절}) l24 {l.차이:+.2f}({l.손절})")
        print(f"{name((bs, None, False)).split(' |')[0]:<15} " + " | ".join(s))

    R = pd.DataFrame(rows).set_index("규칙")
    n_all = len(R)
    four = R[(R.e24 > 0) & (R.eR > 0) & (R.l24 > 0) & (R.lR > 0)]
    print(f"\n시험 규칙 {n_all}개(조건 없음 {int(R.spec.map(lambda s: s[1] is None).sum())} 포함). "
          f"사용자78 네 칸 모두 플러스: {len(four)}개 {list(four.index)}")
    fourz = R[(R.e24 > 0) & (R.eR >= 0) & (R.l24 > 0) & (R.lR >= 0)]
    print(f"(참고) 정규장 0 허용(e24>0, eR>=0, l24>0, lR>=0): {len(fourz)}개 {list(fourz.index)}")
    print(f"early 사용자78 24시간 차이 > 0: {(R.e24 > 0).sum()}개, 그중 정규장만 >= 0: {((R.e24 > 0) & (R.eR >= 0)).sum()}개")

    gated = R[R.spec.map(lambda s: s[1] is not None and s[1][1] != "le")]
    cand = gated[gated.eR >= 0] if (gated.eR >= 0).any() else gated
    order = cand.sort_values("e24", ascending=False)
    best = order.index[0]
    cols = ["e24", "e24_t", "e24_손절", "eR", "l24", "l24_t", "l24_손절", "lR", "e24_137", "eR_137", "l24_137", "lR_137"]
    print("\n=== early 기준 상위 12 (조건 붙은 규칙, rev 제외; 고르기: eR >= 0 안에서 e24 최대; l = late 확인용)")
    print(order[cols].head(12).to_string(float_format=lambda v: f"{v:+.2f}"))

    # 조건 효과: 같은 바탕의 조건 없음과 차이
    print("\n=== 조건 효과(같은 바탕, 조건 붙임 - 조건 없음) 사용자78 차이")
    un = {r.spec[0]: r for _, r in R.iterrows() if r.spec[1] is None}
    eff = []
    for nm, r in R.iterrows():
        if r.spec[1] is None:
            continue
        u = un[r.spec[0]]
        eff.append(dict(규칙=nm, de24=r.e24 - u.e24, deR=r.eR - u.eR, dl24=r.l24 - u.l24, dlR=r.lR - u.lR,
                        손절_e=r.e24_손절, 손절_e_없음=u.e24_손절))
    E = pd.DataFrame(eff).set_index("규칙")
    print(E.to_string(float_format=lambda v: f"{v:+.2f}"))
    gt = E[[R.loc[k].spec[1][1] == "gt" and R.loc[k].spec[1][0] == "dev" and not R.loc[k].spec[2] for k in E.index]]
    print(f"\n1회형 dev>g 조건 효과 early 24시간: 플러스 {(gt.de24 > 0).sum()}/{len(gt)}, 평균 {gt.de24.mean():+.2f}; "
          f"late 24시간(확인): 플러스 {(gt.dl24 > 0).sum()}/{len(gt)}, 평균 {gt.dl24.mean():+.2f}")
    df_ = E[[R.loc[k].spec[1][1] == "gt" and R.loc[k].spec[1][0] == "dev" and R.loc[k].spec[2] for k in E.index]]
    if len(df_):
        print(f"보류형 dev>g 조건 효과 early 24시간: 플러스 {(df_.de24 > 0).sum()}/{len(df_)}, 평균 {df_.de24.mean():+.2f}; "
              f"late 24시간(확인): 플러스 {(df_.dl24 > 0).sum()}/{len(df_)}, 평균 {df_.dl24.mean():+.2f}")

    spec, df = tables[best]
    print(f"\n=== 최선안(early로 고름): {best}  — 전체 표")
    print(fmt_df(df))
    ung = (spec[0], None, False)
    rev = (spec[0], (spec[1][0], "le", spec[1][2], None), spec[2])      # gt면 같은 g, band면 아래 경계(dev<=30) 이하에서만 자름
    print(f"\n같은 손절 조건 없음: {line(name(ung), tables[name(ung)][1])}")
    if rev is not None:
        if name(rev) not in tables:
            run_specs([rev])
        print(f"반대 조건:           {line(name(rev), tables[name(rev)][1])}")
    audits = {}
    for p in ("early", "late"):
        for reg in (False, True):
            a = stop_audit(make(spec), p, reg_only=reg)
            audits[(p, reg)] = a
            print(f"stop_audit {p} {'정규장만' if reg else '24시간'} 사용자78: " +
                  ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))
    for p in ("early", "late"):
        a = stop_audit(make(ung), p)
        print(f"(비교) 조건 없음 stop_audit {p} 24시간: " +
              ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))

    print("\n=== 날짜 쏠림(사용자78 24시간, 최선안 - 기준)")
    dd = {}
    for p in ("early", "late"):
        d = day_dep(spec, p)
        dd[p] = d
        print(f"{p}: 합 {d.sum():+.2f}, 나은 날 {(d > 1e-12).sum()} 나쁜 날 {(d < -1e-12).sum()}, 상위 3일 "
              + ", ".join(f"{k} {v:+.2f}" for k, v in d.head(3).items()) + f", 하위 3일 "
              + ", ".join(f"{k} {v:+.2f}" for k, v in d.tail(3).items()) + f", 최고일 빼면 {d.sum() - d.iloc[0]:+.2f}")

    print("\n=== late 진단(확인용, 선택 뒤): 최선안 바탕 조건 없는 손절의 dev 구간별 득실")
    nn, tot, out, na, _ = bucket_diag(ung, "late")
    print(f"{name(ung)} late: 짝 {nn}건 합 {tot:+.2f} (dev NaN {na}건)\n    " + " | ".join(out))

    for p in ("early", "late"):
        m, b = lookahead_price(spec, p)
        k, w = lookahead_htf(spec, p)
        print(f"미래 참조 점검 {p}: 가격 — 손절 {m}건 봉 i 이후 값 지우고 재판정, 바뀐 건 {b}건; "
              f"40분 RSI — {k}건을 봉 i 시작 전 끝난 5분봉만으로 다시 계산, 최대 차이 {w:.2e}")

    res = dict(best=best, spec=repr(spec), n=n_all, four=list(four.index), order=list(order.index[:6]),
               table=df.to_dict("records"), ung=name(ung), rev=name(rev) if rev else None,
               rows={k: {c: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v))
                         for c, v in r.items()} for k, r in R.iterrows()},
               audits={f"{p}_{'reg' if reg else '24'}": {k: (float(v) if not isinstance(v, str) else v) for k, v in a.items()}
                       for (p, reg), a in audits.items()})
    with open(JS, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"\n끝 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

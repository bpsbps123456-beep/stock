"""칼손절 연구 - [폭포 감지 + 40분봉 RSI 조건] (2026-10-07, 키 g_waterfall).
사용자 가설(10/7): "40분봉 RSI가 30 이하(이미 과매도)면 손절하면 안 좋다." -> 손절 신호가 나와도 40분봉 RSI가 낮으면 보류하고,
40분봉 RSI가 아직 높은데 빠질 때만 자른다.

40분봉 RSI(14): stop_lab.htf_rsi(A, tf=40, n=14). 04:00 ET 기준 40분 큰 봉, 5분봉 parquet로 만든다.
  dev[i]  = 봉 i 시가에 쓸 수 있는 진행 중 40분봉 RSI(그 전 완성 40분봉 RMA 상태에 c[i-1]을 넣음 = 트레이딩뷰 실시간 값)
  conf[i] = 봉 i 시작까지 완성된 마지막 40분봉 RSI
  NaN(early 첫날 오전 등 자리 잡기 전)이면 조건 거짓 -> 손절 안 함(정방향·반대 조건 모두).

규칙(모두 봉 i 시가에 판다, 판단은 c[i-1]·htf 값[i]까지만):
  F  창 안 종가 고정% + 조건: (i-ei) <= T//분 이고 c[i-1] <= 매수가*(1-X) 이고 H[i] > g     (H = dev 또는 conf)
     X = 3·4·5%, T = 15·30·60분, 조건 dev>30·35·40, conf>30·35·40  -> 54개
  S  하락 속도(보유 내내, waterfall.py G6과 같은 정의) + 조건: c[i-1] <= 기준*(1-y) 이고 c[i-1] <= 매수가*(1-z) 이고 dev[i] > g
     기준 = c[i-1-m] (m = M//분 봉), 단 i-1-m < ei 이면 매수가
     (M,y,z) = (15,2%,0) (15,2%,2%) (15,3%,0) (15,3%,2%) (30,3%,0) (30,3%,2%), g = 30·35·40 -> 18개
  비교용(고르는 데 안 씀): 같은 손절의 조건 없는 판(F 9 + S 6 = 15개), 반대 조건 dev<=30 판(F 9개), 최선안의 반대 조건 1개.
고르는 법(미리 정함, waterfall.py와 같음): 조건 붙은 72개 중 early 사용자78 24시간 차이 최대, 단 early 사용자78 정규장만 차이 >= 0.
  late는 전방 확인으로만 본다(late로 다시 고르지 않음).
사용: py stops/g_waterfall.py   (code 폴더에서, 결과: ../results/stops/g_waterfall.txt)
"""
import os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stop_lab import load, compare, stop_audit, base_trades, run, htf_rsi, RAW, UNIV, PNAME
from htf import intraday_bars, rma_state, _rsi_from

OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results", "stops", "g_waterfall.txt")
CACHE = os.path.join(os.path.dirname(RAW), "stop_lab_g_waterfall")
U78 = UNIV["사용자78"]


# ---------- 규칙 ----------
def _gate(A, kind):
    if kind is None:
        return None
    dev, conf = htf_rsi(A, 40, 14)
    return dev if kind == "dev" else conf


def _ok(h, i, g, rev):
    if h is None:
        return True
    v = h[i]
    return bool(v <= g) if rev else bool(v > g)       # NaN이면 둘 다 거짓


def fixed(X, T, kind=None, g=None, rev=False):
    """T분 창 안, i-1 종가 <= 매수가*(1-X), 그리고 40분봉 RSI 조건(rev=True면 반대: <= g)."""
    def make(A):
        c = A["c"]
        k = T // A["mins"]
        h = _gate(A, kind)
        def rule(i, ei, ent):
            return i - ei <= k and c[i - 1] <= ent * (1 - X) and _ok(h, i, g, rev)
        return rule
    return make


def speed(M, y, z, kind=None, g=None, rev=False):
    def make(A):
        c = A["c"]
        m = max(1, M // A["mins"])
        h = _gate(A, kind)
        def rule(i, ei, ent):
            j = i - 1
            ref = c[j - m] if j - m >= ei else ent
            return c[j] <= ref * (1 - y) and c[j] <= ent * (1 - z) and _ok(h, i, g, rev)
        return rule
    return make


def gname(kind, g, rev=False):
    return "" if kind is None else f" & {kind}40 {'<=' if rev else '>'} {g}"


def grid():
    """(계열, 이름, make, 역할) 역할: 'gated'(고르기 대상) / 'ungated' / 'reverse'."""
    G = []
    FS = [(X, T) for X in (0.03, 0.04, 0.05) for T in (15, 30, 60)]
    SS = [(15, 0.02, 0.0), (15, 0.02, 0.02), (15, 0.03, 0.0), (15, 0.03, 0.02), (30, 0.03, 0.0), (30, 0.03, 0.02)]
    for X, T in FS:
        for kind in ("dev", "conf"):
            for g in (30, 35, 40):
                G.append(("F", f"종가 -{X:.0%} {T}분 안{gname(kind, g)}", fixed(X, T, kind, g), "gated"))
    for M, y, z in SS:
        for g in (30, 35, 40):
            G.append(("S", f"{M}분 -{y:.0%} & 매수가 -{z:.0%} 아래{gname('dev', g)}", speed(M, y, z, "dev", g), "gated"))
    for X, T in FS:
        G.append(("F", f"종가 -{X:.0%} {T}분 안", fixed(X, T), "ungated"))
        G.append(("F", f"종가 -{X:.0%} {T}분 안{gname('dev', 30, True)}", fixed(X, T, "dev", 30, True), "reverse"))
    for M, y, z in SS:
        G.append(("S", f"{M}분 -{y:.0%} & 매수가 -{z:.0%} 아래", speed(M, y, z), "ungated"))
    return G, FS, SS


# ---------- 미래 참조 점검 ----------
def lookahead_check(make, n_pts=300, seed=7):
    """봉 i 이후 기준봉 값(o h l c v r ema)을 무작위로 바꿔도 rule(i, ei, ent) 판단이 같아야 한다(waterfall.py 방식).
    40분봉 RSI는 htf_rsi 캐시 값을 쓰는데, dev[i]가 쓰는 기준봉 값은 c[i-1]뿐이라 바뀌지 않는다(5분봉 쪽은 아래 htf_check)."""
    rng = np.random.default_rng(seed)
    bad, tot = 0, 0
    for period in ("early", "late"):
        S = load(period)
        b = base_trades(period)
        t = run(S, make, period)
        pts = [(x.sym, x.ei, x.xi, x.ent) for x in t[t.why == "손절"].itertuples()][: n_pts // 2]
        for x in b.sample(min(len(b), n_pts // 2), random_state=seed).itertuples():
            if x.xi - 1 > x.ei:
                pts.append((x.sym, x.ei, int(rng.integers(x.ei + 1, x.xi)), x.ent))
        for s, ei, i, ent in pts:
            A = S[s]
            r0 = make(A)(i, ei, ent)
            B = dict(A)
            for key in ("o", "h", "l", "c", "v", "r", "ema"):
                arr = A[key].copy()
                arr[i:] = arr[i:] * rng.uniform(0.3, 1.7, len(arr) - i)
                B[key] = arr
            r1 = make(B)(i, ei, ent)
            tot += 1
            bad += int(bool(r0) != bool(r1))
    return bad, tot


def htf_check(n_pts=120, seed=11, tf=40, n=14):
    """40분봉 RSI 독립 재계산: 봉 i 시작 시각 이후에 끝나는 5분봉을 모두 지운 데이터로 dev[i]·conf[i]를 다시 만들어
    htf_rsi 값과 같아야 한다. (5분봉 [s, s+5분)은 s+5분 <= tk[i]일 때만 남김. 지운 뒤에도 같으면 미래 5분봉을 안 쓴 것.)"""
    rng = np.random.default_rng(seed)
    bad, tot, maxd = 0, 0, 0.0
    step = pd.Timedelta(minutes=tf)
    for period in ("early", "late"):
        S = load(period)
        b = base_trades(period)
        for x in b.sample(n_pts // 2, random_state=seed).itertuples():
            if x.xi - 1 <= x.ei:
                continue
            A = S[x.sym]
            i = int(rng.integers(x.ei + 1, x.xi))
            dev, conf = htf_rsi(A, tf, n)
            m5 = pd.read_parquet(os.path.join(RAW, f"5m_{x.sym}.parquet"))
            ti = A["idx"][i]
            cut = m5[m5.index + pd.Timedelta(minutes=5) <= ti]
            hs, hc = intraday_bars(cut, tf)
            hc = hc.astype(float)
            g, l, r = rma_state(hc, n)
            # dev: 봉 i-1이 속한 큰 봉 시작 전의 마지막 완성 큰 봉 상태 + c[i-1]
            tp = A["idx"][i - 1]
            anc = tp.normalize() + pd.Timedelta(hours=4)
            curp = anc + ((tp - anc) // step) * step
            hsn = hs.as_unit("ns").asi8
            j = np.searchsorted(hsn, curp.value, side="left") - 1
            d2 = _rsi_from(np.array([g[j]]), np.array([l[j]]), np.array([hc[j]]), np.array([A["c"][i - 1]]), n)[0] if j >= 0 else np.nan
            anc_i = ti.normalize() + pd.Timedelta(hours=4)
            curi = anc_i + ((ti - anc_i) // step) * step
            jc = np.searchsorted(hsn, curi.value, side="left") - 1
            c2 = r[jc] if jc >= 0 else np.nan
            for a, bb in ((dev[i], d2), (conf[i], c2)):
                tot += 1
                same = (np.isnan(a) and np.isnan(bb)) or (not np.isnan(a) and not np.isnan(bb) and abs(a - bb) < 1e-6)
                bad += int(not same)
                if not np.isnan(a) and not np.isnan(bb):
                    maxd = max(maxd, abs(a - bb))
    return bad, tot, maxd


# ---------- 보조 ----------
def cell(df, period, scope, un="사용자78"):
    return df[(df.기간 == period) & (df.범위 == scope) & (df.종목 == un)].iloc[0]


def day_dep(make, period, reg_only=False):
    t = run(load(period), make, period, reg_only, syms=U78)
    b = base_trades(period, reg_only)
    b = b[b.sym.isin(U78)]
    days = sorted(set(t.day) | set(b.day))
    d = t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0)
    return d[d.abs() > 1e-12].sort_values(ascending=False)


def gate_split(make_un, kind, g, period):
    """조건 없는 손절이 잘랐던 거래(사용자78 24시간)를 손절 순간 40분봉 RSI로 나눠, (손절 수익 - 기준 수익) 합을 본다.
    = 조건이 '살려 둔' 손절(<= g)과 '그대로 둔' 손절(> g) 각각의 직접 득실."""
    S = load(period)
    t = run(S, make_un, period, syms=U78)
    b = base_trades(period)
    b = b[b.sym.isin(U78)][["sym", "ei", "ret"]].rename(columns={"ret": "기준"})
    s = t[t.why == "손절"].merge(b, on=["sym", "ei"], how="inner")
    hv = []
    for x in s.itertuples():
        dev, conf = htf_rsi(S[x.sym], 40, 14)
        hv.append((dev if kind == "dev" else conf)[x.xi])     # 판 봉 xi 시가 시점 값(= rule이 본 값)
    s["h"] = hv
    s["d"] = s.ret - s.기준
    lo, hi, na = s[s.h <= g], s[s.h > g], s[s.h.isna()]
    f = lambda z: f"{len(z)}건 득실합 {z.d.sum():+.2f} 평균 {z.d.mean() if len(z) else float('nan'):+.2%} 헛손절 {(z.기준 > 0).mean() if len(z) else float('nan'):.0%}"
    if len(lo) > 1 and len(hi) > 1:
        se = np.sqrt(lo.d.var(ddof=1) / len(lo) + hi.d.var(ddof=1) / len(hi))
        wt = (hi.d.mean() - lo.d.mean()) / se if se > 0 else float("nan")
    else:
        wt = float("nan")
    return (f"{period}: 조건 없는 손절 {len(s)}건 | {kind}40<={g}: {f(lo)} | >{g}: {f(hi)} | NaN {len(na)}건 | "
            f"평균 차(>g - <=g) Welch t {wt:+.2f}")


def main():
    lines = []
    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    out("칼손절 연구 [폭포 감지 + 40분봉 RSI 조건] - g_waterfall.py (2026-10-07)")
    out(__doc__.split("\n", 1)[1].rsplit("사용:", 1)[0].rstrip())
    out("주의: 시험 전 시간 측정용 탐침 1개(종가 -3% 30분 안 & dev40 > 35)를 돌려 그 규칙의 late 숫자가 화면에 한 번 보였다."
        " 고르는 법은 위에 미리 정한 기계적 기준(early만)이라 영향을 주지 않게 했다.")
    out("")
    for p in ("early", "late"):
        for reg in (False, True):
            b = base_trades(p, reg)
            out(f"기준 {PNAME[p]} {'정규장만' if reg else '24시간'}: " + ", ".join(
                f"{un} {len(b[b.sym.isin(us)])}건 합 {b[b.sym.isin(us)].ret.sum():+.2f}" for un, us in UNIV.items()))

    G, FS, SS = grid()
    roles = pd.Series([x[3] for x in G]).value_counts().to_dict()
    out(f"\n시험 규칙 수: {len(G)} (조건 붙은 고르기 대상 {roles.get('gated', 0)}, 조건 없는 비교 {roles.get('ungated', 0)}, "
        f"반대 조건 비교 {roles.get('reverse', 0)}; 최선안 반대 조건은 뒤에 따로 1개)")
    rows, full = [], {}
    t0 = time.time()
    os.makedirs(CACHE, exist_ok=True)
    for n, (fam, name, make, role) in enumerate(G, 1):
        cp = os.path.join(CACHE, f"{n:03d}.csv")
        df = pd.read_csv(cp) if os.path.exists(cp) else None
        if df is None or df.규칙.iloc[0] != name:
            df = compare(make, name, show=False)
            df.to_csv(cp, index=False)
        full[name] = df
        rec = dict(번호=n, 계열=fam, 역할=role, 규칙=name)
        for p in ("early", "late"):
            for sc, tag in (("24시간", "24"), ("정규장만", "정규")):
                r = cell(df, p, sc)
                rec[f"{p}_{tag}"] = r.차이
                rec[f"{p}_{tag}_t"] = r.t
                rec[f"{p}_{tag}_손절"] = r.손절
            rec[f"{p}_24_137"] = cell(df, p, "24시간", "전체137").차이
            rec[f"{p}_정규_137"] = cell(df, p, "정규장만", "전체137").차이
        rows.append(rec)
        print(f"  [{n}/{len(G)}] {name}: early24 {rec['early_24']:+.2f} early정규 {rec['early_정규']:+.2f} ({time.time() - t0:.0f}s)",
              flush=True)
    R = pd.DataFrame(rows)
    MK = {x[1]: x[2] for x in G}

    cols = ["번호", "계열", "역할", "규칙", "early_24", "early_24_t", "early_24_손절", "early_정규", "early_정규_t",
            "late_24", "late_24_t", "late_24_손절", "late_정규", "late_정규_t", "early_24_137", "late_24_137"]
    fmtf = lambda v: f"{v:+.2f}"
    out("\n=== 전체 결과(사용자78 차이 = 규칙 합 - 기준 합; 137은 전체137 24시간 차이) ===")
    out("  (late 열은 전방 확인용 - 고르는 데 쓰지 않음)")
    out(R[cols].to_string(index=False, float_format=fmtf))

    four = lambda X: X[(X.early_24 > 0) & (X.early_정규 > 0) & (X.late_24 > 0) & (X.late_정규 > 0)]
    Gt = R[R.역할 == "gated"]
    out(f"\n조건 붙은 {len(Gt)}개: early 사용자78 24시간 플러스 {int((Gt.early_24 > 0).sum())}, early 두 칸 플러스 "
        f"{int(((Gt.early_24 > 0) & (Gt.early_정규 > 0)).sum())}, 사용자78 네 칸 모두 플러스 {len(four(Gt))} {list(four(Gt).규칙)}")
    out(f"전체 {len(R)}개(비교용 포함): 네 칸 모두 플러스 {len(four(R))} {list(four(R).규칙)}")

    # 조건 효과 표: 같은 손절의 조건 없는 판 대비
    out("\n=== 조건 효과: 같은 손절에서 조건 없음 / dev40>30·35·40 / conf40>30·35·40 / 반대(dev40<=30) (사용자78 early 24시간 | early 정규 | late 24시간 | late 정규) ===")
    for X, T in FS:
        base = f"종가 -{X:.0%} {T}분 안"
        names = [base] + [base + gname(k, g) for k in ("dev", "conf") for g in (30, 35, 40)] + [base + gname("dev", 30, True)]
        sub = R.set_index("규칙").loc[names, ["early_24", "early_정규", "late_24", "late_정규", "early_24_손절", "late_24_손절"]]
        sub.index = ["  " + (x.replace(base, "") or "조건 없음") for x in names]
        out(f" [{base}]")
        out(sub.to_string(float_format=fmtf))
    for M, y, z in SS:
        base = f"{M}분 -{y:.0%} & 매수가 -{z:.0%} 아래"
        names = [base] + [base + gname("dev", g) for g in (30, 35, 40)]
        sub = R.set_index("규칙").loc[names, ["early_24", "early_정규", "late_24", "late_정규", "early_24_손절", "late_24_손절"]]
        sub.index = ["  " + (x.replace(base, "") or "조건 없음") for x in names]
        out(f" [{base}]")
        out(sub.to_string(float_format=fmtf))

    # 같은 손절 기준 조건 효과 요약(early 24시간): 조건 붙은 판 - 조건 없는 판
    eff = []
    for x in Gt.itertuples():
        base = x.규칙.split(" & dev40")[0].split(" & conf40")[0]
        u = R[R.규칙 == base].iloc[0]
        eff.append(dict(규칙=x.규칙, e24=x.early_24 - u.early_24, e정규=x.early_정규 - u.early_정규,
                        l24=x.late_24 - u.late_24, l정규=x.late_정규 - u.late_정규))
    E = pd.DataFrame(eff)
    out(f"\n조건 붙은 판 - 조건 없는 판 (72개): early24 개선 {int((E.e24 > 0).sum())}/72 중앙 {E.e24.median():+.2f}, "
        f"early정규 개선 {int((E.e정규 > 0).sum())}/72 중앙 {E.e정규.median():+.2f}, "
        f"late24 개선 {int((E.l24 > 0).sum())}/72 중앙 {E.l24.median():+.2f}, late정규 개선 {int((E.l정규 > 0).sum())}/72 중앙 {E.l정규.median():+.2f}")

    # 고르기
    cand = Gt[Gt.early_정규 >= 0].sort_values(["early_24", "early_24_t"], ascending=False)
    if not len(cand):
        cand = Gt.sort_values("early_24", ascending=False)
    best = cand.iloc[0]
    bname = best.규칙
    out(f"\n=== early로 고른 최선안: [{best.번호}] {bname} ===")
    bdf = full[bname]
    out(bdf.assign(**{k: bdf[k].map(fmt.format) for k, fmt in (
        ("평균", "{:+.2%}"), ("승률", "{:.0%}"), ("최악", "{:+.1%}"), ("하위5", "{:+.1%}"), ("합", "{:+.2f}"),
        ("차이", "{:+.2f}"), ("t", "{:+.2f}"), ("차이_10월6일", "{:+.2f}"))}).drop(columns=["손절평균"]).to_string(index=False))
    make = MK[bname]
    for p in ("early", "late"):
        for reg in (False, True):
            a = stop_audit(make, p, reg)
            out(f"stop_audit {p} {'정규장만' if reg else '24시간'} 사용자78: " + ", ".join(
                f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))

    # 같은 손절의 조건 없는 판 / 반대 조건 판
    if best.계열 == "F":
        base = bname.split(" & ")[0]
    else:
        base = bname.rsplit(" & dev40", 1)[0]
    kind = "dev" if " & dev40" in bname else "conf"
    g = int(bname.rsplit(" ", 1)[1])
    un = R[R.규칙 == base].iloc[0]
    if best.계열 == "F":
        X = float(base.split("-")[1].split("%")[0]) / 100
        T = int(base.split(" ")[2].replace("분", ""))
        rev_make = fixed(X, T, kind, g, True)
    else:
        M = int(base.split("분")[0]); y = float(base.split("-")[1].split("%")[0]) / 100; z = float(base.split("-")[2].split("%")[0]) / 100
        rev_make = speed(M, y, z, kind, g, True)
    rev_name = base + gname(kind, g, True)
    rdf = full[rev_name] if rev_name in full else compare(rev_make, rev_name, show=False)
    rv = {f"{p}_{tag}": cell(rdf, p, sc).차이 for p in ("early", "late") for sc, tag in (("24시간", "24"), ("정규장만", "정규"))}
    rvs = {f"{p}_{tag}": cell(rdf, p, sc).손절 for p in ("early", "late") for sc, tag in (("24시간", "24"), ("정규장만", "정규"))}
    out(f"\n조건 없는 같은 손절 [{base}]: early24 {un.early_24:+.2f}(손절 {un.early_24_손절:.0f}) early정규 {un.early_정규:+.2f} "
        f"late24 {un.late_24:+.2f}(손절 {un.late_24_손절:.0f}) late정규 {un.late_정규:+.2f} | 137 early24 {un.early_24_137:+.2f} late24 {un.late_24_137:+.2f}")
    out(f"반대 조건 [{rev_name}]: early24 {rv['early_24']:+.2f}(손절 {rvs['early_24']:.0f}) early정규 {rv['early_정규']:+.2f} "
        f"late24 {rv['late_24']:+.2f}(손절 {rvs['late_24']:.0f}) late정규 {rv['late_정규']:+.2f} | 137 early24 "
        f"{cell(rdf, 'early', '24시간', '전체137').차이:+.2f} late24 {cell(rdf, 'late', '24시간', '전체137').차이:+.2f}")
    for p in ("early", "late"):
        a = stop_audit(rev_make, p, False)
        out(f"  반대 조건 stop_audit {p} 24시간: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))
        a = stop_audit(MK[base], p, False)
        out(f"  조건 없음 stop_audit {p} 24시간: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))

    out("\n=== 조건 없는 손절을 손절 순간 40분봉 RSI로 나눔(직접 득실 = 손절 수익 - 기준 수익, 사용자78 24시간) ===")
    for nm, kd, gg in ((base, kind, g), ("종가 -3% 30분 안", "dev", 30), ("종가 -3% 60분 안", "dev", 30), ("종가 -5% 30분 안", "dev", 30)):
        out(f" [{nm} / {kd}40 {gg}]")
        for p in ("early", "late"):
            out("   " + gate_split(MK[nm], kd, gg, p))

    out("\n=== 최선안 late 24시간 손절 내역(사용자78) ===")
    tl = run(load("late"), make, "late", syms=U78)
    bl = base_trades("late")
    st = tl[tl.why == "손절"].merge(bl[["sym", "ei", "ret", "why"]], on=["sym", "ei"], how="left", suffixes=("", "_기준"))
    out(st[["sym", "t", "ses", "bars", "ret", "ret_기준", "why_기준"]].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    out("  세션별 손절 수(매수 세션): " + ", ".join(f"{k} {v}" for k, v in st.ses.value_counts().items()))

    out("\n=== 최선안 하루 의존(사용자78, 날짜별 합 차이) ===")
    for p in ("early", "late"):
        for reg in (False, True):
            d = day_dep(make, p, reg)
            tot = d.sum()
            top = d.iloc[0] if len(d) else 0.0
            out(f"  {p} {'정규장만' if reg else '24시간'}: 차이 {tot:+.3f}, 바뀐 날 {len(d)}, 최대 이득일 {d.index[0] if len(d) else '-'} {top:+.3f}, "
                f"그날 빼면 {tot - top:+.3f}; 상위3 {', '.join(f'{k}:{v:+.3f}' for k, v in d.head(3).items())}; "
                f"하위3 {', '.join(f'{k}:{v:+.3f}' for k, v in d.tail(3).items())}")

    out("\n=== 차순위(조건 붙은 규칙, early 사용자78 24시간 차이 순, 정규장만 >= 0 조건) ===")
    out(cand.head(8)[["번호", "규칙", "early_24", "early_24_t", "early_정규", "late_24", "late_24_t", "late_정규", "early_24_손절", "late_24_손절"]]
        .to_string(index=False, float_format=fmtf))
    out("\n=== 참고: 조건 무시 early 24시간 상위 5(조건 붙은 규칙) ===")
    out(Gt.sort_values("early_24", ascending=False).head(5)[["번호", "규칙", "early_24", "early_정규", "late_24", "late_정규"]]
        .to_string(index=False, float_format=fmtf))

    out("\n=== 미래 참조 점검 ===")
    bad, tot = lookahead_check(make)
    out(f"  최선안 {bname}: 봉 i 이후 기준봉 값을 바꿔도 판단이 바뀐 점 {bad}/{tot}")
    nm2 = Gt[Gt.계열 == "S"].sort_values("early_24", ascending=False).iloc[0].규칙
    bad, tot = lookahead_check(MK[nm2])
    out(f"  S 계열 early 최고 {nm2}: 바뀐 점 {bad}/{tot}")
    bad, tot, md = htf_check()
    out(f"  40분봉 RSI 독립 재계산(봉 i 시작 뒤에 끝나는 5분봉 삭제): 다른 값 {bad}/{tot}, 최대 차 {md:.2e}")
    out(f"\n총 실행 {time.time() - t0:.0f}s, compare 평가 {len(G) + (0 if rev_name in full else 1)}회")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("저장:", OUT)


if __name__ == "__main__":
    main()

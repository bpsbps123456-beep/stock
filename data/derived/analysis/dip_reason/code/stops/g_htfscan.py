"""칼손절 + 큰 봉 RSI 조건 — 큰 봉 길이·RSI 길이 견고성 지도(g_htfscan, 2026-10-07).

사용자 가설(10/7): 40분봉 RSI가 30 이하(이미 과매도)일 때는 손절하면 안 좋다.
=> 손절 신호가 나와도 큰 봉 RSI <= g 이면 보류하고, 큰 봉 RSI > g 일 때만 자른다.
이 스크립트의 목적: 40분이 특별한지(그 근처만 튀는지) 큰 봉 길이에 따라 매끄럽게 변하는지(견고) 보는 지도.

규칙(모두 봉 i 시가에 판다, 값은 i-1 봉까지 + htf_rsi 결과는 인덱스 i까지):
  기본 손절(base)
    F3  c[i-1] <= ent*0.97                      (종가 -3%, 들고 있는 내내)
    F5  c[i-1] <= ent*0.95                      (종가 -5%)
    BE  fixedtrail 계열 최선안 '본전+2.0%(종가무장)|깨어' = max c[ei..i-1] >= ent*1.02 를 찍은 뒤 c[i-1] <= ent,
        봉 i 시작 시각이 KST 08~24시일 때만 (stops/fixedtrail.py make_rule 그대로 import)
  조건(gate): dev, conf = htf_rsi(A, tf, n)  (stop_lab, 04:00 ET 기준, 5분봉으로 만든 tf분 큰 봉 RSI)
    G = dev 또는 conf, 값은 G[i]
    'gt'  base 이고 G[i] > g  일 때만 판다   (사용자 가설 방향)
    'le'  base 이고 G[i] <= g 일 때만 판다   (반대 조건, 대조용)
    G[i]가 nan(자료 앞쪽 RSI 준비 전)이면 조건을 모른다고 보고 base 그대로 판다(조건 효과 0, 두 방향 같음).
격자(조건 있는 규칙):
  F3: 큰 봉 20·30·40·60·80·120·160분 × RSI 7·14 × dev/conf × g 30·35  = 56
  F5: 같은 7개 큰 봉 × RSI 14 × dev/conf × g 30·35                    = 28
  BE: 같은 7개 큰 봉 × RSI 14 × dev × g 30                            = 7
대조: 조건 없는 F3·F5·BE 3개, 반대 조건(le) F3·F5 40분 RSI14 dev 30 2개 + 최선안의 반대 조건 1개.

선택: 조건 있는 91개 중 early 사용자78 24시간 차이 최대(보조 조건: early 사용자78 정규장만 차이 >= 0). late는 확인만.
사용: py stops/g_htfscan.py   (결과: ../results/stops/g_htfscan.txt, 약 9분)
      py stops/g_htfscan.py audit   (그 뒤 실행: 고정 손절 조건 규칙 early 1·2위와 가설 그대로인 40분 규칙의 stop_audit을 결과 끝에 덧붙임)
"""
import os, sys, time
from datetime import date
from functools import partial
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import stop_lab  # noqa: E402
from stop_lab import compare, stop_audit, load, base_trades, run, metrics, htf_rsi, UNIV, RAW  # noqa: E402
from fixedtrail import make_rule as ft_rule  # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "stops", "g_htfscan.txt"))
TFS = (20, 30, 40, 60, 80, 120, 160)
U78, A137 = UNIV["사용자78"], UNIV["전체137"]
POST = date(2026, 8, 3)          # early 큰 봉 RSI 준비(RMA 수렴) 뒤 구간 민감도: 8/3~9/8
BNAME = {"F3": "종가-3%", "F5": "종가-5%", "BE": "본전+2%|깨어"}


def make_rule(A, base, tf=None, n=14, kind="dev", g=30.0, side="gt"):
    c = A["c"]
    if base == "BE":
        b = ft_rule(A, kind="be", p=0.02, arm="c", ses="깨어")
    else:
        p = 0.03 if base == "F3" else 0.05

        def b(i, ei, ent):
            return c[i - 1] <= ent * (1 - p)
    if tf is None:
        return b
    dev, conf = htf_rsi(A, tf, n)
    G = dev if kind == "dev" else conf

    def rule(i, ei, ent):
        hit = b(i, ei, ent)          # base를 먼저 늘 호출(BE 누적 상태 유지)
        if not hit:
            return False
        x = G[i]
        if x != x:                   # nan: 조건 모름 -> base 그대로
            return True
        return x > g if side == "gt" else x <= g
    return rule


def name_of(sp):
    s = BNAME[sp["base"]]
    if sp.get("tf") is None:
        return s + "|조건없음"
    op = ">" if sp.get("side", "gt") == "gt" else "<="
    return f"{s}|{sp['tf']}분RSI{sp.get('n', 14)}{sp.get('kind', 'dev')}{op}{sp.get('g', 30):g}"


def grid():
    G = []
    G += [dict(base="F3", tf=tf, n=n, kind=k, g=g) for tf in TFS for n in (7, 14) for k in ("dev", "conf") for g in (30, 35)]
    G += [dict(base="F5", tf=tf, n=14, kind=k, g=g) for tf in TFS for k in ("dev", "conf") for g in (30, 35)]
    G += [dict(base="BE", tf=tf, n=14, kind="dev", g=30) for tf in TFS]
    return G


CTRL = [dict(base="F3"), dict(base="F5"), dict(base="BE"),
        dict(base="F3", tf=40, n=14, kind="dev", g=30, side="le"), dict(base="F5", tf=40, n=14, kind="dev", g=30, side="le")]

_S = {}


def _S_of(p):
    if p not in _S:
        _S[p] = load(p)
    return _S[p]


def daydiff(t, b, syms, lo=None):
    t, b = t[t.sym.isin(syms)], b[b.sym.isin(syms)]
    days = sorted(set(b.day) | set(t.day))
    d = (t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0))
    if lo is not None:
        d = d[[x >= lo for x in d.index]]
    return d


def job(sp):
    """compare와 같은 칸(stop_lab.metrics) + 덧붙임: 최대 이득일 하나 뺀 차이, early 8/3 이후 차이."""
    mk = partial(make_rule, **sp)
    rows = []
    for period in ("early", "late"):
        S = _S_of(period)
        for reg in (False, True):
            b = base_trades(period, reg)
            t = run(S, mk, period, reg)
            for un, us in (("사용자78", U78), ("전체137", A137)):
                m = metrics(t, b, us)
                d = daydiff(t, b, us)
                m["차이_최고일뺌"] = d.sum() - d.max() if len(d) else 0.0
                m["최고일"] = str(d.idxmax()) if len(d) and d.max() > 0 else ""
                m["차이_8월3일후"] = daydiff(t, b, us, POST).sum() if period == "early" else np.nan
                rows.append(dict(규칙=name_of(sp), 기간=period, 범위="정규장만" if reg else "24시간", 종목=un, **m))
    return sp, pd.DataFrame(rows)


def job_batch(specs):
    """같은 (큰 봉, RSI 길이)끼리 묶어 한 작업자에서 돌린다. 큰 봉 RSI 캐시는 묶음마다 비워 메모리를 아낀다."""
    stop_lab._HTF.clear()
    out = [job(sp) for sp in specs]
    stop_lab._HTF.clear()
    return out


def batches(specs):
    grp = {}
    for sp in specs:
        grp.setdefault((sp.get("tf"), sp.get("n")), []).append(sp)
    return sorted(grp.values(), key=len, reverse=True)


def cell(df, period, scope, un, col="차이"):
    r = df[(df.기간 == period) & (df.범위 == scope) & (df.종목 == un)]
    return float(r[col].iloc[0])


def srow(nm, df):
    g = partial(cell, df)
    return dict(규칙=nm,
                e24=g("early", "24시간", "사용자78"), e24t=g("early", "24시간", "사용자78", "t"),
                e24손절=int(g("early", "24시간", "사용자78", "손절")),
                eReg=g("early", "정규장만", "사용자78"), eRegt=g("early", "정규장만", "사용자78", "t"),
                l24=g("late", "24시간", "사용자78"), l24t=g("late", "24시간", "사용자78", "t"),
                l24손절=int(g("late", "24시간", "사용자78", "손절")),
                lReg=g("late", "정규장만", "사용자78"), lRegt=g("late", "정규장만", "사용자78", "t"),
                e24_137=g("early", "24시간", "전체137"), eReg_137=g("early", "정규장만", "전체137"),
                l24_137=g("late", "24시간", "전체137"), lReg_137=g("late", "정규장만", "전체137"),
                e24_8월=g("early", "24시간", "사용자78", "차이_8월3일후"),
                e24_최고일뺌=g("early", "24시간", "사용자78", "차이_최고일뺌"),
                l24_최고일뺌=g("late", "24시간", "사용자78", "차이_최고일뺌"),
                e24최악=g("early", "24시간", "사용자78", "최악"), l24최악=g("late", "24시간", "사용자78", "최악"),
                oct6_l24=g("late", "24시간", "사용자78", "차이_10월6일"))


def htf_value_check(period, per=25, seed=0):
    """htf_rsi 값이 '봉 i 시작 전에 끝난 5분봉'만으로 다시 계산한 값과 같은지 확인(5분봉 원자료를 tk[i]에서 잘라 재계산).
    dev[i] = 잘린 자료 + c[i-1](봉 i-1 시작 시각 기준 진행 중 큰 봉), conf[i] = 잘린 자료의 마지막 완성 큰 봉."""
    from htf import rsi_intraday
    S = _S_of(period)
    rng = np.random.default_rng(seed)
    syms = sorted(S)
    bad = tot = 0
    worst = 0.0
    for tf in TFS:
        for n in (7, 14):
            for _ in range(per):
                s = syms[int(rng.integers(len(syms)))]
                A = S[s]
                i = int(rng.integers(200, len(A["c"])))
                dev, conf = htf_rsi(A, tf, n)
                m5 = pd.read_parquet(os.path.join(RAW, f"5m_{s}.parquet"))
                cut = m5[(m5.index.as_unit("ns").asi8 + 5 * 60 * 10**9) <= A["tk"][i]]
                d2, _ = rsi_intraday(cut, tf, n, A["idx"][i - 1:i], np.array([A["c"][i - 1]]))
                _, c2 = rsi_intraday(cut, tf, n, A["idx"][i:i + 1], np.array([np.nan]))
                for a, b2 in ((dev[i], d2[0]), (conf[i], c2[0])):
                    tot += 1
                    if (np.isnan(a) != np.isnan(b2)) or (not np.isnan(a) and abs(a - b2) > 1e-6):
                        bad += 1
                        worst = max(worst, abs(a - b2) if not np.isnan(a - b2) else 999)
            stop_lab._HTF.clear()
    return bad, tot, worst


def rule_trunc_check(sp, period, n=150, seed=1):
    """봉 i 이후 가격(o h l c v r ema)을 nan으로 지운 배열로 만든 규칙이 전체 배열 규칙과 같은 판단을 하는지."""
    S = _S_of(period)
    b = base_trades(period)
    rng = np.random.default_rng(seed)
    rows = b.sample(min(n, len(b)), random_state=seed)
    bad = cnt = 0
    for x in rows.itertuples():
        if x.xi <= x.ei + 1:
            continue
        A = S[x.sym]
        i = int(rng.integers(x.ei + 1, x.xi + 1))
        full = make_rule(A, **sp)
        for j in range(x.ei + 1, i):
            full(j, x.ei, x.ent)
        T = {k: (v[:i + 1] if isinstance(v, np.ndarray) else v) for k, v in A.items()}
        T["idx"] = A["idx"][:i + 1]
        for k in ("o", "h", "l", "c", "v", "r", "ema"):
            T[k] = T[k].astype(float).copy()
            T[k][i] = np.nan
        tr = make_rule(T, **sp)
        for j in range(x.ei + 1, i):
            tr(j, x.ei, x.ent)
        try:
            same = tr(i, x.ei, x.ent) == full(i, x.ei, x.ent)
        except IndexError:
            same = False
        bad += (not same)
        cnt += 1
        for key in [k for k in stop_lab._HTF if k[2] == i + 1]:
            stop_lab._HTF.pop(key, None)
    return bad, cnt


def nan_share(period):
    """early 앞쪽 큰 봉 RSI 준비 전(nan) 봉 비율, 기간 안 봉 기준."""
    S = _S_of(period)
    lo = stop_lab.PERIODS[period][3]
    out = {}
    for tf in TFS:
        for n in (7, 14):
            tot = na = 0
            for A in S.values():
                dev, _ = htf_rsi(A, tf, n)
                m = np.array([d >= lo for d in A["day"]])
                tot += m.sum()
                na += np.isnan(dev[m]).sum()
            out[(tf, n)] = na / max(tot, 1)
            stop_lab._HTF.clear()
    return out


def audit_extra():
    """추가 감사(py stops/g_htfscan.py audit): 사용자 가설 그대로인 40분 규칙과 고정 손절 계열 early 최선안의
    stop_audit·반대 조건·조건 없는 손절을 결과 파일 끝에 덧붙인다. 매개변수는 다시 고르지 않는다(early 순위로 이미 정해진 것만)."""
    refs = [dict(base="F5", tf=30, n=14, kind="conf", g=30),    # 고정 손절(F3·F5) 조건 규칙 중 early 24시간 1위(보조 조건 통과)
            dict(base="F3", tf=40, n=7, kind="dev", g=30),      # 같은 순위 2위
            dict(base="F3", tf=40, n=14, kind="dev", g=30),     # 사용자 가설 그대로(40분봉 RSI14 > 30)
            dict(base="F5", tf=40, n=14, kind="dev", g=30)]
    lines = ["", "== 추가 감사: 고정 손절(F3·F5) 조건 규칙 early 1·2위와 사용자 가설 그대로인 40분 규칙 (사용자78, 24시간/정규장만)"]
    for sp in refs:
        for tag, x in (("조건", sp), ("반대", dict(sp, side="le")), ("조건없음", dict(base=sp["base"]))):
            mk = partial(make_rule, **x)
            for p in ("early", "late"):
                for reg in (False, True):
                    a = stop_audit(mk, p, reg_only=reg)
                    lines.append(f"  [{name_of(sp)}] {tag} {name_of(x)} {p} {'정규장만' if reg else '24시간'}: " +
                                 ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in a.items()))
            stop_lab._HTF.clear()
        lines.append("")
    with open(OUT, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "audit":
    audit_extra()
elif __name__ == "__main__":
    t0 = time.time()
    G = grid()
    with Pool(4) as pool:
        res = [x for part in pool.map(job_batch, batches(G + CTRL), chunksize=1) for x in part]
    spec_of, tables, summ = {}, [], []
    for sp, df in res:
        nm = name_of(sp)
        spec_of[nm] = sp
        tables.append(df)
        summ.append(srow(nm, df))
    sm = pd.DataFrame(summ)
    gated = sm[sm.규칙.isin([name_of(sp) for sp in G])]
    elig = gated[gated.eReg >= 0].sort_values("e24", ascending=False)
    pick = elig.iloc[0].규칙 if len(elig) else gated.sort_values("e24", ascending=False).iloc[0].규칙
    bsp = spec_of[pick]
    rev = dict(bsp, side="le")
    extra = []
    if name_of(rev) not in spec_of:
        sp2, df2 = job(rev)
        spec_of[name_of(rev)] = rev
        tables.append(df2)
        summ.append(srow(name_of(rev), df2))
        extra.append(rev)
    sm = pd.DataFrame(summ)
    n_rules = len(sm)
    allpos = sm[(sm.e24 > 0) & (sm.eReg > 0) & (sm.l24 > 0) & (sm.lReg > 0)]
    full = pd.concat(tables, ignore_index=True)
    S1 = sm.set_index("규칙")

    # 견고성 지도: 설정(base, n, kind, g)별 큰 봉 길이 프로파일
    cfgs = []
    for sp in G:
        key = (sp["base"], sp["n"], sp["kind"], sp["g"])
        if key not in cfgs:
            cfgs.append(key)

    def prof(key, col):
        b_, n_, k_, g_ = key
        return [S1.loc[name_of(dict(base=b_, tf=tf, n=n_, kind=k_, g=g_)), col] for tf in TFS]

    def cname(key):
        return f"{BNAME[key[0]]} RSI{key[1]} {key[2]} >{key[3]:g}"

    maps = {}
    stat_rows = []
    for key in cfgs:
        ung = S1.loc[name_of(dict(base=key[0])), :]
        for col in ("e24", "eReg", "l24", "lReg", "e24_8월", "e24손절", "l24손절"):
            maps.setdefault(col, []).append([cname(key)] + prof(key, col))
        for col in ("e24", "eReg", "l24", "lReg"):
            v = np.array(prof(key, col))
            i40 = TFS.index(40)
            spike = v[i40] - (v[i40 - 1] + v[i40 + 1]) / 2
            adj = np.abs(np.diff(v))
            rank40 = int((v > v[i40]).sum()) + 1
            rho = pd.Series(v).corr(pd.Series(np.log(TFS)), method="spearman")
            stat_rows.append(dict(설정=cname(key), 칸=col, 조건없음=ung[col], 최소=v.min(), 최대=v.max(), 범위=v.max() - v.min(),
                                  값40=v[i40], 순위40=rank40, 튐40=spike, 이웃차평균=adj.mean(), 플러스칸=int((v > 0).sum()),
                                  조건없음보다나은칸=int((v > ung[col]).sum()), 길이상관=rho))
    st = pd.DataFrame(stat_rows)
    # early-late 일치: 조건 있는 91개 전체, 그리고 설정별 큰 봉 프로파일 상관
    rho_all24 = gated.e24.corr(gated.l24, method="spearman")
    rho_allReg = gated.eReg.corr(gated.lReg, method="spearman")
    eff = gated.copy()
    eff["base"] = [spec_of[r]["base"] for r in eff.규칙]
    for col in ("e24", "eReg", "l24", "lReg"):
        eff[col + "_효과"] = eff[col] - [S1.loc[name_of(dict(base=b_)), col] for b_ in eff.base]
    prof_rho = []
    for key in cfgs:
        e, l = np.array(prof(key, "e24")), np.array(prof(key, "l24"))
        prof_rho.append((cname(key), pd.Series(e).corr(pd.Series(l), method="spearman")))

    # 최선안 감사
    stop_lab._HTF.clear()
    mk = partial(make_rule, **bsp)
    aud = {(p, reg): stop_audit(mk, p, reg_only=reg) for p in ("early", "late") for reg in (False, True)}
    ung_sp, rev_sp = dict(base=bsp["base"]), rev
    aud_u = {p: stop_audit(partial(make_rule, **ung_sp), p) for p in ("early", "late")}
    aud_r = {p: stop_audit(partial(make_rule, **rev_sp), p) for p in ("early", "late")}
    # 최선안 손절 체결 세션(사용자78 24시간)
    sess = {}
    for p in ("early", "late"):
        t = run(_S_of(p), mk, p, False, syms=U78)
        s = t[t.why == "손절"]
        sess[p] = (s.ses.value_counts().to_dict(), int((s.px > s.ent).sum()))
    runner = [r for r in elig.규칙 if r != pick][:5]
    la_v = {p: htf_value_check(p) for p in ("early", "late")}
    la_r = {p: rule_trunc_check(bsp, p) for p in ("early", "late")}
    nsh = nan_share("early")
    nsh_l = nan_share("late")

    fmt = lambda v: f"{v:+.2f}"
    with open(OUT, "w", encoding="utf-8") as f:
        w = lambda s="": f.write(s + "\n")
        w("칼손절 + 큰 봉 RSI 조건 — 큰 봉 길이·RSI 길이 견고성 지도(g_htfscan) 2026-10-07")
        w("기준(손절 없음, 사용자78): 24시간 early +27.12 · late +17.04, 정규장만 early +1.47 · late +2.28. 아래 '차이'는 모두 이 기준 대비 수익률 합 차이.")
        w(f"시험한 규칙 수: {n_rules} (조건 있는 격자 {len(G)} + 대조 {len(CTRL)} + 최선안 반대 조건 {len(extra)})")
        w(f"사용자78 네 칸(early/late × 24시간/정규장만) 모두 플러스: {len(allpos)}개 -> {', '.join(allpos.규칙) if len(allpos) else '없음'}")
        w("선택 규칙: 조건 있는 규칙 중 early 사용자78 24시간 차이 최대, 보조 조건 early 사용자78 정규장만 차이 >= 0. late는 확인만.")
        w(f"선택: {pick}")
        w(f"조건 nan 처리: 큰 봉 RSI 준비 전이면 base 그대로 판다. early 기간 안 봉 중 nan 비율 dev: " +
          ", ".join(f"{tf}분/{n}={v:.2%}" for (tf, n), v in nsh.items()))
        w(f"  late nan 비율 최대 {max(nsh_l.values()):.3%}")
        w()
        w("== 대조(조건 없음·반대 조건) 와 최선안")
        keep = [name_of(sp) for sp in CTRL] + [name_of(sp) for sp in extra] + [pick]
        show = sm[sm.규칙.isin(keep)].copy()
        for k in show.columns[1:]:
            if show[k].dtype.kind == "f":
                show[k] = show[k].map(fmt)
        w(show.to_string(index=False))
        w()
        w("== 견고성 지도: 행 = 설정, 열 = 큰 봉 길이(분). 값 = 사용자78 차이")
        for col, lab in (("e24", "early 24시간 (선택용)"), ("eReg", "early 정규장만"), ("e24_8월", "early 24시간 8/3 이후만(큰 봉 RSI 수렴 뒤)"),
                         ("l24", "late 24시간 (확인용)"), ("lReg", "late 정규장만 (확인용)"), ("e24손절", "early 24시간 손절 건수"),
                         ("l24손절", "late 24시간 손절 건수")):
            mp = pd.DataFrame(maps[col], columns=["설정"] + [f"{tf}분" for tf in TFS])
            if col.endswith("손절"):
                for tf in TFS:
                    mp[f"{tf}분"] = mp[f"{tf}분"].map(lambda v: f"{int(v)}")
            else:
                for tf in TFS:
                    mp[f"{tf}분"] = mp[f"{tf}분"].map(fmt)
            w(f"-- {lab}")
            w(mp.to_string(index=False))
            w()
        w("조건 없는 같은 손절(참고): " + " | ".join(
            f"{BNAME[b_]} e24 {S1.loc[name_of(dict(base=b_)), 'e24']:+.2f} eReg {S1.loc[name_of(dict(base=b_)), 'eReg']:+.2f} "
            f"l24 {S1.loc[name_of(dict(base=b_)), 'l24']:+.2f} lReg {S1.loc[name_of(dict(base=b_)), 'lReg']:+.2f}" for b_ in ("F3", "F5", "BE")))
        w()
        w("== 40분 특이성 통계 (설정 × 칸). 순위40 = 7개 큰 봉 중 40분 순위(1=최고), 튐40 = 40분 값 - (30분+60분)/2,")
        w("   이웃차평균 = 이웃 큰 봉 사이 차이 절대값 평균, 길이상관 = log(큰 봉 길이)와 값의 스피어먼 상관, 조건없음보다나은칸 = 7개 중")
        st2 = st.copy()
        for k in ("조건없음", "최소", "최대", "범위", "값40", "튐40", "이웃차평균", "길이상관"):
            st2[k] = st2[k].map(fmt)
        w(st2.to_string(index=False))
        w()
        w("== 칸별 요약(조건 있는 91개)")
        for col in ("e24", "eReg", "l24", "lReg"):
            x = st[st.칸 == col]
            w(f"  {col}: 40분이 1위인 설정 {int((x.순위40 == 1).sum())}/{len(x)}, 40분 순위 평균 {x.순위40.mean():.1f}, "
              f"튐40 평균 {x.튐40.mean():+.2f} (이웃차평균 평균 {x.이웃차평균.mean():.2f}), 길이상관 평균 {x.길이상관.mean():+.2f}, "
              f"조건없음보다 나은 칸 {int(x.조건없음보다나은칸.sum())}/{7 * len(x)}")
        w(f"  early-late 순위 상관(91개, 스피어먼): 24시간 {rho_all24:+.2f}, 정규장만 {rho_allReg:+.2f}")
        w("  설정별 큰 봉 프로파일 early24-late24 스피어먼: " + ", ".join(f"{a} {b_:+.2f}" for a, b_ in prof_rho))
        for b_ in ("F3", "F5", "BE"):
            x = eff[eff.base == b_]
            w(f"  조건 효과(조건 있는 - 조건 없는 같은 손절) {BNAME[b_]} {len(x)}개: e24 평균 {x.e24_효과.mean():+.2f} (플러스 {int((x.e24_효과 > 0).sum())}), "
              f"eReg {x.eReg_효과.mean():+.2f} (플러스 {int((x.eReg_효과 > 0).sum())}), l24 {x.l24_효과.mean():+.2f} (플러스 {int((x.l24_효과 > 0).sum())}), "
              f"lReg {x.lReg_효과.mean():+.2f} (플러스 {int((x.lReg_효과 > 0).sum())})")
        w()
        w(f"== 선택 규칙 '{pick}' 전체 표(stop_lab.metrics, compare와 같은 칸)")
        w(full[full.규칙 == pick].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        w()
        w("== stop_audit (사용자78)")
        for (p, reg), a in aud.items():
            w(f"  최선안 {p} {'정규장만' if reg else '24시간'}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in a.items()))
        for p in ("early", "late"):
            w(f"  조건없음 {name_of(ung_sp)} {p} 24시간: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in aud_u[p].items()))
            w(f"  반대조건 {name_of(rev_sp)} {p} 24시간: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in aud_r[p].items()))
        for p, (vc, up) in sess.items():
            w(f"  최선안 손절 매수 세션 {p}: {vc}, 손절인데 매수가보다 비싸게 판 건수 {up}")
        w()
        w("== 차순위(조건 있는 규칙, early 순위, 보조 조건 통과)")
        rr = sm[sm.규칙.isin(runner)].set_index("규칙").loc[runner].reset_index()
        for k in rr.columns[1:]:
            if rr[k].dtype.kind == "f":
                rr[k] = rr[k].map(fmt)
        w(rr.to_string(index=False))
        w()
        w("== 미래 참조 확인")
        for p in ("early", "late"):
            b_, t_, wv = la_v[p]
            w(f"  {p} 큰 봉 RSI 값: 5분봉을 봉 i 시작 시각에서 잘라 다시 계산한 값과 불일치 {b_}/{t_} (최대 차 {wv:.2g})")
            w(f"  {p} 최선안 규칙: 봉 i 가격을 nan으로 지운 배열 규칙과 판단 불일치/IndexError {la_r[p][0]}/{la_r[p][1]}")
        w()
        w("== 전체 요약(모든 규칙, early 24시간 순)")
        sh = sm.sort_values("e24", ascending=False).copy()
        for k in sh.columns[1:]:
            if sh[k].dtype.kind == "f":
                sh[k] = sh[k].map(fmt)
        w(sh.to_string(index=False))
        w()
        w("== 모든 규칙 전체 표")
        w(full.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(open(OUT, encoding="utf-8").read().split("== 전체 요약")[0])
    print(f"{time.time() - t0:.0f}s")

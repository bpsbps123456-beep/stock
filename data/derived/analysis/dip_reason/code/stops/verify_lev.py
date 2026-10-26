"""반박 검증 — '3배 레버리지 ETF는 이격매수 기준을 2%가 아니라 3%로' (2026-10-08).

lev_study.py 코드는 읽지 않고 아래 정의로 독립 구현했다.
  규칙(dip_scalper.pine과 같게): 기준봉 i에서 EMA(종가)[i-1] x (1-이격)에 저가가 닿으면 min(시가, 그 값)에 산다.
  들고 있는 동안 RSI(와일더)[i-1] > 70 이면 봉 i 시가에 판다. 아니면 매수 봉 시각 뒤 처음 오는 한국시간 08:00
  (매수 봉 시각이 그날 08:00 KST 이상이면 다음 날 08:00) 이후 첫 봉 시가에 판다. 판 봉에서는 다시 사지 않는다(Pine else-if).
  한 번에 한 포지션. 전날 정규장 종가 수익률(1d 일봉 종가 c[D-1]/c[D-2]-1)이 +7% 이상인 ET 날짜 D는
  그날 정규장(09:30~16:00 ET) 봉에서 사지 않는다. EMA·RSI는 Pine처럼 첫 n개 단순평균으로 시작한다.
  뒤 = 1분봉 → 3분봉(자정 기준 3분 구간) EMA60·RSI14, 매수일(ET) 9/11~10/7.
  앞 = 5분봉 EMA36·RSI9, 매수일(ET) 7/16~9/8. 시뮬레이션은 데이터 전 구간으로 돌리고 매수일로 거른다.
  범위: 24h = 모든 세션 매수, reg = 정규장 봉에서만 매수(매도는 모든 세션).
점검: 네 칸 재현, 날짜별 합 차이 t값, ETF별 부호(상승형/인버스), 이웃(2.5·3.5%), 최대 기여 ETF·날짜 제외, ETF 상관과 유효 표본.
사용: py stops/verify_lev.py   (결과: ../../results/verify_lev.txt)
"""
import os, io
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = "C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/dip/raw"
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "verify_lev.txt"))

ETFS = ["SOXL", "SOXS", "TQQQ", "SQQQ", "TECL", "LABU", "NAIL", "DPST", "KORU", "SPXL", "TNA"]
BULL = ["SOXL", "TQQQ", "TECL", "LABU", "NAIL", "DPST", "KORU", "SPXL", "TNA"]
INV = ["SOXS", "SQQQ"]
GAPS = [0.02, 0.025, 0.03, 0.035]
PERIODS = {
    "앞": dict(tf="5m", ema=36, rsi=9, d0="2026-07-16", d1="2026-09-08"),
    "뒤": dict(tf="3m", ema=60, rsi=14, d0="2026-09-11", d1="2026-10-07"),
}
SCOPES = ["24h", "reg"]
REF = {  # 연구자 값(전날 +7% 포함): (2% 합, 2% 건수, 3% 합, 3% 건수)
    ("앞", "24h"): (1.36, 332, 1.71, 206), ("앞", "reg"): (0.76, 202, 1.12, 127),
    ("뒤", "24h"): (0.42, 132, 0.27, 64), ("뒤", "reg"): (0.31, 82, 0.38, 44),
}
NS_DAY = 86400 * 10**9


def pine_ema(x, n):
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    a = 2.0 / (n + 1)
    out[n - 1] = x[:n].mean()
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def pine_rsi(x, n):
    out = np.full(len(x), np.nan)
    if len(x) <= n:
        return out
    d = np.diff(x)
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    au, ad = up[:n].mean(), dn[:n].mean()
    def val(u, w):
        return 100.0 if w == 0 else (0.0 if u == 0 else 100 - 100 / (1 + u / w))
    out[n] = val(au, ad)
    for k in range(n, len(d)):
        au = (au * (n - 1) + up[k]) / n
        ad = (ad * (n - 1) + dn[k]) / n
        out[k + 1] = val(au, ad)
    return out


def load_bars(sym, tf):
    if tf == "5m":
        b = pd.read_parquet(os.path.join(RAW, f"5m_{sym}.parquet"))
    else:
        m = pd.read_parquet(os.path.join(RAW, f"1m_{sym}.parquet"))
        b = m.resample("3min", label="left", closed="left", origin="start_day").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    b = b.dropna(subset=["Open", "High", "Low", "Close"])
    b.index = b.index.tz_convert("America/New_York")
    return b


def hot_dates(sym):
    d = pd.read_parquet(os.path.join(RAW, f"1d_{sym}.parquet"))
    c = d["Close"].astype(float)
    c.index = pd.to_datetime(c.index).normalize()
    r = c / c.shift(1) - 1          # r[k] = 날짜 k의 정규장 수익률
    nxt = {}                        # 다음 거래일 D에 r[k]를 붙인다(전날 수익률)
    idx = list(c.index)
    for k in range(1, len(idx)):
        nxt[idx[k].date()] = r.iloc[k - 1]
    # 1d에 없는 날(오늘 등)은 마지막 거래일 수익률을 쓴다
    return nxt, r


def prepare(sym, per):
    p = PERIODS[per]
    b = load_bars(sym, p["tf"])
    cl = b["Close"].values.astype(float)
    t_et = b.index
    mod = t_et.hour * 60 + t_et.minute
    reg = (mod >= 570) & (mod < 960)
    dates = np.array([x.date() for x in t_et])
    nxt, _ = hot_dates(sym)
    prev_ret = np.array([nxt.get(x, np.nan) for x in dates], dtype=float)
    t_ns = np.asarray(t_et.tz_convert("UTC").as_unit("ns").asi8)
    # 매수 봉별 '처음 오는 08:00 KST' (ns, UTC)
    kst = t_et.tz_convert("Asia/Seoul")
    t0 = (kst.normalize() + pd.Timedelta(hours=8)).tz_convert("UTC").as_unit("ns").asi8
    t0 = np.asarray(t0)
    dead = np.where(t_ns >= t0, t0 + NS_DAY, t0)
    return dict(sym=sym, t=t_et, t_ns=t_ns, o=b["Open"].values.astype(float), l=b["Low"].values.astype(float),
                ema=pine_ema(cl, p["ema"]), rsi=pine_rsi(cl, p["rsi"]), reg=reg, dates=dates,
                prev_ret=prev_ret, dead=dead)


def simulate(P, gap, scope, hot=0.07):
    o, l, ema, rsi, reg, t_ns, dead = P["o"], P["l"], P["ema"], P["rsi"], P["reg"], P["t_ns"], P["dead"]
    hotday = (P["prev_ret"] >= hot) if hot is not None else np.zeros(len(o), bool)
    allow = ~(hotday & reg)
    if scope == "reg":
        allow &= reg
    trades = []
    hold = False
    ent = ei = dl = 0
    for i in range(1, len(o)):
        if hold:
            if (rsi[i - 1] > 70) or (t_ns[i] >= dl):
                trades.append((P["sym"], P["t"][ei], P["dates"][ei], bool(reg[ei]), ent, P["t"][i], o[i], o[i] / ent - 1))
                hold = False
            continue
        if allow[i] and not np.isnan(ema[i - 1]):
            lvl = ema[i - 1] * (1 - gap)
            if l[i] <= lvl:
                ent = min(o[i], lvl)
                ei = i
                dl = dead[i]
                hold = True
    open_pos = (P["sym"], P["t"][ei], P["dates"][ei]) if hold else None
    df = pd.DataFrame(trades, columns=["sym", "bt", "bdate", "breg", "ent", "st", "sx", "ret"])
    return df, open_pos


def main():
    buf = io.StringIO()
    def pr(*a):
        print(*a)
        print(*a, file=buf)

    prep = {(per, s): prepare(s, per) for per in PERIODS for s in ETFS}
    T = {}          # (per, scope, gap, hotflag) -> trades df (period filtered)
    opens = []
    for per, p in PERIODS.items():
        d0, d1 = pd.Timestamp(p["d0"]).date(), pd.Timestamp(p["d1"]).date()
        for scope in SCOPES:
            for gap in GAPS:
                for hk, hot in (("hot7", 0.07), ("off", None)):
                    parts = []
                    for s in ETFS:
                        df, op = simulate(prep[(per, s)], gap, scope, hot)
                        if op is not None and d0 <= op[2] <= d1:
                            opens.append((per, scope, gap, hk) + op)
                        parts.append(df[(df.bdate >= d0) & (df.bdate <= d1)])
                    T[(per, scope, gap, hk)] = pd.concat(parts, ignore_index=True)

    pr("=" * 100)
    pr("verify_lev — 3배 ETF 11개 이격 2% vs 3% 반박 검증 (독립 구현, 2026-10-08)")
    pr("합 = 거래 수익률 합(1.00 = +100%p), 전날 +7% 규칙 포함이 기본")
    pr(f"기간 끝에 안 닫힌 포지션(매수일이 기간 안): {len(opens)}건 {opens[:5]}")
    pr("")

    # 1) 재현 + t값
    pr("[1] 네 칸 재현 (전날 +7% 포함) — 내 값 / 연구자 값")
    pr(f"{'칸':<10}{'2% 합':>8}{'건':>5}{'연구':>7}{'건':>5}{'3% 합':>9}{'건':>5}{'연구':>7}{'건':>5}"
       f"{'3-2':>8}{'날짜n':>7}{'t(활동일)':>10}{'t(전거래일)':>11}{'P(boot>0)':>10}")
    cells = {}
    rng = np.random.default_rng(7)
    for per, p in PERIODS.items():
        d0, d1 = pd.Timestamp(p["d0"]), pd.Timestamp(p["d1"])
        days_all = [x.date() for x in pd.read_parquet(os.path.join(RAW, "1d_SOXL.parquet")).index
                    if d0 <= pd.Timestamp(x) <= d1]
        for scope in SCOPES:
            a, b = T[(per, scope, 0.02, "hot7")], T[(per, scope, 0.03, "hot7")]
            sa, sb = a.groupby("bdate").ret.sum(), b.groupby("bdate").ret.sum()
            dd = sb.sub(sa, fill_value=0.0)
            n = len(dd)
            tval = dd.mean() / (dd.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            dall = dd.reindex(days_all, fill_value=0.0)
            tall = dall.mean() / (dall.std(ddof=1) / np.sqrt(len(dall)))
            vals = dd.values
            boots = np.array([vals[rng.integers(0, n, n)].sum() for _ in range(5000)])
            pboot = (boots > 0).mean()
            r = REF[(per, scope)]
            cells[(per, scope)] = dict(sum2=a.ret.sum(), n2=len(a), sum3=b.ret.sum(), n3=len(b), t=tval, tall=tall,
                                       ndays=n, pboot=pboot, dd=dd)
            c = cells[(per, scope)]
            pr(f"{per + ' ' + scope:<10}{c['sum2']:>+8.2f}{c['n2']:>5}{r[0]:>+7.2f}{r[1]:>5}{c['sum3']:>+9.2f}{c['n3']:>5}"
               f"{r[2]:>+7.2f}{r[3]:>5}{c['sum3'] - c['sum2']:>+8.2f}{n:>7}{tval:>10.2f}{tall:>11.2f}{pboot:>10.2f}")
    pr("  t(활동일) = 둘 중 하나라도 거래가 있는 날짜의 (3% 합 - 2% 합) 평균/표준오차, t(전거래일) = 거래 없는 날 0 포함")
    pr("  P(boot>0) = 날짜 단위로 5000번 다시 뽑아 3%-2% 합이 0보다 큰 비율")
    pr("")
    pr("  참고: 전날 급등 규칙 끔(연구자 표 '끔' 열) 재현")
    REF_OFF = {("앞", "24h"): (1.47, 1.40, 1.76, 1.88), ("앞", "reg"): (0.47, 0.49, 0.80, 0.85),
               ("뒤", "24h"): (0.31, 0.09, 0.19, 0.24), ("뒤", "reg"): (0.25, 0.39, 0.34, 0.38)}
    for (per, scope), ref in REF_OFF.items():
        mine = [T[(per, scope, g, "off")].ret.sum() for g in GAPS]
        pr(f"  {per} {scope:<4} 내 " + " ".join(f"{g*100:.1f}%:{v:+.2f}" for g, v in zip(GAPS, mine))
           + " | 연구 " + " ".join(f"{v:+.2f}" for v in ref))
    pr("")

    # 2) ETF별 부호
    pr("[2] ETF별 3%-2% 합 (전날 +7% 포함)")
    hdr = "".join(f"{s:>7}" for s in ETFS)
    pr(f"{'칸':<10}{hdr}{'  +/-/0':>9}{'상승형':>8}{'인버스':>8}")
    per_etf = {}
    for per in PERIODS:
        for scope in SCOPES:
            a, b = T[(per, scope, 0.02, "hot7")], T[(per, scope, 0.03, "hot7")]
            de = b.groupby("sym").ret.sum().reindex(ETFS, fill_value=0.0) - a.groupby("sym").ret.sum().reindex(ETFS, fill_value=0.0)
            per_etf[(per, scope)] = de
            npos, nneg = int((de > 0.005).sum()), int((de < -0.005).sum())
            nz = len(ETFS) - npos - nneg
            pr(f"{per + ' ' + scope:<10}" + "".join(f"{v:>+7.2f}" for v in de.values)
               + f"{f'{npos}/{nneg}/{nz}':>9}{de[BULL].sum():>+8.2f}{de[INV].sum():>+8.2f}")
    pr("  (±0.005 이내는 0으로 셈)")
    pr("")
    pr("  상승형 vs 인버스 2%·3% 합")
    for per in PERIODS:
        for scope in SCOPES:
            a, b = T[(per, scope, 0.02, "hot7")], T[(per, scope, 0.03, "hot7")]
            pr(f"  {per} {scope:<4} 상승형 2% {a[a.sym.isin(BULL)].ret.sum():+.2f}({(a.sym.isin(BULL)).sum()}건) "
               f"3% {b[b.sym.isin(BULL)].ret.sum():+.2f}({(b.sym.isin(BULL)).sum()}건) | "
               f"인버스 2% {a[a.sym.isin(INV)].ret.sum():+.2f}({(a.sym.isin(INV)).sum()}건) "
               f"3% {b[b.sym.isin(INV)].ret.sum():+.2f}({(b.sym.isin(INV)).sum()}건)")
    pr("")

    # 3) 이웃
    pr("[3] 이웃 이격 합 (전날 +7% 포함), 2% 대비 차")
    for per in PERIODS:
        for scope in SCOPES:
            s = {g: T[(per, scope, g, "hot7")].ret.sum() for g in GAPS}
            n = {g: len(T[(per, scope, g, "hot7")]) for g in GAPS}
            pr(f"  {per} {scope:<4} " + "  ".join(f"{g*100:.1f}%:{s[g]:+.2f}({n[g]}건, {s[g]-s[0.02]:+.2f})" for g in GAPS))
    pr("")

    # 4) 집중도
    pr("[4] 이득 집중도: 최대 기여 ETF·날짜를 빼고 3%-2% 다시")
    conc = {}
    for per in PERIODS:
        for scope in SCOPES:
            a, b = T[(per, scope, 0.02, "hot7")], T[(per, scope, 0.03, "hot7")]
            tot = b.ret.sum() - a.ret.sum()
            de = per_etf[(per, scope)]
            dd = cells[(per, scope)]["dd"]
            if tot >= 0:
                te, td = de.idxmax(), dd.idxmax()
            else:
                te, td = de.idxmin(), dd.idxmin()
            wo_e = tot - de[te]
            wo_d = tot - dd[td]
            a2 = a[(a.sym != te) & (a.bdate != td)]
            b2 = b[(b.sym != te) & (b.bdate != td)]
            wo_both = b2.ret.sum() - a2.ret.sum()
            # 상위 2일 제외
            top2 = (dd.sort_values(ascending=(tot < 0)).index[:2])
            wo_d2 = tot - dd[top2].sum()
            conc[(per, scope)] = dict(tot=tot, te=te, de=de[te], wo_e=wo_e, td=td, dv=dd[td], wo_d=wo_d, wo_both=wo_both, wo_d2=wo_d2)
            pr(f"  {per} {scope:<4} 3-2 {tot:+.2f} | 최대 ETF {te} {de[te]:+.2f} → 빼면 {wo_e:+.2f} | "
               f"최대 날짜 {td} {dd[td]:+.2f} → 빼면 {wo_d:+.2f} | 둘 다 빼면 {wo_both:+.2f} | 상위 2일 빼면 {wo_d2:+.2f}")
    pr("")

    # 5) 표본·상관
    pr("[5] 표본 크기와 ETF 상관")
    for per, p in PERIODS.items():
        d0, d1 = pd.Timestamp(p["d0"]), pd.Timestamp(p["d1"])
        rets = {}
        for s in ETFS:
            _, r = hot_dates(s)
            rets[s] = r[(r.index >= d0) & (r.index <= d1)]
        R = pd.DataFrame(rets).dropna()
        C = R.corr().values
        off = C[~np.eye(len(ETFS), dtype=bool)]
        rho_abs = np.abs(off).mean()
        neff = len(ETFS) / (1 + (len(ETFS) - 1) * rho_abs)
        lam = np.linalg.eigvalsh(C)
        npr = lam.sum() ** 2 / (lam ** 2).sum()      # 고유값 참여비(인버스는 같은 축으로 셈)
        bull_c = R[BULL].corr().values
        bull_off = bull_c[~np.eye(len(BULL), dtype=bool)].mean()
        pr(f"  {per}: 일간 수익률 {len(R)}일, 11개 쌍 평균 |상관| {rho_abs:.2f}, 상승형끼리 평균 상관 {bull_off:.2f}, "
           f"SOXL-SOXS {R['SOXL'].corr(R['SOXS']):+.2f}, TQQQ-SQQQ {R['TQQQ'].corr(R['SQQQ']):+.2f}, "
           f"SOXL-TECL {R['SOXL'].corr(R['TECL']):+.2f} → 유효 독립 종목 수 ≈ {neff:.1f}개(평균상관식), "
           f"{npr:.1f}개(고유값 참여비)")
        for scope in SCOPES:
            c = cells[(per, scope)]
            dd = c["dd"]
            # 날짜별 3%-2% 차이의 ETF 간 동시성: 같은 날 ETF별 차이 부호가 같은 비율
            pr(f"    {scope}: 차이가 나는 날 {c['ndays']}일(0 아닌 날 {(dd.abs() > 1e-9).sum()}일), "
               f"t(활동일) {c['t']:.2f}, 날짜 부트스트랩 P(3%>2%) {c['pboot']:.2f}")
    pr("")

    # 6) 최대 기여 날짜 안쪽: 어떤 거래가 차이를 만들었나 + 가장 큰 단일 거래
    pr("[6] 최대 기여 날짜의 거래 내역과 2%에만 있는 큰 손실 거래")
    for per in PERIODS:
        for scope in SCOPES:
            a, b = T[(per, scope, 0.02, "hot7")], T[(per, scope, 0.03, "hot7")]
            td = conc[(per, scope)]["td"]
            for g, x in ((0.02, a), (0.03, b)):
                y = x[x.bdate == td]
                pr(f"  {per} {scope:<4} {td} {g*100:.0f}%: 합 {y.ret.sum():+.3f} ({len(y)}건) "
                   + ", ".join(f"{r.sym} {r.bt.strftime('%H:%M')}→{r.st.strftime('%m/%d %H:%M')} {r.ret*100:+.1f}%"
                               for r in y.itertuples()))
            # 2%에서 -10% 넘게 잃은 거래 중 3%에 같은 매수 시각이 없는 것
            k3 = set(zip(b.sym, b.bt))
            big = a[(a.ret.values < -0.10) & np.array([(s, t) not in k3 for s, t in zip(a.sym, a.bt)], bool)]
            pr(f"    2%에만 있는 -10% 넘는 손실: " + (", ".join(
                f"{r.sym} {r.bt.strftime('%m/%d %H:%M')} {r.ret*100:+.1f}%" for r in big.itertuples()) or "없음"))
            k2 = set(zip(a.sym, a.bt))
            big3 = b[(b.ret.values < -0.10) & np.array([(s, t) not in k2 for s, t in zip(b.sym, b.bt)], bool)]
            pr(f"    3%에만 있는 -10% 넘는 손실: " + (", ".join(
                f"{r.sym} {r.bt.strftime('%m/%d %H:%M')} {r.ret*100:+.1f}%" for r in big3.itertuples()) or "없음"))
    pr("")

    # 판정
    worse =[k for k, c in cells.items() if c["sum3"] < c["sum2"]]
    rep_bad = [k for k, c in cells.items()
               if abs(c["sum2"] - REF[k][0]) > 0.1 or abs(c["sum3"] - REF[k][2]) > 0.1]
    conc_bad = [k for k, v in conc.items() if v["tot"] > 0 and (v["wo_e"] <= 0 or v["wo_d"] <= 0)]
    pr("[판정 재료]")
    pr(f"  재현 0.1 넘게 차이: {rep_bad}")
    pr(f"  3%가 2%보다 나쁜 칸: {worse} ({len(worse)}/4)")
    pr(f"  3%가 나은 칸 중 최대 ETF 또는 최대 날짜 하나를 빼면 0 이하가 되는 칸: {conc_bad}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print("saved", OUT)


if __name__ == "__main__":
    main()

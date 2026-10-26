"""10/9 사례 정리(2026-10-10 사용자 질문: CRCL 신호 뒤 빠른 반등 vs MRVL 계속 하락, 광통신 급등 뒤 되돌림에서 LITE 회복 vs COHR 못 함).

이 파일은 규칙을 만들지 않는다. 2종목 사례로 문턱을 고르지 않고, 앞선 검증(pat_levels·pat_shape·pat_peers·pat_spike와 verify_*)에서
관찰 후보로 남은 규칙을 정의·문턱 그대로 10/9에 적용해 '미리 맞혔을까'만 본다. 살아남은(확인된) 규칙은 0개다.

결과 정의(측정 전에 고정):
  거래(이격매수): Y1 = 지금 매도 규칙 수익(RSI14>70 다음 봉 시가 / 한국 08시). 11:33 ET까지 안 팔렸으면 11:33 1분봉 종가 평가(미확정).
                  Y2 = 매수 뒤 20봉(60분) 안 RSI 매도.
  급등 되돌림(사건): P = 전날 정규장 종가(일봉), H = 정규장 고점, L = H 뒤 최저가(11:33까지).
                  반납 = (H-L)/(H-P). 회복 = 11:33 가격 >= P + 0.5(H-P)(pat_spike Y2를 종가 대신 11:33로, 미확정).
                  되찾음 = (11:33 가격 - L)/(H - L).
데이터: raw_today 1분봉(10/9 11:33 ET까지, 이후 없음 → 종가·최종 결과 미확정), 동종 이격(ptouch)은 raw_1009(10:18까지, 09:51 전 값만 씀).
특성은 매수 봉 i 시가 시점 값(봉 i-1까지, 지난 날, 지표 선, 체결가)만 쓴다.
출력: ../results/case_1009.txt
사용: py case_1009.py"""
import os, sys
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../../dip_reason/code"))
from events import bars, prep, tod_rvol
from rules import hot_days
import build
import pat_levels as PL
import pat_shape as PS
import pat_peers as PP
import pat_spike as SPK

SP = build.SP
RAWT = os.path.join(SP, "raw_today")
RAW9 = os.path.join(SP, "raw_1009")
RES = os.path.join(HERE, "../results")
DAY = date(2026, 10, 9)
OPT = ["LITE", "COHR", "CIEN", "AAOI", "FN"]
OPT_ALL = OPT + ["VIAV", "IPGP"]          # raw_today에 있는 '광 전송·광모듈' 소분류 종목
CASE = ["CRCL", "MRVL", "LITE", "COHR", "CIEN", "AAOI", "FN"]
LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LINES.append(s)
    print(s)


def pct(x, nd=2):
    return "  n/a" if x is None or not np.isfinite(x) else f"{100 * x:+.{nd}f}%"


def hm(t):
    return "--:--" if t is None else t.strftime("%H:%M")


def load3(s):
    return prep(bars(RAWT, s, 3), 60, 14)


def prev_close(s):
    dd = pd.read_parquet(f"{RAWT}/1d_{s}.parquet")
    dd.index = pd.to_datetime(dd.index).date
    return float(dd.Close.shift(1).get(DAY))


def vprof3(d, ent=None, n=5):
    """직전 n개 정규장 3분봉 거래량 분포(봉 거래량을 저가~고가에 고르게, 0.5% 로그 칸). POC·70% 가치구간·ent 밀도."""
    reg = d[(d.ses == "정규") & (d.date < DAY)]
    days = sorted(reg.date.unique())[-n:]
    x = reg[reg.date.isin(days)]
    lg = np.log(1.005)
    acc = {}
    for lo, hi, v in zip(x.l.values, x.h.values, x.v.values):
        if v <= 0:
            continue
        b0, b1 = int(np.floor(np.log(lo) / lg)), int(np.floor(np.log(hi) / lg))
        if b1 == b0 or hi <= lo:
            acc[b0] = acc.get(b0, 0) + v
            continue
        for b in range(b0, b1 + 1):
            e0, e1 = np.exp(b * lg), np.exp((b + 1) * lg)
            w = (min(hi, e1) - max(lo, e0)) / (hi - lo)
            if w > 0:
                acc[b] = acc.get(b, 0) + v * w
    s = pd.Series(acc).sort_index()
    s = s.reindex(range(s.index.min(), s.index.max() + 1), fill_value=0.0)
    k = int(s.idxmax())
    tot, inc, lo_b, hi_b = s.sum(), s[k], k, k
    while inc < 0.7 * tot:
        dn = s.get(lo_b - 1, -1)
        up = s.get(hi_b + 1, -1)
        if up >= dn:
            hi_b += 1; inc += up
        else:
            lo_b -= 1; inc += dn
    mid = lambda b: np.exp((b + 0.5) * lg)
    out = dict(poc=mid(k), val=np.exp(lo_b * lg), vah=np.exp((hi_b + 1) * lg), days=f"{days[0]:%m/%d}~{days[-1]:%m/%d}")
    if ent is not None:
        out["dens"], _ = PL.vprof(x.l.values, x.h.values, x.v.values, ent)
    return out


def where_va(p, vp):
    if p < vp["val"]:
        return f"가치구간 아래({pct(p / vp['val'] - 1, 1)})"
    if p > vp["vah"]:
        return f"가치구간 위({pct(p / vp['vah'] - 1, 1)})"
    return "가치구간 안"


# ---------------- 1. 광통신 경로 ----------------
def optical_paths(D):
    rows = []
    for s in OPT_ALL:
        d, m = D[s], pd.read_parquet(f"{RAWT}/1m_{s}.parquet")
        P = prev_close(s)
        x = d[d.date == DAY]
        pre, reg = x[x.ses == "프리"], x[x.ses == "정규"]
        m9 = m[m.index.date == DAY]
        mpre = m9[(m9.index.hour * 60 + m9.index.minute) < 570]
        ih = int(np.argmax(reg.h.values))
        H, tH = reg.h.values[ih], reg.index[ih]
        after = reg.iloc[ih:]
        il = int(np.argmin(after.l.values))
        L, tL = after.l.values[il], after.index[il]
        now, tnow = m9.Close.values[-1], m9.index[-1]
        vp = vprof3(d, L)
        has = len(mpre) > 0
        rows.append(dict(sym=s, P=P, pre_hi=mpre.High.max() / P - 1 if has else np.nan, t_pre_hi=mpre.High.idxmax() if has else None,
                         pre_lo=mpre.Low.min() if has else np.nan,
                         open=reg.o.values[0], H=H, tH=tH, L=L, tL=tL, now=now, tnow=tnow,
                         gb=(H - L) / (H - P), gb_min=(tL - tH).total_seconds() / 60,
                         back=(now - L) / (H - L), rec=float(now >= P + 0.5 * (H - P)), vp=vp, iL=reg.index.get_loc(tL),
                         reg=reg))
    return rows


def peer_rel(R, s, t):
    """시각 t 봉(이미 끝난 봉) 기준 자기 vs 동종(광 7종 중 나머지) 중앙값: 전날 대비, 반납 비율."""
    def at(r):
        reg = r["reg"]
        k = reg.index.searchsorted(t, side="right") - 1
        if k < 0:
            return np.nan, np.nan
        c, H = reg.c.values[k], reg.h.values[: k + 1].max()
        return c / r["P"] - 1, (H - c) / (H - r["P"]) if H > r["P"] else np.nan
    own = at(R[s])
    pe = [at(R[q]) for q in R if q != s]
    return own[0] - np.nanmedian([p[0] for p in pe]), own[1] - np.nanmedian([p[1] for p in pe]), own


def part1(D):
    rows = optical_paths(D)
    R = {r["sym"]: r for r in rows}
    say("1. 광통신 10/9 3분봉 경로(전날 종가 P 대비, 데이터 11:33 ET까지 = 장중, 종가·결과 미확정)")
    say(f"{'종목':5} {'P':>8} {'프리고점':>8}(시각) {'시가':>7} {'정규고점':>8}(시각) {'저점':>7}(시각) {'11:33':>7} {'반납':>5} {'고점→저점':>6} {'반납속도':>9} {'되찾음':>6} {'회복':>4}")
    for r in rows:
        P = r["P"]
        spd = (r["H"] - r["L"]) / r["H"] / max(r["gb_min"], 1) * 30
        say(f"{r['sym']:5} {P:8.2f} {pct(r['pre_hi']):>8}({hm(r['t_pre_hi'])}) {pct(r['open'] / P - 1):>7} {pct(r['H'] / P - 1):>8}({hm(r['tH'])}) "
            f"{pct(r['L'] / P - 1):>7}({hm(r['tL'])}) {pct(r['now'] / P - 1):>7} {r['gb']:5.0%} {r['gb_min']:5.0f}분 {pct(-spd):>7}/30분 {r['back']:6.0%} {'예' if r['rec'] else '아니오':>4}")
    say("  반납 = (H-L)/(H-P). 반납속도 = 고점→저점 하락률을 30분당으로. 되찾음 = (11:33-L)/(H-L). 회복 = 11:33 >= P+0.5(H-P).")
    say("  VIAV·IPGP는 같은 소분류(광 전송·광모듈) 참고용.")
    say("")
    say("  3분봉 종가 경로(P 대비): 09:30 시가 / 09:33 / 09:39 / 09:45 / 10:00 / 10:30 / 11:00 / 11:30 봉")
    for r in rows:
        reg = r["reg"]
        pts = [reg.o.values[0]] + [reg.c.get(pd.Timestamp(f"2026-10-09 {x}", tz="America/New_York"), np.nan)
                                    for x in ("09:33", "09:39", "09:45", "10:00", "10:30", "11:00", "11:30")]
        say(f"  {r['sym']:5} " + " ".join(f"{pct(p / r['P'] - 1, 1):>7}" for p in pts))
    say("")
    say("  저점 위치(저점 L 기준) — 5일 3분봉 거래 분포는 직전 5개 정규장(" + rows[0]["vp"]["days"] + "), 0.5% 칸, 가치구간 = 거래 70%")
    say(f"  {'종목':5} {'L/P':>7} {'L/프리저가':>9} {'L/시가':>7} {'L/POC':>7} {'가치구간':>8}~{'':<8} {'L 위치':<20} {'L 밀도':>6} | {'H/POC':>7} {'H 위치':<18}")
    for r in rows:
        vp = r["vp"]
        say(f"  {r['sym']:5} {pct(r['L'] / r['P'] - 1):>7} {pct(r['L'] / r['pre_lo'] - 1):>9} {pct(r['L'] / r['open'] - 1):>7} "
            f"{pct(r['L'] / vp['poc'] - 1):>7} {vp['val']:8.2f}~{vp['vah']:<8.2f} {where_va(r['L'], vp):<20} {vp['dens']:6.2f} | POC {vp['poc']:.2f} "
            f"{pct(r['H'] / vp['poc'] - 1):>7} {where_va(r['H'], vp):<18}")
    say("  L 밀도 = L ±1% 안 거래 몫 / 고르게 퍼졌을 때 기대 몫(1보다 크면 거래가 쌓인 가격대). H 위치 = 그날 고점이 5일 가치구간 어디였나(사후 설명용).")
    say("")
    say("  저점 직전 봉 종가 기준 동종(광 7종 중 나머지 6) 대비 — 상대 상승 = 자기 P 대비 - 동종 중앙값, 상대 반납 = 자기 반납 비율 - 동종 중앙값")
    for r in rows[:5]:
        tb = r["reg"].index[max(r["iL"] - 1, 0)]
        rel_up, rel_gb, own = peer_rel(R, r["sym"], tb)
        say(f"  {r['sym']:5} {hm(tb)} 봉: 자기 {pct(own[0])}, 반납 {own[1]:.0%} | 상대 상승 {pct(rel_up)}, 상대 반납 {100 * rel_gb:+.0f}%p")
    t945 = pd.Timestamp("2026-10-09 09:42", tz="America/New_York")
    say("  같은 시각(09:42 봉 종가, COHR 반납 70% 첫 봉) 기준:")
    for r in rows[:5]:
        rel_up, rel_gb, own = peer_rel(R, r["sym"], t945)
        say(f"  {r['sym']:5} 자기 {pct(own[0])}, 반납 {own[1]:.0%} | 상대 상승 {pct(rel_up)}, 상대 반납 {100 * rel_gb:+.0f}%p")
    say("")
    return R


# ---------------- 2. 10/9 이격 신호(거래) ----------------
def day_trades(D):
    out = []
    for s in CASE:
        d = D[s]
        kst = d.index.tz_convert("Asia/Seoul")
        t8 = kst.normalize() + pd.Timedelta(hours=8)
        dead = np.where(kst < t8, t8.as_unit("ns").asi8, (t8 + pd.Timedelta(days=1)).as_unit("ns").asi8)
        hot = hot_days(RAWT, s, d, 0.07)
        gp = 0.03 if s in build.LEV else 0.02
        held = []
        o, l, r, ema, tk = d.o.values, d.l.values, d.r.values, d.ema.values, d.index.as_unit("ns").asi8
        hold, ent, ei, dl = False, 0.0, 0, 0
        for i in range(61, len(d)):
            if hold:
                why = "RSI" if (i - 1 >= ei and r[i - 1] > 70) else ("8시" if tk[i] >= dl else None)
                if why:
                    held.append((ei, i, ent, o[i], why)); hold = False
            elif not hot[i]:
                lvl = ema[i - 1] * (1 - gp)
                if l[i] <= lvl:
                    ent, ei, hold, dl = min(o[i], lvl), i, True, dead[i]
        if hold:
            held.append((ei, None, ent, np.nan, "보유중"))
        for ei, xi, ent, px, why in held:
            if d.date.iloc[ei] != DAY:
                continue
            out.append(dict(sym=s, i=ei, t=d.index[ei], ses=d.ses.iloc[ei], ent=ent, xi=xi, px=px, why=why))
    return out


def part2(D, R):
    T = day_trades(D)
    say("2. 10/9 이격 신호(3분봉 EMA60[1]x0.98 첫 닿음, 지금 매도 규칙 그대로 재현)")
    m1 = {s: pd.read_parquet(f"{RAWT}/1m_{s}.parquet") for s in CASE}
    rows = []
    for x in T:
        s, d, i = x["sym"], D[x["sym"]], x["i"]
        now = m1[s].Close.values[-1]
        if x["xi"] is not None:
            res = f"{x['xi'] - i}봉 뒤 {x['why']} 매도 {hm(d.index[x['xi']])} {pct(x['px'] / x['ent'] - 1)}"
            y1, y2 = x["px"] / x["ent"] - 1, float(x["why"] == "RSI" and x["xi"] - i <= 20)
        else:
            res = f"11:33까지 미매도, 평가 {pct(now / x['ent'] - 1)}"
            y1 = now / x["ent"] - 1
            y2 = 0.0 if len(d) - 1 - i >= 20 else np.nan
        low_after = d.l.values[i:].min() / x["ent"] - 1
        rows.append(dict(x, res=res, y1=y1, y2=y2, mae=low_after))
        say(f"  {s:5} {hm(x['t'])} {x['ses']} 매수 {x['ent']:.2f} → {res} | 매수 뒤 최저 {pct(low_after)}")
    say("")
    first = {}
    for r in rows:
        if r["ses"] == "정규" and r["sym"] not in first:
            first[r["sym"]] = r
    # 특성 계산
    st = {}
    F = {}
    for s, r in first.items():
        d, i, ent = D[s], r["i"], r["ent"]
        o, h, l, c, rr, ema = (d[k].values.astype(float) for k in ("o", "h", "l", "c", "r", "ema"))
        day, ses, mod = np.asarray(d.date), d.ses.values, d["mod"].values
        rv5 = tod_rvol(d, 5, 10)
        prev, tc = PS.prev_close_map(day, ses, c)
        f = PS.feat_at(o, h, l, c, rr, ema, day, ses, mod, rv5, i, ent, 0.03 if s in build.LEV else 0.02, prev, tc)
        st[s] = PL.Struct(RAWT, s)
        T_ = PL.today_from_prep(d, 5)
        f.update(PL.level_feats(st[s], T_, i, ent, ses[i]))
        vp = vprof3(d, ent)
        f["poc3"] = ent / vp["poc"] - 1
        f["va3"] = where_va(ent, vp)
        f["dens3"] = vp["dens"]
        F[s] = f
    # 동종 특성(pat_peers 그대로, raw_1009 10:18까지 — 09:51 전 값만 씀)
    have = sorted(f[3:-8] for f in os.listdir(RAW9) if f.startswith("1m_"))
    P, lvl, _ = PP.peer_map(have)
    need = sorted(set(first) | {q for s in first for q in P.get(s, [])} | {"SMH", "QQQ"})
    Dp, Tp, DLp, G, grid = PP.build_all(RAW9, 3, 60, 14, need)
    for s, r in first.items():
        pos = Dp[s].index.get_loc(r["t"])
        pf = PP.feats_at(s, pos, Dp, Tp, DLp, G, grid, P)
        F[s].update({k: pf.get(k, np.nan) for k in ("ptouch", "rel_hi", "rel_give", "phi", "lead5", "lead20", "pdev", "rel_dev", "sdev_q")})
        F[s]["n_peer"] = len(P.get(s, []))
        F[s]["peer_lvl"] = lvl.get(s)
    say("  첫 정규장 신호 순간 특성(봉 i 시가 시점)")
    cols = [("t", "매수"), ("tmin", "개장뒤분"), ("d_prev", "P대비"), ("gap", "개장갭"), ("spike", "그날고점"), ("retr", "되돌림"),
            ("kday", "고점뒤봉"), ("rsi1", "RSI(i-1)"), ("ndown", "연속음봉"), ("big1", "최대1봉"), ("vol5", "거래량배수"), ("o_line", "시가/선")]
    say("  " + f"{'종목':5} " + " ".join(f"{n:>8}" for _, n in cols))
    for s in first:
        f = F[s]
        vals = []
        for k, _ in cols:
            v = first[s]["t"] if k == "t" else f.get(k, np.nan)
            if k == "t":
                vals.append(f"{hm(v):>8}")
            elif k in ("tmin", "kday", "ndown"):
                vals.append(f"{v:8.0f}")
            elif k in ("rsi1", "vol5", "retr"):
                vals.append(f"{v:8.2f}")
            else:
                vals.append(f"{pct(v):>8}")
        say(f"  {s:5} " + " ".join(vals))
    cols2 = [("pos20", "20일위치"), ("poc5", "체결/POC5"), ("vp5r", "5일밀도"), ("vp20r", "20일밀도"), ("sw_b", "아래스윙"), ("sw_n", "스윙수"),
             ("pml", "/프리저가"), ("op", "/시가"), ("vw", "/VWAP"), ("h5", "/5일고가"), ("poc3", "/POC3분")]
    say("  " + f"{'종목':5} " + " ".join(f"{n:>9}" for _, n in cols2) + "  3분 가치구간")
    for s in first:
        f = F[s]
        vals = []
        for k, _ in cols2:
            v = f.get(k, np.nan)
            vals.append(f"{v:9.2f}" if k in ("pos20", "vp5r", "vp20r") else (f"{v:9.0f}" if k == "sw_n" else f"{pct(v):>9}"))
        say(f"  {s:5} " + " ".join(vals) + f"  {f['va3']}")
    cols3 = [("ptouch", "동종닿음"), ("pdev", "동종이격"), ("rel_dev", "상대이격"), ("phi", "동종고점"), ("rel_hi", "상대고점"),
             ("rel_give", "상대반납"), ("lead5", "5일상대"), ("lead20", "20일상대")]
    say("  " + f"{'종목':5} " + " ".join(f"{n:>8}" for _, n in cols3) + "  동종")
    for s in first:
        f = F[s]
        vals = [f"{f.get(k, np.nan):8.3f}" if k == "ptouch" else f"{pct(f.get(k, np.nan)):>8}" for k, _ in cols3]
        say(f"  {s:5} " + " ".join(vals) + f"  {f['peer_lvl']} {f['n_peer']}개")
    say("")
    return first, F


# ---------------- 2b. 급등 되돌림 사건(pat_spike V0·V2 그대로, 11:33까지) ----------------
def spike_events(D):
    paths = {DAY: {}}
    for s in OPT_ALL:
        d = D[s]
        f = SPK.daily(s)
        x = d[d.date == DAY]
        reg, pre = x[x.ses == "정규"], x[x.ses == "프리"]
        fr = f.loc[DAY]
        paths[DAY][s] = dict(t=reg.index, o=reg.o.values, h=reg.h.values, l=reg.l.values, c=reg.c.values, v=reg.v.values,
                             runH=np.maximum.accumulate(reg.h.values), P=fr.P, close=np.nan, pml=pre.c.min(), mins=3,
                             **{k: fr[k] for k in ("prev", "tr5", "hi120", "atr20", "vol20", "adv20")})
    ev = {dn: SPK.find_events(paths, "10/9", dn) for dn in ("V0", "V2")}
    return ev, paths


def part2b(D, R):
    ev, paths = spike_events(D)
    say("2b. 급등 되돌림 사건(pat_spike 정의 그대로: 급등 +6%, 반납 70%(V0)·50%(V2) 첫 봉 T, 11:33까지)")
    for dn, e in ev.items():
        for _, r in e.iterrows():
            p = paths[DAY][r.sym]
            k = int(np.searchsorted(p["t"].values, r["T"].to_datetime64()))
            now = R[r.sym]["now"]
            say(f"  {dn} {r.sym:5} T {hm(r['T'])} 급등 {pct(r.spike)} 반납 {r.gbT:.0%} big15 {r.big15:.2f} 고점→T {r.gb_min:.0f}분 "
                f"prev {pct(r.prev)} hi120 {pct(r.hi120)} | T 뒤 11:33 {pct(now / r.cT - 1)}, 회복 {'예' if now >= r.P + 0.5 * (r.H - r.P) else '아니오'}")
        got = set(e.sym)
        miss = [s for s in OPT_ALL if s not in got]
        say(f"  {dn} 사건 없음: {', '.join(miss)}")
    say("")
    return ev


# ---------------- 3. 규칙 적용 ----------------
def judge(a_good, b_bad, good_side):
    """a = 실제 좋은 쪽, b = 실제 나쁜 쪽. good_side(x) -> True(좋음)/False(나쁨)."""
    ga, gb = good_side(a_good), good_side(b_bad)
    if ga is None or gb is None:
        return "판단 불가"
    if ga == gb:
        return "못 가름"
    return "맞음(사후 일치)" if ga and not gb else "반대"


def part3(first, F, ev, R, trades_ok):
    say("3. 관찰 후보 규칙을 문턱 그대로 10/9에 적용(확인된 규칙 0개 — 아래 '맞음'은 모두 사후 일치이고 매매 근거 아님)")
    say("   실제(11:33 기준): CRCL = 좋음(13봉 뒤 RSI 매도), MRVL = 나쁨. 사건: LITE = 회복, COHR = 회복 못 함(둘 다 미확정).")
    e0 = ev["V0"].set_index("sym") if len(ev["V0"]) else pd.DataFrame()
    e2 = ev["V2"].set_index("sym") if len(ev["V2"]) else pd.DataFrame()

    def fv(s, k):
        return F.get(s, {}).get(k, np.nan)

    rules = [
        ("pos20>0.59 → Y1 나쁨", "levels 관찰(약)", "pos20", lambda v: None if not np.isfinite(v) else not (v > 0.59)),
        ("poc5<0 → Y1 좋음", "levels 관찰(뒤 절반서 고름)", "poc5", lambda v: None if not np.isfinite(v) else (True if v < 0 else False)),
        ("kday<=15 → Y2 드묾", "shape 관찰(지표 기억)", "kday", lambda v: None if not np.isfinite(v) else not (v <= 15)),
        ("ptouch>0.25 → Y2 드묾", "peers 추적만", "ptouch", lambda v: None if not np.isfinite(v) else not (v > 0.25)),
        ("gap>1.768% → Y1 나쁨", "shape 뒤 절반만(장세 메모)", "gap", lambda v: None if not np.isfinite(v) else not (v > 0.01768)),
    ]
    say(f"  {'규칙':22} {'지위':24} {'CRCL':>8} {'MRVL':>8} {'CRCL/MRVL':>12} {'LITE':>8} {'COHR':>8} {'LITE/COHR':>12}")
    tally = {}
    for name, stat, k, gs in rules:
        vals = {s: fv(s, k) for s in ("CRCL", "MRVL", "LITE", "COHR")}
        fmt = (lambda v: f"{v:8.2f}") if k in ("pos20", "kday", "ptouch") else (lambda v: f"{pct(v):>8}")
        j1 = judge(vals["CRCL"], vals["MRVL"], gs)
        j2 = judge(vals["LITE"], vals["COHR"], gs)
        tally[name] = (j1, j2)
        say(f"  {name:22} {stat:24} {fmt(vals['CRCL'])} {fmt(vals['MRVL'])} {j1:>12} {fmt(vals['LITE'])} {fmt(vals['COHR'])} {j2:>12}")
    # big15(사건, 날짜 단위 관찰): 같은 날 종목 비교에는 효과 0으로 검증됨
    b = {s: (e0.big15.get(s, np.nan) if len(e0) else np.nan) for s in ("LITE", "COHR")}
    b2 = {s: (e2.big15.get(s, np.nan) if len(e2) else np.nan) for s in ("LITE", "COHR")}
    gsb = lambda v: None if not np.isfinite(v) else not (v > 0.4853)
    gsb2 = lambda v: None if not np.isfinite(v) else not (v > 0.402)
    j = judge(b["LITE"], b["COHR"], gsb)
    j2 = judge(b2["LITE"], b2["COHR"], gsb2)
    fb = lambda v: f"{v:8.2f}" if np.isfinite(v) else f"{'사건없음':>8}"
    say(f"  {'big15>0.485 → 사건 약함':22} {'spike 관찰(날짜 단위만)':24} {'해당없음':>8} {'해당없음':>8} {'-':>12} "
        f"{fb(b['LITE'])} {fb(b['COHR'])} {j:>12}")
    say(f"  {'big15>0.402 (V2 50%)':22} {'spike 관찰(날짜 단위만)':24} {'해당없음':>8} {'해당없음':>8} {'-':>12} {fb(b2['LITE'])} {fb(b2['COHR'])} {j2:>12}")
    say("  판정: '맞음(사후 일치)' = 실제 좋은 쪽을 좋음, 나쁜 쪽을 나쁨으로 둠. '못 가름' = 둘 다 같은 쪽. '반대' = 거꾸로.")
    say("")
    say("  같은 규칙을 10/9 첫 정규장 신호 7건(CRCL MRVL LITE COHR CIEN AAOI FN) 전체에: 규칙이 표시한 쪽(A) vs 나머지(B), Y1은 11:33 평가 포함(하루치·미확정)")
    syms = list(first)
    for name, stat, k, gs in rules:
        A = [s for s in syms if gs(fv(s, k)) is not None and gs(fv(s, k)) == ("좋음" in name)]
        B = [s for s in syms if s not in A]
        ya = np.mean([first[s]["y1"] for s in A]) if A else np.nan
        yb = np.mean([first[s]["y1"] for s in B]) if B else np.nan
        y2a = sum(first[s]["y2"] == 1 for s in A)
        y2b = sum(first[s]["y2"] == 1 for s in B)
        say(f"  {name:22} A {','.join(A) or '-':28} Y1 {pct(ya):>7} Y2 {y2a}/{len(A)} | B {','.join(B) or '-':28} Y1 {pct(yb):>7} Y2 {y2b}/{len(B)}")
    say("  (A = 규칙이 '나쁨/드묾'이면 그 표시가 붙은 쪽, '좋음'이면 좋음 표시 쪽.)")
    say("")
    return tally


def part4():
    say("4. 10/9 광통신 급등 이유(웹 검색 6회, 사후 설명용 — 매매 근거 아님)")
    say("  - 10/9 재료: Lumentum CEO 마이클 헐스턴이 Bloomberg 인터뷰(10/9)에서 'AI 데이터센터용 광부품 생산능력이 2029년 초까지 매진'.")
    say("    6개월 전 '2028년까지'에서 1년 늘어남, 일부 제품은 내년 수요의 약 70%를 못 맞춤. 24/7 Wall St(10/9 장중): AAOI·LITE +7%, COHR +5%.")
    say("  - 10/8 배경: 광통신 일제 하락(일봉: COHR -9.6%, AAOI -13.6%, LITE -5.6%). 기사 이유는 국채금리·유가 상승, 연준 인상 신호, 올해 급등 뒤 차익실현.")
    say("    COHR는 회사 고유 악재 없음(tradingkey 10/8).")
    say("  - LITE = 자기 회사 CEO 발언(직접 재료). COHR = 같은 업종 동반 상승(10/9 고유 재료는 못 찾음). heygotrade: COHR 11:56 ET 308.79(+2.13%).")
    say("  - 'FCC가 중국산 광 트랜시버 수입 금지 추진' 보도(Benzinga, 기사 속 LITE 1046.39 +7.74%, COHR 319.48 +11.0%, 목요일)는")
    say("    일봉과 대조하면 10/1(목, LITE 1045.78 +7.7%, COHR 319.19 +10.9%) 재료다. 10/9 재료가 아니다(8월 초에도 같은 보도).")
    say("  - 가격대 사후 설명(결과를 보고 붙인 것, 규칙 아님):")
    say("    LITE 고점 1143은 5일 가치구간 위 +1.9%(10/6 고가 1137 돌파, 120일 최고 종가 +0.85%). 저점 1077은 가치구간 안·5일 POC -0.8%·밀도 1.60에서 멈춤.")
    say("    COHR 고점 332는 10/1 FCC 보도 뒤 10/2~10/7 거래 구간(5일 가치구간 319~339) 안, 5일 POC 바로 아래(-0.75%), 120일 최고 종가 -22%.")
    say("    저점 302.35는 가치구간 아래 -5.3%, 밀도 0.62, 10/8 종가와 정확히 같은 값.")
    say("    같은 날 다른 광통신이 이 설명을 깬다: 고점이 가치구간 위였던 FN은 회복 못 함, 가치구간 안이던 AAOI·VIAV는 회복(1절 표).")
    say("    가까운 시험 결과도 없다: hi120(120일 고점 거리) 사건 시험 앞 t -0.63, 묶음 안 쌍 일치 앞 61%·뒤 41%(pat_spike),")
    say("    vp5r<1(5일 빈 가격대) 사건 → 뒤 t -0.04, h5<0(5일 고가 아래) → 뒤 t +0.66(pat_levels). 저점 위치(L)는 저점이 지나야 알 수 있는 값이다.")
    say("  출처: 24/7 Wall St 10/9·10/8, Bloomberg 영상 10/9, Benzinga(Lumentum sold out through early 2029; Washington targets Chinese optical transceivers),")
    say("        tikr, tradingkey 10/8, heygotrade COHR.")
    say("")


def main():
    D = {s: load3(s) for s in sorted(set(OPT_ALL + CASE))}
    say("10/9 사례 정리 (case_1009.py, 2026-10-10)")
    say("결과 정의(측정 전 고정): 거래 Y1 = 지금 매도 규칙 수익(11:33까지 미매도면 11:33 평가), Y2 = 60분 안 RSI 매도.")
    say("  사건: 반납 = (H-L)/(H-P), 회복 = 11:33 가격 >= P+0.5(H-P), 되찾음 = (11:33-L)/(H-L). P = 10/8 정규장 종가.")
    say("  데이터가 10/9 11:33 ET에서 끝난다 → 장중 값이고 종가·최종 결과는 확정되지 않았다. 2종목 사례로 규칙을 만들지 않는다.")
    say("")
    R = part1(D)
    first, F = part2(D, R)
    ev = part2b(D, R)
    part3(first, F, ev, R, None)
    part4()
    with open(os.path.join(RES, "case_1009.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()

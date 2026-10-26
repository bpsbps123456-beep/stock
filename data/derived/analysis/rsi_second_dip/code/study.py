"""40분봉 RSI 두 번째 30 이탈 매수 · RSI 70 매도 연구(2026-10-10 사용자 아이디어, 키 rsi_second_dip).

질문(사용자): 급락주는 40분봉 RSI가 30 아래로 여러 번 떨어진다(차트 예: BULL 10/7). 두 번째로 30 아래로 떨어질 때 사고,
  40분봉 RSI가 70 이상이면 팔면 어떤가.

규칙(기본 = 사용자 말 그대로)
  RSI는 Pine ta.rsi와 같은 와일더 RMA, 24시간 봉(04:00~20:00 ET).
  이탈: 봉 종가 RSI <= 30. 한 번 이탈한 뒤에는 종가 RSI가 ARM(기본 30)을 넘어야 다음 이탈을 센다(사이에 30 위로 돌아와야 '두 번째').
  리셋: 종가 RSI >= RESET(기본 70)이면 센 횟수를 0으로(한 하락 국면 = RSI 70과 70 사이).
  매수: 그 국면의 K번째(기본 2) 이탈이 확정된 봉의 다음 봉 시가. 매도: 종가 RSI >= 70인 봉의 다음 봉 시가.
  손절·기한 없음(사용자 규칙). 자료 끝까지 안 팔렸으면 마지막 종가로 평가하고 '미청산'으로 따로 센다.
변형
  봉 안 체결(I): 직전 봉까지의 RMA 상태로 'RSI가 30이 되는 가격'을 구해 저가가 닿으면 min(시가, 그 가격)에 사고,
    'RSI가 70이 되는 가격'에 고가가 닿으면 max(시가, 그 가격)에 판다(트레이딩뷰 실시간 RSI가 30·70에 닿는 순간).
  K = 1·2·3, ARM = 30·40, RESET = 70·50, 1번째~K번째 이탈 간격 제한 WIN = 없음·5거래일.
비교 잣대
  같은종목같은날: 같은 종목, 신호가 확정된 날(ET)의 모든 봉에서 '그 봉 다음 시가에 사서 같은 RSI 70 규칙으로 판' 수익 평균.
    매도 규칙이 '강할 때 판다'라서 아무 때나 사도 플러스가 나오기 쉬우므로, 진입 시점의 값어치는 이것과 비교한다.
  SPY 초과: 같은 보유 구간 SPY 수익을 뺀 값.
  급락주: 신호 시각에 아는 직전 10거래일(그날 정규장 종가를 이미 알면 오늘 포함, 아니면 어제까지) 안에
    정규장 종가 기준 하루 -7% 이하가 있었던 신호. (10/10 첫 판은 장중 신호에도 그날 종가를 써서 미래 정보가 섞였다.)
자료
  1h  premarket_shock/bars 1시간봉(시총 3억 달러 이상 현재 상장 3,570종목, 2023-11~2026-10-07 09:30).
      40분봉 RSI14(560분)를 1시간봉 RSI9(540분)로 근사한다. 2024·2025·2026 상반기·하반기로 나눠 본다.
  40m 5분봉(fetch5m.py, 7/17~10/9)을 04:00 ET 기준 40분봉으로 묶은 진짜 RSI14.
      종목은 8/11 이후 -7% 날이 있던 987종목이라 종목 고르기에 미래 정보가 섞인다 → 8/11 이후 급락일로만 급락주를
      표시하고(crash_since) 8/12 이후 급락주 신호만 센다.
      AI 137종목(dip_reason raw_1010 5분봉)은 미래 정보 없이 고른 묶음이라 전체 신호를 센다.
한계: 현재 상장·시총으로 고른 종목이라 생존 편향이 있다(무너져 상장 폐지된 종목이 빠짐). yfinance 프리·애프터 봉은 거래량 0인 얇은 체결이 섞인다.
사용: py study.py 1h | py study.py 40m   → results/study_<원천>.txt, 거래표 scratchpad rsi2_trades_<원천>.parquet
     py study.py 1h core  → 핵심 6개 변형만(메모리 절약), results/study_1h_core.txt, 거래표는 같은 이름으로 덮어씀
  거래표 열: 규칙 sym day(신호 ET 날짜) t(신호 봉) te·tx(매수·매도 봉 시각, 미청산은 NaT) e_reg·x_reg(정규장 봉인가) ent px ret open hold_d
            spy same_day crash gap_d(첫 이탈~신호 거래일) rsi_sig ex_spy ex_day bk(기간) [src: 40m의 급락987·AI137]
"""
import glob, json, os, sys
from datetime import date
import numpy as np, pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
AN = os.path.normpath(os.path.join(ROOT, ".."))
sys.path.insert(0, os.path.join(AN, "kalman_zones", "code"))
from osc import rsi_state                                           # Pine ta.rsi와 같은 RMA(SMA로 시작)

SP = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/6a57b230-3ac1-4ae6-9d34-b1d8a1c2ce1b/scratchpad"
B1H = os.path.join(AN, "premarket_shock", "bars")
B5M = os.path.join(SP, "raw5m_rsi2")
B5M_AI = os.path.join(SP, "raw_1010")
LO, HI = 30.0, 70.0


# ---------------------------------------------------------------- 자료
def bars_1h(s):
    d = pd.read_parquet(os.path.join(B1H, f"1h_{s}.parquet"))
    return d[d.High > 0]


def bars_40m(s, folder):
    m5 = pd.read_parquet(os.path.join(folder, f"5m_{s}.parquet"))
    m5 = m5[m5.High > 0]
    return m5.resample("40min", origin="start_day", offset="4h", label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna(subset=["Close"])


def prep(d, n, blen, crash_since=None):
    """봉 → 배열 묶음. p30[i]·p70[i] = 봉 i 종가가 그 값이면 RSI가 30·70이 되는 가격(봉 i-1까지의 RMA 상태로 계산).
    급락주 표시는 신호 시각에 아는 값만 쓴다(10/10 독립 재구현 검증에서 미래 정보 발견 → 고침):
      그날 정규장 종가(16:00 ET)를 이미 알면 '오늘 포함 10거래일', 아니면 '어제까지 10거래일' 안에 -7% 이하 날이 있는가.
      crash_C는 봉 끝 시각(C는 봉 마감에 판단), crash_I는 봉 시작 시각(I는 봉 안에서 판단) 기준. blen = 봉 길이(분).
      crash_since = 이 날짜 이전 급락일은 세지 않음(종목을 그 뒤 급락으로 고른 묶음의 미래 정보 차단)."""
    c = d.Close.values.astype(float)
    r, ru, rd = rsi_state(c, n)
    pc, pu, pdn = np.r_[np.nan, c[:-1]], np.r_[np.nan, ru[:-1]], np.r_[np.nan, rd[:-1]]
    R30, R70 = LO / (100 - LO), HI / (100 - HI)
    with np.errstate(invalid="ignore"):
        p30 = np.minimum(pc - (n - 1) * (pu / R30 - pdn), pc * 0.999)   # 이미 30 아래면 직전 종가 바로 아래
        p70 = np.maximum(pc + (n - 1) * (R70 * pdn - pu), pc * 1.001)   # 이미 70 위면 직전 종가 바로 위
    idx = d.index
    mod = idx.hour * 60 + idx.minute
    reg = (mod >= 570) & (mod < 960)
    ordn = np.array([x.toordinal() for x in idx.date])
    _, tday = np.unique(ordn, return_inverse=True)
    # 정규장 종가 기준 하루 수익률 → '직전 10거래일(오늘 포함) 안에 -7% 이하'
    dc = pd.Series(c[reg], index=idx.date[reg]).groupby(level=0).last()
    dr = dc.pct_change()
    crash_day = set(x for x in dr[dr <= -0.07].index if crash_since is None or x >= crash_since)
    days = sorted(set(idx.date))
    pos = {dd: k for k, dd in enumerate(days)}
    inc, prev = np.zeros(len(days), bool), np.zeros(len(days), bool)
    for k in sorted(pos[x] for x in crash_day if x in pos):
        inc[k:k + 10] = True                             # 오늘 포함 10거래일
        prev[k + 1:k + 11] = True                        # 어제까지 10거래일
    return dict(idx=idx, o=d.Open.values.astype(float), h=d.High.values.astype(float), l=d.Low.values.astype(float),
                c=c, r=r, p30=p30, p70=p70, tday=tday.astype(np.int64), day=np.array(idx.date),
                crash_C=np.where(mod + blen >= 960, inc[tday], prev[tday]), crash_I=np.where(mod >= 960, inc[tday], prev[tday]),
                tk=idx.as_unit("ns").asi8, reg=np.asarray(reg))


# ---------------------------------------------------------------- 상태 기계
@njit(cache=True)
def machine(o, h, l, c, r, p30, p70, tday, K, ARM, RESET, WIN, intrabar):
    """반환 거래표: 신호 봉, 매수 봉, 매수가, 매도 봉(-1 = 미청산), 매도가(미청산이면 마지막 종가), 첫 이탈 봉."""
    n = len(c)
    S = np.full(n, -1, np.int64); E = np.full(n, -1, np.int64); X = np.full(n, -1, np.int64); F1 = np.full(n, -1, np.int64)
    EP = np.full(n, np.nan); XP = np.full(n, np.nan)
    m, cnt, first_day, first_bar = 0, 0, -1, -1
    armed, hold, e = False, False, -1
    for i in range(2, n):
        rc = r[i - 1]                                    # 마지막으로 완성된 봉의 RSI
        if rc != rc:
            continue
        if hold:
            sold, px = False, 0.0
            if intrabar:
                if i > e and p70[i] == p70[i] and h[i] >= p70[i]:
                    sold, px = True, max(o[i], p70[i])
            elif i - 1 >= e and rc >= 70.0:
                sold, px = True, o[i]
            if sold:
                X[m], XP[m] = i, px
                m += 1
                hold, cnt, armed = False, 0, False
            continue
        if rc >= RESET:
            cnt = 0
        if rc > ARM:
            armed = True
            if not intrabar:
                continue
        event, ep = False, 0.0
        if intrabar:
            if armed and p30[i] == p30[i] and l[i] <= p30[i]:
                event, ep, sb = True, min(o[i], p30[i]), i
        elif armed and rc <= 30.0:
            event, ep, sb = True, o[i], i - 1
        if not event:
            continue
        armed = False
        if cnt >= 1 and WIN > 0 and tday[sb] - first_day > WIN:
            cnt = 0
        cnt += 1
        if cnt == 1:
            first_day, first_bar = tday[sb], sb
        if cnt == K:
            S[m], E[m], EP[m], F1[m] = sb, i, ep, first_bar
            hold, e = True, i
    if hold:
        XP[m] = c[n - 1]
        m += 1
    return S[:m], E[:m], EP[:m], X[:m], XP[:m], F1[:m]


def base_rets(A):
    """모든 봉 j에서 '다음 봉 시가에 사서, 종가 RSI >= 70인 첫 봉의 다음 봉 시가에 판' 수익(같은종목같은날 잣대)."""
    o, c, r = A["o"], A["c"], A["r"]
    n = len(o)
    hit = np.flatnonzero(r >= HI)
    out = np.full(n, np.nan)
    ent = np.arange(n) + 1
    ok = ent < n
    k = np.searchsorted(hit, ent[ok], side="left")
    xb = np.where(k < len(hit), hit[np.minimum(k, max(len(hit) - 1, 0))] + 1 if len(hit) else n, n)
    xp = np.where(xb < n, o[np.minimum(xb, n - 1)], c[-1])
    out[ok] = xp / o[ent[ok]] - 1
    return out


# ---------------------------------------------------------------- 실행
def universe(src):
    if src == "1h":
        return sorted(os.path.basename(f)[3:-8] for f in glob.glob(os.path.join(B1H, "1h_*.parquet")))
    crash = json.load(open(os.path.join(SP, "rsi2_crash_syms.json")))
    return sorted(set(crash))


def ai137():
    sys.path.insert(0, os.path.join(AN, "dip_reason", "code"))
    from stop_lab import ALL
    return sorted(ALL)


def spy_px(src):
    d = bars_1h("SPY") if src == "1h" else bars_40m("SPY", B5M)
    return d.index.as_unit("ns").asi8, d.Close.values.astype(float)


VARIANTS = {}
for mode in ("C", "I"):
    for K in (1, 2, 3):
        for ARM in (30, 40):
            for RESET in (70, 50):
                for WIN in (0, 5):
                    VARIANTS[f"{mode} K{K} ARM{ARM} R{RESET} W{WIN}"] = (mode == "I", K, float(ARM), float(RESET), WIN)
PRIMARY = ("C K2 ARM30 R70 W0", "I K2 ARM30 R70 W0")
CORE = [f"{m} K{k} ARM30 R70 W0" for m in ("C", "I") for k in (1, 2, 3)]   # 'core' 실행: 메모리를 아끼려고 이것만(10/10 1시간봉 재실행이 메모리 부족으로 멈춤)


def run(src, syms, folder=None, crash_since=None):
    n, blen = (9, 60) if src == "1h" else (14, 40)
    st, sc = spy_px(src)
    rows = []
    for s in syms:
        try:
            d = bars_1h(s) if src == "1h" else bars_40m(s, folder)
        except (FileNotFoundError, OSError):
            continue
        if len(d) < 60 or s in ("SPY", "QQQ"):
            continue
        A = prep(d, n, blen, crash_since)
        br = base_rets(A)
        dmean = pd.Series(br).groupby(A["tday"]).mean().to_dict()        # 같은 종목 같은 날 평균
        for vn, (ib, K, ARM, RESET, WIN) in VARIANTS.items():
            S_, E_, EP_, X_, XP_, F1_ = machine(A["o"], A["h"], A["l"], A["c"], A["r"], A["p30"], A["p70"], A["tday"],
                                                K, ARM, RESET, WIN, ib)
            for j in range(len(S_)):
                sb, eb, xb = S_[j], E_[j], X_[j]
                t0 = A["tk"][eb]
                t1 = A["tk"][xb] if xb >= 0 else A["tk"][-1]
                k0, k1 = np.searchsorted(st, t0, "right") - 1, np.searchsorted(st, t1, "right") - 1
                spy = sc[k1] / sc[k0] - 1 if k0 >= 0 and k1 >= 0 else np.nan
                rows.append((vn, s, A["day"][sb], A["idx"][sb], EP_[j], XP_[j], XP_[j] / EP_[j] - 1, xb < 0,
                             (A["tday"][xb] if xb >= 0 else A["tday"][-1]) - A["tday"][eb], spy, dmean.get(A["tday"][sb], np.nan),
                             bool((A["crash_I"] if ib else A["crash_C"])[sb]), A["tday"][sb] - A["tday"][F1_[j]], A["r"][sb],
                             A["idx"][eb], A["idx"][xb] if xb >= 0 else pd.NaT, bool(A["reg"][eb]), bool(A["reg"][xb]) if xb >= 0 else False))
    t = pd.DataFrame(rows, columns=["규칙", "sym", "day", "t", "ent", "px", "ret", "open", "hold_d", "spy", "same_day", "crash",
                                    "gap_d", "rsi_sig", "te", "tx", "e_reg", "x_reg"])
    t["ex_spy"] = t.ret - t.spy
    t["ex_day"] = t.ret - t.same_day
    return t


def bucket(src, d):
    if src == "1h":
        if d < date(2024, 1, 1):
            return None
        if d < date(2025, 1, 1):
            return "2024"
        if d < date(2026, 1, 1):
            return "2025"
        return "2026 상(1~6월)" if d < date(2026, 7, 1) else "2026 하(7/1~10/7)"
    if d < date(2026, 7, 24):
        return None                                      # RSI 자리 잡기
    return "앞 7/24~8/31" if d < date(2026, 9, 1) else "뒤 9/1~10/9"


def stats(x):
    if not len(x):
        return dict(건수=0)
    dm = x.groupby("day").ret.mean()
    se = dm.std(ddof=1) / np.sqrt(len(dm)) if len(dm) > 1 else np.nan
    de = x.groupby("day").ex_day.mean()
    se2 = de.std(ddof=1) / np.sqrt(len(de)) if len(de) > 1 else np.nan
    return dict(건수=len(x), 날수=x.day.nunique(), 평균=x.ret.mean(), 중앙=x.ret.median(), 승률=(x.ret > 0).mean(),
                하위10=x.ret.quantile(.1), 최악=x.ret.min(), t=dm.mean() / se if se and se > 0 else np.nan,
                SPY초과=x.ex_spy.mean(), 같은날대비=x.ex_day.mean(), 같은날t=de.mean() / se2 if se2 and se2 > 0 else np.nan,
                보유중앙일=x.hold_d.median(), 보유10일초과=(x.hold_d > 10).mean(), 미청산=x.open.mean())


FMT = (("평균", "{:+.2%}"), ("중앙", "{:+.2%}"), ("승률", "{:.0%}"), ("하위10", "{:+.1%}"), ("최악", "{:+.1%}"), ("t", "{:+.2f}"),
       ("SPY초과", "{:+.2%}"), ("같은날대비", "{:+.2%}"), ("같은날t", "{:+.2f}"), ("보유중앙일", "{:.0f}"), ("보유10일초과", "{:.0%}"),
       ("미청산", "{:.0%}"))


def table(src, t, names, filt):
    rows = []
    for vn in names:
        x = t[t.규칙 == vn]
        for b in sorted(set(x.bk.dropna())):
            for fn, f in filt.items():
                y = f(x[x.bk == b])
                rows.append(dict(규칙=vn, 기간=b, 묶음=fn, **stats(y)))
    df = pd.DataFrame(rows)
    out = df.copy()
    for k, fm in FMT:
        if k in out:
            out[k] = out[k].map(lambda v: "" if pd.isna(v) else fm.format(v))
    return df, out


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "1h"
    core = len(sys.argv) > 2 and sys.argv[2] == "core"
    if core:
        for k in [v for v in VARIANTS if v not in CORE]:
            del VARIANTS[k]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 2000)
    if src == "1h":
        t = run("1h", universe("1h"))
        filt = {"전체": lambda x: x, "급락주": lambda x: x[x.crash], "급락주 아님": lambda x: x[~x.crash]}
    else:
        t1 = run("40m", universe("40m"), B5M, crash_since=date(2026, 8, 11))
        t1["src"] = "급락987"
        t1 = t1[t1.crash & (t1.day >= date(2026, 8, 12))]                    # 종목 고르기의 미래 정보 차단
        t2 = run("40m", ai137(), B5M_AI)
        t2["src"] = "AI137"
        t = pd.concat([t1, t2], ignore_index=True)
        filt = {"급락987·급락주(8/12~)": lambda x: x[x.src == "급락987"], "AI137 전체": lambda x: x[x.src == "AI137"],
                "AI137 급락주": lambda x: x[(x.src == "AI137") & x.crash]}
    t["bk"] = [bucket(src, d) for d in t.day]
    t = t[t.bk.notna()]
    t.to_parquet(os.path.join(SP, f"rsi2_trades_{src}.parquet"))
    suffix = "_core" if core else ""
    L = [f"40분봉 RSI 두 번째 30 이탈 매수 · RSI 70 매도 ({src}, {'1시간봉 RSI9 근사' if src == '1h' else '진짜 40분봉 RSI14'}, 2026-10-10)",
         "칸: 평균·중앙·승률·하위10·최악 = 거래 수익, t = 날짜 묶음 t, SPY초과 = 같은 보유 구간 SPY를 뺀 평균,",
         "    같은날대비 = 같은 종목·같은 날 아무 봉에 사서 같은 RSI 70 규칙으로 판 수익 평균을 뺀 값(진입 시점의 값어치), 같은날t = 그 날짜 묶음 t,",
         "    보유중앙일 = 거래일, 미청산 = 자료 끝까지 RSI 70을 못 넘어 마지막 종가로 평가한 비율",
         "규칙 이름: C = 봉 종가 확정 뒤 다음 시가 / I = 봉 안에서 RSI 30·70 가격에 닿을 때, K = 몇 번째 이탈, ARM = 이탈 사이 회복 기준,",
         "    R = 횟수 리셋 RSI, W = 첫 이탈~K번째 최대 거래일(0 = 제한 없음)"]
    kcmp = [f"{m} K{k} ARM30 R70 W0" for m in ("C", "I") for k in (1, 2, 3)]
    df, out = table(src, t, kcmp, filt)
    df.to_csv(os.path.join(SP, f"rsi2_k_{src}{suffix}.csv"), index=False, encoding="utf-8-sig")
    L += ["", "=" * 20 + " 1) 몇 번째 이탈에 살까: K = 1·2·3 (기본 해석 ARM30 R70 W0)", out.to_string(index=False)]
    df2, out2 = table(src, t, list(VARIANTS), filt)
    df2.to_csv(os.path.join(SP, f"rsi2_all_{src}{suffix}.csv"), index=False, encoding="utf-8-sig")
    sens = [v for v in VARIANTS if " K2 " in v]
    L += ["", "=" * 20 + " 2) 해석 민감도(K=2): ARM 30·40 × 리셋 70·50 × 간격 제한 없음·5일 × 체결 C·I",
          out2[out2.규칙.isin(sens)].to_string(index=False)]
    with open(os.path.join(ROOT, "results", f"study_{src}{suffix}.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()

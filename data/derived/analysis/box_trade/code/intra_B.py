"""박스권 매매 B — 장중 바닥 이탈 = 바닥 매수(손절 물량 받기) [키 intra_B] 2026-10-09
사용자 가설(2026-10-09): 지지선을 막 이탈하는 그 점이 오히려 최저점이다(사람들이 손절했을 것). 최근 장세에 더 맞는다.
예: NKE 10/1 장 마감 뒤 실적 → 10/2 갭하락(저가 31.97) → 10/8 34.74.

데이터: stop_lab.load — 앞(고르기) 7/14~9/8 5분봉(EMA36·RSI9 = 3분봉 EMA60·RSI14 근사), 뒤(확인) 9/10~10/6 3분봉(EMA60·RSI14).
종목: 사용자78, 전체137(사용자78 ∪ AI 127). 기본 범위는 정규장(9:30~16:00 ET) 봉만 신호·체결.

지지선(봉 i의 값은 봉 i-1까지로만 정한다):
  N60·N120·N240  직전 정규장 봉 N분의 최저가(전날 정규장으로 이어짐). 박스 상단 = 같은 창의 최고가
  PDL            전 거래일 정규장 저가(상단 = 전일 고가)
  L5D            직전 5거래일 정규장 저가(상단 = 5일 고가)
  ORL            오늘 개장 30분(9:30~10:00) 저가, 10:00 이후만(상단 = 개장 30분 고가)
B 진입(지지선마다 하루 첫 신호 한 번, 같은 날 정규장 안에서 체결):
  LIMy   지정가 = 지지선 × (1-y), y = 0·0.5·1·2%. 봉 저가 < 지정가면 min(시가, 지정가) 체결(갭하락이면 시가)
  CLS    봉 종가 < 지지선 → 다음 봉 시가
  RCLk   첫 이탈 봉 뒤 k분(이탈 봉 포함) 안에 봉 종가 > 이탈 때 지지선(가짜 이탈) → 다음 봉 시가. k = 15·30
  필터   RSI = 직전 봉 RSI ≤ 30(LIM) 또는 이탈 봉 RSI ≤ 30(CLS) / VOL = 이탈 봉 시각 보정 거래량 ≥ 2배(지난 10거래일 같은 시각 평균)
         W2·W4 = 박스 폭(상단/지지선-1) ≤ 2%·4%(N분 지지선만)
비교용(같은 지지선·같은 매도, 시험 규칙 수에 포함):
  A_TOUCH  지지선 × 1.005 지정가(지지선 위에서 버팀 매수)        C_BRK  종가 > 박스 상단 → 다음 봉 시가(위 돌파)
매도(모든 진입에 같은 세트):
  RSI8  지금 지표와 같음: RSI > 70 봉 다음 시가 또는 매수 뒤 첫 한국 아침 8시 이후 첫 봉 시가
  RSI8R RSI8과 같되 RSI > 70 판정·매도를 정규장 봉에서만(애프터 얇은 체결 배제, 견고성 확인용)
  8시   한국 아침 8시(=뉴욕 19:00, 애프터) 첫 봉 시가     H1·H4·D1  매수 봉 시작 + 1시간·4시간·24시간 뒤 첫 정규장 봉 시가
  TP1·TP2  +1%·+2% 지정가(매수 다음 봉부터 정규장 봉 고가로 판정, 시가가 위면 시가), 못 닿으면 8시
  TP2W  실행 가능판: 뉴욕 11:00(한국 24:00) 전 체결이면 TP2, 자는 동안 체결이면 8시(깨어나야 매도 주문을 걸 수 있음)
  TP2S  TP2 + 손절: 정규장 봉 종가 < 이탈 저점(신호~매수 봉 최저가) × 0.99면 다음 봉 시가. 사용자가 깨어 있는 한국 8~24시에만
        (A는 지지선 × 0.99, C는 박스 하단 × 0.99)
  MID   박스 중간((지지선+상단)/2) 지정가, 못 닿으면 8시(C는 해당 없음)
비교 기준:
  무작위1(기본) = 같은 종목·같은 기간(날 무관)·같은 30분 칸의 모든 정규장 봉 시가 매수, 같은 매도(= 같은 시간대·같은 보유 길이의 매수 후 보유).
          거래마다 짝지음. '이탈 뒤 반등이 평소 흐름보다 큰가'를 재는 기준이다.
  무작위3 = 같은 종목·같은 날·같은 30분 칸(intra_A와 같은 기준)     무작위2 = 같은 종목·같은 날의 모든 정규장 봉
          무작위3·2는 그날 경로에 맞춰 짝지어, 떨어진 뒤 사는 규칙이 '같은 칸의 떨어지기 전 봉'보다 싸게 사는 몫이 기계적으로 초과에 들어간다
          (반등이 없어도 플러스, 위 돌파 C는 같은 이유로 마이너스). 그래서 고르기는 무작위1로 하고 무작위3·2는 함께 적는다.
  MID는 목표 거리(%)를 가장 가까운 0.5~10% 격자 값으로 맞춘 무작위 +x% 지정가 매도와 비교한다.
  지금 이격매수 지표(stop_lab.sim 정규장 매수) — 자기 매도(RSI8)와 위 매도 세트 모두
고르기: 앞 기간·정규장·사용자78·RSI8 매도, 거래 30건 이상인 B 규칙 중 '무작위1 대비 초과의 날짜 군집 t' 최대. 뒤 기간은 확인만.
        (첫 실행에서 이 기준으로 정했다. 무작위3 기준으로 고른 결과도 함께 적는다.)
사용: py intra_B.py  → ../results/intra_B.txt, ../results/intra_B_grid.csv
"""
import os, sys, time
from datetime import date
import numpy as np, pandas as pd
from numba import njit

ROOT = r"C:\Users\geunt\OneDrive\Desktop\주식\data\derived\analysis"
DIPC = os.path.join(ROOT, "dip_reason", "code")
sys.path.insert(0, DIPC)
sys.path.insert(0, os.path.join(DIPC, "support"))
from stop_lab import load, sim as base_sim, UNIV, RAW, X as RSI_X   # noqa: E402
import box as BX                                                      # noqa: E402  norm_vol, dct
from events import bars as ev_bars, prep as ev_prep                   # noqa: E402

OUT = os.path.join(ROOT, "box_trade", "results", "intra_B.txt")
GRID_CSV = OUT.replace(".txt", "_grid.csv")
EARN_CSV = os.path.join(ROOT, "box_trade", "results", "earnings_cache.csv")
SCR = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/intra_B"

PER = {"early": (date(2026, 7, 14), date(2026, 9, 8)), "late": (date(2026, 9, 10), date(2026, 10, 7))}
PLAB = {"early": "앞(고르기) 7/14~9/8 5분봉", "late": "뒤(확인) 9/10~10/6 3분봉"}
PSH = {"early": "앞", "late": "뒤"}
U78, U137 = UNIV["사용자78"], UNIV["전체137"]
UNS = (("78", U78), ("137", U137))
LEVELS = ("N60", "N120", "N240", "PDL", "L5D", "ORL")
LEVLAB = {"N60": "직전60분저가", "N120": "직전120분저가", "N240": "직전240분저가", "PDL": "전일저가", "L5D": "5일저가", "ORL": "개장30분저가"}
EXITS = ("RSI8", "RSI8R", "8시", "H1", "H4", "D1", "TP1", "TP2", "TP2W", "TP2S", "MID")
HMIN = {"H1": 60, "H4": 240, "D1": 1440}
TPG = np.array([0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10])
INF = np.inf
_fh = None


def say(*a):
    global _fh
    if _fh is None:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        _fh = open(OUT, "w", encoding="utf-8")
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _fh.write(s + "\n")
    _fh.flush()


def f_pct(v, k=2):
    return "n/a" if v is None or not np.isfinite(v) else f"{v * 100:+.{k}f}%"


# ---------------------------------------------------------------- 규칙 목록
def rules():
    R = []
    meths = [("LIM", 0.0, ""), ("LIM", 0.005, ""), ("LIM", 0.01, ""), ("LIM", 0.02, ""), ("CLS", 0, ""), ("RCL", 15, ""), ("RCL", 30, ""),
             ("LIM", 0.01, "RSI"), ("CLS", 0, "RSI"), ("CLS", 0, "VOL"), ("RCL", 15, "VOL")]
    for lev in LEVELS:
        for m, p, f in meths:
            R.append(dict(fam="B", lev=lev, meth=m, p=p, filt=f))
    for lev in ("N60", "N120", "N240"):
        for f in ("W2", "W4"):
            R.append(dict(fam="B", lev=lev, meth="LIM", p=0.005, filt=f))
    for y in (0.03, 0.05):                                           # 사후 관찰(고르기 풀 밖): 더 깊은 지정가
        R.append(dict(fam="Bx", lev="L5D", meth="LIM", p=y, filt=""))
    for lev in LEVELS:
        R.append(dict(fam="A", lev=lev, meth="TOUCH", p=0.005, filt=""))
        R.append(dict(fam="C", lev=lev, meth="BRK", p=0, filt=""))
    for k, r in enumerate(R):
        r["no"] = k + 1
        r["name"] = rname(r)
    return R


def rname(r):
    m = r["meth"]
    if m == "LIM":
        s = f"LIM{r['p'] * 100:g}%"
    elif m == "RCL":
        s = f"RCL{r['p']}분"
    elif m == "TOUCH":
        s = "A_TOUCH+0.5%"
    elif m == "BRK":
        s = "C_BRK"
    else:
        s = m
    return f"{r['lev']} {s}" + (f" {r['filt']}" if r["filt"] else "")


# ---------------------------------------------------------------- numba 매도 탐색
@njit(cache=False)
def scan(o, h, c, chk, awake, e, tg, sl, xd, n):
    """매수 봉 e 다음 봉부터 xd 직전까지: 정규장 봉(chk) 고가 ≥ tg면 max(시가, tg)에 매도,
    정규장 봉 종가 < sl이고 다음 봉이 깨어 있는 시각이면 다음 봉 시가 매도. 아니면 xd 봉 시가. 반환 (가격, 이유 0=시각 1=목표 2=손절)."""
    m = len(e)
    px = np.full(m, np.nan)
    why = np.full(m, -1, np.int8)
    for q in range(m):
        d = xd[q]
        if d < 0 or d >= n:
            continue
        k = e[q] + 1
        done = False
        while k < d:
            if chk[k]:
                if h[k] >= tg[q]:
                    px[q] = max(o[k], tg[q])
                    why[q] = 1
                    done = True
                    break
                if c[k] < sl[q] and k + 1 < n and awake[k + 1]:
                    px[q] = o[k + 1]
                    why[q] = 2
                    done = True
                    break
            k += 1
        if not done:
            px[q] = o[d]
            why[q] = 0
    return px, why


# ---------------------------------------------------------------- 종목별 준비
DAILY = {}


def daily_table(sym, A):
    """정규장 일봉(저가·고가·시가·종가) — 5분봉(7/14~10/6)에서 만든다. 뒤 기간 3분봉의 전일·5일 저가도 이 표를 쓴다."""
    reg = A["reg"]
    d = np.array([x.toordinal() for x in A["day"]])[reg]
    df = pd.DataFrame({"d": d, "l": A["l"][reg], "h": A["h"][reg], "o": A["o"][reg], "c": A["c"][reg]})
    g = df.groupby("d").agg(l=("l", "min"), h=("h", "max"), o=("o", "first"), c=("c", "last"))
    days = g.index.values.astype(np.int64)
    lo, hi = g.l.values, g.h.values
    l5 = pd.Series(lo).rolling(5, min_periods=5).min().values
    h5 = pd.Series(hi).rolling(5, min_periods=5).max().values
    DAILY[sym] = dict(days=days, l=lo, h=hi, o=g.o.values, c=g.c.values, l5=l5, h5=h5)


def earnings_days():
    """종목별 실적 반응일(정규장 기준): 발표 시각 12시(ET) 이후면 다음 거래일, 이전이면 당일."""
    E = {}
    src = []
    if os.path.exists(EARN_CSV):
        e = pd.read_csv(EARN_CSV)
        src.append(pd.DataFrame({"sym": e.ticker, "t": pd.to_datetime(e.earnings_dt, utc=True)}))
    p2 = os.path.join(RAW, "earnings.parquet")
    if os.path.exists(p2):
        e2 = pd.read_parquet(p2)
        src.append(pd.DataFrame({"sym": e2.sym, "t": pd.to_datetime(e2.t, utc=True)}))
    if not src:
        return E, 0
    e = pd.concat(src, ignore_index=True).drop_duplicates()
    t = e.t.dt.tz_convert("America/New_York")
    e = e.assign(d=t.dt.date.map(lambda x: x.toordinal()), hr=t.dt.hour)
    for s, g in e.groupby("sym"):
        E[s] = list(zip(g.d.values, g.hr.values))
    return E, len(e)


def react_days(sym, days, E):
    out = set()
    for d, hr in E.get(sym, []):
        if hr >= 12:
            k = np.searchsorted(days, d, side="right")
        else:
            k = np.searchsorted(days, d, side="left")
        if k < len(days):
            out.add(int(days[k]))
    return out


P = {}


def prep(period, sym, A, scope="reg", E=None, lohi=None):
    n, mins = len(A["o"]), A["mins"]
    o, h, l, c, v, r, tk, dead, reg, mod = (A[k] for k in ("o", "h", "l", "c", "v", "r", "tk", "dead", "reg", "mod"))
    ema = A["ema"]
    dayi = np.array([x.toordinal() for x in A["day"]], dtype=np.int64)
    lo, hi = lohi if lohi else PER[period]
    inper = (dayi >= lo.toordinal()) & (dayi <= hi.toordinal())
    sc = reg.copy() if scope == "reg" else np.ones(n, bool)
    ar = np.arange(n)
    # 지지선·상단
    LV, HV = {}, {}
    Xs = np.nonzero(sc)[0]
    for N in (60, 120, 240):
        K = N // mins
        lv = np.full(n, np.nan)
        hv = np.full(n, np.nan)
        lv[Xs] = pd.Series(l[Xs]).rolling(K, min_periods=K).min().shift(1).values
        hv[Xs] = pd.Series(h[Xs]).rolling(K, min_periods=K).max().shift(1).values
        LV[f"N{N}"], HV[f"N{N}"] = lv, hv
    D = DAILY[sym]
    k = np.searchsorted(D["days"], dayi, side="left") - 1          # 전 거래일
    okk = k >= 0
    kk = np.maximum(k, 0)
    LV["PDL"] = np.where(okk, D["l"][kk], np.nan)
    HV["PDL"] = np.where(okk, D["h"][kk], np.nan)
    LV["L5D"] = np.where(okk, D["l5"][kk], np.nan)
    HV["L5D"] = np.where(okk, D["h5"][kk], np.nan)
    pclose = np.where(okk, D["c"][kk], np.nan)
    kt = np.searchsorted(D["days"], dayi, side="left")
    kt_ok = (kt < len(D["days"])) & (D["days"][np.minimum(kt, len(D["days"]) - 1)] == dayi)
    dopen = np.where(kt_ok, D["o"][np.minimum(kt, len(D["days"]) - 1)], np.nan)
    gap = dopen / pclose - 1
    orm = reg & (mod >= 570) & (mod < 600)
    og = pd.DataFrame({"d": dayi[orm], "l": l[orm], "h": h[orm]}).groupby("d").agg(l=("l", "min"), h=("h", "max"), k=("l", "size"))
    og = og[og.k >= 30 // mins]
    orl = pd.Series(og.l).reindex(dayi).values
    orh = pd.Series(og.h).reindex(dayi).values
    after10 = mod >= 600
    LV["ORL"] = np.where(after10, orl, np.nan)
    HV["ORL"] = np.where(after10, orh, np.nan)
    for key in LEVELS:                                            # 범위 밖 봉은 신호 없음
        LV[key] = np.where(sc, LV[key], np.nan)
    # 매도 위치
    pos = np.nonzero(r > RSI_X)[0]
    if len(pos):
        q = np.searchsorted(pos, ar)
        krsi = np.where(q < len(pos), pos[np.minimum(q, len(pos) - 1)] + 1, n + 10)
    else:
        krsi = np.full(n, n + 10)
    idead = np.searchsorted(tk, dead, side="left")
    x1 = np.minimum(krsi, idead)
    XI = {"RSI8": np.where(x1 < n, x1, -1), "8시": np.where(idead < n, idead, -1)}
    regn = np.r_[reg[1:], False]
    posr = np.nonzero((r > RSI_X) & reg & regn)[0]
    if len(posr):
        q = np.searchsorted(posr, ar)
        krr = np.where(q < len(posr), posr[np.minimum(q, len(posr) - 1)] + 1, n + 10)
    else:
        krr = np.full(n, n + 10)
    x1r = np.minimum(krr, idead)
    XI["RSI8R"] = np.where(x1r < n, x1r, -1)
    for ex, hm in HMIN.items():
        q = np.searchsorted(tk[Xs], tk + np.int64(hm) * 60 * 10**9, side="left")
        XI[ex] = np.where(q < len(Xs), Xs[np.minimum(q, len(Xs) - 1)], -1)
    kst_h = ((tk // (3600 * 10**9)) + 9) % 24
    awake = kst_h >= 8
    nv = BX.norm_vol(A, dayi) if scope in ("reg", "all") else np.full(n, np.nan)
    ema_p = np.r_[np.nan, ema[:-1]]
    tch = reg & (l <= ema_p * 0.98)                               # 이격매수 선(EMA-2%) 닿음
    ctch = np.r_[0, np.cumsum(tch)]
    rprev = np.r_[np.nan, r[:-1]]
    # 무작위 기준(범위 안·기간 안 모든 봉 시가 매수)
    rb = np.nonzero(sc & inper)[0]
    bucket = mod // 30
    RET = exits_for(o, h, c, reg, awake, XI, idead, n, rb, o[rb], l[rb] * 0.99, np.full(len(rb), np.nan), mod)
    for g, x in enumerate(TPG):
        px, _ = scan(o, h, c, reg, awake, rb, o[rb] * (1 + x), np.full(len(rb), -INF), idead[rb], n)
        RET[f"G{g}"] = px / o[rb] - 1
    R1, R2, R3 = {}, {}, {}
    ud, di = np.unique(dayi[rb], return_inverse=True)
    k3 = di * 48 + bucket[rb]
    for key, ret in RET.items():
        m = np.isfinite(ret)
        sm3 = np.bincount(k3[m], weights=ret[m], minlength=len(ud) * 48)
        ct3 = np.bincount(k3[m], minlength=len(ud) * 48)
        with np.errstate(invalid="ignore", divide="ignore"):
            R3[key] = np.where(ct3 > 0, sm3 / np.maximum(ct3, 1), np.nan)
        sm = np.bincount(bucket[rb][m], weights=ret[m], minlength=48)
        ct = np.bincount(bucket[rb][m], minlength=48)
        with np.errstate(invalid="ignore", divide="ignore"):
            R1[key] = np.where(ct > 0, sm / np.maximum(ct, 1), np.nan)
        sm2 = np.bincount(di[m], weights=ret[m], minlength=len(ud))
        ct2 = np.bincount(di[m], minlength=len(ud))
        with np.errstate(invalid="ignore", divide="ignore"):
            R2[key] = np.where(ct2 > 0, sm2 / np.maximum(ct2, 1), np.nan)
    # 지금 이격매수 거래(정규장 매수)
    bt = [(ei, xi, ent) for ei, xi, ent, px, why in base_sim(A, None, True, True)]
    bei = np.array([x[0] for x in bt], dtype=np.int64)
    bxi = np.array([x[1] for x in bt], dtype=np.int64)
    bent = np.array([x[2] for x in bt], dtype=float)
    earn = react_days(sym, D["days"], E) if E is not None else set()
    dset = set(int(x) for x in D["days"])
    eve = set(int(d_) for d_, hr in (E or {}).get(sym, []) if hr >= 12 and int(d_) in dset)   # 장 마감 뒤 발표일(들고 넘김)
    P[(period, sym, scope)] = dict(n=n, mins=mins, o=o, h=h, l=l, c=c, r=r, rprev=rprev, reg=reg, sc=sc, mod=mod, dayi=dayi,
                                   inper=inper, LV=LV, HV=HV, XI=XI, idead=idead, awake=awake, nv=nv, ctch=ctch, bucket=bucket,
                                   R1=R1, R2=R2, R3=R3, ud=ud, bei=bei, bxi=bxi, bent=bent, gap=gap, earn=earn, eve=eve, tk=tk)


def exits_for(o, h, c, reg, awake, XI, idead, n, e, fill, sl, mid, mod):
    """진입 봉 e·체결가 fill의 매도 세트 수익."""
    out = {}
    for ex in ("RSI8", "RSI8R", "8시", "H1", "H4", "D1"):
        x = XI[ex][e]
        out[ex] = np.where(x >= 0, o[np.maximum(x, 0)] / fill - 1, np.nan)
    m = len(e)
    for ex, tg, s in (("TP1", fill * 1.01, np.full(m, -INF)), ("TP2", fill * 1.02, np.full(m, -INF)), ("TP2S", fill * 1.02, sl),
                      ("MID", np.where(np.isfinite(mid), mid, INF), np.full(m, -INF))):
        px, _ = scan(o, h, c, reg, awake, e, tg, s, idead[e], n)
        out[ex] = px / fill - 1
    out["MID"] = np.where(np.isfinite(mid), out["MID"], np.nan)
    out["TP2W"] = np.where(mod[e] < 660, out["TP2"], out["8시"])
    return out


def first_per_day(idx, dayi):
    if not len(idx):
        return idx
    d = dayi[idx]
    return idx[np.r_[True, d[1:] != d[:-1]]]


def signals(Q, rule, thr=0.0):
    """반환 e(체결 봉), fill, sl(손절선), mid(박스 중간), lv(지지선)."""
    n, o, h, l, c, sc, dayi = Q["n"], Q["o"], Q["h"], Q["l"], Q["c"], Q["sc"], Q["dayi"]
    lv, hv = Q["LV"][rule["lev"]], Q["HV"][rule["lev"]]
    ok = np.isfinite(lv) & Q["inper"]
    m, p, f = rule["meth"], rule["p"], rule["filt"]
    if f in ("W2", "W4"):
        ok &= (hv / lv - 1) <= (0.02 if f == "W2" else 0.04)
    empty = (np.zeros(0, np.int64),) + (np.zeros(0),) * 4
    if m in ("LIM", "TOUCH"):
        lim = lv * (1 - p) if m == "LIM" else lv * (1 + p)
        cand = ok & ((l < lim * (1 - thr)) if m == "LIM" else (l <= lim * (1 - thr)))
        if f == "RSI":
            cand &= Q["rprev"] <= 30
        i = first_per_day(np.nonzero(cand)[0], dayi)
        if not len(i):
            return empty
        fill = np.minimum(o[i], lim[i])
        sl = (l[i] if m == "LIM" else lv[i]) * 0.99
        return i, fill, sl, (lv[i] + hv[i]) / 2, lv[i]
    if m in ("CLS", "BRK"):
        cand = ok & ((c < lv) if m == "CLS" else (c > hv))
        if f == "RSI":
            cand &= Q["r"] <= 30
        if f == "VOL":
            cand &= Q["nv"] >= 2
        j = first_per_day(np.nonzero(cand)[0], dayi)
        j = j[j + 1 < n]
        e = j + 1
        keep = sc[e] & (dayi[e] == dayi[j])
        j, e = j[keep], e[keep]
        if not len(e):
            return empty
        if m == "CLS":
            return e, o[e], np.minimum(l[j], l[e]) * 0.99, (lv[j] + hv[j]) / 2, lv[j]
        return e, o[e], lv[j] * 0.99, np.full(len(e), np.nan), lv[j]
    # RCL: 첫 이탈 봉 jb, 그 뒤 k분 안 종가 > 이탈 때 지지선 → 다음 봉 시가
    cand = ok & (l < lv)
    if f == "VOL":
        cand &= Q["nv"] >= 2
    if f == "RSI":
        cand &= Q["r"] <= 30
    jb = first_per_day(np.nonzero(cand)[0], dayi)
    kb = max(int(p) // Q["mins"], 1)
    E_, F_, S_, M_, L_ = [], [], [], [], []
    for j in jb:
        Lf = lv[j]
        lowr = l[j]
        for t in range(kb):
            mm = j + t
            if mm >= n or not sc[mm] or dayi[mm] != dayi[j]:
                break
            lowr = min(lowr, l[mm])
            if c[mm] > Lf:
                e = mm + 1
                if e < n and sc[e] and dayi[e] == dayi[j]:
                    E_.append(e); F_.append(o[e]); S_.append(min(lowr, l[e]) * 0.99); M_.append((Lf + hv[j]) / 2); L_.append(Lf)
                break
    if not E_:
        return empty
    return np.array(E_, np.int64), np.array(F_), np.array(S_), np.array(M_), np.array(L_)


def trades(period, rule, scope="reg", syms=None, thr=0.0):
    rows = []
    for (p, sym, s), Q in P.items():
        if p != period or s != scope or (syms is not None and sym not in syms):
            continue
        e, fill, sl, mid, lvv = signals(Q, rule, thr)
        if not len(e):
            continue
        rows.append(pack(Q, sym, e, fill, sl, mid, lvv))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def pack(Q, sym, e, fill, sl, mid, lvv):
    n, o = Q["n"], Q["o"]
    RET = exits_for(o, Q["h"], Q["c"], Q["reg"], Q["awake"], Q["XI"], Q["idead"], n, e, fill, sl, mid, Q["mod"])
    bk = Q["bucket"][e]
    di = np.searchsorted(Q["ud"], Q["dayi"][e])
    di_ok = (di < len(Q["ud"])) & (Q["ud"][np.minimum(di, len(Q["ud"]) - 1)] == Q["dayi"][e])
    di = np.minimum(di, len(Q["ud"]) - 1)
    d = dict(sym=sym, dayi=Q["dayi"][e], e=e, mod=Q["mod"][e], fill=fill, dist=fill / lvv - 1)
    dd = np.where(np.isfinite(mid), mid / fill - 1, np.nan)
    g = np.abs(TPG[None, :] - np.clip(np.nan_to_num(dd, nan=0.0), TPG[0], TPG[-1])[:, None]).argmin(axis=1)
    for ex in EXITS:
        d[ex] = RET[ex]
        if ex == "MID":
            r1 = np.array([Q["R1"][f"G{gg}"][b] for gg, b in zip(g, bk)])
            r2 = np.array([Q["R2"][f"G{gg}"][x] for gg, x in zip(g, di)])
            r3 = np.array([Q["R3"][f"G{gg}"][x * 48 + b] for gg, x, b in zip(g, di, bk)])
            bad = ~np.isfinite(dd) | (dd <= 0)
            r1[bad] = np.nan
            r2[bad] = np.nan
            r3[bad] = np.nan
        else:
            r1, r2, r3 = Q["R1"][ex][bk], Q["R2"][ex][di], Q["R3"][ex][di * 48 + bk]
        d["r1_" + ex] = r1
        d["r2_" + ex] = np.where(di_ok, r2, np.nan)
        d["r3_" + ex] = np.where(di_ok, r3, np.nan)
    K = 30 // Q["mins"]
    ct = Q["ctch"]
    a, b = np.maximum(e - K, 0), np.minimum(e + K, n - 1)
    d["ovl"] = (ct[b + 1] - ct[a]) > 0                              # ±30분 안에 이격매수 선 닿음
    j = np.searchsorted(Q["bei"], e, side="right") - 1
    d["hold"] = (j >= 0) & (Q["bxi"][np.maximum(j, 0)] > e) if len(Q["bei"]) else np.zeros(len(e), bool)
    d["earn"] = np.array([x in Q["earn"] for x in Q["dayi"][e]])
    d["eve"] = np.array([int(x) in Q["eve"] for x in Q["dayi"][e]])
    d["gap"] = Q["gap"][e]
    d["awake"] = Q["mod"][e] < 660                                  # 뉴욕 11:00 전 = 한국 24:00 전(서머타임)
    xr = Q["XI"]["RSI8"][e]
    d["xs_RSI8"] = np.where(xr < 0, -1, np.where(Q["reg"][np.maximum(xr, 0)], 0, np.where(Q["mod"][np.maximum(xr, 0)] < 570, 1, 2)))
    d["xrsi"] = (xr >= 0) & (xr < Q["XI"]["8시"][e])                # RSI로 판 거래(8시 전)
    return pd.DataFrame(d)


def base_trades_x(period, syms=None):
    """지금 이격매수(정규장 매수) 거래를 같은 매도 세트로."""
    rows = []
    for (p, sym, s), Q in P.items():
        if p != period or s != "reg" or (syms is not None and sym not in syms):
            continue
        m = Q["inper"][Q["bei"]] if len(Q["bei"]) else np.zeros(0, bool)
        e, fill = Q["bei"][m], Q["bent"][m]
        if not len(e):
            continue
        rows.append(pack(Q, sym, e, fill, Q["l"][e] * 0.99, np.full(len(e), np.nan), fill))
    return pd.concat(rows, ignore_index=True)


_EMPTY = dict(n=0, avg=np.nan, win=np.nan, worst=np.nan, sum=0.0, t=np.nan, days=0,
              **{f"{a}{b}": np.nan for a in ("rnd", "exc", "te") for b in ("", "3", "2")})


def stat(t, syms, ex="RSI8"):
    """rnd·exc·te = 무작위1(기본: 기간 같은 30분 칸), 3 = 같은 날·같은 칸, 2 = 같은 날."""
    if not len(t):
        return dict(_EMPTY)
    t = t[t.sym.isin(syms)]
    t = t[np.isfinite(t[ex])]
    if not len(t):
        return dict(_EMPTY)
    x = t[ex].values
    out = dict(n=len(t), avg=x.mean(), win=(x > 0).mean(), worst=x.min(), sum=x.sum(), t=BX.dct(t.dayi.values, x), days=t.dayi.nunique())
    for suf, col in (("", "r1_"), ("3", "r3_"), ("2", "r2_")):
        r = t[col + ex].values
        e = x - r
        k = np.isfinite(e)
        out["rnd" + suf] = np.nanmean(r) if k.any() else np.nan
        out["exc" + suf] = e[k].mean() if k.any() else np.nan
        out["te" + suf] = BX.dct(t.dayi.values[k], e[k]) if k.sum() > 2 else np.nan
    return out


def line(s, short=False):
    if s["n"] == 0:
        return "n0"
    if short:
        return f"n{s['n']:>4} 평균 {f_pct(s['avg'])} 초과 {f_pct(s['exc'])}(te {s['te']:+.1f})"
    return (f"n{s['n']:>4} 평균 {f_pct(s['avg'])} 승률 {s['win']:.0%} 최악 {f_pct(s['worst'], 1)} 합 {s['sum']:+.2f} t {s['t']:+.2f}"
            f" | 무작위1 {f_pct(s['rnd'])} 초과 {f_pct(s['exc'])} te {s['te']:+.2f} | 무작위3(같은날·칸) 초과 {f_pct(s['exc3'])} te {s['te3']:+.2f}"
            f" | 무작위2(같은날) 초과 {f_pct(s['exc2'])} te {s['te2']:+.2f}")


# ---------------------------------------------------------------- NKE 사례(분봉을 따로 받음)
def nke_arrays(mins):
    el, rl = (60, 14) if mins == 3 else (36, 9)
    d = ev_prep(ev_bars(SCR, "NKE", mins), el, rl)
    idx = d.index
    kst = idx.tz_convert("Asia/Seoul")
    t8 = kst.normalize() + pd.Timedelta(hours=8)
    dead = np.where(kst < t8, t8.as_unit("ns").asi8, (t8 + pd.Timedelta(days=1)).as_unit("ns").asi8).astype(np.int64)
    A = dict(sym="NKE", mins=mins, idx=idx, o=d.o.values.astype(float), h=d.h.values.astype(float), l=d.l.values.astype(float),
             c=d.c.values.astype(float), v=d.v.values.astype(float), r=d.r.values.astype(float), ema=d.ema.values.astype(float),
             tk=idx.as_unit("ns").asi8, dead=dead, reg=(d.ses.to_numpy(dtype="U8") == "정규"), ses=d.ses.to_numpy(dtype="U8"),
             day=np.asarray(d.date), mod=d["mod"].values.astype(np.int64), hot7=np.zeros(len(d), bool))
    return A


# ---------------------------------------------------------------- 본문
def main():
    t0 = time.time()
    E, ne = earnings_days()
    SE = load("early")
    for sym, A in SE.items():
        daily_table(sym, A)
    for period in PER:
        S = SE if period == "early" else load(period)
        for sym, A in S.items():
            prep(period, sym, A, "reg", E)
    print(f"준비 {time.time() - t0:.0f}s", flush=True)
    RL = rules()
    nB = sum(r["fam"] == "B" for r in RL)
    say("박스권 매매 B — 장중 바닥 이탈 = 바닥 매수(손절 물량 받기) [키 intra_B] 2026-10-09")
    say(__doc__.split("사용:")[0].strip())
    say("")
    say(f"시험 규칙: B {nB}개 + 사후 관찰 B {sum(r['fam'] == 'Bx' for r in RL)}개 + 비교용 A {sum(r['fam'] == 'A' for r in RL)}개 + C {sum(r['fam'] == 'C' for r in RL)}개"
        f" = {len(RL)}개(진입 규칙). 매도 세트 {len(EXITS)}개는 같은 진입에 모두 적용.")
    say(f"고르기는 B {nB}개 × 매도 RSI8 1개 = {nB}개 안에서만 한다. 다른 매도 {len(EXITS) - 1}개 표는 참고(선택에 쓰지 않음).")
    say(f"실적 반응일 원천: box_trade/results/earnings_cache.csv + dip/raw/earnings.parquet ({ne}행)")
    say("표기: n 거래수, 평균 거래당 수익, 승률, 최악, 합 = 수익 합, t = 수익의 날짜 군집 t, 초과 = 평균 - 무작위1(기간 같은 30분 칸), te = 초과의 날짜 군집 t")
    say("")

    # (c) 지금 이격매수
    say("== 비교 기준: 지금 이격매수 지표(정규장 매수, EMA-2% 지정가). 같은 매도 세트로도 잼")
    BASE = {}
    for p in PER:
        BASE[p] = base_trades_x(p)
        for un, us in UNS:
            cells = []
            for ex in ("RSI8", "8시", "H1", "H4", "TP2", "TP2S"):
                s = stat(BASE[p], us, ex)
                cells.append(f"{ex} {f_pct(s['avg'])}/{f_pct(s['exc'])}")
            s = stat(BASE[p], us, "RSI8")
            say(f"  {PLAB[p]} {un}: RSI8 {line(s)}")
            say(f"      매도별 평균/초과: " + " ".join(cells))
    say("")

    # 규칙 격자
    say(f"== 1) 규칙 격자 — 정규장, 매도 RSI8. 칸: n 평균 초과(te). ✓ = 초과 > 0")
    G = []
    T = {}
    for rule in RL:
        rec = dict(rule)
        cells = []
        for p in PER:
            tt = trades(p, rule)
            T[(rule["no"], p)] = tt
            for un, us in UNS:
                for ex in EXITS:
                    s = stat(tt, us, ex)
                    for k_, v_ in s.items():
                        rec[f"{p}_{un}_{ex}_{k_}"] = v_
                s = stat(tt, us, "RSI8")
                if un == "78":
                    cells.append(f"{line(s, True)}{'✓' if s['n'] and s['exc'] > 0 else ' '}")
        G.append(rec)
        say(f"  {rule['no']:>2} {rule['fam']} {rule['name']:<26} | 앞78 {cells[0]} | 뒤78 {cells[1]}")
    G = pd.DataFrame(G)
    G.to_csv(GRID_CSV, index=False, encoding="utf-8-sig")
    say("")
    GB = G[G.fam == "B"]
    say("  두 기간 모두 무작위1보다 나은(초과 > 0) 규칙 수 — 매도별(B 72개 기준, 거래 10건 이상인 규칙만 셈). 괄호 = 무작위3(같은 날·칸) 기준")
    BOTH = {}
    for ex in EXITS:
        cells = []
        for un in ("78", "137"):
            ok = (GB[f"early_{un}_{ex}_n"] >= 10) & (GB[f"late_{un}_{ex}_n"] >= 10)
            b = ok & (GB[f"early_{un}_{ex}_exc"] > 0) & (GB[f"late_{un}_{ex}_exc"] > 0)
            pe = ok & (GB[f"early_{un}_{ex}_exc"] > 0)
            b1 = ok & (GB[f"early_{un}_{ex}_exc3"] > 0) & (GB[f"late_{un}_{ex}_exc3"] > 0)
            BOTH[(ex, un)] = int(b.sum())
            cells.append(f"{un}: 앞 플러스 {int(pe.sum())}/{int(ok.sum())} → 두 기간 {int(b.sum())} ({int(b1.sum())})")
        say(f"    {ex:<5} " + " | ".join(cells))
    say("")
    say("  계열 평균(규칙별 값의 평균, 사용자78) — B가 최근에 더 맞는가. 칸: 평균 / 초과1(플러스 규칙 비율) / 초과3")
    for fam in ("B", "A", "C"):
        g = G[G.fam == fam]
        for ex in ("RSI8", "8시", "TP2W"):
            cells = []
            for p in PER:
                v = g[f"{p}_78_{ex}_exc"]
                cells.append(f"{PSH[p]} {f_pct(g[f'{p}_78_{ex}_avg'].mean())} / {f_pct(v.mean())}({(v > 0).mean():.0%}+) / {f_pct(g[f'{p}_78_{ex}_exc3'].mean())}")
            say(f"    {fam}({len(g)}개) {ex:<5} " + " | ".join(cells))
    for lev in LEVELS:
        cells = []
        for fam in ("B", "A", "C"):
            g = G[(G.fam == fam) & (G.lev == lev)]
            cells.append(f"{fam} 앞 {f_pct(g['early_78_RSI8_exc'].mean())} 뒤 {f_pct(g['late_78_RSI8_exc'].mean())}")
        say(f"    {LEVLAB[lev]:<10} " + " | ".join(cells))
    say("")

    # 최선안
    cand = GB[GB["early_78_RSI8_n"] >= 30].sort_values("early_78_RSI8_te", ascending=False)
    say("== 2) 최선안(앞 기간·정규장·사용자78·RSI8, 30건 이상 중 무작위1 대비 초과 te 최대) — 뒤 기간은 확인만")
    c3 = GB[GB["early_78_RSI8_n"] >= 30].sort_values("early_78_RSI8_te3", ascending=False).iloc[0]
    say(f"  (무작위3(같은 날·칸) 기준으로 고르면 #{int(c3.no)} {c3['name']}: 앞 평균 {f_pct(c3.early_78_RSI8_avg)} 초과3 {f_pct(c3.early_78_RSI8_exc3)}(te {c3.early_78_RSI8_te3:+.2f}) 초과1 {f_pct(c3.early_78_RSI8_exc)}"
        f" → 뒤 평균 {f_pct(c3.late_78_RSI8_avg)} 초과3 {f_pct(c3.late_78_RSI8_exc3)}(te {c3.late_78_RSI8_te3:+.2f}) 초과1 {f_pct(c3.late_78_RSI8_exc)})")
    for _, r_ in cand.head(8).iterrows():
        say(f"    #{int(r_.no):>2} {r_['name']:<26} 앞 n{int(r_.early_78_RSI8_n)} 평균 {f_pct(r_.early_78_RSI8_avg)} 초과 {f_pct(r_.early_78_RSI8_exc)}(te {r_.early_78_RSI8_te:+.2f})"
            f" → 뒤 n{int(r_.late_78_RSI8_n)} 평균 {f_pct(r_.late_78_RSI8_avg)} 초과 {f_pct(r_.late_78_RSI8_exc)}(te {r_.late_78_RSI8_te:+.2f})")
    best = cand.iloc[0]
    bno = int(best.no)
    brule = [r for r in RL if r["no"] == bno][0]
    say(f"  선택: #{bno} {brule['name']}")
    say("")
    aref = [r for r in RL if r["fam"] == "A" and r["lev"] == brule["lev"]][0]
    cref = [r for r in RL if r["fam"] == "C" and r["lev"] == brule["lev"]][0]
    for p in PER:
        for un, us in UNS:
            say(f"  [{PLAB[p]} {un}] 같은 매도끼리: 최선 B / A(같은 지지선) / C(같은 박스 상단) / 이격매수")
            for ex in EXITS:
                sb = stat(T[(bno, p)], us, ex)
                sa = stat(T[(aref["no"], p)], us, ex)
                scc = stat(T[(cref["no"], p)], us, ex)
                sx = stat(BASE[p], us, ex)
                say(f"    {ex:<5} B {line(sb)}")
                say(f"          A {line(sa, True)} | C {line(scc, True)} | 이격 {line(sx, True)}")
    say("")

    # 겹침·실적·갭·시각
    say("== 3) 최선안 거래 쪼개 보기(정규장, RSI8 / TP2W 매도)")
    for p in PER:
        tb = T[(bno, p)]
        for un, us in UNS:
            x = tb[tb.sym.isin(us)]
            if not len(x):
                continue
            say(f"  [{PSH[p]} {un}] n{len(x)} 종목 {x.sym.nunique()} 날 {x.dayi.nunique()} | 이격선 ±30분 겹침 {x.ovl.mean():.0%}, 이격 보유 중 {x.hold.mean():.0%},"
                f" 실적 반응일 {x.earn.mean():.0%}, 갭 -3% 이하 {(x.gap <= -0.03).mean():.0%}, 11시 전(깨어 있음) {x.awake.mean():.0%}, 지지선 대비 체결 중앙 {f_pct(x.dist.median())}")
            for lab, m in (("이격 안 겹침", ~x.ovl), ("이격 겹침", x.ovl), ("실적 반응일", x.earn), ("실적 아님", ~x.earn),
                           ("실적 전날(장후 발표)", x.eve), ("갭 -3% 이하", x.gap <= -0.03), ("갭 아님", x.gap > -0.03),
                           ("11시 전", x.awake), ("11시 뒤", ~x.awake)):
                s1, s2 = stat(x[m], us, "RSI8"), stat(x[m], us, "TP2W")
                say(f"      {lab:<10} RSI8 {line(s1, True)} | TP2W {line(s2, True)}")
    say("")

    # 최근 장세: 주별
    say("== 4) 최근 장세 — 주별(정규장, 사용자78, RSI8): 최선 B / A / C / 이격매수의 평균·초과")
    for p in PER:
        for rr_lab, tt in (("B", T[(bno, p)]), ("A", T[(aref["no"], p)]), ("C", T[(cref["no"], p)]), ("이격", BASE[p])):
            x = tt[tt.sym.isin(U78)]
            if not len(x):
                continue
            wk = pd.Series([date.fromordinal(int(d)).isocalendar()[1] for d in x.dayi], index=x.index)
            cells = []
            for w_, g in x.groupby(wk):
                s = stat(g, U78, "RSI8")
                d0 = date.fromordinal(int(g.dayi.min()))
                cells.append(f"{d0.month}/{d0.day}~ n{s['n']} {f_pct(s['avg'], 1)}/{f_pct(s['exc'], 1)}")
            say(f"  {PSH[p]} {rr_lab:<3} " + " · ".join(cells))
    say("")
    say("  B 계열 전체(72개)에서 뒤 기간 후반(9/24~10/6)만 — 규칙별 초과 평균(사용자78, RSI8):")
    late2 = date(2026, 9, 24).toordinal()
    for fam in ("B", "A", "C"):
        v1, v2 = [], []
        for rule in RL:
            if rule["fam"] != fam:
                continue
            tt = T[(rule["no"], "late")]
            if not len(tt):
                continue
            s1 = stat(tt[tt.dayi < late2], U78, "RSI8")
            s2 = stat(tt[tt.dayi >= late2], U78, "RSI8")
            v1.append(s1["exc"]); v2.append(s2["exc"])
        say(f"    {fam}: 9/10~9/23 {f_pct(np.nanmean(v1))}({np.mean(np.array(v1) > 0):.0%}+) · 9/24~10/6 {f_pct(np.nanmean(v2))}({np.mean(np.array(v2) > 0):.0%}+)")
    say("")

    # 24시간 범위(참고)
    say("== 5) 참고 — 최선안을 24시간 범위(프리·애프터 신호·체결 허용)로. 프리·애프터는 거래량 0·얇은 꼬리라 좋게 나올 수 있음")
    for p in PER:
        S = SE if p == "early" else load(p)
        for sym, A in S.items():
            prep(p, sym, A, "all", E)
        tt = trades(p, brule, "all")
        for un, us in UNS:
            say(f"  {PSH[p]} {un} RSI8 {line(stat(tt, us, 'RSI8'))}")
            say(f"        TP2 {line(stat(tt, us, 'TP2'), True)} | 8시 {line(stat(tt, us, '8시'), True)}")
    say("")

    # NKE 사례
    say("== 6) NKE 사례(분봉 따로 받음: 5분봉 7/16~10/8, 1분봉 9/29~10/8 → 3분봉). 상위 B 규칙이 10/1~10/8에 낸 신호")
    try:
        A5 = nke_arrays(5)
        daily_table("NKE", A5)
        A3 = nke_arrays(3)
        lohi = (date(2026, 9, 30), date(2026, 10, 8))
        for lab, A in (("5분봉", A5), ("3분봉", A3)):
            old = BX.RAW
            BX.RAW = SCR
            try:
                prep("nke" + lab, "NKE", A, "reg", E, lohi)
            finally:
                BX.RAW = old
            Q = P[("nke" + lab, "NKE", "reg")]
            for _, r_ in cand.head(5).iterrows():
                rule = [r for r in RL if r["no"] == int(r_.no)][0]
                e, fill, sl, mid, lvv = signals(Q, rule)
                if not len(e):
                    say(f"  {lab} #{rule['no']} {rule['name']}: 신호 없음")
                    continue
                tb = pack(Q, "NKE", e, fill, sl, mid, lvv)
                for _, q in tb.iterrows():
                    ts = A["idx"][int(q.e)]
                    say(f"  {lab} #{rule['no']} {rule['name']:<22} {ts:%m/%d %H:%M} 체결 {q.fill:.2f}(지지선 {f_pct(q.dist)}) 실적일 {'예' if q.earn else '아니오'}"
                        f" → RSI8 {f_pct(q.RSI8)} 8시 {f_pct(q['8시'])} H4 {f_pct(q.H4)} D1 {f_pct(q.D1)} TP2 {f_pct(q.TP2)}")
            # 10/2 장중 저가 시각
            dd = date(2026, 10, 2)
            m = (A["day"] == dd) & A["reg"]
            if m.any():
                k = np.nonzero(m)[0][np.argmin(A["l"][m])]
                say(f"  {lab} 10/2 정규장 저가 {A['l'][k]:.2f} @ {A['idx'][k]:%H:%M} ET, 시가 {A['o'][np.nonzero(m)[0][0]]:.2f}, 종가 {A['c'][np.nonzero(m)[0][-1]]:.2f}")
    except Exception as ex_:  # NKE 자료 없음
        say(f"  NKE 사례 건너뜀: {ex_!r}")
    say("")
    # 견고성
    say("== 7) 견고성(최선안 #%d %s)" % (bno, brule["name"]))
    for thr in (0.001, 0.002):
        for p in PER:
            tt = trades(p, brule, thr=thr)
            cells = [f"{un} RSI8 {line(stat(tt, us, 'RSI8'), True)} TP2W {line(stat(tt, us, 'TP2W'), True)}" for un, us in UNS]
            say(f"  체결을 지정가보다 {thr * 100:.1f}% 더 내려가야 인정 {PSH[p]}: " + " | ".join(cells))
    lim_rules = [r for r in RL if r["fam"] == "B" and r["meth"] == "LIM"]
    for thr in (0.0, 0.001):
        cnt = {un: 0 for un, _ in UNS}
        for rule in lim_rules:
            st = {p: trades(p, rule, thr=thr) for p in PER}
            for un, us in UNS:
                a, b = stat(st["early"], us, "RSI8"), stat(st["late"], us, "RSI8")
                if a["n"] >= 10 and b["n"] >= 10 and a["exc"] > 0 and b["exc"] > 0:
                    cnt[un] += 1
        say(f"  B 지정가 규칙 {len(lim_rules)}개 중 두 기간 초과 > 0(RSI8), 체결 여유 {thr * 100:.1f}%: 사용자78 {cnt['78']}개, 전체137 {cnt['137']}개")
    for p in PER:
        tb = T[(bno, p)]
        for un, us in UNS:
            x = tb[tb.sym.isin(us)]
            xr = x[x.xrsi]
            ses = {0: "정규", 1: "프리", 2: "애프터"}
            sh = " ".join(f"{ses[k_]} {(xr.xs_RSI8 == k_).mean():.0%}" for k_ in (0, 1, 2)) if len(xr) else "-"
            out = []
            for ex in ("RSI8", "RSI8R", "TP2W"):
                xx = x[np.isfinite(x[ex])]
                exc = (xx[ex] - xx["r1_" + ex]).values
                ds = pd.Series(exc).groupby(xx.dayi.values).sum()
                wo = exc[xx.dayi.values != ds.idxmax()].mean() if len(ds) > 1 else np.nan
                out.append(f"{ex} 초과 {f_pct(np.nanmean(exc))} → 최고일 빼면 {f_pct(wo)}")
            ss = x.groupby("sym").RSI8.sum().sort_values(ascending=False)
            top5 = ss.head(5).sum() / ss.sum() if ss.sum() > 0 else np.nan
            say(f"  [{PSH[p]} {un}] RSI로 판 거래 {len(xr)}/{len(x)}건, 매도 세션 {sh} | " + " · ".join(out)
                + f" | 상위 5종목이 RSI8 합의 {top5:.0%}" + f" ({', '.join(ss.head(5).index)})")
    say("  사후 관찰(고르기 풀 밖, 뒤 기간을 본 뒤 추가) — 같은 5일 저가에서 더 깊은 지정가:")
    for rule in RL:
        if rule["fam"] != "Bx":
            continue
        for p in PER:
            tt = trades(p, rule)
            cells = [f"{un} RSI8 {line(stat(tt, us, 'RSI8'), True)} TP2W {line(stat(tt, us, 'TP2W'), True)}" for un, us in UNS]
            say(f"    #{rule['no']} {rule['name']} {PSH[p]}: " + " | ".join(cells))
    say("")
    say(f"요약 수치: B 규칙 {nB}개 중 두 기간 모두 무작위1 초과 > 0(RSI8, 10건 이상): 사용자78 {BOTH[('RSI8', '78')]}개, 전체137 {BOTH[('RSI8', '137')]}개")
    say(f"소요 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

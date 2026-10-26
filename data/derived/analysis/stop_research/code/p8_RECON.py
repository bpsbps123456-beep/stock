"""8단계 갈래 RECON(2026-10-11): CNN 공포탐욕지수(F&G)를 1990년대부터 다시 만들어 '극단 공포에 사기'가 실패하는 장까지 시험한다.

사용자: "그럼 CNN의 공포탐욕지수를 바탕으로 매매에 참고하면? 아니면 이런 것들에서 아이디어 얻어서 통하는 방법을 개발해도 좋고요."
왜 다시 만드나: 실제 F&G(fg_data.fg())는 2011~라 2000~02·2008 같은 긴 하락장이 없다.

구성요소(CNN 공개 설명 기준, 그날 종가 뒤 알 수 있는 값만, 부호는 '클수록 탐욕'). 달력 = ^GSPC 거래일.
  C1 모멘텀    ^GSPC / 125일선 − 1
  C2 변동성    −(^VIX / 50일선)
  C3 안전자산  S&P 20일 수익 − 미국채 20일 수익. 채권 = ^TNX로 근사(−7 × 20일 금리 변화 + 20일 이자). 2002~ IEF로 대조.
  C4 신용      FRED BAA10Y(하루 늦춤)를 먼저 시도. 이 환경에서 fredgraph 응답이 없으면(2026-10-11 실제로 없음)
               하이일드 펀드 VWEHX 20일 총수익 − 5년 듀레이션 국채(^TNX 근사) 20일 수익, 펀드 기준가라 하루 늦춤. 2007~ HYG−IEF로 대조.
  C5 풋콜      −(CBOE 전체 풋콜 비율 5일 평균): CBOE 공개 CSV(pcratioarchive 1995-09~, totalpcarchive 2003-10~,
               totalpc 2006-11~2019-10-04, 뒤 파일 우선). 2019-10-07~2020-08-31은 자료가 없어 그 요소를 뺀다.
               2020-09-01~는 CNN 구성요소 put_call(이미 5일 평균).
  C6 폭        동일가중 − 시총가중 20일 로그 수익 차: RSP−SPY(배당 수정, 2003-05~), 그 전 ^NYA−^GSPC.
  참고(본 지수에 안 넣음) 신고가 강도: p4_stocks 52주 신고가 − 신저가 순비율(2015~, 지금 살아 있는 종목만 = 생존편향).
               '좁은 장' 확인과 CNN 대조에만 쓴다.
정규화(사전 고정): 요소마다 과거 252거래일 백분위(그날 포함, 최소 126일) → 요소 3개 이상인 날의 같은 가중 평균 → 0~100.
  대안 2개만: 504일 백분위(최소 252), 252일 z점수(±2에서 자름, 50+25z). 실제 F&G 분포에 맞추는 보정은 하지 않는다.
검증: 2011~2026 실제 F&G와 상관, ≤20·≥80 날 겹침, 2026 경로.

규칙(FGDIRECT와 같은 형태, 일부만. 그날 종가 F&G → 다음 날 종가 체결 → 사건형은 H거래일 뒤 종가 매도)
  사건형 E1_T  F&G ≤ T 첫날(직전에 T+10 위였던 뒤에만 새 사건), T ∈ {10,20,25}
         E2_δ  F&G ≤ 20에 있다가 20+δ 위로 올라선 첫날, δ ∈ {0,10}
         E3_G  F&G ≥ G 첫날(직전 G−10 아래였던 뒤) = 탐욕 경고. 덜기 근거가 되려면 아무 날보다 나빠야 한다. G ∈ {75,85}
         보유 H ∈ {21,63,126,252}거래일.
  전략형 S1 평소 w0 ∈ {0.6,0.8}, F&G ≤ T(∈{20,25}) → 100%, F&G ≥ 50 → w0.
         S2 평소 100%, F&G ≥ 80 → w1 ∈ {0.5,0.75}, F&G ≤ 60 → 100%.   비중은 매일 맞춤, 나머지 현금 0%.
  신호 4개: R252(본), R504, RZ(대안), CNN(실제 F&G, 2011~, 비교용).
대상: ^GSPC→SPY(1993-02-01~), ^NDX→QQQ(1999-03-11~), ^SOX→SOXX(2001-07-16~) 1배(5단계 XPROD와 같은 이음, ETF는 배당 수정)와
  3배 합성(lev5.synth와 같은 식: 일일 3배 − 2×^IRX − 보수 0.95% − 추가 비용). 추가 비용 SOXL 1.5%·TQQQ 1.3%(5단계 실제 대조),
  UPRO는 같은 방법으로 이 스크립트에서 잼(1.2%). ^IRX는 p7_idx(1971~).
시점: 지수는 옛 시가가 없어 모두 '다음 날 종가 체결'(전략형은 결정일 t의 비중이 t+2일 수익부터 적용). 하루 늦춘 판은 t+3.
비교 기준
  사건형: 같은 대상·같은 보유 기간 '아무 날 매수'(같은 창의 모든 날) 대비 초과, 승률, 추가 −20% 확률(보유 중 종가 ≤ 진입가×0.8), 사건 수,
          겹치지 않는 사건 수. 고르기 창은 매도까지 창 안에서 끝난 거래만(뒤 가격 누출 없음). 2026 창은 10/9까지 미완 보유를 그날 평가하고
          아무 날도 같은 길이로 맞춘다.
  전략형: 효율선초과 = 규칙 − 같은 최대 낙폭보다 깊지 않은 고정 비중(0~1, 0.01 간격, 매일 맞춤) 최고 수익(긴 창은 연율 %p).
          같은노출 = 같은 평균 노출 고정 비중 대비. 비용: 수수료 0, 미끄러짐 0.5% 민감도(비중 변화 1단위당), 현금 이자 0(4% 판 한 줄).
고르기(사전 고정)
  긴 역사(재구성): 대상·형태마다 1990~2010에서 가장 좋은 규칙(신호 R252·R504·RZ 전부 후보) → 2011~2025 확인(> 0이면 통과).
    사건형 매수(E1·E2): 사건 8건 이상, 아무 날 대비 초과 최대. 탐욕 경고(E3): 사건 8건 이상, 초과 최소(가장 나쁨) → 확인 < 0이면 통과.
    전략형: 비중 변경 4번 이상, 효율선초과 최대.
  긴 역사(실제 F&G): 같은 방법, 2011~2018 고르기(사건 5건·변경 2번 이상) → 2019~2025 확인.
  2026: 대상·형태마다 2026 앞(1/2~5/29)에서 최대(사건·변경 1번 이상) → 뒤(6/1~10/9)·2025 확인. 재구성 신호와 실제 신호 따로.
  대표 후보: 긴 역사(재구성) 고르기 → 확인 통과한 규칙 중 3배 합성에서 확인 구간 지표가 가장 큰 것(전략형·사건형 각각).
위약: 신호(유효 구간)를 원형으로 k×⌊n/21⌋일(k=1..20, 모두 63일 이상) 민 20판. ① 고른 규칙 그대로의 순위(21개 중),
  ② '고르기 → 확인' 절차 전체를 위약 판에서 다시 했을 때 확인 값 분포 대비 실제 순위.
구성요소별 예측력: 각 요소 252일 백분위 ≤10·≥90인 날 뒤 63·126일 수익 − 같은 창 아무 날 평균(1990~2010 / 2011~2025).
'좁은 장': C1 백분위 ≥ 70이고 C6 백분위 ≤ 20인 날(신고가 강도판·CNN 구성요소판 따로) 뒤 21·63·126일 수익 − 아무 날.
  10/9는 CNN 자체 점수로는 모멘텀 72.8이지만 252일 백분위로는 34라, 결과를 보기 전에 변형 하나를 더했다: S&P가 125일선 위(C1 > 0) + C6 ≤ 20.
출력: results/p8_RECON.md·json, 자료 cache/p8_RECON_*.parquet. 실행: PYTHONIOENCODING=utf-8 py p8_RECON.py [fetch|report]
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import fg_data
import lev5

warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
ID = "p8_RECON"

# ============================================================================ 사전 고정 상수(확인 구간을 본 뒤 바꾸지 않는다)
MOM_N, VIX_N, SH_N, CR_N, BR_N, PC_N = 125, 50, 20, 20, 20, 5
DUR_10, DUR_HY = 7.0, 5.0
PCT_W, PCT_W2, Z_W, Z_CLIP = 252, 504, 252, 2.0
MIN_COMPS = 3
NORMS = ("R252", "R504", "RZ")
SIGNALS = NORMS + ("CNN",)
E1_T = (10, 20, 25)
E2_T, E2_D = 20, (0, 10)
E3_G = (75, 85)
REARM = 10
H_LIST = (21, 63, 126, 252)
S1_W0, S1_T, S1_EXIT = (0.6, 0.8), (20, 25), 50
S2_G, S2_W1, S2_BACK = 80, (0.5, 0.75), 60
EXTRA_DD = 0.20
SLIP, CASH4, FEE3 = 0.005, 0.04, 0.0095
EXTRA = {"^NDX": 0.013, "^SOX": 0.015}          # 5단계 실제 대조. ^GSPC(UPRO)는 아래 extra_upro()로 잰다
N_PLA, MIN_SHIFT = 20, 63
W_GRID = np.round(np.arange(0, 1.0001, 0.01), 2)
MIN_EV_R, MIN_EV_C = 8, 5
MIN_TR_R, MIN_TR_C, MIN_TR_26 = 4, 2, 1
COMP_LO, COMP_HI, COMP_H = 10, 90, (63, 126)
NAR_MOM, NAR_BR, NAR_H = 70, 20, (21, 63, 126)
LAG_EV, LAG_ST = 1, 2                            # 사건: s+1 종가 매수. 전략: 결정 t → t+2 수익부터
START_CAL, END = "1985-01-01", "2026-10-09"
SOX_CUT = "2001-07-16"

WIN = {  # (시작, 끝)
    "rsel": ("1990-01-01", "2010-12-31"), "rchk": ("2011-01-01", "2025-12-31"),
    "csel": ("2011-01-01", "2018-12-31"), "cchk": ("2019-01-01", "2025-12-31"),
    "long": ("1990-01-01", END),
    "2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", END), "AI쇼크": ("2026-06-22", "2026-07-29"),
    "2026": ("2026-01-02", END), "2025": ("2025-01-02", "2025-12-31"),
}
ERAS = {
    "1990": ("1990-01-02", "1990-12-31"), "1998": ("1998-07-01", "1998-12-31"), "2000~02": ("2000-03-01", "2002-10-31"),
    "2008": ("2007-10-01", "2009-03-31"), "2011": ("2011-04-01", "2011-12-30"), "2015~16": ("2015-07-01", "2016-06-30"),
    "2018 4분기": ("2018-10-01", "2018-12-31"), "2020": ("2020-01-02", "2020-12-31"), "2022": ("2022-01-03", "2022-12-30"),
    "2025": ("2025-01-02", "2025-12-31"), "2026": ("2026-01-02", END),
}
LONGW = {"rsel", "rchk", "csel", "cchk", "long"}
SELCAP = {"rsel", "csel"}               # 사건형: 매도까지 창 안
TRUNC = {"2026앞", "2026뒤", "AI쇼크", "2026"}  # 사건형: 10/9까지 미완 보유를 그날 평가
TARGETS = [("^GSPC", 1), ("^GSPC", 3), ("^NDX", 1), ("^NDX", 3), ("^SOX", 1), ("^SOX", 3)]
TNAME = {("^GSPC", 1): "S&P500 1배(→SPY)", ("^GSPC", 3): "S&P500 3배(UPRO 대리)", ("^NDX", 1): "나스닥100 1배(→QQQ)",
         ("^NDX", 3): "나스닥100 3배(TQQQ 대리)", ("^SOX", 1): "반도체 1배(SOX→SOXX)", ("^SOX", 3): "반도체 3배(SOXL 대리)"}
TKEY = {k: f"{k[0].strip('^')}{k[1]}" for k in TARGETS}

EV_RULES = [("E1", t) for t in E1_T] + [("E2", d) for d in E2_D] + [("E3", g) for g in E3_G]
ST_RULES = [("S1", w0, t) for w0 in S1_W0 for t in S1_T] + [("S2", w1) for w1 in S2_W1]


def ev_ko(r):
    if r[0] == "E1":
        return f"F&G≤{r[1]} 첫날"
    if r[0] == "E2":
        return f"≤20 뒤 >{20 + r[1]} 첫날"
    return f"F&G≥{r[1]} 첫날(탐욕)"


def st_ko(r):
    if r[0] == "S1":
        return f"평소 {r[1]:.0%}, F&G≤{r[2]}→100%, ≥50→{r[1]:.0%}"
    return f"평소 100%, F&G≥80→{r[1]:.0%}, ≤60→100%"


def rk(r):
    return "_".join(str(x) for x in r)


# ============================================================================ 자료 받기
YF_SYMS = ["^TNX", "^NYA", "RSP", "IEF", "HYG", "VWEHX", "VFITX"]
PC_FILES = [("pcratioarchive.csv", 1), ("totalpcarchive.csv", 4), ("totalpc.csv", 4)]  # (파일, P/C 열 위치). 뒤가 우선
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36"}


def fetch():
    import yfinance as yf
    for _ in range(3):
        try:
            f = yf.download(YF_SYMS, start=START_CAL, end="2026-10-10", auto_adjust=False, progress=False, threads=True,
                            group_by="column")
            break
        except Exception as e:
            print("재시도", e)
            time.sleep(5)
    cols = [f[c].stack(future_stack=True).rename(c) for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume")]
    df = pd.concat(cols, axis=1).reset_index()
    df.columns = ["date", "sym", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df.dropna(subset=["Close"])
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df.to_parquet(CACHE / f"{ID}_yf.parquet")
    print(df.groupby("sym").date.agg(["min", "max", "count"]).to_string())
    # CBOE 풋콜
    rows = []
    for fn, col in PC_FILES:
        url = f"https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/{fn}"
        txt = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode("utf-8", "ignore")
        for line in txt.splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) <= col:
                continue
            try:
                d = pd.to_datetime(p[0], format="%m/%d/%Y")
                v = float(p[col])
            except (ValueError, TypeError):
                continue
            rows.append((d.strftime("%Y-%m-%d"), fn, v))
    pc = pd.DataFrame(rows, columns=["date", "src", "pc"])
    pc.to_parquet(CACHE / f"{ID}_pc.parquet")
    print(pc.groupby("src").date.agg(["min", "max", "count"]).to_string())
    # FRED BAA10Y(한 번 시도)
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAA10Y"
        txt = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90).read().decode()
        from io import StringIO
        fr = pd.read_csv(StringIO(txt))
        fr.columns = ["date", "BAA10Y"]
        fr["BAA10Y"] = pd.to_numeric(fr.BAA10Y, errors="coerce")
        fr.dropna().to_parquet(CACHE / f"{ID}_fred.parquet")
        print("FRED BAA10Y", fr.date.min(), fr.date.max(), len(fr))
    except Exception as e:
        print("FRED BAA10Y 실패 → 펀드 대리값 사용:", type(e).__name__, e)
    strength_build()


def strength_build():
    """p4_stocks 52주(252일) 신고가 − 신저가 순비율(%). 종가 기준, 252일 자료가 쌓인 종목만."""
    d = pd.read_parquet(CACHE / "p4_stocks.parquet", columns=["date", "sym", "Close"])
    w = d.pivot_table(index="date", columns="sym", values="Close", aggfunc="last").sort_index()
    del d
    a = w.to_numpy(float)
    hi = pd.DataFrame(a).rolling(252, min_periods=252).max().to_numpy()
    lo = pd.DataFrame(a).rolling(252, min_periods=252).min().to_numpy()
    ok = np.isfinite(hi) & np.isfinite(a)
    nh = ((a >= hi) & ok).sum(1)
    nl = ((a <= lo) & ok).sum(1)
    cnt = ok.sum(1)
    out = pd.DataFrame({"n_high": nh, "n_low": nl, "n": cnt}, index=w.index.astype(str))
    out["net"] = np.where(cnt >= 200, (nh - nl) / np.maximum(cnt, 1) * 100, np.nan)
    out.to_parquet(CACHE / f"{ID}_strength.parquet")
    print("신고가 강도", out.dropna().index.min(), "~", out.index.max(), "종목 수 중앙", int(np.median(cnt)))


# ============================================================================ 자료 읽기
P7 = pd.read_parquet(CACHE / "p7_idx.parquet")


def p7_close(sym, adj=True) -> pd.Series:
    g = P7[P7.sym == sym].sort_values("date").drop_duplicates("date").dropna(subset=["Close"])
    g = g[g.Close > 0]
    c = g["Adj Close"] if adj else g["Close"]
    return pd.Series(c.to_numpy(float), index=g.date.astype(str).to_numpy())


def yf_close(sym, adj=True) -> pd.Series:
    d = pd.read_parquet(CACHE / f"{ID}_yf.parquet")
    g = d[d.sym == sym].sort_values("date").drop_duplicates("date")
    c = g["Adj Close"] if adj else g["Close"]
    return pd.Series(c.to_numpy(float), index=g.date.astype(str).to_numpy()).dropna()


def on_cal(s: pd.Series, cal, limit=5) -> pd.Series:
    s = s[~s.index.duplicated(keep="last")]
    u = sorted(set(s.index) | set(cal))
    return s.reindex(u).ffill(limit=limit).reindex(cal)


CAL = p7_close("^GSPC").index
CAL = np.asarray(CAL[(CAL >= START_CAL) & (CAL <= END)]).astype(str)
NCAL = len(CAL)


def ix(day, side="left"):
    return int(np.searchsorted(CAL, day, side=side))


def win_ix(w):
    a, z = WIN[w] if w in WIN else ERAS[w]
    return ix(a), ix(z, "right") - 1


# ============================================================================ 구성요소
def components():
    spx = on_cal(p7_close("^GSPC"), CAL)
    vix = on_cal(p7_close("^VIX"), CAL)
    tnx = on_cal(yf_close("^TNX", adj=False), CAL) / 100.0
    C = pd.DataFrame(index=CAL)
    C["C1"] = spx / spx.rolling(MOM_N, min_periods=MOM_N).mean() - 1
    C["C2"] = -(vix / vix.rolling(VIX_N, min_periods=VIX_N).mean())
    bond10 = -DUR_10 * (tnx - tnx.shift(SH_N)) + tnx.shift(SH_N) * SH_N / 252
    C["C3"] = (spx / spx.shift(SH_N) - 1) - bond10
    note = {}
    fp = CACHE / f"{ID}_fred.parquet"
    if fp.exists():
        fr = pd.read_parquet(fp)
        baa = pd.Series(fr.BAA10Y.to_numpy(float), index=fr.date.astype(str).to_numpy())
        C["C4"] = -on_cal(baa, CAL).shift(1)
        note["C4"] = "FRED BAA10Y(하루 늦춤)"
    else:
        hy = on_cal(yf_close("VWEHX"), CAL).shift(1)
        bond5 = -DUR_HY * (tnx - tnx.shift(CR_N)) + tnx.shift(CR_N) * CR_N / 252
        C["C4"] = (hy / hy.shift(CR_N) - 1) - bond5.shift(1)
        note["C4"] = "VWEHX 20일 총수익 − 5년 듀레이션 국채(^TNX) 20일, 하루 늦춤(FRED 응답 없음)"
    # 풋콜
    pc = pd.read_parquet(CACHE / f"{ID}_pc.parquet")
    order = {fn: i for i, (fn, _) in enumerate(PC_FILES)}
    pc["o"] = pc.src.map(order)
    pc = pc.sort_values(["date", "o"]).drop_duplicates("date", keep="last").set_index("date")["pc"].sort_index()
    pc = pc[(pc > 0.1) & (pc < 5)]
    pc5 = pc.rolling(PC_N, min_periods=PC_N).mean()
    cnn = fg_data.components()
    cpc = cnn["put_call"].dropna()
    cpc = cpc[cpc.index >= "2020-09-01"]
    pcs = pd.concat([pc5[pc5.index < "2020-09-01"], cpc])
    C["C5"] = -on_cal(pcs, CAL, limit=3)
    note["C5"] = (f"CBOE 전체 풋콜 5일 평균 {pc.index.min()}~{pc.index.max()}, 빈칸~2020-08, "
                  f"2020-09~ CNN put_call")
    # 폭
    rsp, spy = on_cal(yf_close("RSP"), CAL), on_cal(p7_close("SPY"), CAL)
    nya = on_cal(yf_close("^NYA"), CAL)
    ew = np.log(rsp / rsp.shift(BR_N)) - np.log(spy / spy.shift(BR_N))
    nyx = np.log(nya / nya.shift(BR_N)) - np.log(spx / spx.shift(BR_N))
    C["C6"] = ew.where(ew.notna(), nyx)
    note["C6"] = f"RSP−SPY 20일(첫 값 {ew.first_valid_index()}), 그 전 ^NYA−^GSPC"
    # 대조용
    ief = on_cal(yf_close("IEF"), CAL)
    hyg = on_cal(yf_close("HYG"), CAL)
    X = pd.DataFrame(index=CAL)
    X["C3_IEF"] = (spx / spx.shift(SH_N) - 1) - (ief / ief.shift(SH_N) - 1)
    X["C4_HYGIEF"] = (hyg / hyg.shift(CR_N) - 1) - (ief / ief.shift(CR_N) - 1)
    st = pd.read_parquet(CACHE / f"{ID}_strength.parquet")
    X["STR"] = on_cal(st["net"], CAL, limit=3)
    return C, X, note


def pct_roll(s: pd.Series, W: int) -> pd.Series:
    m = W // 2
    r = s.rolling(W, min_periods=m).rank(method="average", pct=True)
    n = s.rolling(W, min_periods=m).count()
    return (100 * (r - 0.5 / n)).where(s.notna())


def z_roll(s: pd.Series, W: int) -> pd.Series:
    mu = s.rolling(W, min_periods=W // 2).mean()
    sd = s.rolling(W, min_periods=W // 2).std()
    return (50 + 25 * ((s - mu) / sd).clip(-Z_CLIP, Z_CLIP)).where(s.notna())


def normalize(C: pd.DataFrame):
    N = {"R252": C.apply(lambda s: pct_roll(s, PCT_W)), "R504": C.apply(lambda s: pct_roll(s, PCT_W2)),
         "RZ": C.apply(lambda s: z_roll(s, Z_W))}
    F = {}
    for k, P in N.items():
        cnt = P.notna().sum(axis=1)
        F[k] = P.mean(axis=1).where(cnt >= MIN_COMPS)
    return N, F


# ============================================================================ 대상
def rf_cal():
    s = p7_close("^IRX", adj=False) / 100.0
    return on_cal(s, CAL, limit=10).fillna(0.0).to_numpy()


def synth3(c1: np.ndarray, extra: float) -> np.ndarray:
    rf = rf_cal()
    r = c1[1:] / c1[:-1] - 1
    r3 = 3 * r - 2 * rf[1:] / 252 - (FEE3 + extra) / 252
    r3 = np.where(np.isfinite(r3), np.maximum(r3, -0.999), np.nan)
    out = np.full(len(c1), np.nan)
    i0 = int(np.flatnonzero(np.isfinite(c1))[0])
    out[i0] = 100.0
    out[i0 + 1:] = 100.0 * np.cumprod(1 + np.nan_to_num(r3[i0:]))
    return out


SPLICE = {"^GSPC": ("SPY", "1993-02-01"), "^NDX": ("QQQ", "1999-03-11"), "^SOX": ("SOXX", SOX_CUT)}  # 5단계 XPROD와 같은 이음


def base_close(sym) -> np.ndarray:
    """지수 일수익으로 시작해 ETF(배당 수정)가 생긴 뒤 ETF 일수익으로 잇는다(이음 날까지는 지수 수익)."""
    first = on_cal(p7_close(sym), CAL)
    etf, cut = SPLICE[sym]
    if etf == "SOXX":
        sx = lev5.bars("SOXX")
        second = on_cal(pd.Series(sx.c, index=sx.days.astype(str)), CAL)
    else:
        second = on_cal(p7_close(etf), CAL)
    r1 = (first / first.shift(1) - 1).to_numpy()
    r2 = (second / second.shift(1) - 1).to_numpy()
    r = np.where(np.arange(NCAL) <= ix(cut), r1, r2)
    i0 = int(np.flatnonzero(first.notna().to_numpy())[0])
    out = np.full(NCAL, np.nan)
    out[i0] = 100.0
    out[i0 + 1:] = 100.0 * np.cumprod(1 + np.nan_to_num(r[i0 + 1:]))
    return out


def extra_upro():
    """^GSPC 3배 합성(추가 비용 0)과 실제 UPRO의 연 로그 비용 차."""
    c1 = base_close("^GSPC")
    s0 = synth3(c1, 0.0)
    up = lev5.bars("UPRO")
    a = on_cal(pd.Series(up.c, index=up.days.astype(str)), CAL).to_numpy()
    i0, i1 = ix(str(up.days[0])) + 1, NCAL - 1
    yrs = (i1 - i0) / 252
    return float(-np.log((a[i1] / a[i0]) / (s0[i1] / s0[i0])) / yrs), str(CAL[i0])


class Target:
    def __init__(self, key, c):
        self.key, self.name, self.c = key, TNAME[key], c
        self.n = len(c)
        self.r = np.r_[np.nan, c[1:] / c[:-1] - 1]
        self.r0 = np.nan_to_num(self.r)
        self.i0 = int(np.flatnonzero(np.isfinite(c))[0])
        self._fwd, self._fr = {}, {}

    def fwd(self, H):
        if H not in self._fwd:
            c = pd.Series(self.c)
            ret = (c.shift(-H) / c - 1).to_numpy()
            mn = c[::-1].rolling(H + 1, min_periods=H + 1).min()[::-1].to_numpy()
            self._fwd[H] = (ret, mn <= (1 - EXTRA_DD) * self.c)
        return self._fwd[H]

    def frontier(self, a, z, cash=0.0):
        key = (a, z, cash)
        if key not in self._fr:
            rr = self.r0[a:z + 1]
            lg = np.log1p(W_GRID[:, None] * rr[None, :] + (1 - W_GRID[:, None]) * cash / 252)
            v = np.exp(np.concatenate([np.zeros((len(W_GRID), 1)), np.cumsum(lg, 1)], 1))
            ret = v[:, -1] - 1
            mdd = (v / np.maximum.accumulate(v, 1) - 1).min(1)
            cagr = (1 + ret) ** (252 / (z - a + 1)) - 1
            self._fr[key] = (ret, mdd, cagr)
        return self._fr[key]


def make_targets():
    T, ex = {}, {}
    up, up0 = extra_upro()
    ex["^GSPC"] = round(up, 3)          # 잰 값을 0.1% 단위로
    ex.update(EXTRA)
    for sym, L in TARGETS:
        c1 = base_close(sym)
        T[(sym, L)] = Target((sym, L), c1 if L == 1 else synth3(c1, ex[sym]))
    return T, {"UPRO_잰값": up, "UPRO_시작": up0, "사용": ex}


# ============================================================================ 규칙
def hyst(F: np.ndarray, on: np.ndarray, off: np.ndarray) -> np.ndarray:
    st = np.where(on, 1.0, np.where(off, 0.0, np.nan))
    st = pd.Series(st).ffill().fillna(0.0).to_numpy()
    return st


def events(F: np.ndarray, r) -> np.ndarray:
    ok = np.isfinite(F)
    Fv = np.where(ok, F, np.nan)
    with np.errstate(invalid="ignore"):
        if r[0] == "E1":
            T = r[1]
            z = hyst(F, ok & (Fv <= T), ok & (Fv > T + REARM))
            return np.flatnonzero((z == 1) & (np.r_[0.0, z[:-1]] == 0))
        if r[0] == "E2":
            z = hyst(F, ok & (Fv <= E2_T), ok & (Fv > E2_T + r[1]))
            return np.flatnonzero((z == 0) & (np.r_[0.0, z[:-1]] == 1))
        G = r[1]
        z = hyst(F, ok & (Fv >= G), ok & (Fv < G - REARM))
        return np.flatnonzero((z == 1) & (np.r_[0.0, z[:-1]] == 0))


def weights(F: np.ndarray, r) -> np.ndarray:
    ok = np.isfinite(F)
    Fv = np.where(ok, F, np.nan)
    with np.errstate(invalid="ignore"):
        if r[0] == "S1":
            z = hyst(F, ok & (Fv <= r[2]), ok & (Fv >= S1_EXIT))
            return np.where(z == 1, 1.0, r[1])
        z = hyst(F, ok & (Fv >= S2_G), ok & (Fv <= S2_BACK))
        return np.where(z == 1, r[1], 1.0)


# ============================================================================ 사건형 통계
def ev_stats(T: Target, ev, H, w, slip=0.0, lag=LAG_EV, sig0=0):
    a, z = win_ix(w)
    a = max(a, sig0, T.i0)
    if z < a:
        return {"n": 0}
    trunc = w in TRUNC
    cap = z if w in SELCAP else T.n - 1
    s = ev[(ev >= a) & (ev <= z)]
    b = s + lag
    b = b[(b < T.n)]
    b = b[np.isfinite(T.c[b])]
    g = (1 - slip) ** 2
    bb = np.arange(a + lag, z + lag + 1)
    bb = bb[(bb < T.n)]
    bb = bb[np.isfinite(T.c[bb])]
    if trunc:
        q = np.minimum(b + H, T.n - 1)
        keep = q > b
        b, q = b[keep], q[keep]
        if len(b) == 0:
            return {"n": 0}
        ret = T.c[q] / T.c[b] * g - 1
        base = np.empty(len(b))
        for i, (bi, qi) in enumerate(zip(b, q)):
            L = qi - bi
            x = bb[bb + L <= T.n - 1]
            base[i] = np.mean(T.c[x + L] / T.c[x]) * g - 1
        lo = np.array([np.min(T.c[bi:qi + 1]) for bi, qi in zip(b, q)])
        hit = lo <= (1 - EXTRA_DD) * T.c[b]
        b_pos = b_hit = np.nan
        bmean = float(np.mean(base))
        ex = ret - base
    else:
        fr, fh = T.fwd(H)
        b = b[(b + H <= cap)]
        b = b[np.isfinite(fr[b])]
        if len(b) == 0:
            return {"n": 0}
        q = b + H
        ret = (1 + fr[b]) * g - 1
        hit = fh[b]
        x = bb[(bb + H <= cap)]
        x = x[np.isfinite(fr[x])]
        br = (1 + fr[x]) * g - 1
        bmean, b_pos, b_hit = float(br.mean()), float((br > 0).mean()), float(fh[x].mean())
        ex = ret - bmean
    endp, n_ind = -1, 0
    for bi, qi in zip(b, q):
        if bi > endp:
            n_ind += 1
            endp = qi
    k = len(b)
    sd = float(np.std(ex, ddof=1)) if k > 1 else np.nan
    t = float(np.mean(ex) / (sd / np.sqrt(n_ind))) if (k > 1 and sd > 0) else np.nan
    return {"n": int(k), "n_ind": int(n_ind), "mean": float(ret.mean()), "ex": float(np.mean(ex)), "pos": float((ret > 0).mean()),
            "base": bmean, "base_pos": b_pos, "hit20": float(hit.mean()), "base_hit20": b_hit, "t": t,
            "days": [str(CAL[i - lag]) for i in b[:40]]}


# ============================================================================ 전략형
def st_run(T: Target, w: np.ndarray, lag=LAG_ST, slip=0.0, cash=0.0):
    x = np.r_[np.full(lag, w[0]), w[:-lag]]
    R = x * T.r0 + (1 - x) * cash / 252 - slip * np.abs(np.r_[0.0, np.diff(x)])
    return R, x


def st_stats(T: Target, R, x, w, sig0, cash=0.0):
    a, z = win_ix(w)
    a = max(a, sig0 + LAG_ST + 1, T.i0 + 1)
    if z - a < 5:
        return None
    seg = R[a:z + 1]
    v = np.r_[1.0, np.cumprod(1 + seg)]
    ret = v[-1] - 1
    mdd = float((v / np.maximum.accumulate(v) - 1).min())
    cagr = (1 + ret) ** (252 / (z - a + 1)) - 1
    fret, fmdd, fcagr = T.frontier(a, z, cash)
    lw = w in LONGW
    met, fm = (cagr, fcagr) if lw else (ret, fret)
    eff = met - fm[fmdd >= mdd - 1e-12].max()
    expo = float(x[a:z + 1].mean())
    same = met - np.interp(expo, W_GRID, fm)
    ch = int((np.abs(np.diff(x[a - 1:z + 1])) > 1e-9).sum())
    return {"ret": float(ret), "cagr": float(cagr), "mdd": mdd, "eff": float(eff), "same": float(same), "expo": expo,
            "sw": ch, "spy": ch / ((z - a + 1) / 252), "full_ret": float(fret[-1]), "full_mdd": float(fmdd[-1]),
            "full_cagr": float(fcagr[-1])}


# ============================================================================ 그리드
EV_WINS = ["rsel", "rchk", "csel", "cchk", "2026앞", "2026뒤", "AI쇼크", "2026", "2025"] + list(ERAS)
ST_WINS = ["rsel", "rchk", "csel", "cchk", "long", "2026앞", "2026뒤", "AI쇼크", "2026", "2025"] + list(ERAS)


def sig_start(F):
    ok = np.flatnonzero(np.isfinite(F))
    return int(ok[0]) if len(ok) else NCAL


def grid(T: dict, SIG: dict, ev_wins=EV_WINS, st_wins=ST_WINS, sigs=SIGNALS):
    EVR, STR = [], []
    for sg in sigs:
        F = SIG[sg]
        s0 = sig_start(F)
        evs = {rk(r): events(F, r) for r in EV_RULES}
        wts = {rk(r): weights(F, r) for r in ST_RULES}
        for tk, Tg in T.items():
            for r in EV_RULES:
                for H in H_LIST:
                    for w in ev_wins:
                        st = ev_stats(Tg, evs[rk(r)], H, w, sig0=s0)
                        EVR.append({"sig": sg, "tgt": TKEY[tk], "rule": rk(r), "H": H, "win": w, **{k: v for k, v in st.items() if k != "days"},
                                    "days": st.get("days")})
            for r in ST_RULES:
                R, x = st_run(Tg, wts[rk(r)])
                for w in st_wins:
                    st = st_stats(Tg, R, x, w, s0)
                    if st:
                        STR.append({"sig": sg, "tgt": TKEY[tk], "rule": rk(r), "win": w, **st})
    return pd.DataFrame(EVR), pd.DataFrame(STR)


# ============================================================================ 고르기
def pick_ev(E: pd.DataFrame, tgt, sigs, sel, min_n, greed=False):
    q = E[(E.tgt == tgt) & (E.sig.isin(sigs)) & (E.win == sel) & (E.n >= min_n)]
    q = q[q.rule.str.startswith("E3") if greed else ~q.rule.str.startswith("E3")]
    q = q[np.isfinite(q.ex)]
    if not len(q):
        return None
    return q.sort_values("ex", ascending=greed).iloc[0].to_dict()


def pick_st(S: pd.DataFrame, tgt, sigs, sel, min_sw):
    q = S[(S.tgt == tgt) & (S.sig.isin(sigs)) & (S.win == sel) & (S.sw >= min_sw)]
    if not len(q):
        return None
    return q.sort_values("eff", ascending=False).iloc[0].to_dict()


def cell(D, **kw):
    q = D
    for k, v in kw.items():
        q = q[q[k] == v]
    return q.iloc[0].to_dict() if len(q) else None


def shifts(n):
    step = n // (N_PLA + 1)
    return [k * step for k in range(1, N_PLA + 1)] if step >= MIN_SHIFT else []


def shifted(F: np.ndarray, k: int) -> np.ndarray:
    s0 = sig_start(F)
    out = F.copy()
    v = F[s0:]
    out[s0:] = np.roll(v, k)
    return out


def rank_of(x, arr):
    arr = [a for a in arr if a is not None and np.isfinite(a)]
    return 1 + sum(a > x for a in arr), len(arr) + 1


# ============================================================================ 구성요소 예측력·좁은 장
def cond_fwd(T: Target, mask: np.ndarray, H, w):
    a, z = win_ix(w)
    a = max(a, T.i0)
    cap = z if w in SELCAP else T.n - 1
    fr, _ = T.fwd(H)
    d = np.arange(a, z + 1)
    b = d + 1
    ok = (b + H <= cap) & np.isfinite(fr[np.minimum(b, T.n - 1)])
    d, b = d[ok], b[ok]
    m = mask[d]
    if m.sum() == 0:
        return {"days": 0}
    st, al = fr[b][m], fr[b]
    idx = d[m]
    ep = 1 + int((np.diff(idx) > 21).sum())
    sd = float(np.std(st, ddof=1)) if len(st) > 1 else np.nan
    return {"days": int(m.sum()), "ep": ep, "mean": float(st.mean()), "base": float(al.mean()), "ex": float(st.mean() - al.mean()),
            "pos": float((st > 0).mean()), "base_pos": float((al > 0).mean()),
            "t": float((st.mean() - al.mean()) / (sd / np.sqrt(ep))) if sd and sd > 0 else np.nan}


# ============================================================================ 출력 도우미
def pct(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def pp(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%p"


def pr(x):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.0f}%"


def rname(rule: str):
    p = rule.split("_")
    if p[0] in ("E1", "E2", "E3"):
        return ev_ko((p[0], int(p[1])))
    if p[0] == "S1":
        return st_ko(("S1", float(p[1]), int(p[2])))
    return st_ko(("S2", float(p[1])))


def jclean(o):
    if isinstance(o, dict):
        return {str(k): jclean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jclean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.ndarray):
        return jclean(o.tolist())
    return o


# ============================================================================ 실행
def main():
    t0 = time.time()
    C, X, note = components()
    N, F = normalize(C)
    cnn = fg_data.fg()
    F["CNN"] = cnn.reindex(CAL)
    SIG = {k: v.to_numpy(float) for k, v in F.items()}
    T, extra = make_targets()
    J = {"id": ID, "구성요소": note, "추가비용": extra}
    print("구성요소·정규화", f"{time.time() - t0:.0f}s", {k: str(CAL[sig_start(v)]) for k, v in SIG.items()}, flush=True)

    # ---------------- 검증: 실제 F&G와 비교
    V = {}
    both = pd.DataFrame({"cnn": F["CNN"], **{k: F[k] for k in NORMS}}).loc["2011-01-03":END].dropna()
    for k in NORMS:
        v = {"상관_2011~": float(both[k].corr(both.cnn)), "상관_2021-02~": float(both.loc["2021-02-01":, k].corr(both.loc["2021-02-01":, "cnn"])),
             "상관_변화21일": float(both[k].diff(21).corr(both.cnn.diff(21))), "평균": float(both[k].mean()), "표준편차": float(both[k].std())}
        for th, side in ((10, "lo"), (20, "lo"), (25, "lo"), (75, "hi"), (80, "hi"), (85, "hi")):
            a = both.cnn <= th if side == "lo" else both.cnn >= th
            b = both[k] <= th if side == "lo" else both[k] >= th
            v[f"{'≤' if side == 'lo' else '≥'}{th}"] = {"실제일수": int(a.sum()), "재구성일수": int(b.sum()),
                                                      "실제중_재구성도": float((a & b).sum() / max(a.sum(), 1)),
                                                      "재구성중_실제도": float((a & b).sum() / max(b.sum(), 1))}
        V[k] = v
    V["실제_평균"] = float(both.cnn.mean())
    V["실제_표준편차"] = float(both.cnn.std())
    rec_all = F["R252"].loc["1990-01-01":END].dropna()
    V["R252_1990~_분포"] = {f"≤{t}": int((rec_all <= t).sum()) for t in (10, 20, 25)} | {f"≥{t}": int((rec_all >= t).sum()) for t in (75, 80, 85)}
    V["R252_≤20_해별"] = rec_all[rec_all <= 20].groupby(rec_all[rec_all <= 20].index.str[:4]).size().to_dict()
    # 구성요소 원값 대조(CNN 2021-02~)
    cc = fg_data.components().loc["2021-02-01":END]
    ccal = cc.reindex(CAL).loc["2021-02-01":END]
    cmp = {"C1~모멘텀": C.C1.loc["2021-02-01":END].corr(ccal.momentum_sp500 / ccal.momentum_sp125 - 1),
           "C2~변동성": C.C2.loc["2021-02-01":END].corr(-(ccal.vix / ccal.vix_50)),
           "C3~안전자산": C.C3.loc["2021-02-01":END].corr(ccal.safe_haven),
           "C4~정크(수준)": C.C4.loc["2021-02-01":END].corr(-ccal.junk),
           "C4~정크(20일 변화)": C.C4.loc["2021-02-01":END].corr(-ccal.junk.diff(20)),
           "C6~맥클렐런폭": C.C6.loc["2021-02-01":END].corr(ccal.breadth),
           "C6~맥클렐런폭(20일 변화)": C.C6.loc["2021-02-01":END].corr(ccal.breadth.diff(20)),
           "C6~신고가강도": C.C6.loc["2021-02-01":END].corr(ccal.strength),
           "신고가강도(p4)~CNN강도": X.STR.loc["2021-02-01":END].corr(ccal.strength),
           "C3~IEF판(2002~)": C.C3.corr(X.C3_IEF), "C4~HYG−IEF(2007~)": C.C4.corr(X.C4_HYGIEF)}
    # 백분위끼리(CNN이 실제로 쓰는 형태에 가깝게)
    cnn_pct = {"mom": pct_roll(ccal.momentum_sp500 / ccal.momentum_sp125 - 1, PCT_W), "vix": pct_roll(-(ccal.vix / ccal.vix_50), PCT_W),
               "sh": pct_roll(ccal.safe_haven, PCT_W), "junk": pct_roll(-ccal.junk, PCT_W), "br": pct_roll(ccal.breadth, PCT_W),
               "str": pct_roll(ccal.strength, PCT_W), "pc": pct_roll(-ccal.put_call, PCT_W)}
    Pn = N["R252"].loc["2021-02-01":END]
    cmp.update({"백분위 C1~모멘텀": Pn.C1.corr(cnn_pct["mom"]), "백분위 C2~변동성": Pn.C2.corr(cnn_pct["vix"]),
                "백분위 C3~안전자산": Pn.C3.corr(cnn_pct["sh"]), "백분위 C4~정크": Pn.C4.corr(cnn_pct["junk"]),
                "백분위 C6~폭": Pn.C6.corr(cnn_pct["br"]), "백분위 C6~강도": Pn.C6.corr(cnn_pct["str"])})
    V["요소대조"] = {k: float(v) for k, v in cmp.items()}
    # 2026 경로
    p26 = pd.DataFrame({"실제": F["CNN"], "R252": F["R252"], "R504": F["R504"], "RZ": F["RZ"]}).loc["2026-01-02":END]
    m = p26.groupby(p26.index.str[:7]).agg(["min", "mean", "max"])
    V["2026월별"] = {mo: {c: [float(m.loc[mo, (c, s)]) for s in ("min", "mean", "max")] for c in ("실제", "R252", "R504", "RZ")} for mo in m.index}
    V["2026상관"] = {k: float(p26[k].corr(p26["실제"])) for k in NORMS}
    keyd = ["2026-03-30", "2026-05-01", "2026-06-26", "2026-07-02", "2026-07-29", "2026-08-05", "2026-10-09"]
    V["2026핵심일"] = {d: {c: (float(p26.loc[d, c]) if d in p26.index and np.isfinite(p26.loc[d, c]) else None) for c in p26.columns} for d in keyd}
    V["2026최저"] = {c: [str(p26[c].idxmin()), float(p26[c].min())] for c in p26.columns}
    V["2026최고"] = {c: [str(p26[c].idxmax()), float(p26[c].max())] for c in p26.columns}
    V["10/9요소백분위"] = {c: float(N["R252"].loc[END, c]) for c in C.columns}
    J["검증"] = V
    print("검증", {k: round(V[k]["상관_2011~"], 3) for k in NORMS}, flush=True)

    # ---------------- 그리드
    E, S = grid(T, SIG)
    print("그리드", len(E), len(S), f"{time.time() - t0:.0f}s", flush=True)
    n_ev = len(EV_RULES) * len(H_LIST) * len(TARGETS)
    n_st = len(ST_RULES) * len(TARGETS)
    J["격자"] = {"사건형_규칙": len(EV_RULES), "보유": list(H_LIST), "전략형_규칙": len(ST_RULES), "대상": len(TARGETS),
               "신호": list(SIGNALS), "사건형_조합_신호당": n_ev, "전략형_조합_신호당": n_st,
               "재구성_조합": (n_ev + n_st) * len(NORMS), "실제_조합": n_ev + n_st, "전체": (n_ev + n_st) * len(SIGNALS)}

    # ---------------- 고르기
    PK = {}
    for tk in TARGETS:
        t = TKEY[tk]
        PK[(t, "ev_r")] = pick_ev(E, t, NORMS, "rsel", MIN_EV_R)
        PK[(t, "g_r")] = pick_ev(E, t, NORMS, "rsel", MIN_EV_R, greed=True)
        PK[(t, "st_r")] = pick_st(S, t, NORMS, "rsel", MIN_TR_R)
        PK[(t, "ev_c")] = pick_ev(E, t, ("CNN",), "csel", MIN_EV_C)
        PK[(t, "g_c")] = pick_ev(E, t, ("CNN",), "csel", MIN_EV_C, greed=True)
        PK[(t, "st_c")] = pick_st(S, t, ("CNN",), "csel", MIN_TR_C)
        PK[(t, "ev_26r")] = pick_ev(E, t, NORMS, "2026앞", 1)
        PK[(t, "st_26r")] = pick_st(S, t, NORMS, "2026앞", MIN_TR_26)
        PK[(t, "ev_26c")] = pick_ev(E, t, ("CNN",), "2026앞", 1)
        PK[(t, "st_26c")] = pick_st(S, t, ("CNN",), "2026앞", MIN_TR_26)
    CHKW = {"ev_r": "rchk", "g_r": "rchk", "st_r": "rchk", "ev_c": "cchk", "g_c": "cchk", "st_c": "cchk",
            "ev_26r": "2026뒤", "st_26r": "2026뒤", "ev_26c": "2026뒤", "st_26c": "2026뒤"}
    SELW = {"ev_r": "rsel", "g_r": "rsel", "st_r": "rsel", "ev_c": "csel", "g_c": "csel", "st_c": "csel",
            "ev_26r": "2026앞", "st_26r": "2026앞", "ev_26c": "2026앞", "st_26c": "2026앞"}

    def look(kind, p, w):
        if p is None:
            return None
        if kind.startswith("st"):
            return cell(S, sig=p["sig"], tgt=p["tgt"], rule=p["rule"], win=w)
        return cell(E, sig=p["sig"], tgt=p["tgt"], rule=p["rule"], H=p["H"], win=w)

    # ---------------- 위약(① 같은 규칙, ② 고르기 절차 전체)
    t1 = time.time()
    sh = {sg: shifts(NCAL - sig_start(SIG[sg])) for sg in SIGNALS}
    PLA_E, PLA_S = [], []
    for j in range(N_PLA):
        SIGj = {sg: shifted(SIG[sg], sh[sg][j]) for sg in SIGNALS if sh[sg]}
        Ej, Sj = grid(T, SIGj, ev_wins=["rsel", "rchk", "csel", "cchk", "2026앞", "2026뒤"],
                      st_wins=["rsel", "rchk", "csel", "cchk", "2026앞", "2026뒤"], sigs=list(SIGj))
        Ej["pj"], Sj["pj"] = j, j
        PLA_E.append(Ej.drop(columns=["days"]))
        PLA_S.append(Sj)
        print(f"  위약 {j + 1}/{N_PLA} {time.time() - t1:.0f}s", flush=True)
    PE, PS = pd.concat(PLA_E), pd.concat(PLA_S)
    J["위약_밀기"] = {sg: sh[sg] for sg in SIGNALS}

    def pla_same(kind, p, w):
        if p is None:
            return None
        real = look(kind, p, w)
        if real is None:
            return None
        met = "eff" if kind.startswith("st") else "ex"
        if kind.startswith("st"):
            q = PS[(PS.sig == p["sig"]) & (PS.tgt == p["tgt"]) & (PS.rule == p["rule"]) & (PS.win == w)]
        else:
            q = PE[(PE.sig == p["sig"]) & (PE.tgt == p["tgt"]) & (PE.rule == p["rule"]) & (PE.H == p["H"]) & (PE.win == w)]
        arr = q[met].to_numpy(float)
        x = real.get(met)
        if x is None or not np.isfinite(x):
            return None
        if kind.startswith("g"):
            r_, n_ = rank_of(-x, list(-arr))
        else:
            r_, n_ = rank_of(x, list(arr))
        return {"순위": r_, "판수": n_, "실제": x, "위약중앙": float(np.nanmedian(arr)) if len(arr) else None}

    def pla_proc(kind, tgt):
        """위약 판마다 같은 고르기 → 확인 값. 실제 확인 값의 순위."""
        p = PK[(tgt, kind)]
        if p is None:
            return None
        selw, chkw = SELW[kind], CHKW[kind]
        sigs = ("CNN",) if kind.endswith("c") else NORMS
        minv = {"ev_r": MIN_EV_R, "g_r": MIN_EV_R, "st_r": MIN_TR_R, "ev_c": MIN_EV_C, "g_c": MIN_EV_C, "st_c": MIN_TR_C,
                "ev_26r": 1, "st_26r": MIN_TR_26, "ev_26c": 1, "st_26c": MIN_TR_26}[kind]
        vals = []
        for j in range(N_PLA):
            if kind.startswith("st"):
                pj = pick_st(PS[PS.pj == j], tgt, sigs, selw, minv)
                c = None if pj is None else cell(PS[PS.pj == j], sig=pj["sig"], tgt=tgt, rule=pj["rule"], win=chkw)
                vals.append(None if c is None else c["eff"])
            else:
                pj = pick_ev(PE[PE.pj == j], tgt, sigs, selw, minv, greed=kind.startswith("g"))
                c = None if pj is None else cell(PE[PE.pj == j], sig=pj["sig"], tgt=tgt, rule=pj["rule"], H=pj["H"], win=chkw)
                vals.append(None if c is None or c.get("n", 0) == 0 else c["ex"])
        real = look(kind, p, chkw)
        met = "eff" if kind.startswith("st") else "ex"
        x = None if real is None else real.get(met)
        if x is None or not np.isfinite(x):
            return {"위약확인": vals}
        if kind.startswith("g"):
            r_, n_ = rank_of(-x, [-v for v in vals if v is not None])
        else:
            r_, n_ = rank_of(x, vals)
        return {"순위": r_, "판수": n_, "위약확인중앙": float(np.nanmedian([v for v in vals if v is not None])) if any(v is not None for v in vals) else None}

    SELJ = {}
    for (t, kind), p in PK.items():
        if p is None:
            SELJ[f"{t}|{kind}"] = None
            continue
        d = {"신호": p["sig"], "규칙": p["rule"], "규칙말": rname(p["rule"]), "H": p.get("H"),
             "고르기": {k: p.get(k) for k in ("n", "n_ind", "ex", "pos", "hit20", "base_hit20", "eff", "same", "cagr", "mdd", "sw", "spy", "expo")
                     if k in p},
             "확인": look(kind, p, CHKW[kind]), "2026앞": look(kind, p, "2026앞"), "2026뒤": look(kind, p, "2026뒤"),
             "AI쇼크": look(kind, p, "AI쇼크"), "2025": look(kind, p, "2025"),
             "위약_같은규칙_고르기창": pla_same(kind, p, SELW[kind]), "위약_같은규칙_확인창": pla_same(kind, p, CHKW[kind]),
             "위약_절차": pla_proc(kind, t)}
        if kind in ("ev_r", "g_r", "st_r"):
            d["시대별"] = {e: look(kind, p, e) for e in ERAS}
            d["2026앞_r"], d["2026뒤_r"] = look(kind, p, "2026앞"), look(kind, p, "2026뒤")
        SELJ[f"{t}|{kind}"] = d
    J["고르기"] = SELJ

    # 고르기 정보(고르기 > 0 중 확인 > 0 비율, 순위상관)
    INFO = {}
    for t in [TKEY[k] for k in TARGETS]:
        for kind, sigs, sw_, cw_, mn in (("st", NORMS, "rsel", "rchk", MIN_TR_R), ("st", ("CNN",), "csel", "cchk", MIN_TR_C),
                                         ("ev", NORMS, "rsel", "rchk", MIN_EV_R), ("ev", ("CNN",), "csel", "cchk", MIN_EV_C),
                                         ("st", NORMS, "2026앞", "2026뒤", 1), ("ev", NORMS, "2026앞", "2026뒤", 1)):
            if kind == "st":
                a = S[(S.tgt == t) & S.sig.isin(sigs) & (S.win == sw_) & (S.sw >= mn)].set_index(["sig", "rule"])["eff"]
                b = S[(S.tgt == t) & S.sig.isin(sigs) & (S.win == cw_)].set_index(["sig", "rule"])["eff"]
            else:
                q = E[~E.rule.str.startswith("E3")]
                a = q[(q.tgt == t) & q.sig.isin(sigs) & (q.win == sw_) & (q.n >= mn)].set_index(["sig", "rule", "H"])["ex"]
                b = q[(q.tgt == t) & q.sig.isin(sigs) & (q.win == cw_) & (q.n >= 1)].set_index(["sig", "rule", "H"])["ex"]
            jn = pd.concat([a.rename("s"), b.rename("c")], axis=1, join="inner").dropna()
            key = f"{t}|{kind}|{sw_}"
            if len(jn) < 3:
                INFO[key] = {"후보": int(len(jn))}
                continue
            pos = jn[jn.s > 0]
            INFO[key] = {"후보": int(len(jn)), "고르기>0": int(len(pos)),
                         "그중_확인>0": float((pos.c > 0).mean()) if len(pos) else None,
                         "전체_확인>0": float((jn.c > 0).mean()), "순위상관": float(jn.s.rank().corr(jn.c.rank())),
                         "확인평균": float(jn.c.mean())}
    J["고르기정보"] = INFO

    # ---------------- 대표 후보(재구성 긴 역사, 3배)
    rep = {}
    for kind in ("st_r", "ev_r"):
        best = None
        for sym, L in TARGETS:
            if L != 3:
                continue
            d = SELJ.get(f"{TKEY[(sym, L)]}|{kind}")
            if not d or not d["확인"]:
                continue
            m = d["확인"].get("eff" if kind == "st_r" else "ex")
            if m is None or not np.isfinite(m) or m <= 0:
                continue
            if best is None or m > best[1]:
                best = (TKEY[(sym, L)], m)
        rep[kind] = best
    J["대표"] = rep

    # ---------------- 민감도(대표·고른 규칙): 미끄러짐, 하루 늦춤, 현금 4%
    def sens(t, kind):
        p = PK[(t, kind)]
        if p is None:
            return None
        tk = [k for k in TARGETS if TKEY[k] == t][0]
        Tg, Fs = T[tk], SIG[p["sig"]]
        s0 = sig_start(Fs)
        r = tuple(p["rule"].split("_"))
        out = {}
        if kind.startswith("st"):
            rr = (r[0], float(r[1])) + ((int(r[2]),) if r[0] == "S1" else ())
            w = weights(Fs, rr)
            for lab, kw in (("미끄러짐0.5%", {"slip": SLIP}), ("하루늦춤", {"lag": LAG_ST + 1}), ("현금4%", {"cash": CASH4})):
                R, x = st_run(Tg, w, **kw)
                out[lab] = {ww: st_stats(Tg, R, x, ww, s0, cash=kw.get("cash", 0.0)) for ww in (SELW[kind], CHKW[kind], "2026뒤", "2025")}
        else:
            ev = events(Fs, (r[0], int(r[1])))
            for lab, kw in (("미끄러짐0.5%", {"slip": SLIP}), ("하루늦춤", {"lag": LAG_EV + 1})):
                out[lab] = {ww: {k: v for k, v in ev_stats(Tg, ev, int(p["H"]), ww, sig0=s0, **kw).items() if k != "days"}
                            for ww in (SELW[kind], CHKW[kind], "2026뒤", "2025")}
        return out

    J["민감도"] = {f"{t}|{k}": sens(t, k) for t in [TKEY[x] for x in TARGETS] for k in ("st_r", "ev_r", "st_c", "ev_c", "st_26r")}

    # ---------------- 7월 AI 쇼크: 규칙이 한 일(날짜)
    sk_a, sk_z = ix("2026-06-15"), ix("2026-08-07", "right") - 1
    shock = {"경로": {str(CAL[i]): {sg: (None if not np.isfinite(SIG[sg][i]) else round(float(SIG[sg][i]), 1)) for sg in SIGNALS}
                      for i in range(sk_a, sk_z + 1)}}
    acts = {}
    for sg in SIGNALS:
        Fs = SIG[sg]
        a26 = ix("2026-01-02")
        for r in EV_RULES:
            ev = events(Fs, r)
            acts[f"{sg}|{rk(r)}"] = [str(CAL[i]) for i in ev if i >= a26]
        for r in ST_RULES:
            w = weights(Fs, r)
            ch = [i for i in range(a26, NCAL) if w[i] != w[i - 1]]
            acts[f"{sg}|{rk(r)}"] = [(str(CAL[i]), float(w[i])) for i in ch]
    shock["2026_사건·비중변경(결정일)"] = acts
    J["7월쇼크"] = shock

    # ---------------- 구성요소별 예측력
    CP = {}
    PC = N["R252"].copy()
    PC["F"] = F["R252"]
    for comp in list(C.columns) + ["F"]:
        v = PC[comp].to_numpy(float)
        for tk in [("^GSPC", 1), ("^NDX", 1), ("^SOX", 1)]:
            for side, m in (("하위10", v <= COMP_LO), ("상위10", v >= COMP_HI)):
                for H in COMP_H:
                    for w in ("rsel", "rchk"):
                        CP[f"{comp}|{TKEY[tk]}|{side}|{H}|{w}"] = cond_fwd(T[tk], m & np.isfinite(v), H, w)
    J["요소예측력"] = CP

    # ---------------- 좁은 장
    NA = {}
    mom = N["R252"]["C1"].to_numpy(float)
    br = N["R252"]["C6"].to_numpy(float)
    strp = pct_roll(X.STR, PCT_W).to_numpy(float)
    cnn_mom = pct_roll((fg_data.components().momentum_sp500 / fg_data.components().momentum_sp125 - 1).reindex(CAL).loc[:END], PCT_W).reindex(CAL).to_numpy(float)
    cnn_br = pct_roll(fg_data.components().breadth.reindex(CAL).loc[:END], PCT_W).reindex(CAL).to_numpy(float)
    cnn_st = pct_roll(fg_data.components().strength.reindex(CAL).loc[:END], PCT_W).reindex(CAL).to_numpy(float)
    cnn_ok = np.asarray(CAL >= "2021-02-01")
    states = {"모멘텀≥70·폭(RSP−SPY)≤20": (mom >= NAR_MOM) & (br <= NAR_BR),
              "모멘텀≥70·신고가강도(p4)≤20": (mom >= NAR_MOM) & (strp <= NAR_BR),
              "125일선위·폭(RSP−SPY)≤20": (C.C1.to_numpy(float) > 0) & (br <= NAR_BR),
              "CNN모멘텀≥70·CNN폭≤20": (cnn_mom >= NAR_MOM) & (cnn_br <= NAR_BR) & cnn_ok,
              "CNN모멘텀≥70·CNN강도≤20": (cnn_mom >= NAR_MOM) & (cnn_st <= NAR_BR) & cnn_ok}
    NA["10/9상태"] = {k: bool(v[ix(END)]) for k, v in states.items()}
    NA["10/9값"] = {"모멘텀백분위": float(mom[ix(END)]), "폭백분위": float(br[ix(END)]), "신고가강도백분위": float(strp[ix(END)]),
                   "CNN모멘텀백분위": float(cnn_mom[ix(END)]), "CNN폭백분위": float(cnn_br[ix(END)]), "CNN강도백분위": float(cnn_st[ix(END)])}
    NA["2026일수"] = {k: int(v[ix("2026-01-02"):].sum()) for k, v in states.items()}
    NA["2026날짜"] = {k: [str(CAL[i]) for i in np.flatnonzero(v) if i >= ix("2026-01-02")] for k, v in states.items()}
    allw = {"rsel": None, "rchk": None, "long": None, **{e: None for e in ERAS}}
    res = {}
    for name, m in states.items():
        for tk in [("^GSPC", 1), ("^NDX", 1), ("^SOX", 1), ("^NDX", 3), ("^SOX", 3)]:
            for H in NAR_H:
                for w in allw:
                    r_ = cond_fwd(T[tk], m, H, w)
                    if r_.get("days", 0):
                        res[f"{name}|{TKEY[tk]}|{H}|{w}"] = r_
    NA["결과"] = res
    J["좁은장"] = NA
    print("요소·좁은 장", f"{time.time() - t0:.0f}s", flush=True)

    # ---------------- 저장
    E.drop(columns=["days"]).to_parquet(CACHE / f"{ID}_grid_ev.parquet")
    S.to_parquet(CACHE / f"{ID}_grid_st.parquet")
    pd.DataFrame({k: v for k, v in F.items()}).to_parquet(CACHE / f"{ID}_index.parquet")
    N["R252"].to_parquet(CACHE / f"{ID}_comp_pct.parquet")
    J["실행초"] = round(time.time() - t0, 1)
    (RESULTS / f"{ID}.json").write_text(json.dumps(jclean(J), ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장", f"{time.time() - t0:.0f}s")
    report()
    return J, E, S


# ============================================================================ 보고서
SIG_KO = {"R252": "재구성(252일 백분위)", "R504": "재구성(504일 백분위)", "RZ": "재구성(z점수)", "CNN": "실제 F&G"}
TGT_KO = {"GSPC1": "S&P500", "GSPC3": "S&P500 3배", "NDX1": "나스닥100", "NDX3": "나스닥100 3배", "SOX1": "반도체",
          "SOX3": "반도체 3배"}


def report():
    J = json.loads((RESULTS / f"{ID}.json").read_text(encoding="utf-8"))
    E = pd.read_parquet(CACHE / f"{ID}_grid_ev.parquet")
    S = pd.read_parquet(CACHE / f"{ID}_grid_st.parquet")
    FX = pd.read_parquet(CACHE / f"{ID}_index.parquet")
    V, G = J["검증"], J["격자"]

    def ev(sg, t, r, H, w):
        return cell(E, sig=sg, tgt=t, rule=r, H=H, win=w)

    def st(sg, t, r, w):
        return cell(S, sig=sg, tgt=t, rule=r, win=w)

    def evtxt(d, n=True):
        if not d or not d.get("n"):
            return "사건 없음"
        return f"{pp(d['ex'])}" + (f"({int(d['n'])}건)" if n else "")

    def sttxt(d):
        if not d:
            return "—"
        return pp(d["eff"])

    def n_events(sg, r):
        f = FX[sg].to_numpy(float)
        ev_ = events(f, (r.split("_")[0], int(r.split("_")[1])))
        d = CAL[ev_]
        d = d[d >= "1990-01-01"]
        yrs = (ix(END) - ix("1990-01-02")) / 252 if sg != "CNN" else (ix(END) - ix("2011-01-03")) / 252
        return len(d), len(d) / yrs

    L = []
    w = L.append
    gp = J["고르기"]
    # ---------------- 사용자 표 행
    rows = [("RZ", "GSPC3", "ev", "E1_10", 126, "대표"), ("RZ", "GSPC1", "ev", "E1_10", 126, ""), ("RZ", "NDX3", "ev", "E1_10", 126, ""),
            ("RZ", "SOX3", "ev", "E1_10", 126, ""), ("R252", "NDX3", "ev", "E1_20", 126, ""), ("R252", "NDX3", "st", "S1_0.8_25", None, ""),
            ("R252", "SOX3", "st", "S1_0.8_25", None, ""), ("R252", "NDX3", "st", "S2_0.5", None, ""),
            ("R252", "NDX3", "ev", "E3_75", 126, "탐욕"), ("CNN", "NDX3", "st", "S1_0.8_25", None, "실제")]
    UT = []
    for sg, t, kind, r, H, tag in rows:
        selw, chkw = ("csel", "cchk") if sg == "CNN" else ("rsel", "rchk")
        if kind == "ev":
            cells = [ev(sg, t, r, H, x) for x in ("2026앞", "2026뒤", "AI쇼크", "2025", selw, chkw)]
            ne_, py_ = n_events(sg, r)
            txt = [evtxt(c) for c in cells] + [f"{ne_}건(연 {py_:.1f})"]
            name = f"{rname(r)} → {H}일 보유"
        else:
            cells = [st(sg, t, r, x) for x in ("2026앞", "2026뒤", "AI쇼크", "2025", selw, chkw)]
            lg = st(sg, t, r, "long" if sg != "CNN" else "cchk")
            txt = [sttxt(c) for c in cells] + [f"연 {lg['spy']:.1f}번" if lg else "—"]
            sk = cells[2]
            txt[2] = "—" if not sk else f"수익 {pct(sk['ret'])}(변경 {sk['sw']}번)"
            name = rname(r)
        UT.append([f"{name} [{SIG_KO[sg]}]" + (" ★" if tag == "대표" else ""), TGT_KO[t], *txt])
    J["사용자표"] = UT

    # ---------------- 본문
    w("# 8단계 RECON: CNN 공포탐욕지수를 1985년부터 다시 만들어 '극단 공포에 사기'를 긴 하락장까지 시험 (2026-10-12)")
    w("")
    w('> 사용자: "그럼 CNN의 공포탐욕지수를 바탕으로 매매에 참고하면? 아니면 이런 것들에서 아이디어 얻어서 통하는 방법을 개발해도 좋고요."')
    w("> 실제 F&G는 2011~라 2000~02·2008 같은 긴 하락장이 없다. CNN이 공개한 구성요소 설명대로 공개 가격으로 다시 만들어 1990~2010으로 고르고 2011~2025로 확인했다.")
    w(f"> 코드 `code/p8_RECON.py`, 자료 `cache/p8_RECON_*.parquet`, 전체 숫자 `results/p8_RECON.json`. 실행 {J.get('실행초', 0):.0f}초.")
    w("")
    w("## 결론")
    w("")
    gs3 = ev("RZ", "GSPC3", "E1_10", 126, "rchk")
    gs3s = ev("RZ", "GSPC3", "E1_10", 126, "rsel")
    g1s, g1c = ev("RZ", "GSPC1", "E1_10", 126, "rsel"), ev("RZ", "GSPC1", "E1_10", 126, "rchk")
    n1s, n1c = ev("RZ", "NDX1", "E1_10", 126, "rsel"), ev("RZ", "NDX1", "E1_10", 126, "rchk")
    pl = gp["GSPC3|ev_r"]
    ne, py = n_events("RZ", "E1_10")
    st_pass = [t for t in ("GSPC1", "GSPC3", "NDX1", "NDX3", "SOX1", "SOX3")
               if gp.get(f"{t}|st_r") and gp[f"{t}|st_r"]["확인"] and gp[f"{t}|st_r"]["확인"]["eff"] > 0]
    st3 = [gp[f"{t}|st_r"]["확인"]["eff"] for t in ("GSPC3", "NDX3", "SOX3")]
    st1 = [gp[f"{t}|st_r"]["확인"]["eff"] for t in ("GSPC1", "NDX1", "SOX1")]
    w(f"1. **재구성 F&G로 비중을 바꾸는 전략(공포에 100%로 올리기·탐욕에 덜기)은 긴 역사에서도 '같은 낙폭 덜 담기'를 못 이겼다.** "
      f"1990~2010에서 고른 전략형 규칙은 2011~2025 확인에서 " + ("6개 대상 모두 마이너스" if not st_pass else f"6개 대상 중 {len(st_pass)}개만 플러스") +
      f"(1배 연 {pp(max(st1))}~{pp(min(st1))}, 3배 연 {pp(max(st3))}~{pp(min(st3))})였다. "
      f"2026에는 재구성 지수가 3월 최저 {V['2026최저']['R252'][1]:.0f}, 7월 쇼크 중 최저 33이라 공포 규칙(≤20·≤25)이 7월에 한 번도 켜지지 않았다.")
    w(f"2. **두 시대에서 버틴 것은 사건형 하나: 재구성 z점수판이 10 이하로 처음 떨어진 날 사서 126거래일 보유.** "
      f"아무 날 대비 S&P500 {pp(g1s['ex'])} → {pp(g1c['ex'])}, 나스닥100 {pp(n1s['ex'])} → {pp(n1c['ex'])}, "
      f"대표 후보 S&P500 3배 {pp(gs3s['ex'])} → {pp(gs3['ex'])}(위약 절차 21판 중 {pl['위약_절차']['순위']}위). "
      f"그러나 1990~2026에 {ne}번(연 {py:.1f}번)뿐이고 2008-10-02(3배 −73%)·2021-11-30(3배 −26%)처럼 하락장 초입에도 켜진다. "
      f"3배는 산 뒤 추가 −20%를 맞을 확률이 아무 날보다 높았다(2011~2025 {pr(gs3['hit20'])} 대 {pr(gs3['base_hit20'])}).")
    w(f"3. **2026 판정은 불가(관찰).** 이 규칙은 2026에 한 번도 켜지지 않았다(z점수판 최저 {V['2026최저']['RZ'][1]:.1f}). "
      f"2025-04-07 한 번이 S&P500 +36%·3배 +127%였다. 새 돈을 넣는 날로 기록만 하고 데스크 규칙·손절 규칙으로 쓰지 않는다. "
      f"10/9는 재구성 {V['2026핵심일']['2026-10-09']['R252']:.0f}(실제 45)로 어떤 신호도 켜져 있지 않다.")
    w("")
    w("## 다시 만든 지수(맞는지)")
    w("")
    k = V["R252"]
    w("| 항목 | 재구성(252일 백분위, 본판) | 비고 |")
    w("|---|---|---|")
    w(f"| 실제 F&G와 상관 2011~2026 | {k['상관_2011~']:.2f} (2021-02~ {k['상관_2021-02~']:.2f}, 21일 변화 {k['상관_변화21일']:.2f}) | 504일판 {V['R504']['상관_2011~']:.2f}, z점수판 {V['RZ']['상관_2011~']:.2f} |")
    w(f"| 실제 ≤20인 날 중 재구성도 ≤20 | {pr(k['≤20']['실제중_재구성도'])} (재구성 ≤20 중 실제도 {pr(k['≤20']['재구성중_실제도'])}) | ≤10은 {pr(k['≤10']['실제중_재구성도'])}·{pr(k['≤10']['재구성중_실제도'])} |")
    w(f"| 실제 ≥80인 날 중 재구성도 ≥80 | {pr(k['≥80']['실제중_재구성도'])} | 탐욕 끝은 덜 맞는다 |")
    w(f"| 2026 최저 | 실제 {V['2026최저']['실제'][1]:.1f}({V['2026최저']['실제'][0][5:]}) · 재구성 {V['2026최저']['R252'][1]:.1f}({V['2026최저']['R252'][0][5:]}) | 평균을 내면 극단이 무뎌진다 |")
    w("| 7월 쇼크 최저(6/22~7/29) | 실제 24.7(6/26) · 재구성 33.1(6/25) | |")
    q = V["10/9요소백분위"]
    w(f"| 10/9 | 실제 45 · 재구성 {V['2026핵심일']['2026-10-09']['R252']:.0f} | 요소 백분위: 모멘텀 {q['C1']:.0f}·VIX {q['C2']:.0f}·안전자산 {q['C3']:.0f}·정크 대리 {q['C4']:.0f}·풋콜 {q['C5']:.0f}·폭 {q['C6']:.0f} |")
    c = V["요소대조"]
    w("")
    w(f"- 요소별(CNN 원값과 2021-02~ 상관): 모멘텀 {c['C1~모멘텀']:.2f}, VIX {c['C2~변동성']:.2f}, 안전자산 {c['C3~안전자산']:.2f}은 같다. "
      f"정크 대리(펀드 20일 수익 차)는 CNN 금리차 수준과 {c['C4~정크(수준)']:.2f}(20일 변화와는 {c['C4~정크(20일 변화)']:.2f}), "
      f"폭 대리(RSP−SPY)는 맥클렐런 폭과 {c['C6~맥클렐런폭']:.2f}로 다르다. FRED가 응답하지 않아 금리차 대신 펀드 기준가를, 맥클렐런 자료가 없어 동일가중 차를 썼다.")
    yr = V["R252_≤20_해별"]
    w(f"- 재구성 ≤20인 날은 1990~2026에 {V['R252_1990~_분포']['≤20']}일(2000~02 {sum(yr.get(x, 0) for x in ('2000', '2001', '2002'))}일, 2008 {yr.get('2008', 0)}일, 2020 {yr.get('2020', 0)}일, 2026 {yr.get('2026', 0)}일).")
    w("")
    w("## 사용자 표")
    w("")
    w("사건형 = 그날 종가 신호 → 다음 날 종가에 사서 보유, 같은 대상·같은 보유 기간 '아무 날 매수' 대비 %p(건수). "
      "전략형 = 효율선초과(같은 최대 낙폭이 나오는 고정 비중 대비 %p, 긴 창은 연율), 7월 쇼크 칸만 그 기간 수익(비중 변경 횟수). "
      "2026 창은 10/9까지 평가. 실제 F&G 행의 고르기·확인은 2011~2018·2019~2025.")
    w("")
    w("| 규칙 | 대상 | 2026 앞(1/2~5/29) | 2026 뒤(6/1~10/9) | 7월 AI 쇼크 | 2025 | 긴 역사 고르기(1990~2010) | 확인(2011~2025) | 사건 수·매매 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in UT:
        w("| " + " | ".join(r) + " |")
    sen = J["민감도"]["GSPC3|ev_r"]
    w("")
    w(f"- ★ 대표 후보. 3배에는 추가 비용(UPRO 1.2%·TQQQ 1.3%·SOXL 1.5%/년)이 들어 있다. 미끄러짐 0.5%면 대표 확인 {pp(sen['미끄러짐0.5%']['rchk']['ex'])}, "
      f"하루 늦게 사면 {pp(sen['하루늦춤']['rchk']['ex'])}. 사건형은 산 기간에만 돈이 들어가 현금 이자 판은 따로 두지 않았다.")
    w("")
    # ---------------- 조합 격자
    w("## 조합 격자(본판 R252, 3배: 1990~2010 고르기 → 2011~2025 확인 → 2026 앞·뒤)")
    w("")
    w(f"시험 수: 사건형 {len(EV_RULES)}규칙 × 보유 {len(H_LIST)}개(21·63·126·252일) × 대상 {len(TARGETS)} × 신호 {len(SIGNALS)} = {G['사건형_조합_신호당'] * len(SIGNALS)}, "
      f"전략형 {len(ST_RULES)} × {len(TARGETS)} × {len(SIGNALS)} = {G['전략형_조합_신호당'] * len(SIGNALS)}, 합 {G['전체']}개(재구성만 {G['재구성_조합']}). "
      f"평가 창 {len(EV_WINS)}개, 위약 {N_PLA}판. 격자는 스크립트 맨 위 상수로 미리 정했고 확인 구간을 본 뒤 바꾸지 않았다.")
    w("")
    w("| 규칙(사건형은 126일 보유) | 나스닥100 3배 1990~2010 | 2011~2025 | 2026 앞 | 2026 뒤 | 반도체 3배 1990~2010 | 2011~2025 | 2026 앞 | 2026 뒤 |")
    w("|---|---|---|---|---|---|---|---|---|")
    WW = ("rsel", "rchk", "2026앞", "2026뒤")
    for r in EV_RULES:
        rr = rk(r)
        a = [ev("R252", t, rr, 126, x) for t in ("NDX3", "SOX3") for x in WW]
        w(f"| {rname(rr)} | " + " | ".join(evtxt(x) for x in a) + " |")
    for r in ST_RULES:
        rr = rk(r)
        a = [st("R252", t, rr, x) for t in ("NDX3", "SOX3") for x in WW]
        txt = []
        for x in a:
            if x and x["sw"] == 0:
                txt.append(f"{sttxt(x)}(변경 0)")
            else:
                txt.append(sttxt(x))
        w(f"| {rname(rr)} | " + " | ".join(txt) + " |")
    w("")
    w("- '(변경 0)'은 그 창에서 비중이 한 번도 안 바뀐 경우다. 이때 효율선초과는 규칙이 아니라 그 창의 등락 때문에 생긴다(떨어진 창에선 낮은 비중이 이긴다).")
    INFO = J["고르기정보"]
    w("")
    w("- 고르기 → 확인이 이어지나(재구성 3판 합쳐, 고르기 창 플러스 규칙 중 확인도 플러스인 비율 · 순위상관):")
    for kind, lab in (("ev", "사건형 매수(60개)"), ("st", "전략형(18개)")):
        parts = []
        for t in ("GSPC1", "GSPC3", "NDX1", "NDX3", "SOX1", "SOX3"):
            d = INFO.get(f"{t}|{kind}|rsel", {})
            if d.get("고르기>0"):
                parts.append(f"{TGT_KO[t]} {pr(d['그중_확인>0'])}·{d['순위상관']:.2f}")
            else:
                parts.append(f"{TGT_KO[t]} 고르기 플러스 없음")
        w(f"  - {lab}: " + ", ".join(parts))
    parts = []
    for t in ("GSPC1", "NDX1", "SOX1", "NDX3"):
        d = INFO.get(f"{t}|ev|csel", {})
        parts.append(f"{TGT_KO[t]} {pr(d.get('그중_확인>0'))}·{d.get('순위상관', float('nan')):.2f}")
    w("  - 실제 F&G 사건형(2011~2018 → 2019~2025): " + ", ".join(parts) + " — 이어지지 않는다.")
    w("")
    # ---------------- 고르기 표
    w("## 고르기 표")
    w("")
    w("### 2026 판정(앞 1/2~5/29에서 고르기 → 뒤 6/1~10/9 확인, 보조 2025)")
    w("")
    SG = {"R252": "백분위", "R504": "504일", "RZ": "z", "CNN": "실제"}

    def short(rule, H=None, sig=None):
        p_ = rule.split("_")
        if p_[0] == "E1":
            s_ = f"≤{p_[1]} 첫날"
        elif p_[0] == "E2":
            s_ = f"≤20 뒤 >{20 + int(p_[1])}"
        elif p_[0] == "E3":
            s_ = f"≥{p_[1]} 첫날"
        elif p_[0] == "S1":
            s_ = f"평소 {float(p_[1]):.0%}·≤{p_[2]}→100%"
        else:
            s_ = f"≥80→{float(p_[1]):.0%}"
        return (f"[{SG[sig]}] " if sig else "") + s_ + (f"·{H}일" if H else "")

    w("셀 = 고른 규칙: 앞 → 뒤 (2025). 사건형은 아무 날 대비 %p, 전략형은 효율선초과 %p.")
    w("")
    w("| 대상 | 전략·재구성 | 전략·실제 F&G | 사건·재구성 | 사건·실제 F&G |")
    w("|---|---|---|---|---|")
    J26 = {}
    for t in ("NDX3", "SOX3", "GSPC3", "NDX1"):
        cells_ = []
        for kind in ("st_26r", "st_26c", "ev_26r", "ev_26c"):
            d = gp.get(f"{t}|{kind}")
            if not d:
                cells_.append("—")
                continue
            if kind.startswith("st"):
                a, b, c_ = (d[x]["eff"] if d[x] else None for x in ("2026앞", "2026뒤", "2025"))
                sw_b = d["2026뒤"]["sw"] if d["2026뒤"] else 0
                verdict = "뒤 변경 0 → 판정 불가" if sw_b == 0 else ("통과" if (b or 0) > 0 else "탈락")
                cells_.append(f"{short(d['규칙'], None, d['신호'])}: {pp(a)} → {pp(b)} ({pp(c_)}) {verdict}")
            else:
                a, b, c_ = d["2026앞"], d["2026뒤"], d["2025"]
                verdict = "뒤 사건 0 → 판정 불가" if not b or not b.get("n") else ("통과" if b["ex"] > 0 else "탈락")
                cells_.append(f"{short(d['규칙'], d['H'], d['신호'])}: {evtxt(a)} → {evtxt(b)} ({evtxt(c_)}) {verdict}")
            J26[f"{t}|{kind}"] = verdict
        w(f"| {TGT_KO[t]} | " + " | ".join(cells_) + " |")
    J["2026판정"] = J26
    w("")
    w("- 2026 앞에는 3월 공포가 있어 규칙이 움직였지만, 뒤(6~10월)에는 재구성 지수가 33 아래로 안 내려가 어떤 고른 규칙도 움직이지 않았다. "
      "앞의 사건형 플러스는 3월 바닥 한두 건이다.")
    w("")
    w("### 긴 역사 판정")
    w("")
    w("셀 = 고른 규칙: 고르기 → 확인 (위약 순위: 같은 규칙 / 고르기 절차 전체, 21판 중, 1이 가장 좋음) 판정. 탐욕경고는 확인이 마이너스여야 통과.")
    w("")
    LS = {}
    for title, kinds, labs in (("재구성: 1990~2010 고르기 → 2011~2025 확인", ("st_r", "ev_r", "g_r"), ("전략형", "사건형 매수", "탐욕경고")),
                               ("실제 F&G: 2011~2018 고르기 → 2019~2025 확인", ("st_c", "ev_c", "g_c"), ("전략형", "사건형 매수", "탐욕경고"))):
        w(f"**{title}**")
        w("")
        w("| 대상 | " + " | ".join(labs) + " |")
        w("|---|---|---|---|")
        for t in ("GSPC1", "GSPC3", "NDX1", "NDX3", "SOX1", "SOX3"):
            cells_ = []
            for kind in kinds:
                d = gp.get(f"{t}|{kind}")
                if not d:
                    cells_.append("—")
                    continue
                met = "eff" if kind.startswith("st") else "ex"
                a = d["고르기"].get(met)
                b = d["확인"].get(met) if d["확인"] else None
                ok = (b is not None) and ((b < 0) if kind.startswith("g") else (b > 0))
                p1 = d["위약_같은규칙_확인창"] or {}
                p2 = d["위약_절차"] or {}
                LS[f"{t}|{kind}"] = {"규칙": rname(d["규칙"]) + (f" → {d['H']}일" if d["H"] else ""), "신호": d["신호"], "고르기": a,
                                     "확인": b, "통과": ok, "위약_같은규칙": p1.get("순위"), "위약_절차": p2.get("순위")}
                cells_.append(f"{short(d['규칙'], d['H'], d['신호'])}: {pp(a)} → {pp(b)} ({p1.get('순위', '—')}/{p2.get('순위', '—')}) "
                              f"{'**통과**' if ok else '탈락'}")
            w(f"| {TGT_KO[t]} | " + " | ".join(cells_) + " |")
        w("")
    J["긴역사판정"] = LS
    w("")
    # ---------------- 시대별
    w("## 시대별: '극단 공포에 사기'가 실패한 장 (126일 보유, 평균 수익 / 그 시대 아무 날 대비, 건수)")
    w("")
    w("| 시대 | 재구성 ≤20 첫날, 나스닥100 | 같은 규칙, 나스닥100 3배 | z점수판 ≤10 첫날, 나스닥100 3배 | z점수판 ≤10, S&P500 3배 |")
    w("|---|---|---|---|---|")
    for e in ERAS:
        cells = [ev("R252", "NDX1", "E1_20", 126, e), ev("R252", "NDX3", "E1_20", 126, e),
                 ev("RZ", "NDX3", "E1_10", 126, e), ev("RZ", "GSPC3", "E1_10", 126, e)]
        txt = ["—" if not x or not x.get("n") else f"{pct(x['mean'])} / {pp(x['ex'])} ({int(x['n'])})" for x in cells]
        w(f"| {e} | " + " | ".join(txt) + " |")
    w("")
    w("- 2026은 10/9까지 미완 보유를 그날 평가. 2000~02는 그 시대 아무 날이 워낙 나빠 '대비'가 플러스여도 절대 수익을 같이 봐야 한다. "
      "2008은 첫 공포 신호 뒤 더 빠져 3배가 크게 잃었다.")
    w("")
    # ---------------- 구성요소별
    CP = J["요소예측력"]
    lab = {"C1": "모멘텀(S&P/125일선)", "C2": "VIX/50일선", "C3": "안전자산(주식−채권 20일)", "C4": "정크 대리(하이일드 펀드−국채 20일)",
           "C5": "풋콜 5일", "C6": "폭(동일가중−시총가중 20일)", "F": "재구성 지수(값 ≤10·≥90)"}
    w("## 구성요소별 예측력 (126일 뒤 수익 − 아무 날, 1990~2010 / 2011~2025, 괄호는 21일 넘게 떨어진 구간 수)")
    w("")
    w("| 요소 | 공포 끝(하위 10%) S&P500 | 탐욕 끝(상위 10%) S&P500 | 공포 끝 나스닥100 | 탐욕 끝 나스닥100 |")
    w("|---|---|---|---|---|")
    for comp in ("C1", "C2", "C3", "C4", "C5", "C6", "F"):
        txt = []
        for t in ("GSPC1", "NDX1"):
            for side in ("하위10", "상위10"):
                a, b = CP.get(f"{comp}|{t}|{side}|126|rsel", {}), CP.get(f"{comp}|{t}|{side}|126|rchk", {})
                txt.append(f"{pp(a.get('ex'))} / {pp(b.get('ex'))} ({a.get('ep', 0)}·{b.get('ep', 0)})")
        w(f"| {lab[comp]} | {txt[0]} | {txt[1]} | {txt[2]} | {txt[3]} |")
    w("")
    w("- 두 시대 같은 방향: 안전자산 공포 끝(+)·탐욕 끝(−), 정크 대리 탐욕 끝(−), 모멘텀 공포 끝(+), 재구성 지수 ≤10(+). "
      "VIX·풋콜·폭은 시대마다 방향이 바뀌었다. 그 상태의 모든 날을 세어 겹침이 크다(구간 수가 독립 표본에 가깝다).")
    w("")
    # ---------------- 좁은 장
    NA = J["좁은장"]
    w("## '좁은 장'(지수는 오르는데 폭은 약함) 뒤 수익 (아무 날 대비)")
    w("")
    w("| 상태 | 10/9 | S&P500 63일 1990~2010 / 2011~2025 | S&P500 126일 | 나스닥100 3배 63일 | 구간 수 |")
    w("|---|---|---|---|---|---|")
    R_ = NA["결과"]
    for name in ("모멘텀≥70·폭(RSP−SPY)≤20", "125일선위·폭(RSP−SPY)≤20", "모멘텀≥70·신고가강도(p4)≤20", "CNN모멘텀≥70·CNN폭≤20", "CNN모멘텀≥70·CNN강도≤20"):
        a63 = (R_.get(f"{name}|GSPC1|63|rsel", {}), R_.get(f"{name}|GSPC1|63|rchk", {}))
        a126 = (R_.get(f"{name}|GSPC1|126|rsel", {}), R_.get(f"{name}|GSPC1|126|rchk", {}))
        n3 = (R_.get(f"{name}|NDX3|63|rsel", {}), R_.get(f"{name}|NDX3|63|rchk", {}))
        w(f"| {name} | {'예' if NA['10/9상태'][name] else '아니오'} | {pp(a63[0].get('ex'))} / {pp(a63[1].get('ex'))} | "
          f"{pp(a126[0].get('ex'))} / {pp(a126[1].get('ex'))} | {pp(n3[0].get('ex'))} / {pp(n3[1].get('ex'))} | "
          f"{a63[0].get('ep', 0)}·{a63[1].get('ep', 0)} |")
    v = NA["10/9값"]
    w("")
    w(f"- 10/9 백분위(지난 252일): 모멘텀 {v['모멘텀백분위']:.0f}, 폭(RSP−SPY) {v['폭백분위']:.0f}, 신고가 강도(p4) {v['신고가강도백분위']:.0f}, "
      f"CNN 폭 {v['CNN폭백분위']:.0f}·강도 {v['CNN강도백분위']:.0f}. CNN 자체 점수는 모멘텀 72.8(탐욕)인데 252일 백분위로는 34라, 사전 정의(모멘텀 ≥70)로는 10/9가 '좁은 장'이 아니다. "
      "가장 가까운 '125일선 위 + 폭 ≤20'(결과 보기 전 추가한 변형)은 1990~2010엔 뒤 수익이 좋았고 2011~2025엔 조금 나빴다(방향이 바뀜) → 경고 신호로 쓸 근거가 없다. "
      "신고가 강도·CNN판은 2015·2021 이후뿐이라 구간이 적다.")
    w("")
    # ---------------- 7월 쇼크
    A = J["7월쇼크"]["2026_사건·비중변경(결정일)"]
    w("## 7월 AI 쇼크(6/22~7/29)에 규칙이 한 일")
    w("")
    w(f"- 재구성 지수(본판)는 6/22 45 → 6/25 33.1(바닥) → 7/10 59 → 7/29 39. 공포 규칙(≤20·≤25), 탐욕 규칙(≥80), z점수판 모두 이 기간 사건 0, 비중 변경 0. "
      f"2026 재구성 공포 사건은 3월에 몰렸다(≤25 첫날 {', '.join(A['R252|E1_25'])}, ≤20 첫날 {', '.join(A['R252|E1_20'])} → 100%, 4/13 원래 비중 복귀).")
    g_n3, g_s3 = ev("R252", "NDX3", "E3_75", 126, "2026뒤"), ev("R252", "SOX3", "E3_75", 126, "2026뒤")
    w(f"- 재구성 탐욕 경고(≥75 첫날)는 6/2에 켜졌다(최고 76.1). 다음 날 종가부터 10/9까지 나스닥100 3배 {pct(g_n3['mean'])}, 반도체 3배 {pct(g_s3['mean'])}. "
      f"한 번뿐이고, 같은 규칙(126일)은 2011~2025에 나스닥100 3배에선 아무 날보다 좋았고({pp(ev('R252', 'NDX3', 'E3_75', 126, 'rchk')['ex'])}) "
      f"반도체 3배에선 조금 나빴다({pp(ev('R252', 'SOX3', 'E3_75', 126, 'rchk')['ex'])}) → 덜기 신호로 쓰기엔 약하다.")
    w("- 실제 F&G는 6/26 24.7로 ≤25 첫날이 켜져, 같은 S1 ≤25 규칙은 6/29 종가에 100%로 올려 8/4까지 들고 7월 쇼크 둘째 다리(7/29까지)를 100%로 맞았다.")
    w("")
    w("## 시험 범위·선택편향·한계")
    w("")
    w(f"- 시험 {G['전체']}개(재구성 {G['재구성_조합']} + 실제 {G['실제_조합']}). 고르기 칸마다 사건형 매수 60·전략형 18 후보 중 최대를 골랐다. "
      "대표는 3배 확인 통과 후보 중 확인 값 최대라 확인 구간을 한 번 더 쓴 선택이다(위약 절차 순위로 보정해 읽는다).")
    w("- 재구성 값은 CNN의 실제 정규화(공개 안 됨)와 다르다. 요소 7개 중 신고가 강도·맥클렐런 폭이 빠졌고, 정크 대리·폭 대리는 CNN 값과 상관이 낮다. "
      "실제 F&G 극단(≤10)의 절반만 재구성도 ≤10이라, 실제 F&G 규칙으로 옮길 때 문턱이 다르다.")
    w("- 지수는 시가가 없어 '다음 날 종가 체결'. 1993·1999 전 S&P·나스닥100은 가격 지수(배당 없음), 뒤는 SPY·QQQ(배당 수정)로 이었다. 3배는 합성(일일 3배 − 2×^IRX − 보수 0.95% − 추가 비용).")
    w(f"- 대표 규칙은 1990~2026 사건 {ne}건, 확인 구간 t≈{gs3['t']:.1f}로 통계적으로 약하다. 2026에 신호가 없어 현재 장세 판정을 못 한다.")
    w("- 2026 창의 사건형은 10/9까지 미완 보유를 그날 평가하고 아무 날도 같은 길이로 맞춘다. 창 끝에 가까운 사건은 비교할 아무 날이 창 앞쪽에만 남아 비교가 약하다.")
    w("- 장세 전환 위험: 2008(첫 신호 뒤 추가 하락)·2021-11(긴 약세장 초입)·1998·2020(3배 반도체)에서 아무 날보다 나빴다.")
    w("- 공포 때 '더 담기'는 평소 비중을 낮춰 둬야 하는 규칙이라, 7종목처럼 늘 100% 들고 가는 계좌엔 '공포 때 새 돈 넣기'로만 옮겨진다.")
    (RESULTS / f"{ID}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (RESULTS / f"{ID}.json").write_text(json.dumps(jclean(J), ensure_ascii=False, indent=1), encoding="utf-8")
    print("보고서", RESULTS / f"{ID}.md")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch()
    elif len(sys.argv) > 1 and sys.argv[1] == "strength":
        strength_build()
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        main()

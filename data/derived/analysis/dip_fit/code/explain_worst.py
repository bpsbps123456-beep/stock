"""이격매수매도 최근 1개월 상·하위 6종목의 손실 거래를 사후 설명한다(2026-10-09). 매매 근거가 아니다.

  1) trades.csv에서 종목별 수익 합 상·하위 6종목을 고른다(전체 세션, 정규장만 함께 표시).
  2) 손실 거래(ret <= -3%)마다 같은 시간 창(매수 봉 시작 ~ 매도 봉 시작)에서
     종목 자체 종가 흐름, 같은 업종 종목(PEERS) 중앙값, 업종 ETF·QQQ를 1분봉 마지막 종가로 잰다.
     혼자 빠짐: 종목 - 업종 중앙값 <= -3%p 이고 업종 중앙값 > -2%
     같이 빠짐: 업종 중앙값 <= -2% 이고 차이 > -3%p
     둘 다:     업종 중앙값 <= -2% 이고 차이 <= -3%p
     잡음:      나머지(손실이 업종과 3%p 안쪽)
     일봉: 그날 종가/전날 종가, 정규장 거래량 / 직전 20일 평균(뉴스가 있으면 커진다).
  3) 12종목 프리·애프터 거래의 체결가 꼬리 의심: 매수가가 매수 시각 ±10분 1분봉 종가 중앙값보다 3% 넘게 낮거나,
     매도가가 매도 시각 ±10분 중앙값보다 3% 넘게 높으면 '꼬리 체결'로 표시(yfinance 단발 체결). 정규장은 뺀다(거래량이 실제라서).
  4) NEWS: 웹 검색(8회 한도)으로 찾은 그날 재료. 사후 설명용.
저장: ../results/explain_worst.txt
사용: py explain_worst.py"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/dip/raw_1009"
RES = os.path.join(HERE, "../results")
LOSS, TAIL = -0.03, 0.03

# 업종 비교군: 사업이 가장 가까운 종목(유니버스 group보다 좁게). 참고 ETF는 업종 ETF가 있으면 그것, 없으면 QQQ.
PEERS = {
    "HUT": (["IREN", "CIFR", "WULF", "APLD", "GLXY"], "QQQ"),          # 비트코인 채굴 → AI 데이터센터 전환주
    "RXRX": (["SDGR", "TEM", "BEAM", "CRSP", "NTLA", "TWST", "TXG"], "XBI"),
    "CRCL": (["COIN", "HOOD", "MSTR", "GLXY"], "QQQ"),               # 코인·스테이블코인
    "OKLO": (["SMR", "CEG", "TLN", "VST"], "QQQ"),                   # 원전·전력
    "ARM": (["NVDA", "AMD", "AVGO", "QCOM", "MRVL"], "SMH"),
    "MDB": (["SNOW", "DDOG", "NET", "CRWD", "GTLB", "DT", "OKTA"], "IGV"),
    "BE": (["VRT", "GEV", "ETN", "CEG", "OKLO", "SMR"], "QQQ"),
    "ILMN": (["TXG", "TWST", "GH", "NTRA", "GRAL"], "XBI"),
    "SDGR": (["RXRX", "TEM", "TXG", "TWST"], "XBI"),
    "GLW": (["COHR", "LITE", "CIEN", "FN", "AAOI"], "QQQ"),
    "MRVL": (["AVGO", "NVDA", "CRDO", "ALAB", "AMD"], "SMH"),
    "ENTG": (["AMAT", "LRCX", "KLAC", "ASML", "TER"], "SMH"),
}

# 웹 검색(2026-10-09, 8회)으로 확인한 그날 재료(사후 설명용). 키 = (종목, ET 날짜). 검색하지 않은 날은 '-'.
NEWS = {
    ("MDB", "2026-09-28"): "CEO CJ Desai 메타 이직 발표(창업자 임시 CEO, 가이던스 유지) → 시가 -24% 갭 [종목 고유]",
    ("OKLO", "2026-09-11"): "OKLO $10억 주식 매각(ATM) 발표 + UBS SMR 매도 하향(목표 10→6) [종목·소형원전 고유, 대형 전력주는 무관]",
    ("RXRX", "2026-10-08"): "공시 없음. 현금 소진·파이프라인 지연 우려 기사뿐(-7.5%) [뚜렷한 재료 없음]",
    ("HUT", "2026-10-08"): "비트코인 약세로 채굴주 동반 하락(-7.5%); MarketBeat의 실적 미스 문구는 지난 분기 재인용(earn=False) [업종+약간 더]",
    ("CRCL", "2026-09-15"): "CLARITY Act 상원 표결 앞 코인주 프리장 동반 매도, CRCL이 가장 약함 [업종(규제) 재료]",
    ("ARM", "2026-09-14"): "앤트로픽·오픈AI·MS 'AI 개발 속도 늦추자' 발언 → 반도체 급락(SOX -5.9%), ARM -9.7% [업종]",
    ("ARM", "2026-09-28"): "9월 +28% 급등 뒤 반도체 매도(QCOM -6%, MRVL -5%) + 소프트뱅크 ARM 담보대출 $250억 확대 [업종+종목]",
    ("ARM", "2026-10-08"): "퀄컴 로열티 소송 배심 재판 + FT 'OpenAI 매출 의문' 보도로 AI 반도체 약세(-6.5%) [업종+종목]",
    ("GLW", "2026-09-14"): "GLW $20억 ATM 증자 + AI 둔화 발언 광통신 급락(COHR -11%, LITE -9%) [업종+종목]",
    ("HUT", "2026-09-14"): "9/14 AI 둔화 발언 매도일(AI 데이터센터주 동반) [업종]",
    ("BE", "2026-10-08"): "10/8 FT 'OpenAI 매출 의문' AI 매도일(같은 날 HUT·ARM·GLW·RXRX도 손실) [업종]",
    ("GLW", "2026-10-08"): "10/8 AI 매도일, AAOI -12% [업종]",
}

# 결론 문장: 아래 숫자를 확인한 뒤 고정 문자열로 둔다(2026-10-09 실행 기준).
SUMMARY = [
    "1) 하위 6종목 순위는 1~2일이 만든다: 최악 1건만 빼면 HUT -11.6→-4.5%, RXRX -9.4→-3.2%, CRCL -8.6→+0.1%, OKLO -8.4→-0.9%, ARM -6.7→-1.6%, MDB -5.9→+10.4%.",
    "2) 하위 6 큰 손실(<=-3%) 15건 -89%: 혼자 빠짐 3건 -30%(MDB CEO 이탈, OKLO $10억 주식 매각, RXRX 재료 없음),"
    " 업종과 같이 빠짐 7건 -30%, 업종도 빠지고 더 빠짐 5건 -29%. 같은 날 몰림: 10/8(AI 매도) 5건, 9/14(AI 둔화 발언) 3건.",
    "3) 하위 손실 15건 중 프리 매수 10건, 8시 매도 11건: 프리장 이격에 사서 RSI 반등 없이 하루 내내 빠진 날이 손실의 본체다.",
    "   정규장만 보면 HUT +4.9%, MDB +4.4%, ARM +2.0%로 플러스. 정규장도 나쁜 종목은 CRCL -10.3%, RXRX -4.3%뿐.",
    "4) 상위 6종목 일부는 데이터 꼬리: BE +76→+18%(9/22 애프터 174.96 단발 체결 +58%), GLW +52→+4%(애프터 2건 +48%),"
    " ILMN +71→+46%(정규장 거래 0건, 2% 기준 꼬리 7건), ENTG +48→+28%. 꼬리 없이 남는 상위: SDGR +53%(정규 +23%), MRVL +43%, ILMN, ENTG.",
    "5) 남는 상위 4종목은 한 달 주가가 올랐다(9/9→10/8 SDGR +37%, ILMN +29%, MRVL +17%, ENTG +13%; 업종 중앙 +4~18%)."
    " 하위는 OKLO -19%, HUT -17%, CRCL -13%로 내렸거나 MDB(+4% vs 업종 +21%)처럼 업종보다 약했다. 단 RXRX(+23%)·ARM(+4%)은 예외.",
    "6) 사후 설명: 한 달 주가·뉴스·업종 동반 하락은 모두 매수 뒤에 안 값이다. 매매 근거로 쓰지 않는다."
    " 매수 전에 쓸 수 있는 후보(실적·증자 일정, 업종 동반 이격)는 다른 에이전트의 앞→뒤 검증 결과를 따른다.",
]

SOURCES = [
    "MDB 9/28: https://www.thestreet.com/investing/stocks/mongodb-stock-crash-ceo-resigns-meta , https://www.fool.com/coverage/stock-market-today/2026/09/28/stock-market-midday-sept-28-stocks-slide-as-yields-rise-mongodb-tumbles/",
    "OKLO 9/11: https://koalagains.com/daily-top-movers/top-losers/details/002892bc-8ab5-4dfe-9918-66680d49bc72 , https://www.benzinga.com/markets/equities/26/09/61885903/shorts-prevail-as-oklo-and-nuscale-stocks-dive-as-smr-risks-rise",
    "HUT 10/8: https://www.marketbeat.com/instant-alerts/price-hut-8-nasdaq-hut-stock-falls-75-heres-what-happened-2026-10-08/",
    "RXRX 10/8: https://www.timothysykes.com/news/recursion-pharmaceuticals-inc-rxrx-news-2026_10_08/",
    "CRCL 9/15: https://finance.yahoo.com/markets/crypto/articles/crypto-stocks-mstr-coin-crcl-104620721.html",
    "ARM 9/28: https://247wallst.com/investing/2026/09/28/arm-sinks-9-as-chip-selloff-deepens-qualcomm-drops-6-marvell-slides-5/ , https://www.tikr.com/blog/arm-holdings-stock-fell-8-in-a-day-heres-what-softbanks-25-billion-loan-means-for-the-stock",
    "9/14 AI 매도·GLW: https://www.semafor.com/article/09/14/2026/ai-slowdown-calls-prompt-chip-selloff , https://finance.yahoo.com/markets/stocks/articles/corning-tumbles-12-2b-market-174950803.html",
    "ARM·10/8: https://www.benzinga.com/trading-ideas/movers/26/10/62243612/why-is-arm-stock-falling-thursday , https://finance.yahoo.com/markets/live/stock-market-today-thursday-october-8-dow-sp-500-nasdaq-080537884.html",
]

_m1, _d1 = {}, {}


def m1(s):
    if s not in _m1:
        x = pd.read_parquet(f"{RAW}/1m_{s}.parquet")
        x.index = x.index.tz_convert("America/New_York")
        _m1[s] = x[x.High > 0]
    return _m1[s]


def d1(s):
    if s not in _d1:
        x = pd.read_parquet(f"{RAW}/1d_{s}.parquet")
        x.index = pd.to_datetime(x.index).date
        _d1[s] = x
    return _d1[s]


def bars3(s):
    """build.py와 같은 3분봉(1분봉을 시계 기준 3분으로 묶음) → 매도 봉 시각을 bars로 되찾는다."""
    return m1(s).resample("3min", label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna(subset=["Close"]).index


def px_at(s, t):
    """t 직전(포함 안 함) 마지막 1분봉 종가."""
    c = m1(s).Close
    c = c[c.index < t]
    return c.iloc[-1] if len(c) else np.nan


def win(s, t0, t1):
    a, b = px_at(s, t0), px_at(s, t1)
    return b / a - 1


def local_med(s, t):
    c = m1(s).Close
    return c[(c.index >= t - pd.Timedelta(minutes=10)) & (c.index <= t + pd.Timedelta(minutes=10))].median()


def daily(s, dy):
    x = d1(s)
    if dy not in x.index:
        return np.nan, np.nan
    i = list(x.index).index(dy)
    cc = x.Close.iloc[i] / x.Close.iloc[i - 1] - 1
    vr = x.Volume.iloc[i] / x.Volume.iloc[max(0, i - 20):i].mean()
    return cc, vr


def kind(ex, pm):
    if pm <= -0.02:
        return "둘 다" if ex <= -0.03 else "같이 빠짐"
    return "혼자 빠짐" if ex <= -0.03 else "잡음"


def main():
    t = pd.read_csv(os.path.join(RES, "trades.csv"))
    t["t"] = pd.to_datetime(t.t, utc=True).dt.tz_convert("America/New_York")
    t["day"] = pd.to_datetime(t.day).dt.date
    g = t.groupby("sym").agg(n=("ret", "size"), s=("ret", "sum"))
    g["s_reg"] = t[t.ses == "정규"].groupby("sym").ret.sum()
    g["n_reg"] = t[t.ses == "정규"].groupby("sym").ret.size()
    g = g.fillna(0).sort_values("s")
    bot, top = list(g.index[:6]), list(g.index[-6:][::-1])
    L = ["이격매수매도 최근 1개월(매수일 ET 2026-09-10~10-08) 상·하위 6종목 손실일 사후 설명",
         "** 사후 설명용이다. 뉴스·업종 비교는 매수 뒤에야 알 수 있는 값이라 매매 근거로 쓰지 않는다. **",
         f"손실 거래 = ret <= {LOSS:.0%}. 창 = 매수 봉 시작 ~ 매도 봉 시작(1분봉 마지막 종가). 혼자 = 종목-업종중앙값 <= -3%p & 업종 > -2%,"
         " 같이 = 업종 <= -2%, 둘 다 = 업종 <= -2% & 차이 <= -3%p, 잡음 = 나머지.",
         f"꼬리 체결(프리·애프터만) = 매수가 < 매수 시각 ±10분 1분봉 중앙값 x(1-{TAIL:.0%}) 또는 매도가 > 매도 시각 중앙값 x(1+{TAIL:.0%}).",
         f"순위(137종목 중) 하위 6: {' '.join(bot)} | 상위 6: {' '.join(top)}", ""]

    rows, tails, tail2 = [], [], []
    for s in bot + top:
        x = t[t.sym == s].sort_values("t")
        idx = bars3(s)
        peers, ref = PEERS[s]
        for _, r in x.iterrows():
            i0 = idx.get_indexer([r.t])[0]
            t1 = idx[i0 + int(r.bars)] if i0 >= 0 and i0 + int(r.bars) < len(idx) else pd.NaT
            mi, mx = local_med(s, r.t), (local_med(s, t1) if pd.notna(t1) else np.nan)
            ext = r.ses != "정규"          # 정규장은 실제 거래량이 있어 빠른 반등을 꼬리로 오인한다(BE 9/25 시가 봉 거래량 58만 주)
            te = ext and r.ent < mi * (1 - TAIL)
            tx = ext and pd.notna(mx) and r.px > mx * (1 + TAIL)
            if ext and r.ent < mi * 0.98:
                tail2.append(s)
            if te or tx:
                tails.append(dict(sym=s, day=r.day, ses=r.ses, ret=r.ret, ent=r.ent, med_in=mi, px=r.px, med_out=mx,
                                  what="매수" if te else "매도"))
            if r.ret > LOSS or pd.isna(t1):
                continue
            own = win(s, r.t, t1)
            pw = pd.Series({p: win(p, r.t, t1) for p in peers}).dropna()
            pm = pw.median()
            rw, qw = win(ref, r.t, t1), win("QQQ", r.t, t1)
            cc, vr = daily(s, r.day)
            pcc = np.nanmedian([daily(p, r.day)[0] for p in peers])
            rows.append(dict(sym=s, side="하위" if s in bot else "상위", day=r.day, t=r.t.strftime("%H:%M"), t1=t1.strftime("%m-%d %H:%M"),
                             ses=r.ses, why=r.why, ret=r.ret, own=own, peer=pm, ex=own - pm, ref=rw, qqq=qw,
                             worst_peer=f"{pw.idxmin()} {pw.min():+.1%}", cc=cc, peer_cc=pcc, vr=vr,
                             kind=kind(own - pm, pm), news=NEWS.get((s, str(r.day)), "")))
    R = pd.DataFrame(rows)
    T = pd.DataFrame(tails)
    R.to_csv(os.path.join(RES, "explain_worst.csv"), index=False, encoding="utf-8-sig")

    c0, c1 = pd.Timestamp("2026-09-09").date(), pd.Timestamp("2026-10-08").date()
    mret = lambda q: d1(q).Close.get(c1) / d1(q).Close.get(c0) - 1
    L.append("== ① 종목 요약 (수익 합 %, 전체 | 정규장만 | 꼬리 체결 빼고 | 최악 1건 빼고 | 한 달 주가 9/9→10/8 종목/업종중앙, 사후) ==")
    for s in bot + top:
        x = t[t.sym == s]
        mo, pmo = mret(s), np.nanmedian([mret(p) for p in PEERS[s][0]])
        tt = T[T.sym == s] if len(T) else T
        tail_sum = tt.ret.sum() if len(tt) else 0.0
        lr = R[R.sym == s] if len(R) else R
        kinds = lr.kind.value_counts().to_dict() if len(lr) else {}
        L.append(f"{s:5s} {'하위' if s in bot else '상위'} {len(x):2d}건 합 {x.ret.sum():+.1%} | 정규 {g.loc[s, 'n_reg']:.0f}건 {g.loc[s, 's_reg']:+.1%}"
                 f" | 애프터 {(x.ses == '애프터').mean():.0%} | 꼬리 {len(tt)}건 {tail_sum:+.1%} → 빼면 {x.ret.sum() - tail_sum:+.1%}"
                 f" (2% 기준 {tail2.count(s)}건) | 최악 1건 빼면 {x.ret.sum() - x.ret.min():+.1%} | 한 달 {mo:+.0%}/{pmo:+.0%}"
                 f" | 손실(<=-3%) {len(lr)}건 {lr.ret.sum() if len(lr) else 0:+.1%} {kinds}")
    L.append("")
    L.append("== ② 손실 거래별 같은 창 비교 (% ) ==")
    L.append("종목 날짜 매수 세션 이유 → 매도 | 거래 | 종목창 업종중앙 차이 업종ETF QQQ | 최악 비교종목 | 일봉 종목/업종 거래량배 | 판정 | 재료")
    for _, r in R.iterrows():
        L.append(f"{r.sym:5s} {r.day} {r.t} {r.ses} {r.why} → {r.t1} | {r.ret:+.1%} | {r.own:+.1%} {r.peer:+.1%} {r.ex:+.1%}p"
                 f" {r.ref:+.1%} {r.qqq:+.1%} | {r.worst_peer} | {r.cc:+.1%}/{r.peer_cc:+.1%} {r.vr:.1f}x | {r.kind} | {r.news or '-'}")
    L.append("")
    L.append("== ③ 꼬리 체결 의심 거래 ==")
    for _, r in T.iterrows():
        L.append(f"{r.sym:5s} {r.day} {r.ses} {r.what} | 거래 {r.ret:+.1%} | 매수가 {r.ent:.2f} (주변 {r.med_in:.2f}) 매도가 {r.px:.2f} (주변 {r.med_out:.2f})")
    L.append("")
    b = R[R.side == "하위"]
    L.append("== ④ 하위 6종목 손실 합 판정별 ==")
    for k, v in b.groupby("kind").ret.agg(["size", "sum"]).iterrows():
        L.append(f"{k}: {int(v['size'])}건 {v['sum']:+.1%}")
    L.append(f"하위 손실 거래 중 프리 매수: {(b.ses == '프리').sum()}/{len(b)}건, 8시 매도: {(b.why == '8시').sum()}/{len(b)}건, 일봉 거래량 2배 이상: {(b.vr >= 2).sum()}건")
    L.append("손실 거래 날짜 몰림(12종목, <=-3%): " + ", ".join(f"{k} {v}건" for k, v in R.day.astype(str).value_counts().items() if v >= 2))
    L.append("")
    L.append("== ⑤ 요약(사후 설명, 매매 근거 아님) ==")
    L.extend(SUMMARY)
    L.append("")
    L.append("== 출처(웹 검색 8회, 사후 확인) ==")
    L.extend(SOURCES)
    with open(os.path.join(RES, "explain_worst.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

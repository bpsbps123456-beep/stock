"""2026-10 반도체 하락(10/6~13) 원인 점검: '헤지펀드 레버리지 → 헤지 매도' 주장 대조.

1) 미국·한국 반도체 단면 수익률(10/2 종가 → 10/13 종가)
2) SK하이닉스 미국 ADR(SKHY, 1 ADS = 0.1주) 대 원주
3) 네이버 투자자별 매매(외국인·기관·개인, 잔차 = 기타법인·자사주)
4) 레버리지·인버스 상품의 10/13 기계적 리밸런싱 추정: L*(L-1)*r*NAV_전일

실행: py data/derived/analysis/chip_selloff_2026-10/code/analyze.py
"""
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from desk.giant import NAVER_TREND, _num  # noqa: E402

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)

GROUPS = {
    "한국 메모리": ["000660.KS", "005930.KS"],
    "미국 메모리·저장": ["MU", "SNDK", "WDC", "STX"],
    "반도체 장비": ["ASML", "AMAT", "LRCX", "KLAC"],
    "AI 연산칩": ["NVDA", "AVGO", "AMD", "MRVL", "TSM"],
    "아날로그·기타": ["TXN", "ADI", "MCHP", "ON", "NXPI", "QCOM", "INTC", "ARM"],
}
INDEX = ["^SOX", "SMH", "SOXL", "SPY", "RSP", "QQQ", "GVIP", "MTUM", "SKHY", "KRW=X"]
START, END = "2026-10-02", "2026-10-13"


SNAPSHOT = OUT / "close_snapshot.csv"  # 2026-10-14 받은 종가. 10/13 미국 종가는 대화 중 TradingView와 대조해 일치


def load_close(tickers: list[str]) -> pd.DataFrame:
    """종가 표. Yahoo가 최근 종가를 빈칸으로 주는 때가 있어(10/14 재요청 시 미국 10/13 전부 NaN)
    저장한 스냅숏을 먼저 쓰고, 기간 끝 값이 비면 멈춘다(앞 값으로 메우지 않음)."""
    if SNAPSHOT.exists():
        close = pd.read_csv(SNAPSHOT, index_col=0, parse_dates=True)
    else:
        close = yf.download([t for t in tickers if t != "KRW=X"], start="2026-06-15", end="2026-10-15",
                            auto_adjust=True, progress=False)["Close"]
        close.to_csv(SNAPSHOT)
    fx = yf.download("KRW=X", start="2026-09-25", end="2026-10-15", progress=False)["Close"].squeeze()
    close = close.join(fx.rename("KRW=X"), how="outer")
    gaps = [t for t in tickers if close.loc[[START, END], t].isna().any()]
    if gaps:
        raise SystemExit(f"기간 끝 종가 없음: {gaps}")
    return close


def cross_section() -> dict:
    tickers = [t for g in GROUPS.values() for t in g] + INDEX
    close = load_close(tickers)
    ret = (close.loc[END] / close.loc[START] - 1) * 100
    out = {g: {"평균": round(ret[ts].mean(), 1), **{t: round(ret[t], 1) for t in ts}} for g, ts in GROUPS.items()}
    out["지수·ETF"] = {t: round(ret[t], 1) for t in INDEX if t != "KRW=X"}
    adr = close["SKHY"] * 10 / (close["000660.KS"] / close["KRW=X"].ffill()) - 1
    out["ADR 대 원주"] = {
        "SKHY 10/8→10/13": round((close.loc[END, "SKHY"] / close.loc["2026-10-08", "SKHY"] - 1) * 100, 1),
        "000660 10/8→10/13": round((close.loc[END, "000660.KS"] / close.loc["2026-10-08", "000660.KS"] - 1) * 100, 1),
        "ADR 프리미엄 10/2·10/8·10/13 %": [round(adr.loc[d] * 100, 1) for d in (START, "2026-10-08", END)],
    }
    return out


def kr_flows() -> dict:
    out = {}
    for code, name in [("000660", "SK하이닉스"), ("005930", "삼성전자")]:
        rows = requests.get(NAVER_TREND.format(code=code), timeout=15,
                            headers={"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)"}).json()
        days = []
        for r in sorted(rows, key=lambda x: x["bizdate"]):
            if r["bizdate"] < "20260915":
                continue
            px = _num(r["closePrice"])
            f, o, i = (_num(r[k]) * px / 1e8 for k in ("foreignerPureBuyQuant", "organPureBuyQuant", "individualPureBuyQuant"))
            days.append({"일자": r["bizdate"], "외국인": round(f), "기관": round(o), "개인": round(i),
                         "기타(자사주 등)": round(-(f + o + i)), "거래대금": round(_num(r["accumulatedTradingVolume"]) * px / 1e8)})
        oct_ = [d for d in days if d["일자"] >= "20261006"]
        sep = [d for d in days if "20260915" <= d["일자"] <= "20261002"]
        out[name] = {
            "단위": "억원",
            "일별": days,
            "10/6~13 합계": {k: sum(d[k] for d in oct_) for k in ("외국인", "기관", "개인", "기타(자사주 등)")},
            "기타 하루 평균 9/15~10/2": round(sum(d["기타(자사주 등)"] for d in sep) / len(sep)),
            "기타 하루 평균 10/6~13": round(sum(d["기타(자사주 등)"] for d in oct_) / len(oct_)),
        }
    return out


def mechanical_rebalance() -> dict:
    """10/13 하루 리밸런싱 추정(억원). NAV는 10/13 종가 시총(네이버 ETF 목록)을 전일로 되돌린 값.

    HK 7709·7747은 TradingView AUM(10/12 NAV 기준, 홍콩달러)을 쓴다. 'Daily Max 2x'라 실제 배율이
    2보다 낮을 수 있고, 설정·환매(개인 순매수)는 빼지 않은 총량이다.
    """
    fx, hkd = 1334.13, 7.78
    r = {"하이닉스": -0.1009, "삼성": -0.0739}
    kr = json.loads(requests.get(
        "https://finance.naver.com/api/sise/etfItemList.nhn?etfType=0&targetColumn=market_sum&sortOrder=desc",
        timeout=20, headers={"User-Agent": "Mozilla/5.0"}).content.decode("euc-kr"))["result"]["etfItemList"]
    cap = lambda word, lev: sum(it["marketSum"] for it in kr if word in it["itemname"] and lev in it["itemname"]
                                and "단일종목" in it["itemname"])
    flow = lambda L, ret, nav0: L * (L - 1) * ret * nav0
    out = {}
    for name, word in [("하이닉스", "SK하이닉스"), ("삼성", "삼성전자")]:
        lev_now, inv_now = cap(word, "레버리지"), cap(word, "인버스2X")
        ret = r[name]
        out[f"KR {name} 2배"] = round(flow(2, ret, lev_now / (1 + 2 * ret)))
        out[f"KR {name} -2배"] = round(flow(-2, ret, inv_now / (1 - 2 * ret)))
    out["HK 7709 하이닉스 2배"] = round(flow(2, r["하이닉스"], 30.815e9 / hkd * fx / 1e8))
    out["HK 7747 삼성 2배"] = round(flow(2, r["삼성"], 8.938e9 / hkd * fx / 1e8))
    out["SOXL+SOXS 지수 1%당(억달러)"] = round((flow(3, -0.01, 239.7) + flow(-3, -0.01, 18.17)), 1)
    return out


if __name__ == "__main__":
    res = {"단면": cross_section(), "한국 수급": kr_flows(), "기계적 리밸런싱 10/13(억원)": mechanical_rebalance()}
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "한국 수급"}, ensure_ascii=False, indent=1))
    for name, v in res["한국 수급"].items():
        print(name, v["10/6~13 합계"], "기타 하루 평균", v["기타 하루 평균 9/15~10/2"], "→", v["기타 하루 평균 10/6~13"])
